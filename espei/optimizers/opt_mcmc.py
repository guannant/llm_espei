import logging
import sys
import time
import warnings

import numpy as np
import emcee
import espei
from espei.priors import PriorSpec, build_prior_specs, rv_zero
from espei.utils import unpack_piecewise, optimal_parameters
from espei.error_functions.context import setup_context
from .opt_base import OptimizerBase

from sklearn.preprocessing import MinMaxScaler

from multiprocessing import Pool

import torch

_log = logging.getLogger(__name__)
logging.getLogger("distributed.nanny").setLevel(logging.ERROR)


class EmceeOptimizer(OptimizerBase):
    """
    An optimizer using an EnsembleSampler based on Goodman and Weare [1]
    implemented in emcee [2]

    Attributes
    ----------
    scheduler : mappable
        An object implementing a `map` function
    save_interval : int
        Interval of iterations to save the tracefile and probfile.
    tracefile : str
        Filename to store the trace with NumPy.save. Array has shape
        (chains, iterations, parameters). Defaults to None.
    probfile : str
        filename to store the log probability with NumPy.save. Has shape (chains, iterations)

    References
    ----------
    [1] Goodman and Weare, Ensemble Samplers with Affine Invariance. Commun. Appl. Math. Comput. Sci. 5, 65-80 (2010).
    [2] Foreman-Mackey, Hogg, Lang, Goodman, emcee: The MCMC Hammer. Publ. Astron. Soc. Pac. 125, 306-312 (2013).
    """
    def __init__(self, dbf, verbosity,logger_filename,phase_models=None, scheduler=None):
        super(EmceeOptimizer, self).__init__(dbf)
        self.phase_models = phase_models
        self.scheduler = scheduler
        self.save_interval = 1
        self.log_verbosity = verbosity
        self.log_filename = logger_filename
        # These are set by the _fit method
        self.sampler = None
        self.tracefile = None
        self.probfile = None

    @staticmethod
    def initialize_new_chains(full_params, active_id,chains_per_parameter, std_deviation, deterministic=True):
        """
        Return an array of num_samples from a Gaussian distribution about each parameter.

        Parameters
        ----------
        params : ndarray
            1D array of initial parameters that will be the mean of the distribution.
        num_samples : int
            Number of chains to initialize.
        chains_per_parameter : int
            number of chains for each parameter. Must be an even integer greater or
            equal to 2. Defaults to 2.
        std_deviation : float
            Fractional standard deviation of the parameters to use for initialization.
        deterministic : bool
            True if the parameters should be generated deterministically.

        Returns
        -------
        ndarray

        Notes
        -----
        Parameters are sampled from ``normal(loc=param, scale=param*std_deviation)``.
        A parameter of zero will produce a standard deviation of zero and
        therefore only zeros will be sampled. This will break emcee's
        StretchMove for this parameter and only zeros will be selected.

        """
        
        params = full_params
        #params = params[active_id]
        _log.trace('Initial active parameters: %s', params)
        
        num_zero_params = np.nonzero(params == 0)[0].size
        if num_zero_params > 0:
            _log.warning(
                "%s initial parameters are initialized to zero. The ensemble of chains "
                "for zero parameters will be all initialized to zero and all proposed "
                "values for these parameter will be zero. If possible, it's better to "
                "make a good guess at a reasonable parameter value to start with. "
                "Alternatively, you can start with a small value near zero and let the "
                "ensemble search parameter space.", num_zero_params)
        nchains = params.size * chains_per_parameter
        _log.info('Initializing %s chains with %s chains per parameter.', nchains, chains_per_parameter)
        if deterministic:
            rng = np.random.RandomState(42)
        else:
            rng = np.random.RandomState()
        # apply a Gaussian random to each parameter with std dev of std_deviation*parameter
        tiled_parameters = np.tile(params, (nchains, 1))
        chains = rng.normal(tiled_parameters, np.abs(std_deviation))
        chains[0] = params #Ensure the initial guess is always included in the set, as the std_deviation may be too large to generate feasible points around a "good" initial point.
        return chains

    @staticmethod
    def initialize_chains_from_trace(restart_trace):
        tr = restart_trace
        walkers = tr[np.nonzero(tr)].reshape((tr.shape[0], -1, tr.shape[2]))[:, -1, :]
        nchains = walkers.shape[0]
        ndim = walkers.shape[1]
        initial_parameters = walkers.mean(axis=0)
        _log.info('Restarting from previous calculation with %s chains (%s per parameter).', nchains, nchains / ndim)
        _log.trace('Means of restarting parameters are %s', initial_parameters)
        _log.trace('Standard deviations of restarting parameters are %s', walkers.std(axis=0))
        return walkers

    @staticmethod
    def get_priors(prior, symbols, params):
        """
        Build priors for a particular set of fitting symbols and initial parameters.
        Returns a dict that should be used to update the context.

        Parameters
        ----------
        prior : dict or PriorSpec or None
            Prior to initialize. See the docs on
        symbols : list of str
            List of symbols that will be fit
        params : list of float
            List of parameter values corresponding to the symbols. These should
            be the initial parameters that the priors will be based off of.

        Returns
        -------

        """
        if isinstance(prior, dict):
            _log.info('Initializing a %s prior for the parameters.', prior['name'])
        elif isinstance(prior, PriorSpec):
            _log.info('Initializing a %s prior for the parameters.', prior.name)
        elif prior is None:
            prior = {'name': 'zero'}
        prior_specs = build_prior_specs(prior, params)
        rv_priors = []
        for spec, param, fit_symbol in zip(prior_specs, params, symbols):
            if isinstance(spec, PriorSpec):
                _log.debug('Initializing a %s prior for %s with parameters: %s.', spec.name, fit_symbol, spec.parameters)
                rv_priors.append(spec.get_prior(param))
            elif hasattr(spec, "logpdf"):
                _log.debug('Using a user-specified prior for %s.', fit_symbol)
                rv_priors.append(spec)
        return {'prior_rvs': rv_priors}

    def save_sampler_state(self):
        """
        Convenience function that saves the trace and lnprob if
        they haven't been set to None by the user.

        Requires that the sampler attribute be set.
        """
        # tau = self.sampler.get_autocorr_time()
        # burnin = int(2 * np.max(tau))
        # thin = int(0.5 * np.min(tau))
        tr = self.tracefile
        if tr is not None:
            _log.trace('Writing trace to %s', tr)
            #np.save(tr, self.sampler.get_chain(discard=burnin, flat=True, thin=thin))
            np.save(tr, self.sampler.get_chain(flat=True))
        prob = self.probfile
        if prob is not None:
            _log.trace('Writing lnprob to %s', prob)
            #np.save(prob, self.sampler.get_log_prob(discard=burnin, flat=True, thin=thin))
            np.save(prob, self.sampler.get_log_prob(flat=True))


    def do_sampling(self,chains, initial_guess, iterations,sample_counts,active_id,surrogate, norm_scaler):
        progbar_width = 30
        _log.info('Running MCMC for %s iterations.', iterations)
        needed_for_retrain = np.tile(np.append(initial_guess,0), (1000,1))
        collected_samples = 0
        cutoff = 0.1
        full_set = np.array([initial_guess]*chains.shape[0])
        old_tau = np.inf
        # starttime = time.time()
        try:
            for i, result in enumerate(self.sampler.sample(chains, iterations=iterations)):
                # progress bar
                # sample_time = time.time()
                n = int((progbar_width) * float(i + 1) / iterations)
                sys.stdout.write(f"\r[{('#' * n).ljust(progbar_width)}] ({i + 1} of {iterations})")
                sys.stdout.flush()
                
                # print("\n Time for sample:",sample_time-starttime)
                # if (i + 1) % self.save_interval == 0:
                #     #self.save_sampler_state()
                #     #_log.trace('Acceptance ratios for parameters: %s', self.sampler.acceptance_fraction)
                #     tau = self.sampler.get_autocorr_time(tol=0)
                #     # Check convergence
                #     converged = np.all(tau * 100 < self.sampler.iteration)
                #     converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                #     print("\n Current convergenc:",np.abs(old_tau - tau) / tau)
                #     if converged:
                #         break
                #     old_tau = tau
                   

                if (self.scheduler is not None) and ((i + 1) % 100 == 0):
                     # Note: resetting the scheduler will reset the logger settings for the workers
                     # You'd typically want to run the following, but the verbosity/filename are out of scope
                     #self.scheduler.run(espei.logger.config_logger, verbosity=log_verbosity, filename=log_filename)
                     print("\n Restarting scheduler")
                     self.scheduler.restart()
                     self.scheduler.run(espei.logger.config_logger, verbosity=self.log_verbosity, filename=self.log_filename)
                     

                # sample_counts[active_id] += chains.shape[0]

                # y_test = result[1]
                # full_set[:,active_id] = result[0]
                # x_test = norm_scaler.transform(full_set)
                # y_pred = surrogate.predict(x_test, return_std=False)
                # error = np.abs((y_pred - y_test)/y_test)
                # failed_count = np.sum(error > cutoff)
                # collected_samples += failed_count
                # if collected_samples>needed_for_retrain.shape[0]:
                # #     # needed_for_retrain[collected_samples-failed_count:collected_samples,:-1] = result[0][error > 0.01]
                # #     # needed_for_retrain[collected_samples-failed_count:collected_samples,-1] = result[1][error > 0.01]
                #     print("Need retrain GP")
                #     break
                # needed_for_retrain[collected_samples-failed_count:collected_samples,active_id] = result[0][error > cutoff]
                # needed_for_retrain[collected_samples-failed_count:collected_samples,-1] = result[1][error > cutoff]
                # GP_time = time.time()
                # print("GP_time:",GP_time-sample_time)
                # n = int((progbar_width) * float(i + 1) / iterations)
                # _log.info("\r[%s%s] (%d of %d)\n", '#' * n, ' ' * (progbar_width - n), i + 1, iterations)
                # starttime = time.time()
        except KeyboardInterrupt:
            pass
        print("total failed points:",collected_samples)
        np.savetxt("data_for_retrain.csv", needed_for_retrain, delimiter=",")
        _log.info('MCMC complete.')
        self.save_sampler_state()
        return sample_counts

    def _fit(self, symbols, ds, W_r, ref_pos, initial_guess,sample_counts = np.array([[0]]*22),norm_scaler = None, surrogate = None,active_id = None, prior=None, iterations=1000,
             chains_per_parameter=2, chain_std_deviation=1, deterministic=True,
             restart_trace=None, tracefile=None, probfile=None,
             mcmc_data_weights=None, approximate_equilibrium=False,
             ):
        """

        Parameters
        ----------
        symbols : list of str
        ds : PickleableTinyDB
        prior : str
            Prior to use to generate priors. Defaults to 'zero', which keeps
            backwards compatibility. Can currently choose 'normal', 'uniform',
            'triangular', or 'zero'.
        iterations : int
            Number of iterations to calculate in MCMC. Default is 1000.
        chains_per_parameter : int
            number of chains for each parameter. Must be an even integer greater
            or equal to 2. Defaults to 2.
        chain_std_deviation : float
            Standard deviation of normal for parameter initialization as a
            fraction of each parameter. Must be greater than 0. Defaults to 0.1.
        deterministic : bool
            If True, the emcee sampler will be seeded to give deterministic sampling
            draws. This will ensure that the runs with the exact same database,
            chains_per_parameter, and chain_std_deviation (or restart_trace) will
            produce exactly the same results.
        restart_trace : np.ndarray
            ndarray of the previous trace. Should have shape (chains, iterations, parameters)
        tracefile : str
            filename to store the trace with NumPy.save. Array has shape
            (chains, iterations, parameters)
        probfile : str
            filename to store the log probability with NumPy.save. Has shape (chains, iterations)
        mcmc_data_weights : dict
            Dictionary of weights for each data type, e.g. {'ZPF': 20, 'HM': 2}

        Returns
        -------
        Dict[str, float]

        """
        # Set NumPy print options so logged arrays print on one line. Reset at the end.
        np.set_printoptions(linewidth=sys.maxsize)
        cbs = self.scheduler is None
        
        ctx = setup_context(self.dbf, ds, symbols, data_weights=mcmc_data_weights, phase_models=self.phase_models, make_callables=cbs)
        symbols_to_fit = ctx['symbols_to_fit']


        if approximate_equilibrium:
            warnings.warn(f"Approximate equilibrium is deprecated and will be removed in ESPEI 0.10. Got {approximate_equilibrium}.", DeprecationWarning)

        prior_dict = self.get_priors(prior, symbols_to_fit, initial_guess)
        ctx.update(prior_dict)
        ctx["active_id"] = active_id
        ctx["initial_set"] = initial_guess
        ctx["W_r"] = W_r
        ctx["ref_pos"] = ref_pos    
        norm_initial_guess = norm_scaler.transform(initial_guess.reshape(1, -1))
        initial_tensor = (torch.tensor(norm_initial_guess) - ref_pos) @ W_r
        if restart_trace is not None:
            chains = self.initialize_chains_from_trace(restart_trace)
            # TODO: check that the shape is valid with the existing parameters
        else:
            chains = self.initialize_new_chains(initial_tensor.cpu().numpy(), active_id,chains_per_parameter, chain_std_deviation, deterministic)
        #ctx["surrogate"] = surrogate
        ctx["scaler"] = norm_scaler
        
        sampler = emcee.EnsembleSampler(chains.shape[0], W_r.shape[-1], self.predict, kwargs=ctx, pool=self.scheduler)
        
        #caculate the total amount of steps needed to reach 1000 samples
        sample_limit = iterations
        if deterministic:
            from espei.rstate import numpy_rstate
            sampler.random_state = numpy_rstate
            _log.info('Using a deterministic ensemble sampler.')
        self.sampler = sampler
        self.tracefile = tracefile
        self.probfile = probfile
        # Run the MCMC simulation
        
        sample_counts = self.do_sampling(chains, initial_guess,sample_limit,sample_counts,active_id,surrogate, norm_scaler)
            
        # Post process
        optimal_params = optimal_parameters(sampler.chain, sampler.lnprobability)
        para_reduced = torch.tensor(optimal_params) @ W_r.T + ref_pos  # shape: (N_train, r)
        full_optimal_params = norm_scaler.inverse_transform(para_reduced.cpu().numpy().reshape(1, -1)).reshape(-1)
        _log.trace('Initial parameters: %s', initial_guess)
        _log.trace('Optimal parameters: %s', full_optimal_params)
        _log.trace('Change in parameters: %s', np.abs(initial_guess - full_optimal_params) / initial_guess)
        parameters = dict(zip(symbols_to_fit, full_optimal_params))
        np.set_printoptions(linewidth=75)
        return parameters, sample_counts

    @staticmethod
    def predict(act_params, **ctx):
        """
        Calculate lnprob = lnlike + lnprior
        """
        # _log.debug('Parameters - %s', act_params)

        # Important to coerce to floats here because the values _must_ be floats if
        # they are used to update PhaseRecords directly
        W_r = ctx["W_r"]
        ref_pos = ctx["ref_pos"]
        scaler = ctx["scaler"]
        para_reduced = torch.tensor(act_params) @ W_r.T + ref_pos  # shape: (N_train, r)
        input_params = scaler.inverse_transform(para_reduced.cpu().numpy().reshape(1, -1))
        params = np.asarray(input_params, dtype=np.float64).reshape(-1)
        
        # lnprior
        prior_rvs = ctx.get('prior_rvs', [rv_zero() for _ in range(params.size)])
        lnprior_multivariate = [rv.logpdf(theta) for rv, theta in zip(prior_rvs, params)]
        _log.debug('Priors: %s', lnprior_multivariate)
        lnprior = np.sum(lnprior_multivariate)
        if np.isneginf(lnprior):
            # It doesn't matter what the likelihood is. We can skip calculating it to save time.
            _log.trace('Proposal - lnprior: %0.4f, lnlike: %0.4f, lnprob: %0.4f', lnprior, np.nan, lnprior)
            return lnprior

        # lnlike
        starttime = time.time()
        lnlike = 0.0
        likelihoods = {}
        for residual_obj in ctx.get("residual_objs", []):
            residual_starttime = time.time()
            likelihood = residual_obj.get_likelihood(params)
            residual_time = time.time() - residual_starttime
            likelihoods[type(residual_obj).__name__] = (likelihood, residual_time)
            lnlike += likelihood
        liketime = time.time() - starttime

        like_str = ". ".join([f"{ky}: {vl[0]:0.3f} ({vl[1]:0.2f} s)" for ky, vl in likelihoods.items()])
        lnlike = np.array(lnlike, dtype=np.float64)
        _log.trace('Likelihood - %0.2fs - %s. Total: %0.3f.', liketime, like_str, lnlike)

        lnprob = lnprior + lnlike
        _log.trace('Proposal - lnprior: %0.4f, lnlike: %0.4f, lnprob: %0.4f', lnprior, lnlike, lnprob)
        # act_params = np.asarray(act_params, dtype=np.float64).reshape((1,-1))
        
        #gp_surro = ctx["surrogate"]
        # x_test = gp_scaler.transform(params)
        # lnprob,gp_std = gp_surro.predict(x_test, return_std=True)
        # print("gp_std:",min(gp_std))
        # print("lnprob:",min(lnprob))
        
        return lnprob
