import logging
import sys
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

from espei.priors import PriorSpec, build_prior_specs, rv_zero
from espei.utils import unpack_piecewise
from espei.error_functions.context import setup_context

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  Matern
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score

import torch
import gpytorch




from SALib.analyze import sobol
from SALib.sample.sobol import sample
from scipy.stats import qmc
from scipy.spatial.distance import euclidean

_log = logging.getLogger(__name__)


class UNOptimizer():
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
    def __init__(self, dbf, phase_models=None, scheduler=None):
        self.dbf = copy.deepcopy(dbf)
        self.phase_models = phase_models
        self.scheduler = scheduler
        self.save_interval = 1
        # These are set by the _fit method
        self.sampler = None
        self.tracefile = None
        self.probfile = None
        self.ctx = None

    @staticmethod
    def initialize_new_chains(params, chains_per_parameter, std_deviation, deterministic=True):
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
        _log.trace('Initial parameters: %s', params)
        params = np.array(params)
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
            np.random.seed(42)
            #rng = np.random.RandomState(1769)
        else:
            rng = np.random.RandomState()
        # apply a Gaussian random to each parameter with std dev of std_deviation*parameter
        tiled_parameters = np.tile(params, (nchains, 1))
        chains = rng.normal(tiled_parameters, np.abs(tiled_parameters * std_deviation))
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
        tr = self.tracefile
        if tr is not None:
            _log.trace('Writing trace to %s', tr)
            np.save(tr, self.result)
        else:
            print("Please provide saving path, Initial samples were not saved.")
        # prob = self.probfile
        # if prob is not None:
        #     _log.trace('Writing lnprob to %s', prob)
        #     np.save(prob, self.sampler.lnprobability)

    def do_sampling(self, initial_guess, sample_n=1000, alpha=0.5,sigma_factor=0.1,range_factor=1,seed=42):
        # Parameters:
        # initial_guess (array): Initial guess of shape (D,)
        # sample_n (int): Number of total samples
        # alpha (float): Fraction of samples centered around A
        # sigma_factor (float): % of range used for Gaussian perturbation
        # range_factor (float): % of A used to estimate implicit bounds
        # seed (int, optional): Random seed for reproducibility
    
        # Returns:
        #     samples (array): Generated samples of shape (N, D)
        if not os.path.exists("samples.csv"):
            progbar_width = 30
            np.random.seed(seed)
            Dim = len(initial_guess) # Number of parameters

            # Estimate bounds as ± range_factor * A
            min_bounds = initial_guess - range_factor * np.abs(initial_guess)  # Ensure no negative values if needed
            max_bounds = initial_guess + range_factor * np.abs(initial_guess)

            # Create LHS samples
            sampler = qmc.LatinHypercube(d=Dim, seed=seed)
            lhs_samples = sampler.random(sample_n)
            lhs_scaled = min_bounds + lhs_samples * (max_bounds - min_bounds)

            # Select alpha*N samples to be centered around initial guess
            num_centered = int(alpha * sample_n)
            num_uniform = sample_n #- num_centered  # Remaining samples for full-space coverage

            # Gaussian perturbation around A
            sigma = abs(sigma_factor * (max_bounds - min_bounds))
            
            centered_samples = np.random.normal(initial_guess, sigma, size=(num_centered, Dim))
            
            # Clip to ensure samples remain within estimated range
            #centered_samples = np.clip(centered_samples, min_bounds, max_bounds)
            
            # Combine samples
            final_samples = np.vstack((centered_samples, lhs_scaled))
            np.random.shuffle(final_samples)
            sample_n = len(final_samples)
            residual = []
            try:
                for idx in range(sample_n):
                    # progress bar
                    residual.append([self.generate_samples(final_samples[idx],idx, **self.ctx)])
                    n = int((progbar_width) * float(idx + 1) / sample_n)
                    #_log.info("\r[%s%s] (%d of %d)\n", '#' * n, ' ' * (progbar_width - n), idx + 1, sample_n)
                    sys.stdout.write(f"\r[{('#' * n).ljust(progbar_width)}] ({idx + 1} of {sample_n})")
                    sys.stdout.flush()
            except KeyboardInterrupt:
                pass

            #pack the parameters with prediction results
            samples = np.hstack((final_samples, np.array(residual)))
            self.results = samples
            _log.info('Initial Space filling complete.')
            np.savetxt("samples.csv", samples, delimiter=",")
        else:
            print("Samples already exist, loading from file.")
            samples = np.loadtxt("samples.csv", delimiter=",")
        best_id = np.argmax(np.array(samples[:,-1]))

        #Training data for surrogate model
        X = samples[:,:-1]  # First 22 columns → Parameters
        Y = samples[:,-1]  # Last 2 columns → CALPHAD evaluations

        if not np.all(Y < 0):
            raise ValueError("All training targets must be negative (log probabilities).")
        #Y = np.log(-Y)
        # Normalize inputs (optional, recommended for GP models)
        scaler_train = MinMaxScaler()
        X_scaled = scaler_train.fit_transform(X)
        _log.info('Trainning the GP surrogate.')
        _log.info(f"X shape: {X.shape}, Y shape: {Y.shape}")

        kernel = Matern(length_scale=1.0, nu=2.5)  # Adjust length_scale based on problem
        #kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    
        # Train the GP model
        gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp_model.fit(X_scaled, Y)

        return samples[best_id,:-1], gp_model,samples,scaler_train
    
    def generate_solbol(self, param_count,gp_model,N = 8192):
        # Parameters:
        # Surrogate (GaussianProcessRegressor): Trained surrogate model
        
        # Returns:
        #     active_id (int): Index of the most informative sample

        problem = {
            'num_vars': param_count,
            'names': [f'param_{i+1}' for i in range(param_count)],
            'bounds': [[0, 1]] * param_count  # Assuming normalized data
        }

        # Generate Sobol samples
        N = 8192  # Number of Monte Carlo samples
        X_sobol = sample(problem, N, calc_second_order=True,seed=42)

        # Evaluate GP surrogate at sampled points
        Y_sobol = gp_model.predict(X_sobol)

        # Compute Sobol sensitivity indices
        Si_1 = sobol.analyze(problem, Y_sobol, print_to_console=False)  # First output variable

        return Si_1['ST']

    def generate_gradient(self,param_count,train_data,gp_model,num_samples=1000):
        # Parameters:
        # Surrogate (GaussianProcessRegressor): Trained surrogate model
        # num_samples (int, optional): Number of samples to estimate gradient
        
        # Returns:
        #     active_id (int): Index of the most informative sample

        # Sample points for gradient evaluation (subset of training data for efficiency)
        num_samples = 1000  # Number of points to estimate gradients
        X = train_data[:,:-1]  
        Y = train_data[:,-1]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_test = X_scaled[:num_samples]

        Y_test_pred = gp_model.predict(X_test, return_std=False)

        gradients_1 = np.zeros((X_test.shape[0], X_test.shape[1]))
        epsilon = 1e-4  # Small perturbation for finite differences

        for i in range(X_test.shape[1]):  # Loop over parameters
            X_perturb = X_test.copy()
            X_perturb[:, i] += epsilon  # Small perturbation in one parameter
            
            Y_perturb = gp_model.predict(X_perturb, return_std=False)
            gradients_1[:, i] = (Y_perturb - Y_test_pred) / epsilon  # Compute finite differences

        # Perform PCA on gradient matrix
        pca1 = PCA(n_components=param_count)
        pca1.fit(gradients_1)

        # Extract parameter importance (absolute loadings of first principal component)
        importance_scores_1 = np.abs(pca1.components_[0])  

        return importance_scores_1
        

    def fit(self, symbols, ds, prior=None, iterations=1000,
             decay_f=0.002, sample_counts=np.array([[0]]*22), deterministic=True,
             mcmc_data_weights=None,
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
        initial_guess = np.array([unpack_piecewise(self.dbf.symbols[s]) for s in symbols_to_fit])

        prior_dict = self.get_priors(prior, symbols_to_fit, initial_guess)
        ctx.update(prior_dict)
        self.ctx = ctx
        
        # Run the initial parameters for guessing purposes:
        _log.trace("start initial LHS sampling")
        #Default Input for do_sampling: sample_n=1000, alpha=0.5,sigma_factor=0.1,range_factor=0.2,seed=42

        best_guess,Surrogate, LHS_samples,scaler_train = self.do_sampling(initial_guess, iterations)
        
        # importance_scores_var = self.generate_solbol(len(symbols_to_fit),Surrogate)
        # importance_scores_grad = self.generate_gradient(len(symbols_to_fit),LHS_samples,Surrogate)
        
        # merged_ranking = np.vstack((importance_scores_grad,importance_scores_var)).T
        # weights = np.exp(-decay_f*sample_counts)
        # scaler = MinMaxScaler()
        # norm_merged = scaler.fit_transform(merged_ranking*weights)
        # num_clusters = 2  # Adjust as needed
        # agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
        # labels = agg_clustering.fit_predict(norm_merged)

        # #Compute Cluster Centroids
        # cluster_centroids = np.array([norm_merged[labels == i].mean(axis=0) for i in range(num_clusters)])

        # #Compute Distance from Sensitivity Origin (0,0,0,0)
        # origin = np.zeros(norm_merged.shape[1])  # Origin in sensitivity space
        # cluster_distances = np.array([euclidean(origin, centroid) for centroid in cluster_centroids])

        # #Rank Clusters by Distance (Larger distance = More Important)
        # active_label = np.argmax(cluster_distances)  # Sort in descending order
        active_id = [1]#np.where(labels == active_label)[0]
        return best_guess, Surrogate, active_id,scaler_train

    
    def generate_samples(self, params, iter, **ctx):
        """
        Calculate lnprob = lnlike + lnprior
        """
        _log.debug('Parameters - %s', params)

        # Important to coerce to floats here because the values _must_ be floats if
        # they are used to update PhaseRecords directly
        params = np.asarray(params, dtype=np.float64)
        # lnprior
        prior_rvs = ctx.get('prior_rvs', [rv_zero() for _ in range(params.size)])
        lnprior_multivariate = [rv.logpdf(theta) for rv, theta in zip(prior_rvs, params)]
        _log.debug('Priors: %s', lnprior_multivariate)
        lnprior = np.sum(lnprior_multivariate)

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

        lnprob = lnprior+lnlike
        _log.trace('Sample _ #: %d, lnlike: %0.4f', iter, lnprob)
        return lnprob
    