import sys
import os
import re


import numpy as np
import pandas as pd
import copy
from collections import defaultdict

from espei.priors import PriorSpec, build_prior_specs, rv_zero
from espei.utils import unpack_piecewise
from espei.error_functions.context import setup_context

from scipy.stats import qmc


class DataGenerator():
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
    def __init__(self, dbf, ds, phase_models=None):
        self.dbf = copy.deepcopy(dbf)
        self.phase_models = phase_models
        
        ctx = setup_context(self.dbf, ds, None, data_weights=None, phase_models=self.phase_models)
        symbols_to_fit = ctx['symbols_to_fit']
        initial_guess = np.array([unpack_piecewise(self.dbf.symbols[s]) for s in symbols_to_fit])
        prior_dict = self.get_priors(None, symbols_to_fit, initial_guess)
        ctx.update(prior_dict)

        self.ctx = ctx
        self.initial_guess = initial_guess
    
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
            # _log.info('Initializing a %s prior for the parameters.', prior['name'])
            print('Initializing a %s prior for the parameters.', prior['name'])
        elif isinstance(prior, PriorSpec):
            #_log.info('Initializing a %s prior for the parameters.', prior.name)
            print('Initializing a %s prior for the parameters.', prior.name)
        elif prior is None:
            prior = {'name': 'zero'}
        prior_specs = build_prior_specs(prior, params)
        rv_priors = []
        for spec, param, fit_symbol in zip(prior_specs, params, symbols):
            if isinstance(spec, PriorSpec):
                #_log.debug('Initializing a %s prior for %s with parameters: %s.', spec.name, fit_symbol, spec.parameters)
                rv_priors.append(spec.get_prior(param))
            elif hasattr(spec, "logpdf"):
                #_log.debug('Using a user-specified prior for %s.', fit_symbol)
                rv_priors.append(spec)
        return {'prior_rvs': rv_priors}


    def generate_sample(self, sample_n=1000, Train=True,use_mc_path=False,sigma_factor=0.1,range_factor=1,seed=42):
        # Parameters:
        # initial_guess (array): Initial guess of shape (D,)
        # sample_n (int): Number of total samples
        # alpha (float): Fraction of samples centered around A
        # sigma_factor (float): % of range used for Gaussian perturbation
        # range_factor (float): % of A used to estimate implicit bounds
        # seed (int, optional): Random seed for reproducibility
    
        # Returns:
        #     samples (array): Generated samples of shape (N, D)
        initial_guess = self.initial_guess
        np.set_printoptions(linewidth=sys.maxsize)

        progbar_width = 30
        np.random.seed(seed)
        Dim = len(initial_guess)
        if use_mc_path:
            mc_path = np.load('/Users/guannantang/Dropbox/Calphad/trace_reset.npy').reshape(88*2000,-1)
            np.random.shuffle(mc_path)
            extracted_mc_path = mc_path[:sample_n,:]
            final_samples = extracted_mc_path
        elif not Train:
            num_centered = sample_n
            sigma = sigma_factor * np.abs(initial_guess)
            centered_samples = np.random.normal(initial_guess, sigma, size=(num_centered, Dim))
            final_samples = centered_samples
        else:
            sampler = qmc.LatinHypercube(d=Dim)
            lhs_raw = sampler.random(n=sample_n)
            abs_center = np.abs(initial_guess)
            perturb_radius = abs_center * range_factor

            lower_bounds = initial_guess - perturb_radius
            upper_bounds = initial_guess + perturb_radius

            final_samples = qmc.scale(lhs_raw, lower_bounds, upper_bounds)
            # num_centered = sample_n
            # sigma = sigma_factor * np.abs(initial_guess)
            # centered_samples = np.random.normal(initial_guess, sigma, size=(num_centered, Dim))
            # final_samples = centered_samples
        
        # Combine samples
        final_samples[0,:] = initial_guess  # Ensure the first sample is the initial guess
        np.random.shuffle(final_samples)
        sample_n = len(final_samples)
        residual = []
        
        for idx in range(sample_n):
            # progress bar
            residual.append(self.eval_samples_sep(final_samples[idx])[0])
            n = int((progbar_width) * float(idx + 1) / sample_n)
            #_log.info("\r[%s%s] (%d of %d)\n", '#' * n, ' ' * (progbar_width - n), idx + 1, sample_n)
            sys.stdout.write(f"\r[{('#' * n).ljust(progbar_width)}] ({idx + 1} of {sample_n})")
            sys.stdout.flush()


        #pack the parameters with prediction results
        samples = np.hstack((final_samples, np.array(residual)))
        self.results = samples
        print('Initial Space filling complete. Data saved in csv file')
        np.savetxt("samples.csv", samples, delimiter=",")
        return

    def generate_query_index(self, initial_point = None, focus = False, sample_n=10000,sigma_factor=0.1):
    
        if initial_point is None:
            initial_guess = self.initial_guess
        else:
            initial_guess = initial_point        
        Dim = len(initial_guess) # Number of parameters

        # Estimate bounds as ± range_factor * A
        
        if focus:
            sigma = abs(sigma_factor * (initial_guess))
            final_samples = np.random.normal(initial_guess, sigma, size=(sample_n, Dim))
        
        else:
            range_factor=0.5
            min_bounds = initial_guess - range_factor * np.abs(initial_guess)  # Ensure no negative values if needed
            max_bounds = initial_guess + range_factor * np.abs(initial_guess)
            sampler = qmc.LatinHypercube(d=Dim)
            lhs_samples = sampler.random(sample_n)
            final_samples = min_bounds + lhs_samples * (max_bounds - min_bounds)
            
        np.random.shuffle(final_samples)
        
        return final_samples
        
    
    def eval_samples(self, params):
        """
        Calculate lnprob = lnlike + lnprior
        """
        ctx = self.ctx
        # Important to coerce to floats here because the values _must_ be floats if
        # they are used to update PhaseRecords directly
        params = np.asarray(params, dtype=np.float64)
        # lnprior
        # prior_rvs = ctx.get('prior_rvs', [rv_zero() for _ in range(params.size)])
        # lnprior_multivariate = [rv.logpdf(theta) for rv, theta in zip(prior_rvs, params)]
        # lnprior = np.sum(lnprior_multivariate)

        # if not np.isfinite(lnprior):
        #     return [-np.inf]*5

        # lnlike
        lnlike = 0.0
        sep_lnlike = [] #[lnprior]
        for residual_obj in ctx.get("residual_objs", []):
            likelihood = residual_obj.get_likelihood(params)
            lnlike += likelihood
            sep_lnlike.append(likelihood)
        lnlike = np.array(lnlike, dtype=np.float64)

        return sep_lnlike
    
    def eval_grouped_error(self, params):
        """
        Calculate lnprob = lnlike + lnprior
        """
        ctx = self.ctx
        # Important to coerce to floats here because the values _must_ be floats if
        # they are used to update PhaseRecords directly
        params = np.asarray(params, dtype=np.float64)
        # lnprior
        # prior_rvs = ctx.get('prior_rvs', [rv_zero() for _ in range(params.size)])
        # lnprior_multivariate = [rv.logpdf(theta) for rv, theta in zip(prior_rvs, params)]
        # lnprior = np.sum(lnprior_multivariate)

        # if not np.isfinite(lnprior):
        #     return [-np.inf]*5

        # lnlike

        sep_lnlike = [] #[lnprior]
        for residual_obj in ctx.get("residual_objs", []):
            likelihood = residual_obj.get_grouped_error(params)
            sep_lnlike.append(likelihood)
        return sep_lnlike
    
    def eval_samples_sep(self, params):
        """
        Calculate lnprob = lnlike + lnprior
        """
        params = np.asarray(params, dtype=np.float64)
        output = self.eval_grouped_error(params)
        #thermochemical data
        keys_1 = list(output[0].keys())
        values_eq_thermalchem = np.array([output[0][k][0][0] - output[0][k][0][1] for k in keys_1])

        #zpf data
        groups = defaultdict(list)
        for input_dict in output[1]:
            for key, val_list in input_dict.items():
                # Extract the comps part (the content within parentheses after 'comps:')
                match = re.search(r"comps:\s*(\([^)]+\))", key)
                if not match:
                    print("find no match")
                    continue  # skip keys that do not contain the comps part
                comps_str = match.group(1)
                
                # Remove the colon and anything inside curly braces.
                # For example, transform "LAVES_C15: {X_MG: 0.31592}" to "LAVES_C15"
                cleaned = re.sub(r":\s*\{[^}]*\}", "", comps_str)
                # Remove extra spaces around commas and parentheses:
                cleaned = re.sub(r"\s*,\s*", ",", cleaned)
                cleaned = re.sub(r"\(\s*", "(", cleaned)
                cleaned = re.sub(r"\s*\)", ")", cleaned)
                
                # Use the cleaned comps string as the group key
                groups[cleaned].extend(val_list)
        zpf_result = {}
        for comps, values in groups.items():
            avg = sum(values) / len(values) if values else None
            zpf_result[comps] = avg
        keys_2 = list(zpf_result.keys())
        values_zpf = np.array([zpf_result[k] for k in keys_2])
        return np.append(values_eq_thermalchem, values_zpf), keys_1 + keys_2


        
    