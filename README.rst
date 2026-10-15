=========
llm_espei
=========

**A fork of ESPEI that serves as the Bayesian CALPHAD engine for Auto-DDM.**

Auto-DDM (Autonomous Data-Driven Modeling) is an LLM-agentic, multi-objective
workflow for weighting heterogeneous datasets in data-driven modeling.
Its CALPHAD example calls this fork once per candidate weight vector to
calibrate a Cu–Mg thermodynamic database and return 22 per-dataset errors.

- Paper: G. Tang and N. Paulson, *An LLM-Agentic Workflow for Data-Driven
  Modeling: From Image Reconstruction to Thermodynamic Modeling*, Research
  Square preprint (2026). doi:`10.21203/rs.3.rs-8574739/v2 <https://doi.org/10.21203/rs.3.rs-8574739/v2>`_
- Workflow code: `guannant/Auto_DataDrivenModeling <https://github.com/guannant/Auto_DataDrivenModeling>`_
- Upstream project: `PhasesResearchLab/ESPEI <https://github.com/PhasesResearchLab/ESPEI>`_ (`espei.org <https://espei.org>`_)

This fork is based on ESPEI 0.9.0. It is not a drop-in replacement for ESPEI:
the MCMC path is specialized for the Auto-DDM Cu–Mg case and is not meant for
general CALPHAD assessments.


How this fork differs from ESPEI
================================

- **Surrogate likelihood.** During MCMC, ``EmceeOptimizer.predict`` no longer
  evaluates PyCalphad. It loads a residual MLP surrogate
  (``final_model_full_data.pt``), its input/output scalers (``scalerX.pkl``,
  ``scalerY.pkl``) and a dataset-weight file (``weights.json``) from the folder
  named by the ``AUTODDM_SURROGATE_DIR`` environment variable. The surrogate
  maps the 22 model parameters to 22 per-dataset errors, and the log-likelihood
  is ``sum(norm.logpdf(error, loc=0, scale=weight))``. Dataset weights
  therefore act as per-dataset standard deviations. A full ESPEI run that
  takes hours on an HPC node becomes a run of tens of seconds on a laptop.
- **Per-dataset error evaluation.** ``espei.optimizers.opt_generate_samples.DataGenerator``
  evaluates a database against the loaded datasets with PyCalphad and returns
  one error per dataset (``eval_samples_sep``): the mean predicted-minus-reference
  value for each thermochemical dataset, and the mean driving force for each
  phase-region group in the ZPF data. Auto-DDM uses this vector as its
  objectives. ``DataGenerator.generate_sample`` draws Latin-hypercube samples
  around the initial parameters and writes ``samples.csv`` for surrogate
  training. ``get_grouped_error`` methods were added to ``ZPFResidual`` and
  ``FixedConfigurationPropertyResidual`` to support this.
- **Residual set.** Only the non-equilibrium thermochemical and ZPF residuals
  are registered. ``ActivityResidual`` and ``EquilibriumPropertyResidual`` are
  disabled.
- **Thermochemical error.** For ``SM_MIX`` data, the ideal mixing entropy is
  subtracted before the difference is taken.
- **Input file.** The ``mcmc`` section requires a new integer key
  ``LHS_points`` (reserved for surrogate data generation). ``cores`` is set to
  ``22 * chains_per_parameter / 2`` and is not read from the input file.
  ``run_espei`` returns ``None`` instead of the database and sampler.
- **Fixed problem size.** The surrogate, chain setup and error grouping assume
  22 fitting parameters and 22 datasets, with the first 10 outputs being
  thermochemical and the remaining 12 being ZPF. Changing the system requires
  retraining the surrogate and adjusting these constants.
- **Smaller changes.** ``_extract_pot_conds`` accepts one-element arrays;
  ``refdata`` uses ``importlib.metadata`` instead of ``pkg_resources``;
  ``espei.plot.plot_all_interaction`` plots interaction data at arbitrary
  temperature; ``torch``, ``joblib`` and ``pandas`` are added as dependencies;
  the dask scheduler logs are quieter.

Everything else (parameter generation, datasets, phase models, priors, plotting)
is unchanged from ESPEI 0.9.0.


Role in the Auto-DDM loop
=========================

For each candidate weight vector Auto-DDM:

1. writes the weights to ``$AUTODDM_SURROGATE_DIR/weights.json``;
2. writes an ESPEI input file and runs ``espei --input run_mcmc.yaml``, which
   produces a calibrated TDB plus trace and log-probability files;
3. loads the TDB with ``DataGenerator`` and calls ``eval_samples_sep`` to get
   the 22 reconstruction errors that drive the next generation.


Installing
==========

Python 3.12 is recommended. Install into a virtual environment:

.. code-block:: bash

   python -m venv auto-ddm-env
   source auto-ddm-env/bin/activate
   pip install -U pip
   pip install git+https://github.com/guannant/llm_espei.git

Auto-DDM's ``requirements.txt`` pins the exact commit it was tested with.


Running
=======

Set the surrogate folder, then run ESPEI as usual.
The folder must contain ``final_model_full_data.pt``, ``scalerX.pkl``,
``scalerY.pkl`` and ``weights.json``. A trained Cu–Mg surrogate ships with
Auto-DDM in ``examples/CALPHAD/Pytorch_MLP_CV/``.

.. code-block:: bash

   export AUTODDM_SURROGATE_DIR=/path/to/Auto_DataDrivenModeling/examples/CALPHAD/Pytorch_MLP_CV
   espei --input run_mcmc.yaml

A minimal ``run_mcmc.yaml``:

.. code-block:: yaml

   system:
     phase_models: phase_models.json
     datasets: input-data
   mcmc:
     input_db: Cu-Mg-generated.tdb
     iterations: 800
     LHS_points: 4500
     chains_per_parameter: 4
     chain_std_deviation: 0.1
     prior:
       name: normal
       loc_relative: 1.0
       scale_relative: 1.0
     scheduler: null
   output:
     output_db: calibrated.tdb
     tracefile: trace.npy
     probfile: lnprob.npy
     logfile: espei_log.txt
     verbosity: 2

``weights.json`` maps dataset names to weights. Thermochemical datasets are
keyed ``<PHASE>_<PROPERTY>`` and ZPF datasets by their phase tuple:

.. code-block:: json

   {
     "LIQUID_HM_MIX": 500,
     "LIQUID_SM_MIX": 0.2,
     "(LIQUID,FCC_A1)": 1000
   }

To compute the per-dataset errors of a calibrated database from Python:

.. code-block:: python

   import json
   from pycalphad import Database
   from espei.datasets import load_datasets, recursive_glob
   from espei.optimizers.opt_generate_samples import DataGenerator

   with open("phase_models.json") as f:
       phase_models = json.load(f)
   dbf = Database("calibrated.tdb")
   datasets = load_datasets(sorted(recursive_glob("input-data", "*.json")))
   gen = DataGenerator(dbf, datasets, phase_models)
   errors, names = gen.eval_samples_sep(gen.initial_guess)


Citing
======

If you use this fork, please cite the Auto-DDM paper above and ESPEI:

B. Bocklund, R. Otis, A. Egorov, A. Obaied, I. Roslyakova, Z.-K. Liu,
ESPEI for efficient thermodynamic database development, modification, and
uncertainty quantification: application to Cu–Mg, MRS Commun. 9 (2019) 618–627.
doi:`10.1557/mrc.2019.59 <https://doi.org/10.1557/mrc.2019.59>`_


License
=======

MIT, inherited from ESPEI. See ``LICENSE``.
ESPEI is Copyright (c) 2015–2019 Richard Otis, Brandon Bocklund and the
Materials Genome Foundation.
