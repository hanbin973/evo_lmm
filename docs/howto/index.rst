How-to guides
=============

Installation
------------

Initialize the GRGL and GRAPP submodules, then install the project with
``uv``:

.. code-block:: console

   git submodule update --init --recursive
   uv sync
   uv run python -c "import evo_lmm, pygrgl, grapp; print('environment ready')"

The native GRGL extension requires CMake and a C++17-capable compiler. The
public API accepts either dense ``(n_individuals, n_variants)`` dosage arrays
or GRGs. For GRGs, pass sample allele frequencies explicitly when they were
computed on a filtered sample; otherwise they are extracted through GRAPP.

See the :doc:`../reference/public_api` for the complete public API.

Calibrated association testing
------------------------------

After a fit, association statistics are computed against the fitted
leave-one-chromosome-out evolutionary covariance
``V_loco = sigma_b2 * (K_evo,loco + delta * I)``, while the tested genotype
columns keep the independent BOLT normalisation. ``association`` performs the
prospective/retrospective moment matching itself:

.. code-block:: python

   fit = fit_evolutionary_bolt_lmm(chrom_grgs, phenotype, covariates=covariates)
   results = association(fit, calibration_variants=30, seed=0)
   print(association_summary(results))

``results`` holds one :class:`~evo_lmm.AssociationResult` per chromosome, with
``beta`` and ``se`` in raw diploid-dosage effect units — ``sigma_b2`` is never
reinterpreted as a standardized genetic variance. A single-variant
linear-regression chi-square is reported alongside the mixed-model statistic so
inflation can be compared directly, and ``association_summary`` reports the mean
chi-square and ``lambda_GC`` for both.

Reuse a calibration when several phenotype transformations share one fit, or
inspect it directly:

.. code-block:: python

   calibration = calibrate_association(fit, count=30, seed=0)
   print(calibration.factor, calibration.std, calibration.inverse_scale)
   results = association(fit, calibration=calibration)

Pass ``calibrate=False`` for the uncalibrated LOCO statistic, or
``use_loco=False`` for the in-sample statistic. Both are diagnostics only: they
are deflated or inflated by construction and are not calibrated test
statistics.
