Benchmark against GRAPP BOLT-LMM
================================

This page compares the GRG-backed evolutionary fit with GRAPP's
``bolt_lmm_inf`` implementation on the same explicit forward-simulation data.
It reuses the simulation source and configured parameters from
:doc:`slim_forward_simplified`:

* 2,000 diploid individuals and a ``4N = 8,000`` generation burn-in;
* ``L = 10^6`` bases and ``V_S = 2N = 4,000``, hence
  ``W_S = V_S/(2N) = 1``;
* ``sigma_a^2 = sigma_b^2 = 1`` and the simplified ``rho^2 = 1`` model; and
* residual variance ``sigma_e^2 = 0.4``.

The benchmark uses seed ``812`` and creates one phenotype from the full GRG.
For the fits, the same tree sequence is divided into two physical blocks.
This is an implementation detail required by GRAPP's leave-one-chromosome-out
calibration: with only one block, there is no chromosome left out of the
calibration set. Both blocks retain all 2,000 individuals and their GRGs are
created with coalescent counts enabled for GRAPP's ``XTX`` traversal.

The reported timings cover the fitting calls only. They include each method's
operator setup, variance-component estimation, and method-specific calibration
or trace work, but exclude the shared SLiM simulation, tskit simplification,
and GRG conversion. The exact seconds are machine-dependent and are printed
when the example is run directly. The checked-in figure was generated locally
so hosted documentation builds do not rerun this compute-heavy benchmark.
Both fits use 15 stochastic trials/probes and a ``5e-4`` CG tolerance, matching
GRAPP's defaults for 2,000 individuals. Both implementations batch probe
columns into GRG ``matmat`` traversals. These controls matter: the earlier
benchmark requested 64 probes and a ``1e-8`` tolerance only from evo-lmm, which
made the wall-clock comparison measure a substantially larger numerical-work
budget.

Cumulative genic variance
-------------------------

For a variant with minor allele frequency ``x``, the genic contribution is
``2*x*(1-x)*E[beta^2 | x]``. The single panel below follows the layout and MAF
bins of Figure A11 in ``stab1_genetics_template.pdf``. It shows the realized
SLiM contribution, the configured evolutionary prior, evo-lmm's fitted prior,
and GRAPP BOLT-LMM's global standardized-GRM allocation.

The GRAPP curve is intentionally described as a global-GRM allocation. BOLT-LMM
assigns the fitted ``sigma_g2`` equally across its ``M`` eligible standardized
markers (``sigma_g2/M`` per marker). In raw-dosage effect notation this is
``E[beta^2 | x] = sigma_g2/(M*2*x*(1-x))``; after multiplying by
``2*x*(1-x)``, each marker contributes ``sigma_g2/M``. Its cumulative curve is
therefore a neutral frequency-independent reference. evo-lmm instead uses the
frequency-dependent simplified prior
``sigma_b^2/(1 + 2*tau*x*(1-x))``.

.. literalinclude:: bolt_benchmark.py
   :language: python
   :linenos:

Run the benchmark from the repository root with:

.. code-block:: console

   uv run python docs/tutorials/bolt_benchmark.py

The figure is pre-generated locally so ReadTheDocs does not rerun SLiM, GRG
conversion, or either fitted model. Regenerate it with
``uv run python docs/generate_figures.py``.

.. image:: ../_static/generated/bolt_benchmark.png
   :alt: Cumulative genic variance benchmark for evo-lmm and GRAPP BOLT-LMM
