Benchmark against GRAPP BOLT-LMM
================================

This tutorial compares the GRG-backed evolutionary model with GRAPP's
``bolt_lmm_inf`` :cite:p:`DeHaasAdonizioPanWei2026` on the forward-simulated populations from
:doc:`slim_forward_simplified`. It asks whether the evolutionary model
reproduces the distribution of genic variance across allele frequencies, and
how much fitting time it requires on the same GRG-backed data.

The shared simulation uses:

* 2,000 diploid individuals and a ``10N = 20,000`` generation burn-in;
* ``L = 10^6`` bases and ``V_S = 2N = 4,000``, hence
  ``W_S = V_S/(2N) = 1``;
* ``sigma_a^2 = sigma_b^2 = 1`` and the simplified ``rho^2 = 1`` model; and
* residual variance ``sigma_e^2 = 0.4``.

The ten deterministic replicates (seeds ``812``--``821``) are shared between
the two tutorials. Each has its own phenotype and full GRG. For fitting, the
tree sequence is split into two blocks so that GRAPP's leave-one-chromosome-out
calibration has a block to leave out. Both blocks retain all 2,000 individuals
and use the Genotype Representation Graph data structure
:cite:p:`DeHaasPanWei2025`, including coalescent counts for GRAPP's ``XTX``
traversal.

Timings cover fitting only: operator setup, variance-component estimation, and
each method's calibration or trace work. They exclude the shared SLiM run,
tskit simplification, and GRG conversion. Both methods use 15 stochastic
trials/probes and a ``5e-4`` CG tolerance. Times will vary by machine; the
example prints the values observed locally.

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
Each point is the mean across the ten forward replicates, and error bars show
the sample standard deviation. The runtime annotation uses the same summary.

.. dropdown:: Show the benchmark implementation
   :color: light

   .. literalinclude:: bolt_benchmark.py
      :language: python
      :linenos:

Generate the ten forward-simulation replicates and their GRGs:

.. code-block:: console

   uv run python docs/tutorials/prepare_bolt_benchmark.py

Then rerun the ten-replicate fitting comparison without repeating SLiM:

.. code-block:: console

   uv run python docs/tutorials/fit_bolt_benchmark.py

The fit-only script reports each replicate together with the mean and sample
standard deviation of both runtimes.

The current ten-replicate fit run reports:

.. list-table::
   :header-rows: 1

   * - Method
     - Mean fit seconds
     - Sample SD seconds
   * - evo-lmm (warm Hutchinson, 15 vectors)
     - 27.24
     - 6.93
   * - GRAPP BOLT-LMM
     - 10.67
     - 0.96

.. image:: ../_static/generated/bolt_benchmark.png
   :alt: Cumulative genic variance benchmark for evo-lmm and GRAPP BOLT-LMM
