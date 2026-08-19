Explicit forward simulation with SLiM
=====================================

This page checks the simplified evolutionary prior on ten independently seeded
tree sequences produced by explicit forward simulations. It follows the tutorial style of the
`PySLiM spatial vignette <https://tskit.dev/pyslim/docs/stable/vignette_space.html>`_:
the workflow is presented as a sequence of simulation, inspection, conversion,
and analysis steps. PySLiM is not used here. The tree sequence is produced by
the ``slim`` executable, and the only Python-side tree-sequence operations are
provided by ``tskit`` and ``pygrgl``.

The SLiM model is adapted from the paper repository's
`burn-in script <https://github.com/hanbin973/param_arch_stab_paper/blob/main/codes/scripts/burnin.slim>`_
and `replicate script <https://github.com/hanbin973/param_arch_stab_paper/blob/main/codes/scripts/main.slim>`_.
In particular:

* mutation type ``m2`` carries normally distributed selection coefficients;
* ``mutationEffect(m2)`` returns one, so the coefficients are treated as trait
  effects rather than direct mutation fitness effects;
* stabilizing fitness is ``exp(-phenotype^2 / (2 * V_S))``;
* ``V_S`` is the fitness width and ``W_S = V_S / (2N)`` is its
  dimensionless diffusion-time rescaling;
* the simplified model uses the paper's ``rho^2 = 1`` construction, so
  ``beta_j = alpha_j``.

SLiM stores the m2 coefficients in each mutation's tree-sequence metadata.
The Python driver reads those coefficients from the tskit mutation table in
the same order as the mutations passed to GRGL. This keeps the effect vector
explicit and aligned after GRG conversion. It does not annotate the tree
sequence with PySLiM metadata.

Prerequisites
-------------

SLiM is an external executable rather than a Python dependency of this
project. Check that it is available before running the example:

.. code-block:: console

   slim -v

The project environment supplies ``tskit``, ``pygrgl``, and ``evo_lmm``. The
complete SLiM source is included as :download:`slim_simplified_prior.slim`.

The forward model
-----------------

The SLiM source is deliberately short enough to read alongside the Python
driver:

.. literalinclude:: slim_simplified_prior.slim
   :language: slim
   :linenos:

It simulates 1,000 diploid individuals for ``4N = 4,000`` generations,
records the tree sequence, and stores the m2 selection coefficients in its
mutation metadata. Here ``V_S = 2N = 2,000`` and therefore ``W_S = 1``.
The production paper
scripts separate burn-in and replicate runs; this documentation version
combines the essential steps so that one command produces the data consumed
by the GRG-backed fit.

SLiM to GRG to evo-lmm
----------------------

For each replicate, the Python driver runs SLiM, loads the resulting tree
sequence to report its size, simplifies the current leaf samples with ordinary
``tskit`` operations, converts that tree sequence with
``pygrgl.grg_from_trees``, computes sample allele frequencies, and applies the
raw GRG dosage operator to the metadata-derived ``alpha`` effects. It then adds
environmental residual noise and fits the simplified prior with the
BOLT-style matrix-free path. The simplification is needed because the full
SLiM recording also marks historical nodes as samples; it is not a PySLiM
operation.

.. literalinclude:: slim_forward_simplified.py
   :language: python
   :linenos:

Run it from the repository root with:

.. code-block:: console

   uv run python docs/tutorials/slim_forward_simplified.py

The fitted ``sigma_b2`` and ``tau`` should be interpreted as a finite-sample
check, not as an exact recovery guarantee. The forward population is short
relative to the paper's large burn-in studies, and each phenotype vector gives
limited information about a frequency-shape parameter. Always inspect the
reported ``FitDiagnostics`` for convergence, conditioning, trace error, and
boundary warnings.

Summary figure
--------------

The script produces a two-panel diagnostic: the left panel compares the
forward-simulated ``alpha_j^2`` values from the first replicate to two
different summaries on the minor-allele-frequency scale. The solid red curve
is the fitted parametric evolutionary prior
``E[beta_j^2 | x_j] = sigma_b2 / (1 + 2 * tau * x_j * (1 - x_j))``;
the dashed purple curve is a local linear regression of the simulated effect
squares. Folding to minor allele frequency removes the visually misleading
uptick near frequency one that appears when plotting the symmetric
``x * (1 - x)`` formula on the full allele-frequency interval. The right
panel shows box plots of the fitted variance components across all ten
replicates. The dotted line segment at each parameter position marks that
component's generating value; the right panel uses a linear y-axis.

.. plot:: tutorials/slim_forward_simplified.py
   :context: reset
   :alt: Forward-simulation effect spectrum and fitted simplified prior summary
