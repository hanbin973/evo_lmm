"""Shared helpers for the GRM-vector benchmark.

The benchmark compares two ways of computing the (un-normalized, un-centred)
GRM-vector product ``G G^T v``:

* ``tskit``: :meth:`tskit.TreeSequence.genetic_relatedness_vector` in ``branch``
  mode, which walks the tree sequence and never forms ``G``.  ``site`` mode is
  not implemented in tskit yet, so this measures the *expected* GRM-vector
  product under the mutation rate (branch lengths scaled by ``mu``) rather than
  the realized one.
* ``grapp``: a GRG-backed :class:`scipy.sparse.linalg.LinearOperator` on the raw
  (non-normalized, non-standardized) genotype matrix ``X``, applied twice.

Simulation is haploid (``ploidy=1``) so that the tskit sample axis and the GRG
sample axis are the same object, and the GRAPP operator is built with
``haploid=True`` so both methods act on the same ``{0, 1}`` matrix.
"""

from __future__ import annotations

import numpy as np


def simulate_tree_sequence(n_samples, seq_len, ne, rho, mu, seed=42):
    """Simulate ancestry and overlay mutations, returning a tree sequence."""
    import msprime

    ts = msprime.sim_ancestry(
        samples=n_samples,
        ploidy=1,
        population_size=ne,
        recombination_rate=rho,
        sequence_length=seq_len,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed + 1, discrete_genome=True)
    return ts


def trees_to_grg(trees_path, grg_path):
    """Convert a ``.trees`` file to a GRG on disk (bi-allelic mutations)."""
    import pygrgl

    grg = pygrgl.grg_from_trees(str(trees_path), binary_mutations=True)
    pygrgl.save_grg(grg, str(grg_path))


def load_grg(grg_path):
    """Load an immutable GRG with up-edges, as the ``X`` operator requires."""
    import pygrgl

    return pygrgl.load_immutable_grg(str(grg_path), load_up_edges=True)


def make_grapp_operator(grg):
    """Return the raw (non-normalized) ``X`` operator for ``grg``."""
    import pygrgl
    from grapp.linalg.ops_scipy import SciPyXOperator

    return SciPyXOperator(
        grg,
        pygrgl.TraversalDirection.UP,
        dtype=np.float64,
        haploid=True,
    )


def compute_grapp_Gv(operator, v):
    """Compute ``G G^T v`` with two raw GRG matrix products.

    ``operator`` is the non-normalized ``X`` (``N x M``); no allele-frequency
    standardization and no ``1 / M`` scaling is applied, so the result matches
    tskit's ``span_normalise=False, centre=False`` convention.
    """
    return operator @ (operator.T @ v)


def compute_tskit_Gv(ts, v, mu=None):
    """Compute the branch-mode relatedness vector, optionally scaled by ``mu``.

    ``mode="branch"`` returns the shared-branch-length form of ``G G^T v``;
    multiplying by the mutation rate puts it on the same scale as the realized
    product returned by :func:`compute_grapp_Gv`.  Pass ``mu=None`` inside timing
    loops to measure the traversal alone.
    """
    y = ts.genetic_relatedness_vector(
        v, mode="branch", span_normalise=False, centre=False
    )
    return y if mu is None else y * mu
