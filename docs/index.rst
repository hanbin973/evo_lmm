evo-lmm documentation
=====================

``evo-lmm`` implements the evolutionary random-effects linear mixed model
with raw diploid dosage data and GRGL-backed genotype representation graphs.
The documentation follows the overview, how-to, tutorial, and reference
organization used by `GRGL <https://grgl.readthedocs.io/en/latest/>`_ and
`GRAPP <https://grapp.readthedocs.io/en/latest/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview/index
   howto/index
   tutorials/index
   reference/index

The model keeps the focal-trait effect ``beta_j`` distinct from the latent
selected-trait effect ``alpha_j``. Frequencies are sample allele frequencies,
and the genotype boundary uses raw diploid dosage without an implicit
``1 / M`` kernel normalization.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
