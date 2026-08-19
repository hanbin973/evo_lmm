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
