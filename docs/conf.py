"""Sphinx configuration for evo-lmm."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "evo-lmm"
copyright = "2026, evo-lmm contributors"
author = "evo-lmm contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "external_links": [
        {
            "url": "https://grgl.readthedocs.io/en/latest/",
            "name": "GRGL documentation",
        },
        {
            "url": "https://grapp.readthedocs.io/en/latest/",
            "name": "GRAPP documentation",
        },
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/hanbin973/fast_lmm",
            "icon": "fa-brands fa-github",
        },
    ],
}

plot_html_show_source_link = False
