"""Sphinx configuration for generated code documentation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
project = "Предиктор транспорта"
author = "Transport Predictor contributors"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
html_theme = "alabaster"
language = "ru"
exclude_patterns = ["_build"]
autodoc_member_order = "bysource"
autodoc_default_options = {"undoc-members": True}
html_baseurl = "https://mdvdv177.github.io/hackaton_v2/"
