"""Sphinx configuration for generated code documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
project = "Предиктор транспорта"
author = "Transport Predictor contributors"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
html_theme = "alabaster"
language = "ru"
exclude_patterns = ["_build"]
autodoc_member_order = "bysource"
