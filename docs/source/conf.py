# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GDF'
copyright = '2025, Srijan Bharati Das, Michael Terres'
author = 'Srijan Bharati Das, Michael Terres'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]
# extensions += ['numpydoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'

html_theme_options = {
    "repository_url": "https://github.com/srijaniiserprinceton/gdf",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

# html_sidebars = {
#     "**": ["sidebar-logo.html", "sbt-sidebar-nav.html"]
# }

html_title = "gdf"
html_static_path = ['_static']

import sys, os
sys.path.insert(0, os.path.abspath('../../gdf/src'))
