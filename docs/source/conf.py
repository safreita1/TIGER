# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
"""
To build the docs, make sure these 3 libraries are installed
1.pip install sphinx
2. pip install sphinxcontrib-bibtex
3. pip install sphinx-rtd-theme
Then, inside the "docs" folder, run "make html".
"""
import os
import sys
import datetime
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'TIGER'
author = 'Scott Freitas'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

# The full version, including alpha/beta/rc tags
release = '0.1.3'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinxcontrib.bibtex']
bibtex_bibfiles = ['refs.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'


html_theme_options = {
   'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
   'analytics_anonymize_ip': False,
   'logo_only': True,
   'display_version': True,
   'prev_next_buttons_location': 'bottom',
   'style_external_links': False,
   'vcs_pageview_mode': '',
   'style_nav_header_background': 'white',
   # Toc options
   'collapse_navigation': False,
   'sticky_navigation': True,
   'navigation_depth': 4,
   'includehidden': True,
   'titles_only': False
}   

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_logo = '../../images/tiger-logo.jpg'
html_static_path = ['_static']
