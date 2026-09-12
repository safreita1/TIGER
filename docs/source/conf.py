"""Sphinx configuration for the TIGER documentation."""
project = 'TIGER'
author = 'Scott Freitas'
release = '0.7.0'
version = release
extensions = []
root_doc = 'index'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {'collapse_navigation': False, 'navigation_depth': 3, 'includehidden': True}
templates_path = ['_templates']
html_static_path = ['_static']
html_extra_path = ['_extra']
html_css_files = ['content.css', 'guide.css']
html_js_files = ['search-index.js', 'site-controls.js']
exclude_patterns = ['_extra', '_static', '_templates']
nitpicky = True
html_show_sourcelink = False
