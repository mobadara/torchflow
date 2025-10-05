import os
import sys
from datetime import datetime

# Add project root to sys.path so autodoc can find the package
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

project = 'torchflow'
author = 'mobadara'
copyright = f"{datetime.now().year}, {author}"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML theme
html_theme = 'furo'
html_static_path = ['_static']

# Autodoc settings
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
