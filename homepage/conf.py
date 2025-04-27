# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Richard Wagner – Overtures"
copyright = '2025, Johannes Hentschel'
author = 'Johannes Hentschel'
release = 'v2.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb", # rendering Jupyter notebooks
    "jupyter_sphinx", # rendering interactive Plotly in notebooks
]

templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**DLC_overview*',
    '**README*',
    '**accents*',
    # '**annotations*',
    '**bass_degrees*',
    # '**cadences*',
    '**chopin_profiles*',
    '**chord_profiles*',
    '**chord_tone_profiles*',
    '**chord_tone_profiles_classification*',
    '**chord_tone_profiles_inspection*',
    '**chromatic_bass*',
    '**cross_entropy*',
    '**couperin_study*'
    '**dft*',
    '**harmonies*',
    '**information_gain*',
    '**ismir*',
    '**keys*',
    '**line_of_fifths*',
    '**modulations*',
    '**modulations_adapted_for_mozart*',
    #'**notes_stats*',
    # '**overview*',
    '**phrase_alignment*',
    '**phrase_diatonics*',
    '**phrase_excerpts*',
    '**phrase_grams*',
    '**phrase_profiles*',
    '**phrase_sankey_draft*',
    '**phrase_stages*',
    '**phrase_unalignment*',
    '**phrases*',
    '**scale_degrees*'
    '**specific*'
    ]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

# -- MyST Notebook configuration-----------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html

nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_execution_allow_errors = False
nb_execution_show_tb = True
# toggle text:
nb_code_prompt_show = "Show {type}"
nb_code_prompt_hide = "Hide {type}"
nb_execution_excludepatterns = exclude_patterns