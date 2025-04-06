# ---
# jupyter:
#   jupytext:
#     formats: md:myst,ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: revamp
#     language: python
#     name: revamp
# ---

# %% [markdown]
# # Chord Profiles

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2
import os

import dimcat as dc
import ms3
import pandas as pd
from dimcat import resources
from dimcat.plotting import write_image
from git import Repo
from matplotlib import pyplot as plt

import utils

plt.rcParams["figure.dpi"] = 300

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

# %%
RESULTS_PATH = os.path.expanduser("~/git/diss/32_profiles/figs")
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename, extension=None):
    if extension:
        extension = "." + extension.lstrip(".")
    else:
        extension = utils.DEFAULT_OUTPUT_FORMAT
    return os.path.join(RESULTS_PATH, f"{filename}{extension}")


def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    if not any(key in kwargs for key in ("height", "width")):
        kwargs["width"] = 1280
        kwargs["height"] = 720
    write_image(fig, filename, directory, **kwargs)


# %% tags=["hide-input"]
package_path = utils.resolve_dir(
    "~/distant_listening_corpus/distant_listening_corpus.datapackage.json"
)
repo = Repo(os.path.dirname(package_path))
utils.print_heading("Data and software versions")
print(f"Data repo '{utils.get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
D = dc.Dataset.from_package(package_path)
D

# %%
chord_slices = utils.get_sliced_notes(D)
chord_slices.head(5)

# %% [markdown]
# ## Document frequencies of chord features

# %%
utils.compare_corpus_frequencies(
    chord_slices,
    [
        "chord_reduced_and_mode",
        ["effective_localkey_is_minor", "numeral"],
        "root",
        "root_fifths_over_global_tonic",
    ],
)

# %% [markdown]
# ## Create chord-tone profiles for multiple chord features
#
# Tokens are `(feature, ..., chord_tone)` tuples.

# %%
chord_reduced: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["chord_reduced_and_mode", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {chord_reduced.shape}")

# %%
numerals: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["effective_localkey_is_minor", "numeral", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {numerals.shape}")
utils.replace_boolean_column_level_with_mode(numerals)

# %%
roots: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {roots.shape}")

# %%
root_fifths_over_global_tonic = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root_fifths_over_global_tonic", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {root_fifths_over_global_tonic.shape}")

# %% [markdown]
# ### Document frequencies of the tokens

# %%
fig = utils.plot_document_frequency(chord_reduced)
save_figure_as(fig, "document_frequency_of_chord_tones")
fig

# %%
utils.plot_document_frequency(numerals, info="numerals")

# %%
utils.plot_document_frequency(roots, info="roots")

# %%
utils.plot_document_frequency(
    root_fifths_over_global_tonic, info="root relative to global tonic"
)

# %% [markdown]
# ## Principal Component Analyses

# %%
# chord_reduced.query("piece in ['op03n12a', 'op03n12b']").dropna(axis=1, how='all')

# %%
metadata = D.get_metadata()
CORPUS_YEARS = utils.corpus_mean_composition_years(metadata)
PIECE_YEARS = metadata.get_composition_years().rename("mean_composition_year")
utils.plot_pca(
    chord_reduced.relative,
    info="chord-tone profiles of reduced chords",
    color=PIECE_YEARS,
)

# %%
utils.plot_pca(
    chord_reduced.combine_results("corpus").relative,
    info="chord-tone profiles of reduced chords",
    color=CORPUS_YEARS,
    size=5,
)

# %%
utils.plot_pca(
    numerals.relative, info="numeral profiles of numerals", color=PIECE_YEARS
)

# %%
utils.plot_pca(
    numerals.combine_results("corpus").relative,
    info="chord-tone profiles of numerals",
    color=CORPUS_YEARS,
    size=5,
)

# %%
utils.plot_pca(
    roots.relative, info="root profiles of chord roots (local)", color=PIECE_YEARS
)

# %%
utils.plot_pca(
    roots.combine_results("corpus").relative,
    info="chord-tone profiles of chord roots (local)",
    color=CORPUS_YEARS,
    size=5,
)

# %%
utils.plot_pca(
    root_fifths_over_global_tonic.relative,
    info="root profiles of chord roots (global)",
    color=PIECE_YEARS,
)

# %%
utils.plot_pca(
    root_fifths_over_global_tonic.combine_results("corpus").relative,
    info="chord-tone profiles of chord roots (global)",
    color=CORPUS_YEARS,
    size=5,
)