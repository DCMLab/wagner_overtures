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
# # Chord profiles for phrases in the DLC
#
# ToDo: Inspect the 7 phrases that have 0-duration when drop_duplicated_ultima_rows=True:

# %% [raw]
# phrase_data = phrase_annotations.get_phrase_data(["chord_and_mode", "duration_qb"], components="phrase")
# phrase_data.groupby(["corpus", "piece", "phrase_id"]).duration_qb.filter(lambda S: S.sum() == 0)

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# # %load_ext autoreload
# # %autoreload 2

import os
from typing import List, Tuple

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
import plotly.express as px
from dimcat import resources
from dimcat.plotting import write_image
from dimcat.utils import get_middle_composition_year
from git import Repo

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

# %%
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "phrases"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(
    filename: str,
    extension=None,
    path=RESULTS_PATH,
) -> str:
    return utils.make_output_path(
        filename,
        extension=extension,
        path=path,
    )


def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
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
chronological_corpus_names = D.get_metadata().get_corpus_names(func=None)
D

# %%
phrase_annotations: resources.PhraseAnnotations = D.get_feature("PhraseAnnotations")
phrase_annotations


# %%
def make_phrase_data(
    phrase_annotations, columns, components=("body", "codetta"), **kwargs
):
    phrase_data = phrase_annotations.get_phrase_data(
        columns, components=components, **kwargs
    )
    return phrase_data


def make_phrase_bigram_table(
    phrase_annotations: resources.PhraseAnnotations, columns: str | List[str]
) -> resources.NgramTable:
    phrase_data = make_phrase_data(
        phrase_annotations,
        columns,
        components=(
            "body",
            "codetta",
        ),  # ideally using "phrase" but see 0-duration-ToDo at the top
        drop_levels="phrase_component",  # otherwise, no bigrams spanning body and codetta
    )
    phrase_bgt = phrase_data.apply_step("BigramAnalyzer")
    return phrase_bgt


def prepare_data(
    phrase_annotations: pd.DataFrame, features: str | List[str], smooth=1e-20
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if isinstance(features, str):
        features = [features]
    if "duration_qb" in features:
        columns = features
    else:
        columns = features + ["duration_qb"]
    phrase_data = make_phrase_data(phrase_annotations, columns)
    phrase_data_df = phrase_data.df.dropna()
    groupby = ["corpus", "piece", "phrase_id"]
    return utils.prepare_tf_idf_data(
        phrase_data_df,
        index=groupby,
        columns=features,
        smooth=smooth,
    )


# %%
phrase_data = make_phrase_data(phrase_annotations, ["localkey_mode", "duration_qb"])
PHRASE_LENGTH = (
    phrase_data.groupby(["corpus", "piece", "phrase_id"])
    .duration_qb.sum()
    .rename("phrase_duration")
)
mode_proportions = phrase_data.groupby(
    ["corpus", "piece", "phrase_id", "localkey_mode"]
).duration_qb.sum()
PHRASE_MODE_BINARY = (
    mode_proportions.groupby(["corpus", "piece", "phrase_id"])
    .apply(lambda S: S.idxmax()[-1])
    .rename("localkey_mode")
)
PHRASE_MODE_TERNARY = (
    phrase_data.groupby(["corpus", "piece", "phrase_id"])
    .localkey_mode.unique()
    .map(lambda arr: "both" if len(arr) > 1 else arr[0])
)
PHRASE_COMPOSITION_YEAR = (
    get_middle_composition_year(D.get_metadata())
    .reindex(PHRASE_MODE_BINARY.index)
    .rename("year")
)
SCATTER_PLOT_SETTINGS = dict(
    color=PHRASE_MODE_TERNARY,
    color_discrete_map=dict(major="blue", minor="red", both="green"),
)

# %% [markdown]
# ## Full chord symbols

# %%
unigram_distribution, f, tf, df, idf = prepare_data(
    phrase_annotations, "chord_and_mode"
)
unigram_distribution

# %% [markdown]
# ### Full chord frequency matrix
#
# Chord symbols carry their mode information, so it is to expected that modes be clearly separated.

# %%
utils.plot_pca(tf, "chord frequency matrix", **SCATTER_PLOT_SETTINGS)

# %% [markdown]
# ### Phrases entirely in major

# %%
pl_log = np.log2(PHRASE_LENGTH)
PL_NORM = pl_log.add(-pl_log.min()).div(pl_log.max() - pl_log.min())
px.histogram(PL_NORM)

# %%
mode_tf = {group: df for group, df in tf.groupby(PHRASE_MODE_TERNARY)}
utils.plot_pca(
    mode_tf["major"],
    "chord frequency matrix for phrases in major",
    color=PHRASE_COMPOSITION_YEAR,
    size=None,
)

# %% [markdown]
# #### Inspection
#
# Inspecting the phrases that make up one of the edges of the tetrahedron (the one from `(-0.24, -0.30, 0.89)` to
# `(-0.47, 0.81, 0.07)`) reveals phrases that have only few and only main chords, i.e. those that are among the highest
# coefficients.

# %%
line = [9556, 14611, 4256, 3023, 5734, 13050, 7073, 3476, 14258]

# %%
phrase_annotations.query("phrase_id in @line").groupby("phrase_id").chord.unique()

# %%
mode_f = {group: df for group, df in f.groupby(PHRASE_MODE_TERNARY)}
utils.plot_pca(
    mode_f["major"],
    "chord proportion matrix for phrases in major",
    color=PHRASE_COMPOSITION_YEAR,
    size=None,
)

# %%
utils.plot_pca(
    mode_tf["minor"],
    "chord frequency matrix for phrases in minor",
    color=PHRASE_COMPOSITION_YEAR,
)

# %%
utils.plot_pca(
    mode_f["minor"],
    "chord proportions matrix for phrases in minor",
    color=PHRASE_COMPOSITION_YEAR,
)

# %% [markdown]
# ### PCA of tf-idf

# %%
utils.plot_pca(tf.mul(idf), "tf-idf matrix", **SCATTER_PLOT_SETTINGS)

# %% [markdown]
# ### For comparison: PCA of t-idf (absolute chord durations weighted by idf)
#
# PCA consistently explains a multiple of the variance for f-idf compared to tf-idf (normalized chord weights)

# %%
utils.plot_pca(f.fillna(0.0).mul(idf), "f-idf matrix", **SCATTER_PLOT_SETTINGS)

# %% [markdown]
# ## Reduced chords (without suspensions, additions, alterations)

# %%
unigram_distribution, f, tf, df, idf = prepare_data(
    phrase_annotations, "chord_reduced_and_mode"
)
unigram_distribution

# %%
utils.plot_pca(tf, "(reduced) chord frequency matrix", **SCATTER_PLOT_SETTINGS)

# %%
utils.plot_pca(f.mul(idf), "f-idf matrix (reduced chords)", **SCATTER_PLOT_SETTINGS)

# %% [markdown]
# ## Only root, regardless of chord type or inversion
#
# PCA plot has straight lines. Th

# %%
unigram_distribution, f, tf, df, idf = prepare_data(phrase_annotations, "root")
unigram_distribution

# %%
utils.plot_pca(tf, "root frequency matrix", **SCATTER_PLOT_SETTINGS)

# %% [markdown]
# ## Grid search on variance explained by PCA components

# %% [raw]
# def do_pca_grid_search(
#     data: pd.DataFrame,
#     features: npt.ArrayLike,
#     max_components: int = 10,
# ):
#     max_features = len(features)
#     n_columns = max_components if max_components > 0 else ceil(max_features / 2)
#     grid_search = np.zeros((max_features, n_columns))
#     for n_features in range(1, max_features + 1):
#         print(n_features, end=" ")
#         selected_features = features[:n_features]
#         selected_data = data.loc(axis=1)[selected_features]
#         if max_components > 0:
#             up_to = min(max_components, n_features)
#         else:
#             up_to = ceil(n_features / 2)
#         for n_components in range(1, up_to + 1):
#             pca = PCA(n_components)
#             _ = pca.fit_transform(selected_data)
#             variance = pca.explained_variance_ratio_.sum()
#             grid_search[n_features - 1, n_components - 1] = variance
#             print(f"{variance:.1%}", end=" ")
#         print()
#     result = pd.DataFrame(
#         grid_search,
#         index=pd.RangeIndex(1, max_features + 1, name="features"),
#         columns=pd.RangeIndex(1, n_columns + 1, name="components"),
#     )
#     return result
#
#
# grid_search_by_occurrence = do_pca_grid_search(tf, df.index[:100])

# %% [raw]
# grid_search_by_duration = do_pca_grid_search(tf, unigram_distribution.index[:100])

# %% [raw]
# grid_search_by_duration - grid_search_by_occurrence