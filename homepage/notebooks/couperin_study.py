# ---
# jupyter:
#   jupytext:
#     formats: md:myst,ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: revamp
#     language: python
#     name: revamp
# ---

# %% [markdown]
# # New

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2

import itertools
import os
from functools import cache
from typing import List, Literal, Optional, Tuple

import ms3
import pandas as pd
import plotly.express as px
from dimcat import Pipeline, plotting

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

# %% mystnb={"code_prompt_hide": "Hide helpers", "code_prompt_show": "Show helpers"} tags=["hide-cell"]
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "couperin_study"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(
    filename: str,
    extension=None,
    path=RESULTS_PATH,
) -> str:
    return utils.make_output_path(filename=filename, extension=extension, path=path)


def save_figure_as(
    fig, filename, formats=("png", "pdf"), directory=RESULTS_PATH, **kwargs
):
    if formats is not None:
        for fmt in formats:
            plotting.write_image(fig, filename, directory, format=fmt, **kwargs)
    else:
        plotting.write_image(fig, filename, directory, **kwargs)


def style_plotly(
    fig,
    save_as=None,
    xaxes: Optional[dict] = None,
    yaxes: Optional[dict] = None,
    match_facet_yaxes=False,
    **layout,
):
    layout_args = dict(utils.STD_LAYOUT, **layout)
    fig.update_layout(**layout_args)
    xaxes_settings = dict(gridcolor="lightgrey")
    if xaxes:
        xaxes_settings.update(xaxes)
    fig.update_xaxes(**xaxes_settings)
    yaxes_settings = dict(gridcolor="lightgrey")
    if yaxes:
        yaxes_settings.update(yaxes)
    fig.update_yaxes(**yaxes_settings)
    if match_facet_yaxes:
        for row_idx, row_figs in enumerate(fig._grid_ref):
            for col_idx, col_fig in enumerate(row_figs):
                fig.update_yaxes(
                    row=row_idx + 1,
                    col=col_idx + 1,
                    matches="y" + str(len(row_figs) * row_idx + 1),
                )
    if save_as:
        save_figure_as(fig, save_as)
    fig.show()


# %% [markdown]
# **Loading data**

# %% tags=["hide-input"]
D = utils.get_dataset("couperin_concerts", corpus_release="v2.2")
D

# %% [markdown]
# **Grouping data**

# %% tags=["hide-input"]
pipeline = Pipeline(["KeySlicer", "ModeGrouper"])
grouped_D = pipeline.process(D)
grouped_D

# %% [markdown]
# **Starting point: DiMCAT's BassNotes feature**

# %% tags=["hide-input"]
bass_notes = grouped_D.get_feature("bassnotes")
bass_notes.df

# %% [markdown]
# **If needed, the `localkey_slice` intervals can be resolved using this table:**

# %% tags=["hide-input"]
local_keys = grouped_D.get_feature("KeyAnnotations")
utils.print_heading("Key Segments")
print(local_keys.groupby("mode").size().to_string())
local_keys.head()

# %% mystnb={"code_prompt_hide": "Hide helpers", "code_prompt_show": "Show helpers"} tags=["hide-cell"]
succession_map = dict(
    ascending_major={
        "1": "2",
        "2": "3",
        "3": "4",
        "4": "5",
        "5": "6",
        "6": "7",
        "7": "1",
    },
    ascending_minor={
        "1": "2",
        "2": "3",
        "3": "4",
        "4": "5",
        "5": "#6",
        "#6": "#7",
        "#7": "1",
    },
    descending={"1": "7", "2": "1", "3": "2", "4": "3", "5": "4", "6": "5", "7": "6"},
)


def inverse_dict(d):
    return {v: k for k, v in d.items()}


predecessor_map = dict(
    ascending_major=inverse_dict(succession_map["ascending_major"]),
    ascending_minor=inverse_dict(succession_map["ascending_minor"]),
    descending=inverse_dict(succession_map["descending"]),
)


def make_precise_preceding_movement_column(df):
    """Expects a dataframe containing the columns bass_degree, preceding_bass_degree, and preceding_movement,"""
    preceding_movement_precise = df.preceding_movement.where(
        df.preceding_movement != "step", df.preceding_interval
    )
    expected_ascending_degree = pd.concat(
        [
            df.loc[["major"], "bass_degree"].map(predecessor_map["ascending_major"]),
            df.loc[["minor"], "bass_degree"].map(predecessor_map["ascending_minor"]),
        ]
    )
    expected_descending_degree = df.bass_degree.map(predecessor_map["descending"])
    preceding_movement_precise = preceding_movement_precise.where(
        df.preceding_bass_degree != expected_ascending_degree, "ascending"
    )
    preceding_movement_precise = preceding_movement_precise.where(
        df.preceding_bass_degree != expected_descending_degree, "descending"
    )
    return preceding_movement_precise


def make_precise_subsequent_movement_column(df):
    """Expects a dataframe containing the columns bass_degree, subsequent_bass_degree, and subsequent_movement,"""
    subsequent_movement_precise = df.subsequent_movement.where(
        df.subsequent_movement != "step", df.subsequent_interval
    )
    expected_ascending_degree = pd.concat(
        [
            df.loc[["major"], "bass_degree"].map(succession_map["ascending_major"]),
            df.loc[["minor"], "bass_degree"].map(succession_map["ascending_minor"]),
        ]
    )
    expected_descending_degree = df.bass_degree.map(succession_map["descending"])
    subsequent_movement_precise = subsequent_movement_precise.where(
        df.subsequent_bass_degree != expected_ascending_degree, "ascending"
    )
    subsequent_movement_precise = subsequent_movement_precise.where(
        df.subsequent_bass_degree != expected_descending_degree, "descending"
    )
    return subsequent_movement_precise


# %% [markdown]
# **This is the main table of this notebook. It corresponds to the `BassNotes` features,
# with a `preceding_` and a `subsequent_` copy of each column concatenated to the right.
# The respective upward and downward shifts are performed within each localkey group,
# leaving first bass degrees with undefined preceding values and last bass degrees without
# undefined subsequent values.**

# %% tags=["hide-input"]
preceding = bass_notes.groupby(["piece", "localkey_slice"]).shift()
preceding.columns = "preceding_" + preceding.columns
subsequent = bass_notes.groupby(["piece", "localkey_slice"]).shift(-1)
subsequent.columns = "subsequent_" + subsequent.columns
BN = pd.concat([bass_notes, preceding, subsequent], axis=1)
BN["preceding_iv"] = BN.bass_note - BN.preceding_bass_note
BN["subsequent_iv"] = BN.subsequent_bass_note - BN.bass_note
BN["preceding_interval"] = ms3.transform(BN.preceding_iv, ms3.fifths2iv, smallest=True)
BN["subsequent_interval"] = ms3.transform(
    BN.subsequent_iv, ms3.fifths2iv, smallest=True
)
BN["preceding_iv_is_step"] = BN.preceding_iv.isin(
    (-5, -2, 2, 5)
).where(  # +m2, -M2, +M2, -m2
    BN.preceding_iv.notna()
)
BN["subsequent_iv_is_step"] = BN.subsequent_iv.isin((-5, -2, 2, 5)).where(
    BN.subsequent_iv.notna()
)
BN["preceding_iv_is_0"] = BN.preceding_iv == 0
BN["subsequent_iv_is_0"] = BN.subsequent_iv == 0
BN["preceding_movement"] = (
    BN.preceding_iv_is_step.map({True: "step", False: "leap"})
    .where(~BN.preceding_iv_is_0, "same")
    .where(BN.preceding_iv.notna(), "none")
)
BN["subsequent_movement"] = (
    BN.subsequent_iv_is_step.map({True: "step", False: "leap"})
    .where(~BN.subsequent_iv_is_0, "same")
    .where(BN.subsequent_iv.notna(), "none")
)
BN["preceding_movement_precise"] = make_precise_preceding_movement_column(BN)
BN["subsequent_movement_precise"] = make_precise_subsequent_movement_column(BN)

BN.head(15)

# %% tags=["hide-input"]
ignore_mask = BN.subsequent_interval.isna() | BN.subsequent_interval.duplicated()
interval2fifths = (  # mapping that allows to order the x-axis with intervals according to LoF
    BN.loc[~ignore_mask, ["subsequent_interval", "subsequent_iv"]]
    .set_index("subsequent_interval")
    .iloc[:, 0]
    .sort_values()
)

# %% [markdown]
# ## Overview of how the bass moves
# ### Intervals

# %% tags=["hide-input"]
interval_data = pd.concat(
    [
        BN.groupby("mode").subsequent_interval.value_counts(normalize=True),
        BN.groupby(["piece", "mode"])
        .subsequent_interval.value_counts(normalize=True)
        .groupby(["mode", "subsequent_interval"])
        .sem()
        .rename("std_err"),
    ],
    axis=1,
).reset_index()
fig = px.bar(
    interval_data,
    x="subsequent_interval",
    y="proportion",
    color="mode",
    barmode="group",
    error_y="std_err",
    color_discrete_map=utils.MAJOR_MINOR_COLORS,
    labels=dict(subsequent_interval="Interval"),
    title="Mode-wise proportion of how often a bass note moves by an interval",
    category_orders=dict(subsequent_interval=interval2fifths.index),
)
style_plotly(fig, "how_often_a_bass_note_moves_by_an_interval")

# %% [markdown]
# ### Types of movement
#
# **The values `ascending` and `descending` designate stepwise movement within the _regola_. Only non-chromatic scale
# degrees can have these values with the exception of `#6` and `#7` which are considered diatonic in the context of
# this study.**

# %% tags=["hide-input"]
PRECISE_CATEGORIES = True

subsequent_movement = (
    "subsequent_movement_precise" if PRECISE_CATEGORIES else "subsequent_movement"
)
movement_data = pd.concat(
    [
        BN.groupby("mode")[subsequent_movement].value_counts(
            normalize=True, dropna=False
        ),
        BN.groupby(["piece", "mode"])[subsequent_movement]
        .value_counts(normalize=True, dropna=False)
        .groupby(["mode", subsequent_movement])
        .sem()
        .rename("std_err"),
    ],
    axis=1,
).reset_index()
movement_data[subsequent_movement] = movement_data[subsequent_movement].fillna("none")
fig = px.bar(
    movement_data,
    x=subsequent_movement,
    y="proportion",
    color="mode",
    barmode="group",
    error_y="std_err",
    color_discrete_map=utils.MAJOR_MINOR_COLORS,
    labels={subsequent_movement: "Movement"},
    title="Mode-wise proportion of a bass note moving in a certain manner",
    category_orders=dict(subsequent_interval=interval2fifths.index),
)
style_plotly(fig, save_as="mode-wise_bass_motion")


# %% [markdown]
# ## Sankey diagrams showing movement types before and after each scale degree

# %% mystnb={"code_prompt_hide": "Hide helpers", "code_prompt_show": "Show helpers"} tags=["hide-cell"]
def make_sankey_data(
    five_major, color_edges=True, precise=True
) -> Tuple[pd.DataFrame, List[str], List[str]] | Tuple[pd.DataFrame, List[str]]:
    preceding_movement = (
        "preceding_movement_precise" if precise else "preceding_movement"
    )
    subsequent_movement = (
        "subsequent_movement_precise" if precise else "subsequent_movement"
    )
    type_counts = five_major["intervals_over_bass"].value_counts()
    preceding_movement_counts = five_major[preceding_movement].value_counts()
    subsequent_movement_counts = five_major[subsequent_movement].value_counts()
    preceding_links = five_major.groupby(
        [preceding_movement]
    ).intervals_over_bass.value_counts()
    subsequent_links = five_major.groupby(
        [subsequent_movement]
    ).intervals_over_bass.value_counts()

    node_labels = []
    label_ids = dict()
    for key, node_sizes in (
        ("preceding", preceding_movement_counts),
        ("intervals", type_counts),
        ("subsequent", subsequent_movement_counts),
    ):
        for label in node_sizes.index:
            label_id = len(node_labels)
            node_labels.append(str(label))
            label_ids[(key, label)] = label_id

    edge_columns = ["source", "target", "value"]
    if color_edges:
        node_colors = utils.make_evenly_distributed_color_map(node_labels)
        edge_columns.append("color")

    links = []
    for (prec_mov, iv), cnt in preceding_links.items():
        source_id = label_ids.get(("preceding", prec_mov))
        target_id = label_ids.get(("intervals", iv))
        if color_edges:
            edge_color = node_colors[source_id]
            links.append((source_id, target_id, cnt, edge_color))
        else:
            links.append((source_id, target_id, cnt))

    for (subs_mov, iv), cnt in subsequent_links.items():
        source_id = label_ids.get(("intervals", iv))
        target_id = label_ids.get(("subsequent", subs_mov))
        if color_edges:
            edge_color = node_colors[target_id]
            links.append((source_id, target_id, cnt, edge_color))
        else:
            links.append((source_id, target_id, cnt))

    edge_data = pd.DataFrame(links, columns=edge_columns)
    if color_edges:
        return edge_data, node_labels, node_colors
    return edge_data, node_labels


def make_bass_degree_sankey(
    bass_degree: str, mode: Literal["major", "minor"], **layout
):
    edge_data, node_labels, node_colors = make_sankey_data(
        BN.loc[mode].query(f"bass_degree == '{bass_degree}'")
    )
    fig = utils.make_sankey(edge_data, node_labels, node_color=node_colors, **layout)
    return fig


# %% [markdown]
# ### Intervals over bass degree 1
# #### Major

# %% tags=["hide-input"]
make_bass_degree_sankey(1, "major")

# %% [markdown]
# #### Minor

# %% tags=["hide-input"]
make_bass_degree_sankey(1, "minor")

# %% [markdown]
# ### Intervals over bass degree 2
# #### Major

# %% tags=["hide-input"]
make_bass_degree_sankey(2, "major")

# %% [markdown]
# #### Minor

# %% tags=["hide-input"]
make_bass_degree_sankey(2, "minor")

# %% [markdown]
# ### Intervals over bass degree 3
# #### Major

# %% tags=["hide-input"]
make_bass_degree_sankey(3, "major")

# %% [markdown]
# #### Minor

# %% tags=["hide-input"]
make_bass_degree_sankey(3, "minor")

# %% [markdown]
# ### Intervals over bass degree 4
# #### Major

# %% tags=["hide-input"]
make_bass_degree_sankey(4, "major")

# %% [markdown]
# #### Minor

# %% tags=["hide-input"]
make_bass_degree_sankey(4, "minor")

# %% [markdown]
# ### Intervals over bass degree 5
# #### Major

# %% tags=["hide-input"]
make_bass_degree_sankey(5, "major")

# %% [markdown]
# #### Minor

# %% tags=["hide-input"]
make_bass_degree_sankey(5, "minor")

# %% [markdown]
# ### Intervals over bass degree 6
# #### Major

# %% tags=["hide-input"]
make_bass_degree_sankey(6, "major")

# %% [markdown]
# #### Minor (ascending)

# %% tags=["hide-input"]
make_bass_degree_sankey("#6", "minor")

# %% [markdown]
# #### Minor (descending)

# %% tags=["hide-input"]
make_bass_degree_sankey(6, "minor")

# %% [markdown]
# ### Intervals over bass degree 7
# #### Major

# %% tags=["hide-input"]
make_bass_degree_sankey(7, "major")

# %% [markdown]
# #### Minor (ascending)

# %% tags=["hide-input"]
make_bass_degree_sankey("#7", "minor")

# %% [markdown]
# #### Minor (descending)

# %% tags=["hide-input"]
make_bass_degree_sankey(7, "minor")

# %% [markdown]
# ## Explanatory power of the RoO
#
# **Most frequent chord for each bass degree**

# %% tags=["hide-input"]
BN.groupby(["mode", "bass_degree"]).intervals_over_bass.apply(
    lambda S: S.value_counts().idxmax()
)

# %% tags=["hide-input"]
maj = ("M3", "P5")
maj6 = ("m3", "m6")
min = ("m3", "P5")
min6 = ("M3", "M6")
Mm56 = ("m3", "d5", "m6")
Mm34 = ("m3", "P4", "M6")
Mm24 = ("M2", "a4", "M6")
mm56 = ("M3", "P5", "M6")
hdim56 = ("m3", "P5", "M6")
hdim34 = ("M3", "a4", "M6")

regole = dict(
    ascending_major=[
        ("1", maj),  # most frequent
        ("2", Mm34),  # most frequent
        ("3", maj6),  # most frequent
        ("4", mm56),  # not most frequent
        ("5", maj),  # most frequent
        ("6", min6),  # not most frequent
        ("7", Mm56),  # most frequent
    ],
    descending_major=[
        ("1", maj),  # same
        ("7", maj6),  # different, not most frequent
        ("6", Mm34),  # different, not most frequent either
        ("5", maj),  # same
        ("4", Mm24),  # different, not most frequent either
        ("3", maj6),  # same
        ("2", Mm34),  # same
    ],
    ascending_minor=[
        ("1", min),  # most frequent
        ("2", Mm34),  # most frequent
        ("3", min6),  # most frequent
        ("4", hdim56),  # most frequent
        ("5", maj),  # most frequent
        ("#6", maj6),  # most frequent
        ("#7", Mm56),  # most frequent
    ],
    descending_minor=[
        ("1", min),  # same
        ("7", min6),  # different, most frequent
        ("6", hdim34),  # different, most frequent
        ("5", maj),  # same
        ("4", Mm24),  # different, not most frequent
        ("3", min6),  # same
        ("2", Mm34),  # same
    ],
)


# %% mystnb={"code_prompt_hide": "Hide helpers", "code_prompt_show": "Show helpers"} tags=["hide-cell"]
@cache
def get_base_df(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    query: Optional[str] = None,
):
    global BN
    try:
        mode, selection = basis.split("_")
    except Exception:
        raise ValueError(f"Invalid keyword for basis: {basis!r}")
    base = BN.loc[[mode]]
    if selection == "all":
        result = base
    elif selection == "diatonic":
        if mode == "major":
            result = base.query("bass_degree in ('1', '2', '3', '4', '5', '6', '7')")
        elif mode == "minor":
            result = base.query(
                "bass_degree in ('1', '2', '3', '4', '5', '6', '#6', '7', '#7')"
            )
    else:
        raise ValueError(f"Unknown keyword for selection: {selection!r}")
    if query:
        result = result.query(query)
    return result


@cache
def get_bass_degree_mask(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    bass_degree: str,
    query: Optional[str] = None,
):
    base = get_base_df(basis, query=query)
    return base.bass_degree == bass_degree


@cache
def get_intervals_mask(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    intervals: tuple,
    query: Optional[str] = None,
):
    base = get_base_df(basis, query=query)
    return base.intervals_over_bass == intervals


@cache
def get_chord_mask(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    bass_degree: str,
    intervals: tuple,
    query: Optional[str] = None,
):
    bass_degree_mask = get_bass_degree_mask(
        basis=basis, bass_degree=bass_degree, query=query
    )
    intervals_mask = get_intervals_mask(basis=basis, intervals=intervals, query=query)
    return bass_degree_mask & intervals_mask


@cache
def get_chord_vocabulary_mask(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    vocabulary: Tuple[Tuple[str, tuple], ...],
    query: Optional[str] = None,
) -> pd.Series:
    base = get_base_df(basis, query=query)
    mask = pd.Series(False, index=base.index, dtype="boolean")
    for bass_degree, intervals in vocabulary:
        mask |= get_chord_mask(
            basis=basis, bass_degree=bass_degree, intervals=intervals, query=query
        )
    return mask


def inspect(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    vocabulary: Tuple[Tuple[str, tuple], ...],
    query: Optional[str] = None,
) -> pd.DataFrame:
    base = get_base_df(basis, query=query)
    mask = get_chord_vocabulary_mask(basis=basis, vocabulary=vocabulary, query=query)
    return base[mask]


def get_vocabulary_coverage(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    vocabulary: Tuple[Tuple[str, tuple], ...],
    query: Optional[str] = None,
) -> float:
    mask = get_chord_vocabulary_mask(basis=basis, vocabulary=vocabulary, query=query)
    return mask.sum() / len(mask)


def get_coverage_values(
    major_vocabulary: Optional[Tuple[Tuple[str, tuple], ...]] = None,
    minor_vocabulary: Optional[Tuple[Tuple[str, tuple], ...]] = None,
    **name2query,
) -> pd.Series:
    if not (major_vocabulary or minor_vocabulary):
        return pd.Series()
    results = {}
    if major_vocabulary:
        results.update(
            {
                ("major", "all"): get_vocabulary_coverage(
                    "major_all", major_vocabulary
                ),
                ("major", "diatonic"): get_vocabulary_coverage(
                    "major_diatonic", major_vocabulary
                ),
            }
        )
        for name, query in name2query.items():
            results[("major", name)] = get_vocabulary_coverage(
                "major_diatonic", major_vocabulary, query=query
            )
    if minor_vocabulary:
        results.update(
            {
                ("minor", "all"): get_vocabulary_coverage(
                    "minor_all", minor_vocabulary
                ),
                ("minor", "diatonic"): get_vocabulary_coverage(
                    "minor_diatonic", minor_vocabulary
                ),
            }
        )
        for name, query in name2query.items():
            results[("minor", name)] = get_vocabulary_coverage(
                "minor_diatonic", minor_vocabulary, query=query
            )
    result = pd.Series(results, name="proportion")
    result.index.names = ["mode", "coverage_of"]
    return result


# %% [markdown]
# ### Which proportion of unigrams are "explained" by Campion's regola
#
# The percentages are based on different sets of unigrams.
# `from` means before/leading to a bass degree, `to` means after/following a bass degree.
#
# * `all`: all bass degrees
# * `diatonic`: all non-chromatic bass degrees (in minor, the chromatic scale degrees `#6` and `#7` are considered
#   diatonic)
# * `to_ascending`: all diatonic bass degrees that ascend within the regola
# * `from_ascending`: all diatonic bass degrees that are reached by ascending within the regola
# * `to_and_from_ascending`: all diatonic bass degrees that are reached by ascending within the regola and proceed
#   ascending within the regola
# * `to_and_from_either`: all diatonic bass degrees whose predecessor and successor are both upper or lower neighbors
#   within the regola
# * `to_leap`: all diatonic bass degrees followed by a leap
# * `to_same`: all diatonic bass degrees followed by the same bass degree
# * etc.

# %% tags=["hide-input"]
regola_vocabulary_major = tuple(
    set(regole["ascending_major"] + regole["descending_major"])
)
regola_vocabulary_minor = tuple(
    set(regole["ascending_minor"] + regole["descending_minor"])
)

features = dict(
    to_ascending="subsequent_movement_precise == 'ascending'",
    to_descending="subsequent_movement_precise == 'descending'",
    to_either="subsequent_movement_precise == ['ascending', 'descending']",
    to_leap="subsequent_movement == 'leap'",
    to_same="subsequent_movement == 'same'",
    last_notes="subsequent_movement == 'none'",
    from_ascending="preceding_movement_precise == 'ascending'",
    from_descending="preceding_movement_precise == 'descending'",
    from_either="preceding_movement_precise == ['ascending', 'descending']",
    from_leap="preceding_movement == 'leap'",
    from_same="preceding_movement == 'same'",
    first_notes="preceding_movement == 'none'",
    to_and_from_ascending="subsequent_movement_precise == 'ascending' & preceding_movement_precise == 'ascending'",
    to_and_from_descending="subsequent_movement_precise == 'descending' & preceding_movement_precise == 'descending'",
    to_and_from_either="subsequent_movement_precise == ['ascending', 'descending'] & "
    "preceding_movement_precise == ['ascending', 'descending']",
    to_and_from_leap="subsequent_movement == 'leap' & preceding_movement == 'leap'",
    to_and_from_same="subsequent_movement == 'same' & preceding_movement == 'same'",
)

regola_coverage = get_coverage_values(
    regola_vocabulary_major, regola_vocabulary_minor, **features
)
regola_coverage


# %% [markdown]
# ### Comparing the regola against all "top k" vocabularies
#
# **Campion's regola comprises 10 different chords for both major and minor.
# For comparison, its values are shown at point 10.5 on the x-axis.
# The lower two plots show how many unigrams are covered by individual chords.
# Hover over the points to see the corresponding chords.**

# %% mystnb={"code_prompt_hide": "Hide helpers", "code_prompt_show": "Show helpers"} tags=["hide-cell"]
def make_coverage_plot_data(
    include_singular_vocabularies=True, **features
) -> pd.DataFrame:
    all_chords = BN[["bass_degree", "intervals_over_bass"]].apply(tuple, axis=1)
    chord_ranking = all_chords.groupby("mode").value_counts(normalize=True)
    major_ranking, minor_ranking = (
        chord_ranking.loc["major"],
        chord_ranking.loc["minor"],
    )
    major_vocab, minor_vocab = [], []
    results = {}
    for i, (maj_chord, min_chord) in enumerate(
        itertools.zip_longest(major_ranking.index, minor_ranking.index), 1
    ):
        if maj_chord:
            major_vocab.append(maj_chord)
        if min_chord:
            minor_vocab.append(min_chord)
        key = ("cumulative", i) if include_singular_vocabularies else i
        values = get_coverage_values(tuple(major_vocab), tuple(minor_vocab), **features)
        chord = pd.Series(str(maj_chord), index=values.index, name="chord")
        chord.loc["minor"] = str(min_chord)
        results[key] = pd.concat([values, chord], axis=1)
        if not include_singular_vocabularies:
            continue
        single_maj_vocab = (maj_chord,) if maj_chord else None
        single_min_vocab = (min_chord,) if min_chord else None
        values = get_coverage_values(single_maj_vocab, single_min_vocab, **features)
        results[("single", i)] = pd.concat([values, chord], axis=1)
    index_levels = ["vocabulary", "rank"] if include_singular_vocabularies else ["rank"]
    return pd.concat(results, names=index_levels)


# %% tags=["hide-input"]

result = make_coverage_plot_data(**features)
regola_results = pd.concat(
    {("cumulative", 10.5): regola_coverage}, names=["vocabulary", "rank"]
).to_frame()
regola_results.loc[:, "chord"] = "regola"
result = pd.concat(
    [
        regola_results,
        result,
    ]
).sort_index()

# %% tags=["hide-input"]
fig = px.line(
    result.reset_index(),
    x="rank",
    y="proportion",
    color="coverage_of",
    facet_col="mode",
    facet_row="vocabulary",
    hover_name="chord",
    log_x=True,
    title="How many unigrams are covered by each top-k vocabulary",
)
style_plotly(
    fig,
    match_facet_yaxes=True,
    height=1500,
    legend=dict(
        orientation="h",
    ),
)

# %% [markdown]
# **In order to inspect these plots you will want to hide traces.
# Click on a legend item to toggle it, double-click on an item to toggle all others.**

# %% [markdown]
# **In order to inspect these plots you will want to hide traces.
# Click on a legend item to toggle it, double-click on an item to toggle all others.**