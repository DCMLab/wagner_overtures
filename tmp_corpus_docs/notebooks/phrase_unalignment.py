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

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]

# %load_ext autoreload
# %autoreload 2
import os
from collections import Counter, defaultdict

import dimcat as dc
import ms3
import pandas as pd
from dimcat import resources
from dimcat.plotting import write_image
from git import Repo
from IPython.display import display

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

# %%
RESULTS_PATH = os.path.expanduser("~/git/diss/33_phrases/figs")
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
            write_image(fig, filename, directory, format=fmt, **kwargs)
    else:
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
composition_years = D.get_metadata().get_composition_years()
composition_years.head()

# %%
phrase_annotations: resources.PhraseAnnotations = D.get_feature("PhraseAnnotations")
phrase_annotations.head()

# %%
stage_data = utils.make_stage_data(
    phrase_annotations,
    columns=[
        "chord",
        "numeral_or_applied_to_numeral",
        "localkey",
        "localkey_is_minor",
        "duration_qb",
    ],
    wide_format=False,
)
original_stage_data = stage_data.copy()
original_stage_data._df = stage_data._df.copy()
# convert the criterion to fifths over local tonic
stage_data._df["numeral_fifths"] = ms3.transform(
    stage_data,
    ms3.roman_numeral2fifths,
    ["numeral_or_applied_to_numeral", "localkey_is_minor"],
)
# then convert the fifths back to roman numerals to normalize them, making them all uppercase and correspond to a
# major scale
stage_data._df["numeral_or_applied_to_numeral"] = stage_data.numeral_fifths.map(
    ms3.fifths2rn
)
stage_data.sample(50)

# %%
width = 1280
height = 800
COLOR_MAP = {
    "VI": "hsv(0%,100%,100%)",
    "I": "hsv(4%,100%,100%)",
    "bbIII": "hsv(8%,100%,100%)",
    "#VII": "hsv(12%,100%,100%)",
    "bbVII": "hsv(15%,100%,100%)",
    "bbVI": "hsv(19%,100%,100%)",
    "#IV": "hsv(23%,100%,100%)",
    "bIV": "hsv(27%,100%,100%)",
    "#II": "hsv(31%,100%,100%)",
    "IV": "hsv(35%,100%,100%)",
    "bV": "hsv(38%,100%,100%)",
    "bVII": "hsv(42%,100%,100%)",
    "VII": "hsv(46%,100%,100%)",
    "#VI": "hsv(50%,100%,100%)",
    "#III": "hsv(54%,100%,100%)",
    "II": "hsv(58%,100%,100%)",
    "bIII": "hsv(62%,100%,100%)",
    "III": "hsv(65%,100%,100%)",
    "#I": "hsv(69%,100%,100%)",
    "V": "hsv(73%,100%,100%)",
    "bVI": "hsv(77%,100%,100%)",
    None: "hsv(81%,100%,100%)",
    "bII": "hsv(85%,100%,100%)",
    "#V": "hsv(88%,100%,100%)",
    "bI": "hsv(92%,100%,100%)",
    pd.NA: "hsv(96%,100%,100%)",
    "@none": "#000000",
}
COLOR_MAP.update({k.lower(): v for k, v in COLOR_MAP.items() if not pd.isnull(k)})


def stages2graph_data(
    stages, ending_on=None, stop_at_modulation=True, cut_at_stage=None
):
    stage_nodes = defaultdict(dict)  # {stage -> {label -> node}}
    edge_weights = Counter()  # {(source_node, target_node) -> weight}
    node_counter = 0
    if ending_on is not None:
        if isinstance(ending_on, str):
            ending_on = {ending_on}
        else:
            ending_on = set(ending_on)
    for phrase_id, progression in stages.groupby("phrase_id"):
        previous_node = None
        for stage, stage_df in progression.groupby("stage"):
            if cut_at_stage and stage > cut_at_stage:
                break
            first_row = stage_df.iloc[0]
            current = first_row.iloc[0]
            if stage == 0:
                if ending_on is not None and current not in ending_on:
                    break
                if stop_at_modulation:
                    localkey = first_row.localkey
            elif stop_at_modulation and first_row.localkey != localkey:
                break
            if current in stage_nodes[stage]:
                current_node = stage_nodes[stage][current]
            else:
                stage_nodes[stage][current] = current_node = node_counter
                node_counter += 1
            if previous_node is not None:
                edge_weights.update([(current_node, previous_node)])
            previous_node = current_node
    return stage_nodes, edge_weights


def make_simple_phrase_sankey(
    stages,
    ending_on=None,
    stop_at_modulation=True,
    cut_at_stage=10,
    height=800,
    color_map=None,
    font_color="blue",
    **kwargs,
):
    stage_nodes, edge_weights = stages2graph_data(
        stages,
        ending_on=ending_on,
        stop_at_modulation=stop_at_modulation,
        cut_at_stage=cut_at_stage,
    )
    node2label = {
        node: label for nodes in stage_nodes.values() for label, node in nodes.items()
    }
    if color_map is None:
        labels = [node2label[i] for i in range(len(node2label))]
        color_map = dict(COLOR_MAP)
        for label in labels:
            if label in color_map:
                continue
            resolved_label = ms3.resolve_relative_keys(label)
            while resolved_label and resolved_label not in color_map:
                resolved_label = resolved_label[1:]
            color_map[label] = color_map[resolved_label]
    settings = dict(font=dict(color="blue"), **kwargs)
    return utils.graph_data2sankey(
        stage_nodes, edge_weights, color_map=color_map, height=height, **settings
    )


# %%
original_stages = original_stage_data.regroup_phrases(
    original_stage_data.numeral_or_applied_to_numeral
).join(composition_years)
original_sankey = make_simple_phrase_sankey(
    original_stages[original_stages.iloc[:, 0].notna()], cut_at_stage=10
)
save_figure_as(original_sankey, "numeral_borrowed_sankey_before", width=800, height=500)

original_sankey

# %%
# created for public defense
original_sankey.update_layout(
    font=dict(size=25, color="blue"),
    title=dict(text="Harmony from a distance (~1600 - ~1930)", automargin=True),
)
original_sankey.write_html("numeral_borrowed_sankey_before.html")

# %%
numeral_criterion = stage_data.numeral_or_applied_to_numeral
stages = stage_data.regroup_phrases(numeral_criterion).join(composition_years)
stages = stages.loc[:, ~stages.columns.duplicated()].copy()
first10_sankey = make_simple_phrase_sankey(stages)
save_figure_as(
    first10_sankey, "numeral_borrowed_sankey_after", width=width, height=height
)
# first10_sankey

# %%
DEFAULT_ERAS = dict(
    prebaroque="mean_composition_year < 1650",
    baroque="1650 <= mean_composition_year <= 1750",
    classical="1750 < mean_composition_year <= 1800",
    extended="1800 < mean_composition_year",
)


def make_era_sankeys(
    stages,
    ending_on={"I"},
    stop_at_modulation=True,
    cut_at_stage=10,
    height=300,
    width=1000,
    eras=None,
):
    if eras is None:
        eras = DEFAULT_ERAS
    for era, query in eras.items():
        era_stages = stages.query(query)
        era_sankey = make_simple_phrase_sankey(
            era_stages,
            ending_on=ending_on,
            stop_at_modulation=stop_at_modulation,
            cut_at_stage=cut_at_stage,
        )
        name = f"{era}_{','.join(ending_on)}_{cut_at_stage}_sankey"
        save_figure_as(era_sankey, name, width=width, height=height)


make_era_sankeys(stages, ending_on={"I"})

# %%
make_era_sankeys(stages, ending_on={"V"})

# %%
stages.index.names

# %%
stages.head()


# %%
def show_era_stats(stages, eras=DEFAULT_ERAS):
    overall = {}
    for era, query in eras.items():
        era_stages = stages.query(query)
        column = era_stages.columns[0]
        end_label = era_stages.groupby("phrase_id")[column].stage_values()
        index_levels = era_stages.index.to_frame()
        n_phrases = index_levels.phrase_id.nunique()
        n_pieces = len(set(index_levels[["corpus", "piece"]].itertuples(index=False)))
        print(f"{era}: {n_phrases} phrases in {n_pieces} pieces")
        vc = utils.value_count_df(end_label)
        display(vc)
        tondom_count = vc.loc[["I", "V"], "counts"].sum()
        nontondom = vc.counts.sum() - tondom_count
        overall[era] = dict(
            n_phrases=n_phrases,
            n_pieces=n_pieces,
            tondom_count=tondom_count,
            nontondom=nontondom,
        )
    return pd.DataFrame(overall).T


stats = show_era_stats(stages)
stats

# %%
tondom_ending = stats.tondom_count.sum()
print(
    f"{tondom_ending} ({tondom_ending / stats.n_phrases.sum():.1%}) phrases end on I or V"
)

# %%
corpus_wise_entropy = stages.groupby("corpus").apply(
    lambda df: pd.Series(
        dict(
            mean_year=df.mean_composition_year.mean(),
            mean_entropy=utils.get_criterion_stage_entropies(df).mean(),
        )
    )
)  # entropy(df
corpus_wise_entropy

# %%
prebaroque = corpus_wise_entropy.query("mean_year < 1650").index.to_list()
baroque = corpus_wise_entropy.query("1650 <= mean_year <= 1750").index.to_list()

# # %%
# import plotly.express as px
#
# px.scatter(
#     corpus_wise_entropy.reset_index(),
#     x="mean_year",
#     y="mean_entropy",
#     hover_name="corpus",
# )

# %%
make_simple_phrase_sankey(stages, ending_on={"I"}, cut_at_stage=20)

# %%
bach_sankey = make_simple_phrase_sankey(stages.query("corpus == 'bach_en_fr_suites'"))
bach_sankey.update_layout(font=dict(size=25, color="blue"))

# %%
type(numeral_criterion)

# %%
enoid = utils.make_effective_numeral_or_its_dominant_criterion(phrase_annotations)
enoid_stages = stage_data.regroup_phrases(enoid)

# %%
make_simple_phrase_sankey(enoid_stages)

# %%
numeral_criterion = stage_data.numeral_or_applied_to_numeral
stages = stage_data.regroup_phrases(numeral_criterion).join(composition_years)
stage_nodes, edge_weights = stages2graph_data(
    stages,  # .query("corpus == 'corelli'"),
    ending_on={"I"},
    stop_at_modulation=True,
    cut_at_stage=10,
)
all_I_sankey = utils.graph_data2sankey(
    stage_nodes, edge_weights, color_map=COLOR_MAP, height=800
)
# save_figure_as(all_I_sankey, "all_I_sankey", width=width, height=height)

# %%
# make_sanke() autocolors
# node2label = {
#         node: label for nodes in stage_nodes.values() for label, node in nodes.items()
#     }
# labels = [node2label[i] for i in range(len(node2label))]
# unique_labels = set(labels)
# color_step = 100 / len(unique_labels)
# color_map = {
#     label: f"hsv({round(i*color_step)}%,100%,100%)"
#     for i, label in enumerate(unique_labels)
# }

# UTILS COLORS
# label_fifths = {label: ms3.roman_numeral2fifths(label) for label in labels}
# label_color = {label: utils.get_fifths_color(fifths)[0] for label, fifths in label_fifths.items()}


utils.graph_data2sankey(stage_nodes, edge_weights, color_map=COLOR_MAP, height=800)

# %%
all_I_sankey.update_traces(dict(arrangement="fixed"))

# %%
corelli_first = stages.query("substage == 0 & corpus == 'corelli'")
corelli_first


def nonmodulating_progression(df):
    first_row = df.iloc[0]
    nonmodulating = df[df.localkey == first_row.localkey]
    return tuple(nonmodulating.iloc[::-1, 0])


corelli_root_progressions = corelli_first.groupby("phrase_id").apply(
    nonmodulating_progression
)
corelli_root_progressions.value_counts()

# %%
ngram_counts = Counter()
for prog in corelli_root_progressions:
    ngram_counts.update(
        (
            ngram
            for n in range(len(prog), 1, -1)
            for ngram in zip(*(prog[i:] for i in range(n)))
        )
    )

# %%
ngram_counts.most_common(50)

# %%
corelli_root_progressions.str[-1].value_counts()

# %%
corelli_I = corelli_root_progressions[corelli_root_progressions.str[-1] == "I"]
print(f"{len(corelli_I)} phrases ending on I to explain")
corelli_I.value_counts().head(30)

# %%
corelli_I.value_counts().sort_index(key=lambda ix: ix.map(len))

# %%
explained = corelli_I.str[-2].isna()
print(f"{explained.sum()} phrases ending on I are the only label.")
V_I = corelli_I.str[-2:] == ("V", "I")
explained |= V_I
print(f"{V_I.sum()} phrases have only V-I.")
only_3 = corelli_I.str[-4].isna()
ii_V_I = corelli_I.str[-3:] == ("ii", "V", "I")
explained |= ii_V_I
print(f"{ii_V_I.sum()} phrases have only ii-V-I.")
IV_V_I = corelli_I.str[-3:] == ("IV", "V", "I")
explained |= IV_V_I
print(f"{IV_V_I.sum()} phrases have only IV-V-I.")
I_IV_V_I = corelli_I.str[-4:] == ("I", "IV", "V", "I")
explained |= I_IV_V_I
print(f"{I_IV_V_I.sum()} phrases have only I-IV-V-I.")
I_ii_V_I = corelli_I.str[-4:] == ("I", "ii", "V", "I")
explained |= I_ii_V_I
print(f"{I_ii_V_I.sum()} phrases have only I-ii-V-I.")
print(f"{explained.sum()} phrases are explained.")

# %%
corelli_I