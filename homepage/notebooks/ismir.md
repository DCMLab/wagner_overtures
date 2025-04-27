---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

# Plots for ISMIR 2023

Notebook created by copying and adapting `annotations.ipynb`.

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
%load_ext autoreload
%autoreload 2
import os

import dimcat as dc
import ms3
import pandas as pd
from dimcat import analyzers, groupers, plotting
from git import Repo
from IPython.display import display
from matplotlib import pyplot as plt

import utils
```

```{code-cell}
RESULTS_PATH = os.path.abspath("/home/laser/git/diss/26_dlc/img/")
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
```

```{code-cell}
:tags: [remove-output]

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
```

```{code-cell}
bigramAnalyzer = analyzers.BigramAnalyzer(features="keyannotations")
bigramAnalyzer.get_feature_specs()
```

```{code-cell}
analyzed_D = bigramAnalyzer.process(D)
localkey_bigram_table = analyzed_D.get_result()
localkey_bigram_table
```

```{code-cell}
localkey_bigram_table.plot_grouped()
```

```{code-cell}
localkey_bigram_table.make_bigram_table()
```

```{code-cell}
localkey_bigram_table.make_ranking_table()
```

```{code-cell}
localkey_bigram_table.plot(max_x=30, max_y=30)
```

```{code-cell}
transitions = localkey_bigram_table.get_transitions()
transitions
```

```{code-cell}
all_metadata = D.get_metadata()
assert len(all_metadata) > 0, "No pieces selected for analysis."
mean_composition_years = utils.corpus_mean_composition_years(all_metadata)
chronological_order = mean_composition_years.index.to_list()
corpus_colors = dict(zip(chronological_order, utils.CORPUS_COLOR_SCALE))
corpus_names = {
    corp: utils.get_corpus_display_name(corp) for corp in chronological_order
}
chronological_corpus_names = list(corpus_names.values())
corpus_name_colors = {
    corpus_names[corp]: color for corp, color in corpus_colors.items()
}
```

## FeatureExtractor

```{code-cell}
notes = D.get_feature("notes")
notes.plot_grouped(
    output=make_output_path("complete_pitch_class_distribution_absolute_bars"),
    height=800,
)
```

## Slicer

```{code-cell}
def globalminor_localkey_expressed_in_globalmajor(key):
    return ms3.abs2rel_key(key, "I", True)


def map_GLEIG_to_list(globalminor_localkeys):
    return list(
        map(globalminor_localkey_expressed_in_globalmajor, globalminor_localkeys)
    )


def piecewise_localkeys_expressed_in_globalmajor(localkeys):
    gpb = localkeys.groupby(["corpus", "piece"])
    piecewise_localkeys = gpb.localkey.apply(list)
    is_major_mask = ~gpb.globalkey_is_minor.first()
    expressed_in_globalmajor = piecewise_localkeys.apply(map_GLEIG_to_list)
    return piecewise_localkeys.where(is_major_mask, expressed_in_globalmajor)


keys = D.get_feature("keyannotations")
keys.load()
piecewise_localkey_transitions = piecewise_localkeys_expressed_in_globalmajor(keys)
```

```{code-cell}
# keys.plot(output=make_output_path("localkey_distributions"), height=5000)
```

```{code-cell}
utils.plot_transition_heatmaps(
    piecewise_localkey_transitions.to_list(),
    top=10,
    bottom_margin=0.05,
    left_margin=0.14,
)
save_pdf_path = os.path.join(RESULTS_PATH, "localkey_transition_matrix.pdf")
plt.savefig(save_pdf_path, dpi=400)
plt.show()
```

## Groupers

```{code-cell}
grouping = dc.Pipeline(
    [
        groupers.CorpusGrouper(),
        groupers.ModeGrouper(),
    ]
)
GD = grouping.process(D)
grouped_keys = GD.get_feature("keyannotations")
grouped_keys_df = grouped_keys.df
grouped_keys_df
```

```{code-cell}
len(set(grouped_keys_df.index.to_frame()[["corpus", "piece"]].itertuples(index=False)))
```

```{code-cell}
segment_duration_per_corpus = (
    grouped_keys.groupby(["corpus", "mode"]).duration_qb.sum().round(2)
)
norm_segment_duration_per_corpus = (
    100
    * segment_duration_per_corpus
    / segment_duration_per_corpus.groupby("corpus").sum()
)
maj_min_ratio_per_corpus = pd.concat(
    [
        segment_duration_per_corpus,
        norm_segment_duration_per_corpus.rename("fraction").round(1).astype(str) + " %",
    ],
    axis=1,
)
maj_min_ratio_per_corpus[
    "corpus_name"
] = maj_min_ratio_per_corpus.index.get_level_values("corpus").map(corpus_names)
fig = plotting.make_bar_plot(
    maj_min_ratio_per_corpus.reset_index(),
    x_col="corpus_name",
    y_col="duration_qb",
    title=None,  # f"Fractions of summed corpus duration that are in major vs. minor",
    color="mode",
    text="fraction",
    color_discrete_map=utils.MAJOR_MINOR_COLORS,
    labels=dict(
        duration_qb="duration in 𝅘𝅥",
        corpus_name="",  # "Key segments grouped by corpus"
    ),
    category_orders=dict(
        corpus_name=[
            name
            for name in chronological_corpus_names
            if name in maj_min_ratio_per_corpus.corpus_name.unique()
        ]
    ),
    layout=dict(
        barmode="stack",
        margin=dict(l=0, r=0, b=0, t=0),
    ),
    x_axis=dict(
        tickangle=45,
        tickfont_size=15,
        showgrid=False,
    ),
)
save_figure_as(
    fig,
    "major_minor_key_segments_corpuswise_absolute_stacked_bars",
    height=400,
    width=1200,
)
fig
```

```{raw-cell}
to_be_filled = grouped_keys_df.quarterbeats_all_endings == ''
grouped_keys_df.quarterbeats_all_endings = grouped_keys_df.quarterbeats_all_endings.where(~to_be_filled,
grouped_keys_df.quarterbeats)
ms3.make_interval_index_from_durations(grouped_keys_df, position_col="quarterbeats_all_endings")
```

## Slicer

```{code-cell}
:tags: [hide-input]

try:
    labels = D.get_feature("harmonylabels")
    all_annotations = labels.df
except Exception:
    all_annotations = pd.DataFrame()
n_annotations = len(all_annotations)
includes_annotations = n_annotations > 0
if includes_annotations:
    all_chords = utils.remove_none_labels(all_annotations)
    all_chords = utils.remove_non_chord_labels(all_chords)
    display(all_chords.head())
    print(f"Concatenated annotation tables contain {n_annotations} rows.")
    no_chord = all_annotations.root.isna()
    print(
        f"Dataset contains {len(all_chords)} tokens and {len(all_chords.chord.unique())} types over "
        f"{len(all_chords.groupby(level=[0,1]))} documents."
    )
    all_annotations["corpus_name"] = all_annotations.index.get_level_values(0).map(
        corpus_names
    )
    all_chords["corpus_name"] = all_chords.index.get_level_values(0).map(corpus_names)
else:
    print("Dataset contains no annotations.")
```

```{code-cell}
group_keys, group_dict = dc.data.resources.utils.make_adjacency_groups(
    all_chords.localkey, groupby=["corpus", "piece"]
)
segment2bass_note_series = {
    seg: bn for seg, bn in all_chords.groupby(group_keys).bass_note
}
full_grams = {
    i: S[(S != S.shift()).fillna(True)].to_list()
    for i, S in segment2bass_note_series.items()
}
full_grams_major, full_grams_minor = [], []
for i, bass_notes in segment2bass_note_series.items():
    # progression = bass_notes[(bass_notes != bass_notes.shift()).fillna(True)].to_list()
    is_minor = group_dict[i].islower()
    progression = ms3.fifths2sd(bass_notes.to_list(), is_minor) + ["∅"]
    if is_minor:
        full_grams_minor.append(progression)
    else:
        full_grams_major.append(progression)
```

```{code-cell}
utils.plot_transition_heatmaps(full_grams_major, full_grams_minor, top=20)
plt.savefig(make_output_path("bass_degree_bigrams"), dpi=400)
plt.show()
```

```{code-cell}
# font_dict = {'font': {'size': 20}}2
width = 1600
height = 900
layout=dict(
    utils.STD_LAYOUT,
    margin=dict(l=0, r=0, b=0, t=0),
)
fig = utils.plot_cum(
    all_chords.chord,
    font_size=35,
    markersize=10,
    percent=True,
    height=height,
    width=width,
    **layout,
)
for trace, color in zip(fig.data, (utils.TailwindColorsHex.get_color(c) for c in ("PURPLE_800", "EMERALD_800"))):
    trace.marker.color = color
save_figure_as(
    fig,
    "chord_type_distribution_cumulative",
    width=width,
    height=height,
)
fig.show()
```

```{code-cell}
grouped_chords = groupers.ModeGrouper().process(labels)
grouped_chords.default_groupby
```

```{code-cell}
utils.value_count_df(grouped_chords.chord)
```

```{code-cell}
ugs_dict = {
    mode: utils.value_count_df(chords).reset_index()
    for mode, chords in grouped_chords.groupby("mode").chord
}
ugs_df = pd.concat(ugs_dict, axis=1)
ugs_df.columns = ["_".join(map(str, col)) for col in ugs_df.columns]
ugs_df.index = (ugs_df.index + 1).rename("k")
ugs_df
```