---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

# Phrases in the DLC

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
import re
from typing import Optional

import dimcat as dc
import ms3
import pandas as pd
from dimcat.plotting import make_pie_chart, write_image
from git import Repo

from utils import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FOLDER,
    get_repo_name,
    make_stage_data,
    print_heading,
    resolve_dir,
    value_count_df,
)

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "phrases"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename, extension=None):
    if extension:
        extension = "." + extension.lstrip(".")
    else:
        extension = DEFAULT_OUTPUT_FORMAT
    return os.path.join(RESULTS_PATH, f"{filename}{extension}")


def make_and_store_stage_data(
    phrase_feature,
    name: Optional[str] = None,
    columns="chord",
    components="body",
    drop_levels=3,
    reverse=True,
    level_name="stage",
    wide_format=True,
    query=None,
):
    phrase_data = make_stage_data(
        phrase_feature=phrase_feature,
        columns=columns,
        components=components,
        drop_levels=drop_levels,
        reverse=reverse,
        level_name=level_name,
        wide_format=wide_format,
        query=query,
    )
    if name:
        phrase_data.to_csv(make_output_path(name, "tsv"), sep="\t")
    return phrase_data


def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, **kwargs)
```

```{code-cell}
:tags: [hide-input]

package_path = resolve_dir(
    "~/distant_listening_corpus/distant_listening_corpus.datapackage.json"
)
repo = Repo(os.path.dirname(package_path))
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
D = dc.Dataset.from_package(package_path)
D
```

```{code-cell}
phrase_annotations = D.get_feature("PhraseAnnotations")
phrase_annotations
```

```{code-cell}
phrase_components = phrase_annotations.extract_feature("PhraseComponents")
phrase_components
```

```{code-cell}
phrases = phrase_components.extract_feature("PhraseLabels")
phrases
```

```{code-cell}
vc = value_count_df(phrases.end_chord, rank_index=True)
vc.head(50)
```

```{code-cell}
stages = make_and_store_stage_data(
    phrases, name="stages", columns=["localkey", "chord"]
)
stages.head()
```

```{code-cell}
onekey_major = make_and_store_stage_data(
    phrases,
    name="onekey_major",
    query="body_n_modulations == 0 & localkey_mode == 'major'",
)
onekey_minor = make_and_store_stage_data(
    phrases,
    name="onekey_minor",
    query="body_n_modulations == 0 & localkey_mode == 'minor'",
)
one_key_major_I = make_and_store_stage_data(
    phrases,
    name="onekey_major_I",
    query="body_n_modulations == 0 & localkey_mode == 'major' & end_chord == 'I'",
)  # end_chord.str.contains('^I(?![iIvV\/])')")
one_key_minor_i = make_and_store_stage_data(
    phrases,
    name="one_key_minor_i",
    query="body_n_modulations == 0 & localkey_mode == 'minor' & end_chord == 'i'",
)
```

```{code-cell}
def show_stage(df, column: int, **kwargs):
    """Pie chart for a given stage."""
    stage = df.loc(axis=1)[column]
    if len(stage.shape) > 1:
        stage = stage.iloc(axis=1)[0]
    vc = stage.value_counts().to_frame()
    settings = dict(
        traces_settings=dict(textposition="inside"),
        layout=dict(uniformtext_minsize=20, uniformtext_mode="hide"),
    )
    settings.update(kwargs)
    return make_pie_chart(vc, **settings)


show_stage(one_key_major_I, 1)
```

```{code-cell}
numeral_regex = r"^(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none)"


def get_numeral(label: str) -> str:
    """From a chord label, get the root or, if an applied chord, the root that the chord is applied to."""
    considered = label.split("/")[-1]
    try:
        return re.match(numeral_regex, considered).group()
    except AttributeError:
        return pd.NA


numeral_criterion = one_key_major_I.dataframe.chord.map(
    get_numeral, na_action="ignore"
).rename("root_numeral")
result = one_key_major_I.regroup_phrases(numeral_criterion)
result.to_csv(make_output_path("one_key_major_I.stages", "tsv"), sep="\t")
result
```

```{code-cell}
# regrouped = result._format_dataframe(
#     result.drop(columns="root_numeral"), "WIDE"
# )  # ToDo: Unstack needs to take into account the new index levels
regrouped = result["chord"].unstack(level=[-2, -1]).sort_index(axis=1)
show_stage(regrouped, 1)
```

```{code-cell}
regrouped
```

```{raw-cell}
# this recursive approach is very inefficient

@cache
def make_regexes(numeral) -> Tuple[str, str]:
    """Returns two regular expressions, one that matches chords with a given root and are not applied chords, and one
    that matches chords that are applied to that root.
    """
    return rf"^{numeral}[^iIvV\/]*$", rf"^.+/{numeral}\]?$"


def _merge_subsequent_into_stage(df, *regex, fill_value=".", printout=False):
    """Considers the left-most column as the current stage, the head, and the adjacent column (the first column of the
    tail) as subsequent stage. The items from the latter that match any of the given regular expressions are merged
    into the former by inserting a new column containing these items representing a substage (filled with the
    fill_value). The corresponding rows in the tail, from which items were merged, are rolled to the left and newly
    empty columns at the end of tail are dropped. Then the tail is processed with recursively_merge_adjacent_roots()
    before concatenating it to the substages on the left.
    """
    _, n_cols = df.shape
    if n_cols < 2:
        return df
    substage = itertools.count(start=1)
    column_iloc = df.iloc(axis=1)
    stage_column = column_iloc[0]
    stage = stage_column.name[0]
    head = [stage_column]
    tail = column_iloc[1:]
    right = column_iloc[1]

    def make_mask(series):
        mask = np.zeros_like(right, bool)
        for rgx in regex:
            mask |= series.str.contains(rgx, na=False)
        return mask

    mask = make_mask(right)
    while mask.any():
        new_column = right.where(mask, other=fill_value).rename((stage, next(substage)))
        head.append(new_column)
        tail.loc[mask] = tail.loc[mask].shift(-1, axis=1)
        right = tail.iloc(axis=1)[0]
        mask = make_mask(right)
    tail_curtailed = tail.dropna(how="all", axis=1)
    if printout:
        print(
            f"Added {len(head)} substages to {stage}, reducing the tail from {n_cols} to {tail_curtailed.shape[1]}"
        )
    recursively_merged_tail = recursively_merge_adjacent_roots(
        tail_curtailed, printout=printout
    )
    return pd.concat(head + [recursively_merged_tail], axis=1)


def recursively_merge_adjacent_roots(phrase_data, printout=False):
    """Recursively inserts substage columns by merging labels from later stages into earlier ones when they have
    the same root. Chords applied to the root are also merged."""
    n_rows, n_cols = phrase_data.shape
    if printout:
        print(f"merge_adjacent_roots({n_rows}x{n_cols})")
    if n_cols < 2:
        return phrase_data
    numeral_groups = phrase_data.iloc(axis=1)[0].map(get_numeral, na_action="ignore")
    results = []
    for numeral, df in phrase_data.groupby(numeral_groups):
        regex_a, regex_b = make_regexes(numeral)
        df_curtailed = df.dropna(how="all", axis=1)
        if printout:
            print(f"_merge_subsequent_into_stage({len(df)}/{n_rows}) for {numeral}")
        numeral_df = _merge_subsequent_into_stage(
            df_curtailed, regex_a, regex_b, printout=printout
        )
        results.append(numeral_df)
    return pd.concat(results).sort_index(axis=1)

result = recursively_merge_adjacent_roots(phrase_data)
```