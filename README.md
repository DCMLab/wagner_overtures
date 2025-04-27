![Version](https://img.shields.io/github/v/release/DCMLab/wagner_overtures?display_name=tag)
[![DOI](https://zenodo.org/badge/388192508.svg)](https://doi.org/10.5281/zenodo.14997120)
![GitHub repo size](https://img.shields.io/github/repo-size/DCMLab/wagner_overtures)
![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-9cf)


This is a README file for a data repository originating from the [DCML corpus initiative](https://github.com/DCMLab/dcml_corpora)
and serves as welcome page for both 

* the GitHub repo [https://github.com/DCMLab/wagner_overtures](https://github.com/DCMLab/wagner_overtures) and the corresponding
* documentation page [https://dcmlab.github.io/wagner_overtures](https://dcmlab.github.io/wagner_overtures)

For information on how to obtain and use the dataset, please refer to [this documentation page](https://dcmlab.github.io/wagner_overtures/introduction).

When you use (parts of) this dataset in your work, please read and cite the accompanying data report:

_Hentschel, J., Rammos, Y., Neuwirth, M., & Rohrmeier, M. (2025). A corpus and a modular infrastructure for the 
empirical study of (an)notated music. Scientific Data, 12(1), 685. https://doi.org/10.1038/s41597-025-04976-z_

# Richard Wagner – Overtures (A corpus of annotated scores)

Here we have two contrasting Wagner overtures in piano reduction: in Tristan und Isolde, one of his most futuristic
efforts, and in Die Meistersinger von Nürnberg, one of his most traditional. The contrast is all the more interesting in
the context of the knowledge that they were composed at about the same time; their stylistic differences thus reflect a
difference in the themes of their associated operas rather than a development of the composer's technique. In the case
of Meistersinger, our annotations identify the rich layers of granular detail with which Wagner has decorated what are
ostensibly rustic and simple harmonies. Conversely, in Tristan, which famously contains very few resolutions to the
tonic triad, we have quantified just how far Wagner was able to go in delaying harmonic closure, and these annotations
will prove useful in future research modeling extreme harmonic phenomena.

## Getting the data

* download repository as a [ZIP file](https://github.com/DCMLab/wagner_overtures/archive/main.zip)
* download a [Frictionless Datapackage](https://specs.frictionlessdata.io/data-package/) that includes concatenations
  of the TSV files in the four folders (`measures`, `notes`, `chords`, and `harmonies`) and a JSON descriptor:
  * [wagner_overtures.zip](https://github.com/DCMLab/wagner_overtures/releases/latest/download/wagner_overtures.zip)
  * [wagner_overtures.datapackage.json](https://github.com/DCMLab/wagner_overtures/releases/latest/download/wagner_overtures.datapackage.json)
* clone the repo: `git clone https://github.com/DCMLab/wagner_overtures.git` 


## Data Formats

Each piece in this corpus is represented by five files with identical name prefixes, each in its own folder. 
For example, the “Vorspiel” of *Tristan und Isolde* has the following files:

* `MS3/WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia.mscx`: Uncompressed MuseScore 3.6.2 file including the music and annotation labels.
* `notes/WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia.notes.tsv`: A table of all note heads contained in the score and their relevant features (not each of them represents an onset, some are tied together)
* `measures/WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia.measures.tsv`: A table with relevant information about the measures in the score.
* `chords/WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia.chords.tsv`: A table containing layer-wise unique onset positions with the musical markup (such as dynamics, articulation, lyrics, figured bass, etc.).
* `harmonies/WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia.harmonies.tsv`: A table of the included harmony labels (including cadences and phrases) with their positions in the score.

Each TSV file comes with its own JSON descriptor that describes the meanings and datatypes of the columns ("fields") it contains,
follows the [Frictionless specification](https://specs.frictionlessdata.io/tabular-data-resource/),
and can be used to validate and correctly load the described file. 

### Opening Scores

After navigating to your local copy, you can open the scores in the folder `MS3` with the free and open source score
editor [MuseScore](https://musescore.org). Please note that the scores have been edited, annotated and tested with
[MuseScore 3.6.2](https://github.com/musescore/MuseScore/releases/tag/v3.6.2). 
MuseScore 4 has since been released which renders them correctly but cannot store them back in the same format.

### Opening TSV files in a spreadsheet

Tab-separated value (TSV) files are like Comma-separated value (CSV) files and can be opened with most modern text
editors. However, for correctly displaying the columns, you might want to use a spreadsheet or an addon for your
favourite text editor. When you use a spreadsheet such as Excel, it might annoy you by interpreting fractions as
dates. This can be circumvented by using `Data --> From Text/CSV` or the free alternative
[LibreOffice Calc](https://www.libreoffice.org/download/download/). Other than that, TSV data can be loaded with
every modern programming language.

### Loading TSV files in Python

Since the TSV files contain null values, lists, fractions, and numbers that are to be treated as strings, you may want
to use this code to load any TSV files related to this repository (provided you're doing it in Python). After a quick
`pip install -U ms3` (requires Python 3.10 or later) you'll be able to load any TSV like this:

```python
import ms3

labels = ms3.load_tsv("harmonies/WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia.harmonies.tsv")
notes = ms3.load_tsv("notes/WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia.notes.tsv")
```


## Version history

See the [GitHub releases](https://github.com/DCMLab/wagner_overtures/releases).

## Questions, Suggestions, Corrections, Bug Reports

Please [create an issue](https://github.com/DCMLab/wagner_overtures/issues) and/or feel free to fork and submit pull requests.

## Cite as

> Hentschel, J., Rammos, Y., Neuwirth, M., & Rohrmeier, M. (2025). A corpus and a modular infrastructure for the empirical study of (an)notated music. Scientific Data, 12(1), 685. https://doi.org/10.1038/s41597-025-04976-z

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

![cc-by-nc-sa-image](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)

## Overview
|                        file_name                         |measures|labels|standard| annotators |
|----------------------------------------------------------|-------:|-----:|--------|------------|
|WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia    |     111|   359|2.1.0   |Adrian Nagel|
|WWV096-Meistersinger_01_Vorspiel-Prelude_SchottKleinmichel|     222|  1074|2.1.0   |Adrian Nagel|


*Overview table automatically updated using [ms3](https://ms3.readthedocs.io/).*
