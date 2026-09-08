# Training variance in cross-dataset evaluation of ECG classifiers

Code and per-seed results for the paper

> **Significant but not replicable: training variance in cross-dataset evaluation of ECG classifiers**
> Submitted to *Physiological Measurement* (manuscript PMEA-107521).

Four detectors of ECG abnormality are trained on lead II of PTB-XL and transferred, frozen and without adaptation, to the MIT-BIH Arrhythmia Database. Every deep result is replicated under five random seeds, and the whole experiment is repeated with source and target exchanged.

## The finding this repository supports

Cross-dataset results are almost always reported from a single trained model with an evaluation-set confidence interval. That measures one source of uncertainty and omits another — the variance of the training procedure itself.

Here the omitted term was the larger one:

| | |
|:--|:--|
| Single trained pair, autoencoder vs CNN on MIT-BIH | +0.035 AUC, DeLong **p = 7.3 × 10⁻⁷** |
| Same comparison across five seeds | 0.011 ± 0.020 AUC, **p = 0.30** |
| Evaluation-set standard error implied by that DeLong statistic | 0.007 |
| Across-seed standard deviation | 0.020 |

`results/multiseed_runs.json` contains the per-seed numbers behind that table, so the discrepancy can be checked without rerunning anything.

## Layout

```
src/
  preprocess_ptbxl.py       PTB-XL   -> processed/ptbxl_*.npy
  preprocess_mitbih.py      MIT-BIH  -> processed/mitbih_*.npy
  train_baseline_cnn.py     supervised 1D CNN
  train_autoencoder.py      convolutional autoencoder, normal records only
  cross_dataset_eval.py     frozen transfer of both deep models to MIT-BIH
  statistical_analysis.py   stratified bootstrap CIs and paired DeLong tests
  multiseed_experiment.py   five-seed replication, forward direction
  reverse_transfer.py       MIT-BIH -> PTB-XL, patient-disjoint split
  classical_baseline.py     28 interpretable features, logistic regression + random forest
  make_figures.py           figures 1-3 and supplementary figure S1
results/
  multiseed_runs.json       per-seed results as used in the paper
docs/
  mapping_to_paper.md       which script produces which table, figure and number
```

## Requirements

Python 3.9 or newer, and:

```
numpy
scipy
pandas
scikit-learn
torch
matplotlib
wfdb
tqdm
```

`pip install -r requirements.txt` installs these. Everything runs on CPU; no GPU is required.

**A caveat that matters for a paper about replication.** `requirements.txt` gives lower bounds, not the exact versions that produced the published numbers. Pinning would be better. If you are reproducing this work and want an exact environment, run `pip freeze` and compare; small differences in BLAS, PyTorch or scikit-learn versions can move the fourth decimal place of an AUC, which is well inside the seed-to-seed spread this paper is about.

## Data

Neither database is redistributed here. Both are public:

- PTB-XL v1.0.3 — <https://physionet.org/content/ptb-xl/1.0.3/> (DOI [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45))
- MIT-BIH Arrhythmia Database v1.0.0 — <https://physionet.org/content/mitdb/1.0.0/> (DOI [10.13026/C2F305](https://doi.org/10.13026/C2F305))

Download them and place the record files so that the tree looks like:

```
data/
  ptbxl/     ptbxl_database.csv, scp_statements.csv, records100/, records500/
  mitbih/    100.dat, 100.hea, 100.atr, ...
```

The paths `data/ptbxl` and `data/mitbih` are set at the top of the two preprocessing scripts.

## Running it

Run from the repository root, in this order. The scripts communicate through `processed/` and `results/`, which are created on first use.

```bash
python src/preprocess_ptbxl.py        # ~5 min, writes processed/ptbxl_*.npy
python src/preprocess_mitbih.py       # ~2 min, writes processed/mitbih_*.npy

python src/train_baseline_cnn.py      # supervised CNN, writes baseline_cnn.pt
python src/train_autoencoder.py       # autoencoder + Youden threshold on validation
python src/cross_dataset_eval.py      # frozen transfer to MIT-BIH

python src/statistical_analysis.py    # bootstrap CIs, DeLong tests
python src/classical_baseline.py      # feature models, both domains
python src/multiseed_experiment.py    # five seeds, forward direction
python src/reverse_transfer.py        # five seeds, reverse direction
python src/make_figures.py            # figures, from cached results only
```

Two of the long scripts take options:

```bash
python src/multiseed_experiment.py --quick          # 2 epochs, for a smoke test
python src/multiseed_experiment.py --seeds 0 1 2    # a subset of seeds
python src/reverse_transfer.py --quick
python src/classical_baseline.py --quick
```

`multiseed_experiment.py` and `reverse_transfer.py` are resumable: a seed already present in the output JSON is skipped, so an interrupted run can be restarted with the same command.

`make_figures.py` recomputes nothing. It reads the cached results, so figures can be regenerated without retraining.

## Cost

All ten deep trainings together take under an hour on a laptop CPU, because the models are deliberately compact — about 1.1 × 10⁴ parameters for the CNN and 1.8 × 10⁴ for the autoencoder. Feature extraction runs at roughly 1 ms per window. The full study, including the 2000-replicate bootstrap, is a few hours of CPU time.

That number is part of the argument. Replicating a training procedure five times is cheap at this scale, and the paper argues it should be routine.

## A note on the EPS figures

`make_figures.py` writes EPS, TIFF and PNG. Microsoft Arial stores no glyph names in its `post` table, and matplotlib's PostScript backend responds by emitting glyph references that the embedded font does not define — so the EPS renders with **every label silently missing** in Ghostscript, in Distiller and in publishers' submission systems, while looking correct in the PNG. The script therefore passes each EPS through Ghostscript with `-dNoOutputFonts`, which converts the text to vector outlines. If Ghostscript is not on your PATH the script warns and leaves the EPS unconverted; use the TIFF versions in that case.

## Reproducing the paper's numbers

`docs/mapping_to_paper.md` lists every table, figure and quoted statistic in the manuscript against the script and output file that produces it.

## Citing

If you use this code, please cite the paper. Until it has a DOI, cite the manuscript reference:

> Abdukarimov S, Gaibnazarov B, Khaydarov A, Kurbonov S, Kimizbayeva O, Urokov A, Makhmudov M and Alimova Z. Significant but not replicable: training variance in cross-dataset evaluation of ECG classifiers. Submitted to *Physiological Measurement*, manuscript PMEA-107521.

Please also cite the two databases; their citations are on the PhysioNet pages linked above.

## Licence

MIT — see [LICENSE](LICENSE). The databases are not covered by this licence and remain subject to the terms set by their custodians.
