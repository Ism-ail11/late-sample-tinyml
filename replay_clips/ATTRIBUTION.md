# Accelerometer fixture attribution

These fixtures contain short excerpts from two CC BY 4.0 datasets distributed by the UCI Machine Learning Repository. They preserve the original tri-axial acceleration values and per-clip activity labels. The extraction selects held-out subjects and renames the selected segments. See [the CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

| File | Source | Selection |
| --- | --- | --- |
| `wisdm.npz` | Gary Weiss, *WISDM Smartphone and Smartwatch Activity and Biometrics Dataset*, [DOI 10.24432/C5HK59](https://doi.org/10.24432/C5HK59) | 72 clips of 120 smartwatch accelerometer samples, four per activity |
| `mhealth.npz` | Oresti Banos, Rafael Garcia, and Alejandro Saez, *MHEALTH*, [DOI 10.24432/C5TW22](https://doi.org/10.24432/C5TW22) | 48 clips of 120 right-lower-arm accelerometer samples, four per activity |

See [the dataset description](https://archive.ics.uci.edu/ml/machine-learning-databases/00507/WISDM-dataset-description.pdf) and [mHealthDroid](https://doi.org/10.1007/978-3-319-13105-4_14) for the respective recording protocols. The original archives are downloaded by `tinyml.data` when retraining or rebuilding the fixtures.

The six files in `folds/` are a second selection of excerpts from the same sources under the same license. The three outer test folds cover 51 WISDM and 10 MHEALTH participants. The fixtures preserve each selected clip's 120 original accelerometer readings, subject, activity, and segment offset. `python -m tinyml.folds` reproduces the selection.
