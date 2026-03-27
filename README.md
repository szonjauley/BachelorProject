# BachelorProject

# AU and Gaze Analysis Pipeline

This repository contains a full pipeline for preprocessing, feature extraction, aggregation, statistical analysis, and prediction modeling for **AU** (Action Unit) and **gaze** data.

The AU pipeline **must** be run first as the gaze pipeline depends on `xxx_speaker_segments.csv`.

## Repository Structure

At the root of the project, the following folders should exist:

```text
root/
├── output/
│   ├── au/
│   │   └── boxplots/
│   └── gaze/
│       └── boxplots/
└── data/
    ├── depression.csv
    ├── participant_folders/
    └── splits/
        ├── train_split_Depression_AVEC2017.csv
        ├── full_test_split.csv
        └── dev_split_Depression_AVEC2017.csv
```

### Required data files

- `data/depression.csv`
- `data/participant_folders/`
- `data/splits/`
- `data/splits/train_split_Depression_AVEC2017.csv`
- `data/splits/full_test_split.csv`
- `data/splits/dev_split_Depression_AVEC2017.csv`

---

## AU Pipeline

### 1. Create labeled AU segment files
Run:

```bash
python au_split_automation.py
```

This script runs:

- `au_split.py`
- `ellie_participant_split.py`

It creates `xxx_speaker_segments.csv` files and then uses them to generate `xxx_AUs_labeled.csv` files inside each participant folder in `participant_folders/`.

### 2. Create AU aggregation file
Run:

```bash
python au_aggregation.py
```

This script uses:

- `xxx_AUs_labeled.csv`
- `data/depression.csv`

It creates `au_aggregations.csv`, a long-format CSV containing one row per participant. The file is saved in the `data/` folder.

### 3. Check AU normality
Run:

```bash
python au_normality.py
```

This script uses `au_aggregations.csv` to test whether each AU is normally distributed.

Output:

- `au_normality.csv` saved in `output/au/`

### 4. Create AU boxplots
Run:

```bash
python au_boxplot_analysis.py
```

This script uses `au_aggregation.py` output to create boxplots for each AU.

Output:

- `boxplots/` folder with 10 boxplots saved in `output/au/`

### 5. Run AU statistical tests
Run:

```bash
python au_statistical_tests.py
```

This script uses `au_aggregation.py` output to perform statistical tests for each AU.

Output:

- `statistical_tests/` folder saved in `output/au/`

### 6. Run AU permutation test
Run:

```bash
python au_permutation.py
```

This script uses `au_aggregation.py` output to perform a permutation test on the interaction.

Output:

- `au_permutation.csv` saved in `output/au/`

### 7. Run AU regression analysis
Run:

```bash
python au_regression_analysis.py
```

This script uses `au_aggregation.py` output to perform interaction regression on the AU.

Output:

- `au_interaction_regression_mean.csv`
- `au_interaction_regression_std.csv`

Both files are saved in `output/au/`.

### 8. Run AU prediction model
Run:

```bash
python au_prediction_model.py
```

This script uses:

- `train_split_Depression_AVEC2017.csv`
- `full_test_split.csv`
- `dev_split_Depression_AVEC2017.csv`
- `au_aggregations.csv`

Output:

- Best performing **Logistic Regression** and **Random Forest** models for each interaction type:
  - `all`
  - `listening`
  - `speaking`

---

## Gaze Pipeline

### 1. Create labeled gaze segment files
Run:

```bash
python gaze_label.py
```

This script uses:

- `xxx_CLNF_gaze.txt`
- `xxx_speaker_segments.csv`

It creates `xxx_CLNF_gaze_labeled.csv` inside each participant folder in `participant_folders/`.

### 2. Preprocess gaze data
Run:

```bash
python gaze_preprocessing.py
```

This script uses `participant_folders/` and `depression.csv` to create a cleaned labeled gaze file with a confidence threshold of `0.7`.

Output:

- `gaze_cleaned_labeled_0.7.csv` saved in `data/`

### 3. Extract gaze features
Run:

```bash
python gaze_features.py
```

This script uses `gaze_cleaned_labeled_0.7.csv` to create gaze delta features by interaction type.

Outputs:

- `combined_gaze_deltas.csv`
- `listening_gaze_deltas.csv`
- `speaking_gaze_deltas.csv`

All files are saved in `data/`.

### 4. Create gaze aggregation file
Run:

```bash
python gaze_aggregation.py
```

This script uses the three gaze delta files to create a long-format CSV containing each participant’s mean and standard deviation for the `delta_degree` feature.

Output:

- `gaze_aggregation.csv` saved in `data/`

### 5. Check gaze normality
Run:

```bash
python gaze_normality.py
```

This script uses `gaze_aggregation.csv` to test whether the delta feature is normally distributed.

Output:

- `gaze_normality.csv` saved in `output/gaze/`

### 6. Create gaze boxplots
Run:

```bash
python gaze_boxplot_analysis.py
```

This script uses `gaze_aggregation.py` output to create boxplots for the gaze features.

Output:

- `boxplots/` folder saved in `output/gaze/`

### 7. Run gaze statistical tests
Run:

```bash
python gaze_statistical_tests.py
```

This script uses `gaze_aggregation.py` output to perform statistical tests on the gaze delta features.

Output:

- `gaze_stat_test_mean.csv`
- `gaze_stat_test_std.csv`
  
Both files are saved in `output/gaze/`.

### 8. Run gaze permutation test
Run:

```bash
python gaze_permutation.py
```

This script uses `gaze_aggregation.py` output to perform a permutation test on the interaction.

Output:

- `gaze_permutation.csv` saved in `output/gaze/`

### 9. Run gaze regression analysis
Run:

```bash
python gaze_regression_analysis.py
```

This script uses `gaze_aggregation.py` output to perform interaction regression on the gaze features.

Output:

- `gaze_interaction_regression_mean.csv`
- `gaze_interaction_regression_std.csv`

Both files are saved in `gaze/output/`.

### 10. Run gaze prediction model
Run:

```bash
python gaze_prediction_model.py
```

This script uses:

- `train_split_Depression_AVEC2017.csv`
- `full_test_split.csv`
- `dev_split_Depression_AVEC2017.csv`
- `gaze_aggregation.csv`

Output:

- Best performing **Logistic Regression** and **Random Forest** models for each interaction type:
  - `all`
  - `listening`
  - `speaking`

---

## Notes

- Make sure the required data files are placed in the expected locations before running the scripts.

---

## Suggested Run Order

1. `au_split_automation.py`
2. `au_aggregation.py`
3. `au_normality.py`
4. `au_boxplot_analysis.py`
5. `au_statistical_tests.py`
6. `au_permutation.py`
7. `au_regression_analysis.py`
8. `au_prediction_model.py`
9. `gaze_label.py`
10. `gaze_preprocessing.py`
11. `gaze_features.py`
12. `gaze_aggregation.py`
13. `gaze_normality.py`
14. `gaze_boxplot_analysis.py`
15. `gaze_statistical_tests.py`
16. `gaze_permutation.py`
17. `gaze_regression_analysis.py`
18. `gaze_prediction_model.py`

You can also run the gaze pipeline after running the `au_split_automation.py` file.

