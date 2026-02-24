import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# ======================================================
# CONTROL SWITCHES
# ======================================================
RUN_TEST = False
USE_REDUCED_FEATURES = True   # True = only mean + std


# ======================================================
# PATHS
# ======================================================
ALL_ALL_PATH = "all_processed_data/all_all/AU_stats_wide_person_level_all_all_0.7.csv"
LISTENING_ALL_PATH = "all_processed_data/listening_all/AU_stats_wide_person_level_listening_all_0.7.csv"
SPEAKING_ALL_PATH = "all_processed_data/speaking_all/AU_stats_wide_person_level_speaking_all_0.7.csv"

TRAIN_SPLIT = "all_processed_data/train_split_Depression_AVEC2017.csv"
DEV_SPLIT   = "all_processed_data/dev_split_Depression_AVEC2017.csv"
TEST_SPLIT  = "all_processed_data/full_test_split.csv"


# ======================================================
# LOAD SPLITS
# ======================================================
train_df = pd.read_csv(TRAIN_SPLIT)
dev_df   = pd.read_csv(DEV_SPLIT)
test_df  = pd.read_csv(TEST_SPLIT)

train_dev_df = pd.concat([train_df, dev_df], axis=0)

train_ids = train_dev_df["Participant_ID"].values
test_ids = test_df["Participant_ID"].values


# ======================================================
# CROSS VALIDATION
# ======================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ======================================================
# EXPERIMENT FUNCTION
# ======================================================
def run_experiment(feature_path, group_name):

    print(f"\n================ {group_name} ================")

    df_features = pd.read_csv(feature_path)

    # FILTER DATA
    train_features = df_features[df_features["person_ID"].isin(train_ids)]
    test_features  = df_features[df_features["person_ID"].isin(test_ids)]

    train_merged = train_features.merge(
        train_dev_df[["Participant_ID", "PHQ8_Binary"]],
        left_on="person_ID",
        right_on="Participant_ID"
    )

    test_merged = test_features.merge(
        test_df[["Participant_ID", "PHQ8_Binary"]],
        left_on="person_ID",
        right_on="Participant_ID"
    )

    # =========================
    # FEATURE SELECTION
    # =========================
    if USE_REDUCED_FEATURES:
        print("Using REDUCED feature set (mean + std)")
        feature_cols = [
            col for col in train_merged.columns
            if col.endswith("_mean") or col.endswith("_std")
        ]
    else:
        print("Using FULL feature set")
        feature_cols = train_merged.drop(
            columns=["person_ID", "Participant_ID", "PHQ8_Binary"]
        ).columns

    X_train = train_merged[feature_cols].values
    y_train = train_merged["PHQ8_Binary"].values

    X_test = test_merged[feature_cols].values
    y_test = test_merged["PHQ8_Binary"].values


    # =========================
    # LOGISTIC REGRESSION
    # =========================
    log_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=10000, class_weight="balanced"))
    ])

    log_param_grid = {
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }

    log_grid = GridSearchCV(
        log_pipe,
        log_param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    log_grid.fit(X_train, y_train)

    print("\nBest Logistic Regression:")
    print("Params:", log_grid.best_params_)
    print("CV F1:", log_grid.best_score_)


    # =========================
    # RANDOM FOREST
    # =========================
    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        max_features=None
    )

    rf_param_grid = {
        "n_estimators": [100, 500, 1000],
        "max_depth": [None, 2, 4, 6, 8, 10],
        "min_samples_split": [None, 2, 4, 6, 8, 10],
        "min_samples_leaf": [None, 2, 4, 6, 8, 10]
    }

    rf_grid = GridSearchCV(
        rf,
        rf_param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)

    print("\nBest Random Forest:")
    print("Params:", rf_grid.best_params_)
    print("CV F1:", rf_grid.best_score_)


    # ======================================================
    # OPTIONAL TEST EVALUATION
    # ======================================================
    if RUN_TEST:
        print("\n🔎 Running FINAL TEST evaluation...")

        best_log = log_grid.best_estimator_
        best_rf = rf_grid.best_estimator_

        best_log.fit(X_train, y_train)
        best_rf.fit(X_train, y_train)

        log_preds = best_log.predict(X_test)
        rf_preds = best_rf.predict(X_test)

        print("\nLogistic Regression (TEST)")
        print("Accuracy:", accuracy_score(y_test, log_preds))
        print("F1:", f1_score(y_test, log_preds))

        print("\nRandom Forest (TEST)")
        print("Accuracy:", accuracy_score(y_test, rf_preds))
        print("F1:", f1_score(y_test, rf_preds))


    return log_grid, rf_grid


# ======================================================
# RUN ALL GROUPS
# ======================================================
log_combined, rf_combined = run_experiment(ALL_ALL_PATH, "COMBINED")
log_listening, rf_listening = run_experiment(LISTENING_ALL_PATH, "LISTENING")
log_speaking, rf_speaking = run_experiment(SPEAKING_ALL_PATH, "SPEAKING")