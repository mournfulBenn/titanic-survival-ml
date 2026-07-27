"""Train, evaluate, and export a reproducible Titanic classifier."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import TitanicFeatureEngineer
from src.validation import passenger_groups


RANDOM_STATE = 42
TARGET = "Survived"

CORE_NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "TicketGroupSize",
    "FarePerPerson",
    "NameLength",
]
CORE_CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "CabinDeck",
    "TicketPrefix",
]
NUMERIC_FEATURES = CORE_NUMERIC_FEATURES + [
    "FarePerFamilyMember",
    "CabinKnown",
    "IsChild",
    "IsMother",
]
CATEGORICAL_FEATURES = CORE_CATEGORICAL_FEATURES + [
    "FamilySizeCategory",
]


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_model(
    model,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", TitanicFeatureEngineer()),
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            ("model", model),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    return {
        "Dummy baseline": build_model(DummyClassifier(strategy="most_frequent")),
        "Logistic regression": build_model(
            LogisticRegression(
                max_iter=2_000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
        ),
        "Random forest": build_model(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=7,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "Extra trees": build_model(
            ExtraTreesClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "Gradient boosting": build_model(
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=3,
                subsample=0.9,
                random_state=RANDOM_STATE,
            )
        ),
    }


def feature_ablation_models() -> dict[str, Pipeline]:
    return {
        "Random forest (core)": build_model(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=7,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            CORE_NUMERIC_FEATURES,
            CORE_CATEGORICAL_FEATURES,
        ),
        "Random forest (enhanced)": candidate_models()["Random forest"],
        "Extra trees (core)": build_model(
            ExtraTreesClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            CORE_NUMERIC_FEATURES,
            CORE_CATEGORICAL_FEATURES,
        ),
        "Extra trees (enhanced)": candidate_models()["Extra trees"],
    }


def evaluate_models(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    rows: list[dict[str, float | str]] = []

    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            groups=groups,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        row: dict[str, float | str] = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        row["fit_time_mean_seconds"] = float(np.mean(scores["fit_time"]))
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["accuracy_mean", "roc_auc_mean"], ascending=False)
        .reset_index(drop=True)
    )


def select_robust_model(
    comparison: pd.DataFrame,
    group_comparison: pd.DataFrame,
    evaluation_count: int,
) -> tuple[str, dict[str, object]]:
    """Apply a one-standard-error shortlist, then prefer group robustness."""

    eligible = comparison[comparison["model"] != "Dummy baseline"].copy()
    best_row = eligible.sort_values("accuracy_mean", ascending=False).iloc[0]
    standard_error = float(best_row["accuracy_std"]) / np.sqrt(evaluation_count)
    threshold = float(best_row["accuracy_mean"]) - standard_error
    shortlist = eligible[eligible["accuracy_mean"] >= threshold]

    group_scores = group_comparison.set_index("model")["accuracy_mean"]
    shortlist = shortlist.assign(
        group_accuracy=shortlist["model"].map(group_scores)
    ).sort_values(
        ["group_accuracy", "accuracy_mean", "roc_auc_mean"],
        ascending=False,
    )
    selected = str(shortlist.iloc[0]["model"])
    rationale = {
        "rule": (
            "Best group-aware accuracy among non-baseline models within one "
            "standard error of the best repeated-CV accuracy"
        ),
        "best_repeated_cv_accuracy": float(best_row["accuracy_mean"]),
        "best_model_standard_error": standard_error,
        "shortlist_threshold": threshold,
        "shortlisted_models": shortlist["model"].tolist(),
    }
    return selected, rationale


def save_eda_figures(train: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    missing_rate = train.isna().mean().sort_values(ascending=False)
    missing_rate = missing_rate[missing_rate > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    missing_rate.mul(100).sort_values().plot.barh(ax=ax, color="#3B82F6")
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("")
    ax.set_title("Missingness in the training data")
    fig.tight_layout()
    fig.savefig(figures_dir / "missing_values.png", dpi=180)
    plt.close(fig)

    survival = train.groupby(["Sex", "Pclass"])[TARGET].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    survival.plot.bar(ax=ax, color=["#0EA5E9", "#F59E0B", "#EF4444"])
    ax.set_ylabel("Survival rate")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.set_title("Survival rate by sex and passenger class")
    ax.legend(title="Passenger class")
    fig.tight_layout()
    fig.savefig(figures_dir / "survival_by_sex_and_class.png", dpi=180)
    plt.close(fig)


def save_confusion_matrix(matrix: np.ndarray, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, str(value), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Not survived", "Survived"],
        yticklabels=["Not survived", "Survived"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Out-of-fold confusion matrix",
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def grouped_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv: StratifiedGroupKFold,
) -> pd.DataFrame:
    """Estimate raw-column importance only on held-out passenger groups."""

    fold_importances: list[np.ndarray] = []
    for fold, (train_indices, validation_indices) in enumerate(
        cv.split(X, y, groups)
    ):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X.iloc[train_indices], y.iloc[train_indices])
        result = permutation_importance(
            fold_pipeline,
            X.iloc[validation_indices],
            y.iloc[validation_indices],
            scoring="accuracy",
            n_repeats=5,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )
        fold_importances.append(result.importances_mean)

    values = np.vstack(fold_importances)
    return (
        pd.DataFrame(
            {
                "source_feature": X.columns,
                "importance_mean": values.mean(axis=0),
                "importance_std_across_folds": values.std(axis=0),
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def save_feature_importance(
    importance: pd.DataFrame, destination: Path, top_n: int = 10
) -> None:
    plot_data = importance.head(top_n).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        plot_data["source_feature"],
        plot_data["importance_mean"],
        xerr=plot_data["importance_std_across_folds"],
        color="#14B8A6",
        alpha=0.9,
    )
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_xlabel("Decrease in held-out accuracy after permutation")
    ax.set_ylabel("")
    ax.set_title("Group-aware permutation importance")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def validate_data(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_required = {
        "PassengerId",
        TARGET,
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked",
    }
    test_required = train_required.difference({TARGET})
    if missing := train_required.difference(train.columns):
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    if missing := test_required.difference(test.columns):
        raise ValueError(f"Test data is missing columns: {sorted(missing)}")
    if train["PassengerId"].duplicated().any() or test["PassengerId"].duplicated().any():
        raise ValueError("PassengerId must be unique within each dataset")
    if not set(train[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("Survived must be a binary target")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.report_dir / "figures"

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    validate_data(train, test)

    X = train.drop(columns=[TARGET])
    y = train[TARGET].astype(int)
    models = candidate_models()
    repeated_cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=RANDOM_STATE
    )
    comparison = evaluate_models(models, X, y, cv=repeated_cv)
    comparison.to_csv(args.report_dir / "model_comparison.csv", index=False)

    groups = passenger_groups(X)
    grouped_cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    group_comparison = evaluate_models(
        models, X, y, cv=grouped_cv, groups=groups
    )
    group_comparison.to_csv(
        args.report_dir / "group_validation_comparison.csv", index=False
    )

    ablation = evaluate_models(
        feature_ablation_models(), X, y, cv=repeated_cv
    )
    ablation.to_csv(args.report_dir / "feature_ablation.csv", index=False)
    group_ablation = evaluate_models(
        feature_ablation_models(),
        X,
        y,
        cv=grouped_cv,
        groups=groups,
    )
    group_ablation.to_csv(
        args.report_dir / "feature_ablation_group.csv", index=False
    )

    best_model_name, selection_rationale = select_robust_model(
        comparison, group_comparison, evaluation_count=15
    )
    best_pipeline = models[best_model_name]
    importance = grouped_permutation_importance(
        best_pipeline, X, y, groups, grouped_cv
    )
    importance.to_csv(
        args.report_dir / "permutation_importance.csv", index=False
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    out_of_fold_predictions = cross_val_predict(
        best_pipeline, X, y, cv=cv, method="predict", n_jobs=-1
    )
    report = classification_report(
        y, out_of_fold_predictions, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y, out_of_fold_predictions)

    best_pipeline.fit(X, y)
    test_predictions = best_pipeline.predict(test).astype(int)
    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"].astype(int), TARGET: test_predictions}
    )
    if len(submission) != len(test) or not set(submission[TARGET]).issubset({0, 1}):
        raise RuntimeError("Generated submission failed validation")

    submission.to_csv(args.output_dir / "submission_candidate.csv", index=False)
    joblib.dump(best_pipeline, args.output_dir / "titanic_pipeline.joblib")
    with (args.report_dir / "classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    summary = {
        "random_state": RANDOM_STATE,
        "cross_validation": "5-fold stratified, shuffled, repeated 3 times",
        "stress_test_validation": (
            "5-fold stratified group validation using connected family-ticket groups"
        ),
        "selection_metric": "one-standard-error shortlist plus group robustness",
        "selection_rationale": selection_rationale,
        "selected_model": best_model_name,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "positive_target_rate": float(y.mean()),
        "candidate_submission_file": "outputs/submission_candidate.csv",
        "selected_model_metrics": comparison.loc[
            comparison["model"] == best_model_name
        ].iloc[0].to_dict(),
        "selected_model_group_metrics": group_comparison.loc[
            group_comparison["model"] == best_model_name
        ].iloc[0].to_dict(),
        "top_source_features": importance.head(5)["source_feature"].tolist(),
    }
    with (args.report_dir / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_eda_figures(train, figures_dir)
    save_confusion_matrix(matrix, figures_dir / "confusion_matrix.png")
    save_feature_importance(
        importance, figures_dir / "permutation_importance.png"
    )

    display_columns = [
        "model",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "roc_auc_mean",
    ]
    print(comparison[display_columns].to_string(index=False, float_format="%.4f"))
    print("\nGroup-aware validation stress test:")
    print(
        group_comparison[display_columns].to_string(
            index=False, float_format="%.4f"
        )
    )
    print("\nFeature ablation:")
    print(ablation[display_columns].to_string(index=False, float_format="%.4f"))
    print("\nFeature ablation under group-aware validation:")
    print(
        group_ablation[display_columns].to_string(
            index=False, float_format="%.4f"
        )
    )
    print(f"\nSelected model: {best_model_name}")
    print(f"Submission saved to: {args.output_dir / 'submission_candidate.csv'}")


if __name__ == "__main__":
    main()
