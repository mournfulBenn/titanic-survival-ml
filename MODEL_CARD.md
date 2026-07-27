# Model Card: Titanic Survival Classifier

## Model overview

- **Task:** binary classification of passenger survival
- **Selected estimator:** Extra Trees classifier
- **Training rows:** 891
- **Inference rows:** 418 competition test records
- **Selection rule:** one-standard-error shortlist followed by the highest
  family-ticket-group-aware validation accuracy
- **Primary implementation:** `src/train.py`

## Intended use

This model is an educational and portfolio artifact demonstrating reproducible
tabular machine learning, leakage-aware validation, and model interpretation.
It is not intended for safety-critical, historical, demographic, insurance, or
person-level decision making.

## Inputs and features

The pipeline consumes the original Titanic competition columns. Learned
preprocessing is fitted inside each validation fold. Derived features include
title, family size, travelling alone, ticket-group size, fare per passenger,
cabin deck, ticket prefix, name length, child status, and mother status.

The target is never used to impute age or construct predictor values.

## Evaluation

| Measurement | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Repeated stratified CV | 0.8279 +/- 0.0169 | 0.7640 | 0.8797 |
| Family-ticket group CV | 0.8238 +/- 0.0268 | 0.7582 | 0.8658 |
| Kaggle v2 leaderboard | 0.78229 | Not available | Not available |

The Kaggle score and local cross-validation are reported separately because they
measure performance on different labels and splits.

## Interpretation

Permutation importance is computed on held-out family-ticket groups. Sex is the
strongest source feature, followed by name-derived information, family
relationships, and passenger class. Importance values describe predictive use
within this historical dataset; they do not establish causal effects.

## Limitations

- The dataset is small and validation estimates have meaningful variance.
- Related passengers violate the independent-and-identically-distributed
  assumption; group-aware validation reduces but does not eliminate this issue.
- Historical survival patterns encode social conditions and evacuation
  practices from a specific disaster.
- Kaggle leaderboard feedback can encourage test-set adaptation, so only two
  scored submissions are used as external evidence.
- Passenger names, sex, and travel class are sensitive or proxy attributes in
  many modern decision contexts.

## Reproducibility

The project fixes random seeds, records package requirements, keeps preprocessing
inside the model pipeline, saves all evaluation tables, and fingerprints scored
submission files with SHA-256. Unit tests run locally and in GitHub Actions.

