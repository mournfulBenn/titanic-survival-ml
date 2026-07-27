# GitHub Release Checklist

## Recommended repository settings

- **Repository name:** `titanic-survival-ml`
- **Visibility:** Public
- **Description:** Leakage-aware Titanic ML pipeline with group-aware validation,
  model interpretation, and reproducible Kaggle results.
- **Topics:** `python`, `machine-learning`, `data-science`, `kaggle`, `titanic`,
  `scikit-learn`, `feature-engineering`, `cross-validation`,
  `model-interpretability`

## Before the first commit

- [x] Confirm the Git commit display name: `GYX`.
- [x] Confirm the Git commit email: `gx22@uw.edu`.
- [x] Choose the MIT License.
- [ ] Review `README.md` and `README.zh-CN.md`.
- [ ] Confirm raw Kaggle CSV files are ignored.
- [ ] Confirm the unedited student-era script is ignored.
- [ ] Run unit tests and dependency checks.

## Suggested commit history

1. `feat: build leakage-safe Titanic training pipeline`
2. `feat: add group-aware validation and model selection`
3. `docs: add bilingual project documentation and model card`
4. `ci: add automated unit tests`

Because the repository has not yet been committed, this history can be created
cleanly without rewriting any published commits.

## After publishing

- [ ] Verify the GitHub Actions workflow passes.
- [ ] Add the repository URL to the resume project entry.
- [ ] Pin the repository on the GitHub profile.
- [ ] Add the Kaggle submission link if the profile is intended to be public.
- [ ] Check all images and relative links on GitHub.
