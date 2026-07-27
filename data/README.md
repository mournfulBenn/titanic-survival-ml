# Data

The local files in `raw/` come from Kaggle's
[Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
competition:

- `train.csv`: 891 labelled passenger records;
- `test.csv`: 418 unlabelled passenger records;
- `gender_submission.csv`: Kaggle's example submission format.

The raw CSV files are intentionally excluded from Git. To reproduce the project:

1. Join the competition and accept its rules on Kaggle.
2. Download the competition data from the **Data** tab.
3. Place `train.csv`, `test.csv`, and `gender_submission.csv` in `data/raw/`.

If the Kaggle CLI is installed and authenticated, the equivalent download is:

```powershell
kaggle competitions download -c titanic -p data/raw
Expand-Archive data/raw/titanic.zip -DestinationPath data/raw -Force
```

Generated submission files are model predictions, not copies of the sample
submission.

