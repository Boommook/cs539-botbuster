# X Bot Detector: BotBuster

BotBuster is a Dash-based machine learning dashboard for detecting whether
Twitter/X accounts are likely human-operated or automated bot accounts. The
application uses account metadata, profile-completeness signals, posting
activity, and engineered text features to classify accounts as `human` or `bot`.

## Project Features

- About page with project description and team members
- Manual prediction form for entering account feature values
- CSV or Excel batch prediction workflow
- Model explainability through Random Forest feature importance
- Model diagnostics including accuracy, precision, recall, F1 score, confusion
  matrix, ROC curve, and holdout predictions

## Project Structure

```
cs539-botbuster/
  data/
    raw/
      twitter_human_bots_dataset.csv
    processed/
      demo_accounts.csv
  models/
    bot_detector_rf.pkl
    feature_columns.pkl
  scripts/
    train_model.py
  utils/
    feature_engineering.py
  webapp/
    assets/
      styles.css
    html/
      config.json
      index.html
    botbuster_app.py
  requirements.txt
  .python-version
```

## Local Setup

Clone the GitHub repository, move into the project folder, create a virtual
environment, and install the project dependencies from `requirements.txt`.

```powershell
git clone https://github.com/Boommook/cs539-botbuster.git
cd cs539-botbuster
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

The `requirements.txt` file installs the Python packages required to run the
Dash dashboard, load the trained model, process uploaded account data, and
generate the model diagnostic visualizations.

## Run Locally

After installing the dependencies, start the dashboard from the `webapp`
directory.

```powershell
cd webapp
python botbuster_app.py
```

Open the dashboard at [http://127.0.0.1:8050/].

## Input Data

The batch prediction page accepts:

- raw account data with the same columns as
  `data/raw/twitter_human_bots_dataset.csv`
- engineered data with the columns saved in `models/feature_columns.pkl`, such
  as `data/processed/demo_accounts.csv`

## Rebuild The Model

Use the training script when dependencies change, the model needs to be
regenerated, or the saved model becomes incompatible with the installed
scikit-learn version.

```powershell
cd scripts
python train_model.py
```

The script saves:

- `models/bot_detector_rf.pkl`
- `models/feature_columns.pkl`
- `data/processed/demo_accounts.csv`

## Live Website

The public website is available at
[https://cs539-botbuster.onrender.com/](https://cs539-botbuster.onrender.com/).

## Notes

- Do not commit local virtual environments such as `venv/` or `.venv/`.
- Do not commit local agent or editor state such as `.agents/`.
- The trained model and feature-column files are required for the deployed app
  to run.
