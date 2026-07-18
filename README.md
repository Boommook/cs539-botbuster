# X Bot Detector: BotBuster

BotBuster is a Dash-based machine learning dashboard for detecting whether
Twitter/X accounts are likely human-operated or automated bot accounts. The
application uses account metadata, profile-completeness signals, posting
activity, and engineered text features to classify accounts as `human` or `bot`.

## Live Website

The public website is available at:

```text
https://cs539-botbuster.onrender.com/
```

## Project Features

- About page with project description and team members
- Manual prediction form for entering account feature values
- CSV or Excel batch prediction workflow
- Model explainability through Random Forest feature importance
- Model diagnostics including accuracy, precision, recall, F1 score, confusion
  matrix, ROC curve, and holdout predictions

## Project Structure

```text
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

Create and activate a virtual environment, then install the project
dependencies.

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

```powershell
.\venv\Scripts\python.exe webapp\botbuster_app.py
```

Open the dashboard at:

```text
http://127.0.0.1:8050/
```

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
.\venv\Scripts\python.exe scripts\train_model.py
```

The script saves:

- `models/bot_detector_rf.pkl`
- `models/feature_columns.pkl`
- `data/processed/demo_accounts.csv`

## Deploy On Render

Create a Render **Web Service** connected to the GitHub repository.

Recommended Render settings:

```text
Language: Python 3
Branch: main
Build Command: pip install -r requirements.txt
Start Command: gunicorn webapp.botbuster_app:server --bind 0.0.0.0:$PORT
```

The repository includes `.python-version` so Render can use the intended Python
runtime.

## Notes

- Do not commit local virtual environments such as `venv/` or `.venv/`.
- Do not commit local agent or editor state such as `.agents/`.
- The trained model and feature-column files are required for the deployed app
  to run.
