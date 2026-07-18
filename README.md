# BotBuster

BotBuster is a Dash web app that classifies Twitter/X accounts as `human` or
`bot` from account metadata.

## Run the app

```powershell
.\venv\Scripts\python.exe webapp\botbuster_app.py
```

Open http://127.0.0.1:8050/ after the server starts.

You can upload either:

- raw account data with the same columns as `data/raw/twitter_human_bots_dataset.csv`
- engineered data with the columns in `models/feature_columns.pkl`, such as
  `data/processed/demo_accounts.csv`

## Rebuild the model

Use this when dependencies change or the model pickle becomes incompatible with
the installed scikit-learn version.

```powershell
.\venv\Scripts\python.exe scripts\train_model.py
```

The script saves:

- `models/bot_detector_rf.pkl`
- `models/feature_columns.pkl`
- `data/processed/demo_accounts.csv`