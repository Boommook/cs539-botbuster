from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.feature_engineering import engineer_features


RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "twitter_human_bots_dataset.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "bot_detector_rf.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
DEMO_DATA_PATH = PROCESSED_DATA_DIR / "demo_accounts.csv"


def main():
    df = pd.read_csv(RAW_DATA_PATH)
    model_df = engineer_features(df, add_log_features=False)

    x = model_df.drop(columns=["account_type", "account_is_bot"])
    y = model_df["account_is_bot"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=["human", "bot"]))
    print(confusion_matrix(y_test, y_pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns = x_train.columns.tolist()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

    demo_df = x_test.copy()
    demo_df["true_label"] = df.loc[x_test.index, "account_type"]
    demo_df["screen_name"] = df.loc[x_test.index, "screen_name"]
    demo_df["twitter_id"] = df.loc[x_test.index, "id"]
    demo_df.to_csv(DEMO_DATA_PATH, index=True)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved feature columns to {FEATURE_COLUMNS_PATH}")
    print(f"Saved demo data to {DEMO_DATA_PATH}")


if __name__ == "__main__":
    main()
