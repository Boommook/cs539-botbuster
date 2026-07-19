import numpy as np
import pandas as pd


def engineer_features(df, add_log_features=True):
    """
    Clean, remove, and add features to the dataset.
    Returns a copy of the dataset containing the newly engineered features.
    """
    model_df = df.copy()

    if "account_type" in model_df.columns:
        model_df["account_is_bot"] = (
            model_df["account_type"]
            .map({
                "human": 0,
                "bot": 1
            })
            .astype(int)
        )

    # ---------------------------------------------------------
    # Clean/remove features
    # ---------------------------------------------------------
    # remove columns that are identifiers, dupes, or unlikely to generalize
    model_df = model_df.drop(
        columns=[
            "Unnamed: 0",
            "id",
            "created_at",
            "profile_background_image_url",
            "profile_image_url",
            "location",
            "lang"
        ],
        errors="ignore"
    )

    # ---------------------------------------------------------
    # Missing-value features
    # ---------------------------------------------------------
    description = df.get("description", pd.Series("", index=df.index)).fillna("")
    location = df.get("location", pd.Series("", index=df.index))
    language = df.get("lang", pd.Series("", index=df.index))
    screen_name = df.get("screen_name", pd.Series("", index=df.index)).fillna("")

    model_df["description_missing"] = description.eq("")
    model_df["location_unknown"] = (
        location.isna()
        | location.fillna("").str.lower().eq("unknown")
        | location.fillna("").eq("")
    )
    model_df["language_missing"] = language.isna() | language.fillna("").eq("")

    # Description features
    model_df["description_length"] = description.str.len()
    model_df["description_word_count"] = description.str.split().str.len()
    model_df["description_mention_count"] = description.str.count(r"@\w+")
    model_df["description_hashtag_count"] = description.str.count(r"#\w+")
    model_df["description_digit_count"] = description.str.count(r"\d")
    model_df["description_exclamation_count"] = description.str.count("!")

    # Screen-name features
    model_df["screen_name_length"] = screen_name.str.len()
    model_df["screen_name_digit_count"] = screen_name.str.count(r"\d")
    model_df["screen_name_underscore_count"] = screen_name.str.count("_")

    # ---------------------------------------------------------
    # Drop raw text columns now that features are extracted
    # ---------------------------------------------------------
    model_df = model_df.drop(
        columns=["description", "screen_name"],
        errors="ignore"
    )

    # ---------------------------------------------------------
    # Log-transform skewed numeric features
    # ---------------------------------------------------------
    if add_log_features:
        skewed_cols = [
            "followers_count",
            "friends_count",
            "favourites_count",
            "statuses_count",
            "average_tweets_per_day",
        ]
        for col in skewed_cols:
            model_df[f"{col}_log"] = np.log1p(model_df[col])

    # ---------------------------------------------------------
    # Cast boolean columns to int (0/1) for modeling
    # ---------------------------------------------------------
    bool_cols = model_df.select_dtypes(include="bool").columns
    model_df[bool_cols] = model_df[bool_cols].astype(int)

    ## return completed dataframe
    return model_df
