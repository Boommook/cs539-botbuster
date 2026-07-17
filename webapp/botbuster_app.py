from pathlib import Path
import base64
import datetime
import io
import os
import sys

import dash_ag_grid as dag
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.feature_engineering import engineer_features


MODEL_PATH = ROOT_DIR / "models" / "bot_detector_rf.pkl"
FEATURE_COLUMNS_PATH = ROOT_DIR / "models" / "feature_columns.pkl"
DEMO_DATA_PATH = ROOT_DIR / "data" / "processed" / "demo_accounts.csv"

RAW_INPUT_COLUMNS = [
    "created_at",
    "default_profile",
    "default_profile_image",
    "description",
    "favourites_count",
    "followers_count",
    "friends_count",
    "geo_enabled",
    "id",
    "lang",
    "location",
    "profile_background_image_url",
    "profile_image_url",
    "screen_name",
    "statuses_count",
    "verified",
    "average_tweets_per_day",
    "account_age_days",
]

BOOLEAN_COLUMNS = {
    "default_profile",
    "default_profile_image",
    "geo_enabled",
    "verified",
    "description_missing",
    "location_unknown",
    "language_missing",
}

PAGE_TITLES = {
    "about": "About Project",
    "manual": "Manual Prediction",
    "batch": "CSV Batch Prediction",
    "explain": "Model Explainability",
    "diagnostics": "Model Diagnostics",
}

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
bot_class_index = list(model.classes_).index(1) if hasattr(model, "classes_") and 1 in model.classes_ else 1

demo_df = pd.read_csv(DEMO_DATA_PATH) if DEMO_DATA_PATH.exists() else pd.DataFrame(columns=feature_columns)
manual_defaults = {
    column: int(demo_df[column].mode().iloc[0]) if column in BOOLEAN_COLUMNS and column in demo_df else 0
    for column in feature_columns
}
for column in feature_columns:
    if column in BOOLEAN_COLUMNS:
        continue
    manual_defaults[column] = float(demo_df[column].median()) if column in demo_df else 0.0

app = Dash(
    __name__,
    title="BotBuster",
    assets_folder=str(ROOT_DIR / "webapp" / "assets"),
    suppress_callback_exceptions=True,
)
server = app.server


def display_name(column):
    return column.replace("_", " ").title()


def metric_card(label, value, tone="blue"):
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
        ],
        className=f"metric-card metric-{tone}",
    )


def page_title(title):
    return html.Div(
        [
            html.Div("BotBuster", className="page-eyebrow"),
            html.H2(title, className="page-title"),
        ],
        className="page-heading",
    )


def read_uploaded_file(contents, filename):
    _content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    filename_lower = filename.lower()

    if filename_lower.endswith(".csv"):
        return pd.read_csv(io.StringIO(decoded.decode("utf-8-sig")))
    if filename_lower.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(decoded))

    raise ValueError("Unsupported file type. Upload a CSV or Excel file.")


def coerce_feature_frame(model_df):
    clean_df = model_df.copy()
    for column in feature_columns:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce").fillna(0)
    return clean_df[feature_columns]


def prepare_model_input(read_df):
    if set(feature_columns).issubset(read_df.columns):
        return coerce_feature_frame(read_df[feature_columns])

    missing_raw_columns = [column for column in RAW_INPUT_COLUMNS if column not in read_df.columns]
    if missing_raw_columns:
        missing_preview = ", ".join(missing_raw_columns[:6])
        if len(missing_raw_columns) > 6:
            missing_preview += ", ..."
        raise ValueError(f"Missing required account columns: {missing_preview}")

    model_df = engineer_features(read_df[RAW_INPUT_COLUMNS], add_log_features=False)
    missing_model_columns = [column for column in feature_columns if column not in model_df.columns]
    if missing_model_columns:
        raise ValueError(f"Feature engineering did not create: {', '.join(missing_model_columns)}")

    return coerce_feature_frame(model_df)


def predict_features(model_df):
    predictions = model.predict(model_df)
    labels = np.array(["bot" if int(prediction) == 1 else "human" for prediction in predictions])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_df)[:, bot_class_index]
    else:
        probabilities = predictions.astype(float)

    return labels, probabilities


def add_predictions(read_df):
    model_df = prepare_model_input(read_df)
    result_df = read_df.copy()
    labels, probabilities = predict_features(model_df)

    result_df["bot_detected"] = labels
    result_df["bot_probability"] = (probabilities * 100).round(2)

    priority_columns = [
        "bot_detected",
        "bot_probability",
        "true_label",
        "screen_name",
        "twitter_id",
        "id",
    ]
    ordered_columns = [column for column in priority_columns if column in result_df.columns]
    ordered_columns += [column for column in result_df.columns if column not in ordered_columns]
    return result_df[ordered_columns]


def feature_importance_frame():
    importances = getattr(model, "feature_importances_", np.zeros(len(feature_columns)))
    frame = pd.DataFrame({"feature": feature_columns, "importance": importances})
    return frame.sort_values("importance", ascending=False)


def feature_importance_figure(limit=15):
    frame = feature_importance_frame().head(limit).sort_values("importance")
    fig = go.Figure(
        go.Bar(
            x=frame["importance"],
            y=[display_name(value) for value in frame["feature"]],
            orientation="h",
            marker={"color": "#2563eb"},
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 18, "b": 36},
        height=470,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Importance",
        yaxis_title="",
    )
    return fig


def local_signal_figure(input_df):
    if demo_df.empty:
        baseline = pd.Series(0, index=feature_columns)
        spread = pd.Series(1, index=feature_columns)
    else:
        baseline = demo_df[feature_columns].median(numeric_only=True).reindex(feature_columns).fillna(0)
        spread = demo_df[feature_columns].std(numeric_only=True).reindex(feature_columns).replace(0, 1).fillna(1)

    importances = feature_importance_frame().set_index("feature")["importance"].reindex(feature_columns).fillna(0)
    scores = ((input_df.iloc[0] - baseline).abs() / spread) * importances
    frame = (
        pd.DataFrame({"feature": feature_columns, "signal": scores.values})
        .sort_values("signal", ascending=False)
        .head(10)
        .sort_values("signal")
    )

    fig = go.Figure(
        go.Bar(
            x=frame["signal"],
            y=[display_name(value) for value in frame["feature"]],
            orientation="h",
            marker={"color": "#0f766e"},
            hovertemplate="%{y}<br>Signal: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 18, "b": 36},
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Relative Signal",
        yaxis_title="",
    )
    return fig


def diagnostics_data():
    if demo_df.empty or "true_label" not in demo_df.columns:
        return pd.DataFrame(), np.array([]), np.array([]), np.array([])

    predicted = add_predictions(demo_df)
    y_true = predicted["true_label"].astype(str).str.lower().map({"human": 0, "bot": 1}).to_numpy()
    y_pred = predicted["bot_detected"].astype(str).str.lower().map({"human": 0, "bot": 1}).to_numpy()
    y_score = predicted["bot_probability"].to_numpy() / 100
    return predicted, y_true, y_pred, y_score


def confusion_figure(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Predicted Human", "Predicted Bot"],
            y=["Actual Human", "Actual Bot"],
            colorscale="Blues",
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 18, "b": 36},
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def roc_figure(y_true, y_score):
    fpr, tpr, _thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC {roc_auc:.3f}",
            line={"color": "#2563eb", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Baseline",
            line={"color": "#94a3b8", "dash": "dash"},
        )
    )
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 18, "b": 36},
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


def prediction_grid(result_df, height="570px"):
    return dag.AgGrid(
        rowData=result_df.to_dict("records"),
        columnDefs=[{"field": column} for column in result_df.columns],
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        dashGridOptions={"pagination": True, "paginationPageSize": 20},
        className="ag-theme-alpine",
        style={"height": height, "width": "100%"},
    )


def make_feature_control(column):
    label = display_name(column)
    value = manual_defaults.get(column, 0)

    if column in BOOLEAN_COLUMNS:
        control = dcc.Dropdown(
            id={"type": "manual-input", "column": column},
            options=[
                {"label": "False", "value": 0},
                {"label": "True", "value": 1},
            ],
            value=int(value),
            clearable=False,
            className="dash-control",
        )
    else:
        control = dcc.Input(
            id={"type": "manual-input", "column": column},
            type="number",
            value=float(value),
            className="number-control",
        )

    return html.Label(
        [
            html.Span(label),
            control,
        ],
        className="field-control",
    )


def about_page():
    team_members = ["Cole Bennett", "Nathaniel Ince", "Sheema Farzin", "Kamal Dhital"]

    return html.Div(
        [
            page_title("About Project"),
            html.Div(
                [
                    html.H3("Project Description", className="section-title"),
                    html.P(
                        "BotBuster is a machine-learning dashboard for detecting whether Twitter/X accounts "
                        "are likely human-operated or automated bot accounts. The project analyzes account "
                        "metadata, profile completeness, posting activity, and text-derived features to produce "
                        "bot or human predictions."
                    ),
                    html.P(
                        "The Dash application supports manual account prediction, CSV batch prediction, model "
                        "explainability through feature importance, and diagnostic views such as confusion "
                        "matrix, ROC curve, and holdout prediction results."
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.H3("Team Members", className="section-title"),
                    html.Ul(
                        [html.Li(member) for member in team_members],
                        className="team-list",
                    ),
                ],
                className="panel",
            ),
        ]
    )


def manual_page():
    return html.Div(
        [
            page_title("Manual Prediction"),
            html.Div(
                [make_feature_control(column) for column in feature_columns],
                className="form-grid",
            ),
            html.Button("Predict", id="manual-predict", n_clicks=0, className="primary-button"),
            html.Div(id="manual-result", className="result-space"),
        ]
    )


def batch_page():
    return html.Div(
        [
            page_title("CSV Batch Prediction"),
            dcc.Upload(
                id="upload-data",
                children=html.Div(["Drag and drop or ", html.Span("select a CSV or Excel file")]),
                className="upload-zone",
                multiple=True,
            ),
            html.Div(id="output-data-upload", className="result-space"),
        ]
    )


def explainability_page():
    frame = feature_importance_frame()
    return html.Div(
        [
            page_title("Model Explainability"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Global Feature Importance", className="section-title"),
                            dcc.Graph(figure=feature_importance_figure(), config={"displayModeBar": False}),
                        ],
                        className="panel wide-panel",
                    ),
                    html.Div(
                        [
                            html.H3("Ranked Features", className="section-title"),
                            prediction_grid(frame.round(5), height="470px"),
                        ],
                        className="panel",
                    ),
                ],
                className="two-column-layout",
            ),
        ]
    )


def diagnostics_page():
    predicted, y_true, y_pred, y_score = diagnostics_data()
    if predicted.empty:
        return html.Div(
            [
                page_title("Model Diagnostics"),
                html.Div("No labeled demo data is available.", className="empty-state"),
            ]
        )

    metrics = [
        metric_card("Accuracy", f"{accuracy_score(y_true, y_pred):.1%}", "blue"),
        metric_card("Precision", f"{precision_score(y_true, y_pred):.1%}", "teal"),
        metric_card("Recall", f"{recall_score(y_true, y_pred):.1%}", "amber"),
        metric_card("F1 Score", f"{f1_score(y_true, y_pred):.1%}", "rose"),
    ]

    sample_columns = [
        column
        for column in ["bot_detected", "bot_probability", "true_label", "screen_name", "twitter_id"]
        if column in predicted.columns
    ]

    return html.Div(
        [
            page_title("Model Diagnostics"),
            html.Div(metrics, className="metric-grid"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Confusion Matrix", className="section-title"),
                            dcc.Graph(figure=confusion_figure(y_true, y_pred), config={"displayModeBar": False}),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H3("ROC Curve", className="section-title"),
                            dcc.Graph(figure=roc_figure(y_true, y_score), config={"displayModeBar": False}),
                        ],
                        className="panel",
                    ),
                ],
                className="two-column-layout",
            ),
            html.Div(
                [
                    html.H3("Holdout Predictions", className="section-title"),
                    prediction_grid(predicted[sample_columns].head(250), height="430px"),
                ],
                className="panel",
            ),
        ]
    )


def render_page(page):
    if page == "about":
        return about_page()
    if page == "batch":
        return batch_page()
    if page == "explain":
        return explainability_page()
    if page == "diagnostics":
        return diagnostics_page()
    return manual_page()


app.layout = html.Div(
    [
        html.A("Skip to content", href="#page-content", className="skip-link"),
        html.Aside(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("BotBuster", className="brand-title"),
                                html.Div("Advanced Bot Detection", className="brand-subtitle"),
                            ]
                        ),
                    ],
                    className="brand-row",
                ),
                dcc.RadioItems(
                    id="page-nav",
                    options=[{"label": title, "value": key} for key, title in PAGE_TITLES.items()],
                    value="about",
                    className="nav-list",
                    inputClassName="nav-radio",
                    labelClassName="nav-item",
                ),
            ],
            className="sidebar",
        ),
        html.Main(
            [
                html.Div(
                    [
                        html.Div("CS539 Group Project", className="header-kicker"),
                        html.H1("X Bot Detector: BotBuster", className="app-title"),
                    ],
                    className="topbar",
                ),
                html.Div(id="page-content", className="content-panel"),
                html.Footer(
                    [
                        html.Span(
                            [
                                html.Strong("© BotBuster"),
                                " 2026 Summer CS539 Group Project",
                            ]
                        ),
                    ],
                    className="footer",
                ),
            ],
            id="page-content-wrapper",
            className="main-content",
        ),
    ],
    className="app-shell",
)


@app.callback(Output("page-content", "children"), Input("page-nav", "value"))
def update_page(page):
    return render_page(page)


@app.callback(
    Output("manual-result", "children"),
    Input("manual-predict", "n_clicks"),
    [State({"type": "manual-input", "column": column}, "value") for column in feature_columns],
    prevent_initial_call=True,
)
def update_manual_prediction(n_clicks, *values):
    if not n_clicks:
        return html.Div()

    input_data = {
        column: (0 if value is None else value)
        for column, value in zip(feature_columns, values)
    }
    input_df = coerce_feature_frame(pd.DataFrame([input_data], columns=feature_columns))
    labels, probabilities = predict_features(input_df)
    label = labels[0]
    probability = probabilities[0]
    tone = "rose" if label == "bot" else "teal"

    return html.Div(
        [
            html.Div(
                [
                    metric_card("Prediction", label.upper(), tone),
                    metric_card("Bot Probability", f"{probability:.1%}", "blue"),
                ],
                className="metric-grid compact-metrics",
            ),
            html.Div(
                [
                    html.H3("Local Signal Ranking", className="section-title"),
                    dcc.Graph(figure=local_signal_figure(input_df), config={"displayModeBar": False}),
                ],
                className="panel",
            ),
        ]
    )


def parse_contents(contents, filename, date):
    try:
        read_df = read_uploaded_file(contents, filename)
        result_df = add_predictions(read_df)
    except Exception as exc:
        return html.Div(
            [
                html.H4(filename),
                html.Div(f"There was an error processing this file: {exc}"),
            ],
            className="error-card",
        )

    uploaded_at = datetime.datetime.fromtimestamp(date).strftime("%Y-%m-%d %H:%M:%S")
    bot_rate = (result_df["bot_detected"].eq("bot").mean() * 100) if "bot_detected" in result_df else 0
    children = [
        html.Div(
            [
                html.Div(
                    [
                        html.H3(filename, className="section-title"),
                        html.Div(f"Uploaded {uploaded_at}", className="muted-text"),
                    ]
                ),
                html.Div(
                    [
                        metric_card("Rows", f"{len(result_df):,}", "blue"),
                        metric_card("Bot Rate", f"{bot_rate:.1f}%", "rose"),
                    ],
                    className="metric-grid compact-metrics",
                ),
                prediction_grid(result_df),
            ],
            className="panel",
        )
    ]

    if "true_label" in result_df.columns:
        true_values = result_df["true_label"].astype(str).str.lower().map({"human": 0, "bot": 1})
        pred_values = result_df["bot_detected"].astype(str).str.lower().map({"human": 0, "bot": 1})
        valid = true_values.notna() & pred_values.notna()
        if valid.any():
            children.insert(
                0,
                html.Div(
                    [
                        metric_card("File Accuracy", f"{accuracy_score(true_values[valid], pred_values[valid]):.1%}", "teal"),
                        metric_card("File F1", f"{f1_score(true_values[valid], pred_values[valid]):.1%}", "amber"),
                    ],
                    className="metric-grid compact-metrics",
                ),
            )

    return html.Div(children)


@app.callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("upload-data", "last_modified"),
    prevent_initial_call=True,
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        return html.Div()

    return [
        parse_contents(contents, name, modified)
        for contents, name, modified in zip(list_of_contents, list_of_names, list_of_dates)
    ]


if __name__ == "__main__":
    debug_enabled = os.getenv("DASH_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(debug=debug_enabled, host="127.0.0.1", port=int(os.getenv("PORT", "8050")))
