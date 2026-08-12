import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

LOG_DIR = Path(os.environ.get("LOG_DIR", "/logs"))
LOG_FILE = LOG_DIR / "prediction_logs.json"
DATA_PATH = Path(os.environ.get("DATA_PATH", "/app/IMDB Dataset.csv"))

st.set_page_config(page_title="Model Monitoring", layout="wide")


def load_logs() -> pd.DataFrame:
    if not LOG_FILE.exists():
        return pd.DataFrame()
    records = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def load_training_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


df = load_logs()
train_df = load_training_data()

if not df.empty:
    df["correct"] = df["predicted_sentiment"] == df["true_sentiment"]
    accuracy = df["correct"].mean() * 100
    if accuracy < 80:
        st.error(
            f"ALERT: model accuracy is {accuracy:.2f}%, "
            "below the 80% threshold. Investigate for model degradation or drift."
        )

st.title("Sentiment Model Monitoring")

if st.button("Refresh"):
    st.rerun()

if df.empty:
    st.info(
        f"No predictions logged yet at {LOG_FILE}. "
        "Send requests to the API to populate this dashboard."
    )
    st.stop()

total = len(df)
last_seen = df["timestamp"].max()


def precision_pct(label: str):
    predicted = df[df["predicted_sentiment"] == label]
    if predicted.empty:
        return None
    return (predicted["true_sentiment"] == label).mean() * 100


pos_precision = precision_pct("positive")
neg_precision = precision_pct("negative")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total predictions", f"{total:,}")
col2.metric("Accuracy", f"{accuracy:.2f}%")
col3.metric(
    "Precision (positive)",
    f"{pos_precision:.2f}%" if pos_precision is not None else "n/a",
)
col4.metric(
    "Precision (negative)",
    f"{neg_precision:.2f}%" if neg_precision is not None else "n/a",
)
col5.metric("Last prediction", last_seen.strftime("%Y-%m-%d %H:%M UTC"))

st.subheader("Data drift: review length distribution")
if train_df.empty:
    st.warning(
        f"Training data not found at {DATA_PATH}; cannot compare length distributions."
    )
else:
    train_lengths = train_df["review"].str.split().str.len()
    request_lengths = df["request_text"].str.split().str.len()
    upper = float(max(train_lengths.quantile(0.95), request_lengths.quantile(0.95)))
    bins = np.linspace(0, upper, 31)

    def length_proportions(lengths: pd.Series) -> np.ndarray:
        counts, _ = np.histogram(lengths.clip(upper=upper), bins=bins)
        return counts / max(len(lengths), 1)

    midpoints = ((bins[:-1] + bins[1:]) / 2).round().astype(int)
    hist_df = pd.DataFrame(
        {
            "training data": length_proportions(train_lengths),
            "inference requests": length_proportions(request_lengths),
        },
        index=midpoints,
    )
    hist_df.index.name = "review length (words)"
    st.caption(
        "Proportion of reviews per word-count bucket "
        "(top 5% longest reviews clipped into the last bucket)."
    )
    st.bar_chart(hist_df, stack=False)

st.subheader("Target drift: sentiment distribution")
if train_df.empty:
    st.warning(
        f"Training data not found at {DATA_PATH}; "
        "showing predicted distribution only."
    )
    st.bar_chart(df["predicted_sentiment"].value_counts(normalize=True))
else:
    target_df = pd.DataFrame(
        {
            "training data": train_df["sentiment"].value_counts(normalize=True),
            "inference requests (predicted)": df["predicted_sentiment"].value_counts(
                normalize=True
            ),
        }
    ).fillna(0)
    target_df.index.name = "sentiment"
    st.caption("Proportion of each sentiment class.")
    st.bar_chart(target_df, stack=False)

st.subheader("Rolling accuracy (last 50 predictions per point)")
rolling = (
    df.sort_values("timestamp")["correct"]
    .rolling(window=50, min_periods=1)
    .mean()
    .mul(100)
    .reset_index(drop=True)
)
st.line_chart(rolling)

st.subheader("Recent predictions")
st.dataframe(
    df.sort_values("timestamp", ascending=False).head(25),
    use_container_width=True,
)
