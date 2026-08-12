import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

LOG_DIR = Path(os.environ.get("LOG_DIR", "/logs"))
LOG_FILE = LOG_DIR / "prediction_logs.json"

st.set_page_config(page_title="Model Monitoring", layout="wide")
st.title("Sentiment Model Monitoring")


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


df = load_logs()

if st.button("Refresh"):
    st.rerun()

if df.empty:
    st.info(
        f"No predictions logged yet at {LOG_FILE}. "
        "Send requests to the API to populate this dashboard."
    )
    st.stop()

df["correct"] = df["predicted_sentiment"] == df["true_sentiment"]

total = len(df)
accuracy = df["correct"].mean() * 100
last_seen = df["timestamp"].max()

col1, col2, col3 = st.columns(3)
col1.metric("Total predictions", f"{total:,}")
col2.metric("Accuracy", f"{accuracy:.2f}%")
col3.metric("Last prediction", last_seen.strftime("%Y-%m-%d %H:%M UTC"))

left, right = st.columns(2)

with left:
    st.subheader("Predicted sentiment distribution")
    st.bar_chart(df["predicted_sentiment"].value_counts())

with right:
    st.subheader("True sentiment distribution")
    st.bar_chart(df["true_sentiment"].value_counts())

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
