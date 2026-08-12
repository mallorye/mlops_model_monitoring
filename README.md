# Sentiment Model Monitoring

A two-container MLOps stack for serving and monitoring a sentiment analysis model:

- **API service** (`api/`) — a FastAPI app that serves a scikit-learn sentiment model at `POST /predict` and appends every prediction (with its ground-truth label) to a JSONL log file.
- **Monitoring service** (`monitoring/`) — a Streamlit dashboard that reads those prediction logs and compares them against the training data to surface accuracy, precision, data drift, target drift, and alerts.

## Architecture

```
                 ┌─────────────────┐        ┌────────────────────┐
 curl /          │  sentiment-api   │        │ sentiment-monitoring│
 evaluate.py ───▶│  FastAPI :8000   │        │  Streamlit :8501    │
                 └────────┬────────┘        └─────────┬──────────┘
                          │ writes                    │ reads
                          ▼                           ▼
                 ┌─────────────────────────────────────────────┐
                 │   Docker volume: sentiment-logs (/logs)      │
                 │        /logs/prediction_logs.json            │
                 └─────────────────────────────────────────────┘
                          Docker network: sentiment-net
```

Both containers run on a shared Docker network (`sentiment-net`) and mount the same named volume (`sentiment-logs`) at `/logs`. The API writes one JSON line per prediction to `/logs/prediction_logs.json`; the dashboard reads the same file, so predictions show up in the monitoring UI without any coupling between the two services.  In other words, FastAPI is the serving path; Streamlit is an observer reading its logs off a shared volume; the dependency runs strictly from observer to logs to server, never backward.

## Prerequisites

- Docker
- `make`
- Python 3 with the `requests` package (only needed to run `evaluate.py` from the host)

## Running the stack

Step 1 — build both images:

```bash
make build
```

Step 2 — create the network and volume, and start both containers:

```bash
make run
```

The services are then available at:

| Service    | URL                    |
|------------|------------------------|
| API        | http://localhost:8000  |
| Monitoring | http://localhost:8501  |

Step 3 — when you're done, stop the containers and remove the network and volume:

```bash
make clean
```

Note: `make clean` deletes the `sentiment-logs` volume, so all logged predictions are removed with it.

## Using the API

`POST /predict` takes the review text plus the ground-truth label (used for monitoring), and returns the model's prediction and confidence.

Positive example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely wonderful, I loved every minute.", "true_sentiment": "positive"}'
```

Negative example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "A complete waste of time, the plot made no sense.", "true_sentiment": "negative"}'
```

Example response:

```json
{"predicted_sentiment": "positive", "confidence": 0.9731}
```

Every request is appended to `/logs/prediction_logs.json` inside the shared volume and appears on the monitoring dashboard.

## Evaluating the model

[evaluate.py](evaluate.py) sends every example in [test.json](test.json) to the running API and reports accuracy. With the stack up (`make run`):

```bash
python3 evaluate.py
```

It prints the total number of examples, how many predictions matched the true label, and the final accuracy:

```
Total: 174
Correct: 159
Accuracy: 91.38%
```

Because every request goes through the API, running the evaluation also populates the monitoring dashboard — open http://localhost:8501 afterwards to see the results.

## Monitoring dashboard

The Streamlit dashboard (http://localhost:8501) compares live inference traffic against the training data (`IMDB Dataset.csv`, mounted into the container and located via the `DATA_PATH` env var, default `/app/IMDB Dataset.csv`). It shows:

- **Accuracy alerting** — if live accuracy drops below **80%**, a red alert banner (`st.error`) appears at the very top of the dashboard, flagging possible model degradation or drift.
- **Model quality metrics** — total prediction count, overall accuracy, **precision for the positive class**, precision for the negative class, and the timestamp of the most recent prediction.
- **Data drift** — a histogram comparing the review-length (word count) distribution of the training data against incoming request texts. A mismatch (e.g. short one-line requests vs. long training reviews) signals that live traffic no longer looks like the data the model was trained on.
- **Target drift** — a grouped bar chart comparing the model's predicted sentiment distribution against the sentiment distribution of the training data. A skew toward one class relative to training suggests the input distribution (or the model's behavior) has shifted.
- **Rolling accuracy** — accuracy over a rolling window of the last 50 predictions, to spot degradation over time rather than just in aggregate.
- **Recent predictions** — a table of the 25 most recent requests with their predicted and true labels.

Use the **Refresh** button to re-read the logs; predictions appear as soon as the API handles them.

## Project layout

```
.
├── api/
│   ├── Dockerfile
│   ├── main.py               # FastAPI service
│   ├── requirements.txt
│   └── sentiment_model.pkl   # trained scikit-learn pipeline
├── monitoring/
│   ├── Dockerfile
│   ├── app.py                # Streamlit dashboard
│   └── requirements.txt
├── IMDB Dataset.csv          # training data, used by the dashboard for drift comparison
├── evaluate.py               # accuracy evaluation against the running API
├── test.json                 # labeled evaluation set
├── Makefile                  # build / run / clean
└── README.md
```
