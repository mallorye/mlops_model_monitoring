import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent / "sentiment_model.pkl"
LOG_DIR = Path(os.environ.get("LOG_DIR", "/logs"))
LOG_FILE = LOG_DIR / "prediction_logs.json"

LABEL_NAMES = {"0": "negative", "1": "positive"}

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str
    true_sentiment: str


class PredictResponse(BaseModel):
    predicted_sentiment: str
    confidence: float


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    predicted = model.predict([request.text])[0]
    proba = model.predict_proba([request.text])[0]
    class_index = list(model.classes_).index(predicted)
    confidence = round(float(proba[class_index]), 4)
    predicted = LABEL_NAMES.get(str(predicted), str(predicted))

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_text": request.text,
        "predicted_sentiment": str(predicted),
        "true_sentiment": request.true_sentiment,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return PredictResponse(predicted_sentiment=str(predicted), confidence=confidence)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
