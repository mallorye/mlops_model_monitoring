FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py sentiment_model.pkl ./

# Predictions are logged as JSONL here; mount a volume at /logs to share
# them with the monitoring service.
ENV LOG_DIR=/logs

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
