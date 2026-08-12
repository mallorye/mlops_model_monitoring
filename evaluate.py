import json
import sys

import requests

API_URL = "http://localhost:8000/predict"


def main():
    with open("test.json") as f:
        items = json.load(f)

    total = 0
    correct = 0

    for item in items:
        payload = {
            "text": item["text"],
            "true_sentiment": item["true_label"],
        }
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(f"Error: could not connect to {API_URL}.")
            print("Please start the API first with: make run")
            sys.exit(1)

        predicted = response.json()["predicted_sentiment"]
        total += 1
        if predicted == item["true_label"]:
            correct += 1

    accuracy = (correct / total * 100) if total else 0.0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
