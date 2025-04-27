"""
Load feedback, train model, retrain model
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


def load_feedback(feedback_path):
    """
    Loads feedback JSONL file and returns X (embeddings) and y (labels).
    """
    X = []
    y = []

    with open(feedback_path, "r") as f:
        for line in f:
            record = json.loads(line)
            X.append(record["embedding"])
            y.append(record["feedback"])

    X = np.array(X)
    y = np.array(y)

    return X, y

def train_feedback_model(X, y):
    """
    Trains a logistic regression model on embeddings.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model trained. Validation accuracy: {acc:.2f}")
    return model

def retrain_model(feedback_path="user_feedback.jsonl", model_path="user_feedback_model.joblib"):
    """
    Retrains the model from all collected feedback and saves it.
    """
    X, y = load_feedback(feedback_path)
    model = train_feedback_model(X, y)
    joblib.dump(model, model_path)
    print(f"Model retrained and saved to {model_path}")
