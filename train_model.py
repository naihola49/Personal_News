from src.trainer import load_feedback, train_feedback_model
import joblib

if __name__ == "__main__":
    X, y = load_feedback("data/user_feedback.jsonl") # load labeled feedback
    model = train_feedback_model(X, y)
    # save model
    joblib.dump(model, "models/user_feedback_model.joblib")
    print("Model saved")

