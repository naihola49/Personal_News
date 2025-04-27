from src.feedback import collect_feedback

if __name__ == "__main__":
    collect_feedback("data/saved_articles.jsonl", "data/user_feedback.jsonl")
