"""
Collects user feedback, append new feedback
"""

import json

def collect_feedback(jsonl_path, feedback_path):
    """
    Goes through articles and records user feedback.
    Saves feedback to a JSONL file: each line is {"title": ..., "feedback": 1 or 0}
    """
    print("Starting feedback collection...")

    with open(jsonl_path, "r") as f:
        articles = [json.loads(line) for line in f]

    feedback_records = []

    for article in articles:
        print("\n-----------------------------------")
        print(f" Title: {article['title']}")
        print(f"Summary: {article['summary'][:200]}...")
        print(f"Link: {article['link']}")
        print("-----------------------------------")
        feedback = input("Like [l], Dislike [d] ").strip().lower()

        while True:
            if feedback == "l":
                label = 1
                break
            elif feedback == "d":
                label = 0
                break
            else:
                print("Please Enter a Valid Argument")

        feedback_records.append({
            "title": article["title"],
            "link": article["link"],
            "feedback": label,
            "embedding": article["embedding"]
        })

    # Save collected feedback
    with open(feedback_path, "w") as f:
        for record in feedback_records:
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(feedback_records)} feedback entries to {feedback_path}")

def append_feedback_record(article, label, embedding, feedback_path="user_feedback.jsonl"): # new feedback on the fly
    """
    Appends a new feedback record (title, link, label, embedding) into user_feedback.jsonl.
    """
    record = {
        "title": article["title"],
        "link": article["link"],
        "feedback": label,
        "embedding": embedding.tolist() # save as list -> numpy array cannot handle JSON
    }
    with open(feedback_path, "a") as f:
        f.write(json.dumps(record) + "\n")
