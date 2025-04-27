from src.rss_parser import fetch_nyt_articles, fetch_wsj_articles, fetch_ft_articles, remove_duplicates
from src.embedder import embed_articles
from src.feedback import append_feedback_record
from src.saver import save_articles_with_embeddings
from src.trainer import retrain_model
import random
import joblib
import numpy as np
import os
from datetime import datetime

def load_model(model_path="models/user_feedback_model.joblib"):
    """
    Loads saved sklearn model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Train first.")
    return joblib.load(model_path)

def main_session():
    print(f"Starting Personalized News Session: {datetime.now().strftime('%B %d, %Y')}")
    print("-" * 80)

    # Fetch articles
    nyt_articles = fetch_nyt_articles()
    wsj_articles = fetch_wsj_articles()
    ft_articles = fetch_ft_articles()
    articles = nyt_articles + wsj_articles + ft_articles

    # Remove duplicates
    articles = remove_duplicates(articles)
    print(f"Total unique articles fetched: {len(articles)}")
    print("-" * 80)

    # Embed articles
    embeddings = embed_articles(articles)

    # Load trained model
    model = load_model()

    # Predict scores
    prob = model.predict_proba(embeddings)[:, 1]  # Probability of liking the article

    # Select top articles and sample others
    top_indices = np.argsort(prob)[::-1]
    top_10_indices = top_indices[:10]
    remaining_indices = list(set(range(len(articles))) - set(top_10_indices))
    random_sample_indices = random.sample(remaining_indices, min(7, len(remaining_indices)))

    final_indices = list(top_10_indices) + random_sample_indices
    random.shuffle(final_indices)

    print("Presenting selected articles\n")

    feedback_count = 0
    like_count = 0
    dislike_count = 0

    for idx in final_indices:
        article = articles[idx]
        score = prob[idx]

        print("=" * 80)
        print(f"Source: {article['source']} | Score: {score:.2f}")
        print(f"Title: {article['title']}")
        print(f"Summary: {article['summary']}")
        print(f"Link: {article['link']}")
        print(f"Published: {article['published']}")
        print("=" * 80)

        # Get user feedback
        while True:
            feedback = input("Feedback ([l]ike / [d]islike): ").strip().lower()
            if feedback == "l":
                label = 1
                like_count += 1
                break
            elif feedback == "d":
                label = 0
                dislike_count += 1
                break
            else:
                print("Invalid input. Enter 'l' or 'd'.")

        # Save feedback
        append_feedback_record(article, label, embedding=embeddings[idx])
        feedback_count += 1

    print("\nSession Completed")
    print("-" * 80)
    print(f"Articles Liked: {like_count}")
    print(f"Articles Disliked: {dislike_count}")
    print(f"Total Feedback Collected: {feedback_count}")
    print("-" * 80)

    # Retrain the model
    if feedback_count > 0:
        retrain_model()
        print("Model updated based on new feedback.")
    else:
        print("No feedback collected. Model not updated.")

if __name__ == "__main__":
    main_session()
