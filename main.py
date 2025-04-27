from src.rss_parser import fetch_nyt_articles, fetch_wsj_articles, fetch_ft_articles
from src.embedder import embed_articles
from src.feedback import append_feedback_record
from src.saver import save_articles_with_embeddings
from src.trainer import retrain_model
import random
import joblib
import numpy as np
import os

def load_model(model_path="models/user_feedback_model.joblib"):
    """
    Loads saved sklearn from disk
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found. Train first!")
    return joblib.load(model_path)

def main_session():
    print("Fetching new articles")
    # article block
    nyt_articles = fetch_nyt_articles()
    wsj_articles = fetch_wsj_articles()
    ft_articles = fetch_ft_articles()
    articles = nyt_articles + wsj_articles + ft_articles
    print(f"Total Articles Fetched {len(articles)}")

    # embed -> BERT
    embeddings = embed_articles(articles)

    # load model
    model = load_model()

    # predict scores
    prob = model.predict_proba(embeddings)[:, 1] # prob of being in like class
    """
    The goal here is to show a top 10
    Randomly sample the rest to provide exposure
    """
    top_indices = np.argsort(prob)[::-1] # sort by top scoring articles
    top_10_indices = top_indices[:10]
    remaining_indices = list(set(range(len(articles))) - set(top_10_indices))

    random_sample_indices = random.sample(remaining_indices, 7)
    final_indicies = list(top_10_indices) + list(random_sample_indices) # merge
    random.shuffle(final_indices)


    print("Starting Personalized News Feed")

    feedback_count = 0

    for idx in final_indices:
        article = articles[idx]
        score = prob[idx]
        print(f"{article['source']} | {score:.2f}")
        print(f"Title: {article['title']}")
        print(f"Summary: {article['summary'][:200]}...")
        print(f"Link: {article['link']}")
        print(f"Published: {article['published']}\n")
    
        # getting user feedback
        while True:
            feedback = input("Did you like this? [l]ike / [d]islike ").strip().lower()
            if feedback == "l":
                label = 1
                break
            elif feedback == "d":
                label = 0
                break
            else:
                print("Invalid Input. Enter 'l' or 'd' ")
    
        # append new feedback
        append_feedback_record(article, label, embedding=embeddings[idx])
        feedback_count += 1
    print(f"Collected feedback on {feedback_count} articles.")

    # retrain model
    retrain_model()
    print("Model Updated! Check back in tomorrow")

if __name__ == "__main__":
    main_session()
