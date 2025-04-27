import json

def save_articles_with_embeddings(articles, embeddings, filepath):
    """
    Saves articles and corresponding embeddings into a JSONL file.
    Each line = one article+embedding.
    """
    with open(filepath, "w") as f:
        for article, emb in zip(articles, embeddings):
            record = {
                "source": article["source"],
                "title": article["title"],
                "summary": article["summary"],
                "link": article["link"],
                "published": article["published"],
                "embedding": emb.tolist()  # numpy array to list
            }
            f.write(json.dumps(record) + "\n")
