from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast + accurate

def embed_articles(articles):
    """
    Takes in a list of articles (with 'title' and 'summary'), returns BERT embeddings.
    """
    texts = [
        f"{a['title']} {a['summary']}" for a in articles
    ]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings
