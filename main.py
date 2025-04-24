from src.rss_parser import fetch_nyt_articles
from src.embedder import embed_articles

if __name__ == "__main__":
    articles = fetch_nyt_articles()

    print(f"\nTotal NYT articles fetched: {len(articles)}\n")

    for article in articles[:5]:
        print(f"Title: {article['title']}")
        print(f"Summary: {article['summary'][:100]}...")
        print(f"Link: {article['link']}")
        print(f"Date: {article['published']}\n")
    
    embeddings = embed_articles(articles)
    print(f"Pulled {len(embeddings)} embeddings, size {len(embeddings[0])}")

