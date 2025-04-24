import requests
from bs4 import BeautifulSoup

NYT_RSS_URL = "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"

def fetch_nyt_articles():
    response = requests.get(NYT_RSS_URL, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.content, features="xml")

    items = soup.find_all("item")
    print(f"NYT: Found {len(items)} items")

    articles = []
    for item in items:
        articles.append({
            "source": "New York Times",
            "title": item.title.text,
            "summary": item.description.text,
            "link": item.link.text,
            "published": item.pubDate.text
        })

    return articles
