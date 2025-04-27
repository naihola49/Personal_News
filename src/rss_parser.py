import requests
from bs4 import BeautifulSoup

# NYT
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
# WSJ
WSJ_RSS_URL = "https://feeds.a.dj.com/rss/RSSWorldNews.xml"

def fetch_wsj_articles():
    response = requests.get(WSJ_RSS_URL, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.content, features="xml")

    items = soup.find_all("item")
    print(f"WSJ: Found {len(items)} items")

    articles = []
    for item in items:
        articles.append({
            "source": "Wall Street Journal",
            "title": item.title.text,
            "summary": item.description.text,
            "link": item.link.text,
            "published": item.pubDate.text
        })

    return articles
# FT
FT_RSS_URL = "https://www.ft.com/world?format=rss"

def fetch_ft_articles():
    response = requests.get(FT_RSS_URL, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.content, features="xml")

    items = soup.find_all("item")
    print(f"FT: Found {len(items)} items")

    articles = []
    for item in items:
        articles.append({
            "source": "Financial Times",
            "title": item.title.text,
            "summary": item.description.text,
            "link": item.link.text,
            "published": item.pubDate.text
        })

    return articles