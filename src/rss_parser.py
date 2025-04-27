import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz

# NYT
NYT_RSS_URLS = {
    "New York Times - World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "New York Times - Business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "New York Times - Technology": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "New York Times - US": "https://rss.nytimes.com/services/xml/rss/nyt/US.xml"
}

def fetch_nyt_articles():
    articles = []
    today = datetime.now(pytz.utc).date()

    for label, url in NYT_RSS_URLS.items():
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.find_all("item")
        
        for item in items:
            pub_date = item.pubDate.text.strip()
            try:
                published_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
                if published_dt.date() == today:
                    articles.append({
                        "source": label,
                        "title": item.title.text,
                        "summary": item.description.text,
                        "link": item.link.text,
                        "published": pub_date
                    })
            except Exception as e:
                print(f"Error parsing date: {pub_date} -> {e}")
    
    print(f"NYT: Found {len(articles)} articles from today")
    return articles

# WSJ
WSJ_RSS_URLS = {
    "Wall Street Journal - Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "Wall Street Journal - Technology": "https://feeds.a.dj.com/rss/WSJcomUSBusinessTechnology.xml"
}

def fetch_wsj_articles():
    articles = []
    today = datetime.now(pytz.utc).date()

    for label, url in WSJ_RSS_URLS.items():
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.find_all("item")
        
        for item in items:
            pub_date = item.pubDate.text.strip()
            try:
                published_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
                if published_dt.date() == today:
                    articles.append({
                        "source": label,
                        "title": item.title.text,
                        "summary": item.description.text,
                        "link": item.link.text,
                        "published": pub_date
                    })
            except Exception as e:
                print(f"Error parsing date: {pub_date} -> {e}")

    print(f"WSJ: Found {len(articles)} articles from today")
    return articles

# FT
FT_RSS_URLS = {
    "Financial Times - Companies": "https://www.ft.com/companies?format=rss",
    "Financial Times - US": "https://www.ft.com/us?format=rss"
}

def fetch_ft_articles():
    articles = []
    today = datetime.now(pytz.utc).date()

    for label, url in FT_RSS_URLS.items():
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.find_all("item")
        
        for item in items:
            pub_date = item.pubDate.text.strip()
            try:
                published_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                if published_dt.date() == today:
                    articles.append({
                        "source": label,
                        "title": item.title.text,
                        "summary": item.description.text,
                        "link": item.link.text,
                        "published": pub_date
                    })
            except Exception as e:
                print(f"Error parsing date: {pub_date} -> {e}")
    
    print(f"FT: Found {len(articles)} articles from today")
    return articles

def remove_duplicates(articles):
    seen_links = set()
    unique_articles = []
    for article in articles:
        if article["link"] not in seen_links:
            unique_articles.append(article)
            seen_links.add(article["link"])
    print(f"Total after removing duplicates: {len(unique_articles)}")
    return unique_articles
