"""
RSS Parser for the Personal News Aggregator.
Fetches articles from configured RSS feeds with proper error handling.
"""

import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import List, Dict, Optional
from pathlib import Path

from .exceptions import RSSFetchError
from .models import Article
from config import RSS_FEEDS, SESSION_CONFIG

logger = logging.getLogger(__name__)


class RSSParser:
    """Handles RSS feed parsing and article extraction."""
    
    def __init__(self, user_agent: Optional[str] = None):
        """Initialize the RSS parser.
        
        Args:
            user_agent: Custom user agent string for HTTP requests
        """
        self.user_agent = user_agent or SESSION_CONFIG["user_agent"]
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
    
    def fetch_all_feeds(self) -> List[Article]:
        """Fetch articles from all configured RSS feeds.
        
        Returns:
            List of Article objects from all feeds
            
        Raises:
            RSSFetchError: If fetching from any feed fails
        """
        all_articles = []
        
        for source, feeds in RSS_FEEDS.items():
            try:
                logger.info(f"Fetching articles from {source.upper()}")
                articles = self._fetch_source_articles(source, feeds)
                all_articles.extend(articles)
                logger.info(f"Successfully fetched {len(articles)} articles from {source.upper()}")
            except Exception as e:
                logger.error(f"Failed to fetch articles from {source.upper()}: {e}")
                # Continue with other sources instead of failing completely
                continue
        
        # Remove duplicates
        unique_articles = self._remove_duplicates(all_articles)
        logger.info(f"Total unique articles fetched: {len(unique_articles)}")
        
        return unique_articles
    
    def _fetch_source_articles(self, source: str, feeds: Dict[str, str]) -> List[Article]:
        """Fetch articles from a specific news source.
        
        Args:
            source: Name of the news source (e.g., 'nyt', 'wsj')
            feeds: Dictionary of feed names to URLs
            
        Returns:
            List of Article objects from the source
            
        Raises:
            RSSFetchError: If fetching fails
        """
        articles = []
        today = datetime.now(timezone.utc).date()
        
        for feed_name, url in feeds.items():
            try:
                feed_articles = self._fetch_feed_articles(feed_name, url, today)
                articles.extend(feed_articles)
            except Exception as e:
                logger.warning(f"Failed to fetch feed {feed_name}: {e}")
                continue
        
        return articles
    
    def _fetch_feed_articles(self, feed_name: str, url: str, target_date: datetime.date) -> List[Article]:
        """Fetch articles from a specific RSS feed.
        
        Args:
            feed_name: Name of the feed
            url: RSS feed URL
            target_date: Target date for articles
            
        Returns:
            List of Article objects from the feed
            
        Raises:
            RSSFetchError: If fetching fails
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.find_all("item")
            
            articles = []
            for item in items:
                try:
                    article = self._parse_rss_item(item, feed_name, target_date)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse RSS item: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from {feed_name}")
            return articles
            
        except requests.RequestException as e:
            raise RSSFetchError(feed_name, url, e)
        except Exception as e:
            raise RSSFetchError(feed_name, url, e)
    
    def _parse_rss_item(self, item, feed_name: str, target_date: datetime.date) -> Optional[Article]:
        """Parse a single RSS item into an Article object.
        
        Args:
            item: BeautifulSoup RSS item element
            feed_name: Name of the feed
            target_date: Target date for articles
            
        Returns:
            Article object if parsing succeeds and date matches, None otherwise
        """
        try:
            # Extract required fields
            title = self._extract_text(item, "title")
            summary = self._extract_text(item, "description")
            link = self._extract_text(item, "link")
            pub_date = self._extract_text(item, "pubDate")
            
            if not all([title, summary, link, pub_date]):
                logger.debug(f"Skipping item with missing required fields")
                return None
            
            # Parse publication date
            published_dt = self._parse_publication_date(pub_date)
            if not published_dt or published_dt.date() != target_date:
                return None
            
            # Create Article object
            return Article(
                source=feed_name,
                title=title.strip(),
                summary=summary.strip(),
                link=link.strip(),
                published=pub_date.strip()
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse RSS item: {e}")
            return None
    
    def _extract_text(self, item, tag_name: str) -> Optional[str]:
        """Extract text content from an XML tag.
        
        Args:
            item: BeautifulSoup element
            tag_name: Name of the tag to extract
            
        Returns:
            Text content if found, None otherwise
        """
        tag = item.find(tag_name)
        return tag.text if tag else None
    
    def _parse_publication_date(self, date_string: str) -> Optional[datetime]:
        """Parse publication date string into datetime object.
        
        Args:
            date_string: Date string from RSS feed
            
        Returns:
            Parsed datetime object if successful, None otherwise
        """
        date_formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # Standard RSS format
            "%a, %d %b %Y %H:%M:%S %Z",  # Alternative format
            "%Y-%m-%dT%H:%M:%S%z",       # ISO format
            "%Y-%m-%d %H:%M:%S"          # Simple format
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date string: {date_string}")
        return None
    
    def _remove_duplicates(self, articles: List[Article]) -> List[Article]:
        """Remove duplicate articles based on link.
        
        Args:
            articles: List of Article objects
            
        Returns:
            List of unique Article objects
        """
        seen_links = set()
        unique_articles = []
        
        for article in articles:
            if article.link not in seen_links:
                unique_articles.append(article)
                seen_links.add(article.link)
        
        logger.info(f"Removed {len(articles) - len(unique_articles)} duplicate articles")
        return unique_articles
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Backward compatibility functions
def fetch_nyt_articles() -> List[Article]:
    """Fetch articles from NYT feeds (backward compatibility)."""
    with RSSParser() as parser:
        return parser._fetch_source_articles("nyt", RSS_FEEDS["nyt"])


def fetch_wsj_articles() -> List[Article]:
    """Fetch articles from WSJ feeds (backward compatibility)."""
    with RSSParser() as parser:
        return parser._fetch_source_articles("wsj", RSS_FEEDS["wsj"])


def fetch_ft_articles() -> List[Article]:
    """Fetch articles from FT feeds (backward compatibility)."""
    with RSSParser() as parser:
        return parser._fetch_source_articles("ft", RSS_FEEDS["ft"])


def remove_duplicates(articles: List[Article]) -> List[Article]:
    """Remove duplicate articles (backward compatibility)."""
    with RSSParser() as parser:
        return parser._remove_duplicates(articles)
