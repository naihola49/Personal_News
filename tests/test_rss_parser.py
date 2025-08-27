"""
Unit tests for the RSS parser module.
Tests RSS feed fetching and article parsing functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from pathlib import Path

from src.rss_parser import RSSParser
from src.models import Article
from src.exceptions import RSSFetchError


class TestRSSParser:
    """Test cases for the RSSParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = RSSParser()
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.parser, 'session'):
            self.parser.close()
    
    def test_parser_initialization(self):
        """Test RSS parser initialization."""
        assert self.parser.user_agent is not None
        assert self.parser.session is not None
    
    def test_context_manager(self):
        """Test RSS parser as context manager."""
        with RSSParser() as parser:
            assert parser.session is not None
        # Session should be closed after context exit
    
    @patch('src.rss_parser.requests.Session.get')
    def test_fetch_feed_articles_success(self, mock_get):
        """Test successful RSS feed fetching."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = '''
        <rss>
            <item>
                <title>Test Article</title>
                <description>Test Summary</description>
                <link>https://example.com/test</link>
                <pubDate>Mon, 01 Jan 2025 12:00:00 +0000</pubDate>
            </item>
        </rss>
        '''
        mock_get.return_value = mock_response
        
        # Mock today's date
        with patch('src.rss_parser.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
            mock_datetime.strptime.side_effect = lambda date_str, fmt: datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
            
            articles = self.parser._fetch_feed_articles("Test Feed", "https://example.com/feed", datetime(2025, 1, 1).date())
            
            assert len(articles) == 1
            assert articles[0].title == "Test Article"
            assert articles[0].summary == "Test Summary"
            assert articles[0].link == "https://example.com/test"
    
    @patch('src.rss_parser.requests.Session.get')
    def test_fetch_feed_articles_http_error(self, mock_get):
        """Test RSS feed fetching with HTTP error."""
        mock_get.side_effect = Exception("HTTP Error")
        
        with pytest.raises(RSSFetchError):
            self.parser._fetch_feed_articles("Test Feed", "https://example.com/feed", datetime(2025, 1, 1).date())
    
    def test_parse_rss_item_valid(self):
        """Test parsing valid RSS item."""
        # Mock BeautifulSoup item
        mock_item = Mock()
        mock_item.find.side_effect = lambda tag: Mock(text=f"Test {tag}") if tag in ["title", "description", "link", "pubDate"] else None
        
        with patch('src.rss_parser.datetime') as mock_datetime:
            mock_datetime.strptime.side_effect = lambda date_str, fmt: datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
            
            article = self.parser._parse_rss_item(mock_item, "Test Feed", datetime(2025, 1, 1).date())
            
            assert article is not None
            assert article.title == "Test title"
            assert article.source == "Test Feed"
    
    def test_parse_rss_item_missing_fields(self):
        """Test parsing RSS item with missing fields."""
        mock_item = Mock()
        mock_item.find.return_value = None
        
        article = self.parser._parse_rss_item(mock_item, "Test Feed", datetime(2025, 1, 1).date())
        
        assert article is None
    
    def test_parse_rss_item_wrong_date(self):
        """Test parsing RSS item with wrong date."""
        mock_item = Mock()
        mock_item.find.side_effect = lambda tag: Mock(text=f"Test {tag}") if tag in ["title", "description", "link", "pubDate"] else None
        
        with patch('src.rss_parser.datetime') as mock_datetime:
            mock_datetime.strptime.side_effect = lambda date_str, fmt: datetime(2024, 12, 31, 12, 0, tzinfo=timezone.utc)
            
            article = self.parser._parse_rss_item(mock_item, "Test Feed", datetime(2025, 1, 1).date())
            
            assert article is None
    
    def test_extract_text_success(self):
        """Test successful text extraction."""
        mock_item = Mock()
        mock_tag = Mock()
        mock_tag.text = "Test Text"
        mock_item.find.return_value = mock_tag
        
        result = self.parser._extract_text(mock_item, "title")
        assert result == "Test Text"
    
    def test_extract_text_missing_tag(self):
        """Test text extraction with missing tag."""
        mock_item = Mock()
        mock_item.find.return_value = None
        
        result = self.parser._extract_text(mock_item, "title")
        assert result is None
    
    def test_parse_publication_date_success(self):
        """Test successful date parsing."""
        date_string = "Mon, 01 Jan 2025 12:00:00 +0000"
        result = self.parser._parse_publication_date(date_string)
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 1
    
    def test_parse_publication_date_invalid(self):
        """Test date parsing with invalid format."""
        date_string = "Invalid Date Format"
        result = self.parser._parse_publication_date(date_string)
        assert result is None
    
    def test_remove_duplicates(self):
        """Test duplicate article removal."""
        articles = [
            Article(source="Test", title="Title 1", summary="Summary 1", link="https://example.com/1", published="2025-01-01"),
            Article(source="Test", title="Title 2", summary="Summary 2", link="https://example.com/2", published="2025-01-01"),
            Article(source="Test", title="Title 3", summary="Summary 3", link="https://example.com/1", published="2025-01-01"),  # Duplicate link
        ]
        
        unique_articles = self.parser._remove_duplicates(articles)
        assert len(unique_articles) == 2
        assert unique_articles[0].link == "https://example.com/1"
        assert unique_articles[1].link == "https://example.com/2"


class TestRSSParserBackwardCompatibility:
    """Test backward compatibility functions."""
    
    @patch('src.rss_parser.RSSParser')
    def test_fetch_nyt_articles(self, mock_parser_class):
        """Test backward compatibility for NYT articles."""
        mock_parser = Mock()
        mock_parser._fetch_source_articles.return_value = []
        mock_parser_class.return_value.__enter__.return_value = mock_parser
        
        from src.rss_parser import fetch_nyt_articles
        result = fetch_nyt_articles()
        
        assert result == []
        mock_parser._fetch_source_articles.assert_called_once()
    
    @patch('src.rss_parser.RSSParser')
    def test_fetch_wsj_articles(self, mock_parser_class):
        """Test backward compatibility for WSJ articles."""
        mock_parser = Mock()
        mock_parser._fetch_source_articles.return_value = []
        mock_parser_class.return_value.__enter__.return_value = mock_parser
        
        from src.rss_parser import fetch_wsj_articles
        result = fetch_wsj_articles()
        
        assert result == []
        mock_parser._fetch_source_articles.assert_called_once()
    
    @patch('src.rss_parser.RSSParser')
    def test_fetch_ft_articles(self, mock_parser_class):
        """Test backward compatibility for FT articles."""
        mock_parser = Mock()
        mock_parser._fetch_source_articles.return_value = []
        mock_parser_class.return_value.__enter__.return_value = mock_parser
        
        from src.rss_parser import fetch_ft_articles
        result = fetch_ft_articles()
        
        assert result == []
        mock_parser._fetch_source_articles.assert_called_once()
