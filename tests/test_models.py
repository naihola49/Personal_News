"""
Unit tests for the data models.
Tests data validation and model behavior.
"""

import pytest
from datetime import datetime
from src.models import Article, FeedbackRecord, ModelMetrics, SessionStats, RSSFeedConfig


class TestArticle:
    """Test cases for the Article model."""
    
    def test_valid_article_creation(self):
        """Test creating a valid article."""
        article = Article(
            source="Test Source",
            title="Test Title",
            summary="Test Summary",
            link="https://example.com",
            published="2025-01-01"
        )
        
        assert article.source == "Test Source"
        assert article.title == "Test Title"
        assert article.summary == "Test Summary"
        assert article.link == "https://example.com"
        assert article.published == "2025-01-01"
        assert article.embedding is None
    
    def test_article_with_embedding(self):
        """Test creating an article with embedding."""
        embedding = [0.1, 0.2, 0.3]
        article = Article(
            source="Test Source",
            title="Test Title",
            summary="Test Summary",
            link="https://example.com",
            published="2025-01-01",
            embedding=embedding
        )
        
        assert article.embedding == embedding
    
    def test_article_properties(self):
        """Test article computed properties."""
        article = Article(
            source="Test Source",
            title="Test Title",
            summary="Test Summary",
            link="https://example.com",
            published="2025-01-01"
        )
        
        assert article.title_length == 10
        assert article.summary_length == 13
    
    def test_article_validation_empty_title(self):
        """Test that empty title raises validation error."""
        with pytest.raises(ValueError, match="Article title cannot be empty"):
            Article(
                source="Test Source",
                title="",
                summary="Test Summary",
                link="https://example.com",
                published="2025-01-01"
            )
    
    def test_article_validation_empty_link(self):
        """Test that empty link raises validation error."""
        with pytest.raises(ValueError, match="Article link cannot be empty"):
            Article(
                source="Test Source",
                title="Test Title",
                summary="Test Summary",
                link="",
                published="2025-01-01"
            )
    
    def test_article_validation_empty_summary(self):
        """Test that empty summary raises validation error."""
        with pytest.raises(ValueError, match="Article summary cannot be empty"):
            Article(
                source="Test Source",
                title="Test Title",
                summary="",
                link="https://example.com",
                published="2025-01-01"
            )


class TestFeedbackRecord:
    """Test cases for the FeedbackRecord model."""
    
    def test_valid_feedback_record_creation(self):
        """Test creating a valid feedback record."""
        feedback_record = FeedbackRecord(
            title="Test Title",
            link="https://example.com",
            feedback=1,
            embedding=[0.1, 0.2, 0.3]
        )
        
        assert feedback_record.title == "Test Title"
        assert feedback_record.link == "https://example.com"
        assert feedback_record.feedback == 1
        assert feedback_record.embedding == [0.1, 0.2, 0.3]
        assert isinstance(feedback_record.timestamp, datetime)
    
    def test_feedback_record_validation_invalid_feedback(self):
        """Test that invalid feedback value raises validation error."""
        with pytest.raises(ValueError, match="Feedback must be 0 \\(dislike\\) or 1 \\(like\\)"):
            FeedbackRecord(
                title="Test Title",
                link="https://example.com",
                feedback=2,
                embedding=[0.1, 0.2, 0.3]
            )
    
    def test_feedback_record_validation_empty_title(self):
        """Test that empty title raises validation error."""
        with pytest.raises(ValueError, match="Article title cannot be empty"):
            FeedbackRecord(
                title="",
                link="https://example.com",
                feedback=1,
                embedding=[0.1, 0.2, 0.3]
            )
    
    def test_feedback_record_validation_empty_link(self):
        """Test that empty link raises validation error."""
        with pytest.raises(ValueError, match="Article link cannot be empty"):
            FeedbackRecord(
                title="Test Title",
                link="",
                feedback=1,
                embedding=[0.1, 0.2, 0.3]
            )
    
    def test_feedback_record_validation_empty_embedding(self):
        """Test that empty embedding raises validation error."""
        with pytest.raises(ValueError, match="Embedding must be a non-empty list"):
            FeedbackRecord(
                title="Test Title",
                link="https://example.com",
                feedback=1,
                embedding=[]
            )


class TestModelMetrics:
    """Test cases for the ModelMetrics model."""
    
    def test_valid_model_metrics_creation(self):
        """Test creating valid model metrics."""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            training_samples=1000,
            validation_samples=250,
            best_params={"C": 1.0},
            training_time=45.2
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.82
        assert metrics.recall == 0.88
        assert metrics.f1_score == 0.85
        assert metrics.training_samples == 1000
        assert metrics.validation_samples == 250
        assert metrics.best_params == {"C": 1.0}
        assert metrics.training_time == 45.2
    
    def test_model_metrics_validation_invalid_accuracy(self):
        """Test that invalid accuracy raises validation error."""
        with pytest.raises(ValueError, match="Metric 1.5 must be between 0.0 and 1.0"):
            ModelMetrics(
                accuracy=1.5,
                precision=0.82,
                recall=0.88,
                f1_score=0.85,
                training_samples=1000,
                validation_samples=250,
                best_params={"C": 1.0},
                training_time=45.2
            )
    
    def test_model_metrics_validation_negative_samples(self):
        """Test that negative sample counts raise validation error."""
        with pytest.raises(ValueError, match="Training samples must be positive"):
            ModelMetrics(
                accuracy=0.85,
                precision=0.82,
                recall=0.88,
                f1_score=0.85,
                training_samples=-100,
                validation_samples=250,
                best_params={"C": 1.0},
                training_time=45.2
            )


class TestSessionStats:
    """Test cases for the SessionStats model."""
    
    def test_valid_session_stats_creation(self):
        """Test creating valid session stats."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        stats = SessionStats(
            total_articles=20,
            articles_liked=15,
            articles_disliked=5,
            feedback_count=20,
            session_duration=300.0,
            start_time=start_time,
            end_time=end_time
        )
        
        assert stats.total_articles == 20
        assert stats.articles_liked == 15
        assert stats.articles_disliked == 5
        assert stats.feedback_count == 20
        assert stats.session_duration == 300.0
        assert stats.start_time == start_time
        assert stats.end_time == end_time
    
    def test_session_stats_like_ratio(self):
        """Test like ratio calculation."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        stats = SessionStats(
            total_articles=20,
            articles_liked=15,
            articles_disliked=5,
            feedback_count=20,
            session_duration=300.0,
            start_time=start_time,
            end_time=end_time
        )
        
        assert stats.like_ratio == 0.75
    
    def test_session_stats_like_ratio_zero_feedback(self):
        """Test like ratio with zero feedback."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        stats = SessionStats(
            total_articles=20,
            articles_liked=0,
            articles_disliked=0,
            feedback_count=0,
            session_duration=300.0,
            start_time=start_time,
            end_time=end_time
        )
        
        assert stats.like_ratio == 0.0
    
    def test_session_stats_feedback_rate(self):
        """Test feedback rate calculation."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        stats = SessionStats(
            total_articles=20,
            articles_liked=15,
            articles_disliked=5,
            feedback_count=20,
            session_duration=300.0,
            start_time=start_time,
            end_time=end_time
        )
        
        assert stats.feedback_rate == 4.0  # 20 feedback / 5 minutes


class TestRSSFeedConfig:
    """Test cases for the RSSFeedConfig model."""
    
    def test_valid_rss_feed_config_creation(self):
        """Test creating valid RSS feed config."""
        config = RSSFeedConfig(
            name="Test Feed",
            url="https://example.com/feed",
            category="Technology"
        )
        
        assert config.name == "Test Feed"
        assert config.url == "https://example.com/feed"
        assert config.category == "Technology"
        assert config.enabled is True
        assert config.update_frequency == 3600
    
    def test_rss_feed_config_custom_values(self):
        """Test creating RSS feed config with custom values."""
        config = RSSFeedConfig(
            name="Test Feed",
            url="https://example.com/feed",
            category="Technology",
            enabled=False,
            update_frequency=1800
        )
        
        assert config.enabled is False
        assert config.update_frequency == 1800
    
    def test_rss_feed_config_validation_empty_name(self):
        """Test that empty name raises validation error."""
        with pytest.raises(ValueError, match="Feed name cannot be empty"):
            RSSFeedConfig(
                name="",
                url="https://example.com/feed",
                category="Technology"
            )
    
    def test_rss_feed_config_validation_empty_url(self):
        """Test that empty URL raises validation error."""
        with pytest.raises(ValueError, match="Feed URL cannot be empty"):
            RSSFeedConfig(
                name="Test Feed",
                url="",
                category="Technology"
            )
    
    def test_rss_feed_config_validation_invalid_frequency(self):
        """Test that invalid update frequency raises validation error."""
        with pytest.raises(ValueError, match="Update frequency must be positive"):
            RSSFeedConfig(
                name="Test Feed",
                url="https://example.com/feed",
                category="Technology",
                update_frequency=0
            )
