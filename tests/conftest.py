"""
Pytest configuration and common fixtures for the Personal News Aggregator tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

from src.models import Article, FeedbackRecord


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_articles():
    """Create sample articles for testing."""
    return [
        Article(
            source="Test Source 1",
            title="Test Article 1",
            summary="This is a test summary for article 1",
            link="https://example.com/article1",
            published="2025-01-01"
        ),
        Article(
            source="Test Source 2",
            title="Test Article 2",
            summary="This is a test summary for article 2",
            link="https://example.com/article2",
            published="2025-01-01"
        ),
        Article(
            source="Test Source 3",
            title="Test Article 3",
            summary="This is a test summary for article 3",
            link="https://example.com/article3",
            published="2025-01-01"
        )
    ]


@pytest.fixture
def sample_feedback_records():
    """Create sample feedback records for testing."""
    return [
        FeedbackRecord(
            title="Test Article 1",
            link="https://example.com/article1",
            feedback="like",
            embedding=[0.1, 0.2, 0.3, 0.4]
        ),
        FeedbackRecord(
            title="Test Article 2",
            link="https://example.com/article2",
            feedback="dislike",
            embedding=[0.5, 0.6, 0.7, 0.8]
        ),
        FeedbackRecord(
            title="Test Article 3",
            link="https://example.com/article3",
            feedback="like",
            embedding=[0.9, 1.0, 1.1, 1.2]
        )
    ]


@pytest.fixture
def mock_rss_response():
    """Create a mock RSS response for testing."""
    return '''
    <rss version="2.0">
        <channel>
            <title>Test RSS Feed</title>
            <description>Test RSS Feed Description</description>
            <link>https://example.com/feed</link>
            <item>
                <title>Test Article Title</title>
                <description>Test Article Description</description>
                <link>https://example.com/article</link>
                <pubDate>Mon, 01 Jan 2025 12:00:00 +0000</pubDate>
            </item>
        </channel>
    </rss>
    '''


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for testing."""
    mock_model = Mock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4]]
    return mock_model


@pytest.fixture
def mock_ml_model():
    """Create a mock machine learning model for testing."""
    mock_model = Mock()
    mock_model.predict.return_value = [1, 0, 1]  # Binary predictions
    mock_model.predict_proba.return_value = [[0.2, 0.8], [0.9, 0.1], [0.3, 0.7]]  # Probabilities
    return mock_model


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        "rss_feeds": {
            "test": {
                "name": "Test Feed",
                "url": "https://example.com/feed",
                "update_frequency": 3600
            }
        },
        "model_config": {
            "embedding_model": "test-model",
            "max_iter": 1000
        },
        "session_config": {
            "top_articles_count": 5,
            "user_agent": "Test User Agent"
        }
    }


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )


# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark tests with "integration" in name as integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark tests with "slow" in name as slow tests
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark tests that might require external services
        if any(keyword in item.name.lower() for keyword in ["rss", "http", "api"]):
            item.add_marker(pytest.mark.external)


# Test session setup and teardown
def pytest_sessionstart(session):
    """Setup before test session starts."""
    print(f"\nStarting test session with {len(session.items)} tests")


def pytest_sessionfinish(session, exitstatus):
    """Cleanup after test session finishes."""
    print(f"\nTest session finished with status: {exitstatus}")
    if exitstatus == 0:
        print("All tests passed!")
    else:
        print("Some tests failed!")
