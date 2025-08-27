"""
Data models for the Personal News Aggregator.
Uses dataclasses for type safety and data validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class Article:
    """Represents a news article with all its metadata."""
    
    source: str
    title: str
    summary: str
    link: str
    published: str
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate article data after initialization."""
        if not self.title.strip():
            raise ValueError("Article title cannot be empty")
        if not self.link.strip():
            raise ValueError("Article link cannot be empty")
        if not self.summary.strip():
            raise ValueError("Article summary cannot be empty")
    
    @property
    def title_length(self) -> int:
        """Get the length of the article title."""
        return len(self.title)
    
    @property
    def summary_length(self) -> int:
        """Get the length of the article summary."""
        return len(self.summary)


@dataclass
class FeedbackRecord:
    """Represents user feedback for an article."""
    
    title: str
    link: str
    feedback: int  # 1 for like, 0 for dislike
    embedding: List[float]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate feedback data after initialization."""
        if self.feedback not in [0, 1]:
            raise ValueError("Feedback must be 0 (dislike) or 1 (like)")
        if not self.title.strip():
            raise ValueError("Article title cannot be empty")
        if not self.link.strip():
            raise ValueError("Article link cannot be empty")
        if not isinstance(self.embedding, list) or len(self.embedding) == 0:
            raise ValueError("Embedding must be a non-empty list")


@dataclass
class ModelMetrics:
    """Represents model performance metrics."""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_samples: int
    validation_samples: int
    best_params: Dict[str, Any]
    training_time: float
    
    def __post_init__(self):
        """Validate metrics data after initialization."""
        for metric in [self.accuracy, self.precision, self.recall, self.f1_score]:
            if not 0.0 <= metric <= 1.0:
                raise ValueError(f"Metric {metric} must be between 0.0 and 1.0")
        if self.training_samples <= 0:
            raise ValueError("Training samples must be positive")
        if self.validation_samples <= 0:
            raise ValueError("Validation samples must be positive")
        if self.training_time < 0:
            raise ValueError("Training time cannot be negative")


@dataclass
class SessionStats:
    """Represents statistics for a news session."""
    
    total_articles: int
    articles_liked: int
    articles_disliked: int
    feedback_count: int
    session_duration: float
    start_time: datetime
    end_time: datetime
    
    @property
    def like_ratio(self) -> float:
        """Calculate the ratio of liked articles."""
        if self.feedback_count == 0:
            return 0.0
        return self.articles_liked / self.feedback_count
    
    @property
    def feedback_rate(self) -> float:
        """Calculate feedback rate per minute."""
        if self.session_duration == 0:
            return 0.0
        return self.feedback_count / (self.session_duration / 60)


@dataclass
class RSSFeedConfig:
    """Configuration for an RSS feed."""
    
    name: str
    url: str
    category: str
    enabled: bool = True
    update_frequency: int = 3600  # seconds
    
    def __post_init__(self):
        """Validate RSS feed configuration."""
        if not self.name.strip():
            raise ValueError("Feed name cannot be empty")
        if not self.url.strip():
            raise ValueError("Feed URL cannot be empty")
        if self.update_frequency <= 0:
            raise ValueError("Update frequency must be positive")
