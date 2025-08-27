"""
Feedback Collection for the Personal News Aggregator.
Handles user feedback collection and storage with proper validation.
"""

import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .exceptions import DataError, ValidationError
from .models import Article, FeedbackRecord
from config import USER_FEEDBACK_PATH

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Handles collection and storage of user feedback."""
    
    def __init__(self, feedback_path: Optional[Path] = None):
        """Initialize the feedback collector.
        
        Args:
            feedback_path: Path to the feedback JSONL file
        """
        self.feedback_path = Path(feedback_path) if feedback_path else USER_FEEDBACK_PATH
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized FeedbackCollector with path: {self.feedback_path}")
    
    def collect_feedback(self, articles_path: Path, feedback_path: Optional[Path] = None) -> List[FeedbackRecord]:
        """Collect feedback for articles from a saved articles file.
        
        Args:
            articles_path: Path to the saved articles JSONL file
            feedback_path: Path to save feedback (uses default if None)
            
        Returns:
            List of collected FeedbackRecord objects
            
        Raises:
            DataError: If collection fails
        """
        try:
            save_path = Path(feedback_path) if feedback_path else self.feedback_path
            
            logger.info(f"Starting feedback collection from {articles_path}")
            
            # Load articles
            articles = self._load_articles(articles_path)
            if not articles:
                logger.warning("No articles found for feedback collection")
                return []
            
            # Collect feedback for each article
            feedback_records = []
            for article in articles:
                try:
                    feedback_record = self._collect_single_feedback(article)
                    if feedback_record:
                        feedback_records.append(feedback_record)
                except Exception as e:
                    logger.warning(f"Failed to collect feedback for article '{article.title}': {e}")
                    continue
            
            # Save collected feedback
            self._save_feedback_records(feedback_records, save_path)
            
            logger.info(f"Successfully collected feedback for {len(feedback_records)} articles")
            return feedback_records
            
        except Exception as e:
            logger.error(f"Feedback collection failed: {e}")
            raise DataError("feedback_collection", str(articles_path), e)
    
    def append_feedback_record(self, article: Article, label: int, embedding: List[float], 
                             feedback_path: Optional[Path] = None) -> FeedbackRecord:
        """Append a new feedback record for an article.
        
        Args:
            article: Article object
            label: Feedback label (1 for like, 0 for dislike)
            embedding: Article embedding as list
            feedback_path: Path to save feedback (uses default if None)
            
        Returns:
            Created FeedbackRecord object
            
        Raises:
            ValidationError: If input validation fails
            DataError: If saving fails
        """
        try:
            save_path = Path(feedback_path) if feedback_path else self.feedback_path
            
            # Validate inputs
            self._validate_feedback_inputs(article, label, embedding)
            
            # Create feedback record
            feedback_record = FeedbackRecord(
                title=article.title,
                link=article.link,
                feedback=label,
                embedding=embedding
            )
            
            # Save to file
            self._append_feedback_to_file(feedback_record, save_path)
            
            logger.info(f"Successfully saved feedback for article: {article.title[:50]}...")
            return feedback_record
            
        except Exception as e:
            if isinstance(e, (ValidationError, DataError)):
                raise
            logger.error(f"Failed to append feedback record: {e}")
            raise DataError("feedback_append", str(self.feedback_path), e)
    
    def get_feedback_history(self, feedback_path: Optional[Path] = None) -> List[FeedbackRecord]:
        """Load all feedback records from file.
        
        Args:
            feedback_path: Path to load feedback from (uses default if None)
            
        Returns:
            List of FeedbackRecord objects
            
        Raises:
            DataError: If loading fails
        """
        try:
            load_path = Path(feedback_path) if feedback_path else self.feedback_path
            
            if not load_path.exists():
                logger.info(f"Feedback file does not exist: {load_path}")
                return []
            
            feedback_records = []
            with open(load_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record_data = json.loads(line.strip())
                        feedback_record = FeedbackRecord(**record_data)
                        feedback_records.append(feedback_record)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Invalid feedback record at line {line_num}: {e}")
                        continue
            
            logger.info(f"Loaded {len(feedback_records)} feedback records from {load_path}")
            return feedback_records
            
        except Exception as e:
            logger.error(f"Failed to load feedback history: {e}")
            raise DataError("feedback_load", str(self.feedback_path), e)
    
    def get_feedback_stats(self, feedback_path: Optional[Path] = None) -> Dict[str, Any]:
        """Get statistics about collected feedback.
        
        Args:
            feedback_path: Path to load feedback from (uses default if None)
            
        Returns:
            Dictionary containing feedback statistics
        """
        try:
            feedback_records = self.get_feedback_history(feedback_path)
            
            if not feedback_records:
                return {
                    "total_feedback": 0,
                    "likes": 0,
                    "dislikes": 0,
                    "like_ratio": 0.0
                }
            
            total = len(feedback_records)
            likes = sum(1 for record in feedback_records if record.feedback == 1)
            dislikes = total - likes
            like_ratio = likes / total if total > 0 else 0.0
            
            stats = {
                "total_feedback": total,
                "likes": likes,
                "dislikes": dislikes,
                "like_ratio": like_ratio,
                "first_feedback": min(record.timestamp for record in feedback_records),
                "last_feedback": max(record.timestamp for record in feedback_records)
            }
            
            logger.info(f"Feedback stats: {total} total, {likes} likes, {dislikes} dislikes")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return {"error": str(e)}
    
    def _load_articles(self, articles_path: Path) -> List[Article]:
        """Load articles from JSONL file.
        
        Args:
            articles_path: Path to articles file
            
        Returns:
            List of Article objects
        """
        articles = []
        try:
            with open(articles_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        article_data = json.loads(line.strip())
                        article = Article(**article_data)
                        articles.append(article)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Invalid article at line {line_num}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Failed to load articles from {articles_path}: {e}")
            raise DataError("articles_load", str(articles_path), e)
        
        return articles
    
    def _collect_single_feedback(self, article: Article) -> Optional[FeedbackRecord]:
        """Collect feedback for a single article.
        
        Args:
            article: Article object
            
        Returns:
            FeedbackRecord if feedback collected, None if skipped
        """
        # Display article information
        print("\n" + "="*80)
        print(f"Source: {article.source}")
        print(f"Title: {article.title}")
        print(f"Summary: {article.summary[:200]}...")
        print(f"Link: {article.link}")
        print("="*80)
        
        # Get user feedback
        while True:
            try:
                feedback_input = input("Feedback ([l]ike / [d]islike / [s]kip): ").strip().lower()
                
                if feedback_input in ['l', 'like']:
                    label = 1
                    break
                elif feedback_input in ['d', 'dislike']:
                    label = 0
                    break
                elif feedback_input in ['s', 'skip']:
                    logger.info(f"User skipped article: {article.title[:50]}...")
                    return None
                else:
                    print("Invalid input. Enter 'l' for like, 'd' for dislike, or 's' to skip.")
            except (EOFError, KeyboardInterrupt):
                logger.info("User interrupted feedback collection")
                return None
        
        # Create feedback record
        if article.embedding:
            embedding = article.embedding
        else:
            logger.warning(f"No embedding found for article: {article.title}")
            embedding = []
        
        return FeedbackRecord(
            title=article.title,
            link=article.link,
            feedback=label,
            embedding=embedding
        )
    
    def _validate_feedback_inputs(self, article: Article, label: int, embedding: List[float]) -> None:
        """Validate feedback input parameters.
        
        Args:
            article: Article object
            label: Feedback label
            embedding: Article embedding
            
        Raises:
            ValidationError: If validation fails
        """
        if not article.title.strip():
            raise ValidationError("title", article.title, "Title cannot be empty")
        
        if not article.link.strip():
            raise ValidationError("link", article.link, "Link cannot be empty")
        
        if label not in [0, 1]:
            raise ValidationError("label", str(label), "Label must be 0 (dislike) or 1 (like)")
        
        if not isinstance(embedding, list) or len(embedding) == 0:
            raise ValidationError("embedding", str(embedding), "Embedding must be a non-empty list")
    
    def _save_feedback_records(self, feedback_records: List[FeedbackRecord], save_path: Path) -> None:
        """Save feedback records to file.
        
        Args:
            feedback_records: List of FeedbackRecord objects
            save_path: Path to save feedback
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w") as f:
            for record in feedback_records:
                f.write(json.dumps(record.__dict__, default=str) + "\n")
        
        logger.info(f"Saved {len(feedback_records)} feedback records to {save_path}")
    
    def _append_feedback_to_file(self, feedback_record: FeedbackRecord, save_path: Path) -> None:
        """Append a single feedback record to file.
        
        Args:
            feedback_record: FeedbackRecord to append
            save_path: Path to append feedback to
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "a") as f:
            f.write(json.dumps(feedback_record.__dict__, default=str) + "\n")


# Backward compatibility functions
def collect_feedback(jsonl_path: str, feedback_path: str) -> None:
    """Collect feedback (backward compatibility)."""
    collector = FeedbackCollector()
    collector.collect_feedback(Path(jsonl_path), Path(feedback_path))


def append_feedback_record(article: Article, label: int, embedding: List[float], 
                         feedback_path: str = "user_feedback.jsonl") -> FeedbackRecord:
    """Append feedback record (backward compatibility)."""
    collector = FeedbackCollector()
    return collector.append_feedback_record(article, label, embedding, Path(feedback_path))
