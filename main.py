"""
Main application for the Personal News Aggregator.
Orchestrates the news session with proper error handling and logging.
"""

import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.rss_parser import RSSParser
from src.embedder import ArticleEmbedder
from src.feedback import FeedbackCollector
from src.saver import DataSaver
from src.trainer import ModelTrainer
from src.models import Article, SessionStats
from src.exceptions import NewsAggregatorError
from src.logging_config import setup_logging, log_execution_time
from config import SESSION_CONFIG, USER_FEEDBACK_PATH, MODEL_PATH

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class NewsAggregator:
    """Main application class for the Personal News Aggregator."""
    
    def __init__(self):
        """Initialize the news aggregator."""
        self.rss_parser = RSSParser()
        self.embedder = ArticleEmbedder()
        self.feedback_collector = FeedbackCollector()
        self.data_saver = DataSaver()
        self.model_trainer = ModelTrainer()
        
        logger.info("News Aggregator initialized successfully")
    
    @log_execution_time
    def run_session(self) -> SessionStats:
        """Run a complete news session.
        
        Returns:
            SessionStats object containing session information
            
        Raises:
            NewsAggregatorError: If session fails
        """
        session_start = datetime.now()
        logger.info(f"Starting Personalized News Session: {session_start.strftime('%B %d, %Y')}")
        
        try:
            # Fetch articles
            articles = self._fetch_articles()
            if not articles:
                raise NewsAggregatorError("No articles fetched from any source")
            
            # Embed articles
            embeddings = self._embed_articles(articles)
            
            # Load trained model
            model = self._load_model()
            
            # Select articles for presentation
            selected_articles = self._select_articles(articles, embeddings, model)
            
            # Collect feedback
            session_stats = self._collect_feedback(selected_articles, embeddings, session_start)
            
            # Retrain model if feedback was collected
            if session_stats.feedback_count > 0:
                self._retrain_model()
            
            logger.info("News session completed successfully")
            return session_stats
            
        except Exception as e:
            logger.error(f"News session failed: {e}")
            raise NewsAggregatorError(f"Session failed: {e}")
    
    def _fetch_articles(self) -> List[Article]:
        """Fetch articles from RSS feeds.
        
        Returns:
            List of Article objects
            
        Raises:
            NewsAggregatorError: If fetching fails
        """
        try:
            logger.info("Fetching articles from RSS feeds")
            articles = self.rss_parser.fetch_all_feeds()
            
            if not articles:
                logger.warning("No articles fetched from any source")
                return []
            
            logger.info(f"Successfully fetched {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch articles: {e}")
            raise NewsAggregatorError(f"Article fetching failed: {e}")
    
    def _embed_articles(self, articles: List[Article]) -> List[List[float]]:
        """Embed articles using BERT model.
        
        Args:
            articles: List of Article objects
            
        Returns:
            List of article embeddings
        """
        try:
            logger.info("Generating article embeddings")
            embeddings = self.embedder.embed_articles(articles)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Failed to embed articles: {e}")
            raise NewsAggregatorError(f"Article embedding failed: {e}")
    
    def _load_model(self):
        """Load the trained recommendation model.
        
        Returns:
            Loaded sklearn model
            
        Raises:
            NewsAggregatorError: If model loading fails
        """
        try:
            logger.info("Loading trained recommendation model")
            model = self.model_trainer.load_model()
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise NewsAggregatorError(f"Model loading failed: {e}")
    
    def _select_articles(self, articles: List[Article], embeddings: List[List[float]], model) -> List[Article]:
        """Select articles for presentation using the model.
        
        Args:
            articles: List of Article objects
            embeddings: List of article embeddings
            model: Trained recommendation model
            
        Returns:
            List of selected Article objects
        """
        try:
            logger.info("Selecting articles for presentation")
            
            # Get model predictions
            import numpy as np
            embeddings_array = np.array(embeddings)
            probabilities = model.predict_proba(embeddings_array)[:, 1]
            
            # Select top articles
            top_count = SESSION_CONFIG["top_articles_count"]
            random_count = SESSION_CONFIG["random_sample_count"]
            
            # Get top articles by probability
            top_indices = np.argsort(probabilities)[::-1][:top_count]
            top_articles = [articles[i] for i in top_indices]
            
            # Get random sample from remaining articles
            remaining_indices = list(set(range(len(articles))) - set(top_indices))
            if len(remaining_indices) > 0:
                random_indices = random.sample(remaining_indices, min(random_count, len(remaining_indices)))
                random_articles = [articles[i] for i in random_indices]
            else:
                random_articles = []
            
            # Combine and shuffle
            selected_articles = top_articles + random_articles
            random.shuffle(selected_articles)
            
            logger.info(f"Selected {len(selected_articles)} articles for presentation")
            return selected_articles
            
        except Exception as e:
            logger.error(f"Failed to select articles: {e}")
            raise NewsAggregatorError(f"Article selection failed: {e}")
    
    def _collect_feedback(self, articles: List[Article], embeddings: List[List[float]], 
                         session_start: datetime) -> SessionStats:
        """Collect user feedback for selected articles.
        
        Args:
            articles: List of Article objects
            embeddings: List of article embeddings
            session_start: Session start time
            
        Returns:
            SessionStats object
        """
        try:
            logger.info("Starting feedback collection")
            
            like_count = 0
            dislike_count = 0
            feedback_count = 0
            
            for i, article in enumerate(articles):
                try:
                    # Display article
                    self._display_article(article, i + 1, len(articles))
                    
                    # Get user feedback
                    feedback = self._get_user_feedback(article)
                    if feedback is None:
                        continue
                    
                    # Save feedback
                    embedding = embeddings[articles.index(article)] if article in articles else []
                    self.feedback_collector.append_feedback_record(article, feedback, embedding)
                    
                    # Update counts
                    if feedback == 1:
                        like_count += 1
                    else:
                        dislike_count += 1
                    feedback_count += 1
                    
                except KeyboardInterrupt:
                    logger.info("User interrupted feedback collection")
                    break
                except Exception as e:
                    logger.warning(f"Failed to collect feedback for article '{article.title}': {e}")
                    continue
            
            session_end = datetime.now()
            session_duration = (session_end - session_start).total_seconds()
            
            # Create session stats
            session_stats = SessionStats(
                total_articles=len(articles),
                articles_liked=like_count,
                articles_disliked=dislike_count,
                feedback_count=feedback_count,
                session_duration=session_duration,
                start_time=session_start,
                end_time=session_end
            )
            
            logger.info(f"Feedback collection completed: {feedback_count} articles rated")
            return session_stats
            
        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            raise NewsAggregatorError(f"Feedback collection failed: {e}")
    
    def _display_article(self, article: Article, article_num: int, total_articles: int) -> None:
        """Display article information to user.
        
        Args:
            article: Article object to display
            article_num: Current article number
            total_articles: Total number of articles
        """
        print("\n" + "="*80)
        print(f"Article {article_num} of {total_articles}")
        print(f"Source: {article.source}")
        print(f"Title: {article.title}")
        print(f"Summary: {article.summary}")
        print(f"Link: {article.link}")
        print(f"Published: {article.published}")
        print("="*80)
    
    def _get_user_feedback(self, article: Article) -> Optional[int]:
        """Get user feedback for an article.
        
        Args:
            article: Article object
            
        Returns:
            Feedback label (1 for like, 0 for dislike) or None if skipped
        """
        while True:
            try:
                feedback_input = input("Feedback ([l]ike / [d]islike / [s]kip): ").strip().lower()
                
                if feedback_input in ['l', 'like']:
                    return 1
                elif feedback_input in ['d', 'dislike']:
                    return 0
                elif feedback_input in ['s', 'skip']:
                    logger.info(f"User skipped article: {article.title[:50]}...")
                    return None
                else:
                    print("Invalid input. Enter 'l' for like, 'd' for dislike, or 's' to skip.")
                    
            except (EOFError, KeyboardInterrupt):
                logger.info("User interrupted feedback input")
                return None
    
    def _retrain_model(self) -> None:
        """Retrain the recommendation model with new feedback."""
        try:
            logger.info("Retraining model with new feedback")
            self.model_trainer.retrain_model(USER_FEEDBACK_PATH, MODEL_PATH)
            logger.info("Model retraining completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to retrain model: {e}")
            # Don't raise here - model retraining failure shouldn't crash the session


def main():
    """Main entry point for the application."""
    try:
        aggregator = NewsAggregator()
        session_stats = aggregator.run_session()
        
        # Display session summary
        print("\n" + "="*80)
        print("Session Completed")
        print("="*80)
        print(f"Articles Liked: {session_stats.articles_liked}")
        print(f"Articles Disliked: {session_stats.articles_disliked}")
        print(f"Total Feedback Collected: {session_stats.feedback_count}")
        print(f"Session Duration: {session_stats.session_duration:.1f} seconds")
        print(f"Like Ratio: {session_stats.like_ratio:.2%}")
        print("="*80)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\nApplication interrupted by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print(f"\nApplication failed: {e}")
        raise


if __name__ == "__main__":
    main()
