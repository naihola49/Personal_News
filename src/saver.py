"""
Data Saver for the Personal News Aggregator.
Handles saving articles and embeddings with proper error handling.
"""

import logging
import json
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime
import numpy as np

from .exceptions import DataError
from .models import Article
from config import SAVED_ARTICLES_PATH

logger = logging.getLogger(__name__)


class DataSaver:
    """Handles saving articles and embeddings to disk."""
    
    def __init__(self, save_path: Optional[Path] = None):
        """Initialize the data saver.
        
        Args:
            save_path: Path to save articles (uses default if None)
        """
        self.save_path = Path(save_path) if save_path else SAVED_ARTICLES_PATH
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DataSaver with path: {self.save_path}")
    
    def save_articles_with_embeddings(self, articles: List[Article], 
                                    embeddings: Union[np.ndarray, List[List[float]]], 
                                    save_path: Optional[Path] = None) -> None:
        """Save articles with their embeddings to JSONL file.
        
        Args:
            articles: List of Article objects
            embeddings: Article embeddings as numpy array or list of lists
            save_path: Path to save articles (uses default if None)
            
        Raises:
            DataError: If saving fails
        """
        try:
            save_path = Path(save_path) if save_path else self.save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not articles:
                logger.warning("No articles provided for saving")
                return
            
            # Convert embeddings to list format if needed
            if isinstance(embeddings, np.ndarray):
                embedding_list = embeddings.tolist()
            else:
                embedding_list = embeddings
            
            if len(articles) != len(embedding_list):
                raise DataError("save", str(save_path), 
                              ValueError(f"Mismatch between articles ({len(articles)}) and embeddings ({len(embedding_list)})"))
            
            logger.info(f"Saving {len(articles)} articles with embeddings to {save_path}")
            
            # Save articles with embeddings
            with open(save_path, "w") as f:
                for article, embedding in zip(articles, embedding_list):
                    record = {
                        "source": article.source,
                        "title": article.title,
                        "summary": article.summary,
                        "link": article.link,
                        "published": article.published,
                        "embedding": embedding
                    }
                    f.write(json.dumps(record) + "\n")
            
            logger.info(f"Successfully saved {len(articles)} articles to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save articles: {e}")
            raise DataError("articles_save", str(save_path), e)
    
    def save_articles(self, articles: List[Article], save_path: Optional[Path] = None) -> None:
        """Save articles without embeddings to JSONL file.
        
        Args:
            articles: List of Article objects
            save_path: Path to save articles (uses default if None)
            
        Raises:
            DataError: If saving fails
        """
        try:
            save_path = Path(save_path) if save_path else self.save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not articles:
                logger.warning("No articles provided for saving")
                return
            
            logger.info(f"Saving {len(articles)} articles to {save_path}")
            
            with open(save_path, "w") as f:
                for article in articles:
                    record = {
                        "source": article.source,
                        "title": article.title,
                        "summary": article.summary,
                        "link": article.link,
                        "published": article.published
                    }
                    if article.embedding:
                        record["embedding"] = article.embedding
                    f.write(json.dumps(record) + "\n")
            
            logger.info(f"Successfully saved {len(articles)} articles to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save articles: {e}")
            raise DataError("articles_save", str(save_path), e)
    
    def load_articles(self, load_path: Optional[Path] = None) -> List[Article]:
        """Load articles from JSONL file.
        
        Args:
            load_path: Path to load articles from (uses default if None)
            
        Returns:
            List of Article objects
            
        Raises:
            DataError: If loading fails
        """
        try:
            load_path = Path(load_path) if load_path else self.save_path
            
            if not load_path.exists():
                logger.info(f"Articles file does not exist: {load_path}")
                return []
            
            articles = []
            with open(load_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        article_data = json.loads(line.strip())
                        article = Article(**article_data)
                        articles.append(article)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Invalid article at line {line_num}: {e}")
                        continue
            
            logger.info(f"Successfully loaded {len(articles)} articles from {load_path}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to load articles: {e}")
            raise DataError("articles_load", str(load_path), e)
    
    def append_article(self, article: Article, save_path: Optional[Path] = None) -> None:
        """Append a single article to the articles file.
        
        Args:
            article: Article object to append
            save_path: Path to append article to (uses default if None)
            
        Raises:
            DataError: If appending fails
        """
        try:
            save_path = Path(save_path) if save_path else self.save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            record = {
                "source": article.source,
                "title": article.title,
                "summary": article.summary,
                "link": article.link,
                "published": article.published
            }
            if article.embedding:
                record["embedding"] = article.embedding
            
            with open(save_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            
            logger.info(f"Successfully appended article: {article.title[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to append article: {e}")
            raise DataError("article_append", str(save_path), e)
    
    def get_articles_info(self, load_path: Optional[Path] = None) -> dict:
        """Get information about saved articles.
        
        Args:
            load_path: Path to load articles from (uses default if None)
            
        Returns:
            Dictionary containing articles information
        """
        try:
            articles = self.load_articles(load_path)
            
            if not articles:
                return {
                    "total_articles": 0,
                    "sources": {},
                    "articles_with_embeddings": 0
                }
            
            # Count articles by source
            sources = {}
            articles_with_embeddings = 0
            
            for article in articles:
                source = article.source
                sources[source] = sources.get(source, 0) + 1
                
                if article.embedding:
                    articles_with_embeddings += 1
            
            info = {
                "total_articles": len(articles),
                "sources": sources,
                "articles_with_embeddings": articles_with_embeddings,
                "embedding_coverage": articles_with_embeddings / len(articles) if articles else 0.0
            }
            
            logger.info(f"Articles info: {len(articles)} total, {articles_with_embeddings} with embeddings")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get articles info: {e}")
            return {"error": str(e)}
    
    def backup_articles(self, backup_path: Optional[Path] = None) -> Path:
        """Create a backup of the articles file.
        
        Args:
            backup_path: Path for backup (uses default with timestamp if None)
            
        Returns:
            Path to the backup file
            
        Raises:
            DataError: If backup fails
        """
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.save_path.parent / f"articles_backup_{timestamp}.jsonl"
            
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the current file
            import shutil
            shutil.copy2(self.save_path, backup_path)
            
            logger.info(f"Successfully created backup at {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise DataError("backup", str(self.save_path), e)


# Backward compatibility function
def save_articles_with_embeddings(articles: List[Article], embeddings: Union[np.ndarray, List[List[float]]], 
                                filepath: str) -> None:
    """Save articles with embeddings (backward compatibility)."""
    saver = DataSaver()
    saver.save_articles_with_embeddings(articles, embeddings, Path(filepath))
