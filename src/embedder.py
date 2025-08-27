"""
Article Embedder for the Personal News Aggregator.
Converts article text into BERT embeddings for machine learning.
"""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from .exceptions import EmbeddingError
from .models import Article
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class ArticleEmbedder:
    """Handles article text embedding using BERT models."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedder.
        
        Args:
            model_name: Name of the BERT model to use
        """
        self.model_name = model_name or MODEL_CONFIG["embedding_model"]
        self._model = None
        logger.info(f"Initializing embedder with model: {self.model_name}")
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the BERT model."""
        if self._model is None:
            try:
                logger.info(f"Loading BERT model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Successfully loaded model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load BERT model {self.model_name}: {e}")
                raise EmbeddingError("model_loading", e)
        return self._model
    
    def embed_articles(self, articles: List[Article]) -> np.ndarray:
        """Embed a list of articles.
        
        Args:
            articles: List of Article objects to embed
            
        Returns:
            numpy array of embeddings with shape (n_articles, embedding_dim)
            
        Raises:
            EmbeddingError: If embedding fails
        """
        if not articles:
            logger.warning("No articles provided for embedding")
            return np.array([])
        
        try:
            # Prepare text for embedding
            texts = self._prepare_texts(articles)
            logger.info(f"Embedding {len(texts)} articles")
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Update articles with embeddings
            self._update_articles_with_embeddings(articles, embeddings)
            
            logger.info(f"Successfully embedded {len(articles)} articles")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed articles: {e}")
            raise EmbeddingError("batch_embedding", e)
    
    def embed_single_article(self, article: Article) -> np.ndarray:
        """Embed a single article.
        
        Args:
            article: Article object to embed
            
        Returns:
            numpy array of embedding with shape (embedding_dim,)
            
        Raises:
            EmbeddingError: If embedding fails
        """
        try:
            text = self._prepare_single_text(article)
            logger.debug(f"Embedding single article: {article.title[:50]}...")
            
            embedding = self.model.encode(
                [text], 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Update article with embedding
            article.embedding = embedding[0].tolist()
            
            logger.debug(f"Successfully embedded article: {article.title[:50]}...")
            return embedding[0]
            
        except Exception as e:
            logger.error(f"Failed to embed article '{article.title}': {e}")
            raise EmbeddingError(article.title, e)
    
    def _prepare_texts(self, articles: List[Article]) -> List[str]:
        """Prepare article texts for embedding.
        
        Args:
            articles: List of Article objects
            
        Returns:
            List of prepared text strings
        """
        texts = []
        for article in articles:
            text = self._prepare_single_text(article)
            texts.append(text)
        return texts
    
    def _prepare_single_text(self, article: Article) -> str:
        """Prepare a single article's text for embedding.
        
        Args:
            article: Article object
            
        Returns:
            Prepared text string
        """
        # Combine title and summary with proper formatting
        title = article.title.strip()
        summary = article.summary.strip()
        
        # Remove any HTML tags that might be in the summary
        if "<" in summary and ">" in summary:
            from bs4 import BeautifulSoup
            summary = BeautifulSoup(summary, "html.parser").get_text()
        
        # Combine with separator
        combined_text = f"{title} {summary}"
        
        # Truncate if too long (BERT models have token limits)
        max_length = 512  # Conservative limit for most BERT models
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length] + "..."
        
        return combined_text
    
    def _update_articles_with_embeddings(self, articles: List[Article], embeddings: np.ndarray):
        """Update Article objects with their embeddings.
        
        Args:
            articles: List of Article objects
            embeddings: numpy array of embeddings
        """
        if len(articles) != len(embeddings):
            logger.warning(f"Mismatch between articles ({len(articles)}) and embeddings ({len(embeddings)})")
            return
        
        for article, embedding in zip(articles, embeddings):
            article.embedding = embedding.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension
        """
        try:
            # Create a dummy embedding to get the dimension
            dummy_text = "test"
            dummy_embedding = self.model.encode([dummy_text])
            return dummy_embedding.shape[1]
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}")
            # Return a default value based on the model name
            if "L6" in self.model_name:
                return 384
            elif "L12" in self.model_name:
                return 768
            else:
                return 768  # Default fallback
    
    def __del__(self):
        """Cleanup when the embedder is destroyed."""
        if hasattr(self, '_model') and self._model is not None:
            del self._model


# Backward compatibility function
def embed_articles(articles: List[Article]) -> np.ndarray:
    """Embed articles (backward compatibility)."""
    embedder = ArticleEmbedder()
    return embedder.embed_articles(articles)
