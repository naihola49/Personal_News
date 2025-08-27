"""
Unit tests for the embedder module.
Tests BERT text embedding functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.embedder import ArticleEmbedder
from src.models import Article
from src.exceptions import EmbeddingError


class TestArticleEmbedder:
    """Test cases for the ArticleEmbedder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.embedder = ArticleEmbedder()
    
    def test_embedder_initialization(self):
        """Test embedder initialization."""
        assert self.embedder.model_name == "all-MiniLM-L6-v2"
        assert self.embedder._model is None
    
    @patch('src.embedder.SentenceTransformer')
    def test_model_lazy_loading(self, mock_transformer_class):
        """Test that BERT model is loaded lazily."""
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        
        # Model should not be loaded yet
        assert self.embedder._model is None
        
        # Access model property to trigger loading
        model = self.embedder.model
        
        # Model should now be loaded
        assert self.embedder._model is not None
        mock_transformer_class.assert_called_once_with("all-MiniLM-L6-v2")
    
    @patch('src.embedder.SentenceTransformer')
    def test_model_loading_error(self, mock_transformer_class):
        """Test error handling during model loading."""
        mock_transformer_class.side_effect = Exception("Model loading failed")
        
        with pytest.raises(EmbeddingError, match="model_loading"):
            _ = self.embedder.model
    
    @patch('src.embedder.SentenceTransformer')
    def test_embed_articles_success(self, mock_transformer_class):
        """Test successful article embedding."""
        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_transformer_class.return_value = mock_model
        
        # Create test articles
        articles = [
            Article(source="Test", title="Title 1", summary="Summary 1", link="https://example.com/1", published="2025-01-01"),
            Article(source="Test", title="Title 2", summary="Summary 2", link="https://example.com/2", published="2025-01-01")
        ]
        
        # Test embedding
        embeddings = self.embedder.embed_articles(articles)
        
        assert embeddings.shape == (2, 3)
        assert mock_model.encode.called
        
        # Check that articles have embeddings
        assert articles[0].embedding == [0.1, 0.2, 0.3]
        assert articles[1].embedding == [0.4, 0.5, 0.6]
    
    def test_embed_articles_empty_list(self):
        """Test embedding with empty article list."""
        embeddings = self.embedder.embed_articles([])
        assert embeddings.shape == (0,)
    
    @patch('src.embedder.SentenceTransformer')
    def test_embed_articles_encoding_error(self, mock_transformer_class):
        """Test error handling during article encoding."""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_transformer_class.return_value = mock_model
        
        articles = [
            Article(source="Test", title="Title 1", summary="Summary 1", link="https://example.com/1", published="2025-01-01")
        ]
        
        with pytest.raises(EmbeddingError, match="batch_embedding"):
            self.embedder.embed_articles(articles)
    
    @patch('src.embedder.SentenceTransformer')
    def test_embed_single_article(self, mock_transformer_class):
        """Test single article embedding."""
        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer_class.return_value = mock_model
        
        article = Article(source="Test", title="Title 1", summary="Summary 1", link="https://example.com/1", published="2025-01-01")
        
        embedding = self.embedder.embed_single_article(article)
        
        assert embedding.shape == (3,)
        assert article.embedding == [0.1, 0.2, 0.3]
    
    @patch('src.embedder.SentenceTransformer')
    def test_embed_single_article_error(self, mock_transformer_class):
        """Test error handling during single article embedding."""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_transformer_class.return_value = mock_model
        
        article = Article(source="Test", title="Title 1", summary="Summary 1", link="https://example.com/1", published="2025-01-01")
        
        with pytest.raises(EmbeddingError, match="Title 1"):
            self.embedder.embed_single_article(article)
    
    def test_prepare_single_text(self):
        """Test text preparation for embedding."""
        article = Article(source="Test", title="Test Title", summary="Test Summary", link="https://example.com", published="2025-01-01")
        
        text = self.embedder._prepare_single_text(article)
        
        assert "Test Title" in text
        assert "Test Summary" in text
        assert text == "Test Title Test Summary"
    
    def test_prepare_single_text_with_html(self):
        """Test text preparation with HTML tags."""
        article = Article(source="Test", title="Test Title", summary="<p>Test Summary</p>", link="https://example.com", published="2025-01-01")
        
        with patch('src.embedder.BeautifulSoup') as mock_bs:
            mock_soup = Mock()
            mock_soup.get_text.return_value = "Test Summary"
            mock_bs.return_value = mock_soup
            
            text = self.embedder._prepare_single_text(article)
            
            assert text == "Test Title Test Summary"
            mock_bs.assert_called_once()
    
    def test_prepare_single_text_truncation(self):
        """Test text truncation for long content."""
        long_summary = "A" * 1000  # Very long summary
        article = Article(source="Test", title="Test Title", summary=long_summary, link="https://example.com", published="2025-01-01")
        
        text = self.embedder._prepare_single_text(article)
        
        assert len(text) <= 512 + 3  # 512 chars + "..."
        assert text.endswith("...")
    
    def test_update_articles_with_embeddings(self):
        """Test updating articles with embeddings."""
        articles = [
            Article(source="Test", title="Title 1", summary="Summary 1", link="https://example.com/1", published="2025-01-01"),
            Article(source="Test", title="Title 2", summary="Summary 2", link="https://example.com/2", published="2025-01-01")
        ]
        
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        self.embedder._update_articles_with_embeddings(articles, embeddings)
        
        assert articles[0].embedding == [0.1, 0.2, 0.3]
        assert articles[1].embedding == [0.4, 0.5, 0.6]
    
    def test_update_articles_with_embeddings_mismatch(self):
        """Test handling of article/embedding count mismatch."""
        articles = [
            Article(source="Test", title="Title 1", summary="Summary 1", link="https://example.com/1", published="2025-01-01")
        ]
        
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # More embeddings than articles
        
        # Should not raise error, just log warning
        self.embedder._update_articles_with_embeddings(articles, embeddings)
        
        # Only first article should be updated
        assert articles[0].embedding == [0.1, 0.2, 0.3]
    
    @patch('src.embedder.SentenceTransformer')
    def test_get_embedding_dimension(self, mock_transformer_class):
        """Test getting embedding dimension."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])  # 4-dimensional
        mock_transformer_class.return_value = mock_model
        
        dimension = self.embedder.get_embedding_dimension()
        assert dimension == 4
    
    @patch('src.embedder.SentenceTransformer')
    def test_get_embedding_dimension_error(self, mock_transformer_class):
        """Test getting embedding dimension with error."""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_transformer_class.return_value = mock_model
        
        # Should return default dimension
        dimension = self.embedder.get_embedding_dimension()
        assert dimension == 384  # Default for L6 model
    
    def test_get_embedding_dimension_model_name_based(self):
        """Test getting embedding dimension based on model name."""
        # Test L6 model
        embedder = ArticleEmbedder("all-MiniLM-L6-v2")
        dimension = embedder.get_embedding_dimension()
        assert dimension == 384
        
        # Test L12 model
        embedder = ArticleEmbedder("all-MiniLM-L12-v2")
        dimension = embedder.get_embedding_dimension()
        assert dimension == 768
        
        # Test unknown model
        embedder = ArticleEmbedder("unknown-model")
        dimension = embedder.get_embedding_dimension()
        assert dimension == 768  # Default fallback


class TestArticleEmbedderBackwardCompatibility:
    """Test backward compatibility functions."""
    
    @patch('src.embedder.ArticleEmbedder')
    def test_embed_articles_backward_compatibility(self, mock_embedder_class):
        """Test backward compatibility for embed_articles function."""
        mock_embedder = Mock()
        mock_embedder.embed_articles.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_embedder_class.return_value = mock_embedder
        
        from src.embedder import embed_articles
        
        articles = [Mock()]  # Mock article
        result = embed_articles(articles)
        
        assert result.shape == (1, 3)
        mock_embedder.embed_articles.assert_called_once_with(articles)
