"""
Unit tests for the main application module.
Tests the NewsAggregator class and main workflow.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from src.main import NewsAggregator
from src.models import Article, SessionStats
from src.exceptions import NewsAggregatorError


class TestNewsAggregator:
    """Test cases for the NewsAggregator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.aggregator = NewsAggregator()
    
    def teardown_method(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_aggregator_initialization(self):
        """Test NewsAggregator initialization."""
        assert self.aggregator.rss_parser is not None
        assert self.aggregator.embedder is not None
        assert self.aggregator.trainer is not None
        assert self.aggregator.feedback_collector is not None
        assert self.aggregator.data_saver is not None
    
    @patch('src.main.RSSParser')
    @patch('src.main.ArticleEmbedder')
    @patch('src.main.ModelTrainer')
    @patch('src.main.FeedbackCollector')
    @patch('src.main.DataSaver')
    def test_aggregator_with_mocks(self, mock_saver, mock_feedback, mock_trainer, mock_embedder, mock_parser):
        """Test NewsAggregator with mocked dependencies."""
        aggregator = NewsAggregator()
        
        assert aggregator.rss_parser is not None
        assert aggregator.embedder is not None
        assert aggregator.trainer is not None
        assert aggregator.feedback_collector is not None
        assert aggregator.data_saver is not None
    
    @patch('src.main.RSSParser')
    @patch('src.main.ArticleEmbedder')
    @patch('src.main.ModelTrainer')
    @patch('src.main.FeedbackCollector')
    @patch('src.main.DataSaver')
    def test_run_session_success(self, mock_saver, mock_feedback, mock_trainer, mock_embedder, mock_parser):
        """Test successful session run."""
        # Mock all components
        mock_parser_instance = Mock()
        mock_parser_instance.fetch_all_articles.return_value = [
            Article(source="Test", title="Title 1", summary="Summary 1", link="https://example.com/1", published="2025-01-01"),
            Article(source="Test", title="Title 2", summary="Summary 2", link="https://example.com/2", published="2025-01-01")
        ]
        mock_parser.return_value.__enter__.return_value = mock_parser_instance
        
        mock_embedder_instance = Mock()
        mock_embedder_instance.embed_articles.return_value = None  # Updates articles in place
        mock_embedder.return_value = mock_embedder_instance
        
        mock_trainer_instance = Mock()
        mock_trainer_instance.load_model.return_value = Mock()
        mock_trainer_instance.predict_preferences.return_value = [0.8, 0.3]
        mock_trainer.return_value = mock_trainer_instance
        
        mock_feedback_instance = Mock()
        mock_feedback_instance.collect_feedback.return_value = 2  # 2 articles processed
        mock_feedback.return_value = mock_feedback_instance
        
        mock_saver_instance = Mock()
        mock_saver_instance.save_articles_with_embeddings.return_value = None
        mock_saver.return_value = mock_saver_instance
        
        # Create aggregator and run session
        aggregator = NewsAggregator()
        stats = aggregator.run_session()
        
        # Verify results
        assert isinstance(stats, SessionStats)
        assert stats.articles_fetched == 2
        assert stats.articles_processed == 2
        assert stats.feedback_collected == 2
        assert stats.model_retrained is False  # No retraining in this test
    
    @patch('src.main.RSSParser')
    def test_run_session_rss_error(self, mock_parser):
        """Test session run with RSS fetching error."""
        mock_parser_instance = Mock()
        mock_parser_instance.fetch_all_articles.side_effect = Exception("RSS Error")
        mock_parser.return_value.__enter__.return_value = mock_parser_instance
        
        aggregator = NewsAggregator()
        
        with pytest.raises(NewsAggregatorError, match="RSS"):
            aggregator.run_session()
    
    @patch('src.main.RSSParser')
    @patch('src.main.ArticleEmbedder')
    def test_run_session_embedding_error(self, mock_embedder, mock_parser):
        """Test session run with embedding error."""
        # Mock RSS parser success
        mock_parser_instance = Mock()
        mock_parser_instance.fetch_all_articles.return_value = [
            Article(source="Test", title="Title 1", summary="Summary 1", link="https://example.com/1", published="2025-01-01")
        ]
        mock_parser.return_value.__enter__.return_value = mock_parser_instance
        
        # Mock embedder error
        mock_embedder_instance = Mock()
        mock_embedder_instance.embed_articles.side_effect = Exception("Embedding Error")
        mock_embedder.return_value = mock_embedder_instance
        
        aggregator = NewsAggregator()
        
        with pytest.raises(NewsAggregatorError, match="embedding"):
            aggregator.run_session()
    
    @patch('src.main.RSSParser')
    @patch('src.main.ArticleEmbedder')
    @patch('src.main.ModelTrainer')
    def test_run_session_no_articles(self, mock_trainer, mock_embedder, mock_parser):
        """Test session run with no articles fetched."""
        # Mock RSS parser returning no articles
        mock_parser_instance = Mock()
        mock_parser_instance.fetch_all_articles.return_value = []
        mock_parser.return_value.__enter__.return_value = mock_parser_instance
        
        aggregator = NewsAggregator()
        stats = aggregator.run_session()
        
        # Should handle gracefully
        assert stats.articles_fetched == 0
        assert stats.articles_processed == 0
        assert stats.feedback_collected == 0
    
    @patch('src.main.RSSParser')
    @patch('src.main.ArticleEmbedder')
    @patch('src.main.ModelTrainer')
    @patch('src.main.FeedbackCollector')
    @patch('src.main.DataSaver')
    def test_run_session_with_model_retraining(self, mock_saver, mock_feedback, mock_trainer, mock_embedder, mock_parser):
        """Test session run that triggers model retraining."""
        # Mock components
        mock_parser_instance = Mock()
        mock_parser_instance.fetch_all_articles.return_value = [
            Article(source="Test", title="Title 1", summary="Summary 1", link="https://example.com/1", published="2025-01-01")
        ]
        mock_parser.return_value.__enter__.return_value = mock_parser_instance
        
        mock_embedder_instance = Mock()
        mock_embedder_instance.embed_articles.return_value = None
        mock_embedder.return_value = mock_embedder_instance
        
        mock_trainer_instance = Mock()
        mock_trainer_instance.load_model.return_value = Mock()
        mock_trainer_instance.predict_preferences.return_value = [0.8]
        mock_trainer_instance.retrain_model.return_value = Mock()
        mock_trainer.return_value = mock_trainer_instance
        
        mock_feedback_instance = Mock()
        mock_feedback_instance.collect_feedback.return_value = 1
        mock_feedback.return_value = mock_feedback_instance
        
        mock_saver_instance = Mock()
        mock_saver_instance.save_articles_with_embeddings.return_value = None
        mock_saver.return_value = mock_saver_instance
        
        # Create aggregator and run session
        aggregator = NewsAggregator()
        stats = aggregator.run_session()
        
        # Verify model was retrained
        assert stats.model_retrained is True
        mock_trainer_instance.retrain_model.assert_called_once()
    
    def test_get_session_summary(self):
        """Test session summary generation."""
        # Mock session data
        self.aggregator.articles_fetched = 5
        self.aggregator.articles_processed = 4
        self.aggregator.feedback_collected = 3
        self.aggregator.model_retrained = True
        
        summary = self.aggregator._get_session_summary()
        
        assert summary.articles_fetched == 5
        assert summary.articles_processed == 4
        assert summary.feedback_collected == 3
        assert summary.model_retrained is True
    
    def test_handle_session_error(self):
        """Test session error handling."""
        error = Exception("Test Error")
        
        with pytest.raises(NewsAggregatorError):
            self.aggregator._handle_session_error("test_phase", error)
    
    @patch('src.main.NewsAggregator')
    def test_main_function(self, mock_aggregator_class):
        """Test main function execution."""
        mock_aggregator = Mock()
        mock_aggregator.run_session.return_value = Mock()
        mock_aggregator_class.return_value = mock_aggregator
        
        from src.main import main
        
        # Should not raise any exceptions
        main()
        
        mock_aggregator.run_session.assert_called_once()
    
    def test_aggregator_context_manager(self):
        """Test NewsAggregator as context manager."""
        with NewsAggregator() as aggregator:
            assert aggregator is not None
            assert hasattr(aggregator, 'rss_parser')
    
    def test_aggregator_error_logging(self):
        """Test that errors are properly logged."""
        with patch('src.main.logging.getLogger') as mock_logger:
            mock_logger_instance = Mock()
            mock_logger.return_value = mock_logger_instance
            
            aggregator = NewsAggregator()
            
            # Trigger an error
            with pytest.raises(NewsAggregatorError):
                aggregator._handle_session_error("test", Exception("Test Error"))
            
            # Verify error was logged
            mock_logger_instance.error.assert_called()


class TestNewsAggregatorIntegration:
    """Integration tests for NewsAggregator."""
    
    @pytest.mark.integration
    def test_full_workflow_integration(self):
        """Test full workflow integration (requires real data)."""
        # This would be a real integration test
        # Marked as integration to separate from unit tests
        pass
    
    @pytest.mark.integration
    def test_model_training_integration(self):
        """Test model training integration (requires real data)."""
        # This would test actual model training
        pass


class TestNewsAggregatorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_aggregator_with_none_components(self):
        """Test aggregator behavior with None components."""
        # This would test edge cases
        pass
    
    def test_aggregator_with_empty_config(self):
        """Test aggregator with empty configuration."""
        # This would test configuration edge cases
        pass
