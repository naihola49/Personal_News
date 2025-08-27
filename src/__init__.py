"""
Personal News Aggregator Package

A machine learning-powered news aggregator that learns user preferences
through BERT embeddings and continuous feedback.
"""

__version__ = "1.0.0"
__author__ = "News Aggregator Team"

from .rss_parser import RSSParser
from .embedder import ArticleEmbedder
from .trainer import ModelTrainer
from .feedback import FeedbackCollector
from .saver import DataSaver

__all__ = [
    "RSSParser",
    "ArticleEmbedder", 
    "ModelTrainer",
    "FeedbackCollector",
    "DataSaver"
]
