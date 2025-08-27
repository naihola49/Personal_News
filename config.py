"""
Configuration file for the Personal News Aggregator.
Centralizes all configuration values for easy maintenance.
"""

import os
from pathlib import Path
from typing import Dict, List

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# File paths
USER_FEEDBACK_PATH = DATA_DIR / "user_feedback.jsonl"
SAVED_ARTICLES_PATH = DATA_DIR / "saved_articles.jsonl"
MODEL_PATH = MODELS_DIR / "user_feedback_model.joblib"

RSS_FEEDS = {
    "nyt": {
        "New York Times - World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "New York Times - Business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "New York Times - Technology": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
        "New York Times - US": "https://rss.nytimes.com/services/xml/rss/nyt/US.xml"
    },
    "wsj": {
        "Wall Street Journal - Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "Wall Street Journal - Technology": "https://feeds.a.dj.com/rss/WSJcomUSBusinessTechnology.xml"
    },
    "ft": {
        "Financial Times - Companies": "https://www.ft.com/companies?format=rss",
        "Financial Times - US": "https://www.ft.com/us?format=rss"
    }
}

# configs
MODEL_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "test_size": 0.2,
    "random_state": 42,
    "max_iter": 1000,
    "cv_folds": 5
}

# tuning
HYPERPARAM_GRID = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__solver': ['liblinear', 'lbfgs', 'saga']
}

# session
SESSION_CONFIG = {
    "top_articles_count": 10,
    "random_sample_count": 7,
    "user_agent": "Mozilla/5.0 (compatible; NewsAggregator/1.0)"
}

# logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "news_aggregator.log"
}
LOGGING_CONFIG["file"].parent.mkdir(exist_ok=True)
