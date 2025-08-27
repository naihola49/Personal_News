"""
Custom exceptions for the Personal News Aggregator.
Provides specific error types for different failure scenarios.
"""


class NewsAggregatorError(Exception):
    """Base exception for all news aggregator errors."""
    pass


class RSSFetchError(NewsAggregatorError):
    """Raised when RSS feed fetching fails."""
    
    def __init__(self, source: str, url: str, original_error: Exception = None):
        self.source = source
        self.url = url
        self.original_error = original_error
        super().__init__(f"Failed to fetch RSS feed from {source} at {url}: {original_error}")


class EmbeddingError(NewsAggregatorError):
    """Raised when article embedding fails."""
    
    def __init__(self, article_title: str, original_error: Exception = None):
        self.article_title = article_title
        self.original_error = original_error
        super().__init__(f"Failed to embed article '{article_title}': {original_error}")


class ModelError(NewsAggregatorError):
    """Raised when model operations fail."""
    
    def __init__(self, operation: str, original_error: Exception = None):
        self.operation = operation
        self.original_error = original_error
        super().__init__(f"Model operation '{operation}' failed: {original_error}")


class DataError(NewsAggregatorError):
    """Raised when data operations fail."""
    
    def __init__(self, operation: str, file_path: str, original_error: Exception = None):
        self.operation = operation
        self.file_path = file_path
        self.original_error = original_error
        super().__init__(f"Data operation '{operation}' failed for {file_path}: {original_error}")


class ConfigurationError(NewsAggregatorError):
    """Raised when configuration is invalid."""
    
    def __init__(self, config_key: str, message: str):
        self.config_key = config_key
        super().__init__(f"Configuration error for '{config_key}': {message}")


class ValidationError(NewsAggregatorError):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, value: str, message: str):
        self.field = field
        self.value = value
        super().__init__(f"Validation error for field '{field}' with value '{value}': {message}")
