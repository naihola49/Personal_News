"""
Model training script for the Personal News Aggregator.
Trains the recommendation model from collected feedback data.
"""

import sys
import logging
from pathlib import Path

from src.trainer import ModelTrainer
from src.logging_config import setup_logging
from config import USER_FEEDBACK_PATH, MODEL_PATH

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    try:
        logger.info("Starting model training process")
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Check if feedback data exists
        if not USER_FEEDBACK_PATH.exists():
            logger.error(f"Feedback file not found: {USER_FEEDBACK_PATH}")
            print(f"Error: No feedback data found at {USER_FEEDBACK_PATH}")
            print("Please run the news aggregator first to collect some feedback.")
            sys.exit(1)
        
        # Train the model
        logger.info("Training model from feedback data")
        model = trainer.retrain_model(USER_FEEDBACK_PATH, MODEL_PATH)
        
        # Get model info
        model_info = trainer.get_model_info()
        
        # Display results
        print("Model training completed successfully!")
        print(f"Model saved to: {MODEL_PATH}")
        
        if trainer.metrics:
            print(f"Validation Accuracy: {trainer.metrics.accuracy:.3f}")
            print(f"Precision: {trainer.metrics.precision:.3f}")
            print(f"Recall: {trainer.metrics.recall:.3f}")
            print(f"F1 Score: {trainer.metrics.f1_score:.3f}")
            print(f"Training Time: {trainer.metrics.training_time:.2f} seconds")
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        print(f"Error: Model training failed - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

