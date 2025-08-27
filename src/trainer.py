"""
Model Trainer for the Personal News Aggregator.
Handles model training, validation, and persistence with proper error handling.
"""

import logging
import json
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from .exceptions import ModelError, DataError
from .models import ModelMetrics, FeedbackRecord
from config import MODEL_CONFIG, HYPERPARAM_GRID, MODEL_PATH

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, validation, and persistence."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the model trainer.
        
        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.model = None
        self.metrics = None
        logger.info(f"Initialized ModelTrainer with model path: {self.model_path}")
    
    def load_feedback(self, feedback_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load feedback data from JSONL file.
        
        Args:
            feedback_path: Path to the feedback JSONL file
            
        Returns:
            Tuple of (features, labels) as numpy arrays
            
        Raises:
            DataError: If loading fails
        """
        try:
            logger.info(f"Loading feedback data from {feedback_path}")
            
            if not feedback_path.exists():
                raise DataError("load", str(feedback_path), FileNotFoundError("Feedback file not found"))
            
            features = []
            labels = []
            
            with open(feedback_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        features.append(record["embedding"])
                        labels.append(record["feedback"])
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Invalid feedback record at line {line_num}: {e}")
                        continue
            
            if not features:
                raise DataError("load", str(feedback_path), ValueError("No valid feedback records found"))
            
            X = np.array(features)
            y = np.array(labels)
            
            logger.info(f"Successfully loaded {len(features)} feedback records")
            return X, y
            
        except Exception as e:
            if isinstance(e, DataError):
                raise
            raise DataError("load", str(feedback_path), e)
    
    def train_feedback_model(self, X: np.ndarray, y: np.ndarray) -> Pipeline:
        """Train a logistic regression model on feedback data.
        
        Args:
            X: Feature matrix (embeddings)
            y: Target labels (0 for dislike, 1 for like)
            
        Returns:
            Trained sklearn Pipeline
            
        Raises:
            ModelError: If training fails
        """
        try:
            logger.info(f"Training model on {len(X)} samples")
            start_time = time.time()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=MODEL_CONFIG["test_size"], 
                random_state=MODEL_CONFIG["random_state"]
            )
            
            logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
            
            # pipeline creation
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(max_iter=MODEL_CONFIG["max_iter"]))
            ])
            
            # use grid search from pipeline
            logger.info("Performing hyperparameter tuning with GridSearchCV")
            grid_search = GridSearchCV(
                pipeline, 
                HYPERPARAM_GRID, 
                cv=MODEL_CONFIG["cv_folds"], 
                scoring='accuracy', 
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # eval
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            training_time = time.time() - start_time
            
            # metrics
            self.metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_samples=len(X_train),
                validation_samples=len(X_test),
                best_params=grid_search.best_params_,
                training_time=training_time
            )
            
            logger.info(f"Model training completed successfully")
            logger.info(f"Validation accuracy: {accuracy:.3f}")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Training time: {training_time:.2f} seconds")
            
            return best_model
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise ModelError("training", e)
    
    def save_model(self, model: Pipeline, model_path: Optional[Path] = None) -> None:
        """Save the trained model to disk.
        
        Args:
            model: Trained sklearn Pipeline
            model_path: Path to save the model (uses default if None)
            
        Raises:
            ModelError: If saving fails
        """
        try:
            save_path = Path(model_path) if model_path else self.model_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving model to {save_path}")
            joblib.dump(model, save_path)
            
            # Save metrics alongside model
            metrics_path = save_path.with_suffix('.metrics.json')
            if self.metrics:
                with open(metrics_path, 'w') as f:
                    json.dump(self.metrics.__dict__, f, indent=2, default=str)
                logger.info(f"Saved model metrics to {metrics_path}")
            
            logger.info(f"Model successfully saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise ModelError("saving", e)
    
    def load_model(self, model_path: Optional[Path] = None) -> Pipeline:
        """Load a trained model from disk.
        
        Args:
            model_path: Path to load the model from (uses default if None)
            
        Returns:
            Loaded sklearn Pipeline
            
        Raises:
            ModelError: If loading fails
        """
        try:
            load_path = Path(model_path) if model_path else self.model_path
            
            if not load_path.exists():
                raise ModelError("loading", FileNotFoundError(f"Model file not found: {load_path}"))
            
            logger.info(f"Loading model from {load_path}")
            model = joblib.load(load_path)
            
            metrics_path = load_path.with_suffix('.metrics.json')
            if metrics_path.exists():
                try:
                    with open(metrics_path, 'r') as f:
                        metrics_data = json.load(f)
                    self.metrics = ModelMetrics(**metrics_data)
                    logger.info(f"Loaded model metrics from {metrics_path}")
                except Exception as e:
                    logger.warning(f"Failed to load model metrics: {e}")
            
            logger.info(f"Model successfully loaded from {load_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelError("loading", e)
    
    def retrain_model(self, feedback_path: Path, model_path: Optional[Path] = None) -> Pipeline:
        """Retrain the model from all collected feedback.
        
        Args:
            feedback_path: Path to the feedback JSONL file
            model_path: Path to save the retrained model
            
        Returns:
            Retrained sklearn Pipeline
            
        Raises:
            ModelError: If retraining fails
        """
        try:
            logger.info("Starting model retraining")
            
            # feedback
            X, y = self.load_feedback(feedback_path)
            
            # train
            model = self.train_feedback_model(X, y)
            
            # saveskiiii
            self.save_model(model, model_path)
            
            logger.info("Model retraining completed successfully")
            return model
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            raise ModelError("retraining", e)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "metrics": self.metrics.__dict__ if self.metrics else None
        }
        
        if self.model_path.exists():
            info["model_size_mb"] = self.model_path.stat().st_size / (1024 * 1024)
        
        return info


# Backward compatibility functions
def load_feedback(feedback_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load feedback data (backward compatibility)."""
    trainer = ModelTrainer()
    return trainer.load_feedback(Path(feedback_path))


def train_feedback_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Train feedback model (backward compatibility)."""
    trainer = ModelTrainer()
    return trainer.train_feedback_model(X, y)


def retrain_model(feedback_path: str = "user_feedback.jsonl", model_path: str = "user_feedback_model.joblib") -> None:
    """Retrain model (backward compatibility)."""
    trainer = ModelTrainer()
    trainer.retrain_model(Path(feedback_path), Path(model_path))
