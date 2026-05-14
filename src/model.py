import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from typing import Tuple, Dict, Any
from loguru import logger

from .config import settings


class StudentPredictor:
    """Logistic Regression model for predicting student pass/fail outcomes."""
    
    def __init__(self):
        self.model = LogisticRegression(
            max_iter=settings.model_max_iter,
            solver=settings.model_solver,
            C=settings.model_c,
            random_state=settings.random_state
        )
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model.
        
        Args:
            X: Training features (hours studied)
            y: Training labels (pass/fail)
        """
        logger.info("Training logistic regression model...")
        self.model.fit(X, y)
        self.is_trained = True
        logger.success("Model training completed")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict pass/fail for given hours.
        
        Args:
            X: Hours studied
            
        Returns:
            Predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of passing.
        
        Args:
            X: Hours studied
            
        Returns:
            Probability array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        
        report = classification_report(
            y, y_pred, 
            target_names=["Fail", "Pass"],
            output_dict=True
        )
        
        cm = confusion_matrix(y, y_pred)
        
        try:
            auc_score = roc_auc_score(y, y_prob)
            fpr, tpr, thresholds = roc_curve(y, y_prob)
        except:
            auc_score = None
            fpr, tpr, thresholds = None, None, None
        
        results = {
            "classification_report": report,
            "confusion_matrix": cm,
            "auc_score": auc_score,
            "roc_curve": (fpr, tpr, thresholds),
            "predictions": y_pred,
            "probabilities": y_prob
        }
        
        logger.info(f"Model evaluation completed. AUC: {auc_score:.4f}" if auc_score else "Model evaluation completed")
        return results
    
    def get_model_params(self) -> Dict[str, float]:
        """
        Get model coefficients and intercept.
        
        Returns:
            Dictionary with coefficient and intercept
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        return {
            "coefficient": float(self.model.coef_[0][0]),
            "intercept": float(self.model.intercept_[0])
        }
    
    def save(self, filepath: Path) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Path) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
