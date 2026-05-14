import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

from .config import settings


class Visualizer:
    """Visualization utilities for the student predictor model."""
    
    @staticmethod
    def plot_sigmoid_curve(
        X: np.ndarray,
        y: np.ndarray,
        hours_range: np.ndarray,
        probabilities: np.ndarray,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot the sigmoid curve with training data.
        
        Args:
            X: Training features
            y: Training labels
            hours_range: Range of hours for plotting
            probabilities: Predicted probabilities for hours_range
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(hours_range, probabilities, color='blue', linewidth=2, label='Probability of Passing')
        plt.scatter(X, y, color='red', s=100, alpha=0.6, label='Training Data', edgecolors='black')
        plt.axhline(settings.decision_threshold, color='gray', linestyle='--', linewidth=1.5, label=f'Decision Boundary ({settings.decision_threshold})')
        plt.xlabel("Hours Studied", fontsize=12)
        plt.ylabel("Probability of Passing", fontsize=12)
        plt.title("Logistic Regression - Pass/Fail Prediction", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sigmoid curve saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Optional path to save plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=["Fail", "Pass"],
               yticklabels=["Fail", "Pass"],
               title="Confusion Matrix",
               ylabel='True Label',
               xlabel='Predicted Label')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_roc_curve(
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            auc_score: AUC score
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_prediction_distribution(
        hours: np.ndarray,
        probabilities: np.ndarray,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot distribution of prediction probabilities.
        
        Args:
            hours: Hours studied
            probabilities: Predicted probabilities
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 6))
        colors = ['red' if p < settings.decision_threshold else 'green' for p in probabilities]
        plt.bar(hours.flatten(), probabilities, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(settings.decision_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold ({settings.decision_threshold})')
        plt.xlabel("Hours Studied", fontsize=12)
        plt.ylabel("Probability of Passing", fontsize=12)
        plt.title("Prediction Probability Distribution", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distribution saved to {save_path}")
        
        plt.show()
