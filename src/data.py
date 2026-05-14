import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from loguru import logger

from .config import settings


def generate_sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample study hours and pass/fail labels.
    
    Returns:
        Tuple of (X, y) where X is hours studied and y is pass/fail label
    """
    logger.info("Generating sample data...")
    
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    
    logger.info(f"Generated {len(X)} samples")
    return X, y


def save_data_to_csv(X: np.ndarray, y: np.ndarray, filepath: Path) -> None:
    """
    Save data to CSV file.
    
    Args:
        X: Feature array (hours studied)
        y: Label array (pass/fail)
        filepath: Path to save CSV file
    """
    df = pd.DataFrame({
        'hours_studied': X.flatten(),
        'passed': y
    })
    df.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}")


def load_data_from_csv(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Tuple of (X, y)
    """
    df = pd.read_csv(filepath)
    X = df['hours_studied'].values.reshape(-1, 1)
    y = df['passed'].values
    logger.info(f"Loaded {len(X)} samples from {filepath}")
    return X, y


def create_prediction_range(start: float = 0, end: float = 11, points: int = 100) -> np.ndarray:
    """
    Create a range of values for prediction visualization.
    
    Args:
        start: Start value
        end: End value
        points: Number of points
        
    Returns:
        Array of values
    """
    return np.linspace(start, end, points).reshape(-1, 1)
