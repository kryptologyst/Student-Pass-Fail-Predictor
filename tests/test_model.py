import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.model import StudentPredictor
from src.data import generate_sample_data


class TestStudentPredictor:
    
    @pytest.fixture
    def sample_data(self):
        """Fixture to provide sample data."""
        return generate_sample_data()
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Fixture to provide a trained model."""
        X, y = sample_data
        model = StudentPredictor()
        model.train(X, y)
        return model
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = StudentPredictor()
        assert model.is_trained is False
        assert model.model is not None
    
    def test_model_training(self, sample_data):
        """Test model training."""
        X, y = sample_data
        model = StudentPredictor()
        model.train(X, y)
        assert model.is_trained is True
    
    def test_prediction_before_training(self):
        """Test that prediction fails before training."""
        model = StudentPredictor()
        X = np.array([[5.0]])
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(X)
    
    def test_prediction(self, trained_model):
        """Test model prediction."""
        X_test = np.array([[5.0], [2.0], [8.0]])
        predictions = trained_model.predict(X_test)
        assert len(predictions) == 3
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba(self, trained_model):
        """Test probability prediction."""
        X_test = np.array([[5.0]])
        proba = trained_model.predict_proba(X_test)
        assert len(proba) == 1
        assert 0 <= proba[0] <= 1
    
    def test_evaluate(self, trained_model, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        results = trained_model.evaluate(X, y)
        
        assert "classification_report" in results
        assert "confusion_matrix" in results
        assert "auc_score" in results
        assert "predictions" in results
        assert "probabilities" in results
    
    def test_get_model_params(self, trained_model):
        """Test getting model parameters."""
        params = trained_model.get_model_params()
        assert "coefficient" in params
        assert "intercept" in params
        assert isinstance(params["coefficient"], float)
        assert isinstance(params["intercept"], float)
    
    def test_save_and_load(self, trained_model):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.joblib"
            
            trained_model.save(filepath)
            assert filepath.exists()
            
            new_model = StudentPredictor()
            new_model.load(filepath)
            assert new_model.is_trained is True
            
            X_test = np.array([[5.0]])
            pred1 = trained_model.predict(X_test)
            pred2 = new_model.predict(X_test)
            assert pred1[0] == pred2[0]
    
    def test_save_untrained_model(self):
        """Test that saving untrained model raises error."""
        model = StudentPredictor()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.joblib"
            with pytest.raises(ValueError, match="Cannot save untrained model"):
                model.save(filepath)
