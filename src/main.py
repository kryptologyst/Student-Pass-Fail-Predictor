import typer
from pathlib import Path
from loguru import logger
import sys

from .config import settings
from .data import generate_sample_data, save_data_to_csv, load_data_from_csv, create_prediction_range
from .model import StudentPredictor
from .visualizer import Visualizer

app = typer.Typer(help="Student Pass/Fail Predictor CLI")

logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def train(
    save_model: bool = typer.Option(True, help="Save trained model"),
    visualize: bool = typer.Option(True, help="Generate visualizations")
):
    """Train the student predictor model."""
    logger.info("Starting training pipeline...")
    
    X, y = generate_sample_data()
    
    csv_path = settings.data_dir / "sample_data.csv"
    save_data_to_csv(X, y, csv_path)
    
    predictor = StudentPredictor()
    predictor.train(X, y)
    
    params = predictor.get_model_params()
    logger.info(f"Model Coefficient (Slope): {params['coefficient']:.4f}")
    logger.info(f"Model Intercept: {params['intercept']:.4f}")
    
    results = predictor.evaluate(X, y)
    
    logger.info("\nClassification Report:")
    for label, metrics in results['classification_report'].items():
        if isinstance(metrics, dict):
            logger.info(f"{label}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}")
    
    if visualize:
        vis = Visualizer()
        
        hours_range = create_prediction_range(0, 11, 100)
        probabilities = predictor.predict_proba(hours_range)
        
        vis.plot_sigmoid_curve(
            X, y, hours_range, probabilities,
            save_path=settings.plots_dir / "sigmoid_curve.png"
        )
        
        vis.plot_confusion_matrix(
            results['confusion_matrix'],
            save_path=settings.plots_dir / "confusion_matrix.png"
        )
        
        if results['auc_score']:
            fpr, tpr, _ = results['roc_curve']
            vis.plot_roc_curve(
                fpr, tpr, results['auc_score'],
                save_path=settings.plots_dir / "roc_curve.png"
            )
    
    if save_model:
        model_path = settings.models_dir / "student_predictor.joblib"
        predictor.save(model_path)
    
    logger.success("Training pipeline completed!")


@app.command()
def predict(
    hours: float = typer.Option(..., help="Number of hours studied"),
    model_path: Path = typer.Option(
        settings.models_dir / "student_predictor.joblib",
        help="Path to saved model"
    )
):
    """Make a prediction for given study hours."""
    logger.info(f"Making prediction for {hours} hours studied...")
    
    predictor = StudentPredictor()
    
    if model_path.exists():
        predictor.load(model_path)
    else:
        logger.warning("No saved model found. Training new model...")
        X, y = generate_sample_data()
        predictor.train(X, y)
    
    import numpy as np
    test_hours = np.array([[hours]])
    
    prediction = predictor.predict(test_hours)[0]
    probability = predictor.predict_proba(test_hours)[0]
    
    outcome = "Pass" if prediction == 1 else "Fail"
    logger.success(f"\nPrediction: {outcome}")
    logger.info(f"Probability of passing: {probability:.2%}")
    
    if probability < 0.3:
        logger.warning("⚠️  High risk of failure. Recommend additional study time.")
    elif probability < 0.7:
        logger.info("⚡ Moderate chance. Consider studying a bit more.")
    else:
        logger.success("✅ High probability of passing!")


@app.command()
def interactive():
    """Interactive prediction mode."""
    logger.info("Starting interactive mode...")
    logger.info("Type 'quit' or 'exit' to stop.\n")
    
    predictor = StudentPredictor()
    model_path = settings.models_dir / "student_predictor.joblib"
    
    if model_path.exists():
        predictor.load(model_path)
    else:
        logger.warning("No saved model found. Training new model...")
        X, y = generate_sample_data()
        predictor.train(X, y)
    
    import numpy as np
    
    while True:
        try:
            user_input = input("\nEnter hours studied (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting interactive mode. Goodbye!")
                break
            
            hours = float(user_input)
            
            if hours < 0:
                logger.error("Hours cannot be negative!")
                continue
            
            test_hours = np.array([[hours]])
            prediction = predictor.predict(test_hours)[0]
            probability = predictor.predict_proba(test_hours)[0]
            
            outcome = "Pass" if prediction == 1 else "Fail"
            print(f"\n{'='*50}")
            print(f"  Prediction: {outcome}")
            print(f"  Probability: {probability:.2%}")
            print(f"{'='*50}")
            
        except ValueError:
            logger.error("Invalid input! Please enter a number.")
        except KeyboardInterrupt:
            logger.info("\nExiting interactive mode. Goodbye!")
            break


@app.command()
def info():
    """Display project information."""
    logger.info(f"Project: {settings.project_name}")
    logger.info(f"Version: {settings.version}")
    logger.info(f"Base Directory: {settings.base_dir}")
    logger.info(f"Data Directory: {settings.data_dir}")
    logger.info(f"Output Directory: {settings.output_dir}")
    logger.info(f"Random State: {settings.random_state}")
    logger.info(f"Decision Threshold: {settings.decision_threshold}")


if __name__ == "__main__":
    app()
