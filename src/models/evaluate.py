from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate(model, X_test, y_test) -> dict:
    """
    Evaluates the trained model on test data.
                                                                                        
    Args:
        model: Trained Random Forest model                                             
        X_test: Test features                                                        
        y_test: True test labels

    Returns:
        metrics: Dictionary of evaluation results
    """                                                                                
    logger.info("Evaluating model...")

    y_pred = model.predict(X_test)

    # The regression metrics 

    mae = mean_absolute_error(y_test,y_pred)

    mse = mean_squared_error(y_test,y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "r2_score": r2
    }

    logger.info(f"mean_absolute_error: {mae:.4f}")
    logger.info(f"mean_squared_error: {mse:.4f}")
    logger.info(f"root_mean_squared_error: {rmse:.4f}")
    logger.info(f"r2_score: {r2:.4f}")

    return metrics



