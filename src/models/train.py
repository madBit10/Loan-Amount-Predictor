from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train(X,y,config:dict):
    """
    Trains a Linear model

    Args:
        X: Input features
        y: Target column
        config: 

    Returns:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    """
    logger.info("Starting model training...")

    # train test split 

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    logger.info(f"Train size: {X_train.shape}, Test_size: {X_test.shape}")

    # model training

    model = LinearRegression()

    model.fit(X_train, y_train)

    logger.info("Model traning complete")

    return model,X_test,y_test