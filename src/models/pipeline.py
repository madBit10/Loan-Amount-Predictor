from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_and_train_pipeline(X_train, y_train):

    logger.info("Starting the build and train pipeline...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]).fit(X_train,y_train)

    logger.info("The build and train pipeline executed...")

    return pipeline