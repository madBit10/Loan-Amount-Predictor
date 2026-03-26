from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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

def build_and_train_ridge(X_train, y_train):
    logger.info("Starting the build and train pipeline for ridge regression...")

    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10))
    ]).fit(X_train, y_train)

    logger.info("The build and train pipeline for ridge executed...")

    return ridge_pipeline

def build_and_train_lasso(X_train, y_train):

    logger.info("Starting the build and train pipeline for lasso regression...")

    lasso_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=10))
    ]).fit(X_train, y_train)

    logger.info("The build and train pipeline for lasso executed....")

    return lasso_pipeline