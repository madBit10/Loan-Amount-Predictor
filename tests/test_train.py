import pytest
from sklearn.model_selection import train_test_split
from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.models.pipeline import build_and_train_pipeline
from src.models.evaluate import evaluate

config = {
    "test_size": 0.2,
    "random_state": 42,
}

def test_train():
    df = load_data("data/credit_risk_dataset.csv")
    X, y = preprocess(df, "loan_amnt")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    pipeline = build_and_train_pipeline(X_train, y_train)

    # check pipeline exists
    assert pipeline is not None

    # check test set has rows
    assert X_test.shape[0] > 0

def test_evaluate():
    df = load_data("data/credit_risk_dataset.csv")
    X, y = preprocess(df, "loan_amnt")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    pipeline = build_and_train_pipeline(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)

    # check metrics exist
    assert "mean_absolute_error" in metrics
    assert "r2_score" in metrics

    # check accuracy is reasonable
    assert metrics["r2_score"] > 0
