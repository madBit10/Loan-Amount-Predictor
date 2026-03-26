import pytest
import pandas as pd

from src.data.load_data import load_data
from src.data.preprocess import preprocess

def test_load_data():
    df = load_data("data/credit_risk_dataset.csv")

    # check it returns a DataFrame
    assert isinstance(df, pd.DataFrame)

    # check if it has rows and columns
    assert df.shape[0] > 0
    assert df.shape[1] > 1

    # check key columns exist
    assert "loan_amnt" in df.columns

def test_preprocess():
    df = load_data("data/credit_risk_dataset.csv")
    X,y = preprocess(df, "loan_amnt")

    # check shapes
    assert X.shape[0] == y.shape[0]

    # check no missing values
    assert X.isnull().sum().sum() == 0

    # now we are checking the continuous values here so check if the asset y.min() is greater than 0
    assert y.min() > 0