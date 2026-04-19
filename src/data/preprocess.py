# Data processing layer

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)

# remove outliers function using IQR

def remove_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:

    # rows before removing the outliers

    rows_before = len(df)

    for column in columns:
        # calculate the first quartile
        Q1 = df[column].quantile(0.25)

        # calculate the 3rd quartile
        Q3 = df[column].quantile(0.75)

        # IQR
        IQR = Q3 - Q1

        # lower fence
        lower_fence = Q1 - 1.5 * IQR
        
        # upper fence 
        upper_fence = Q3 + 1.5 * IQR

        

        # masking to remove the rows that are not within the range of the fence

        mask = (df[column] >= lower_fence) & (df[column] <= upper_fence)
        df = df[mask]

    # rows after removing the outliers
    
    rows_after = len(df)

    rows_removed = rows_before - rows_after


    logger.info(f"Number of rows that were removed: {rows_removed}")

    return df


def create_features(df):
    # feature engineering

    # removed the loan_to_income_ratio as it would use the loan_amnt to get created and the user does not know the loan_amnt at the time of the prediction 
    # df['loan_to_income_ratio'] = df['loan_amnt']/ (df['person_income'] + 1) # the feature gives the ratio of the loan amount/ person income

    # income_per_year_emp = df['person_income']/df['person_emp_length'] # the feature gives the ratio between the person income and the emp length of a person

    # df['income_per_year_emp'] = np.where(
    #     df['person_emp_length'] > 0,
    #     df['person_income'] / df['person_emp_length'],df['person_income']
    # )


    df['income_per_year_emp'] = df['person_income']/ (df['person_emp_length'] + 1)

    # logger.info(f"Inf values: {np.isinf(df.select_dtypes(include=np.number)).sum().sum()}")                                                                                            
    # logger.info(f"NaN values: {df.isnull().sum().sum()}")

    return df


def preprocess(df: pd.DataFrame, target_column: str):
    """
    Cleans and preprocess raw data for training

    Args:
        df: Raw Dataframe
        target_column: Name of the target column
    
    Returns: 
        X: input features
        y: target columns

    """

    logger.info("Starting preprocessing...")


    # step 1 - Drop duplicates

    df = df.drop_duplicates().copy()
    logger.info(f"After dropping the duplicates: {df.shape}")


    # step2 - Fill missing values with median 

    # Fill the loan int rate with median missing values handling
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    # fill person emp_length with median handling missing values 

    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())

    logger.info("Missing values filled")

    # step - outlier detection and removal

    outlier_cols = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate"]
    df = remove_outliers(df, outlier_cols)

    logger.info("Detected the outliers and removed them")

    
    # step - create features

    df = create_features(df)

    logger.info("Applied feature engineering and created the necessary features")


    # step 3 - Convert Y/N to 1/0

    # convert Y/N to 1/0

    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({"Y": 1, "N":0})


    # step 4 - Label encode loan_grade column

    # using label encoder on the loan grade column as it has the data that follows a specific order

    le = LabelEncoder() 
    df["loan_grade"] = le.fit_transform(df["loan_grade"])

    logger.info("loan_grade label encoded")


    # Step 5 - one hot label encoding to the columns 

    # let's change the home_ownership and loan_intent columns through one hot encoding

    df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent"], drop_first=True)

    logger.info("Categorical columns one hot encoded")


    # step 6 - fix boolean columns to int

    df = df.astype({col: int for col in df.select_dtypes(include="bool").columns})


    # step 7 - Split X and y 

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    logger.info(f"Preprocessing complete. X shape: {X.shape}, y shape: {y.shape}")

    return X,y

