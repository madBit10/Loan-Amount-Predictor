# It orchestrates ("manages") the entire traning process

"""
calls:

step1 - load_data.py "Gets the raw data"
step2 - preprocess.py "Clean and prepare it"
step3 - train.py "Train the model"
step4 - evaluate.py "Check how good it is"
step5 - save_artifacts.py "Save the model and results"

"""

import yaml
from src.data.load_data import load_data
from src.data.preprocess import preprocess
# from src.models.train import train
from src.models.evaluate import evaluate
from src.utils.save_artifacts import save_artifacts
from src.utils.logger import get_logger
from sklearn.model_selection import train_test_split
from src.models.pipeline import build_and_train_pipeline

logger = get_logger(__name__)

def run_training_pipeline():
    """
    Orchestrates the full training process.
    Calls each step in order and passes data between them.


    """

    logger.info("Starting training pipeline...")

    #step 1 - load data

    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded")

    # step 2 - load data
    df = load_data(config["data"]["raw_data_path"])

    # step 3 - Preprocess

    X,y = preprocess(df, config["target_column"])

    # train test split 

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    logger.info(f"Train size: {X_train.shape}, Test_size: {X_test.shape}")

    # step 4 - train

    pipeline = build_and_train_pipeline(X_train, y_train)

    # step 5 - evaluate

    metrics = evaluate(pipeline, X_test, y_test)

    # step 6 - save artifacts

    save_artifacts(pipeline, X.columns.to_list(), metrics, {
        "model_path": config["artifacts"]["model_path"],
        "features_path": config["artifacts"]["features_path"],
        "metrics_path": config["artifacts"]["metrics_path"]
    })

    logger.info("Training pipeline complete")