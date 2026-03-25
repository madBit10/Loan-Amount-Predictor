# Basic entry point of a project (CLI starts here)

# The main.py job is to do 3 things
# 1. Listen - What did the user type? (--train? --predict?)
# 2. Decide - okay, they want to train, let me call the traning pipeline
# 3. Hand off - it calls the right pipline and let it do the actual work

# Front door of the application 

import argparse
from src.pipeline.training_pipeline import run_training_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """
    CLI entry point 
    Listens for user commands and calls the right pipeline


    """

    parser = argparse.ArgumentParser(description="loan amount predictor")

    parser.add_argument(
        "--train", 
        action="store_true",
        help="Run the training pipeline"
    )

    args = parser.parse_args()

    if args.train:
        logger.info("Training mode selected")
        run_training_pipeline()
    else:
        logger.info("No mode provided. Use --train to train the model")


if __name__ == "__main__":
    main()