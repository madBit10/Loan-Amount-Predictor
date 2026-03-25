# Loan Amount Prediction

A regression ML project that predicts how much loan a person will borrow, using Linear Regression with an Sklearn Pipeline.

## Project Structure

```
loan_amount_prediction/
├── data/                  # Dataset and exploration notebook
├── src/
│   ├── main.py            # CLI entry point
│   ├── config/            # Configuration management
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Training, pipeline, and evaluation
│   ├── pipeline/          # Training and prediction pipelines
│   └── utils/             # Logger and artifact saving
├── artifacts/
│   └── metrics/           # Model performance metrics
├── tests/                 # Unit tests
├── requirements.txt
└── run_training.sh
```

## Dataset

- Source: Kaggle Credit Risk Dataset
- 32,581 rows, 12 columns
- Target: `loan_amnt` (continuous — loan amount in dollars)

## Model

- Algorithm: Linear Regression (Sklearn Pipeline with StandardScaler)
- MAE: $2,846
- RMSE: $4,041
- R²: 0.598

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/madBit10/loan-amount-prediction
cd loan_amount_prediction
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add dataset
Place `credit_risk_dataset.csv` in the `data/` folder.

## Usage

### Train the model
```bash
python3 -m src.main --train
```

### Run tests then train
```bash
bash run_training.sh
```

## Results

| Metric | Value |
|---|---|
| MAE | $2,846 |
| RMSE | $4,041 |
| R² | 0.598 |

## Key Design Decisions

- **Sklearn Pipeline** — bundles StandardScaler + LinearRegression into one object, prevents data leakage
- **Median imputation** — robust to outliers in financial data
- **loan_status kept as feature** — default history is a strong signal for loan amount
- **Feature columns saved** — ensures prediction uses the same feature set as training
