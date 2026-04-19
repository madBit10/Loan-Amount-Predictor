# Loan Amount Prediction

A regression ML project that predicts how much loan a person will borrow, using Linear Regression, Ridge, and Lasso with Sklearn Pipelines.

## Project Structure

```
loan_amount_prediction/
├── data/
│   ├── credit_risk_dataset.csv   # Training dataset
│   └── sample_input.csv          # Sample input for prediction
├── src/
│   ├── main.py                   # CLI entry point (--train / --predict)
│   ├── config/                   # Configuration management
│   ├── data/                     # Data loading and preprocessing
│   ├── models/                   # Training, pipeline, and evaluation
│   ├── pipeline/                 # Training and prediction pipelines
│   └── utils/                    # Logger and artifact saving
├── artifacts/
│   ├── models/                   # Saved model and feature columns
│   └── metrics/                  # Model performance metrics
├── tests/                        # Unit tests
├── requirements.txt
└── run_training.sh
```

## Dataset

- Source: Kaggle Credit Risk Dataset
- 32,581 rows, 12 columns
- Target: `loan_amnt` (continuous — loan amount in dollars)

## Models

Three regression models compared, all using Sklearn Pipeline with StandardScaler:

- Linear Regression
- Ridge Regression (alpha=10)
- Lasso Regression (alpha=10)

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

### Predict loan amount
```bash
python3 -m src.main --predict data/sample_input.csv
```

### Run tests then train
```bash
bash run_training.sh
```

## Results

| Metric | Linear Regression | Ridge | Lasso |
|---|---|---|---|
| MAE | $1,507 | $1,507 | $1,503 |
| RMSE | $2,161 | $2,161 | $2,161 |
| R² | 0.8026 | 0.8026 | 0.8026 |

## Key Design Decisions

- **Sklearn Pipeline** — bundles StandardScaler + model into one object, prevents data leakage
- **Median imputation** — robust to outliers in financial data
- **IQR outlier removal** — removed 4,825 rows (14.9%) with extreme values in age, income, employment length, loan amount, and interest rate. R² improved from 0.598 to 0.8026
- **Feature engineering** — created `income_per_year_emp` (person_income / (person_emp_length + 1)) to capture income stability. `loan_to_income_ratio` was initially created but removed because it uses the target variable (`loan_amnt`), which is unavailable at prediction time
- **Feature columns saved** — ensures prediction uses the same feature set as training
