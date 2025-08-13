# iFood Data Science Case Study

A machine learning solution to predict customer offer completion using transaction, profile, and offer data from iFood's marketing campaigns.

## Project Overview

This project implements a binary classification model to predict whether customers will complete marketing offers. The solution uses historical transaction data, customer profiles, and offer characteristics to build predictive features and train a Random Forest classifier.

### Key Results
- **ROC-AUC**: 0.89
- **Accuracy**: 80%
- **Model**: Random Forest with engineered features
- **Data Split**: Time-based (80th percentile at day 21)

## Project Structure

```
ifood-case/
├── data/
│   ├── raw/                          # Original data files
│   │   ├── offers.json              # Marketing offer details
│   │   ├── profile.json             # Customer demographics
│   │   └── transactions.json        # Transaction and event data
│   ├── processed/                    # Cleaned datasets with features
│   └── predictions/                  # Model evaluation results by date
├── notebooks/
│   ├── 0_eda.ipynb                  # Exploratory Data Analysis
│   ├── 1_data_processing.ipynb      # Data cleaning & feature engineering
│   └── 2_modelling.ipynb            # Model training & evaluation
├── presentation/                     # Presentation materials
├── requirements.txt                 # Python dependencies
└── README.md
```

## Dataset Description

### Profile Data (17,000 customers)
- **age**: Customer age at account creation
- **registered_on**: Account creation date
- **gender**: Customer gender (M/F/None)
- **credit_card_limit**: Registered card limit

### Offers Data (10 unique offers)
- **offer_type**: bogo, discount, informational
- **duration**: Offer validity period (days)
- **min_value**: Minimum spend to activate offer
- **discount_value**: Discount amount
- **channels**: Distribution channels (email, mobile, social, web)

### Transactions Data (306,534 events)
- **event**: transaction, offer received, offer viewed, offer completed
- **account_id**: Customer identifier
- **time_since_test_start**: Days since experiment start
- **amount**: Transaction amount (for purchases)
- **offer_id**: Related offer identifier

## Feature Engineering

The model uses several categories of engineered features:

### Customer Features
- Age and tenure segments (new, continuous, tenured, high tenured, extreme tenured)
- Credit card limit
- Customer lifetime metrics

### Historical Features
- Historical spending patterns (`hist_spent`, `hist_count`)
- Historical offer completion rate
- 30-day rolling spending and transaction metrics

### Offer Features
- Discount value and minimum spend requirements
- Offer duration and type
- Distribution channels (email, mobile, social, web)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ifood-case
```

### 2. Create Conda Environment
```bash
conda create -n ifood-case python=3.9
conda activate ifood-case
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter
```bash
jupyter notebook
```

## Usage

Run the notebooks in sequence:

1. **Exploratory Data Analysis** (`0_eda.ipynb`)
   - Data quality assessment
   - Distribution analysis
   - Customer segmentation insights

2. **Data Processing** (`1_data_processing.ipynb`)
   - Data cleaning and preprocessing
   - Feature engineering pipeline
   - Dataset creation for modeling

3. **Modeling** (`2_modelling.ipynb`)
   - Model training and evaluation
   - Feature importance analysis
   - Performance metrics and reporting

## Model Details

### Algorithm
- **Random Forest Classifier**
- Handles mixed data types (numerical, categorical, boolean)
- Built-in feature importance for interpretability

### Preprocessing Pipeline
- **Categorical features**: Imputation (missing → 'missing') + OneHot encoding
- **Numerical features**: Used as-is (no scaling needed for tree-based models)
- **Feature selection**: Domain-informed feature engineering

### Evaluation Strategy
- **Time-based split**: Training on first 80% of timeline, testing on remaining 20%
- **Metrics**: Precision, Recall, F1-Score, ROC-AUC
- **Feature analysis**: Importance ranking and contribution analysis

### Key Insights
- Most important features: Historical spending patterns and offer characteristics
- Customer tenure segments provide strong predictive power
- Rolling 30-day metrics capture recent behavioral changes
- Channel preferences (email, mobile) influence completion rates
  