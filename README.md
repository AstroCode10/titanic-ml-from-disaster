# Titanic Survival Prediction 🛳️

This repository contains a complete machine learning pipeline for predicting passenger survival on the Titanic. The project is based on the Kaggle competition **Titanic: Machine Learning from Disaster**.


## Project Overview

The goal of this project is to build a supervised machine learning model that predicts whether a passenger survived the Titanic disaster using demographic and travel-related features.

The workflow includes:
- Cleaning and imputing missing data
- Creating meaningful engineered features
- Building a preprocessing and modeling pipeline
- Hyperparameter tuning using cross-validation
- Generating a submission file compatible with Kaggle

---
## Project Structure

**File descriptions:**

- `train.csv` – Training dataset containing features and the target variable `Survived`
- `test.csv` – Test dataset without survival labels
- `submission.csv` – Model predictions formatted for Kaggle submission
- `titanic_model.py` – Python script containing the full pipeline
- `README.md` – Project documentation

---

## Dataset

The dataset is provided by Kaggle:

**Titanic: Machine Learning from Disaster**  
<https://www.kaggle.com/competitions/titanic>

### Target Variable

- `Survived`  
  - `0` = Did not survive  
  - `1` = Survived

---

## Data Preprocessing

### Missing Values

Missing values are handled using statistics computed from the training data only.

#### Numerical Features
- `Age`
- `Fare`
- `SibSp`
- `Parch`
- `Pclass`

Missing values are filled using the **median**.

#### Categorical Features
- `Sex`
- `Embarked`

Missing values are filled using the **mode**.

This approach avoids data leakage when processing the test dataset.

---

## Feature Engineering

### Title Extraction

Passenger titles are extracted from the `Name` column using regular expressions.

Rare titles are grouped into a single category:
- `Lady`
- `Countess`
- `Capt`
- `Col`
- `Don`
- `Dr`
- `Major`
- `Rev`
- `Sir`
- `Jonkheer`
- `Dona`

These are replaced with the value `Rare`.

---

### Family Features

Two family-related features are created:

- `FamilySize = SibSp + Parch + 1`
- `IsAlone`
  - `1` if `FamilySize == 1`
  - `0` otherwise

---

### Age Groups

Passengers are categorized into age groups using fixed bins:

| Age Range | Group |
|----------|------|
| 0–12 | Child |
| 13–18 | Teen |
| 19–35 | YoungAdult |
| 36–60 | Adult |
| 61+ | Senior |

---

### Fare Bins

Passenger fares are discretized into quartiles using `pd.qcut` and labeled from 1 to 4.

---

### Interaction Features

Two interaction features are added:

- `Age_Fare = Age × Fare`
- `Pclass_FamilySize = Pclass × FamilySize`

---

### Dropped Columns

The following columns are removed from both datasets:

- `PassengerId`
- `Name`
- `Ticket`
- `Cabin`

---

## Preprocessing Pipeline

A unified preprocessing pipeline is built using `ColumnTransformer`.

### Numerical Features

- Standardized using `StandardScaler`

### Categorical Features

- One-hot encoded using `OneHotEncoder`
- `handle_unknown="ignore"`
- `drop="first"`

The preprocessing pipeline is applied consistently to both training and test data.

---

## Model

### Algorithm

- **Random Forest Classifier**

### Hyperparameter Tuning

Hyperparameters are optimized using `GridSearchCV` with 5-fold cross-validation.

Tuned parameters include:
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

### Evaluation Metric

- Accuracy

---

## Training and Prediction

- The model is trained on the full training dataset.
- The best-performing model is selected from grid search.
- Predictions are generated for the test dataset.

---

## Submission

Predictions are saved in a Kaggle-compatible CSV file with the following format:


PassengerId,Survived
892,0
893,1
...

The output file is:
submission.csv

## How to Install

1. Install dependencies: pip install pandas scikit-learn
2. Place train.csv and test.csv in the project directory.
3. Run the script: python titanic_model.py
