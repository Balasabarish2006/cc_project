# Credit Card Fraud Detection using Machine Learning

A comprehensive comparative study evaluating supervised machine learning algorithms for detecting fraudulent credit card transactions on an imbalanced financial dataset.

---

## 📌 Project Overview

Credit card fraud detection is a critical application of machine learning in financial technology. Fraudulent transactions constitute a minuscule fraction of overall credit card activity, making this an **extremely imbalanced classification task**. Standard accuracy metrics often fail on such data, as a naive classifier predicting zero fraud would easily achieve over 99% accuracy while providing zero real-world utility.

This project implements an end-to-end machine learning pipeline that:

* Preprocesses raw credit card transaction data.
* Handles missing target values and scales continuous features.
* Trains **8 distinct classification algorithms** (ranging from linear estimators to ensemble methods and non-parametric classifiers).
* Benchmarks models using evaluation metrics designed for severe class imbalance (**Precision**, **Recall**, and **F1-Score**).

---

## 🔄 Working Flowchart

```
                 +--------------------------------+
                 |    Load Dataset (CSV File)     |
                 +--------------------------------+
                                 |
                                 v
                 +--------------------------------+
                 |       Data Preprocessing       |
                 | - Drop missing target values   |
                 | - Separate Features (X) & Y    |
                 +--------------------------------+
                                 |
                                 v
                 +--------------------------------+
                 |      Train-Test Split 80/20    |
                 |   (Stratified Sampling: y)     |
                 +--------------------------------+
                                 |
                 +---------------+---------------+
                 |                               |
                 v                               v
    +-------------------------+     +-------------------------+
    |     Feature Scaling     |     |   Unscaled Data Stream  |
    |    (StandardScaler)     |     |  (Tree-Based Models)    |
    +-------------------------+     +-------------------------+
                 |                               |
                 v                               v
    +-------------------------+     +-------------------------+
    |  Distance / Linear      |     |     Decision Tree       |
    |  Based Algorithms:      |     |     Random Forest       |
    | - Linear Regression     |     +-------------------------+
    | - Logistic Regression   |                  |
    | - K-Nearest Neighbors   |                  |
    | - SVM (RBF Kernel)      |                  |
    | - Gaussian Naive Bayes  |                  |
    | - Lasso & Ridge Reg.    |                  |
    +-------------------------+                  |
                 |                               |
                 +---------------+---------------+
                                 |
                                 v
                 +--------------------------------+
                 |    Model Predictions & Metric  |
                 |          Evaluation            |
                 | (Accuracy, Precision, Recall,  |
                 |           F1-Score)            |
                 +--------------------------------+

```

---

## 📊 Dataset Overview & Exploratory Summary

The dataset contains anonymized credit card transactions, where features `V1` through `V28` are numerical values resulting from a Principal Component Analysis (PCA) transformation. Features `Time` and `Amount` remain untransformed.

* **Source File:** `creditcard.csv`
* **Total Records Analyzed:** 9,965
* **Features:** 30 predictors (`Time`, `V1`–`V28`, `Amount`)
* **Target Variable:** `Class` (`0.0` = Legitimate, `1.0` = Fraudulent)
* **Class Distribution:**
* **Legitimate (`0.0`):** 9,926 transactions (99.61%)
* **Fraudulent (`1.0`):** 38 transactions (0.39%)
* **Missing (`NaN`):** 1 record (removed during data cleaning)



---

## ⚙️ Data Preprocessing & Modeling Pipeline

1. **Data Cleaning:** Identified and removed rows containing missing values in the target feature (`Class`).
2. **Train-Test Split:** Partitioned the dataset into **80% Training** (7,971 samples) and **20% Testing** (1,993 samples).
* **Stratification:** Applied `stratify=y` during the split to ensure identical minority class proportions in both train and test sets.


3. **Feature Scaling:** Fitted `StandardScaler` on `X_train` and transformed both `X_train` and `X_test` to normalize feature distributions (mean = 0, variance = 1) for distance-based models (KNN, SVM, Logistic Regression, Naive Bayes). Tree-based models (Decision Tree, Random Forest) were trained directly on unscaled data.

---

## 🔬 Evaluated Algorithms

| Algorithm Category | Model | Key Configuration / Parameters |
| --- | --- | --- |
| **Linear Models** | Linear Regression | Threshold conversion ($y \ge 0.5 \to 1$) |
|  | Logistic Regression | L2 Penalty (`lbfgs` solver, max_iter=1000) |
|  | Lasso Logistic Regression | L1 Penalty (`liblinear` solver, max_iter=1000) |
|  | Ridge Logistic Regression | L2 Penalty (`lbfgs` solver, max_iter=1000) |
| **Non-Parametric Models** | K-Nearest Neighbors (KNN) | $k=5$, Minkowski distance |
|  | Support Vector Machine (SVM) | Radial Basis Function (RBF) Kernel |
|  | Gaussian Naive Bayes | Continuous feature probability density |
| **Tree & Ensemble Models** | Decision Tree Classifier | Gini impurity, random_state=42 |
|  | Random Forest Classifier | $100$ Estimators, random_state=42 |

---

## 📈 Model Performance & Comparative Analysis

Because the test dataset contains only **8 actual fraud cases out of 1,993 samples**, accuracy alone is insufficient. Evaluation focuses on **Fraud Class (1.0)** metrics:

* **Precision:** Ratio of true fraud cases out of all predicted fraud cases (lowers False Positives).
* **Recall:** Ratio of true fraud cases detected out of actual fraud cases (lowers False Negatives).
* **F1-Score:** Harmonic mean of Precision and Recall.

### Evaluation Metrics Summary

| Model | Test Accuracy | Fraud Precision | Fraud Recall | Fraud F1-Score | Fraud Support |
| --- | --- | --- | --- | --- | --- |
| **Decision Tree** | **99.90%** | **0.88** | **0.88** | **0.88** | 8 |
| **K-Nearest Neighbors (KNN)** | 99.85% | 1.00 | 0.62 | 0.77 | 8 |
| **Support Vector Machine (SVM)** | 99.85% | 1.00 | 0.62 | 0.77 | 8 |
| **Random Forest** | 99.85% | 1.00 | 0.62 | 0.77 | 8 |
| **L1 Logistic Regression (Lasso)** | 99.85% | 1.00 | 0.62 | 0.77 | 8 |
| **L2 Logistic Regression (Ridge)** | 99.80% | 0.83 | 0.62 | 0.71 | 8 |
| **Linear Regression (Threshold = 0.5)** | 99.75% | 1.00 | 0.38 | 0.55 | 8 |
| **Gaussian Naive Bayes** | 98.60% | 0.19 | 0.75 | 0.30 | 8 |

---

## 💡 Key Findings & Insights

1. **Best Overall Model:** **Decision Tree** achieved the highest overall balance on minority class detection with an **F1-Score of 0.88** (identifying 7 out of 8 fraud cases with only 1 false positive).
2. **Zero False-Positive Models:** **KNN**, **SVM**, **Random Forest**, and **L1 Logistic Regression** reached **100% Precision** (zero false alarms), but missed 3 out of 8 fraud cases (Recall = 0.62).
3. **High Sensitivity (High False Alarm Rate):** **Gaussian Naive Bayes** achieved high recall (0.75), catching 6 out of 8 fraud cases, but suffered from extremely low precision (0.19) due to independence assumptions across correlated features.
4. **Regularization Impact:** L1 Regularization (Lasso) slightly outperformed standard L2 Regularization (Ridge) on precision (1.00 vs 0.83) by performing implicit feature selection on the sparse PCA predictors.

---

## 🚀 Potential Future Improvements

* **Resampling Techniques:** Implement **SMOTE** (Synthetic Minority Over-sampling Technique) or Random Under-sampling to rebalance the training set before fitting.
* **Cost-Sensitive Learning:** Adjust class weights (`class_weight='balanced'`) in SVM and Logistic Regression to heavily penalize misclassified fraud cases.
* **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to optimize tree depth, KNN neighbor count $k$, and SVM gamma parameters.
* **Advanced Ensembles:** Benchmark gradient boosting implementations like **XGBoost**, **LightGBM**, or **CatBoost**.

---

## 🛠️ Project Setup & Usage

### 1. Prerequisites

Ensure Python 3.8+ is installed along with the required scientific computing libraries:

```bash
pip install pandas numpy scikit-learn

```

### 2. Repository Structure

```text
├── creditcard.csv       # Dataset file (place in project root or path)
├── fraud_detection.py  # Main Python pipeline script
└── README.md           # Project documentation

```

### 3. Execution

Clone the repository and run the main script:

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python fraud_detection.py

```
