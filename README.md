# Drug Solubility Prediction using Machine Learning

## 📌 Project Overview

This project predicts the aqueous solubility (LogS) of chemical compounds using Machine Learning regression models. The dataset contains molecular descriptors such as molecular weight, lipophilicity, aromatic proportion, and the number of rotatable bonds.

The objective is to compare different regression algorithms and determine which model provides the best prediction accuracy for drug solubility.

### Features
- Data Loading and Exploration
- Data Preprocessing
- Train-Test Splitting
- Linear Regression Model
- Random Forest Regression Model
- Model Evaluation using MSE and R² Score
- Model Comparison
- Prediction Visualization

---

## 📂 Dataset

The dataset contains 1,144 molecules with the following features:

| Feature | Description |
|----------|-------------|
| MolLogP | Molecular Lipophilicity |
| MolWt | Molecular Weight |
| NumRotatableBonds | Number of Rotatable Bonds |
| AromaticProportion | Aromatic Atom Proportion |
| logS | Solubility (Target Variable) |

Dataset Source:

https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv

---

## 🎯 Problem Statement

Predict the solubility (logS) of chemical compounds using molecular descriptors and evaluate the performance of different Machine Learning regression models.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn

---

## 📊 Machine Learning Workflow

### 1. Load Dataset

```python
import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv"
)
```

### 2. Data Preparation

```python
X = df.drop('logS', axis=1)
y = df['logS']
```

### 3. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=100
)
```

### 4. Linear Regression Model

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)
```

### 5. Random Forest Regression Model

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    max_depth=2,
    random_state=100
)

rf.fit(X_train, y_train)

y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)
```

### 6. Model Evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# Random Forest
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)
```

---

## 🏆 Results

| Model | Training MSE | Training R² | Test MSE | Test R² |
|---------|------------|------------|----------|----------|
| Linear Regression | 1.0075 | 0.7645 | 1.0207 | 0.7892 |
| Random Forest | 1.0282 | 0.7597 | 1.4077 | 0.7092 |

### Best Model

**Linear Regression**

- Test R² Score = 0.7892
- Test MSE = 1.0207

Linear Regression achieved better performance than Random Forest on the testing dataset.

---

## 📈 Data Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

plt.scatter(
    x=y_test,
    y=y_lr_test_pred,
    c="#7CAE00",
    alpha=0.3
)

z = np.polyfit(y_test, y_lr_test_pred, 1)
p = np.poly1d(z)

plt.plot(y_test, p(y_test), "#F8766D")

plt.xlabel("Experimental LogS")
plt.ylabel("Predicted LogS")

plt.show()
```

---

## 📁 Project Structure

```
Drug-Solubility-Prediction/
│
├── data/
│   └── delaney_solubility_with_descriptors.csv
│
├── notebooks/
│   └── solubility_prediction.ipynb
│
├── src/
│   └── train_model.py
│
├── README.md
│
├── requirements.txt
│
└── results/
    └── model_comparison.png
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YushBytes/drug-solubility-prediction.git
cd drug-solubility-prediction
```

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Run the project:

```bash
python train_model.py
```

---

## 📌 Future Improvements

- Hyperparameter Tuning using GridSearchCV
- Cross Validation
- Feature Engineering
- XGBoost Regressor
- Gradient Boosting Regressor
- Model Deployment using Streamlit
- Drug Solubility Prediction Web Application

---

## 👨‍💻 Author

Ayush Bidwai

B.Tech Computer Science Engineering

Machine Learning Project – Drug Solubility Prediction

---

## 📜 License

This project is open-source and intended for educational and research purposes.
