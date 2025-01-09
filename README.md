<p align="center">
  <img src= "https://raw.githubusercontent.com/Jackhammer9/RedML/main/images/logo.png" height=250px width=250px>
</p>

<h1 align="center">RedML</h1>

![GitHub stars](https://img.shields.io/github/stars/Jackhammer9/RedML?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/Jackhammer9/RedML?style=for-the-badge)
![GitHub followers](https://img.shields.io/github/followers/Jackhammer9?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/Jackhammer9/RedML?style=for-the-badge)
![GitHub license](https://img.shields.io/github/license/Jackhammer9/RedML?style=for-the-badge)
![PyPI version](https://img.shields.io/pypi/v/RedML?style=for-the-badge)
![Python version](https://img.shields.io/badge/python-3.6%2B-blue?style=for-the-badge)


A Python package for machine learning models built from scratch, including **Linear Regression** and **K-Nearest Neighbors Regression**. RedML is designed for learning, experimentation, and lightweight use cases.

---

## **Features**

- **Linear Regression**:
  - Train and predict using gradient descent.
  - Supports single and multi-feature datasets.
  - Visualize the regression line for single-feature datasets.
  - Evaluate using the R² score.

- **K-Nearest Neighbors Regression**:
  - Predict continuous target values using \( k \)-nearest neighbors.
  - Supports single and multi-feature datasets.
  - Visualize regression curves for single-feature datasets.
  - Evaluate using the R² score.

---

## **Installation**

Install RedML using pip:

```bash
pip install RedML
```

Or clone the repository:

```bash
git clone https://github.com/Jackhammer9/RedML.git
cd RedML
pip install .
```

---

## **Usage**

### **1. Linear Regression**

#### **Example Usage**
```python
from RedML import LinearRegressionClassifier

# Training data
X = [[1], [2], [3], [4], [5]]  # Single feature dataset
y = [1.5, 2.5, 3.5, 4.5, 5.5]

# Initialize the model
model = LinearRegressionClassifier(X, y, learningRate=0.01, maxIter=1000)

# Train the model
model.fit()

# Predict target values
X_test = [[6], [7]]
predictions = model.predict(X_test)
print("Predictions:", predictions)

# Visualize the regression line (for single-feature datasets)
model.visualize()

# Evaluate the model using R² score
r2_score = model.score(X, y)
print("R² Score:", r2_score)
```

---

### **2. K-Nearest Neighbors Regression**

#### **Example Usage**
```python
from RedML import KNNRegressionClassifier

# Training data
X = [[1], [2], [3], [4], [5]]  # Single feature dataset
y = [1.5, 2.5, 3.5, 4.5, 5.5]

# Initialize the model
knn = KNNRegressionClassifier(X, y, k=2)

# Fit the model
knn.fit()

# Predict target values
X_test = [[2.5], [3.5]]
predictions = knn.predict(X_test)
print("Predictions:", predictions)

# Visualize the regression curve (for single-feature datasets)
knn.visualize()

# Evaluate the model using R² score
r2_score = knn.score(X, y)
print("R² Score:", r2_score)
```
---

## **Contributing**

Contributions are welcome! If you'd like to improve RedML, please fork the repository and submit a pull request.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
