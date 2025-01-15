import numpy as np
import math
import matplotlib.pyplot as plt


class PolynomialRegressionClassifier:
    """
    A class for polynomial regression with gradient descent optimization.

    This model supports single-feature datasets and includes methods for fitting,
    predicting, visualizing, and scoring the regression model.
    """

    def __init__(self, X, y, degree=2, maxIter=1000, learningRate=0.001):
        """
        Initialize the PolynomialRegressionClassifier.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input features.
        y : array-like, shape (n_samples,)
            The target values.
        degree : int, optional (default=2)
            The degree of the polynomial features.
        maxIter : int, optional (default=1000)
            The maximum number of iterations for gradient descent.
        learningRate : float, optional (default=0.001)
            The learning rate for gradient descent.
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.degree = degree
        self.maxIter = maxIter
        self.learningRate = learningRate
        self.coef = None

    def fit(self):
        """
        Fit the polynomial regression model to the input data using gradient descent.

        This method calculates the optimal coefficients by minimizing the mean squared error.
        """
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(self.X, self.degree)

        # Initialize coefficients
        self.coef = np.zeros(X_poly.shape[1])

        oldCost = math.inf
        for _ in range(self.maxIter):
            predictions = X_poly @ self.coef
            error = predictions - self.y
            gradient = (1 / X_poly.shape[0]) * (X_poly.T @ error)
            self.coef -= self.learningRate * gradient
            cost = (1 / (2 * X_poly.shape[0])) * np.sum(error ** 2)
            if abs(cost - oldCost) <= 0.0001:
                break
            oldCost = cost

    def predict(self, X):
        """
        Predict the target values for the given input features.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input features.

        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            The predicted target values.
        """
        # Generate polynomial features for X
        X_poly = self._generate_polynomial_features(X, self.degree)
        return X_poly @ self.coef

    def _generate_polynomial_features(self, X, degree):
        """
        Generate polynomial features for the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input features.
        degree : int
            The degree of the polynomial.

        Returns:
        --------
        X_poly : array-like, shape (n_samples, degree + 1)
            The polynomial features.
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # shape (n_samples, 1)

        n_samples = X.shape[0]
        X_poly = np.ones((n_samples, degree + 1))
        for d in range(1, degree + 1):
            X_poly[:, d] = X[:, 0] ** d
        return X_poly

    def visualize(self):
        """
        Visualize the polynomial regression fit for a single-feature dataset.

        Raises:
        -------
        ValueError:
            If the input data has more than one feature.
        """
        if self.X.shape[1] > 1:
            raise ValueError("Visualization is only supported for single-feature datasets.")
        
        # Scatter plot of the original data
        plt.scatter(self.X, self.y, color='blue', label='Actual Data')
        
        # Sort for a smooth polynomial curve
        X_sorted = np.sort(self.X, axis=0)
        y_pred = self.predict(X_sorted)
        
        plt.plot(X_sorted, y_pred, color='red', label=f'Polynomial Fit (Degree {self.degree})')
        plt.xlabel("X (Feature)")
        plt.ylabel("y (Target)")
        plt.title(f"Polynomial Regression (Degree {self.degree})")
        plt.legend()
        plt.show()

    def score(self, X, y):
        """
        Calculate the R^2 score for the model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input features.
        y : array-like, shape (n_samples,)
            The true target values.

        Returns:
        --------
        r2_score : float
            The R^2 score, indicating the goodness of fit.
        """
        y = np.array(y)
        predictions = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - predictions) ** 2)
        return 1 - (ss_residual / ss_total)


if __name__ == "__main__":
    # Example usage
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2.3, 3.1, 4.8, 8.5, 10.1])
    
    # Create and train the model
    model = PolynomialRegressionClassifier(X, y, degree=3, maxIter=3000, learningRate=0.0001)
    model.fit()
    
    # Predict for a new data point
    print(model.predict([[4]]))
    
    # Evaluate the model
    r2 = model.score(X, y)
    print(f"R^2 Score: {r2:.4f}")