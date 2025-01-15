import numpy as np
import math
import matplotlib.pyplot as plt


class LinearRegressionClassifier:
    """
    A Linear Regression model implemented from scratch.

    This class supports both single-feature and multi-feature regression,
    gradient descent optimization, and model evaluation using R^2 score.

    Attributes:
        X: Training feature matrix.
        y: Training target values.
        learningRate (float): Learning rate for gradient descent.
        maxIter (int): Maximum number of iterations for gradient descent.
        coef: Coefficients (weights) of the regression model.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, learningRate: float = 0.001, maxIter: int = 3000):
        """
        Initializes the Linear Regression Classifier.

        Args:
            X: Training feature matrix.
            y: Target vector.
            learningRate (float): Learning rate for gradient descent. Default is 0.0001.
            maxIter (int): Maximum number of iterations for gradient descent. Default is 1000.
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.learningRate = learningRate
        self.maxIter = maxIter
        self.coef = None

    def fit(self) -> None:
        """
        Trains the linear regression model using gradient descent.
        """
        # Add intercept column to X
        self.intercepted = np.c_[np.ones(self.X.shape[0]), self.X]

        # Initialize coefficients (weights)
        self.coef = np.zeros(self.intercepted.shape[1])

        # Perform gradient descent
        oldCost = math.inf
        for _ in range(self.maxIter):
            predictions = self.predict(self.intercepted)
            error = predictions - self.y
            gradientDescent = (1 / self.intercepted.shape[0]) * (self.intercepted.T @ error)
            self.coef -= self.learningRate * gradientDescent
            cost = (1 / (2 * self.intercepted.shape[0])) * np.sum(error ** 2)
            if abs(cost - oldCost) <= 0.0001:  # Convergence check
                break
            oldCost = cost

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for the given input data.

        Args:
            X: Input feature matrix.

        Returns:
            np.ndarray: Predicted target values of shape (n_samples,).
        """
        X = np.array(X)
        # Add intercept column if needed
        if len(X.shape) > 1:
            if X.shape[1] + 1 == len(self.coef):
                X = np.c_[np.ones(X.shape[0]), X]
        else:
            X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coef

    def visualize(self) -> None:
        """
        Visualizes the regression line (for single-feature datasets only).

        Raises:
            ValueError: If the dataset has more than one feature.
        """
        if self.X.shape[1] > 1:
            raise ValueError("Visualization is only supported for single-feature datasets.")
        
        # Scatter plot for actual data
        plt.scatter(self.X, self.y, color='blue', label='Actual Data')

        # Plot regression line
        predictions = self.predict(self.intercepted)
        plt.plot(self.X, predictions, color='red', label='Regression Line')

        # Add labels, title, and legend
        plt.xlabel("X (Feature)")
        plt.ylabel("y (Target)")
        plt.title("Linear Regression Visualization")
        plt.legend()
        plt.show()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluates the model's performance using the R^2 score.

        Args:
            X: Test feature matrix of shape.
            y: True target values.

        Returns:
            float: R^2 score.
        """
        X = np.array(X)
        y = np.array(y)
        predictions = self.predict(X)
        total_variance = np.sum((y - np.mean(y)) ** 2)
        residual_variance = np.sum((y - predictions) ** 2)
        return 1 - (residual_variance / total_variance)


# Example Usage
if __name__ == "__main__":
    # Sample single-feature dataset
    X = [[2], [4], [6], [8], [10]]
    y = [1, 2, 3, 4, 5]

    # Initialize and train the model
    model = LinearRegressionClassifier(X, y)
    model.fit()

    # Visualize the regression results
    model.visualize()

    # Evaluate the model
    print("R^2 Score:", model.score(X, y))
