import numpy as np
import matplotlib.pyplot as plt


class KNNRegressionClassifier:
    """
    A K-Nearest Neighbors (KNN) Regression model.

    This model predicts continuous target values based on the average of the
    k-nearest neighbors in the feature space.

    Attributes:
        X: Training feature matrix.
        y: Training target vector.
        k (int): Number of neighbors to consider for prediction.

    Methods:
        predict(X): Predicts target values for input data.
        score(X, y): Calculates the R² score of the model.
        visualize(resolution): Visualizes the regression curve for single-feature datasets.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, k: int = 3):
        """
        Initializes the KNNRegressionClassifier.

        Args:
            X: Training feature matrix
            y: Target vector of
            k (int): Number of nearest neighbors to consider for predictions. Default is 3.
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.k = k

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for the given input data.

        Args:
            X: Input feature matrix

        Returns:
            np.ndarray: Predicted target values of shape (n_samples,).
        """
        X = np.array(X)
        predictions = []

        for x in X:
            # Compute distances to all training points
            distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            # Find indices of the k-nearest neighbors
            nearest_indices = np.argsort(distances)[:self.k]
            # Retrieve the target values of the k-nearest neighbors
            nearest_targets = self.y[nearest_indices]
            # Compute the mean of the neighbors' target values
            prediction = np.mean(nearest_targets)
            predictions.append(prediction)

        return np.array(predictions)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluates the model's performance using the R² score.

        Args:
            X: Test feature matrix.
            y: True target values.

        Returns:
            float: R² score.
        """
        predictions = self.predict(X)
        y = np.array(y)
        total_variance = np.sum((y - np.mean(y)) ** 2)
        residual_variance = np.sum((y - predictions) ** 2)
        return 1 - (residual_variance / total_variance)

    def visualize(self, resolution: int = 100) -> None:
        """
        Visualizes the regression curve for single-feature datasets.

        Args:
            resolution (int): Number of points to use for the regression curve. Default is 100.

        Raises:
            ValueError: If the dataset has more than one feature.
        """
        if self.X.ndim > 1 and self.X.shape[1] > 1:
            raise ValueError("Visualization is only supported for single-feature datasets.")

        # Generate evenly spaced feature values for plotting
        X_plot = np.linspace(self.X.min(), self.X.max(), resolution).reshape(-1, 1)
        # Predict the target values for the generated feature values
        y_plot = self.predict(X_plot)

        # Scatter plot of training data
        plt.scatter(self.X, self.y, color='blue', label='Training Data')
        # Regression curve
        plt.plot(X_plot, y_plot, color='red', label='KNN Regression Curve')

        # Add labels, title, and legend
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.title("KNN Regression Visualization")
        plt.legend()
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Example dataset (single feature)
    X_train = [[1], [2], [3], [4], [5]]
    y_train = [1.2, 1.9, 3.1, 4.0, 5.1]

    # Initialize and train the model
    knn = KNNRegressionClassifier(X_train, y_train, k=2)
    knn.fit()

    # Predict on new data
    X_test = [[1.5], [2.5], [3.5]]
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)

    # Evaluate the model
    print("R² Score:", knn.score(X_train, y_train))

    # Visualize the regression
    knn.visualize()
