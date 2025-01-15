import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class EpsilonSVRClassifier:
    """
    A simple epsilon-SVR model with a kernel function, using explicit alpha and alpha* (alpha_star) in the dual problem.
    """

    def __init__(self, X, y, epsilon=0.1, C=1.0, kernel='rbf', gamma=0.1):
        """
        Initialize the SVR model.

        Parameters:
        - X: Input features, shape (n_samples, n_features).
        - y: Target values, shape (n_samples,).
        - epsilon: Epsilon margin of tolerance for the loss function.
        - C: Regularization parameter.
        - kernel: Type of kernel function ('linear', 'rbf', 'poly').
        - gamma: Parameter for the RBF kernel (ignored for linear kernel).
        """
        self.X = np.array(X, dtype=float)
        self.y = np.array(y, dtype=float)
        self.epsilon = epsilon
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma

        # Lagrange multipliers for the dual problem
        self.alpha = None
        self.alpha_star = None

        # Bias term
        self.b = 0.0

    def _kernel(self, x1, x2):
        """
        Compute the kernel function based on the specified kernel type.

        Parameters:
        - x1, x2: Input vectors.

        Returns:
        - The computed kernel value.
        """
        if self.kernel_type == 'linear':
            return np.dot(x1, x2)
        elif self.kernel_type == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel_type == 'poly':
            return (np.dot(x1, x2) + 1) ** 2
        else:
            raise ValueError("Unsupported kernel type.")

    def fit(self):
        """
        Train the SVR model using the dual formulation of epsilon-SVR.
        """
        n_samples = self.X.shape[0]

        # Precompute the kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel(self.X[i], self.X[j])

        # Dual optimization objective
        def objective(z):
            alpha = z[:n_samples]
            alpha_star = z[n_samples:]
            alpha_diff = alpha - alpha_star

            half_term = 0.5 * alpha_diff.dot(K).dot(alpha_diff)
            eps_term = self.epsilon * np.sum(alpha + alpha_star)
            y_term = -np.sum(self.y * alpha_diff)

            return half_term + eps_term + y_term

        # Gradient of the objective
        def gradient(z):
            alpha = z[:n_samples]
            alpha_star = z[n_samples:]
            alpha_diff = alpha - alpha_star

            grad_alpha_diff = K.dot(alpha_diff)

            grad_alpha = grad_alpha_diff + self.epsilon - self.y
            grad_alpha_star = -grad_alpha_diff + self.epsilon + self.y

            return np.concatenate([grad_alpha, grad_alpha_star])

        # Equality constraint: sum(alpha - alpha_star) = 0
        def eq_constraint(z):
            alpha = z[:n_samples]
            alpha_star = z[n_samples:]
            return np.sum(alpha) - np.sum(alpha_star)

        # Jacobian of the equality constraint
        def eq_constraint_jac(z):
            grad = np.zeros_like(z)
            grad[:n_samples] = 1.0
            grad[n_samples:] = -1.0
            return grad

        # Bounds: 0 <= alpha, alpha_star <= C
        bounds = [(0, self.C)] * (2 * n_samples)

        # Initial guess
        z0 = np.zeros(2 * n_samples)

        # Solve the dual optimization problem
        cons = [{
            'type': 'eq',
            'fun': eq_constraint,
            'jac': eq_constraint_jac
        }]

        result = minimize(
            fun=objective,
            x0=z0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=cons,
            options={'disp': True}
        )

        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)

        # Extract alpha and alpha_star from the result
        z_opt = result.x
        self.alpha = z_opt[:n_samples]
        self.alpha_star = z_opt[n_samples:]

        # Compute bias term
        alpha_diff = self.alpha - self.alpha_star
        b_list = []
        for i in range(n_samples):
            if (1e-6 < self.alpha[i] < self.C - 1e-6) or (1e-6 < self.alpha_star[i] < self.C - 1e-6):
                w_phi = np.sum(alpha_diff * K[i, :])
                b_list.append(self.y[i] - w_phi)
        self.b = np.mean(b_list)

    def predict(self, X):
        """
        Predict target values for input features.

        Parameters:
        - X: Input features, shape (n_samples, n_features).

        Returns:
        - Predictions, shape (n_samples,).
        """
        X = np.array(X, dtype=float)
        alpha_diff = self.alpha - self.alpha_star

        predictions = []
        for x in X:
            prediction = np.sum([
                alpha_diff[i] * self._kernel(self.X[i], x)
                for i in range(self.X.shape[0])
            ]) + self.b
            predictions.append(prediction)
        return np.array(predictions)

    def visualize(self):
        """
        Visualize the SVR fit for single-feature datasets.
        """
        if self.X.shape[1] > 1:
            raise ValueError("Visualization is only supported for single-feature datasets.")
        plt.scatter(self.X, self.y, color="blue", label="Data")
        X_sorted = np.sort(self.X, axis=0)
        y_pred = self.predict(X_sorted)
        plt.plot(X_sorted, y_pred, color="red", label="SVR Fit")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def score(self, X, y):
        """
        Calculate the R^2 score.

        Parameters:
        - X: Input features, shape (n_samples, n_features).
        - y: True target values, shape (n_samples,).

        Returns:
        - R^2 score.
        """
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


# Example usage
if __name__ == "__main__":
    # Input data
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]).astype(float)
    y = np.array([2.3, 3.1, 4.8, 8.5, 10.1, 11.8, 13.2, 14.4, 16, 17.6]).astype(float)

    # Initialize and train the SVR model
    svr = EpsilonSVRClassifier(X, y, epsilon=0.1, C=10, kernel="rbf", gamma=0.5)
    svr.fit()

    # Predict and evaluate
    X_test = np.array([[6]]).astype(float)
    predictions = svr.predict(X_test)
    print("Predictions on X_test:", predictions)

    # Visualize the results
    svr.visualize()

    # R^2 score
    r2_score = svr.score(X, y)
    print("R^2 Score:", r2_score)