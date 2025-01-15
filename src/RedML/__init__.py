from .models.regression.KNNRegression import KNNRegressionClassifier
from .models.regression.LinearRegression import LinearRegressionClassifier
from .models.regression.PolynomialRegression import PolynomialRegressionClassifier
from .models.regression.SupportVectorRegression import EpsilonSVRClassifier

__all__ = ["KNNRegressionClassifier", "LinearRegressionClassifier"
           , "PolynomialRegressionClassifier", "EpsilonSVRClassifier"]
