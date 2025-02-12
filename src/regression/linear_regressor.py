import numpy as np
from regression.regressor import Regressor

class LinearRegressor(Regressor):
    """
    Implementación de regresión lineal usando la fórmula de mínimos cuadrados con numpy.
    """

    def __init__(self):
        self.coefficients = None  # Inicialmente sin entrenar

    def train(self, X: list[list[float]], y: list[float]) -> None:
        """
        Entrena el modelo ajustando los coeficientes usando la pseudoinversa.

        :param X: Lista de listas con características.
        :param y: Lista de valores objetivo.
        """
        X = np.array(X)
        y = np.array(y)

        # Agregar columna de 1s para el término de sesgo (bias)
        X = np.column_stack((np.ones(X.shape[0]), X))

        # Calcular coeficientes beta con la pseudoinversa en lugar de la inversa
        self.coefficients = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, x: list[float]) -> float:
        """
        Predice el valor para una sola muestra.

        :param x: Lista con características de una muestra.
        :return: Valor predicho.
        """
        if self.coefficients is None:
            raise ValueError("El modelo no ha sido entrenado aún.")

        x = np.array([1] + x)  # Agregar bias term
        return float(x @ self.coefficients)
