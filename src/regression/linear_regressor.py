import numpy as np
from .regressor import Regressor

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
        x_arr = np.array(X)
        y_arr = np.array(y)

        # Agregar columna de 1s para el término de sesgo (bias)
        x = np.column_stack((np.ones(x_arr.shape[0]), x_arr))

        # Calcular coeficientes beta con la pseudoinversa en lugar de la inversa
        self.coefficients = np.linalg.pinv(x.T @ x) @ x.T @ y_arr

    def predict(self, x: list[float]) -> float:
        """
        Predice el valor para una sola muestra.

        :param x: Lista con características de una muestra.
        :return: Valor predicho.
        """
        if self.coefficients is None:
            raise ValueError("El modelo no ha sido entrenado aún.")

        new_x = np.array([1] + x)  # Agregar bias term
        return float(new_x @ self.coefficients)
