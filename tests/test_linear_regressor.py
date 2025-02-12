import unittest
import numpy as np
from regression.linear_regressor import LinearRegressor

class TestLinearRegressor(unittest.TestCase):
    """
    Pruebas unitarias para la clase LinearRegressor.
    """

    def test_train_sets_correct_coefficients(self):
        """Verifica que el entrenamiento ajusta correctamente los coeficientes."""
        regressor = LinearRegressor()
        X = [[1, 2], [2, 3], [3, 4]]  # Características
        y = [10, 15, 20]  # Valores esperados

        regressor.train(X, y)

        # Calcular los coeficientes esperados usando numpy
        X_np = np.array(X)
        X_np = np.column_stack((np.ones(X_np.shape[0]), X_np))  # Agregar bias
        y_np = np.array(y)
        expected_coefficients = np.linalg.pinv(X_np.T @ X_np) @ X_np.T @ y_np

        np.testing.assert_almost_equal(regressor.coefficients, expected_coefficients, decimal=5)

    def test_predict_returns_correct_value(self):
        """Verifica que `predict` devuelve valores esperados tras el entrenamiento."""
        regressor = LinearRegressor()
        X = [[1, 2], [2, 3], [3, 4]]  # Características
        y = [10, 15, 20]  # Valores esperados

        regressor.train(X, y)
        prediction = regressor.predict([4, 5])  # Predicción para una nueva muestra

        # Calcular la predicción esperada manualmente
        X_np = np.array(X)
        X_np = np.column_stack((np.ones(X_np.shape[0]), X_np))  # Agregar bias
        y_np = np.array(y)
        coefficients = np.linalg.pinv(X_np.T @ X_np) @ X_np.T @ y_np
        expected_prediction = float(np.array([1, 4, 5]) @ coefficients)

        self.assertAlmostEqual(prediction, expected_prediction, places=5)

    def test_predict_fails_without_training(self):
        """Verifica que `predict` lanza un error si el modelo no ha sido entrenado."""
        regressor = LinearRegressor()

        with self.assertRaises(ValueError):
            regressor.predict([1, 2])

    def test_linear_regressor_public_members(self):
        """Verifica que `LinearRegressor` solo tiene los atributos y métodos públicos especificados en el enunciado."""
        public_methods = [method for method in dir(LinearRegressor) if not method.startswith("_")]
        self.assertEqual(public_methods, ['predict', 'train'])

if __name__ == "__main__":
    unittest.main()
