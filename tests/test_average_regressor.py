import unittest
from regression.average_regressor import AverageRegressor

class TestAverageRegressor(unittest.TestCase):
    """
    Pruebas unitarias para la clase AverageRegressor.
    """

    def test_predict_returns_average(self):
        """Verifica que `predict` devuelve siempre el coste promedio."""
        regressor = AverageRegressor()
        X = [[1, 2, 3, 4], [2, 3, 4, 5], [0, 1, 2, 3]]  # Ignorados
        y = [100, 150, 200]  # Costes

        regressor.train(X, y)
        prediction = regressor.predict([5, 6, 7, 8])  # Los valores de entrada se ignoran
        expected_average = sum(y) / len(y)

        self.assertEqual(prediction, expected_average)

    def test_predict_fails_without_training(self):
        """Verifica que `predict` lanza un error si el modelo no ha sido entrenado."""
        regressor = AverageRegressor()

        with self.assertRaises(ValueError):
            regressor.predict([1, 2, 3, 4])

    def test_average_regressor_public_members(self):
        """Verifica que `AverageRegressor` solo tiene los atributos y métodos públicos especificados en el enunciado."""
        public_methods = [method for method in dir(AverageRegressor) if not method.startswith("_")]
        self.assertEqual(public_methods, ['predict', 'train'])

if __name__ == "__main__":
    unittest.main()
