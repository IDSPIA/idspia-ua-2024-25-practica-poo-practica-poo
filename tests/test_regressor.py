import unittest
from abc import ABCMeta
from regression.regressor import Regressor

class TestRegressorInterface(unittest.TestCase):
    """
    Pruebas unitarias para la interfaz Regressor.
    """

    def test_regressor_is_abstract(self):
        """Verifica que `Regressor` no puede instanciarse directamente."""
        with self.assertRaises(TypeError):
            _ = Regressor()  # Intentar instanciar la interfaz debe fallar

    def test_subclass_must_implement_methods(self):
        """Verifica que una subclase debe implementar los métodos abstractos."""

        class IncompleteRegressor(Regressor):
            """Clase de prueba que no implementa los métodos requeridos."""
            pass

        with self.assertRaises(TypeError):
            _ = IncompleteRegressor()  # No debe poder instanciarse

    def test_subclass_implements_methods(self):
        """Verifica que una subclase con métodos implementados puede instanciarse."""

        class MockRegressor(Regressor):
            """Clase de prueba con los métodos correctamente implementados."""

            def train(self, X, y):
                pass

            def predict(self, x):
                return 0.0

        try:
            regressor = MockRegressor()  # Esto no debe lanzar errores
            self.assertTrue(hasattr(regressor, "train"))
            self.assertTrue(hasattr(regressor, "predict"))
        except TypeError:
            self.fail("La subclase que implementa los métodos no debería lanzar TypeError.")

    def test_regressor_public_members(self):
        """Verifica que `Regressor` solo tiene los atributos y métodos públicos especificados en el enunciado."""
        public_methods = [method for method in dir(Regressor) if not method.startswith("_")]
        self.assertEqual(public_methods, ['predict', 'train'])    


if __name__ == "__main__":
    unittest.main()
