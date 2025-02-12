import unittest

class TestDataVisibility(unittest.TestCase):
    """
    Pruebas unitarias para verificar la visibilidad de las clases dentro del módulo `data`.
    """

    def test_dataloader_is_accessible(self):
        """Verifica que `DataLoader` es accesible desde `data`."""
        from data import DataLoader
        self.assertTrue(hasattr(DataLoader, "__init__"))

    def test_dataset_is_accessible(self):
        """Verifica que `DataSet` es accesible desde `data`."""
        from data import DataSet
        self.assertTrue(hasattr(DataSet, "__init__"))

    def test_util_is_not_accessible(self):
        """Comprueba que `Util` NO se puede importar desde `data`."""
        try:
            from data import Util  # Intentamos importar Util
            self.fail("Se pudo importar `Util`, pero no debería estar accesible.")  # Falla si no lanza ImportError
        except ImportError:
            pass  # El error esperado ocurrió, el test pasa

if __name__ == "__main__":
    unittest.main()
