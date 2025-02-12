import unittest
from data.dataset import DataSet

class TestDataSet(unittest.TestCase):
    """
    Pruebas unitarias para la clase DataSet.
    """

    def test_dataset_initialization(self):
        """Verifica que un DataSet se inicializa correctamente con un proveedor."""
        dataset = DataSet("AWS")
        self.assertEqual(dataset.provider, "AWS")  # El proveedor debe ser el correcto
        self.assertEqual(len(dataset), 0)  # Inicialmente debe estar vacío

    def test_provider_is_read_only(self):
        """Verifica que el atributo provider es de solo lectura."""
        dataset = DataSet("Google")
        with self.assertRaises(AttributeError):  # Intentar modificarlo debe lanzar error
            dataset.provider = "Azure"

    def test_add_valid_entry(self):
        """Verifica que se puede agregar una entrada válida."""
        dataset = DataSet("AWS")
        dataset.add_entry([2, 3, 1, 10, 195])
        self.assertEqual(len(dataset), 1)  # Ahora debe tener una entrada

    def test_add_invalid_entry(self):
        """Verifica que agregar una entrada con menos de 5 valores lanza un ValueError."""
        dataset = DataSet("Azure")
        with self.assertRaises(ValueError):
            dataset.add_entry([2, 3, 1, 10])  # Falta el coste

    def test_get_features_and_labels(self):
        """Verifica que get_features_and_labels separa correctamente X e y."""
        dataset = DataSet("Google")
        dataset.add_entry([2, 3, 1, 10, 195])
        dataset.add_entry([1, 2, 3, 8, 150])
        
        X, y = dataset.get_features_and_labels()
        self.assertEqual(X, [[2, 3, 1, 10], [1, 2, 3, 8]])
        self.assertEqual(y, [195, 150])

    def test_get_features_and_labels_empty_dataset(self):
        """Verifica que get_features_and_labels lanza un error si el dataset está vacío."""
        dataset = DataSet("Azure")
        with self.assertRaises(ValueError):
            dataset.get_features_and_labels()

    def test_dataset_iteration(self):
        """Verifica que DataSet es iterable y devuelve las filas correctamente."""
        dataset = DataSet("AWS")
        dataset.add_entry([2, 3, 1, 10, 195])
        dataset.add_entry([1, 2, 3, 8, 150])

        entries = list(dataset)  # Convertir en lista para comparar
        self.assertEqual(entries, [[2, 3, 1, 10, 195], [1, 2, 3, 8, 150]])
    
    def test_dataset_str_representation(self):
        """Verifica que la representación en cadena de un DataSet es correcta."""
        dataset = DataSet("Google")
        self.assertEqual(str(dataset), "DataSet(provider='Google', entries=0)")
    
    def test_dataset_public_members(self):
        """Verifica que `DataSet` solo tiene los atributos y métodos públicos especificados en el enunciado."""
        public_methods = [method for method in dir(DataSet) if not method.startswith("_")]
        self.assertEqual(public_methods, ['add_entry', 'get_features_and_labels', 'provider'])    

if __name__ == "__main__":
    unittest.main()
