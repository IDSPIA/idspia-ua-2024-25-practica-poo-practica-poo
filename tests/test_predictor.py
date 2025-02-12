import unittest
import tempfile
import os
import shutil
from regression.predictor import Predictor
from regression.average_regressor import AverageRegressor
from regression.linear_regressor import LinearRegressor
from data.dataloader import DataLoader
from data.dataset import DataSet

class TestPredictor(unittest.TestCase):
    """
    Pruebas unitarias para la clase Predictor.
    """

    def setUp(self):
        """Crea un entorno de prueba con datos ficticios en una carpeta temporal."""
        self.test_dir = tempfile.TemporaryDirectory()

        # Crear subcarpetas para los proveedores
        self.aws_dir = os.path.join(self.test_dir.name, "AWS")
        self.gcp_dir = os.path.join(self.test_dir.name, "GoogleCloud")
        os.makedirs(self.aws_dir, exist_ok=True)
        os.makedirs(self.gcp_dir, exist_ok=True)

        # Crear archivos de log con datos ficticios
        self._create_log_file(self.aws_dir, "aws_log.txt", ["2 3 1 10 195", "1 2 3 5 120"])
        self._create_log_file(self.gcp_dir, "gcp_log.txt", ["0 2 3 8 250", "4 1 2 6 180"])

    def tearDown(self):
        """Elimina la carpeta temporal después de cada test."""
        self.test_dir.cleanup()

    def _create_log_file(self, directory, filename, lines):
        """Crea un archivo de log con las líneas dadas."""
        with open(os.path.join(directory, filename), "w") as f:
            f.write("\n".join(lines) + "\n")

    def test_invalid_regressor_type(self):
        """Verifica que se lanza un error si el tipo de regresor es inválido."""
        with self.assertRaises(ValueError):
            Predictor("INVALID")

    def test_train_with_average_regressor(self):
        """Verifica que el entrenamiento con AverageRegressor funciona correctamente."""
        predictor = Predictor(Predictor.AVERAGE)
        predictor.train(self.test_dir.name)

        # Asegurar que se han entrenado regresores del tipo correcto
        for provider in predictor.providers:
            self.assertIsInstance(predictor._regressors[provider], AverageRegressor)

    def test_train_with_linear_regressor(self):
        """Verifica que el entrenamiento con LinearRegressor funciona correctamente."""
        predictor = Predictor(Predictor.LINEAR)
        predictor.train(self.test_dir.name)

        # Asegurar que se han entrenado regresores del tipo correcto
        for provider in predictor.providers:
            self.assertIsInstance(predictor._regressors[provider], LinearRegressor)

    def test_predict_without_training(self):
        """Verifica que se lanza un error si se intenta predecir sin entrenar."""
        predictor = Predictor(Predictor.LINEAR)

        with self.assertRaises(ValueError):
            predictor.predict("AWS", [1, 2, 3, 4])

    def test_predict_with_trained_regressor(self):
        """Verifica que `predict` devuelve un valor tras entrenar el modelo."""
        predictor = Predictor(Predictor.AVERAGE)
        predictor.train(self.test_dir.name)

        prediction = predictor.predict("AWS", [2, 3, 1, 10])
        self.assertIsInstance(prediction, float)  # Debe ser un número

    def test_predict_with_invalid_provider(self):
        """Verifica que se lanza un error si se intenta predecir con un proveedor no registrado."""
        predictor = Predictor(Predictor.LINEAR)
        predictor.train(self.test_dir.name)

        with self.assertRaises(ValueError):
            predictor.predict("NonExistentProvider", [2, 3, 1, 10])
    
    def test_predict_with_invalid_input(self):
        """Verifica que se lanza un error si se intenta predecir con una entrada inválida."""
        predictor = Predictor(Predictor.LINEAR)
        predictor.train(self.test_dir.name)

        with self.assertRaises(ValueError):
            predictor.predict("AWS", [2, 3, 1])

    def test_predictor_public_members(self):
        """Verifica que `Predictor` solo tiene los atributos y métodos públicos especificados en el enunciado."""
        public_methods = [method for method in dir(Predictor) if not method.startswith("_")]
        self.assertEqual(public_methods, ['AVERAGE', 'LINEAR', 'predict', 'providers', 'train'])

if __name__ == "__main__":
    unittest.main()
