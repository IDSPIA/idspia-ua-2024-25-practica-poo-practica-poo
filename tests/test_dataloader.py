import unittest
import os
import tempfile
import logging
from data.dataloader import DataLoader

class TestDataLoader(unittest.TestCase):
    """
    Pruebas unitarias para la clase DataLoader.
    """

    def setUp(self):
        """Configura un directorio temporal con archivos de prueba antes de cada test."""
        self.test_dir = tempfile.TemporaryDirectory()

        # Crear subcarpetas para proveedores
        aws_dir = os.path.join(self.test_dir.name, "AWS")
        google_dir = os.path.join(self.test_dir.name, "GoogleCloud")
        os.makedirs(aws_dir, exist_ok=True)
        os.makedirs(google_dir, exist_ok=True)

        # Crear archivos de log con datos válidos
        self._create_log_file(aws_dir, "aws_log1.txt", ["2 3 1 10 195", "1 2 3 5 120"])
        self._create_log_file(google_dir, "gcp_log1.txt", ["0 2 3 8 250", "4 1 2 6 180"])

        # Archivo con datos inválidos
        self._create_log_file(aws_dir, "aws_invalid.txt", ["2 3 x 10 195", "1 2 3"])

        # Archivo vacío
        self._create_log_file(google_dir, "empty_log.txt", [])

        # Archivo que no debería afectar la carga
        unrelated_file = os.path.join(self.test_dir.name, "unrelated.txt")
        with open(unrelated_file, "w") as f:
            f.write("Este archivo no debería ser procesado.\n")

        # Crear un directorio temporal con ficheros que usan un separador distinto
        self.test_dir_separator = tempfile.TemporaryDirectory()
        aws_dir = os.path.join(self.test_dir_separator.name, "AWS")
        google_dir = os.path.join(self.test_dir_separator.name, "GoogleCloud")
        os.makedirs(aws_dir, exist_ok=True)
        os.makedirs(google_dir, exist_ok=True)
        self._create_log_file(aws_dir, "aws_log1.txt", ["2,3,1,10,195", "1,2,3,5,120"])
        self._create_log_file(google_dir, "gcp_log1.txt", ["0,2,3,8,250", "4,1,2,6,180"])

    def tearDown(self):
        """Elimina el directorio temporal después de cada test."""
        self.test_dir.cleanup()

    def _create_log_file(self, directory, filename, lines):
        """Crea un archivo de log con las líneas dadas."""
        with open(os.path.join(directory, filename), "w") as f:
            f.write("\n".join(lines) + "\n")

    def test_load_data_creates_datasets(self):
        """Verifica que DataLoader carga correctamente los archivos y devuelve una lista de DataSet."""
        dataloader = DataLoader()
        datasets = dataloader.load_data(self.test_dir.name)

        # Debe haber dos datasets (AWS y GoogleCloud)
        self.assertEqual(len(datasets), 2)
        self.assertEqual(set(ds.provider for ds in datasets), {"AWS", "GoogleCloud"})

        # Verificar número de registros cargados
        for dataset in datasets:
            if dataset.provider == "AWS":
                self.assertEqual(len(dataset), 2)  # Solo las líneas válidas
            elif dataset.provider == "GoogleCloud":
                self.assertEqual(len(dataset), 2)  # Solo las líneas válidas
    
    def test_load_data_with_separator(self):
        """Verifica que DataLoader maneja correctamente un separador distinto en los archivos de log."""
        dataloader = DataLoader(separator=",")
        datasets = dataloader.load_data(self.test_dir_separator.name)

        # Debe haber dos datasets (AWS y GoogleCloud)
        self.assertEqual(len(datasets), 2)
        self.assertEqual(set(ds.provider for ds in datasets), {"AWS", "GoogleCloud"})

        # Verificar número de registros cargados
        for dataset in datasets:
            if dataset.provider == "AWS":
                self.assertEqual(len(dataset), 2)  # Solo las líneas válidas
            elif dataset.provider == "GoogleCloud":
                self.assertEqual(len(dataset), 2)  # Solo las líneas válidas

    def test_load_data_ignores_invalid_lines(self):
        """Verifica que las líneas inválidas sean ignoradas sin detener la carga."""
        dataloader = DataLoader()
        datasets = dataloader.load_data(self.test_dir.name)

        aws_dataset = next(ds for ds in datasets if ds.provider == "AWS")
        self.assertEqual(len(aws_dataset), 2)  # No cuenta las líneas inválidas

    def test_logging_for_invalid_lines(self):
        """Verifica que las líneas inválidas generen advertencias en el log."""
        dataloader = DataLoader()

        with self.assertLogs("data.dataloader", level="WARNING") as cm:
            _ = dataloader.load_data(self.test_dir.name)

        warnings = [record for record in cm.output if "Línea inválida" in record]
        self.assertTrue(len(warnings) > 0)  # Debe haber al menos una advertencia

    def test_non_existent_directory_raises_error(self):
        """Verifica que si la carpeta no existe, se lanza un FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            dataloader = DataLoader()
            dataloader.load_data("ruta_inexistente")

    def test_load_data_skips_empty_files(self):
        """Verifica que archivos vacíos no afecten la carga."""
        dataloader = DataLoader()
        datasets = dataloader.load_data(self.test_dir.name)

        google_dataset = next(ds for ds in datasets if ds.provider == "GoogleCloud")
        self.assertEqual(len(google_dataset), 2)  # No se cuentan líneas de archivos vacíos

    def test_logging_for_empty_providers(self):
        """Verifica que un proveedor sin datos válidos genere una advertencia en el log."""
        empty_provider_dir = os.path.join(self.test_dir.name, "Azure")
        os.makedirs(empty_provider_dir, exist_ok=True)

        dataloader = DataLoader()

        with self.assertLogs("data.dataloader", level="WARNING") as cm:
            datasets = dataloader.load_data(self.test_dir.name)

        warnings = [record for record in cm.output if "No se encontraron datos válidos" in record]
        self.assertTrue(len(warnings) > 0)  # Debe haber al menos una advertencia

    def test_ignores_non_directory_files(self):
        """Verifica que archivos sueltos en la carpeta raíz sean ignorados."""
        dataloader = DataLoader()
        datasets = dataloader.load_data(self.test_dir.name)

        # No debe afectar la carga de datasets válidos
        self.assertEqual(len(datasets), 2)  # Solo AWS y GoogleCloud

    def test_dataloader_public_members(self):
        """Verifica que `DataLoader` solo tiene los atributos y métodos públicos especificados en el enunciado."""
        public_methods = [method for method in dir(DataLoader) if not method.startswith("_")]
        self.assertEqual(public_methods, ['load_data'])    

if __name__ == "__main__":
    unittest.main()
