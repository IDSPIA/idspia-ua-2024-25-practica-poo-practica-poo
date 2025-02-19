import unittest
import inspect
from data.util import Util

class TestUtil(unittest.TestCase):
    def test_parse_line_valid(self):
        """Verifica que `parse_line` convierte una cadena en una lista de números float."""
        self.assertEqual(Util.parse_line("2 3 1 10 195"), [2.0, 3.0, 1.0, 10.0, 195.0])
    
    def test_parse_line_separator(self):
        """Verifica que `parse_line` maneja correctamente un separador distinto al espacio."""
        self.assertEqual(Util.parse_line("2,3,1,10,195", ","), [2.0, 3.0, 1.0, 10.0, 195.0])

    def test_parse_line_empty(self):
        """Verifica que `parse_line` maneja correctamente una cadena vacía."""
        with self.assertLogs("data.util", level='WARNING') as cm:
            self.assertIsNone(Util.parse_line(""))
        self.assertIn("Línea vacía encontrada. Será ignorada.", cm.output[0])

    def test_parse_line_invalid_numbers(self):
        """Verifica que `parse_line` maneja correctamente valores no numéricos."""
        with self.assertLogs("data.util", level='WARNING') as cm:
            self.assertIsNone(Util.parse_line("2 3 x 10 195"))
        self.assertIn("Línea con valores no numéricos", cm.output[0])

    def test_parse_line_wrong_length(self):
        """Verifica que `parse_line` maneja correctamente líneas con número incorrecto de valores."""
        with self.assertLogs("data.util", level='WARNING') as cm:
            self.assertIsNone(Util.parse_line("2 3 1 10"))
        self.assertIn("Línea con número incorrecto de valores", cm.output[0]) 
            
        with self.assertLogs("data.util", level='WARNING') as cm:
            self.assertIsNone(Util.parse_line("2 3 1 10 2 40"))
        self.assertIn("Línea con número incorrecto de valores", cm.output[0])   
        
    def test_parse_line_is_static(self):
        """Verifica que `parse_line` es un método estático."""
        method = inspect.getattr_static(Util, "parse_line")
        self.assertIsInstance(method, classmethod)

    def test_util_public_members(self):
        """Verifica que `Util` solo tiene los atributos y métodos públicos especificados en el enunciado."""
        public_methods = [method for method in dir(Util) if not method.startswith("_")]
        self.assertEqual(public_methods, ["parse_line"])

if __name__ == "__main__":
    unittest.main()
