import logging

class Util:
    @staticmethod
    def parse_line(line: str, separator: str = " "):
        if line == "":
            logging.getLogger("data.util").warning("Línea vacía encontrada. Será ignorada.")
            return None
        try:
            data = [float(v) for v in line.split(separator)]
            assert len(data) == 5
        except ValueError:
            logging.getLogger('data.util').warning("Línea con valores no numéricos")
            return None
        except AssertionError:
            logging.getLogger('data.util').warning("Línea con número incorrecto de valores")
            return None
        else:
            return data
