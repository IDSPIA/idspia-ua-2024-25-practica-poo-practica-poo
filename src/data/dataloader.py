from typing import List
from .dataset import DataSet
import logging
from .util import Util
import os

class DataLoader:
    def __init__(self, separator: str = " "):
        self.__separator = separator

    def load_data(self, folder_path: str) -> List[DataSet]:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The provided path '{folder_path}' does not exist")

        datasets: List[DataSet] = []
        for subfolder in os.listdir(folder_path):
            if not os.path.isdir(os.path.join(folder_path, subfolder)):
                continue
            files = os.listdir(os.path.join(folder_path, subfolder))
            if files:
                dataset = DataSet(subfolder)
                for file in files:
                    with open(os.path.join(folder_path, subfolder, file), "r") as f:
                        for line in f:
                            row = Util.parse_line(line[:-1], self.__separator)
                            if row is not None:
                                dataset.add_entry(row)
                            else:
                                logging.getLogger('data.dataloader').warning("Línea inválida")
                datasets.append(dataset)
            else:
                logging.getLogger('data.dataloader').warning("No se encontraron datos válidos")

        return datasets
