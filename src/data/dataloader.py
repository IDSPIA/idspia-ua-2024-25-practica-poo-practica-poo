from typing import List
from .dataset import DataSet
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
            dataset = DataSet(subfolder)
            for file in files:
                with open(os.path.join(folder_path, subfolder, file), "r") as f:
                    for line in f:
                        row = Util.parse_line(line[:-1], self.__separator)
                        if row is not None and len(row) == 5:
                            dataset.add_entry(row)
            datasets.append(dataset)

        return datasets
