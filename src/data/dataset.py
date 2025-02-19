from typing import List, Tuple

class DataSet:
    #  Does the DataSet receive an argument for all the data of a provider or does it have to iterate itself to obtain it?
    def __init__(self, provider: str):
        self.__provider = provider
        self.__data: List[List[float]] = []

    @property
    def provider(self):
        return self.__provider

    def __iter__(self):
        self.__index = 0
        return self
    
    def __next__(self):
        if self.__index < len(self.__data):
            data = self.__data[self.__index]
            self.__index += 1
            return data
        else:
            raise StopIteration

    def __len__(self):
        #  Does this return the total number of entries across all files of a provider of the number of files?
        return len(self.__data)

    def __str__(self):
        return f"DataSet(provider='{self.provider}', entries={len(self)})"

    def add_entry(self, entry: List[float]):
        if len(entry) != 5:
            raise ValueError("Expected a list of 5 values")

        self.__data.append(entry)

    def get_features_and_labels(self) -> Tuple[List[List[float]], List[float]]:
        if len(self.__data) == 0:
            raise ValueError("The DataSet is empty, cannot obtain features and labels.")

        x = [data[:-1] for data in self.__data]
        y = [data[-1] for data in self.__data]
        return x, y
