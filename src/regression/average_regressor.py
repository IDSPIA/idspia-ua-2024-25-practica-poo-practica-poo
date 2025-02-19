import numpy as np
from typing import cast
from .regressor import Regressor

class AverageRegressor(Regressor):
    def __init__(self):
        self.__average_cost = None
        self.__trained = False

    def train(self, X: list[list[float]], y: list[float]) -> None:
        self.__average_cost = float(np.mean(y))
        self.__trained = True

    def predict(self, x: list[float]) -> float:
        if self.__trained:
            return cast(float, self.__average_cost)
        else:
            raise ValueError("The model hasn't been trained, so its 'predict' method cannot be called")
