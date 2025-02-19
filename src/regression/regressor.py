from abc import abstractmethod, ABC

class Regressor(ABC):

    @abstractmethod
    def train(self, X: list[list[float]], y: list[float]) -> None:
        pass

    @abstractmethod
    def predict(self, x: list[float]) -> float:
        pass
