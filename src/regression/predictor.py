from typing import Dict, List
from data import DataSet, DataLoader
from .regressor import Regressor
from .average_regressor import AverageRegressor
from .linear_regressor import LinearRegressor


class Predictor:
    AVERAGE = "AVERAGE"
    LINEAR = "LINEAR"
    def __init__(self, regressor_type: str):
        if regressor_type != Predictor.AVERAGE and regressor_type != Predictor.LINEAR: 
            raise ValueError("The regressor_type can only be Average or Linear")
        self.__regressor_type = regressor_type
        self.__regressors: Dict[str, Regressor]= {}
        self.__providers: List[str]= []

    @property
    def providers(self):
        return self.__providers
    
    def train(self, folder_path: str):
        datasets: List[DataSet] = DataLoader().load_data(folder_path)
        for dataset in datasets:
            regressor = LinearRegressor() if self.__regressor_type == Predictor.LINEAR else AverageRegressor()
            regressor.train(*dataset.get_features_and_labels())
            self.__regressors[dataset.provider] = regressor

    def predict(self, provider: str, x: List[float]) -> float:
        if provider not in self.__regressors.keys():
            raise ValueError("The given provider is not one found within the training dataset.")
        if len(x) != 4:
            raise ValueError(f"Expected 4 arguments for model prediction, instead found {len(x)}.")

        prediction = self.__regressors[provider].predict(x)
        return prediction
