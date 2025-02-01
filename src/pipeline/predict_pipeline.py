import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, UDI: int, 
                 Type: str, 
                 air_temperature: float,
                 process_temperature: float,
                 rotational_speed: float,  # Correct parameter name
                 torque: float,
                 tool_wear: int,
                 target: int):
        self.UDI = UDI
        self.Type = Type
        self.air_temperature = air_temperature
        self.process_temperature = process_temperature
        self.rotational_speed = rotational_speed  # Correct attribute name
        self.torque = torque
        self.tool_wear = tool_wear
        self.target = target

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "UDI": [self.UDI],
                "Type": [self.Type],
                "Air temperature [K]": [self.air_temperature],
                "Process temperature [K]": [self.process_temperature],
                "Rotational speed [rpm]": [self.rotational_speed],
                "Torque [Nm]": [self.torque],
                "Tool wear [min]": [self.tool_wear],
                "Target": [self.target]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)