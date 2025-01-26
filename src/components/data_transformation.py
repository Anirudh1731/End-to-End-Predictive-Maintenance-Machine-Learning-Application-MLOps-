import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transoformer_object(self):

        ''' This function is to get the preprocessor'''
        try:
            numerical_columns=['UDI','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]','Target']
            categorical_features=['Type']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler(with_mean=False))
                ]


            )

            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical Columns standard scaling completed")

            logging.info("Categorical Columns encoding completed")

            #Combining both pipelines

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("categorical_pipeline",categorical_pipeline,categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)   
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            train_df.drop(columns=['Product ID'],inplace=True)
            test_df.drop(columns=['Product ID'],inplace=True)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transoformer_object()

            target_col_name='Failure Type'

            input_feature_train_df=train_df.drop(columns=[target_col_name],axis=1)
            # Encode the target variable
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(train_df[target_col_name])

            input_feature_test_df=test_df.drop(columns=[target_col_name],axis=1)
            target_feature_test_df = label_encoder.transform(test_df[target_col_name])

            logging.info("Applying preprocssing object to dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            #print(input_feature_train_arr[:,-1])
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
            logging.info('Saved preprocessing object')
            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)