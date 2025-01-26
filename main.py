from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation,DataTransformationConfig


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.inititate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,tes_arr,preprocessor_path=data_transformation.initiate_data_transformation(train_data,test_data)
    print(train_arr)
    
