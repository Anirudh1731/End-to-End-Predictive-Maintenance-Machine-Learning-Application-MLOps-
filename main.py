from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.inititate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,preprocessor_path=data_transformation.initiate_data_transformation(train_data,test_data)
    print(train_arr)
    model_trainer=ModelTrainer()
    best_score=model_trainer.initiate_model_trainer(train_arr,test_arr,preprocessor_path)
    print(train_arr)
    print(best_score)
    
