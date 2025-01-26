import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from src.utils import evaluate_models

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import mlflow
@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def track_mlflow(self,best_model,score):
        with mlflow.start_run():
            mlflow.log_metric('accuracy',score)
            mlflow.sklearn.log_model(best_model,"model")

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Splitting training and testing input data")

            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            models={
                "RandomForestClassifier":RandomForestClassifier(),
                "Decision Tree":DecisionTreeClassifier(),
                "Gradient Boosting":GradientBoostingClassifier(),
                "Logistic Regression":LogisticRegression(),
                "KNN":KNeighborsClassifier(),
                "XGB Classifier":XGBClassifier(),
                "CatBoostClassifier":CatBoostClassifier(),
                "AdaBoostClassifier":AdaBoostClassifier()
            }

        #     params = {
        #         "RandomForestClassifier": {
        #         'n_estimators': [50, 100, 200],
        #         'max_depth': [None, 10, 20, 30],
        #         'min_samples_split': [2, 5, 10],
        #         'min_samples_leaf': [1, 2, 4],
        #         'bootstrap': [True, False]
        #     },
        #     "Decision Tree": {
        #         'criterion': ['gini', 'entropy'],
        #         'max_depth': [None, 10, 20, 30],
        #         'min_samples_split': [2, 5, 10],
        #         'min_samples_leaf': [1, 2, 4]
        #     },
        #     "Gradient Boosting": {
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'n_estimators': [50, 100, 200],
        #         'max_depth': [3, 5, 10],
        #         'subsample': [0.8, 1.0]
        #     },
        #     "Logistic Regression": {
        #         'C': [0.01, 0.1, 1, 10],
        #         'solver': ['liblinear', 'lbfgs', 'saga'],
        #         'max_iter': [100, 200, 500]
        #     },
        #     "KNN": {
        #         'n_neighbors': [3, 5, 10, 20],
        #         'weights': ['uniform', 'distance'],
        #         'metric': ['euclidean', 'manhattan', 'minkowski']
        #     },
        #     "XGB Classifier": {
        #         'n_estimators': [50, 100, 200],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'max_depth': [3, 5, 10],
        #         'colsample_bytree': [0.6, 0.8, 1.0],
        #         'subsample': [0.6, 0.8, 1.0]
        #     },
        #     "CatBoostClassifier": {
        #         'iterations': [100, 200, 500],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'depth': [3, 5, 10]
        #     },
        #     "AdaBoostClassifier": {
        #         'n_estimators': [50, 100, 200],
        #         'learning_rate': [0.01, 0.1, 0.5],
        #         'algorithm': ['SAMME', 'SAMME.R']
        #     }
        # }


            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            #Track the experiments
            self.track_mlflow(best_model,best_model_score)

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found is {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            accuracy_score_best=accuracy_score(y_test,predicted)

            return accuracy_score_best
        except Exception as e:
            raise CustomException(e,sys)