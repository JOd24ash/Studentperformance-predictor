import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
)

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import(
    mean_absolute_error,mean_squared_error,r2_score
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class Model_trainer_Config:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class Model_trainer:
    def __init__(self):
        self.model_trainer_config=Model_trainer_Config()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spliting the traing and test input data")
            X_train,y_train,X_test,y_test=(train_array[:,:-1],
                                           train_array[:,-1],
                                           test_array[:,:-1],
                                           test_array[:,-1]
                                           )
            models={
                "Linear Regression":LinearRegression(),
                "Lasso":Lasso(max_iter=10000),
                "Ridge":Ridge(max_iter=10000),
                "K-NeighborsRegressor":KNeighborsRegressor(),
                "Decsion tree" :DecisionTreeRegressor(),
                "Random Forest Regressor":RandomForestRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
            } 
            model_report:dict
            model_all_report:dict
            param_grids = {
    "Linear Regression": {
        'fit_intercept': [True, False]
    },
    
    "Lasso": {
        'alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0],
        'selection': ['cyclic', 'random']
    },
    
    "Ridge": {
        'alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
    },
    
    "K-NeighborsRegressor": {
        'n_neighbors': [3, 5, 7, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    
    "Decsion tree": {
        'criterion': ['squared_error', 'absolute_error', 'poisson'], # Removed friedman_mse
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    "Random Forest Regressor": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None]
    },
    
    "XGBRegressor": {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    
    "CatBoost Regressor": {
        'iterations': [100, 200],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5]
    },
    
    "AdaBoost Regressor": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'loss': ['linear', 'square', 'exponential']
    }
}

            
            model_report,model_all_report=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param_grid=param_grids)
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("no best model found")
            logging.info(f"best model found  om both traing and test dataset")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_=r2_score(y_test,predicted)
            return (r2_,model_all_report)
        except Exception as e:
            raise CustomException(e,sys)
