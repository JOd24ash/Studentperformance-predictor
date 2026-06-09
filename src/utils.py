import os
import numpy as np
import pandas as pd
import sys
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb')as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)      

def evaluate_model(true,predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true,predicted)
    rmse=np.sqrt(mean_squared_error(true,predicted))
    r2_sqaure=r2_score(true,predicted)
    return rmse,r2_sqaure,mae     

def evaluate_models(X_train,y_train,X_test,y_test,models,param_grid):
    try:
        report={}
        model_train_report=[]
        model_test_report=[]
        all_report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            param=param_grid[list(models.keys())[i]]
            gs=GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            



            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            model_train_rmse,model_train_r2,model_train_mae=evaluate_model(y_train,y_train_pred)
            model_test_rmse,model_test_r2,model_test_mae=evaluate_model(y_test,y_test_pred)
            
            model_train_report.append( (model_train_rmse,model_train_r2,model_train_mae))
            model_test_report.append((model_test_rmse,model_test_r2,model_test_mae))

            report[list(models.keys())[i]]=model_test_r2
            all_report[list(models.keys())[i]]=[model_train_report,model_test_report]

        return (report,all_report)
    except Exception as e:
        raise CustomException(e,sys)    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)    