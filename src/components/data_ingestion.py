import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import  train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_dat_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestioin_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the the data ingestoni method or component ")    
        try:
            df=pd.read_csv(r"notebook\data\stud.csv")
            logging.info("exported the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestioin_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestioin_config.raw_dat_path,index=False,header=True)
            logging.info("train test split intiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestioin_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestioin_config.test_data_path,index=False,header=True)
            logging.info("ingestioin of dat is done")

            return(
                self.ingestioin_config.train_data_path
                ,self.ingestioin_config.test_data_path,
                self.ingestioin_config.raw_dat_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()
