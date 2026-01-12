import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import sys
import os

from sklearn.preprocessing import StandardScaler
#importing column transformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from dataclasses import dataclass
from sklearn.impute import SimpleImputer

from src.utils import save_object



class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")

            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info("Numerical and categorical pipelines created")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            X_train = train_df.drop(columns=['math_score'], axis=1)
            y_train = train_df['math_score']

            X_test = test_df.drop(columns=['math_score'], axis=1)
            y_test = test_df['math_score']
            logging.info("Separated input features and target feature from train and test data")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training and testing datasets")

            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)
            
            #np.c_ is a NumPy object that concatenates arrays column-wise
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            logging.info("saved transformed train and test arrays")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)