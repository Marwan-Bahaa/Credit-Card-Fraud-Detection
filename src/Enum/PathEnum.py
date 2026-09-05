from enum import Enum
class PathEnum(str, Enum): 
    TRAIN_PATH = '/mnt/e/Machine Learning/Credit-Card-Fraud-Detection/src/data/split/train.csv'
    TRAIN_VAL_PATH = '/mnt/e/Machine Learning/Credit-Card-Fraud-Detection/src/data/split/trainval.csv' 
    VAL_PATH = '/mnt/e/Machine Learning/Credit-Card-Fraud-Detection/src/data/split/val.csv' 
    TEST_PATH = '/mnt/e/Machine Learning/Credit-Card-Fraud-Detection/src/data/split/test.csv'


    