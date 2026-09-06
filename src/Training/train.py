import argparse
import sys
import os
import joblib 

# Gets the parent directory of your current notebook folder
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Adds it to the search path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir) 

from Enum.PathEnum import PathEnum
from data.data_helper import load_data
from Preprocessing.preprocess import Preprocessing
from models.LR_model import LogisticRegression_model 
from models.RF_model import  RandomForest_model


class Train: 
    def __init__(self, model_name:str): 
        self.model_name = model_name  
        self.model = None  
        self.scaler = None 
        self.target_col = None
        self.factor = None
        # Stores learned bounds per column: {'col_name': (lower, upper)}
        self.clip_bounds = None 
        self.Preprocess_pipline = Preprocessing()

    def prepare_data(self): 
        df_train = load_data(PathEnum.TRAIN_PATH.value) 
        X_train_scaled, y_train, self.clip_bounds, self.scaler = self.Preprocess_pipline.fit_transform(df_train)
        return X_train_scaled, y_train 

    def train_model(self, X, y): 
        if self.model_name == 'LR': 
            self.model = LogisticRegression_model(
                                                  metric='f1', 
                                                  random_state=42
                                                  )
        elif self.model_name == 'RF': 
            self.model = RandomForest_model( 
                    metric='f1', 
                    n_estimators=100, 
                    max_depth=10, 
                    class_weights={0: 1, 1: 3}, 
                    random_state=42
                    )
        else: 
            return None      

        model = self.model.fit(X, y) 

        return model 
        
# ...
if __name__ == "__main__":
    Train_pipline = Train(model_name='RF')
    X, y = Train_pipline.prepare_data() 
    print(X.shape)
    print(y.shape) 
    model = Train_pipline.train_model(X, y) 
    
    compress_model = {
        'model': model, 
        'threshould': model.best_threshold_ 
    }  

    compress_preprocess = {
        'scaler':Train_pipline.scaler,
        'clip_limts':Train_pipline.clip_bounds
    }    

    joblib.dump(compress_preprocess,f'preprocess_parmters.pkl')


 

        

         