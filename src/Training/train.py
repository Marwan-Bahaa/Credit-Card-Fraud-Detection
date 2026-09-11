import argparse
import sys
import os
import joblib 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold 
from sklearn.metrics import f1_score, make_scorer 


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
    def __init__(self, model_name:str = None): 
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

    def train_LR(self, x_train_scaled, y_train):   
        best_parms={} 

        par_list = { 
                 'C':[0.1, 1.0, 10.0],
                'class_weight': ['balanced', None, {0: 0.35, 1: 0.65}, {0: 0.25, 1: 0.75}, {0: 0.15, 1: 0.85}],
                'solver':       ['sag', 'lbfgs', 'saga', ' newton-cg'],  
                'max_iter':     [400, 500, 600, 800],   
        } 

        lr=LogisticRegression() 

        score=make_scorer(f1_score, pos_label=1) 

        stratified_kfold = StratifiedKFold(
            n_splits=5, 
            shuffle=True, 
            random_state=42
        )        

        grid_search = GridSearchCV(
            lr, 
            par_list, 
            cv=stratified_kfold, 
            scoring=score, 
            n_jobs=-1 
        )
        grid_search.fit(x_train_scaled, y_train)
        best_parms = grid_search.best_params_ 
        print("Best Hyperparameters:", best_parms)
        lr = LogisticRegression_model(**best_parms, random_state=42) 
        optimal_threshould=lr.fit(x_train_scaled, y_train).best_threshold_ 
        return {'model':lr, 'optimal_threshould':optimal_threshould, 'best_parms':best_parms}
        


         

    def train_RF(self, x_train_scaled, y_train): 
        param_distributions = {
            'n_estimators': [200, 400, 600 ,800],
            'min_samples_leaf': [2, 5, 10, 15],
            'min_samples_split': [5, 10, 20],
            'class_weight': [{0: 0.20, 1: 0.80}, 'balanced_subsample', {0: 0.15, 1: 0.85}],
        }

        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, pos_label=1)

        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(n_jobs=-1, bootstrap=True, random_state=42),
            param_distributions=param_distributions,
            scoring=scorer,
            cv=stratified_kfold,
            n_iter=20,  
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        random_search.fit(x_train_scaled, y_train)
        parameters = random_search.best_params_
        print("Best Hyperparameters for Random Forest:", parameters)
        RF = RandomForest_model(**parameters, random_state=42) 
        opt_thr = RF.fit(x_train_scaled, y_train).best_threshold_ 
        return {'model':RF, 'optimal_thershould':opt_thr, 'best_paramters':parameters} 
     
        
            

    def train_model(self, X, y): 
        model_par = {}
        if self.model_name == 'LR': 
          model_par = self.train_LR(X, y)
        elif self.model_name == 'RF': 
           model_par = self.train_RF(X, y) 
        else: 
            return None      
        return model_par 


        
# ...
if __name__ == "__main__":
    Train_pipline = Train()
    X_scaled, y = Train_pipline.prepare_data()

    for model_name in ['LR','RF']:
        Train_model = Train(model_name=model_name)
        model_pars = Train_model.train_model(X=X_scaled, y=y) 
        joblib.dump(model_pars,f'{Train_model.model_name}_parmters.pkl') 
        

    compress_preprocess = {
        'scaler':Train_pipline.scaler,
        'clip_limts':Train_pipline.clip_bounds
    } 
       
    joblib.dump(compress_preprocess,f'preprocess_parmters.pkl')      