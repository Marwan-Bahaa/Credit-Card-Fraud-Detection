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
from sklearn.metrics import f1_score, make_scorer, classification_report 
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score  
from testing.evaluate import cls_report, plot_pr_and_threshold_curves 
from sklearn.neural_network import MLPClassifier 
from sklearn.neighbors import KNeighborsClassifier 


class Train: 
    def __init__(self): 
        self.model_name = None 
        self.model = None  
        self.scaler = None 
        self.target_col = None
        self.factor = None
        self.clip_bounds = None 
        self.Preprocess_pipline = None 


    def train_LR(self,x_train, y_train, x_val, y_val):
        n_inner_cv = 3
        n_outer_cv = 5
        n_iter = 50

        param_distributions ={
                    'penalty': ['l2'],
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['lbfgs', 'sag', 'saga', 'newton-cg'],
                    'class_weight': [
                        'balanced',
                        None,
                        {0: 0.35, 1: 0.65},
                        {0: 0.25, 1: 0.75},
                        {0: 0.15, 1: 0.85},
                    ],
                    'max_iter': [500, 1000, 1500],
                }

            
        lr = LogisticRegression_model(random_state=42)
        scorer = make_scorer(f1_score, pos_label=1)

        inner_cv = StratifiedKFold(
            n_splits=n_inner_cv, shuffle=True, random_state=42
        ) 

        outer_cv = StratifiedKFold(
            n_splits=n_outer_cv, shuffle=True, random_state=42
        )

        inner_search = RandomizedSearchCV(
            estimator=lr,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=inner_cv,
            scoring=scorer,
            n_jobs=-1,
            random_state=42
        )

        nested_scores = cross_val_score(
            estimator=inner_search,
            X=x_train,
            y=y_train,
            cv=outer_cv,
            scoring=scorer,
            n_jobs=-1,
        )

        print("Fitting final model...")
        inner_search.fit(x_train, y_train)
        best_parms = inner_search.best_params_ 
        print(f'best parmters : {best_parms}')  

        model = LogisticRegression_model(**best_parms) 
        model.fit(x_train, y_train)  
        y_pred = model.predict(x_val)
        y_prob = model.predict_proba(x_val) 

        print(classification_report(y_val, y_pred))
        plot_pr_and_threshold_curves(y_val, y_prob)

        return {'model':model, 'best thershould': model.best_threshould, 'best parmters': best_parms}



    

    def train_RF(self, x_train, y_train, x_val, y_val):
        n_inner_cv = 3
        n_outer_cv = 5
        n_iter = 50

        param_distributions = {
            'n_estimators': [200, 400, 600 ,800],
            'min_samples_leaf': [2, 5, 10, 15],
            'min_samples_split': [5, 10, 20],
            'class_weight': [{0: 0.20, 1: 0.80}, 'balanced_subsample', {0: 0.15, 1: 0.85}],
        }
            
        lr = RandomForest_model(random_state=42)
        scorer = make_scorer(f1_score, pos_label=1)

        inner_cv = StratifiedKFold(
            n_splits=n_inner_cv, shuffle=True, random_state=42
        ) 

        outer_cv = StratifiedKFold(
            n_splits=n_outer_cv, shuffle=True, random_state=42
        )

        inner_search = RandomizedSearchCV(
            estimator=lr,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=inner_cv,
            scoring=scorer,
            n_jobs=-1,
            random_state=42
        )

        nested_scores = cross_val_score(
            estimator=inner_search,
            X=x_train,
            y=y_train,
            cv=outer_cv,
            scoring=scorer,
            n_jobs=-1,
        )

        print("Fitting final model...")
        inner_search.fit(x_train, y_train)
        best_parms = inner_search.best_params_ 
        print(f'best parmters : {best_parms}') 
        model = LogisticRegression_model(**best_parms) 

        model.fit(x_train, y_train) 
        y_pred = model.predict(x_val)
        y_prob = model.predict_proba(x_val) 
    
        print(classification_report(y_val, y_pred))
        plot_pr_and_threshold_curves(y_val, y_prob)

        return {'model':model, 'best thershould': model.best_threshould, 'best_parmters': best_parms} 
    

    def train_nn(self, x_train, y_train, x_val, y_val):
        n_inner_cv = 3
        n_outer_cv = 5
        n_iter = 50

        param_dist = {
        'activation': ['relu'],
        'hidden_layer_sizes': [
            (30, 20), 
            (30, 20, 10), 
            (40, 30, 20), 
            (64, 32, 16),
            (64, 32, 32, 16)
        ],
        'solver': ['adam', 'sgd'],
        'batch_size': [64, 128, 512],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'alpha': [0.001, 0.01, 0.025],
        'max_iter': [800, 1000, 2000, 3000]
        }

        nn_cv = MLPClassifier(random_state=42)
        scorer = make_scorer(f1_score, pos_label=1)

        inner_cv = StratifiedKFold(
            n_splits=n_inner_cv, shuffle=True, random_state=42
        ) 

        outer_cv = StratifiedKFold(
            n_splits=n_outer_cv, shuffle=True, random_state=42
        )

        inner_search = RandomizedSearchCV(
            estimator=nn_cv,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=inner_cv,
            scoring=scorer,
            n_jobs=-1,
            random_state=42
        )

        nested_scores = cross_val_score(
            estimator=inner_search,
            X=x_train,
            y=y_train,
            cv=outer_cv,
            scoring=scorer,
            n_jobs=-1,
        )

        print("Fitting final model...")
        inner_search.fit(x_train, y_train)
        best_parms = inner_search.best_params_ 
        print(f'best parmters : {best_parms}') 
        model = MLPClassifier(**best_parms, early_stopping=True) 

        model.fit(x_train, y_train) 
        y_pred = model.predict(x_val)
        y_prob = model.predict_proba(x_val) 
    
        print(classification_report(y_val, y_pred))
        plot_pr_and_threshold_curves(y_val, y_prob)

        return {'model':model, 'best thershould': model.best_threshold_, 'best_parmters': best_parms}
        


    def train_knn(self, X_train, y_train, X_val, y_val, random_seed=42):

        param_distributions = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        }

        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
        scorer = make_scorer(f1_score, pos_label=1)

        random_search = RandomizedSearchCV(
            estimator=KNeighborsClassifier(n_jobs=-1),
            param_distributions=param_distributions,
            scoring=scorer,
            cv=stratified_kfold,
            n_iter=20,  
            n_jobs=-1,
            verbose=2,
            random_state=random_seed 
        )

        random_search.fit(X_train, y_train)

        parameters = random_search.best_params_
        print("Best Hyperparameters for KNN:", parameters)
    
        knn = KNeighborsClassifier(**parameters, n_jobs=-1)

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val) 
        
        print(classification_report(y_val, y_pred)) 

        return {"model": knn , "parameters": parameters}



if __name__ == "__main__":
    from Enum.PathEnum import PathEnum 
    from data.data_helper import load_data 
    from Preprocessing.preprocess import Preprocessing
    from models.save_model import save_model_pkl 

    df_train=load_data(PathEnum.TRAIN_PATH.value)
    df_val=load_data(PathEnum.VAL_PATH.value) 
    df_test=load_data(PathEnum.TEST_PATH.value)

    preprocessing = Preprocessing()
    X_train_scaled, y_train, clip_bounds, scaler = preprocessing.fit_transform(df_train)
    X_eval_scaled, y_eval = preprocessing.transform(df_val, clip_bounds, scaler)
    X_test_scaled, y_test = preprocessing.transform(df_test, clip_bounds, scaler)

    Train_pipline = Train()     
    print('Train LR')
    dic_lr = Train_pipline.train_LR(X_train_scaled, y_train, X_eval_scaled, y_eval)  
    print('Train RF')
    dic_rf = Train_pipline.train_RF(X_train_scaled, y_train, X_eval_scaled, y_eval) 
    print('Train NN')
    dic_nn = Train_pipline.train_nn(X_train_scaled, y_train, X_eval_scaled, y_eval)  
    print('Train KNN')
    dic_knn = Train_pipline.train_knn(X_train_scaled, y_train, X_eval_scaled, y_eval)  
    
    test_data = {'X':X_test_scaled, 'Y':y_test}

    save_model_pkl(model_pack=dic_lr, model_name='LR')  
    save_model_pkl(model_pack=dic_rf, model_name='RF') 
    save_model_pkl(model_pack=dic_nn, model_name='NN') 
    save_model_pkl(model_pack=dic_nn, model_name='KNN') 
    save_model_pkl(model_pack=test_data, model_name='test_data_prepared') 
