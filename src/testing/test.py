import sys
import os
import joblib
from sklearn.metrics import classification_report
# Gets the parent directory of your current notebook folder
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Adds it to the search path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir) 

from evaluate import cls_report, plot_pr_and_threshold_curves 
from models.load_pkl import load_model_pkl 
from Enum.PathEnum import PathEnum
from data.data_helper import load_data 
from Preprocessing.preprocess import Preprocessing


if __name__ == '__main__': 
    df_train=load_data(PathEnum.TRAIN_PATH.value)
    df_val=load_data(PathEnum.VAL_PATH.value) 
    df_test=load_data(PathEnum.TEST_PATH.value)

    preprocessing=Preprocessing()

    X_train_scaled, y_train, clip_bounds, scaler = preprocessing.fit_transform(df_train)
    X_eval_scaled, y_eval = preprocessing.transform(df_val, clip_bounds, scaler)
    X_test_scaled, y_test = preprocessing.transform(df_test, clip_bounds, scaler)
    

    lr=load_model_pkl('/home/marwan/Downloads/Machine Learning/Credit-Card-Fraud-Detection/src/models/trained_models/LR')['model'] 
    rf=load_model_pkl('/home/marwan/Downloads/Machine Learning/Credit-Card-Fraud-Detection/src/models/trained_models/RF')['model'] 
    nn=load_model_pkl('/home/marwan/Downloads/Machine Learning/Credit-Card-Fraud-Detection/src/models/trained_models/NN')['model'] 
    vc=load_model_pkl('/home/marwan/Downloads/Machine Learning/Credit-Card-Fraud-Detection/src/models/trained_models/voting_classifier')['model']
    data_dic = {'train' : [X_train_scaled, y_train], 'eval': [X_eval_scaled, y_eval], 'test': [X_test_scaled, y_test]}
    models = [lr, rf, nn, vc]

    print(vc.__class__.__name__) 

    for model in models:
        for d_type in data_dic.keys():
            x, y = data_dic[d_type] 
            y_pred = model.predict(x)
            if model.__class__.__name__ in ['MLPClassifier','VotingClassifier']:
                y_prob = model.predict_proba(x)[:, 1]
            else:
                y_prob = model.predict_proba(x)  
            print(f"Classification report for {model.__class__.__name__} on {d_type} data:")
            print(classification_report(y, y_pred))
            plot_pr_and_threshold_curves(y, y_prob,model_name=f'{model.__class__.__name__}_{d_type}')
            print("\n" + "=" * 40 + "\n")  



   
    