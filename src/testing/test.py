import sys
import os
import joblib
from sklearn.metrics import classification_report
# Gets the parent directory of your current notebook folder
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Adds it to the search path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir) 

from Enum.PathEnum import PathEnum
from data.data_helper import load_data 
from Preprocessing.preprocess import Preprocessing 
from testing.evaluate import plot_pr_curve 
from evaluate import cls_report


# if __name__ == '__main__': 
#   model_par = joblib.load(f'{parent_dir}/Training/preprocess_parmters.pkl') 
#   LR_m = joblib.load(f'{parent_dir}/Training/LR_parmters.pkl')  
#   RF_m = joblib.load(f'{parent_dir}/Training/RF_parmters.pkl')   
   
#   scaler = model_par['scaler']
#   clip_limts = model_par['clip_limts'] 
 
#   lr_model = RF_m['model']  
#   rf_model = RF_m['model'] 

#   df_train = load_data(PathEnum.TRAIN_PATH.value) 
#   df_val = load_data(PathEnum.VAL_PATH.value) 
#   df_test = load_data(PathEnum.TEST_PATH.value) 

#   preprocess = Preprocessing() 

#   x_train, y_train = preprocess.transform(df_train, clip_limts, scaler)
#   x_val, y_val = preprocess.transform(df_val, clip_limts, scaler)
#   x_test, y_test = preprocess.transform(df_test, clip_limts, scaler) 

#   yt_pred = rf_model.predict(x_train)
#   yv_pred = rf_model.predict(x_val)
#   yts_pred = rf_model.predict(x_test) 

#   train = [y_train, yt_pred]
#   val   = [y_val, yv_pred]
#   test  = [y_test, yts_pred] 

#   cls_report(lr_model, train, val, test)  
   
if __name__=='__main__': 
    from models.LR_model import LogisticRegression_model 
    from models.RF_model import RandomForest_model
    from data.data_helper import load_data 
    from Enum.PathEnum import PathEnum 
    from Preprocessing.preprocess import Preprocessing
    from sklearn.metrics import classification_report

    df=load_data(PathEnum.TRAIN_PATH.value)
    df_val=load_data(PathEnum.VAL_PATH.value)
    preprocess = Preprocessing() 
    x_scaled, y, limts, scaler = preprocess.fit_transform(df)

    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # y_train: الأهداف (Labels) الخاصة بك
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='', classes=classes, y=y)

    class_weights_dict = dict(zip(classes, weights))


    xval_scaled, y_val = preprocess.transform(df_val, limts, scaler) 

    lr=LogisticRegression_model(class_weight=class_weights_dict) 
    rf=RandomForest_model(class_weight=class_weights_dict)   
    
    lr.fit(x_scaled, y)
    rf.fit(x_scaled, y) 

    ylr_pre=lr.predict(x_scaled)
    yrf_pre=rf.predict(x_scaled)
    ylr_preval=lr.predict(xval_scaled)
    yrf_preval=rf.predict(xval_scaled)

    print(classification_report(y, ylr_pre))
    print(classification_report(y_val, ylr_preval))

    print(classification_report(y, yrf_pre))
    print(classification_report(y_val, yrf_preval))  

    lr.plot_threshold_curve(x_scaled, y)
    rf.plot_threshold_curve(x_scaled, y)


   

