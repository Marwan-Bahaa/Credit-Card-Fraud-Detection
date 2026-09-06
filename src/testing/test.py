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

if __name__ == '__main__': 
  model_par = joblib.load(f'{parent_dir}/Training/preprocess_parmters.pkl') 
  LR_m = joblib.load(f'{parent_dir}/Training/LR.pkl')  
  RF_m = joblib.load(f'{parent_dir}/Training/RF.pkl')  
  scaler = model_par['scaler']
  clip_limts = model_par['clip_limts'] 
  lr_model = LR_m['model']  
  rf_model = RF_m['model'] 
  df_test = load_data(PathEnum.TEST_PATH.value) 
  x_test, y_test = Preprocessing().transform(df_test, clip_limts, scaler) 

  print('\t=========LR==========')  
  y_predict1 = lr_model.predict(x_test)   
  print(classification_report(y_true=y_test, y_pred=y_predict1))

  print('\t=========RF==========')  
  y_predict2 = rf_model.predict(x_test)   
  print(classification_report(y_true=y_test, y_pred=y_predict2))

  plot_pr_curve(y_test, y_predict1, model_name='LR')  
  plot_pr_curve(y_test, y_predict2, model_name='RF')  
  