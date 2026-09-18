import joblib 
import os 

def save_model_pkl(model_pack, model_name): 
    path = '/home/marwan/Downloads/Machine Learning/Credit-Card-Fraud-Detection/src/models/trained_models'
    os.makedirs(path, exist_ok=True)
    joblib.dump(model_pack, f'{path}/{model_name}.pkl') 


if __name__=='__main__': 
    ...