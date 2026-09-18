import pandas as pd

import sys
import os

# Gets the parent directory of your current notebook folder
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Adds it to the search path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir) 

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

def load_data(path:str): 
    if path == None: 
        raise ValueError('this is invalid path')
    return pd.read_csv(path)



def separate_X_y(df:pd.DataFrame, target='Class'):
    """Separates feature matrix X and target y."""
    cols = df.columns 
    print(cols)
    if target in cols:
        X = df.drop(columns=[target])
        y = df[target]
        return X, y
    return df, None


def balance_data_transformation(X_train, y_train, balance_type='smote',sampling_strategy='auto',k=5,random_state=42):
    """
    Balance the input training data using the specified balancing strategy.

    Parameters:
    - X_train (numpy array or pandas DataFrame): The training data features.
    - y_train (numpy array or pandas DataFrame): The training data labels.
    - balance_type (str, optional): The type of balancing strategy to be used.
      Options:
      - 'under_sampling': for random under-sampling
      - 'over_sampling': for random over-sampling
      - 'smote': for SMOTE
      - 'SMOTEENN': for SMOTEENN combination
      - 'SMOTETomek': for SMOTETomek combination
      Default is 'smote'.
    - random_state (int, optional): The random state for reproducibility. Default is None.

    Returns:
    - X_resampled (numpy array or pandas DataFrame): The balanced training data features.
    - y_resampled (numpy array or pandas DataFrame): The balanced training data labels.
    """
    
 
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.values.ravel() 
    
    print("Dataset before balancing:")
    print(f"Number of Non-fraud transactions: {len(y_train[y_train == 0])}")
    print(f"Number of fraud transactions:     {len(y_train[y_train == 1])}")

    if balance_type == 'under':
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    elif balance_type == 'over':
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    elif balance_type == 'smote':
        sampler = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=k)
    elif balance_type == 'SMOTEENN':
        sampler = SMOTEENN(
                          random_state=random_state, 
                          sampling_strategy=sampling_strategy,
                          smote=SMOTE(random_state=random_state,
                                      sampling_strategy=sampling_strategy,
                                      k_neighbors=k),  
                         )
    elif balance_type == 'SMOTETomek':
        sampler = SMOTETomek(
                             random_state=random_state, 
                             sampling_strategy=sampling_strategy,
                             smote=SMOTE(random_state=random_state,
                                         sampling_strategy=sampling_strategy,
                                         k_neighbors=k),
                             n_jobs=-1
                        )
    else:
        raise ValueError("Invalid balance type. Please choose 'under_sampling', 'over_sampling', 'smote', 'SMOTEENN', or 'SMOTETomek'.")
    
    # Fit and resample the training data using the chosen sampler
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    print("\n Dataset after balancing:")
    print(f"Number of Non-fraud transactions: {len(y_resampled[y_resampled == 0])}")
    print(f"Number of fraud transactions:     {len(y_resampled[y_resampled == 1])}")

    return X_resampled, y_resampled

if __name__ == '__main__': 
    from Enum.PathEnum import PathEnum 

    df = load_data(PathEnum.TRAIN_PATH.value) 
    print(df.columns)

    x, y = separate_X_y(df=df, target='Class') 

    print(x) 

    print(y.value_counts())