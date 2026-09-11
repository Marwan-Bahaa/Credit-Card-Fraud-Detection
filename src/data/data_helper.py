import pandas as pd

import sys
import os

# Gets the parent directory of your current notebook folder
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Adds it to the search path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir) 



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



if __name__ == '__main__': 
    from Enum.PathEnum import PathEnum 

    df = load_data(PathEnum.TRAIN_PATH.value) 
    print(df.columns)

    x, y = separate_X_y(df=df, target='Class') 

    print(x) 

    print(y.value_counts())