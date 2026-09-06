import pandas as pd
def load_data(path:str): 
    if path == None: 
        raise(ValueError)
    return pd.read_csv(path)

def split_x_y(df:pd.DataFrame, target:str):  
    x = df.drop([target], axis=1) 
    y=df[target] 
    return x, y 
