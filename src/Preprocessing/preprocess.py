import sys
import os

# Gets the parent directory of your current notebook folder
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Adds it to the search path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir) 

import numpy as np
from sklearn.preprocessing import StandardScaler
from data.data_helper import separate_X_y, balance_data_transformation
from helper_functions import load_config 

config = load_config('/home/marwan/Downloads/Machine Learning/Credit-Card-Fraud-Detection/src/Config/config.yaml')


class Preprocessing:
    def __init__(self, target_col='Class', factor=1.5, clip_bounds=None):
        self.target_col = target_col
        self.factor = factor
        
        # Stores learned bounds per column: {'col_name': (lower, upper)}
        self.clip_bounds = clip_bounds if clip_bounds is not None else {}
        self.scaler = StandardScaler()
       


    def _apply_log(self, df, cols=['Time', 'Amount']):
        """Applies log1p transformation to specified features."""
        df_out = df.copy()
        for col in cols:
            if col in df_out.columns:
                df_out[f'{col}_log'] = np.log1p(np.maximum(0, df_out[col]))
                df_out = df_out.drop(columns=[col])
        return df_out

    def _calc_boundaries(self, df, columns=None, factor=None):
        """
        Calculates and stores IQR lower and upper bounds ONLY on training features.
        """
        f = factor if factor is not None else self.factor
        features = columns if columns is not None else [c for c in df.columns if c != self.target_col]
        
        bounds = {}
        for col in features:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - f * iqr
            upper_bound = q3 + f * iqr
            bounds[col] = (lower_bound, upper_bound)

        self.clip_bounds = bounds
        return bounds

    def _clip_outliers(self, df, bounds=None):
        """
        Clips values using provided bounds dictionary or self.clip_bounds.
        """
        active_bounds = bounds if bounds is not None else self.clip_bounds
        if not active_bounds:
            raise ValueError("No clip boundaries found. Call '_calc_boundaries' or pass a 'bounds' dict.")

        df_clipped = df.copy()
        for col, (lower_bound, upper_bound) in active_bounds.items():
            if col in df_clipped.columns:
                df_clipped[col] = df_clipped[col].clip(lower=lower_bound, upper=upper_bound)

        return df_clipped



    def fit_transform(self, train_df):
        """
        Executes pipeline on TRAIN data:
        Calculate Bounds -> Clip Outliers -> Log Transform -> Standard Scaling (fits scaler)
        """
        if config['preprocessing']['drop_duplicates']:
            train_df = train_df.drop_duplicates().reset_index(drop=True)

        if config['preprocessing']['clip_outliers']:
           self._calc_boundaries(train_df)

        if config['preprocessing']['clip_outliers']:
           self._calc_boundaries(train_df)
           df_clipped = self._clip_outliers(train_df)
        else:
            df_clipped = train_df

        if config['preprocessing']['log_transform']:
            df_log = self._apply_log(df_clipped, cols=['Time', 'Amount'])

        if config['preprocessing']['standard_scaling']:
            X_train, y_train = separate_X_y(df_log, self.target_col)

            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train, y_train = separate_X_y(df_log, self.target_col)
            X_train_scaled = X_train

        if config['preprocessing']['balancing']['do_balance']:
           X_train_scaled, y_train = balance_data_transformation(X_train_scaled, y_train, config['preprocessing']['balancing'], config['preprocessing']['balancing']['sampling_strategy'], config['preprocessing']['balancing']['k_neighbors'], config['preprocessing']['balancing']['random_state'])

        return X_train_scaled, y_train, self.clip_bounds, self.scaler

    def transform(self, df, custom_bounds=None, scaler=None):
        """
        Executes pipeline on VAL, TEST, or INFERENCE data:
        Clip Outliers (uses stored/passed bounds) -> Log Transform -> Standard Scaling (uses train scaler)
        """
        
        if config['preprocessing']['drop_duplicates']:
            df = df.drop_duplicates().reset_index(drop=True)

        
        if config['preprocessing']['clip_outliers']:
            df_clipped = self._clip_outliers(df, bounds=custom_bounds)
        else:
            df_clipped = df

        if config['preprocessing']['log_transform']:
            df_log = self._apply_log(df_clipped, cols=['Time', 'Amount'])
        else:
            df_log = df_clipped

        X_eval, y_eval = separate_X_y(df_log, self.target_col)

        if config['preprocessing']['standard_scaling']:
            X_eval_scaled = scaler.transform(X_eval)
        else:
            X_eval_scaled = X_eval 

        if config['preprocessing']['balancing']['do_balance']:
            X_eval_scaled, y_eval = balance_data_transformation(X_eval_scaled, y_eval, config['preprocessing']['balancing'], config['preprocessing']['balancing']['sampling_strategy'], config['preprocessing']['balancing']['k_neighbors'], config['preprocessing']['balancing']['random_state'])    

        return X_eval_scaled, y_eval




if __name__ == '__main__':
    from Enum.PathEnum import PathEnum
    from data.data_helper import load_data 

    preprocessor = Preprocessing(target_col='Class', factor=1.5)
     

    train_df = load_data(PathEnum.TRAIN_PATH.value)
    val_df = load_data(PathEnum.VAL_PATH.value)


    X_train, y_train, clip_bounds, scaler = preprocessor.fit_transform(train_df)


    X_val, y_val = preprocessor.transform(val_df, clip_bounds, scaler)
   
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")

    print(train_df.describe()) 
    print(val_df.describe()) 

    print(X_train[:]) 
    print(X_val[:]) 

