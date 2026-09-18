import joblib 

def load_model_pkl(model_name):
    """
    Load a model from a pickle file.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        The loaded model.
    """
    return joblib.load(f"{model_name}.pkl")