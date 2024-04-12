import numpy as np
import pandas as pd

import pickle

penguins_categories = {'species': ['Adelie', 'Chinstrap', 'Gentoo'],
         'island': ['Biscoe', 'Dream', 'Torgersen'],
         'sex': ['Female', 'Male']
    }

penguins_model = "penguins_tree.pkl"

def penguins_preprocessing(X: list) -> pd.DataFrame:
    """
    This function should accept list of distionaries of data we'd like to prepare for our model.
    Each dictionary key is feature name, value is text/number.
    Function returns the DataFrame with processed data.
    """

    df = pd.DataFrame(X)
    for feature, categories in penguins_categories.items():
        if feature in df.columns:
            df[feature] = df[feature].apply(lambda x: categories.index(x))
            df[feature] = df[feature].astype(np.int64)

    return df

def penguins_classifier(data: list) -> list:
    """
    Classifier function should accept list of crieterias and return model predictions
    """
    X = penguins_preprocessing(data)

    with open(penguins_model, 'rb') as f:
        pickled_tree = pickle.load(f)

    y_pred = pickled_tree.predict(X)
    
    return list(map(lambda x: penguins_categories['species'][x], y_pred))