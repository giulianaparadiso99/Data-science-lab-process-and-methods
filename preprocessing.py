import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils import fill_missing_values, finding_cities_states
from utils import add_square_feet_quantile, impute_by_square_feet_quantile,estimate_from_field, handle_amenities

def preprocessing_dev(df):
    df = df.copy()

    # Dropping items with missing values in "latutude" and "longitude"
    df = df.dropna(subset = ["latitude", "longitude"])

    # Filling missing values for "pets allowed" and "amenities"
    df = fill_missing_values(df)

    # Outliers removal
    #df = df[(df["square_feet"]<20000) & (df["price"]<40000)]

    #df = finding_cities_states(df)

    df = add_square_feet_quantile(df)
    print(f"square feet quantile: {df['square_feet_quantile']}")
    if df['bedrooms'].isnull().any():
        df['bedrooms'] = df.groupby('square_feet_quantile')['bedrooms'].transform(lambda x: x.fillna(x.mean()).round().astype(int))

    if df['bathrooms'].isnull().any():
        df['bathrooms'] = df.groupby('square_feet_quantile')['bathrooms'].transform(lambda x: x.fillna(x.mean()).round().astype(int))


    df["max_from_body"], df["min_from_body"], df["std_from_body"], df["mean_from_body"] = estimate_from_field(df, "body")
    df["max_from_title"], df["min_from_title"], df["std_from_title"], df["mean_from_title"] = estimate_from_field(df, "title")
    df = handle_amenities(df)

    # PROVVISORIO PER FARE RUN
    # togliamo title, body e amenities perchè dobbiamo ancora encodarli
    # togliamo currency e pricetype perchè non è utile per il modello
    df = df.drop(columns=["currency", "price_type", "address", "cityname", "state", "body", "title", "body"])

    return df

def re_assign_for_eval(df, eval, feature):
    if eval[feature].isnull().any():
        # Get the bin edges from df's quantiles
        quantile_bins = pd.qcut(df['square_feet'], q=10, retbins=True, duplicates='drop')[1]
        # Assign quantiles to eval using the same bins
        eval['square_feet_quantile'] = pd.cut(eval['square_feet'], bins=quantile_bins, labels=False, include_lowest=True)
        # Compute means per quantile from df
        quantile_means = df.groupby('square_feet_quantile')[feature].mean().round().astype(int)
        # Fill missing values in eval using the quantile means
        mask = eval[feature].isnull()
        eval.loc[mask, feature] = eval.loc[mask, 'square_feet_quantile'].map(quantile_means)
    return eval

def preprocessing_eval(df, eval):
    df = df.copy()
    eval = eval.copy()
    
    eval = fill_missing_values(eval)

    # Filling missing values for "bedroom" and "bathrooms"
    eval = re_assign_for_eval(df,eval,"bedrooms")
    eval = re_assign_for_eval(df,eval,"bathrooms")
    if (eval["bathrooms"].isnull().any()):
        eval.loc[eval["bathrooms"].isnull(), "bathrooms"] = 10

    eval["max_from_body"], eval["min_from_body"], eval["std_from_body"], eval["mean_from_body"] = estimate_from_field(eval, "body")
    eval["max_from_title"], eval["min_from_title"], eval["std_from_title"], eval["mean_from_title"] = estimate_from_field(eval, "title")
    eval = handle_amenities(eval)

    eval = eval.drop(columns=["currency", "price_type", "address", "cityname", "state", "body", "title", "body"])
    eval["latitude"] = eval["latitude"].fillna(eval["latitude"].mean())
    eval["longitude"] = eval["longitude"].fillna(eval["longitude"].mean())
    
    
    return eval