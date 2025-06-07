import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils import fill_missing_values, finding_cities_states
from utils import impute_by_square_feet_bins_fit_and_filter, impute_by_square_feet_bins_transform

def preprocessing(df,bin_means, bins):

    # Filling missing values (pets allowed, amenities)
    df = fill_missing_values(df)

    # Filling missing values (city, state)

    #df= df_train.dropna(subset=["latitude", "longitude"])
    df["latitude"] = df["latitude"].fillna(df["latitude"].mean())
    df["longitude"] = df["longitude"].fillna(df["longitude"].mean())

    print("Missing values, columns latitude:", df.isnull()["latitude"].sum())
    print("Missing values, columns longitude:", df.isnull()["longitude"].sum())

    df = finding_cities_states(df)

    df = impute_by_square_feet_bins_transform(df, bin_means, bins)
    
    print("Missing values, columns bathrooms:", df.isnull()["bathrooms"].sum())
    print("Missing values, columns bedrooms:", df.isnull()["bedrooms"].sum())

    # Drop adrress
    df = df.drop(columns="address")

    # PROVVISORIO PER FARE RUN
    # togliamo title, body e amenities perchè dobbiamo ancora encodarli
    # togliamo currency e pricetype perchè non è utile per il modello
    df = df.drop(columns=["title", "body", "amenities", "currency", "price_type"])

    return df



