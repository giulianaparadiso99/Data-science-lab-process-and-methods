import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils import fill_missing_values, finding_cities_states
from utils import impute_by_square_feet_bins_fit_and_filter, impute_by_square_feet_bins_transform

def preprocessing(df_train, df_test):

    # Filling missing values (pets allowed, amenities)
    df_train = fill_missing_values(df_train)
    df_test = fill_missing_values(df_test)

    # Filling missing values (city, state)

    df_train = df_train.dropna(subset=["latitude", "longitude"])

    print("Missing values in train set, columns latitude:", df_train.isnull()["latitude"].sum())
    print("Missing values in train set, columns longitude:", df_train.isnull()["longitude"].sum())

    df_train = finding_cities_states(df_train)
    
    df_test["latitude"] = df_test["latitude"].fillna(df_test["latitude"].mean())
    df_test["longitude"] = df_test["longitude"].fillna(df_test["longitude"].mean())
    
    print("Missing values in test set, columns latitude:", df_test.isnull()["latitude"].sum())
    print("Missing values in test set, columns longitude:", df_test.isnull()["longitude"].sum())

    df_test = finding_cities_states(df_test)


    # Filling missing values (bathrooms, bedrooms)
    df_train, bin_means, bins = impute_by_square_feet_bins_fit_and_filter(df_train)

    print("Bins means:", bin_means)
    print("Bins:", bins)

    df_train = impute_by_square_feet_bins_transform(df_train, bin_means, bins)
    df_test = impute_by_square_feet_bins_transform(df_test, bin_means, bins)
    
    print("Missing values in train set, columns bathrooms:", df_train.isnull()["bathrooms"].sum())
    print("Missing values in train set, columns bedrooms:", df_train.isnull()["bedrooms"].sum())
    print("Missing values in test set, columns bathrooms:", df_test.isnull()["bathrooms"].sum())
    print("Missing values in test set, columns bedrooms:", df_test.isnull()["bedrooms"].sum())

    # Drop adrress
    df_train = df_train.drop(columns="address")
    df_test = df_test.drop(columns="address")

    # PROVVISORIO PER FARE RUN
    # togliamo title, body e amenities perchè dobbiamo ancora encodarli
    # togliamo currency e pricetype perchè non è utile per il modello
    df_train = df_train.drop(columns=["title", "body", "amenities", "currency", "price_type"])
    df_test = df_test.drop(columns=["title", "body", "amenities", "currency", "price_type"])


    
    return df_train, df_test



