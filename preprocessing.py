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
    df_train = finding_cities_states(df_train)
    df_test = finding_cities_states(df_test)

    # Filling missing values (bathrooms, bedrooms)
    df_train, bin_means, bins = impute_by_square_feet_bins_fit_and_filter(df_train)
    df = impute_by_square_feet_bins_transform(df, bin_means, bins)

    # Drop adrress
    df_train = df_train.drop(columns="address")
    df_test = df_test.drop(columns="address")

    # PROVVISORIO PER FARE RUN
    # togliamo title, body e amenities perchè dobbiamo ancora encodarli
    # togliamo currency e pricetype perchè non è utile per il modello
    df_train = df_train.drop(columns=["title", "body", "amenities", "currency", "pricetype"])
    df_test = df_test.drop(columns=["title", "body", "amenities", "currency", "pricetype"])
    df_test["latitude"] = df["latitude"].fillna(df["latitude"].mean())
    df_test["longitude"] = df["longitude"].fillna(df["longitude"].mean())
    df_train = df_train.dropna(subset=["latitude", "longitude"])

    
    return df_train, df_test



