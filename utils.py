import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from prettytable import PrettyTable
from collections import Counter
from sklearn.neighbors import BallTree

def print_stats(keys, values, feature):
    import matplotlib.pyplot as plt

    keys = [str(k) for k in keys]  # Convert all keys to strings

    plt.figure(figsize=(8, 5))
    plt.bar(keys, values)
    plt.ylabel('Count')
    plt.xlabel('Category')
    plt.title(f'Number of Listings per {feature}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def stats(df):
    category_numbers = Counter(df["category"])
    bathroom_numbers = Counter(df["bathrooms"].astype(str))
    bedroom_numbers = Counter(df["bedrooms"].astype(str))
    pets = Counter(df["pets_allowed"])
    currencies = Counter(df["currency"])
    states = Counter(df["state"])
    cities = Counter(df["cityname"])

    print_stats(category_numbers.keys(), category_numbers.values(), "category")
    print_stats(bathroom_numbers.keys(), bathroom_numbers.values(), "bathrooms")
    print_stats(bedroom_numbers.keys(), bedroom_numbers.values(), "bedrooms")
    print_stats(pets.keys(), pets.values(), "pets")
    print_stats(currencies.keys(), currencies.values(), "currencies")
    print_stats(states.keys(), states.values(), "states")
    print_stats(cities.keys(), cities.values(), "cities")
    



def exploration_plots(df, features):
    for feature in features:
        plt.figure(figsize=(12, 5))
        
        # Scatter plot: feature vs price
        plt.subplot(1, 2, 1)
        plt.scatter(df[feature], df["price"], alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel("price")
        plt.title(f"{feature} vs Price (Scatter)")
        
        # Histogram: average price per feature value
        plt.subplot(1, 2, 2)
        avg_price = df.groupby(feature)["price"].mean()
        avg_price.plot(kind="bar")
        plt.ylabel("Average price")
        plt.xlabel(feature)
        plt.title(f"Average Price by {feature} (Histogram)")
        plt.tight_layout()
        plt.show()



def fill_missing_values(df):
    # Fill missing values efficiently
    df["pets_allowed"] = df["pets_allowed"].fillna("no")
    df["amenities"] = df["amenities"].fillna("none")
    return df

# Converti lat/lon in radianti
def to_radians(df):
    return np.deg2rad(df[['latitude', 'longitude']].values)

def finding_cities_states(df):
    # Dividi in completi e incompleti
    df_complete = df[df['cityname'].notnull() & df['state'].notnull()]
    df_missing = df[df['cityname'].isnull() | df['state'].isnull()]

    # Costruisci BallTree
    tree = BallTree(to_radians(df_complete), metric='haversine')

    # Trova vicino più vicino per ogni record mancante
    dist, ind = tree.query(to_radians(df_missing), k=1)

    # Imputa valori
    nearest_neighbors = df_complete.iloc[ind.flatten()]
    df_missing.loc[:, 'cityname'] = nearest_neighbors['cityname'].values
    df_missing.loc[:, 'state'] = nearest_neighbors['state'].values

    # Ricombina dataset
    df_imputed = pd.concat([df_complete, df_missing])
    return df_imputed


def impute_by_square_feet_bins_fit_and_filter(df, sq_col='square_feet', cols_to_impute=['bathrooms', 'bedrooms'], q=10):
    """
    Rimuove gli outlier su 'sq_col', crea bin per quantili, calcola le medie di imputazione.
    
    Restituisce:
    - df filtrato (senza outlier e con size_bin)
    - bin_means: medie per bin
    - bins: i bin usati per qcut (serve per trasformazioni successive)
    """
    df = df.copy()

    # Step 1: Rimozione outlier
    Q1 = df[sq_col].quantile(0.25)
    Q3 = df[sq_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_filtered = df[(df[sq_col] >= lower) & (df[sq_col] <= upper)].copy()

    # Step 2: Bin quantili
    df_filtered['size_bin'], bins = pd.qcut(df_filtered[sq_col], q=q, duplicates='drop', retbins=True)

    # Step 3: Medie per bin
    bin_means = df_filtered.groupby('size_bin')[cols_to_impute].mean()

    return df_filtered, bin_means, bins

def impute_by_square_feet_bins_transform(df, bin_means, bins, sq_col='square_feet', cols_to_impute=['bathrooms', 'bedrooms']):
    """
    Applica l'imputazione usando i bin e le medie calcolate sul training.
    """
    """
    Applica l'imputazione usando i bin e le medie calcolate sul training.
    Dopo l'imputazione, riempie eventuali valori mancanti residui con la mediana della colonna.
    """
    df = df.copy()

    # Crea i bin a partire dai limiti salvati
    df['size_bin'] = pd.cut(df[sq_col], bins=bins, include_lowest=True)

    # Imputa i valori mancanti riga per riga
    def impute_row(row):
        bin_label = row['size_bin']
        for col in cols_to_impute:
            if pd.isnull(row[col]) and bin_label in bin_means.index:
                row[col] = bin_means.loc[bin_label, col]
        return row

    df = df.apply(impute_row, axis=1)
    df.drop(columns='size_bin', inplace=True)

    # Fallback: riempi eventuali valori mancanti residui con la mediana della colonna
    for col in cols_to_impute:
        df[col] = df[col].fillna(df[col].median())

    return df



