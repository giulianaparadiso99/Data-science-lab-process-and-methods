import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from prettytable import PrettyTable
from collections import Counter
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns

    
def compare_distributions(df1, df2, numerical_features=None):
   
    if numerical_features is None:
        numerical_features = df1.select_dtypes(include='number').columns
    
    numerical_features = [col for col in numerical_features if col != 'id']

    num_features = len(numerical_features)
    ncols = 2
    nrows = (num_features + 1) // ncols
    
    plt.figure(figsize=(12, 5*nrows))
    for i, col in enumerate(numerical_features, 1):
        plt.subplot(nrows, ncols, i)
        df1[col].hist(bins=30, density=True, alpha=0.7, color='#1f77b4', label='Development')
        df2[col].hist(bins=30, density=True, alpha=0.7, color='#ff7f0e', label='Evaluation')
        plt.title(f'{col} Distribution')
        plt.legend()

        plt.xlabel(col)
        plt.ylabel('Proportion')

    plt.tight_layout()
    plt.show()




#################################################



def compare_categorical_distributions(df1, df2, categorical_features=None, threshold=0.01):
    
    if categorical_features is None:
        categorical_features = df1.select_dtypes(include='object').columns
    
    num_features = len(categorical_features)
    ncols = 2
    nrows = (num_features + 1) // ncols
    
    plt.figure(figsize=(12, 5*nrows))
    for i, col in enumerate(categorical_features, 1):
        plt.subplot(nrows, ncols, i)
        
        # Frequencies calculation
        freq_df1 = df1[col].value_counts(normalize=True)
        freq_df2 = df2[col].value_counts(normalize=True)
        
        # Function to group rare categories into "Other"
        def group_rare_categories(series, threshold):
            mask = series >= threshold
            grouped = series[mask].copy()
            if (~mask).any():
                grouped['Other'] = series[~mask].sum()
            return grouped
        
        freq_df1_grouped = group_rare_categories(freq_df1, threshold)
        freq_df2_grouped = group_rare_categories(freq_df2, threshold)
        
        # Creation of a combined DataFrame
        freq = pd.DataFrame({'Development': freq_df1_grouped, 'Evaluation': freq_df2_grouped}).fillna(0)
        freq = freq.sort_index()
        
        # Plot
        freq.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], ax=plt.gca())    

        cleaned_labels = [label.replace('$', r'\$') for label in freq.index]
        plt.xticks(range(len(cleaned_labels)), cleaned_labels, rotation=30, ha='right')
        plt.title(f'{col} Distribution (threshold={threshold})')
        plt.xlabel(col)
        plt.ylabel('Proportion')
        plt.legend()

    plt.tight_layout()
    plt.show()




#################################################




def boxplot_features(df1, df2, numerical_features=None):
    
    if numerical_features is None:
        numerical_features = df1.select_dtypes(include='number').columns
    
    numerical_features = [col for col in numerical_features if col != 'id']

    num_features = len(numerical_features)
    ncols = 2
    nrows = (num_features + 1) // ncols
    
    plt.figure(figsize=(12, 5*nrows))
    for i, col in enumerate(numerical_features, 1):
        plt.subplot(nrows, ncols, i)
        data = [df1[col].dropna(), df2[col].dropna()]
        plt.boxplot(data, patch_artist=True,
                    boxprops=dict(facecolor='#1f77b4', color='#1f77b4'),
                    medianprops=dict(color='black'),
                    flierprops=dict(marker='o', color='#1f77b4'))
        plt.xticks([1, 2], ['Development', 'Evaluation'])

        plt.title(f'{col} Box Plot')

    plt.tight_layout()
    plt.show()


def correlation_matrix(df):
   
    numerical_features = df.select_dtypes(include='number').columns
    
    corr = df[numerical_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title("Correlation Matrix delle caratteristiche numeriche")
    plt.tight_layout()
    plt.show()


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





#################################################




def estimate_from_field(df, field):
    series = df[field]
    max_vals, min_vals, std_vals, mean_vals = [], [], [], []
    for line in series:
        matches = re.findall(r'\$\d+(?:\.\d+)?[^\s]*', str(line))
        values = []
        if matches:
            for value in matches:
                match = re.findall(r'\$(\d+(?:\.\d+)?)', value)
                values.append(float(match[0]) if match else 0)
        max_vals.append(np.max(values) if values else 0)
        min_vals.append(np.min(values) if values else 0)
        std_vals.append(np.std(values) if values else 0)
        mean_vals.append(np.mean(values) if values else 0)
    return max_vals, min_vals, std_vals, mean_vals




#################################################




def handle_amenities(df):
    # Cleaning and splitting amenities into lists
    df['amenities'] = df['amenities'].str.replace(r'[\[\]{}"]', '', regex=True)
    df['amenities'] = df['amenities'].str.split(',')

    # Stripping extra spaces from each item, handling NaN by converting to empty list
    df['amenities'] = df['amenities'].apply(lambda x: [item.strip() for item in x] if isinstance(x, list) else [])

    # Encoding
    mlb = MultiLabelBinarizer()
    amenities_encoded = pd.DataFrame(mlb.fit_transform(df['amenities']),
                                    columns=mlb.classes_,
                                    index=df.index)

    # Combining with original DataFrame
    df = pd.concat([df.drop(columns=['amenities']), amenities_encoded], axis=1)
    return df





#################################################




def fill_missing_values(df):
    # Filling missing values for "pets_allowed" and "amenities"
    df["pets_allowed"] = df["pets_allowed"].fillna("no")
    df["amenities"] = df["amenities"].fillna("none")
    return df

# Transforming latitude and longitude values in radians
def to_radians(df):
    return np.deg2rad(df[['latitude', 'longitude']].values)

def finding_cities_states(df):
    # Dividing DataFrame in two subsets, one with missing values and one without missing values
    df_complete = df[df['cityname'].notnull() & df['state'].notnull()]
    df_missing = df[df['cityname'].isnull() | df['state'].isnull()]

    # BallTree Construction
    tree = BallTree(to_radians(df_complete), metric='haversine')

    # Finding the closest latitude/longitude record for each record with missing values
    dist, ind = tree.query(to_radians(df_missing), k=1)

    # Imputing missing values
    nearest_neighbors = df_complete.iloc[ind.flatten()]
    df_missing.loc[:, 'cityname'] = nearest_neighbors['cityname'].values
    df_missing.loc[:, 'state'] = nearest_neighbors['state'].values

    # Reconstruction of the DataFrame
    df_imputed = pd.concat([df_complete, df_missing])
    return df_imputed





#################################################





def add_square_feet_quantile(df, sq_col='square_feet', n_quantiles=10, new_col='square_feet_quantile'):
    """
    Adds a new feature to the dataframe indicating the quantile bin of the square_feet column.
    """
    df = df.copy()
    # Assigning quantile labels (from 0 to n_quantiles-1)
    df[new_col] = pd.qcut(df[sq_col], q=n_quantiles, labels=False, duplicates='drop')
    return df


def impute_by_square_feet_quantile(df, quantiles, quantile_means, sq_col='square_feet', col_to_impute='bathrooms'):
    """
    Impute missing values in col_to_impute using the mean of each square_feet quantile group.
    - quantiles: array-like, the bin edges (as from pd.qcut(..., retbins=True))
    - quantile_means: list or array, mean value for each quantile (same order as quantiles bins)
    - sq_col: the column to use for quantile assignment (default 'square_feet')
    - col_to_impute: the column to fill missing values in
    """
    df = df.copy()
    # Assigning quantile index to each row
    quantile_idx = pd.cut(df[sq_col], bins=quantiles, labels=False, include_lowest=True)
    # Filling missing values with the mean of the corresponding quantile
    for i in range(len(quantile_means)):
        mask = (quantile_idx == i) & (df[col_to_impute].isnull())
        df.loc[mask, col_to_impute] = quantile_means[i]
    return df



