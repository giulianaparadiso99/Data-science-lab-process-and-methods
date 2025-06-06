import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from prettytable import PrettyTable


def print_stats(keys, values, feature):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.bar(keys, values)
    plt.ylabel('Count')
    plt.xlabel('Category')
    plt.title(f'Number of Listings per {feature}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def stats(df):
    category_numbers = {}
    bathroom_numbers = {}
    bedroom_numbers = {}
    pets = {}
    for i in range (len (df)):
        category = df.iloc[i]["category"]
        bathrooms = df.iloc[i]["bathrooms"]
        bedrooms = df.iloc[i]["bedrooms"]
        pets_all = df.iloc[i]["pets_allowed"]

        if category not in category_numbers:
            category_numbers[category] = 0
        category_numbers[category]+=1

        if str(bathrooms) not in bathroom_numbers:
            bathroom_numbers[str(bathrooms)] = 0
        bathroom_numbers[str(bathrooms)]+=1

        if str(bedrooms) not in bedroom_numbers:
            bedroom_numbers[str(bedrooms)] = 0
        bedroom_numbers[str(bedrooms)]+=1

        if pets_all not in pets:
            pets[pets_all] = 0
        pets[pets_all]+=1

    print_stats(category_numbers.keys(), category_numbers.values(), "category")
    print_stats(bathroom_numbers.keys(), bathroom_numbers.values(), "bathrooms")
    print_stats(bedroom_numbers.keys(), bedroom_numbers.values(), "bedrooms")
    print_stats(pets.keys(), pets.values(),"pets")


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