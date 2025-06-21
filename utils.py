import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def plot_apartment_choropleth(df, shapefile_path='geopandas_files/states/ne_110m_admin_1_states_provinces.shp'):
    
    # GeoDataFrame from DataFrame
    gdf_points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
        crs="EPSG:4326"
    )

    # Upload shapefiles
    states = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # Join GeoDataFrame of points with states
    gdf_with_state = gpd.sjoin(gdf_points, states, how='left', predicate='within')

    # Count apartments per state
    state_counts = gdf_with_state['name'].value_counts().reset_index()
    state_counts.columns = ['name', 'count']
    states_with_counts = states.merge(state_counts, on='name', how='left')
    states_with_counts['count'] = states_with_counts['count'].fillna(0)

    # Colormap
    custom_color_map = LinearSegmentedColormap.from_list("custom_cmap", ['#1f77b4', '#ff7f0e'])

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    states_with_counts.plot(
        column='count',
        cmap=custom_color_map,
        linewidth=0.4,
        edgecolor='black',
        legend=True,
        ax=ax,
        cax=cax
    )

    ax.set_title('Number of Apartment Listings by US State', fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    plt.show()


    
def compare_distributions(df1, df2, numerical_features=None):
   
    if numerical_features is None:
        numerical_features = df1.select_dtypes(include='number').columns
    
    numerical_features = [col for col in numerical_features if col != 'id']

    num_features = len(numerical_features)
    num_cols = 2
    num_rows = (num_features + 1) // num_cols
    
    plt.figure(figsize=(12, 5*num_rows))
    for i, col in enumerate(numerical_features, 1):
        plt.subplot(num_rows, num_cols, i)
        df1[col].hist(bins=30, density=True, alpha=0.7, color='#1f77b4', label='Development')
        df2[col].hist(bins=30, density=True, alpha=0.7, color='#ff7f0e', label='Evaluation')
        plt.title(f'{col} Distribution')
        plt.legend()

        plt.xlabel(col)
        plt.ylabel('Proportion')

    plt.tight_layout()
    plt.show()


def compare_categorical_distributions(df1, df2, categorical_features=None, threshold=0.01):
    
    if categorical_features is None:
        categorical_features = df1.select_dtypes(include='object').columns
    
    num_features = len(categorical_features)
    num_cols = 2
    num_rows = (num_features + 1) // num_cols
    
    plt.figure(figsize=(12, 5*num_rows))
    for i, col in enumerate(categorical_features, 1):
        plt.subplot(num_rows, num_cols, i)
        
        # Frequencies calculation
        freq_df1 = df1[col].value_counts(normalize=True)
        freq_df2 = df2[col].value_counts(normalize=True)
        
        # Function to group rare categories into "Other" category
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


def correlation_matrix(df, target='price'):
    numerical_features = df.select_dtypes(include='number').columns.tolist()
    
    if target not in numerical_features and target in df.columns:
        numerical_features.append(target)
    
    correlation = df[numerical_features].corr(method='spearman')

    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ffffff', '#ff7f0e']
    cmap = LinearSegmentedColormap.from_list('blue_orange', colors, N=256)
    
    sns.heatmap(correlation, cmap=cmap, annot=True, fmt=".2f", center=0)
    plt.title(f"Correlation Matrix between numerical features, includind the '{target}' target variable")
    plt.tight_layout()
    plt.show()

def compare_scatter_outliers(df, evaluate_df):
    features = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude', 'time']

    for feature in features:
        if feature not in df.columns or feature not in evaluate_df.columns:
            continue

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Scatter comparison: {feature}", fontsize=14)

        # Left: development dataset with price
        axs[0].scatter(df[feature], df['price'], alpha=0.3, label="Development", color="blue")
        axs[0].set_title("Development")
        axs[0].set_xlabel(feature)
        axs[0].set_ylabel("Price")
        axs[0].grid(True)
        axs[0].tick_params(axis='x', labelrotation=45)

        # Right: evaluation dataset (feature only)
        axs[1].scatter(evaluate_df[feature], [0]*len(evaluate_df), alpha=0.3, label="Evaluation", color="orange")
        axs[1].set_title("Evaluation")
        axs[1].set_xlabel(feature)
        axs[1].set_yticks([])
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()


def plot_square_feet_comparison(df, evaluate_df):
    
    if 'square_feet' not in df.columns or 'square_feet' not in evaluate_df.columns:
        print("Missing 'square_feet' in one of the datasets.")
        return

    fig, axs = plt.subplots(2, 1, figsize=(5, 8))
    fig.suptitle("Visual Comparison of Square Footage Feature Across Datasets", fontsize=14)

    # Development: scatter vs price
    axs[0].scatter(df['square_feet'], df['price'], alpha=0.3, color='#1f77b4')
    axs[0].set_title("Development: square_feet vs price")
    axs[0].set_xlabel("square_feet")
    axs[0].set_ylabel("price")
    axs[0].grid(True)
    axs[0].tick_params(axis='x', labelrotation=45)  

    # Evaluation: distribution only
    axs[1].scatter(evaluate_df['square_feet'], [0]*len(evaluate_df), alpha=0.3, color='#ff7f0e')
    axs[1].set_title("Evaluation: square_feet distribution")
    axs[1].set_xlabel("square_feet")
    axs[1].set_yticks([])
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


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


def fill_missing_values(df):
    # Filling missing values for "pets_allowed" and "amenities"
    df["pets_allowed"] = df["pets_allowed"].fillna("no")
    df["amenities"] = df["amenities"].fillna("none")
    return df


def add_square_feet_quantile(df, sq_col='square_feet', n_quantiles=10, new_col='square_feet_quantile'):
    df = df.copy()
    # Assigning quantile labels (from 0 to n_quantiles-1)
    df[new_col] = pd.qcut(df[sq_col], q=n_quantiles, labels=False, duplicates='drop')
    return df


def impute_by_square_feet_quantile(df, quantiles, quantile_means, sq_col='square_feet', col_to_impute='bathrooms'):
    df = df.copy()
    # Assigning quantile index to each row
    quantile_idx = pd.cut(df[sq_col], bins=quantiles, labels=False, include_lowest=True)
    # Filling missing values with the mean of the corresponding quantile
    for i in range(len(quantile_means)):
        mask = (quantile_idx == i) & (df[col_to_impute].isnull())
        df.loc[mask, col_to_impute] = quantile_means[i]
    return df

# Initialize geocoder
geolocator = Nominatim(user_agent="ds-lab-project")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

def build_query_string(title, body):
    text = f"{title} {body}"

    # Looking for ZIP code
    zip_match = re.search(r'\b\d{5}\b', text)
    if zip_match:
        zip_code = zip_match.group(0)
        return f"{zip_code} USA"

    # Looking for cityname and state
    city_state_match = re.search(r'\b([A-Z][a-z]+)\s([A-Z]{2})\b', text)
    if city_state_match:
        return f"{city_state_match.group(1)}, {city_state_match.group(2)}"

    return None

def forward_geocode_from_text(title, body):
    query = build_query_string(title, body)
    if query:
        location = geocode(query)
        if location:
            return {
                'latitude': location.latitude,
                'longitude': location.longitude,
                'address': location.raw.get('display_name', ''),
                'city': location.raw.get('address', {}).get('city') or
                        location.raw.get('address', {}).get('town') or
                        location.raw.get('address', {}).get('village'),
                'state': location.raw.get('address', {}).get('state')
            }
    return None

def impute_geolocation_with_nominatim(df):
    df = df.copy()

    for idx, row in df[df[['latitude', 'longitude', 'cityname', 'state']].isna().all(axis=1)].iterrows():
        result = forward_geocode_from_text(row['title'], row['body'])
        if result:
            df.at[idx, 'latitude'] = result['latitude']
            df.at[idx, 'longitude'] = result['longitude']
            df.at[idx, 'cityname'] = result['city']
            df.at[idx, 'state'] = result['state']
            print(f"Imputato {idx}: {result['latitude']}, {result['longitude']}, {result['city']}, {result['state']}")
        else:
            print(f"Nessun risultato per {idx}")

    return df
