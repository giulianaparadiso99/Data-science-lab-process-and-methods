# 🏠 Apartment Price Prediction Model

## Project Overview

This project develops and evaluates several **Machine Learning regression models** to accurately predict apartment rental prices based on a comprehensive set of features, including geographical data, property characteristics, amenities, and text-based features derived from listing descriptions.

The process includes rigorous **Exploratory Data Analysis (EDA)**, **feature engineering**, **data cleaning**, and a systematic **model comparison** to identify the best-performing algorithm and hyperparameters.

## 💾 Dataset

The project utilizes two main datasets:
1.  `development.csv`: The primary training and validation dataset, which includes the target variable (`price`).
2.  `evaluation.csv`: The held-out test dataset for generating final predictions.

## ⚙️ Dependencies

The following major libraries are required to run the notebook:

* **Python 3.x**
* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn` (for models, splitting, scaling, and error metrics)
* `xgboost`
* `catboost`
* `prettytable`
* `geopy` (for geographical data processing, as seen in the preprocessing file)

## 📊 Exploratory Data Analysis (EDA)

The notebook includes several key visualizations to understand the data:

* **Geographical Distribution (Choropleth Maps):** Visualizing apartment density across different US states and cities for both the development and evaluation datasets.
* **Feature Distributions:** Comparison of normalized frequency distributions for numerical features (e.g., `square_feet`, `bathrooms`, `bedrooms`) across both datasets to ensure consistency.
* **Categorical Distributions:** Comparison of distributions for categorical features (e.g., `currency`, `fee`, `has_photo`).
* **Correlation Matrix:** A heatmap to identify linear relationships between numerical features and the target variable (`price`).
* **Outlier Detection:** Scatter plots comparing `price` vs. `square_feet` to visualize potential outliers and assess their consistency between the datasets.

## 🧹 Data Preprocessing & Feature Engineering

A custom `preprocessing.py` module is used to handle data preparation, which includes:

1.  **Handling Missing Values:** Missing values in `bathrooms`, `bedrooms`, and geographical features (`cityname`, `state`, `latitude`, `longitude`) are imputed. Notably, attempts were made to **geolocate missing city/state data** by searching the `title` and `body` fields for potential location keywords.
2.  **Feature Extraction from Text:** New features like `max_from_body`, `min_from_body`, `std_from_body`, and `mean_from_body` are engineered by processing text fields (`title`, `body`).
3.  **Encoding Amenities:** The `amenities` column is processed to create **binary (one-hot encoded) features** for common amenities (e.g., `Pool`, `Gym`, `Washer Dryer`).
4.  **Standardization:** Numerical features are scaled using `StandardScaler` to prepare the data for distance-based models (like SVR and MLP).

## 🧠 Model Training and Selection

### 1. Initial Model Comparison

The performance of various models with default/initial hyperparameters is compared using the **Mean Absolute Error (MAE)** metric on non-standardized data:

* **Tree-based models:** Random Forest Regressor, Extra Trees Regressor, HistGradientBoosting Regressor, CatBoost Regressor, Gradient Boosting Regressor, and XGBoost Regressor.
* **Other models:** SVR (Support Vector Regressor) and MLP (Multi-layer Perceptron Regressor), both of which are run on *standardized data*.

### 2. Hyperparameter Tuning (Grid Search)

A **Grid Search** approach is used to systematically tune the hyperparameters for the top-performing tree-based models, aiming to minimize the MAE.

### 3. Final Evaluation

The model with the best-tuned hyperparameters (based on test set MAE) is selected as the final model. This model is then trained on the **full `development.csv`** dataset and used to generate predictions on the `evaluation.csv` dataset, saving the results to a `predictions.csv` file.

### 📈 Feature Importance

The project concludes with an analysis of **feature importances** from the best-performing tree-based model (e.g., Extra Trees Regressor), highlighting the most influential variables for apartment price prediction, such as `longitude`, `latitude`, `square_feet`, and `bathrooms`.
