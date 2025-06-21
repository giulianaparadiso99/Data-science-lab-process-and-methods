# Apartment Rent Price Prediction

This project develops a regression model to predict the monthly rent price of apartments in the US, based on structured features and unstructured textual data.

Files

- `development.csv`: training data with the target variable `price`.
- `evaluation.csv`: test data without the target variable.
- `main.ipynb`: main notebook used for preprocessing, model training, hyperparameter tuning, and prediction.
- `predictions.csv`: submission file containing the predictions for the evaluation set.
- `preprocessing.py`: contains the preprocessing functions applied to both `development` and `evaluation` datasets.
- `utils.py`: utility functions used throughout the project. Includes helper functions for preprocessing and for creating visualizations.
- `report.pdf`: final report describing the methodology, preprocessing steps, models used, and results.


## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- xgboost
- catboost
- matplotlib
- seaborn

## How to Use the Code

The main logic of the project is contained in the `main.ipynb` notebook.

- The notebook is divided into sequential code cells.
- Each cell is preceded by a brief description explaining its purpose.
- To run the project, simply execute the cells in order from top to bottom. The workflow is linear and fairly intuitive.

No manual file editing or configuration is required. All necessary functions are imported from `preprocessing.py` and `utils.py`.
