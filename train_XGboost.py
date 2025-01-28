import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


def load_data(file_path):
    """
    Load the dataset from an Excel file.
    
    Parameters:
        file_path (str): Path to the Excel file containing the dataset.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_excel(file_path)


def split_data(df, target_column):
    """
    Split the dataset into features (X) and target (y) and perform train-test split.
    
    Parameters:
        df (pd.DataFrame): The dataset as a Pandas DataFrame.
        target_column (str): Name of the target column.
    
    Returns:
        X_train, X_test, y_train, y_test: Split data.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_xgboost(X_train, y_train):
    """
    Train an XGBoost regression model.
    
    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target values.
    
    Returns:
        model: Trained XGBoost model.
    """
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_test, y_pred):
    """
    Evaluate the model using Mean Squared Error (MSE) and R-squared (R2).
    
    Parameters:
        y_test (array-like): True target values.
        y_pred (array-like): Predicted target values.
    
    Returns:
        mse (float): Mean Squared Error.
        r2 (float): R-squared value.
    """
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def plot_results(y_test, y_pred):
    """
    Plot true values vs predicted values.
    
    Parameters:
        y_test (array-like): True target values.
        y_pred (array-like): Predicted target values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Equality Line')
    plt.title('True Values vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Specify the path to the dataset (replace with your dataset's path)
    file_path = "data/GWSGWLPREBEST.xlsx"  # Dummy path for the dataset
    target_column = "GWL"  # Update with the actual target column name
    
    # Load the data
    print("Loading data...")
    df = load_data(file_path)
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    
    # Train the model
    print("Training the model...")
    model = train_xgboost(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print("Evaluating the model...")
    mse, r2 = evaluate_model(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")
    
    # Plot results
    print("Plotting results...")
    plot_results(y_test, y_pred)
    print("Done!")


if __name__ == "__main__":
    main()
