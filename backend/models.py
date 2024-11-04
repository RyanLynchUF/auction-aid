from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import settings

CURR_LEAGUE_YR = settings.CURR_LEAGUE_YR

def train_model(input_features:pd.DataFrame, mode:str='testing', model_type:str='random_forest'):
    """
    
    mode: testing, pre-draft, drafting
    """

    input_features = input_features.drop(labels=['pos'], axis=1)
    # Feature Selections
    # Determine which features would introduce multi-collinearity to the models
    multicollinearity_results = detect_multicollinearity(input_features.loc[:, input_features.columns != 'curr_year_bid_amt'], 
                                    correlation_threshold=0.8,  # Adjust as needed
                                    vif_threshold=10.0)         # Adjust as needed

    
    if mode == 'testing':
        # Print the results in a readable format
        print_multicollinearity_analysis(multicollinearity_results)

    # Remove columns with high multi-collinearity
    multicollinearity_columns_to_remove = multicollinearity_results[2]
    input_features = input_features[[col for col in input_features if col not in multicollinearity_columns_to_remove]]
        
    trained_model = tune_model(input_features, model_type=model_type, mode=mode)
    if mode == 'testing':
        print_model_results(trained_model)

    return trained_model, multicollinearity_columns_to_remove


def tune_model(features:pd.DataFrame, model_type:str, 
                        test_size:float=0.2, random_state:int=42, mode:str='testing'):
    """
    Train a Random Forest model to predict auction values in fantasy football.
    
    Parameters:
    -----------
    features : pandas.DataFrame
        DataFrame containing fantasy football features and target variable
    test_size : float
        Proportion of dataset to include in the test split
    random_state : int
        Random state for reproducibility
    """
    # Separate features and target
    X = features.drop('curr_year_bid_amt', axis=1)
    y = features['curr_year_bid_amt']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    if model_type == 'random_forest':
        # Define hyperparameter search space
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        # Initialize base model
        base_model = RandomForestRegressor(random_state=random_state)
        
        # Initialize RandomizedSearchCV
        model = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=25,
            cv=5,
            verbose=1,
            random_state=random_state,
            n_jobs=-1,
            scoring='neg_mean_squared_error'
        )

        print("Training Random Forest model...")
    else:

        model = LinearRegression()
        print("Training Linear Regression model...")
        
    # Fit RandomizedSearchCV
    print("Training model with hyperparameter tuning...")
    model.fit(X_train, y_train)
    
    # Get best model
    best_model = model.best_estimator_ if model_type == 'random_forest' else model
    
    # Make predictions
    train_predictions = best_model.predict(X_train)
    test_predictions = best_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_r2': r2_score(y_train, train_predictions),
        'test_r2': r2_score(y_test, test_predictions),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_predictions)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_predictions)),
        'train_mae': mean_absolute_error(y_train, train_predictions),
        'test_mae': mean_absolute_error(y_test, test_predictions)
    }
    
    if model_type == 'random_forest':
        importance_values = best_model.feature_importances_
        importance_name = 'importance'
    else:
        importance_values = best_model.coef_
        importance_name = 'coefficient'

    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        importance_name: importance_values
    }).sort_values(importance_name, ascending=False)
    
    
    return {
        'model_type': model_type,
        'model': best_model,
        'best_params': model.best_params_ if model_type == 'random_forest' else None,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'predictions': {
            'train': train_predictions,
            'test': test_predictions
        },
        'actual_values': {
            'train': y_train,
            'test': y_test
        },
    }


def print_model_results(results):
    """
    Print the results of the fantasy football random forest model.
    """
    print("\nModel Performance Metrics:")
    print("-" * 50)
    print(f"Training R² Score: {results['metrics']['train_r2']:.4f}")
    print(f"Testing R² Score: {results['metrics']['test_r2']:.4f}")
    print(f"Training RMSE: ${results['metrics']['train_rmse']:.2f}")
    print(f"Testing RMSE: ${results['metrics']['test_rmse']:.2f}")
    print(f"Training MAE: ${results['metrics']['train_mae']:.2f}")
    print(f"Testing MAE: ${results['metrics']['test_mae']:.2f}")
    
    if results['best_params']:
        print("\nBest Hyperparameters:")
        print("-" * 50)
        for param, value in results['best_params'].items():
            print(f"{param}: {value}")
    
    importance_name = 'importance' if results['model_type'] == 'random_forest' else 'coefficient'
    print(f"\nTop 10 Most Important Features ({importance_name}):")
    print("-" * 50)
    print(results['feature_importance'].head(10))


def predict_auction_value(model_results, prediction_features):
    """
    Make predictions for new players using the trained model.
    
    Parameters:
    -----------
    model_results : dict
        Results dictionary from train_fantasy_rf_model
    new_player_data : pandas.DataFrame
        DataFrame containing new player data to predict
    """

    # Create a copy to avoid modifying the original DataFrame
    df = prediction_features.copy()
    df = df.drop(labels=['pos'], axis=1)
    
    # Make prediction and add as new column
    df['expected_auction_value'] = model_results['model'].predict(df)
    
    return df


def detect_multicollinearity(df: pd.DataFrame, 
                           correlation_threshold: float = 0.8,
                           vif_threshold: float = 5.0) -> Tuple[Dict, pd.DataFrame, List[str]]:
    """
    Detect multicollinearity in a dataframe using correlation analysis and VIF.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numerical columns
    correlation_threshold : float, optional (default=0.8)
        Threshold for correlation coefficient to flag high correlation
    vif_threshold : float, optional (default=5.0)
        Threshold for VIF to flag high multicollinearity
    
    Returns:
    --------
    Tuple containing:
    - Dictionary of highly correlated pairs with their correlation coefficients
    - DataFrame with VIF scores for each variable
    - List of columns recommended for removal
    """
    # Convert to numeric, dropping non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Step 1: Correlation Analysis
    correlation_matrix = numeric_df.corr()
    
    # Find highly correlated pairs
    high_correlation_pairs = {}
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i,j]) >= correlation_threshold:
                col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                high_correlation_pairs[(col1, col2)] = correlation_matrix.iloc[i,j]
    
    # Step 2: VIF Analysis
    vif_data = pd.DataFrame()
    vif_data["Variable"] = numeric_df.columns
    vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) 
                       for i in range(numeric_df.shape[1])]
    
    # Step 3: Identify problematic columns
    problematic_columns = set()
    
    # Add columns with high VIF
    high_vif_columns = vif_data[vif_data["VIF"] > vif_threshold]["Variable"].tolist()
    problematic_columns.update(high_vif_columns)
    
    # Add columns from highly correlated pairs
    for (col1, col2), corr_value in high_correlation_pairs.items():
        # For each pair, suggest removing the column with higher VIF
        vif1 = vif_data[vif_data["Variable"] == col1]["VIF"].values[0]
        vif2 = vif_data[vif_data["Variable"] == col2]["VIF"].values[0]
        problematic_columns.add(col1 if vif1 > vif2 else col2)
    
    return (high_correlation_pairs, 
            vif_data.sort_values("VIF", ascending=False), 
            list(problematic_columns))

# Function to print the results in a readable format
def print_multicollinearity_analysis(results: Tuple[Dict, pd.DataFrame, List[str]]) -> None:
    """
    Print the multicollinearity analysis results in a readable format.
    
    Parameters:
    -----------
    results : Tuple
        Results from detect_multicollinearity function
    """
    high_correlations, vif_data, problem_columns = results
    
    print("1. Highly Correlated Pairs:")
    print("-" * 50)
    if high_correlations:
        for (col1, col2), corr in high_correlations.items():
            print(f"{col1} -- {col2}: {corr:.3f}")
    else:
        print("No highly correlated pairs found")
    
    print("\n2. VIF Analysis:")
    print("-" * 50)
    print(vif_data.to_string(index=False))
    
    print("\n3. Recommended Columns to Remove:")
    print("-" * 50)
    if problem_columns:
        for col in problem_columns:
            print(f"- {col}")
    else:
        print("No columns recommended for removal")

"""
Not Used - Alternative approach which creates a model for each position group.

def train_model_by_position(data: pd.DataFrame, model_type: str = 'random_forest'):
    Train and evaluate separate random forest models for each position.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the input features and target variable
    model_type (str): Type of model to train, either 'random_forest' or 'linear_regression'
    
    Returns:
    dict: A dictionary containing the model performance metrics for each position
  
    # Get the unique positions
    positions = ['WR', 'RB', 'QB', 'K', 'TE', 'DST']

    model_performance = {}
    for pos in positions:
        # Filter the data for the current position
        X_pos = data.drop(['curr_year_bid_amt', 'pos'], axis=1)[data['pos'] == pos]
        y_pos = data['curr_year_bid_amt'][data['pos'] == pos]

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_pos, y_pos, test_size=0.2, random_state=42)

        # Train the model
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

        model_performance[pos] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'trained_model': model
        }

     # Calculate the overall performance metrics
    all_train_r2 = sum(metrics['train_r2'] for metrics in model_performance.values()) / len(model_performance)
    all_test_r2 = sum(metrics['test_r2'] for metrics in model_performance.values()) / len(model_performance)
    all_train_rmse = np.sqrt(sum(metrics['train_rmse']**2 for metrics in model_performance.values()) / len(model_performance))
    all_test_rmse = np.sqrt(sum(metrics['test_rmse']**2 for metrics in model_performance.values()) / len(model_performance))

    overall_performance = {
        'all_train_r2': all_train_r2,
        'all_test_r2': all_test_r2,
        'all_train_rmse': all_train_rmse,
        'all_test_rmse': all_test_rmse
    }

    # Print the summary of model performance
    print("Training - Position Model Performance Summary:")
    for pos, metrics in model_performance.items():
        print(f"Position: {pos}")
        print(f"  Train R^2: {metrics['train_r2']:.2f}")
        print(f"  Test R^2: {metrics['test_r2']:.2f}")
        print(f"  Train RMSE: {metrics['train_rmse']:.2f}")
        print(f"  Test RMSE: {metrics['test_rmse']:.2f}")


    print("\nTraining - Position Overall Performance:")
    print(f"  Train R^2: {overall_performance['all_train_r2']:.2f}")
    print(f"  Test R^2: {overall_performance['all_test_r2']:.2f}")
    print(f"  Train RMSE: {overall_performance['all_train_rmse']:.2f}")
    print(f"  Test RMSE: {overall_performance['all_test_rmse']:.2f}")

    return model_performance, overall_performance


def predict_by_position(prediction_features: pd.DataFrame, models: dict):

    Use the trained models to predict values on a new dataset and assess the performance.
    
    Parameters:
    prediction_features (pd.DataFrame): DataFrame containing the input features and target variable for the new dataset
    models (dict): A dictionary of trained models, with positions as keys and models as values
    
    Returns:
    dict: A dictionary containing the prediction performance metrics for the new dataset
    
    # Get the unique positions
    positions = ['WR', 'RB', 'QB', 'K', 'TE', 'DST']

    # Create a copy to avoid modifying the original DataFrame
    df = prediction_features.copy()
    
    predictions_list_by_position = []
    for pos in positions:
        # Filter the data for the current position
        X_pos = df.drop(['pos'], axis=1)[df['pos'] == pos]
        
        # Make predictions using the trained model
        model = models[pos]['trained_model']
        predictions = model.predict(X_pos)
        X_pos['predicted_bid_amt'] = predictions
        predictions_list_by_position.append(X_pos)

    all_predictions = pd.concat(predictions_list_by_position)
        
    return all_predictions
    
    """