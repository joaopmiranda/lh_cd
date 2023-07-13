import logging
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def hyperparameter_tuning(model, x, y, param_grid, cv=5, scoring='r2'):
    """
    Performs hyperparameter tuning using GridSearchCV.

    Args:
        model: The model to be tuned.
        x (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        param_grid (dict): The dictionary of hyperparameters and their possible values.
        cv (int): The number of cross-validation folds.
        scoring (str): The scoring metric to evaluate.

    Returns:
        GridSearchCV: The tuned model with best hyperparameters.
    """
    logging.info("Performing hyperparameter tuning...")

    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(x, y)

    logging.info("Hyperparameter tuning complete.")
    return grid_search


# Usage example:
# tuned_model = hyperparameter_tuning(model, x_train, y_train, param_grid, cv=5, scoring='r2')
# print(tuned_model.best_params_)


def cross_validation_report(model, x, y, cv=5, scoring='r2'):
    """
    Performs cross-validation and generates a report.

    Args:
        model: The trained model.
        x (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        cv (int): The number of cross-validation folds.
        scoring (str): The scoring metric to evaluate.

    Returns:
        dict: The cross-validation scores.
    """
    logging.info("Performing cross-validation...")

    scores = cross_val_score(model, x, y, cv=cv, scoring=scoring)

    report = {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores
    }

    logging.info("Cross-validation complete.")
    return report


def train_model(x, y, **kwargs):
    """
    Trains an XGBoostRegressor model.

    Args:
        x (pd.DataFrame): The input features.
        y (pd.Series): The target variable.

    Returns:
        XGBRegressor: The trained XGBoost model.
    """
    logging.info("Training the model...")
    model = XGBRegressor(**kwargs)
    model.fit(x, y)
    logging.info("Training complete.")
    return model

def evaluate_model(model, x_test, y_test):
    """
    Evaluates the trained model.

    Args:
        model (XGBRegressor): The trained XGBoost model.
        x_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.

    Returns:
        float: The R² score of the model predictions.
    """
    logging.info("Evaluating the model...")
    predictions = model.predict(x_test)
    score = r2_score(y_test, predictions)
    logging.info("Evaluation complete.")
    return score
