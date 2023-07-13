import pandas as pd
import logging
import plotly.graph_objects as go


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_hyperparameter_tuning_results(results):
    """
    Plots the results of hyperparameter tuning.

    Args:
        results (dict): The results of hyperparameter tuning.

    Returns:
        None
    """
    logging.info("Plotting hyperparameter tuning results...")

    param_names = list(results['params'][0].keys())
    param_values = {param: [] for param in param_names}
    mean_scores = results['mean_test_score']

    for param_set in results['params']:
        for param in param_names:
            param_values[param].append(param_set[param])

    fig = go.Figure()

    for param in param_names:
        fig.add_trace(go.Scatter(x=param_values[param], y=mean_scores,
                                 mode='markers', name=param))

    fig.update_layout(
        title='Hyperparameter Tuning Results',
        xaxis_title='Hyperparameter Values',
        yaxis_title='Mean Score',
        legend_title='Hyperparameters'
    )

    fig.show()
    logging.info("Hyperparameter tuning results plotted.")


def plot_model_results(x, y_true, y_pred):
    """
    Plots the results of the model.

    Args:
        x (array-like): The input features.
        y_true (array-like): The true target values.
        y_pred (array-like): The predicted target values.

    Returns:
        None
    """
    logging.info("Plotting model results...")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y_true,
                             mode='markers', name='True'))

    fig.add_trace(go.Scatter(x=x, y=y_pred,
                             mode='lines', name='Predicted'))

    fig.update_layout(
        title='Model Results',
        xaxis_title='X',
        yaxis_title='Y',
        legend_title='Data'
    )

    fig.show()
    logging.info("Model results plotted.")


def display_results_table(results, columns):
    """
    Displays the results in a tabular format.

    Args:
        results (dict): The results to display.
        columns (list): The column names.

    Returns:
        None
    """
    logging.info("Displaying results table...")

    data = {col: results[col] for col in columns}
    df = pd.DataFrame(data)
    df = df.sort_values('Mean Score', ascending=False)
    df = df.reset_index(drop=True)
    print(df)

    logging.info("Results table displayed.")


def display_model_results_table(x, y_true, y_pred):
    """
    Displays the model results in a tabular format.

    Args:
        x (array-like): The input features.
        y_true (array-like): The true target values.
        y_pred (array-like): The predicted target values.

    Returns:
        None
    """
    logging.info("Displaying model results table...")

    df = pd.DataFrame({'X': x, 'True': y_true, 'Predicted': y_pred})
    print(df)

    logging.info("Model results table displayed.")
