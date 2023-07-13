import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(file_path, *args, **kwargs):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path of the CSV file.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        pandas.DataFrame: Loaded data from the CSV file.
    """
    try:
        logger.info(f"Loading data from file: {file_path}")
        data = pd.read_csv(file_path, *args, **kwargs)
        logger.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logger.error("File not found.")
        raise
