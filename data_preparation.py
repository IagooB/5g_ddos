import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import logging
import os

# Global constants
NUM_CLIENTS = 10  # Total number of clients in federated learning
FRACTION = 0.50   # Fraction of data to be used for training

# Logging configuration
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'execution.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Path to the dataset
file_path = "5g_ddos.csv"

def load_and_preprocess_data(file_path):
    """
    Load and preprocess data from a CSV file.

    This function loads the dataset, selects the required columns, 
    performs random sampling to reduce the dataset size, and handles missing data.
    
    Args:
        file_path (str): Path to the CSV file.

    Returns:
        X (DataFrame): Features of the dataset.
        y_label (Series): Labels for the data points.
        y_slice (Series): Property labels (indicating presence of a specific feature).

    Raises:
        AssertionError: If the dataset does not contain enough data after preprocessing.
    """
    logger.info(f"Loading and preprocessing data from {file_path}")
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Select necessary columns for the experiment
    necessary_columns = [
        'Src IP', 'Src Port', 'Dst Port', 'Protocol', 'Flow Duration', 'Total Fwd Packet',
        'Fwd Packet Length Std', 'ACK Flag Count', 'Fwd Seg Size Min', 'label', 'Slice',
    ]
    df = df[necessary_columns]

    # Randomly sample the data based on the FRACTION constant and drop missing values
    df = df.sample(frac=FRACTION)
    df = df.dropna()

    # Separate features (X) from the labels (y_label) and the property indicator (y_slice)
    X = df.drop(['label', 'Slice'], axis=1)
    y_label = df['label']
    y_slice = df['Slice']
    
    # Ensure that the dataset contains enough data for further processing
    assert len(X) > 0, "The dataset does not contain enough data after preprocessing."

    return X, y_label, y_slice


def create_client_data(X, y_label, y_slice, num_clients=NUM_CLIENTS):
    """
    Split the dataset into subsets for multiple clients in federated learning.

    This function divides the dataset into balanced subsets of clients, where half of the clients 
    have a specific property (based on y_slice) and the other half do not. 
    
    Args:
        X (np.array): Feature matrix.
        y_label (np.array): Label array.
        y_slice (np.array): Property indicator array.
        num_clients (int): Number of clients to create (default is 10).
        
    Returns:
        client_data (list): A list of dictionaries where each dictionary contains data for one client.

    Raises:
        AssertionError: If the dataset does not contain enough data with or without the specified property.
    """
    logger.info(f"Creating data for {num_clients} clients")

    # Convert data to NumPy arrays for better performance
    X = np.array(X)
    y_label = np.array(y_label)
    y_slice = np.array(y_slice)

    # Create masks for data that has the property (y_slice == 1) and that doesn't
    mask_with_property = y_slice == 1
    X_with_property = X[mask_with_property]
    y_label_with_property = y_label[mask_with_property]

    X_without_property = X[~mask_with_property]
    y_label_without_property = y_label[~mask_with_property]

    # Calculate the minimum data size to ensure balanced distribution among clients
    min_data_size_with_property = len(X_with_property) // (num_clients // 2)
    min_data_size_without_property = len(X_without_property) // (num_clients // 2)

    min_data_size = min(min_data_size_with_property, min_data_size_without_property)

    # Ensure that there are enough data points with and without the property
    assert len(X_with_property) > 0, "Not enough data with the property."
    assert len(X_without_property) > 0, "Not enough data without the property."

    client_data = []

    # Create half the clients with the property and half without
    for i in range(num_clients // 2):
        # Assign data to a client with the property
        idx = np.random.choice(len(X_with_property), size=min_data_size, replace=False)
        client_data.append({
            'X': X_with_property[idx],
            'y_label': y_label_with_property[idx],
            'y_slice': np.ones(len(idx)),
            'has_property': True
        })
        # Remove the assigned data to avoid duplication
        X_with_property = np.delete(X_with_property, idx, axis=0)
        y_label_with_property = np.delete(y_label_with_property, idx)

        # Assign data to a client without the property
        idx = np.random.choice(len(X_without_property), size=min_data_size, replace=False)
        client_data.append({
            'X': X_without_property[idx],
            'y_label': y_label_without_property[idx],
            'y_slice': np.zeros(len(idx)),
            'has_property': False
        })
        # Remove the assigned data to avoid duplication
        X_without_property = np.delete(X_without_property, idx, axis=0)
        y_label_without_property = np.delete(y_label_without_property, idx)

    return client_data
