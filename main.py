import numpy as np
from tqdm import tqdm
import random
import logging
import os
import tensorflow as tf

from data_preparation import *
from outputs import *
from models import *

####################
#### PARAMETERS ####
####################

RANDOM_SEED = 42  # Seed for reproducibility
BATCH_SIZE = 32   # Batch size for training
NUM_ROUNDS = 100  # Number of federated learning rounds
NUM_CLIENTS = 10  # Number of federated clients
NUM_SHADOW_MODELS = 20  # Number of shadow models for property inference
GLOBAL_MODEL_EPOCHS = 50  # Number of epochs for global model training
SHADOW_TRAIN_ROUNDS = 10  # Rounds for shadow model training
PROPERTY_THRESHOLD = 0.5  # Threshold for property inference
DATA_FILE_PATH = 'label_bi.csv'  # Path to the dataset
RESULTS_CSV_FILE = 'results/federated_learning_results.csv'  # File to save results
SHADOW_WEIGHTS_CSV_FILE = 'weights/shadow_model_weights.csv'  # File to save shadow model weights

####################

# Logging configuration
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'execution.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
####################

# GPU configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU device

def simulated_federated_learning(clients, shadow_models, global_model, num_rounds=NUM_ROUNDS, threshold=PROPERTY_THRESHOLD):
    """
    Simulate federated learning with multiple clients and infer a property using shadow models.

    This function simulates a federated learning process over multiple rounds. In each round, clients are 
    selected, their updates are aggregated, and property inference is performed using shadow models. 
    Results such as loss, accuracy, and property probabilities are logged and saved.

    Args:
        clients (list): List of clients participating in federated learning.
        shadow_models (list): List of trained shadow models for property inference.
        global_model (Model): The global model to be updated in each round.
        num_rounds (int): Number of federated learning rounds (default: 100).
        threshold (float): Threshold to determine if a property is present based on the inferred probability (default: 0.5).

    Returns:
        results (list): A list of dictionaries containing results for each round.
    """
    logger.info(f"Starting simulated federated learning with {num_rounds} rounds")

    results = []  # List to store results for each round

    # Separate clients based on whether they have the property
    clients_with_property = [client for client in clients if client.data['has_property']]
    clients_without_property = [client for client in clients if not client.data['has_property']]

    for round in tqdm(range(1, num_rounds + 1), desc="Federated Learning Progress"):
        logger.info(f"Starting round {round}")
        
        # Randomly decide whether to use clients with or without the property in this round
        use_property_clients = random.choice([True, False])
        selected_clients = clients_with_property if use_property_clients else clients_without_property

        # Fetch current global model weights
        global_weights = global_model.get_weights()

        client_weights = []
        client_updates = []
        
        # Collect client updates
        for client in selected_clients:
            w, _, update_info = client.fit(global_weights)  # Fit client model and get updates
            client_weights.append(w)
            client_updates.append(update_info['updates'])

        # Average client weights and set them as the global model's new weights
        averaged_weights = [np.mean(layer, axis=0) for layer in zip(*client_weights)]
        global_model.set_weights(averaged_weights)

        # Save averaged weights to a CSV file for shadow model training
        save_weights_to_csv(averaged_weights, round, SHADOW_WEIGHTS_CSV_FILE)

        # Average the client updates
        averaged_updates = np.mean(client_updates, axis=0)

        # Infer the presence of the property using shadow models
        property_prob = infer_property(shadow_models, averaged_updates)

        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        # Evaluate the global model on all clients
        for client in clients:
            loss, num_samples, metrics = client.evaluate(global_model.get_weights())
            total_loss += loss * num_samples
            total_accuracy += metrics['accuracy'] * num_samples
            total_samples += num_samples

        # Compute average loss and accuracy for the round
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples

        # Store results for the current round
        results.append({
            'round': round,
            'property_probability': property_prob,
            'has_property': use_property_clients,
            'prediction': property_prob > threshold,
            'clients_with_property': 5 if use_property_clients else 0,
            'clients_without_property': 0 if use_property_clients else 5,
            'loss': avg_loss,
            'accuracy': avg_accuracy
        })

        # Log the results of the round
        logger.info(f"Round {round} completed. Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}")
        logger.info(f"  Property probability: {property_prob:.4f}")
        logger.info(f"  The round has property: {use_property_clients}")

    logger.info("Federated learning simulation completed")
    return results


def main():
    """
    Main function to run the federated learning experiment. This function:
    1. Loads and preprocesses the dataset.
    2. Initializes clients and a global model.
    3. Trains shadow models for property inference.
    4. Runs the federated learning simulation.
    5. Logs and saves experiment results, including plots and metrics.
    """
    logger.info("Starting main experiment process")

    # Step 1: Load and preprocess the data
    logger.info(f"Loading data from {DATA_FILE_PATH}")
    X, y_label, y_slice = load_and_preprocess_data(DATA_FILE_PATH)

    # Step 2: Create data for the clients
    logger.info(f"Creating data for {NUM_CLIENTS} clients")
    client_data = create_client_data(X, y_label, y_slice, num_clients=NUM_CLIENTS)

    # Step 3: Create and initialize the global model
    logger.info("Creating global model for reuse")
    global_model = create_global_model(client_data[0]['X'].shape[1])

    # Step 4: Initialize clients with the global model's weights
    logger.info("Initializing simulated clients with reused global model weights")
    clients = []
    for i, data in enumerate(client_data):
        client = SimulatedFlowerClient(i, data, GLOBAL_MODEL_EPOCHS, BATCH_SIZE)
        client.model.set_weights(global_model.get_weights())  # Reuse the global model's weights
        clients.append(client)

    # Step 5: Train shadow models for property inference
    logger.info(f"Training {NUM_SHADOW_MODELS} shadow models")
    shadow_models, shadow_data_size = train_shadow_models(client_data, NUM_SHADOW_MODELS, SHADOW_TRAIN_ROUNDS, GLOBAL_MODEL_EPOCHS, BATCH_SIZE)

    # Step 6: Run the federated learning simulation
    logger.info(f"Starting federated learning simulation with {NUM_ROUNDS} rounds")
    results = simulated_federated_learning(clients, shadow_models, global_model, num_rounds=NUM_ROUNDS, threshold=PROPERTY_THRESHOLD)

    # Step 7: Plot and save combined results (Loss, Accuracy, Probability Evolution)
    logger.info("Plotting and saving combined results (Loss, Accuracy, Probability Evolution)")
    plot_results(results)

    # Step 8: Calculate shadow model metrics (ROC, Precision, etc.)
    logger.info("Calculating shadow model metrics")
    auc_roc, cm, precision, optimal_threshold = calculate_shadow_model_metrics(results)

    # Step 9: Calculate custom precision for shadow model predictions
    correct_predictions, incorrect_predictions = calculate_correct_predictions(results)
    total_predictions = correct_predictions + incorrect_predictions
    custom_precision = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    logger.info(f"Custom Precision (Shadow Model): {custom_precision:.2f}%")

    # Step 10: Save final experiment results to CSV
    save_experiment_results_to_csv(
        dataset_size=len(X),
        batch_size=BATCH_SIZE,
        num_shadow_models=NUM_SHADOW_MODELS,
        global_model_epochs=GLOBAL_MODEL_EPOCHS,
        shadow_train_rounds=SHADOW_TRAIN_ROUNDS,
        auc_roc=auc_roc,
        optimal_threshold=optimal_threshold,
        precision=precision,
        custom_precision=f"{custom_precision:.2f}",
        tp=cm[1, 1],
        fp=cm[0, 1],
        tn=cm[0, 0],
        fn=cm[1, 0],
        filename='results/final_experiment_results.csv'
    )

    # Step 11: Plot and save ROC-AUC and Threshold plots
    logger.info("Plotting and saving ROC-AUC and Optimal Threshold")
    fpr, tpr, thresholds = roc_curve([r['has_property'] for r in results], [r['property_probability'] for r in results])
    plot_results_roc_and_threshold(fpr, tpr, thresholds, auc_roc, optimal_threshold)

    # Step 12: Save all detailed results to CSV
    logger.info(f"Saving detailed results to {RESULTS_CSV_FILE}")
    save_results_to_csv_with_precision(results, RESULTS_CSV_FILE, custom_precision)
    logger.info(f"Results successfully saved to {RESULTS_CSV_FILE}")

    # Step 13: Print and log experiment characteristics
    logger.info("Printing experiment characteristics")
    print_experiment_characteristics(
        X, client_data, shadow_models, shadow_data_size,
        metrics={'roc_auc': auc_roc, 'optimal_threshold': optimal_threshold, 'accuracy': precision},
        num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS, global_model_epochs=GLOBAL_MODEL_EPOCHS,
        batch_size=BATCH_SIZE, num_shadow_models=NUM_SHADOW_MODELS,
        property_threshold=PROPERTY_THRESHOLD, shadow_train_rounds=SHADOW_TRAIN_ROUNDS, random_seed=RANDOM_SEED
    )

    logger.info("Main experiment process completed")


if __name__ == "__main__":
    main()