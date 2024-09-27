import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import csv
import logging
import pandas as pd
import os
import numpy as np


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


# Initialize logger
logger = logging.getLogger(__name__)

def plot_results(results):
    """
    Plots the results of federated learning, including property probability, loss, and accuracy per round.
    
    Args:
        results (list): List of dictionaries containing results for each round.
    """
    rounds = [r['round'] for r in results]
    probabilities = [r['property_probability'] for r in results]
    property_present = [r['has_property'] for r in results]
    losses = [r['loss'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Property probability plot
    ax1.plot(rounds, probabilities, marker='o', label='Property Probability')
    for i, (round_, prob, present) in enumerate(zip(rounds, probabilities, property_present)):
        if present:
            ax1.scatter(round_, prob, color='red', zorder=5, label='Rounds with Property' if i == 0 else "")
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Property Probability')
    ax1.set_title('Property Probability by Round')
    ax1.grid(True)
    ax1.legend()

    # Loss and Accuracy plot
    ax2.plot(rounds, losses, label='Loss', color='red')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax3 = ax2.twinx()  # Create a second y-axis for accuracy
    ax3.plot(rounds, accuracies, label='Accuracy', color='blue')
    ax3.set_ylabel('Accuracy', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')

    ax2.set_title('Loss and Accuracy by Round')
    fig.tight_layout()
    plt.show()

    logger.info("Plots generated and displayed")

def calculate_metrics(results):
    """
    Calculates key performance metrics, such as AUC-ROC and confusion matrix.
    
    Args:
        results (list): List of results per round.
    
    Returns:
        tuple: AUC-ROC score and confusion matrix.
    """
    y_true = [r['has_property'] for r in results]
    y_pred = [r['prediction'] for r in results]
    y_prob = [r['property_probability'] for r in results]

    auc_roc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    return auc_roc, cm

def print_metrics(auc_roc, cm):
    """
    Prints model performance metrics such as AUC-ROC and confusion matrix.
    
    Args:
        auc_roc (float): Area under the ROC curve.
        cm (np.array): Confusion matrix.
    """
    print(f"AUC ROC: {auc_roc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nTrue Negative:", cm[0][0])
    print("False Positive:", cm[0][1])
    print("False Negative:", cm[1][0])
    print("True Positive:", cm[1][1])

def save_results_to_csv(results, filename):
    """
    Saves the results of each round to a CSV file.
    
    Args:
        results (list): List of results per round.
        filename (str): Name of the output CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Round', 'Prediction', 'Probability', 'Clients with Property', 'Clients without Property', 'Has Property', 'Loss', 'Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for r in results:
            writer.writerow({
                'Round': r['round'],
                'Prediction': r['prediction'],
                'Probability': r['property_probability'],
                'Clients with Property': r['clients_with_property'],
                'Clients without Property': r['clients_without_property'],
                'Has Property': r['has_property'],
                'Loss': r['loss'],
                'Accuracy': r['accuracy']
            })

    logger.info(f"Results saved to {filename}")

def save_weights_to_csv(weights, round_num, filename='shadow_model_weights.csv'):
    """
    Saves the model weights passed to shadow models during each round to a CSV file.
    
    Args:
        weights (list): List of weight arrays from the global model.
        round_num (int): Current round number.
        filename (str): Name of the CSV file to save the weights.
    """
    flattened_weights = np.concatenate([w.flatten() for w in weights])
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if round_num == 1:  # Write header only for the first round
            writer.writerow(['Round'] + [f'Weight_{i}' for i in range(len(flattened_weights))])
        writer.writerow([round_num] + flattened_weights.tolist())

    logger.info(f"Weights for round {round_num} saved to {filename}")

def print_experiment_characteristics(X, client_data, shadow_models, shadow_data_size, metrics, 
                                     NUM_CLIENTS, NUM_ROUNDS, GLOBAL_MODEL_EPOCHS, 
                                     BATCH_SIZE, PROPERTY_THRESHOLD, SHADOW_TRAIN_ROUNDS, RANDOM_SEED):
    """
    Prints key characteristics of the experiment, such as dataset size, client details, and performance metrics.
    
    Args:
        X (np.array): The original dataset.
        client_data (list): List of client datasets.
        shadow_models (list): List of trained shadow models.
        shadow_data_size (int): Size of the data used to train shadow models.
        metrics (dict): Dictionary of performance metrics (ROC-AUC, threshold, accuracy).
        NUM_CLIENTS (int): Number of clients in the experiment.
        NUM_ROUNDS (int): Number of federated learning rounds.
        GLOBAL_MODEL_EPOCHS (int): Number of epochs for global model training.
        BATCH_SIZE (int): Batch size used for training.
        PROPERTY_THRESHOLD (float): Threshold used to infer property presence.
        SHADOW_TRAIN_ROUNDS (int): Number of training rounds for shadow models.
        RANDOM_SEED (int): Random seed used for reproducibility.
    """
    print("\n--- Experiment Characteristics ---")
    print(f"Total dataset size: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    
    print(f"Shadow models dataset size: {shadow_data_size}")
    print(f"Shadow model architecture: Same as global model")
    
    client_data_sizes = [len(client['X']) for client in client_data]
    print(f"Number of clients: {NUM_CLIENTS}")
    print(f"Average client dataset size: {np.mean(client_data_sizes):.2f}")
    print(f"Min client dataset size: {np.min(client_data_sizes)}")
    print(f"Max client dataset size: {np.max(client_data_sizes)}")
    
    print(f"Number of rounds: {NUM_ROUNDS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Number of shadow models: {len(shadow_models)}")
    print(f"Global model epochs: {GLOBAL_MODEL_EPOCHS}")
    print(f"Shadow train rounds: {SHADOW_TRAIN_ROUNDS}")
    print(f"Property threshold: {PROPERTY_THRESHOLD}")

    print("\nClient data shapes:")
    for i, client in enumerate(client_data):
        print(f"  Client {i}: {client['X'].shape}")
    
    print("\nPerformance Metrics:")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"Accuracy (based on probability changes): {metrics['accuracy']:.4f}")

    print("\n--- FINAL ---")


def analyze_results(results_file):
    """
    Analyzes the results from the federated learning experiment by calculating ROC-AUC, optimal thresholds, 
    and accuracy based on changes in prediction probabilities.
    
    Args:
        results_file (str): Path to the CSV file containing the results.
    
    Returns:
        dict: Dictionary containing calculated metrics (ROC-AUC, optimal threshold, accuracy).
    """
    # Try different encodings to handle potential issues with CSV encoding
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            # Load the results CSV file
            df = pd.read_csv(results_file, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Unable to read the CSV file with any of the attempted encodings: {encodings}")

    # Rename columns if necessary due to encoding issues
    df = df.rename(columns={
        'Predicción': 'Prediction',
        'Pérdida': 'Loss',
        'Precisión': 'Accuracy'
    })

    # Calculate ROC-AUC
    fpr, tpr, thresholds = roc_curve(df['Has Property'], df['Probability'])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    # Calculate the optimal threshold based on ROC curve
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot threshold vs. TPR and FPR
    plt.figure()
    plt.plot(thresholds, tpr, label='True Positive Rate')
    plt.plot(thresholds, fpr, label='False Positive Rate')
    plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('TPR and FPR vs. Threshold')
    plt.legend()
    plt.savefig('threshold_plot.png')
    plt.close()

    # Calculate accuracy based on changes in probability
    correct_predictions = 0
    total_predictions = len(df) - 1  # Cannot calculate for the first round
    
    for i in range(1, len(df)):
        prev_prob = df.loc[i-1, 'Probability']
        curr_prob = df.loc[i, 'Probability']
        prev_property = df.loc[i-1, 'Has Property']
        curr_property = df.loc[i, 'Has Property']
        
        # Correct if the probability increased when property is present, or decreased when it is absent
        if (curr_property and curr_prob > prev_prob) or (not curr_property and curr_prob < prev_prob):
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    
    return {
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy
    }

def save_results_to_csv_with_precision(results, filename, custom_precision):
    """
    Saves the federated learning results to a CSV file, including a custom precision value.
    
    Args:
        results (list): List of results per round.
        filename (str): Name of the output CSV file.
        custom_precision (float): Custom precision value to be included in the CSV.
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Round', 'Prediction', 'Probability', 'Clients with Property', 'Clients without Property', 'Has Property', 'Loss', 'Accuracy', 'Custom Precision']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for r in results:
            writer.writerow({
                'Round': r['round'],
                'Prediction': r['prediction'],
                'Probability': r['property_probability'],
                'Clients with Property': r['clients_with_property'],
                'Clients without Property': r['clients_without_property'],
                'Has Property': r['has_property'],
                'Loss': r['loss'],
                'Accuracy': r['accuracy'],
                'Custom Precision': f"{custom_precision:.2f}%"
            })

    logger.info(f"Results with custom precision saved to {filename}")

def plot_results_roc_and_threshold(fpr, tpr, thresholds, auc_roc, optimal_threshold):
    """
    Plots and saves both the ROC-AUC curve and the TPR/FPR vs. Threshold plot in a single figure.
    
    Args:
        fpr (array): False positive rates for the ROC curve.
        tpr (array): True positive rates for the ROC curve.
        thresholds (array): Threshold values used to generate the ROC curve.
        auc_roc (float): Area under the ROC curve.
        optimal_threshold (float): The best threshold based on ROC curve analysis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve plot
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")

    # Threshold vs. TPR/FPR plot
    ax2.plot(thresholds, tpr, label='True Positive Rate', color='green')
    ax2.plot(thresholds, fpr, label='False Positive Rate', color='red')
    ax2.axvline(optimal_threshold, color='blue', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.2f})')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Rate')
    ax2.set_title('TPR and FPR vs. Threshold')
    ax2.legend()

    # Save the combined plot
    plt.tight_layout()
    plt.savefig('roc_threshold_plot.png')
    plt.close()
    logger.info("ROC-AUC and Optimal Threshold plots saved")

def calculate_correct_predictions(results, threshold_percentage=0.1):
    """
    Calculates the number of correct and incorrect predictions based on probability movements over rounds.
    
    Args:
        results (list): List of results from the federated learning simulation.
        threshold_percentage (float): Threshold to consider probability movement as significant (default 10%).
    
    Returns:
        tuple: Number of correct predictions and incorrect predictions.
    """
    correct_predictions = 0
    incorrect_predictions = 0

    min_prob = float('inf')
    max_prob = float('-inf')

    for i in range(1, len(results)):  # Start from the second round
        current_prob = results[i]['property_probability']
        previous_prob = results[i - 1]['property_probability']
        has_property = results[i]['has_property']

        # Update min and max probabilities
        min_prob = min(min_prob, current_prob)
        max_prob = max(max_prob, current_prob)

        # Calculate significant movement threshold based on range
        range_value = max_prob - min_prob
        if range_value == 0:
            range_value = 1  # To avoid division by zero

        significant_movement_threshold = range_value * threshold_percentage
        movement_diff = abs(current_prob - previous_prob)

        if movement_diff < significant_movement_threshold:
            # Insignificant movement: treat current probability as the same as the previous one
            current_prob = previous_prob

        # Correct prediction if property is present and probability increases, or if absent and probability decreases
        if (has_property and current_prob >= previous_prob) or (not has_property and current_prob <= previous_prob):
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    return correct_predictions, incorrect_predictions

def calculate_shadow_model_metrics(results):
    """
    Calculates performance metrics for shadow models used in property inference.

    Args:
        results (list): List of results from the federated learning simulation.

    Returns:
        tuple: 
            - auc_roc (float): Area under the ROC Curve for the shadow models.
            - cm (np.array): Confusion matrix based on property inference.
            - precision (float): Precision of the shadow models.
            - optimal_threshold (float): Optimal threshold derived from the ROC curve.
    """
    # Extract true labels and predicted probabilities from results
    y_true = [r['has_property'] for r in results]
    y_prob = [r['property_probability'] for r in results]

    # Calculate AUC-ROC for shadow models
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_roc = auc(fpr, tpr)

    # Calculate the optimal threshold based on the ROC curve
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Derive predictions based on the optimal threshold
    y_pred = [1 if prob > optimal_threshold else 0 for prob in y_prob]

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate precision
    tp = cm[1, 1]
    fp = cm[0, 1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return auc_roc, cm, precision, optimal_threshold

def save_experiment_results_to_csv(dataset_size, batch_size, num_shadow_models, global_model_epochs, shadow_train_rounds, 
                                   auc_roc, optimal_threshold, precision, custom_precision, tp, fp, tn, fn, filename='final_experiment_results.csv'):
    """
    Saves the final results of the experiment to a CSV file.

    Args:
        dataset_size (int): Size of the dataset.
        batch_size (int): Batch size used in training.
        num_shadow_models (int): Number of shadow models used.
        global_model_epochs (int): Number of epochs used for global model training.
        shadow_train_rounds (int): Number of rounds for shadow model training.
        auc_roc (float): AUC-ROC score for the shadow models.
        optimal_threshold (float): Optimal threshold derived from ROC curve.
        precision (float): Precision of the shadow models.
        custom_precision (float): Custom precision of the shadow models.
        tp (int): True positives from the confusion matrix.
        fp (int): False positives from the confusion matrix.
        tn (int): True negatives from the confusion matrix.
        fn (int): False negatives from the confusion matrix.
        filename (str): Name of the output CSV file (default is 'final_experiment_results.csv').
    """
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['DATASET_SIZE', 'BATCH_SIZE', 'NUM_SHADOW_MODELS', 'GLOBAL_MODEL_EPOCHS', 'SHADOW_TRAIN_ROUNDS', 
                      'AUC-ROC', 'Optimal_Threshold', 'Precision', 'Custom_Precision', 'TP', 'FP', 'TN', 'FN']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file is empty
        if csvfile.tell() == 0:
            writer.writeheader()
        
        # Write the experiment results
        writer.writerow({
            'DATASET_SIZE': dataset_size,
            'BATCH_SIZE': batch_size,
            'NUM_SHADOW_MODELS': num_shadow_models,
            'GLOBAL_MODEL_EPOCHS': global_model_epochs,
            'SHADOW_TRAIN_ROUNDS': shadow_train_rounds,
            'AUC-ROC': auc_roc,
            'Optimal_Threshold': optimal_threshold,
            'Precision': precision,
            'Custom_Precision': custom_precision,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn
        })

    logger.info(f"Final experiment results saved to {filename}")