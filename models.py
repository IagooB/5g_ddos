import tensorflow as tf
import numpy as np
import logging
import os
from concurrent.futures import ThreadPoolExecutor

# Set random seed for reproducibility
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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

def create_global_model(input_shape):
    """
    Creates and compiles the global TensorFlow model used for federated learning.
    
    Args:
        input_shape (int): The shape of the input features.
    
    Returns:
        model (tf.keras.Model): A compiled TensorFlow model.
    """
    logger.info(f"Creating global model with input shape {input_shape}")
    
    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),  # Input layer with the specified input shape
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer with 64 units and ReLU activation
        tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer with 32 units and ReLU activation
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    # Compile the model with Adam optimizer and binary crossentropy loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_shadow_model(input_shape):
    """
    Creates and returns a shadow model. The architecture is the same as the global model.
    
    Args:
        input_shape (int): The shape of the input features.
    
    Returns:
        model (tf.keras.Model): A TensorFlow model identical to the global model architecture.
    """
    logger.info(f"Creating shadow model with input shape {input_shape}")
    return create_global_model(input_shape)

def initialize_clients(client_data, global_model_weights, global_model_epochs, batch_size):
    """
    Initializes the clients for federated learning by assigning the global model weights.
    Clients are initialized in parallel using a thread pool for efficiency.
    
    Args:
        client_data (list): A list of dictionaries containing client-specific data.
        global_model_weights (list): The weights from the global model to initialize clients.
        global_model_epochs (int): The number of epochs for each client's local training.
        batch_size (int): The batch size for training on each client.
    
    Returns:
        clients (list): A list of initialized SimulatedFlowerClient instances.
    """
    logger.info("Initializing simulated clients in parallel")

    clients = []
    
    def init_client(client_id, data):
        """
        Helper function to initialize a single client with given data and model weights.
        
        Args:
            client_id (int): The unique identifier for the client.
            data (dict): The data specific to the client.
        
        Returns:
            client (SimulatedFlowerClient): Initialized client with the global model's weights.
        """
        client = SimulatedFlowerClient(client_id, data, global_model_epochs, batch_size)
        client.model.set_weights(global_model_weights)  # Set the global model's weights to the client
        return client

    # Parallelize client initialization using ThreadPoolExecutor for efficiency
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(init_client, i, data) for i, data in enumerate(client_data)]
        clients = [f.result() for f in futures]  # Gather initialized clients

    return clients

class SimulatedFlowerClient:
    """
    Simulates a client in federated learning.
    Each client holds its own data and a local model, which is updated with global model parameters during training.
    """
    def __init__(self, cid, data, global_model_epochs, batch_size):
        """
        Initializes a simulated federated learning client.
        
        Args:
            cid (int): Client ID.
            data (dict): Client's dataset, including features (X) and labels (y_label).
            global_model_epochs (int): Number of epochs for training the client's model.
            batch_size (int): Batch size for model training.
        """
        logger.info(f"Initializing simulated client {cid}")
        self.cid = cid
        self.data = data
        self.global_model_epochs = global_model_epochs  # Number of training epochs
        self.batch_size = batch_size  # Batch size for training
        self.model = create_global_model(self.data['X'].shape[1])  # Initialize the client's model

    def get_parameters(self):
        """
        Retrieves the current model parameters (weights) of the client.
        
        Returns:
            list: Model's weights.
        """
        return self.model.get_weights()

    def fit(self, parameters):
        """
        Trains the client's model with the given global model parameters.
        
        Args:
            parameters (list): Weights from the global model.
        
        Returns:
            tuple: Updated weights, number of training samples, and updates (flattened weight differences).
        """
        logger.info(f"Training client model {self.cid}")
        
        old_weights = self.model.get_weights()  # Store old weights
        self.model.set_weights(parameters)  # Set the global model weights
        self.model.fit(self.data['X'], self.data['y_label'], epochs=self.global_model_epochs, batch_size=self.batch_size, verbose=0)
        new_weights = self.model.get_weights()  # Get the updated weights after training

        # Calculate the difference (updates) between the new and old weights
        updates = [new_w - old_w for new_w, old_w in zip(new_weights, old_weights)]
        flattened_updates = np.concatenate([u.flatten() for u in updates])  # Flatten the updates for easy processing

        return new_weights, len(self.data['X']), {"updates": flattened_updates}

    def evaluate(self, parameters):
        """
        Evaluates the client's model using the given global model parameters.
        
        Args:
            parameters (list): Weights from the global model.
        
        Returns:
            tuple: Loss, number of samples, and accuracy.
        """
        self.model.set_weights(parameters)  # Set global weights for evaluation
        loss, accuracy = self.model.evaluate(self.data['X'], self.data['y_label'], verbose=0)
        return loss, len(self.data['X']), {"accuracy": accuracy}

def train_shadow_models(client_data, num_shadow_models, SHADOW_TRAIN_ROUNDS, GLOBAL_MODEL_EPOCHS, BATCH_SIZE):
    """
    Trains multiple shadow models in parallel using federated client updates.
    
    Args:
        client_data (list): List of client data dictionaries.
        num_shadow_models (int): Number of shadow models to train.
        SHADOW_TRAIN_ROUNDS (int): Number of rounds to collect updates from clients for shadow model training.
        GLOBAL_MODEL_EPOCHS (int): Number of epochs for global model training.
        BATCH_SIZE (int): Batch size for training models.
    
    Returns:
        tuple: List of trained shadow models and the size of the shadow training dataset.
    """
    logger.info(f"Training {num_shadow_models} shadow models in parallel")

    input_shape = client_data[0]['X'].shape[1]

    # Use MirroredStrategy for multi-GPU support if available
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        global_model = create_global_model(input_shape)  # Create the global model to be trained

    X_shadow, y_shadow = [], []

    # Collect updates from clients over multiple rounds
    for _ in range(SHADOW_TRAIN_ROUNDS):
        for client in client_data:
            old_weights = global_model.get_weights()
            global_model.fit(client['X'], client['y_label'], epochs=GLOBAL_MODEL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            new_weights = global_model.get_weights()

            # Compute updates (difference between new and old weights)
            updates = [new_w - old_w for new_w, old_w in zip(new_weights, old_weights)]
            X_shadow.append(np.concatenate([u.flatten() for u in updates]))  # Flatten and store updates
            y_shadow.append(int(client['has_property']))  # Label based on whether the client has the property

    X_shadow = np.array(X_shadow)
    y_shadow = np.array(y_shadow)

    shadow_models = []

    # Helper function to train a single shadow model
    def train_single_shadow_model(i):
        logger.info(f"Training shadow model {i + 1}/{num_shadow_models}")
        with strategy.scope():
            shadow_model = create_global_model(X_shadow.shape[1])  # Create the shadow model
            shadow_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            shadow_model.fit(X_shadow, y_shadow, epochs=GLOBAL_MODEL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        return shadow_model

    # Train shadow models in parallel
    with ThreadPoolExecutor(max_workers=num_shadow_models) as executor:
        shadow_models = list(executor.map(train_single_shadow_model, range(num_shadow_models)))

    return shadow_models, len(X_shadow)

def parallel_client_training(global_model_weights, selected_clients):
    """
    Trains selected clients in parallel and collects their model updates.
    
    Args:
        global_model_weights (list): Weights from the global model to initialize clients.
        selected_clients (list): List of selected SimulatedFlowerClient instances.
    
    Returns:
        tuple: Updated client weights and flattened client updates.
    """
    logger.info("Training clients in parallel")

    # Helper function to train a single client
    def train_single_client(client):
        w, _, update_info = client.fit(global_model_weights)
        return w, update_info['updates']
    
    # Train clients in parallel
    with ThreadPoolExecutor(max_workers=len(selected_clients)) as executor:
        results = list(executor.map(train_single_client, selected_clients))
    
    client_weights, client_updates = zip(*results)  # Separate weights and updates
    return client_weights, client_updates

def infer_property(shadow_models, updates):
    """
    Infers the presence of a property based on model updates using shadow models.
    
    Args:
        shadow_models (list): List of trained shadow models.
        updates (np.array): Flattened updates from the global model.
    
    Returns:
        float: Mean probability that the updates indicate the presence of the property.
    """
    flattened_updates = updates.flatten().reshape(1, -1)  # Flatten updates for prediction
    probabilities = [model.predict(flattened_updates, verbose=0)[0][0] for model in shadow_models]  # Predict with each shadow model
    return np.mean(probabilities)  # Return the mean probability across all shadow models
