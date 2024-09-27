# 5G Network Slice DoS/DDoS Attack Detection Using Federated Learning and Shadow Models

## Overview

This project implements a federated learning approach to detect shadow models attacks in 5G network slices. The dataset used for this experiment is derived from the paper "[DoS/DDoS Attack Dataset of 5G Network Slicing](https://ieeexplore.ieee.org/document/10056693)" by Khan et al. The objective of this project is to simulate federated learning across multiple clients using shadow models to infer specific properties (such as whether the data corresponds to an specific slice).


The dataset consists of traffic collected from a simulated 5G network with slicing. It includes benign and malicious traffic, distributed across two slices (Slice 1 and Slice 2) and over two days of data collection (Day 1 and Day 2).

### Dataset

The processed dataset used for this experiment can be downloaded in Kaggle: "[Dos/DDoS Attacks on 5g networks
](https://www.kaggle.com/datasets/iagobs/dosddos-attacks-on-5g-networks)"

Selected features for the experiments:
- **Flow Duration**
- **Src IP**
- **Dst Port**
- **Fwd Packet Length Std**
- **Src Port**
- **ACK Flag Count**
- **Protocol**
- **Total Fwd Packet**
- **Fwd Seg Size Min**

**Preprocessing**:
- IP addresses were converted to numeric format.
- Attack and benign labels were encoded (0 for benign, 1 for attack).
- Missing values were removed.
- StandardScaler was applied to normalize the features.

The final dataset contains over **6 million rows** and **13 columns**.

## Project Structure

The code is organized into the following main modules:

- **`data_preparation.py`**: Loads, cleans, and preprocesses the dataset. It also prepares data for federated clients.
- **`models.py`**: Defines the global model used for federated learning and the shadow models for property inference.
- **`outputs.py`**: Handles the generation of plots and saving results such as model weights and performance metrics.
- **`main.py`**: The main script that orchestrates the federated learning process, including initializing clients, training models, and running simulations.

## Installation

### Prerequisites

This project requires Python 3.7 or later. The following Python libraries are necessary:

- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `concurrent.futures`

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

### Dataset

The dataset can be downloaded from the [IEEE Dataport](https://ieee-dataport.org/documents/dosddos-attack-dataset-5g-network-slicing#files). After downloading, place the dataset file (`label_bi.csv`) in the root directory of this project.

## How to Run the Project

### 1. Preprocessing the Data

The data preprocessing is handled by the `data_preparation.py` module, which:
- Loads the data from the `5g_ddos.csv` file.
- Cleans and preprocesses the data by removing missing values, scaling numerical features, and preparing labels.

```bash
python main.py
```

### 2. Federated Learning Simulation

The main federated learning simulation is initiated by running the `main.py` script. This script:
- Creates multiple simulated clients, each with a subset of the data.
- Trains shadow models to infer whether each client has a specific property.
- Aggregates client updates to update a global model.
- Evaluates the performance of the global model and shadow models, saving the results to CSV files.

You can configure various parameters, such as the number of clients, rounds, and shadow models in the `main.py` script.

### 3. Output and Results

After running the experiments, the following outputs are generated:
- **Logs**: Stored in the `logs/` directory.
- **Model Weights**: Saved for each round in `shadow_model_weights.csv`.
- **Results**: Final experiment results are saved in `federated_learning_results.csv` and `final_experiment_results.csv`.

### Plotting Results

The `outputs.py` module generates plots for key metrics:
- **ROC Curve**: AUC-ROC of shadow models.
- **Loss and Accuracy**: Over federated learning rounds.
- **Optimal Threshold**: Based on the ROC curve for property inference.

These plots are automatically saved in the working directory.

## Key Functions

- **`load_and_preprocess_data(file_path)`**: Loads and cleans the dataset, returning feature matrices and labels.
- **`create_global_model(input_shape)`**: Creates the global TensorFlow model.
- **`create_shadow_model(input_shape)`**: Creates a shadow model with the same architecture as the global model.
- **`simulated_federated_learning(clients, shadow_models, global_model)`**: Simulates federated learning, aggregates updates, and trains shadow models.
- **`analyze_results(results_file)`**: Analyzes the results stored in the CSV file and computes metrics such as AUC-ROC and optimal thresholds.

## Configuration

The following key parameters are configurable in the `main.py` file:

- **`NUM_CLIENTS`**: Number of clients to simulate in federated learning (default: 10).
- **`NUM_ROUNDS`**: Number of federated learning rounds (default: 100).
- **`GLOBAL_MODEL_EPOCHS`**: Number of epochs for training the global model (default: 50).
- **`NUM_SHADOW_MODELS`**: Number of shadow models for property inference (default: 20).
- **`BATCH_SIZE`**: Batch size for model training (default: 32).

## Citation

If you use this dataset or code in your research, please cite the following paper:

```
@article{khan2022dosddos,
  title={DoS/DDoS Attack Dataset of 5G Network Slicing},
  author={Khan, Author 1 and Others},
  journal={IEEE Dataport},
  year={2022},
  url={https://ieee-dataport.org/documents/dosddos-attack-dataset-5g-network-slicing#files}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
