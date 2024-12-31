import torch
import torch.nn as nn
import csv
import pandas as pd
import json
import ast
from network.modules import PAWNN, PAWNN_DataModule
import wandb

wandb.login()

def load_input_feature_map(file_path):
    with open(file_path, 'r') as file:
        input_feature_map = json.load(file)
    print(f"Input feature map loaded from {file_path}")
    return input_feature_map

# Paths and data setup
data_path = 'path/to/data/'
dataset = 'dataset_complete_2018_base_no_pmu_targets.csv'
with open(data_path + 'target_cols.csv', 'r') as file:
    reader = csv.reader(file)
    target_columns = next(reader)

diameter = 8
target_columns = [item for item in target_columns]
min_angle = -torch.pi
max_angle = torch.pi
angle_target_indices = [idx for idx, target in enumerate(target_columns) if 'angle' in target]

bus_measurements_df = pd.read_csv(data_path + 'bus_measurement_mapping.csv')
for col in ['neighbor_buses', 'lines', 'loads', 'transformers']:
    bus_measurements_df[col] = bus_measurements_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

bus_measurements = {}
for idx, row in bus_measurements_df.iterrows():
    bus_measurements[row['bus']] = {
        'neighbor_buses': row['neighbor_buses'],
        'lines': row['lines'],
        'loads': row['loads'],
        'transformers': row['transformers']
    }

input_feature_map = load_input_feature_map(data_path + 'input_feature_map.json')

# Sweep Configuration

sweep_config = {
    "method": "random",  # Random search
    "metric": {"name": "val_mse", "goal": "minimize"},
    "parameters": {
        "batch_size": {"values": [4, 8, 16, 32, 64, 128]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "hidden_sizes_per_bus": {
            "values": [
                [64, 32, 16, 8, 8, 4, 4, 2],
                [64, 32, 16, 8, 8, 4, 2, 2],
                [64, 32, 16, 8, 4, 4, 2, 2],
                [64, 32, 16, 8, 4, 4, 2, 2],
                [64, 64, 32, 16, 8, 4, 4, 2],
                [64, 32, 32, 16, 8, 8, 4, 2],
                [32, 16, 8, 8, 4, 4, 2, 2],
                [32, 16, 8, 4, 4, 4, 2, 2],
                [32, 16, 8, 8, 4, 2, 2, 2],
                [32, 32, 16, 8, 8, 4, 4, 2],
                [32, 16, 8, 8, 4, 4, 2, 2],
                [32, 32, 32, 16, 8, 4, 4, 2],
                [16, 16, 8, 8, 4, 4, 2, 2],
                [16, 16, 16, 8, 4, 4, 2, 2],
                [16, 16, 8, 8, 4, 2, 2, 2],
                [16, 16, 16, 8, 8, 4, 4, 2],
                [16, 8, 8, 8, 4, 4, 4, 2],
                [16, 16, 16, 16, 8, 4, 4, 2]
            ]
        },
        "gamma": {"min": 0.7, "max": 0.9},
        "weight_decay": {"distribution": "uniform", "min": 0.0, "max": 1e-4},
        "activation": {"values": ["ReLU", "Tanh", "LeakyReLU"]},
        "epochs": {"min": 25, "max": 200}
    },
}

# Function to map activation names to PyTorch functions

activation_map = {
    "ReLU": nn.ReLU(),
    "Tanh": nn.Tanh(),
    "Sigmoid": nn.Sigmoid(),
    "LeakyReLU": nn.LeakyReLU(),
}


def train_with_sweep():
    with wandb.init() as run:
        config = wandb.config

        hparams = {
            'input_size': 7448,
            'output_size': len(target_columns),
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_layers': diameter,
            'hidden_sizes_per_bus': config.hidden_sizes_per_bus,
            'gamma': config.gamma,
            'weight_decay': config.weight_decay,
            'activation': activation_map[config.activation],
            'epochs': config.epochs,
            'num_workers': 60,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        # Set up the data module
        data_module = PAWNN_DataModule(
            data_path=data_path + dataset,
            target_columns=target_columns,
            batch_size=hparams["batch_size"],
            num_workers=hparams["num_workers"],
            root=data_path
        )
        data_module.setup()
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        device = hparams["device"]

        # Create model
        model = PAWNN(hparams, bus_measurements, input_feature_map, target_columns).to(device)

        # Create optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=hparams["gamma"])

        # Training and validation
        for epoch in range(hparams['epochs']):
            model.train()
            train_mse, train_mape, train_r2 = 0.0, 0.0, 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)

                pred = model(features)
                pred[:, angle_target_indices] = torch.clamp(pred[:, angle_target_indices], min=min_angle, max=max_angle)
                loss = nn.MSELoss()(pred, targets)

                # Compute metrics
                train_mse += loss.item()
                train_mape += model.mean_absolute_percentage_error(targets, pred)
                train_r2 += model.r2_score(targets, pred)

                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            val_mse, val_mape, val_r2 = 0.0, 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device)
                    targets = batch['targets'].to(device)

                    pred = model(features)
                    pred[:, angle_target_indices] = torch.clamp(pred[:, angle_target_indices], min=min_angle, max=max_angle)
                    loss = nn.MSELoss()(pred, targets)
                    
                    # Compute metrics
                    val_mse += loss.item()
                    val_mape += model.mean_absolute_percentage_error(targets, pred)
                    val_r2 += model.r2_score(targets, pred)

            # Normalize metrics
            train_mse /= len(train_loader)
            train_mape /= len(train_loader)
            train_r2 /= len(train_loader)

            val_mse /= len(val_loader)
            val_mape /= len(val_loader)
            val_r2 /= len(val_loader)

            # Log metrics to WandB
            wandb.log({
                "train_mse": train_mse,
                "train_mape": train_mape,
                "train_r2": train_r2,
                "val_mse": val_mse,
                "val_mape": val_mape,
                "val_r2": val_r2,
            })

            print(f"Epoch [{epoch+1}/{hparams['epochs']}] - Train MSE: {loss.item():.4f}, Validation MSE: {val_mse:.4f}")

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="PAWNN_Hyperparameter_Tuning_AUS_P1R_complete_base")

# Run the sweep
wandb.agent(sweep_id, train_with_sweep)