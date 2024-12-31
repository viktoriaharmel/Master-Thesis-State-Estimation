import torch
import torch.nn as nn
import csv
import pandas as pd
import numpy as np
import os
import io
import time
import tempfile
import ast
import shutil
from network.modules import FC, FC_DataModule
import wandb
import os.path
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, min_delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait before stopping when no improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            path (str): Path to save the best model checkpoint.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            #self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            #self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset the counter if improvement is found

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)  # Reset the pointer to the start of the buffer

        # Log the model as an artifact
        artifact = wandb.Artifact("trained_fc", type="model")
        # Write buffer contents to a temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pth") as tmp_file:
            tmp_file.write(buffer.getvalue())  # Write the buffer content (bytes) to the file
            tmp_file.flush()  # Ensure data is written to disk

            # Add the temporary file to the artifact
            artifact.add_file(tmp_file.name, name="model_state_dict.pth")
        wandb.log_artifact(artifact)

        print("Model saved as an artifact to WandB.")

def clear_directory(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # Remove all contents
        shutil.rmtree(directory_path)
        # Recreate the empty directory if needed
        os.makedirs(directory_path)
        print(f"Cleared all contents from {directory_path}")
    else:
        print(f"Directory {directory_path} does not exist or is not a directory.")
    
wandb.login()

# Define the paths and columns as needed for the data module
data_path = 'path/to/data/'
dataset = 'dataset_complete_2018_base_no_pmu_targets.csv'
with open(data_path + 'target_cols.csv', 'r') as file:
    reader = csv.reader(file)
    target_columns = next(reader)  # Read the first row

target_columns = [item for item in target_columns]   

# Set up hyperparameters
hparams = {
    "batch_size": 128,
    'num_layers': 8,
    "learning_rate": 5e-4,
    "gamma": 0.7522,
    "weight_decay": 3e-5,
    "input_size": 7448,
    "hidden_sizes_per_bus": [64, 32, 32, 16, 16, 8, 4, 2],
    "output_size": len(target_columns),
    "num_workers": 40,
    "epochs": 150,
    "activation": nn.ReLU(),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

with wandb.init(project="FC Training AUS P1R complete base", config=hparams):
    config = wandb.config

    min_angle = -torch.pi
    max_angle = torch.pi

    angle_target_indices = [idx for idx, target in enumerate(target_columns) if 'angle' in target]

    bus_measurements_df = pd.read_csv(data_path + 'bus_measurement_mapping.csv')

    # Ensure that each field is a list
    for col in ['neighbor_buses', 'lines', 'loads', 'transformers']:
        bus_measurements_df[col] = bus_measurements_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Flatten partition elements into a dictionary for easy lookup
    bus_measurements = {}
    for idx, row in bus_measurements_df.iterrows():
        bus_measurements[row['bus']] = {
            'neighbor_buses': row['neighbor_buses'],
            'lines': row['lines'],
            'loads': row['loads'],
            'transformers': row['transformers']
        }

    # Set up the data module
    data_module = FC_DataModule(
        data_path=data_path + dataset,
        target_columns=target_columns,
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        root=data_path
    )

    # Prepare the data by splitting and applying transformations
    data_module.setup()

    device = hparams["device"]
    
    def log(loss_type, loss, example_ct, epoch, mode='train'):
        wandb.log({"epoch": epoch, f"{mode}_{loss_type}": loss, f"{mode}_example_ct":example_ct})
        print(f"{mode} {loss_type} after {str(example_ct).zfill(5)} examples: {loss:.3f}")


    def train_model(model, train_loader, val_loader, loss_func, epochs=10, name="default"):
        """
        Train the classifier for a number of epochs.
        """
        wandb.watch(model, loss_func, log="all", log_freq=10)

        optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * len(train_loader) / 5), gamma=hparams["gamma"])
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=5e-4,         # Matches initial learning rate
            max_lr=1e-3,          # Maximum learning rate
            step_size_up=220,    # Number of iterations to reach max_lr
            mode='triangular2',    # Oscillation mode with decreasing amplitude
            cycle_momentum=False
        )
        scheduler_stopping = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-6)
        early_stopping = EarlyStopping(patience=10, min_delta=1e-4, path=name + "trained_fc.pth")
        
        wandb.define_metric("train_example_ct")
        wandb.define_metric("val_example_ct")
        wandb.define_metric("epoch")
        wandb.define_metric("train_mse", step_metric="train_example_ct")
        wandb.define_metric("train_mape", step_metric="train_example_ct")
        wandb.define_metric("train_r2", step_metric="train_example_ct")
        wandb.define_metric("val_mse", step_metric="val_example_ct")
        wandb.define_metric("val_mape", step_metric="val_example_ct")
        wandb.define_metric("val_r2", step_metric="val_example_ct")
        wandb.define_metric("learning_rate", step_metric="epoch")
        wandb.define_metric("train_mse_epoch", step_metric="epoch")
        wandb.define_metric("val_mse_epoch", step_metric="epoch")
        wandb.define_metric("train_mape_epoch", step_metric="epoch")
        wandb.define_metric("val_mape_epoch", step_metric="epoch")
        wandb.define_metric("train_r2_epoch", step_metric="epoch")
        wandb.define_metric("val_r2_epoch", step_metric="epoch")

        train_batch_ct = 0
        train_example_ct = 0
        val_batch_ct = 0
        val_example_ct = 0
        for epoch in range(epochs):

            # Training stage, where we want to update the parameters.
            model.train()  # Set the model to training mode

            print(f'Training epoch {epoch}')
            train_mse, train_mape, train_r2 = 0.0, 0.0, 0.0
            for batch in train_loader:
                optimizer.zero_grad() # Reset the gradients
                # Extract features and targets from batch dictionary
                features = batch['features']
                targets = batch['targets']

                # Move to device if necessary (optional)
                features, targets = features.to(device), targets.to(device) # Send the data to the device (GPU or CPU) - it has to be the same device as the model.

                pred = model(features) # Stage 1: Forward().
                pred[:, angle_target_indices] = torch.clamp(pred[:, angle_target_indices], min=min_angle, max=max_angle)
                loss = loss_func(pred, targets) # Compute the loss over the predictions and the ground truth.
                
                train_example_ct +=  len(features)
                train_batch_ct += 1
 
                train_mse += loss.item()
                train_mape += model.mean_absolute_percentage_error(targets, pred)
                train_r2 += model.r2_score(targets, pred)

                loss.backward()  # Stage 2: Backward().
                optimizer.step() # Stage 3: Update the parameters.
                scheduler.step() # Update the learning rate.
 
            # Validation stage
            model.eval()

            with torch.no_grad():
                val_mse, val_mape, val_r2 = 0.0, 0.0, 0.0
                for batch in val_loader:
                    features = batch['features']
                    targets = batch['targets']
                    features, targets = features.to(device), targets.to(device)

                    pred = model(features)
                    pred[:, angle_target_indices] = torch.clamp(pred[:, angle_target_indices], min=min_angle, max=max_angle)
                    loss = loss_func(pred, targets)
                    val_example_ct +=  len(features)
                    val_batch_ct += 1

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

            scheduler_stopping.step(val_mse)
            # Log epoch-level metrics
            wandb.log({
                "epoch": epoch,
                "train_mse_epoch": train_mse,
                "val_mse_epoch": val_mse,
                "train_mape_epoch": train_mape,
                "val_mape_epoch": val_mape,
                "train_r2_epoch": train_r2,
                "val_r2_epoch": val_r2,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

            # Check for early stopping
            early_stopping(val_mse, model)

            # Break the loop if early stopping is triggered
            if early_stopping.early_stop:
                print("Early stopping triggered. Training stopped.")
                break
        
        # Save the model in the exchangeable pth format
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)  # Reset the pointer to the start of the buffer

        # Log the model as an artifact
        artifact = wandb.Artifact("trained_fc", type="model")
        # Write buffer contents to a temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pth") as tmp_file:
            tmp_file.write(buffer.getvalue())  # Write the buffer content (bytes) to the file
            tmp_file.flush()  # Ensure data is written to disk

            # Add the temporary file to the artifact
            artifact.add_file(tmp_file.name, name="model_state_dict.pth")
        wandb.log_artifact(artifact)

        print("Model saved as an artifact to WandB.")
    

    def test_model(model, test_loader, loss_func, target_columns, device, results_path="test_results.xlsx"):
        """
        Test the model on the test dataset and log results.
        
        Parameters:
            model: Trained PyTorch model.
            test_loader: DataLoader for the test dataset.
            loss_func: Loss function used for testing.
            target_columns: List of target column names.
            device: Device to run the model on (e.g., "cuda" or "cpu").
            results_path: Path to save the test results in Excel format.
        """
        model.eval()  # Set the model to evaluation mode
        test_loss = []
        test_mape = []
        test_r2 = []

        test_results = {
            'Index': [],
            'Target': [],
            'Target Value': [],
            'Prediction': []
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)

                # Forward pass
                pred = model(features)
                loss = loss_func(pred, targets)

                # Compute additional metrics
                mape = model.mean_absolute_percentage_error(targets, pred).item()
                r2 = model.r2_score(targets, pred).item()

                # Log metrics
                test_loss.append(loss.item())
                test_mape.append(mape)
                test_r2.append(r2)

                for index, (target_vals, predictions) in enumerate(zip(targets.cpu().numpy(), pred.cpu().detach().numpy())):
                    for target, target_value, prediction in zip(target_columns, target_vals, predictions):
                        test_results['Index'].append(batch_idx * test_loader.batch_size + index)
                        test_results['Target'].append(target)
                        test_results['Target Value'].append(target_value)
                        test_results['Prediction'].append(prediction)

        test_results_df = pd.DataFrame.from_dict(test_results)

        buffer = io.BytesIO()
        test_results_df.to_csv(buffer)
        buffer.seek(0)

        # Create a WandB artifact
        artifact = wandb.Artifact("test_results", type="dataset")

        # Write the DataFrame directly to a temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix=".csv") as tmp_file:
            test_results_df.to_csv(tmp_file.name, index=False)  # Save DataFrame as CSV to the temp file
            tmp_file.flush()  # Ensure data is written to disk

            # Add the temporary file to the artifact
            artifact.add_file(tmp_file.name, name="test_results.csv")

        # Log the artifact to WandB
        wandb.log_artifact(artifact)

        print("Test results saved as an artifact to WandB.")

        # Compute mean metrics
        avg_loss = np.mean(test_loss)
        avg_mape = np.mean(test_mape)
        avg_r2 = np.mean(test_r2)

        print(f"Test Results: Loss: {avg_loss:.4f}, MAPE: {avg_mape:.2f}%, R²: {avg_r2:.4f}")
        return avg_loss, avg_mape, avg_r2


    path = "logs"
    num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
    path = os.path.join(path, f'run_{num_of_runs + 1}')

    # Train the classifier.
    labled_train_loader = data_module.train_dataloader()
    labled_val_loader = data_module.val_dataloader()

    epochs = hparams.get('epochs', 4)
    loss_func = nn.MSELoss() # The loss function used for classification.
    model = FC(hparams, bus_measurements, target_columns).to(device)
    train_model(model, labled_train_loader, labled_val_loader, loss_func, epochs=epochs, name="fc_complete_base")
    

    print()
    print("Finished training!")
    
    # Get the test DataLoader
    labeled_test_loader = data_module.test_dataloader()

    # Test the model
    test_loss, test_mape, test_r2 = test_model(
        model=model,
        test_loader=labeled_test_loader,
        loss_func=nn.MSELoss(),
        target_columns=target_columns,
        device=device,
        results_path=data_path + "test_results_fc_complete_base.xlsx"
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAPE: {test_mape:.2f}%")
    print(f"Test R²: {test_r2:.4f}")

    time.sleep(60)

    clear_directory(os.path.expanduser('~/.cache/wandb/artifacts/obj/'))
    clear_directory(os.path.expanduser('~/.local/share/wandb/artifacts/staging/'))