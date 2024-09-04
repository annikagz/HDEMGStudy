import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from networks import TemporalConvNet
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainEval:
    def __init__(
        self,
        training_data,
        testing_data,
        n_channels,
        lr=1e-3,
        stagnation_margin=5,
        k_fold=True,
        model_name="",
    ):
        self.training_data = training_data
        self.testing_data = testing_data
        self.lr = lr
        self.stagnation_margin = stagnation_margin
        self.model = TemporalConvNet(num_inputs=1, num_channels=n_channels).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), betas=(0.9, 0.999), lr=self.lr, weight_decay=0
        )
        self.criterion = nn.MSELoss()
        self.epochs = 100
        self.model_name = model_name
        self.saved_model_path = "" + model_name
        if k_fold:
            self.train_kfolds()
        else:
            self.train_regular_model(self.training_data[0], self.training_data[1])
        self.model_performance = pd.DataFrame(
            columns=[
                "Model name" "Model type",
                "Epochs",
                "Training loss",
                "Validation loss",
                "Testing loss",
            ]
        )

    def training_block(self, x_train, y_train):
        optimizer = self.optimizer
        self.model.train()
        epoch_losses = []
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch in tqdm(
                range(x_train.shape[0]), desc=f"Epoch {epoch + 1}/{self.epochs}"
            ):
                inputs = torch.tensor(x_train[batch], dtype=torch.float32)
                targets = torch.tensor(y_train[batch], dtype=torch.float32)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / x_train.shape[0]
            epoch_losses.append(epoch_loss)
            print(f"Training Loss: {epoch_loss:.4f}")
            if (
                len(epoch_losses) > 20
                and np.std(epoch_losses[-self.stagnation_margin : :]) < self.lr
            ):
                break  # Stop training if the loss is stagnant
        return epoch_losses

    def train_regular_model(self):
        epoch_losses = self.training_block(self.training_data[0], self.training_data[1])
        torch.save(self.model.state_dict(), self.saved_model_path + "_regular.pth")
        testing_loss = self.evaluate_model(self.testing_data[0], self.testing_data[1])
        self.model_performance.loc[len(self.model_performance) + 1] = [
            self.model_name,
            "Regular",
            len(epoch_losses),
            epoch_losses[-1],
            np.nan,
            testing_loss,
        ]

    def train_kfolds(self):
        self.model.train()
        training_losses = []
        validation_losses = []
        epochs_ran = []
        for fold, (x_train, x_val, y_train, y_val) in enumerate(self.training_data):
            print(f"Fold {fold + 1}/{len(self.training_data)}")
            self.model.reset_model_weights()
            epoch_losses = self.training_block(x_train, y_train)
            # Do the validation evaluation
            val_loss = self.evaluate_model(x_val, y_val)
            validation_losses.append(val_loss)
            training_losses.append(epoch_losses[-1])
            epochs_ran.append(len(epoch_losses))
            print(f"Validation Loss for fold {fold + 1}: {val_loss:.4f}")
            torch.save(self.model.state_dict(), self.saved_model_path + "_kfolds.pth")
        testing_loss = self.evaluate_model(self.testing_data[0], self.testing_data[1])
        self.model_performance.loc[len(self.model_performance) + 1] = [
            self.model_name,
            "KFolds",
            np.mean(epochs_ran),
            np.mean(training_losses),
            np.mean(validation_losses),
            testing_loss,
        ]

    def evaluate_model(self, x_test, y_test):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in range(x_test.shape[0]):
                inputs = torch.tensor(x_test[batch], dtype=torch.float32)
                targets = torch.tensor(y_test[batch], dtype=torch.float32)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
        running_loss /= x_test.shape[0]
        return running_loss
