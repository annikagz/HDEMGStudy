import numpy as np
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch
from sklearn.model_selection import KFold

from DataProcessing.preprocessing import standardize_emg, normalize_labels


class NetworkFormatter:
    def __init__(
        self,
        emg_data,
        label,
        fs=2000,
        window_size=250,
        step_size=10,
        test_size=0.05,
        batch_size=4,
        n_folds=None,  # otheriwse this is an int
    ):
        self.emg_data = emg_data
        self.label = label
        self.fs = fs
        self.window_size = int((window_size / 1000) * self.fs)
        self.step_size = int((step_size / 1000) * self.fs)
        self.prediction_horizon = 1
        self.n_folds = n_folds
        self.test_size = test_size
        self.batch_size = batch_size
        self.windowed_emg = None
        self.windowed_labels = None
        self.training_data = None
        self.testing_data = None

        # ACTIONS
        self.window_the_data()
        self.batch_the_data()
        self.reserve_testing_data()
        self.shuffle_training_windows()
        if n_folds:
            self.apply_k_fold_formatting(n_folds)
            # Note that if we apply this, then self.folds[0] = train_data, val_data, train_labels, val_labels
        else:
            self.training_data[0], self.testing_data[0] = standardize_emg(
                self.training_data[0], self.testing_data[0]
            )
            self.training_data[1], self.testing_data[1] = normalize_labels(
                self.training_data[1], self.testing_data[1]
            )

    def window_the_data(self):
        windowed_emg_data = []
        windowed_label = []
        # Window the data
        for i in range(
            0,
            self.emg_data.shape[0] - self.window_size - self.prediction_horizon,
            self.step_size,
        ):
            windowed_emg_data.append(
                np.expand_dims(self.emg_data[i : i + self.window_size, :], axis=0)
            )
            windowed_label.append(
                np.expand_dims(
                    self.label[i + self.window_size + self.prediction_horizon, :],
                    axis=0,
                )
            )
        self.windowed_emg = np.concatenate(windowed_emg_data, axis=0).transpose(
            (0, 2, 1)
        )  # So now the data is of shape (n_windows, n_channels, window_length)
        self.windowed_labels = np.concatenate(windowed_label, axis=0)

    def batch_the_data(self):
        num_batches = self.windowed_emg.shape[0] // self.batch_size

        data_batches = []
        label_batches = []

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size

            batch_data = self.windowed_emg[start_idx:end_idx]
            batch_labels = self.windowed_labels[start_idx:end_idx]
            data_batches.append(np.expand_dims(batch_data, 0))
            label_batches.append(np.expand_dims(batch_labels, 0))

        # Handle the remaining data if it's not enough to form a full batch
        if self.windowed_emg.shape[0] % self.batch_size != 0:
            batch_data = self.windowed_emg[num_batches * self.batch_size :]
            batch_labels = self.windowed_labels[num_batches * self.batch_size :]
            data_batches.append(np.expand_dims(batch_data, 0))
            label_batches.append(np.expand_dims(batch_labels, 0))

        data_batches = np.stack(data_batches, axis=0)
        label_batches = np.stack(label_batches, axis=0)
        # Right now, the data is of shape (n_batches, batch_size, n_channels, window_length)
        # The labels are of shape (n_batches, batch_size, 1)
        self.windowed_emg = data_batches
        self.windowed_labels = label_batches

    def reserve_testing_data(self):
        emg_train, emg_test, label_train, label_test = train_test_split(
            self.windowed_emg,
            self.windowed_labels,
            test_size=self.test_size,
            shuffle=False,
        )
        self.training_data = (emg_train, label_train)
        self.testing_data = (emg_test, label_test)

    def apply_k_fold_formatting(self, n_folds):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = []

        for train_index, val_index in kf.split(self.training_data[0]):
            train_data, val_data = (
                self.training_data[0][train_index],
                self.training_data[0][val_index],
            )
            train_labels, val_labels = (
                self.training_data[1][train_index],
                self.training_data[1][val_index],
            )
            train_data, val_data = standardize_emg(train_data, val_data)
            train_labels, val_labels = normalize_labels(train_labels, val_labels)
            folds.append((train_data, val_data, train_labels, val_labels))
        self.training_data = folds

    def shuffle_training_windows(self):
        assert (
            self.training_data[0].shape[0] == self.training_data[1].shape[0]
        ), "Arrays must have the same length along axis 0"

        # Generate a random permutation of indices based on the length of axis 0
        indices = np.random.permutation(self.training_data[0].shape[0])

        # Apply the permutation to both arrays
        self.training_data[0] = self.training_data[0][indices]
        self.training_data[1] = self.training_data[1][indices]
