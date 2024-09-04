import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

from variance_metrics import get_all_metrics
from DataProcessing.utils import (
    segment_array_with_label_peaks,
    rank_values,
)
from DataProcessing.preprocessing import envelope
from DataProcessing.hdemg_to_bipolar import HDEMG2Bipolar
from DataProcessing.formatting import NetworkFormatter
from training import TrainEval


class MuscleData:
    def __init__(self, subject_number, muscle, fs=2000):
        self.subject_path = ""
        self.subject_name = "DS0" + str(subject_number)
        self.muscle = muscle
        self.flipped = [True if self.subject_name in [""] else False][0]
        self.fs = fs

        self.raw_data = None
        self.combinations = None
        self.bipolar_data = None
        self.label = None

        self.variance_metrics = None  # This will be the CoV for the max peak, the peak location, and the auc
        self.rankings = None

        self.model_performances = None

    def get_data(self):
        data = pd.load_cvs(self.subject_path)
        sampler = HDEMG2Bipolar(
            data, radius=3.2, fov=95, patch_shape=(13, 5), flipped=self.flipped
        )
        self.raw_data = sampler.HDEMGData
        self.combinations = sampler.list_of_combinations
        self.bipolar_data = sampler.bipolar_data

    def filter_the_data(self):
        self.bipolar_data = envelope(self.bipolar_data)

    def compute_variance_metrics(self):
        cycle_emg_data = segment_array_with_label_peaks(
            self.bipolar_data, knee_angle=self.label, cycle_length=1000
        )
        variance_metrics = get_all_metrics(cycle_emg_data)
        self.variance_metrics = pd.DataFrame()
        self.variance_metrics["Bipolar combinations"] = self.combinations
        self.variance_metrics["Peak amplitude"] = variance_metrics[0]
        self.variance_metrics["Peak location"] = variance_metrics[1]
        self.variance_metrics["Area Under Curve"] = variance_metrics[2]

    def get_variance_rankings(self):
        max_amps, peak_loc, auc = self.variance_metrics
        amplitude_ranking = rank_values(max_amps)
        peak_loc_ranking = rank_values(peak_loc)
        auc_ranking = rank_values(auc)
        rankings = pd.DataFrame()
        rankings["Bipolar combinations"] = self.combinations
        rankings["Peak amplitude"] = amplitude_ranking
        rankings["Peak location"] = peak_loc_ranking
        rankings["Area Under Curve"] = auc_ranking
        self.rankings = rankings

    def get_preliminary_results(self):
        pass

    def get_agreement_ranking(self):
        agreement_ranking = np.sum(self.rankings.iloc[:, 1::].to_numpy, axis=1)
        agreement_ranking = rank_values(agreement_ranking)
        self.rankings["Agreement"] = agreement_ranking

    def train_on_all_combinations(self, window_size=250, batch_size=64, n_folds=5):
        formatted_data = NetworkFormatter(
            self.bipolar_data,
            self.label,
            fs=self.fs,
            window_size=window_size,
            test_size=0.05,
            batch_size=batch_size,
            n_folds=n_folds,
            step_size=1,
        )
        # record the important information from here
        model_training = TrainEval(
            formatted_data.training_data,
            formatted_data.testing_data,
            n_channels=len(self.bipolar_data),
            k_fold=True,
            model_name="all_combinations",
        )
        model_performance = model_training.model_performance
        if self.model_performances is None:
            self.model_performances = model_performance
        else:
            self.model_performances = pd.concat(
                [self.model_performances, model_performance], ignore_index=True
            )

    def train_on_reduced_patch_size(self):
        # Get the combination of channels required for this
        # select the relevant channel columns
        # train the model as before
        pass

    def train_on_randomly_sampled_channels(
        self, n_combinations=[3, 5, 7], n_randomisations=10
    ):
        # This will be mainly to see if it is possible to get
        pass
