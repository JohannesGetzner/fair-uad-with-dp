from ._experiment import Experiment
from ._experiment import DEFAULT_DATASET_CONFIG, DEFAULT_RUN_CONFIG, DEFAULT_DP_CONFIG
from typing import Dict
from src_refactored.datasets.DataManager import ATTRIBUTE_MAPPINGS

class DataSetSizeExperiment(Experiment):
    def __init__(self, run_config: Dict = DEFAULT_RUN_CONFIG, dp_config: Dict = DEFAULT_DP_CONFIG, dataset_config: Dict = DEFAULT_DATASET_CONFIG, percent_of_data_to_use=0.5):
        super().__init__(run_config, dp_config, dataset_config)
        self.percent_of_data_to_use = percent_of_data_to_use

    def run(self):
        pass

    def run_DP(self):
        pass

    def custom_data_loading_hook(self, train_A, train_B):
        print(f"ATTENTION: custom data loading hook is used! Reducing training dataset size to {self.percent_of_data_to_use}")
        train_A = train_A.sample(
            frac=self.percent_of_data_to_use,
            random_state=self.dataset_config["random_state"],
            replace=False
        )
        train_B = train_B.sample(
            frac=self.percent_of_data_to_use,
            random_state=self.dataset_config["random_state"],
            replace=False
        )
        print("New number of training samples for",
            f"{ATTRIBUTE_MAPPINGS[self.dataset_config['protected_attr']]['A']}/{ATTRIBUTE_MAPPINGS[self.dataset_config['protected_attr']]['B']}:",
            f"{len(train_A)}/{len(train_B)}")
        return train_A, train_B