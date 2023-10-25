from ._experiment import Experiment
from typing import Dict, Tuple, List
import wandb
import pandas as pd
import numpy as np
from src_refactored.datasets.data_manager import DataManager, ATTRIBUTE_MAPPINGS


class UpsamplingExperiment(Experiment):
    def __init__(self,
                 run_config: Dict,
                 dp_config: Dict,
                 dataset_config: Dict,
                 model_config: Dict,
                 wandb_config: Dict,
                 upsampling_strategy: Tuple[str, str, str] = ("even", "age")
                 ):
        super().__init__(run_config, dp_config, dataset_config, model_config, wandb_config)
        self.upsampling_strategy = upsampling_strategy

    def start_experiment(self, data_manager: DataManager, *args, **kwargs):
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(self.custom_data_loading_hook)
        for hidden_dim in self.hidden_dims:
            self.model_config["hidden_dims"] = hidden_dim
            for seed in range(self.run_config["num_seeds"]):
                if self.run_config["dp"]:
                    self._run_DP(train_loader, val_loader, test_loader)
                else:
                    self._run(train_loader, val_loader, test_loader)
                wandb.finish()

    def custom_data_loading_hook(self, train_A, train_B, *args, **kwargs):
        print("ATTENTION: custom data loading hook is used! Up-sampling training dataset.")
        if self.dataset_config["protected_attribute_percent"] in [0.0, 1.0]:
            raise ValueError("Cannot up-sample when one of the classes is at 100%.")
        num_add_samples = abs(len(train_A) - len(train_B))
        if len(train_A) > len(train_B):
            train_B = self.upsample_dataset(train_B, self.upsampling_strategy[0], num_add_samples)
        elif len(train_A) < len(train_B):
            train_A = self.upsample_dataset(train_A, self.upsampling_strategy[0], num_add_samples)
        else:
            raise ValueError("Number of samples for both classes is already equal.")
        print("Number of training samples for",
            f"{ATTRIBUTE_MAPPINGS[self.dataset_config['protected_attr']]['A']}/{ATTRIBUTE_MAPPINGS[self.dataset_config['protected_attr']]['B']}:",
            f"{len(train_A)}/{len(train_B)}")
        return train_A, train_B

    @staticmethod
    def upsample_dataset(data: pd.DataFrame, strategy: str, num_add_samples: int):
        n = len(data)
        if strategy.startswith("even"):
            replication_factor = (n + num_add_samples) / n
            data_new = data.loc[data.index.repeat(np.floor(replication_factor))]
            if replication_factor % 1 != 0:
                num_remaining_replications = int(np.rint((replication_factor % 1) * n))
                additional_samples = data.sample(n=num_remaining_replications, replace=False, random_state=42)
                data_new = pd.concat([data_new, additional_samples])
            data = data_new
        else:
            data = pd.concat([data, data.sample(n=num_add_samples, replace=True, random_state=42)])
        print("Up-sampling by", num_add_samples)
        return data
