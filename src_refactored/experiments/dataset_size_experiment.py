from ._experiment import Experiment
from typing import Dict
from src_refactored.datasets.anomaly_dataset import ATTRIBUTE_MAPPINGS, AnomalyDataset
import wandb

class DataSetSizeExperiment(Experiment):
    def __init__(self,
                 run_config: Dict,
                 dp_config: Dict,
                 dataset_config: Dict,
                 model_config: Dict,
                 wandb_config: Dict,
                 percent_of_data_to_use=0.5
                 ):
        super().__init__(run_config, dp_config, dataset_config, model_config, wandb_config)
        self.percent_of_data_to_use = percent_of_data_to_use

    def start_experiment(self, data_manager: AnomalyDataset, *args, **kwargs):
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(self.custom_data_loading_hook)
        job_type_mod = f"data-set-size={self.percent_of_data_to_use}"
        for seed in range(self.run_config["num_seeds"]):
            self.run_config["seed"] = self.run_config["initial_seed"] + seed
            if self.run_config["dp"]:
                self._run_DP(train_loader, val_loader, test_loader, job_type_mod=job_type_mod, group_name_mod=kwargs["group_name_mod"])
            else:
                self._run(train_loader, val_loader, test_loader, job_type_mod=job_type_mod, group_name_mod=kwargs["group_name_mod"])
            wandb.finish()

    def custom_data_loading_hook(self, train_A, train_B, *args, **kwargs):
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