from ._experiment import Experiment
from typing import Dict
import wandb
from src_refactored.datasets.data_manager import DataManager


class FineTuningExperiment(Experiment):
    def __init__(self,
                 run_config: Dict,
                 dp_config: Dict,
                 dataset_config: Dict,
                 model_config: Dict,
                 wandb_config: Dict,
                 fine_tuning_epsilon=3,
                 fine_tuning_steps=None,
                 fine_tuning_protected_attr_percent=0.0
                 ):
        super().__init__(run_config, dp_config, dataset_config, model_config, wandb_config)
        self.fine_tuning_epsilon = fine_tuning_epsilon
        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_dataset_config = self.dataset_config.copy()
        self.fine_tuning_dataset_config["protected_attr_percent"] = fine_tuning_protected_attr_percent
        # also add here because this is what gets logged
        self.dataset_config["fine_tuning_protected_attr_percent"] = fine_tuning_protected_attr_percent

    def start_experiment(self, data_manager: DataManager, *args, **kwargs):
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(self.custom_data_loading_hook)
        # save normal steps
        self.run_config["initial_steps"] = self.run_config["num_steps"]
        for seed in range(self.run_config["num_seeds"]):
            self.run_config["seed"] = self.run_config["initial_seed"] + seed
            if self.run_config["dp"]:
                self.dp_config["initial_epsilon"] = self.dp_config["epsilon"]
                self.dp_config["epsilon"] = self.dp_config["epsilon"] - self.fine_tuning_epsilon
                self._run_DP(train_loader, val_loader, test_loader)
            else:
                self._run(train_loader, val_loader, test_loader)
            # run fine-tuning
            print("Starting fine-tuning...")
            self.run_config["num_steps"] = self.fine_tuning_steps
            fine_tuning_data_manager = DataManager(self.fine_tuning_dataset_config)
            fine_tuning_train_loader, _, _ = fine_tuning_data_manager.get_dataloaders(self.custom_data_loading_hook)
            if self.run_config["dp"]:
                self.dp_config["epsilon"] = self.fine_tuning_epsilon
                self._run_DP(fine_tuning_train_loader, val_loader, test_loader)
            else:
                self._run(fine_tuning_train_loader, val_loader, test_loader)
            wandb.finish()
