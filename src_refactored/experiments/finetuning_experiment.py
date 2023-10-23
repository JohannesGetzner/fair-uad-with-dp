from ._experiment import Experiment
from ._experiment import DEFAULT_DATASET_CONFIG, DEFAULT_RUN_CONFIG, DEFAULT_DP_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_WANDB_CONFIG
from typing import Dict
import wandb
from src_refactored.datasets.data_manager import DataManager


class FineTuningExperiment(Experiment):
    def __init__(self,
                 run_config: Dict = DEFAULT_RUN_CONFIG,
                 dp_config: Dict = DEFAULT_DP_CONFIG,
                 dataset_config: Dict = DEFAULT_DATASET_CONFIG,
                 model_config: Dict = DEFAULT_MODEL_CONFIG,
                 wandb_config: Dict = DEFAULT_WANDB_CONFIG,
                 fine_tuning_epsilon=3,
                 fine_tuning_steps=None,
                 fine_tuning_protected_attr_percent=0.0
                 ):
        super().__init__(run_config, dp_config, dataset_config)
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
