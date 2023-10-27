from ._experiment import Experiment
from typing import Dict, Tuple
import wandb
from src_refactored.datasets.data_manager import DataManager


class LossWeighingExperiment(Experiment):
    def __init__(self,
                 run_config: Dict,
                 dp_config: Dict,
                 dataset_config: Dict,
                 model_config: Dict,
                 wandb_config: Dict,
                 loss_weight: float = 1.0,
                 pv_to_weigh: Tuple[str, str] = ("age", "old")
                 ):
        super().__init__(run_config, dp_config, dataset_config, model_config, wandb_config)
        self.loss_weight = loss_weight
        self.pv_to_weigh = pv_to_weigh

    def start_experiment(self, data_manager: DataManager, *args, **kwargs):
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(self.custom_data_loading_hook)
        if self.pv_to_weigh[0] == "age":
            if self.pv_to_weigh[1] == "old":
                loss_weights = (self.loss_weight, 1.0)
            else:
                loss_weights = (1.0, self.loss_weight)
        elif self.pv_to_weigh[0] == "sex":
            if self.pv_to_weigh[1] == "male":
                loss_weights = (self.loss_weight, 1.0)
            else:
                loss_weights = (1.0, self.loss_weight)
        else:
            raise ValueError(f"Unknown protected variable {self.pv_to_weigh[0]}")

        for seed in range(self.run_config["num_seeds"]):
            if self.run_config["dp"]:
                self._run_DP(train_loader, val_loader, test_loader, loss_weights=loss_weights)
            else:
                self._run(train_loader, val_loader, test_loader, loss_weights=loss_weights)
            wandb.finish()