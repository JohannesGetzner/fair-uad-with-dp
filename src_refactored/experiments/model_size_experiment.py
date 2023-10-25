from ._experiment import Experiment
from typing import Dict, Tuple, List
import wandb
from src_refactored.datasets.data_manager import DataManager


class ModelSizeExperiment(Experiment):
    def __init__(self,
                 run_config: Dict,
                 dp_config: Dict,
                 dataset_config: Dict,
                 model_config: Dict,
                 wandb_config: Dict,
                 hidden_dims: List[List[int]] = [(100, 150, 200, 300)],
                 ):
        super().__init__(run_config, dp_config, dataset_config, model_config, wandb_config)
        self.hidden_dims = hidden_dims

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
