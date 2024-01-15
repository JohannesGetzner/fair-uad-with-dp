from ._experiment import Experiment
from typing import Dict, Tuple, List
import wandb
from src.datasets.anomaly_dataset import AnomalyDataset

class ModelSizeExperiment(Experiment):
    def __init__(self,
                 run_config: Dict,
                 dp_config: Dict,
                 dataset_config: Dict,
                 model_config: Dict,
                 wandb_config: Dict,
                 reduce_hidden_dims: bool = False,
                 ):
        super().__init__(run_config, dp_config, dataset_config, model_config, wandb_config)
        self.reduce_hidden_dims = reduce_hidden_dims
        self.hidden_dims_variants = self.create_hidden_dims(self.model_config["hidden_dims"])

    def start_experiment(self, data_manager: AnomalyDataset, *args, **kwargs):
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(self.custom_data_loading_hook)
        for hidden_dims in self.hidden_dims_variants:
            self.model_config["hidden_dims"] = hidden_dims
            job_type_mod = f"hidden_dims={hidden_dims}"
            for seed in range(self.run_config["num_seeds"]):
                self.run_config["seed"] = self.run_config["initial_seed"] + seed
                if self.run_config["dp"]:
                    self._run_DP(train_loader, val_loader, test_loader, job_type_mod=job_type_mod, group_name_mod=kwargs["group_name_mod"])
                else:
                    self._run(train_loader, val_loader, test_loader, job_type_mod=job_type_mod, group_name_mod=kwargs["group_name_mod"])
                wandb.finish()

    def create_hidden_dims(self, hidden_dims: List[int]):
        hidden_dims_variants = []
        if self.reduce_hidden_dims:
            for i in range(len(hidden_dims)):
                hidden_dims_variants.append(hidden_dims[:i+1])
        return hidden_dims_variants

