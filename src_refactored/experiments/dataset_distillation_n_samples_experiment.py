from ._experiment import Experiment
from ._experiment import DEFAULT_DATASET_CONFIG, DEFAULT_RUN_CONFIG, DEFAULT_DP_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_WANDB_CONFIG
from typing import Dict, Tuple, List
import wandb
from torch import Tensor, Generator
from src_refactored.datasets.data_manager import DataManager
from src_refactored.datasets.datasets import NormalDataset
from src_refactored.datasets.data_utils import get_load_fn, get_transforms
from torch.utils.data import DataLoader


class DatasetDistillationNSamplesExperiment(Experiment):
    def __init__(self,
                 run_config: Dict = DEFAULT_RUN_CONFIG,
                 dp_config: Dict = DEFAULT_DP_CONFIG,
                 dataset_config: Dict = DEFAULT_DATASET_CONFIG,
                 model_config: Dict = DEFAULT_MODEL_CONFIG,
                 wandb_config: Dict = DEFAULT_WANDB_CONFIG,
                 num_training_samples: int = 1,
                 ):
        super().__init__(run_config, dp_config, dataset_config, model_config, wandb_config)
        self.num_training_samples = num_training_samples

    def start_experiment(self, data_manager: DataManager, *args, **kwargs):
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(self.custom_data_loading_hook)
        train_data = train_loader.dataset.data
        train_labels = train_loader.dataset.labels
        train_meta = train_loader.dataset.meta
        n_samples_train_loaders = []
        prev = 0
        for i in range(self.num_training_samples, len(train_data), self.num_training_samples):
            temp_dataset = NormalDataset(
                train_data[prev:i],
                train_labels[prev:i],
                train_meta[prev:i],
                transform=get_transforms(self.dataset_config["dataset"]),
                load_fn=get_load_fn(self.dataset_config["dataset"]),
            )
            temp_dataloader = DataLoader(
                temp_dataset,
                batch_size=self.dataset_config["batch_size"],
                shuffle=True,
                num_workers=4,
                generator=Generator().manual_seed(2147483647))
            n_samples_train_loaders.append(temp_dataloader)
            prev = i
        for n_samples_train_loader in n_samples_train_loaders:
            for seed in range(self.run_config["num_seeds"]):
                if self.run_config["dp"]:
                    self._run_DP(n_samples_train_loader, val_loader, test_loader)
                else:
                    self._run(n_samples_train_loader, val_loader, test_loader)
                wandb.finish()
