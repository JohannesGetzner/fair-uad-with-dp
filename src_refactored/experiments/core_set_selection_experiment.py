from ._experiment import Experiment
from typing import Dict
import wandb
from torch import Tensor, Generator
from datasets.data_utils import get_transforms, get_load_fn
from src_refactored.datasets.anomaly_dataset import AnomalyDataset, MEMMAP_NormalDataset
from torch.utils.data import DataLoader


class CoreSetSelectionExperiment(Experiment):
    def __init__(self,
                 run_config: Dict,
                 dp_config: Dict,
                 dataset_config: Dict,
                 model_config: Dict,
                 wandb_config: Dict,
                 num_training_samples: int = 1,
                 cutoff: int = 1000
                 ):
        super().__init__(run_config, dp_config, dataset_config, model_config, wandb_config)
        self.num_training_samples = num_training_samples
        self.cutoff = cutoff

    def start_experiment(self, data_manager: AnomalyDataset, *args, **kwargs):
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(self.custom_data_loading_hook)
        train_data = train_loader.dataset.data
        train_labels = train_loader.dataset.labels
        train_meta = train_loader.dataset.meta
        train_idx_map = train_loader.dataset.index_mapping
        train_filenames = train_loader.dataset.filenames
        n_samples_train_loaders = []
        transform = get_transforms(self.dataset_config["img_size"])
        load_fn = get_load_fn(self.dataset_config["dataset"])
        for i in range(self.num_training_samples, len(train_idx_map), self.num_training_samples):
            temp_dataset = MEMMAP_NormalDataset(
                train_data,
                train_labels,
                train_meta,
                transform=transform,
                index_mapping=train_idx_map,
                load_fn=load_fn,
                filenames=train_filenames
            )
            temp_dataloader = DataLoader(
                temp_dataset,
                batch_size=self.dataset_config["batch_size"],
                shuffle=True,
                num_workers=4,
                generator=Generator().manual_seed(2147483647)
            )
            n_samples_train_loaders.append(temp_dataloader)
            if i > self.cutoff:
                break
        for idx, n_samples_train_loader in enumerate(n_samples_train_loaders):
            job_type_name = f"train_loader_idx={idx}"
            for seed in range(self.run_config["num_seeds"]):
                self.run_config["seed"] = self.run_config["initial_seed"] + seed
                if self.run_config["dp"]:
                    self._run_DP(n_samples_train_loader, val_loader, test_loader, job_type_mod=job_type_name, group_name_mod=kwargs["group_name_mod"])
                else:
                    self._run(n_samples_train_loader, val_loader, test_loader, job_type_mod=job_type_name, group_name_mod=kwargs["group_name_mod"])
                wandb.finish()
