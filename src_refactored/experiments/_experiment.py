from abc import ABC, abstractmethod
from typing import Dict

DEFAULT_RUN_CONFIG = {
    "initial_seed": 1,
    "num_seeds": 5,
    "debug": False,
    # training loop
    "val_steps": 50,
    "num_steps": 8000,
    "val_frequency": 200,
    # logging
    "log_frequency": 100,
    "log_img_freq": 1000,
    "num_imgs_log": 8,
    # dp
    "dp": False
}

DEFAULT_DATASET_CONFIG = {
    "dataset": "rsna",
    "protected_attr": "age",
    "protected_attr_percent": 0.9,
    "batch_size": 32,
    "img_size": 128,
    "random_state": 42
}

DEFAULT_DP_CONFIG = {
    "epsilon": 8,
    "delta": None,
    "max_grad_norm": 0.01,
    "num_steps": 45000,
    "batch_size": 32,
}

class Experiment(ABC):
    def __init__(self, run_config=None, dp_config=None, dataset_config=None):
        self.run_config = DEFAULT_RUN_CONFIG.copy()
        run_config.update(run_config)
        self.dp_config = DEFAULT_DP_CONFIG.copy()
        self.dp_config.update(dp_config)
        self.dataset_config = DEFAULT_DATASET_CONFIG.copy()
        self.dataset_config.update(dataset_config)
        if self.run_config["dp"]:
            self.run_config["num_steps"] = self.dp_config["num_steps"]
            self.run_config["batch_size"] = self.dp_config["batch_size"]

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def run_DP(self):
        pass

    @abstractmethod
    def custom_data_loading_hook(self, *args, **kwargs):
        pass