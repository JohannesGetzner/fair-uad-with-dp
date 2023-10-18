from datasets.data_manager import DataManager
from experiments.default_experiment import DefaultExperiment
import os

if __name__ == '__main__':
    experiment = DefaultExperiment()
    datamanager = DataManager(experiment.dataset_config)
    train_dataloader, val_dataloader, test_dataloader = datamanager.get_dataloaders(experiment)
    experiment.run(train_dataloader, val_dataloader, test_dataloader)
