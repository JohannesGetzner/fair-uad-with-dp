from src_refactored.datasets.data_manager import DataManager
from src_refactored.experiments.dataset_size_experiment import DataSetSizeExperiment

if __name__ == '__main__':
    experiment = DataSetSizeExperiment()
    datamanager = DataManager(experiment.dataset_config)
    datamanager.get_dataloaders(experiment)