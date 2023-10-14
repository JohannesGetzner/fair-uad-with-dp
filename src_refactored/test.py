from src_refactored.datasets.DataManager import DataManager
from src_refactored.experiments.DatasetSizeExperiment import DataSetSizeExperiment

if __name__ == '__main__':
    experiment = DataSetSizeExperiment()
    datamanager = DataManager(experiment.dataset_config)
    datamanager.get_dataloaders(experiment)