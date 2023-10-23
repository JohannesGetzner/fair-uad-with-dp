from datasets.data_manager import DataManager
from experiments._experiment import Experiment
import os

if __name__ == '__main__':
    experiment = Experiment()
    datamanager = DataManager(experiment.dataset_config)
    experiment.run(datamanager)
