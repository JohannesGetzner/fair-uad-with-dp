import yaml
import os
with open(os.path.join(os.path.dirname(__file__), "default_experiment_settings.yml"), 'r') as f:
    DEFAULT_DICTS = yaml.safe_load(f)
from ._experiment import Experiment
from .dataset_size_experiment import DataSetSizeExperiment
from .dataset_distillation_n_samples_experiment import DatasetDistillationNSamplesExperiment
from .finetuning_experiment import FineTuningExperiment
from .loss_weighing_experiment import LossWeighingExperiment
from .model_size_experiment import ModelSizeExperiment
from .upsampling_experiment import UpsamplingExperiment
