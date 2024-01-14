import os
import sys
import yaml
from argparse import ArgumentParser, BooleanOptionalAction
from datasets.cxr14 import CXR14AnomalyDataset
from datasets.rsna import RsnaAnomalyDataset
from experiments import DEFAULT_DICTS
import experiments


def get_args_from_arg_group(arg_group):
    args = {}
    for action in arg_group._group_actions:
        if getattr(RUN_PARAMS, action.dest) != action.default:
            args[action.dest] = getattr(RUN_PARAMS, action.dest)
    return args


def load_configurations(file_path:str):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


EXPERIMENT_MAP = {
    'default': experiments.Experiment,
    'fine_tuning': experiments.FineTuningExperiment,
    'dataset_size': experiments.DataSetSizeExperiment,
    'loss_weighing': experiments.LossWeighingExperiment,
    'model_size': experiments.ModelSizeExperiment,
    'upsampling': experiments.UpsamplingExperiment,
    'core_set_selection': experiments.CoreSetSelectionExperiment,
}

parser = ArgumentParser()
parser.add_argument('--experiment', default="default", type=str, choices=EXPERIMENT_MAP.keys())
parser.add_argument('--dataset', default=None, type=str, choices=['cxr14', 'rsna'])
parser.add_argument('--protected_attr', default=None, type=str, choices=['age', 'sex'])
parser.add_argument('--dp', default=False, action=BooleanOptionalAction)
parser.add_argument('--debug', default=False, action=BooleanOptionalAction)
# override experiment settings
override_args = parser.add_argument_group('override')
override_args.add_argument('--batch_size', default=None, type=int)
override_args.add_argument('--initial_seed', default=None, type=int)
override_args.add_argument('--num_steps', default=None, type=int)

# fine-tuning experiment settings
fine_tuning_args = parser.add_argument_group('fine_tuning')
fine_tuning_args.add_argument('--fine_tuning_epsilon', default=None, type=float)
fine_tuning_args.add_argument('--fine_tuning_steps', default=None, type=float)
fine_tuning_args.add_argument('--fine_tuning_protected_attr_percent', default=None, type=float)

# dataset-size experiment settings
dataset_size_args = parser.add_argument_group('dataset_size')
dataset_size_args.add_argument('--percent_of_data_to_use', default=None, type=float)

# loss-weight experiment settings
loss_weight_args = parser.add_argument_group('loss_weighing')
loss_weight_args.add_argument('--loss_weight', default=None, type=float)
loss_weight_args.add_argument('--pv_to_weigh', default=None, type=str, nargs='+')

# model size experiment settings
model_size_args = parser.add_argument_group('model_size')
model_size_args.add_argument('--reduce_hidden_dims', default=False, type=bool, action=BooleanOptionalAction)

# upsampling experiment settings
upsampling_args = parser.add_argument_group('upsampling')
upsampling_args.add_argument('--upsampling_strategy', default=None, type=str, nargs='+')

# dataset distillation experiment settings
core_set_selection_args = parser.add_argument_group('core_set_selection')
core_set_selection_args.add_argument('--num_training_samples', default=None, type=int)
RUN_PARAMS = parser.parse_args()

if __name__ == '__main__':
    exp_key = RUN_PARAMS.experiment
    arg_group_matches = [arg_group for arg_group in parser._action_groups if arg_group.title == exp_key]
    if len(arg_group_matches) == 0:
        exp_args = {}
    else:
        exp_args = get_args_from_arg_group(arg_group_matches[0])
    base_config = load_configurations(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yml'))
    if RUN_PARAMS.dp:
        base_config = base_config['dp']
    elif RUN_PARAMS.debug:
        base_config = base_config['debug']
    else:
        base_config = base_config['normal']
    override_params = get_args_from_arg_group(override_args)
    base_config.update(override_params)
    if RUN_PARAMS.dataset: base_config['dataset'] = RUN_PARAMS.dataset
    if RUN_PARAMS.protected_attr: base_config['protected_attr'] = RUN_PARAMS.protected_attr

    config_dicts = DEFAULT_DICTS.copy()
    for dict_name, default_config in config_dicts.items():
        for key, value in base_config.items():
            if key in default_config.keys():
                default_config[key] = value

    experiment = EXPERIMENT_MAP[exp_key](**config_dicts, **exp_args)
    if experiment.dataset_config["dataset"] == "rsna":
        dataset = RsnaAnomalyDataset(experiment.dataset_config)
    elif experiment.dataset_config["dataset"] == "cxr14":
        dataset = CXR14AnomalyDataset(experiment.dataset_config)
    else:
        raise ValueError("Dataset not supported")
    group_name_mod = f"experiment={exp_key}"
    experiment.start_experiment(dataset, group_name_mod=group_name_mod)

















