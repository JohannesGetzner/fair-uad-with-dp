import os
import random
from argparse import Namespace
from numbers import Number
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def save_checkpoint(path: str, model: nn.Module, step: int, config: Dict):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    checkpoint = {
        'model': model.state_dict(),
        'config': config,
        'step': step
    }
    torch.save(checkpoint, path)


class TensorboardLogger(SummaryWriter):
    def __init__(
        self,
        log_dir: str = None,
        config: Namespace = None,
        enabled: bool = True,
        comment: str = '',
        purge_step: int = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = ''
    ):
        self.enabled = enabled
        if self.enabled:
            super().__init__(
                log_dir=log_dir,
                comment=comment,
                purge_step=purge_step,
                max_queue=max_queue,
                flush_secs=flush_secs,
                filename_suffix=filename_suffix
            )
        else:
            return

        # Add config
        if config is not None:
            self.add_hparams(
                {k: v for k, v in vars(config).items() if isinstance(v, (int, float, str, bool, torch.Tensor))},
                {}
            )

    def log(self, data: Dict[str, Any], step: int) -> None:
        """Log each entry in data as its corresponding data type"""
        if self.enabled:
            for k, v in data.items():
                is_array = isinstance(v, np.ndarray) or isinstance(v, torch.Tensor)
                # Scalars
                if isinstance(v, Number) or (is_array and len(v.shape) == 0):
                    self.add_scalar(k, v, step)

                # Images
                elif is_array and len(v.shape) >= 3:
                    if len(v.shape) == 3:
                        self.add_image(k, v, step)
                    elif len(v.shape) == 4:
                        self.add_images(k, v, step)
                    else:
                        raise ValueError(f'Unsupported image shape: {v.shape}')

                else:
                    raise ValueError(f'Unsupported data type: {type(v)}')
