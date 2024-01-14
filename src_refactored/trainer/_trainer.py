from abc import ABC, abstractmethod
import torch
import wandb
import os
from collections import defaultdict
import pandas as pd
from src_refactored.utils.metrics import AvgDictMeter, build_metrics
from time import time
from src_refactored.utils.utils import get_remaining_time_as_str
from datetime import datetime
from torch import nn


class Trainer(ABC):
    def __init__(self,  optimizer, train_loader, val_loader, test_loader, config, log_dir, previous_steps):
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.log_dir = log_dir
        self.previous_steps = previous_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.privacy_engine = None

    @abstractmethod
    def train(self, model, **kwargs):
        pass

    @abstractmethod
    def train_step(self, model, x, loss_weights=None):
        pass

    def validate(self, model, step, log_imgs=False):
        i_step = 0
        device = next(model.parameters()).device
        x, y, meta = next(iter(self.val_loader))
        metrics = build_metrics(subgroup_names=list(x.keys()))
        losses = defaultdict(AvgDictMeter)
        imgs = defaultdict(list)
        anomaly_maps = defaultdict(list)

        for x, y, meta in self.val_loader:
            # x, y, anomaly_map: [b, 1, h, w]
            # Compute loss, anomaly map and anomaly score
            for i, k in enumerate(x.keys()):
                loss_dict, anomaly_map, anomaly_score = self.val_step(model, x[k], y[k])

                # Update metrics
                group = torch.tensor([i] * len(anomaly_score))
                metrics.update(group, anomaly_score, y[k])
                losses[k].add(loss_dict)
                imgs[k].append(x[k])
                if anomaly_map is not None:
                    anomaly_maps[k].append(anomaly_map)
                else:
                    log_imgs = False

            i_step += 1
            if i_step >= self.config["val_steps"]:
                break
        # Compute and flatten metrics and losses
        metrics_c = metrics.compute()
        losses_c = {k: v.compute() for k, v in losses.items()}
        losses_c = {f'{k}_{m}': v[m] for k, v in losses_c.items() for m in v.keys()}
        if log_imgs:
            imgs = {f"{k}_imgs": wandb.Image(torch.cat(v)[:self.config['num_imgs_log']]) for k, v in imgs.items()}
            anomaly_maps = {f"{k}_anomaly_maps": wandb.Image(torch.cat(v)[:self.config['num_imgs_log']]) for k, v in
                            anomaly_maps.items()}
            imgs_log = {**imgs, **anomaly_maps}
            wandb.log(imgs_log, step=step)

        # Compute metrics
        results = {**metrics_c, **losses_c}

        # Print validation results
        print("\nval results:")
        log_msg = "\n".join([f'{k}: {v:.4f}' for k, v in results.items()])
        log_msg += "\n"
        print(log_msg)

        # Save checkpoint
        if not self.config["debug"]:
            ckpt_name = os.path.join(self.log_dir, 'ckpt_last.pth')
            print(f'Saving checkpoint to {ckpt_name}')
            self.save_checkpoint(ckpt_name, model, step)
        return results

    def val_step(self, model, x, y):
        model.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            if self.config["dp"]:
                loss_dict = model._module.loss(x)
                anomaly_map, anomaly_score = model._module.predict_anomaly(x)
            else:
                loss_dict = model.loss(x)
                anomaly_map, anomaly_score = model.predict_anomaly(x)
        x = x.cpu()
        anomaly_score = anomaly_score.cpu() if anomaly_score is not None else None
        anomaly_map = anomaly_map.cpu() if anomaly_map is not None else None
        return loss_dict, anomaly_map, anomaly_score

    def test(self, model):
        print("Testing...")
        model.eval()
        x, y, meta = next(iter(self.test_loader))
        metrics = build_metrics(subgroup_names=list(x.keys()))
        losses = defaultdict(AvgDictMeter)
        anomaly_scores = []
        labels = []
        subgroup_names = []

        for x, y, meta in self.test_loader:
            # x, y, anomaly_map: [b, 1, h, w]
            # Compute loss, anomaly map and anomaly score
            for i, k in enumerate(x.keys()):
                loss_dict, _, anomaly_score = self.val_step(model, x[k], y[k])

                # Update metrics
                group = torch.tensor([i] * len(anomaly_score))
                metrics.update(group, anomaly_score, y[k])
                losses[k].add(loss_dict)

                # Store anomaly scores, labels, and subgroup names
                anomaly_scores.append(anomaly_score)
                labels.append(y[k])
                subgroup_names += [k] * len(anomaly_score)

        # Aggregate anomaly scores, labels, and subgroup names
        anomaly_scores = torch.cat(anomaly_scores)
        labels = torch.cat(labels)

        # Compute and flatten metrics and losses
        metrics_c = metrics.compute()
        losses_c = {k: v.compute() for k, v in losses.items()}
        losses_c = {f'{k}_{m}': v[m] for k, v in losses_c.items() for m in v.keys()}

        results = {**metrics_c, **losses_c}

        # Print validation results
        print("\nTest results:")
        log_msg = "\n".join([f'{k}: {v.mean():.4f}' for k, v in results.items()])
        log_msg += "\n"
        print(log_msg)

        # Write test results to wandb summary
        for k, v in results.items():
            wandb.run.summary[k] = v

        # Save test results to csv
        if not self.config["debug"]:
            csv_path = os.path.join(self.log_dir, 'test_results.csv')
            # create dataframe from dict, keys are the columns and values are single row
            metrics_c = {k: v.item() for k, v in metrics_c.items()}
            df = pd.DataFrame.from_dict(metrics_c, orient='index').T
            for k, v in self.config.items():
                df[k] = pd.Series([v])
            df.to_csv(csv_path, index=False)

        # Save anomaly scores and labels to csv
        if not self.config["debug"]:
            # os.makedirs(log_dir, exist_ok=True)
            csv_path = os.path.join(self.log_dir, 'anomaly_scores.csv')
            df = pd.DataFrame({'anomaly_score': anomaly_scores, 'label': labels, 'subgroup_name': subgroup_names})
            df.to_csv(csv_path, index=False)

    def log_training_progress(self, train_losses, i_step, i_epoch, t_start):
        train_results = train_losses.compute()
        # Print training loss
        log_msg = " - ".join([f'{k}: {v:.4f}' for k, v in train_results.items()])
        log_msg = f"Iteration {i_step} from {self.config['num_steps']} - " + log_msg
        # Elapsed time
        elapsed_time = datetime.utcfromtimestamp(time() - t_start)
        log_msg += f" - time: {elapsed_time.strftime('%d-%H:%M:%S')}s"
        # Estimate remaining time
        time_per_epoch = ((time() - t_start) / i_epoch) if i_epoch > 0 else time() - t_start
        remaining_time = (self.config["epochs"] - i_epoch) * time_per_epoch
        log_msg += f" - remaining time: {get_remaining_time_as_str(remaining_time)}"
        print(log_msg)
        # Log to w&b or tensorboard
        wandb.log({f'train/{k}': v for k, v in train_results.items()}, step=i_step)

    def save_checkpoint(self, path: str, model: nn.Module, step: int):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        checkpoint = {'model': model.state_dict(), 'config': self.config, 'step': step,
                      "optimizer": self.optimizer.state_dict()}
        if self.config["dp"]:
            checkpoint["accountant"] = self.privacy_engine.accountant
            checkpoint["optimizer"] = self.optimizer.original_optimizer.state_dict()
        torch.save(checkpoint, path)
