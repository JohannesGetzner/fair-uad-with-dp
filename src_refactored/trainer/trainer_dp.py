import torch
import wandb
from opacus.utils.batch_memory_manager import BatchMemoryManager
from src.utils.metrics import AvgDictMeter
from time import time
from ._trainer import Trainer


class DPTrainer(Trainer):
    def __init__(self, optimizer, train_loader, val_loader, test_loader, config, log_dir, privacy_engine):
        super().__init__(optimizer, train_loader, val_loader, test_loader, config, log_dir)
        self.privacy_engine = privacy_engine

    def train(self, model, **kwargs):
        print('Starting training...')
        i_step = self.previous_step
        i_epoch = 0
        train_losses = AvgDictMeter()
        t_start = time()
        log_imgs = False

        while True:
            with BatchMemoryManager(
                    data_loader=self.train_loader,
                    max_physical_batch_size=self.config["max_physical_batch_size"],
                    optimizer=self.optimizer
            ) as new_train_loader:
                model.train()
                for x, y, meta in new_train_loader:
                    i_step += 1
                    x = x.to(self.device)
                    y = y.to(self.device)
                    if "loss_weights" in kwargs.keys():
                        per_sample_loss_weights = torch.where(meta == 0, kwargs["loss_weights"][0],
                            kwargs["loss_weights"][1]).to(self.device)
                        loss_dict = self.train_step(model, x, per_sample_loss_weights)
                    else:
                        loss_dict = self.train_step(model, x)
                    train_losses.add(loss_dict)
                    if i_step % self.config["log_frequency"] == 0:
                        self.log_training_progress(train_losses, i_step, i_epoch, t_start)
                        train_losses.reset()

                    # validation
                    eps = self.privacy_engine.get_epsilon(self.config.delta)
                    if i_step % self.config["val_frequency"] == 0:
                        log_imgs = i_step % self.config["log_img_freq"] == 0 and self.config["num_imgs_log"] != 0
                        val_results = self.validate(model, i_step, log_imgs)
                        print(f"ɛ: {eps:.2f} (target: {self.config['epsilon']})")
                        val_results['epsilon'] = eps
                        # Log to w&b
                        wandb.log(val_results, step=i_step)

                    if eps >= self.config["epsilon"]:
                        print(f'Reached maximum ɛ {eps}/{self.config["epsilon"]}.', 'Finished training.')
                        # Final validation
                        print("Final validation...")
                        self.validate(model, i_step, log_imgs,)
                        return model

                i_epoch += 1
                if i_epoch % 100 == 0:
                    print(f"Finished epoch {i_epoch}/{self.config['epochs']}, ({i_step} iterations)")
                if i_epoch >= self.config["epochs"]:
                    print(f"Reached {self.config['epochs']} epochs.', 'Finished training.")
                    print(f"ɛ: {eps:.2f} (target: {self.config['epsilon']})")
                    # Final validation
                    print("Final validation...")
                    self.validate(model, i_step, log_imgs)
                    return model, i_step

    def train_step(self, model, x, loss_weights=None):
        model.train()
        self.optimizer.zero_grad()
        loss_dict = model.loss(x, per_sample_loss_weights=loss_weights)
        loss = loss_dict['loss']
        loss.backward()
        self.optimizer.step()
        return loss_dict
