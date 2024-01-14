import torch
import wandb
from src_refactored.utils.metrics import AvgDictMeter
from time import time
from ._trainer import Trainer

class StandardTrainer(Trainer):
    def __init__(self, optimizer, train_loader, val_loader, test_loader, config, log_dir, previous_steps = 0):
        super().__init__(optimizer, train_loader, val_loader, test_loader, config, log_dir, previous_steps)

    def train(self, model, **kwargs):
        print('Starting training...')
        i_step = self.previous_steps
        i_epoch = 0
        train_losses = AvgDictMeter()
        t_start = time()
        log_imgs = False
        
        while True:
            model.train()
            for x, y, meta in self.train_loader:
                i_step += 1
                x = x.to(self.device)
                y = y.to(self.device)
                if "loss_weights" in kwargs.keys():
                    per_sample_loss_weights = torch.where(
                        meta == 0, 
                        kwargs["loss_weights"][0], 
                        kwargs["loss_weights"][1]
                    ).to(self.device)
                    loss_dict = self.train_step(model, x, per_sample_loss_weights)
                else:
                    loss_dict = self.train_step(model, x)
                train_losses.add(loss_dict)
                if i_step % self.config["log_frequency"] == 0:
                    self.log_training_progress(train_losses, i_step, i_epoch, t_start)
                    train_losses.reset()

                # validation
                if i_step % self.config["val_frequency"] == 0:
                    log_imgs = i_step % self.config["log_img_freq"] == 0 and self.config["num_imgs_log"] != 0
                    val_results = self.validate(model, i_step, log_imgs)
                    # Log to w&b
                    wandb.log(val_results, step=i_step)

            i_epoch += 1
            if i_epoch % 10 == 0:
                print(f"Finished epoch {i_epoch}/{self.config['epochs']}, ({i_step} iterations)")
            if i_epoch >= self.config["epochs"]:
                print(f"Reached {self.config['epochs']} epoch(s).Finished training.")
                # Final validation
                print("Final validation...")
                self.validate(model, i_step, log_imgs)
                self.previous_steps = i_step
                return model

    def train_step(self, model, x, loss_weights=None):
        model.train()
        self.optimizer.zero_grad()
        loss_dict = model.loss(x, per_sample_loss_weights=loss_weights)
        loss = loss_dict['loss']
        loss.backward()
        self.optimizer.step()
        return loss_dict
