from ._experiment import Experiment
from ._experiment import DEFAULT_DATASET_CONFIG, DEFAULT_RUN_CONFIG, DEFAULT_DP_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_WANDB_CONFIG
from src_refactored.trainer import StandardTrainer, DPTrainer
from opacus import PrivacyEngine
import wandb


class DefaultExperiment(Experiment):
    def __init__(self,
                 run_config=DEFAULT_RUN_CONFIG,
                 dp_config=DEFAULT_DP_CONFIG,
                 dataset_config=DEFAULT_DATASET_CONFIG,
                 model_config=DEFAULT_MODEL_CONFIG,
                 wandb_config=DEFAULT_WANDB_CONFIG,
                 percent_of_data_to_use=0.5):
        super().__init__(run_config, dp_config, dataset_config, model_config, wandb_config)
        self.percent_of_data_to_use = percent_of_data_to_use

    def run(self, train_loader, val_loader, test_loader, wandb_config=DEFAULT_WANDB_CONFIG):
        for seed in range(self.run_config["num_seeds"]):
            self.run_config["seed"] = self.run_config["initial_seed"] + seed

            # TODO calculate epochs
            self.run_config["epochs"] = 3
            # TODO run args
            model, optimizer, log_dir = self.prep_run()
            trainer = StandardTrainer(
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=self.run_config,
                log_dir=log_dir
            )
            model = trainer.train(model)
            trainer.test(model)
            wandb.finish()

    def run_DP(self,  train_loader, val_loader, test_loader):
        for seed in range(self.run_config["num_seeds"]):
            self.run_config["seed"] = self.run_config["initial_seed"] + seed
            self.dp_config["delta"] = 1 / len(train_loader.dataset)
            # TODO calculate epochs
            self.run_config["epochs"] = 3
            privacy_engine = PrivacyEngine(accountant="rdp")
            # TODO run args
            model, optimizer, log_dir = self.prep_run()
            model, optimizer, dp_train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=self.dp_config["epsilon"],
                target_delta=self.dp_config["delta"],
                max_grad_norm=self.dp_config["max_grad_norm"],
                epochs=self.run_config["epochs"]
            )
            trainer = DPTrainer(
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=self.run_config,
                log_dir=log_dir,
                privacy_engine=privacy_engine
            )
            model = trainer.train(model)
            trainer.test(model)
            wandb.finish()

    def custom_data_loading_hook(self, train_A, train_B):
        return train_A, train_B
