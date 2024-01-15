from ._experiment import Experiment
from typing import Dict
import wandb
from src.datasets.anomaly_dataset import AnomalyDataset
from src.utils.utils import seed_everything
from opacus import PrivacyEngine
from src.trainer import StandardTrainer, DPTrainer



class FineTuningExperiment(Experiment):
    def __init__(self,
                 run_config: Dict,
                 dp_config: Dict,
                 dataset_config: Dict,
                 model_config: Dict,
                 wandb_config: Dict,
                 fine_tuning_epsilon=3,
                 fine_tuning_steps=None,
                 fine_tuning_protected_attr_percent=0.0
                 ):
        super().__init__(run_config, dp_config, dataset_config, model_config, wandb_config)
        self.fine_tuning_epsilon = fine_tuning_epsilon
        self.fine_tuning_steps = fine_tuning_steps if fine_tuning_steps is not None else int(run_config["num_steps"] // 3)
        self.fine_tuning_dataset_config = self.dataset_config.copy()
        self.fine_tuning_dataset_config["protected_attr_percent"] = fine_tuning_protected_attr_percent
        # also add here because this is what gets logged
        self.dataset_config["fine_tuning_protected_attr_percent"] = fine_tuning_protected_attr_percent
        self.base_model = None
        self.previous_steps = 0

    def start_experiment(self, data_manager: AnomalyDataset, *args, **kwargs):
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(self.custom_data_loading_hook)
        # save normal steps
        self.run_config["initial_steps"] = self.run_config["num_steps"]
        for seed in range(self.run_config["num_seeds"]):
            self.run_config["seed"] = self.run_config["initial_seed"] + seed
            if self.run_config["dp"]:
                self.dp_config["initial_epsilon"] = self.dp_config["epsilon"]
                self.dp_config["epsilon"] = self.dp_config["epsilon"] - self.fine_tuning_epsilon
                self._run_DP(train_loader, val_loader, test_loader, group_name_mod=kwargs["group_name_mod"])
            else:
                self._run(train_loader, val_loader, test_loader, group_name_mod=kwargs["group_name_mod"])
            # run fine-tuning
            print("Starting fine-tuning...")
            self.run_config["num_steps"] = self.run_config["num_steps"] + self.fine_tuning_steps
            data_manager.config = self.fine_tuning_dataset_config
            fine_tuning_train_loader, _, _ = data_manager.get_dataloaders(self.custom_data_loading_hook)
            if self.run_config["dp"]:
                self.dp_config["epsilon"] = self.fine_tuning_epsilon
                self._run_DP(fine_tuning_train_loader, val_loader, test_loader)
            else:
                self._run(fine_tuning_train_loader, val_loader, test_loader)
            wandb.finish()

    def _run(self, train_loader, val_loader, test_loader, **kwargs):
        seed_everything(self.run_config["seed"])
        self.run_config["epochs"] = self.steps_to_epochs(train_loader)
        model, optimizer, log_dir = self.prep_run()
        if self.base_model is not None:
            model = self.base_model
        trainer = StandardTrainer(
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=self.run_config,
            log_dir=log_dir,
            previous_steps=self.previous_steps if self.base_model else 0
        )
        model = trainer.train(model, **kwargs)
        trainer.test(model)
        if self.base_model is None:
            self.base_model = model
            self.previous_steps = trainer.previous_steps

    def _run_DP(self, train_loader, val_loader, test_loader, **kwargs):
        seed_everything(self.run_config["seed"])
        self.dp_config["delta"] = 1 / len(train_loader.dataset)
        self.run_config["epochs"] = self.steps_to_epochs(train_loader)
        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, log_dir = self.prep_run()
        if self.base_model is not None:
            model = self.base_model
        model, optimizer, dp_train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=self.dp_config["epsilon"],
            target_delta=self.dp_config["delta"],
            max_grad_norm=self.dp_config["max_grad_norm"],
            epochs=self.run_config["epochs"])
        trainer = DPTrainer(
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=self.run_config,
            log_dir=log_dir,
            privacy_engine=privacy_engine,
            previous_steps=self.previous_steps if self.base_model else 0
        )
        model = trainer.train(model, **kwargs)
        trainer.test(model)
        if self.base_model is None:
            self.base_model = model
            self.previous_steps = trainer.previous_steps


    def steps_to_epochs(self, train_loader):
        if self.base_model is None:
            epochs =  self.run_config["num_steps"] // len(train_loader)
        else:
            epochs = self.fine_tuning_steps // len(train_loader)
        assert epochs > 0, f"Number of epochs is {epochs}. Increase your steps or fine-tuning steps."
        return epochs