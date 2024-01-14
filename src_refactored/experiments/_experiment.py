import torch
import os
import wandb
from abc import ABC, abstractmethod
from opacus.validators import ModuleValidator
from src_refactored.models.DeepSVDD.deepsvdd import DeepSVDD
from src_refactored.models.FAE.fae import FeatureReconstructor
from src_refactored.trainer import StandardTrainer, DPTrainer
from opacus import PrivacyEngine
from datetime import datetime
from src_refactored.utils.utils import seed_everything
from src_refactored.datasets.anomaly_dataset import AnomalyDataset
CURRENT_TIMESTAMP = str(datetime.strftime(datetime.now(), format="%Y-%m-%d %H:%M:%S"))


class Experiment(ABC):
    def __init__(self, run_config=None, dp_config=None, dataset_config=None, model_config=None, wandb_config=None):
        self.run_config = run_config
        self.dp_config = dp_config
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.wandb_config = wandb_config
        self.model_config["img_size"] = self.dataset_config["img_size"]
        if self.run_config["dp"]:
            self.run_config["num_steps"] = self.dp_config["num_steps"]
            self.run_config["batch_size"] = self.dp_config["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def _run(self, train_loader, val_loader, test_loader, **kwargs):
        seed_everything(self.run_config["seed"])
        self.run_config["epochs"] = self.steps_to_epochs(train_loader)
        model, optimizer, log_dir = self.prep_run(**kwargs)
        trainer = StandardTrainer(
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=self.run_config,
            log_dir=log_dir
        )
        model = trainer.train(model, **kwargs)
        trainer.test(model)

    def _run_DP(self, train_loader, val_loader, test_loader, **kwargs):
        seed_everything(self.run_config["seed"])
        self.dp_config["delta"] = 1 / len(train_loader.dataset)
        self.run_config["epochs"] = self.steps_to_epochs(train_loader)
        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, log_dir = self.prep_run(**kwargs)
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
            privacy_engine=privacy_engine
        )
        model = trainer.train(model, **kwargs)
        trainer.test(model)

    def start_experiment(self, data_manager: AnomalyDataset, *args, **kwargs):
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(self.custom_data_loading_hook)
        for seed in range(self.run_config["num_seeds"]):
            self.run_config["seed"] = self.run_config["initial_seed"] + seed
            if self.run_config["dp"]:
                self._run_DP(train_loader, val_loader, test_loader)
            else:
                self._run(train_loader, val_loader, test_loader)
            wandb.finish()

    def custom_data_loading_hook(self, train_A, train_B, *args, **kwargs):
        return train_A, train_B

    def _init_model(self):
        print("Initializing model...")
        if self.model_config["model_type"] == 'FAE':
            model = FeatureReconstructor(self.model_config)
        elif self.model_config["model_type"] == 'DeepSVDD':
            model = DeepSVDD(self.model_config)
        else:
            raise ValueError(f'Unknown model type {self.model_config["model_type"]}')
        model = model.to(self.device)

        # perform model surgery if DP is enabled
        if self.run_config["dp"]:
            model = ModuleValidator.fix(model)

        # Init optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.model_config["lr"],
            weight_decay=self.model_config["weight_decay"]
        )
        return model, optimizer

    def _construct_log_dir(self, timestamp, job_type_mod, group_name_mod):
        log_path = self.run_config["log_dir"]

        # job_type is always the same
        job_type = f"male_percent" if self.dataset_config["protected_attr"] == "sex" else "old_percent"
        # add protected_attr_percent
        job_type += "_" + str(self.dataset_config["protected_attr_percent"]).replace('.', '')
        job_type += f"_{job_type_mod}" if job_type_mod else ""
        # build log_dir and group_name
        group_name = f"{timestamp}-{self.model_config['model_type']}-{self.dataset_config['dataset'].upper()}-{self.dataset_config['protected_attr'].upper()}"
        group_name += f"-{group_name_mod}" if group_name_mod else ""
        group_name += f"-DP" if self.run_config['dp'] else "-noDP"
        log_path = os.path.join(log_path, group_name, job_type, f"seed_{self.run_config['seed']}")
        return log_path, group_name, job_type

    def _init_wandb(self, log_dir, group_name: str, job_type: str):
        log_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # os.makedirs(log_dir, exist_ok=True)
        run_name = "seed_" + str(self.run_config["seed"])
        config = {
                "run": self.run_config,
                "dataset": self.dataset_config,
                "model": self.model_config
        }
        if self.run_config["dp"]:
            config["DP"] = self.dp_config
        run = wandb.init(
            project=self.wandb_config["project"],
            config=config,
            dir=os.path.join(log_dir),
            group=group_name,
            tags=[],
            job_type=job_type,
            name=run_name
        )
        return run

    def prep_run(self, timestamp=CURRENT_TIMESTAMP, job_type_mod="", group_name_mod="", **kwargs):
        print(job_type_mod)
        model, optimizer = self._init_model()
        log_dir, group_name, job_type = self._construct_log_dir(timestamp, job_type_mod, group_name_mod)
        self._init_wandb(log_dir, group_name, job_type)
        return model, optimizer, log_dir

    def steps_to_epochs(self, train_loader):
        return self.run_config["num_steps"] // len(train_loader)


