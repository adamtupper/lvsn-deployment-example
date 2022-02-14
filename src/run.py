"""
Train ResNet-18 models on CIFAR-10 using varying levels of supervision to create
some baselines for later comparisons.

See params.yaml for hyperparameter settings.
"""
import argparse
import sys

import pytorch_lightning as pl
import torch
import yaml
from lightning_modules import LightningResNet18
from mlutils.lightning.cifar import PartiallyLabelledCIFARDataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.utilities.seed import seed_everything
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback


def parse_args(args: list) -> argparse.Namespace:
    """Parse command line parameters.

    :param args: command line parameters as list of strings (for example
        ``["--help"]``).
    :return: command line parameters namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the models for this experiment."
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--dataset-path",
        default="/home-local2/adtup.extra.nobkp/data",
        help="the path to the dataset",
        type=str,
    )
    parser.add_argument(
        "--cpus-per-trial",
        default=1,
        help="the number of CPU cores to use per trial",
        type=int,
    )
    parser.add_argument(
        "--project-name",
        help="the name of the Weights and Biases project to save the results",
        required=True,
        type=str,
    )

    return parser.parse_args(args)


def run_trial(
    config: dict, params: dict, args: argparse.Namespace, num_gpus: int = 0
) -> None:
    """Train a single model according to the configuration provided.

    :param config: The trial and model configuration.
    :param params: The hyperparameters.
    :param args: The program arguments.
    """
    seed_everything(config["seed"], workers=True)

    # Initialise the model
    model = LightningResNet18(config, num_classes=10)

    data_module = PartiallyLabelledCIFARDataModule(
        batch_size=params["batch_size"],
        batch_size_test=params["batch_size"],
        dataset_dir=args.dataset_path,
        proportion_labelled=config["proportion_labelled"],
        version="CIFAR10",
    )

    wandb_logger = pl_loggers.WandbLogger(
        project=args.project_name,
        id=tune.get_trial_id(),
        log_model=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_acc", patience=config["patience"] + 5, mode="max"
    )
    lr_monitor_callback = LearningRateMonitor(
        logging_interval="step", log_momentum=True
    )
    tune_report_callback = TuneReportCallback(
        {"val_loss": "val_loss", "val_acc": "val_acc"},
        on="validation_end",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=".",
        filename="{epoch}-{val_acc:.2f}",
        mode="max",
        monitor="val_acc",
        every_n_epochs=1,
        save_top_k=1,
    )
    pl_callbacks = [
        lr_monitor_callback,
        tune_report_callback,
        checkpoint_callback,
        early_stopping_callback,
    ]

    trainer = pl.Trainer(
        max_epochs=1 if args.dry_run else params["epochs"],
        gpus=num_gpus,
        logger=wandb_logger,
        callbacks=pl_callbacks,
        progress_bar_refresh_rate=0,  # disable progress bar
        deterministic=True,
    )

    wandb_logger.watch(model)
    trainer.fit(model, datamodule=data_module)
    wandb_logger.experiment.finish()


def run_experiment(params: dict, args: argparse.Namespace) -> None:
    """Run the experiment using Ray Tune.

    :param params: The hyperparameters.
    :param args: The program arguments.
    """
    config = {
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "momentum": params["momentum"],
        "lr_decay": params["lr_decay"],
        "patience": params["patience"],
        "proportion_labelled": tune.grid_search(params["labelled_proportions"]),
        "seed": tune.grid_search(params["seeds"]),
    }

    reporter = CLIReporter(
        parameter_columns=[
            "proportion_labelled",
            "seed",
        ],
        metric_columns=[
            "epoch",
            "val_loss",
            "val_acc",
        ],
    )

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpus_per_trial = 1 if use_cuda else 0

    tune.run(
        tune.with_parameters(
            run_trial, params=params, args=args, num_gpus=gpus_per_trial
        ),
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": gpus_per_trial},
        metric="val_acc",
        mode="max",
        config=config,
        progress_reporter=reporter,
        name=args.project_name,
    )


def main(args: list) -> None:
    """Parse command line args, load training params, and initiate training.

    :param args: command line parameters as list of strings.
    """
    args = parse_args(args)
    params = yaml.safe_load(open("params.yaml"))

    run_experiment(params, args)


def run() -> None:
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`
    This function can be used as entry point to create console scripts.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
