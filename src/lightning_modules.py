"""PyTorch Lightning modules for ResNet-18.
"""
from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from resnet_models.imagenet.resnet import resnet18
from torchmetrics.functional import accuracy


class LightningResNet(pl.LightningModule):
    """Base Lightning module for a CIFAR10 ResNet model."""

    def __init__(
        self,
        config: dict,
        model: Callable,
        num_classes: int = 10,
    ) -> None:
        """Initialise a new Lightning ResNet module.

        :param config: The hyperparameter configuration.
        :param model: A function that returns the ResNet model to train.
        :param num_classes: The number of output classes, defaults to 10.
        """
        super().__init__()
        self.save_hyperparameters("config", "num_classes")

        self.model = model(num_classes=num_classes)
        self.init_lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.momentum = config["momentum"]
        self.lr_decay = config["lr_decay"]
        self.patience = config["patience"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: The input batch.
        :return: The predicted class probabilities for each example in the
            batch.
        """
        output = self.model(x)
        return output

    def configure_optimizers(self) -> tuple:
        """Configure optimizers.
        :return: The configured optimizers and schedulers.
        """
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.init_lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )

        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=self.lr_decay, patience=self.patience
            ),
            "monitor": "val_acc",
        }

        return [optimizer], [scheduler]

    def training_step(
        self, training_batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step.

        :param training_batch: A batch of training examples
        :param batch_idx: The batch index
        :return: The training loss over the batch.
        """
        x, y = training_batch
        output = self.forward(x)

        mask = y.ge(0.0)
        masked_y = torch.masked_select(y, mask)

        loss = F.cross_entropy(output[mask, :], masked_y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Perform a single validation step.

        :param val_batch: A batch of validation examples
        :param batch_idx: The batch index.
        :return: The validation loss over the batch.
        """
        x, y = val_batch
        output = self.forward(x)

        loss = F.cross_entropy(output, y)
        self.log("val_loss", loss)

        probs = F.softmax(output, dim=1)
        acc = accuracy(probs, y)
        self.log("val_acc", acc)

        return loss

    def test_step(self, test_batch: torch.tensor, batch_idx: int) -> torch.Tensor:
        """Perform a single test step.

        :param test_batch: A batch of test examples
        :param batch_idx: The batch index.
        :return: The test loss over the batch.
        """
        x, y = test_batch
        output = self.forward(x)

        loss = F.cross_entropy(output, y)
        self.log("test_loss", loss)

        probs = F.softmax(output, dim=1)
        acc = accuracy(probs, y)
        self.log("test_acc", acc)

        return loss


class LightningResNet18(LightningResNet):
    """Lightning module for a CIFAR10/100 ResNet-18 model."""

    def __init__(self, config: dict, num_classes: int = 10) -> None:
        """Initialise a new Lightning CIFAR10/100 ResNet-18 module.

        :param config: The hyperparameter configuration.
        :param num_classes: The number of output classes, defaults to 10.
        """
        super().__init__(config, resnet18, num_classes)
