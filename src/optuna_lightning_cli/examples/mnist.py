"""MNIST example components for Optuna tuning with Lightning.

The training loop lives in the :class:`MnistClassifier` LightningModule, while
the :class:`MnistDataModule` owns dataset download and data loading. The CLI
only wires the pieces together, patches Optuna parameters into the config, and
hands the result to ``Trainer.fit()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import LightningDataModule, LightningModule
from torch import Tensor, nn
from torch.utils.data import DataLoader


class MnistClassifier(LightningModule):
    """A small MLP for MNIST classification.

    Args:
        lr: Learning rate used by the Adam optimizer.
        hidden_size: Width of the hidden layer in the classifier.
    """

    def __init__(self, lr: float = 1e-3, hidden_size: int = 128) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass and return logits for the 10 MNIST classes."""

        return self.network(x)

    def training_step(self, batch: tuple[Tensor, Tensor], _batch_idx: int) -> Tensor:
        """Compute the training loss for a batch."""

        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
        )
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], _batch_idx: int) -> Tensor:
        """Compute validation loss and accuracy for a batch."""

        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).float().mean()
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
        )
        self.log(
            "val_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create the Adam optimizer used by the example."""

        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class MnistDataModule(LightningDataModule):
    """LightningDataModule for downloading and loading MNIST.

    Args:
        data_dir: Directory used to store the MNIST files.
        batch_size: Batch size for the training and validation loaders.
        download: Whether to download MNIST if it is missing locally.
        num_workers: Number of worker processes used by the data loaders.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        download: bool = True,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.download = download
        self.num_workers = num_workers
        self.train_dataset: Any | None = None
        self.val_dataset: Any | None = None

    def prepare_data(self) -> None:
        """Download MNIST if needed."""

        mnist = self._mnist_dataset()
        transform = self._transform()
        mnist(self.data_dir, train=True, download=self.download, transform=transform)
        mnist(self.data_dir, train=False, download=self.download, transform=transform)

    def setup(self, stage: str | None = None) -> None:
        """Create the train and validation datasets."""

        mnist = self._mnist_dataset()
        transform = self._transform()
        if stage in (None, "fit", "validate"):
            self.train_dataset = mnist(
                self.data_dir,
                train=True,
                download=False,
                transform=transform,
            )
            self.val_dataset = mnist(
                self.data_dir,
                train=False,
                download=False,
                transform=transform,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training data loader."""

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation data loader."""

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @staticmethod
    def _mnist_dataset():
        from torchvision.datasets import MNIST

        return MNIST

    @staticmethod
    def _transform():
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
