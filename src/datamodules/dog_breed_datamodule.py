import os
import zipfile
from pathlib import Path
from typing import Union

import gdown
import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils.logging_utils import logger


class DogBreedImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        dl_path: Union[str, Path] = "data",
        num_workers: int = 0,
        batch_size: int = 32,
    ):
        super().__init__()
        self._dl_path = Path(dl_path).resolve()  # Get the absolute path
        self._num_workers = num_workers
        self._batch_size = batch_size
        self.dataset = None

    def prepare_data(self):
        """Download images and prepare images datasets."""
        logger.info(f"Data path: {self._dl_path}")
        if not self._dl_path.exists():
            logger.info(f"Creating directory: {self._dl_path}")
            self._dl_path.mkdir(parents=True)

        dataset_path = self._dl_path / "dog_breed_dataset.zip"
        extracted_path = self._dl_path / "dataset"

        logger.info(f"Checking for extracted dataset at: {extracted_path}")
        if not extracted_path.exists():
            if not dataset_path.exists():
                logger.info(f"Downloading dataset to: {dataset_path}")
                url = "https://drive.google.com/uc?id=1QhNYSUoDESCUSEzelXZPigBcpZJno_lJ"
                gdown.download(url, str(dataset_path), quiet=False)

            logger.info(f"Extracting dataset to: {self._dl_path}")
            with zipfile.ZipFile(dataset_path, "r") as zip_ref:
                zip_ref.extractall(self._dl_path)

            logger.info("Dataset extracted successfully.")
        else:
            logger.info("Dataset already exists. Skipping download and extraction.")

        logger.info(f"Contents of {self._dl_path}:")
        for item in os.listdir(self._dl_path):
            logger.info(f"  {item}")

        if extracted_path.exists():
            logger.info(f"Contents of {extracted_path}:")
            for item in os.listdir(extracted_path):
                logger.info(f"  {item}")
        else:
            logger.error(f"Extracted path does not exist: {extracted_path}")

    @property
    def data_path(self):
        return self._dl_path / "dataset"

    @property
    def normalize_transform(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def val_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    def setup(self, stage: str = None):
        logger.info(f"Setting up dataset from: {self.data_path}")
        if not self.data_path.exists():
            logger.error(f"Data path does not exist: {self.data_path}")
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        if self.dataset is None:
            self.dataset = ImageFolder(
                root=self.data_path, transform=self.train_transform
            )

        logger.info(f"Number of classes: {len(self.dataset.classes)}")
        logger.info(f"Total number of images: {len(self.dataset)}")

        # Split the dataset
        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

        # Update transforms for validation and test sets
        self.val_dataset.dataset.transform = self.val_transform
        self.test_dataset.dataset.transform = self.val_transform

        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Validation dataset size: {len(self.val_dataset)}")
        logger.info(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self._batch_size, num_workers=self._num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )
