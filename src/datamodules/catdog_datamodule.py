from pathlib import Path
from typing import Union

import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


class CatDogImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        dl_path: Union[str, Path] = "data",
        num_workers: int = 0,
        batch_size: int = 8,
    ):
        super().__init__()
        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size

    def prepare_data(self):
        download_and_extract_archive(
            url="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
            download_root=self._dl_path,
            remove_finished=True,
        )

    @property
    def data_path(self):
        return Path(self._dl_path).joinpath("cats_and_dogs_filtered")

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
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def valid_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool):
        if train:
            dataset = self.create_dataset(
                self.data_path.joinpath("train"), self.train_transform
            )
        else:
            dataset = self.create_dataset(
                self.data_path.joinpath("validation"), self.valid_transform
            )
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=train,
        )

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)

    def test_dataloader(self):
        return self.__dataloader(train=False)  # Using validation dataset for testing
