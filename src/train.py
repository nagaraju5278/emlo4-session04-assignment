import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datamodules.dog_breed_datamodule import DogBreedImageDataModule
from models.dog_breed_classifier import DogBreedClassifier
from utils.callbacks import get_rich_model_summary, get_rich_progress_bar
from utils.logging_utils import logger, setup_logging, task_wrapper


@task_wrapper
def train_and_test(data_module, model):
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="checkpoint_model_dg_bread",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    rich_progress_bar = get_rich_progress_bar()
    rich_model_summary = get_rich_model_summary()

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=1,
        callbacks=[checkpoint_callback, rich_progress_bar, rich_model_summary],
        accelerator="auto",
        logger=TensorBoardLogger(save_dir="logs", name="dog_breed_classification"),
        log_every_n_steps=1,  # Log every step to see progress with small datasets
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

    # Save the final model
    trainer.save_checkpoint("checkpoints/checkpoint_model_dg_bread.ckpt")
    logger.info(f"Final model saved to checkpoints/checkpoint_model_dg_bread.ckpt")


def main():
    setup_logging()
    L.seed_everything(42, workers=True)

    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    logger.info("Initializing DataModule and Model")
    # Initialize DataModule
    data_module = DogBreedImageDataModule(dl_path="data", batch_size=32, num_workers=2)
    data_module.prepare_data()
    data_module.setup()

    # Get the number of classes
    num_classes = len(data_module.dataset.classes)

    # Initialize Model
    model = DogBreedClassifier(num_classes=num_classes, lr=1e-3)

    train_and_test(data_module, model)


if __name__ == "__main__":
    main()
