import argparse
import os

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from rich.console import Console
from rich.table import Table

from datamodules.dog_breed_datamodule import DogBreedImageDataModule
from models.dog_breed_classifier import DogBreedClassifier
from utils.callbacks import get_rich_progress_bar
from utils.logging_utils import logger, setup_logging, task_wrapper


@task_wrapper
def evaluate(trainer, model, data_module):
    # Evaluate the model on the validation set
    results = trainer.validate(model, data_module)
    return results[0]  # Return the first (and only) results dict


def print_results(results):
    console = Console()
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in results.items():
        table.add_row(key, f"{value:.4f}")

    console.print(table)


@task_wrapper
def main(args):
    setup_logging("eval.log")
    L.seed_everything(42, workers=True)

    # Initialize DataModule
    logger.info("Initializing DataModule")
    data_module = DogBreedImageDataModule(dl_path="data", batch_size=32, num_workers=2)
    data_module.prepare_data()
    data_module.setup()

    # Load the model from checkpoint
    logger.info(f"Loading model from checkpoint: {args.ckpt_path}")
    model = DogBreedClassifier.load_from_checkpoint(args.ckpt_path)

    # Initialize Trainer
    trainer = L.Trainer(
        accelerator="auto",
        logger=TensorBoardLogger(save_dir="logs", name="dog_breed_evaluation"),
        callbacks=[get_rich_progress_bar()],
    )

    # Evaluate the model
    results = evaluate(trainer, model, data_module)

    # Print the results
    print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for Dog Breed Classifier"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the model checkpoint"
    )
    args = parser.parse_args()

    main(args)
