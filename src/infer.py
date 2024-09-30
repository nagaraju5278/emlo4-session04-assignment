import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from rich.progress import Progress, TaskID
from torchvision import transforms

from models.dog_breed_classifier import DogBreedClassifier
from utils.logging_utils import logger, setup_logging, task_wrapper


@task_wrapper
def inference(model, image_path, class_names):
    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply the transform to the image
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Move the input tensor to the same device as the model
    img_tensor = img_tensor.to(model.device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_breed = class_names[predicted_class]
    confidence = probabilities[0][predicted_class].item()

    return predicted_breed, confidence, img


@task_wrapper
def main(args):
    setup_logging("inference.log")

    # Load the model
    logger.info(f"Loading model from {args.ckpt_path}")
    model = DogBreedClassifier.load_from_checkpoint(args.ckpt_path)

    # Get class names
    class_names = model.hparams.get("class_names", None)
    if class_names is None:
        logger.warning(
            "Class names not found in model checkpoint. Using index as class name."
        )
        class_names = [str(i) for i in range(model.hparams.num_classes)]

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Get list of image files
    image_files = [
        f
        for f in os.listdir(args.input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    with Progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))

        # Process each image in the input folder
        for filename in image_files:
            image_path = os.path.join(args.input_folder, filename)
            predicted_breed, confidence, img = inference(model, image_path, class_names)

            # Save the result
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Predicted: {predicted_breed} (Confidence: {confidence:.2f})")
            output_path = os.path.join(args.output_folder, f"pred_{filename}")
            plt.savefig(output_path)
            plt.close()

            logger.info(
                f"Processed {filename}: Predicted {predicted_breed} with confidence {confidence:.2f}"
            )
            progress.update(task, advance=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for Dog Breed Classifier"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder to save prediction results",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/checkpoint_model_dg_bread.ckpt",
        help="Path to the model checkpoint",
    )
    args = parser.parse_args()

    main(args)
