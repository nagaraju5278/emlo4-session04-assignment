import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def show_batch(dataloader, num_images=4):
    batch = next(iter(dataloader))
    images, labels = batch
    grid = make_grid(images[:num_images])
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Sample batch from the dataset")
    plt.show()


def visualize_model(model):
    from torchviz import make_dot

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render("model_architecture", format="png")
    # You might want to use a different method to display the image in a non-notebook environment
    # For example, you could save it to a file instead of displaying it inline
    dot.render("model_architecture", format="png", cleanup=True)
    print("Model architecture saved as 'model_architecture.png'")
