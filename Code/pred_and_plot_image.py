import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import torchvision.transforms as transforms

def pred_and_plot_image(model: torch.nn.Module, image_path: str, transform=None, plot=False):
    """
    Perform prediction on an input image using a given model and plot the original and predicted images.

    Parameters:
    - model (torch.nn.Module): The pretrained deep learning model for inference.
    - image_path (str): Path to the input image file.
    - transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. Defaults to None, which will apply a default transform of converting the image to a tensor.
    - plot (bool): If True, the original and predicted images will be plotted. Defaults to False.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the original and predicted images as numpy arrays.
    """

    # Automatic device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and convert image
    try:
        hr_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None

    hr_img_ycbcr = hr_img.convert('YCbCr')
    y_channel = np.array(hr_img_ycbcr)[:, :, 0]  # Take the Y channel

    if transform is None:
        transform = transforms.ToTensor()
    
    test_img_tensor = transform(y_channel).unsqueeze(0).to(device)

    # Model inference
    try:
        model.eval().to(device)
        with torch.no_grad():
            y_predicted = model(test_img_tensor).squeeze().cpu().numpy() * 255
    except Exception as e:
        print(f"Error during model inference: {e}")
        return None, None

    y_predicted = y_predicted.astype(np.uint8)

    # Visualization
    if plot:
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(y_channel, cmap='gray')
        plt.title("Original Image")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.imshow(y_predicted, cmap='gray')
        plt.title("Predicted Image")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return y_channel, y_predicted

