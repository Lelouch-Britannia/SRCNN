import cv2 as cv
import numpy as np
from typing import Tuple

def generate_lr_image(image: np.ndarray, operation: str, scale: int, kernel_size: Tuple[int, int], sigma: float) -> np.ndarray:
    """
    Processes the low-resolution image based on the specified operation: Gaussian Blur, Down-and-Up sampling, or both,
    for both grayscale and color images.

    Parameters:
    - image (np.ndarray): The input image to process. Can be grayscale or color.
    - operation (str): The operation to perform - 'blur', 'downup', 'both'.
    - scale (int): The downscaling factor for down-and-up sampling.
    - kernel_size (Tuple[int, int]): The kernel size for Gaussian blur.
    - sigma (float): The sigma value for Gaussian blur.

    Returns:
    - np.ndarray: Low-resolution image.
    """
    # Check if the image is grayscale or color
    if len(image.shape) == 2:
        # Grayscale image
        h, w = image.shape
    else:
        # Color image
        h, w, _ = image.shape

    lr_image = image

    if operation in ['blur', 'both']:
        # Apply Gaussian Blur
        lr_image = cv.GaussianBlur(lr_image, kernel_size, sigma)

    if operation in ['downup', 'both']:
        # Downscale and then upscale
        if len(image.shape) == 2:
            # Grayscale
            lr_image = cv.resize(cv.resize(lr_image, (w // scale, h // scale), interpolation=cv.INTER_CUBIC), (w, h), interpolation=cv.INTER_CUBIC)
        else:
            # Color
            lr_image = cv.resize(cv.resize(lr_image, (w // scale, h // scale), interpolation=cv.INTER_CUBIC), (w, h), interpolation=cv.INTER_CUBIC)

    return lr_image

