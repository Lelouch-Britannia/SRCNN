from PIL import Image
import numpy as np

def modcrop(image: Image.Image, scale: int) -> Image.Image:
    """Crop image to make it divisible by the scale."""

    height, width = image.shape
    height = height - np.mod(height, scale)
    widht = width -  np.mod(width, scale)
    image =  image[0:height, 0:widht]

    return image
