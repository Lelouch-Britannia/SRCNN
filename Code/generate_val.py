import os
import sys
import cv2 as cv
import numpy as np
import argparse
from PIL import Image
from typing import Tuple

sys.path.append('Helper')
from modcrop import modcrop
from generate_lr_image import generate_lr_image

def generate_val(filepath: str, patches_label_dir: str, patches_data_dir: str, operation: str, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 0, size_input:int = 33, size_label:int = 21, scale:int = 3, stride:int = 21):
    """
    Generates validation patches from a single image file and saves them to specified directories.

    This function takes an image file, processes it to generate low-resolution (LR) and high-resolution (HR) patches, and saves these patches as individual image files in specified directories. LR patches are generated by downsampling and then upsizing the original HR image using bicubic interpolation. The function then extracts patches of specified sizes from both the LR and HR images using a specified stride.

    Parameters:
    - filename (str): Path to the original HR image file.
    - patches_label_dir (str): Directory path where HR (label) patches will be saved.
    - patches_data_dir (str): Directory path where LR (input/data) patches will be saved.
    - size_input (int, optional): Size of the LR patches. Default is 33.
    - size_label (int, optional): Size of the HR patches. Default is 21.
    - scale (int, optional): Factor by which the HR image is downsampled to generate the LR image. Default is 3.
    - stride (int, optional): Stride with which patches are extracted from the images, controlling the overlap. Default is 14.

    Returns:
    - int: The total number of patches generated from the image.

    Raises:
    - TypeError: If the input image is not a PIL Image object.
    """
    # Create directories for patches if they don't exist
    os.makedirs(patches_label_dir, exist_ok=True)
    os.makedirs(patches_data_dir, exist_ok=True)

    filename = os.path.basename(filepath).split('.')[0]

    padding = (size_input - size_label) // 2

    img = Image.open(filepath).convert('RGB')
    
    if not isinstance(img, Image.Image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    
    img_ycbcr = img.convert('YCbCr')

    # Convert images to numpy arrays
    img = np.array(img_ycbcr)

    # Take the y channel
    y_channel = img[:, :, 0]

    im_label = modcrop(y_channel, scale)
    h, w = im_label.shape
    
    im_input = generate_lr_image(im_label, operation, scale, kernel_size, sigma)
    
  
    total_patches = 0
    
    # Save each patch to the respective directories
    for x in range(0, h-size_input+1, stride):
        for y in range(0,  w-size_input+1, stride):
            # Extract the data and label patch         
            data_patch = im_input[x:x+size_input, y:y+size_input]
            label_patch = im_label[x+padding:x+padding+size_label, y+padding:y+padding+size_label]
            
            # Convert patches back to images
            data_patch_img = Image.fromarray(data_patch)
            label_patch_img = Image.fromarray(label_patch)

            total_patches += 1

            # Save the patch
            data_patch_img.save(os.path.join(patches_data_dir, f'{filename}_{x}_{y}.png'))
            label_patch_img.save(os.path.join(patches_label_dir, f'{filename}_{x}_{y}.png'))

    return total_patches

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate patches from validation data for SRCNN.',
                                     usage='''python script_name.py --val_data_dir <val_data_dir> 
                                            [--patches_label_dir <patches_label_dir>] 
                                            [--patches_data_dir <patches_data_dir>] 
                                            [--size_input <size_input>] 
                                            [--size_label <size_label>] 
                                            [--scale <scale>] 
                                            [--stride <stride>]''')

    # Required arguments
    parser.add_argument('-d', '--val_data_dir', type=str, required=True, help='Directory containing validation images')
    
    default_label_dir = os.path.join('..', 'Data', 'Validation', 'pHR')
    default_data_dir = os.path.join('..', 'Data', 'Validation', 'pLR')
    # Optional arguments

    parser.add_argument('-o', '--operation', type=str, choices=['blur', 'downup', 'both'], default='downup', help='Operation to perform on images: blur, downup, both')
    parser.add_argument('--patches_label_dir', type=str, default=default_label_dir, help='Directory to save label patches')
    parser.add_argument('--patches_data_dir', type=str, default=default_data_dir, help='Directory to save data patches')
    parser.add_argument('--size_input', type=int, default=33, help='Input patch size')
    parser.add_argument('--size_label', type=int, default=21, help='Label patch size')
    parser.add_argument('--scale', type=int, default=3, help='Downscaling factor')
    parser.add_argument('--stride', type=int, default=21, help='Stride for patchify')
    parser.add_argument('--kernel_size', type=int, nargs=2, default=[5, 5], help='Kernel size for Gaussian blur as Tuple')
    parser.add_argument('--sigma', type=float, default=0, help='Sigma value for Gaussian blur. Default is 0.')
    args = parser.parse_args()

    # Parse kernel_size argument into the format expected by cv2.GaussianBlur
    kernel_size = tuple(args.kernel_size)

    # Print message about the processing type
    if args.operation == 'blur':
        print(f"Applying Gaussian Blur with kernel size {kernel_size} and sigma {args.sigma}.")
    elif args.operation == 'downup':
        print("Applying Down-and-Up sampling.")
    elif args.operation == 'both':
        print(f"Applying both Gaussian Blur with kernel size {kernel_size} and sigma {args.sigma}, and Down-and-Up sampling.")

    total = 0
    for filename in os.listdir(args.val_data_dir):
        filepath = os.path.join(args.val_data_dir, filename)
        if os.path.isfile(filepath):
            patches_count = generate_val(filepath=filepath,
                    patches_label_dir=args.patches_label_dir,
                    patches_data_dir=args.patches_data_dir,
                    operation=args.operation,
                    kernel_size=kernel_size,
                    sigma=args.sigma,
                    size_input=args.size_input,
                    size_label=args.size_label,
                    scale=args.scale,
                    stride=args.stride)
            total += patches_count
            print(f'Generated {patches_count} patches from {filename}')


    print(f'Total patches generated: {total}')
