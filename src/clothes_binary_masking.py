import cv2
from tqdm import tqdm
import os
import numpy as np
import argparse
import glob
import itertools


SUPPORTED_EXTS = ["jpg", "jpeg", "png"]


def iter_images_in_dir(dir_path):
    """
    Return an iterable containing path to all supported image
    files in the provided directory
    """
    return itertools.chain(
        *[glob.iglob(os.path.join(dir_path, f"*.{ext}")) for ext in SUPPORTED_EXTS]
    )


def segment_image(image_path):
    """
    Returns a binary mask of the image at the given path
    """
    img = cv2.imread(image_path)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    fg = np.zeros((1, 65), dtype="float")
    bg = np.zeros((1, 65), dtype="float")
    rect = (2, 2, img.shape[1] - 2, img.shape[0] - 2)
    mask, bg, fg = cv2.grabCut(
        img, mask, rect, bg, fg, iterCount=25, mode=cv2.GC_INIT_WITH_RECT
    )
    mask[mask > 2] = 255
    mask[mask <= 2] = 0
    return mask


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image-dir",
        "-I",
        required=True,
        type=str,
        help="path to clothing image directory",
    )
    ap.add_argument(
        "--output-dir",
        "-O",
        required=True,
        type=str,
        help="path to output directory to save segmentation masks",
    )
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    for image_file in tqdm(iter_images_in_dir(args.image_dir)):
        binary_mask = segment_image(image_file)
        # save mask to directory with the same file name as the original image
        output_image_path = os.path.join(args.output_dir, os.path.basename(image_file))
        cv2.imwrite(output_image_path, binary_mask)
