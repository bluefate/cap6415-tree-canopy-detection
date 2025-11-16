import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image( image: np.ndarray, title: str = "" ) -> None:
    """
    Show an image using matplotlib.
    Handles RGB and Greyscale automatically.
    """
    # Convert float images safely
    if image.dtype != np.uint8:
        img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img = image

    plt.figure(figsize = (5, 5))
    if image.ndim == 2:
        plt.imshow(image, cmap = "Greys")
    else:
        plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_mask( mask: np.ndarray, title: str = "" ) -> None:
    """
    Show a binary mask.
    """
    plt.figure(figsize = (5, 5))
    plt.imshow(mask, cmap = "Greys")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_overlay( image: np.ndarray, mask: np.ndarray, alpha: float = 0.4, title: str = "" ) -> None:
    """
    Show an image with a red mask overlay.
    """
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    mask_u8 = (mask * 255).astype(np.uint8)
    mask_rgb = np.zeros_like(img_u8)
    mask_rgb[:, :, 0] = mask_u8

    overlay = cv2.addWeighted(img_u8, 1 - alpha, mask_rgb, alpha, 0)

    plt.figure(figsize = (5, 5))
    plt.imshow(overlay)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_stages( stages: dict ) -> None:
    """
    Display multiple enhancement or filtering stages.
    Keys must be stage names. Values are images.
    """
    n = len(stages)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    plt.figure(figsize = (4 * cols, 4 * rows))

    for i, (name, img) in enumerate(stages.items(), 1):
        plt.subplot(rows, cols, i)
        if img.ndim == 2:
            plt.imshow(img, cmap = "Greys")
        else:
            plt.imshow(img)
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def compare_images( a: np.ndarray, b: np.ndarray, titles = ("A", "B") ) -> None:
    """
    Display two images side by side for comparison.
    """
    plt.figure(figsize = (10, 5))

    plt.subplot(1, 2, 1)
    if a.ndim == 2:
        plt.imshow(a, cmap = "Greys")
    else:
        plt.imshow(a)
    plt.title(titles[0])
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if b.ndim == 2:
        plt.imshow(b, cmap = "Greys")
    else:
        plt.imshow(b)
    plt.title(titles[1])
    plt.axis("off")

    plt.show()
