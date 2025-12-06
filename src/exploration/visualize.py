from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_side_by_side(
    *images: np.ndarray,
    titles: Optional[Tuple[str, ...]] = None,
    cmaps: Optional[Tuple[str, ...]] = None,
    maxcolumns: Optional[int] = None,
    preserve_values: bool = False,
    vmax=None,
    kernel=None,
    kernel_title=None,
    kernel_cmap="seismic",
    mask_colors=None,
    figsize=4,
) -> None:
    """Show multiple images side by side."""

    count = len(images)
    # auto adjust long grids
    if maxcolumns is not None and maxcolumns > 10:
        titles = tuple(str(i) for i in range(count))
        scale = max(1.0, maxcolumns / 10)
        title_fontsize = min(22, 20 * scale)
    else:
        title_fontsize = 10

    # Default titles
    if titles is None:
        # default = ["Image", "Mask", "Overlay"]
        # titles = default[:count]
        titles = tuple(f"Image {i + 1}" for i in range(count))

    if cmaps is None:
        cmaps = tuple([None] * count)

    # ---- grid layout ----
    if maxcolumns is None or maxcolumns >= count:
        ncols = count
        nrows = 1
    else:
        ncols = maxcolumns
        nrows = int(np.ceil(count / maxcolumns))

    add_kernel = kernel is not None
    if add_kernel:
        ncols = ncols + 2  # reserve 2 columns for 2D + 3D kernel plots

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize * ncols, figsize * nrows))
    fig.patch.set_facecolor("white")

    # Flatten axes for easy iteration
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Iterate over images
    for ax, img, title, cmap in zip(axes[:count], images, titles, cmaps):
        # ---------------------------------------------------
        # color-mask logic FIRST and ALWAYS before type checks
        # ---------------------------------------------------
        # if "mask" in title.lower() and isinstance(img, tuple) and len(img) == 2:
        if mask_colors and isinstance(img, tuple) and len(img) == 2:

            mask, cls = img
            # choose color
            # if mask_colors and cls in mask_colors:
            if cls in mask_colors:
                color = tuple(mask_colors[cls])
            else:
                color = (255, 255, 255)

            # convert single-channel mask to 0/255
            if mask.dtype != np.uint8:
                mask_u8 = (mask > 0.5).astype(np.uint8) * 255
            else:
                mask_u8 = mask * 255 if mask.max() <= 1 else mask

            # create rgb mask
            img = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
            img[mask_u8 > 0] = color

        # ---------------------------------------------------
        # now handle normal ndarray inputs
        # ---------------------------------------------------
        elif isinstance(img, np.ndarray):
            # Handle raw image arrays
            if not preserve_values:
                # Handle [0,1] range images properly
                if img.max() <= 1.0:
                    # Image is in [0,1] range - scale to [0,255]
                    img = (img * 255).astype(np.uint8)
                else:
                    # Image is in larger range - clip to [0,255]
                    img = np.clip(img, 0, 255).astype(np.uint8)
            # else: keep raw values (signed floats/ints)

        # ---------------------------------------------------
        # matplotlib figure
        # ---------------------------------------------------
        elif hasattr(img, "canvas"):
            # render the source figure into an image
            img.canvas.draw()

            # get RGBA buffer from source figure
            buf = np.asarray(img.canvas.buffer_rgba())

            # convert RGBA to RGB by dropping alpha
            img = buf[:, :, :3]

        # ---------------------------------------------------
        # unsupported type
        # ---------------------------------------------------
        else:
            raise TypeError(
                f"Unsupported type {type(img)} passed to show_side_by_side. "
                "Expected numpy.ndarray, (mask,class_name) tuple, or matplotlib Figure."
            )

        # Visualization
        # if title == "Mask":
        #     ax.imshow(img, "gray")
        # if "mask" in title.lower():
        if "mask" in title.lower() and isinstance(img, np.ndarray) and img.ndim == 2:
            ax.imshow(img, "gray")
        elif cmap is not None:
            # If preserve_values is True, we need symmetric vmin/vmax
            if preserve_values and np.issubdtype(img.dtype, np.number):
                # convert before passing
                # lap_img = apply_kernel(img_gray, lk).astype(np.float32)
                if vmax is None:
                    vmax = float(np.max(np.abs(img)))

                ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)

        ax.set_facecolor("white")
        if maxcolumns is not None and maxcolumns > 10:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            ax.set_title(title)
        ax.axis("off")

    # if add_kernel:
    #     for spine in ax2d.spines.values():
    #         spine.set_visible(True)
    #         spine.set_edgecolor("black")
    #         spine.set_linewidth(2)

    # remove empty axes
    for ax in axes[count:]:
        ax.remove()

    plt.tight_layout()
    plt.show()


def show_image(
    image: np.ndarray, title: str = "", return_img: bool = False, cmap="gray"
):
    """
    Show an image using matplotlib.
    """
    # Convert float images safely - handle [0,1] range properly
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            # Image is in [0,1] range - convert to [0,255]
            img = (image * 255).astype(np.uint8)
        else:
            # Image is in larger range - clip to [0,255]
            img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img = image

    if return_img:
        return img
    else:
        plt.figure(figsize=(5, 5))
        if img.ndim == 2:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()


# using from PIL import Image to be able to show pure white and black
def show_mask(mask: np.ndarray, title: str = "", return_img: bool = False, cmap="gray"):
    """
    Show a mask with proper handling of class indices.
    """
    # Handle multi-class masks (class indices 0,1,2)
    if mask.dtype != np.uint8:
        if mask.max() <= 2:
            # Multi-class mask with indices [0,1,2] - scale to visible range
            mask_img = (mask * 127).astype(np.uint8)  # 0->0, 1->127, 2->254
        else:
            # Binary mask [0,1] - convert to [0,255]
            mask_img = (mask > 0.5).astype(np.uint8) * 255
    else:
        if mask.max() <= 2:
            # Already uint8 with class indices [0,1,2]
            mask_img = mask * 127
        elif mask.max() <= 1:
            # Binary mask [0,1] as uint8
            mask_img = mask * 255
        else:
            # Already in proper range
            mask_img = mask

    if return_img:
        return mask_img

    # use PIL for exact grayscale
    from PIL import Image

    img = Image.fromarray(mask_img, mode="L")

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    title: str = "",
    return_img: bool = False,
):
    """
    Show an image with a red mask overlay.
    """
    # Convert image to uint8
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    # Handle mask properly - don't multiply class indices by 255
    if mask.max() <= 2:
        # Multi-class mask [0,1,2] - convert to binary [0,1] then to [0,255]
        mask_binary = (mask > 0).astype(np.float32)  # Any class > 0 = tree
        mask_u8 = (mask_binary * 255).astype(np.uint8)
    else:
        # Already processed mask
        mask_u8 = mask.astype(np.uint8)

    mask_rgb = np.zeros_like(img_u8)
    mask_rgb[:, :, 0] = mask_u8  # Red channel for trees

    overlay = cv2.addWeighted(img_u8, 1 - alpha, mask_rgb, alpha, 0)

    if return_img:
        return overlay
    else:
        plt.figure(figsize=(5, 5))
        plt.imshow(overlay)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()


def show_stages(stages: dict, cmaps=None, maxcolumns=5) -> None:
    """
    Display multiple enhancement or filtering stages.
    Keys must be stage names. Values are images.
    """
    images = list(stages.values())
    titles = list(stages.keys())
    if cmaps is None:
        cmaps = ["gray"] * len(images)

    show_side_by_side(*images, titles=titles, cmaps=cmaps)

    # n = len(stages)
    # cols = min(4, n)
    # rows = int(np.ceil(n / cols))
    #
    # plt.figure(figsize = (4 * cols, 4 * rows))
    #
    # for i, (name, img) in enumerate(stages.items(), 1):
    #     plt.subplot(rows, cols, i)
    #     if img.ndim == 2:
    #         plt.imshow(img, cmap = cmap)
    #     else:
    #         plt.imshow(img)
    #     plt.title(name)
    #     plt.axis("off")
    #
    # plt.tight_layout()
    # plt.show()
