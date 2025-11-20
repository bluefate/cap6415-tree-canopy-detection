# From C:\github\Tree-Canopy-Detection\src\exploration\class_explorer.py
from pathlib import Path

import cv2
import numpy as np

from src.data.annotations import AnnotationEntry
from src.exploration.visualize import (
    show_side_by_side,
)
from src.utils.config import Config
from src.utils.helpers import p


# -----------------------------------------------------------
# Core helpers
# -----------------------------------------------------------

config = Config.load()
CLASS_NAMES = ["individual_tree", "group_of_trees"]


def load_image(image_dir: Path, entry: AnnotationEntry):
    path = image_dir / entry.image_path.name
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError("Failed to read image " + str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mask_for_class(entry: AnnotationEntry, cls: str):
    H = entry.height
    W = entry.width
    mask = np.zeros((H, W), dtype=np.uint8)

    for item in entry.items:
        if item.cls != cls:
            continue
        seg = item.segmentation
        if seg is None or len(seg) < 4:
            continue
        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], 1)

    return mask


def available_classes(entries):
    classes = set()
    for e in entries:
        for item in e.items:
            classes.add(item.cls)
    return sorted(classes)


def mask_all(entry: AnnotationEntry):
    H = entry.height
    W = entry.width
    mask = np.zeros((H, W), dtype=np.uint8)
    for item in entry.items:
        seg = item.segmentation
        if seg is None or len(seg) < 4:
            continue
        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], 1)
    return mask


# -----------------------------------------------------------
# Visualization utilities
# -----------------------------------------------------------


def color_mask(entry: AnnotationEntry):
    H = entry.height
    W = entry.width
    mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for item in entry.items:
        seg = item.segmentation
        if seg is None or len(seg) < 4:
            continue

        cls = item.cls
        color = config.MASK_COLORS.get(cls, (255, 255, 255))
        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask_rgb, [poly], color)

    return mask_rgb


def draw_bboxes(img, entry: AnnotationEntry):
    out = img.copy()
    for item in entry.items:
        seg = item.segmentation
        if seg is None or len(seg) < 4:
            continue

        xs = seg[0::2]
        ys = seg[1::2]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

        if item.cls == "individual_tree":
            color = (0, 255, 0)
        elif item.cls == "group_of_trees":
            color = (255, 0, 0)
        else:
            color = (0, 128, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = item.cls
        cv2.putText(
            out, label, (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
        )
    return out


def draw_bboxes_for_class(img, entry: AnnotationEntry, cls: str):
    out = img.copy()
    for item in entry.items:
        if item.cls != cls:
            continue
        seg = item.segmentation
        if seg is None or len(seg) < 4:
            continue

        xs = seg[0::2]
        ys = seg[1::2]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

        if cls == "individual_tree":
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            out, cls, (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 3
        )

    return out


def explore_image(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)

    mask_ind = mask_for_class(entry, "individual_tree")
    mask_grp = mask_for_class(entry, "group_of_trees")

    color_ind = np.zeros_like(img)
    color_grp = np.zeros_like(img)

    col_ind = tuple(config.MASK_COLORS.get("individual_tree", [0, 255, 0]))
    col_grp = tuple(config.MASK_COLORS.get("group_of_trees", [255, 0, 0]))

    color_ind[mask_ind == 1] = col_ind
    color_grp[mask_grp == 1] = col_grp

    # over_ind = show_overlay(img, mask_ind, return_img=True)
    # over_grp = show_overlay(img, mask_grp, return_img=True)
    over_ind = cv2.addWeighted(img, 0.6, color_ind, 0.4, 0)
    over_grp = cv2.addWeighted(img, 0.6, color_grp, 0.4, 0)

    show_side_by_side(
        # img,
        (mask_ind, "individual_tree"),
        (mask_grp, "group_of_trees"),
        over_ind,
        over_grp,
        titles=(
            # "Image",
            "Mask individual",
            "Mask group",
            "Overlay individual",
            "Overlay group",
        ),
        mask_colors=config.MASK_COLORS,
    )


def show_single_class(entry: AnnotationEntry, image_dir: Path, cls: str):
    img = load_image(image_dir, entry)
    mask = mask_for_class(entry, cls)
    show_side_by_side(
        img,
        (mask, cls),
        titles=("Image", f"Mask {cls}"),
        mask_colors=config.MASK_COLORS,
    )


def show_all_classes(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)

    # build combined mask for both classes
    mask = mask_all(entry)

    # build colored mask (per polygon, per class)
    mask_rgb = color_mask(entry)

    # overlay with mask colors
    overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)

    show_side_by_side(
        img,
        mask_rgb,
        overlay,
        titles=("Image", "Mask All (color)", "Overlay All"),
        mask_colors=config.MASK_COLORS,
    )


def show_per_class(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)
    classes = ["individual_tree", "group_of_trees"]

    # masks must be tuples (mask, class_name) for coloring
    imgs = [img] + [(mask_for_class(entry, c), c) for c in classes]

    titles = ["Image"] + classes

    show_side_by_side(
        *imgs,
        titles=tuple(titles),
        mask_colors=config.MASK_COLORS,
    )


def show_overlay_all(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)
    # mask = mask_all(entry)
    mask = color_mask(entry)

    # show_overlay(img, mask, title="Overlay All")
    overlay = cv2.addWeighted(img, 0.6, mask, 0.4, 0)

    show_side_by_side(
        overlay,
        titles=("Overlay All",),
        mask_colors=config.MASK_COLORS,
    )


def show_overlay_by_class(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)

    m_ind = mask_for_class(entry, "individual_tree")
    m_grp = mask_for_class(entry, "group_of_trees")

    col_ind = tuple(config.MASK_COLORS.get("individual_tree", [0, 255, 0]))
    col_grp = tuple(config.MASK_COLORS.get("group_of_trees", [255, 0, 0]))

    # color masks
    color_ind = np.zeros_like(img)
    color_grp = np.zeros_like(img)

    color_ind[m_ind == 1] = col_ind
    color_grp[m_grp == 1] = col_grp

    # overlays
    overlay_ind = cv2.addWeighted(img, 0.6, color_ind, 0.4, 0)
    overlay_grp = cv2.addWeighted(img, 0.6, color_grp, 0.4, 0)

    show_side_by_side(
        overlay_ind,
        overlay_grp,
        titles=("Overlay individual", "Overlay group"),
        mask_colors=config.MASK_COLORS,
    )


# -----------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------


def explore_color_overlay(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)
    mask_rgb = color_mask(entry)
    overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)

    show_side_by_side(
        img, mask_rgb, overlay, titles=("Image", "Color mask", "Overlay color mask")
    )


def explore_bboxes(entry: AnnotationEntry, image_dir: Path):
    img = load_image(image_dir, entry)
    b_all = draw_bboxes(img, entry)
    b_ind = draw_bboxes_for_class(img, entry, "individual_tree")
    b_grp = draw_bboxes_for_class(img, entry, "group_of_trees")

    show_side_by_side(
        b_ind, b_grp, b_all, titles=("Individual bboxes", "Group bboxes", "All bboxes")
    )


def find_images_with_both(entries):
    out = []
    for e in entries:
        classes = {item.cls for item in e.items}
        if "individual_tree" in classes and "group_of_trees" in classes:
            out.append(e)
    return out


def count_classes(entry: AnnotationEntry):
    counts = {"individual_tree": 0, "group_of_trees": 0}
    for item in entry.items:
        if item.cls in counts:
            counts[item.cls] += 1
    return counts


def class_distribution(entries):
    if isinstance(entries, AnnotationEntry):
        entries = [entries]
    total = {"individual_tree": 0, "group_of_trees": 0}
    for e in entries:
        for item in e.items:
            if item.cls in total:
                total[item.cls] += 1
    return total


def dataset_report(entries, image_dir: Path, sample_count=3):
    dist = class_distribution(entries)
    p("Class counts:", dist)

    both = find_images_with_both(entries)
    p("Images containing both classes:", len(both))

    samples = entries[:sample_count]

    for e in samples:
        p("Image:", e.image_path.name)
        explore_color_overlay(e, image_dir)
        explore_bboxes(e, image_dir)
        explore_image(e, image_dir)


# From C:\github\Tree-Canopy-Detection\src\exploration\enhancement.py
import cv2
import numpy as np


def to_gray( image: np.ndarray ) -> np.ndarray:
    """
    Convert an RGB image to grayscale.
    """
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def equalize_hist( gray: np.ndarray ) -> np.ndarray:
    """
    Apply histogram equalization to a grayscale image.
    """
    return cv2.equalizeHist(gray)


def clahe_enhance( gray: np.ndarray, clip: float = 2.0, tile: int = 8 ) -> np.ndarray:
    """
    Apply CLAHE to improve local contrast.
    """
    clahe = cv2.createCLAHE(clipLimit = clip, tileGridSize = (tile, tile))
    return clahe.apply(gray)


def sharpen( gray: np.ndarray ) -> np.ndarray:
    """
    Apply a basic sharpening filter to enhance edges.
    """
    kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype = np.float32,
    )
    out = cv2.filter2D(gray, -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize( gray: np.ndarray ) -> np.ndarray:
    """
    Normalize pixel values to zero to one.
    """
    g = gray.astype(np.float32)
    m = g.min()
    M = g.max()
    if M <= m:
        return g
    return (g - m) / (M - m)


def enhance_image_for_segmentation( image: np.ndarray ) -> tuple:
    """
    Full enhancement pipeline used in notebooks.
    Returns enhanced image and intermediate stages.

    Steps:
    1. convert to grayscale
    2. equalize histogram
    3. apply CLAHE
    4. sharpen
    5. normalize to zero to one
    """

    stages = { }

    gray = to_gray(image)
    stages["gray"] = gray

    eq = equalize_hist(gray)
    stages["equalized"] = eq

    clahe = clahe_enhance(eq)
    stages["clahe"] = clahe

    sharp = sharpen(clahe)
    stages["sharpened"] = sharp

    norm = normalize(sharp)
    stages["normalized"] = norm

    return norm, stages


# From C:\github\Tree-Canopy-Detection\src\exploration\filters.py
import cv2
import numpy as np


def cv2_apply_gaussian(
    image: np.ndarray, ksize: int = 5, sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian blur to an RGB image.
    """
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def cv2_apply_sobel(image: np.ndarray) -> np.ndarray:
    """
    Apply Sobel edge detection to a grayscale or RGB image.
    If RGB, converts to grayscale internally.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(dx * dx + dy * dy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag


def cv2_apply_laplacian(image: np.ndarray) -> np.ndarray:
    """
    Apply Laplacian edge detection.
    Converts to grayscale if needed.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    out = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    return out


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize to range zero to one.
    """
    img = image.astype(np.float32)
    m = img.min()
    M = img.max()
    if M <= m:
        return img
    return (img - m) / (M - m)


# From C:\github\Tree-Canopy-Detection\src\exploration\kernels.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.exploration.visualize import show_side_by_side


def laplacian_kernel() -> np.ndarray:
    """
    Standard 3x3 Laplacian kernel.
    """
    return np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )


def visualize_kernel(kernel, title="Kernel", cmap=None):
    """
    Visualize a 2D convolution kernel as an image and 3D surface.
    """

    # Auto-select colormap if not provided
    if cmap is None:
        if np.any(kernel < 0):
            cmap = "seismic"  # diverging: shows negative vs positive clearly
        else:
            cmap = "gray"  # sequential: good for blur kernels

    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.patch.set_facecolor("white")

    # 2D heatmap
    ax2d = fig.add_subplot(1, 2, 1)
    ax2d.imshow(kernel, cmap=cmap)
    ax2d.set_title(f"{title.replace('_', ' ')} (2D heatmap)", fontsize=8)
    ax2d.axis("off")

    # 3D surface
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    x = np.arange(kernel.shape[1])
    y = np.arange(kernel.shape[0])
    X, Y = np.meshgrid(x, y)
    ax3d.plot_surface(X, Y, kernel, cmap=cmap, edgecolor="k")
    ax3d.set_title(f"{title.replace('_', ' ')} (3D surface)", fontsize=8)

    plt.show()


def apply_kernel_using_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a kernel to a grayscale image using convolution.
    """
    from scipy.signal import convolve2d

    if image.ndim == 3:
        raise ValueError("apply_kernel expects a single channel image")

    out = convolve2d(image, kernel, mode="same", boundary="symm")
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def apply_custom_kernel(image, kernel, show_image=False):
    """
    Apply a custom kernel to an image and visualize the response.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    filtered = cv2.filter2D(gray, -1, kernel)

    if show_image:
        show_side_by_side(
            gray,
            filtered,
            titles=["Original", "Kernel Response"],
            cmaps=["gray", "gray"],
        )

    return filtered


def get_kernels(name: str = "all"):
    """
    Return a dictionary of named standard kernels for exploration.
    """
    kernels = {
        "Identity": np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        ),
        "Sobel_X": np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ],
            dtype=np.float32,
        ),
        "Sobel_Y": np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ],
            dtype=np.float32,
        ),
        "Prewitt_X": np.array(
            [
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1],
            ],
            dtype=np.float32,
        ),
        "Prewitt_Y": np.array(
            [
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1],
            ],
            dtype=np.float32,
        ),
        "Scharr_X": np.array(
            [
                [-3, 0, 3],
                [-10, 0, 10],
                [-3, 0, 3],
            ],
            dtype=np.float32,
        ),
        "Scharr_Y": np.array(
            [
                [-3, -10, -3],
                [0, 0, 0],
                [3, 10, 3],
            ],
            dtype=np.float32,
        ),
        "Roberts_X": np.array(
            [
                [1, 0],
                [0, -1],
            ],
            dtype=np.float32,
        ),
        "Roberts_Y": np.array(
            [
                [0, 1],
                [-1, 0],
            ],
            dtype=np.float32,
        ),
        "Laplacian_3x3": np.array(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0],
            ],
            dtype=np.float32,
        ),
        "Laplacian_5x5": np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float32,
        ),
        "Box_Blur_3x3": np.ones((3, 3), dtype=np.float32) / 9,
        "Box_Blur_5x5": np.ones((5, 5), dtype=np.float32) / 25,
        "cv2_Gaussian_3x3": cv2.getGaussianKernel(3, 1) @ cv2.getGaussianKernel(3, 1).T,
        "cv2_Gaussian_5x5": cv2.getGaussianKernel(5, 1) @ cv2.getGaussianKernel(5, 1).T,
        "Sharpen_Basic": np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        ),
        "High_Boost": np.array(
            [
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1],
            ],
            dtype=np.float32,
        ),
        "Emboss_1": np.array(
            [
                [-2, -1, 0],
                [-1, 1, 1],
                [0, 1, 2],
            ],
            dtype=np.float32,
        ),
        "Emboss_2": np.array(
            [
                [-1, -1, 0],
                [-1, 0, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        ),
        "Edge_Enhance": np.array(
            [
                [0, 0, 0],
                [-1, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        ),
        "Edge_Enhance_Strong": np.array(
            [
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1],
            ],
            dtype=np.float32,
        ),
        "Motion_Blur_5x5": np.eye(5, dtype=np.float32) / 5,
        "Motion_Blur_9x9": np.eye(9, dtype=np.float32) / 9,
        "Gradient_Magnitude": np.array(
            [
                [1, 1, 1],
                [1, -8, 1],
                [1, 1, 1],
            ],
            dtype=np.float32,
        ),
        "High_Pass_3x3": np.array(
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1],
            ],
            dtype=np.float32,
        ),
    }

    sobel_x = kernels["Sobel_X"]
    sobel_y = kernels["Sobel_Y"]
    kernels["Sobel"] = np.sqrt(sobel_x**2 + sobel_y**2)

    def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        Generate a 2D Gaussian kernel.
        """
        k = size // 2
        x = np.arange(-k, k + 1)
        y = np.arange(-k, k + 1)
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
        kernel /= kernel.sum()
        return kernel.astype(np.float32)

    for i in range(1, 3):
        kernels[f"Gaussian_2x2_sigma{i}"] = gaussian_kernel(size=2, sigma=i)
        kernels[f"Gaussian_3x3_sigma{i}"] = gaussian_kernel(size=3, sigma=i)
        kernels[f"Gaussian_5x5_sigma{i}"] = gaussian_kernel(size=5, sigma=i)
        kernels[f"Gaussian_7x7_sigma{i}"] = gaussian_kernel(size=7, sigma=i)
        kernels[f"Gaussian_9x9_sigma{i}"] = gaussian_kernel(size=9, sigma=i)

    if name == "keys":
        return kernels.keys()

    if name == "all":
        return kernels

    for key in kernels:
        if key.lower() == name.lower():
            return kernels[key]

    raise ValueError(f"Unknown kernel '{name}'")


def display_standard_kernels(cmap="Greens_r"):
    """
    Display all standard kernels from get_standard_kernels as heatmaps.
    """
    kernels = get_kernels()

    n = len(kernels)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = axes.flatten()

    for ax, (name, kernel) in zip(axes, kernels.items()):
        vmax = np.max(np.abs(kernel))
        im = ax.imshow(kernel, cmap=cmap, vmin=-vmax, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)

        all_ints = np.allclose(kernel, np.round(kernel))
        if not all_ints:
            decimals = []
            for val in kernel.flatten():
                if val != 0:
                    s = f"{val:.8f}".rstrip("0").rstrip(".")
                    if "." in s:
                        decimals.append(len(s.split(".")[1]))
            max_decimals = max(decimals) if decimals else 0
            if max_decimals == 1:
                fmt = "{:.1f}"
            elif max_decimals == 2:
                fmt = "{:.2f}"
            else:
                fmt = "{:.3f}"
        else:
            fmt = "{}"

        for (i, j), val in np.ndenumerate(kernel):
            rgba = cmap_obj(norm(val))
            brightness = np.dot(rgba[:3], [0.299, 0.587, 0.114])
            text_color = "black" if brightness > 0.5 else "white"
            if all_ints or val == 0:
                text_str = f"{int(val)}"
            else:
                text_str = fmt.format(val)

            ax.text(
                j, i, text_str, ha="center", va="center", fontsize=6, color=text_color
            )

        ax.set_title(name.replace("_", " "), fontsize=6)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(kernels) :]:
        ax.axis("off")

    plt.suptitle("Standard Kernels (Values and Structure)", fontsize=8, y=1.02)
    plt.tight_layout()
    plt.show()


def make_motion_kernel(size: int = 9, angle: float = 0.0) -> np.ndarray:
    """
    Create a motion blur kernel of a given size and angle.
    """
    if size % 2 == 0:
        size += 1

    kernel = np.zeros((size, size), dtype=np.float32)
    cv2.line(kernel, (size // 2, 0), (size // 2, size - 1), color=1.0, thickness=1)

    # Rotate the vertical line to desired angle
    center = (size // 2, size // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (size, size))

    # Normalize so sum = 1
    s = kernel.sum()
    if s != 0:
        kernel /= s
    return kernel


def make_gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Create a 2D Gaussian blur kernel.
    """
    if size % 2 == 0:
        size += 1

    gk = cv2.getGaussianKernel(size, sigma)
    kernel = gk @ gk.T
    kernel /= kernel.sum()
    return kernel


def make_directional_edge_kernel(
    size: int = 3, direction: str = "horizontal"
) -> np.ndarray:
    """
    Create a simple directional edge detection kernel.

    """
    if size < 3:
        raise ValueError("size must be at least 3")

    base = np.zeros((size, size), dtype=float)

    if direction == "horizontal":
        base[size // 2, :] = np.linspace(-1, 1, size)
    elif direction == "vertical":
        base[:, size // 2] = np.linspace(-1, 1, size)
    elif direction == "diag_pos":
        np.fill_diagonal(base[:, ::-1], np.linspace(-1, 1, size))
    elif direction == "diag_neg":
        np.fill_diagonal(base, np.linspace(-1, 1, size))
    else:
        raise ValueError(
            "direction must be one of: horizontal, vertical, diag_pos, diag_neg"
        )

    denom = np.sum(np.abs(base))
    if denom != 0:
        base /= denom
    return base


# From C:\github\Tree-Canopy-Detection\src\exploration\visualize.py
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
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
                # Default: clip to 0–255 and cast to uint8
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
    # Convert float images safely
    if image.dtype != np.uint8:
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
    Show a binary mask as pure black and white.
    Uses PIL to avoid Matplotlib auto scaling side effects.
    """

    # normalize mask to 0 and 255
    if mask.dtype != np.uint8:
        # convert float or int mask to binary (0 or 255)
        mask_img = (mask > 0.5).astype(np.uint8) * 255
    else:
        if mask.max() <= 1:
            mask_img = mask * 255
        else:
            # if mask is uint8 but noisy, re-binarize
            mask_img = (mask > 127).astype(np.uint8) * 255

    if return_img:
        return mask_img

    # use PIL for exact grayscale
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
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    mask_u8 = (mask * 255).astype(np.uint8)
    mask_rgb = np.zeros_like(img_u8)
    mask_rgb[:, :, 0] = mask_u8

    overlay = cv2.addWeighted(img_u8, 1 - alpha, mask_rgb, alpha, 0)

    if return_img:
        return overlay
    else:
        if image.max() <= 1.0:
            img_u8 = (image * 255).astype(np.uint8)
        else:
            img_u8 = image.astype(np.uint8)

        mask_u8 = (mask * 255).astype(np.uint8)
        mask_rgb = np.zeros_like(img_u8)
        mask_rgb[:, :, 0] = mask_u8

        overlay = cv2.addWeighted(img_u8, 1 - alpha, mask_rgb, alpha, 0)

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


# From C:\github\Tree-Canopy-Detection\src\exploration\__init__.py


