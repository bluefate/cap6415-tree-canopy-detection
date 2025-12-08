import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.exploration.visualize import show_side_by_side


def laplacian_kernel() -> np.ndarray:
    """
    Generate standard 3x3 Laplacian edge detection kernel.
    
    Returns:
        np.ndarray: 3x3 Laplacian kernel for edge detection.
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
    Visualize a 2D convolution kernel as heatmap and 3D surface plot.
    
    Automatically selects appropriate colormap (seismic for kernels with negative 
    values, gray for non-negative kernels).
    
    Args:
        kernel (np.ndarray): 2D convolution kernel to visualize.
        title (str): Title for the visualization. Defaults to "Kernel".
        cmap (str, optional): Matplotlib colormap name. If None, auto-selected.
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
    Apply a kernel to a grayscale image using scipy convolution.
    
    Args:
        image (np.ndarray): Grayscale input image.
        kernel (np.ndarray): 2D convolution kernel.
    
    Returns:
        np.ndarray: Convolved image with values clipped to [0, 255] as uint8.
    
    Raises:
        ValueError: If image has more than one channel.
    """
    from scipy.signal import convolve2d

    if image.ndim == 3:
        raise ValueError("apply_kernel expects a single channel image")

    out = convolve2d(image, kernel, mode="same", boundary="symm")
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def apply_custom_kernel(image, kernel, show_image=False):
    """
    Apply a custom convolution kernel to an image.
    
    Converts RGB to grayscale internally and optionally displays side-by-side 
    comparison of original and filtered result.
    
    Args:
        image (np.ndarray): Input RGB image.
        kernel (np.ndarray): 2D convolution kernel.
        show_image (bool): Whether to display comparison plot. Defaults to False.
    
    Returns:
        np.ndarray: Filtered grayscale image.
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
    Return dictionary of named standard convolution kernels for exploration.
    
    Includes edge detection (Sobel, Prewitt, Scharr, Roberts, Laplacian), 
    blur (Box, Gaussian, Motion), sharpening, embossing, and high-pass filters.
    
    Args:
        name (str): Kernel name or 'all' for all kernels. Defaults to "all".
                   Use 'keys' to get list of available names.
    
    Returns:
        dict or np.ndarray: If name='all' returns dict of all kernels. 
                           If name='keys' returns dict_keys of kernel names.
                           Otherwise returns specific kernel array.
    
    Raises:
        ValueError: If kernel name not found.
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
        Generate a 2D Gaussian kernel of specified size and standard deviation.
        
        Args:
            size (int): Kernel dimension. Defaults to 5.
            sigma (float): Gaussian standard deviation. Defaults to 1.0.
        
        Returns:
            np.ndarray: Normalized 2D Gaussian kernel.
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
    Display all standard kernels as a grid of annotated heatmaps.
    
    Creates a subplot grid showing kernel structure with values overlaid. 
    Color indicates kernel value magnitude, text shows exact values.
    
    Args:
        cmap (str): Matplotlib colormap name. Defaults to "Greens_r".
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
    Create a motion blur kernel with specified size and rotation angle.
    
    Args:
        size (int): Kernel size (will be made odd). Defaults to 9.
        angle (float): Rotation angle in degrees. Defaults to 0.0.
    
    Returns:
        np.ndarray: Normalized motion blur kernel.
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
    
    Args:
        size (int): Kernel size (will be made odd). Defaults to 5.
        sigma (float): Gaussian standard deviation. Defaults to 1.0.
    
    Returns:
        np.ndarray: Normalized 2D Gaussian kernel.
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
    Create a directional edge detection kernel.
    
    Generates a kernel with gradient values along specified direction.
    
    Args:
        size (int): Kernel size. Defaults to 3.
        direction (str): Direction for gradient. Must be one of 'horizontal', 'vertical',
                        'diag_pos' (top-left to bottom-right), 'diag_neg' (top-right to 
                        bottom-left). Defaults to 'horizontal'.
    
    Returns:
        np.ndarray: Normalized directional edge kernel.
    
    Raises:
        ValueError: If size < 3 or direction not recognized.
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
