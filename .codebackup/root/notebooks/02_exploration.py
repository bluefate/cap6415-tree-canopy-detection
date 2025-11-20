# %% [markdown]
#
# # Notebook: 02 Exploration
# ### Purpose: explore frequency space, filters, kernels, enhancement pipelines, and visualize transformations.

# %%
import os
import sys


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.exploration.enhancement import enhance_image_for_segmentation
from src.exploration.filters import cv2_apply_gaussian, cv2_apply_laplacian, cv2_apply_sobel
from src.exploration.kernels import (apply_custom_kernel, apply_kernel_using_convolution, get_kernels, laplacian_kernel,
                                     make_directional_edge_kernel, make_gaussian_kernel, make_motion_kernel,
                                     )
from src.exploration.visualize import show_image, show_side_by_side, show_stages
from src.utils.config import Config
from src.utils.helpers import c, init_notebook, p, t


config = Config.load()

init_notebook(config.train.seed)

train_dir = config.paths.train_images
files = sorted([f for f in Path(train_dir).glob("*.*")])
sample_path = random.choice(files)

# %% [markdown]
# #### load sample image

# %%
t(sample_path.name)
img = cv2.imread(str(sample_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
##img = load_image(sample_path)

show_image(img)


# %% [markdown]
# #### Frequency analysis
#
# Magnitude map samples to help:
#
# - bright center = strong low frequency content
# - bright edges = strong high frequency noise or fine texture
# - patterns or lines = directional structure
# - rings = scale-specific periodic textures

# %%
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
fft = np.fft.fftshift(np.fft.fft2(gray))
mag = np.log1p(np.abs(fft))

cmaps = [
    # Uniform
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    # Sequential
    'Greys', 'Blues', 'Purples', 'Oranges', 'Reds',
    # Diverging
    'coolwarm', 'bwr', 'seismic', 'PiYG', 'PRGn',
    # Qualitative
    'tab10', 'tab20', 'Set1', 'Set2', 'Pastel1',
    # Miscellaneous
    'flag', 'prism', 'ocean', 'terrain', 'rainbow'
]

max_per_row = 5
n_rows = int(np.ceil(len(cmaps) / max_per_row))
n_cols = min(len(cmaps), max_per_row)

fig, axes = plt.subplots(n_rows, n_cols, figsize = (2 * n_cols, 2 * n_rows))
fig.suptitle("FFT magnitude", fontsize = 18, weight = "bold")

axes = axes.flatten()

for i, cmap in enumerate(cmaps):
    im = axes[i].imshow(mag, cmap = cmap)
    axes[i].set_title(f"{cmap}")
    axes[i].axis("off")

# hide any unused subplots
for j in range(len(cmaps), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()


# %% [markdown]
# Bright center with dimmed edges.
#
# - strong concentration of low-frequency energy
# - image is dominated by broad, smooth shapes
#
# No strong diagonal streaks.
# - when present diagonal streaks would suggest strong directional texture, such as repeating patterns or ridges
#

# %% [markdown]
# # Kernels
#
# **Hint** Adding kernel outputs as new chennels can improve the segmentation seen as modesl benefit from texture and edge cues that aren't obvious in raw RGB

# %%
gk = get_kernels("Gaussian_7x7_sigma2")
p("Gaussian kernel", gk)

lk = laplacian_kernel()
p("Laplacian kernel", lk)

sb = get_kernels("Sobel")
sx = get_kernels("Sobel_X")
sy = get_kernels("Sobel_Y")

p("Sobel kernels shapes (Sobel, Sx, Sy)", (sb.shape, sx.shape, sy.shape))


# %% [markdown]
# #### Apply kernels using exploration utilities
#
# - Laplacian output often contains negative and positive values (because it’s a derivative).
#
# CMAPS to use:
# - General Laplacian visualization "gray" Neutral, shows edges clearly
# - Very dark/light images	"seismic" or "bwr"	Diverging colormaps: negative values in one color, positive in another
# - Want high contrast edges	"hot" or "inferno"	Bright edges stand out against dark background
# - Teaching / presentation	"coolwarm"	Easy to interpret positive vs negative transitions

# %%
# img_red = img[:, :, 0]
# img_green = img[:, :, 1]
# img_blue = img[:, :, 2]


# Build  RGB image for visualization
# green_only, green_blue_only = np.zeros_like(img)
# green_only[:, :, 1] = img[:, :, 1]
# green_blue_only = img[:, :, [1, 2]]


img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def temp_app_kernels( img ):
    lap_img = apply_kernel_using_convolution(img, lk)
    sobel_x = apply_kernel_using_convolution(img, sx)
    sobel_y = apply_kernel_using_convolution(img, sy)
    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return lap_img, sobel_x, sobel_y, sobel


# Apply kernels to full grayscale
lap_gray, sobel_x_gray, sobel_y_gray, sobel_gray = temp_app_kernels(img_gray)

# Show results side by side
show_side_by_side(
        lap_gray, sobel_gray, sobel_x_gray, sobel_y_gray,
        titles = ["Laplacian", "Sobel", "Sobel X", "Sobel Y"],
        #cmaps = ["seismic"] * 4,
        cmaps = [None, "seismic", "seismic", "seismic"],
        # maxcolumns = 3,
        preserve_values = True,
        # vmax = vmax
)



# %% [markdown]
# #### Built in filters

# %%



ga = cv2_apply_gaussian(img, 5, 1.2)
so = cv2_apply_sobel(img)
la = cv2_apply_laplacian(img)

show_side_by_side(
        ga, so, la, img_gray,
        titles = ["Gaussian", "Sobel", "Laplacian", "Gray"],
        cmaps = [None, "seismic", "seismic", "seismic"]
)



# %% [markdown]
# # Enhancement pipeline

# %%
enhanced, stages = enhance_image_for_segmentation(img)
show_stages(stages)

p("enhanced.ndim", enhanced.ndim)
#show_side_by_side(img, enhanced, cmaps = [None, "grey"], titles = ["Original", "Final enhanced"])




# %% [markdown]
# #### Kernel samples

# %%



def demo_parameterized_kernels( image: np.ndarray ) -> None:
    """
    Show how parameterized kernels affect an image.
    """

    t("=== === === === === === === === === Motion blur tests === === === === === === === === ===")
    for angle in [0, 45, 90]:
        kernel = make_motion_kernel(size = 15, angle = angle)
        p("Motion Kernel", f"Angle {angle}", color1 = c.BLUE, color2 = c.BLACK)
        #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # visualize_kernel(kernel, title = f"Motion Blur {angle}°")
        result = apply_custom_kernel(image, kernel)
        show_side_by_side(
                gray, result,
                titles = ["Gray", f"Motion Blur {angle}°"],
                cmaps = ["gray", "seismic"],
                kernel = kernel
        )
    p("\n")
    t("=== === === === === === === === ===  Gaussian blur tests === === === === === === === === ===")
    for sigma in [0.5, 1.5, 3]:
        kernel = make_gaussian_kernel(size = 9, sigma = sigma)
        p("Gaussian Kernel", f"sigma={sigma}", color1 = c.BLUE, color2 = c.BLACK)
        # visualize_kernel(kernel, title = f"Gaussian σ={sigma}", )
        result = apply_custom_kernel(image, kernel)
        show_side_by_side(
                image, result,
                titles = ["Orignal", f"Gaussian σ={sigma}"],
                cmaps = [None, "seismic"],
                kernel = kernel
        )
    p("\n")
    t("=== === === === === === === === === Directional edge tests === === === === === === === === ===")
    for direction in ["horizontal", "vertical", "diag_pos", "diag_neg"]:
        kernel = make_directional_edge_kernel(size = 7, direction = direction)
        p("Directional Edge", direction, color1 = c.BLUE, color2 = c.BLACK)
        # visualize_kernel(kernel, title = f"Edge {direction}", )
        result = apply_custom_kernel(image, kernel)
        show_side_by_side(
                image, result,
                titles = ["Orignal", f"Edge {direction}"],
                cmaps = [None, "seismic"],
                kernel = kernel
        )


demo_parameterized_kernels(img)


# %%
def demo_kernels( image, names = ("Sobel_X", "Sobel_Y", "Laplacian_3x3") ):
    """
    Apply selected kernels from get_kernels() and show results with kernel visualization.
    """

    kernels = get_kernels()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    for name in names:
        if name not in kernels:
            p("Missing kernel", name, color = "red")
            continue

        kernel = kernels[name].astype(np.float32)

        # apply kernel
        filtered = cv2.filter2D(gray, -1, kernel)

        # visualize everything together
        show_side_by_side(
                gray,
                filtered,
                titles = ("gray", name),
                cmaps = ("gray", "seismic"),
                kernel = kernel,
                kernel_title = name
        )


#demo_kernels(img, names = ["Sobel_X", "Sobel_Y", "Sobel", "Laplacian_3x3"])

all_k = list(get_kernels().keys())
#demo_kernels(img, names=all_k[:6])
demo_kernels(img, names = all_k)
