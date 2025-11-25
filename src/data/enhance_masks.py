from exploration.enhancement import clahe_enhance
from src.data.loaders import ImageMaskDataset


class EnhancedImageMaskDataset(ImageMaskDataset):
    """
    Extended dataset that applies filter enhancements during loading.

    Can operate in 3 modes:
    1. 'rgb' - Original 3-channel RGB
    2. 'filtered' - Top 3 filters as RGB channels
    3. 'concat' - 6-channel (RGB + 3 filters)
    """

    def __init__(
        self,
        entries,
        image_dir,
        mode="rgb",
        filter_names=None,
        classes=None,
        transform=None,
    ):
        super().__init__(entries, image_dir, classes, transform)
        self.mode = mode
        self.filter_names = filter_names or ["laplacian", "sobel", "clahe"]

    def apply_filters(self, img):
        """
        Apply specified filters and return as 3-channel image.

        Handles both kernel-based and algorithmic filters with consistent
        naming through the centralized filter registry approach.
        """
        from src.exploration.kernels import get_kernels
        from src.exploration.enhancement import to_gray
        import cv2
        import numpy as np

        gray = to_gray(img)

        # loading kernel bank dynamically
        kernel_bank = get_kernels("all")

        # create callable filters dynamically from kernel bank
        dynamic_kernels = {
            name.lower(): lambda k=kernel: cv2.filter2D(gray, -1, k)
            for name, kernel in kernel_bank.items()
        }

        dynamic_kernels.update(
            {
                # Edge detection
                "laplacian": lambda: cv2.Laplacian(gray, cv2.CV_64F),
                "sobel": lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0)
                + cv2.Sobel(gray, cv2.CV_64F, 0, 1),
                "sobel_x_cv": lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0),
                "sobel_y_cv": lambda: cv2.Sobel(gray, cv2.CV_64F, 0, 1),
                "canny": lambda: cv2.Canny(gray, 50, 150),
                # Enhancement
                "clahe": lambda: clahe_enhance(gray),
                "histogram_eq": lambda: cv2.equalizeHist(gray),
                # Gaussian blur variants - THIS FIXES THE gaussian_3x3 issue
                "gaussian_3x3": lambda: cv2.GaussianBlur(gray, (3, 3), 1.0),
                "gaussian_5x5": lambda: cv2.GaussianBlur(gray, (5, 5), 1.5),
                "gaussian_7x7": lambda: cv2.GaussianBlur(gray, (7, 7), 2.0),
                "gaussian_9x9": lambda: cv2.GaussianBlur(gray, (9, 9), 2.5),
                # Other useful filters
                "bilateral": lambda: cv2.bilateralFilter(gray, 9, 75, 75),
                "median_3x3": lambda: cv2.medianBlur(gray, 3),
                "median_5x5": lambda: cv2.medianBlur(gray, 5),
            }
        )

        channels = []
        for fname in self.filter_names:
            key = fname.lower()
            if key not in dynamic_kernels:
                raise KeyError(
                    f"Unknown filter '{fname}'. Available: {list(dynamic_kernels.keys())}"
                )

            filtered = dynamic_kernels[key]()
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            channels.append(filtered)

        # Ensure at least 3 channels
        if len(channels) == 0:
            raise RuntimeError(f"No filters produced output for {self.filter_names}")

        while len(channels) < 3:
            channels.append(channels[-1])

        return np.stack(channels[:3], axis=2)

    @classmethod
    def get_available_filters(cls):
        """
        Get list of all available filter names.
        """
        from src.exploration.kernels import get_kernels

        # Get kernel-based filters
        kernel_bank = get_kernels("all")
        kernel_names = [name.lower() for name in kernel_bank.keys()]

        # Algorithmic filters (always available)
        algorithmic = [
            "laplacian",
            "sobel",
            "clahe",
            "gaussian_3x3",
            "gaussian_5x5",
            "gaussian_7x7",
            "gaussian_9x9",
            "sobel_x_cv",
            "sobel_y_cv",
            "canny",
            "histogram_eq",
            "bilateral",
            "median_3x3",
            "median_5x5",
        ]

        return sorted(set(kernel_names + algorithmic))
