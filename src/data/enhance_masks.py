import cv2
import numpy as np
import torch

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

    def __getitem__(self, idx: int):
        """Override to ensure mask dtype is float for BCEWithLogitsLoss."""
        entry = self.entries[idx]
        img_path = self.image_dir / entry.image_path.name

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Build mask
        if self.classes is None:
            from src.data.masks import build_multiclass_mask

            mask = build_multiclass_mask(entry)
        else:
            from src.data.annotations import AnnotationEntry
            from src.data.masks import build_multiclass_mask

            filtered_items = [item for item in entry.items if item.cls in self.classes]
            filtered_entry = AnnotationEntry(
                entry.image_path, entry.width, entry.height, filtered_items
            )
            mask = build_multiclass_mask(filtered_entry)

        # Apply filters based on mode
        if self.mode == "filtered":
            image = self.apply_filters(image)
        elif self.mode == "concat":
            filtered_img = self.apply_filters(image)
            # Concatenate RGB + filtered
            image = np.concatenate([image, filtered_img], axis=2)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert image to tensor
        if isinstance(image, torch.Tensor):
            img_t = image.float()
            if img_t.ndim == 3 and img_t.shape[0] != 3:
                img_t = img_t.permute(2, 0, 1)
            if img_t.max() > 1.0:
                img_t = img_t / 255.0
        else:
            img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Convert mask to tensor - LONG for CrossEntropyLoss (multi-class)
        if isinstance(mask, torch.Tensor):
            mask_t = mask.long()
        else:
            mask_t = torch.from_numpy(mask).long()

        # Ensure mask is [H, W] for multi-class CrossEntropyLoss
        while mask_t.ndim > 2:
            mask_t = mask_t.squeeze(0)

        # Verify shape is correct
        assert mask_t.ndim == 2, f"Mask should be 2D [H, W], got {mask_t.shape}"

        return img_t, mask_t

    def apply_filters(self, img):
        """
        Apply specified filters and return as 3-channel image.
        """
        from src.data.image_loader import create_enhanced_image

        return create_enhanced_image(img, self.filter_names)

    # def apply_filters(self, img):
    #     """
    #     Apply specified filters and return as 3-channel image.
    #
    #     Handles both kernel-based and algorithmic filters with consistent
    #     naming through the centralized filter registry approach.
    #     """
    #     from src.exploration.kernels import get_kernels
    #     from src.exploration.enhancement import to_gray
    #     import cv2
    #     import numpy as np
    #
    #     gray = to_gray(img)
    #
    #     # loading kernel bank dynamically
    #     kernel_bank = get_kernels("all")
    #
    #     # create callable filters dynamically from kernel bank
    #     dynamic_kernels = {
    #         name.lower(): lambda k=kernel: cv2.filter2D(gray, -1, k)
    #         for name, kernel in kernel_bank.items()
    #     }
    #
    #     dynamic_kernels.update(
    #         {
    #             # Edge detection
    #             "laplacian": lambda: cv2.Laplacian(gray, cv2.CV_64F),
    #             "sobel": lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    #             + cv2.Sobel(gray, cv2.CV_64F, 0, 1),
    #             "sobel_x_cv": lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0),
    #             "sobel_y_cv": lambda: cv2.Sobel(gray, cv2.CV_64F, 0, 1),
    #             "canny": lambda: cv2.Canny(gray, 50, 150),
    #             # Enhancement
    #             "clahe": lambda: clahe_enhance(gray),
    #             "histogram_eq": lambda: cv2.equalizeHist(gray),
    #             # Gaussian blur variants - THIS FIXES THE gaussian_3x3 issue
    #             "gaussian_3x3": lambda: cv2.GaussianBlur(gray, (3, 3), 1.0),
    #             "gaussian_5x5": lambda: cv2.GaussianBlur(gray, (5, 5), 1.5),
    #             "gaussian_7x7": lambda: cv2.GaussianBlur(gray, (7, 7), 2.0),
    #             "gaussian_9x9": lambda: cv2.GaussianBlur(gray, (9, 9), 2.5),
    #             # Other useful filters
    #             "bilateral": lambda: cv2.bilateralFilter(gray, 9, 75, 75),
    #             "median_3x3": lambda: cv2.medianBlur(gray, 3),
    #             "median_5x5": lambda: cv2.medianBlur(gray, 5),
    #         }
    #     )
    #
    #     channels = []
    #     for fname in self.filter_names:
    #         key = fname.lower()
    #         if key not in dynamic_kernels:
    #             raise KeyError(
    #                 f"Unknown filter '{fname}'. Available: {list(dynamic_kernels.keys())}"
    #             )
    #
    #         filtered = dynamic_kernels[key]()
    #         filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(
    #             np.uint8
    #         )
    #         channels.append(filtered)
    #
    #     # Ensure at least 3 channels
    #     if len(channels) == 0:
    #         raise RuntimeError(f"No filters produced output for {self.filter_names}")
    #
    #     while len(channels) < 3:
    #         channels.append(channels[-1])
    #
    #     return np.stack(channels[:3], axis=2)

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
