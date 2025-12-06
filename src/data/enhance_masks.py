import cv2
import numpy as np
import torch

from src.data.loaders import ImageMaskDataset
from src.data.masks import build_multiclass_mask


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
        """Fixed tensor processing to avoid double normalization."""
        entry = self.entries[idx]
        img_path = self.image_dir / entry.image_path.name

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Build mask
        if self.classes is None:
            mask = build_multiclass_mask(entry)
        else:
            from src.data.annotations import AnnotationEntry

            filtered_items = [item for item in entry.items if item.cls in self.classes]
            filtered_entry = AnnotationEntry(
                entry.image_path, entry.width, entry.height, filtered_items
            )
            mask = build_multiclass_mask(filtered_entry)

        # Apply filters BEFORE transforms to avoid double normalization
        if self.mode == "filtered":
            image = self.apply_filters_to_enhanced_image(image)
        elif self.mode == "concat":
            filtered_img = self.apply_filters_to_enhanced_image(image)
            image = np.concatenate([image, filtered_img], axis=2)

        # Apply transforms (includes normalization)
        if self.transform:
            processed = self.transform(image=image, mask=mask)
            image = processed["image"]  # Should be tensorized by ToTensorV2
            mask = processed["mask"]

            # Normalize mask to class indices 0,1,2
            if mask.max() > 2:
                mask = (mask / 255).astype(np.uint8)

        # Safety check: Convert image to tensor if not already done
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        else:
            # Ensure proper format if already tensor
            if image.ndim == 3 and image.shape[0] not in [3, 6]:  # Not CHW format
                image = image.permute(2, 0, 1)
            if image.max() > 1.0:  # Not normalized
                image = image / 255.0

        # Convert mask to tensor
        if isinstance(mask, torch.Tensor):
            mask_t = mask.long()
        else:
            mask_t = torch.from_numpy(mask).long()

        while mask_t.ndim > 2:
            mask_t = mask_t.squeeze(0)

        assert mask_t.ndim == 2, f"Mask should be 2D [H, W], got {mask_t.shape}"

        return image, mask_t

    def apply_filters_to_enhanced_image(self, img):
        """
        Apply specified filters and return as 3-channel image.
        """
        from src.data.image_loader import create_enhanced_image

        return create_enhanced_image(img, self.filter_names)

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
