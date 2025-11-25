from typing import List, Optional

import cv2
import numpy as np
import torch

from src.data.loaders import ImageMaskDataset
from src.data.masks import build_multi_mask


class EnhancedImageMaskDataset(ImageMaskDataset):
    """
    Extended dataset that applies filter enhancements during loading.

    Can operate in 3 modes:
    1. 'rgb' - Original 3-channel RGB
    2. 'filtered' - Top 3 filters as RGB channels
    3. 'concat' - 6-channel (RGB + 3 filters)
    """

    # Default filters if none specified
    DEFAULT_FILTERS = ['laplacian', 'sobel', 'clahe']

    def __init__(
            self, entries,
            image_dir,
            mode = 'rgb',
            filter_names: Optional[List[str]] = None,
            classes: Optional[List[str]] = None,
            transform = None
    ):
        super().__init__(entries, image_dir, classes, transform)
        self.mode = mode
        self.filter_names = filter_names or self.DEFAULT_FILTERS

        # Validate mode
        if mode not in ['rgb', 'filtered', 'concat']:
            raise ValueError(
                    f"Invalid mode '{mode}'. Must be one of: 'rgb', 'filtered', 'concat'"
            )

        # Validate filter names on initialization (fail-fast)
        if mode in ['filtered', 'concat']:
            self._validate_filters()

    def _validate_filters(self) -> None:
        """
        Validate filter names against available filters.
        """
        available = self._get_available_filters()

        invalid_filters = []
        for fname in self.filter_names:
            if fname.lower() not in available:
                invalid_filters.append(fname)

        if invalid_filters:
            suggestions = []
            for inv in invalid_filters:
                sugg = self._suggest_filter(inv, available)
                if sugg:
                    suggestions.append(f"'{inv}' -> '{sugg}'")

            suggestion_str = ""
            if suggestions:
                suggestion_str = f" Suggestions: {', '.join(suggestions)}"

            raise KeyError(
                    f"Unknown filter(s): {invalid_filters}. "
                    f"Available filters: {sorted(available)[:20]}...{suggestion_str}"
            )

    @staticmethod
    def _apply_clahe(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        return clahe.apply(gray)

    @staticmethod
    def _suggest_filter(invalid_name: str, available: set) -> Optional[str]:
        """
        Suggest a correction for an invalid filter name.

        Args:
            invalid_name: Invalid filter name
            available: Set of available filter names

        Returns:
            Suggested filter name or None
        """
        invalid_lower = invalid_name.lower()

        # Try partial match
        for name in available:
            if invalid_lower in name or name in invalid_lower:
                return name

        return None

    def _get_available_filters(self) -> set:
        """
        Get set of available filter names.
        """
        # Import here to avoid circular imports
        from src.exploration.kernels import get_kernels

        # Get kernel-based filters
        kernel_bank = get_kernels("all")
        kernel_names = {name.lower() for name in kernel_bank.keys()}

        # Algorithmic filters (always available)
        algorithmic = {
            'laplacian', 'sobel', 'clahe',
            'gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7', 'gaussian_9x9',
            'sobel_x_cv', 'sobel_y_cv', 'canny',
            'histogram_eq', 'bilateral',
            'median_3x3', 'median_5x5',
        }

        return kernel_names | algorithmic


    def apply_filters(self, img):
        """
        Apply specified filters and return as 3-channel image.

        Handles both kernel-based and algorithmic filters with consistent
        naming through the centralized filter registry approach.
        """
        from src.exploration.kernels import get_kernels

        # Convert to grayscale for filter application
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        target_h, target_w = img.shape[:2]

        # loading kernel bank dynamically
        kernel_bank = get_kernels("all")
        filter_registry = {}

        for kname, kernel in kernel_bank.items():
            filter_registry[kname.lower()] = lambda k=kernel: cv2.filter2D(gray, -1, k)

        # Add algorithmic filters (OpenCV native)
        filter_registry.update({
            'laplacian': lambda: cv2.Laplacian(gray, cv2.CV_64F),
            'sobel': lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0) + cv2.Sobel(gray, cv2.CV_64F, 0, 1),
            'sobel_x_cv': lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0),
            'sobel_y_cv': lambda: cv2.Sobel(gray, cv2.CV_64F, 0, 1),
            'canny': lambda: cv2.Canny(gray, 50, 150),
            'clahe': lambda: self._apply_clahe(gray),
            'histogram_eq': lambda: cv2.equalizeHist(gray),
            'gaussian_3x3': lambda: cv2.GaussianBlur(gray, (3, 3), 1.0),
            'gaussian_5x5': lambda: cv2.GaussianBlur(gray, (5, 5), 1.5),
            'gaussian_7x7': lambda: cv2.GaussianBlur(gray, (7, 7), 2.0),
            'gaussian_9x9': lambda: cv2.GaussianBlur(gray, (9, 9), 2.5),
            'bilateral': lambda: cv2.bilateralFilter(gray, 9, 75, 75),
            'median_3x3': lambda: cv2.medianBlur(gray, 3),
            'median_5x5': lambda: cv2.medianBlur(gray, 5),
        })


        # Apply requested filters
        channels = []
        for fname in self.filter_names:
            key = fname.lower()

            if key not in filter_registry:
                # This shouldn't happen if validation passed, but handle gracefully
                available = sorted(filter_registry.keys())
                raise KeyError(
                        f"Unknown filter '{fname}'. "
                        f"Available filters ({len(available)}): {available[:20]}..."
                )

            try:
                # Apply filter
                filtered = filter_registry[key]()

                # Ensure 2D (grayscale)
                if filtered.ndim == 3:
                    filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)

                # Resize if shape doesn't match (shouldn't happen, but safety check)
                if filtered.shape != (target_h, target_w):
                    filtered = cv2.resize(
                            filtered, (target_w, target_h),
                            interpolation=cv2.INTER_LINEAR
                    )

                # Normalize to uint8
                if filtered.dtype != np.uint8:
                    filtered = cv2.normalize(
                            filtered, None, 0, 255, cv2.NORM_MINMAX
                    ).astype(np.uint8)

                channels.append(filtered)

            except Exception as e:
                raise RuntimeError(f"Filter '{fname}' failed: {e}")

        # Ensure we have at least 3 channels
        if len(channels) == 0:
            raise RuntimeError(f"No filters produced output for {self.filter_names}")

        while len(channels) < 3:
            channels.append(channels[-1].copy())

        # Stack into 3-channel image
        return np.stack(channels[:3], axis=2)





    def __getitem__( self, idx ):
        """Override to apply filter enhancement."""
        entry = self.entries[idx]
        img_path = self.image_dir / entry.image_path.name

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        # Generate mask
        if self.classes is None:
            polys = [item.segmentation for item in entry.items]
        else:
            polys = [item.segmentation for item in entry.items if item.cls in self.classes]

        mask = build_multi_mask(polys, W, H)

        # Apply filter enhancement based on mode
        if self.mode == 'filtered':
            image = self.apply_filters(image)
        elif self.mode == 'concat':
            filtered = self.apply_filters(image)
            # Concatenate RGB + filtered (6 channels)
            image = np.concatenate([image, filtered], axis = 2)
        # else: mode == 'rgb', use original image

            # Apply augmentations
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # Convert to tensors with proper shape handling
            img_t = self._to_image_tensor(image)
            mask_t = self._to_mask_tensor(mask)

            return img_t, mask_t

        @staticmethod
        def _to_image_tensor(image) -> torch.Tensor:
            """
            Convert image to tensor with shape (C, H, W).
            """
            if isinstance(image, torch.Tensor):
                img_t = image.float()
                # Ensure CHW format
                if img_t.ndim == 3 and img_t.shape[0] not in [3, 6]:
                    img_t = img_t.permute(2, 0, 1)
            else:
                # numpy HWC array
                img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

            return img_t

        @staticmethod
        def _to_mask_tensor(mask) -> torch.Tensor:
            """
            Convert mask to tensor with shape (1, H, W).
            """
            if isinstance(mask, torch.Tensor):
                mask_t = mask.float()
                if mask_t.ndim == 2:
                    mask_t = mask_t.unsqueeze(0)
            else:
                mask = mask.astype('float32')
                if mask.ndim == 2:
                    mask = mask[None, ...]  # Add channel dimension
                mask_t = torch.from_numpy(mask)

            return mask_t

        @classmethod
        def get_available_filters(cls) -> List[str]:
            """
            Get list of all available filter names.
            """
            # Create dummy instance to get filters
            from src.exploration.kernels import get_kernels

            kernel_bank = get_kernels("all")
            kernel_names = [name.lower() for name in kernel_bank.keys()]

            algorithmic = [
                'laplacian', 'sobel', 'clahe',
                'gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7', 'gaussian_9x9',
                'sobel_x_cv', 'sobel_y_cv', 'canny',
                'histogram_eq', 'bilateral',
                'median_3x3', 'median_5x5',
            ]

            return sorted(set(kernel_names + algorithmic))