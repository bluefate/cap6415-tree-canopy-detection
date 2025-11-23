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

    def __init__( self, entries, image_dir, mode = 'rgb', filter_names = None, classes = None, transform = None ):
        super().__init__(entries, image_dir, classes, transform)
        self.mode = mode
        self.filter_names = filter_names or ['laplacian', 'sobel', 'clahe']

    def apply_filters( self, img ):
        """Apply the specified filters and return as channels."""
        from src.exploration.enhancement import clahe_enhance, to_gray
        from src.exploration.filters import cv2_apply_laplacian, cv2_apply_sobel

        gray = to_gray(img)
        target_h, target_w = img.shape[:2]

        filter_map = {
            'laplacian': lambda: cv2_apply_laplacian(img),
            'sobel':     lambda: cv2_apply_sobel(img),
            'clahe':     lambda: clahe_enhance(gray),
        }

        channels = []
        for fname in self.filter_names:
            if fname in filter_map:
                filtered = filter_map[fname]()
                #
                # # Ensure 2D (grayscale)
                # if filtered.ndim == 3:
                #     filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
                #
                # # Resize if needed
                # if filtered.shape != (target_h, target_w):
                #     filtered = cv2.resize(filtered, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                #
                # Normalize to 0-255
                if filtered.dtype != np.uint8:
                    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                channels.append(filtered)

        # Ensure we have exactly 3 channels
        while len(channels) < 3:
            #channels.append(channels[-1].copy())
            channels.append(channels[-1])

        return np.stack(channels[:3], axis = 2)

    def __getitem__( self, idx ):
        """Override to apply filter enhancement."""
        entry = self.entries[idx]
        img_path = self.image_dir / entry.image_path.name

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

        from src.data.masks import build_multi_mask

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
            augmented = self.transform(image = image, mask = mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors
        if isinstance(image, torch.Tensor):
            img_t = image.float()
            if img_t.ndim == 3 and img_t.shape[0] not in [3, 6]:
                img_t = img_t.permute(2, 0, 1)
        else:
            img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if isinstance(mask, torch.Tensor):
            mask_t = mask.float()
            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0)
        else:
            mask = mask.astype('float32')
            if mask.ndim == 2:
                mask = mask[None, ...]
            mask_t = torch.from_numpy(mask)

        return img_t, mask_t