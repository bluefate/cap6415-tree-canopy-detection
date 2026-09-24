import cv2
import numpy as np


def crop_bbox_with_context(image_dir, entry, item, context_factor=1.5):
    """
    Crop bounding box with natural background context.
    
    Expands bbox edges to include surrounding context without artificial padding,
    instead providing real background pixels for training.
    
    Args:
        image_dir (Path): Directory containing images.
        entry: Annotation entry with image metadata.
        item: Annotation item with segmentation and bbox.
        context_factor (float): Expansion factor (1.5 = 50% more on each side). 
                               Defaults to 1.5.
    
    Returns:
        tuple: (cropped_image_rgb, context_bbox) where context_bbox is 
               (x1, y1, x2, y2) coordinates in original image space.
    
    Raises:
        FileNotFoundError: If image file not found.
        ValueError: If image cannot be loaded.
    """
    img_path = image_dir / entry.image_path.name
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]

    # Get original bbox
    x1, y1, x2, y2 = item.bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Calculate context expansion
    context_w = int(bbox_w * (context_factor - 1) / 2)
    context_h = int(bbox_h * (context_factor - 1) / 2)

    # Expand bbox with context, but clip to image boundaries
    new_x1 = max(0, x1 - context_w)
    new_y1 = max(0, y1 - context_h)
    new_x2 = min(img_w, x2 + context_w)
    new_y2 = min(img_h, y2 + context_h)

    # Crop with natural context
    crop = img[new_y1:new_y2, new_x1:new_x2]

    return crop, (new_x1, new_y1, new_x2, new_y2)


def crop_mask_with_context(entry, item, context_bbox):
    """
    Extract mask for context-expanded crop region.
    
    Creates binary mask covering the segmentation polygon, then crops to 
    match the context-expanded bbox.
    
    Args:
        entry: Annotation entry with height/width.
        item: Annotation item with segmentation.
        context_bbox (tuple): Expanded bbox coordinates (x1, y1, x2, y2) in original space.
    
    Returns:
        np.ndarray: Binary mask for cropped region.
    """
    new_x1, new_y1, new_x2, new_y2 = context_bbox

    # Create full image mask
    mask = np.zeros((entry.height, entry.width), dtype=np.uint8)
    seg = item.segmentation
    if seg and len(seg) >= 4:
        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], 1)

    # Crop mask to match context crop
    mask_crop = mask[new_y1:new_y2, new_x1:new_x2]
    return mask_crop


def crop_bbox(image_dir, entry, item):
    """
    Crop image to bounding box without context padding.
    
    Args:
        image_dir (Path): Directory containing images.
        entry: Annotation entry with image metadata.
        item: Annotation item with bbox coordinates.
    
    Returns:
        np.ndarray: Cropped image (RGB).
    
    Raises:
        FileNotFoundError: If image file not found.
        ValueError: If image cannot be loaded.
    """
    img_path = image_dir / entry.image_path.name
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = item.bbox
    return img[y1:y2, x1:x2]


def crop_mask(entry, item):
    """
    Extract binary mask for bounding box region without context.
    
    Args:
        entry: Annotation entry with height/width.
        item: Annotation item with segmentation and bbox.
    
    Returns:
        np.ndarray: Binary mask for bbox region.
    """
    mask = np.zeros((entry.height, entry.width), dtype=np.uint8)
    seg = item.segmentation
    if seg and len(seg) >= 4:
        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], 1)
    x1, y1, x2, y2 = item.bbox
    return mask[y1:y2, x1:x2]


def pad_to_size(img, target_h, target_w):
    """
    Pad image with zeros to reach target dimensions.
    
    Padding is centered - extra space is split equally between sides.
    
    Args:
        img (np.ndarray): Input image.
        target_h (int): Target height.
        target_w (int): Target width.
    
    Returns:
        np.ndarray: Padded image with shape (target_h, target_w, channels).
    """
    h, w = img.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )


def expand_crop_with_background(img, mask, bbox, target_h, target_w):
    """
    Expand crop to target size by including natural background context.
    
    Instead of zero-padding, expands bbox to include surrounding image context. 
    Falls back to resizing if not enough background available.
    
    Args:
        img (np.ndarray): Full image.
        mask (np.ndarray): Full image mask.
        bbox (tuple): Original bbox coordinates (x1, y1, x2, y2).
        target_h (int): Target height.
        target_w (int): Target width.
    
    Returns:
        tuple: (expanded_crop_image, expanded_crop_mask) with shape (target_h, target_w).
    """
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    crop_h, crop_w = y2 - y1, x2 - x1

    # Calculate how much more we need
    need_more_h = target_h - crop_h
    need_more_w = target_w - crop_w

    if need_more_h <= 0 and need_more_w <= 0:
        # Already big enough, just return the crop
        return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    # Expand the bbox to get more background
    expand_h = max(0, need_more_h // 2)
    expand_w = max(0, need_more_w // 2)

    # Calculate new bounds, staying within image
    new_y1 = max(0, y1 - expand_h)
    new_x1 = max(0, x1 - expand_w)
    new_y2 = min(img_h, new_y1 + target_h)
    new_x2 = min(img_w, new_x1 + target_w)

    # If we still don't have enough space, adjust the other side
    if new_y2 - new_y1 < target_h:
        new_y1 = max(0, new_y2 - target_h)
    if new_x2 - new_x1 < target_w:
        new_x1 = max(0, new_x2 - target_w)

    # Extract the expanded crop
    expanded_img = img[new_y1:new_y2, new_x1:new_x2]
    expanded_mask = mask[new_y1:new_y2, new_x1:new_x2]

    # If we still need to reach exact target size, resize
    if expanded_img.shape[:2] != (target_h, target_w):
        expanded_img = cv2.resize(
            expanded_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
        expanded_mask = cv2.resize(
            expanded_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )

    return expanded_img, expanded_mask
