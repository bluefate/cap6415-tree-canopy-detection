class CustomNormalize6Channel:
    """
    Custom normalization for 6-channel images (RGB + 3 filter channels).
    
    Applies separate normalization to RGB channels (ImageNet stats) and 
    filter channels (custom stats). Used in Albumentations pipelines for 
    concat mode where first 3 channels are RGB and last 3 are filtered outputs.
    """

    def __init__(self):
        """
        Initialize normalization parameters for 6-channel images.
        
        Sets up ImageNet statistics for RGB channels and custom statistics for filter channels.
        """
        # ImageNet normalization stats for RGB channels (0:3)
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]

        # Custom stats for filter channels (3:6)
        # Can be replaced with computed statistics from filter outputs
        self.filter_mean = [0.5, 0.5, 0.5]
        self.filter_std = [0.5, 0.5, 0.5]

    def __call__(self, image, **kwargs):
        """
        Apply normalization to 6-channel image.
        
        Args:
            image (np.ndarray): Input image with shape (H, W, 6).
        
        Returns:
            dict: Dictionary with normalized image under 'image' key.
        """
        if len(image.shape) == 3 and image.shape[2] == 6:
            # Normalize RGB channels (0:3)
            for i in range(3):
                image[:, :, i] = (
                    image[:, :, i] / 255.0 - self.rgb_mean[i]
                ) / self.rgb_std[i]

            # Normalize filter channels (3:6)
            for i in range(3, 6):
                image[:, :, i] = (
                    image[:, :, i] / 255.0 - self.filter_mean[i - 3]
                ) / self.filter_std[i - 3]

        return {"image": image}
