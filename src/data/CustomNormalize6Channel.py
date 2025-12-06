class CustomNormalize6Channel:
    """Custom normalization for 6-channel images (RGB + 3 filters)."""

    def __init__(self):
        # ImageNet stats for RGB channels
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]

        # Custom stats for filter channels (you might want to compute these)
        self.filter_mean = [0.5, 0.5, 0.5]
        self.filter_std = [0.5, 0.5, 0.5]

    def __call__(self, image, **kwargs):
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
