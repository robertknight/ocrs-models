from torchvision import transforms


def text_recognition_data_augmentations():
    """
    Create a set of data augmentations for use with text recognition.
    """

    # Fill color for empty space created by transforms.
    # This is the "black" value for normalized images.
    transform_fill = -0.5

    augmentations = transforms.RandomApply(
        [
            transforms.RandomChoice(
                [
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.RandomRotation(
                        degrees=5,
                        fill=transform_fill,
                        expand=True,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.Pad(padding=(5, 5), fill=transform_fill),
                ]
            )
        ],
        p=0.5,
    )
    return augmentations
