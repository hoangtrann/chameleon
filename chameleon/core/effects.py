"""Effect filters for selfie and background processing."""

import os

import cv2
import numpy as np
from cmapy import cmap


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift image by dx, dy pixels.

    Args:
        img: Input image
        dx: Horizontal shift (positive = right)
        dy: Vertical shift (positive = down)

    Returns:
        Shifted image with zeros in empty areas
    """
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy > 0:
        img[:dy, :] = 0
    elif dy < 0:
        img[dy:, :] = 0
    if dx > 0:
        img[:, :dx] = 0
    elif dx < 0:
        img[:, dx:] = 0
    return img


def hologram_effect(img: np.ndarray) -> np.ndarray:
    """Apply hologram effect to image.

    Args:
        img: Input image

    Returns:
        Image with hologram effect applied
    """
    # Add a blue tint
    holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)

    # Add a halftone effect
    band_length, band_gap = 3, 4
    for y in range(holo.shape[0]):
        if y % (band_length + band_gap) < band_length:
            # Use deterministic value if TEST_DETERMINISTIC env var is set
            if os.environ.get("TEST_DETERMINISTIC"):
                holo[y, :, :] = holo[y, :, :] * 0.2  # Fixed value instead of random
            else:
                holo[y, :, :] = holo[y, :, :] * np.random.uniform(0.1, 0.3)

    # Add some ghosting
    holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
    holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0)

    # Combine with the original color, oversaturated
    out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
    return out


def blur_effect(frame: np.ndarray, value: int = 90, fast_mode: bool = True) -> np.ndarray:
    """Apply blur effect to image with optional downscale optimization.

    Args:
        frame: Input frame
        value: Blur intensity (0-100)
        fast_mode: Use downscale-blur-upscale for 3-4x speedup (default: True)

    Returns:
        Blurred frame
    """
    value = min(100, max(0, int(value)))
    if value == 0:
        return frame

    # Fast mode: downscale -> blur -> upscale (3-4x faster, imperceptible quality difference)
    if fast_mode and frame.shape[0] > 360:  # Only worth it for larger images
        h, w = frame.shape[:2]

        # Downscale to 50%
        small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)

        # Blur at lower resolution (adjust kernel for smaller size)
        kernel_size = int((value / 100) * 49) + 1  # Half the kernel size
        kernel_size = _get_next_odd_number(kernel_size)
        blurred_small = cv2.GaussianBlur(small, (kernel_size, kernel_size), 0)

        # Upscale back
        return cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        # Original method: full resolution blur
        kernel_size = int((value / 100) * 99) + 1
        kernel_size = _get_next_odd_number(kernel_size)
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def solid_effect(frame: np.ndarray, color: tuple) -> np.ndarray:
    """Fill frame with solid color.

    Args:
        frame: Input frame (used for shape only)
        color: BGR color tuple

    Returns:
        Frame filled with solid color
    """
    frame[:] = color
    return frame


def cmap_effect(frame: np.ndarray, map_name: str) -> np.ndarray:
    """Apply color map to image.

    Args:
        frame: Input frame
        map_name: Color map name from cmapy

    Returns:
        Frame with color map applied
    """
    cv2.applyColorMap(frame, cmap(map_name), dst=frame)
    return frame


def brightness_effect(frame: np.ndarray, value: int = 100) -> np.ndarray:
    """Apply brightness adjustment to image.

    Args:
        frame: Input frame
        value: Brightness percentage (0-200, 100=normal)

    Returns:
        Frame with adjusted brightness
    """
    brightness = min(200, max(0, int(value))) / 100.0  # Allow up to 200% brightness
    return np.clip(frame * brightness, 0, 255).astype(np.uint8)


def opacity_effect(frame: np.ndarray, value: int = 100) -> np.ndarray:
    """Apply opacity/transparency to image.

    Args:
        frame: Input frame
        value: Opacity percentage (0-100)

    Returns:
        Frame with adjusted opacity
    """
    opacity = min(100, max(0, int(value))) / 100.0
    # Create a transparent version by blending with black
    black = np.zeros_like(frame)
    return cv2.addWeighted(frame, opacity, black, 1 - opacity, 0)


def tile_effect(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Tile image to fill target dimensions.

    Args:
        img: Input image
        target_width: Target width
        target_height: Target height

    Returns:
        Tiled image
    """
    h, w = img.shape[:2]

    # Calculate how many tiles we need
    tiles_x = (target_width + w - 1) // w
    tiles_y = (target_height + h - 1) // h

    # Tile the image
    tiled = np.tile(img, (tiles_y, tiles_x, 1))

    # Crop to exact target size
    return tiled[:target_height, :target_width]


def crop_effect(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Crop image to maintain aspect ratio.

    Args:
        img: Input image
        target_width: Target width
        target_height: Target height

    Returns:
        Cropped and resized image
    """
    h, w = img.shape[:2]
    target_aspect = target_width / target_height
    img_aspect = w / h

    if img_aspect > target_aspect:
        # Image is wider than target, crop width
        new_width = int(h * target_aspect)
        x_offset = (w - new_width) // 2
        cropped = img[:, x_offset : x_offset + new_width]
    else:
        # Image is taller than target, crop height
        new_height = int(w / target_aspect)
        y_offset = (h - new_height) // 2
        cropped = img[y_offset : y_offset + new_height, :]

    # Resize to target dimensions
    return cv2.resize(cropped, (target_width, target_height))


def _get_next_odd_number(number: int) -> int:
    """Get next odd number (required for kernel sizes).

    Args:
        number: Input number

    Returns:
        Next odd number
    """
    if number % 2 == 0:
        return number + 1
    return number


def apply_effects(frame: np.ndarray, config: dict) -> np.ndarray:
    """Apply multiple effects based on configuration.

    Args:
        frame: Input frame
        config: Effect configuration dictionary

    Returns:
        Frame with effects applied
    """
    # Apply hologram effect
    if config.get("hologram", False):
        frame = hologram_effect(frame)

    # Apply blur effect
    if config.get("blur") is not None:
        frame = blur_effect(frame, config["blur"])

    # Apply solid color
    if config.get("solid") is not None:
        frame = solid_effect(frame, config["solid"])

    # Apply color map
    if config.get("cmap") is not None:
        frame = cmap_effect(frame, config["cmap"])

    # Apply brightness
    if config.get("brightness") is not None:
        frame = brightness_effect(frame, config["brightness"])

    # Apply opacity
    if config.get("opacity") is not None:
        frame = opacity_effect(frame, config["opacity"])

    return frame
