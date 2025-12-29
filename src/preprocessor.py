"""
Image preprocessing pipeline for preparing frames for model inference.
"""

import cv2
import numpy as np
from PIL import Image
import base64
import io
import logging
from typing import Tuple, Optional


logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing for vision-language models."""

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
        cache_size: int = 10
    ):
        """
        Initialize image preprocessor.

        Args:
            target_size: Target size for resizing (width, height), None to keep original
            normalize: Whether to normalize pixel values
            cache_size: Number of preprocessed frames to cache
        """
        self.target_size = target_size
        self.normalize = normalize
        self.cache_size = cache_size
        self.cache = {}

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame.

        Args:
            frame: Input frame in BGR format (OpenCV default)

        Returns:
            Preprocessed frame in RGB format
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize if target size specified
            if self.target_size:
                rgb_frame = cv2.resize(
                    rgb_frame,
                    self.target_size,
                    interpolation=cv2.INTER_LINEAR
                )

            # Normalize if required
            if self.normalize:
                rgb_frame = rgb_frame.astype(np.float32) / 255.0

            return rgb_frame

        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return frame

    def bgr_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert BGR frame to RGB.

        Args:
            frame: BGR frame

        Returns:
            RGB frame
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def to_pil_image(self, frame: np.ndarray) -> Image.Image:
        """
        Convert numpy array frame to PIL Image.

        Args:
            frame: Frame as numpy array (BGR format)

        Returns:
            PIL Image in RGB format
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)

            # Resize if target size specified
            if self.target_size:
                pil_image = pil_image.resize(
                    self.target_size,
                    Image.Resampling.LANCZOS
                )

            return pil_image

        except Exception as e:
            logger.error(f"Error converting to PIL Image: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='black')

    def to_base64(self, frame: np.ndarray, format: str = 'JPEG') -> str:
        """
        Convert frame to base64 encoded string.

        Args:
            frame: Frame as numpy array
            format: Image format ('JPEG', 'PNG')

        Returns:
            Base64 encoded string
        """
        try:
            # Convert to PIL Image
            pil_image = self.to_pil_image(frame)

            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format)
            img_bytes = buffer.getvalue()
            base64_str = base64.b64encode(img_bytes).decode('utf-8')

            return base64_str

        except Exception as e:
            logger.error(f"Error converting to base64: {e}")
            return ""

    def resize_frame(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        keep_aspect_ratio: bool = True
    ) -> np.ndarray:
        """
        Resize frame to specified dimensions.

        Args:
            frame: Input frame
            width: Target width
            height: Target height
            keep_aspect_ratio: Whether to maintain aspect ratio

        Returns:
            Resized frame
        """
        try:
            if keep_aspect_ratio:
                # Calculate aspect ratio
                h, w = frame.shape[:2]
                aspect = w / h

                if width / height > aspect:
                    # Height is the limiting factor
                    new_height = height
                    new_width = int(height * aspect)
                else:
                    # Width is the limiting factor
                    new_width = width
                    new_height = int(width / aspect)

                resized = cv2.resize(
                    frame,
                    (new_width, new_height),
                    interpolation=cv2.INTER_LINEAR
                )

                # Create canvas and center the image
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                y_offset = (height - new_height) // 2
                x_offset = (width - new_width) // 2
                canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

                return canvas
            else:
                return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        except Exception as e:
            logger.error(f"Error resizing frame: {e}")
            return frame

    def apply_clahe(self, frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast.

        Args:
            frame: Input frame
            clip_limit: Threshold for contrast limiting

        Returns:
            Enhanced frame
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)

            # Merge channels
            enhanced_lab = cv2.merge([l_enhanced, a, b])

            # Convert back to BGR
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            return enhanced_bgr

        except Exception as e:
            logger.error(f"Error applying CLAHE: {e}")
            return frame

    def denoise(self, frame: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Apply denoising filter to frame.

        Args:
            frame: Input frame
            strength: Denoising strength (higher = more denoising)

        Returns:
            Denoised frame
        """
        try:
            denoised = cv2.fastNlMeansDenoisingColored(
                frame,
                None,
                strength,
                strength,
                7,
                21
            )
            return denoised
        except Exception as e:
            logger.error(f"Error denoising frame: {e}")
            return frame

    def get_optimal_model_size(self, model_name: str) -> Tuple[int, int]:
        """
        Get optimal input size for specific model.

        Args:
            model_name: Name of the vision model

        Returns:
            (width, height) tuple
        """
        # Common model input sizes
        size_map = {
            'llava': (336, 336),
            'qwen': (448, 448),
            'cogvlm': (224, 224),
            'clip': (224, 224),
            'blip': (384, 384)
        }

        # Check model name for known models
        model_lower = model_name.lower()
        for key, size in size_map.items():
            if key in model_lower:
                return size

        # Default size
        return (224, 224)

    def batch_preprocess(self, frames: list) -> list:
        """
        Preprocess multiple frames at once.

        Args:
            frames: List of frames

        Returns:
            List of preprocessed frames
        """
        return [self.preprocess(frame) for frame in frames]

    def clear_cache(self):
        """Clear the preprocessing cache."""
        self.cache.clear()
        logger.debug("Preprocessing cache cleared")
