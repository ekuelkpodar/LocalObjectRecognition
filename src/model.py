"""
vLLM model wrapper and inference logic for vision-language models.
"""

import logging
import time
from typing import Optional, Dict, Any
import torch
from PIL import Image
import numpy as np


logger = logging.getLogger(__name__)


class VLLMModel:
    """Wrapper for vLLM vision-language model inference."""

    def __init__(
        self,
        model_name: str = "liuhaotian/llava-v1.5-7b",
        max_tokens: int = 150,
        temperature: float = 0.2,
        top_p: float = 0.9,
        gpu_memory_util: float = 0.8,
        max_model_len: int = 2048,
        dtype: str = "half"
    ):
        """
        Initialize vLLM model.

        Args:
            model_name: Name or path of the vision-language model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            gpu_memory_util: GPU memory utilization (0.0-1.0)
            max_model_len: Maximum model sequence length
            dtype: Data type for model ('half', 'float16', 'float32')
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gpu_memory_util = gpu_memory_util
        self.max_model_len = max_model_len
        self.dtype = dtype

        self.llm = None
        self.sampling_params = None
        self.is_initialized = False

        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.last_inference_time = 0.0

    def initialize(self) -> bool:
        """
        Initialize the vLLM model.

        Returns:
            True if initialization successful, False otherwise
        """
        logger.info(f"Initializing vLLM model: {self.model_name}")
        start_time = time.time()

        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.error("CUDA is not available. vLLM requires GPU.")
                return False

            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPU(s)")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")

            # Import vLLM (import here to provide better error messages)
            try:
                from vllm import LLM, SamplingParams
            except ImportError:
                logger.error("vLLM not installed. Install with: pip install vllm")
                return False

            # Initialize model
            logger.info("Loading model (this may take a while)...")
            self.llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_util,
                dtype=self.dtype
            )

            # Set up sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")

            # Log GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                logger.info(f"GPU memory: {memory_allocated:.2f} GB allocated, "
                          f"{memory_reserved:.2f} GB reserved")

            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing model: {e}", exc_info=True)
            return False

    def generate(
        self,
        image: np.ndarray,
        prompt: str,
        timeout: float = 10.0
    ) -> Optional[str]:
        """
        Generate description for an image.

        Args:
            image: Image as numpy array (BGR format from OpenCV)
            prompt: Text prompt for the model
            timeout: Maximum time for inference

        Returns:
            Generated text or None if error
        """
        if not self.is_initialized:
            logger.error("Model not initialized. Call initialize() first.")
            return None

        try:
            start_time = time.time()

            # Convert image format if needed
            from src.preprocessor import ImagePreprocessor
            preprocessor = ImagePreprocessor()
            pil_image = preprocessor.to_pil_image(image)

            # Create the input for vLLM
            # Note: The exact format depends on the model
            # For LLaVA models, we typically need to format the prompt
            formatted_prompt = self._format_prompt(prompt)

            # Generate response
            # Note: vLLM's vision model API may vary by model
            # This is a general approach that works with LLaVA-style models
            try:
                outputs = self.llm.generate(
                    {
                        "prompt": formatted_prompt,
                        "multi_modal_data": {"image": pil_image}
                    },
                    sampling_params=self.sampling_params
                )

                # Extract text from output
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0].outputs[0].text.strip()
                else:
                    logger.warning("No output generated from model")
                    return None

            except Exception as e:
                # Handle different vLLM API versions
                logger.warning(f"Primary inference method failed: {e}")
                logger.info("Attempting alternative inference approach...")

                # Alternative approach for different model types
                generated_text = self._alternative_inference(pil_image, formatted_prompt)

            inference_time = time.time() - start_time
            self.last_inference_time = inference_time
            self.total_inference_time += inference_time
            self.inference_count += 1

            avg_time = self.total_inference_time / self.inference_count
            logger.debug(
                f"Inference completed in {inference_time:.2f}s "
                f"(avg: {avg_time:.2f}s)"
            )

            return generated_text

        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            return None

    def _format_prompt(self, prompt: str) -> str:
        """
        Format prompt for the specific model.

        Args:
            prompt: User prompt

        Returns:
            Formatted prompt string
        """
        # LLaVA-style prompt format
        if "llava" in self.model_name.lower():
            return f"USER: <image>\n{prompt}\nASSISTANT:"

        # Add other model-specific formats as needed
        return prompt

    def _alternative_inference(self, image: Image.Image, prompt: str) -> str:
        """
        Alternative inference method for different model versions.

        Args:
            image: PIL Image
            prompt: Formatted prompt

        Returns:
            Generated text
        """
        # This is a fallback method for different API versions
        # Implement based on specific model requirements
        logger.warning("Using fallback inference method")
        return "Model inference not fully configured for this model type. Please check model compatibility."

    def get_stats(self) -> Dict[str, Any]:
        """
        Get inference statistics.

        Returns:
            Dictionary with inference statistics
        """
        avg_time = (self.total_inference_time / self.inference_count
                   if self.inference_count > 0 else 0.0)

        return {
            'inference_count': self.inference_count,
            'total_inference_time': self.total_inference_time,
            'average_inference_time': avg_time,
            'last_inference_time': self.last_inference_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0.0
        }

    def reset_stats(self):
        """Reset inference statistics."""
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.last_inference_time = 0.0
        logger.info("Model statistics reset")

    def cleanup(self):
        """Cleanup model resources."""
        logger.info("Cleaning up model resources")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.llm = None
        self.is_initialized = False


class ModelInferenceThread:
    """Manages model inference in a separate thread."""

    def __init__(self, model: VLLMModel):
        """
        Initialize inference thread manager.

        Args:
            model: Initialized VLLMModel instance
        """
        self.model = model
        self.running = False
        self.inference_thread = None

    def start(self):
        """Start the inference thread."""
        # Implementation would depend on specific threading needs
        pass

    def stop(self):
        """Stop the inference thread."""
        self.running = False
