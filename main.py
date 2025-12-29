#!/usr/bin/env python3
"""
Real-Time Object Recognition System using vLLM and Webcam

Main application that orchestrates camera capture, model inference,
and real-time display with multi-threading support.
"""

import os
import sys
import time
import threading
import queue
import logging
import argparse
import signal
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config, PromptTemplateManager
from src.camera import CameraCapture
from src.model import VLLMModel
from src.preprocessor import ImagePreprocessor
from src.display import DisplayManager
from src.utils import (
    setup_logging,
    print_system_info,
    get_gpu_info,
    generate_filename,
    ensure_dir,
    save_detection_log,
    Timer,
    PerformanceMonitor,
    check_dependencies
)


logger = logging.getLogger(__name__)


class ObjectRecognitionSystem:
    """Main application class for real-time object recognition."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the object recognition system.

        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.running = False
        self.processing = False

        # Components
        self.camera = None
        self.model = None
        self.preprocessor = None
        self.display = None
        self.prompt_manager = None

        # Threading
        self.inference_thread = None
        self.inference_queue = queue.Queue(
            maxsize=self.config.processing['max_queue_size']
        )
        self.result_queue = queue.Queue()

        # State
        self.frame_counter = 0
        self.current_prompt = None
        self.last_detection = ""

        # Performance monitoring
        self.perf_monitor = PerformanceMonitor(window_size=100)

        # Shutdown event
        self.shutdown_event = threading.Event()

    def initialize(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if initialization successful, False otherwise
        """
        logger.info("Initializing Object Recognition System")

        try:
            # Print system information
            print_system_info()

            # Check GPU
            gpu_info = get_gpu_info()
            if not gpu_info['available']:
                logger.error("No GPU available. vLLM requires CUDA-capable GPU.")
                return False

            # Initialize prompt manager
            logger.info("Loading prompt templates")
            self.prompt_manager = PromptTemplateManager()
            self.current_prompt = self.prompt_manager.get_default_prompt()
            logger.info(f"Default prompt: {self.current_prompt[:50]}...")

            # Initialize preprocessor
            logger.info("Initializing image preprocessor")
            self.preprocessor = ImagePreprocessor(
                target_size=None,  # Will use model's optimal size
                normalize=False
            )

            # Initialize camera
            logger.info("Initializing camera")
            self.camera = CameraCapture(
                device_index=self.config.camera['device_index'],
                width=self.config.camera['capture_width'],
                height=self.config.camera['capture_height'],
                fps=self.config.camera['fps'],
                max_queue_size=self.config.processing['max_queue_size']
            )

            if not self.camera.start():
                logger.error("Failed to start camera")
                return False

            # Initialize display
            logger.info("Initializing display")
            self.display = DisplayManager(
                width=self.config.processing['display_width'],
                height=self.config.processing['display_height'],
                show_fps=True
            )
            self.display.create_window()
            self.display.set_status("Loading model...")

            # Initialize model (this may take a while)
            logger.info("Initializing vLLM model (this may take a while...)")
            self.model = VLLMModel(
                model_name=self.config.model['name'],
                max_tokens=self.config.model['max_tokens'],
                temperature=self.config.model['temperature'],
                top_p=self.config.model['top_p'],
                gpu_memory_util=self.config.model['gpu_memory_util'],
                max_model_len=self.config.model['max_model_len'],
                dtype=self.config.model['dtype']
            )

            if not self.model.initialize():
                logger.error("Failed to initialize model")
                return False

            self.display.set_status("Model loaded - Ready")

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            return False

    def start(self):
        """Start the main processing loop."""
        logger.info("Starting object recognition system")
        self.running = True

        # Start inference thread
        self.inference_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True
        )
        self.inference_thread.start()

        # Main loop
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.stop()

    def _main_loop(self):
        """Main processing loop."""
        process_every_n = self.config.processing['process_every_n_frames']

        logger.info("Entering main loop")
        self.display.set_status("Running")

        while self.running:
            try:
                # Get frame from camera
                frame = self.camera.get_latest_frame()

                if frame is None:
                    time.sleep(0.01)
                    continue

                self.frame_counter += 1

                # Process frame for inference every N frames
                if self.frame_counter % process_every_n == 0:
                    self._queue_frame_for_inference(frame.copy())

                # Check for results
                self._check_results()

                # Update display
                self._update_display(frame)

                # Handle keyboard input
                if not self._handle_keyboard():
                    break

            except Exception as e:
                logger.error(f"Error in main loop iteration: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Main loop ended")

    def _queue_frame_for_inference(self, frame: np.ndarray):
        """
        Queue a frame for inference.

        Args:
            frame: Frame to process
        """
        try:
            self.inference_queue.put_nowait((self.frame_counter, frame))
            logger.debug(f"Frame {self.frame_counter} queued for inference")
        except queue.Full:
            logger.debug("Inference queue full, skipping frame")

    def _inference_worker(self):
        """Worker thread for model inference."""
        logger.info("Inference worker started")

        while self.running and not self.shutdown_event.is_set():
            try:
                # Get frame from queue
                try:
                    frame_num, frame = self.inference_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                self.processing = True
                logger.debug(f"Processing frame {frame_num}")

                # Perform inference
                timer = Timer()
                timer.start()

                detection_text = self.model.generate(
                    frame,
                    self.current_prompt,
                    timeout=self.config.processing.get('inference_timeout', 10.0)
                )

                inference_time = timer.stop()

                # Track performance
                self.perf_monitor.add_metric('inference_time', inference_time)

                # Put result in queue
                if detection_text:
                    self.result_queue.put((frame_num, detection_text, inference_time))
                    logger.debug(
                        f"Frame {frame_num} processed in {inference_time:.2f}s"
                    )

                self.processing = False

            except Exception as e:
                logger.error(f"Error in inference worker: {e}", exc_info=True)
                self.processing = False
                time.sleep(0.1)

        logger.info("Inference worker stopped")

    def _check_results(self):
        """Check for and process inference results."""
        try:
            while not self.result_queue.empty():
                frame_num, detection_text, inference_time = self.result_queue.get_nowait()

                self.last_detection = detection_text
                logger.info(f"Detection result: {detection_text[:100]}...")

                # Save to log if configured
                if self.config.logging.get('log_detections', False):
                    log_file = "logs/detections.txt"
                    save_detection_log(
                        log_file,
                        frame_num,
                        time.time(),
                        detection_text,
                        inference_time
                    )

        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error checking results: {e}")

    def _update_display(self, frame: np.ndarray):
        """
        Update the display with current frame and information.

        Args:
            frame: Current frame to display
        """
        try:
            # Get statistics
            camera_stats = self.camera.get_stats()
            model_stats = self.model.get_stats()

            # Update display
            self.display.update(
                frame=frame,
                detection_text=self.last_detection,
                capture_fps=camera_stats['fps'],
                inference_fps=model_stats['fps'],
                queue_size=self.inference_queue.qsize(),
                processing=self.processing
            )

        except Exception as e:
            logger.error(f"Error updating display: {e}")

    def _handle_keyboard(self) -> bool:
        """
        Handle keyboard input.

        Returns:
            True to continue, False to quit
        """
        key = cv2.waitKey(1) & 0xFF

        if key == 255:  # No key pressed
            return True

        # Q or ESC - Quit
        if key in [ord('q'), ord('Q'), 27]:
            logger.info("Quit requested")
            return False

        # P - Pause/Resume
        elif key in [ord('p'), ord('P')]:
            self.display.toggle_pause()

        # S - Save frame
        elif key in [ord('s'), ord('S')]:
            self._save_current_frame()

        # C - Clear detections
        elif key in [ord('c'), ord('C')]:
            self.display.clear_detections()
            self.last_detection = ""

        # 1-5 - Switch prompts
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            self._switch_prompt(chr(key))

        return True

    def _switch_prompt(self, number: str):
        """
        Switch to a different prompt template.

        Args:
            number: Prompt number (1-5)
        """
        new_prompt = self.prompt_manager.get_template_by_number(int(number))
        if new_prompt:
            self.current_prompt = new_prompt
            logger.info(f"Switched to prompt {number}: {new_prompt[:50]}...")
            self.display.set_status(f"Using prompt template {number}")

    def _save_current_frame(self):
        """Save the current frame and detection results."""
        try:
            # Create output directory
            output_dir = "output"
            ensure_dir(output_dir)

            # Generate filename
            filename = generate_filename("detection", "jpg")
            filepath = os.path.join(output_dir, filename)

            # Get latest frame
            frame = self.camera.get_latest_frame()
            if frame is not None:
                self.display.save_frame(frame, filepath)

                # Save detection text
                text_file = filepath.replace('.jpg', '.txt')
                with open(text_file, 'w') as f:
                    f.write(f"Frame: {self.frame_counter}\n")
                    f.write(f"Timestamp: {time.time()}\n")
                    f.write(f"Prompt: {self.current_prompt}\n")
                    f.write(f"\nDetection:\n{self.last_detection}\n")

                logger.info(f"Saved frame and detection to {filepath}")

        except Exception as e:
            logger.error(f"Error saving frame: {e}")

    def stop(self):
        """Stop the system and cleanup resources."""
        logger.info("Stopping object recognition system")
        self.running = False
        self.shutdown_event.set()

        # Stop components
        if self.camera:
            self.camera.stop()

        if self.display:
            self.display.destroy()

        if self.model:
            self.model.cleanup()

        # Wait for inference thread
        if self.inference_thread and self.inference_thread.is_alive():
            logger.info("Waiting for inference thread to stop")
            self.inference_thread.join(timeout=5.0)

        logger.info("System stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time Object Recognition using vLLM and Webcam"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        print("ERROR: Missing required dependencies. Please install them first.")
        return 1

    # Load config for logging setup
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        return 1

    # Setup logging
    log_level = "DEBUG" if args.debug else config.logging['level']
    setup_logging(
        level=log_level,
        log_to_file=config.logging['log_to_file'],
        log_file=config.logging['log_file'],
        log_format=config.logging['format']
    )

    logger.info("=" * 60)
    logger.info("Real-Time Object Recognition System")
    logger.info("=" * 60)

    # Create and run system
    system = ObjectRecognitionSystem(args.config)

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Signal received, shutting down...")
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize
    if not system.initialize():
        logger.error("Failed to initialize system")
        return 1

    # Start
    system.start()

    return 0


if __name__ == "__main__":
    sys.exit(main())
