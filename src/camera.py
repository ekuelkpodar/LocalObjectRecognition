"""
Camera capture module with threading support for real-time video processing.
"""

import cv2
import threading
import queue
import time
import logging
from typing import Optional, Tuple
import numpy as np


logger = logging.getLogger(__name__)


class CameraCapture:
    """Manages webcam capture with threading for non-blocking operation."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        max_queue_size: int = 5
    ):
        """
        Initialize camera capture.

        Args:
            device_index: Camera device index (0 for default)
            width: Capture frame width
            height: Capture frame height
            fps: Target frames per second
            max_queue_size: Maximum size of frame queue
        """
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.max_queue_size = max_queue_size

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None

        # Statistics
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

    def start(self) -> bool:
        """
        Start camera capture.

        Returns:
            True if started successfully, False otherwise
        """
        logger.info(f"Starting camera capture on device {self.device_index}")

        try:
            self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera device {self.device_index}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            logger.info(
                f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS"
            )

            # Start capture thread
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            logger.info("Camera capture thread started")
            return True

        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        logger.debug("Capture loop started")
        frame_time = 1.0 / self.fps if self.fps > 0 else 0.033

        while self.running:
            try:
                start_time = time.time()

                ret, frame = self.cap.read()

                if not ret:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

                self.frames_captured += 1

                # Try to put frame in queue
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # Queue is full, drop the frame
                    self.frames_dropped += 1
                    if self.frames_dropped % 10 == 0:
                        logger.debug(f"Frame queue full, dropped {self.frames_dropped} frames")

                # Update FPS calculation
                elapsed = time.time() - self.last_fps_time
                if elapsed >= 1.0:
                    self.current_fps = self.frames_captured / elapsed
                    self.frames_captured = 0
                    self.last_fps_time = time.time()

                # Maintain frame rate
                processing_time = time.time() - start_time
                sleep_time = max(0, frame_time - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)

        logger.debug("Capture loop ended")

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the next available frame.

        Args:
            timeout: Maximum time to wait for a frame

        Returns:
            Frame as numpy array or None if timeout
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame, discarding older frames.

        Returns:
            Latest frame or None if no frames available
        """
        frame = None
        try:
            # Empty the queue and get the latest frame
            while True:
                try:
                    frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Error getting latest frame: {e}")

        return frame

    def stop(self):
        """Stop camera capture and cleanup resources."""
        logger.info("Stopping camera capture")
        self.running = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()
            logger.info("Camera released")

        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def is_running(self) -> bool:
        """Check if camera is currently running."""
        return self.running

    def get_fps(self) -> float:
        """Get current capture FPS."""
        return self.current_fps

    def get_stats(self) -> dict:
        """
        Get capture statistics.

        Returns:
            Dictionary with capture statistics
        """
        return {
            'fps': self.current_fps,
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'queue_size': self.frame_queue.qsize(),
            'max_queue_size': self.max_queue_size
        }

    def clear_queue(self):
        """Clear all frames from the queue."""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        logger.debug("Frame queue cleared")

    def __del__(self):
        """Cleanup on destruction."""
        self.stop()
