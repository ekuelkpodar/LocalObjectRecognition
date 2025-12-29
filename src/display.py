"""
Display and UI module for visualizing real-time object recognition results.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, List, Tuple
from collections import deque


logger = logging.getLogger(__name__)


class DisplayManager:
    """Manages the display window and visualization of detection results."""

    def __init__(
        self,
        window_name: str = "Object Recognition - vLLM",
        width: int = 1280,
        height: int = 720,
        show_fps: bool = True,
        max_text_lines: int = 10
    ):
        """
        Initialize display manager.

        Args:
            window_name: Name of the display window
            width: Window width
            height: Window height
            show_fps: Whether to show FPS counter
            max_text_lines: Maximum number of text lines to display
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        self.show_fps = show_fps
        self.max_text_lines = max_text_lines

        self.window_created = False
        self.paused = False

        # Text display
        self.current_detection = ""
        self.detection_history = deque(maxlen=max_text_lines)
        self.status_message = "Initializing..."

        # FPS tracking
        self.capture_fps = 0.0
        self.inference_fps = 0.0
        self.display_fps = 0.0
        self.fps_update_time = time.time()
        self.frame_count = 0

        # Colors (BGR format)
        self.colors = {
            'background': (30, 30, 30),
            'text': (255, 255, 255),
            'highlight': (0, 255, 0),
            'status': (100, 200, 255),
            'error': (0, 0, 255),
            'info': (255, 200, 100)
        }

    def create_window(self):
        """Create the display window."""
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)
            self.window_created = True
            logger.info(f"Display window created: {self.width}x{self.height}")

    def update(
        self,
        frame: np.ndarray,
        detection_text: Optional[str] = None,
        capture_fps: float = 0.0,
        inference_fps: float = 0.0,
        queue_size: int = 0,
        processing: bool = False
    ) -> bool:
        """
        Update the display with new frame and information.

        Args:
            frame: Current video frame
            detection_text: Latest detection text
            capture_fps: Capture frames per second
            inference_fps: Inference frames per second
            queue_size: Current processing queue size
            processing: Whether inference is currently processing

        Returns:
            True if display updated successfully, False to quit
        """
        if not self.window_created:
            self.create_window()

        try:
            # Update FPS
            self.capture_fps = capture_fps
            self.inference_fps = inference_fps
            self._update_display_fps()

            # Update detection text
            if detection_text:
                self.current_detection = detection_text
                self.detection_history.append(detection_text)

            # Create display frame
            display_frame = self._create_display_frame(
                frame,
                queue_size,
                processing
            )

            # Show frame
            cv2.imshow(self.window_name, display_frame)

            return True

        except Exception as e:
            logger.error(f"Error updating display: {e}")
            return False

    def _create_display_frame(
        self,
        frame: np.ndarray,
        queue_size: int,
        processing: bool
    ) -> np.ndarray:
        """
        Create the complete display frame with overlays.

        Args:
            frame: Base video frame
            queue_size: Processing queue size
            processing: Whether processing is active

        Returns:
            Display frame with overlays
        """
        # Resize frame to display size
        display_frame = cv2.resize(frame, (self.width, self.height))

        # Calculate panel dimensions
        panel_width = 400
        panel_height = self.height

        # Create side panel
        panel = self._create_info_panel(
            panel_width,
            panel_height,
            queue_size,
            processing
        )

        # Combine frame and panel
        combined = np.hstack([display_frame, panel])

        # Add top status bar
        combined = self._add_status_bar(combined)

        # Add keyboard shortcuts help
        combined = self._add_help_overlay(combined)

        return combined

    def _create_info_panel(
        self,
        width: int,
        height: int,
        queue_size: int,
        processing: bool
    ) -> np.ndarray:
        """
        Create information panel with detection results and stats.

        Args:
            width: Panel width
            height: Panel height
            queue_size: Processing queue size
            processing: Processing status

        Returns:
            Panel image
        """
        panel = np.ones((height, width, 3), dtype=np.uint8) * np.array(self.colors['background'], dtype=np.uint8)

        y_offset = 30
        line_height = 25

        # Title
        self._put_text(
            panel,
            "OBJECT DETECTION",
            (10, y_offset),
            font_scale=0.7,
            color=self.colors['highlight'],
            thickness=2
        )
        y_offset += line_height * 2

        # FPS Information
        self._put_text(panel, "Performance:", (10, y_offset), color=self.colors['info'])
        y_offset += line_height

        self._put_text(
            panel,
            f"  Capture: {self.capture_fps:.1f} FPS",
            (10, y_offset),
            font_scale=0.5
        )
        y_offset += line_height

        self._put_text(
            panel,
            f"  Inference: {self.inference_fps:.2f} FPS",
            (10, y_offset),
            font_scale=0.5
        )
        y_offset += line_height

        self._put_text(
            panel,
            f"  Display: {self.display_fps:.1f} FPS",
            (10, y_offset),
            font_scale=0.5
        )
        y_offset += line_height * 2

        # Queue status
        queue_color = self.colors['error'] if queue_size > 3 else self.colors['text']
        self._put_text(
            panel,
            f"Queue: {queue_size}",
            (10, y_offset),
            color=queue_color
        )
        y_offset += line_height

        # Processing status
        status_text = "Processing..." if processing else "Idle"
        status_color = self.colors['highlight'] if processing else self.colors['text']
        self._put_text(
            panel,
            f"Status: {status_text}",
            (10, y_offset),
            color=status_color
        )
        y_offset += line_height * 2

        # Current detection
        self._put_text(panel, "Latest Detection:", (10, y_offset), color=self.colors['info'])
        y_offset += line_height

        # Display detection text (word wrapped)
        if self.current_detection:
            wrapped_lines = self._wrap_text(self.current_detection, width - 20, 0.5)
            for line in wrapped_lines[:8]:  # Limit to 8 lines
                self._put_text(
                    panel,
                    line,
                    (10, y_offset),
                    font_scale=0.5,
                    color=self.colors['text']
                )
                y_offset += line_height

        # Paused indicator
        if self.paused:
            pause_y = height - 100
            self._put_text(
                panel,
                "PAUSED",
                (width // 2 - 50, pause_y),
                font_scale=1.0,
                color=self.colors['error'],
                thickness=2
            )

        return panel

    def _add_status_bar(self, frame: np.ndarray) -> np.ndarray:
        """
        Add status bar at the top of the frame.

        Args:
            frame: Input frame

        Returns:
            Frame with status bar
        """
        bar_height = 40
        bar = np.ones((bar_height, frame.shape[1], 3), dtype=np.uint8) * np.array(self.colors['background'], dtype=np.uint8)

        self._put_text(
            bar,
            self.status_message,
            (10, 28),
            font_scale=0.6,
            color=self.colors['status']
        )

        return np.vstack([bar, frame])

    def _add_help_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Add keyboard shortcuts help overlay.

        Args:
            frame: Input frame

        Returns:
            Frame with help overlay
        """
        help_text = [
            "Q/ESC: Quit",
            "P: Pause",
            "S: Save",
            "C: Clear",
            "1-5: Prompts"
        ]

        x_offset = frame.shape[1] - 150
        y_offset = frame.shape[0] - 150

        for text in help_text:
            self._put_text(
                frame,
                text,
                (x_offset, y_offset),
                font_scale=0.4,
                color=self.colors['info']
            )
            y_offset += 20

        return frame

    def _put_text(
        self,
        img: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 0.6,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 1
    ):
        """
        Put text on image with shadow for better visibility.

        Args:
            img: Image to draw on
            text: Text to display
            position: (x, y) position
            font_scale: Font scale
            color: Text color (BGR)
            thickness: Text thickness
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw shadow
        cv2.putText(
            img,
            text,
            (position[0] + 1, position[1] + 1),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA
        )

        # Draw text
        cv2.putText(
            img,
            text,
            position,
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    def _wrap_text(self, text: str, max_width: int, font_scale: float) -> List[str]:
        """
        Wrap text to fit within specified width.

        Args:
            text: Text to wrap
            max_width: Maximum width in pixels
            font_scale: Font scale

        Returns:
            List of wrapped text lines
        """
        words = text.split()
        lines = []
        current_line = []

        font = cv2.FONT_HERSHEY_SIMPLEX

        for word in words:
            test_line = ' '.join(current_line + [word])
            (width, _), _ = cv2.getTextSize(test_line, font, font_scale, 1)

            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def _update_display_fps(self):
        """Update display FPS calculation."""
        self.frame_count += 1
        elapsed = time.time() - self.fps_update_time

        if elapsed >= 1.0:
            self.display_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_update_time = time.time()

    def set_status(self, message: str):
        """
        Set status message.

        Args:
            message: Status message to display
        """
        self.status_message = message
        logger.debug(f"Status: {message}")

    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        status = "paused" if self.paused else "resumed"
        self.set_status(f"Playback {status}")
        logger.info(f"Display {status}")

    def is_paused(self) -> bool:
        """Check if display is paused."""
        return self.paused

    def clear_detections(self):
        """Clear detection history."""
        self.current_detection = ""
        self.detection_history.clear()
        self.set_status("Detection history cleared")
        logger.info("Detection history cleared")

    def save_frame(self, frame: np.ndarray, filepath: str):
        """
        Save current frame to file.

        Args:
            frame: Frame to save
            filepath: Output file path
        """
        try:
            cv2.imwrite(filepath, frame)
            self.set_status(f"Frame saved: {filepath}")
            logger.info(f"Frame saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            self.set_status(f"Error saving frame")

    def destroy(self):
        """Destroy the display window."""
        if self.window_created:
            cv2.destroyAllWindows()
            self.window_created = False
            logger.info("Display window destroyed")

    def __del__(self):
        """Cleanup on destruction."""
        self.destroy()
