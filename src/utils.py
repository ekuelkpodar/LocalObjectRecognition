"""
Utility functions and logging setup for the object recognition system.
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import psutil
import torch


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_file: Path to log file
        log_format: Log message format

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_to_file and log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[]
    )

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_to_file and log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Logging to file: {log_file}")

    return root_logger


def get_gpu_info() -> dict:
    """
    Get GPU information and memory usage.

    Returns:
        Dictionary with GPU information
    """
    info = {
        'available': False,
        'count': 0,
        'devices': []
    }

    try:
        if torch.cuda.is_available():
            info['available'] = True
            info['count'] = torch.cuda.device_count()

            for i in range(info['count']):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / 1e9,
                    'allocated_memory_gb': torch.cuda.memory_allocated(i) / 1e9,
                    'reserved_memory_gb': torch.cuda.memory_reserved(i) / 1e9,
                    'free_memory_gb': (torch.cuda.get_device_properties(i).total_memory -
                                      torch.cuda.memory_allocated(i)) / 1e9
                }
                info['devices'].append(device_info)

    except Exception as e:
        logging.error(f"Error getting GPU info: {e}")

    return info


def get_system_info() -> dict:
    """
    Get system information including CPU and memory.

    Returns:
        Dictionary with system information
    """
    info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_total_gb': psutil.virtual_memory().total / 1e9,
        'memory_available_gb': psutil.virtual_memory().available / 1e9,
        'memory_percent': psutil.virtual_memory().percent
    }

    return info


def print_system_info():
    """Print comprehensive system information."""
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    # System info
    sys_info = get_system_info()
    print(f"\nCPU:")
    print(f"  Cores: {sys_info['cpu_count']}")
    print(f"  Usage: {sys_info['cpu_percent']}%")

    print(f"\nMemory:")
    print(f"  Total: {sys_info['memory_total_gb']:.2f} GB")
    print(f"  Available: {sys_info['memory_available_gb']:.2f} GB")
    print(f"  Usage: {sys_info['memory_percent']}%")

    # GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU:")
    if gpu_info['available']:
        print(f"  Count: {gpu_info['count']}")
        for device in gpu_info['devices']:
            print(f"\n  Device {device['id']}: {device['name']}")
            print(f"    Total Memory: {device['total_memory_gb']:.2f} GB")
            print(f"    Allocated: {device['allocated_memory_gb']:.2f} GB")
            print(f"    Free: {device['free_memory_gb']:.2f} GB")
    else:
        print("  No CUDA-capable GPU detected")

    print("\n" + "=" * 60 + "\n")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def generate_filename(prefix: str = "capture", extension: str = "jpg") -> str:
    """
    Generate a unique filename with timestamp.

    Args:
        prefix: Filename prefix
        extension: File extension

    Returns:
        Unique filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def ensure_dir(directory: str):
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_detection_log(
    filepath: str,
    frame_number: int,
    timestamp: float,
    detection_text: str,
    inference_time: float
):
    """
    Save detection results to log file.

    Args:
        filepath: Path to log file
        frame_number: Frame number
        timestamp: Timestamp
        detection_text: Detection result text
        inference_time: Inference time in seconds
    """
    try:
        ensure_dir(os.path.dirname(filepath))

        with open(filepath, 'a') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Frame: {frame_number}\n")
            f.write(f"Timestamp: {datetime.fromtimestamp(timestamp)}\n")
            f.write(f"Inference Time: {inference_time:.3f}s\n")
            f.write(f"Detection:\n{detection_text}\n")

        logging.debug(f"Detection logged to {filepath}")

    except Exception as e:
        logging.error(f"Error saving detection log: {e}")


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        """Initialize timer."""
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """
        Stop the timer.

        Returns:
            Elapsed time in seconds
        """
        self.end_time = time.time()
        return self.elapsed()

    def elapsed(self) -> float:
        """
        Get elapsed time.

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Context manager enter."""
        self.start()
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of samples to keep for averaging
        """
        self.window_size = window_size
        self.metrics = {}

    def add_metric(self, name: str, value: float):
        """
        Add a metric value.

        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(value)

        # Keep only last window_size values
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name].pop(0)

    def get_average(self, name: str) -> float:
        """
        Get average value for a metric.

        Args:
            name: Metric name

        Returns:
            Average value
        """
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])

    def get_latest(self, name: str) -> float:
        """
        Get latest value for a metric.

        Args:
            name: Metric name

        Returns:
            Latest value
        """
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return self.metrics[name][-1]

    def get_stats(self, name: str) -> dict:
        """
        Get statistics for a metric.

        Args:
            name: Metric name

        Returns:
            Dictionary with min, max, avg
        """
        if name not in self.metrics or not self.metrics[name]:
            return {'min': 0.0, 'max': 0.0, 'avg': 0.0}

        values = self.metrics[name]
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values)
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}


def check_dependencies() -> bool:
    """
    Check if all required dependencies are installed.

    Returns:
        True if all dependencies available, False otherwise
    """
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'pillow',
        'yaml': 'pyyaml',
        'torch': 'torch'
    }

    missing = []

    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        logging.error(f"Missing dependencies: {', '.join(missing)}")
        logging.error(f"Install with: pip install {' '.join(missing)}")
        return False

    return True
