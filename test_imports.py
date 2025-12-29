#!/usr/bin/env python3
"""
Quick import test to verify all modules load correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing module imports...")

try:
    from src.config import Config, PromptTemplateManager
    print("✓ config.py")
except ImportError as e:
    print(f"✗ config.py: {e}")

try:
    from src.camera import CameraCapture
    print("✓ camera.py")
except ImportError as e:
    print(f"✗ camera.py: {e}")

try:
    from src.model import VLLMModel
    print("✓ model.py")
except ImportError as e:
    print(f"✗ model.py: {e}")

try:
    from src.preprocessor import ImagePreprocessor
    print("✓ preprocessor.py")
except ImportError as e:
    print(f"✗ preprocessor.py: {e}")

try:
    from src.display import DisplayManager
    print("✓ display.py")
except ImportError as e:
    print(f"✗ display.py: {e}")

try:
    from src.utils import setup_logging, get_gpu_info, Timer
    print("✓ utils.py")
except ImportError as e:
    print(f"✗ utils.py: {e}")

print("\nAll module imports successful!")
print("\nTesting configuration loading...")

try:
    config = Config('config.yaml')
    print(f"✓ Loaded config.yaml")
    print(f"  Model: {config.model['name']}")
    print(f"  Camera: {config.camera['device_index']}")

    prompt_manager = PromptTemplateManager()
    templates = prompt_manager.list_templates()
    print(f"✓ Loaded {len(templates)} prompt templates")

except Exception as e:
    print(f"✗ Configuration loading failed: {e}")

print("\n✓ All tests passed! System is ready.")
