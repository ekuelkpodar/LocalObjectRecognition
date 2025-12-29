#!/usr/bin/env python3
"""
Setup verification script for Object Recognition System.

Run this script to verify that all dependencies are installed
and the system is ready to run.
"""

import sys
import importlib


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ERROR: Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"  OK: Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_module(module_name, package_name=None):
    """Check if a module is installed."""
    if package_name is None:
        package_name = module_name

    try:
        importlib.import_module(module_name)
        print(f"  OK: {package_name}")
        return True
    except ImportError:
        print(f"  MISSING: {package_name}")
        return False


def check_dependencies():
    """Check all required dependencies."""
    print("\nChecking dependencies...")

    dependencies = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('PIL', 'pillow'),
        ('yaml', 'pyyaml'),
        ('torch', 'torch'),
        ('psutil', 'psutil'),
    ]

    all_ok = True
    missing = []

    for module, package in dependencies:
        if not check_module(module, package):
            all_ok = False
            missing.append(package)

    # Check vLLM separately as it might not be installed in all environments
    print("\nChecking optional/GPU dependencies...")
    try:
        import vllm
        print("  OK: vllm")
    except ImportError:
        print("  MISSING: vllm (required for model inference)")
        missing.append("vllm")
        all_ok = False

    try:
        import transformers
        print("  OK: transformers")
    except ImportError:
        print("  MISSING: transformers")
        missing.append("transformers")
        all_ok = False

    try:
        import accelerate
        print("  OK: accelerate")
    except ImportError:
        print("  MISSING: accelerate")
        missing.append("accelerate")
        all_ok = False

    if not all_ok:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")

    return all_ok


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  OK: CUDA available")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")

            return True
        else:
            print("  WARNING: CUDA not available")
            print("  This system requires a CUDA-capable GPU for vLLM")
            return False

    except Exception as e:
        print(f"  ERROR checking CUDA: {e}")
        return False


def check_camera():
    """Check if camera is accessible."""
    print("\nChecking camera...")

    try:
        import cv2

        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("  OK: Camera accessible (device 0)")
            cap.release()
            return True
        else:
            print("  WARNING: Cannot open camera device 0")
            print("  You may need to specify a different device index")
            return False

    except Exception as e:
        print(f"  ERROR checking camera: {e}")
        return False


def check_config_files():
    """Check if configuration files exist."""
    print("\nChecking configuration files...")

    import os

    files = [
        'config.yaml',
        'prompts/templates.yaml',
        'requirements.txt'
    ]

    all_ok = True

    for file in files:
        if os.path.exists(file):
            print(f"  OK: {file}")
        else:
            print(f"  MISSING: {file}")
            all_ok = False

    return all_ok


def check_project_structure():
    """Check if project structure is correct."""
    print("\nChecking project structure...")

    import os

    directories = [
        'src',
        'prompts',
        'logs'
    ]

    files = [
        'src/__init__.py',
        'src/camera.py',
        'src/model.py',
        'src/preprocessor.py',
        'src/display.py',
        'src/config.py',
        'src/utils.py',
        'main.py'
    ]

    all_ok = True

    for dir in directories:
        if os.path.isdir(dir):
            print(f"  OK: {dir}/")
        else:
            print(f"  MISSING: {dir}/")
            all_ok = False

    for file in files:
        if os.path.exists(file):
            print(f"  OK: {file}")
        else:
            print(f"  MISSING: {file}")
            all_ok = False

    return all_ok


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Object Recognition System - Setup Verification")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA/GPU", check_cuda),
        ("Camera", check_camera),
        ("Configuration Files", check_config_files),
        ("Project Structure", check_project_structure),
    ]

    results = {}

    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    for name, status in results.items():
        status_str = "PASS" if status else "FAIL"
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}: {status_str}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)

    if all_passed:
        print("All checks passed! ✓")
        print("\nYou can now run the application:")
        print("  python main.py")
    else:
        print("Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print("  2. Ensure CUDA is properly installed for GPU support")
        print("  3. Check camera permissions and connections")

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
