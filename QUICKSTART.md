# Quick Start Guide

Get up and running with the Object Recognition System in minutes!

## Prerequisites

- Python 3.8+
- CUDA-capable GPU with 8GB+ VRAM
- Webcam
- CUDA 11.8+ installed

## Installation (5 minutes)

### 1. Set up virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: This will download ~4GB of dependencies including PyTorch and vLLM.

### 3. Verify setup

```bash
python verify_setup.py
```

This will check:
- Python version
- All dependencies
- CUDA/GPU availability
- Camera access
- Project structure

All checks should pass ✓

## First Run (2 minutes)

### Start the application

```bash
python main.py
```

**What happens:**
1. System initializes (shows GPU info)
2. Model downloads (first run only, ~15GB for LLaVA-1.5-7B)
3. Model loads into GPU (~20-30 seconds)
4. Camera starts
5. Display window appears

### You should see:
- Live webcam feed on the left
- Info panel on the right showing:
  - Performance metrics (FPS)
  - Processing status
  - Detection results

## Basic Usage

### Keyboard Controls

| Key | Action |
|-----|--------|
| **Q** | Quit application |
| **P** | Pause/Resume |
| **S** | Save current frame & results |
| **C** | Clear detection history |
| **1-5** | Switch prompt templates |

### Try This:

1. **Basic Detection** (Press `1`)
   - Shows simple list of objects

2. **Detailed Analysis** (Press `2`)
   - Objects with locations (left/right/center)

3. **Save a Detection** (Press `S`)
   - Saves to `output/` folder
   - Both image and text file

## Understanding the Display

### Top Bar
- Current status message

### Left Panel (Main View)
- Live webcam feed
- Real-time video at ~30 FPS

### Right Panel (Info)
- **Performance**: Capture/Inference/Display FPS
- **Queue**: Number of frames waiting for processing
- **Status**: Idle or Processing
- **Latest Detection**: Most recent result

### Bottom Right
- Keyboard shortcuts reminder

## Expected Performance

With default settings (LLaVA-1.5-7B on RTX 3090):

- **Display FPS**: 20-30 FPS (smooth video)
- **Inference FPS**: ~0.8 FPS (1 detection per 1.2 seconds)
- **Latency**: 1-2 seconds from action to detection

## Common First-Run Issues

### 1. "No GPU available"
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```
Should print: `True`

**Fix**: Install CUDA 11.8+ and compatible PyTorch

### 2. "Failed to open camera"
```bash
# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```
Should print: `True`

**Fix**:
- Check camera permissions
- Try different device index in `config.yaml`

### 3. "CUDA out of memory"
Edit `config.yaml`:
```yaml
model:
  gpu_memory_util: 0.6  # Reduce from 0.8
```

### 4. Model download slow/fails
- Requires good internet connection
- First run downloads ~15GB
- Subsequent runs use cached model

## Customization Quick Tips

### Use a different model

Edit `config.yaml`:
```yaml
model:
  name: "liuhaotian/llava-v1.5-13b"  # Larger, more accurate
```

### Faster performance

Edit `config.yaml`:
```yaml
processing:
  process_every_n_frames: 5  # Process fewer frames

model:
  max_tokens: 100  # Shorter responses
```

### Better accuracy

Edit `config.yaml`:
```yaml
model:
  temperature: 0.1  # More deterministic
  max_tokens: 200   # Longer descriptions
```

### Custom prompts

Edit `prompts/templates.yaml`:
```yaml
  my_prompt:
    name: "My Custom Prompt"
    key: "6"
    prompt: "Your custom prompt here"
    description: "What this prompt does"
```

Then press `6` while running!

## What to Try

### Test Different Scenes

1. **Desktop objects**: Keyboard, mouse, monitor
2. **Household items**: Books, plants, furniture
3. **Multiple objects**: See how it handles complexity
4. **Actions**: Wave, point, hold objects
5. **Different lighting**: Test in various conditions

### Test Different Prompts

Press `1`-`5` to cycle through:
1. Basic detection
2. Detailed with locations
3. Structured output
4. Scene understanding
5. Object counting

Notice how responses change!

### Save Interesting Results

Press `S` to save:
- Image: `output/detection_YYYYMMDD_HHMMSS.jpg`
- Text: `output/detection_YYYYMMDD_HHMMSS.txt`

## Next Steps

### Explore Advanced Features

See full [README.md](README.md) for:
- Multi-camera support
- Custom model selection
- Performance tuning
- API usage
- Troubleshooting

### Check Logs

```bash
# View logs
tail -f logs/object_recognition.log
```

### Monitor Performance

Watch the FPS counters:
- **Capture FPS**: Should be ~30
- **Inference FPS**: 0.5-1.0 is good
- **Display FPS**: Should be 20-30

## Getting Help

1. **Verify setup**: `python verify_setup.py`
2. **Check logs**: `logs/object_recognition.log`
3. **Read README**: [README.md](README.md)
4. **Common issues**: See Troubleshooting in README

## System Requirements Recap

### Minimum
- GPU: 8GB VRAM (RTX 3060, RTX 2080)
- RAM: 16GB
- CPU: 4 cores
- Storage: 30GB free

### Recommended
- GPU: 16GB+ VRAM (RTX 3090, RTX 4080)
- RAM: 32GB
- CPU: 8+ cores
- Storage: 50GB free (for multiple models)

## Tips for Best Results

1. **Good lighting**: Helps the model see clearly
2. **Clear camera**: Clean your webcam lens
3. **Stable position**: Keep camera steady
4. **Give it time**: First inference takes longer
5. **Experiment**: Try different prompts and settings

## Success Indicators

You know it's working when:
- ✓ Display shows live video
- ✓ Right panel shows FPS metrics
- ✓ "Processing..." appears periodically
- ✓ Detection text updates every few seconds
- ✓ Results make sense for what's visible

## Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| Slow inference | Increase `process_every_n_frames` to 5 |
| GPU memory error | Reduce `gpu_memory_util` to 0.6 |
| No camera | Change `device_index` in config |
| Choppy video | Normal - inference is separate from display |
| Poor detections | Try different prompts (keys 1-5) |

## Ready to Go!

```bash
# Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the application
python main.py

# Have fun! Press Q to quit when done
```

---

**Enjoy your real-time object recognition system!**

For detailed documentation, see [README.md](README.md)
