# Chameleon

Real-time background replacement and effects for Linux webcams. Add blur, custom backgrounds, and overlays to any video conferencing app.

## Why Chameleon?

Most video conferencing apps on Linux lack background effects support. Chameleon creates a virtual webcam with professional background replacement, blur, and overlay effects that work with Zoom, Teams, Meet, and any other video app.

## ✨ Highlights

- 🎭 **Background replacement** - Images or animated videos
- 🌫️ **Background blur** - Adjustable intensity (0-100%)
- 👤 **Selfie effects** - Hologram, blur, opacity, brightness
- 🖼️ **Foreground overlays** - Logos, branding, watermarks
- ⚡ **Hardware acceleration** - CUDA, VAAPI, VDPAU support
- 🚀 **High performance** - 60 FPS @ 1080p with YOLO11n-seg + GPU
- 🔄 **On-demand processing** - Low CPU usage when inactive
- 🎨 **Color maps** - Artistic visual effects (viridis, plasma, etc.)

### Performance

| Model | GPU Speed | CPU Speed | FPS @ 1080p |
|-------|-----------|-----------|-------------|
| **YOLO11n-seg** (recommended) | 2-5ms | 15-25ms | 60+ / 30-40 |
| MediaPipe (default) | N/A | 80-150ms | 6-12 |
| FastSAM-s | 10-15ms | 40-60ms | 20-30 |

## 🚀 Quick Start

### Prerequisites

**v4l2loopback kernel module** (required for virtual camera):

```bash
# Debian/Ubuntu
sudo apt-get install v4l2loopback-dkms

# Fedora
sudo dnf install v4l2loopback

# Arch Linux
sudo pacman -S v4l2loopback-dkms

# Load module
sudo modprobe v4l2loopback devices=1 exclusive_caps=1 video_nr=10 max_buffers=2 card_label="Virtual Cam"

# Or install persistently
./scripts/install-virtual-cam.sh
```

**Python 3.11 or 3.12** (MediaPipe not compatible with 3.13+):

```bash
# For systems with Python 3.13+ (e.g., Debian 13)
# use uv for best experience
uv python install 3.12

# Install pyenv first, then:
pyenv install 3.12
pyenv local 3.12
```

### Installation

```bash
# Clone repository
git clone https://github.com/hoangtrann/chameleon
cd chameleon

# Create virtual environment
uv venv --python 3.12

# or
python -m venv venv

# activate the venv
source venv/bin/activate

# Install, drop uv prefix if using pyenv
uv pip install --upgrade .
```

### Basic Usage

```bash
# Default (MediaPipe)
chameleon run

# or with configs
chameleon run --threshold 50 --fast-blur --background no,blur=50 --mask no

# Recommended - Much faster with YOLO11n-seg
chameleon run --model yolo11n-seg --use-clahe --temporal-smoothing 0.3 --adaptive-threshold --fast-blur --background no,blur=50

# Use in video apps
# Select "fake-cam" or /dev/video10 as your camera in Zoom, Teams, etc.
```

## 📖 Usage

Recommend using config file and simply run `chameleon run`

### Configuration File

You can copy the example config file and modify to your own preferences

```bash
cp config.example.conf config.conf
```

```conf
[DEFAULT]
webcam_path = /dev/video0
v4l2loopback_path = /dev/video10
width = 1280
height = 720
fps = 30

background = file=background.jpg,blur=20
selfie = brightness=110
mask = foreground=logo.png,opacity=80
```

```bash
chameleon run -c config.conf
```

### Effects Syntax

**Background:**
```bash
--background=file=mybg.jpg              # Image/video file
--background=no,blur=50                 # Blur intensity
--background=solid=255,0,0              # Solid color (B,G,R)
--background=cmap=viridis               # Color map
--background=file=bg.jpg,blur=30        # Combined effects
```

**Selfie:**
```bash
--selfie=hologram                       # Hologram effect
--selfie=blur=40                        # Blur person
--selfie=brightness=120                 # Adjust brightness
--selfie=opacity=80                     # Transparency
```

**Overlay:**
```bash
--mask=foreground=logo.png              # Overlay image
--mask=foreground=logo.png,opacity=60   # With transparency
```

### Performance Optimization

```

## 🛠️ Development

### Setup

```bash
# Development install
pip install --upgrade -e .

# Benchmark
./scripts/benchmark.sh
```

### Project Structure

```
chameleon/
├── cli.py              # CLI interface
├── config.py           # Configuration models
├── core/
│   ├── effects.py      # Image effects
│   └── pipeline.py     # Processing pipeline
├── input/
│   └── camera.py       # Webcam input
├── output/
│   └── virtual.py      # Virtual camera output
└── processing/
    ├── segmentation.py # Person segmentation
    └── compositor.py   # Image composition
```

## 🤝 Contributing

Contributions welcome! We especially need help with:

- Testing on different hardware configurations
- Hardware acceleration (CUDA, VAAPI, VDPAU)
- ONNX model integration
- Documentation improvements
- Unit tests

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run code quality checks (`scripts/lint.sh`)
5. Commit with descriptive messages
6. Push to your fork
7. Open a Pull Request

## 📋 Troubleshooting

**Camera not detected in apps:**
```bash
# Ensure exclusive_caps=1 is set
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=1 exclusive_caps=1 video_nr=10 max_buffers=10 card_label="Virtual Cam"
```

**Poor performance:**
- Lower resolution: `--width 640 --height 480`
- Use default model: don't specify `--model` when running to use the default model `mediapipe`
- Enable fast blur: `--fast-blur`

**Python 3.13+ compatibility:**
- Use `uv` or `pyenv`` to install Python 3.12 (MediaPipe requirement)

For more issues, see [GitHub Issues](https://github.com/hoangtrann/chameleon/issues).

## 📝 License

GPLv3 - See [LICENSE](LICENSE) for details.

```
Chameleon - Virtual webcam with background effects for Linux
Copyright (C) 2020-2025 Hoang Tran and contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```

## 🌟 Contributors

[![Contributors](https://contributors-img.web.app/image?repo=hoangtrann/chameleon)](https://github.com/hoangtrann/chameleon/graphs/contributors)

---

**⭐ Star this repository if you find it useful!**
