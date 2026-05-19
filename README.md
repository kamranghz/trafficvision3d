<div align="center">

# TrafficVision3D

**Turn a monocular traffic video into an animated 3D USD scene — ready for NVIDIA Isaac Sim or USD Composer**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=github&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![USD](https://img.shields.io/badge/USD-OpenUSD-76b900?logo=nvidia&logoColor=white)](https://openusd.org/)
[![Isaac Sim](https://img.shields.io/badge/NVIDIA-Isaac%20Sim-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/isaac-sim)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*Vehicle detection · Multi-object tracking · 2D-to-3D mapping · Animated USD export*

[Overview](#overview) · [How It Works](#how-it-works) · [Installation](#installation) · [Quick Start](#quick-start) · [Output](#output) · [Viewers](#opening-in-nvidia-tools) · [Roadmap](#roadmap)

</div>

---

> **Demo**
>
> <p align="center">
>   <video src="docs/demo.mp4" controls width="900"></video>
> </p>
>
> [Watch demo video](docs/demo.mp4)

---

## Overview

TrafficVision3D is a fully automated pipeline that processes an ordinary monocular traffic video and produces a physics-ready, animated 3D scene in Universal Scene Description (USD) format. No depth camera or LiDAR required.

**What the pipeline delivers:**

- Detects vehicles (cars, buses, trucks) using YOLOv8
- Assigns and maintains persistent track IDs across frames
- Estimates each vehicle's 3D world position and heading from 2D image motion
- Exports a complete animated highway scene — road surface, lighting, and vehicles — as a single `.usd` file
- Writes per-track metadata to JSON for downstream analysis

The output opens directly in **NVIDIA Isaac Sim** or **NVIDIA USD Composer** with no post-processing required.

---

## How It Works

```
Traffic Video (.mp4 / .avi / .mov)
         │
         ▼
  ┌─────────────┐
  │  Detection  │  YOLOv8 — detects car, bus, truck per frame
  └──────┬──────┘
         │ bounding boxes
         ▼
  ┌─────────────┐
  │  Tracking   │  Distance-based assignment → stable per-vehicle IDs
  └──────┬──────┘
         │ trajectories
         ▼
  ┌──────────────┐
  │  3D Mapping  │  2D image coords → road-space (X, Y, Z) coordinates
  └──────┬───────┘
         │ world positions
         ▼
  ┌─────────────────┐
  │  Orientation    │  Heading angle computed from trajectory direction
  └──────┬──────────┘
         │ pose + heading
         ▼
  ┌──────────────┐
  │  USD Export  │  Scene, road, lighting, and vehicle animations
  └──────┬───────┘
         │
         ▼
  output/traffic_animation.usd
  output/realistic_animation_summary.json
```

### Pipeline stages

| Stage | Method |
|---|---|
| **Detection** | YOLOv8 — COCO traffic classes (`car`, `bus`, `truck`) |
| **Tracking** | Distance-based multi-object assignment with persistent IDs |
| **3D mapping** | Projective 2D → road-plane homography |
| **Orientation** | Heading estimated from frame-to-frame trajectory vector |
| **USD export** | OpenUSD API — scene graph, road mesh, point lights, animated Xform prims |

---

## Project Structure

```text
trafficvision3d/
│
├── main.py                   # Pipeline entry point
├── requirements.txt
│
├── output/                   # Generated files (created at runtime)
│   ├── traffic_animation.usd
│   └── realistic_animation_summary.json
│
├── docs/
│   └── demo.mp4              # Demo video (place here for GitHub preview)
│
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.10+

> **Note on `pxr` (USD Python bindings):** The `pxr` package is bundled with NVIDIA Omniverse and Isaac Sim environments. If you are running outside those environments, install [OpenUSD Python bindings](https://openusd.org/release/python_support.html) compatible with your platform separately.

```bash
# 1. Clone the repository
git clone https://github.com/kamranghz/trafficvision3d.git
cd trafficvision3d

# 2. Create and activate a virtual environment
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Basic run

```bash
python main.py --video path/to/traffic_video.mp4
```

### With explicit output options

```bash
python main.py \
  --video path/to/traffic_video.mp4 \
  --output traffic_animation.usd \
  --output-dir output
```

Supported input formats: `.mp4`, `.avi`, `.mov`

---

## Output

A successful run produces two files in the `output/` directory:

| File | Description |
|---|---|
| `traffic_animation.usd` | Animated 3D scene — road, lighting, and all tracked vehicles |
| `realistic_animation_summary.json` | Per-track metadata and processing summary |

### JSON summary format

```json
{
  "total_frames": 450,
  "tracks": [
    {
      "id": 3,
      "class": "car",
      "first_frame": 12,
      "last_frame": 401,
      "avg_speed_ms": 14.2
    }
  ]
}
```

---

## Opening in NVIDIA Tools

### Isaac Sim *(recommended for robotics / simulation workflows)*

1. Launch **Isaac Sim**
2. Go to `File → Open`
3. Select `output/traffic_animation.usd`
4. Press **Play** on the timeline

### USD Composer *(recommended for scene inspection and USD editing)*

1. Launch **USD Composer**
2. Go to `File → Open`
3. Select `output/traffic_animation.usd`
4. Use the timeline playback controls to review the animation

---

## Limitations

- **Approximate depth** — 3D position estimation uses a projective road-plane mapping, not calibrated photogrammetry. Metric accuracy depends on camera viewpoint and video quality.
- **COCO classes only** — vehicle detection is currently limited to the traffic-relevant classes available in the YOLOv8 COCO model (`car`, `bus`, `truck`).
- **Monocular input** — no stereo or depth-sensor support at this time.

---

## Roadmap

- [ ] Camera calibration input for metric-accurate 3D reconstruction
- [ ] Full MOT pipeline — appearance features + Kalman filter + Hungarian algorithm
- [ ] Config file for detection thresholds and world-mapping parameters
- [ ] Optional loading of real vehicle USD asset libraries
- [ ] Multi-camera support

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Kamran Gholizadeh HamlAbadi**
PhD Candidate, University of Ottawa · MCRLab
[github.com/kamranghz](https://github.com/kamranghz) · [LinkedIn](https://www.linkedin.com/in/kamrangh)
