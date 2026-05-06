# TrafficVision3D

Turn a traffic video into an animated 3D USD scene using YOLOv8 detection, multi-object tracking, and automatic 2D-to-3D mapping.

The output is ready to inspect in **NVIDIA Isaac Sim** or **NVIDIA USD Composer**.

---

## What this project does

- Reads a monocular traffic video (`.mp4`, `.avi`, `.mov`)
- Detects vehicles with YOLOv8
- Tracks vehicles across frames with persistent IDs
- Estimates 3D world positions and heading from motion
- Generates an animated USD highway scene
- Exports track metadata for analysis

---

## Demo on GitHub (Video in README)

To show video on the GitHub main page, place a demo file in your repo, for example:

- `docs/demo.mp4`

Then use this in `README.md`:

```html
<p align="center">
  <video src="docs/demo.mp4" controls width="900"></video>
</p>
```

If a browser does not render inline video, add a fallback link:

```md
[Watch demo video](docs/demo.mp4)
```

---

## Output viewers

You can open the generated `.usd` scene in either:

- **Isaac Sim** (recommended when you want robotics/simulation workflow)
- **USD Composer** (recommended when you want scene inspection and USD editing)

Both can load `output/traffic_animation.usd`.

---

## Project structure

```text
.
├── main.py
├── requirements.txt
├── output/
│   └── (generated .usd + summary .json)
└── README.md
```

---

## Quick start

### 1) Create environment

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> Note: `pxr` (USD Python API) is typically provided by Omniverse/Isaac Sim environments.  
> If your system does not provide it, install USD Python bindings compatible with your platform.

### 3) Run pipeline

```bash
python main.py --video path/to/traffic_video.mp4
```

Optional:

```bash
python main.py --video path/to/traffic_video.mp4 --output traffic_animation.usd --output-dir output
```

---

## Generated files

After a successful run:

- `output/traffic_animation.usd` - animated 3D scene
- `output/realistic_animation_summary.json` - processing and track summary

---

## Open in Isaac Sim

1. Launch Isaac Sim
2. `File -> Open`
3. Select `output/traffic_animation.usd`
4. Press **Play** on timeline

---

## Open in USD Composer

1. Launch USD Composer
2. `File -> Open`
3. Select `output/traffic_animation.usd`
4. Use timeline playback to review animation

---

## How it works (high level)

1. **Detection** - YOLOv8 detects `car`, `bus`, `truck`
2. **Tracking** - a distance-based assignment keeps IDs stable
3. **3D mapping** - 2D image coordinates are mapped to road-space coordinates
4. **Orientation** - heading is computed from trajectory direction
5. **USD export** - scene, lighting, road, and animated vehicles are written to USD

---

## Current limitations

- Monocular depth estimation is approximate (not calibrated photogrammetry)
- Accuracy depends on camera viewpoint and video quality
- Vehicle classes are currently limited to COCO traffic classes used in script

---

## Roadmap ideas

- Camera calibration input for improved metric accuracy
- Better MOT data association (appearance features + Kalman/Hungarian full pipeline)
- Config file for thresholds and world mapping parameters
- Optional real vehicle USD asset loading

---

## License

MIT

# trafficvision3d
