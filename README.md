# OmniTool – Gaussian Splat Scene Builder (Windows)

This project is a **Windows-first Gradio app** that takes **one or many images**, reconstructs a **navigable 3D Gaussian Splat scene** with Nerfstudio Splatfacto + COLMAP, and lets you **explore + download** the result in-browser.

## One-command run (Windows 10/11)

```bash
python app.py
```

On first run, the app will:
- Create a local `.venv` if needed.
- Install dependencies from `requirements.txt`.
- Download COLMAP (Windows build) into `./third_party/colmap`.
- Create `./inputs`, `./outputs`, and `./models`.

Then it will launch Gradio on `http://localhost:7860`.

> If you are already in a virtual environment, the app will reuse it and just ensure dependencies are installed.

## Recommended capture
- **20–100 images** with good overlap.
- Walk around the scene with **consistent lighting** and a **static subject**.
- Avoid motion blur and auto-exposure flicker.

## App workflow

### Tab 1 – Create Scene (Multi-image)
1. Upload a **ZIP of images** or select multiple images directly.
2. Set iterations and downscale.
3. Click **Run**.
4. Wait for processing, training, and export.
5. Explore the scene in the embedded viewer.
6. Download the `.splat` file or the full job bundle.

### Tab 2 – Load / Explore Existing Scene
- Pick a previous job from the dropdown **or** upload your own `.splat`/`.ply` file.
- The embedded viewer updates immediately.

### Tab 3 – Single Image (Limited)
Single-image reconstruction is not enabled in this build. Use the multi-image pipeline for full 3D scenes.

## Outputs
Each run creates a unique job folder at:

```
outputs/<timestamp>_<shortid>/
```

Contents include:
- `settings.json`
- `logs.txt`
- `preview.jpg/png`
- `processed/` (COLMAP + Nerfstudio processed data)
- `train/` (training outputs + config.yml)
- `export/scene.splat`
- `job_bundle.zip`

## Troubleshooting

### COLMAP fails to run
- Check that the auto-downloaded COLMAP folder exists: `third_party/colmap/bin`.
- Ensure the scene has enough overlap and texture.

### CUDA not available
- The app will fall back to CPU mode automatically.
- Expect significantly longer training times.

### Viewer doesn’t load
- The embedded viewer uses a CDN-loaded WebGL library. If your network blocks it, download the viewer library and update `static/viewer/viewer.js` to load it locally.

## Notes
- The viewer expects `.splat` outputs. `.ply` uploads are supported in the “Load” tab via a Three.js point cloud viewer.
- For best performance, use a machine with an NVIDIA GPU and CUDA drivers installed.
