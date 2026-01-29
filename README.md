# OmniTool 3D Scene Builder (Windows)

Build navigable 3D scenes from multiple images using **Nerfstudio Splatfacto** + **COLMAP**. The app handles first-run dependency installs, downloads COLMAP for Windows automatically, and launches a Gradio UI with an embedded WebGL viewer.

## One-command run

```bash
python app.py
```

On the first run (Windows):
- Creates a local `.venv` if you are not already in a virtual environment.
- Installs Python dependencies.
- Downloads the COLMAP Windows binary to `./third_party/colmap`.

Then Gradio launches at `http://localhost:7860`.

## Recommended capture workflow

- Capture **20–100 images** with strong overlap.
- Keep the scene static and evenly lit.
- Avoid motion blur and reflective surfaces when possible.

## App workflow

### Tab 1: Create Scene (Multi-image)
1. Upload a **ZIP** of images (preferred) or select multiple images.
2. Set training iterations and downscale factor.
3. Click **Run pipeline** and follow logs.
4. Explore the scene in the embedded viewer (WASD/mouse or orbit controls).
5. Download the exported `scene.splat` or the job bundle zip.

### Tab 2: Load / Explore Existing Scene
- Choose a prior job from the dropdown.
- Or upload an external `.splat` / `.ply`.

### Tab 3: Single Image (Limited)
Single-image reconstruction is not enabled in this build; use multi-image capture for best results.

## Outputs
Each run creates a job folder:

```
outputs/<timestamp>_<shortid>/
  ├─ processed/
  ├─ trained/
  ├─ export/
  │   └─ scene.splat
  ├─ job_bundle.zip
  ├─ preview.jpg
  ├─ pipeline.log
  └─ settings.json
```

## Troubleshooting

**COLMAP fails or missing binaries**
- Delete `third_party/colmap` and re-run `python app.py` to force a re-download.

**CUDA missing / GPU not detected**
- The pipeline will run on CPU with smaller defaults, but will be much slower.

**Nerfstudio CLI not found**
- The app attempts to install it automatically. If it fails, run:
  ```bash
  pip install nerfstudio
  ```

**Viewer shows a blank screen**
- The embedded viewer uses CDN-hosted WebGL modules (GaussianSplats3D + Three.js). Ensure your browser has internet access or vendor those scripts locally under `static/viewer`.
