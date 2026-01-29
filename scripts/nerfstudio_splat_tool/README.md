# Nerfstudio Splatfacto Tool (Windows)

This standalone Gradio app builds navigable 3D scenes from multi-image captures using Nerfstudio Splatfacto and COLMAP.

## Quick start (Windows)

1. Double-click `start.bat`.
2. The script creates a local virtual environment in `.venv`, installs dependencies, downloads COLMAP, and launches the UI.
3. Open the Gradio URL in your browser.

If you already activated a virtual environment, you can also run:

```bash
python app.py
```

## Recommended capture

- 20–100 images with good overlap and stable lighting.
- Walk around the subject, keep it static, and avoid motion blur.
- Higher overlap yields better reconstructions.

## Pipeline flow

1. Upload a ZIP of images (preferred) or select multiple images.
2. Click **Run Pipeline**. The app runs:
   - `ns-process-data images`
   - `ns-train splatfacto`
   - `ns-export gaussian-splat` and `ns-export pointcloud`
3. Once complete, navigate the scene in the embedded viewer and download the outputs.

The job bundle ZIP includes inputs, processed data, configs, and exported files.

## Troubleshooting

- **COLMAP fails**: Ensure the image set has sufficient overlap and texture; try increasing the downscale factor.
- **CUDA missing**: The app falls back to CPU mode automatically but will run slower.
- **Dependencies fail to install**: Update your GPU drivers, and ensure Python 3.10+ is installed.

## Notes

- The viewer currently loads `.ply` outputs for navigation (with a `.splat` download provided). If you want a dedicated splat viewer, you can swap in a WebGL splat viewer and point it at the exported `.splat` file.
