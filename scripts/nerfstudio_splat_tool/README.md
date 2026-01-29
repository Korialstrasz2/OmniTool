# Nerfstudio Splatfacto Tool (Windows)

This standalone Gradio app builds navigable 3D scenes from multi-image captures using Nerfstudio Splatfacto and COLMAP. It also supports single-image reconstruction using depth estimation for one-shot captures.

## Quick start (Windows)

1. Double-click `start.bat`.
2. The script creates a local virtual environment in `.venv`, installs dependencies, downloads COLMAP, and launches the UI.
3. Open the Gradio URL in your browser.

If you already activated a virtual environment, you can also run:

```bash
python app.py
```

## CUDA install (Windows, this tool)

This tool pins Nerfstudio 1.1.4 and PyTorch 2.1.2. To avoid CUDA/torch mismatches, install the CUDA build **inside** the tool's virtual environment:

```bash
.\scripts\nerfstudio_splat_tool\.venv\Scripts\activate
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Notes:
- You do **not** need the full CUDA toolkit; the PyTorch CUDA wheels bundle the runtime. You only need an up-to-date NVIDIA driver.
- If you reinstall dependencies, re-run the CUDA install command to ensure the CUDA build stays in place.

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

The job bundle ZIP includes inputs, processed data, configs, and exported files, plus a `viewer.html`
file that can be opened locally alongside `scene.ply`.

## Single-image flow

1. Open the **Single Image (Depth)** tab or upload a single image in the multi-image tab.
2. Pick the depth model (Depth Anything V2 or Apple SHARP on macOS) and quality setting.
3. Click **Run Single-Image Pipeline** (or run the multi-image pipeline with a single input).
4. The app estimates depth, builds a colored point cloud, and exports `scene.ply`.

## Troubleshooting

- **COLMAP fails**: Ensure the image set has sufficient overlap and texture; try increasing the downscale factor.
- **COLMAP `--SiftExtraction.use_gpu` option error**: The tool now auto-drops this flag when the bundled COLMAP build does not support it. If the error persists, delete any custom COLMAP installations from your PATH and rerun so the bundled version is used.
- **COLMAP `Failed to read faiss index` / vocab tree error**: Newer COLMAP builds require a FAISS-based vocab tree, but Nerfstudio bundles a legacy FLANN index. The tool auto-retries feature matching with the exhaustive matcher. If you still see the error, update COLMAP or regenerate the vocab tree with COLMAP's `vocab_tree_upgrader`.
- **Single-image mode**: If only one image is provided, the app automatically switches to the depth-based pipeline to avoid COLMAP errors.
- **Depth model download fails/offline**: Ensure the Depth Anything V2 model is cached locally (Hugging Face cache) or install from a local checkpoint. You can also pre-download with `python -m huggingface_hub.snapshot_download --repo-id depth-anything/Depth-Anything-V2-Large-hf`.
- **Apple SHARP**: Requires macOS + CoreML. On Windows, use Depth Anything V2 instead.
- **COLMAP download fails (404/blocked)**: Download the Windows ZIP from the COLMAP releases page and extract it into `scripts/nerfstudio_splat_tool/third_party/colmap`, then rerun the app.
- **CUDA missing**: The app falls back to CPU mode automatically but will run slower.
- **CUDA available but not detected**: Ensure PyTorch is installed with CUDA support (the CPU-only build will always report no CUDA).
- **Downscale flag error**: The app auto-detects the correct `ns-process-data` downscale flag for your Nerfstudio version. If you still see an error, rerun after updating Nerfstudio.
- **Dependencies fail to install**: Update your GPU drivers, and ensure Python 3.10+ is installed.
- **UnicodeEncodeError (emoji / cp1252)**: This happens on legacy Windows terminals when Rich prints emoji. Ensure `PYTHONUTF8=1` and `RICH_NO_EMOJI=1` are set in your environment, or run the tool from a UTF-8-capable terminal (Windows Terminal / VS Code).

## Notes

- The viewer currently loads `.ply` outputs for navigation (with a `.splat` download provided). If you want a dedicated splat viewer, you can swap in a WebGL splat viewer and point it at the exported `.splat` file.
