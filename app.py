import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import quote

import gradio as gr
import requests

BASE_DIR = Path(__file__).parent
INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
VIEWER_DIR = STATIC_DIR / "viewer"
THIRD_PARTY_DIR = BASE_DIR / "third_party"
COLMAP_DIR = THIRD_PARTY_DIR / "colmap"
COLMAP_ZIP_URL = (
    "https://github.com/colmap/colmap/releases/download/3.9/colmap-x64-windows.zip"
)

DEFAULT_ITERATIONS = 2000
DEFAULT_DOWNSCALE = 2

CURRENT_PROCESS: Optional[subprocess.Popen] = None
PROCESS_LOCK = threading.Lock()
CANCEL_EVENT = threading.Event()


def ensure_directories() -> None:
    for path in [INPUTS_DIR, OUTPUTS_DIR, MODELS_DIR, STATIC_DIR, VIEWER_DIR, THIRD_PARTY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def in_virtualenv() -> bool:
    return sys.prefix != sys.base_prefix


def run_subprocess(command: List[str], env: Optional[dict] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )


def ensure_python_version() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10+ is required.")


def ensure_requirements() -> None:
    requirements_file = BASE_DIR / "requirements.txt"
    marker_file = BASE_DIR / ".deps_installed"
    if marker_file.exists():
        return
    command = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
    result = run_subprocess(command)
    if result.returncode != 0:
        raise RuntimeError(f"Dependency installation failed:\n{result.stdout}")
    marker_file.write_text(str(time.time()))


def ensure_nerfstudio() -> None:
    if shutil.which("ns-train"):
        return
    result = run_subprocess([sys.executable, "-m", "pip", "install", "nerfstudio"])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install nerfstudio:\n{result.stdout}")


def find_colmap_bin() -> Optional[Path]:
    if not COLMAP_DIR.exists():
        return None
    for path in COLMAP_DIR.rglob("colmap.exe"):
        return path.parent
    return None


def download_colmap() -> Path:
    COLMAP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = COLMAP_DIR / "colmap.zip"
    response = requests.get(COLMAP_ZIP_URL, stream=True, timeout=60)
    response.raise_for_status()
    with zip_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(COLMAP_DIR)
    zip_path.unlink(missing_ok=True)
    colmap_bin = find_colmap_bin()
    if not colmap_bin:
        raise RuntimeError("COLMAP download completed but colmap.exe was not found.")
    return colmap_bin


def ensure_colmap() -> Path:
    colmap_bin = find_colmap_bin()
    if colmap_bin:
        return colmap_bin
    return download_colmap()


def get_pipeline_env(colmap_bin: Path) -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{colmap_bin};{env.get('PATH', '')}"
    return env


def check_torch_cuda() -> Tuple[bool, str]:
    try:
        import torch  # pylint: disable=import-error

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, "CUDA not available"
    except Exception as exc:  # pragma: no cover - optional dependency
        return False, f"Torch unavailable: {exc}"


def bootstrap() -> None:
    ensure_python_version()
    ensure_directories()

    if os.name == "nt" and not in_virtualenv():
        venv_dir = BASE_DIR / ".venv"
        if not venv_dir.exists():
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        venv_python = venv_dir / "Scripts" / "python.exe"
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "-r", str(BASE_DIR / "requirements.txt")])
        subprocess.check_call([str(venv_python), str(BASE_DIR / "app.py")])
        sys.exit(0)

    ensure_requirements()
    ensure_nerfstudio()
    ensure_colmap()


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            if member.is_dir():
                continue
            filename = Path(member.filename)
            target_path = destination / filename.name
            with zip_ref.open(member) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def copy_images(image_paths: Iterable[Path], destination: Path) -> None:
    for image_path in image_paths:
        destination_path = destination / image_path.name
        shutil.copy(image_path, destination_path)


def create_job_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def list_output_jobs() -> List[str]:
    if not OUTPUTS_DIR.exists():
        return []
    return sorted([p.name for p in OUTPUTS_DIR.iterdir() if p.is_dir()], reverse=True)


def build_viewer_html(scene_path: Optional[Path]) -> str:
    viewer_path = VIEWER_DIR / "index.html"
    if not viewer_path.exists():
        return "<div>Viewer assets missing.</div>"

    if scene_path is None:
        return "<div class='viewer-placeholder'>No scene loaded yet.</div>"

    resolved_scene = scene_path.resolve()
    scene_url = f"/file={resolved_scene.as_posix()}"
    viewer_url = f"/file={viewer_path.as_posix()}?scene={quote(scene_url)}"
    return (
        "<iframe src=\"{url}\" style=\"width: 100%; height: 520px; border: none;\"></iframe>"
    ).format(url=viewer_url)


def write_settings(job_dir: Path, settings: dict) -> None:
    with (job_dir / "settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)


def save_preview_image(job_dir: Path, images_dir: Path) -> None:
    for image in images_dir.iterdir():
        if image.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            shutil.copy(image, job_dir / "preview.jpg")
            return


def create_job_bundle(job_dir: Path, input_dir: Path) -> Path:
    bundle_path = job_dir / "job_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zip_ref:
        for root, _, files in os.walk(job_dir):
            for file in files:
                file_path = Path(root) / file
                zip_ref.write(file_path, arcname=f"outputs/{file_path.relative_to(job_dir)}")
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = Path(root) / file
                zip_ref.write(file_path, arcname=f"inputs/{file_path.relative_to(input_dir)}")
    return bundle_path


def terminate_process_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], check=False)
    else:
        process.terminate()


def stream_command(
    command: List[str],
    log_path: Path,
    env: dict,
) -> Iterable[str]:
    global CURRENT_PROCESS
    with PROCESS_LOCK:
        CURRENT_PROCESS = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    assert CURRENT_PROCESS.stdout is not None
    with log_path.open("a", encoding="utf-8") as log_handle:
        for line in CURRENT_PROCESS.stdout:
            log_handle.write(line)
            log_handle.flush()
            yield line
            if CANCEL_EVENT.is_set():
                terminate_process_tree(CURRENT_PROCESS)
                break

    with PROCESS_LOCK:
        if CURRENT_PROCESS:
            CURRENT_PROCESS.wait()
            return_code = CURRENT_PROCESS.returncode
            CURRENT_PROCESS = None
        else:
            return_code = 0

    if return_code != 0 and not CANCEL_EVENT.is_set():
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def gather_scene_file(export_dir: Path) -> Optional[Path]:
    for name in ["scene.splat", "scene.ply", "point_cloud.ply"]:
        candidate = export_dir / name
        if candidate.exists():
            return candidate
    for candidate in export_dir.glob("*.splat"):
        return candidate
    for candidate in export_dir.glob("*.ply"):
        return candidate
    return None


def prepare_inputs(zip_upload: Optional[str], image_files: Optional[List[str]], job_dir: Path) -> Path:
    input_dir = INPUTS_DIR / job_dir.name / "images"
    input_dir.mkdir(parents=True, exist_ok=True)

    if zip_upload:
        safe_extract_zip(Path(zip_upload), input_dir)
    elif image_files:
        copy_images([Path(file) for file in image_files], input_dir)

    return input_dir


def run_pipeline(
    zip_upload: Optional[str],
    image_files: Optional[List[str]],
    iterations: int,
    downscale: int,
    device_mode: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    CANCEL_EVENT.clear()
    logs = ""
    status = "Initializing job..."
    iterations = int(iterations)
    downscale = int(downscale)
    job_dir = create_job_dir()
    log_path = job_dir / "pipeline.log"
    input_dir = prepare_inputs(zip_upload, image_files, job_dir)

    if not any(input_dir.iterdir()):
        raise gr.Error("No images found. Upload a zip or multiple images.")

    colmap_bin = ensure_colmap()
    env = get_pipeline_env(colmap_bin)

    gpu_available, gpu_detail = check_torch_cuda()
    if device_mode == "Force CPU":
        env["CUDA_VISIBLE_DEVICES"] = ""
        status = "CPU-only mode enabled."
    elif not gpu_available:
        status = f"GPU not detected ({gpu_detail}). Using CPU defaults."
        env["CUDA_VISIBLE_DEVICES"] = ""

    settings = {
        "iterations": iterations,
        "downscale": downscale,
        "device_mode": device_mode,
        "input_dir": str(input_dir),
        "job_dir": str(job_dir),
    }
    write_settings(job_dir, settings)
    save_preview_image(job_dir, input_dir)

    proc_dir = job_dir / "processed"
    train_dir = job_dir / "trained"
    export_dir = job_dir / "export"
    proc_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        (
            [
                "ns-process-data",
                "images",
                "--data",
                str(input_dir),
                "--output-dir",
                str(proc_dir),
                "--downscale-factor",
                str(downscale),
            ],
            "Processing images with COLMAP...",
        ),
        (
            [
                "ns-train",
                "splatfacto",
                "--data",
                str(proc_dir),
                "--output-dir",
                str(train_dir),
                "--max-num-iterations",
                str(iterations),
            ],
            "Training Gaussian splats (Splatfacto)...",
        ),
    ]

    if device_mode == "Force CPU":
        commands[1][0].extend(["--pipeline.device", "cpu"])

    total_steps = len(commands) + 1
    progress(0, desc="Starting pipeline")

    for index, (command, step_status) in enumerate(commands, start=1):
        status = step_status
        progress(index / total_steps, desc=step_status)
        for line in stream_command(command, log_path, env):
            logs += line
            yield logs, status, build_viewer_html(None), None, None, gr.update(choices=list_output_jobs())
        if CANCEL_EVENT.is_set():
            status = "Pipeline canceled."
            progress(1, desc=status)
            yield logs, status, build_viewer_html(None), None, None, gr.update(choices=list_output_jobs())
            return

    config_path = train_dir / "config.yml"
    export_command = [
        "ns-export",
        "gaussian-splat",
        "--load-config",
        str(config_path),
        "--output-dir",
        str(export_dir),
        "--output-filename",
        "scene.splat",
    ]
    status = "Exporting splat scene..."
    progress((total_steps - 0.5) / total_steps, desc=status)
    for line in stream_command(export_command, log_path, env):
        logs += line
        yield logs, status, build_viewer_html(None), None, None, gr.update(choices=list_output_jobs())

    scene_file = gather_scene_file(export_dir)
    if not scene_file:
        raise gr.Error("Export finished but no scene file was found.")

    bundle_path = create_job_bundle(job_dir, input_dir.parent)
    status = "Done. Scene ready for exploration."
    progress(1, desc=status)
    yield (
        logs,
        status,
        build_viewer_html(scene_file),
        str(scene_file),
        str(bundle_path),
        gr.update(choices=list_output_jobs(), value=job_dir.name),
    )


def cancel_pipeline() -> str:
    CANCEL_EVENT.set()
    with PROCESS_LOCK:
        if CURRENT_PROCESS:
            terminate_process_tree(CURRENT_PROCESS)
            return "Cancel signal sent."
    return "No active pipeline to cancel."


def load_existing_job(job_name: str):
    if not job_name:
        return build_viewer_html(None), None
    job_dir = OUTPUTS_DIR / job_name
    export_dir = job_dir / "export"
    scene_file = gather_scene_file(export_dir)
    if not scene_file:
        return build_viewer_html(None), None
    return build_viewer_html(scene_file), str(scene_file)


def load_external_scene(scene_file: Optional[str]):
    if not scene_file:
        return build_viewer_html(None)
    return build_viewer_html(Path(scene_file))


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="OmniTool 3D Scene Builder") as demo:
        gr.Markdown(
            "# OmniTool: Multi-image Gaussian Splatting Builder\n"
            "Upload a zip of images or multiple images to reconstruct a navigable scene."
        )

        with gr.Tab("Create Scene (Multi-image)"):
            with gr.Row():
                zip_upload = gr.File(label="Upload ZIP of images", file_types=[".zip"])
                image_upload = gr.File(
                    label="Or upload multiple images",
                    file_count="multiple",
                    file_types=["image"],
                )

            with gr.Row():
                iterations = gr.Number(
                    label="Max training iterations",
                    value=DEFAULT_ITERATIONS,
                    precision=0,
                )
                downscale = gr.Number(
                    label="Downscale factor",
                    value=DEFAULT_DOWNSCALE,
                    precision=0,
                )
                device_mode = gr.Dropdown(
                    label="Device mode",
                    choices=["Auto", "Force CPU"],
                    value="Auto",
                )

            with gr.Row():
                run_button = gr.Button("Run pipeline", variant="primary")
                cancel_button = gr.Button("Cancel")

            status_text = gr.Markdown("Status: Ready")
            log_box = gr.Textbox(
                label="Logs",
                value="",
                lines=16,
                interactive=False,
            )
            viewer_html = gr.HTML(build_viewer_html(None))
            with gr.Row():
                scene_download = gr.File(label="Download scene file")
                bundle_download = gr.File(label="Download job bundle")

        with gr.Tab("Load / Explore Existing Scene"):
            job_dropdown = gr.Dropdown(
                label="Saved jobs",
                choices=list_output_jobs(),
                interactive=True,
            )
            load_button = gr.Button("Load selected job")
            external_file = gr.File(label="Load external .splat/.ply", file_types=[".splat", ".ply"])
            viewer_existing = gr.HTML(build_viewer_html(None))
            scene_download_existing = gr.File(label="Scene file")

        with gr.Tab("Single Image (Limited)"):
            gr.Markdown(
                "Single-image reconstruction is optional and limited.\n"
                "For now, use the multi-image workflow for best results."
            )

        run_button.click(
            run_pipeline,
            inputs=[zip_upload, image_upload, iterations, downscale, device_mode],
            outputs=[log_box, status_text, viewer_html, scene_download, bundle_download, job_dropdown],
        )
        cancel_button.click(cancel_pipeline, outputs=status_text)

        load_button.click(
            load_existing_job,
            inputs=job_dropdown,
            outputs=[viewer_existing, scene_download_existing],
        )
        external_file.change(
            load_external_scene,
            inputs=external_file,
            outputs=viewer_existing,
        )

    return demo


def main() -> None:
    bootstrap()
    demo = build_ui()
    demo.launch(
        allowed_paths=[str(OUTPUTS_DIR), str(STATIC_DIR)],
        server_name="0.0.0.0",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
