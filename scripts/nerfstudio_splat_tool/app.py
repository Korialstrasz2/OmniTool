import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
VIEWER_DIR = STATIC_DIR / "viewer"
INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
THIRD_PARTY_DIR = BASE_DIR / "third_party"
COLMAP_DIR = THIRD_PARTY_DIR / "colmap"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"

COLMAP_ZIP_URL = (
    "https://github.com/colmap/colmap/releases/download/3.8/colmap-x64-windows.zip"
)

DEFAULT_ITERATIONS = 2000
DEFAULT_DOWNSCALE = 2

RUNNER_LOCK = threading.Lock()
ACTIVE_PROCESS: Optional[subprocess.Popen] = None
CANCEL_EVENT = threading.Event()


@dataclass
class JobPaths:
    job_id: str
    job_dir: Path
    input_dir: Path
    processed_dir: Path
    train_dir: Path
    export_dir: Path
    bundle_zip: Path
    preview_image: Path


def ensure_directories() -> None:
    for path in (INPUTS_DIR, OUTPUTS_DIR, MODELS_DIR, THIRD_PARTY_DIR, COLMAP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def run_subprocess(command: List[str], env: Optional[Dict[str, str]] = None) -> None:
    subprocess.check_call(command, env=env)


def install_requirements() -> None:
    if not REQUIREMENTS_FILE.exists():
        return
    run_subprocess([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])


def ensure_gradio():
    if importlib.util.find_spec("gradio") is None:
        install_requirements()
    import gradio as gr  # noqa: WPS433 - runtime import is intentional

    return gr


gr = ensure_gradio()


def download_colmap() -> Path:
    import urllib.request

    archive_path = COLMAP_DIR / "colmap.zip"
    if archive_path.exists():
        archive_path.unlink()
    COLMAP_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(COLMAP_ZIP_URL) as response:
        archive_path.write_bytes(response.read())
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(COLMAP_DIR)
    archive_path.unlink(missing_ok=True)
    return find_colmap_bin()


def find_colmap_bin() -> Path:
    for path in COLMAP_DIR.rglob("COLMAP.exe"):
        return path.parent
    for path in COLMAP_DIR.rglob("colmap.exe"):
        return path.parent
    raise FileNotFoundError("COLMAP executable not found after extraction.")


def ensure_colmap() -> Path:
    try:
        return find_colmap_bin()
    except FileNotFoundError:
        return download_colmap()


def check_python_version() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10+ is required.")


def check_dependencies() -> Tuple[Dict[str, str], List[str]]:
    check_python_version()
    ensure_directories()

    logs: List[str] = []
    if not in_virtualenv():
        logs.append(
            "Warning: Running outside a virtual environment. "
            "It is recommended to use start.bat or activate the tool's venv."
        )

    try:
        import torch  # noqa: F401
    except ImportError:
        logs.append("Installing Python dependencies...")
        install_requirements()

    try:
        import torch

        if not torch.cuda.is_available():
            logs.append("CUDA not detected. Training will run in CPU mode with smaller defaults.")
    except Exception as exc:  # pragma: no cover - depends on torch install
        logs.append(f"Torch check failed: {exc}")

    env = os.environ.copy()
    colmap_bin = ensure_colmap()
    env["PATH"] = f"{colmap_bin}{os.pathsep}{env.get('PATH', '')}"

    if not shutil.which("ns-process-data", path=env.get("PATH")):
        logs.append("Nerfstudio CLI not found. Installing nerfstudio...")
        run_subprocess([sys.executable, "-m", "pip", "install", "nerfstudio"], env=env)

    return env, logs


def make_job_paths() -> JobPaths:
    job_id = time.strftime("%Y%m%d_%H%M%S") + f"_{uuid4().hex[:6]}"
    job_dir = OUTPUTS_DIR / job_id
    input_dir = job_dir / "inputs"
    processed_dir = job_dir / "processed"
    train_dir = job_dir / "train"
    export_dir = job_dir / "export"
    bundle_zip = job_dir / "bundle.zip"
    preview_image = job_dir / "preview.jpg"
    for path in (job_dir, input_dir, processed_dir, train_dir, export_dir):
        path.mkdir(parents=True, exist_ok=True)
    return JobPaths(
        job_id=job_id,
        job_dir=job_dir,
        input_dir=input_dir,
        processed_dir=processed_dir,
        train_dir=train_dir,
        export_dir=export_dir,
        bundle_zip=bundle_zip,
        preview_image=preview_image,
    )


def extract_inputs(zip_file: Optional[str], images: List[str], job: JobPaths) -> None:
    if zip_file is not None:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(job.input_dir)
    else:
        for image in images:
            dest = job.input_dir / Path(image).name
            shutil.copy(image, dest)


def write_preview(job: JobPaths) -> None:
    try:
        from PIL import Image

        candidates = list(job.input_dir.glob("*"))
        if not candidates:
            return
        image_path = candidates[0]
        with Image.open(image_path) as img:
            img.thumbnail((640, 360))
            img.save(job.preview_image)
    except Exception:
        return


def find_config_file(train_dir: Path) -> Path:
    for config in train_dir.rglob("config.yml"):
        return config
    raise FileNotFoundError("Training config.yml not found.")


def build_commands(
    job: JobPaths,
    iterations: int,
    downscale: int,
) -> List[List[str]]:
    return [
        [
            "ns-process-data",
            "images",
            "--data",
            str(job.input_dir),
            "--output-dir",
            str(job.processed_dir),
            "--downscale-factor",
            str(downscale),
        ],
        [
            "ns-train",
            "splatfacto",
            "--data",
            str(job.processed_dir),
            "--output-dir",
            str(job.train_dir),
            "--max-num-iterations",
            str(iterations),
        ],
    ]


def build_export_commands(train_dir: Path, export_dir: Path) -> List[List[str]]:
    config_path = find_config_file(train_dir)
    return [
        [
            "ns-export",
            "gaussian-splat",
            "--load-config",
            str(config_path),
            "--output-dir",
            str(export_dir),
            "--output-filename",
            "scene.splat",
        ],
        [
            "ns-export",
            "pointcloud",
            "--load-config",
            str(config_path),
            "--output-dir",
            str(export_dir),
            "--output-filename",
            "scene.ply",
        ],
    ]


def run_command_stream(
    command: List[str],
    env: Dict[str, str],
    log_lines: List[str],
) -> Iterable[str]:
    global ACTIVE_PROCESS

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(BASE_DIR),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    ACTIVE_PROCESS = process
    if process.stdout:
        for line in process.stdout:
            log_lines.append(line.rstrip())
            yield line
    process.wait()
    ACTIVE_PROCESS = None
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def terminate_active_process() -> None:
    global ACTIVE_PROCESS
    if ACTIVE_PROCESS is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(ACTIVE_PROCESS.pid)],
                check=False,
                capture_output=True,
            )
        else:
            ACTIVE_PROCESS.terminate()
    finally:
        ACTIVE_PROCESS = None


def zip_job(job: JobPaths) -> None:
    if job.bundle_zip.exists():
        job.bundle_zip.unlink()
    with zipfile.ZipFile(job.bundle_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in job.job_dir.rglob("*"):
            if path == job.bundle_zip:
                continue
            zipf.write(path, path.relative_to(job.job_dir))


def resolve_download_scene(export_dir: Path) -> Optional[Path]:
    splat = export_dir / "scene.splat"
    ply = export_dir / "scene.ply"
    if splat.exists():
        return splat
    if ply.exists():
        return ply
    return None


def resolve_viewer_scene(export_dir: Path) -> Optional[Path]:
    ply = export_dir / "scene.ply"
    splat = export_dir / "scene.splat"
    if ply.exists():
        return ply
    if splat.exists():
        return splat
    return None


def viewer_html(scene_path: Optional[Path]) -> str:
    viewer_path = VIEWER_DIR / "index.html"
    if scene_path is None:
        return "<div class='viewer-placeholder'>No scene loaded yet.</div>"
    viewer_url = f"/file={quote(str(viewer_path))}?scene=/file={quote(str(scene_path))}"
    return (
        "<iframe class='viewer-frame' "
        f"src='{viewer_url}' "
        "frameborder='0'></iframe>"
    )


def run_pipeline(
    zip_input: Optional[str],
    images: List[str],
    iterations: int,
    downscale: int,
    device_mode: str,
    progress: gr.Progress = gr.Progress(),
):
    images = images or []
    if not zip_input and not images:
        yield "", "Please upload a ZIP or images.", viewer_html(None), None, None
        return

    CANCEL_EVENT.clear()
    env, dep_logs = check_dependencies()
    log_lines: List[str] = []
    for line in dep_logs:
        log_lines.append(line)
    yield "\n".join(log_lines), "Preparing job...", viewer_html(None), None, None

    job = make_job_paths()
    extract_inputs(zip_input, images, job)
    write_preview(job)

    settings = {
        "iterations": iterations,
        "downscale": downscale,
        "device_mode": device_mode,
    }
    with open(job.job_dir / "settings.json", "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)

    if device_mode == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        log_lines.append("CPU mode selected. This will be significantly slower.")
    elif device_mode == "auto":
        log_lines.append("Auto device mode selected.")

    progress(0.05, desc="Processing images")
    try:
        for command in build_commands(job, iterations, downscale):
            log_lines.append(f"$ {' '.join(command)}")
            for line in run_command_stream(command, env, log_lines):
                yield "\n".join(log_lines), "Running pipeline...", viewer_html(None), None, None
                if CANCEL_EVENT.is_set():
                    raise RuntimeError("Cancelled")
    except Exception as exc:
        terminate_active_process()
        log_lines.append(f"Pipeline failed: {exc}")
        yield "\n".join(log_lines), "Pipeline failed.", viewer_html(None), None, None
        return

    progress(0.7, desc="Exporting scene")
    try:
        for command in build_export_commands(job.train_dir, job.export_dir):
            log_lines.append(f"$ {' '.join(command)}")
            for line in run_command_stream(command, env, log_lines):
                yield "\n".join(log_lines), "Exporting scene...", viewer_html(None), None, None
                if CANCEL_EVENT.is_set():
                    raise RuntimeError("Cancelled")
    except Exception as exc:
        terminate_active_process()
        log_lines.append(f"Export failed: {exc}")
        yield "\n".join(log_lines), "Export failed.", viewer_html(None), None, None
        return

    progress(0.9, desc="Packaging output")
    zip_job(job)
    scene_file = resolve_download_scene(job.export_dir)
    viewer_scene = resolve_viewer_scene(job.export_dir)
    if scene_file is None:
        log_lines.append("No exported scene file found.")
        yield "\n".join(log_lines), "Export finished, but no scene found.", viewer_html(None), None, None
        return

    log_lines.append("Pipeline completed successfully.")
    yield (
        "\n".join(log_lines),
        f"Completed job {job.job_id}.",
        viewer_html(viewer_scene),
        str(scene_file),
        str(job.bundle_zip),
    )


def cancel_pipeline() -> str:
    CANCEL_EVENT.set()
    terminate_active_process()
    return "Cancellation requested."


def list_output_jobs() -> List[str]:
    if not OUTPUTS_DIR.exists():
        return []
    return sorted([path.name for path in OUTPUTS_DIR.iterdir() if path.is_dir()], reverse=True)


def load_existing_scene(job_id: str) -> Tuple[str, str]:
    if not job_id:
        return viewer_html(None), "Select a job to load."
    export_dir = OUTPUTS_DIR / job_id / "export"
    scene_file = resolve_viewer_scene(export_dir)
    if scene_file is None:
        return viewer_html(None), "No exported scene found for this job."
    return viewer_html(scene_file), f"Loaded {scene_file.name}."


def load_uploaded_scene(file: Optional[str]) -> Tuple[str, str]:
    if not file:
        return viewer_html(None), "Upload a .splat or .ply file."
    job_dir = OUTPUTS_DIR / f"external_{uuid4().hex[:6]}"
    export_dir = job_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    dest = export_dir / Path(file).name
    shutil.copy(file, dest)
    return viewer_html(dest), f"Loaded {dest.name}."


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Nerfstudio Splatfacto Tool") as demo:
        gr.Markdown(
            "# Nerfstudio Splatfacto Scene Builder\n"
            "Upload multi-image captures, run the Splatfacto pipeline, and explore the output."
        )
        with gr.Tab("Create Scene (Multi-image)"):
            with gr.Row():
                zip_input = gr.File(label="Upload ZIP of images", file_types=[".zip"], type="filepath")
                images_input = gr.Files(
                    label="Or select multiple images",
                    file_types=[".jpg", ".jpeg", ".png", ".webp"],
                    type="filepath",
                )
            with gr.Row():
                iterations = gr.Slider(
                    minimum=500,
                    maximum=10000,
                    step=100,
                    value=DEFAULT_ITERATIONS,
                    label="Max training iterations",
                )
                downscale = gr.Slider(
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=DEFAULT_DOWNSCALE,
                    label="Downscale factor",
                )
                device_mode = gr.Dropdown(
                    choices=[("Auto", "auto"), ("CPU only", "cpu")],
                    value="auto",
                    label="GPU/CPU mode",
                )
            with gr.Row():
                run_btn = gr.Button("Run Pipeline", variant="primary")
                cancel_btn = gr.Button("Cancel", variant="stop")
            status = gr.Markdown("Idle.")
            viewer = gr.HTML(viewer_html(None))
            logs = gr.Textbox(label="Logs", lines=12, value="", interactive=False)
            with gr.Row():
                download_scene = gr.File(label="Download scene file")
                download_bundle = gr.File(label="Download job bundle")

            run_btn.click(
                run_pipeline,
                inputs=[zip_input, images_input, iterations, downscale, device_mode],
                outputs=[logs, status, viewer, download_scene, download_bundle],
            )
            cancel_btn.click(cancel_pipeline, outputs=status)

        with gr.Tab("Load / Explore Existing Scene"):
            existing_jobs = gr.Dropdown(label="Existing output jobs", choices=list_output_jobs())
            refresh_jobs = gr.Button("Refresh job list")
            load_job_btn = gr.Button("Load selected job")
            external_upload = gr.File(label="Upload .splat or .ply file", file_types=[".splat", ".ply"])
            load_external_btn = gr.Button("Load uploaded scene")
            load_status = gr.Markdown("Select a job or upload a file.")
            viewer_existing = gr.HTML(viewer_html(None))

            refresh_jobs.click(
                lambda: gr.Dropdown.update(choices=list_output_jobs()),
                outputs=existing_jobs,
            )
            load_job_btn.click(load_existing_scene, inputs=existing_jobs, outputs=[viewer_existing, load_status])
            load_external_btn.click(load_uploaded_scene, inputs=external_upload, outputs=[viewer_existing, load_status])

    return demo


def main() -> None:
    ensure_directories()
    demo = build_ui()
    demo.launch(allowed_paths=[str(OUTPUTS_DIR), str(STATIC_DIR)], show_error=True)


if __name__ == "__main__":
    main()
