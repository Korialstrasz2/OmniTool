import contextlib
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple
from urllib.parse import quote
from urllib.request import urlretrieve

import gradio as gr

BASE_DIR = Path(__file__).resolve().parent
INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
THIRD_PARTY_DIR = BASE_DIR / "third_party"
COLMAP_DIR = THIRD_PARTY_DIR / "colmap"
COLMAP_BIN_DIR = COLMAP_DIR / "bin"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
VIEWER_HTML = STATIC_DIR / "viewer" / "index.html"

COLMAP_WINDOWS_URL = (
    "https://github.com/colmap/colmap/releases/download/3.9/colmap-x64-windows.zip"
)

RUN_LOCK = threading.Lock()
RUN_STATE = {
    "process": None,
    "cancel_requested": False,
}


def ensure_directories() -> None:
    for path in (INPUTS_DIR, OUTPUTS_DIR, MODELS_DIR, STATIC_DIR, COLMAP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def in_virtualenv() -> bool:
    return bool(os.environ.get("VIRTUAL_ENV") or (hasattr(sys, "real_prefix")))


def ensure_venv_and_requirements() -> None:
    if in_virtualenv():
        ensure_requirements_installed()
        return

    venv_dir = BASE_DIR / ".venv"
    if not venv_dir.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    python_exe = venv_dir / "Scripts" / "python.exe"
    pip_exe = venv_dir / "Scripts" / "pip.exe"
    if not python_exe.exists():
        raise RuntimeError("Virtual environment python.exe not found.")

    subprocess.check_call([str(pip_exe), "install", "--upgrade", "pip"])
    subprocess.check_call([str(pip_exe), "install", "-r", str(REQUIREMENTS_FILE)])
    os.execv(str(python_exe), [str(python_exe), str(__file__), *sys.argv[1:]])


def ensure_requirements_installed() -> None:
    required = ["gradio", "nerfstudio", "torch"]
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])


def check_python_version() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10+ is required.")


def check_torch() -> Tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "Torch not installed."
    if torch.cuda.is_available():
        return True, f"CUDA available: {torch.cuda.get_device_name(0)}"
    return False, "CUDA not available. Running in CPU mode."


def ensure_colmap() -> None:
    if COLMAP_BIN_DIR.exists() and any(COLMAP_BIN_DIR.glob("colmap*")):
        return
    COLMAP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = COLMAP_DIR / "colmap.zip"
    urlretrieve(COLMAP_WINDOWS_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(COLMAP_DIR)
    zip_path.unlink(missing_ok=True)

    nested_bin = next(COLMAP_DIR.glob("**/bin"), None)
    if nested_bin and nested_bin != COLMAP_BIN_DIR:
        COLMAP_BIN_DIR.mkdir(parents=True, exist_ok=True)
        for item in nested_bin.iterdir():
            shutil.move(str(item), COLMAP_BIN_DIR / item.name)


def ensure_nerfstudio_cli() -> None:
    if shutil.which("ns-train"):
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nerfstudio"])


def check_dependencies() -> Dict[str, str]:
    check_python_version()
    ensure_venv_and_requirements()
    ensure_colmap()
    ensure_nerfstudio_cli()
    cuda_available, cuda_message = check_torch()
    return {
        "cuda_available": str(cuda_available),
        "cuda_message": cuda_message,
    }


def update_env_for_colmap() -> Dict[str, str]:
    env = os.environ.copy()
    env_path = env.get("PATH", "")
    colmap_path = str(COLMAP_BIN_DIR)
    if colmap_path not in env_path:
        env["PATH"] = colmap_path + os.pathsep + env_path
    return env


def resolve_device_mode(mode: str) -> str:
    if mode != "Auto":
        return mode
    try:
        import torch

        return "GPU" if torch.cuda.is_available() else "CPU"
    except ImportError:
        return "CPU"


def create_job_id() -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    return f"{timestamp}_{short_id}"


def normalize_inputs(zip_file: Optional[str], images: Optional[List[str]]) -> Path:
    job_id = create_job_id()
    job_input_dir = INPUTS_DIR / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)

    if zip_file is not None:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(job_input_dir)
    elif images:
        for image in images:
            dest = job_input_dir / Path(image).name
            shutil.copy(image, dest)
    else:
        raise ValueError("Please upload a zip file or multiple images.")

    return job_input_dir


def save_settings(output_dir: Path, settings: Dict[str, str]) -> None:
    with open(output_dir / "settings.json", "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)


def make_job_bundle(output_dir: Path, input_dir: Path) -> Path:
    bundle_path = output_dir / "job_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zip_ref:
        for folder in (input_dir, output_dir):
            for path in folder.rglob("*"):
                if path.is_file():
                    zip_ref.write(path, path.relative_to(folder.parent))
    return bundle_path


def write_logs(output_dir: Path, logs: List[str]) -> None:
    with open(output_dir / "logs.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(logs))


def copy_preview_image(input_dir: Path, output_dir: Path) -> None:
    images = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    if images:
        preview = output_dir / f"preview{images[0].suffix}"
        shutil.copy(images[0], preview)


def build_viewer_html(scene_path: Optional[Path]) -> str:
    if scene_path is None or not scene_path.exists():
        return "<div class='viewer-placeholder'>No scene loaded.</div>"
    scene_url = f"/file={scene_path.resolve()}"
    viewer_url = f"/file={VIEWER_HTML.resolve()}?scene={quote(scene_url)}"
    return (
        "<iframe "
        "src=\"{viewer_url}\" "
        "style=\"width: 100%; height: 520px; border: none; border-radius: 8px;\" "
        "allow=\"fullscreen\"></iframe>"
    ).format(viewer_url=viewer_url)


def run_subprocess(command: List[str], cwd: Optional[Path], env: Dict[str, str]) -> Iterable[str]:
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    RUN_STATE["process"] = process
    for line in iter(process.stdout.readline, ""):
        if RUN_STATE["cancel_requested"]:
            terminate_process_tree(process)
            break
        yield line.rstrip()
    process.stdout.close()
    return_code = process.wait()
    RUN_STATE["process"] = None
    if return_code != 0:
        yield f"Command failed with exit code {return_code}."


def terminate_process_tree(process: subprocess.Popen) -> None:
    with contextlib.suppress(Exception):
        import psutil

        proc = psutil.Process(process.pid)
        for child in proc.children(recursive=True):
            child.terminate()
        proc.terminate()
    with contextlib.suppress(Exception):
        process.terminate()


def find_latest_config(train_dir: Path) -> Optional[Path]:
    configs = list(train_dir.rglob("config.yml"))
    if not configs:
        return None
    return max(configs, key=lambda path: path.stat().st_mtime)


def run_pipeline(
    zip_file: Optional[str],
    images: Optional[List[str]],
    iterations: int,
    downscale: int,
    device_mode: str,
    progress=gr.Progress(track_tqdm=False),
) -> Generator[Tuple[str, str, str, Optional[str], Optional[str]], None, None]:
    if RUN_LOCK.locked():
        yield "Busy", "Another job is currently running.", build_viewer_html(None), None, None
        return

    with RUN_LOCK:
        RUN_STATE["cancel_requested"] = False
        logs: List[str] = []
        status = "Preparing job"
        yield status, "\n".join(logs), build_viewer_html(None), None, None

        requested_mode = device_mode
        device_mode = resolve_device_mode(device_mode)
        iterations = max(1, int(iterations))
        downscale = max(1, int(downscale))

        if requested_mode == "GPU" and device_mode == "CPU":
            logs.append("GPU requested but CUDA is unavailable. Falling back to CPU mode.")

        job_input_dir = normalize_inputs(zip_file, images)
        job_id = job_input_dir.name
        job_output_dir = OUTPUTS_DIR / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        proc_dir = job_output_dir / "processed"
        train_dir = job_output_dir / "train"
        export_dir = job_output_dir / "export"
        proc_dir.mkdir(parents=True, exist_ok=True)
        train_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        settings = {
            "job_id": job_id,
            "iterations": str(iterations),
            "downscale_factor": str(downscale),
            "device_mode": device_mode,
        }
        save_settings(job_output_dir, settings)
        copy_preview_image(job_input_dir, job_output_dir)

        env = update_env_for_colmap()
        if device_mode == "CPU":
            env["CUDA_VISIBLE_DEVICES"] = ""

        commands = [
            [
                "ns-process-data",
                "images",
                "--data",
                str(job_input_dir),
                "--output-dir",
                str(proc_dir),
                "--downscale-factor",
                str(downscale),
            ],
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
        ]
        if device_mode == "CPU":
            commands[1].extend(["--pipeline.device", "cpu"])

        export_config = find_latest_config(train_dir)
        if not export_config:
            logs.append("Unable to locate training config.yml. Export skipped.")
            yield "Failed", "\n".join(logs), build_viewer_html(None), None, None
            return
        export_command = [
            "ns-export",
            "gaussian-splat",
            "--load-config",
            str(export_config),
            "--output-dir",
            str(export_dir),
            "--output-filename",
            "scene.splat",
        ]

        for command in commands + [export_command]:
            if RUN_STATE["cancel_requested"]:
                status = "Cancelled"
                logs.append("Job cancelled.")
                write_logs(job_output_dir, logs)
                yield status, "\n".join(logs), build_viewer_html(None), None, None
                return
            status = f"Running: {' '.join(command[:2])}"
            progress(0, desc=status)
            for line in run_subprocess(command, cwd=BASE_DIR, env=env):
                logs.append(line)
                yield status, "\n".join(logs), build_viewer_html(None), None, None
            logs.append(f"Completed: {' '.join(command[:2])}")
            write_logs(job_output_dir, logs)

        scene_file = export_dir / "scene.splat"
        bundle = make_job_bundle(job_output_dir, job_input_dir)
        status = "Completed"
        viewer_html = build_viewer_html(scene_file)
        write_logs(job_output_dir, logs)
        yield status, "\n".join(logs), viewer_html, str(scene_file), str(bundle)


def cancel_run() -> Tuple[str, str]:
    RUN_STATE["cancel_requested"] = True
    process = RUN_STATE.get("process")
    if process:
        terminate_process_tree(process)
    return "Cancelled", "Cancellation requested."


def list_jobs() -> List[str]:
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(
        [p.name for p in OUTPUTS_DIR.iterdir() if p.is_dir() and p.name != "external"]
    )


def refresh_jobs() -> gr.Dropdown:
    return gr.Dropdown.update(choices=list_jobs())


def load_job_scene(job_id: str) -> Tuple[str, Optional[str]]:
    if not job_id:
        return build_viewer_html(None), None
    export_dir = OUTPUTS_DIR / job_id / "export"
    scene_path = export_dir / "scene.splat"
    if not scene_path.exists():
        ply_path = export_dir / "scene.ply"
        if ply_path.exists():
            scene_path = ply_path
        else:
            return build_viewer_html(None), None
    return build_viewer_html(scene_path), str(scene_path)


def load_external_scene(file: Optional[str]) -> Tuple[str, Optional[str]]:
    if file is None:
        return build_viewer_html(None), None
    job_dir = OUTPUTS_DIR / "external" / uuid.uuid4().hex[:8]
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / Path(file).name
    shutil.copy(file, dest)
    return build_viewer_html(dest), str(dest)


def build_ui() -> gr.Blocks:
    jobs = list_jobs()
    with gr.Blocks(title="Gaussian Splat Scene Builder") as demo:
        gr.Markdown(
            "# Gaussian Splat Scene Builder\n"
            "Create and explore 3D Gaussian Splat scenes using Nerfstudio Splatfacto."
        )

        with gr.Tab("Create Scene (Multi-image)"):
            with gr.Row():
                zip_upload = gr.File(label="Upload ZIP of images", file_types=[".zip"], type="filepath")
                image_upload = gr.File(
                    label="Or upload multiple images",
                    file_types=["image"],
                    file_count="multiple",
                    type="filepath",
                )
            with gr.Row():
                iterations = gr.Number(label="Max iterations", value=1000, precision=0)
                downscale = gr.Number(label="Downscale factor", value=2, precision=0)
                device_mode = gr.Radio(["Auto", "GPU", "CPU"], value="Auto", label="Device")
            with gr.Row():
                run_btn = gr.Button("Run", variant="primary")
                cancel_btn = gr.Button("Cancel", variant="stop")

            status_text = gr.Textbox(label="Status", value="Idle", interactive=False)
            logs = gr.Textbox(label="Logs", lines=15, interactive=False)
            viewer = gr.HTML(value=build_viewer_html(None), label="Viewer")
            with gr.Row():
                download_scene = gr.File(label="Download scene", interactive=False)
                download_bundle = gr.File(label="Download job bundle", interactive=False)

            run_btn.click(
                run_pipeline,
                inputs=[zip_upload, image_upload, iterations, downscale, device_mode],
                outputs=[status_text, logs, viewer, download_scene, download_bundle],
            )
            cancel_btn.click(cancel_run, outputs=[status_text, logs])

        with gr.Tab("Load / Explore Existing Scene"):
            with gr.Row():
                job_dropdown = gr.Dropdown(choices=jobs, label="Existing jobs")
                refresh_btn = gr.Button("Refresh jobs")
            external_scene = gr.File(
                label="Load external .splat/.ply",
                file_types=[".splat", ".ply"],
                type="filepath",
            )
            viewer_existing = gr.HTML(value=build_viewer_html(None))
            download_existing = gr.File(label="Download scene", interactive=False)

            job_dropdown.change(load_job_scene, inputs=job_dropdown, outputs=[viewer_existing, download_existing])
            external_scene.change(load_external_scene, inputs=external_scene, outputs=[viewer_existing, download_existing])
            refresh_btn.click(refresh_jobs, outputs=job_dropdown)

        with gr.Tab("Single Image (Limited)"):
            gr.Markdown(
                "Single-image reconstruction is not enabled in this build. "
                "Use the multi-image pipeline for full 3D scenes."
            )

    return demo


def main() -> None:
    ensure_directories()
    dep_status = check_dependencies()
    print(f"Dependency check: {dep_status['cuda_message']}")
    app = build_ui()
    app.queue()
    app.launch(
        allowed_paths=[str(OUTPUTS_DIR), str(INPUTS_DIR), str(STATIC_DIR)],
        server_name="0.0.0.0",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
