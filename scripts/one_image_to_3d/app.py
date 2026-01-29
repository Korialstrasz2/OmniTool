import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ENGINE_SPECS = {
    "Depth Anything V2 Small (fast)": {
        "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "description": "Fast previews with solid geometry.",
    },
    "Depth Anything V2 Large (high detail)": {
        "model_id": "depth-anything/Depth-Anything-V2-Large-hf",
        "description": "Sharper depth boundaries and improved fine detail.",
    },
    "ZoeDepth NYU/KITTI (metric)": {
        "model_id": "Intel/zoedepth-nyu-kitti",
        "description": "Metric-aware depth with strong multi-surface consistency.",
    },
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    min_val = float(depth.min())
    max_val = float(depth.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(depth)
    return (depth - min_val) / (max_val - min_val)


@lru_cache(maxsize=len(ENGINE_SPECS))
def load_depth_model(model_id: str) -> Tuple[AutoImageProcessor, AutoModelForDepthEstimation, str]:
    device = pick_device()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model, device


def estimate_depth(image: Image.Image, model_id: str) -> np.ndarray:
    processor, model, device = load_depth_model(model_id)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return depth.cpu().numpy()


def build_mesh(
    image: Image.Image,
    depth: np.ndarray,
    depth_scale: float,
    xy_scale: float,
) -> trimesh.Trimesh:
    height, width = depth.shape
    xs = (np.arange(width) - width / 2) / width * xy_scale
    ys = (np.arange(height) - height / 2) / height * xy_scale
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_z = depth * depth_scale

    vertices = np.column_stack(
        [
            grid_x.reshape(-1),
            -grid_y.reshape(-1),
            grid_z.reshape(-1),
        ]
    )

    color_image = image.resize((width, height), Image.Resampling.LANCZOS)
    colors = np.asarray(color_image).reshape(-1, 3)

    index_grid = np.arange(height * width).reshape(height, width)
    faces_a = np.stack(
        [
            index_grid[:-1, :-1].ravel(),
            index_grid[1:, :-1].ravel(),
            index_grid[1:, 1:].ravel(),
        ],
        axis=1,
    )
    faces_b = np.stack(
        [
            index_grid[:-1, :-1].ravel(),
            index_grid[1:, 1:].ravel(),
            index_grid[:-1, 1:].ravel(),
        ],
        axis=1,
    )
    faces = np.concatenate([faces_a, faces_b], axis=0)

    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)


def ensure_model_assets() -> None:
    """Download all configured models at startup so the UI stays responsive."""
    for engine in ENGINE_SPECS.values():
        model_id = engine["model_id"]
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        device = pick_device()
        model.to(device)
        model.eval()
        del processor, model


def load_images_from_paths(paths: Iterable[str], max_resolution: int) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if max(image.size) > max_resolution:
            image.thumbnail((max_resolution, max_resolution), Image.Resampling.LANCZOS)
        images.append(image)
    return images


def parse_view_angles(view_angles: str, count: int) -> List[float]:
    cleaned = [chunk.strip() for chunk in view_angles.split(",") if chunk.strip()]
    if cleaned:
        angles = []
        for angle in cleaned:
            try:
                angles.append(float(angle))
            except ValueError:
                continue
        if angles:
            return angles[:count]
    if count <= 0:
        return []
    step = 360.0 / count
    return [i * step for i in range(count)]


def rotate_and_offset_mesh(mesh: trimesh.Trimesh, angle_deg: float, orbit_radius: float) -> trimesh.Trimesh:
    angle_rad = np.deg2rad(angle_deg)
    rotation = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])
    offset = trimesh.transformations.translation_matrix(
        [orbit_radius * np.sin(angle_rad), 0.0, orbit_radius * np.cos(angle_rad)]
    )
    transformed = mesh.copy()
    transformed.apply_transform(offset @ rotation)
    return transformed


def generate_scene(
    input_image: Image.Image,
    multi_images: List[str],
    mode: str,
    engine: str,
    max_resolution: int,
    depth_scale: float,
    xy_scale: float,
    orbit_radius: float,
    view_angles: str,
    include_depth_preview: bool,
):
    if mode == "single":
        if input_image is None:
            raise gr.Error("Please upload an image first.")
        images = [input_image.convert("RGB")]
    else:
        if not multi_images:
            raise gr.Error("Please upload at least two images for multi-view.")
        images = load_images_from_paths(multi_images, max_resolution)

    model_id = ENGINE_SPECS[engine]["model_id"]
    meshes = []
    depth_preview = None
    for idx, image in enumerate(images):
        if max(image.size) > max_resolution:
            image.thumbnail((max_resolution, max_resolution), Image.Resampling.LANCZOS)
        depth = estimate_depth(image, model_id)
        depth_norm = normalize_depth(depth)
        mesh = build_mesh(image, depth_norm, depth_scale, xy_scale)
        meshes.append(mesh)
        if include_depth_preview and depth_preview is None:
            depth_image = Image.fromarray((depth_norm * 255).astype(np.uint8))
            depth_preview = depth_image.convert("L")

    if mode == "multi":
        angles = parse_view_angles(view_angles, len(meshes))
        transformed_meshes = []
        for mesh, angle in zip(meshes, angles):
            transformed_meshes.append(rotate_and_offset_mesh(mesh, angle, orbit_radius))
        mesh = trimesh.util.concatenate(transformed_meshes)
    else:
        mesh = meshes[0]

    output_dir = Path(tempfile.mkdtemp(prefix="one-image-3d-"))
    mesh_path = output_dir / "scene.glb"
    mesh.export(mesh_path)

    return str(mesh_path), depth_preview, str(mesh_path)


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # One Image → Explorable 3D Scene

            This tool estimates depth from a single image and converts it into a 3D mesh that you can orbit and inspect.
            """
        )
        with gr.Row():
            with gr.Column():
                mode = gr.Radio(
                    ["single", "multi"],
                    value="single",
                    label="Scene input mode",
                    info="Single image for quick depth or multi-view to blend angles.",
                )
                input_image = gr.Image(label="Single input image", type="pil")
                multi_images = gr.Files(
                    label="Multi-view images",
                    file_types=["image"],
                    type="filepath",
                )
                engine = gr.Dropdown(
                    list(ENGINE_SPECS.keys()),
                    value="Depth Anything V2 Large (high detail)",
                    label="Depth engine",
                    info="Choose the strongest available depth backbone.",
                )
                max_resolution = gr.Slider(
                    256,
                    2048,
                    value=1024,
                    step=32,
                    label="Max resolution (higher = sharper)",
                )
                depth_scale = gr.Slider(
                    0.5,
                    4.0,
                    value=2.6,
                    step=0.1,
                    label="Depth scale",
                )
                xy_scale = gr.Slider(
                    0.5,
                    4.0,
                    value=2.4,
                    step=0.1,
                    label="XY scale",
                )
                orbit_radius = gr.Slider(
                    0.0,
                    4.0,
                    value=1.2,
                    step=0.1,
                    label="Multi-view orbit radius",
                )
                view_angles = gr.Textbox(
                    label="View angles (degrees, comma-separated)",
                    placeholder="0, 90, 180, 270 (leave blank to auto-space)",
                )
                include_depth_preview = gr.Checkbox(
                    value=False,
                    label="Generate depth preview",
                    info="Disable to save time; meshes still render.",
                )
                generate_button = gr.Button("Generate 3D Scene")
            with gr.Column():
                model_view = gr.Model3D(label="Explorable scene")
                depth_preview = gr.Image(label="Depth preview", type="pil")
                download_mesh = gr.File(label="Download mesh")

        generate_button.click(
            generate_scene,
            inputs=[
                input_image,
                multi_images,
                mode,
                engine,
                max_resolution,
                depth_scale,
                xy_scale,
                orbit_radius,
                view_angles,
                include_depth_preview,
            ],
            outputs=[model_view, depth_preview, download_mesh],
        )
    return demo


if __name__ == "__main__":
    ensure_model_assets()
    build_demo().launch()
