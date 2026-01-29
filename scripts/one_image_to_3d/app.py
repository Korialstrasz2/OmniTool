import os
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
    "Depth Anything V2 Giant (ultra detail)": {
        "model_id": "depth-anything/Depth-Anything-V2-Giant-hf",
        "description": "Highest-detail Depth Anything backbone for max fidelity meshes.",
    },
    "Depth Anything V2 Large (high detail)": {
        "model_id": "depth-anything/Depth-Anything-V2-Large-hf",
        "description": "Sharper depth boundaries and improved fine detail.",
    },
    "MiDaS DPT Large (balanced)": {
        "model_id": "Intel/dpt-large",
        "description": "Strong all-around depth with stable surfaces.",
    },
    "MiDaS DPT Hybrid (detail boost)": {
        "model_id": "Intel/dpt-hybrid-midas",
        "description": "Sharper edges with reasonable speed for higher quality meshes.",
    },
    "ZoeDepth NYU/KITTI (metric)": {
        "model_id": "Intel/zoedepth-nyu-kitti",
        "description": "Metric-aware depth with strong multi-surface consistency.",
    },
    "Apple SHARP (external depth map)": {
        "model_id": "",
        "description": "Use Apple SHARP depth maps exported externally (Windows).",
        "external_only": True,
    },
    "Custom / external model ID": {
        "model_id": "",
        "description": "Use any compatible depth model ID or external depth maps.",
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
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
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


def load_depth_map_image(path: str, size: Tuple[int, int]) -> np.ndarray:
    depth_image = Image.open(path).convert("L")
    if depth_image.size != size:
        depth_image = depth_image.resize(size, Image.Resampling.BILINEAR)
    depth_array = np.asarray(depth_image).astype(np.float32)
    return normalize_depth(depth_array)


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
        if engine.get("external_only"):
            continue
        if not model_id:
            continue
        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
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


def match_image_size(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    if image.size != size:
        return image.resize(size, Image.Resampling.LANCZOS)
    return image


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
    custom_model_id: str,
    depth_source: str,
    depth_maps: List[str],
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

    engine_spec = ENGINE_SPECS[engine]
    model_id = engine_spec["model_id"]
    if engine_spec.get("external_only"):
        depth_source = "external"
    if engine == "Custom / external model ID":
        if not custom_model_id.strip():
            if depth_source == "model":
                raise gr.Error("Enter a custom model ID or switch to external depth maps.")
        else:
            model_id = custom_model_id.strip()

    depth_paths: List[str] = depth_maps or []
    if depth_source == "external":
        if mode == "single" and not depth_paths:
            raise gr.Error("Upload a depth map to use the external depth option.")
        if mode != "single" and len(depth_paths) != len(images):
            raise gr.Error("Provide one depth map per image for external depth blending.")
    if depth_source == "model" and not model_id:
        raise gr.Error("Select a model-based engine or provide a custom model ID.")
    meshes = []
    depth_preview = None
    depth_stack: List[np.ndarray] = []
    color_stack: List[np.ndarray] = []
    target_size = images[0].size
    for idx, image in enumerate(images):
        if max(image.size) > max_resolution:
            image.thumbnail((max_resolution, max_resolution), Image.Resampling.LANCZOS)
        image = match_image_size(image, target_size)
        if depth_source == "external":
            depth_norm = load_depth_map_image(depth_paths[idx], image.size)
        else:
            depth = estimate_depth(image, model_id)
            depth_norm = normalize_depth(depth)
        if mode == "multi_merge":
            depth_stack.append(depth_norm)
            color_stack.append(np.asarray(image))
        else:
            mesh = build_mesh(image, depth_norm, depth_scale, xy_scale)
            meshes.append(mesh)
        if include_depth_preview and depth_preview is None:
            depth_image = Image.fromarray((depth_norm * 255).astype(np.uint8))
            depth_preview = depth_image.convert("L")

    if mode == "multi_merge":
        depth_merged = np.median(np.stack(depth_stack, axis=0), axis=0)
        color_merged = np.median(np.stack(color_stack, axis=0), axis=0).astype(np.uint8)
        merged_image = Image.fromarray(color_merged)
        mesh = build_mesh(merged_image, depth_merged, depth_scale, xy_scale)
    elif mode == "multi_orbit":
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

    depth_output = gr.update(value=depth_preview, visible=include_depth_preview)
    return str(mesh_path), depth_output, str(mesh_path)


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # One Image → Explorable 3D Scene

            This tool estimates depth from a single image (or merges multiple images) and converts it into a 3D mesh
            that you can orbit and inspect. Use high-resolution settings and external depth maps (Apple SHARP, etc.)
            when you need maximum quality. Models load on demand; set `ONE_IMAGE_3D_PRELOAD_MODELS=1` to prefetch.
            """
        )
        with gr.Row():
            with gr.Column():
                mode = gr.Radio(
                    ["single", "multi_orbit", "multi_merge"],
                    value="single",
                    label="Scene input mode",
                    info="Single image, multi-view orbit, or multi-image merge for detail boost.",
                )
                input_image = gr.Image(label="Single input image", type="pil")
                multi_images = gr.Files(
                    label="Multi-view images (orbit or merge)",
                    file_types=["image"],
                    type="filepath",
                )
                engine = gr.Dropdown(
                    list(ENGINE_SPECS.keys()),
                    value="Depth Anything V2 Large (high detail)",
                    label="Depth engine",
                    info="Choose a built-in model or supply a custom model ID.",
                )
                custom_model_id = gr.Textbox(
                    label="Custom model ID (optional)",
                    placeholder="apple/sharp-depth-v1 or org/model-id",
                )
                depth_source = gr.Radio(
                    ["model", "external"],
                    value="model",
                    label="Depth source",
                    info="Model = built-in/Custom model ID. External = upload depth maps (Apple SHARP, etc.).",
                )
                depth_maps = gr.Files(
                    label="External depth maps (optional, match image order)",
                    file_types=["image"],
                    type="filepath",
                )
                max_resolution = gr.Slider(
                    256,
                    10240,
                    value=2048,
                    step=64,
                    label="Max resolution (higher = sharper, up to 5x)",
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
            with gr.Column(scale=2):
                model_view = gr.Model3D(label="Explorable scene", height=720)
                download_mesh = gr.File(label="Download mesh")
                with gr.Accordion("Depth preview (optional)", open=False):
                    depth_preview = gr.Image(label="Depth preview", type="pil", visible=False)

        generate_button.click(
            generate_scene,
            inputs=[
                input_image,
                multi_images,
                mode,
                engine,
                custom_model_id,
                depth_source,
                depth_maps,
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
    if os.environ.get("ONE_IMAGE_3D_PRELOAD_MODELS") == "1":
        ensure_model_assets()
    build_demo().launch()
