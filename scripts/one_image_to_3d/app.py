import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


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


@lru_cache(maxsize=1)
def load_depth_model() -> Tuple[AutoImageProcessor, AutoModelForDepthEstimation, str]:
    device = pick_device()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model, device


def estimate_depth(image: Image.Image) -> np.ndarray:
    processor, model, device = load_depth_model()
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


def generate_scene(
    input_image: Image.Image,
    max_resolution: int,
    depth_scale: float,
    xy_scale: float,
):
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    image = input_image.convert("RGB")
    if max(image.size) > max_resolution:
        image.thumbnail((max_resolution, max_resolution), Image.Resampling.LANCZOS)

    depth = estimate_depth(image)
    depth_norm = normalize_depth(depth)
    mesh = build_mesh(image, depth_norm, depth_scale, xy_scale)

    output_dir = Path(tempfile.mkdtemp(prefix="one-image-3d-"))
    mesh_path = output_dir / "scene.glb"
    mesh.export(mesh_path)

    depth_image = Image.fromarray((depth_norm * 255).astype(np.uint8))
    depth_image = depth_image.convert("L")
    return str(mesh_path), depth_image, str(mesh_path)


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
                input_image = gr.Image(label="Input image", type="pil")
                max_resolution = gr.Slider(
                    128,
                    512,
                    value=320,
                    step=16,
                    label="Max resolution (lower = faster)",
                )
                depth_scale = gr.Slider(
                    0.5,
                    4.0,
                    value=2.0,
                    step=0.1,
                    label="Depth scale",
                )
                xy_scale = gr.Slider(
                    0.5,
                    4.0,
                    value=2.0,
                    step=0.1,
                    label="XY scale",
                )
                generate_button = gr.Button("Generate 3D Scene")
            with gr.Column():
                model_view = gr.Model3D(label="Explorable scene")
                depth_preview = gr.Image(label="Depth preview", type="pil")
                download_mesh = gr.File(label="Download mesh")

        generate_button.click(
            generate_scene,
            inputs=[input_image, max_resolution, depth_scale, xy_scale],
            outputs=[model_view, depth_preview, download_mesh],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
