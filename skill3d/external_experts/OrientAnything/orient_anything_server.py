import argparse
import base64
import importlib
import io
import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image, ImageDraw


app = Flask(__name__)

repo_dir = None
device = None
ckpt_path = None
weights_dir = None

loaded_repo_dir = None
model = None
val_preprocess = None
DINOv2_MLP = None
background_preprocess = None
output_dim = None


def _default_device(device_name: Optional[str] = None) -> str:
    if device_name:
        return device_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_repo_loaded(base_dir: str):
    global loaded_repo_dir
    global DINOv2_MLP
    global background_preprocess
    global val_preprocess

    resolved_base_dir = str(Path(base_dir).expanduser().resolve())

    if loaded_repo_dir == resolved_base_dir:
        return

    if resolved_base_dir not in sys.path:
        sys.path.insert(0, resolved_base_dir)

    old_cwd = os.getcwd()
    try:
        os.chdir(resolved_base_dir)
        vision_module = importlib.import_module("vision_tower")
        utils_module = importlib.import_module("utils")
        paths_module = importlib.import_module("paths")
        from transformers import AutoImageProcessor

        DINOv2_MLP = vision_module.DINOv2_MLP
        background_preprocess = utils_module.background_preprocess
        dino_large = getattr(paths_module, "DINO_LARGE")
        val_preprocess = AutoImageProcessor.from_pretrained(dino_large, cache_dir="./")
    finally:
        os.chdir(old_cwd)

    loaded_repo_dir = resolved_base_dir


def _resolve_checkpoint_path(explicit_path: Optional[str]) -> str:
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    candidates = []
    if ckpt_path:
        candidates.append(ckpt_path)
    if weights_dir:
        candidates.extend(sorted(glob(os.path.join(weights_dir, "*.pt"))))
        candidates.extend(sorted(glob(os.path.join(weights_dir, "*.pth"))))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Orient-Anything checkpoint not found. "
        f"Tried explicit path {explicit_path!r}, ckpt_path={ckpt_path!r}, weights_dir={weights_dir!r}."
    )


def _load_model(base_dir: str, explicit_ckpt_path: Optional[str] = None):
    global model, output_dim
    if model is not None:
        return model

    _ensure_repo_loaded(base_dir)
    resolved_ckpt = _resolve_checkpoint_path(explicit_ckpt_path)
    state = torch.load(resolved_ckpt, map_location="cpu")

    inferred_out_dim = None
    probe_weight = state.get("down_sampler.net1.0.weight")
    if probe_weight is not None and hasattr(probe_weight, "shape") and len(probe_weight.shape) >= 1:
        inferred_out_dim = int(probe_weight.shape[0])
    if inferred_out_dim is None:
        raise RuntimeError("Failed to infer Orient-Anything output dimension from checkpoint.")

    output_dim = inferred_out_dim

    model_instance = DINOv2_MLP(
        dino_mode="large",
        in_dim=1024,
        out_dim=output_dim,
        evaluate=True,
        mask_dino=False,
        frozen_back=False,
    )
    model_instance.load_state_dict(state)
    model_instance.eval()
    model_instance = model_instance.to(device)
    model = model_instance
    app.logger.info("Loaded Orient-Anything v1 model from %s with out_dim=%s", resolved_ckpt, output_dim)
    return model


def _decode_pil_image(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _image_to_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _wrap_angle(value: float, low: float, high: float) -> float:
    width = high - low
    while value < low:
        value += width
    while value >= high:
        value -= width
    return value


def _normalize_crop_box(crop_box, image_size):
    if crop_box is None:
        raise ValueError("crop_box is required for Orient-Anything inference.")
    if len(crop_box) != 4:
        raise ValueError("crop_box must contain exactly 4 values: [x1, y1, x2, y2]")

    width, height = image_size
    x1, y1, x2, y2 = [int(round(float(value))) for value in crop_box]
    left = max(0, min(x1, x2))
    top = max(0, min(y1, y2))
    right = min(width, max(x1, x2))
    bottom = min(height, max(y1, y2))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop_box after normalization: {[left, top, right, bottom]}")
    return [left, top, right, bottom]


def _crop_image(image: Image.Image, crop_box):
    normalized = _normalize_crop_box(crop_box, image.size)
    cropped = image.crop(tuple(normalized))
    return cropped, normalized


def _predict_pose(image: Image.Image, remove_background: bool, infer_aug: bool):
    prepared = background_preprocess(image, remove_background)
    images = [prepared]
    if infer_aug:
        utils_module = importlib.import_module("utils")
        images = list(utils_module.get_crop_images(image, num=3)) + list(utils_module.get_crop_images(prepared, num=3))

    image_inputs = val_preprocess(images=images)
    image_inputs["pixel_values"] = torch.from_numpy(np.array(image_inputs["pixel_values"])).to(device)
    with torch.no_grad():
        predictions = model(image_inputs)
    angles = _decode_predictions(predictions)

    return {
        "azimuth": float(angles[0]),
        "elevation": float(angles[1]),
        "rotation": float(angles[2]),
        "confidence": float(angles[3]),
        "processed_image": prepared,
    }


def _decode_predictions(predictions: torch.Tensor) -> torch.Tensor:
    preds = predictions.detach()
    if preds.ndim == 1:
        preds = preds.unsqueeze(0)

    current_out_dim = int(preds.shape[-1])
    if current_out_dim not in {722, 902}:
        raise RuntimeError(f"Unsupported Orient-Anything output dimension: {current_out_dim}")

    rotation_bins = 180 if current_out_dim == 722 else 360
    az_logits = preds[:, 0:360]
    el_logits = preds[:, 360:540]
    rot_logits = preds[:, 540:540 + rotation_bins]
    conf_logits = preds[:, 540 + rotation_bins:540 + rotation_bins + 2]

    az_pred = torch.argmax(az_logits, dim=-1).to(torch.float32)
    el_pred = torch.argmax(el_logits, dim=-1).to(torch.float32) - 90.0
    rot_pred = torch.argmax(rot_logits, dim=-1).to(torch.float32)
    rot_pred = rot_pred - (90.0 if rotation_bins == 180 else 180.0)
    conf_pred = torch.mean(F.softmax(conf_logits, dim=-1), dim=0)[0]

    if preds.shape[0] > 1:
        utils_module = importlib.import_module("utils")
        az_value = float(utils_module.remove_outliers_and_average_circular(az_pred))
        el_value = float(utils_module.remove_outliers_and_average(el_pred))
        rot_value = float(utils_module.remove_outliers_and_average(rot_pred))
    else:
        az_value = float(az_pred[0])
        el_value = float(el_pred[0])
        rot_value = float(rot_pred[0])

    angles = torch.zeros(4)
    angles[0] = az_value
    angles[1] = el_value
    angles[2] = rot_value
    angles[3] = conf_pred
    return angles


def _overlay_pose(image: Image.Image, pose: dict) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    center = (width // 2, height // 2)
    radius = max(30, min(width, height) // 5)

    az = math.radians(float(pose.get("azimuth", 0.0)))
    el = math.radians(float(pose.get("elevation", 0.0)))
    rot = math.radians(float(pose.get("rotation", 0.0)))

    front = (math.cos(az), -math.sin(el) - 0.3 * math.sin(az))
    right = (math.sin(az), -0.6 * math.cos(rot))
    top = (0.4 * math.sin(rot), -math.cos(el))

    axes = [
        (front, (255, 0, 0, 255), "F"),
        (right, (0, 200, 0, 255), "R"),
        (top, (0, 90, 255, 255), "U"),
    ]
    for vec, color, label in axes:
        end = (center[0] + int(radius * vec[0]), center[1] + int(radius * vec[1]))
        draw.line([center, end], fill=color, width=4)
        draw.ellipse((end[0] - 5, end[1] - 5, end[0] + 5, end[1] + 5), fill=color)
        draw.text((end[0] + 6, end[1] + 2), label, fill=color)

    return canvas.convert("RGB")


def _overlay_pose_on_original(image: Image.Image, crop_box, pose: dict) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = crop_box
    draw.rectangle([left, top, right, bottom], outline=(255, 170, 0, 255), width=4)

    width = right - left
    height = bottom - top
    center = (left + width // 2, top + height // 2)
    radius = max(18, min(width, height) // 3)

    az = math.radians(float(pose.get("azimuth", 0.0)))
    el = math.radians(float(pose.get("elevation", 0.0)))
    rot = math.radians(float(pose.get("rotation", 0.0)))

    front = (math.cos(az), -math.sin(el) - 0.3 * math.sin(az))
    right_vec = (math.sin(az), -0.6 * math.cos(rot))
    top_vec = (0.4 * math.sin(rot), -math.cos(el))
    axes = [
        (front, (255, 0, 0, 255), "F"),
        (right_vec, (0, 200, 0, 255), "R"),
        (top_vec, (0, 90, 255, 255), "U"),
    ]
    for vec, color, label in axes:
        end = (center[0] + int(radius * vec[0]), center[1] + int(radius * vec[1]))
        draw.line([center, end], fill=color, width=4)
        draw.ellipse((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4), fill=color)
        draw.text((end[0] + 4, end[1] + 2), label, fill=color)
    return canvas.convert("RGB")


def _compute_relative_pose(ref_pose: dict, tgt_pose: dict) -> dict:
    return {
        "azimuth": _wrap_angle(float(tgt_pose["azimuth"]) - float(ref_pose["azimuth"]), -180.0, 180.0),
        "elevation": float(tgt_pose["elevation"]) - float(ref_pose["elevation"]),
        "rotation": _wrap_angle(float(tgt_pose["rotation"]) - float(ref_pose["rotation"]), -180.0, 180.0),
    }


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy" if model is not None else "starting",
            "model_loaded": model is not None,
            "device": device,
            "repo_dir": repo_dir,
            "weights_dir": weights_dir,
            "renderer_available": False,
            "renderer_backend": "v1_2d_overlay",
        }
    )


@app.route("/test", methods=["GET"])
def test():
    demo_path = os.path.join(repo_dir, "assets", "demo.png")
    if not os.path.exists(demo_path):
        return jsonify({"success": False, "error": f"Demo image not found: {demo_path}"}), 500

    image = Image.open(demo_path).convert("RGB")
    pose = _predict_pose(image, remove_background=True, infer_aug=False)
    return jsonify(
        {
            "success": True,
            "reference_pose": {
                "azimuth": pose["azimuth"],
                "elevation": pose["elevation"],
                "rotation": pose["rotation"],
                "confidence": pose["confidence"],
            },
        }
    )


@app.route("/infer", methods=["POST"])
def infer():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        if "image_ref" not in data:
            return jsonify({"success": False, "error": "Missing image_ref"}), 400

        render_visualization = bool(data.get("render_visualization", False))
        remove_background = bool(data.get("remove_background", True))
        infer_aug = bool(data.get("infer_aug", False))

        ref_image_full = _decode_pil_image(data["image_ref"])
        ref_crop_image, ref_crop_box = _crop_image(ref_image_full, data.get("crop_box_ref"))
        ref_pose = _predict_pose(ref_crop_image, remove_background=remove_background, infer_aug=infer_aug)

        result = {
            "success": True,
            "reference_crop_box": ref_crop_box,
            "reference_crop_size": [ref_crop_image.width, ref_crop_image.height],
            "reference_pose": {
                "azimuth": ref_pose["azimuth"],
                "elevation": ref_pose["elevation"],
                "rotation": ref_pose["rotation"],
                "confidence": ref_pose["confidence"],
            },
            "target_pose": None,
            "relative_pose": None,
            "rendered": False,
            "renderer_backend": None,
            "infer_aug": infer_aug,
        }

        if data.get("image_tgt"):
            tgt_image_full = _decode_pil_image(data["image_tgt"])
            tgt_crop_image, tgt_crop_box = _crop_image(tgt_image_full, data.get("crop_box_tgt"))
            tgt_pose = _predict_pose(tgt_crop_image, remove_background=remove_background, infer_aug=infer_aug)
            result["target_crop_box"] = tgt_crop_box
            result["target_crop_size"] = [tgt_crop_image.width, tgt_crop_image.height]
            result["target_pose"] = {
                "azimuth": tgt_pose["azimuth"],
                "elevation": tgt_pose["elevation"],
                "rotation": tgt_pose["rotation"],
                "confidence": tgt_pose["confidence"],
            }
            result["relative_pose"] = _compute_relative_pose(result["reference_pose"], result["target_pose"])
        else:
            tgt_pose = None

        if render_visualization:
            result["reference_overlay"] = _image_to_b64(
                _overlay_pose_on_original(ref_image_full, ref_crop_box, result["reference_pose"])
            )
            if tgt_pose is not None:
                result["target_overlay"] = _image_to_b64(
                    _overlay_pose_on_original(tgt_image_full, tgt_crop_box, result["target_pose"])
                )
            result["rendered"] = True
            result["renderer_backend"] = "v1_2d_overlay"

        return jsonify(result)
    except Exception as e:
        app.logger.exception("Orient-Anything v1 inference failed")
        return jsonify({"success": False, "error": str(e)}), 500


def main():
    global repo_dir, ckpt_path, weights_dir, device

    parser = argparse.ArgumentParser(description="Orient-Anything v1 HTTP service")
    parser.add_argument(
        "--repo_dir",
        type=str,
        default=os.environ.get("ORIENT_ANYTHING_REPO_DIR", os.path.join("third_party", "Orient-Anything")),
        help="Path to the Orient-Anything v1 repo.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.environ.get("ORIENT_ANYTHING_CKPT_PATH", os.path.join("checkpoints", "orient_anything", "dino_weight.pt")),
        help="Path to the Orient-Anything v1 checkpoint.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=os.environ.get("ORIENT_ANYTHING_WEIGHTS_DIR", os.path.join("checkpoints", "orient_anything")),
        help="Directory used when searching for checkpoints.",
    )
    parser.add_argument("--port", type=int, default=20034, help="Server port.")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda or cpu.")
    args = parser.parse_args()

    repo_dir = str(Path(args.repo_dir).expanduser().resolve())
    ckpt_path = str(Path(args.ckpt_path).expanduser().resolve()) if args.ckpt_path else args.ckpt_path
    weights_dir = str(Path(args.weights_dir).expanduser().resolve())
    device = _default_device(args.device)

    if not os.path.exists(repo_dir):
        raise FileNotFoundError(f"Orient-Anything repo directory not found: {repo_dir}")
    os.makedirs(weights_dir, exist_ok=True)

    _load_model(repo_dir, ckpt_path)
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
