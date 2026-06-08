import argparse
import base64
import importlib.util
import logging
import os
import traceback
from glob import glob
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request


logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

SwinIRNet = None
device = None
repo_dir = None
weights_dir = None
model_cache: Dict[Tuple, Dict[str, object]] = {}


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_repo_file(base_dir: str, relative_path: str) -> str:
    path = os.path.join(base_dir, relative_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required SwinIR file not found: {path}")
    return path


def _ensure_repo_loaded(base_dir: str):
    global SwinIRNet
    if SwinIRNet is not None:
        return
    network_path = _resolve_repo_file(base_dir, os.path.join("models", "network_swinir.py"))
    module = _load_module_from_path("swinir_network_swinir", network_path)
    SwinIRNet = module.SwinIR


def _default_device(device_name: Optional[str] = None) -> str:
    if device_name:
        return device_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_weight_name(
    task: str,
    scale: int,
    noise: int,
    jpeg: int,
    large_model: bool,
) -> str:
    if task == "classical_sr":
        return f"001_classicalSR_DF2K_s64w8_SwinIR-M_x{scale}.pth"
    if task == "lightweight_sr":
        return f"002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth"
    if task == "real_sr":
        if large_model:
            return "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
        return "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    if task == "gray_dn":
        return f"004_grayDN_DFWB_s128w8_SwinIR-M_noise{noise}.pth"
    if task == "color_dn":
        return f"005_colorDN_DFWB_s128w8_SwinIR-M_noise{noise}.pth"
    if task == "jpeg_car":
        return f"006_CAR_DFWB_s126w7_SwinIR-M_jpeg{jpeg}.pth"
    if task == "color_jpeg_car":
        return f"006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg{jpeg}.pth"
    raise ValueError(f"Unsupported SwinIR task: {task}")


def _resolve_weight_path(
    task: str,
    scale: int,
    noise: int,
    jpeg: int,
    large_model: bool,
    model_path: Optional[str],
) -> str:
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SwinIR checkpoint not found: {model_path}")
        return model_path

    filename = _default_weight_name(task, scale, noise, jpeg, large_model)
    local_path = os.path.join(weights_dir, filename)
    if os.path.exists(local_path):
        return local_path

    for candidate in _candidate_weight_paths(task, scale, noise, jpeg, large_model):
        if os.path.exists(candidate):
            logger.info("Resolved SwinIR local checkpoint %s for task=%s", candidate, task)
            return candidate

    available = sorted(os.path.basename(path) for path in glob(os.path.join(weights_dir, "*.pth")))
    raise FileNotFoundError(
        f"SwinIR checkpoint not found for task={task}, scale={scale}. "
        f"Expected {local_path} or a compatible local file under {weights_dir}. "
        f"Available files: {available}"
    )


def _candidate_weight_paths(
    task: str,
    scale: int,
    noise: int,
    jpeg: int,
    large_model: bool,
):
    candidates = []

    if task == "real_sr":
        exact_patterns = []
        if large_model:
            exact_patterns.extend(
                [
                    f"*realSR*DFOWMFC*SwinIR-L*x{scale}*.pth",
                    f"*realSR*SwinIR-L*x{scale}*.pth",
                ]
            )
        else:
            exact_patterns.extend(
                [
                    f"*realSR*DFO*SwinIR-M*x{scale}*.pth",
                    f"*realSR*SwinIR-M*x{scale}*.pth",
                    f"*realSR*SwinIR-L*x{scale}*.pth",
                    f"*realSR*x{scale}*.pth",
                ]
            )
        for pattern in exact_patterns:
            candidates.extend(sorted(glob(os.path.join(weights_dir, pattern))))

    elif task == "classical_sr":
        candidates.extend(sorted(glob(os.path.join(weights_dir, f"*classicalSR*x{scale}*.pth"))))
    elif task == "lightweight_sr":
        candidates.extend(sorted(glob(os.path.join(weights_dir, f"*lightweightSR*x{scale}*.pth"))))
    elif task == "gray_dn":
        candidates.extend(sorted(glob(os.path.join(weights_dir, f"*grayDN*noise{noise}*.pth"))))
    elif task == "color_dn":
        candidates.extend(sorted(glob(os.path.join(weights_dir, f"*colorDN*noise{noise}*.pth"))))
    elif task == "jpeg_car":
        candidates.extend(sorted(glob(os.path.join(weights_dir, f"*CAR*jpeg{jpeg}*.pth"))))
    elif task == "color_jpeg_car":
        candidates.extend(sorted(glob(os.path.join(weights_dir, f"*colorCAR*jpeg{jpeg}*.pth"))))

    seen = set()
    deduped = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _effective_large_model(task: str, requested_large_model: bool, resolved_path: str) -> bool:
    if task != "real_sr":
        return requested_large_model

    basename = os.path.basename(resolved_path)
    if "SwinIR-L" in basename or "DFOWMFC" in basename:
        return True
    if "SwinIR-M" in basename or "DFO" in basename:
        return False
    return requested_large_model


def _build_model(task: str, scale: int, training_patch_size: int, large_model: bool):
    if task == "classical_sr":
        model = SwinIRNet(
            upscale=scale,
            in_chans=3,
            img_size=training_patch_size,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )
        return model, "params", 8

    if task == "lightweight_sr":
        model = SwinIRNet(
            upscale=scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffledirect",
            resi_connection="1conv",
        )
        return model, "params", 8

    if task == "real_sr":
        if large_model:
            model = SwinIRNet(
                upscale=scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=240,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="3conv",
            )
        else:
            model = SwinIRNet(
                upscale=scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="1conv",
            )
        return model, "params_ema", 8

    if task == "gray_dn":
        model = SwinIRNet(
            upscale=1,
            in_chans=1,
            img_size=128,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )
        return model, "params", 8

    if task == "color_dn":
        model = SwinIRNet(
            upscale=1,
            in_chans=3,
            img_size=128,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )
        return model, "params", 8

    if task == "jpeg_car":
        model = SwinIRNet(
            upscale=1,
            in_chans=1,
            img_size=126,
            window_size=7,
            img_range=255.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )
        return model, "params", 7

    if task == "color_jpeg_car":
        model = SwinIRNet(
            upscale=1,
            in_chans=3,
            img_size=126,
            window_size=7,
            img_range=255.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )
        return model, "params", 7

    raise ValueError(f"Unsupported SwinIR task: {task}")


def _validate_request(task: str, scale: int, noise: int, jpeg: int):
    if task in {"classical_sr", "lightweight_sr"} and scale not in {2, 3, 4, 8}:
        raise ValueError(f"{task} only supports scale in {{2, 3, 4, 8}}")
    if task == "real_sr" and scale != 4:
        raise ValueError("real_sr only supports scale=4")
    if task in {"gray_dn", "color_dn"} and noise not in {15, 25, 50}:
        raise ValueError("Denoising tasks only support noise in {15, 25, 50}")
    if task in {"jpeg_car", "color_jpeg_car"} and jpeg not in {10, 20, 30, 40}:
        raise ValueError("JPEG restoration tasks only support jpeg in {10, 20, 30, 40}")


def _load_model_entry(
    task: str,
    scale: int,
    noise: int,
    jpeg: int,
    training_patch_size: int,
    large_model: bool,
    model_path: Optional[str],
):
    _validate_request(task, scale, noise, jpeg)
    _ensure_repo_loaded(repo_dir)
    resolved_path = _resolve_weight_path(task, scale, noise, jpeg, large_model, model_path)
    effective_large_model = _effective_large_model(task, large_model, resolved_path)
    key = (task, scale, noise, jpeg, training_patch_size, effective_large_model, resolved_path)
    if key in model_cache:
        return model_cache[key]

    if task == "real_sr" and effective_large_model != large_model:
        logger.info(
            "Adjusted SwinIR real_sr architecture to match local weights: requested large_model=%s, resolved=%s",
            large_model,
            effective_large_model,
        )
    model, param_key, window_size = _build_model(task, scale, training_patch_size, effective_large_model)
    state = torch.load(resolved_path, map_location="cpu")
    model.load_state_dict(state[param_key] if param_key in state else state, strict=True)
    model = model.to(device).eval()
    entry = {
        "model": model,
        "window_size": window_size,
        "scale": scale,
        "weights_path": resolved_path,
    }
    model_cache[key] = entry
    logger.info("Loaded SwinIR model for %s from %s", key, resolved_path)
    return entry


def _decode_image(image_b64: str, grayscale: bool) -> np.ndarray:
    image_bytes = base64.b64decode(image_b64)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), flag)
    if image is None:
        raise ValueError("Invalid image data")
    image = image.astype(np.float32) / 255.0
    if grayscale:
        image = image[..., None]
    return image


def _run_tiled_inference(
    img_tensor: torch.Tensor,
    model: torch.nn.Module,
    scale: int,
    tile: Optional[int],
    tile_overlap: int,
    window_size: int,
) -> torch.Tensor:
    if tile is None:
        return model(img_tensor)

    b, c, h, w = img_tensor.size()
    tile = min(tile, h, w)
    if tile % window_size != 0:
        raise ValueError("tile size should be a multiple of window_size")
    stride = tile - tile_overlap
    if stride <= 0:
        raise ValueError("tile_overlap must be smaller than tile")

    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    output_acc = torch.zeros(b, c, h * scale, w * scale, device=img_tensor.device, dtype=img_tensor.dtype)
    weight_acc = torch.zeros_like(output_acc)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_tensor[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
            out_patch = model(in_patch)
            out_mask = torch.ones_like(out_patch)
            output_acc[..., h_idx * scale : (h_idx + tile) * scale, w_idx * scale : (w_idx + tile) * scale].add_(out_patch)
            weight_acc[..., h_idx * scale : (h_idx + tile) * scale, w_idx * scale : (w_idx + tile) * scale].add_(out_mask)

    return output_acc.div_(weight_acc)


def _infer_single_image(
    image: np.ndarray,
    model: torch.nn.Module,
    scale: int,
    tile: Optional[int],
    tile_overlap: int,
    window_size: int,
) -> np.ndarray:
    if image.shape[2] == 1:
        chw = np.transpose(image, (2, 0, 1))
    else:
        chw = np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))
    img_tensor = torch.from_numpy(chw).float().unsqueeze(0).to(device)

    _, _, h_old, w_old = img_tensor.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [2])], dim=2)[:, :, : h_old + h_pad, :]
    img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], dim=3)[:, :, :, : w_old + w_pad]

    with torch.no_grad():
        output = _run_tiled_inference(
            img_tensor=img_tensor,
            model=model,
            scale=scale,
            tile=tile,
            tile_overlap=tile_overlap,
            window_size=window_size,
        )
    output = output[..., : h_old * scale, : w_old * scale]
    output = output.squeeze(0).float().cpu().clamp_(0, 1).numpy()

    if output.ndim == 3 and output.shape[0] > 1:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    elif output.ndim == 3 and output.shape[0] == 1:
        output = output[0]

    output = (output * 255.0).round().astype(np.uint8)
    return output


@app.route("/health", methods=["GET"])
def health_check():
    try:
        loaded = [
            {
                "task": key[0],
                "scale": key[1],
                "noise": key[2],
                "jpeg": key[3],
                "weights_path": value["weights_path"],
            }
            for key, value in model_cache.items()
        ]
        return jsonify(
            {
                "status": "healthy",
                "repo_dir": repo_dir,
                "weights_dir": weights_dir,
                "device": device,
                "loaded_models": loaded,
            }
        )
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/test", methods=["GET"])
def test():
    try:
        entry = _load_model_entry(
            task="real_sr",
            scale=4,
            noise=15,
            jpeg=40,
            training_patch_size=64,
            large_model=False,
            model_path=None,
        )
        test_image = np.zeros((64, 64, 3), dtype=np.float32)
        test_image[:, 16:48, :] = 1.0
        output = _infer_single_image(
            image=test_image,
            model=entry["model"],
            scale=entry["scale"],
            tile=None,
            tile_overlap=32,
            window_size=entry["window_size"],
        )
        return jsonify({"success": True, "shape": list(output.shape)})
    except Exception as e:
        logger.error("SwinIR test failed: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/infer", methods=["POST"])
def infer():
    try:
        data = request.get_json(force=True)
        if "image" not in data:
            return jsonify({"error": "Missing image"}), 400

        task = data.get("task", "real_sr")
        scale = int(data.get("scale", 4 if task in {"real_sr", "classical_sr", "lightweight_sr"} else 1))
        noise = int(data.get("noise", 15))
        jpeg = int(data.get("jpeg", 40))
        training_patch_size = int(data.get("training_patch_size", 64))
        large_model = bool(data.get("large_model", False))
        tile = data.get("tile")
        tile = int(tile) if tile is not None else None
        tile_overlap = int(data.get("tile_overlap", 32))
        model_path = data.get("model_path")

        grayscale = task in {"gray_dn", "jpeg_car"}
        image = _decode_image(data["image"], grayscale=grayscale)
        entry = _load_model_entry(
            task=task,
            scale=scale,
            noise=noise,
            jpeg=jpeg,
            training_patch_size=training_patch_size,
            large_model=large_model,
            model_path=model_path,
        )
        output = _infer_single_image(
            image=image,
            model=entry["model"],
            scale=entry["scale"],
            tile=tile,
            tile_overlap=tile_overlap,
            window_size=entry["window_size"],
        )
        success, buffer = cv2.imencode(".png", output)
        if not success:
            raise RuntimeError("Failed to encode SwinIR output")

        return jsonify(
            {
                "success": True,
                "enhanced_image": base64.b64encode(buffer).decode("utf-8"),
                "shape": list(output.shape),
                "task": task,
                "scale": entry["scale"],
                "weights_path": entry["weights_path"],
            }
        )
    except Exception as e:
        logger.error("SwinIR inference failed: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinIR Server")
    parser.add_argument(
        "--repo_dir",
        type=str,
        default=os.environ.get("SWINIR_REPO_DIR", "third_party/SwinIR"),
        help="Path to the official SwinIR repository checkout.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=os.environ.get("SWINIR_WEIGHTS_DIR", "checkpoints/swinir"),
        help="Directory containing SwinIR checkpoint files.",
    )
    parser.add_argument("--port", type=int, default=20032, help="Server port.")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda or cpu.")
    args = parser.parse_args()

    repo_dir = args.repo_dir
    weights_dir = args.weights_dir
    device = _default_device(args.device)

    if not os.path.exists(repo_dir):
        raise FileNotFoundError(f"SwinIR repo directory not found: {repo_dir}")
    os.makedirs(weights_dir, exist_ok=True)
    _ensure_repo_loaded(repo_dir)

    logger.info("Starting SwinIR server on port %s", args.port)
    logger.info("Repo dir: %s", repo_dir)
    logger.info("Weights dir: %s", weights_dir)
    logger.info("Device: %s", device)
    app.run(host="0.0.0.0", port=args.port, debug=False)
