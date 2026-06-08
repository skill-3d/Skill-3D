import argparse
import base64
import io
import logging
import os
import traceback
from typing import Optional

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request

from depth_anything_3.api import DepthAnything3

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model = None
device = None
model_dir = None


def _infer_model_variant(pretrained_dir: Optional[str]) -> str:
    model_name = os.path.basename(str(pretrained_dir or "")).lower()
    if "nested" in model_name:
        return "nested"
    if "metric" in model_name:
        return "metric"
    if "mono" in model_name:
        return "mono"
    return "unknown"


def _prediction_is_metric(prediction) -> bool:
    try:
        return bool(int(getattr(prediction, "is_metric", 0)))
    except Exception:
        return _infer_model_variant(model_dir) in {"nested", "metric"}


def _parse_intrinsics_matrix(payload) -> Optional[np.ndarray]:
    if payload is None:
        return None
    matrix = np.asarray(payload, dtype=np.float32)
    if matrix.shape == (3, 3):
        return matrix[None, ...]
    if matrix.shape == (1, 3, 3):
        return matrix
    raise ValueError("camera_intrinsics must be a 3x3 matrix or shape [1, 3, 3]")


def load_model(pretrained_dir: str, device_name: str = None) -> bool:
    """加载 Depth Anything 3 模型"""
    global model, device, model_dir
    try:
        model_dir = pretrained_dir
        if device_name:
            device = device_name
        else:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        logger.info(f"正在加载模型: {pretrained_dir}")
        logger.info(f"使用设备: {device}")
        model = DepthAnything3.from_pretrained(pretrained_dir).to(device)
        model.eval()

        # 简单验证
        logger.info("正在验证模型...")
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        with torch.no_grad():
            _ = model.inference([dummy], export_dir=None, export_format="mini_npz")
        logger.info("模型验证成功")
        return True
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.error(traceback.format_exc())
        return False


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    try:
        status = {
            "status": "健康" if model is not None else "不健康",
            "model_loaded": model is not None,
            "model_dir": model_dir,
            "device": device,
            "model_variant": _infer_model_variant(model_dir),
            "depth_units": "meters" if _infer_model_variant(model_dir) in {"nested", "metric"} else "relative",
        }
        logger.info(f"健康检查结果：{status}")
        return jsonify(status)
    except Exception as e:
        logger.error(f"健康检查失败：{e}")
        return jsonify({"status": "不健康", "error": str(e)}), 500


@app.route("/test", methods=["GET"])
def test():
    """测试接口"""
    global model
    try:
        if model is None:
            return jsonify({"success": False, "error": "模型未加载"}), 500

        logger.info("正在创建测试图像...")
        test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            test_image[:, i] = i

        logger.info("正在进行测试推理...")
        with torch.no_grad():
            prediction = model.inference([test_image], export_dir=None, export_format="mini_npz")
            depth = prediction.depth[0]

        logger.info(f"推理成功，输出尺寸：{depth.shape}")
        return jsonify({"success": True, "message": "测试推理成功", "shape": list(depth.shape)})
    except Exception as e:
        logger.error(f"测试推理失败：{e}")
        logger.error(f"错误追踪：{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/infer", methods=["POST"])
def infer():
    """深度估计推理接口"""
    global model
    if model is None:
        return jsonify({"error": "模型未加载"}), 500

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "缺少图像数据"}), 400

        process_res = data.get("process_res", 504)
        process_res_method = data.get("process_res_method", "upper_bound_resize")
        return_colored = data.get("return_colored", True)
        return_raw_depth = data.get("return_raw_depth", False)
        return_intrinsics = data.get("return_intrinsics", True)
        return_confidence = data.get("return_confidence", False)
        try:
            intrinsics = _parse_intrinsics_matrix(data.get("camera_intrinsics"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            image_bytes = base64.b64decode(data["image"])
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return jsonify({"error": "图像数据无效"}), 400

        if image is None:
            return jsonify({"error": "图像数据无效"}), 400

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logger.info("正在进行深度估计...")
        with torch.no_grad():
            prediction = model.inference(
                [image_rgb],
                export_dir=None,
                export_format="mini_npz",
                process_res=process_res,
                process_res_method=process_res_method,
                intrinsics=intrinsics,
            )

        depth = prediction.depth[0].astype(np.float32)
        conf = None
        if getattr(prediction, "conf", None) is not None:
            conf = prediction.conf[0].astype(np.float32)
        predicted_intrinsics = None
        if getattr(prediction, "intrinsics", None) is not None:
            predicted_intrinsics = np.asarray(prediction.intrinsics[0], dtype=np.float32)
        is_metric = _prediction_is_metric(prediction)
        min_val = float(depth.min())
        max_val = float(depth.max())
        denom = max(max_val - min_val, 1e-6)
        depth_norm = ((depth - min_val) / denom * 255.0).astype(np.uint8)

        if return_colored:
            import matplotlib

            cmap = matplotlib.colormaps.get_cmap("Spectral_r")
            depth_vis = (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        else:
            depth_vis = depth_norm

        _, buffer = cv2.imencode(".png", depth_vis)
        depth_b64 = base64.b64encode(buffer).decode("utf-8")

        raw_depth_b64 = None
        if return_raw_depth:
            raw_buffer = io.BytesIO()
            np.savez_compressed(raw_buffer, depth=depth)
            raw_depth_b64 = base64.b64encode(raw_buffer.getvalue()).decode("utf-8")

        conf_b64 = None
        if return_confidence and conf is not None:
            conf_buffer = io.BytesIO()
            np.savez_compressed(conf_buffer, conf=conf)
            conf_b64 = base64.b64encode(conf_buffer.getvalue()).decode("utf-8")

        logger.info("深度估计完成")
        return jsonify(
            {
                "success": True,
                "depth_map": depth_b64,
                "shape": list(depth_vis.shape),
                "raw_depth_shape": list(depth.shape),
                "input_image_shape": list(image.shape[:2]),
                "return_colored": return_colored,
                "depth_min": min_val,
                "depth_max": max_val,
                "depth_units": "meters" if is_metric else "relative",
                "is_metric": is_metric,
                "model_variant": _infer_model_variant(model_dir),
                "scale_factor": getattr(prediction, "scale_factor", None),
                "raw_depth_npz": raw_depth_b64,
                "intrinsics": predicted_intrinsics.tolist() if return_intrinsics and predicted_intrinsics is not None else None,
                "confidence_npz": conf_b64,
                "confidence_shape": list(conf.shape) if conf_b64 is not None else None,
            }
        )

    except Exception as e:
        logger.error(f"推理失败：{e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"推理失败：{str(e)}"}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Anything V3 Server")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints/depth_anything/DA3NESTED_GIANT_LARGE_1.1",
        help="Path to depth model directory (default: checkpoints/depth_anything/DA3NESTED_GIANT_LARGE_1.1)",
    )
    parser.add_argument("--port", type=int, default=20019, help="Port to run the server on (default: 20019)")
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g., cuda, cpu, mps)")

    args = parser.parse_args()

    logger.info("正在启动服务器...")
    logger.info(f"模型路径: {args.model_dir}")
    logger.info(f"服务端口: {args.port}")

    if not os.path.exists(args.model_dir):
        logger.error(f"找不到模型目录：{args.model_dir}")
        exit(1)

    if not load_model(pretrained_dir=args.model_dir, device_name=args.device):
        logger.error("无法启动服务器：模型加载失败")
        exit(1)

    logger.info("模型加载成功，正在启动服务器...")
    app.run(host="0.0.0.0", port=args.port, debug=False)
