import base64
import io
import json
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
import requests

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _scale_intrinsics_to_size(
    intrinsics: Optional[np.ndarray],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> Optional[np.ndarray]:
    if intrinsics is None:
        return None
    if src_width <= 0 or src_height <= 0:
        return intrinsics
    if src_width == dst_width and src_height == dst_height:
        return intrinsics

    scaled = np.asarray(intrinsics, dtype=np.float32).copy()
    scale_x = dst_width / float(src_width)
    scale_y = dst_height / float(src_height)
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


class DepthClient:
    def __init__(self, server_url: str):
        """
        初始化客户端

        Args:
            server_url: 服务器地址，如 'http://localhost:20019'
        """
        self.server_url = server_url.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False

    def health_check(self):
        """检查服务器健康状态"""
        try:
            logger.info("正在检查服务器状态...")
            response = self.session.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return None

    def test_infer(self):
        """使用服务器内置的测试数据进行推理"""
        try:
            logger.info("发送测试推理请求...")
            response = self.session.get(f"{self.server_url}/test", timeout=30)
            response.raise_for_status()
            result = response.json()
            logger.info("测试推理完成")
            return result
        except Exception as e:
            logger.error(f"测试推理失败: {e}")
            return None

    def infer(
        self,
        image_path: str,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        return_raw_depth: bool = True,
        return_intrinsics: bool = True,
        return_confidence: bool = False,
        camera_intrinsics: Optional[np.ndarray] = None,
    ):
        """
        发送图片进行深度估计

        Args:
            image_path: 图片路径
            process_res: 处理分辨率
            process_res_method: 处理方式

        Returns:
            推理结果，如果失败则返回None
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"图片文件不存在: {image_path}")
                return None

            logger.info(f"读取图片: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图片: {image_path}")
                return None

            logger.info(f"图片尺寸: {image.shape}")

            _, buffer = cv2.imencode(".jpg", image)
            image_b64 = base64.b64encode(buffer).decode("utf-8")

            data = {
                "image": image_b64,
                "process_res": process_res,
                "process_res_method": process_res_method,
                "return_colored": True,
                "return_raw_depth": return_raw_depth,
                "return_intrinsics": return_intrinsics,
                "return_confidence": return_confidence,
            }
            if camera_intrinsics is not None:
                intrinsics = np.asarray(camera_intrinsics, dtype=np.float32)
                if intrinsics.shape != (3, 3):
                    raise ValueError("camera_intrinsics must be a 3x3 matrix")
                data["camera_intrinsics"] = intrinsics.tolist()

            logger.info("发送推理请求...")
            response = self.session.post(
                f"{self.server_url}/infer",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()

            if result.get("success"):
                depth_bytes = base64.b64decode(result["depth_map"])
                depth_array = cv2.imdecode(
                    np.frombuffer(depth_bytes, np.uint8),
                    cv2.IMREAD_COLOR if result.get("return_colored", True) else cv2.IMREAD_GRAYSCALE,
                )

                original_height, original_width = image.shape[:2]
                depth_height, depth_width = depth_array.shape[:2]

                if depth_width != original_width:
                    depth_array = cv2.resize(
                        depth_array, (original_width, depth_height * original_width // depth_width)
                    )

                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                if len(depth_array.shape) == 2:
                    depth_array = cv2.cvtColor(depth_array, cv2.COLOR_GRAY2BGR)

                combined_image = np.vstack([image, depth_array])

                input_filename = os.path.basename(image_path)
                name_without_ext = os.path.splitext(input_filename)[0]

                os.makedirs("outputs", exist_ok=True)

                combined_filename = f"outputs/depth_v3_combined_{name_without_ext}.png"
                cv2.imwrite(combined_filename, combined_image)
                logger.info(f"拼接图像已保存至: {combined_filename}")

                depth_only_filename = f"outputs/depth_v3_only_{name_without_ext}.png"
                cv2.imwrite(depth_only_filename, depth_array)
                logger.info(f"深度图已保存至: {depth_only_filename}")

                raw_depth = None
                raw_depth_shape = result.get("raw_depth_shape")
                if isinstance(raw_depth_shape, (list, tuple)) and len(raw_depth_shape) >= 2:
                    raw_depth_height = int(raw_depth_shape[0])
                    raw_depth_width = int(raw_depth_shape[1])
                else:
                    raw_depth_height = depth_array.shape[0]
                    raw_depth_width = depth_array.shape[1]

                raw_depth_payload = result.get("raw_depth_npz")
                if raw_depth_payload:
                    try:
                        raw_bytes = base64.b64decode(raw_depth_payload)
                        with np.load(io.BytesIO(raw_bytes)) as raw_file:
                            raw_depth = raw_file["depth"]
                        raw_depth_height, raw_depth_width = raw_depth.shape[:2]
                        if raw_depth is not None:
                            if raw_depth_width != original_width or raw_depth_height != original_height:
                                raw_depth = cv2.resize(
                                    raw_depth,
                                    (original_width, original_height),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                    except Exception as exc:
                        logger.warning(f"Failed to decode raw depth payload: {exc}")

                intrinsics = None
                intrinsics_payload = result.get("intrinsics")
                if intrinsics_payload is not None:
                    try:
                        intrinsics = np.asarray(intrinsics_payload, dtype=np.float32)
                        if intrinsics.shape != (3, 3):
                            intrinsics = None
                        else:
                            intrinsics = _scale_intrinsics_to_size(
                                intrinsics,
                                src_width=raw_depth_width,
                                src_height=raw_depth_height,
                                dst_width=original_width,
                                dst_height=original_height,
                            )
                    except Exception as exc:
                        logger.warning(f"Failed to decode intrinsics payload: {exc}")
                        intrinsics = None

                confidence = None
                conf_payload = result.get("confidence_npz")
                if conf_payload:
                    try:
                        conf_bytes = base64.b64decode(conf_payload)
                        with np.load(io.BytesIO(conf_bytes)) as conf_file:
                            confidence = conf_file["conf"]
                        if confidence.shape[:2] != (original_height, original_width):
                            confidence = cv2.resize(
                                confidence,
                                (original_width, original_height),
                                interpolation=cv2.INTER_LINEAR,
                            )
                    except Exception as exc:
                        logger.warning(f"Failed to decode confidence payload: {exc}")
                        confidence = None

                return {
                    "depth_array": depth_array,
                    "combined_array": combined_image,
                    "shape": result.get("shape"),
                    "output_path": combined_filename,
                    "depth_only_path": depth_only_filename,
                    "raw_depth": raw_depth,
                    "intrinsics": intrinsics,
                    "confidence": confidence,
                    "depth_min": result.get("depth_min"),
                    "depth_max": result.get("depth_max"),
                    "depth_units": result.get("depth_units", "relative"),
                    "is_metric": bool(result.get("is_metric")),
                    "model_variant": result.get("model_variant"),
                    "scale_factor": result.get("scale_factor"),
                    "success": True,
                }

            logger.error(f"服务器返回错误: {result.get('error')}")
            return None

        except Exception as e:
            logger.error(f"推理请求失败: {e}")
            return None


def main():
    """主程序"""
    SERVER_URL = "http://0.0.0.0:20019"

    client = DepthClient(SERVER_URL)

    logger.info("\n=== 执行健康检查 ===")
    health = client.health_check()
    if health:
        logger.info(f"服务器状态: {health}")
    else:
        logger.error("服务器健康检查失败，退出程序")
        return

    time.sleep(2)

    logger.info("\n=== 执行测试推理 ===")
    test_result = client.test_infer()
    if test_result:
        logger.info(f"测试推理成功: {test_result}")
    else:
        logger.error("测试推理失败，退出程序")
        return

    time.sleep(2)

    logger.info("\n=== 处理图片 ===")
    image_path = "assets/example.png"
    result = client.infer(image_path)

    if result:
        logger.info("图片处理成功！")
        logger.info(f"- 输入图片: {image_path}")
        logger.info(f"- 拼接图像: {result['output_path']}")
        logger.info(f"- 深度图像: {result['depth_only_path']}")
        logger.info(f"- 图片尺寸: {result['shape']}")
    else:
        logger.error("图片处理失败")


if __name__ == "__main__":
    main()
