import base64
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SwinIRClient:
    def __init__(self, server_url: str = "http://localhost:20032"):
        self.server_url = server_url.rstrip("/")

    def health_check(self):
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("SwinIR health check failed: %s", e)
            return None

    def test(self):
        try:
            response = requests.get(f"{self.server_url}/test", timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("SwinIR test failed: %s", e)
            return None

    def infer(
        self,
        image_path: str,
        task: str = "real_sr",
        scale: int = 4,
        large_model: bool = False,
        tile: Optional[int] = None,
        tile_overlap: int = 32,
        noise: int = 15,
        jpeg: int = 40,
        training_patch_size: int = 64,
    ):
        try:
            if not os.path.exists(image_path):
                logger.error("Image file not found: %s", image_path)
                return None

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                logger.error("Failed to read image: %s", image_path)
                return None

            success, buffer = cv2.imencode(".png", image)
            if not success:
                logger.error("Failed to encode image: %s", image_path)
                return None

            payload = {
                "image": base64.b64encode(buffer).decode("utf-8"),
                "task": task,
                "scale": scale,
                "large_model": large_model,
                "tile": tile,
                "tile_overlap": tile_overlap,
                "noise": noise,
                "jpeg": jpeg,
                "training_patch_size": training_patch_size,
            }
            response = requests.post(
                f"{self.server_url}/infer",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
            if not result.get("success"):
                logger.error("SwinIR server returned error: %s", result)
                return None

            image_bytes = base64.b64decode(result["enhanced_image"])
            output_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if output_array is None:
                logger.error("Failed to decode SwinIR output")
                return None

            os.makedirs("outputs/swinir", exist_ok=True)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join("outputs", "swinir", f"{stem}_{task}.png")
            cv2.imwrite(output_path, output_array)

            return {
                "success": True,
                "output_path": output_path,
                "shape": result.get("shape"),
                "task": result.get("task"),
                "scale": result.get("scale"),
                "weights_path": result.get("weights_path"),
                "enhanced_array": output_array,
            }
        except Exception as e:
            logger.error("SwinIR inference request failed: %s", e)
            return None


def main():
    client = SwinIRClient()
    logger.info("=== SwinIR health ===")
    logger.info(client.health_check())
    time.sleep(1)
    logger.info("=== SwinIR test ===")
    logger.info(client.test())


if __name__ == "__main__":
    main()
