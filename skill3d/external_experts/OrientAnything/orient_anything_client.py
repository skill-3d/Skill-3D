import base64
import logging
import os
import time
from typing import Optional

import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrientAnythingClient:
    def __init__(self, server_url: str = "http://127.0.0.1:20034", max_retries: int = 2, retry_backoff: float = 2.0):
        self.server_url = server_url.rstrip("/")
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))

    def health_check(self):
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Orient-Anything health check failed: %s", e)
            return None

    def test(self):
        try:
            response = requests.get(f"{self.server_url}/test", timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Orient-Anything test failed: %s", e)
            return None

    @staticmethod
    def _file_to_b64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _save_b64_image(image_b64: str, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image_bytes = base64.b64decode(image_b64)
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        return output_path

    def infer(
        self,
        image_path: str,
        crop_box,
        target_image_path: Optional[str] = None,
        target_crop_box=None,
        remove_background: bool = True,
        render_visualization: bool = False,
        infer_aug: bool = False,
    ):
        try:
            if not os.path.exists(image_path):
                logger.error("Reference image not found: %s", image_path)
                return {"success": False, "error": f"Reference image not found: {image_path}"}
            if target_image_path and not os.path.exists(target_image_path):
                logger.error("Target image not found: %s", target_image_path)
                return {"success": False, "error": f"Target image not found: {target_image_path}"}

            payload = {
                "image_ref": self._file_to_b64(image_path),
                "crop_box_ref": list(crop_box) if crop_box is not None else None,
                "image_tgt": self._file_to_b64(target_image_path) if target_image_path else None,
                "crop_box_tgt": list(target_crop_box) if target_crop_box is not None else None,
                "remove_background": remove_background,
                "render_visualization": render_visualization,
                "infer_aug": infer_aug,
            }
            result = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = requests.post(
                        f"{self.server_url}/infer",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=300,
                    )
                    response.raise_for_status()
                    result = response.json()
                    break
                except requests.HTTPError as e:
                    response = getattr(e, "response", None)
                    detail = response.text[:2000] if response is not None and response.text else None
                    status_code = response.status_code if response is not None else None
                    retryable = status_code in {500, 502, 503, 504}
                    if retryable and attempt < self.max_retries:
                        wait_seconds = self.retry_backoff * (attempt + 1)
                        logger.warning(
                            "Orient-Anything infer attempt %s/%s failed with HTTP %s. Retrying in %.1fs",
                            attempt + 1,
                            self.max_retries + 1,
                            status_code,
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)
                        continue
                    logger.error(
                        "Orient-Anything inference request failed with HTTP %s: %s",
                        status_code,
                        detail or e,
                    )
                    return {
                        "success": False,
                        "error": str(e),
                        "status_code": status_code,
                        "response_text": detail,
                    }
                except requests.RequestException as e:
                    if attempt < self.max_retries:
                        wait_seconds = self.retry_backoff * (attempt + 1)
                        logger.warning(
                            "Orient-Anything infer attempt %s/%s failed: %s. Retrying in %.1fs",
                            attempt + 1,
                            self.max_retries + 1,
                            e,
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)
                        continue
                    logger.error("Orient-Anything inference request failed: %s", e)
                    return {"success": False, "error": str(e)}

            if result is None:
                return {"success": False, "error": "Orient-Anything inference produced no result"}

            if not result.get("success"):
                logger.error("Orient-Anything server returned error: %s", result)
                return {
                    "success": False,
                    "error": result.get("error", "Orient-Anything server returned success=false"),
                    "server_result": result,
                }

            output_path = None
            target_output_path = None
            output_paths = []
            stem = os.path.splitext(os.path.basename(image_path))[0]
            if result.get("reference_overlay"):
                output_path = self._save_b64_image(
                    result["reference_overlay"],
                    os.path.join("outputs", "orient_anything", f"{stem}_reference_overlay.png"),
                )
                output_paths.append(output_path)
            if target_image_path and result.get("target_overlay"):
                target_stem = os.path.splitext(os.path.basename(target_image_path))[0]
                target_output_path = self._save_b64_image(
                    result["target_overlay"],
                    os.path.join("outputs", "orient_anything", f"{target_stem}_target_overlay.png"),
                )
                output_paths.append(target_output_path)

            return {
                "success": True,
                "output_path": output_path,
                "target_output_path": target_output_path,
                "output_paths": output_paths,
                "reference_pose": result.get("reference_pose"),
                "relative_pose": result.get("relative_pose"),
                "target_pose": result.get("target_pose"),
                "rendered": result.get("rendered", False),
                "renderer_backend": result.get("renderer_backend"),
                "infer_aug": bool(result.get("infer_aug", False)),
            }
        except Exception as e:
            logger.error("Orient-Anything inference request failed: %s", e)
            return {"success": False, "error": str(e)}


def main():
    client = OrientAnythingClient()
    logger.info("=== Orient-Anything health ===")
    logger.info(client.health_check())
    time.sleep(1)
    logger.info("=== Orient-Anything test ===")
    logger.info(client.test())


if __name__ == "__main__":
    main()
