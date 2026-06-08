"""
SwinIR Tool

This module contains the SwinIRTool that wraps SwinIR image restoration
for the SPAgent system.
"""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from skill3d.core.tool import Tool

logger = logging.getLogger(__name__)


class SwinIRTool(Tool):
    """Tool for image restoration and super-resolution using SwinIR."""

    def __init__(self, use_mock: bool = True, server_url: str = "http://localhost:20032"):
        super().__init__(
            name="swinir_tool",
            description=(
                "Enhance low-quality images with SwinIR. Use it for super-resolution, denoising, "
                "or JPEG artifact reduction before running downstream perception tools on blurry, "
                "tiny, noisy, or compressed images. It also supports crop_box so you can enhance a "
                "detected object region instead of the whole image."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.use_mock:
            class SimpleMockSwinIR:
                def infer(self, image_path, task="real_sr", scale=4, **kwargs):
                    try:
                        image = Image.open(image_path)
                    except Exception:
                        return {"success": False, "error": f"Failed to read image: {image_path}"}

                    if task in {"real_sr", "classical_sr", "lightweight_sr"}:
                        output = image.resize((image.width * scale, image.height * scale), Image.BICUBIC)
                    else:
                        output = image

                    Path("outputs/swinir").mkdir(parents=True, exist_ok=True)
                    output_path = Path("outputs/swinir") / f"{Path(image_path).stem}_{task}_mock.png"
                    output.save(output_path)
                    return {
                        "success": True,
                        "output_path": str(output_path),
                        "shape": list(reversed(output.size)) + ([len(output.getbands())] if len(output.getbands()) > 1 else []),
                        "task": task,
                        "scale": scale,
                    }

            self._client = SimpleMockSwinIR()
            logger.info("Using mock SwinIR service")
            return

        from skill3d.external_experts.SwinIR.swinir_client import SwinIRClient

        self._client = SwinIRClient(server_url=self.server_url)
        logger.info("Using real SwinIR service at %s", self.server_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image that needs enhancement.",
                },
                "crop_box": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Optional crop box [x1, y1, x2, y2]. Useful after detect_objects_tool when only one detected region needs local zoom enhancement.",
                },
                "crop_margin": {
                    "type": "integer",
                    "description": "Optional pixel margin added around crop_box before enhancement.",
                    "default": 0,
                },
                "task": {
                    "type": "string",
                    "enum": [
                        "real_sr",
                        "classical_sr",
                        "lightweight_sr",
                        "gray_dn",
                        "color_dn",
                        "jpeg_car",
                        "color_jpeg_car",
                    ],
                    "description": "Enhancement mode. real_sr is the best default for blurry or low-resolution real images.",
                    "default": "real_sr",
                },
                "scale": {
                    "type": "integer",
                    "description": "Upscale factor for super-resolution tasks. Use 4 for real_sr.",
                    "default": 4,
                },
                "large_model": {
                    "type": "boolean",
                    "description": "Use the larger real_sr checkpoint. Only applies to task=real_sr.",
                    "default": False,
                },
                "tile": {
                    "type": "integer",
                    "description": "Optional tiled inference size for large images to reduce GPU memory usage.",
                },
                "tile_overlap": {
                    "type": "integer",
                    "description": "Tile overlap used when tile is set.",
                    "default": 32,
                },
                "noise": {
                    "type": "integer",
                    "description": "Noise level for denoising checkpoints. Valid values: 15, 25, 50.",
                    "default": 15,
                },
                "jpeg": {
                    "type": "integer",
                    "description": "JPEG quality setting for JPEG restoration checkpoints. Valid values: 10, 20, 30, 40.",
                    "default": 40,
                },
            },
            "required": ["image_path"],
        }

    @staticmethod
    def _normalize_crop_box(
        crop_box: Sequence[float],
        image_size: Tuple[int, int],
        crop_margin: int = 0,
    ) -> Tuple[int, int, int, int]:
        if len(crop_box) != 4:
            raise ValueError("crop_box must contain exactly 4 values: [x1, y1, x2, y2]")

        width, height = image_size
        x1, y1, x2, y2 = [int(round(float(value))) for value in crop_box]
        margin = max(0, int(crop_margin))

        left = max(0, min(x1, x2) - margin)
        top = max(0, min(y1, y2) - margin)
        right = min(width, max(x1, x2) + margin)
        bottom = min(height, max(y1, y2) + margin)

        if right <= left or bottom <= top:
            raise ValueError(f"Invalid crop_box after normalization: {[left, top, right, bottom]}")
        return left, top, right, bottom

    def call(
        self,
        image_path: str,
        crop_box: Optional[Sequence[float]] = None,
        crop_margin: int = 0,
        task: str = "real_sr",
        scale: int = 4,
        large_model: bool = False,
        tile: Optional[int] = None,
        tile_overlap: int = 32,
        noise: int = 15,
        jpeg: int = 40,
    ) -> Dict[str, Any]:
        try:
            if not Path(image_path).exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}

            inference_image_path = image_path
            applied_crop_box = None
            crop_size = None

            if crop_box is not None:
                with Image.open(image_path) as source_image:
                    normalized_box = self._normalize_crop_box(
                        crop_box=crop_box,
                        image_size=source_image.size,
                        crop_margin=crop_margin,
                    )
                    cropped_image = source_image.crop(normalized_box)
                    crop_size = [cropped_image.width, cropped_image.height]
                    with tempfile.TemporaryDirectory(prefix="swinir_crop_") as tmp_dir:
                        crop_path = Path(tmp_dir) / f"{Path(image_path).stem}_crop.png"
                        cropped_image.save(crop_path)
                        result = self._client.infer(
                            image_path=str(crop_path),
                            task=task,
                            scale=scale,
                            large_model=large_model,
                            tile=tile,
                            tile_overlap=tile_overlap,
                            noise=noise,
                            jpeg=jpeg,
                        )
                    inference_image_path = image_path
                    applied_crop_box = list(normalized_box)
            else:
                result = self._client.infer(
                    image_path=image_path,
                    task=task,
                    scale=scale,
                    large_model=large_model,
                    tile=tile,
                    tile_overlap=tile_overlap,
                    noise=noise,
                    jpeg=jpeg,
                )
            if result and result.get("success"):
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get("output_path"),
                    "input_image_path": inference_image_path,
                    "crop_box": applied_crop_box,
                    "crop_size": crop_size,
                    "shape": result.get("shape"),
                    "task": result.get("task", task),
                    "scale": result.get("scale", scale),
                }

            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            return {"success": False, "error": f"SwinIR failed: {error_msg}"}
        except Exception as e:
            logger.error("SwinIR tool error: %s", e)
            return {"success": False, "error": str(e)}
