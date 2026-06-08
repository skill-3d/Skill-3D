"""
Orient-Anything Tool

This module contains the OrientAnythingTool that wraps Orient-Anything
for the SPAgent system.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).parent.parent))

from skill3d.core.tool import Tool

logger = logging.getLogger(__name__)


class OrientAnythingTool(Tool):
    """Tool for absolute orientation and relative rotation estimation."""

    def __init__(self, use_mock: bool = True, server_url: str = "http://localhost:20034"):
        super().__init__(
            name="orient_anything_tool",
            description=(
                "Estimate object orientation from one image and relative rotation between two images of the same object. "
                "Useful for azimuth, elevation, in-plane rotation, symmetry ambiguity, and viewpoint comparison. "
                "It also supports crop_box so you can run orientation reasoning on a detected object region."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.use_mock:
            class SimpleMockOrientAnything:
                def infer(self, image_path, target_image_path=None, **kwargs):
                    render_visualization = bool(kwargs.get("render_visualization", False))
                    output_dir = Path("outputs/orient_anything")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    output_path = None
                    target_output_path = None
                    output_paths = []

                    if render_visualization:
                        image = Image.open(image_path).convert("RGB")
                        draw = ImageDraw.Draw(image)
                        center = (image.width // 2, image.height // 2)
                        axis_len = max(24, min(image.width, image.height) // 4)
                        axis_specs = [
                            ((center[0] + axis_len, center[1]), (255, 0, 0), "X"),
                            ((center[0], center[1] - axis_len), (0, 200, 0), "Y"),
                            ((center[0] - axis_len // 2, center[1] + axis_len // 2), (0, 90, 255), "Z"),
                        ]
                        for end, color, label in axis_specs:
                            draw.line([center, end], fill=color, width=4)
                            draw.text((end[0] + 4, end[1] + 4), label, fill=color)

                        output_path = output_dir / f"{Path(image_path).stem}_reference_overlay_mock.png"
                        image.save(output_path)
                        output_paths.append(str(output_path))

                        if target_image_path:
                            target_image = Image.open(target_image_path).convert("RGB")
                            target_draw = ImageDraw.Draw(target_image)
                            target_center = (target_image.width // 2, target_image.height // 2)
                            for end, color, label in axis_specs:
                                dx = end[0] - center[0]
                                dy = end[1] - center[1]
                                target_end = (target_center[0] + dx, target_center[1] + dy)
                                target_draw.line([target_center, target_end], fill=color, width=4)
                                target_draw.text((target_end[0] + 4, target_end[1] + 4), label, fill=color)

                            target_output_path = output_dir / f"{Path(target_image_path).stem}_target_overlay_mock.png"
                            target_image.save(target_output_path)
                            output_paths.append(str(target_output_path))

                    return {
                        "success": True,
                        "output_path": str(output_path) if output_path else None,
                        "target_output_path": str(target_output_path) if target_output_path else None,
                        "output_paths": output_paths,
                        "reference_pose": {
                            "azimuth": 45.0,
                            "elevation": 10.0,
                            "rotation": 0.0,
                            "symmetry_alpha": 1,
                        },
                        "relative_pose": {
                            "azimuth": 30.0,
                            "elevation": 0.0,
                            "rotation": 0.0,
                        }
                        if target_image_path
                        else None,
                        "target_pose": {
                            "azimuth": 75.0,
                            "elevation": 10.0,
                            "rotation": 0.0,
                        }
                        if target_image_path
                        else None,
                        "rendered": render_visualization,
                        "renderer_backend": "mock_2d" if render_visualization else None,
                    }

            self._client = SimpleMockOrientAnything()
            logger.info("Using mock Orient-Anything service")
            return

        from skill3d.external_experts.OrientAnything.orient_anything_client import OrientAnythingClient

        self._client = OrientAnythingClient(server_url=self.server_url)
        logger.info("Using real Orient-Anything service at %s", self.server_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the reference image of the object.",
                },
                "crop_box": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Required crop box [x1, y1, x2, y2] for the reference image. Use detect_objects_tool first to isolate the target object before orientation inference.",
                },
                "crop_margin": {
                    "type": "integer",
                    "description": "Optional pixel margin added around crop_box before orientation inference.",
                    "default": 0,
                },
                "target_image_path": {
                    "type": "string",
                    "description": "Optional second image of the same object for relative rotation estimation.",
                },
                "target_crop_box": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Optional crop box [x1, y1, x2, y2] for the target image when target_image_path is provided.",
                },
                "remove_background": {
                    "type": "boolean",
                    "description": "Whether to remove the background before orientation inference.",
                    "default": True,
                },
                "render_visualization": {
                    "type": "boolean",
                    "description": "Whether to render axis overlays onto the input image(s).",
                    "default": False,
                },
            },
            "required": ["image_path", "crop_box"],
        }

    def call(
        self,
        image_path: str,
        crop_box: Optional[Sequence[float]] = None,
        crop_margin: int = 0,
        target_image_path: Optional[str] = None,
        target_crop_box: Optional[Sequence[float]] = None,
        remove_background: bool = True,
        render_visualization: bool = False,
    ) -> Dict[str, Any]:
        try:
            if not Path(image_path).exists():
                return {"success": False, "error": f"Reference image not found: {image_path}"}
            if target_image_path and not Path(target_image_path).exists():
                return {"success": False, "error": f"Target image not found: {target_image_path}"}
            if crop_box is None:
                return {"success": False, "error": "crop_box is required for orient_anything_tool"}
            if target_image_path and target_crop_box is None:
                return {"success": False, "error": "target_crop_box is required when target_image_path is provided"}

            result = self._client.infer(
                image_path=image_path,
                crop_box=list(crop_box),
                target_image_path=target_image_path,
                target_crop_box=list(target_crop_box) if target_crop_box is not None else None,
                remove_background=remove_background,
                render_visualization=render_visualization,
            )
            if result and result.get("success"):
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get("output_path"),
                    "target_output_path": result.get("target_output_path"),
                    "output_paths": result.get("output_paths", []),
                    "crop_box": result.get("reference_crop_box") or list(crop_box),
                    "crop_size": result.get("reference_crop_size"),
                    "target_crop_box": result.get("target_crop_box") or (list(target_crop_box) if target_crop_box is not None else None),
                    "target_crop_size": result.get("target_crop_size"),
                    "reference_pose": result.get("reference_pose"),
                    "relative_pose": result.get("relative_pose"),
                    "target_pose": result.get("target_pose"),
                    "rendered": result.get("rendered", False),
                    "renderer_backend": result.get("renderer_backend"),
                }

            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            return {"success": False, "error": f"Orient-Anything failed: {error_msg}"}
        except Exception as e:
            logger.error("Orient-Anything tool error: %s", e)
            return {"success": False, "error": str(e)}
