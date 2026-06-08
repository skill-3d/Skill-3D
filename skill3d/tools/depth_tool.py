"""
Depth Estimation Tool

This module contains the DepthEstimationTool that wraps
Depth-AnythingV3 functionality for the SPAgent system.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from skill3d.core.tool import Tool

logger = logging.getLogger(__name__)


class DepthEstimationTool(Tool):
    """Tool for depth estimation using Depth-AnythingV3"""
    
    def __init__(self, use_mock: bool = True, server_url: str = "http://127.0.0.1:20019"):
        """
        Initialize depth estimation tool
        
        Args:
            use_mock: Whether to use mock client for testing
            server_url: URL of the depth estimation server
        """
        super().__init__(
            name="depth_estimation_tool",
            description=(
                "Generate a dense metric depth map for the input image. "
                "This tool supports nearest/farthest reasoning, object-level depth summaries from crop boxes, "
                "and pixel-level metric measurements such as per-point range and 3D distance between selected points."
            )
        )
        
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the depth estimation client"""
        if self.use_mock:
            try:
                from skill3d.external_experts.Depth_AnythingV3.mock_depth_service import MockDepthService
                self._client = MockDepthService()
                logger.info("Using mock depth estimation service")
            except ImportError:
                class SimpleMockDepthService:
                    def infer(self, image_path, **kwargs):
                        stem = Path(image_path).stem
                        raw_depth = np.tile(
                            np.linspace(1.0, 5.0, 256, dtype=np.float32),
                            (256, 1),
                        )
                        depth_vis = ((raw_depth - raw_depth.min()) / max(raw_depth.max() - raw_depth.min(), 1e-6) * 255.0).astype(np.uint8)
                        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
                        return {
                            "success": True,
                            "output_path": f"outputs/depth_v3_mock_{stem}.png",
                            "depth_only_path": f"outputs/depth_v3_mock_only_{stem}.png",
                            "shape": [256, 256],
                            "depth_array": depth_vis,
                            "raw_depth": raw_depth,
                            "intrinsics": np.array(
                                [[256.0, 0.0, 128.0], [0.0, 256.0, 128.0], [0.0, 0.0, 1.0]],
                                dtype=np.float32,
                            ),
                            "depth_min": float(raw_depth.min()),
                            "depth_max": float(raw_depth.max()),
                            "depth_units": "meters",
                            "is_metric": True,
                            "model_variant": "mock_metric",
                        }

                self._client = SimpleMockDepthService()
                logger.info("Using simple mock depth estimation service")
        else:
            try:
                from skill3d.external_experts.Depth_AnythingV3.depth_client import DepthClient
                self._client = DepthClient(server_url=self.server_url)
                logger.info(f"Using real depth estimation service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import real depth client: {e}")
                raise
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": (
                        "The path to the input image for dense depth estimation. "
                        "Useful for nearest/farthest, front/back, and distance-ranking questions."
                    )
                },
                "crop_box": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional [x1, y1, x2, y2] region to summarize after detection when one target box matters."
                },
                "crop_boxes": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "Optional list of regions [[x1, y1, x2, y2], ...] for multi-object depth comparison."
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional region labels aligned with crop_boxes, e.g. ['chair', 'table']."
                },
                "point_coords": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "Optional list of precise pixel queries [[x, y], ...] for per-point metric depth and range."
                },
                "point_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional labels aligned with point_coords."
                },
                "point_pairs": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "description": "Optional list of point index pairs [[0, 1], ...] to compute 3D Euclidean distances between queried points."
                },
                "camera_intrinsics": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "Optional 3x3 camera intrinsics matrix. If omitted, DA3 will use its predicted intrinsics when available."
                }
            },
            "required": ["image_path"]
        }

    def _normalize_crop_box(
        self,
        crop_box: Sequence[float],
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int]:
        if len(crop_box) != 4:
            raise ValueError("crop_box must contain exactly 4 values: [x1, y1, x2, y2]")

        x1, y1, x2, y2 = [int(round(float(value))) for value in crop_box]
        left = max(0, min(x1, x2))
        top = max(0, min(y1, y2))
        right = min(image_width, max(x1, x2))
        bottom = min(image_height, max(y1, y2))
        if right <= left or bottom <= top:
            raise ValueError(f"Invalid crop_box after normalization: {[left, top, right, bottom]}")
        return left, top, right, bottom

    def _depth_stats(self, values: np.ndarray) -> Dict[str, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("No valid depth values available")
        return {
            "mean": round(float(np.mean(finite)), 4),
            "median": round(float(np.median(finite)), 4),
            "min": round(float(np.min(finite)), 4),
            "max": round(float(np.max(finite)), 4),
            "std": round(float(np.std(finite)), 4),
        }

    def _normalize_intrinsics(
        self,
        camera_intrinsics: Optional[Sequence[Sequence[float]]],
    ) -> Optional[np.ndarray]:
        if camera_intrinsics is None:
            return None
        intrinsics = np.asarray(camera_intrinsics, dtype=np.float32)
        if intrinsics.shape != (3, 3):
            raise ValueError("camera_intrinsics must be a 3x3 matrix")
        return intrinsics

    def _normalize_point(
        self,
        point: Sequence[float],
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int]:
        if len(point) != 2:
            raise ValueError("Each point must contain exactly 2 values: [x, y]")
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if x < 0 or x >= image_width or y < 0 or y >= image_height:
            raise ValueError(f"Point {point} is outside the image bounds")
        return x, y

    def _point_to_camera_geometry(
        self,
        x: int,
        y: int,
        depth_z: float,
        intrinsics: Optional[np.ndarray],
    ) -> Tuple[Optional[List[float]], Optional[float]]:
        if intrinsics is None or intrinsics.shape != (3, 3):
            return None, None

        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])
        if abs(fx) < 1e-6 or abs(fy) < 1e-6:
            return None, None

        x_cam = (x - cx) * depth_z / fx
        y_cam = (y - cy) * depth_z / fy
        z_cam = depth_z
        camera_distance = float(np.linalg.norm([x_cam, y_cam, z_cam]))
        return [round(float(x_cam), 4), round(float(y_cam), 4), round(float(z_cam), 4)], round(camera_distance, 4)

    def _measure_point(
        self,
        raw_depth: np.ndarray,
        point: Sequence[float],
        label: str,
        intrinsics: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        height, width = raw_depth.shape[:2]
        x, y = self._normalize_point(point, width, height)
        depth_z = float(raw_depth[y, x])
        if not np.isfinite(depth_z):
            raise ValueError(f"Depth at point {[x, y]} is invalid")

        camera_point, camera_distance = self._point_to_camera_geometry(x, y, depth_z, intrinsics)
        return {
            "label": label,
            "pixel": [x, y],
            "camera_axis_depth": round(depth_z, 4),
            "camera_point": camera_point,
            "camera_distance": camera_distance,
        }

    def _build_point_measurements(
        self,
        raw_depth: np.ndarray,
        intrinsics: Optional[np.ndarray],
        point_coords: Optional[Sequence[Sequence[float]]] = None,
        point_labels: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        measurements: List[Dict[str, Any]] = []
        point_labels = list(point_labels or [])
        for idx, point in enumerate(point_coords or []):
            label = point_labels[idx] if idx < len(point_labels) and point_labels[idx] else f"point_{idx + 1}"
            measurements.append(self._measure_point(raw_depth, point, label, intrinsics))
        return measurements

    def _build_point_pair_measurements(
        self,
        point_measurements: List[Dict[str, Any]],
        point_pairs: Optional[Sequence[Sequence[int]]],
    ) -> List[Dict[str, Any]]:
        pair_measurements: List[Dict[str, Any]] = []
        for idx, pair in enumerate(point_pairs or []):
            if len(pair) != 2:
                raise ValueError("Each point pair must contain exactly 2 point indices")
            left_idx = int(pair[0])
            right_idx = int(pair[1])
            if left_idx < 0 or left_idx >= len(point_measurements) or right_idx < 0 or right_idx >= len(point_measurements):
                raise ValueError(f"Point pair {pair} references an out-of-range point index")

            left = point_measurements[left_idx]
            right = point_measurements[right_idx]
            distance_3d = None
            if left.get("camera_point") is not None and right.get("camera_point") is not None:
                left_point = np.asarray(left["camera_point"], dtype=np.float32)
                right_point = np.asarray(right["camera_point"], dtype=np.float32)
                distance_3d = round(float(np.linalg.norm(left_point - right_point)), 4)

            pair_measurements.append(
                {
                    "label": f"pair_{idx + 1}",
                    "point_indices": [left_idx, right_idx],
                    "from_label": left.get("label"),
                    "to_label": right.get("label"),
                    "distance_3d": distance_3d,
                }
            )
        return pair_measurements

    def _format_depth_value(self, value: Optional[float], units: str) -> str:
        if value is None:
            return "unknown"
        suffix = " m" if units == "meters" else ""
        return f"{value:.3f}{suffix}"

    def _build_region_depth_stats(
        self,
        raw_depth: np.ndarray,
        intrinsics: Optional[np.ndarray],
        crop_box: Optional[Sequence[float]] = None,
        crop_boxes: Optional[Sequence[Sequence[float]]] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        height, width = raw_depth.shape[:2]
        requested_boxes: List[Sequence[float]] = []
        if crop_box is not None:
            requested_boxes.append(crop_box)
        if crop_boxes:
            requested_boxes.extend(list(crop_boxes))

        region_stats: List[Dict[str, Any]] = []
        labels = list(labels or [])
        for idx, box in enumerate(requested_boxes):
            left, top, right, bottom = self._normalize_crop_box(box, width, height)
            region = raw_depth[top:bottom, left:right]
            stats = self._depth_stats(region)
            finite_mask = np.isfinite(region)
            if not np.any(finite_mask):
                raise ValueError(f"No valid depth values inside crop_box {[left, top, right, bottom]}")

            safe_region = np.where(finite_mask, region, np.inf)
            nearest_local = np.unravel_index(int(np.argmin(safe_region)), safe_region.shape)
            farthest_local = np.unravel_index(int(np.argmax(np.where(finite_mask, region, -np.inf))), region.shape)

            center_x = int(round((left + right - 1) / 2.0))
            center_y = int(round((top + bottom - 1) / 2.0))
            nearest_x = left + int(nearest_local[1])
            nearest_y = top + int(nearest_local[0])
            farthest_x = left + int(farthest_local[1])
            farthest_y = top + int(farthest_local[0])
            label = labels[idx] if idx < len(labels) and labels[idx] else f"region_{idx + 1}"

            region_stats.append(
                {
                    "label": label,
                    "crop_box": [left, top, right, bottom],
                    "pixel_count": int(region.size),
                    "depth_mean": stats["mean"],
                    "depth_median": stats["median"],
                    "depth_min": stats["min"],
                    "depth_max": stats["max"],
                    "depth_std": stats["std"],
                    "center_measurement": self._measure_point(raw_depth, [center_x, center_y], f"{label}_center", intrinsics),
                    "nearest_surface_measurement": self._measure_point(raw_depth, [nearest_x, nearest_y], f"{label}_nearest_surface", intrinsics),
                    "farthest_surface_measurement": self._measure_point(raw_depth, [farthest_x, farthest_y], f"{label}_farthest_surface", intrinsics),
                }
            )
        return region_stats

    def _render_region_overlay(
        self,
        image_path: str,
        depth_array: Optional[np.ndarray],
        region_stats: List[Dict[str, Any]],
    ) -> Optional[str]:
        if depth_array is None or not region_stats:
            return None
        overlay = depth_array.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        for idx, item in enumerate(region_stats):
            left, top, right, bottom = item["crop_box"]
            color = (
                int((37 * (idx + 1)) % 255),
                int((91 * (idx + 2)) % 255),
                int((173 * (idx + 3)) % 255),
            )
            cv2.rectangle(overlay, (left, top), (right, bottom), color, 2)
            label = f"{item['label']}:{item['depth_median']:.2f}"
            cv2.putText(
                overlay,
                label,
                (left, max(18, top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = output_dir / f"depth_v3_regions_{Path(image_path).stem}.png"
        cv2.imwrite(str(overlay_path), overlay)
        return str(overlay_path)

    def _build_depth_description(
        self,
        units: str,
        global_stats: Optional[Dict[str, float]],
        region_stats: List[Dict[str, Any]],
        point_measurements: List[Dict[str, Any]],
        point_pair_measurements: List[Dict[str, Any]],
    ) -> str:
        if point_pair_measurements:
            pair_text = "; ".join(
                f"{item['from_label']} -> {item['to_label']}: {self._format_depth_value(item['distance_3d'], units)}"
                for item in point_pair_measurements
            )
            return (
                "Metric point-to-point measurement generated from the dense depth map. "
                f"3D distances: {pair_text}. "
                "camera_axis_depth is the optical-axis depth Z, while camera_distance is the Euclidean range from the camera center."
            )

        if point_measurements:
            point_text = "; ".join(
                f"{item['label']} depth={self._format_depth_value(item['camera_axis_depth'], units)}, "
                f"range={self._format_depth_value(item.get('camera_distance'), units)}"
                for item in point_measurements
            )
            return (
                "Metric point queries generated from the dense depth map. "
                f"{point_text}. "
                "camera_axis_depth is the optical-axis depth Z, while camera_distance is the Euclidean range from the camera center."
            )

        if region_stats:
            ranked = sorted(region_stats, key=lambda item: item["depth_median"])
            nearest = ranked[0]
            farthest = ranked[-1]
            ranking_text = " < ".join(
                f"{item['label']} ({self._format_depth_value(item['depth_median'], units)})"
                for item in ranked
            )
            return (
                "Structured object-level depth comparison generated from the dense depth map. "
                f"Near-to-far ranking by median depth: {ranking_text}. "
                f"Nearest region: {nearest['label']}. "
                f"Farthest region: {farthest['label']}. "
                "For each region, center_measurement and nearest_surface_measurement also provide metric camera_distance values."
            )

        if global_stats:
            return (
                "Dense depth evidence generated for the full image. "
                f"Estimated depth range: {self._format_depth_value(global_stats['min'], units)} to "
                f"{self._format_depth_value(global_stats['max'], units)}. "
                "Use this to reason about near/far ordering, front/back relations, or follow up with localized regions or point queries."
            )

        return (
            "Dense depth evidence generated. Use this result to compare near/far ordering, "
            "front/back relations, or candidate object distances after localizing the targets."
        )

    def call(
        self,
        image_path: str,
        crop_box: Optional[Sequence[float]] = None,
        crop_boxes: Optional[Sequence[Sequence[float]]] = None,
        labels: Optional[Sequence[str]] = None,
        point_coords: Optional[Sequence[Sequence[float]]] = None,
        point_labels: Optional[Sequence[str]] = None,
        point_pairs: Optional[Sequence[Sequence[int]]] = None,
        camera_intrinsics: Optional[Sequence[Sequence[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute depth estimation
        
        Args:
            image_path: Path to input image
            
        Returns:
            Depth estimation result dictionary
        """
        try:
            logger.info(f"Running depth estimation on: {image_path}")
            
            # Check if image exists
            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }

            normalized_intrinsics = self._normalize_intrinsics(camera_intrinsics)
            
            # Call depth estimation
            if hasattr(self._client, 'infer'):
                result = self._client.infer(
                    image_path,
                    return_raw_depth=True,
                    return_intrinsics=True,
                    camera_intrinsics=normalized_intrinsics,
                )
            else:
                # Fallback for different client interfaces
                result = self._client.process_image(image_path)
            
            if result and result.get('success'):
                logger.info("Depth estimation completed successfully")
                raw_depth = result.get("raw_depth")
                depth_units = result.get("depth_units", "relative")
                intrinsics = result.get("intrinsics")
                if intrinsics is None:
                    intrinsics = normalized_intrinsics
                global_stats = None
                region_stats: List[Dict[str, Any]] = []
                depth_ranking: List[str] = []
                nearest_region = None
                farthest_region = None
                vis_path = None
                point_measurements: List[Dict[str, Any]] = []
                point_pair_measurements: List[Dict[str, Any]] = []

                if isinstance(raw_depth, np.ndarray):
                    global_stats = self._depth_stats(raw_depth)
                    try:
                        region_stats = self._build_region_depth_stats(
                            raw_depth=raw_depth,
                            intrinsics=intrinsics,
                            crop_box=crop_box,
                            crop_boxes=crop_boxes,
                            labels=labels,
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to compute region depth stats: {exc}")
                        region_stats = []

                    try:
                        point_measurements = self._build_point_measurements(
                            raw_depth=raw_depth,
                            intrinsics=intrinsics,
                            point_coords=point_coords,
                            point_labels=point_labels,
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to compute point measurements: {exc}")
                        point_measurements = []

                    try:
                        point_pair_measurements = self._build_point_pair_measurements(
                            point_measurements=point_measurements,
                            point_pairs=point_pairs,
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to compute point-pair measurements: {exc}")
                        point_pair_measurements = []

                    if region_stats:
                        ranked = sorted(region_stats, key=lambda item: item["depth_median"])
                        depth_ranking = [item["label"] for item in ranked]
                        nearest_region = ranked[0]
                        farthest_region = ranked[-1]
                        vis_path = self._render_region_overlay(
                            image_path=image_path,
                            depth_array=result.get("depth_array"),
                            region_stats=region_stats,
                        )

                description = self._build_depth_description(
                    units=depth_units,
                    global_stats=global_stats,
                    region_stats=region_stats,
                    point_measurements=point_measurements,
                    point_pair_measurements=point_pair_measurements,
                )
                result_summary = {
                    "output_path": result.get("output_path"),
                    "depth_only_path": result.get("depth_only_path"),
                    "shape": result.get("shape"),
                    "depth_min": result.get("depth_min"),
                    "depth_max": result.get("depth_max"),
                    "depth_units": depth_units,
                    "is_metric": bool(result.get("is_metric")),
                    "model_variant": result.get("model_variant"),
                    "scale_factor": result.get("scale_factor"),
                }
                return {
                    "success": True,
                    "description": description,
                    "result": result_summary,
                    "output_path": result.get('output_path'),
                    "depth_only_path": result.get('depth_only_path'),
                    "vis_path": vis_path,
                    "shape": result.get('shape'),
                    "depth_units": depth_units,
                    "is_metric": bool(result.get("is_metric")),
                    "model_variant": result.get("model_variant"),
                    "camera_intrinsics": intrinsics.tolist() if isinstance(intrinsics, np.ndarray) else None,
                    "global_depth_stats": global_stats,
                    "region_depth_stats": region_stats,
                    "depth_ranking_near_to_far": depth_ranking,
                    "nearest_region": nearest_region,
                    "farthest_region": farthest_region,
                    "point_measurements": point_measurements,
                    "point_pair_distances": point_pair_measurements,
                    "measurement_reference_note": (
                        "camera_axis_depth is depth along the optical axis; camera_distance is Euclidean range from the camera center."
                    ),
                    "raw_depth_available": isinstance(raw_depth, np.ndarray),
                }
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.error(f"Depth estimation failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"Depth estimation failed: {error_msg}"
                }
                
        except Exception as e:
            logger.error(f"Depth estimation tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            } 
