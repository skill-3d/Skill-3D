"""
Object Detection Tool

This module contains the ObjectDetectionTool that wraps
GroundingDINO functionality for the SPAgent system.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from skill3d.core.tool import Tool

logger = logging.getLogger(__name__)


class ObjectDetectionTool(Tool):
    """Tool for object detection using GroundingDINO"""
    
    def __init__(self, use_mock: bool = True, server_url: str = "http://10.8.131.51:30969"):
        """
        Initialize object detection tool
        
        Args:
            use_mock: Whether to use mock client for testing
            server_url: URL of the GroundingDINO server
        """
        super().__init__(
            name="detect_objects_tool",
            description=(
                "Detect and locate objects in the image using GroundingDINO for open-vocabulary object detection. "
                "Returns bounding boxes that can be reused by downstream tools such as depth_estimation_tool, "
                "swinir_tool, or orient_anything_tool via crop_box-style arguments."
            )
        )
        
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the GroundingDINO client"""
        if self.use_mock:
            try:
                from skill3d.external_experts.GroundingDINO.mock_gdino_service import MockGroundingDINOService
                self._client = MockGroundingDINOService()
                logger.info("Using mock GroundingDINO service")
            except ImportError:
                # Fallback to creating a simple mock
                class SimpleMockGDINO:
                    def infer_image(self, image_path, text_prompt, **kwargs):
                        return {
                            "success": True,
                            "boxes": [[100, 100, 200, 200]],
                            "labels": ["object"],
                            "confidence": [0.8],
                            "vis_path": f"outputs/gdino_mock_{Path(image_path).stem}.jpg"
                        }
                self._client = SimpleMockGDINO()
                logger.info("Using simple mock GroundingDINO service")
        else:
            try:
                from skill3d.external_experts.GroundingDINO.grounding_dino_client import GroundingDINOClient
                self._client = GroundingDINOClient(server_url=self.server_url)
                logger.info(f"Using real GroundingDINO service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import real GroundingDINO client: {e}")
                raise
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the input image for object detection."
                },
                "text_prompt": {
                    "type": "string",
                    "description": "Text description of objects to detect."
                },
                "box_threshold": {
                    "type": "number",
                    "description": "Confidence threshold for box detection.",
                    "default": 0.35
                },
                "text_threshold": {
                    "type": "number",
                    "description": "Confidence threshold for text matching.",
                    "default": 0.25
                }
            },
            "required": ["image_path", "text_prompt"]
        }
    
    def call(
        self, 
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Execute object detection
        
        Args:
            image_path: Path to input image
            text_prompt: Text description of objects to detect
            box_threshold: Confidence threshold for box detection
            text_threshold: Confidence threshold for text matching
            
        Returns:
            Object detection result dictionary
        """
        try:
            logger.info(f"Running object detection on: {image_path} with prompt: {text_prompt}")
            
            # Check if image exists
            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }
            
            # Call object detection
            if hasattr(self._client, 'infer_image'):
                result = self._client.infer_image(
                    image_path=image_path,
                    text_prompt=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
            elif hasattr(self._client, 'infer'):
                result = self._client.infer(
                    image_path=image_path,
                    text_prompt=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
            else:
                # Fallback for different client interfaces
                result = self._client.detect(
                    image_path=image_path,
                    text_prompt=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
            
            if result and result.get('success'):
                logger.info("Object detection completed successfully")
                detections = result.get("detections", []) if isinstance(result, dict) else []
                boxes = result.get('boxes', []) if isinstance(result, dict) else []
                labels = result.get('labels', []) if isinstance(result, dict) else []
                confidence = result.get('confidence', []) if isinstance(result, dict) else []

                if detections and not boxes:
                    boxes = [item.get("bbox") for item in detections if isinstance(item, dict) and item.get("bbox") is not None]
                if detections and not labels:
                    labels = [item.get("label") for item in detections if isinstance(item, dict)]
                if detections and not confidence:
                    confidence = [item.get("confidence") for item in detections if isinstance(item, dict)]

                return {
                    "success": True,
                    "result": result,
                    "detections": detections,
                    "boxes": boxes,
                    "labels": labels,
                    "confidence": confidence,
                    "vis_path": result.get('vis_path'),
                    "output_path": result.get('output_path')
                }
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.error(f"Object detection failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"Object detection failed: {error_msg}"
                }
                
        except Exception as e:
            logger.error(f"Object detection tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            } 
