"""
Segmentation Tool

This module contains the SegmentationTool that wraps
SAM3 functionality for the Skill-3D system.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from skill3d.core.tool import Tool

logger = logging.getLogger(__name__)


class SegmentationTool(Tool):
    """Tool for image segmentation using SAM3"""
    
    def __init__(self, use_mock: bool = True, server_url: str = "http://127.0.0.1:20020"):
        """
        Initialize segmentation tool
        
        Args:
            use_mock: Whether to use mock client for testing
            server_url: URL of the SAM3 server
        """
        super().__init__(
            name="segment_image_tool",
            description="Segment objects in the image based on user's request. Can use points, boxes to guide segmentation."
        )
        
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the SAM3 client"""
        if self.use_mock:
            try:
                from skill3d.external_experts.SAM3.mock_sam3_service import MockSAM3Service
                self._client = MockSAM3Service()
                logger.info("Using mock SAM3 service")
            except ImportError:
                # Fallback to creating a simple mock
                class SimpleMockSAM3:
                    def infer(self, image_path, **kwargs):
                        stem = Path(image_path).stem
                        
                        # 模拟多个掩码（2-4个随机数量的对象）
                        import random
                        num_objects = random.randint(2, 4)
                        masks_data = []
                        
                        for i in range(num_objects):
                            masks_data.append({
                                'mask': f'mock_mask_data_{i}',
                                'id': i
                            })
                        
                        return {
                            "success": True,
                            "output_path": f"outputs/sam3_combined_{stem}.jpg",
                            "overlay_path": f"outputs/sam3_overlay_{stem}.jpg",
                            "mask_path": f"outputs/sam3_mask_{stem}.png",
                            "vis_path": f"outputs/sam3_mock_{stem}.jpg",  # Backward compatibility
                            "masks": masks_data,  # 多个掩码支持随机颜色
                            "shape": [1024, 1024]
                        }
                self._client = SimpleMockSAM3()
                logger.info("Using simple mock SAM3 service")
        else:
            try:
                from skill3d.external_experts.SAM3.sam3_client import SAM3Client
                self._client = SAM3Client(server_url=self.server_url)
                logger.info(f"Using real SAM3 service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import real SAM3 client: {e}")
                raise
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the input image for segmentation."
                },
                "point_coords": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "Optional list of point coordinates [[x1,y1], [x2,y2], ...]"
                },
                "point_labels": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional list of point labels (1 for foreground, 0 for background)"
                },
                "box": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional bounding box coordinates [x1,y1,x2,y2]"
                }
            },
            "required": ["image_path"]
        }
    
    def call(
        self, 
        image_path: str,
        point_coords: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        box: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Execute image segmentation
        
        Args:
            image_path: Path to input image
            point_coords: Optional point coordinates for guided segmentation
            point_labels: Optional point labels (1=foreground, 0=background)
            box: Optional bounding box coordinates
            
        Returns:
            Segmentation result dictionary
        """
        try:
            logger.info(f"Running segmentation on: {image_path}")
            
            # Check if image exists
            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }
            
            # Prepare arguments for segmentation
            prompt_args = {}
            
            if point_coords is not None:
                prompt_args["point_coords"] = point_coords
            if point_labels is not None:
                prompt_args["point_labels"] = point_labels
            if box is not None:
                prompt_args["box"] = box
            
            # Call segmentation. The SAM3 client supports text/box prompts; this
            # wrapper exposes box prompts for tool compatibility.
            if hasattr(self._client, 'infer'):
                try:
                    result = self._client.infer(image_path=image_path, box=box)
                except TypeError:
                    result = self._client.infer(image_path=image_path, prompts=prompt_args or None)
            else:
                # Fallback for different client interfaces
                seg_args = {"image_path": image_path}
                seg_args.update(prompt_args)
                result = self._client.segment(**seg_args)
            
            if result and result.get('success'):
                logger.info("Segmentation completed successfully")
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get('output_path'),  # Combined image
                    "overlay_path": result.get('overlay_path'),  # Mask visualization
                    "mask_path": result.get('mask_path'),  # Original mask
                    "vis_path": result.get('vis_path'),  # Backward compatibility
                    "shape": result.get('shape'),
                    "masks": result.get('masks', [])
                }
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.error(f"Segmentation failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"Segmentation failed: {error_msg}"
                }
                
        except Exception as e:
            logger.error(f"Segmentation tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            } 
