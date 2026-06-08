import sys
from pathlib import Path
import cv2
from typing import List

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from skill3d.core.tool import Tool, ToolRegistry
from skill3d.core.tool_normalization import (
    parse_tool_call_block,
    normalize_tool_name,
    normalize_tool_arguments,
)


class DummyTool(Tool):
    def __init__(self, name, parameters):
        super().__init__(name=name, description="dummy")
        self._parameters = parameters

    @property
    def parameters(self):
        return self._parameters

    def call(self, **kwargs):
        return {"success": True, "result": kwargs}


def main():
    # parse_tool_call_block
    block = '{"name": "pi3_tool", "arguments": {"azimuth": 45, "elevation": 30}}'
    parsed = parse_tool_call_block(block)
    assert isinstance(parsed, dict)
    assert parsed["name"] == "pi3_tool"

    block = "{'name': 'detect_objects_tool', 'arguments': {'camera_view': False}}"
    parsed = parse_tool_call_block(block)
    assert isinstance(parsed, dict)
    assert parsed["arguments"]["camera_view"] is False

    # normalize_tool_name
    registry = ToolRegistry()
    registry.register(DummyTool("pi3_tool", {"type": "object", "properties": {}}))
    registry.register(DummyTool("detect_objects_tool", {"type": "object", "properties": {}}))
    assert normalize_tool_name("Pi3_3D_Tool", registry) == "pi3_tool"
    assert normalize_tool_name("OBJECT_DETECTION_TOOL", registry) == "detect_objects_tool"

    # normalize_tool_arguments: pi3
    pi3_tool = DummyTool(
        "pi3_tool",
        {
            "type": "object",
            "properties": {
                "image_path": {"type": "list"},
                "azimuth_angle": {"type": "number"},
                "elevation_angle": {"type": "number"},
                "rotation_reference_camera": {"type": "integer"},
                "camera_view": {"type": "boolean"},
            },
        },
    )
    args = {
        "image_path": "temp_frames/frame_0.jpg",
        "angle": "(45, 30)",
        "camera": "2",
        "camera_view": "true",
        "extra": "drop_me",
    }
    norm = normalize_tool_arguments(pi3_tool, args)
    assert norm["image_path"] == ["temp_frames/frame_0.jpg"]
    assert norm["azimuth_angle"] == 45.0
    assert norm["elevation_angle"] == 30.0
    assert norm["rotation_reference_camera"] == 2
    assert norm["camera_view"] is True
    assert "extra" not in norm

    # normalize_tool_arguments: detect
    detect_tool = DummyTool(
        "detect_objects_tool",
        {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "text_prompt": {"type": "string"},
                "box_threshold": {"type": "number"},
            },
        },
    )
    args = {
        "image_paths": ["a.jpg", "b.jpg"],
        "prompt": "chair",
        "box_threshold": 0.1,
    }
    norm = normalize_tool_arguments(detect_tool, args)
    assert norm["image_path"] == "a.jpg"
    assert norm["text_prompt"] == "chair"
    assert norm["box_threshold"] == 0.1

    # normalize_tool_arguments: depth
    depth_tool = DummyTool(
        "depth_estimation_tool",
        {
            "type": "object",
            "properties": {"image_path": {"type": "string"}},
        },
    )
    args = {"image_path": ["only.jpg"]}
    norm = normalize_tool_arguments(depth_tool, args)
    assert norm["image_path"] == "only.jpg"

    print("tool_normalization checks: OK")


if __name__ == "__main__":
    main()
