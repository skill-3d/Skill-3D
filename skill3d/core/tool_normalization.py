"""
Tool call normalization helpers.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

from .tool import Tool, ToolRegistry


def parse_tool_call_block(block: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single <tool_call> JSON-like block with fallbacks for common formatting issues.
    """
    try:
        parsed = json.loads(block)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(block)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass

    fixed = re.sub(r"\bTrue\b", "true", block)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    try:
        parsed = json.loads(fixed)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    repaired = _parse_nonstandard_tool_call_block(block)
    if repaired is not None:
        return repaired

    return None


def _split_top_level_parts(text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    in_single = False
    in_double = False
    prev = ""
    for char in text:
        if char == "'" and not in_double and prev != "\\":
            in_single = not in_single
        elif char == '"' and not in_single and prev != "\\":
            in_double = not in_double
        elif not in_single and not in_double:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
            elif char == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                prev = char
                continue
        current.append(char)
        prev = char
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _coerce_value(value: str) -> Any:
    raw = str(value or "").strip()
    if not raw:
        return ""
    fixed = re.sub(r"\btrue\b", "True", raw, flags=re.IGNORECASE)
    fixed = re.sub(r"\bfalse\b", "False", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\bnull\b", "None", fixed, flags=re.IGNORECASE)
    try:
        return ast.literal_eval(fixed)
    except Exception:
        pass
    quoted = raw.strip('"').strip("'").strip()
    number_match = re.fullmatch(r"-?\d+(?:\.\d+)?", quoted)
    if number_match:
        return float(quoted) if "." in quoted else int(quoted)
    return quoted


def _default_args_from_freeform(tool_name: str, value: str) -> Dict[str, Any]:
    text = str(value or "").strip().strip('"').strip("'").strip()
    if not text:
        return {}
    if tool_name == "detect_objects_tool":
        return {"text_prompt": text}
    if tool_name == "moondream_tool":
        return {"task": "point", "object_name": text}
    if tool_name == "swinir_tool":
        return {"task": "real_sr", "target_object": text}
    if tool_name == "pi3_tool":
        camera_match = re.search(r"(?:cam|camera|image)[_ ]?(\d+)", text, flags=re.IGNORECASE)
        angle_match = re.search(r"-?\d+(?:\.\d+)?", text)
        args: Dict[str, Any] = {}
        if camera_match:
            args["rotation_reference_camera"] = int(camera_match.group(1))
        if angle_match:
            args["azimuth_angle"] = _coerce_value(angle_match.group(0))
            args.setdefault("elevation_angle", 0)
        return args
    return {}


def _parse_argument_items(items: List[str], tool_name: str) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for item in items:
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            args[key.strip()] = _coerce_value(value)
            continue
        if ":" in item:
            key, value = item.split(":", 1)
            args[key.strip().strip('"').strip("'")] = _coerce_value(value)
            continue

        args.update(_default_args_from_freeform(tool_name, item))
    return args


def _parse_nonstandard_tool_call_block(block: str) -> Optional[Dict[str, Any]]:
    text = str(block or "").strip()
    if not text:
        return None

    bare_brace_match = re.fullmatch(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}", text)
    if bare_brace_match:
        return {"name": bare_brace_match.group(1), "arguments": {}}

    bare_match = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text)
    if bare_match:
        return {"name": text, "arguments": {}}

    fn_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", text, re.DOTALL)
    if fn_match:
        tool_name = fn_match.group(1).strip()
        body = fn_match.group(2).strip()
        args = _parse_argument_items(_split_top_level_parts(body), tool_name)
        return {"name": tool_name, "arguments": args}

    colon_match = re.fullmatch(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)\s*\}", text, re.DOTALL)
    if colon_match:
        tool_name = colon_match.group(1).strip()
        body = colon_match.group(2).strip()
        args = _parse_argument_items(_split_top_level_parts(body), tool_name)
        return {"name": tool_name, "arguments": args}

    return None

    return None


def normalize_tool_call_payload(
    payload: Dict[str, Any],
    registry: Optional[ToolRegistry] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize JSON-ish tool payloads into {'name': ..., 'arguments': ...}.

    This accepts the canonical format plus a few common malformed variants
    produced by models in the wild, for example:
    - {"tool": {"name": "...", "params": {...}}}
    - {"tool": "segment_image_tool", "image": "..."}
    - {"image": "...", "text_prompt": "segment detected tv"}
    """
    if not isinstance(payload, dict):
        return None

    registry = registry or ToolRegistry()

    def _normalize_name(name: Any) -> str:
        raw = str(name or "").strip()
        return normalize_tool_name(raw, registry) if raw else raw

    if "name" in payload:
        args = payload.get("arguments")
        if args is None:
            args = payload.get("parameters")
        if isinstance(args, dict):
            return {"name": _normalize_name(payload.get("name")), "arguments": dict(args)}

    tool_blob = payload.get("tool")
    if isinstance(tool_blob, dict):
        name = tool_blob.get("name")
        params = tool_blob.get("params")
        if name:
            return {
                "name": _normalize_name(name),
                "arguments": dict(params) if isinstance(params, dict) else {},
            }

    if isinstance(tool_blob, str):
        arguments = {key: value for key, value in payload.items() if key != "tool"}
        return {"name": _normalize_name(tool_blob), "arguments": arguments}

    name = payload.get("tool_name")
    params = payload.get("params")
    if params is None:
        params = payload.get("parameters")
    if name:
        return {
            "name": _normalize_name(name),
            "arguments": dict(params) if isinstance(params, dict) else {},
        }

    # Heuristic fallback: infer a segmentation call from image + segment-like text.
    image_value = None
    for key in ("image_path", "image", "img", "path"):
        if key in payload:
            image_value = payload[key]
            break
    text_value = None
    for key in ("text_prompt", "prompt", "instruction", "query"):
        if key in payload and isinstance(payload[key], str):
            text_value = payload[key]
            break

    if image_value is not None and text_value:
        normalized_text = text_value.strip().lower()
        if normalized_text.startswith("segment ") or "segment detected" in normalized_text or "segmentation" in normalized_text:
            return {
                "name": _normalize_name("segment_image_tool"),
                "arguments": {"image_path": image_value},
            }

    return None


def _norm_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalize_tool_name(tool_name: str, registry: ToolRegistry) -> str:
    if not tool_name:
        return tool_name
    raw = tool_name.strip()
    lower = raw.lower()
    alias_map = {
        "pi3_tool": "pi3_tool",
        "pi3_3d_tool": "pi3_tool",
        "pi3_3d_reconstruction_tool": "pi3_tool",
        "pi3_3d_reconstruction": "pi3_tool",
        "pi3_reconstruction_tool": "pi3_tool",
        "pi3_tool_render": "pi3_tool",
        "3d_reconstruction_tool": "pi3_tool",
        "3d_reconstruction": "pi3_tool",
        "reconstruct_scene_tool": "pi3_tool",
        "reconstruct_3d_scene": "pi3_tool",
        "explore_3d_structure_tool": "pi3_tool",
        "analyze_structure_tool": "pi3_tool",
        "analyze_scene_structure_tool": "pi3_tool",
        "visualize_scene_tool": "pi3_tool",
        "change_view_angle_tool": "pi3_tool",
        "rotation_tool": "pi3_tool",
        "rotate_scene_tool": "pi3_tool",
        "3d_view_tool": "pi3_tool",
        "detect_objects_tool": "detect_objects_tool",
        "object_detection_tool": "detect_objects_tool",
        "depth_estimation_tool": "depth_estimation_tool",
        "segment_image_tool": "segment_image_tool",
        "segmentation_tool": "segment_image_tool",
        "yoloe_tool": "yoloe_tool",
        "swinir_tool": "swinir_tool",
        "super_resolution_tool": "swinir_tool",
        "image_enhancement_tool": "swinir_tool",
        "image_restoration_tool": "swinir_tool",
        "orient_anything_tool": "orient_anything_tool",
        "orientation_tool": "orient_anything_tool",
        "pose_estimation_tool": "orient_anything_tool",
        "rotation_estimation_tool": "orient_anything_tool",
    }
    if raw in alias_map:
        return alias_map[raw]
    if lower in alias_map:
        return alias_map[lower]

    for registered in registry.list_tools():
        if registered.lower() == lower:
            return registered

    norm_raw = _norm_token(raw)
    for registered in registry.list_tools():
        if _norm_token(registered) == norm_raw:
            return registered

    if "pi3" in lower or "3d" in lower:
        if registry.get("pi3_tool"):
            return "pi3_tool"
    if "detect" in lower or "dino" in lower:
        if registry.get("detect_objects_tool"):
            return "detect_objects_tool"
    if "depth" in lower:
        if registry.get("depth_estimation_tool"):
            return "depth_estimation_tool"
    if "segment" in lower or "mask" in lower:
        if registry.get("segment_image_tool"):
            return "segment_image_tool"
    if "super" in lower or "restore" in lower or "enhance" in lower:
        if registry.get("swinir_tool"):
            return "swinir_tool"
    if "orient" in lower or "rotation" in lower or "pose" in lower:
        if registry.get("orient_anything_tool"):
            return "orient_anything_tool"

    return raw


def _extract_angle_pair(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(numbers) >= 2:
            return [float(numbers[0]), float(numbers[1])]
    return None


def normalize_tool_arguments(tool: Tool, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}
    args = dict(arguments)

    for key, val in list(args.items()):
        if isinstance(val, str) and val.lower() in {"true", "false"}:
            args[key] = val.lower() == "true"

    if "image_path" not in args:
        for alt in ["image_paths", "images", "image", "img_path", "img", "path"]:
            if alt in args:
                args["image_path"] = args.pop(alt)
                break

    if tool.name == "pi3_tool":
        if "azimuth_angle" not in args:
            for alt in ["azimuth", "angle_azimuth", "azim", "azimuth_deg"]:
                if alt in args:
                    args["azimuth_angle"] = args.pop(alt)
                    break
        if "elevation_angle" not in args:
            for alt in ["elevation", "angle_elevation", "elev", "elevation_deg"]:
                if alt in args:
                    args["elevation_angle"] = args.pop(alt)
                    break

        for alt in [
            "view_angle",
            "viewing_angle",
            "angle",
            "angles",
            "rotation_angle",
            "new_view_angle",
            "camera_angle",
        ]:
            if alt in args:
                angle_pair = _extract_angle_pair(args.pop(alt))
                if angle_pair:
                    args.setdefault("azimuth_angle", angle_pair[0])
                    args.setdefault("elevation_angle", angle_pair[1])

        if "rotation_reference_camera" not in args:
            for alt in ["camera", "camera_id", "ref_camera", "rotation_camera"]:
                if alt in args:
                    try:
                        args["rotation_reference_camera"] = int(args.pop(alt))
                    except (TypeError, ValueError):
                        args.pop(alt, None)
                    break

        if "camera_view" in args and isinstance(args["camera_view"], str):
            args["camera_view"] = args["camera_view"].lower() == "true"

        if "image_path" in args and isinstance(args["image_path"], str):
            args["image_path"] = [args["image_path"]]

    elif tool.name == "detect_objects_tool":
        if "text_prompt" not in args:
            for alt in [
                "prompt",
                "text",
                "labels",
                "query",
                "object",
                "objects",
                "object_name",
                "target_object",
                "target_objects",
                "categories",
                "category_names",
                "what",
            ]:
                if alt in args:
                    value = args.pop(alt)
                    if isinstance(value, list):
                        args["text_prompt"] = ", ".join(str(item) for item in value if str(item).strip())
                    else:
                        args["text_prompt"] = value
                    break
        if "image_path" in args and isinstance(args["image_path"], list):
            if args["image_path"]:
                args["image_path"] = args["image_path"][0]

    elif tool.name in {"depth_estimation_tool", "segment_image_tool", "yoloe_tool"}:
        if tool.name == "segment_image_tool" and "box" not in args:
            for alt in ["bbox", "bounding_box", "crop_box"]:
                if alt in args:
                    args["box"] = args.pop(alt)
                    break
        if "image_path" in args and isinstance(args["image_path"], list):
            if args["image_path"]:
                args["image_path"] = args["image_path"][0]
    elif tool.name == "swinir_tool":
        if "crop_box" not in args:
            for alt in ["box", "bbox", "reference_box", "reference_bbox", "crop"]:
                if alt in args:
                    args["crop_box"] = args.pop(alt)
                    break
        if "task" not in args:
            for alt in ["mode", "enhancement_task", "enhancement_mode"]:
                if alt in args:
                    args["task"] = args.pop(alt)
                    break
        if "scale" not in args:
            for alt in ["upscale", "upscale_factor", "sr_scale"]:
                if alt in args:
                    args["scale"] = args.pop(alt)
                    break
        if "image_path" in args and isinstance(args["image_path"], list):
            if args["image_path"]:
                args["image_path"] = args["image_path"][0]
    elif tool.name == "orient_anything_tool":
        if "image_path" not in args:
            for alt in ["reference_image_path", "reference_path", "ref_image_path"]:
                if alt in args:
                    args["image_path"] = args.pop(alt)
                    break
        if "crop_box" not in args:
            for alt in ["box", "bbox", "reference_box", "reference_bbox", "crop"]:
                if alt in args:
                    args["crop_box"] = args.pop(alt)
                    break
        if "target_image_path" not in args:
            for alt in ["target_path", "comparison_image_path", "second_image_path", "image_path_2"]:
                if alt in args:
                    args["target_image_path"] = args.pop(alt)
                    break
        if "target_crop_box" not in args:
            for alt in ["target_box", "target_bbox", "comparison_box", "comparison_bbox"]:
                if alt in args:
                    args["target_crop_box"] = args.pop(alt)
                    break
        if "image_path" in args and isinstance(args["image_path"], list):
            if args["image_path"]:
                args["image_path"] = args["image_path"][0]

    properties = tool.parameters.get("properties", {})
    if properties:
        args = {k: v for k, v in args.items() if k in properties}

    return args
