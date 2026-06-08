"""
Prompt templates with lightweight skill references.
"""

from typing import List, Dict, Any, Optional
import json
import os
import re

from .skill_normalization import build_default_skills, skills_to_json


_TOOL_CARD_SUMMARIES = {
    "detect_objects_tool": "Open-vocabulary detection for reusable boxes. Use it when category-level localization or crop reuse is needed, not as the universal default first step.",
    "supervision_tool": "Fast scene overview or visualization aid.",
    "swinir_tool": "Image super-resolution and restoration for tiny, blurry, far-away, or low-resolution regions. Prefer it on a localized crop before concluding from unreadable local detail.",
    "segment_image_tool": "Pixel-level masks for boundaries, contact regions, and closest-point reasoning.",
    "depth_estimation_tool": "Dense metric depth with crop_box/crop_boxes or point_coords/point_pairs for nearest/farthest, front/back, and distance checks. Use only schema-listed arguments.",
    "orient_anything_tool": "Pose and orientation prediction for facing direction, front/back interpretation, axis alignment, and relative rotation. Use it on a detected target crop, not on a cluttered full image.",
    "pi3_tool": "Heavy 3D viewpoint / geometry / layout check. Delay it until direct 2D evidence remains insufficient unless occlusion, viewpoint change, or overall scene layout is the real blocker.",
}

_TOOL_PARAMETER_HINTS = {
    "detect_objects_tool": {
        "image_path": "Input image to run open-vocabulary detection on.",
        "text_prompt": "Object names or phrases to detect.",
        "box_threshold": "Box confidence threshold.",
        "text_threshold": "Text-to-box matching threshold.",
    },
    "supervision_tool": {
        "image_path": "Input image to visualize quickly.",
        "task": "Use 'image_det' for boxes or 'image_seg' for segmentation overlays.",
    },
    "depth_estimation_tool": {
        "image_path": "Input image for dense metric depth estimation. Strong for nearest/farthest, front/back, and metric distance questions; combine with detection when target objects must be localized first.",
        "crop_box": "Optional [x1, y1, x2, y2] single region for object-level depth statistics after detection.",
        "crop_boxes": "Optional list of boxes for multi-object depth comparison in one call.",
        "labels": "Optional labels aligned with crop_boxes, used in the returned near-to-far ranking summary.",
        "point_coords": "Optional pixel queries [[x, y], ...] for precise per-point depth/range measurement.",
        "point_labels": "Optional labels aligned with point_coords.",
        "point_pairs": "Optional point index pairs [[0, 1], ...] to compute 3D Euclidean distances between queried points.",
        "camera_intrinsics": "Optional 3x3 camera intrinsics matrix if known; improves metric 3D distance measurement.",
    },
    "pi3_tool": {
        "image_path": "Input images that define the scene to reconstruct.",
        "azimuth_angle": "Requested azimuth for a new view in degrees.",
        "elevation_angle": "Requested elevation for a new view in degrees.",
        "rotation_reference_camera": "1-based input camera index used as the rotation reference.",
        "camera_view": "True for ego-view from that camera, False for global view.",
    },
    "segment_image_tool": {
        "image_path": "Input image to segment.",
        "point_coords": "Optional [[x, y], ...] guidance points.",
        "point_labels": "Labels for the guidance points: 1 foreground, 0 background.",
        "box": "Optional [x1, y1, x2, y2] box prompt for segmentation.",
    },
    "swinir_tool": {
        "image_path": "Input image to enhance or super-resolve.",
        "crop_box": "Preferred [x1, y1, x2, y2] crop for local enhancement after detection. Use a target crop when the low-quality issue is local rather than global.",
        "crop_margin": "Optional pixel margin added around crop_box.",
        "task": "Enhancement mode such as real_sr, denoising, or JPEG restoration.",
        "scale": "Upscale factor for super-resolution modes.",
        "large_model": "Whether to use the larger real_sr checkpoint.",
        "tile": "Optional tile size for large-image inference.",
        "tile_overlap": "Tile overlap when tiled inference is used.",
        "noise": "Noise level for denoising checkpoints.",
        "jpeg": "JPEG quality setting for JPEG restoration checkpoints.",
    },
    "orient_anything_tool": {
        "image_path": "Reference image of the object to orient.",
        "crop_box": "Required [x1, y1, x2, y2] crop for the target object. Detect or localize the object first, then pass its crop.",
        "crop_margin": "Optional pixel margin added around crop_box.",
        "target_image_path": "Optional second image of the same object for relative rotation.",
        "target_crop_box": "Required crop for the target image when comparing two views of the same object.",
        "remove_background": "Whether to remove background before orientation inference.",
        "render_visualization": "Set true to render the axis overlay on the image.",
    },
}


def _tool_name_from_schema(tool_schema: Dict[str, Any]) -> str:
    function = tool_schema.get("function", {}) if isinstance(tool_schema, dict) else {}
    return str(function.get("name", "")).strip()


def _collapse_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _first_sentence(text: Any) -> str:
    normalized = _collapse_text(text)
    if not normalized:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)
    return parts[0]


def _compact_parameter_schema(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(parameters, dict):
        return parameters

    compact = {
        key: value
        for key, value in parameters.items()
        if key != "properties"
    }
    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        compact["properties"] = properties
        return compact

    compact_properties: Dict[str, Any] = {}
    tool_overrides = _TOOL_PARAMETER_HINTS.get(tool_name, {})
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            compact_properties[prop_name] = prop_schema
            continue
        prop_copy = dict(prop_schema)
        short_description = tool_overrides.get(prop_name) or _first_sentence(prop_copy.get("description"))
        if short_description:
            prop_copy["description"] = short_description
        compact_properties[prop_name] = prop_copy

    compact["properties"] = compact_properties
    return compact


def _compact_tool_schema(tool_schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tool_schema, dict):
        return tool_schema
    function = tool_schema.get("function", {})
    if not isinstance(function, dict):
        return tool_schema

    tool_name = str(function.get("name", "")).strip()
    compact_function = dict(function)
    compact_function["description"] = _TOOL_CARD_SUMMARIES.get(tool_name) or _first_sentence(function.get("description"))
    compact_function["parameters"] = _compact_parameter_schema(tool_name, function.get("parameters", {}))

    compact_schema = dict(tool_schema)
    compact_schema["function"] = compact_function
    return compact_schema


def _build_tool_cards(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cards = []
    for tool in tools or []:
        tool_name = _tool_name_from_schema(tool)
        if not tool_name:
            continue
        cards.append(
            {
                "name": tool_name,
                "summary": _TOOL_CARD_SUMMARIES.get(tool_name) or _first_sentence(tool.get("function", {}).get("description")),
            }
        )
    return cards


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _compact_text(value: Any, limit: int = 360) -> str:
    text = _collapse_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _skill_id(skill: Dict[str, Any]) -> str:
    return str(skill.get("id") or skill.get("skill_id") or skill.get("name") or "").strip()


def _skill_source(skill: Dict[str, Any]) -> str:
    source = str(skill.get("source") or skill.get("source_type") or "").strip().lower()
    if source:
        return source
    skill_id = _skill_id(skill)
    return "static" if skill_id.startswith("seed::") else "dynamic"


def _skill_tools(skill: Dict[str, Any]) -> List[str]:
    tools = skill.get("required_tools") or skill.get("tools") or skill.get("tool_candidates") or []
    return [str(tool) for tool in _as_list(tools) if str(tool).strip()]


def _skill_score(skill: Dict[str, Any]) -> Optional[float]:
    return _safe_float(skill.get("retrieval_score") or skill.get("score"))


def _rank_scene_task_skills(skill_bundle: Dict[str, Any], top_k: int = 6) -> List[Dict[str, Any]]:
    """Merge retrieved and focus skills into the compact candidate set used by the paper-style prompt."""
    top_k = max(1, int(top_k or 6))
    sections = [
        ("retrieved_skills", skill_bundle.get("retrieved_skills", []), 0),
        ("focus_skills", skill_bundle.get("focus_skills", []), 1),
        ("skill_catalog", skill_bundle.get("skill_catalog", []), 2),
    ]
    ranked = []
    seen = set()
    fallback_rank = 0
    for section, values, section_priority in sections:
        for raw_skill in _as_list(values):
            if not isinstance(raw_skill, dict):
                continue
            skill_id = _skill_id(raw_skill)
            if not skill_id or skill_id in seen:
                continue
            seen.add(skill_id)
            skill = dict(raw_skill)
            skill["_source_section"] = section
            score = _skill_score(skill)
            score_key = -score if score is not None else 0.0
            ranked.append(((section_priority, score_key, fallback_rank), skill))
            fallback_rank += 1
    ranked.sort(key=lambda item: item[0])
    return [skill for _, skill in ranked[:top_k]]


def _compact_skill_candidate(skill: Dict[str, Any], rank: int) -> Dict[str, Any]:
    meta = skill.get("observed_support") if isinstance(skill.get("observed_support"), dict) else {}
    fallbacks = [fb for fb in _as_list(skill.get("fallbacks")) if isinstance(fb, dict)]
    failure_memory = skill.get("failure_memory") if isinstance(skill.get("failure_memory"), dict) else {}
    candidate = {
        "rank": rank,
        "id": _skill_id(skill),
        "name": skill.get("name") or skill.get("title") or _skill_id(skill),
        "source": _skill_source(skill),
        "trigger": _compact_text(skill.get("when") or skill.get("trigger") or skill.get("when_to_apply"), 280),
        "required_tools": _skill_tools(skill)[:6],
        "strategy": [_compact_text(item, 260) for item in _as_list(skill.get("strategy") or skill.get("decision_logic"))[:4]],
        "preconditions": [_compact_text(item, 220) for item in _as_list(skill.get("preconditions"))[:3]],
        "skip_conditions": [_compact_text(item, 220) for item in _as_list(skill.get("skip_conditions"))[:3]],
        "stop_conditions": [_compact_text(item, 220) for item in _as_list(skill.get("stop_conditions"))[:3]],
        "fallbacks": [
            {
                "when": _compact_text(fb.get("failure_type") or fb.get("condition"), 160),
                "action": _compact_text(fb.get("next_best_tool") or fb.get("action") or fb.get("fallback"), 180),
            }
            for fb in fallbacks[:2]
        ],
    }
    score = _skill_score(skill)
    if score is not None:
        candidate["retrieval_score"] = round(score, 4)
    if meta:
        candidate["historical_support"] = {
            key: meta.get(key)
            for key in ("total", "success", "correct")
            if key in meta
        }
    if failure_memory:
        candidate["failure_lessons"] = [
            _compact_text(f"{key}: {value.get('last_error') if isinstance(value, dict) else value}", 220)
            for key, value in list(failure_memory.items())[:2]
        ]
    return {key: value for key, value in candidate.items() if value not in (None, "", [], {})}


def _build_scene_task_context(skill_bundle: Dict[str, Any]) -> Dict[str, Any]:
    problem = skill_bundle.get("problem_card", {}) if isinstance(skill_bundle.get("problem_card"), dict) else {}
    return {
        "matched_problem_id": skill_bundle.get("runtime_problem_id") or skill_bundle.get("matched_problem_id"),
        "seed_problem_id": skill_bundle.get("seed_problem_id") or problem.get("seed_problem_id") or problem.get("id"),
        "task_type": problem.get("title") or problem.get("id") or skill_bundle.get("seed_problem_id"),
        "task_intent": _compact_text(problem.get("intent"), 360),
        "required_evidence": [_compact_text(item, 180) for item in _as_list(problem.get("required_evidence"))[:8]],
        "priority_tools": [str(item) for item in _as_list(problem.get("priority_tools"))[:8]],
        "helper_tool_groups": _as_list(problem.get("helper_tool_groups"))[:4],
        "defer_tools": [str(item) for item in _as_list(problem.get("defer_tools"))[:6]],
        "defer_tool_note": _compact_text(problem.get("defer_tool_note"), 260),
    }


def _compact_memory_item(item: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "question",
        "task",
        "scene",
        "scene_id",
        "tool_calls",
        "used_tools",
        "answer",
        "final_answer",
        "outcome",
        "success",
        "failure_type",
        "lesson",
        "evidence",
    )
    compact: Dict[str, Any] = {}
    for key in keys:
        if key not in item:
            continue
        value = item.get(key)
        if isinstance(value, str):
            compact[key] = _compact_text(value, 260)
        elif isinstance(value, list):
            compact[key] = [_compact_text(x, 180) if not isinstance(x, dict) else _compact_memory_item(x) for x in value[:5]]
        elif isinstance(value, dict):
            compact[key] = _compact_memory_item(value)
        else:
            compact[key] = value
    return compact


def _build_scene_memory(skill_bundle: Dict[str, Any], scene_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    episodes = [item for item in _as_list(skill_bundle.get("retrieved_episodes")) if isinstance(item, dict)]
    evidence = [item for item in _as_list(skill_bundle.get("retrieved_evidence")) if isinstance(item, dict)]
    scene_summary = scene_summary if isinstance(scene_summary, dict) else _build_scene_task_context(skill_bundle)
    return {
        "task_type": scene_summary.get("task_type"),
        "seed_problem_id": scene_summary.get("seed_problem_id"),
        "task_intent": scene_summary.get("task_intent"),
        "required_evidence": scene_summary.get("required_evidence") or [],
        "priority_tools": scene_summary.get("priority_tools") or [],
        "retrieved_memory": [
            _compact_memory_item(item)
            for item in [*episodes[:2], *evidence[:2]]
        ],
    }


def create_system_prompt_with_skills(
    tools: List[Dict[str, Any]],
    skill_bundle: Optional[Dict[str, Any]] = None,
    skills: Optional[List[Dict[str, Any]]] = None,
    allow_model_skill_choice: bool = True,
) -> str:
    """
    Create system prompt with available tools and advisory skill references.

    Args:
        tools: List of tool function schemas

    Returns:
        System prompt string
    """
    if not tools:
        return "You are a helpful assistant that can analyze images and answer questions."

    compact_tools = [_compact_tool_schema(tool) for tool in tools]
    tools_json = json.dumps(compact_tools, indent=2, ensure_ascii=False)
    tool_cards_json = json.dumps(_build_tool_cards(tools), indent=2, ensure_ascii=False)
    if skill_bundle is None:
        if skills is None:
            skills = skills_to_json(build_default_skills())
        skill_bundle = {
            "problem_card": {
                "id": "general",
                "seed_problem_id": "general",
                "title": "General Spatial Reasoning",
                "intent": "Decide which evidence pattern best fits the current question.",
                "required_evidence": ["collect only the evidence that changes the answer"],
                "near_misses": [],
                "keywords": [],
                "example_questions": [],
                "priority_tools": [],
                "helper_tool_groups": [],
                "must_try_tools": [],
                "defer_tools": [],
                "defer_tool_note": "",
            },
            "focus_skills": [],
            "skill_catalog": [
                {
                    "id": skill.get("name", f"skill-{idx}"),
                    "name": skill.get("name"),
                    "source": "static",
                    "when": skill.get("when"),
                    "tools": [],
                }
                for idx, skill in enumerate(skills)
            ],
        }

    retrieval_top_k = int(skill_bundle.get("retrieval_top_k") or skill_bundle.get("top_skill_k") or 6)
    skill_candidates = [
        _compact_skill_candidate(skill, rank=idx + 1)
        for idx, skill in enumerate(_rank_scene_task_skills(skill_bundle, top_k=retrieval_top_k))
    ]
    scene_summary = _build_scene_task_context(skill_bundle)
    scene_memory_json = json.dumps(_build_scene_memory(skill_bundle, scene_summary=scene_summary), indent=2, ensure_ascii=False)
    skill_candidates_json = json.dumps(skill_candidates, indent=2, ensure_ascii=False)
    prompt_mode = str(os.environ.get("SPAGENT_SKILL_PROMPT_MODE", "full")).strip().lower()
    force_skill_tool = str(os.environ.get("SPAGENT_FORCE_SKILL_TOOL_CALL", "0")).strip().lower() not in {"0", "false", "no", "off"}
    if allow_model_skill_choice:
        tool_turn_shape = """  <think>Task type: ...\nScene context: ...\nMissing evidence: ...\nSelected skill: ...\nTool plan: ...</think>
  <skill_choice>{"id":"skill_id","source":"static_or_dynamic","name":"skill name","reason":"why this skill helps"}</skill_choice>
  <tool_call>{"name":"tool_name","arguments":{...}}</tool_call>"""
        final_turn_shape = """  <think>Task type: ...\nScene context: ...\nUsed evidence: ...\nFinal reasoning: ...</think>
  <answer>final option/number only</answer>"""
        skill_selection_protocol = """- Use <skill_choice> only when one candidate skill materially guides the next tool step.
- Choose exactly one valid skill id from <skill_candidates> before the first tool call.
- If you output <skill_choice>, the same assistant turn must also output a <tool_call> that follows the chosen skill's workflow."""
        skill_binding_policy = """- Skills in this prompt are references, not mandatory plans unless you explicitly choose one in <skill_choice>.
- Skill/tool binding is strict: if you choose a candidate skill id, execute the minimum sufficient tool flow implied by that skill before producing the final answer.
- Tool-call turns contain thinking, skill choice, and tool calls; final turns contain thinking and answer."""
        
    else:
        tool_turn_shape = """  <think>Task type: ...\nScene context: ...\nMissing evidence: ...\nTool plan: ...</think>
  <tool_call>{"name":"tool_name","arguments":{...}}</tool_call>"""
        final_turn_shape = """  <think>Task type: ...\nScene context: ...\nUsed evidence: ...\nFinal reasoning: ...</think>
  <answer>final option/number only</answer>"""
        skill_selection_protocol = """- Do not output <skill_choice>. Skill selection is handled internally by the retrieval module.
- Use the scene-task context, scene memory, and candidate skills directly to decide tool calls."""
        skill_binding_policy = """- Skills in this prompt are retrieved references for the current question; use their evidence pattern and required_tools/tool sequence directly when useful.
- Tool-call turns contain thinking and one or more <tool_call> blocks; final turns contain thinking and one <answer> block."""

    compact_tool_format_rule = (
        "- Before any final answer, you MUST output <skill_choice>...</skill_choice> followed by <tool_call>...</tool_call> and wait for tool results."
        if allow_model_skill_choice and force_skill_tool
        else "- If more evidence is needed, output one optional <skill_choice>...</skill_choice> followed by one <tool_call>...</tool_call>."
        if allow_model_skill_choice
        else "- If more evidence is needed, output one <tool_call>...</tool_call> after <think>."
    )
    compact_answer_rule = (
        "- Do not output <answer> until at least one selected skill has been paired with a successful tool call and its tool result has been observed."
        if force_skill_tool
        else "- If enough evidence is available, output one <answer>...</answer> after <think>."
    )
    compact_reason_rule = (
        "- After observing tool results, the final turn MUST put the evidence-based reason inside <think> before <answer>; mention which tool output supports the answer."
    )
    compact_force_rule = (
        "- For this rollout, <skill_choice> and <tool_call> are mandatory before the first <answer>; direct RGB-only answering is a format violation."
        if force_skill_tool
        else ""
    )

    if prompt_mode in {"compact", "sft", "rollout", "sft_rollout"}:
        return f"""You are Skill-3D, a scene-aware spatial reasoning agent.

# Required Format
- Every assistant turn must start with exactly one <think>...</think> block.
{compact_tool_format_rule}
{compact_answer_rule}
{compact_reason_rule}
- Do not write text outside these tags.

When a tool is required, output exactly:
{tool_turn_shape}

When the final answer is supported, output exactly:
{final_turn_shape}

# Scene Memory
<scene_memory>
{scene_memory_json}
</scene_memory>

# Candidate Skills
Use only these top-{retrieval_top_k} static/dynamic skills when selecting a skill:
<skill_candidates>
{skill_candidates_json}
</skill_candidates>

# Skill Rules
{skill_selection_protocol}
{compact_force_rule}
- Prefer a matching dynamic skill; otherwise use the most relevant static skill.
- Choose the minimum sufficient tool workflow. Avoid duplicate calls.
- Metric distance/depth questions need depth or 3D evidence; boundary-sensitive questions need segmentation; counting/localization needs detection; orientation questions need orientation evidence; cross-view layout may need Pi3.

# Tools
<tools>
{tools_json}
</tools>

# Tool Cards
<tool_cards>
{tool_cards_json}
</tool_cards>

# Tool Policy
{skill_binding_policy}
- Use only schema-supported argument names.
- Stop as soon as tool evidence is sufficient for the final answer.
"""

    return f"""You are Skill-3D, a scene-aware multimodal spatial reasoning agent.
Your goal is to answer visual spatial reasoning questions accurately by combining scene context memory, static and dynamic skills, external perception or geometry tools, and final reasoning grounded in tool evidence.

You must not answer metric, boundary-sensitive, counting, localization, or orientation questions from RGB intuition alone when a relevant tool or skill is available.

# Core Procedure
For each user question, follow this loop:
1. Scene understanding: identify the scene type, target objects, visible views, and whether the input is single-view, multi-view, room-level, or object-to-object.
2. Skill retrieval: use the provided scene-task context and top candidate skills. Prefer a matched dynamic skill when available; otherwise fall back to the corresponding static seed skill.
3. Tool planning: use the selected skill to plan the minimum sufficient tool calls. Match the evidence need to detection, segmentation, depth estimation, orientation estimation, super-resolution, or 3D reconstruction.
4. Tool execution and adoption: execute tools only when they are necessary, then explicitly use successful and relevant tool outputs in later reasoning.
5. Final answer: answer the original question directly, with the requested unit or option format.

# Interaction Tag Protocol
- The FIRST non-whitespace token of every assistant response MUST be `<think>`.
- Put all reasoning text inside exactly one paired `<think>...</think>` block.
- Do not write any natural-language reasoning before `<think>` or after the final closing tag.
- If you need more evidence, use this shape:
{tool_turn_shape}
- If you are ready to answer, use this shape:
{final_turn_shape}
- Never output `<answer>` before tool results have been observed when you just requested a tool.

# Scene Memory
This is compressed scene-aware memory. It is context for skill selection, not a mandatory plan:
<scene_memory>
{scene_memory_json}
</scene_memory>

# Candidate Skills
These are the top-{retrieval_top_k} static/dynamic skills selected by scene-task retrieval. This is the only skill list you should consider:
<skill_candidates>
{skill_candidates_json}
</skill_candidates>

# Skill Selection Rules
{skill_selection_protocol}
- Prefer dynamic skills when their trigger, scene context, and required evidence match the current question.
- Use a static skill when no dynamic skill is specific enough.
- If several skills overlap, choose the most specific skill and ignore redundant ones.
- Do not invent skill ids outside <skill_candidates>.
- Avoid duplicate tool calls unless they verify genuinely uncertain evidence.

# Available Tools
You have access to the following callable tools. Treat this as API reference, not as a recommendation:
<tools>
{tools_json}
</tools>

# Tool Cards
These are lightweight discovery cards. They are intentionally brief:
<tool_cards>
{tool_cards_json}
</tool_cards>

# How to call a tool
When you need to use a tool, return a JSON object with the function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

You can call multiple tools if needed by using multiple <tool_call> blocks.

# Output Protocol
- Every response turn must begin with a concise <think></think> block; no text is allowed before `<think>`.
- Final answers must use <answer></answer> only after the collected observations and reasoning are sufficient.

# Tool Use Policy
Decide for yourself which tools to call and in what order.
{skill_binding_policy}
- Use any tool sequence that best resolves uncertainty for the current question.
- Start with the minimum-evidence tool that can materially change the answer.
- Use detection or grounding tools for object identity/localization.
- Use segmentation when closest-point, contact, size, or boundary precision matters.
- Use depth estimation for metric depth, approximate physical distance, nearest/farthest, or front/back evidence.
- Use orientation estimation for facing direction, front/back interpretation, and relative rotation.
- Use Pi3/reconstruction for cross-view, occlusion, view selection, or room-level 3D layout when direct 2D cues are insufficient.
- Do not jump to pi3_tool before direct 2D cues such as depth_estimation_tool, segment_image_tool, swinir_tool, or detect_objects_tool unless viewpoint change, occlusion, or overall 3D layout/room geometry is the actual blocker.
- If a tool call fails or returns coarse evidence, switch to a complementary tool family rather than repeating the same family with the same uncertainty.
- Use only parameter names that exist in the tool schema. Do not invent unsupported arguments such as `box1`, `box2`, `image_path_a`, `image_path_b`, `point_prompts`, `distance_type`, or `use_detected`.
- If the question explicitly asks for an answer `in meters`, do not answer from RGB intuition alone.
- Stop when more tool calls are unlikely to change the answer.

# Progressive Disclosure
- Before calling a tool, use scene-task context, scene memory, candidate skills, and tool cards as lightweight guidance.
- Once you decide to call a tool, use its schema to choose narrowly targeted arguments.
- Stay schema-faithful: prefer `crop_box`, `crop_boxes`, `point_coords`, `point_pairs`, and other listed parameters over invented shortcut fields.
- Do not read a tool's presence in this prompt as proof that it should be used.

# Multi-Step Workflow
You can perform MULTIPLE rounds of tool calls. The selected skill should guide evidence acquisition, but the final answer must be grounded in observed tool evidence.

Stop early when additional tool calls will not change the answer. Only put the final option/number in <answer></answer> tags (no extra text).
"""


def create_skill_reflection_prompt(
    question: str,
    tool_calls: List[Dict[str, Any]],
    tool_results: Dict[str, Any],
    final_answer: str,
    success: bool,
) -> str:
    """
    Ask model to summarize trajectory as a reusable skill JSON.
    """
    compact_calls = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        compact_calls.append(
            {
                "name": call.get("name"),
                "arguments": call.get("arguments", {}),
            }
        )

    compact_results = {}
    for key, value in (tool_results or {}).items():
        if not isinstance(value, dict):
            continue
        compact_results[key] = {
            "success": bool(value.get("success")),
            "error": value.get("error"),
            "description": value.get("description"),
        }

    payload = {
        "question": question,
        "success": bool(success),
        "tool_calls": compact_calls,
        "tool_results": compact_results,
        "final_answer": final_answer[:600],
    }

    return f"""You are optimizing tool-use skills for a spatial reasoning agent.

Analyze the episode below and produce ONE reusable skill candidate.

Episode:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return strict JSON only (no markdown, no explanation) with this schema:
{{
  "name": "short skill name",
  "when": "when to use this skill",
  "strategy": ["step1", "step2", "step3"],
  "tool_usage": "tool_a -> tool_b -> tool_c",
  "confidence": 0.0
}}

Rules:
- confidence must be within [0, 1]
- strategy should be concise and actionable
- infer tool_usage from successful calls when possible
- focus on reusable evidence patterns rather than superficial wording quirks
"""
