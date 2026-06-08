"""
SPAgent - Spatial Intelligence Agent

This module contains the main SPAgent class that orchestrates problem solving
using external expert tools and VLLM models.
"""

import json
import ast
import re
import logging
import asyncio
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from pathlib import Path

from .tool import Tool, ToolRegistry
from .model import Model
from .prompts import create_system_prompt, create_follow_up_prompt, create_user_prompt, create_fallback_prompt
from .prompts_with_skills import create_system_prompt_with_skills, create_skill_reflection_prompt
from .data_collector import DataCollector
from .tool_normalization import (
    parse_tool_call_block,
    normalize_tool_name,
    normalize_tool_arguments,
    normalize_tool_call_payload,
)
from .skill_learning import AdaptiveSkillManager

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

logger = logging.getLogger(__name__)


class SPAgent:
    """
    Spatial Intelligence Agent
    
    An agent that can solve spatial intelligence problems by combining
    VLLM models with external expert tools.
    """
    
    def __init__(
        self, 
        model: Model,
        tools: Optional[List[Tool]] = None,
        max_workers: int = 4,
        data_collector: Optional[DataCollector] = None,
        prompt_style: str = "normal",
        skill_manager: Optional[AdaptiveSkillManager] = None,
        enable_skill_learning: bool = True,
        enable_skill_update: bool = True,
        enable_skill_reflection: bool = True,
        skill_storage_path: str = "statics/learned_skills.json",
        memory_mode: str = "symbolic",
        retrieval_mode: str = "symbolic",
        retrieval_top_k: int = 6,
        retrieval_trace_path: Optional[str] = None,
    ):
        """
        Initialize SPAgent
        
        Args:
            model: VLLM model wrapper to use
            tools: List of external expert tools (optional)
            max_workers: Maximum number of parallel tool executions
            data_collector: Optional DataCollector for training data collection
        """
        self.model = model
        self.tool_registry = ToolRegistry()
        self.max_workers = max_workers
        self.data_collector = data_collector
        self.prompt_style = prompt_style
        self.enable_skill_learning = enable_skill_learning
        self.enable_skill_update = enable_skill_update
        self.enable_skill_reflection = enable_skill_reflection
        self.memory_mode = memory_mode
        self.retrieval_mode = retrieval_mode
        self.retrieval_top_k = int(retrieval_top_k)
        self.retrieval_trace_path = retrieval_trace_path or os.environ.get("SPAGENT_RETRIEVAL_TRACE_PATH")
        self.auto_select_skills = self._env_flag("SPAGENT_AUTO_SELECT_SKILLS", True)
        self.auto_select_skill_top_k = max(1, int(os.environ.get("SPAGENT_AUTO_SELECT_SKILL_TOP_K", "1")))
        self.allow_model_skill_choice = self._env_flag("SPAGENT_ALLOW_MODEL_SKILL_CHOICE", True)
        self.force_skill_tool_call = self._env_flag("SPAGENT_FORCE_SKILL_TOOL_CALL", False)
        self.enforce_must_try_tools = self._env_flag("SPAGENT_ENFORCE_MUST_TRY_TOOLS", False)
        self.repair_missing_think = self._env_flag("SPAGENT_REPAIR_MISSING_THINK", False)
        self.strip_disallowed_skill_choice = self._env_flag("SPAGENT_STRIP_DISALLOWED_SKILL_CHOICE", True)
        self.oracle_rollout = self._env_flag("SPAGENT_ORACLE_ROLLOUT", False)
        self.stop_on_answer = self._env_flag("SPAGENT_STOP_ON_ANSWER", True)
        self.skill_manager = None
        if self.prompt_style == "skills" and self.enable_skill_learning:
            self.skill_manager = skill_manager or AdaptiveSkillManager(
                storage_path=skill_storage_path,
                update_enabled=enable_skill_update,
            )
        
        # Register provided tools
        if tools:
            for tool in tools:
                self.add_tool(tool)
        
        logger.info(f"Initialized SPAgent with model: {model.model_name}")
        if data_collector:
            logger.info("Data collection enabled")
        logger.info(f"Prompt style: {self.prompt_style}")
        if self.skill_manager is not None:
            logger.info("Adaptive skill learning enabled")
        logger.info(f"Memory mode: {self.memory_mode}, retrieval mode: {self.retrieval_mode}, retrieval_top_k={self.retrieval_top_k}")
        logger.info(
            "Automatic skill selection: %s, top_k=%s",
            self.auto_select_skills,
            self.auto_select_skill_top_k,
        )
        logger.info("Force model skill+tool call before answer: %s", self.force_skill_tool_call)
        logger.info("Enforce must-try tools: %s", self.enforce_must_try_tools)
        logger.info("Stop workflow immediately on complete <answer>: %s", self.stop_on_answer)
        if self.oracle_rollout:
            logger.warning("Oracle rollout mode is enabled; ground-truth answers may be injected into prompts.")

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _write_retrieval_trace(
        self,
        question: str,
        skill_bundle: Optional[Dict[str, Any]],
        selected_skill_choices: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        is_correct: Optional[bool] = None,
    ) -> None:
        if not self.retrieval_trace_path or not isinstance(skill_bundle, dict):
            return
        try:
            path = Path(self.retrieval_trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics = skill_bundle.get("retrieval_diagnostics", {})
            record = {
                "question": question,
                "predicted_class": diagnostics.get("symbolic_routed_class"),
                "symbolic_candidates": diagnostics.get("symbolic_skill_candidates", []),
                "dense_retrieved_topk": diagnostics.get("dense_retrieved_candidates", []),
                "reranked_topk": diagnostics.get("reranked_final_list", []),
                "injected_retrieved_skills": [
                    item.get("id") for item in skill_bundle.get("retrieved_skills", []) if isinstance(item, dict)
                ],
                "selected_skill_choices": selected_skill_choices or [],
                "tool_chain": [call.get("name") for call in (tool_calls or []) if isinstance(call, dict)],
                "is_correct": is_correct,
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to write retrieval trace: %s", exc)

    def _build_skill_choice_reporting_block(
        self,
        skill_bundle: Optional[Dict[str, Any]],
        include_catalog_snapshot: bool = False,
    ) -> str:
        if self.prompt_style != "skills" or not skill_bundle:
            return ""
        if not self.allow_model_skill_choice:
            return ""
        prompt_mode = str(os.environ.get("SPAGENT_SKILL_PROMPT_MODE", "")).strip().lower()
        if prompt_mode in {"compact", "sft", "rollout", "sft_rollout"}:
            return ""

        lines = [
            "# Skill Choice Reporting",
            "Every response must include a concise <think></think> block before any tool call or final answer.",
            "<skill_choice></skill_choice> blocks are optional. Use them only when a catalog skill is useful for the current step.",
            "Each block must contain strict JSON with keys: id, source, name, reason.",
            "Use source='static' or source='dynamic' when you pick a catalog skill.",
            "If no catalog skill is useful for this step, omit <skill_choice> or output one block with id='none', source='none', name='none'.",
            "Binding rule: if you pick any catalog skill id other than 'none', you must execute that skill's required tool flow before the final <answer>.",
            "Never output <tool_call> and <answer> in the same response.",
            "Do not output <tool_choice>; use <tool_call> for executable tool decisions.",
            "If you combine multiple skills, output multiple <skill_choice> blocks.",
            "Example:",
            "<think>I need object localization evidence before answering the spatial relation.</think>",
            "<skill_choice>",
            '{"id":"seed::spatial_localization","source":"static","name":"Skill 1: Spatial Localization & Relation","reason":"This step is mainly about a left/right spatial relation."}',
            "</skill_choice>",
        ]

        if include_catalog_snapshot:
            snapshot = []
            for card in skill_bundle.get("skill_catalog", []):
                if not isinstance(card, dict):
                    continue
                snapshot.append(
                    {
                        "id": card.get("id"),
                        "name": card.get("name"),
                        "source": card.get("source"),
                        "when": card.get("when"),
                    }
                )
            lines.extend(
                [
                    "",
                    "Current skill catalog snapshot:",
                    "<skill_catalog_snapshot>",
                    json.dumps(snapshot, ensure_ascii=False, indent=2),
                    "</skill_catalog_snapshot>",
                ]
            )

        return "\n".join(lines)

    def _build_skill_catalog_index(
        self,
        skill_bundle: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(skill_bundle, dict):
            return {}

        index: Dict[str, Dict[str, Any]] = {}
        for section in ("skill_catalog", "focus_skills", "retrieved_skills"):
            for card in skill_bundle.get(section, []) or []:
                if not isinstance(card, dict):
                    continue
                skill_id = str(card.get("id", "")).strip()
                if not skill_id:
                    continue
                index[skill_id] = card
        return index

    def _active_skill_choices(self, skill_choices: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        active: List[Dict[str, Any]] = []
        for choice in skill_choices or []:
            if not isinstance(choice, dict):
                continue
            skill_id = str(choice.get("id", "")).strip()
            source = str(choice.get("source", "")).strip().lower()
            if not skill_id or skill_id in {"none", "no_skill", "direct", "direct_answer"} or source == "none":
                continue
            active.append(choice)
        return active

    def _auto_selected_skill_choices(
        self,
        skill_bundle: Optional[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Select skills from retrieval output without requiring model-emitted tags.

        This mirrors SkillRL-style prompt-time skill retrieval: use dense/hybrid
        retrieved skills when available, otherwise fall back to symbolic focus
        skills. The result is treated as an internal selector decision, so tool
        constraints can use the skill even if the model does not output
        <skill_choice>.
        """
        if not self.auto_select_skills or self.prompt_style != "skills" or not isinstance(skill_bundle, dict):
            return []

        limit = max(1, int(top_k or self.auto_select_skill_top_k))
        candidates: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for section in ("retrieved_skills", "focus_skills"):
            for card in skill_bundle.get(section, []) or []:
                if not isinstance(card, dict):
                    continue
                skill_id = str(card.get("id", "")).strip()
                if not skill_id or skill_id in seen:
                    continue
                source = str(card.get("source") or card.get("source_type") or "").strip().lower()
                if not source:
                    source = "static" if skill_id.startswith("seed::") else "dynamic"
                candidates.append(
                    {
                        "id": skill_id,
                        "source": source,
                        "name": card.get("name") or card.get("skill_name") or skill_id,
                        "reason": (
                            "Automatically selected from prompt-time skill retrieval "
                            f"section={section}; trigger={card.get('when') or card.get('trigger') or ''}"
                        ),
                        "auto_selected": True,
                    }
                )
                seen.add(skill_id)
                if len(candidates) >= limit:
                    return candidates
        return candidates

    def _merge_skill_choices(
        self,
        parsed_choices: Optional[List[Dict[str, Any]]],
        automatic_choices: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for choice in (parsed_choices or []) + (automatic_choices or []):
            if not isinstance(choice, dict):
                continue
            skill_id = str(choice.get("id", "")).strip()
            if not skill_id or skill_id in seen:
                continue
            seen.add(skill_id)
            merged.append(choice)
        return merged

    def _fallback_required_tools_for_seed_skill(self, skill_id: str) -> List[str]:
        seed_id = str(skill_id or "").replace("seed::", "", 1)
        anchors = {
            "spatial_localization": ["detect_objects_tool"],
            "depth_distance": ["depth_estimation_tool"],
            "multiview_occlusion": ["pi3_tool"],
            "detection_counting": ["detect_objects_tool"],
            "segmentation_boundary": ["segment_image_tool"],
            "fine_grained_pointing": ["moondream_tool"],
            "custom_category": ["detect_objects_tool"],
        }
        return anchors.get(seed_id, [])

    def _skill_required_tools(
        self,
        skill_bundle: Optional[Dict[str, Any]],
        skill_choices: Optional[List[Dict[str, Any]]],
    ) -> List[str]:
        if not isinstance(skill_bundle, dict):
            return []

        skill_index = self._build_skill_catalog_index(skill_bundle)
        required: List[str] = []
        seen: Set[str] = set()

        for choice in self._active_skill_choices(skill_choices):
            skill_id = str(choice.get("id", "")).strip()
            card = skill_index.get(skill_id, {})
            candidate_tools = (
                card.get("required_tools")
                or card.get("skill_required_tools")
                or (card.get("tool_candidates") if str(card.get("source", "")).lower() == "dynamic" else None)
                or []
            )
            if not candidate_tools and str(card.get("source", "")).lower() == "dynamic":
                candidate_tools = card.get("tools") or []
            if not candidate_tools and skill_id.startswith("seed::"):
                candidate_tools = self._fallback_required_tools_for_seed_skill(skill_id)

            for tool_name in candidate_tools or []:
                normalized = normalize_tool_name(str(tool_name).strip(), self.tool_registry)
                if not normalized or normalized in seen or self.tool_registry.get(normalized) is None:
                    continue
                seen.add(normalized)
                required.append(normalized)

        return required

    def _question_signals(
        self,
        question: str,
        skill_bundle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        text = str(question or "").lower()
        problem_card = (skill_bundle or {}).get("problem_card", {}) if isinstance(skill_bundle, dict) else {}
        problem_id = str(problem_card.get("seed_problem_id") or problem_card.get("id") or "").strip()
        depth_keywords = [
            "closer",
            "farther",
            "distance",
            "in meters",
            "meter",
            "metre",
            "nearest",
            "farthest",
            "front",
            "behind",
            "closest point",
            "point-to-point",
            "point to point",
            "point pair",
        ]
        needs_metric_point_distance = ("closest point" in text or "point-to-point" in text or "point to point" in text or "point pair" in text) and ("distance" in text or "in meters" in text)
        return {
            "needs_depth": problem_id == "depth_distance" or any(token in text for token in depth_keywords),
            "needs_boundary": any(token in text for token in ["closest point", "boundary", "mask", "outline", "touch", "contact", "surface"]),
            "needs_exact_points": any(token in text for token in ["point", "pixel", "corner", "handle", "tip", "selected"]),
            "needs_metric_point_distance": needs_metric_point_distance,
            "needs_orientation": any(
                token in text
                for token in [
                    "orientation",
                    "facing",
                    "front side",
                    "back side",
                    "parallel",
                    "perpendicular",
                    "pose",
                    "rotation",
                    "rotated",
                    "tilted",
                ]
            ),
            "needs_view_change": any(
                token in text
                for token in [
                    "another view",
                    "other view",
                    "different view",
                    "viewpoint",
                    "hidden",
                    "occluded",
                    "occlusion",
                    "visible from another",
                    "camera can see",
                    "room size",
                    "combined space",
                    "layout",
                ]
            ),
            "needs_low_quality_recovery": any(
                token in text
                for token in [
                    "tiny",
                    "small",
                    "blurry",
                    "blurred",
                    "far away",
                    "low resolution",
                    "hard to see",
                    "unclear",
                    "fuzzy",
                ]
            ),
            "requires_localization": problem_id in {"detection_counting", "custom_category", "spatial_localization"},
            "is_counting": problem_id == "detection_counting" or "how many" in text or "count " in text,
            "is_multiview_problem": problem_id == "multiview_occlusion",
        }

    def _missing_must_try_tools(
        self,
        skill_bundle: Optional[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        skill_choices: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        if not self.enforce_must_try_tools:
            return []
        if not isinstance(skill_bundle, dict):
            return []

        problem_card = skill_bundle.get("problem_card", {}) if isinstance(skill_bundle.get("problem_card"), dict) else {}
        must_try_tools = [str(tool).strip() for tool in problem_card.get("must_try_tools", []) if str(tool).strip()]
        must_try_tools.extend(self._skill_required_tools(skill_bundle, skill_choices))
        if not must_try_tools:
            return []

        ordered_must_try: List[str] = []
        seen_must_try: Set[str] = set()
        for tool_name in must_try_tools:
            normalized = normalize_tool_name(str(tool_name).strip(), self.tool_registry)
            if not normalized or normalized in seen_must_try or self.tool_registry.get(normalized) is None:
                continue
            seen_must_try.add(normalized)
            ordered_must_try.append(normalized)

        attempted: Set[str] = set()
        for call in tool_calls or []:
            if not isinstance(call, dict):
                continue
            raw_name = str(call.get("name", "")).strip()
            if not raw_name:
                continue
            attempted.add(normalize_tool_name(raw_name, self.tool_registry))

        return [tool for tool in ordered_must_try if tool not in attempted]

    def _extract_tool_failures(
        self,
        all_tool_results: Dict[str, Any],
    ) -> Dict[str, int]:
        failure_counts: Dict[str, int] = {}
        for key, result in (all_tool_results or {}).items():
            if not isinstance(result, dict):
                continue
            if result.get("success"):
                continue
            tool_name = str(result.get("tool_name") or str(key).split("_iter")[0]).strip()
            if not tool_name:
                continue
            failure_counts[tool_name] = failure_counts.get(tool_name, 0) + 1
        return failure_counts

    def _repair_tool_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        args = dict(arguments or {})
        if tool_name == "moondream_tool":
            if "task" not in args and str(args.get("object_name", "")).strip():
                args["task"] = "point"
            for alias in ("objects", "object", "target_objects"):
                if "object_name" not in args and str(args.get(alias, "")).strip():
                    args["object_name"] = args.pop(alias)
                    break
        return args

    def _required_tool_parameters(
        self,
        tool_name: str,
    ) -> List[str]:
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            return []
        required = tool.parameters.get("required", []) if isinstance(tool.parameters, dict) else []
        return [str(item).strip() for item in required if str(item).strip()]

    def _format_rejected_tool_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        reason: str,
        failure_type: str = "selector_rejected",
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "error": reason,
            "failure_type": failure_type,
            "tool_name": tool_name,
            "tool_arguments": dict(arguments or {}),
            "information_gain": 0.0,
            "selector_feedback": reason,
        }

    def _validate_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        skill_bundle: Optional[Dict[str, Any]],
        question: str,
        all_tool_calls: List[Dict[str, Any]],
        all_tool_results: Dict[str, Any],
        view_state: Dict[str, Any],
        skill_choices: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[str], Optional[str]]:
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            return False, arguments, f"Tool not registered: {tool_name}", "selector_rejected"

        repaired = self._repair_tool_arguments(tool_name, arguments)
        normalized = normalize_tool_arguments(tool, repaired)
        required = self._required_tool_parameters(tool_name)
        missing_required = [key for key in required if key not in normalized or normalized.get(key) in (None, "", [], {})]
        if missing_required:
            return False, normalized, f"{tool_name} is missing required parameters: {', '.join(missing_required)}", "schema_error"

        problem_card = (skill_bundle or {}).get("problem_card", {}) if isinstance(skill_bundle, dict) else {}
        signals = self._question_signals(question, skill_bundle=skill_bundle)
        missing_must_try = set(
            self._missing_must_try_tools(
                skill_bundle,
                all_tool_calls,
                skill_choices=skill_choices,
            )
        )
        failure_counts = self._extract_tool_failures(all_tool_results)

        if tool_name == "detect_objects_tool":
            if signals["needs_depth"] and "depth_estimation_tool" in missing_must_try and not signals["requires_localization"]:
                return False, normalized, "Reject detect_objects_tool because depth_estimation_tool is still a required unmet attempt for this depth question.", "selector_rejected"

        if tool_name == "depth_estimation_tool":
            if not signals["needs_depth"]:
                pass
            if signals["needs_metric_point_distance"]:
                has_point_measurement = bool(normalized.get("point_coords")) and bool(normalized.get("point_pairs"))
                has_region_measurement = bool(normalized.get("crop_box")) or bool(normalized.get("crop_boxes"))
                if not has_point_measurement and not has_region_measurement:
                    return (
                        False,
                        normalized,
                        "Reject depth_estimation_tool because closest-point metric distance questions need localized points or regions first. Use point_coords+point_pairs or crop_box/crop_boxes, not only image_path.",
                        "schema_error",
                    )

        if tool_name == "pi3_tool":
            azimuth = float(normalized.get("azimuth_angle", 0) or 0)
            elevation = float(normalized.get("elevation_angle", 0) or 0)
            ref_camera = int(normalized.get("rotation_reference_camera", 1) or 1)
            camera_view = bool(normalized.get("camera_view", False))
            view_key = f"{azimuth:.1f}:{elevation:.1f}:ref{ref_camera}:cam{int(camera_view)}"
            tried_views = set(str(item) for item in view_state.get("tried_views", []) or [])
            if azimuth == 0.0 and elevation == 0.0:
                return False, normalized, "Reject pi3_tool because (0, 0) repeats the original input viewpoint.", "repeated_view"
            if view_key in tried_views:
                return False, normalized, f"Reject pi3_tool because view {view_key} has already been tried.", "repeated_view"
            if not signals["needs_view_change"] and "pi3_tool" in [str(tool).strip() for tool in problem_card.get("defer_tools", [])]:
                return False, normalized, "Reject pi3_tool because the current question does not yet justify 3D viewpoint escalation.", "selector_rejected"

        if tool_name == "swinir_tool" and not signals["needs_low_quality_recovery"]:
            return False, normalized, "Reject swinir_tool because the question does not indicate a low-quality visual blocker.", "selector_rejected"
        if (
            tool_name == "swinir_tool"
            and "crop_box" not in normalized
            and (signals["requires_localization"] or signals["needs_exact_points"] or signals["needs_orientation"] or signals["needs_depth"])
        ):
            return False, normalized, "Reject swinir_tool without crop_box because this question first needs a localized target region.", "selector_rejected"

        if tool_name == "orient_anything_tool" and not signals["needs_orientation"]:
            return False, normalized, "Reject orient_anything_tool because the current blocker is not orientation or pose.", "selector_rejected"
        if tool_name == "orient_anything_tool" and "crop_box" not in normalized:
            return False, normalized, "Reject orient_anything_tool because it now requires a target crop_box; localize the object first.", "schema_error"
        if tool_name == "orient_anything_tool" and normalized.get("target_image_path") and "target_crop_box" not in normalized:
            return False, normalized, "Reject orient_anything_tool because target_image_path requires target_crop_box for the second view.", "schema_error"

        if tool_name in failure_counts and failure_counts[tool_name] >= 2:
            repeated_same_call = any(
                normalize_tool_name(str(call.get("name", "")).strip(), self.tool_registry) == tool_name
                and normalize_tool_arguments(tool, dict(call.get("arguments", {}) or {})) == normalized
                for call in all_tool_calls
                if isinstance(call, dict)
            )
            if repeated_same_call:
                return False, normalized, f"Reject {tool_name} because the same call already failed recently without new evidence.", "selector_rejected"

        return True, normalized, None, None

    def _score_tool_call(
        self,
        tool_name: str,
        skill_bundle: Optional[Dict[str, Any]],
        question: str,
        all_tool_calls: List[Dict[str, Any]],
        all_tool_results: Dict[str, Any],
        skill_choices: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        score = 0.0
        problem_card = (skill_bundle or {}).get("problem_card", {}) if isinstance(skill_bundle, dict) else {}
        signals = self._question_signals(question, skill_bundle=skill_bundle)
        priority_tools = [str(tool).strip() for tool in problem_card.get("priority_tools", []) if str(tool).strip()]
        must_try_tools = [str(tool).strip() for tool in problem_card.get("must_try_tools", []) if str(tool).strip()]
        skill_required_tools = self._skill_required_tools(skill_bundle, skill_choices)
        helper_tools = {
            str(tool).strip()
            for group in problem_card.get("helper_tool_groups", []) or []
            if isinstance(group, dict)
            for tool in group.get("tools", []) or []
            if str(tool).strip()
        }
        failure_counts = self._extract_tool_failures(all_tool_results)
        attempted_tools = {
            normalize_tool_name(str(call.get("name", "")).strip(), self.tool_registry)
            for call in all_tool_calls
            if isinstance(call, dict)
        }

        if tool_name in priority_tools:
            score += 5.0
        if tool_name in must_try_tools and tool_name not in attempted_tools:
            score += 6.0
        if tool_name in skill_required_tools and tool_name not in attempted_tools:
            score += 6.0
        if tool_name in helper_tools:
            score += 2.0
        if tool_name == "detect_objects_tool":
            score -= 1.0
        if tool_name == "pi3_tool":
            score -= 2.0
        if tool_name == "depth_estimation_tool" and signals["needs_depth"]:
            score += 3.0
            if signals["needs_metric_point_distance"]:
                score += 2.0
        if tool_name == "segment_image_tool" and signals["needs_boundary"]:
            score += 2.5
        if tool_name == "moondream_tool" and signals["needs_exact_points"]:
            score += 2.5
        if tool_name == "swinir_tool" and signals["needs_low_quality_recovery"]:
            score += 3.0
        if tool_name == "orient_anything_tool" and signals["needs_orientation"]:
            score += 3.0
        if tool_name == "pi3_tool" and signals["needs_view_change"]:
            score += 2.5

        score -= min(3.0, float(failure_counts.get(tool_name, 0)) * 0.75)
        return score

    def _select_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        skill_choices: List[Dict[str, Any]],
        skill_bundle: Optional[Dict[str, Any]],
        question: str,
        all_tool_calls: List[Dict[str, Any]],
        all_tool_results: Dict[str, Any],
        iteration: int,
        view_state: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
        approved_calls: List[Dict[str, Any]] = []
        rejected_calls: List[Dict[str, Any]] = []
        decision_log: List[Dict[str, Any]] = []

        normalized_calls: List[Dict[str, Any]] = []
        for raw_call in tool_calls:
            if not isinstance(raw_call, dict):
                continue
            raw_name = str(raw_call.get("name", "")).strip()
            norm_name = normalize_tool_name(raw_name, self.tool_registry)
            normalized_calls.append(
                {
                    "name": norm_name,
                    "arguments": dict(raw_call.get("arguments", {}) or {}),
                    "raw_name": raw_name,
                }
            )

        missing_must_try = self._missing_must_try_tools(skill_bundle, all_tool_calls, skill_choices=skill_choices)
        proposed_names = {call["name"] for call in normalized_calls}
        selector_note = None
        if missing_must_try and not any(tool_name in proposed_names for tool_name in missing_must_try):
            selector_note = (
                "Selector override: before answering, attempt the required tool(s): "
                + ", ".join(missing_must_try)
                + ". Use schema-valid arguments and prefer them over default detection."
            )

        for call in normalized_calls:
            valid, normalized_args, rejection_reason, failure_type = self._validate_tool_call(
                call["name"],
                call["arguments"],
                skill_bundle=skill_bundle,
                question=question,
                all_tool_calls=all_tool_calls,
                all_tool_results=all_tool_results,
                view_state=view_state,
                skill_choices=skill_choices,
            )
            score = self._score_tool_call(
                call["name"],
                skill_bundle=skill_bundle,
                question=question,
                all_tool_calls=all_tool_calls,
                all_tool_results=all_tool_results,
                skill_choices=skill_choices,
            )

            decision = {
                "iteration": iteration,
                "tool_name": call["name"],
                "raw_name": call["raw_name"],
                "score": round(score, 4),
                "approved": bool(valid),
                "arguments": normalized_args,
            }
            if rejection_reason:
                decision["reason"] = rejection_reason
                decision["failure_type"] = failure_type or "selector_rejected"
                rejected_calls.append(
                    {
                        "name": call["name"],
                        "arguments": normalized_args,
                        "selector_reason": rejection_reason,
                        "failure_type": failure_type or "selector_rejected",
                    }
                )
            else:
                approved_calls.append({"name": call["name"], "arguments": normalized_args})
            decision_log.append(decision)

        approved_calls.sort(
            key=lambda item: self._score_tool_call(
                item["name"],
                skill_bundle=skill_bundle,
                question=question,
                all_tool_calls=all_tool_calls,
                all_tool_results=all_tool_results,
                skill_choices=skill_choices,
            ),
            reverse=True,
        )

        selector_decision = {
            "iteration": iteration,
            "skill_choices": [choice.get("id") for choice in skill_choices if isinstance(choice, dict)],
            "skill_required_tools": self._skill_required_tools(skill_bundle, skill_choices),
            "missing_must_try_tools": missing_must_try,
            "approved_tool_calls": approved_calls,
            "rejected_tool_calls": rejected_calls,
            "selector_log": decision_log,
        }
        if not selector_note and missing_must_try and not approved_calls:
            selector_note = (
                "Selector override: the required tool(s) are still missing or invalid: "
                + ", ".join(missing_must_try)
                + ". Retry them with schema-valid arguments."
            )
        if selector_note:
            selector_decision["selector_reason"] = selector_note

        return approved_calls, rejected_calls, selector_decision, selector_note

    def _classify_tool_failure(
        self,
        tool_name: str,
        error_message: str,
    ) -> str:
        error_text = str(error_message or "").lower()
        if "missing required" in error_text or "required parameters" in error_text or "invalid" in error_text:
            return "schema_error"
        if tool_name == "detect_objects_tool" and any(token in error_text for token in ["no result returned", "未检测到目标", "not detect", "detection failed"]):
            return "no_detection"
        if tool_name == "pi3_tool" and any(token in error_text for token in ["already been tried", "original input viewpoint"]):
            return "repeated_view"
        return "tool_runtime_error"

    def _estimate_information_gain(
        self,
        tool_name: str,
        result: Dict[str, Any],
    ) -> float:
        if not isinstance(result, dict) or not result.get("success"):
            return 0.0

        score = 0.3
        if tool_name == "detect_objects_tool":
            evidence = len(result.get("boxes", []) or []) or len(result.get("detections", []) or [])
            score = min(1.0, 0.2 + 0.15 * evidence)
        elif tool_name == "segment_image_tool":
            evidence = len(result.get("masks", []) or [])
            score = min(1.0, 0.4 + 0.15 * evidence)
        elif tool_name == "depth_estimation_tool":
            score = 0.45
            if result.get("point_measurements"):
                score += 0.2
            if result.get("point_pair_distances"):
                score += 0.2
            if result.get("region_depth_stats"):
                score += 0.15
            score = min(1.0, score)
        elif tool_name == "moondream_tool":
            evidence = len(result.get("result", {}).get("points", []) or []) + int(result.get("result", {}).get("total_points") or 0)
            score = min(1.0, 0.35 + 0.1 * max(evidence, 1))
        elif tool_name == "pi3_tool":
            azimuth = float(result.get("azimuth_angle", 0) or 0)
            elevation = float(result.get("elevation_angle", 0) or 0)
            score = 0.55
            if azimuth != 0.0 or elevation != 0.0:
                score += 0.2
            if result.get("cached"):
                score -= 0.1
            score = max(0.0, min(1.0, score))
        return round(score, 4)

    def _standardize_tool_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = dict(result or {}) if isinstance(result, dict) else {
            "success": False,
            "error": str(result),
        }
        payload["tool_name"] = tool_name
        payload["tool_arguments"] = dict(arguments or {})
        if payload.get("success"):
            payload["failure_type"] = None
        else:
            if payload.get("transient_api_error"):
                payload["failure_type"] = "tool_api_transient"
            else:
                payload["failure_type"] = payload.get("failure_type") or self._classify_tool_failure(tool_name, payload.get("error", ""))
        payload["information_gain"] = payload.get("information_gain")
        if payload["information_gain"] is None:
            payload["information_gain"] = self._estimate_information_gain(tool_name, payload)
        payload["low_information_gain"] = bool(payload.get("success")) and float(payload.get("information_gain", 0.0)) < 0.1
        payload.setdefault("selector_feedback", "")
        if tool_name == "pi3_tool":
            azimuth = float(payload.get("azimuth_angle", 0) or 0)
            elevation = float(payload.get("elevation_angle", 0) or 0)
            ref_camera = int(arguments.get("rotation_reference_camera", 1) or 1)
            camera_view = bool(arguments.get("camera_view", False))
            payload["view_key"] = f"{azimuth:.1f}:{elevation:.1f}:ref{ref_camera}:cam{int(camera_view)}"
        return payload

    def _get_env_int(self, key: str, default: int) -> int:
        try:
            return int(os.environ.get(key, str(default)))
        except Exception:
            return default

    def _get_env_float(self, key: str, default: float) -> float:
        try:
            return float(os.environ.get(key, str(default)))
        except Exception:
            return default

    def _tool_retry_config(self) -> Dict[str, float]:
        return {
            "max_attempts": float(self._get_env_int("SPAGENT_TOOL_RECOVERY_MAX_ATTEMPTS", 0)),
            "max_total_wait": max(0.0, self._get_env_float("SPAGENT_TOOL_RECOVERY_MAX_TOTAL_WAIT", 300.0)),
            "initial_wait": max(1.0, self._get_env_float("SPAGENT_TOOL_RECOVERY_INITIAL_WAIT", 10.0)),
            "max_wait": max(1.0, self._get_env_float("SPAGENT_TOOL_RECOVERY_MAX_WAIT", 300.0)),
            "backoff": max(1.0, self._get_env_float("SPAGENT_TOOL_RECOVERY_BACKOFF", 2.0)),
            "jitter": max(0.0, self._get_env_float("SPAGENT_TOOL_RECOVERY_JITTER", 0.1)),
        }

    def _retry_after_from_exception(self, exc: Exception) -> Optional[float]:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) if response is not None else {}
        retry_after = None
        if isinstance(headers, dict):
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is None:
            return None
        try:
            return max(0.0, float(retry_after))
        except Exception:
            return None

    def _is_retryable_tool_exception(self, exc: Exception) -> bool:
        if requests is not None and isinstance(
            exc,
            (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ProxyError,
                requests.exceptions.SSLError,
            ),
        ):
            return True
        status_code = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        try:
            status_code = int(status_code) if status_code is not None else None
        except Exception:
            status_code = None
        if status_code in {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}:
            return True
        if status_code in {400, 401, 403, 404, 422}:
            return False
        return self._error_text_is_retryable(str(exc))

    def _error_text_is_retryable(self, error_text: str) -> bool:
        text = str(error_text or "").lower()
        retryable_markers = [
            "connection error",
            "connection refused",
            "connection reset",
            "connection aborted",
            "remote disconnected",
            "read timed out",
            "timed out",
            "timeout",
            "max retries exceeded",
            "bad gateway",
            "gateway timeout",
            "service unavailable",
            "temporarily unavailable",
            "too many requests",
            "rate limit",
            "no route to host",
            "network is unreachable",
            "502",
            "503",
            "504",
            "522",
            "524",
        ]
        return any(marker in text for marker in retryable_markers)

    def _tool_service_unhealthy(self, tool: Tool) -> Optional[bool]:
        client = getattr(tool, "_client", None)
        if client is None:
            return None
        health = getattr(client, "health_check", None)
        if not callable(health):
            return None
        try:
            result = health()
        except Exception as exc:
            logger.warning("Tool %s health check raised during recovery probe: %s", tool.name, exc)
            return True
        if result is None:
            return True
        if isinstance(result, dict):
            if result.get("success") is False:
                return True
            status = str(result.get("status", "")).lower()
            if any(token in status for token in ["healthy", "ok", "ready", "running"]):
                return False
            if any(token in status for token in ["unhealthy", "error", "failed", "down"]):
                return True
            if result.get("model_loaded") is False:
                return True
        return False

    def _is_retryable_tool_result(self, tool: Tool, result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict) or result.get("success"):
            return False
        if result.get("transient_api_error"):
            return True
        error_text = str(result.get("error") or "")
        if self._error_text_is_retryable(error_text):
            return True
        ambiguous_empty = any(
            marker in error_text.lower()
            for marker in [
                "no result returned",
                "unknown error",
                "returned none",
            ]
        )
        if ambiguous_empty:
            return self._tool_service_unhealthy(tool) is True
        return False

    def _sleep_before_tool_retry(
        self,
        tool_name: str,
        attempt: int,
        error_summary: str,
        wait_s: float,
        retry_after: Optional[float] = None,
        max_sleep_s: Optional[float] = None,
    ) -> Tuple[float, float]:
        config = self._tool_retry_config()
        sleep_s = retry_after if retry_after is not None else wait_s
        if retry_after is None and config["jitter"] > 0:
            sleep_s += random.uniform(0.0, sleep_s * config["jitter"])
        if max_sleep_s is not None:
            sleep_s = min(sleep_s, max(0.0, max_sleep_s))
        logger.warning(
            "Tool service/API transient error for %s; attempt=%s; waiting %.1fs before retry. Error: %s",
            tool_name,
            attempt,
            sleep_s,
            error_summary,
        )
        print(
            "[工具恢复重试] "
            f"{tool_name} 遇到可恢复的服务/API错误，attempt={attempt}，"
            f"等待 {sleep_s:.1f}s 后继续重试；这不会计为系统/skill失败。"
        )
        time.sleep(sleep_s)
        return min(config["max_wait"], wait_s * config["backoff"]), sleep_s

    def _tool_recovery_exhausted_result(
        self,
        tool_name: str,
        attempt: int,
        error_summary: str,
        total_wait_s: float,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "error": (
                f"Tool service/API for {tool_name} stayed unavailable after "
                f"{attempt} attempt(s) and {total_wait_s:.1f}s recovery wait. "
                "This is a transient external service failure, not usable scene evidence. "
                "Continue with another available tool if possible, otherwise answer with explicit uncertainty."
            ),
            "transient_api_error": True,
            "transient_recovery_exhausted": True,
            "transient_retry_attempts": attempt,
            "transient_total_wait_s": round(total_wait_s, 3),
            "last_transient_error": error_summary,
        }

    def _call_tool_with_recovery(self, tool: Tool, arguments: Dict[str, Any]) -> Dict[str, Any]:
        config = self._tool_retry_config()
        max_attempts = int(config["max_attempts"])
        max_total_wait_s = float(config["max_total_wait"])
        wait_s = min(config["initial_wait"], config["max_wait"])
        attempt = 0
        total_wait_s = 0.0

        while True:
            attempt += 1
            try:
                result = tool.call(**arguments)
            except Exception as exc:
                if not self._is_retryable_tool_exception(exc):
                    raise
                error_summary = f"{type(exc).__name__}: {exc}"
                if (max_attempts > 0 and attempt >= max_attempts) or (
                    max_total_wait_s > 0 and total_wait_s >= max_total_wait_s
                ):
                    return self._tool_recovery_exhausted_result(tool.name, attempt, error_summary, total_wait_s)
                remaining_wait_s = None
                if max_total_wait_s > 0:
                    remaining_wait_s = max(0.0, max_total_wait_s - total_wait_s)
                    if remaining_wait_s <= 0:
                        return self._tool_recovery_exhausted_result(tool.name, attempt, error_summary, total_wait_s)
                wait_s, slept_s = self._sleep_before_tool_retry(
                    tool.name,
                    attempt,
                    error_summary,
                    wait_s,
                    retry_after=self._retry_after_from_exception(exc),
                    max_sleep_s=remaining_wait_s,
                )
                total_wait_s += slept_s
                continue

            if not isinstance(result, dict):
                result = {
                    "success": False,
                    "error": f"Tool returned non-dict result: {type(result).__name__}",
                }

            if self._is_retryable_tool_result(tool, result):
                error_summary = str(result.get("error") or result)
                if (max_attempts > 0 and attempt >= max_attempts) or (
                    max_total_wait_s > 0 and total_wait_s >= max_total_wait_s
                ):
                    payload = self._tool_recovery_exhausted_result(tool.name, attempt, error_summary, total_wait_s)
                    payload.update({k: v for k, v in dict(result).items() if k not in payload})
                    payload["transient_api_error"] = True
                    payload["transient_recovery_exhausted"] = True
                    return payload
                remaining_wait_s = None
                if max_total_wait_s > 0:
                    remaining_wait_s = max(0.0, max_total_wait_s - total_wait_s)
                    if remaining_wait_s <= 0:
                        payload = self._tool_recovery_exhausted_result(tool.name, attempt, error_summary, total_wait_s)
                        payload.update({k: v for k, v in dict(result).items() if k not in payload})
                        payload["transient_api_error"] = True
                        payload["transient_recovery_exhausted"] = True
                        return payload
                wait_s, slept_s = self._sleep_before_tool_retry(
                    tool.name,
                    attempt,
                    error_summary,
                    wait_s,
                    max_sleep_s=remaining_wait_s,
                )
                total_wait_s += slept_s
                continue

            if attempt > 1:
                result = dict(result)
                result["transient_recovered"] = True
                result["transient_retry_attempts"] = attempt - 1
                result["transient_total_wait_s"] = round(total_wait_s, 3)
            return result

    def _update_view_state(
        self,
        view_state: Dict[str, Any],
        tool_results: Dict[str, Any],
    ) -> None:
        if not isinstance(view_state, dict):
            return
        view_state.setdefault("tried_views", [])
        view_state.setdefault("tried_reference_cameras", [])
        view_state.setdefault("view_gain_history", [])
        for result in (tool_results or {}).values():
            if not isinstance(result, dict):
                continue
            if str(result.get("tool_name", "")).strip() != "pi3_tool":
                continue
            view_key = str(result.get("view_key", "")).strip()
            ref_camera = int(result.get("tool_arguments", {}).get("rotation_reference_camera", 1) or 1)
            if view_key and view_key not in view_state["tried_views"]:
                view_state["tried_views"].append(view_key)
            if ref_camera not in view_state["tried_reference_cameras"]:
                view_state["tried_reference_cameras"].append(ref_camera)
            view_state["view_gain_history"].append(
                {
                    "view_key": view_key,
                    "information_gain": float(result.get("information_gain", 0.0) or 0.0),
                    "success": bool(result.get("success")),
                }
            )

    def _build_output_path_sources(
        self,
        tool_results: Dict[str, Any],
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for result in (tool_results or {}).values():
            if not isinstance(result, dict):
                continue
            tool_name = str(result.get("tool_name", "")).strip()
            if not tool_name:
                continue
            for key in ("output_path", "target_output_path", "vis_path", "overlay_path", "mask_path"):
                path = result.get(key)
                if path:
                    mapping[str(path)] = tool_name
            for path in result.get("output_paths", []) or []:
                if path:
                    mapping[str(path)] = tool_name
        return mapping

    def _select_analysis_images(
        self,
        image_paths: List[str],
        additional_images: List[str],
        tool_results: Dict[str, Any],
        skill_bundle: Optional[Dict[str, Any]],
        limit: int = 4,
    ) -> List[str]:
        valid_images = self._sort_additional_images_by_input_order(image_paths, additional_images)
        if not valid_images:
            return image_paths

        path_to_tool = self._build_output_path_sources(tool_results)
        problem_card = (skill_bundle or {}).get("problem_card", {}) if isinstance(skill_bundle, dict) else {}
        priority_tools = [str(tool).strip() for tool in problem_card.get("priority_tools", []) if str(tool).strip()]

        def rank(path: str) -> Tuple[int, int]:
            tool_name = path_to_tool.get(path, "")
            priority_bonus = 2 if tool_name in priority_tools else 0
            pi3_bonus = 1 if tool_name == "pi3_tool" else 0
            recency_bonus = 1
            return (priority_bonus + pi3_bonus + recency_bonus, 0)

        ranked = sorted(valid_images, key=rank, reverse=True)
        selected = ranked[: max(1, int(limit))]
        return image_paths + selected
    
    def add_tool(self, tool: Tool):
        """
        Add a tool to the agent
        
        Args:
            tool: Tool instance to add
        """
        self.tool_registry.register(tool)
    
    def remove_tool(self, tool_name: str):
        """
        Remove a tool from the agent
        
        Args:
            tool_name: Name of tool to remove
        """
        self.tool_registry.unregister(tool_name)
    
    def list_tools(self) -> List[str]:
        """
        Get list of available tool names
        
        Returns:
            List of tool names
        """
        return self.tool_registry.list_tools()
    
    def set_model(self, model: Model):
        """
        Set the VLLM model
        
        Args:
            model: Model instance to use
        """
        self.model = model
        logger.info(f"Updated model to: {model.model_name}")

    def update_skills_with_feedback(
        self,
        question: str,
        tool_calls: List[Dict[str, Any]],
        tool_results: Dict[str, Any],
        reward_score: Optional[float] = None,
        is_correct: Optional[bool] = None,
        success: bool = True,
        task_type: Optional[str] = None,
    ):
        """
        External feedback hook for delayed supervision (e.g., evaluation correctness).
        """
        if self.skill_manager is None:
            return
        if not self.enable_skill_update:
            return
        self.skill_manager.record_episode(
            question=question,
            tool_calls=tool_calls,
            tool_results=tool_results,
            success=success,
            reward_score=reward_score,
            is_correct=is_correct,
            reflection=None,
            task_type=task_type,
        )
    
    def solve_problem(
        self, 
        image_path: Union[str, List[str]], 
        question: str,
        max_iterations: int = 3,
        video_path: Optional[str] = None,
        pi3_num_frames: int = 7,
        use_baseline_comparison: bool = False,
        reward_score: Optional[float] = None,
        is_correct: Optional[bool] = None,
        oracle_ground_truth: Optional[str] = None,
        auto_update_skills: bool = True,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Solve a spatial intelligence problem
        
        Args:
            image_path: Path to image or list of image paths
            question: User's question about the image(s)
            max_iterations: Maximum number of tool-call iterations (default: 1)
            video_path: Optional path to original video (for pi3 tool re-sampling)
            pi3_num_frames: Number of frames to uniformly sample for pi3 tool (default 10)
            use_baseline_comparison: If True, run a naive baseline (no tools) in parallel
                                    and synthesize final answer from both results (default: False)
            reward_score: Optional reward in [0, 1] from external evaluator
            is_correct: Optional correctness flag from external evaluator
            oracle_ground_truth: Optional ground-truth answer used only when
                SPAGENT_ORACLE_ROLLOUT=1 for teacher-forced SFT rollout collection
            auto_update_skills: Whether to auto-update learned skills at end of this run
            **model_kwargs: Additional arguments for model inference
            
        Returns:
            Dictionary containing:
            - answer: Final answer text
            - initial_response: Model's initial response
            - tool_calls: List of tool calls made
            - tool_results: Results from tool execution
            - used_tools: List of tools that were used
            - additional_images: List of additional images generated by tools
            - iterations: Number of iterations performed
            - baseline_answer: Naive baseline answer (if use_baseline_comparison=True)
        """
        logger.info(f"Starting problem solving for question: {question} (max_iterations={max_iterations})")
        total_start = time.perf_counter()
        timing_profile: Dict[str, Any] = {
            "schema_version": 1,
            "total_wall_time_s": 0.0,
            "input_validation_s": 0.0,
            "tool_schema_s": 0.0,
            "memory_retrieval_s": 0.0,
            "prompt_build_s": 0.0,
            "working_memory_s": 0.0,
            "model_inference_s": 0.0,
            "data_collection_s": 0.0,
            "skill_choice_parsing_s": 0.0,
            "tool_call_parsing_s": 0.0,
            "tool_selection_s": 0.0,
            "baseline_inference_s": 0.0,
            "tool_execution_wall_s": 0.0,
            "view_state_update_s": 0.0,
            "post_tool_processing_s": 0.0,
            "final_prompt_build_s": 0.0,
            "baseline_synthesis_s": 0.0,
            "cleanup_s": 0.0,
            "answer_extraction_s": 0.0,
            "skill_reflection_s": 0.0,
            "skill_update_s": 0.0,
            "retrieval_trace_s": 0.0,
            "model_calls": [],
            "tool_calls": [],
            "tool_execution_by_name": {},
        }

        def _add_duration(key: str, duration: float) -> None:
            timing_profile[key] = float(timing_profile.get(key, 0.0) or 0.0) + max(0.0, duration)

        def _add_elapsed(key: str, start: float) -> None:
            _add_duration(key, time.perf_counter() - start)
        
        # Validate inputs
        input_start = time.perf_counter()
        if isinstance(image_path, str):
            image_paths = [image_path]
        else:
            image_paths = image_path
        
        for path in image_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Image not found: {path}")
        _add_elapsed("input_validation_s", input_start)

        session_id = f"infer_{hash(question) & 0xffffffff}"

        def _write_working_note(note_type: str, payload: Dict[str, Any]) -> None:
            if self.skill_manager is None:
                return
            note_start = time.perf_counter()
            self.skill_manager.write_working_note(
                session_id=session_id,
                note_type=note_type,
                payload=payload,
            )
            _add_elapsed("working_memory_s", note_start)

        def _run_model_inference(images: List[str], prompt: str, phase: str) -> str:
            call_start = time.perf_counter()
            success = False
            try:
                if len(images) == 1:
                    response = self.model.single_image_inference(
                        images[0],
                        prompt,
                        **model_kwargs,
                    )
                else:
                    response = self.model.multiple_images_inference(
                        images,
                        prompt,
                        **model_kwargs,
                    )
                success = True
                return response
            finally:
                duration = time.perf_counter() - call_start
                _add_duration("model_inference_s", duration)
                timing_profile.setdefault("model_calls", []).append({
                    "phase": phase,
                    "image_count": len(images),
                    "duration_s": duration,
                    "success": success,
                })
        
        # Start data collection session if enabled
        if self.data_collector:
            data_start = time.perf_counter()
            self.data_collector.start_session(question, image_paths)
            _add_elapsed("data_collection_s", data_start)
        
        # Create system prompt with available tools
        schema_start = time.perf_counter()
        tool_schemas = self.tool_registry.get_function_schemas()
        _add_elapsed("tool_schema_s", schema_start)
        skill_bundle = None
        if self.prompt_style == "skills":
            if self.skill_manager is not None:
                memory_start = time.perf_counter()
                skill_bundle = self.skill_manager.build_prompt_bundle(
                    question=question,
                    memory_mode=self.memory_mode,
                    retrieval_mode=self.retrieval_mode,
                    retrieval_top_k=self.retrieval_top_k,
                )
                _add_elapsed("memory_retrieval_s", memory_start)
            prompt_start = time.perf_counter()
            system_prompt = create_system_prompt_with_skills(
                tool_schemas,
                skill_bundle=skill_bundle,
                allow_model_skill_choice=self.allow_model_skill_choice,
            )
            skill_choice_block = self._build_skill_choice_reporting_block(
                skill_bundle,
                include_catalog_snapshot=False,
            )
            if skill_choice_block:
                system_prompt = system_prompt + "\n\n" + skill_choice_block
            _add_elapsed("prompt_build_s", prompt_start)
            logger.info("Skill reference bundle injected into system prompt")
        else:
            prompt_start = time.perf_counter()
            system_prompt = create_system_prompt(tool_schemas)
            _add_elapsed("prompt_build_s", prompt_start)
        # system_prompt += '\nThis time, you need to call the function anyway in <tool_call></tool_call> format.' # Debug Only!
        prompt_start = time.perf_counter()
        user_prompt = create_user_prompt(question, image_paths, tool_schemas)
        oracle_hint = ""
        if self.oracle_rollout and oracle_ground_truth is not None:
            oracle_hint = self._build_oracle_rollout_hint(oracle_ground_truth)
            user_prompt = user_prompt + "\n\n" + oracle_hint
        _add_elapsed("prompt_build_s", prompt_start)
        
        # Initialize tracking variables for multi-step workflow
        all_tool_calls = []
        all_tool_results = {}
        all_additional_images = []
        all_successful_tools = []
        current_images = image_paths
        initial_response = None
        last_response = None
        last_response_has_usable_answer = False
        iteration = 0
        baseline_answer = None  # For naive baseline comparison
        baseline_triggered = False  # Track if baseline has been triggered
        all_skill_choices = []
        skill_choices: List[Dict[str, Any]] = []
        auto_skill_choices: List[Dict[str, Any]] = self._auto_selected_skill_choices(skill_bundle)
        if auto_skill_choices:
            logger.info("Auto-selected skills from retrieval: %s", auto_skill_choices)
            for choice in auto_skill_choices:
                annotated = dict(choice)
                annotated["iteration"] = 0
                annotated["phase"] = "auto_retrieval"
                all_skill_choices.append(annotated)
        last_logged_skill_choice_response = None
        format_violations: List[Dict[str, Any]] = []
        selector_decisions = []
        view_state = {
            "tried_views": [],
            "tried_reference_cameras": [],
            "view_gain_history": [],
            "unresolved_regions": [],
            "occlusion_hypotheses": [],
        }
        pending_selector_note = None
        
        # Multi-step workflow loop
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"=== Iteration {iteration}/{max_iterations} ===")
            _write_working_note(
                note_type="iteration_start",
                payload={
                    "iteration": iteration,
                    "question": question,
                    "current_images": current_images,
                    "view_state": view_state,
                },
            )
            
            # Step 1: Get model response
            if iteration == 1:
                # First iteration: use initial prompt
                prompt_start = time.perf_counter()
                prompt = system_prompt + "\n\n" + user_prompt
                _add_elapsed("prompt_build_s", prompt_start)
                logger.info("Getting initial model response...")
            else:
                # Subsequent iterations: create continuation prompt
                prompt_start = time.perf_counter()
                prompt = self._create_continuation_prompt(
                    question, 
                    last_response, 
                    all_tool_results, 
                    image_paths,
                    all_additional_images,
                    iteration,
                    max_iterations,
                    skill_bundle=skill_bundle,
                    selector_note=pending_selector_note,
                    oracle_hint=oracle_hint,
                )
                _add_elapsed("prompt_build_s", prompt_start)
                logger.info(f"Getting continuation response for iteration {iteration}...")
                pending_selector_note = None
            
            current_response = _run_model_inference(
                current_images,
                prompt,
                phase=f"iteration_{iteration}",
            )
            repaired_response = self._repair_missing_think_response(current_response)
            if repaired_response != current_response:
                logger.info("Repaired response by wrapping pre-tag reasoning in <think> for iteration %s", iteration)
                current_response = repaired_response
            cleaned_response = self._strip_disallowed_skill_choice_blocks(current_response)
            if cleaned_response != current_response:
                logger.info("Removed model-emitted <skill_choice> blocks because model skill choice is disabled")
                current_response = cleaned_response
            
            logger.info(f"Response: {current_response}")
            
            # Save initial response
            if iteration == 1:
                initial_response = current_response
            last_response = current_response
            
            # Record inference data if collection is enabled
            if self.data_collector:
                data_start = time.perf_counter()
                self.data_collector.record_inference(
                    iteration=iteration,
                    images=current_images,
                    prompt=prompt,
                    response=current_response,
                    context={
                        "tool_calls_history": all_tool_calls,
                        "tool_results_history": all_tool_results,
                        "additional_images_history": all_additional_images
                    }
                )
                _add_elapsed("data_collection_s", data_start)
            _write_working_note(
                note_type="model_response",
                payload={
                    "iteration": iteration,
                    "response": current_response,
                },
            )

            response_format_errors = self._response_format_errors(current_response)
            has_complete_answer = self._has_answer_tags(current_response)
            if response_format_errors:
                violation = {
                    "iteration": iteration,
                    "phase": "iteration",
                    "errors": response_format_errors,
                }
                format_violations.append(violation)
                logger.warning("Response format violation in iteration %s: %s", iteration, response_format_errors)
                _write_working_note(note_type="format_violation", payload=violation)
                if self.stop_on_answer and has_complete_answer:
                    logger.info(
                        "Ending workflow immediately because iteration %s produced complete <answer>; "
                        "format errors will be recorded but no continuation/tool execution will run.",
                        iteration,
                    )
                    last_response_has_usable_answer = True
                    _write_working_note(
                        note_type="answer_stop",
                        payload={
                            "iteration": iteration,
                            "reason": "complete_answer_with_format_violation",
                            "format_errors": response_format_errors,
                        },
                    )
                    break
                repairable_format_errors = {
                    "missing_think",
                    "empty_think",
                    "answer_without_prior_think_reason",
                }
                if repairable_format_errors & set(response_format_errors) and iteration < max_iterations:
                    allowed_outputs = "<tool_call> or <answer>"
                    if self.allow_model_skill_choice:
                        allowed_outputs = "optional <skill_choice>, <tool_call>, or <answer>"
                    pending_selector_note = (
                        "Format violation: every response turn must start with a non-empty <think></think> block. "
                        f"Restate the reasoning in <think> before any {allowed_outputs}. "
                        "If you are giving the final answer after tool use, explain inside <think> which tool evidence supports it before <answer>."
                    )
                    continue
            
            # Step 2: Parse skill choices and tool calls from response
            parse_start = time.perf_counter()
            skill_choices = self._parse_skill_choices(
                current_response,
                skill_bundle=skill_bundle,
            )
            _add_elapsed("skill_choice_parsing_s", parse_start)
            if skill_choices:
                logger.info("Skill choices for iteration %s: %s", iteration, skill_choices)
                for choice in skill_choices:
                    annotated = dict(choice)
                    annotated["iteration"] = iteration
                    annotated["phase"] = "iteration"
                    all_skill_choices.append(annotated)
                last_logged_skill_choice_response = current_response
            effective_skill_choices = self._merge_skill_choices(skill_choices, auto_skill_choices)
            _write_working_note(
                note_type="parsed_skill_choices",
                payload={
                    "iteration": iteration,
                    "skill_choices": skill_choices,
                    "auto_skill_choices": auto_skill_choices,
                    "effective_skill_choices": effective_skill_choices,
                },
            )

            parse_start = time.perf_counter()
            tool_calls = self._parse_tool_calls(current_response)
            _add_elapsed("tool_call_parsing_s", parse_start)

            has_answer = has_complete_answer
            if self.stop_on_answer and has_answer:
                if tool_calls:
                    logger.warning(
                        "Ending workflow immediately because <answer> was produced in iteration %s; "
                        "ignoring %s same-turn tool call(s). Format reward/evaluation should penalize mixed tags.",
                        iteration,
                        len(tool_calls),
                    )
                else:
                    logger.info("Ending workflow immediately because <answer> was produced in iteration %s", iteration)
                last_response_has_usable_answer = True
                _write_working_note(
                    note_type="answer_stop",
                    payload={
                        "iteration": iteration,
                        "reason": "complete_answer",
                        "ignored_same_turn_tool_calls": tool_calls,
                    },
                )
                break

            # Trigger naive baseline if enabled and tool calls detected
            if use_baseline_comparison and tool_calls and not baseline_triggered:
                baseline_triggered = True
                logger.info("Tool calls detected - triggering naive baseline agent...")
                baseline_start = time.perf_counter()
                baseline_answer = self._get_naive_baseline_answer(image_paths, question, **model_kwargs)
                _add_elapsed("baseline_inference_s", baseline_start)
                logger.info(f"Naive baseline answer: {baseline_answer}")
            
            # Check if response has <answer> tags (indicating completion)
            if tool_calls and has_answer:
                logger.info("Ignoring same-turn <answer> because <tool_call> was also provided; tool results must be incorporated in a later turn.")
                has_answer = False

            if self.force_skill_tool_call and self.allow_model_skill_choice and not all_tool_calls:
                valid_model_skill_choices = [
                    choice for choice in self._active_skill_choices(skill_choices)
                    if choice.get("resolved_from_catalog")
                ]
                force_errors = []
                if not valid_model_skill_choices:
                    force_errors.append("missing_valid_skill_choice_before_first_tool")
                if not tool_calls:
                    force_errors.append("missing_tool_call_before_first_answer")
                if force_errors:
                    violation = {
                        "iteration": iteration,
                        "phase": "forced_skill_tool_protocol",
                        "errors": force_errors,
                    }
                    format_violations.append(violation)
                    _write_working_note(note_type="format_violation", payload=violation)
                    logger.warning("Forced skill+tool protocol violation in iteration %s: %s", iteration, force_errors)
                    if iteration < max_iterations:
                        pending_selector_note = (
                            "Protocol violation: before the first final answer, you must choose one valid skill id from "
                            "<skill_candidates> using <skill_choice>{...}</skill_choice>, then immediately output "
                            "one executable <tool_call>{...}</tool_call>. Do not answer directly from RGB. "
                            "Use the selected skill's required_tools/tool workflow and schema-valid arguments."
                        )
                        continue
                    has_answer = False
            last_response_has_usable_answer = has_answer
            selection_start = time.perf_counter()
            missing_must_try = self._missing_must_try_tools(
                skill_bundle,
                all_tool_calls,
                skill_choices=effective_skill_choices,
            )
            _add_elapsed("tool_selection_s", selection_start)

            if not tool_calls:
                logger.info(f"No tool calls found in iteration {iteration}")
                if has_answer and missing_must_try and iteration < max_iterations:
                    pending_selector_note = (
                        "Selector override: do not answer yet. Required tool(s) still not attempted: "
                        + ", ".join(missing_must_try)
                        + ". Attempt them before the final answer."
                    )
                    logger.info("Continuing because must-try tools are still missing: %s", missing_must_try)
                    continue
                if has_answer or iteration == max_iterations:
                    logger.info("Ending workflow: final answer provided or max iterations reached")
                    break
                continue

            selection_start = time.perf_counter()
            approved_tool_calls, rejected_tool_calls, selector_decision, selector_note = self._select_tool_calls(
                tool_calls=tool_calls,
                skill_choices=effective_skill_choices,
                skill_bundle=skill_bundle,
                question=question,
                all_tool_calls=all_tool_calls,
                all_tool_results=all_tool_results,
                iteration=iteration,
                view_state=view_state,
            )
            _add_elapsed("tool_selection_s", selection_start)
            selector_decisions.append(selector_decision)
            if rejected_tool_calls:
                logger.info("Selector rejected %s tool call(s) in iteration %s", len(rejected_tool_calls), iteration)
            if not approved_tool_calls:
                logger.info("Selector approved no tool calls in iteration %s", iteration)
                if selector_note and iteration < max_iterations:
                    pending_selector_note = selector_note
                    continue
                if has_answer and missing_must_try and iteration < max_iterations:
                    pending_selector_note = selector_note or (
                        "Selector override: required tools are still missing: " + ", ".join(missing_must_try)
                    )
                    continue
                if has_answer or iteration == max_iterations:
                    break
                continue
            
            # Step 3: Execute tools
            logger.info(f"Executing {len(approved_tool_calls)} tool calls in iteration {iteration}...")
            tool_start = time.perf_counter()
            tool_results = self._execute_tools(
                approved_tool_calls,
                video_path=video_path,
                pi3_num_frames=pi3_num_frames,
                timing_profile=timing_profile,
            )
            _add_elapsed("tool_execution_wall_s", tool_start)
            rejected_results = {
                f"{call['name']}_rejected_{idx}": self._format_rejected_tool_result(
                    call["name"],
                    call.get("arguments", {}),
                    call.get("selector_reason", "Selector rejected tool call."),
                    failure_type=call.get("failure_type", "selector_rejected"),
                )
                for idx, call in enumerate(rejected_tool_calls)
            }
            tool_results.update(rejected_results)
            view_start = time.perf_counter()
            self._update_view_state(view_state, tool_results)
            _add_elapsed("view_state_update_s", view_start)
            _write_working_note(
                note_type="tool_execution",
                payload={
                    "iteration": iteration,
                    "tool_calls": approved_tool_calls,
                    "tool_results": tool_results,
                },
            )
            
            # Step 4: Collect results
            post_tool_start = time.perf_counter()
            iteration_additional_images = []
            for tool_name, result in tool_results.items():
                if result.get('success'):
                    all_successful_tools.append(f"{tool_name}_iter{iteration}")
                    # Collect additional images
                    candidate_paths = []
                    for key in ("output_path", "target_output_path", "vis_path"):
                        path = result.get(key)
                        if path is not None:
                            candidate_paths.append(path)
                    for path in result.get("output_paths", []) or []:
                        if path is not None:
                            candidate_paths.append(path)

                    for path in candidate_paths:
                        if Path(path).exists() and path not in iteration_additional_images:
                            iteration_additional_images.append(path)
            
            # Update tracking variables
            all_tool_calls.extend(approved_tool_calls)
            all_tool_results.update({f"{k}_iter{iteration}": v for k, v in tool_results.items()})
            all_additional_images.extend(iteration_additional_images)
            
            # Update current images for next iteration
            if iteration_additional_images:
                # Combine original images with ALL accumulated images (not just current iteration)
                # This ensures all previously generated images are available for next iteration
                valid_all_images = self._sort_additional_images_by_input_order(image_paths, all_additional_images)
                if valid_all_images:
                    # Keep original images + all accumulated additional images for comprehensive analysis
                    current_images = image_paths + valid_all_images
                # else: keep current_images unchanged
            _add_elapsed("post_tool_processing_s", post_tool_start)
            
            # Check if we should stop (has answer tag and we're not forcing more iterations)
            if has_answer and iteration < max_iterations:
                logger.info(f"Answer provided in iteration {iteration}, but continuing workflow if more tool calls exist")
        
        # Generate final response if needed
        if not last_response_has_usable_answer and all_successful_tools:
            logger.info("Generating final response...")
            final_prompt_start = time.perf_counter()
            tool_description = None
            if all_tool_results:
                last_result = list(all_tool_results.values())[-1]
                if last_result.get('description'):
                    tool_description = last_result.get('description')
            
            follow_up_prompt = create_follow_up_prompt(
                question, 
                initial_response, 
                all_tool_results,
                image_paths,
                all_additional_images,
                tool_description
            )
            skill_choice_block = self._build_skill_choice_reporting_block(
                skill_bundle,
                include_catalog_snapshot=True,
            )
            if skill_choice_block:
                follow_up_prompt = follow_up_prompt + "\n\n" + skill_choice_block
            
            final_images = self._select_analysis_images(
                image_paths=image_paths,
                additional_images=all_additional_images,
                tool_results=all_tool_results,
                skill_bundle=skill_bundle,
            )
            _add_elapsed("final_prompt_build_s", final_prompt_start)
            
            final_response = _run_model_inference(
                final_images,
                follow_up_prompt,
                phase="final_synthesis",
            )
            repaired_final_response = self._repair_missing_think_response(final_response)
            if repaired_final_response != final_response:
                logger.info("Repaired final synthesis response by wrapping pre-tag reasoning in <think>")
                final_response = repaired_final_response
            final_response = self._strip_disallowed_skill_choice_blocks(final_response)
            
            # Record final synthesis inference if collection is enabled
            if self.data_collector:
                data_start = time.perf_counter()
                self.data_collector.record_inference(
                    iteration=iteration + 1,  # Final synthesis is an additional step
                    images=final_images,
                    prompt=follow_up_prompt,
                    response=final_response,
                    context={
                        "type": "final_synthesis",
                        "tool_calls_history": all_tool_calls,
                        "tool_results_history": all_tool_results,
                        "additional_images_history": all_additional_images
                    }
                )
                _add_elapsed("data_collection_s", data_start)
                
        elif not last_response_has_usable_answer:
            # Generate fallback if no answer tags
            logger.warning("No answer tags found, generating fallback response")
            final_prompt_start = time.perf_counter()
            fallback_prompt = create_fallback_prompt(question, last_response)
            skill_choice_block = self._build_skill_choice_reporting_block(
                skill_bundle,
                include_catalog_snapshot=True,
            )
            if skill_choice_block:
                fallback_prompt = fallback_prompt + "\n\n" + skill_choice_block
            _add_elapsed("final_prompt_build_s", final_prompt_start)
            final_response = _run_model_inference(
                current_images,
                fallback_prompt,
                phase="fallback",
            )
            repaired_final_response = self._repair_missing_think_response(final_response)
            if repaired_final_response != final_response:
                logger.info("Repaired fallback response by wrapping pre-tag reasoning in <think>")
                final_response = repaired_final_response
            final_response = self._strip_disallowed_skill_choice_blocks(final_response)
            
            # Record fallback inference if collection is enabled
            if self.data_collector:
                data_start = time.perf_counter()
                self.data_collector.record_inference(
                    iteration=iteration + 1,
                    images=current_images,
                    prompt=fallback_prompt,
                    response=final_response,
                    context={
                        "type": "fallback",
                        "tool_calls_history": all_tool_calls,
                        "tool_results_history": all_tool_results
                    }
                )
                _add_elapsed("data_collection_s", data_start)
        else:
            final_response = last_response
        
        # Synthesize final answer with baseline comparison if enabled
        if use_baseline_comparison and baseline_answer is not None:
            logger.info("Synthesizing final answer from tool-based and baseline responses...")
            baseline_response = final_response  # Save pre-synthesis response
            synthesis_start = time.perf_counter()
            final_response = self._synthesize_with_baseline(
                question, 
                final_response, 
                baseline_answer, 
                image_paths,
                all_additional_images,
                **model_kwargs
            )
            repaired_final_response = self._repair_missing_think_response(final_response)
            if repaired_final_response != final_response:
                logger.info("Repaired baseline synthesis response by wrapping pre-tag reasoning in <think>")
                final_response = repaired_final_response
            final_response = self._strip_disallowed_skill_choice_blocks(final_response)
            _add_elapsed("baseline_synthesis_s", synthesis_start)
            
            # Record baseline synthesis inference if collection is enabled
            if self.data_collector:
                final_images = self._select_analysis_images(
                    image_paths=image_paths,
                    additional_images=all_additional_images,
                    tool_results=all_tool_results,
                    skill_bundle=skill_bundle,
                )
                data_start = time.perf_counter()
                self.data_collector.record_inference(
                    iteration=iteration + 2,  # Baseline synthesis is an additional step
                    images=final_images,
                    prompt=f"Synthesizing tool-based and baseline answers for: {question}",
                    response=final_response,
                    context={
                        "type": "baseline_synthesis",
                        "tool_based_answer": baseline_response,
                        "baseline_answer": baseline_answer,
                        "tool_calls_history": all_tool_calls,
                        "tool_results_history": all_tool_results
                    }
                )
                _add_elapsed("data_collection_s", data_start)
        
        # Clean up temporary pi3 frames
        cleanup_start = time.perf_counter()
        self._cleanup_pi3_frames()
        _add_elapsed("cleanup_s", cleanup_start)

        if final_response != last_logged_skill_choice_response:
            parse_start = time.perf_counter()
            final_skill_choices = self._parse_skill_choices(
                final_response,
                skill_bundle=skill_bundle,
            )
            _add_elapsed("skill_choice_parsing_s", parse_start)
            if final_skill_choices:
                logger.info("Skill choices for final response: %s", final_skill_choices)
                for choice in final_skill_choices:
                    annotated = dict(choice)
                    annotated["iteration"] = iteration
                    annotated["phase"] = "final_response"
                    all_skill_choices.append(annotated)
                last_logged_skill_choice_response = final_response
            final_format_errors = self._response_format_errors(final_response)
            if final_format_errors:
                violation = {
                    "iteration": iteration,
                    "phase": "final_response",
                    "errors": final_format_errors,
                }
                format_violations.append(violation)
                logger.warning("Final response format violation: %s", final_format_errors)

        answer_start = time.perf_counter()
        extracted_answer = self._extract_answer(final_response)
        _add_elapsed("answer_extraction_s", answer_start)

        # End data collection session if enabled
        if self.data_collector:
            success = extracted_answer is not None  # Success if we have an answer
            
            data_start = time.perf_counter()
            self.data_collector.end_session(
                success=success,
                final_answer=extracted_answer or final_response,
                error_message=None if success else "No answer tags found",
                metadata={
                    "iterations": iteration,
                    "num_tool_calls": len(all_tool_calls),
                    "used_tools": all_successful_tools,
                    "num_additional_images": len(all_additional_images),
                    "format_violations": format_violations,
                }
            )
            _add_elapsed("data_collection_s", data_start)

        # Optional model reflection to suggest a reusable skill candidate
        reflection = None
        do_skill_update = bool(auto_update_skills and self.enable_skill_update)

        if do_skill_update and self.skill_manager is not None and self.enable_skill_reflection:
            reflection_start = time.perf_counter()
            reflection = self._generate_skill_reflection(
                question=question,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                final_response=final_response,
                success=extracted_answer is not None,
            )
            _add_elapsed("skill_reflection_s", reflection_start)

        # Update adaptive skills from this runtime trajectory
        if do_skill_update and self.skill_manager is not None:
            try:
                skill_update_start = time.perf_counter()
                self.skill_manager.record_episode(
                    question=question,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    success=extracted_answer is not None,
                    reward_score=reward_score,
                    is_correct=is_correct,
                    reflection=reflection,
                )
                _add_elapsed("skill_update_s", skill_update_start)
            except Exception as exc:
                logger.warning("Adaptive skill update failed: %s", exc)

        trace_start = time.perf_counter()
        self._write_retrieval_trace(
            question=question,
            skill_bundle=skill_bundle,
            selected_skill_choices=all_skill_choices,
            tool_calls=all_tool_calls,
            is_correct=is_correct,
        )
        _add_elapsed("retrieval_trace_s", trace_start)
        timing_profile["total_wall_time_s"] = time.perf_counter() - total_start
        accounted_keys = [
            "input_validation_s",
            "tool_schema_s",
            "memory_retrieval_s",
            "prompt_build_s",
            "working_memory_s",
            "model_inference_s",
            "data_collection_s",
            "skill_choice_parsing_s",
            "tool_call_parsing_s",
            "tool_selection_s",
            "baseline_inference_s",
            "tool_execution_wall_s",
            "view_state_update_s",
            "post_tool_processing_s",
            "final_prompt_build_s",
            "baseline_synthesis_s",
            "cleanup_s",
            "answer_extraction_s",
            "skill_reflection_s",
            "skill_update_s",
            "retrieval_trace_s",
        ]
        accounted = sum(float(timing_profile.get(key, 0.0) or 0.0) for key in accounted_keys)
        timing_profile["accounted_wall_time_s"] = accounted
        timing_profile["other_wall_time_s"] = max(0.0, timing_profile["total_wall_time_s"] - accounted)
        
        return {
            "answer": final_response,
            "initial_response": initial_response or final_response,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "skill_choices": all_skill_choices,
            "selector_decisions": selector_decisions,
            "format_violations": format_violations,
            "used_tools": all_successful_tools,
            "additional_images": all_additional_images,
            "iterations": iteration,
            "baseline_answer": baseline_answer,  # Naive baseline answer (if enabled)
            "view_state": view_state,
            "memory_trace": {
                "memory_mode": self.memory_mode,
                "retrieval_mode": self.retrieval_mode,
                "retrieved_skills": (skill_bundle or {}).get("retrieved_skills", []) if isinstance(skill_bundle, dict) else [],
                "retrieved_episodes": (skill_bundle or {}).get("retrieved_episodes", []) if isinstance(skill_bundle, dict) else [],
                "retrieved_evidence": (skill_bundle or {}).get("retrieved_evidence", []) if isinstance(skill_bundle, dict) else [],
                "retrieval_diagnostics": (skill_bundle or {}).get("retrieval_diagnostics", {}) if isinstance(skill_bundle, dict) else {},
            },
            "timing_profile": timing_profile,
            "prompts": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "follow_up_prompt": None,  # This is now handled in continuation prompts
                "skill_bundle": skill_bundle,
            }
        }
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from model response
        
        Args:
            response: Model response text
            
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        
        # Find all tool_call blocks
        pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            tool_call = parse_tool_call_block(match)
            normalized_call = normalize_tool_call_payload(tool_call, self.tool_registry) if tool_call else None
            if normalized_call and 'name' in normalized_call and 'arguments' in normalized_call:
                args = normalized_call.get("arguments", {})
                image_paths = None
                if isinstance(args, dict):
                    raw_paths = args.get("image_paths") or args.get("images")
                    if isinstance(raw_paths, list) and normalized_call.get("name") != "pi3_tool":
                        image_paths = [path for path in raw_paths if path]
                if image_paths:
                    for image_path in image_paths:
                        expanded_args = dict(args)
                        expanded_args.pop("image_paths", None)
                        expanded_args.pop("images", None)
                        expanded_args["image_path"] = image_path
                        tool_calls.append({"name": normalized_call["name"], "arguments": expanded_args})
                else:
                    tool_calls.append(normalized_call)
            else:
                logger.warning(f"Invalid tool call format: {match}")
        
        return tool_calls

    def _parse_skill_choices(
        self,
        response: str,
        skill_bundle: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse skill choice blocks from model response.

        Expected format:
        <skill_choice>{"id":"...", "source":"static|dynamic|none", "name":"...", "reason":"..."}</skill_choice>
        """
        if not response:
            return []

        skill_choices: List[Dict[str, Any]] = []
        skill_index = self._build_skill_catalog_index(skill_bundle)
        pattern = r'<skill_choice>\s*(.*?)\s*</skill_choice>'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            payload = None
            try:
                payload = json.loads(match)
            except Exception:
                try:
                    payload = ast.literal_eval(match)
                except Exception:
                    logger.warning("Invalid skill choice format: %s", match)
                    continue

            records = payload if isinstance(payload, list) else [payload]
            for record in records:
                if not isinstance(record, dict):
                    continue
                skill_id = str(record.get("id", "")).strip()
                reported_source = str(record.get("source", "")).strip().lower()
                reported_name = str(record.get("name", "")).strip()
                reason = str(record.get("reason", "")).strip()
                if not skill_id:
                    continue

                if skill_id == "none":
                    source = "none"
                    name = "none"
                    catalog_match = False
                else:
                    catalog_card = skill_index.get(skill_id, {})
                    source = str(catalog_card.get("source", "")).strip().lower() or reported_source or "unknown"
                    name = str(catalog_card.get("name", "")).strip() or reported_name or skill_id
                    catalog_match = bool(catalog_card)

                choice_record = {
                    "id": skill_id,
                    "source": source,
                    "name": name,
                    "reason": reason,
                    "resolved_from_catalog": catalog_match,
                }
                if reported_source and reported_source != source:
                    choice_record["reported_source"] = reported_source
                if reported_name and reported_name != name:
                    choice_record["reported_name"] = reported_name
                skill_choices.append(choice_record)

        return skill_choices
    
    def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        video_path: Optional[str] = None,
        pi3_num_frames: int = 10,
        timing_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute tool calls in parallel when possible
        
        Args:
            tool_calls: List of tool call dictionaries
            video_path: Optional path to original video (for pi3 tool re-sampling)
            pi3_num_frames: Number of frames to uniformly sample for pi3 tool
            
        Returns:
            Dictionary of tool_name -> result
        """
        tool_results = {}

        def _add_profile_duration(key: str, duration: float) -> None:
            if timing_profile is None:
                return
            timing_profile[key] = float(timing_profile.get(key, 0.0) or 0.0) + max(0.0, duration)

        def _record_tool_timing(tool_name: str, result_key: str, duration: float, result: Dict[str, Any]) -> None:
            if timing_profile is None:
                return
            success = bool(result.get("success")) if isinstance(result, dict) else False
            timing_profile.setdefault("tool_calls", []).append({
                "tool_name": tool_name,
                "result_key": result_key,
                "duration_s": duration,
                "success": success,
            })
            by_name = timing_profile.setdefault("tool_execution_by_name", {})
            stats = by_name.setdefault(
                tool_name,
                {
                    "calls": 0,
                    "success": 0,
                    "failure": 0,
                    "wall_time_s_sum": 0.0,
                },
            )
            stats["calls"] += 1
            stats["success" if success else "failure"] += 1
            stats["wall_time_s_sum"] += duration

        def _timed_safe_tool_call(tool: Tool, arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
            call_start = time.perf_counter()
            result = self._safe_tool_call(tool, arguments)
            return result, time.perf_counter() - call_start
        
        # Normalize tool names and arguments
        normalized_calls = []
        for call in tool_calls:
            raw_name = call.get("name", "")
            name = normalize_tool_name(raw_name, self.tool_registry)
            arguments = call.get("arguments", {}) if isinstance(call, dict) else {}
            normalized_calls.append({"name": name, "arguments": arguments})

        # Group tool calls by tool name to handle multiple calls to same tool
        tool_groups = {}
        for i, call in enumerate(normalized_calls):
            tool_name = call['name']
            if tool_name not in tool_groups:
                tool_groups[tool_name] = []
            tool_groups[tool_name].append((i, call))
        
        # Execute tools in parallel, but Pi3Tool and Pi3MultiimgTool sequentially to avoid server issues
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tool = {}
            
            # Handle Pi3Tool and Pi3MultiimgTool calls sequentially first
            pi3_calls = []
            other_calls = {}
            
            for tool_name, calls in tool_groups.items():
                if tool_name in ['pi3_tool', 'pi3_multiimg_tool']:
                    pi3_calls.extend(calls)
                else:
                    other_calls[tool_name] = calls
            
            # Execute Pi3Tool and Pi3MultiimgTool calls sequentially
            if pi3_calls:
                logger.info(f"Executing {len(pi3_calls)} Pi3 tool calls sequentially...")
                
                # Extract more frames for pi3 if video_path is provided
                pi3_frame_paths = []
                if video_path and Path(video_path).exists():
                    logger.info(f"Extracting {pi3_num_frames} frames for pi3 tool from video: {video_path}")
                    frame_start = time.perf_counter()
                    pi3_frame_paths = self._extract_frames_for_pi3(video_path, pi3_num_frames)
                    _add_profile_duration("pi3_frame_extraction_s", time.perf_counter() - frame_start)
                
                for call_idx, call in pi3_calls:
                    tool_name = call['name']
                    tool = self.tool_registry.get(tool_name)
                    if tool:
                        # If we extracted frames for pi3, update the arguments
                        arguments = call['arguments'].copy()
                        if pi3_frame_paths:
                            # Update image_path in arguments
                            if 'image_path' in arguments:
                                logger.info(f"Updating pi3 tool arguments with {len(pi3_frame_paths)} newly extracted frames")
                                arguments['image_path'] = pi3_frame_paths
                            elif 'image_path' not in arguments:
                                arguments['image_path'] = pi3_frame_paths
                        
                        result, duration = _timed_safe_tool_call(tool, arguments)
                        result_key = tool_name if len(pi3_calls) == 1 else f"{tool_name}_{call_idx}"
                        tool_results[result_key] = result
                        _record_tool_timing(tool_name, result_key, duration, result)
                        # Add small delay between Pi3 tool calls
                        time.sleep(1)
                    else:
                        logger.error(f"{tool_name} not found")
                        result_key = tool_name if len(pi3_calls) == 1 else f"{tool_name}_{call_idx}"
                        tool_results[result_key] = {
                            "success": False,
                            "error": f"{tool_name} not found"
                        }
                        _record_tool_timing(tool_name, result_key, 0.0, tool_results[result_key])
            
            # Execute other tools in parallel as before
            for tool_name, calls in other_calls.items():
                tool = self.tool_registry.get(tool_name)
                if tool is None:
                    logger.error(f"Tool not found: {tool_name}")
                    for _, call in calls:
                        tool_results[f"{tool_name}_{_}"] = {
                            "success": False,
                            "error": f"Tool not found: {tool_name}"
                        }
                        _record_tool_timing(tool_name, f"{tool_name}_{_}", 0.0, tool_results[f"{tool_name}_{_}"])
                    continue
                
                # Submit tool execution for each call
                for call_idx, call in calls:
                    arguments = dict(call['arguments'])
                    future = executor.submit(_timed_safe_tool_call, tool, arguments)
                    future_to_tool[future] = (tool_name, call_idx)
            
            # Collect results from parallel execution
            for future in as_completed(future_to_tool):
                tool_name, call_idx = future_to_tool[future]
                try:
                    result, duration = future.result()
                    # Use unique key for multiple calls to same tool
                    result_key = tool_name if len([t for t in other_calls.get(tool_name, [])]) == 1 else f"{tool_name}_{call_idx}"
                    tool_results[result_key] = result
                    _record_tool_timing(tool_name, result_key, duration, result)
                except Exception as e:
                    logger.error(f"Tool execution failed for {tool_name}: {e}")
                    result_key = tool_name if len([t for t in other_calls.get(tool_name, [])]) == 1 else f"{tool_name}_{call_idx}"
                    tool_results[result_key] = {
                        "success": False,
                        "error": str(e)
                    }
                    _record_tool_timing(tool_name, result_key, 0.0, tool_results[result_key])
        
        return tool_results
    
    def _safe_tool_call(self, tool: Tool, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely execute a tool call with error handling
        
        Args:
            tool: Tool instance
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            logger.info(f"Executing tool: {tool.name} with args: {arguments}")
            result = self._call_tool_with_recovery(tool, arguments)
            standardized = self._standardize_tool_result(tool.name, arguments, result)
            if standardized.get("success"):
                logger.info(f"Tool {tool.name} completed successfully")
            else:
                logger.warning(
                    "Tool %s returned failure_type=%s error=%s",
                    tool.name,
                    standardized.get("failure_type"),
                    standardized.get("error"),
                )
            return standardized
        except Exception as e:
            logger.error(f"Tool {tool.name} execution failed: {e}")
            return self._standardize_tool_result(tool.name, arguments, {
                "success": False,
                "error": str(e)
            })
    
    def _sort_additional_images_by_input_order(self, image_paths: List[str], additional_images: List[str]) -> List[str]:
        """
        Sort additional images to match the order of input image_paths
        
        Args:
            image_paths: Original input image paths in order
            additional_images: Generated additional images (may be in different order)
            
        Returns:
            List of valid additional images sorted by input order
        """
        # Filter out None and invalid paths first
        valid_additional_images = [img for img in additional_images if img is not None and Path(img).exists()]
        
        if not valid_additional_images:
            return []
        
        # Create a sorted list of additional images based on input order
        sorted_additional_images = []
        
        for input_path in image_paths:
            input_stem = Path(input_path).stem
            
            # Find all additional images that correspond to this input image
            matching_images = []
            
            for additional_img in valid_additional_images:
                if additional_img in sorted_additional_images:
                    continue  # Already matched
                
                additional_stem = Path(additional_img).stem
                
                # Check if this additional image corresponds to the current input image
                if self._is_image_match(input_stem, additional_stem):
                    matching_images.append(additional_img)
            
            # Add matching images in a consistent order (by filename)
            matching_images.sort()
            sorted_additional_images.extend(matching_images)
        
        # Add any remaining unmatched additional images at the end
        for additional_img in valid_additional_images:
            if additional_img not in sorted_additional_images:
                sorted_additional_images.append(additional_img)
        
        logger.info(f"Sorted additional images: {[Path(img).name for img in sorted_additional_images]}")
        return sorted_additional_images
    
    def _is_image_match(self, input_stem: str, additional_stem: str) -> bool:
        """
        Check if an additional image corresponds to an input image
        
        Args:
            input_stem: Stem of input image filename
            additional_stem: Stem of additional image filename
            
        Returns:
            True if they match, False otherwise
        """
        # Since the suffix of saved images is always the same as the original image name,
        # we check if the additional image name ends with the input image name
        # This handles any prefix automatically without needing to maintain a list
        return additional_stem.endswith(input_stem)
    
    def _has_answer_tags(self, response: str) -> bool:
        """
        Check if response contains <answer> tags
        
        Args:
            response: Response text to check
            
        Returns:
            True if response contains <answer> tags, False otherwise
        """
        return '<answer>' in response and '</answer>' in response

    def _has_think_tags(self, response: str) -> bool:
        """
        Check whether a response includes the mandatory thinking block.
        """
        return bool(response) and "<think>" in response and "</think>" in response

    def _repair_missing_think_response(self, response: str) -> str:
        """
        Optionally wrap leading free-form reasoning in <think> for SFT rollout collection.

        Some local/student models follow the tool/answer tags but put the reasoning
        text before the first tag. For data collection this can be repaired without
        changing the selected tool or final answer.
        """
        text = str(response or "").strip()
        if not self.repair_missing_think or not text or self._has_think_tags(text):
            return response

        tag_positions = [
            pos for pos in (
                text.find("<skill_choice>"),
                text.find("<tool_call>"),
                text.find("<answer>"),
            )
            if pos >= 0
        ]
        if not tag_positions:
            return response

        first_tag = min(tag_positions)
        reasoning = text[:first_tag].strip()
        tagged_tail = text[first_tag:].lstrip()
        if not reasoning or not tagged_tail:
            return response

        return f"<think>\n{reasoning}\n</think>\n{tagged_tail}"

    def _strip_disallowed_skill_choice_blocks(self, response: str) -> str:
        """
        Remove model-emitted skill_choice blocks when prompt-time retrieval is
        managing skill selection internally.
        """
        if self.allow_model_skill_choice or not self.strip_disallowed_skill_choice:
            return response
        text = str(response or "")
        if "<skill_choice>" not in text:
            return response
        cleaned = re.sub(r"\s*<skill_choice>\s*.*?\s*</skill_choice>\s*", "\n", text, flags=re.DOTALL)
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    def _response_format_errors(self, response: str) -> List[str]:
        """
        Validate the Skill-3D rollout tag protocol for one model turn.
        """
        errors: List[str] = []
        text = response or ""
        if not self._has_think_tags(text):
            errors.append("missing_think")
        else:
            think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
            think_text = think_match.group(1).strip() if think_match else ""
            if not think_text:
                errors.append("empty_think")
            if "<answer>" in text:
                answer_pos = text.find("<answer>")
                think_start = text.find("<think>")
                think_end = text.find("</think>")
                if think_start < 0 or think_end < 0 or think_start > answer_pos or think_end > answer_pos:
                    errors.append("answer_without_prior_think_reason")
        if "<tool_choice>" in text or "</tool_choice>" in text:
            errors.append("tool_choice_tag_disallowed")
        if "<tool_call>" in text and "<answer>" in text:
            errors.append("same_turn_tool_call_and_answer")
        return errors
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """
        Extract answer content from <answer> tags
        
        Args:
            response: Response text to extract from
            
        Returns:
            Extracted answer text or None if no answer tags found
        """
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _parse_jsonish_object(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        raw = text.strip()
        if "```" in raw:
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        try:
            obj = ast.literal_eval(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        # Last fallback: extract first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        block = m.group(0)
        try:
            obj = json.loads(block)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _generate_skill_reflection(
        self,
        question: str,
        tool_calls: List[Dict[str, Any]],
        tool_results: Dict[str, Any],
        final_response: str,
        success: bool,
    ) -> Optional[Dict[str, Any]]:
        if not tool_calls:
            return None
        try:
            prompt = create_skill_reflection_prompt(
                question=question,
                tool_calls=tool_calls,
                tool_results=tool_results,
                final_answer=final_response,
                success=success,
            )
            reflection_text = self.model.text_only_inference(
                prompt=prompt,
                temperature=0.2,
                max_tokens=400,
            )
            reflection = self._parse_jsonish_object(reflection_text)
            if not reflection:
                return None
            # Minimal schema guard
            if not all(k in reflection for k in ("name", "when", "strategy", "tool_usage")):
                return None
            if not isinstance(reflection.get("strategy"), list):
                return None
            if "confidence" in reflection:
                try:
                    reflection["confidence"] = max(0.0, min(1.0, float(reflection["confidence"])))
                except Exception:
                    reflection["confidence"] = 0.5
            return reflection
        except Exception as exc:
            logger.warning("Skill reflection generation failed: %s", exc)
            return None
    
    def _create_continuation_prompt(
        self, 
        question: str, 
        last_response: str, 
        all_tool_results: Dict[str, Any],
        original_images: List[str],
        additional_images: List[str],
        current_iteration: int,
        max_iterations: int,
        skill_bundle: Optional[Dict[str, Any]] = None,
        selector_note: Optional[str] = None,
        oracle_hint: str = "",
    ) -> str:
        """
        Create continuation prompt for multi-step workflow
        
        Args:
            question: Original user question
            last_response: Last model response
            all_tool_results: All tool results so far
            original_images: Original image paths
            additional_images: All additional images generated
            current_iteration: Current iteration number
            max_iterations: Maximum iterations allowed
            
        Returns:
            Continuation prompt string
        """
        tool_summary = []
        angle_info = []
        
        for tool_name, result in all_tool_results.items():
            if result.get('success'):
                tool_summary.append(f"- {tool_name}: Successfully executed")
                
                # Extract angle information if available
                if 'azimuth_angle' in result and 'elevation_angle' in result:
                    azim = result.get('azimuth_angle')
                    elev = result.get('elevation_angle')
                    angle_info.append(f"  └─ Viewing angle: azimuth={azim}°, elevation={elev}°")
                
                if result.get('description'):
                    tool_summary.append(f"  Description: {result.get('description')}")
            else:
                error_text = result.get("error", "Unknown error")
                if result.get("transient_api_error"):
                    wait_s = result.get("transient_total_wait_s")
                    wait_note = f" after {wait_s:.1f}s recovery" if isinstance(wait_s, (int, float)) else ""
                    tool_summary.append(
                        f"- {tool_name}: Temporarily unavailable{wait_note} - {error_text}"
                    )
                    tool_summary.append(
                        "  Note: treat this as an external service/API outage, not as visual evidence from the scene."
                    )
                else:
                    tool_summary.append(f"- {tool_name}: Failed - {error_text}")
        
        tool_summary_text = "\n".join(tool_summary) if tool_summary else "None yet"
        angle_info_text = "\n".join(angle_info) if angle_info else ""
        
        original_images_info = "\n".join([f"- {path}" for path in original_images]) if original_images else "None"
        additional_images_info = "\n".join([f"- {path}" for path in additional_images]) if additional_images else "None yet"
        
        remaining = max_iterations - current_iteration
        used_pi3 = any(str(tool_name).startswith("pi3_tool") for tool_name in all_tool_results.keys())
        pi3_follow_up_notes = ""
        if used_pi3:
            pi3_follow_up_notes = """

   Pi3 follow-up notes (because pi3_tool has already been used in this trajectory):
   - The original input images already correspond to (azimuth=0°, elevation=0°). Ask pi3 for a genuinely new viewpoint, not (0°, 0°).
   - Prefer coarse new angles first, such as azimuth ±45/±90 or elevation 30-60.
   - If multiple input images exist, use rotation_reference_camera=1,2,3,... to rotate around different input cameras.
   - Use camera_view=true only when first-person evidence is needed; otherwise keep camera_view=false.
"""
        skill_choice_block = self._build_skill_choice_reporting_block(
            skill_bundle,
            include_catalog_snapshot=True,
        )
        skill_choice_notes = ""
        if skill_choice_block:
            skill_choice_notes = "\n\n" + skill_choice_block
        selector_guidance = ""
        if selector_note:
            selector_guidance = f"\nSelector guidance:\n- {selector_note}\n"

        prompt_mode = str(os.environ.get("SPAGENT_SKILL_PROMPT_MODE", "")).strip().lower()
        if self.prompt_style == "skills" and prompt_mode in {"compact", "sft", "rollout", "sft_rollout"}:
            if self.force_skill_tool_call and not all_tool_results:
                tool_rule = (
                    "- Before the first final answer, output one valid <skill_choice> from the provided candidates "
                    "and one executable <tool_call>; do not answer directly."
                )
            else:
                tool_rule = "- If more evidence is needed, output <skill_choice> when useful, then <tool_call>."

            prompt = f"""# Required Format
- Start every assistant turn with exactly one <think>...</think> block.
{tool_rule}
- If tool evidence is sufficient, first explain inside <think> which tool output supports the decision, then output <answer>...</answer>.
- Do not output <tool_choice>. Do not put text outside the required tags.

# Current State
Original Question: {question}

Previous Assistant Turn:
{last_response}

Tool Results So Far:
{tool_summary_text}
{angle_info_text}

Original Images:
{original_images_info}

Generated Images:
{additional_images_info}
{selector_guidance}
{oracle_hint}

# Next Step
You have {remaining} more iteration(s). Use schema-valid tool arguments. Avoid duplicate tool calls unless parameters or images change the evidence. Stop with <answer> only when the answer is supported by observed evidence.
"""
            return prompt
        
        prompt = f"""=== Multi-Step Analysis: Iteration {current_iteration}/{max_iterations} ===

Original Question: {question}

Your Previous Response: 
{last_response}

Tool Execution Summary:
{tool_summary_text}
{angle_info_text}

Original Images:
{original_images_info}

Generated Images Available for Analysis:
{additional_images_info}
{selector_guidance}

	=== Next Steps ===

	You have {remaining} more iteration(s) available. You can:

	1. **Continue investigating**:
	   - Choose the next tool call only if it can add new evidence.
	   - Avoid repeating the same tool call unless you change parameters or it adds new evidence.
	   - If the blocker is boundary precision, exact points, or tiny/blurry detail, switch to a complementary helper such as segment_image_tool, moondream_tool, or swinir_tool instead of repeating the same coarse tool.
	   - Do not reuse detect_objects_tool as the default next step when another helper better matches the remaining uncertainty.
	   - Use only schema-listed parameters; avoid invented keys such as box1, box2, image_path_a, image_path_b, point_prompts, distance_type, or use_detected.
	   - Use any tool order that best matches your reasoning.
{pi3_follow_up_notes}

2. **Provide final answer** - If you have sufficient information from current viewpoints:
   - Output your comprehensive analysis in <think></think> tags
   - Reference the specific viewpoints that helped you understand the structure

	Instructions:
	- Every response must start with a concise <think></think> block.
	- <skill_choice></skill_choice> is optional. Use it only when a catalog skill is useful for this step.
	- If you choose a non-none skill, you must execute its required evidence flow through <tool_call></tool_call> before final answering.
	- If you need more evidence: output one or more <tool_call></tool_call> blocks.
	- If you are ready: output your final option/number in <answer></answer> only after <think></think> (no extra text).
	- Do not output <tool_choice>; executable tool decisions must use <tool_call>.
{skill_choice_notes}
{oracle_hint}

	Please continue:"""
        
        return prompt

    def _build_oracle_rollout_hint(self, oracle_ground_truth: str) -> str:
        """Build teacher-forcing guidance for SFT data collection.

        This is guarded by SPAGENT_ORACLE_ROLLOUT and should only be used on
        training splits. It provides the known answer so the student model can
        produce a clean trajectory, while still requiring evidence-based tool
        use when tools are useful.
        """
        answer = str(oracle_ground_truth).strip()
        return f"""# Oracle SFT Rollout Mode
This is a supervised training-data collection rollout. The ground-truth answer is known:
<oracle_answer>{answer}</oracle_answer>

Use the oracle answer only to keep the trajectory correct and concise. Still reason from the images, retrieved skills, and tool observations. If a tool is useful, choose the appropriate skill and call the tool before answering. The final <answer> must match the oracle answer exactly in meaning."""
    def _extract_frames_for_pi3(self, video_path: str, num_frames: int = 10) -> List[str]:
        """
        Extract frames from video for pi3 tool by uniformly sampling
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to uniformly sample from video
            
        Returns:
            List of paths to extracted frame images
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.error("cv2 is required for video frame extraction. Please install opencv-python.")
            return []
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = total_frames / original_fps
        
        # Use the specified number of frames directly
        frame_interval = total_frames / num_frames
        
        frame_paths = []
        temp_dir = Path("temp_frames_pi3")
        temp_dir.mkdir(exist_ok=True)
        
        # Extract video filename (without extension)
        video_filename = Path(video_path).stem
        
        # Extract frames evenly
        for i in range(num_frames):
            frame_idx = int(i * frame_interval)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = temp_dir / f"{video_filename}_pi3_frame_{i}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
        
        cap.release()
        logger.info(f"Extracted {len(frame_paths)} frames from video for pi3 tool (duration: {total_duration:.2f}s, original fps: {original_fps:.2f}, uniformly sampled {num_frames} frames)")
        return frame_paths
    
    def _cleanup_pi3_frames(self):
        """
        Clean up temporary frames extracted for pi3 tool
        """
        import os
        import shutil
        
        temp_dir = Path("temp_frames_pi3")
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary pi3 frames")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary pi3 frames: {e}")
    
    def _get_naive_baseline_answer(
        self, 
        image_paths: List[str], 
        question: str,
        **model_kwargs
    ) -> str:
        """
        Get answer from naive baseline agent (no tools)
        
        Args:
            image_paths: List of image paths
            question: User's question
            **model_kwargs: Additional arguments for model inference
            
        Returns:
            Baseline answer string
        """
        # Create simple prompt without tool information
        naive_prompt = f"""You are a helpful AI assistant specialized in spatial intelligence and visual reasoning.

Please analyze the image(s) and answer the following question directly, using only what you can see in the image(s).

Question: {question}

Please provide your answer directly in <answer></answer> tags."""
        
        try:
            if len(image_paths) == 1:
                response = self.model.single_image_inference(
                    image_paths[0],
                    naive_prompt,
                    **model_kwargs
                )
            else:
                response = self.model.multiple_images_inference(
                    image_paths,
                    naive_prompt,
                    **model_kwargs
                )
            return response
        except Exception as e:
            logger.error(f"Failed to get naive baseline answer: {e}")
            return f"Error: {str(e)}"
    
    def _synthesize_with_baseline(
        self, 
        question: str,
        tool_based_answer: str,
        baseline_answer: str,
        image_paths: List[str],
        additional_images: List[str],
        **model_kwargs
    ) -> str:
        """
        Synthesize final answer by comparing tool-based and baseline answers
        
        Args:
            question: Original question
            tool_based_answer: Answer from tool-enhanced reasoning
            baseline_answer: Answer from naive baseline (no tools)
            image_paths: Original image paths
            additional_images: Additional images generated by tools
            **model_kwargs: Additional arguments for model inference
            
        Returns:
            Synthesized final answer
        """
        synthesis_prompt = f"""You are a helpful AI assistant tasked with providing the most accurate answer by synthesizing information from two different approaches.

Original Question: {question}

Approach 1 - Tool-Enhanced Analysis:
{tool_based_answer}

Approach 2 - Direct Visual Analysis (Baseline):
{baseline_answer}

=== Your Task ===

Compare the two approaches and provide the MOST ACCURATE answer to the original question.

Instructions:
1. Carefully analyze both answers
2. Consider which approach provides more reliable information:
   - Tool-enhanced analysis may have additional insights from specialized tools (depth, 3D, segmentation, etc.)
   - Direct visual analysis relies purely on what's visible in the image
3. Identify where they agree or disagree
4. When they disagree, determine which is more likely to be correct and why
5. Synthesize the information to provide the best possible answer

Important:
- If both approaches agree, confirm the answer with confidence
- If they disagree, explain your reasoning for choosing one over the other
- Use information from the tool-enhanced approach when it provides clear additional insights
- Use information from the baseline when the tool-enhanced approach seems uncertain or incorrect
- Your goal is to provide the MOST ACCURATE answer, not just combine both answers

Please provide your final synthesized answer in <answer></answer> tags."""
        
        try:
            # Use additional images if available, otherwise use original images
            final_images = self._select_analysis_images(
                image_paths=image_paths,
                additional_images=additional_images,
                tool_results={},
                skill_bundle=None,
            )
            
            if len(final_images) == 1:
                response = self.model.single_image_inference(
                    final_images[0],
                    synthesis_prompt,
                    **model_kwargs
                )
            else:
                response = self.model.multiple_images_inference(
                    final_images,
                    synthesis_prompt,
                    **model_kwargs
                )
            
            logger.info(f"Synthesized answer: {response}")
            return response
        except Exception as e:
            logger.error(f"Failed to synthesize answer: {e}")
            # Fallback to tool-based answer if synthesis fails
            return tool_based_answer
