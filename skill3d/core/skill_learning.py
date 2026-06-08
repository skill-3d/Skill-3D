"""
Adaptive skill learning for SPAgent.

Memory now uses a two-level nested JSON structure:
1) Question Class Memory: classify and evolve question types.
2) Skill Memory: learn/update skills per question class.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .skill_normalization import build_default_skills, skills_to_json
from .skill_retrieval import (
    SkillRetrievalIndex,
    filter_similar_skill_candidates,
    rerank_skill_candidates,
)

logger = logging.getLogger(__name__)


HEAVY_MEMORY_KEYS = {
    "annotated_image",
    "image",
    "images",
    "image_array",
    "mask",
    "masks",
    "mask_image",
    "segmentation",
    "segmentation_mask",
    "depth",
    "depth_map",
    "raw_depth",
    "point_cloud",
    "points3d",
    "vertices",
    "colors",
    "heatmap",
    "overlay",
}

TIME_OUTPUT_KEYS = {
    "created_at",
    "updated_at",
    "last_seen",
    "first_seen",
    "last_updated",
    "timestamp",
    "start_time",
    "end_time",
}

DEFAULT_MONOLITH_LOAD_LIMIT_MB = 256
MAX_WORKING_SESSIONS = 32
MAX_WORKING_NOTES_PER_SESSION = 24
MAX_MEMORY_LIST_ITEMS = 128
MAX_EXAMPLE_QUESTIONS_PER_CLASS = 8
DEFAULT_HIERARCHICAL_EPISODE_LIMIT = 5000
DEFAULT_HIERARCHICAL_FACT_LIMIT = 8000


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _strip_time_fields_for_output(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if str(key).strip().lower() in TIME_OUTPUT_KEYS:
                continue
            cleaned[key] = _strip_time_fields_for_output(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_time_fields_for_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_time_fields_for_output(item) for item in value)
    return value


class AdaptiveSkillManager:
    """
    Keep seed skills and incrementally learn skills from trajectories.

    - Question Class Memory: classify known/new question types and keep class-level priors.
    - Skill Memory: maintain trajectories + learned skills as candidate references.
    """

    def __init__(
        self,
        storage_path: str = "statics/learned_skills.json",
        question_memory_export_path: Optional[str] = None,
        max_learned_skills: int = 24,
        stale_days: int = 14,
        update_enabled: bool = True,
    ):
        self.storage_path = Path(storage_path)
        if question_memory_export_path is None:
            self.question_memory_export_path = self.storage_path.with_name("questions.json")
        else:
            qpath = Path(question_memory_export_path)
            if qpath.is_absolute():
                self.question_memory_export_path = qpath
            elif self.storage_path.is_absolute() and qpath.parts and qpath.parts[0] == self.storage_path.parent.name:
                self.question_memory_export_path = self.storage_path.parent / Path(*qpath.parts[1:])
            elif self.storage_path.is_absolute():
                self.question_memory_export_path = self.storage_path.parent / qpath
            else:
                self.question_memory_export_path = qpath
        self.max_learned_skills = max(1, int(max_learned_skills))
        self.stale_days = max(1, int(stale_days))
        self.update_enabled = bool(update_enabled)
        self._needs_storage_refresh = False
        self._monolith_load_limit_bytes = int(
            os.environ.get("SPAGENT_MONOLITH_LOAD_LIMIT_MB", str(DEFAULT_MONOLITH_LOAD_LIMIT_MB))
        ) * 1024 * 1024
        self._hierarchical_root = Path(
            os.environ.get("SKILL3D_HIERARCHICAL_MEMORY_DIR")
            or os.environ.get("SPAGENT_HIERARCHICAL_MEMORY_DIR")
            or str(self.storage_path.with_suffix(""))
        )
        if not self._hierarchical_root.is_absolute() and self.storage_path.is_absolute():
            self._hierarchical_root = self.storage_path.parent / self._hierarchical_root
        self._hierarchical_manifest_version = 1
        self._hierarchical_episode_limit = max(
            1,
            int(os.environ.get("SPAGENT_HIERARCHICAL_EPISODE_LIMIT", str(DEFAULT_HIERARCHICAL_EPISODE_LIMIT))),
        )
        self._hierarchical_fact_limit = max(
            1,
            int(os.environ.get("SPAGENT_HIERARCHICAL_FACT_LIMIT", str(DEFAULT_HIERARCHICAL_FACT_LIMIT))),
        )

        self._seed_skills = skills_to_json(build_default_skills())
        self._seed_pattern_order = {
            "spatial_localization": 0,
            "depth_distance": 1,
            "multiview_occlusion": 2,
            "detection_counting": 3,
            "segmentation_boundary": 4,
            "fine_grained_pointing": 5,
            "custom_category": 6,
            "comprehensive_scene_understanding": 7,
        }
        self._seed_class_keywords = {
            "spatial_localization": ["where", "left", "right", "between", "relative", "position"],
            "depth_distance": [
                "closer",
                "farther",
                "distance",
                "front",
                "behind",
                "near",
                "nearest",
                "farthest",
                "closest",
                "depth",
                "closer to",
                "farther from",
                "distance between",
            ],
            "multiview_occlusion": ["occlusion", "camera", "viewpoint", "visible", "see"],
            "detection_counting": ["how many", "count", "identify", "all objects"],
            "segmentation_boundary": ["segment", "boundary", "outline", "mask"],
            "fine_grained_pointing": ["point", "locate exactly", "exactly where"],
            "custom_category": ["detect all", "category", "class", "specific object"],
            "comprehensive_scene_understanding": ["analyze", "understand", "reason", "scene"],
        }
        self._known_tools = {
            "depth_estimation_tool",
            "segment_image_tool",
            "detect_objects_tool",
            "supervision_tool",
            "yoloe_detection_tool",
            "moondream_tool",
            "swinir_tool",
            "orient_anything_tool",
            "pi3_tool",
        }
        self._progressive_root = self.storage_path.parent / "progressive_skills"
        self._static_root = self._progressive_root / "static"
        self._dynamic_root = self._progressive_root / "dynamic"
        self._static_problem_dir = self._static_root / "problem-types"
        self._static_skill_dir = self._static_root / "skills"
        self._dynamic_problem_dir = self._dynamic_root / "problem-types"
        self._dynamic_skill_dir = self._dynamic_root / "skills"
        self._dynamic_state_dir = self._dynamic_root / "state"
        self._dynamic_problem_state_path = self._dynamic_state_dir / "problem-state.json"
        self._dynamic_skill_state_path = self._dynamic_state_dir / "skill-state.json"
        self._retrieval_index: Optional[SkillRetrievalIndex] = None
        self._retrieval_memory_version: Optional[int] = None
        self._stopwords = {
            "the", "a", "an", "to", "of", "in", "on", "at", "for", "and", "or", "is", "are", "be", "from",
            "this", "that", "with", "what", "which", "how", "can", "could", "would", "should", "please",
        }

        self._memory = self._load_memory()
        sanitized = self._sanitize_loaded_skill_memory()
        seed_synced = self._sync_seed_question_classes()
        renamed = self._rename_runtime_question_classes()
        if self._needs_storage_refresh or sanitized or seed_synced or renamed:
            self._save_memory()
        else:
            self._materialize_progressive_disclosure()

    # ---------------------------------------------------------------------
    # Five-layer memory compatibility helpers
    # ---------------------------------------------------------------------
    def _rule_memory(self) -> Dict[str, Any]:
        layer = self._memory.setdefault("rule_memory", {})
        layer.setdefault("version", 1)
        layer.setdefault("rules", {})
        return layer

    def _working_memory(self) -> Dict[str, Any]:
        layer = self._memory.setdefault("working_memory", {})
        layer.setdefault("version", 1)
        layer.setdefault("sessions", {})
        return layer

    def _episode_memory(self) -> Dict[str, Any]:
        layer = self._memory.setdefault("episode_memory", {})
        layer.setdefault("version", 1)
        layer.setdefault("episodes", [])
        return layer

    def _evidence_memory(self) -> Dict[str, Any]:
        layer = self._memory.setdefault("evidence_memory", {})
        layer.setdefault("version", 1)
        layer.setdefault("facts", [])
        return layer

    def write_working_note(
        self,
        session_id: str,
        note_type: str,
        payload: Dict[str, Any],
    ) -> None:
        session_id = str(session_id or "").strip()
        if not session_id:
            return
        layer = self._working_memory()
        session_state = layer["sessions"].setdefault(
            session_id,
            {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "notes": [],
            },
        )
        session_state["notes"].append(
            {
                "type": str(note_type or "generic").strip() or "generic",
                "payload": self._compact_for_memory(payload if isinstance(payload, dict) else {"value": payload}),
                "timestamp": datetime.now().isoformat(),
            }
        )
        if len(session_state["notes"]) > MAX_WORKING_NOTES_PER_SESSION:
            del session_state["notes"][:-MAX_WORKING_NOTES_PER_SESSION]
        session_state["updated_at"] = datetime.now().isoformat()
        sessions = layer.get("sessions", {})
        if len(sessions) > MAX_WORKING_SESSIONS:
            ordered = sorted(
                sessions.items(),
                key=lambda item: str((item[1] or {}).get("updated_at", "")) if isinstance(item[1], dict) else "",
            )
            for stale_id, _ in ordered[: max(0, len(sessions) - MAX_WORKING_SESSIONS)]:
                sessions.pop(stale_id, None)

    def close_episode(
        self,
        session_id: str,
        question: str,
        tool_calls: List[Dict[str, Any]],
        tool_results: Dict[str, Any],
        final_answer: Optional[str],
        success: bool,
        reward_score: Optional[float] = None,
        is_correct: Optional[bool] = None,
        question_class: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        session_id = str(session_id or "").strip()
        if not session_id:
            return None
        working = self._working_memory().get("sessions", {}).pop(session_id, None)
        normalized_question = self.normalize_question(question)
        full_sequence = self._extract_full_sequence(tool_calls)
        effective_sequence = self._extract_effective_sequence(tool_calls, tool_results) or list(dict.fromkeys(full_sequence))
        episode_id = f"episode_{hashlib.sha1((session_id + normalized_question).encode('utf-8')).hexdigest()[:12]}"
        episode = {
            "episode_id": episode_id,
            "session_id": session_id,
            "question": normalized_question,
            "scene_signature": self._build_scene_signature(tool_calls, tool_results),
            "task": question_class or self._classify_seed_pattern(normalized_question),
            "dataset_task": self._normalize_task_type(task_type),
            "success": bool(success),
            "reward_score": _safe_float(reward_score, 0.0),
            "is_correct": None if is_correct is None else bool(is_correct),
            "trajectory_summary": {
                "tool_sequence": full_sequence,
                "effective_tool_sequence": effective_sequence,
                "final_answer": final_answer,
            },
            "error_pattern": self._episode_error_pattern(tool_results),
            "objects": self._extract_object_labels(tool_results),
            "relations": self._extract_relation_hints(normalized_question),
            "reward_trace": {
                "reward_score": _safe_float(reward_score, 0.0),
                "success": bool(success),
                "is_correct": None if is_correct is None else bool(is_correct),
            },
            "embedding_text": self._build_episode_retrieval_text(
                normalized_question,
                effective_sequence,
                tool_results,
                success=bool(success),
                final_answer=final_answer,
            ),
            "provenance": {
                "source": "runtime_episode",
                "created_at": datetime.now().isoformat(),
            },
            "working_memory_snapshot": self._compact_for_memory(working or {}),
        }
        episodes = self._episode_memory()["episodes"]
        episodes.append(episode)
        if len(episodes) > 2000:
            del episodes[:-2000]
        return episode

    def promote_skill_candidate(
        self,
        episode: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(episode, dict):
            return None
        sequence = episode.get("trajectory_summary", {}).get("effective_tool_sequence") or []
        if not sequence:
            return None
        return {
            "candidate_id": f"candidate_{hashlib.sha1(json.dumps(sequence, ensure_ascii=False).encode('utf-8')).hexdigest()[:10]}",
            "source_episode_id": episode.get("episode_id"),
            "task_pattern": episode.get("task"),
            "question": episode.get("question"),
            "tool_candidates": sequence,
            "confidence": episode.get("reward_score", 0.0),
        }

    def _promotion_gate(
        self,
        episode: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Dict[str, Any]]:
        if not isinstance(episode, dict):
            return False, {"reason": "missing_episode"}
        reward_score = _safe_float(episode.get("reward_score"), 0.0)
        success = bool(episode.get("success"))
        sequence = episode.get("trajectory_summary", {}).get("effective_tool_sequence") or []
        objects = episode.get("objects") or []
        gate = {
            "success": success,
            "reward_score": reward_score,
            "tool_count": len(sequence),
            "object_count": len(objects),
        }
        if not success:
            gate["reason"] = "episode_not_successful"
            return False, gate
        if reward_score < 0.3:
            gate["reason"] = "reward_below_threshold"
            return False, gate
        if not sequence:
            gate["reason"] = "no_effective_tool_sequence"
            return False, gate
        gate["reason"] = "passed"
        return True, gate

    def patch_skill(
        self,
        skill_id: str,
        patch: Dict[str, Any],
    ) -> bool:
        for skill in self._skill_memory().get("skills", []):
            if isinstance(skill, dict) and str(skill.get("id")) == str(skill_id):
                for key, value in (patch or {}).items():
                    skill[key] = value
                self._save_memory()
                return True
        return False

    def deprecate_skill(self, skill_id: str, reason: str = "") -> bool:
        for skill in self._skill_memory().get("skills", []):
            if isinstance(skill, dict) and str(skill.get("id")) == str(skill_id):
                meta = skill.setdefault("meta", {})
                meta["deprecated"] = True
                meta["deprecation_reason"] = str(reason or "").strip()
                self._save_memory()
                return True
        return False

    def recall_skills(
        self,
        query: str,
        class_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        index = self._get_retrieval_index()
        dense = index.retrieve(query=query, top_k=max(top_k * 3, top_k), class_filter=[class_id] if class_id else None)
        reranked = rerank_skill_candidates(query=query, candidates=dense, class_id=class_id, top_k=top_k)
        return _strip_time_fields_for_output([self._retrieval_card_from_unit(item["unit"], item) for item in reranked])

    def recall_episodes(
        self,
        query: str,
        task: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        query_text = self.normalize_question(query).lower()
        scored = []
        for episode in self._episode_memory().get("episodes", []):
            if not isinstance(episode, dict):
                continue
            if task and str(episode.get("task")) != str(task):
                continue
            text = str(episode.get("embedding_text", "")).lower()
            overlap = sum(1 for token in query_text.split() if token and token in text)
            score = overlap + 0.5 * _safe_float(episode.get("reward_score"), 0.0)
            scored.append((score, episode))
        scored.sort(key=lambda item: item[0], reverse=True)
        return _strip_time_fields_for_output([episode for _, episode in scored[: max(1, int(top_k))]])

    def recall_facts(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        query_text = self.normalize_question(query).lower()
        scored = []
        for fact in self._evidence_memory().get("facts", []):
            if not isinstance(fact, dict):
                continue
            text = " ".join([
                str(fact.get("entity", "")),
                str(fact.get("relation", "")),
                str(fact.get("summary", "")),
            ]).lower()
            overlap = sum(1 for token in query_text.split() if token and token in text)
            score = overlap + 0.5 * _safe_float(fact.get("confidence"), 0.0)
            scored.append((score, fact))
        scored.sort(key=lambda item: item[0], reverse=True)
        return _strip_time_fields_for_output([fact for _, fact in scored[: max(1, int(top_k))]])

    def flush_memory(self) -> None:
        self._save_memory()

    def search_provenance(self, keyword: str, top_k: int = 10) -> List[Dict[str, Any]]:
        target = str(keyword or "").lower().strip()
        results: List[Dict[str, Any]] = []
        if not target:
            return results
        for episode in self._episode_memory().get("episodes", []):
            if not isinstance(episode, dict):
                continue
            haystack = json.dumps(episode.get("provenance", {}), ensure_ascii=False).lower()
            if target in haystack:
                results.append({"source": "episode", "record": _strip_time_fields_for_output(episode)})
        for fact in self._evidence_memory().get("facts", []):
            if not isinstance(fact, dict):
                continue
            haystack = json.dumps(fact.get("provenance", {}), ensure_ascii=False).lower()
            if target in haystack:
                results.append({"source": "fact", "record": _strip_time_fields_for_output(fact)})
        return results[: max(1, int(top_k))]

    def export_skills(self, question: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return skills shown to the model: seed + learned skills.

        Learned skills are returned in stored order so selection stays with the model.
        """
        if not self.update_enabled:
            return self._seed_skills

        changed = False
        if question:
            _, changed = self._classify_question(
                question,
                create_if_missing=False,
                touch=False,
            )

        skill_mem = self._skill_memory()
        learned = [s for s in skill_mem.get("skills", []) if isinstance(s, dict)]

        if changed:
            self._save_memory()

        return _strip_time_fields_for_output(self._seed_skills + learned)

    def build_prompt_bundle(
        self,
        question: str,
        memory_mode: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
        retrieval_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build a progressive-disclosure prompt bundle.

        The bundle keeps static knowledge and dynamic memory separate:
        - problem_card: compact current problem summary
        - skill_catalog: the full archived skill catalog, without prompt-time truncation
        """
        memory_mode = str(memory_mode or os.environ.get("SPAGENT_MEMORY_MODE", "symbolic")).strip().lower()
        default_retrieval_mode = memory_mode if memory_mode in {"dense", "hybrid"} else "symbolic"
        retrieval_mode = str(
            retrieval_mode
            or os.environ.get("SPAGENT_RETRIEVAL_MODE")
            or default_retrieval_mode
        ).strip().lower()
        retrieval_top_k = int(retrieval_top_k or os.environ.get("SPAGENT_RETRIEVAL_TOP_K", "6"))
        if memory_mode not in {"none", "symbolic", "dense", "hybrid"}:
            memory_mode = "symbolic"
        if retrieval_mode not in {"symbolic", "dense", "hybrid"}:
            retrieval_mode = memory_mode if memory_mode in {"dense", "hybrid"} else "symbolic"

        normalized_question = self.normalize_question(question)
        empty_payload = {
            "normalized_question": normalized_question,
            "matched_problem_id": None,
            "runtime_class_known": False,
            "runtime_problem_id": None,
            "seed_problem_id": "none",
            "recommended_seed_skill_id": "none",
            "problem_card": {},
            "focus_skills": [],
            "retrieved_skills": [],
            "retrieved_episodes": [],
            "retrieved_evidence": [],
            "skill_catalog": [],
            "retrieval_diagnostics": {
                "memory_mode": memory_mode,
                "retrieval_mode": retrieval_mode,
                "disabled": True,
            },
        }
        if memory_mode == "none":
            return _strip_time_fields_for_output(empty_payload)

        matched_problem_id, changed = self._classify_question(
            normalized_question,
            create_if_missing=False,
            touch=False,
        )
        if changed:
            self._save_memory()

        runtime_problem_id = matched_problem_id if matched_problem_id and not self._is_seed_problem_id(matched_problem_id) else None
        seed_problem_id = (
            self._resolve_seed_parent(matched_problem_id, question=normalized_question)
            if matched_problem_id
            else self._classify_seed_pattern(normalized_question)
        )
        include_dynamic = bool(matched_problem_id)

        catalog = self._build_skill_catalog()
        focus_skills = self._build_focus_skills(seed_problem_id, matched_problem_id=matched_problem_id, normalized_question=normalized_question)
        retrieved_skills, retrieval_diagnostics = self._retrieve_prompt_skills(
            query=normalized_question,
            matched_problem_id=matched_problem_id,
            seed_problem_id=seed_problem_id,
            focus_skills=focus_skills,
            catalog=catalog,
            memory_mode=memory_mode,
            retrieval_mode=retrieval_mode,
            top_k=retrieval_top_k,
        )
        retrieved_episodes: List[Dict[str, Any]] = []
        retrieved_evidence: List[Dict[str, Any]] = []
        if memory_mode in {"dense", "hybrid"}:
            retrieved_episodes = self.recall_episodes(
                query=normalized_question,
                task=matched_problem_id or seed_problem_id,
                top_k=min(2, max(1, retrieval_top_k)),
            )
            retrieved_evidence = self.recall_facts(
                query=normalized_question,
                top_k=min(3, max(1, retrieval_top_k)),
            )
        retrieval_diagnostics["episode_hits"] = len(retrieved_episodes)
        retrieval_diagnostics["evidence_hits"] = len(retrieved_evidence)

        prompt_payload = {
            "normalized_question": normalized_question,
            "matched_problem_id": matched_problem_id,
            "runtime_class_known": bool(runtime_problem_id),
            "runtime_problem_id": runtime_problem_id,
            "seed_problem_id": seed_problem_id,
            "recommended_seed_skill_id": f"seed::{seed_problem_id}",
            "problem_card": self._build_problem_card(
                seed_problem_id,
                normalized_question,
                matched_problem_id=matched_problem_id,
                runtime_problem_id=runtime_problem_id,
                include_dynamic=include_dynamic,
            ),
            "focus_skills": focus_skills,
            "retrieved_skills": retrieved_skills,
            "retrieved_episodes": retrieved_episodes,
            "retrieved_evidence": retrieved_evidence,
            "skill_catalog": catalog,
            "retrieval_diagnostics": retrieval_diagnostics,
        }
        return _strip_time_fields_for_output(prompt_payload)

    def _get_retrieval_index(self) -> SkillRetrievalIndex:
        memory_version = int(self._memory.get("global_version", 0))
        if self._retrieval_index is None or self._retrieval_memory_version != memory_version:
            self._retrieval_index = SkillRetrievalIndex.from_memory(self.storage_path)
            self._retrieval_memory_version = memory_version
        return self._retrieval_index

    def _retrieval_card_from_unit(self, unit, score_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "id": unit.skill_id,
            "name": unit.skill_name,
            "source": unit.source_type,
            "problem_id": unit.task_pattern,
            "seed_problem_id": unit.metadata.get("seed_parent") or unit.question_class,
            "when": unit.trigger,
            "tools": unit.tool_candidates,
            "trigger": unit.trigger,
            "blocker_type": unit.metadata.get("blocker_type") or "",
            "preconditions": unit.metadata.get("preconditions") or [],
            "skip_conditions": unit.metadata.get("skip_conditions") or [],
            "tool_candidates": unit.tool_candidates,
            "fallbacks": unit.fallbacks,
            "stop_conditions": unit.metadata.get("stop_conditions") or [],
            "view_policy": unit.view_policy,
            "maturity": unit.maturity,
            "observed_support": unit.metadata.get("observed_support") or {},
            "failure_memory": unit.failure_memory,
            "retrieval_text": unit.retrieval_text,
        }
        if score_payload:
            payload["retrieval_score"] = score_payload.get("score")
            payload["retrieval_score_breakdown"] = score_payload.get("score_breakdown")
            if score_payload.get("skill_filter"):
                payload["skill_filter"] = score_payload.get("skill_filter")
        return payload

    def _retrieve_prompt_skills(
        self,
        query: str,
        matched_problem_id: Optional[str],
        seed_problem_id: str,
        focus_skills: List[Dict[str, Any]],
        catalog: List[Dict[str, Any]],
        memory_mode: str,
        retrieval_mode: str,
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "memory_mode": memory_mode,
            "retrieval_mode": retrieval_mode,
            "symbolic_routed_class": matched_problem_id or seed_problem_id,
            "candidate_class_ids": [],
            "symbolic_skill_candidates": [card.get("id") for card in focus_skills if isinstance(card, dict)],
            "dense_retrieved_candidates": [],
            "reranked_final_list": [],
            "skill_filter": {},
        }
        if memory_mode == "symbolic" or retrieval_mode == "symbolic":
            return [], diagnostics

        class_id = matched_problem_id or seed_problem_id
        candidate_classes: List[str] = []
        if memory_mode == "hybrid" or retrieval_mode == "hybrid":
            candidate_classes = [class_id, seed_problem_id]
            candidate_classes.extend(
                str(card.get("id", "")).replace("seed::", "")
                for card in focus_skills
                if isinstance(card, dict) and str(card.get("id", "")).startswith("seed::")
            )
            candidate_classes = list(dict.fromkeys([c for c in candidate_classes if c]))
        diagnostics["candidate_class_ids"] = candidate_classes

        index = self._get_retrieval_index()
        candidate_pool_k = max(
            top_k * int(os.environ.get("SPAGENT_SKILL_FILTER_CANDIDATE_MULTIPLIER", "6")),
            top_k * 3,
            top_k,
        )
        dense_candidates = index.retrieve(
            query=query,
            top_k=candidate_pool_k,
            class_filter=candidate_classes if candidate_classes else None,
        )
        if len(dense_candidates) < top_k:
            existing_ids = {item["unit"].skill_id for item in dense_candidates}
            global_candidates = index.retrieve(query=query, top_k=max(candidate_pool_k, top_k * 4, top_k), class_filter=None)
            for item in global_candidates:
                if item["unit"].skill_id in existing_ids:
                    continue
                dense_candidates.append(item)
                existing_ids.add(item["unit"].skill_id)
                if len(dense_candidates) >= candidate_pool_k:
                    break
        diagnostics["dense_retrieved_candidates"] = [
            {
                "skill_id": item["unit"].skill_id,
                "source_type": item["unit"].source_type,
                "question_class": item["unit"].question_class,
                "semantic_score": item.get("semantic_score"),
                "rank": item.get("rank"),
            }
            for item in dense_candidates
        ]
        reranked_pool = rerank_skill_candidates(
            query=query,
            candidates=dense_candidates,
            class_id=class_id if candidate_classes else None,
            top_k=candidate_pool_k,
        )
        reranked, filter_diagnostics = filter_similar_skill_candidates(
            reranked_pool,
            top_k=top_k,
        )
        diagnostics["skill_filter"] = filter_diagnostics
        diagnostics["reranked_final_list"] = [
            {
                "skill_id": item["unit"].skill_id,
                "source_type": item["unit"].source_type,
                "final_score": item.get("score"),
                "score_breakdown": item.get("score_breakdown"),
                "skill_filter": item.get("skill_filter", {}),
                "reason": "semantic + class/tool/performance/freshness/failure-aware rerank + diversity filtering",
            }
            for item in reranked
        ]
        return [self._retrieval_card_from_unit(item["unit"], item) for item in reranked], diagnostics

    def normalize_question(self, question: str) -> str:
        """Normalize evaluation wrappers so problem routing focuses on task semantics."""
        text = (question or "").strip()
        if not text:
            return ""

        wrappers = [
            r"^Based on these\s+\d+\s+uniformly sampled frames from a video,\s*please answer:\s*",
            r"^Based on these\s+\d+\s+frames from a video,\s*please answer:\s*",
            r"^Based on these\s+\d+\s+images,\s*please answer:\s*",
            r"^Please answer:\s*",
        ]
        for pattern in wrappers:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        text = re.sub(
            r"\n+Select from the following choices\..*$",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text or (question or "").strip()

    def _seed_pattern_score(self, pattern: str, text: str, tokens: List[str]) -> float:
        keywords = self._seed_class_keywords.get(pattern, [])
        hit_score = 0.0
        for kw in keywords:
            kw = str(kw).lower().strip()
            if not kw:
                continue
            if kw in text:
                hit_score += 1.0 if " " in kw else 0.6

        if tokens and keywords:
            kw_set = set(str(k).lower() for k in keywords)
            overlap = len([t for t in tokens if t in kw_set])
            hit_score += 0.4 * overlap
        return hit_score

    def _classify_seed_pattern(self, question: str) -> str:
        normalized_question = self.normalize_question(question)
        text = normalized_question.lower().strip()
        tokens = self._tokenize_question(normalized_question)
        best_pattern = "comprehensive_scene_understanding"
        best_score = -1.0

        for pattern in self._seed_pattern_order.keys():
            score = self._seed_pattern_score(pattern, text, tokens)
            if score > best_score:
                best_score = score
                best_pattern = pattern

        return best_pattern

    def _normalize_task_type(self, task_type: Optional[str]) -> str:
        raw = re.sub(r"[^a-zA-Z0-9_]+", "_", str(task_type or "").strip().lower()).strip("_")
        aliases = {
            "object_rel_direction_easy": "object_relative_direction",
            "object_rel_direction_medium": "object_relative_direction",
            "object_rel_direction_hard": "object_relative_direction",
            "object_rel_distance": "object_relative_distance",
            "object_abs_distance": "object_absolute_distance",
            "obj_appearance_order": "object_appearance_order",
        }
        return aliases.get(raw, raw)

    def _task_class_profile(self, task_type: str) -> Dict[str, Any]:
        task = self._normalize_task_type(task_type)
        profiles = {
            "object_absolute_distance": {
                "name": "Object Absolute Distance",
                "description": "Dataset task class for metric object-to-object distance estimation.",
                "seed_parent": "depth_distance",
                "keywords": ["distance", "meters", "measuring", "closest point", "between", "metric"],
            },
            "object_relative_distance": {
                "name": "Object Relative Distance",
                "description": "Dataset task class for comparing relative distance or near/far relations.",
                "seed_parent": "depth_distance",
                "keywords": ["closer", "farther", "nearest", "farthest", "distance", "relative distance"],
            },
            "object_counting": {
                "name": "Object Counting",
                "description": "Dataset task class for counting object instances across the scene.",
                "seed_parent": "detection_counting",
                "keywords": ["how many", "count", "number of", "instances", "visible"],
            },
            "object_size_estimation": {
                "name": "Object Size Estimation",
                "description": "Dataset task class for object length, width, height, or size estimation.",
                "seed_parent": "spatial_localization",
                "keywords": ["longest dimension", "width", "height", "length", "size", "larger"],
            },
            "object_relative_direction": {
                "name": "Object Relative Direction",
                "description": "Dataset task class for egocentric front-left/front-right/back-left/back-right reasoning.",
                "seed_parent": "spatial_localization",
                "keywords": ["front-left", "front-right", "back-left", "back-right", "quadrants", "standing", "facing"],
            },
            "object_appearance_order": {
                "name": "Object Appearance Order",
                "description": "Dataset task class for ordering objects by appearance or spatial sequence in views.",
                "seed_parent": "spatial_localization",
                "keywords": ["appearance order", "order", "first", "second", "third", "sequence"],
            },
            "route_planning": {
                "name": "Route Planning",
                "description": "Dataset task class for navigation action sequence and turn decision problems.",
                "seed_parent": "spatial_localization",
                "keywords": ["navigate", "go forward", "turn left", "turn right", "turn back", "route"],
            },
            "room_size_estimation": {
                "name": "Room Size Estimation",
                "description": "Dataset task class for estimating room size or combined floor area.",
                "seed_parent": "depth_distance",
                "keywords": ["room size", "square meters", "square meter", "combined space", "floor area"],
            },
        }
        return profiles.get(
            task,
            {
                "name": self._titleize_identifier(task),
                "description": "Dataset task class observed during rollout.",
                "seed_parent": self._classify_seed_pattern(task.replace("_", " ")),
                "keywords": [token for token in task.split("_") if token],
            },
        )

    def _ensure_task_question_class(
        self,
        task_type: Optional[str],
        question: str,
    ) -> Optional[str]:
        class_id = self._normalize_task_type(task_type)
        if not class_id:
            return None
        profile = self._task_class_profile(class_id)
        qcm = self._question_class_memory()
        classes = qcm.setdefault("classes", {})
        entry = classes.get(class_id)
        now = datetime.now().isoformat()
        if entry is None:
            entry = {
                "class_id": class_id,
                "name": profile["name"],
                "description": profile["description"],
                "seed_parent": profile["seed_parent"],
                "keywords": list(profile.get("keywords", [])),
                "example_questions": [],
                "hits": 0,
                "first_seen": now,
                "last_seen": now,
                "source": "dataset_task",
                "solution_stats": {},
            }
            classes[class_id] = entry
            if class_id not in qcm.setdefault("class_order", []):
                qcm["class_order"].append(class_id)
            qcm["global_version"] = int(qcm.get("global_version", 0)) + 1
        else:
            entry["name"] = entry.get("name") or profile["name"]
            entry["description"] = entry.get("description") or profile["description"]
            entry["seed_parent"] = profile["seed_parent"]
            entry["source"] = entry.get("source") or "dataset_task"
            keywords = [str(k).lower() for k in entry.get("keywords", []) if str(k).strip()]
            for keyword in profile.get("keywords", []):
                keyword = str(keyword).lower().strip()
                if keyword and keyword not in keywords:
                    keywords.append(keyword)
            entry["keywords"] = keywords[:24]
        return class_id

    def _is_seed_problem_id(self, problem_id: Optional[str]) -> bool:
        return str(problem_id or "").strip() in self._seed_pattern_order

    def _resolve_seed_parent(
        self,
        problem_id: Optional[str],
        question: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> str:
        candidate_id = str(problem_id or "").strip()
        if self._is_seed_problem_id(candidate_id):
            return candidate_id

        if not isinstance(info, dict):
            info = self._question_class_memory().get("classes", {}).get(candidate_id, {}) if candidate_id else {}

        stored_parent = str(info.get("seed_parent") or "").strip() if isinstance(info, dict) else ""
        if self._is_seed_problem_id(stored_parent):
            return stored_parent

        probe_parts: List[str] = []
        if question:
            probe_parts.append(self.normalize_question(question))
        if isinstance(info, dict):
            probe_parts.extend(
                self.normalize_question(item)
                for item in info.get("example_questions", [])
                if str(item).strip()
            )
            probe_parts.extend(str(item).strip() for item in info.get("keywords", []) if str(item).strip())
            if str(info.get("description") or "").strip():
                probe_parts.append(str(info.get("description")).strip())
            if str(info.get("name") or "").strip():
                probe_parts.append(str(info.get("name")).strip())

        probe_text = " ".join(part for part in probe_parts if part).strip()
        if probe_text:
            return self._classify_seed_pattern(probe_text)
        return "comprehensive_scene_understanding"

    def _build_problem_card(
        self,
        problem_id: str,
        normalized_question: str,
        matched_problem_id: Optional[str] = None,
        runtime_problem_id: Optional[str] = None,
        include_dynamic: bool = True,
    ) -> Dict[str, Any]:
        classes = self._question_class_memory().get("classes", {})
        memory_problem_id = matched_problem_id or problem_id
        info = classes.get(memory_problem_id, {})
        examples = info.get("example_questions", [])[-3:]
        profile = self._get_static_problem_profile(problem_id)
        static_path = self._static_problem_dir / self._slug(problem_id) / "PROBLEM.md"
        dynamic_path = self._dynamic_problem_dir / self._slug(memory_problem_id) / "PROBLEM.md"
        priority_tools = self._priority_tools_for_problem(problem_id, normalized_question)
        helper_tool_groups = self._helper_tool_groups_for_problem(problem_id, normalized_question)
        must_try_tools = self._must_try_tools_for_problem(problem_id, normalized_question)
        defer_tools = self._deferred_tools_for_problem(problem_id, normalized_question)

        return {
            "id": problem_id,
            "seed_problem_id": problem_id,
            "matched_problem_id": matched_problem_id,
            "title": info.get("name") or problem_id,
            "normalized_question": normalized_question,
            "intent": profile["intent"],
            "required_evidence": profile["required_evidence"],
            "near_misses": profile["near_misses"],
            "keywords": info.get("keywords", [])[:8],
            "example_questions": examples,
            "static_path": str(static_path),
            "dynamic_path": str(dynamic_path) if include_dynamic and matched_problem_id else "",
            "runtime_class_known": bool(runtime_problem_id),
            "runtime_problem_id": runtime_problem_id,
            "profile_source_problem_id": problem_id,
            "priority_tools": priority_tools,
            "priority_tool_note": self._priority_tool_note_for_problem(problem_id, normalized_question),
            "helper_tool_groups": helper_tool_groups,
            "must_try_tools": must_try_tools,
            "must_try_tool_note": self._must_try_tool_note_for_problem(problem_id, normalized_question),
            "defer_tools": defer_tools,
            "defer_tool_note": self._defer_tool_note_for_problem(problem_id, normalized_question),
        }

    def _build_focus_skills(
        self,
        seed_problem_id: str,
        matched_problem_id: Optional[str] = None,
        normalized_question: str = "",
    ) -> List[Dict[str, Any]]:
        focus: List[Dict[str, Any]] = []
        seen = set()

        def add_card(card: Optional[Dict[str, Any]]):
            if not isinstance(card, dict):
                return
            card_id = str(card.get("id") or "").strip()
            if not card_id or card_id in seen:
                return
            seen.add(card_id)
            focus.append(card)

        add_card(self._build_static_skill_card(seed_problem_id, normalized_question=normalized_question))

        dynamic_pattern = matched_problem_id or seed_problem_id
        for skill in self._skill_memory().get("skills", []):
            if not isinstance(skill, dict):
                continue
            meta = skill.get("meta", {})
            if str(meta.get("maturity") or "mature").strip() == "micro_policy":
                continue
            if str(meta.get("pattern") or "").strip() != dynamic_pattern:
                continue
            add_card(self._build_dynamic_skill_card(skill, normalized_question=normalized_question))

        return focus

    def _build_skill_catalog(self) -> List[Dict[str, Any]]:
        catalog: List[Dict[str, Any]] = []
        seen = set()

        def add_card(card: Optional[Dict[str, Any]]):
            if not card:
                return
            card_id = card.get("id")
            if not card_id or card_id in seen:
                return
            seen.add(card_id)
            catalog.append(card)

        # Preserve source/insertion order only; do not reorder by the current question.
        for problem_id in self._seed_pattern_order.keys():
            add_card(self._build_static_skill_card(problem_id))

        skill_mem = self._skill_memory()
        for skill in skill_mem.get("skills", []):
            if isinstance(skill, dict):
                meta = skill.get("meta", {}) if isinstance(skill.get("meta"), dict) else {}
                if str(meta.get("maturity") or "mature").strip() == "micro_policy":
                    continue
                add_card(self._build_dynamic_skill_card(skill))

        return catalog

    def _build_static_skill_card(
        self,
        problem_id: str,
        normalized_question: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        seed_skill = self._get_seed_skill_for_pattern(problem_id)
        if not seed_skill:
            return None
        profile = self._get_static_problem_profile(problem_id)
        slug = self._slug(problem_id)
        base_dir = self._static_skill_dir / slug
        tools = self._extract_tools_from_usage(seed_skill.get("tool_usage", ""))
        priority_tools = self._priority_tools_for_problem(problem_id, normalized_question)
        required_tools = self._skill_required_tools_for_problem(problem_id, normalized_question or "")
        return {
            "id": f"seed::{problem_id}",
            "name": seed_skill.get("name", problem_id),
            "source": "static",
            "when": seed_skill.get("when"),
            "tools": tools,
            "required_tools": required_tools,
            "priority_tools": priority_tools,
            "decision_logic": profile["default_plan"],
            "escalate_when": profile["escalate_when"],
            "stop_when": profile["stop_when"],
            "skill_path": str(base_dir / "SKILL.md"),
            "resource_paths": [
                str(base_dir / "references" / "guide.md"),
                str(base_dir / "references" / "examples.md"),
                str(base_dir / "references" / "tool-hints.md"),
                str(base_dir / "references" / "failure-recovery.md"),
            ],
        }

    def _build_dynamic_skill_card(
        self,
        skill: Optional[Dict[str, Any]],
        normalized_question: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not skill:
            return None
        meta = skill.get("meta", {})
        pattern = meta.get("pattern", "comprehensive_scene_understanding")
        seed_parent = self._resolve_seed_parent(pattern, info=self._question_class_memory().get("classes", {}).get(pattern, {}))
        stored_parent = str(meta.get("seed_parent") or "").strip()
        if self._is_seed_problem_id(stored_parent):
            seed_parent = stored_parent
        profile = self._get_static_problem_profile(seed_parent)
        skill_dir = self._dynamic_skill_dir / self._slug(skill.get("id", pattern))
        sequence = self._normalize_tool_sequence(meta.get("sequence"))
        if not sequence and str(skill.get("tool_usage") or "").strip() != "no_tool":
            sequence = self._extract_tools_from_usage(skill.get("tool_usage", ""))
        priority_tools = self._priority_tools_for_problem(seed_parent, normalized_question)
        if not sequence and str(skill.get("tool_usage") or "").strip() == "no_tool":
            required_tools = []
        else:
            required_tools = self._ordered_unique_tools(
                [str(tool).strip() for tool in sequence if str(tool).strip() in self._known_tools]
            ) or self._skill_required_tools_for_problem(seed_parent, normalized_question or "")
        failure_memory = meta.get("failure_memory", {}) if isinstance(meta.get("failure_memory"), dict) else {}
        return {
            "id": skill.get("id"),
            "name": skill.get("name"),
            "source": "dynamic",
            "problem_id": pattern,
            "seed_problem_id": seed_parent,
            "when": skill.get("when"),
            "tools": sequence,
            "required_tools": required_tools,
            "trigger": skill.get("trigger") or skill.get("when"),
            "blocker_type": meta.get("blocker_type", "object_localization"),
            "preconditions": skill.get("preconditions") or [],
            "skip_conditions": skill.get("skip_conditions") or [],
            "tool_candidates": skill.get("tool_candidates") or sequence,
            "fallbacks": skill.get("fallbacks") or [],
            "stop_conditions": skill.get("stop_conditions") or profile["stop_when"],
            "required_evidence": skill.get("required_evidence") or meta.get("required_evidence") or profile["required_evidence"],
            "answer_pattern": skill.get("answer_pattern") or meta.get("answer_pattern") or "",
            "coverage": skill.get("coverage") or meta.get("coverage") or {},
            "view_policy": skill.get("view_policy") or {},
            "maturity": meta.get("maturity", "mature"),
            "priority_tools": priority_tools,
            "decision_logic": skill.get("strategy") or profile["default_plan"],
            "escalate_when": profile["escalate_when"],
            "stop_when": profile["stop_when"],
            "observed_support": {
                "success": int(meta.get("success", 0)),
                "total": int(meta.get("total", 0)),
            },
            "failure_memory": failure_memory,
            "failure_lessons": skill.get("failure_lessons") or meta.get("failure_lessons") or {},
            "skill_path": str(skill_dir / "SKILL.md"),
            "resource_paths": [
                str(skill_dir / "references" / "guide.md"),
                str(skill_dir / "references" / "examples.md"),
                str(skill_dir / "references" / "tool-hints.md"),
                str(skill_dir / "references" / "failure-recovery.md"),
            ],
        }

    def _get_static_problem_profile(self, problem_id: str) -> Dict[str, Any]:
        defaults = {
            "intent": "Resolve the current spatial reasoning question with the minimum reliable evidence.",
            "required_evidence": [
                "identify the target objects or regions",
                "collect the evidence that directly changes the answer",
                "stop when additional tool calls no longer reduce uncertainty",
            ],
            "near_misses": [
                "questions that are purely descriptive with no spatial decision",
                "questions whose answer can be read directly without tool escalation",
            ],
            "default_plan": [
                "Identify the question type and target objects first.",
                "Choose the tools that can most directly test the needed evidence.",
                "Add complementary evidence only if it may change the answer.",
            ],
            "escalate_when": [
                "the current evidence is ambiguous or contradictory",
                "another modality or viewpoint could materially change the answer",
                "the current evidence cannot support a confident final choice",
            ],
            "stop_when": [
                "the answer is stable across the collected evidence",
                "another tool call would only repeat existing evidence",
            ],
        }
        overrides = {
            "spatial_localization": {
                "intent": "Resolve left/right/between/navigation-style spatial relationships.",
                "required_evidence": ["target object identity", "relative layout", "viewpoint ambiguity if any"],
                "near_misses": ["pure counting", "room-size estimation"],
                "default_plan": [
                    "Locate the target objects first.",
                    "Use whichever tool or combination best clarifies the relation.",
                    "Add orientation, segmentation, or 3D evidence only when it can change the answer.",
                ],
            },
            "detection_counting": {
                "intent": "Count or enumerate target objects in the scene.",
                "required_evidence": ["target category", "object instances", "deduplicated count"],
                "near_misses": ["relative position", "room-size estimation"],
            },
            "fine_grained_pointing": {
                "intent": "Locate an object precisely enough to point to it.",
                "required_evidence": ["precise target identity", "pointing coordinates or local region", "visual verification"],
            },
            "comprehensive_scene_understanding": {
                "intent": "Answer complex questions that need multiple complementary signals.",
                "required_evidence": ["scene overview", "target-specific evidence", "cross-check from at least one complementary tool"],
            },
            "depth_distance": {
                "intent": "Reason about metric depth, distance, front/back, near/far, and point-to-point or object-to-object measurements, with depth_estimation_tool (DA3 / DA3NESTED) as a strong direct cue.",
                "required_evidence": [
                    "localized target objects, regions, or precise image points",
                    "region-level or point-level metric depth evidence",
                    "boundary or point precision when closest-point measurement matters",
                    "the correct interpretation of camera_axis_depth versus camera_distance when actual distance is asked",
                    "3D verification when occlusion, viewpoint change, or room/layout geometry makes direct 2D depth evidence insufficient",
                ],
                "near_misses": ["pure counting", "identity-only recognition without a depth decision"],
                "default_plan": [
                    "If the question is nearest/farthest, front/back, closest-point distance, depth ranking, or asks for meters, keep depth_estimation_tool as the early anchor instead of treating it as a last-resort cue.",
                    "Do not default to detect_objects_tool first on every sample. Use it only when reusable crop boxes are the missing evidence for depth_estimation_tool, swinir_tool, or orient_anything_tool.",
                    "If the unresolved blocker is closest-point precision, boundaries, or contact regions, use segment_image_tool or moondream_tool before assuming a coarse box is good enough.",
                    "If the unresolved blocker is tiny or blurry local detail, use swinir_tool on the local crop before repeating localization or depth.",
                    "For exact metric measurement between selected image locations, call depth_estimation_tool with point_coords and point_pairs so it can return point_measurements and point_pair_distances in meters.",
                    "Remember that camera_axis_depth is optical-axis depth Z, while camera_distance is Euclidean range from the camera center. Use camera_distance or point_pair_distances when the question asks for actual metric distance.",
                    "If the blocker is occlusion, viewpoint change, or overall room/space geometry, bring in pi3_tool earlier once you have a rough depth or localization anchor. Otherwise keep pi3_tool as a late escalation.",
                ],
            },
            "multiview_occlusion": {
                "intent": "Reason about visibility, occlusion, and what different viewpoints can see.",
                "required_evidence": [
                    "camera/viewpoint identity",
                    "occlusion evidence",
                    "multi-view comparison",
                    "3D verification of hidden or newly visible regions",
                ],
                "near_misses": [
                    "single-view depth ranking without a viewpoint change",
                    "identity-only recognition without occlusion reasoning",
                ],
                "default_plan": [
                    "Treat pi3_tool as the primary tool when the answer depends on hidden regions, viewpoint change, or what another view would reveal.",
                    "Use detect_objects_tool or moondream_tool only to anchor reusable regions or specific points across views, not as a substitute for 3D visibility reasoning.",
                    "If front/back or facing direction controls the occlusion pattern, use orient_anything_tool as a direct helper.",
                    "If the relevant evidence is tiny or blurry, use swinir_tool before rechecking visibility.",
                ],
            },
            "segmentation_boundary": {
                "intent": "Trace object masks, outlines, or precise boundaries.",
                "required_evidence": ["target localization", "mask quality", "boundary correctness"],
            },
            "custom_category": {
                "intent": "Detect user-specified custom categories or uncommon object labels.",
                "required_evidence": ["category definition", "instance detections", "visual verification"],
            },
        }
        profile = dict(defaults)
        profile.update(overrides.get(problem_id, {}))
        return profile

    def _contains_any_phrase(self, text: str, phrases: List[str]) -> bool:
        normalized = str(text or "").lower()
        return any(phrase in normalized for phrase in phrases)

    def _depth_distance_signals(self, normalized_question: str) -> Dict[str, bool]:
        text = str(normalized_question or "").lower()
        return {
            "boundary_precision": self._contains_any_phrase(
                text,
                [
                    "closest point",
                    "closest points",
                    "contact",
                    "touch",
                    "touching",
                    "surface",
                    "surfaces",
                    "edge",
                    "edges",
                    "boundary",
                    "boundaries",
                    "outline",
                    "outlines",
                    "mask",
                ],
            ),
            "exact_points": self._contains_any_phrase(
                text,
                [
                    "selected point",
                    "selected points",
                    "point-to-point",
                    "point to point",
                    "point pair",
                    "point pairs",
                    "exact point",
                    "exact points",
                    "pixel",
                    "pixels",
                    "measurement point",
                    "measurement points",
                    "tip",
                    "corner",
                    "handle",
                ],
            ),
            "low_quality": self._contains_any_phrase(
                text,
                [
                    "small",
                    "tiny",
                    "blurry",
                    "blurred",
                    "hard to see",
                    "far away",
                    "distant",
                    "low resolution",
                    "low-res",
                    "unclear",
                    "fuzzy",
                ],
            ),
            "orientation": self._contains_any_phrase(
                text,
                [
                    "orientation",
                    "oriented",
                    "facing",
                    "front side",
                    "back side",
                    "upright",
                    "upside down",
                    "clockwise",
                    "counterclockwise",
                    "pose",
                    "rotation",
                    "rotated",
                    "parallel",
                    "perpendicular",
                    "tilted",
                    "angle",
                ],
            ),
            "viewpoint": self._contains_any_phrase(
                text,
                [
                    "occluded",
                    "occlusion",
                    "hidden",
                    "blocked",
                    "not visible",
                    "another view",
                    "other view",
                    "different view",
                    "viewpoint",
                    "camera view",
                ],
            ),
            "room_layout": self._contains_any_phrase(
                text,
                [
                    "room size",
                    "size of the room",
                    "how big is the room",
                    "how large is the room",
                    "dimensions of the room",
                    "overall room",
                    "overall space",
                    "combined space",
                    "room layout",
                    "space layout",
                    "scene layout",
                    "layout of the room",
                    "layout of the space",
                    "floor area",
                    "overall area",
                    "area of the room",
                    "area of this room",
                    "square meter",
                    "square meters",
                    "square metre",
                    "square metres",
                    "sq m",
                ],
            ),
        }

    def _ordered_unique_tools(self, tool_names: List[str]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for tool_name in tool_names:
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)
            ordered.append(tool_name)
        return ordered

    def _prefer_pi3_early_for_depth_distance(self, signals: Dict[str, bool]) -> bool:
        return bool(signals.get("viewpoint") or signals.get("room_layout"))

    def _priority_tools_for_problem(self, problem_id: str, normalized_question: Optional[str] = None) -> List[str]:
        if problem_id == "multiview_occlusion":
            signals = self._depth_distance_signals(normalized_question or "")
            priority_tools = ["pi3_tool"]
            if signals["orientation"]:
                priority_tools.append("orient_anything_tool")
            if signals["low_quality"]:
                priority_tools.append("swinir_tool")
            return self._ordered_unique_tools(priority_tools)

        if problem_id != "depth_distance":
            return []

        signals = self._depth_distance_signals(normalized_question or "")
        priority_tools = ["depth_estimation_tool"]
        if self._prefer_pi3_early_for_depth_distance(signals):
            priority_tools.append("pi3_tool")
        if signals["boundary_precision"]:
            priority_tools.append("segment_image_tool")
        if signals["boundary_precision"] or signals["exact_points"]:
            priority_tools.append("moondream_tool")
        if signals["low_quality"]:
            priority_tools.append("swinir_tool")
        if signals["orientation"]:
            priority_tools.append("orient_anything_tool")
        return self._ordered_unique_tools(priority_tools)

    def _skill_required_tools_for_problem(self, problem_id: str, normalized_question: str) -> List[str]:
        """Minimal tool anchors required once a catalog skill is actively selected.

        Static skill `tools` are broad candidate sets, not literal sequences. These
        anchors make a selected skill operational without forcing every helper in
        its common-tool list.
        """
        anchors = {
            "spatial_localization": ["detect_objects_tool"],
            "depth_distance": ["depth_estimation_tool"],
            "multiview_occlusion": ["pi3_tool"],
            "detection_counting": ["detect_objects_tool"],
            "segmentation_boundary": ["segment_image_tool"],
            "fine_grained_pointing": ["moondream_tool"],
            "custom_category": ["detect_objects_tool"],
        }
        return self._ordered_unique_tools(anchors.get(problem_id, []))

    def _helper_tool_groups_for_problem(self, problem_id: str, normalized_question: str) -> List[Dict[str, Any]]:
        if problem_id == "multiview_occlusion":
            signals = self._depth_distance_signals(normalized_question)
            helper_groups: List[Dict[str, Any]] = [
                {
                    "when": "If the answer depends on hidden regions, changed viewpoints, or cross-view geometry.",
                    "tools": ["pi3_tool"],
                    "note": "Prefer pi3_tool as the primary 3D visibility check instead of repeatedly retrying generic 2D localization.",
                },
                {
                    "when": "If reusable boxes or object identity are still missing across views.",
                    "tools": ["detect_objects_tool", "moondream_tool"],
                    "note": "Use detect_objects_tool for reusable boxes across views. Prefer moondream_tool when the key evidence is a specific point, local patch, or partially visible target.",
                },
            ]
            if signals["orientation"]:
                helper_groups.append(
                    {
                        "when": "If front/back side or facing direction determines the occlusion pattern.",
                        "tools": ["orient_anything_tool"],
                        "note": "Prefer orient_anything_tool when pose or facing direction changes what should be visible from each viewpoint.",
                    }
                )
            if signals["low_quality"]:
                helper_groups.append(
                    {
                        "when": "If the visible evidence is tiny, blurry, or hard to inspect.",
                        "tools": ["swinir_tool", "detect_objects_tool"],
                        "note": "Use swinir_tool before rechecking visibility or identity on the critical crop.",
                    }
                )
            return helper_groups

        if problem_id != "depth_distance":
            return []

        signals = self._depth_distance_signals(normalized_question)
        helper_groups: List[Dict[str, Any]] = [
            {
                "when": "If reusable boxes or coarse object localization are still missing.",
                "tools": ["detect_objects_tool", "moondream_tool"],
                "note": "Prefer detect_objects_tool for reusable crop boxes. Prefer moondream_tool when exact target points matter more than boxes.",
            }
        ]

        if signals["boundary_precision"] or signals["exact_points"]:
            helper_groups.append(
                {
                    "when": "If closest points, boundaries, contact regions, or exact measurement points matter.",
                    "tools": ["segment_image_tool", "moondream_tool"],
                    "note": "Use segmentation or precise pointing so depth_estimation_tool can receive crop_box, crop_boxes, point_coords, or point_pairs aligned with the real measurement surface.",
                }
            )

        if signals["low_quality"]:
            helper_groups.append(
                {
                    "when": "If the relevant object region is tiny, blurry, or hard to inspect.",
                    "tools": ["swinir_tool", "detect_objects_tool"],
                    "note": "Prefer swinir_tool as the first recovery step for unreadable local crops, then rerun localization or reasoning on the enhanced region.",
                }
            )

        if signals["orientation"]:
            helper_groups.append(
                {
                    "when": "If pose or facing direction could change which surface or direction is measured.",
                    "tools": ["orient_anything_tool"],
                    "note": "Prefer orient_anything_tool when pose, facing direction, front/back side, or axis alignment is the missing evidence, instead of repeating generic detection.",
                }
            )

        if self._prefer_pi3_early_for_depth_distance(signals):
            helper_groups.append(
                {
                    "when": "If occlusion, viewpoint change, or overall room/space geometry is part of the answer.",
                    "tools": ["pi3_tool"],
                    "note": "Promote pi3_tool earlier when visible-pixel depth alone cannot resolve hidden structure, viewpoint ambiguity, or room-scale geometry.",
                }
            )
        else:
            helper_groups.append(
                {
                    "when": "If viewpoint change or occlusion is still the blocker after direct 2D evidence.",
                    "tools": ["pi3_tool"],
                    "note": "Delay pi3_tool until depth/localization/segmentation/enhancement evidence still leaves the answer unstable.",
                }
            )
        return helper_groups

    def _priority_tool_note_for_problem(self, problem_id: str, normalized_question: str) -> str:
        if problem_id == "multiview_occlusion":
            signals = self._depth_distance_signals(normalized_question)
            notes = [
                "This is a multi-view occlusion problem. Prefer pi3_tool as the primary 3D visibility check instead of defaulting to generic 2D relocalization.",
                "Use detect_objects_tool or moondream_tool only when reusable regions or object identity are still missing across views.",
            ]
            if signals["orientation"]:
                notes.append("If front/back side or facing direction determines the occlusion pattern, use orient_anything_tool as the direct helper.")
            if signals["low_quality"]:
                notes.append("If the relevant region is tiny or blurry, use swinir_tool before rechecking visibility.")
            return " ".join(notes)

        if problem_id != "depth_distance":
            return ""

        signals = self._depth_distance_signals(normalized_question)
        notes = [
            "This is a depth/distance problem. Keep depth_estimation_tool as the anchor tool, but do not default to detect_objects_tool on every sample.",
            "Use detect_objects_tool only when reusable crop boxes are actually needed.",
        ]
        if signals["boundary_precision"]:
            notes.append("Because closest points or boundaries matter here, prefer segment_image_tool or moondream_tool before relying on a coarse box.")
        elif signals["exact_points"]:
            notes.append("Because exact measurement points matter here, prefer moondream_tool before repeating coarse localization.")
        if signals["low_quality"]:
            notes.append("If the relevant region is tiny or blurry, run swinir_tool on the local crop before measuring depth again.")
        if signals["orientation"]:
            notes.append("If pose or facing direction changes which surface matters, prioritize orient_anything_tool as the direct helper rather than jumping straight to 3D or repeating generic detection.")
        if self._prefer_pi3_early_for_depth_distance(signals):
            notes.append("Because the question depends on occlusion, viewpoint change, or overall room/space geometry, bring pi3_tool in earlier once a rough depth or localization anchor is available.")
        else:
            notes.append("Delay pi3_tool until direct 2D cues remain insufficient.")
        return " ".join(notes)

    def _deferred_tools_for_problem(self, problem_id: str, normalized_question: str) -> List[str]:
        if problem_id == "depth_distance":
            signals = self._depth_distance_signals(normalized_question)
            if self._prefer_pi3_early_for_depth_distance(signals):
                return []
            return ["pi3_tool"]
        return []

    def _defer_tool_note_for_problem(self, problem_id: str, normalized_question: str) -> str:
        if problem_id != "depth_distance":
            return ""

        signals = self._depth_distance_signals(normalized_question)
        if self._prefer_pi3_early_for_depth_distance(signals):
            return ""
        return (
            "Delay pi3_tool until depth_estimation_tool and lighter 2D helpers such as detect_objects_tool, moondream_tool, segment_image_tool, or swinir_tool are still insufficient."
        )
    def _must_try_tools_for_problem(self, problem_id: str, normalized_question: str) -> List[str]:
        text = str(normalized_question or "").lower()
        if problem_id == "depth_distance" and "in meters" in text:
            return ["depth_estimation_tool"]
        return []

    def _must_try_tool_note_for_problem(self, problem_id: str, normalized_question: str) -> str:
        text = str(normalized_question or "").lower()
        if problem_id == "depth_distance" and "in meters" in text:
            return (
                "The question explicitly asks for a metric answer in meters. Before giving the final answer, attempt depth_estimation_tool at least once. Do not answer this meters-valued depth/distance question from RGB intuition alone."
            )
        return ""

    def _infer_blocker_type(
        self,
        pattern: str,
        normalized_question: str,
        sequence: List[str],
    ) -> str:
        signals = self._depth_distance_signals(normalized_question)
        if pattern == "multiview_occlusion" or "pi3_tool" in sequence:
            return "viewpoint_change"
        if pattern == "depth_distance":
            if "depth_estimation_tool" in sequence:
                return "metric_distance"
            if signals["boundary_precision"]:
                return "boundary_precision"
            if signals["exact_points"]:
                return "exact_points"
            return "depth_reasoning"
        if pattern == "segmentation_boundary" or "segment_image_tool" in sequence:
            return "boundary_precision"
        if pattern == "fine_grained_pointing" or "moondream_tool" in sequence:
            return "exact_points"
        if pattern == "custom_category":
            return "custom_localization"
        if pattern == "detection_counting":
            return "instance_counting"
        return "object_localization"

    def _default_preconditions(
        self,
        pattern: str,
        blocker_type: str,
        sequence: List[str],
    ) -> List[str]:
        preconditions: List[str] = []
        if "detect_objects_tool" in sequence:
            preconditions.append("Use detect_objects_tool only when reusable crop boxes or category-level localization are still missing.")
        if "depth_estimation_tool" in sequence:
            preconditions.append("Prefer depth_estimation_tool only when the blocker is metric depth/distance, front/back, or a meters-valued answer.")
        if "segment_image_tool" in sequence:
            preconditions.append("Use segment_image_tool when object boundaries, closest-point reasoning, or contact regions matter.")
        if "moondream_tool" in sequence:
            preconditions.append("Use moondream_tool only when exact points matter more than coarse boxes.")
        if "swinir_tool" in sequence:
            preconditions.append("Use swinir_tool only when local evidence is tiny, blurry, or hard to inspect.")
        if "orient_anything_tool" in sequence:
            preconditions.append("Use orient_anything_tool only when pose, facing direction, or axis alignment is the missing evidence.")
        if "pi3_tool" in sequence or blocker_type == "viewpoint_change":
            preconditions.append("Escalate to pi3_tool only when viewpoint change, hidden geometry, or room-layout ambiguity remains unresolved.")
        if not preconditions:
            preconditions.append(f"Trigger this policy only when the active blocker is {blocker_type.replace('_', ' ')}.")
        return preconditions

    def _default_skip_conditions(
        self,
        pattern: str,
        blocker_type: str,
        sequence: List[str],
    ) -> List[str]:
        conditions = [
            "Skip this policy if the current evidence is already sufficient to answer confidently.",
            "Skip repeated tool calls when the same arguments previously failed without producing new evidence.",
        ]
        if pattern == "depth_distance":
            conditions.append("Do not default to detect_objects_tool first if depth_estimation_tool can already operate on the required points or regions.")
        if blocker_type == "viewpoint_change":
            conditions.append("Do not escalate to pi3_tool when the blocker is boundary completion or exact point selection rather than viewpoint insufficiency.")
        return conditions

    def _next_best_tool_for_failure(
        self,
        pattern: str,
        normalized_question: str,
        failed_tool: str,
        sequence: List[str],
    ) -> Optional[str]:
        candidates = self._priority_tools_for_problem(pattern, normalized_question)
        for group in self._helper_tool_groups_for_problem(pattern, normalized_question):
            if isinstance(group, dict):
                candidates.extend(str(tool).strip() for tool in group.get("tools", []) if str(tool).strip())
        candidates.extend(sequence)
        ordered = self._ordered_unique_tools([candidate for candidate in candidates if candidate and candidate != failed_tool])
        return ordered[0] if ordered else None

    def _default_fallbacks(
        self,
        pattern: str,
        normalized_question: str,
        sequence: List[str],
        failure_memory: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        fallbacks: List[Dict[str, Any]] = []
        for payload in (failure_memory or {}).values():
            if not isinstance(payload, dict):
                continue
            next_best_tool = str(payload.get("next_best_tool") or "").strip()
            if not next_best_tool:
                continue
            fallbacks.append(
                {
                    "failed_tool": str(payload.get("tool_name") or "").strip(),
                    "failure_type": str(payload.get("failure_type") or "").strip(),
                    "next_best_tool": next_best_tool,
                }
            )
        if fallbacks:
            return fallbacks
        if len(sequence) == 1:
            next_best_tool = self._next_best_tool_for_failure(pattern, normalized_question, sequence[0], sequence)
            if next_best_tool:
                return [
                    {
                        "failed_tool": sequence[0],
                        "failure_type": "generic_recovery",
                        "next_best_tool": next_best_tool,
                    }
                ]
        return []

    def _default_stop_conditions(
        self,
        pattern: str,
    ) -> List[str]:
        profile = self._get_static_problem_profile(pattern if self._is_seed_problem_id(pattern) else self._resolve_seed_parent(pattern))
        return list(profile.get("stop_when", []))

    def _extract_failure_events(
        self,
        tool_results: Dict[str, Any],
        pattern: str,
        normalized_question: str,
        sequence: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        failures: Dict[str, Dict[str, Any]] = {}
        for result_key, result in (tool_results or {}).items():
            if not isinstance(result, dict):
                continue
            if result.get("success"):
                continue
            if result.get("transient_api_error") or str(result.get("failure_type") or "") == "tool_api_transient":
                continue
            tool_name = str(result.get("tool_name") or str(result_key).split("_iter")[0]).strip()
            if not tool_name:
                continue
            failure_type = str(result.get("failure_type") or "tool_runtime_error").strip() or "tool_runtime_error"
            failure_key = f"{tool_name}:{failure_type}"
            entry = failures.setdefault(
                failure_key,
                {
                    "tool_name": tool_name,
                    "failure_type": failure_type,
                    "count": 0,
                    "last_error": "",
                    "next_best_tool": self._next_best_tool_for_failure(pattern, normalized_question, tool_name, sequence),
                    "last_seen": None,
                },
            )
            entry["count"] += 1
            entry["last_error"] = str(result.get("error") or entry.get("last_error") or "")
            entry["last_seen"] = datetime.now().isoformat()
        return failures

    def _required_evidence_types_for_question(self, seed_parent: str, normalized_question: str) -> List[str]:
        profile = self._get_static_problem_profile(seed_parent)
        required = [str(item).strip() for item in profile.get("required_evidence", []) if str(item).strip()]
        signals = self._depth_distance_signals(normalized_question)
        evidence_types: List[str] = []

        if seed_parent in {"spatial_localization", "detection_counting", "custom_category"}:
            evidence_types.append("object_level_evidence")
        if seed_parent == "depth_distance":
            evidence_types.append("metric_depth_evidence")
            if signals["boundary_precision"]:
                evidence_types.append("boundary_evidence")
            if signals["exact_points"]:
                evidence_types.append("point_level_evidence")
            if signals["viewpoint"] or signals["room_layout"]:
                evidence_types.append("multiview_geometry")
        if seed_parent == "segmentation_boundary" or signals["boundary_precision"]:
            evidence_types.append("boundary_evidence")
        if seed_parent == "fine_grained_pointing" or signals["exact_points"]:
            evidence_types.append("point_level_evidence")
        if seed_parent == "multiview_occlusion" or signals["viewpoint"]:
            evidence_types.append("multiview_geometry")
        if signals["orientation"]:
            evidence_types.append("orientation_evidence")
        if not evidence_types:
            evidence_types.append("scene_context_evidence")

        return list(dict.fromkeys(required + evidence_types))

    def _compact_tool_evidence_summary(self, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        successful_tools: List[str] = []
        failed_tools: List[str] = []
        output_keys: List[str] = []
        for result_key, result in (tool_results or {}).items():
            if not isinstance(result, dict):
                continue
            tool_name = str(result.get("tool_name") or str(result_key).split("_iter")[0]).strip()
            if result.get("success"):
                if tool_name and tool_name not in successful_tools:
                    successful_tools.append(tool_name)
            else:
                if tool_name and tool_name not in failed_tools:
                    failed_tools.append(tool_name)
            for key in (
                "boxes",
                "detections",
                "masks",
                "point_measurements",
                "point_pair_distances",
                "region_depth_stats",
                "depth_ranking_near_to_far",
                "output_path",
                "output_paths",
                "vis_path",
                "description",
            ):
                value = result.get(key)
                if value not in (None, "", [], {}) and key not in output_keys:
                    output_keys.append(key)
        return {
            "successful_tools": successful_tools,
            "failed_tools": failed_tools,
            "output_keys": output_keys,
            "objects": self._extract_object_labels(tool_results),
        }

    def _build_scene_task_context(
        self,
        question: str,
        question_class: str,
        seed_parent: str,
        tool_calls: List[Dict[str, Any]],
        tool_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        relations = self._extract_relation_hints(question)
        labels = self._extract_object_labels(tool_results)
        tokens = self._tokenize_question(question)
        target_entities = labels[:10]
        if not target_entities:
            target_entities = [
                token
                for token in tokens
                if token not in {"many", "distance", "between", "closer", "farther", "left", "right", "front", "behind"}
            ][:8]
        return {
            "task_category": question_class,
            "seed_task_category": seed_parent,
            "target_entities": target_entities,
            "scene_signature": self._build_scene_signature(tool_calls, tool_results),
            "relations": relations,
            "required_evidence": self._required_evidence_types_for_question(seed_parent, question),
        }

    def _diagnose_rollout_errors(
        self,
        question: str,
        seed_parent: str,
        sequence: List[str],
        full_sequence: List[str],
        tool_results: Dict[str, Any],
        success: bool,
        reward: float,
        is_correct: Optional[bool],
    ) -> List[str]:
        if bool(success) and reward >= 0.5 and is_correct is not False:
            return []

        errors: List[str] = []
        required_tools = self._skill_required_tools_for_problem(seed_parent, question)
        attempted = set(full_sequence)
        effective = set(sequence)
        failed_results = [
            result
            for result in (tool_results or {}).values()
            if isinstance(result, dict)
            and not result.get("success")
            and not result.get("transient_api_error")
            and str(result.get("failure_type") or "") != "tool_api_transient"
        ]
        successful_results = [result for result in (tool_results or {}).values() if isinstance(result, dict) and result.get("success")]

        if not full_sequence and required_tools:
            errors.append("missing_tool_call_when_required")
        if any(tool not in attempted for tool in required_tools):
            errors.append("missing_evidence")
        if failed_results:
            failure_types = {str(result.get("failure_type") or "").strip() for result in failed_results}
            if "schema_error" in failure_types:
                errors.append("invalid_tool_input")
            if "selector_rejected" in failure_types:
                errors.append("wrong_tool_selection")
            if "repeated_view" in failure_types or len(full_sequence) != len(list(dict.fromkeys(full_sequence))):
                errors.append("redundant_tool_calls")
            if not errors:
                errors.append("tool_execution_failure")
        if successful_results and not effective:
            errors.append("ignored_tool_output")
        if successful_results and (is_correct is False or reward < 0.5):
            errors.append("ignored_or_misread_valid_evidence")
        if not errors:
            errors.append("answer_not_supported")
        return list(dict.fromkeys(errors))

    def _find_compatible_workflow_skill(
        self,
        question_class: str,
        seed_parent: str,
        sequence: List[str],
    ) -> Optional[str]:
        if not sequence:
            return None
        sequence_key = "->".join(sequence)
        for skill in self._skill_memory().get("skills", []) or []:
            if not isinstance(skill, dict):
                continue
            meta = skill.get("meta", {}) if isinstance(skill.get("meta"), dict) else {}
            meta_sequence = self._normalize_tool_sequence(meta.get("effective_sequence") or meta.get("sequence"))
            if "->".join(meta_sequence) != sequence_key:
                continue
            if str(meta.get("pattern") or "") == question_class:
                return str(skill.get("id") or "")
            if str(meta.get("seed_parent") or "") == seed_parent:
                return str(skill.get("id") or "")
        return None

    def _coverage_delta(
        self,
        scene_context: Dict[str, Any],
        sequence: List[str],
    ) -> Dict[str, Any]:
        return {
            "scene_signatures": [scene_context.get("scene_signature")] if scene_context.get("scene_signature") else [],
            "target_entities": scene_context.get("target_entities", [])[:10],
            "evidence_types": scene_context.get("required_evidence", [])[:12],
            "tool_sequences": ["->".join(sequence)] if sequence else [],
        }

    def _merge_skill_coverage(self, current: Any, incoming: Any) -> Dict[str, Any]:
        merged = current if isinstance(current, dict) else {}
        incoming = incoming if isinstance(incoming, dict) else {}
        result: Dict[str, Any] = {}
        for key in {"scene_signatures", "target_entities", "evidence_types", "tool_sequences"}:
            items: List[str] = []
            for source in (merged.get(key, []), incoming.get(key, [])):
                for item in source or []:
                    item_text = str(item or "").strip()
                    if item_text and item_text not in items:
                        items.append(item_text)
            result[key] = items[-64:]
        return result

    def _answer_pattern_for_workflow(
        self,
        seed_parent: str,
        sequence: List[str],
    ) -> str:
        if seed_parent == "depth_distance":
            if "depth_estimation_tool" in sequence:
                return "Use metric depth, region statistics, or point-pair distances as the premise for the final distance/near-far answer."
            return "Answer only after collecting localized depth or boundary evidence."
        if seed_parent == "detection_counting":
            return "Deduplicate detected instances across views before giving the final count."
        if seed_parent == "multiview_occlusion":
            return "Use cross-view or reconstructed geometry to justify visibility/layout conclusions."
        if seed_parent == "segmentation_boundary":
            return "Use mask or boundary evidence as the premise for closest-point or boundary-sensitive answers."
        return "Use the selected tool evidence as an explicit premise for the final answer."

    def _build_skill_management_decision(
        self,
        question: str,
        question_class: str,
        seed_parent: str,
        sequence: List[str],
        full_sequence: List[str],
        tool_results: Dict[str, Any],
        success: bool,
        reward: float,
        is_correct: Optional[bool],
        reflection: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        scene_context = self._build_scene_task_context(
            question=question,
            question_class=question_class,
            seed_parent=seed_parent,
            tool_calls=[{"name": name} for name in full_sequence],
            tool_results=tool_results,
        )
        required_evidence = scene_context.get("required_evidence", [])
        error_types = self._diagnose_rollout_errors(
            question=question,
            seed_parent=seed_parent,
            sequence=sequence,
            full_sequence=full_sequence,
            tool_results=tool_results,
            success=success,
            reward=reward,
            is_correct=is_correct,
        )
        is_successful_rollout = bool(success) and reward >= 0.5 and is_correct is not False
        compatible_skill_id = self._find_compatible_workflow_skill(question_class, seed_parent, sequence)
        coverage_delta = self._coverage_delta(scene_context, sequence)

        if is_successful_rollout and compatible_skill_id:
            action = "merge_or_update_dynamic_skill"
            reason = "successful workflow is compatible with an existing skill; merge only scene/evidence/tool coverage that adds information"
        elif is_successful_rollout and not sequence:
            action = "insert_direct_dynamic_skill"
            reason = "successful rollout provides a reusable direct-answer policy for this seen problem type"
        elif is_successful_rollout:
            action = "insert_dynamic_skill"
            reason = "successful workflow provides a reusable scene-aware tool-use pattern"
        elif error_types:
            if sequence:
                action = "attach_failure_lesson_or_patch_skill"
                reason = "failed workflow produced diagnosable lesson or fallback rule"
            else:
                action = "attach_missing_tool_lesson"
                reason = "failure occurred without executable tool evidence"
        else:
            action = "ignore_no_reusable_signal"
            reason = "rollout does not provide reusable success or diagnosable failure signal"

        return {
            "schema_version": 1,
            "action": action,
            "reason": reason,
            "question": question,
            "question_class": question_class,
            "seed_parent": seed_parent,
            "scene_task_context": scene_context,
            "required_evidence": required_evidence,
            "tool_sequence": sequence,
            "full_tool_sequence": full_sequence,
            "compatible_skill_id": compatible_skill_id,
            "coverage_delta": coverage_delta,
            "answer_pattern": self._answer_pattern_for_workflow(seed_parent, sequence),
            "error_types": error_types,
            "tool_evidence_summary": self._compact_tool_evidence_summary(tool_results),
            "reward": round(float(reward), 6),
            "is_correct": None if is_correct is None else bool(is_correct),
            "reflection": reflection if isinstance(reflection, dict) else None,
        }

    def _record_skill_management_decision(self, decision: Dict[str, Any]) -> None:
        if not isinstance(decision, dict):
            return
        management = self._skill_memory().setdefault(
            "skill_management",
            {
                "schema_version": 1,
                "decisions": [],
                "failure_lessons": {},
                "success_workflows": {},
            },
        )
        compact = self._compact_for_memory(decision)
        decisions = management.setdefault("decisions", [])
        decisions.append(compact)
        if len(decisions) > 500:
            del decisions[:-500]

        sequence = self._normalize_tool_sequence(decision.get("tool_sequence"))
        workflow_key = self._sequence_key(
            str(decision.get("question_class") or "unknown"),
            sequence or ["no_tool"],
        )
        if str(decision.get("action", "")).startswith(("insert", "merge")) and sequence:
            workflows = management.setdefault("success_workflows", {})
            workflow = workflows.setdefault(
                workflow_key,
                {
                    "question_class": decision.get("question_class"),
                    "seed_parent": decision.get("seed_parent"),
                    "tool_sequence": sequence,
                    "total": 0,
                    "success": 0,
                    "coverage": {},
                    "answer_pattern": decision.get("answer_pattern", ""),
                },
            )
            workflow["total"] = int(workflow.get("total", 0)) + 1
            workflow["success"] = int(workflow.get("success", 0)) + 1
            workflow["coverage"] = self._merge_skill_coverage(
                workflow.get("coverage", {}),
                decision.get("coverage_delta", {}),
            )

        error_types = [str(item).strip() for item in decision.get("error_types", []) if str(item).strip()]
        if error_types:
            lesson_key = self._sequence_key(
                str(decision.get("seed_parent") or decision.get("question_class") or "unknown"),
                sequence or ["no_tool"],
            )
            lessons = management.setdefault("failure_lessons", {})
            lesson = lessons.setdefault(
                lesson_key,
                {
                    "seed_parent": decision.get("seed_parent"),
                    "question_class": decision.get("question_class"),
                    "tool_sequence": sequence,
                    "count": 0,
                    "error_types": {},
                    "fallback_tools": {},
                    "example_questions": [],
                },
            )
            lesson["count"] = int(lesson.get("count", 0)) + 1
            for error_type in error_types:
                counts = lesson.setdefault("error_types", {})
                counts[error_type] = int(counts.get(error_type, 0)) + 1
            evidence_summary = decision.get("tool_evidence_summary", {})
            if isinstance(evidence_summary, dict):
                for tool_name in evidence_summary.get("failed_tools", []) or []:
                    fallback = self._next_best_tool_for_failure(
                        str(decision.get("seed_parent") or ""),
                        str(decision.get("question") or ""),
                        str(tool_name),
                        sequence,
                    )
                    if fallback:
                        fallbacks = lesson.setdefault("fallback_tools", {})
                        fallbacks[str(tool_name)] = fallback
            examples = lesson.setdefault("example_questions", [])
            question = str(decision.get("question") or "").strip()
            if question and question not in examples:
                examples.append(question)
                lesson["example_questions"] = examples[-8:]

    def _skill_maturity(
        self,
        sequence: List[str],
        total: int,
        failure_memory: Dict[str, Any],
    ) -> str:
        if len(sequence) == 0 and total >= 1:
            return "direct_policy"
        if len(sequence) > 1:
            return "mature"
        if len(sequence) == 1 and total >= 5 and any(isinstance(item, dict) and item.get("count", 0) > 0 for item in (failure_memory or {}).values()):
            return "mature"
        return "micro_policy"

    def _slug(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
        return slug or "item"

    def _titleize_identifier(self, value: str) -> str:
        words = [word for word in re.split(r"[_-]+", value or "") if word]
        return " ".join(word.capitalize() for word in words) or "Runtime Learned Question Class"

    def _ensure_unique_class_id(self, base_id: str, existing_ids: Any) -> str:
        if isinstance(existing_ids, dict):
            taken = set(existing_ids.keys())
        else:
            taken = set(existing_ids or [])
        candidate = base_id
        suffix = 2
        while candidate in taken:
            candidate = f"{base_id}_{suffix}"
            suffix += 1
        return candidate

    def _build_runtime_class_identity(
        self,
        question: str,
        tokens: List[str],
        info: Optional[Dict[str, Any]] = None,
        reserved_ids: Optional[Any] = None,
    ) -> Dict[str, str]:
        naming_stopwords = {
            "based", "these", "uniformly", "sampled", "frames", "frame", "video", "answer",
            "select", "following", "choices", "choice", "shown", "show", "image", "images",
            "room", "rooms", "object", "objects", "please", "estimate", "combined", "space",
        }
        info = info or {}
        example_questions = [self.normalize_question(item) for item in info.get("example_questions", []) if item]
        keywords = [str(item).lower().strip() for item in info.get("keywords", []) if str(item).strip()]
        combined_text = " ".join(
            part for part in [self.normalize_question(question), *example_questions, " ".join(keywords)] if part
        ).lower()

        keyword_pool: List[str] = []
        for token in list(tokens) + keywords:
            tok = re.sub(r"[^a-z0-9]+", "", str(token).lower())
            if not tok or tok in self._stopwords or tok in naming_stopwords:
                continue
            if tok not in keyword_pool:
                keyword_pool.append(tok)

        rules = [
            (
                "navigation_action_sequence",
                "Navigation Action Sequence",
                "Auto-created class for navigation-style action sequences.",
                ["navigate to", "go forward until", "turn left", "turn right", "turn back"],
            ),
            (
                "room_size_estimation",
                "Room Size Estimation",
                "Auto-created class for estimating room size or floor area.",
                ["square meters", "square meter", "size of this room", "combined space"],
            ),
            (
                "object_dimension_estimation",
                "Object Dimension Estimation",
                "Auto-created class for estimating object length, width, height, or longest dimension.",
                ["longest dimension", "width", "height", "length"],
            ),
            (
                "object_count_estimation",
                "Object Count Estimation",
                "Auto-created class for counting target objects.",
                ["how many", "count", "number of"],
            ),
            (
                "object_presence_verification",
                "Object Presence Verification",
                "Auto-created class for verifying whether a target object is present.",
                ["is there", "does the room contain", "can you see"],
            ),
        ]

        best_rule = None
        best_score = 0
        for base_id, title, description, phrases in rules:
            score = sum(combined_text.count(phrase) for phrase in phrases)
            if score > best_score:
                best_rule = (base_id, title, description)
                best_score = score

        if best_rule is not None:
            base_id, name, description = best_rule
        else:
            base_tokens = keyword_pool[:3] or ["scene", "reasoning"]
            if len(base_tokens) == 1:
                base_tokens.append("task")
            base_id = "_".join(base_tokens)
            if not any(base_id.startswith(prefix) for prefix in ("scene_", "room_", "object_", "navigation_", "layout_", "spatial_", "custom_")):
                base_id = f"custom_{base_id}"
            name = self._titleize_identifier(base_id)
            description = "Auto-created class for a previously unseen question pattern."

        class_id = self._ensure_unique_class_id(base_id, reserved_ids or self._question_class_memory().get("classes", {}))
        return {
            "class_id": class_id,
            "name": name if class_id == base_id else self._titleize_identifier(class_id),
            "description": description,
        }

    def _merge_trajectory_node(self, current: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(current)
        merged["total"] = int(current.get("total", 0)) + int(incoming.get("total", 0))
        merged["success"] = int(current.get("success", 0)) + int(incoming.get("success", 0))
        merged["correct"] = int(current.get("correct", 0)) + int(incoming.get("correct", 0))
        merged["reward_sum"] = round(_safe_float(current.get("reward_sum"), 0.0) + _safe_float(incoming.get("reward_sum"), 0.0), 6)
        merged["last_reflection"] = incoming.get("last_reflection") or current.get("last_reflection")
        merged["effective_sequence"] = incoming.get("effective_sequence") or current.get("effective_sequence") or merged.get("sequence", [])
        merged["full_sequence"] = incoming.get("full_sequence") or current.get("full_sequence") or merged.get("sequence", [])
        merged["seed_parent"] = incoming.get("seed_parent") or current.get("seed_parent")
        timestamps = [stamp for stamp in [current.get("last_seen"), incoming.get("last_seen")] if isinstance(stamp, str) and stamp]
        merged["last_seen"] = max(timestamps) if timestamps else None
        return merged

    def _rename_runtime_question_classes(self) -> bool:
        qcm = self._question_class_memory()
        classes = qcm.get("classes", {})
        legacy_ids = [
            class_id
            for class_id, info in classes.items()
            if info.get("source") == "runtime_learning" and class_id.startswith("learned_question_class_")
        ]
        if not legacy_ids:
            return False

        rename_map: Dict[str, Dict[str, str]] = {}
        reserved_ids = {class_id for class_id in classes.keys() if class_id not in legacy_ids}
        for old_id in legacy_ids:
            info = classes.get(old_id, {})
            example_questions = [item for item in info.get("example_questions", []) if item]
            question = next(iter(example_questions), info.get("description") or info.get("name") or old_id)
            tokens: List[str] = []
            for item in example_questions[:10]:
                tokens.extend(self._tokenize_question(item))
            if not tokens:
                tokens = [str(item).lower() for item in info.get("keywords", []) if str(item).strip()]
            identity = self._build_runtime_class_identity(
                question,
                tokens,
                info=info,
                reserved_ids=reserved_ids,
            )
            reserved_ids.add(identity["class_id"])
            rename_map[old_id] = identity

        if not rename_map:
            return False

        new_classes: Dict[str, Dict[str, Any]] = {}
        for class_id, info in classes.items():
            updated = dict(info)
            if class_id in rename_map:
                identity = rename_map[class_id]
                updated["class_id"] = identity["class_id"]
                updated["name"] = identity["name"]
                updated["description"] = identity["description"]
                new_classes[identity["class_id"]] = updated
            else:
                new_classes[class_id] = updated
        qcm["classes"] = new_classes

        new_order: List[str] = []
        seen = set()
        for class_id in qcm.get("class_order", []):
            new_id = rename_map.get(class_id, {}).get("class_id", class_id)
            if new_id not in seen:
                new_order.append(new_id)
                seen.add(new_id)
        for class_id in new_classes.keys():
            if class_id not in seen:
                new_order.append(class_id)
                seen.add(class_id)
        qcm["class_order"] = new_order

        skill_mem = self._skill_memory()
        renamed_trajectories: Dict[str, Dict[str, Any]] = {}
        for node in skill_mem.get("trajectories", {}).values():
            if not isinstance(node, dict):
                continue
            updated = dict(node)
            old_pattern = updated.get("pattern", "comprehensive_scene_understanding")
            new_pattern = rename_map.get(old_pattern, {}).get("class_id", old_pattern)
            updated["pattern"] = new_pattern
            updated["seed_parent"] = updated.get("seed_parent") or self._resolve_seed_parent(new_pattern)
            sequence = updated.get("sequence", [])
            new_key = self._sequence_key(new_pattern, sequence)
            if new_key in renamed_trajectories:
                renamed_trajectories[new_key] = self._merge_trajectory_node(renamed_trajectories[new_key], updated)
            else:
                renamed_trajectories[new_key] = updated
        skill_mem["trajectories"] = renamed_trajectories

        skill_id_map: Dict[str, str] = {}
        existing_skills: Dict[str, Dict[str, Any]] = {}
        for skill in skill_mem.get("skills", []):
            if not isinstance(skill, dict):
                continue
            updated = dict(skill)
            meta = dict(skill.get("meta", {}))
            sequence = meta.get("sequence", []) or self._extract_tools_from_usage(skill.get("tool_usage", ""))
            old_pattern = meta.get("pattern", "comprehensive_scene_understanding")
            new_pattern = rename_map.get(old_pattern, {}).get("class_id", old_pattern)
            meta["pattern"] = new_pattern
            meta["seed_parent"] = meta.get("seed_parent") or self._resolve_seed_parent(new_pattern)
            meta["sequence"] = sequence
            updated["meta"] = meta
            old_id = skill.get("id")
            new_id = self._skill_id(new_pattern, sequence)
            updated["id"] = new_id
            if isinstance(updated.get("name"), str) and updated["name"].startswith("Learned Skill ("):
                updated["name"] = re.sub(r"Learned Skill \([^)]*\)", f"Learned Skill ({new_pattern})", updated["name"], count=1)
            if isinstance(updated.get("when"), str) and updated["when"].startswith("Learned from runtime trajectories for "):
                updated["when"] = f"Learned from runtime trajectories for {new_pattern}"
            if new_id not in existing_skills:
                existing_skills[new_id] = updated
            if old_id:
                skill_id_map[old_id] = new_id
        skill_mem["skills"] = list(existing_skills.values())

        for info in qcm["classes"].values():
            if "preferred_skill_id" in info:
                info.pop("preferred_skill_id", None)

        self._refresh_learned_skills()
        self._memory["global_version"] = int(self._memory.get("global_version", 0)) + 1
        return True

    def _materialize_progressive_disclosure(self) -> None:
        try:
            self._materialize_static_library()
            self._materialize_dynamic_library()
        except Exception as exc:
            logger.warning("Failed to materialize skill reference docs: %s", exc)

    def _prune_generated_dirs(self, root: Path, keep_slugs: List[str]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        keep = set(keep_slugs)
        for child in root.iterdir():
            if child.is_dir() and child.name not in keep:
                shutil.rmtree(child)

    def _get_static_example_questions(self, problem_id: str) -> List[str]:
        examples = {
            "spatial_localization": [
                "Is the lamp to the left of the sofa?",
                "Which way should the robot turn to face the TV?",
                "Is the chair between the table and the couch?",
            ],
            "depth_distance": [
                "Is the bed closer than the desk?",
                "Which object is farther from the camera, the chair or the lamp?",
                "Measuring from the closest point of each object, which is closer to the TV, the chair or the sofa?",
                "Which detected object is nearest to the camera?",
                "What is the 3D distance in meters between these two selected points on the image?",
                "After detecting the trash bin and the recycling bin, what is the metric distance between their nearest visible surfaces?",
            ],
            "multiview_occlusion": [
                "Would the vase become visible from a right-side view?",
                "Is the cabinet blocking the view of the chair?",
            ],
            "detection_counting": [
                "How many chairs are visible in the room?",
                "Count the windows in this scene.",
                "How many trash bins are present?",
            ],
            "segmentation_boundary": [
                "Outline the boundary of the sofa.",
                "Segment the exact mask of the table.",
            ],
            "fine_grained_pointing": [
                "Point to the mug on the desk.",
                "Locate the exact position of the remote control.",
            ],
            "custom_category": [
                "Detect all emergency exits in the image.",
                "Find every object that looks like a storage basket.",
            ],
            "comprehensive_scene_understanding": [
                "Which object should the robot approach first to reach the sink safely?",
                "What evidence supports the final answer to this multi-step scene question?",
            ],
        }
        return examples.get(
            problem_id,
            [
                "Identify the minimum evidence needed for this problem type.",
                "Use the most relevant evidence pattern instead of following a fixed tool order.",
            ],
        )

    def _materialize_static_library(self) -> None:
        keep_problem_slugs: List[str] = []
        keep_skill_slugs: List[str] = []
        for problem_id in self._seed_pattern_order.keys():
            slug = self._slug(problem_id)
            keep_problem_slugs.append(slug)
            keep_skill_slugs.append(slug)
            problem_dir = self._static_problem_dir / slug
            skill_dir = self._static_skill_dir / slug
            problem_dir.mkdir(parents=True, exist_ok=True)
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "references").mkdir(parents=True, exist_ok=True)
            seed_skill = self._get_seed_skill_for_pattern(problem_id)
            (problem_dir / "PROBLEM.md").write_text(
                self._render_static_problem_markdown(problem_id),
                encoding="utf-8",
            )
            if seed_skill:
                (skill_dir / "SKILL.md").write_text(
                    self._render_skill_markdown(problem_id, seed_skill, source="static"),
                    encoding="utf-8",
                )
                self._write_static_skill_references(skill_dir, problem_id, seed_skill)
        self._prune_generated_dirs(self._static_problem_dir, keep_problem_slugs)
        self._prune_generated_dirs(self._static_skill_dir, keep_skill_slugs)

    def _materialize_dynamic_library(self) -> None:
        qcm = self._question_class_memory()
        skill_mem = self._skill_memory()
        self._dynamic_state_dir.mkdir(parents=True, exist_ok=True)
        self._dynamic_problem_state_path.write_text(
            json.dumps(_strip_time_fields_for_output(qcm), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._dynamic_skill_state_path.write_text(
            json.dumps(_strip_time_fields_for_output(skill_mem), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        keep_problem_slugs: List[str] = []
        for problem_id, info in qcm.get("classes", {}).items():
            if not info.get("hits") and not info.get("solution_stats") and info.get("source") == "seed":
                continue
            slug = self._slug(problem_id)
            keep_problem_slugs.append(slug)
            problem_dir = self._dynamic_problem_dir / slug
            problem_dir.mkdir(parents=True, exist_ok=True)
            (problem_dir / "PROBLEM.md").write_text(
                self._render_dynamic_problem_markdown(problem_id, info),
                encoding="utf-8",
            )
        self._prune_generated_dirs(self._dynamic_problem_dir, keep_problem_slugs)

        keep_skill_slugs: List[str] = []
        for skill in skill_mem.get("skills", []):
            slug = self._slug(skill.get("id", "skill"))
            keep_skill_slugs.append(slug)
            skill_dir = self._dynamic_skill_dir / slug
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "references").mkdir(parents=True, exist_ok=True)
            pattern = skill.get("meta", {}).get("pattern", "comprehensive_scene_understanding")
            info = qcm.get("classes", {}).get(pattern, {})
            (skill_dir / "SKILL.md").write_text(
                self._render_skill_markdown(pattern, skill, source="dynamic"),
                encoding="utf-8",
            )
            self._write_dynamic_skill_references(skill_dir, pattern, skill, info)
        self._prune_generated_dirs(self._dynamic_skill_dir, keep_skill_slugs)

    def _render_static_problem_markdown(self, problem_id: str) -> str:
        profile = self._get_static_problem_profile(problem_id)
        required_text = "\n".join(f"- {item}" for item in profile["required_evidence"])
        near_miss_text = "\n".join(f"- {item}" for item in profile["near_misses"])
        example_text = "\n".join(f"- {example}" for example in self._get_static_example_questions(problem_id))
        skill_path = self._static_skill_dir / self._slug(problem_id) / "SKILL.md"
        return (
            f"# Problem Type\n\n"
            f"{problem_id}\n\n"
            f"## Intent\n\n- {profile['intent']}\n\n"
            f"## Required Evidence\n\n{required_text}\n\n"
            f"## Near Misses\n\n{near_miss_text}\n\n"
            f"## Reference Skill\n\n- `{skill_path}`\n\n"
            f"## Example Questions\n\n{example_text}\n\n"
            f"## Notes\n\n- Source: static\n- This file stores stable problem guidance only.\n"
        )

    def _render_dynamic_problem_markdown(self, problem_id: str, info: Dict[str, Any]) -> str:
        examples = info.get("example_questions", [])[-5:]
        sequences = list((info.get("solution_stats") or {}).values())
        seed_parent = self._resolve_seed_parent(problem_id, info=info)
        sequence_lines = []
        for item in sorted(sequences, key=lambda row: str(row.get("last_seen") or ""), reverse=True)[:3]:
            seq = " -> ".join(item.get("sequence", [])) or "(none)"
            sequence_lines.append(f"- `{seq}` (success={item.get('success', 0)}/{item.get('total', 0)})")
        if not sequence_lines:
            sequence_lines.append("- No runtime sequences recorded yet")
        example_lines = [f"- {example}" for example in examples] or ["- No representative examples recorded yet"]
        keyword_lines = [f"- {keyword}" for keyword in info.get("keywords", [])[:8]] or ["- No observed keywords yet"]
        static_path = self._static_problem_dir / self._slug(seed_parent) / "PROBLEM.md"
        keyword_text = "\n".join(keyword_lines)
        sequence_text = "\n".join(sequence_lines)
        example_text = "\n".join(example_lines)
        return (
            f"# Problem Type\n\n"
            f"{info.get('name', problem_id)}\n\n"
            f"## Runtime Summary\n\n"
            f"- Source: dynamic\n"
            f"- Hits: {int(info.get('hits', 0))}\n"
            f"- Seed archetype: `{seed_parent}`\n"
            f"- Static profile: `{static_path}`\n\n"
            f"## Observed Keywords\n\n{keyword_text}\n\n"
            f"## Representative Sequences\n\n{sequence_text}\n\n"
            f"## Example Questions\n\n{example_text}\n\n"
            f"## Notes\n\n- This file stores runtime evidence and examples only.\n"
        )

    def _render_skill_markdown(self, problem_id: str, skill: Dict[str, Any], source: str) -> str:
        profile_problem_id = problem_id if source == "static" else self._resolve_seed_parent(problem_id)
        profile = self._get_static_problem_profile(profile_problem_id)
        tool_usage = skill.get("tool_usage", "")
        tool_order = self._extract_tools_from_usage(tool_usage)
        strategy = skill.get("strategy") or profile["default_plan"]
        plan_title = "Reference Plan" if source == "static" else "Observed Plan"
        tool_title = "Common Tools" if source == "static" else "Observed Tools"
        tool_text = "\n".join(f"- `{tool}`" for tool in tool_order) if tool_order else "- Use the best available tool sequence for this problem."
        runtime_section = ""
        if source == "dynamic":
            meta = skill.get("meta", {})
            runtime_section = (
                f"# Runtime Signals\n\n"
                f"- Pattern: `{meta.get('pattern', problem_id)}`\n"
                f"- Seed archetype: `{meta.get('seed_parent', profile_problem_id)}`\n"
                f"- Support: {int(meta.get('success', 0))}/{int(meta.get('total', 0))}\n"
                f"- Correct rate: {_safe_float(meta.get('correct_rate', 0.0), 0.0):.3f}\n"
                f"- Exploratory gap: {int(meta.get('fallback_gap', 0))}\n\n"
            )
        plan_text = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(strategy))
        return (
            f"---\n"
            f"name: {self._slug(skill.get('name', problem_id))}\n"
            f"description: {skill.get('when', profile['intent'])}\n"
            f"---\n\n"
            f"# Trigger\n\n- {skill.get('when', profile['intent'])}\n\n"
            f"# {plan_title}\n\n{plan_text}\n\n"
            f"# {tool_title}\n\n{tool_text}\n\n"
            f"{runtime_section}"
            f"# Read More\n\n"
            f"- `references/guide.md`\n"
            f"- `references/examples.md`\n"
            f"- `references/tool-hints.md`\n"
            f"- `references/failure-recovery.md`\n\n"
            f"# Notes\n\n- Source: {source}\n"
        )

    def _write_static_skill_references(
        self,
        skill_dir: Path,
        problem_id: str,
        skill: Dict[str, Any],
    ) -> None:
        profile = self._get_static_problem_profile(problem_id)
        refs_dir = skill_dir / "references"
        refs_dir.mkdir(parents=True, exist_ok=True)
        sequence = self._extract_tools_from_usage(skill.get("tool_usage", ""))
        examples = self._get_static_example_questions(problem_id)
        guide_text = (
            "# Purpose\n\n"
            f"- {profile['intent']}\n\n"
            "# When to use\n\n"
            f"- {skill.get('when', profile['intent'])}\n\n"
            "# How to apply\n\n"
            + "\n".join(f"- {step}" for step in (skill.get("strategy") or profile["default_plan"]))
            + "\n\n# Constraints\n\n"
            "- Prefer the minimum sufficient evidence.\n"
            "- Treat this skill as a reference, not a mandatory sequence.\n"
            "- Add tools only when they can change the answer.\n\n"
            "# Notes\n\n- Source: static\n- This guide is stable and should not include runtime examples.\n"
        )
        (refs_dir / "guide.md").write_text(guide_text, encoding="utf-8")
        sequence_text = "- `" + " -> ".join(sequence) + "`" if sequence else "- No common tool set recorded"
        examples_text = (
            "# Example Questions\n\n"
            + "\n".join(f"- {example}" for example in examples)
            + "\n\n# Common Tools\n\n"
            + sequence_text
            + "\n\n# Notes\n\n- Runtime examples are stored under the matching dynamic problem type.\n"
        )
        (refs_dir / "examples.md").write_text(examples_text, encoding="utf-8")
        hint_lines = [
            "- Choose the tool mix that best matches the current evidence need.",
            "- Combine tools when they provide complementary signals.",
            "- Avoid repeating the same tool call if it will not change the answer.",
        ]
        if "depth_estimation_tool" in sequence:
            hint_lines.extend(
                [
                    "- DA3 is not only for a pretty depth map. Use it as direct metric evidence for depth, near/far ranking, and distance-in-meters questions.",
                    "- If the question compares named objects, detect them first and pass crop_box or crop_boxes into depth_estimation_tool so it can return region_depth_stats, nearest_region, farthest_region, and depth_ranking_near_to_far.",
                    "- If the question asks for an exact metric measurement between selected image locations, call depth_estimation_tool with point_coords and point_pairs; then read point_measurements and point_pair_distances.",
                    "- Interpret camera_axis_depth as optical-axis depth Z. Interpret camera_distance as Euclidean range from the camera center. For actual metric distance, prefer camera_distance or point_pair_distances over raw Z depth alone.",
                ]
            )
        if "pi3_tool" in sequence:
            hint_lines.append("- If Pi3 is needed, avoid azimuth=0 and elevation=0 because the input image already provides that view.")
        (refs_dir / "tool-hints.md").write_text("# Tool Hints\n\n" + "\n".join(hint_lines) + "\n", encoding="utf-8")
        failure_text = (
            "# Failure Recovery\n\n"
            "- If a tool fails, switch to another tool that can provide similar evidence.\n"
            "- If two consecutive tools fail, fall back to direct reasoning and answer with explicit uncertainty.\n"
            "- If the current question does not match this static skill, reroute instead of forcing this pattern.\n"
        )
        (refs_dir / "failure-recovery.md").write_text(failure_text, encoding="utf-8")

    def _write_dynamic_skill_references(
        self,
        skill_dir: Path,
        problem_id: str,
        skill: Dict[str, Any],
        info: Dict[str, Any],
    ) -> None:
        refs_dir = skill_dir / "references"
        refs_dir.mkdir(parents=True, exist_ok=True)
        meta = skill.get("meta", {})
        sequence = meta.get("sequence") or self._extract_tools_from_usage(skill.get("tool_usage", ""))
        examples = info.get("example_questions", [])[-5:]
        seed_parent = str(meta.get("seed_parent") or self._resolve_seed_parent(problem_id, info=info)).strip()
        static_skill_path = self._static_skill_dir / self._slug(seed_parent) / "SKILL.md"
        guide_text = (
            "# Runtime Summary\n\n"
            f"- Pattern: `{meta.get('pattern', problem_id)}`\n"
            f"- Seed archetype: `{seed_parent}`\n"
            f"- Total attempts: {int(meta.get('total', 0))}\n"
            f"- Successful attempts: {int(meta.get('success', 0))}\n"
            f"- Exploratory gap: {int(meta.get('fallback_gap', 0))}\n"
            f"- Static fallback: `{static_skill_path}`\n\n"
            "# Notes\n\n- Source: dynamic\n- This guide stores observed runtime behavior only.\n"
        )
        (refs_dir / "guide.md").write_text(guide_text, encoding="utf-8")
        sequence_text = "- `" + " -> ".join(sequence) + "`" if sequence else "- No tool sequence recorded yet"
        full_sequence = meta.get("full_sequence") if isinstance(meta.get("full_sequence"), list) else sequence
        full_sequence_text = ""
        if isinstance(full_sequence, list) and full_sequence and full_sequence != sequence:
            full_sequence_text = "\n\n# Expanded Trajectory\n\n- `" + " -> ".join(full_sequence) + "`\n"
        examples_text = (
            "# Observed Questions\n\n"
            + ("\n".join(f"- {example}" for example in examples) if examples else "- No representative examples recorded yet")
            + "\n\n# Observed Sequence\n\n"
            + sequence_text
            + "\n"
            + full_sequence_text
        )
        (refs_dir / "examples.md").write_text(examples_text, encoding="utf-8")
        hint_lines = [
            "- Treat the observed sequence as a reference, not a mandatory order.",
            "- If the current question diverges from the observed pattern, adapt the tool choice freely.",
            "- Avoid repeating low-yield tool calls once the answer becomes stable.",
        ]
        if "pi3_tool" in sequence:
            hint_lines.append("- If Pi3 is used, avoid azimuth=0 and elevation=0 because the input image already provides that view.")
        (refs_dir / "tool-hints.md").write_text("# Tool Hints\n\n" + "\n".join(hint_lines) + "\n", encoding="utf-8")
        failure_text = (
            "# Failure Recovery\n\n"
            "- If the learned sequence stops improving the answer, switch to another evidence pattern instead of forcing it.\n"
            "- If the current question no longer matches the observed examples, reroute from the problem card before using more tools.\n"
            "- If runtime evidence is too sparse, answer with explicit uncertainty instead of overfitting to the learned sequence.\n"
        )
        (refs_dir / "failure-recovery.md").write_text(failure_text, encoding="utf-8")

    def record_episode(
        self,
        question: str,
        tool_calls: List[Dict[str, Any]],
        tool_results: Dict[str, Any],
        success: bool,
        reward_score: Optional[float] = None,
        is_correct: Optional[bool] = None,
        reflection: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
    ) -> None:
        """
        Update Question Class Memory and Skill Memory from one completed episode.

        reward_score: numeric feedback in [0,1] if available.
        is_correct: evaluation correctness signal if available.
        reflection: optional model-generated skill suggestion JSON.
        """
        if not self.update_enabled:
            return

        full_sequence = self._extract_full_sequence(tool_calls)
        sequence = self._extract_effective_sequence(tool_calls, tool_results)
        if not sequence:
            sequence = list(dict.fromkeys(full_sequence))

        normalized_question = self.normalize_question(question)
        question_class = self._ensure_task_question_class(task_type, normalized_question)
        if not question_class:
            question_class, _ = self._classify_question(
                normalized_question,
                create_if_missing=True,
                touch=False,
            )
        if not question_class:
            return

        reward = self._normalize_reward(reward_score, success, is_correct)
        seed_parent = self._resolve_seed_parent(question_class, question=normalized_question)
        blocker_type = self._infer_blocker_type(seed_parent, normalized_question, sequence)
        failure_events = self._extract_failure_events(tool_results, seed_parent, normalized_question, sequence)
        skill_mem = self._skill_memory()
        episode_key = self._sequence_key(question_class, sequence or ["no_tool"])

        management_decision = self._build_skill_management_decision(
            question=normalized_question,
            question_class=question_class,
            seed_parent=seed_parent,
            sequence=sequence,
            full_sequence=full_sequence,
            tool_results=tool_results,
            success=success,
            reward=reward,
            is_correct=is_correct,
            reflection=reflection,
        )
        self._record_skill_management_decision(management_decision)

        node: Optional[Dict[str, Any]] = None
        should_track_trajectory = bool(sequence) or bool(success)
        if should_track_trajectory:
            key = self._sequence_key(question_class, sequence)
            trajectories = skill_mem.setdefault("trajectories", {})
            node = trajectories.setdefault(
                key,
                {
                    "pattern": question_class,
                    "seed_parent": seed_parent,
                    "sequence": sequence,
                    "effective_sequence": sequence,
                    "full_sequence": full_sequence,
                    "total": 0,
                    "success": 0,
                    "correct": 0,
                    "reward_sum": 0.0,
                    "last_seen": None,
                    "last_reflection": None,
                    "failure_memory": {},
                    "blocker_type": blocker_type,
                    "scene_task_context": management_decision.get("scene_task_context", {}),
                    "required_evidence": management_decision.get("required_evidence", []),
                    "answer_pattern": management_decision.get("answer_pattern", ""),
                    "coverage": {},
                },
            )
            node["total"] += 1
            node["success"] += int(bool(success))
            if is_correct is not None:
                node["correct"] += int(bool(is_correct))
            node["reward_sum"] = _safe_float(node.get("reward_sum"), 0.0) + reward
            node["last_seen"] = datetime.now().isoformat()
            node["sequence"] = sequence
            node["seed_parent"] = seed_parent
            node["effective_sequence"] = sequence
            node["full_sequence"] = full_sequence
            node["blocker_type"] = blocker_type
            node["scene_task_context"] = management_decision.get("scene_task_context", {})
            node["required_evidence"] = management_decision.get("required_evidence", [])
            node["answer_pattern"] = management_decision.get("answer_pattern", "")
            node["coverage"] = self._merge_skill_coverage(
                node.get("coverage", {}),
                management_decision.get("coverage_delta", {}),
            )
            if reflection:
                node["last_reflection"] = reflection
            stored_failures = node.setdefault("failure_memory", {})
            for failure_key, payload in failure_events.items():
                existing = stored_failures.setdefault(
                    failure_key,
                    {
                        "tool_name": payload.get("tool_name"),
                        "failure_type": payload.get("failure_type"),
                        "count": 0,
                        "last_error": "",
                        "next_best_tool": payload.get("next_best_tool"),
                        "last_seen": None,
                    },
                )
                existing["count"] = int(existing.get("count", 0)) + int(payload.get("count", 0))
                existing["last_error"] = payload.get("last_error") or existing.get("last_error") or ""
                existing["next_best_tool"] = payload.get("next_best_tool") or existing.get("next_best_tool")
                existing["last_seen"] = payload.get("last_seen") or datetime.now().isoformat()

        self._update_question_class_memory(question_class, normalized_question, sequence, success)
        episode = self.close_episode(
            session_id=f"session_{hashlib.sha1((normalized_question + episode_key).encode('utf-8')).hexdigest()[:12]}",
            question=normalized_question,
            tool_calls=tool_calls,
            tool_results=tool_results,
            final_answer=None,
            success=success,
            reward_score=reward_score,
            is_correct=is_correct,
            question_class=question_class,
            task_type=task_type,
        )
        self._update_evidence_memory(
            question=normalized_question,
            question_class=question_class,
            tool_results=tool_results,
            episode=episode,
        )
        promoted_candidate = self.promote_skill_candidate(episode)
        promotion_allowed, promotion_gate = self._promotion_gate(episode)
        if promoted_candidate is not None:
            promoted_candidate["promotion_gate"] = promotion_gate
            if episode is not None:
                episode["skill_candidate"] = promoted_candidate
        if should_track_trajectory:
            self._refresh_learned_skills()
        if episode is not None:
            episode["promotion_allowed"] = promotion_allowed
            episode["promotion_gate"] = promotion_gate
            episode["skill_management_decision"] = management_decision
        self._save_memory()

    def _update_evidence_memory(
        self,
        question: str,
        question_class: str,
        tool_results: Dict[str, Any],
        episode: Optional[Dict[str, Any]] = None,
    ) -> None:
        facts = self._evidence_memory().setdefault("facts", [])
        labels = self._extract_object_labels(tool_results)
        relations = self._extract_relation_hints(question)
        now = datetime.now().isoformat()
        for label in labels[:8]:
            fact_id = hashlib.sha1(f"{question_class}:{label}".encode("utf-8")).hexdigest()[:12]
            facts.append(
                {
                    "fact_id": f"fact_{fact_id}",
                    "entity": label,
                    "relation": ", ".join(relations) if relations else "observed_in_episode",
                    "summary": f"{label} appeared in a {question_class} episode.",
                    "confidence": 0.6,
                    "updated_at": now,
                    "contradictions": [],
                    "provenance": {
                        "episode_id": episode.get("episode_id") if isinstance(episode, dict) else None,
                        "question_class": question_class,
                    },
                }
            )
        if len(facts) > 4000:
            del facts[:-4000]

    def _question_class_memory(self) -> Dict[str, Any]:
        qcm = self._memory.setdefault("question_class_memory", {})
        qcm.setdefault("classes", {})
        qcm.setdefault("class_order", [])
        qcm.setdefault("global_version", 0)
        return qcm

    def _skill_memory(self) -> Dict[str, Any]:
        sm = self._memory.setdefault("skill_memory", {})
        sm.setdefault("trajectories", {})
        sm.setdefault("skills", [])
        sm.setdefault(
            "skill_management",
            {
                "schema_version": 1,
                "decisions": [],
                "failure_lessons": {},
                "success_workflows": {},
            },
        )
        return sm

    def _normalize_tool_sequence(self, sequence: Any) -> List[str]:
        return [str(tool).strip() for tool in (sequence or []) if str(tool).strip()]

    def _strip_legacy_skill_meta(self, meta: Dict[str, Any]) -> bool:
        removed = False
        for key in [
            "score",
            "base_score",
            "combo_bonus",
            "combo_patterns",
            "pi3_penalty",
            "pi3_penalty_reasons",
            "is_champion",
            "policy_action",
            "policy_note",
            "status",
        ]:
            if key in meta:
                meta.pop(key, None)
                removed = True
        return removed

    def _sanitize_loaded_skill_memory(self) -> bool:
        changed = False
        skill_mem = self._skill_memory()
        for key in ["pattern_champions", "pattern_policy"]:
            if key in skill_mem:
                skill_mem.pop(key, None)
                changed = True

        classes = self._question_class_memory().get("classes", {})
        for class_id, class_info in classes.items():
            if not isinstance(class_info, dict):
                continue
            desired_seed_parent = class_id if self._is_seed_problem_id(class_id) else self._resolve_seed_parent(class_id, info=class_info)
            if class_info.get("seed_parent") != desired_seed_parent:
                class_info["seed_parent"] = desired_seed_parent
                changed = True

        cleaned_trajectories: Dict[str, Dict[str, Any]] = {}
        for key, node in skill_mem.get("trajectories", {}).items():
            if not isinstance(node, dict):
                changed = True
                continue
            updated = dict(node)
            pattern = str(updated.get("pattern", "comprehensive_scene_understanding")).strip() or "comprehensive_scene_understanding"
            updated["pattern"] = pattern
            desired_seed_parent = self._resolve_seed_parent(pattern)
            if updated.get("seed_parent") != desired_seed_parent:
                updated["seed_parent"] = desired_seed_parent
                changed = True
            cleaned_trajectories[str(key)] = updated
        if skill_mem.get("trajectories", {}) != cleaned_trajectories:
            skill_mem["trajectories"] = cleaned_trajectories
            changed = True

        cleaned_skills: List[Dict[str, Any]] = []
        seen_ids = set()
        for skill in skill_mem.get("skills", []):
            if not isinstance(skill, dict):
                changed = True
                continue

            meta = skill.setdefault("meta", {})
            changed = self._strip_legacy_skill_meta(meta) or changed
            pattern = str(meta.get("pattern", "comprehensive_scene_understanding")).strip() or "comprehensive_scene_understanding"
            sequence = self._normalize_tool_sequence(meta.get("sequence"))
            if not sequence:
                sequence = self._extract_tools_from_usage(skill.get("tool_usage", ""))
            effective_sequence = self._normalize_tool_sequence(meta.get("effective_sequence") or sequence)
            full_sequence = self._normalize_tool_sequence(meta.get("full_sequence") or effective_sequence or sequence)
            seed_parent = self._resolve_seed_parent(pattern)
            blocker_type = str(meta.get("blocker_type") or self._infer_blocker_type(seed_parent, "", effective_sequence or sequence)).strip()
            failure_memory = meta.get("failure_memory", {}) if isinstance(meta.get("failure_memory"), dict) else {}
            maturity = str(meta.get("maturity") or self._skill_maturity(effective_sequence or sequence, int(meta.get("total", 0) or 0), failure_memory)).strip()

            meta["pattern"] = pattern
            meta["sequence"] = sequence
            meta["effective_sequence"] = effective_sequence
            meta["full_sequence"] = full_sequence
            meta["seed_parent"] = seed_parent
            meta["fallback_gap"] = max(0, len(full_sequence) - len(effective_sequence))
            meta["blocker_type"] = blocker_type
            meta["failure_memory"] = failure_memory
            meta["maturity"] = maturity

            skill_id = str(skill.get("id") or "").strip()
            if not skill_id:
                if not sequence:
                    changed = True
                    continue
                skill_id = self._skill_id(pattern, sequence)
                skill["id"] = skill_id
                changed = True

            if skill_id in seen_ids:
                changed = True
                continue
            seen_ids.add(skill_id)
            cleaned_skills.append(skill)

        if skill_mem.get("skills", []) != cleaned_skills:
            skill_mem["skills"] = cleaned_skills
            changed = True

        for class_info in classes.values():
            if isinstance(class_info, dict) and "preferred_skill_id" in class_info:
                class_info.pop("preferred_skill_id", None)
                changed = True

        if changed:
            self._memory["global_version"] = int(self._memory.get("global_version", 0)) + 1
        return changed

    def _sync_seed_question_classes(self) -> bool:
        changed = False
        qcm = self._question_class_memory()
        classes = qcm.setdefault("classes", {})
        class_order = qcm.setdefault("class_order", [])
        now = datetime.now().isoformat()

        for class_id, keywords in self._seed_class_keywords.items():
            desired_keywords = list(dict.fromkeys(str(item).strip().lower() for item in keywords if str(item).strip()))
            entry = classes.get(class_id)
            if not isinstance(entry, dict):
                classes[class_id] = {
                    "class_id": class_id,
                    "name": class_id,
                    "description": f"Seed question class for {class_id}",
                    "seed_parent": class_id,
                    "keywords": desired_keywords,
                    "example_questions": [],
                    "hits": 0,
                    "first_seen": now,
                    "last_seen": now,
                    "source": "seed",
                    "solution_stats": {},
                }
                changed = True
                continue

            if entry.get("keywords") != desired_keywords:
                entry["keywords"] = desired_keywords
                changed = True
            if entry.get("source") != "seed":
                entry["source"] = "seed"
                changed = True
            if entry.get("class_id") != class_id:
                entry["class_id"] = class_id
                changed = True
            if entry.get("seed_parent") != class_id:
                entry["seed_parent"] = class_id
                changed = True
            if "solution_stats" not in entry or not isinstance(entry.get("solution_stats"), dict):
                entry["solution_stats"] = {}
                changed = True
            if "example_questions" not in entry or not isinstance(entry.get("example_questions"), list):
                entry["example_questions"] = []
                changed = True
            if not entry.get("first_seen"):
                entry["first_seen"] = now
                changed = True
            if not entry.get("last_seen"):
                entry["last_seen"] = now
                changed = True

        desired_seed_order = list(self._seed_pattern_order.keys())
        trailing = [class_id for class_id in class_order if class_id not in desired_seed_order]
        next_order = desired_seed_order + trailing
        if class_order != next_order:
            qcm["class_order"] = next_order
            changed = True

        if changed:
            qcm["global_version"] = int(qcm.get("global_version", 0)) + 1
            self._memory["global_version"] = int(self._memory.get("global_version", 0)) + 1
        return changed

    def _extract_tools_from_usage(self, usage: str) -> List[str]:
        if not usage:
            return []
        normalized = usage.replace("→", "->")
        tokens = [
            t.strip() for t in normalized.replace("/", " ").replace("(", " ").replace(")", " ").replace(",", " ").split("->")
        ]
        flat: List[str] = []
        for token in tokens:
            for word in token.split():
                if word in self._known_tools:
                    flat.append(word)
        deduped = []
        seen = set()
        for tool in flat:
            if tool not in seen:
                deduped.append(tool)
                seen.add(tool)
        return deduped

    def _get_seed_skill_for_pattern(self, pattern: str) -> Optional[Dict[str, Any]]:
        idx = self._seed_pattern_order.get(pattern)
        if idx is None:
            idx = self._seed_pattern_order.get("comprehensive_scene_understanding")
        if idx is None:
            return None
        if idx < 0 or idx >= len(self._seed_skills):
            return None
        return self._seed_skills[idx]

    def _extract_full_sequence(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[str]:
        sequence: List[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "").strip()
            if name:
                sequence.append(name)
        return sequence

    def _extract_effective_sequence(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: Dict[str, Any],
    ) -> List[str]:
        success_by_name = defaultdict(bool)
        for result_key, result in tool_results.items():
            if isinstance(result, dict) and result.get("success"):
                name = result_key.split("_iter")[0]
                success_by_name[name] = True

        ordered_names: List[str] = []
        seen = set()
        for call in tool_calls:
            name = call.get("name")
            if not name or name in seen:
                continue
            if success_by_name and not success_by_name.get(name, False):
                continue
            ordered_names.append(name)
            seen.add(name)
        return ordered_names

    def _extract_sequence(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: Dict[str, Any],
    ) -> List[str]:
        return self._extract_effective_sequence(tool_calls, tool_results)

    def _tokenize_question(self, question: str) -> List[str]:
        normalized_question = self.normalize_question(question)
        raw = re.findall(r"[a-zA-Z]{3,}", normalized_question.lower())
        return [tok for tok in raw if tok not in self._stopwords]

    def _classify_question(
        self,
        question: str,
        create_if_missing: bool = True,
        touch: bool = True,
    ) -> Tuple[Optional[str], bool]:
        """
        Classify question into known class; create a new class when unknown.

        Returns:
            (question_class_id, changed_memory)
        """
        now = datetime.now().isoformat()
        qcm = self._question_class_memory()
        classes = qcm.setdefault("classes", {})

        normalized_question = self.normalize_question(question)
        text = normalized_question.lower().strip()
        tokens = self._tokenize_question(normalized_question)

        best_class = None
        best_score = -1.0

        for class_id, info in classes.items():
            keywords = info.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []

            hit_score = 0.0
            for kw in keywords:
                kw = str(kw).lower().strip()
                if not kw:
                    continue
                if kw in text:
                    hit_score += 1.0 if " " in kw else 0.6

            if tokens and keywords:
                kw_set = set(str(k).lower() for k in keywords)
                overlap = len([t for t in tokens if t in kw_set])
                hit_score += 0.4 * overlap

            if hit_score > best_score:
                best_score = hit_score
                best_class = class_id

        changed = False
        is_known = best_class is not None and best_score >= 1.0

        if is_known:
            if touch:
                entry = classes[best_class]
                entry["hits"] = int(entry.get("hits", 0)) + 1
                entry["last_seen"] = now
                examples = entry.setdefault("example_questions", [])
                if normalized_question and normalized_question not in examples:
                    examples.append(normalized_question)
                    entry["example_questions"] = examples[-MAX_EXAMPLE_QUESTIONS_PER_CLASS:]
                changed = True
            return best_class, changed

        if not create_if_missing:
            return None, changed

        new_class = self._create_new_question_class(
            normalized_question,
            tokens,
            initial_hits=1 if touch else 0,
            example_question=normalized_question if touch else None,
        )
        classes[new_class["class_id"]] = new_class
        qcm.setdefault("class_order", []).append(new_class["class_id"])
        qcm["global_version"] = int(qcm.get("global_version", 0)) + 1
        changed = True
        return new_class["class_id"], changed

    def _create_new_question_class(
        self,
        question: str,
        tokens: List[str],
        initial_hits: int = 1,
        example_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        identity = self._build_runtime_class_identity(question, tokens)

        keyword_candidates = []
        for token in tokens:
            if token not in keyword_candidates:
                keyword_candidates.append(token)
            if len(keyword_candidates) >= 8:
                break
        if not keyword_candidates:
            keyword_candidates = ["complex", "scene", "reasoning"]

        return {
            "class_id": identity["class_id"],
            "name": identity["name"],
            "description": identity["description"],
            "seed_parent": self._classify_seed_pattern(question),
            "keywords": keyword_candidates,
            "example_questions": [example_question] if example_question else [],
            "hits": max(0, int(initial_hits)),
            "first_seen": now,
            "last_seen": now,
            "source": "runtime_learning",
            "solution_stats": {},
        }

    def _update_question_class_memory(self, question_class: str, question: str, sequence: List[str], success: bool) -> None:
        qcm = self._question_class_memory()
        classes = qcm.setdefault("classes", {})
        entry = classes.get(question_class)
        if entry is None:
            entry = self._create_new_question_class(question, self._tokenize_question(question))
            entry["class_id"] = question_class
            classes[question_class] = entry
        desired_seed_parent = question_class if self._is_seed_problem_id(question_class) else self._resolve_seed_parent(question_class, question=question, info=entry)
        if entry.get("seed_parent") != desired_seed_parent:
            entry["seed_parent"] = desired_seed_parent

        now = datetime.now().isoformat()
        entry["hits"] = int(entry.get("hits", 0)) + 1
        entry["last_seen"] = now

        examples = entry.setdefault("example_questions", [])
        if question and question not in examples:
            examples.append(question)
            entry["example_questions"] = examples[-MAX_EXAMPLE_QUESTIONS_PER_CLASS:]

        sequence_key = "->".join(sequence)
        stats = entry.setdefault("solution_stats", {}).setdefault(
            sequence_key,
            {"sequence": sequence, "total": 0, "success": 0, "last_seen": None},
        )
        stats["total"] += 1
        stats["success"] += int(bool(success))
        stats["last_seen"] = now

        if success:
            token_updates = self._tokenize_question(question)
            kws = [str(k).lower() for k in entry.get("keywords", []) if str(k).strip()]
            for tok in token_updates:
                if tok not in kws:
                    kws.append(tok)
                if len(kws) >= 12:
                    break
            entry["keywords"] = kws

    def _sequence_key(self, pattern: str, sequence: List[str]) -> str:
        return pattern + "::" + "->".join(sequence)

    def _skill_id(self, pattern: str, sequence: List[str]) -> str:
        token = pattern + "::" + "->".join(sequence)
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:10]
        return f"{pattern}_{digest}"

    def _normalize_reward(
        self,
        reward_score: Optional[float],
        success: bool,
        is_correct: Optional[bool],
    ) -> float:
        if reward_score is not None:
            val = max(0.0, min(1.0, _safe_float(reward_score, 0.0)))
            return val
        if is_correct is not None:
            return 1.0 if is_correct else 0.0
        return 1.0 if success else 0.0

    def _refresh_learned_skills(self) -> None:
        skill_mem = self._skill_memory()
        trajectories = skill_mem.get("trajectories", {})

        existing = {s.get("id"): s for s in skill_mem.get("skills", []) if isinstance(s, dict)}
        refreshed_by_id: Dict[str, Dict[str, Any]] = {}
        trajectory_order: List[str] = []

        for item in trajectories.values():
            if not isinstance(item, dict):
                continue

            pattern = str(item.get("pattern", "unknown")).strip() or "unknown"
            seed_parent = str(item.get("seed_parent") or self._resolve_seed_parent(pattern)).strip() or "comprehensive_scene_understanding"
            effective_sequence = self._normalize_tool_sequence(item.get("effective_sequence") or item.get("sequence", []))
            full_sequence = self._normalize_tool_sequence(item.get("full_sequence") or effective_sequence)
            total = int(item.get("total", 0))
            failure_memory = item.get("failure_memory", {}) if isinstance(item.get("failure_memory"), dict) else {}
            blocker_type = str(item.get("blocker_type") or self._infer_blocker_type(seed_parent, "", effective_sequence)).strip()
            maturity = self._skill_maturity(effective_sequence, total, failure_memory)
            preconditions = self._default_preconditions(seed_parent, blocker_type, effective_sequence)
            skip_conditions = self._default_skip_conditions(seed_parent, blocker_type, effective_sequence)
            fallbacks = self._default_fallbacks(seed_parent, "", effective_sequence, failure_memory)
            stop_conditions = self._default_stop_conditions(seed_parent)
            scene_task_context = item.get("scene_task_context", {}) if isinstance(item.get("scene_task_context"), dict) else {}
            required_evidence = item.get("required_evidence", []) if isinstance(item.get("required_evidence"), list) else []
            coverage = item.get("coverage", {}) if isinstance(item.get("coverage"), dict) else {}
            answer_pattern = str(item.get("answer_pattern") or self._answer_pattern_for_workflow(seed_parent, effective_sequence)).strip()
            management = skill_mem.get("skill_management", {}) if isinstance(skill_mem.get("skill_management"), dict) else {}
            lesson_key = self._sequence_key(seed_parent, effective_sequence)
            pattern_lesson_key = self._sequence_key(pattern, effective_sequence)
            failure_lessons = {}
            if isinstance(management.get("failure_lessons"), dict):
                failure_lessons = (
                    management["failure_lessons"].get(lesson_key)
                    or management["failure_lessons"].get(pattern_lesson_key)
                    or {}
                )
            view_policy = {
                "require_need_view_change": blocker_type == "viewpoint_change" or "pi3_tool" in effective_sequence,
                "disallow_origin_view": "pi3_tool" in effective_sequence,
                "prefer_coarse_angles_first": "pi3_tool" in effective_sequence,
            }

            reflection = item.get("last_reflection") if isinstance(item.get("last_reflection"), dict) else None
            skill_id = self._skill_id(pattern, effective_sequence)
            trajectory_order.append(skill_id)

            prev = existing.get(skill_id, {})
            prev_meta = prev.get("meta", {}) if isinstance(prev, dict) else {}
            prev_name = prev.get("name")
            prev_when = prev.get("when")
            prev_strategy = prev.get("strategy")
            prev_is_tuned = (
                isinstance(prev_name, str)
                and prev_name.startswith("Tuned Skill (")
            ) or (
                isinstance(prev_when, str)
                and prev_when.startswith("Auto-tuned for pattern ")
            )
            if prev_is_tuned:
                prev_name = None
                prev_when = None
                prev_strategy = None
            reflection_name = reflection.get("name") if reflection else None
            reflection_when = reflection.get("when") if reflection else None
            reflection_strategy = reflection.get("strategy") if reflection else None
            reflection_usage = reflection.get("tool_usage") if reflection else None

            if effective_sequence:
                default_name = f"Learned Skill ({pattern}): {' -> '.join(effective_sequence)}"
                default_when = f"Learned from runtime trajectories for {pattern}"
                default_strategy = [f"Use {tool} when this pattern appears" for tool in effective_sequence]
                default_usage = " -> ".join(effective_sequence)
            else:
                default_name = f"Learned Direct Skill ({pattern})"
                default_when = f"Use for {pattern} questions when the observed views already contain enough evidence."
                default_strategy = [
                    "Answer directly from the current visual evidence when the relation/count/order is clear.",
                    "Do not force an external tool call only to satisfy a template.",
                    "Escalate to the seed or dynamic tool skill only if the current evidence is ambiguous.",
                ]
                default_usage = "no_tool"

            text_changed = any(
                bool(x)
                for x in [
                    reflection_name and reflection_name != prev.get("name"),
                    reflection_when and reflection_when != prev.get("when"),
                    reflection_usage and reflection_usage != prev.get("tool_usage"),
                ]
            )
            prev_version = int(prev_meta.get("skill_version", 0))
            next_version = prev_version + 1 if (prev_version == 0 or text_changed) else prev_version

            success = int(item.get("success", 0))
            correct = int(item.get("correct", 0))
            reward_sum = round(_safe_float(item.get("reward_sum"), 0.0), 6)
            total_denominator = max(total, 1)

            refreshed_by_id[skill_id] = {
                "id": skill_id,
                "name": reflection_name or prev_name or default_name,
                "when": reflection_when or prev_when or default_when,
                "trigger": reflection_when or prev_when or default_when,
                "strategy": reflection_strategy if isinstance(reflection_strategy, list) and reflection_strategy else (prev_strategy or default_strategy),
                "tool_usage": reflection_usage or prev.get("tool_usage") or default_usage,
                "preconditions": prev.get("preconditions") or preconditions,
                "skip_conditions": prev.get("skip_conditions") or skip_conditions,
                "tool_candidates": prev.get("tool_candidates") or effective_sequence,
                "fallbacks": prev.get("fallbacks") or fallbacks,
                "stop_conditions": prev.get("stop_conditions") or stop_conditions,
                "required_evidence": prev.get("required_evidence") or required_evidence,
                "answer_pattern": prev.get("answer_pattern") or answer_pattern,
                "coverage": self._merge_skill_coverage(prev.get("coverage", {}), coverage),
                "failure_lessons": failure_lessons,
                "view_policy": prev.get("view_policy") or view_policy,
                "meta": {
                    "source": "runtime_learning",
                    "pattern": pattern,
                    "seed_parent": seed_parent,
                    "blocker_type": blocker_type,
                    "sequence": effective_sequence,
                    "effective_sequence": effective_sequence,
                    "full_sequence": full_sequence,
                    "fallback_gap": max(0, len(full_sequence) - len(effective_sequence)),
                    "failure_memory": failure_memory,
                    "failure_lessons": failure_lessons,
                    "scene_task_context": scene_task_context,
                    "required_evidence": required_evidence,
                    "coverage": coverage,
                    "answer_pattern": answer_pattern,
                    "maturity": maturity,
                    "total": total,
                    "success": success,
                    "correct": correct,
                    "reward_sum": reward_sum,
                    "success_rate": round(success / total_denominator, 6),
                    "reward_avg": round(reward_sum / total_denominator, 6),
                    "correct_rate": round(correct / total_denominator, 6) if correct > 0 else 0.0,
                    "skill_version": next_version,
                    "reflection": reflection or prev_meta.get("reflection"),
                },
            }

        ordered_skills: List[Dict[str, Any]] = []
        seen_ids = set()
        for skill in skill_mem.get("skills", []):
            if not isinstance(skill, dict):
                continue
            skill_id = str(skill.get("id") or "").strip()
            if not skill_id or skill_id not in refreshed_by_id or skill_id in seen_ids:
                continue
            ordered_skills.append(refreshed_by_id[skill_id])
            seen_ids.add(skill_id)

        for skill_id in trajectory_order:
            if skill_id in refreshed_by_id and skill_id not in seen_ids:
                ordered_skills.append(refreshed_by_id[skill_id])
                seen_ids.add(skill_id)

        skill_mem["skills"] = ordered_skills

        self._memory["global_version"] = int(self._memory.get("global_version", 0)) + 1

    def _build_default_question_classes(self) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        classes: Dict[str, Dict[str, Any]] = {}
        for class_id, keywords in self._seed_class_keywords.items():
            classes[class_id] = {
                "class_id": class_id,
                "name": class_id,
                "description": f"Seed question class for {class_id}",
                "seed_parent": class_id,
                "keywords": keywords,
                "example_questions": [],
                "hits": 0,
                "first_seen": now,
                "last_seen": now,
                "source": "seed",
                "solution_stats": {},
            }
        return classes

    def _default_memory_v3(self) -> Dict[str, Any]:
        classes = self._build_default_question_classes()
        return {
            "version": 3,
            "global_version": 0,
            "rule_memory": {
                "version": 1,
                "rules": {
                    "seed_problem_order": list(self._seed_pattern_order.keys()),
                    "known_tools": sorted(self._known_tools),
                },
            },
            "working_memory": {
                "version": 1,
                "sessions": {},
            },
            "episode_memory": {
                "version": 1,
                "episodes": [],
            },
            "question_class_memory": {
                "version": 1,
                "global_version": 0,
                "class_order": list(classes.keys()),
                "classes": classes,
            },
            "skill_memory": {
                "version": 1,
                "trajectories": {},
                "skills": [],
                "skill_management": {
                    "schema_version": 1,
                    "decisions": [],
                    "failure_lessons": {},
                    "success_workflows": {},
                },
            },
            "evidence_memory": {
                "version": 1,
                "facts": [],
            },
        }

    def _migrate_legacy_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        migrated = self._default_memory_v3()
        migrated["global_version"] = int(data.get("global_version", 0))

        skill_mem = migrated["skill_memory"]
        skill_mem["trajectories"] = data.get("trajectories", {}) if isinstance(data.get("trajectories", {}), dict) else {}
        skill_mem["skills"] = data.get("skills", []) if isinstance(data.get("skills", []), list) else []

        qcm = migrated["question_class_memory"]
        classes = qcm["classes"]

        for key, node in skill_mem["trajectories"].items():
            if not isinstance(node, dict):
                continue
            class_id = node.get("pattern") or "comprehensive_scene_understanding"
            node["seed_parent"] = node.get("seed_parent") or (class_id if self._is_seed_problem_id(class_id) else self._resolve_seed_parent(class_id))
            if class_id not in classes:
                now = datetime.now().isoformat()
                classes[class_id] = {
                    "class_id": class_id,
                    "name": class_id,
                    "description": "Migrated class from legacy pattern",
                    "seed_parent": class_id if self._is_seed_problem_id(class_id) else self._resolve_seed_parent(class_id),
                    "keywords": [],
                    "example_questions": [],
                    "hits": 0,
                    "first_seen": now,
                    "last_seen": now,
                    "source": "migration",
                    "solution_stats": {},
                }
                qcm["class_order"].append(class_id)

            seq = node.get("sequence", [])
            if not isinstance(seq, list):
                seq = []
            if seq:
                seq_key = "->".join(seq)
                stats = classes[class_id]["solution_stats"].setdefault(
                    seq_key,
                    {"sequence": seq, "total": 0, "success": 0, "last_seen": None},
                )
                stats["total"] += int(node.get("total", 0))
                stats["success"] += int(node.get("success", 0))
                stats["last_seen"] = node.get("last_seen")

        migrated["version"] = 3
        return migrated

    def _resolve_hierarchical_child(self, root: Path, value: str) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else root / path

    def _load_json_file(self, path: Path, default: Any) -> Any:
        try:
            if not path.exists():
                return default
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if payload is not None else default
        except Exception as exc:
            logger.warning("Failed to read memory shard %s: %s", path, exc)
            return default

    def _load_jsonl_tail(self, path: Path, limit: int) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        rows.append(payload)
            if len(rows) > limit:
                return rows[-limit:]
            return rows
        except Exception as exc:
            logger.warning("Failed to read memory shard %s: %s", path, exc)
            return []

    def _load_hierarchical_memory(self, manifest: Optional[Dict[str, Any]], default: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        root = self._hierarchical_root
        if isinstance(manifest, dict):
            manifest_root = manifest.get("root") or manifest.get("memory_root")
            if manifest_root:
                root = Path(str(manifest_root))
                if not root.is_absolute():
                    root = self.storage_path.parent / root
        if not root.exists():
            return None
        self._hierarchical_root = root

        memory = self._default_memory_v3()
        if isinstance(manifest, dict):
            memory["version"] = int(manifest.get("version", 3))
            memory["global_version"] = int(manifest.get("global_version", 0))

        rule_memory = self._load_json_file(root / "rules.json", {})
        if isinstance(rule_memory, dict):
            memory["rule_memory"].update(rule_memory)

        qcm = self._load_json_file(root / "question_classes" / "index.json", {})
        if isinstance(qcm, dict):
            memory["question_class_memory"].update(qcm)

        skill_mem = self._load_json_file(root / "skills" / "state.json", {})
        if isinstance(skill_mem, dict):
            memory["skill_memory"].update(skill_mem)

        episodes = self._load_jsonl_tail(
            root / "episodes" / "recent.jsonl",
            self._hierarchical_episode_limit,
        )
        if episodes:
            memory["episode_memory"]["episodes"] = episodes

        facts = self._load_jsonl_tail(
            root / "evidence" / "facts.jsonl",
            self._hierarchical_fact_limit,
        )
        if facts:
            memory["evidence_memory"]["facts"] = facts

        return memory

    def _load_progressive_state_memory(self, default: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        qcm = self._load_json_file(self._dynamic_problem_state_path, {})
        skill_mem = self._load_json_file(self._dynamic_skill_state_path, {})
        if not isinstance(qcm, dict) and not isinstance(skill_mem, dict):
            return None
        if not qcm and not skill_mem:
            return None
        memory = self._default_memory_v3()
        if isinstance(qcm, dict) and qcm:
            memory["question_class_memory"].update(qcm)
            memory["global_version"] = max(
                int(memory.get("global_version", 0)),
                int(qcm.get("global_version", 0)),
            )
        if isinstance(skill_mem, dict) and skill_mem:
            memory["skill_memory"].update(skill_mem)
        self._needs_storage_refresh = True
        return memory

    def _load_memory(self) -> Dict[str, Any]:
        default = self._default_memory_v3()
        if not self.storage_path.exists():
            loaded = self._load_hierarchical_memory(None, default)
            if loaded is not None:
                return loaded
            return default
        try:
            file_size = self.storage_path.stat().st_size
            if file_size > self._monolith_load_limit_bytes:
                logger.warning(
                    "Skipping monolithic skill memory load from %s because it is %.2f GB; "
                    "falling back to hierarchical/progressive state.",
                    self.storage_path,
                    file_size / (1024 ** 3),
                )
                loaded = self._load_hierarchical_memory(None, default)
                if loaded is not None:
                    return loaded
                loaded = self._load_progressive_state_memory(default)
                if loaded is not None:
                    return loaded
                self._needs_storage_refresh = True
                return default

            with self.storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return default

            if data.get("storage_format") in {"skill3d_hierarchical_memory", "spagent_hierarchical_memory"}:
                loaded = self._load_hierarchical_memory(data, default)
                if loaded is not None:
                    return loaded
                self._needs_storage_refresh = True
                return default

            if "question_class_memory" in data and "skill_memory" in data:
                merged = default
                merged.update({
                    "version": int(data.get("version", 3)),
                    "global_version": int(data.get("global_version", 0)),
                })

                if isinstance(data.get("rule_memory"), dict):
                    merged["rule_memory"].update(data["rule_memory"])
                if isinstance(data.get("working_memory"), dict):
                    merged["working_memory"].update(data["working_memory"])
                if isinstance(data.get("episode_memory"), dict):
                    merged["episode_memory"].update(data["episode_memory"])
                if isinstance(data.get("question_class_memory"), dict):
                    merged["question_class_memory"].update(data["question_class_memory"])
                if isinstance(data.get("skill_memory"), dict):
                    merged["skill_memory"].update(data["skill_memory"])
                if isinstance(data.get("evidence_memory"), dict):
                    merged["evidence_memory"].update(data["evidence_memory"])

                merged["version"] = 3
                return merged

            logger.info("Migrating legacy learned_skills.json to two-level memory schema")
            self._needs_storage_refresh = True
            return self._migrate_legacy_memory(data)

        except Exception as exc:
            logger.warning("Failed to load learned skills from %s: %s", self.storage_path, exc)
            loaded = self._load_hierarchical_memory(None, default)
            if loaded is not None:
                return loaded
            loaded = self._load_progressive_state_memory(default)
            if loaded is not None:
                return loaded
            return default

    def _save_memory(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_hierarchical_memory()
            self._save_question_class_memory()
            self._materialize_progressive_disclosure()
        except Exception as exc:
            logger.warning("Failed to save learned skills to %s: %s", self.storage_path, exc)

    def _save_hierarchical_memory(self) -> None:
        root = self._hierarchical_root
        root.mkdir(parents=True, exist_ok=True)
        (root / "question_classes").mkdir(parents=True, exist_ok=True)
        (root / "skills" / "by_id").mkdir(parents=True, exist_ok=True)
        (root / "trajectories" / "by_class").mkdir(parents=True, exist_ok=True)
        (root / "episodes").mkdir(parents=True, exist_ok=True)
        (root / "evidence").mkdir(parents=True, exist_ok=True)
        (root / "working").mkdir(parents=True, exist_ok=True)

        rule_memory = self._memory.get("rule_memory", {})
        qcm = self._memory.get("question_class_memory", {})
        skill_mem = self._memory.get("skill_memory", {})
        episode_mem = self._memory.get("episode_memory", {})
        evidence_mem = self._memory.get("evidence_memory", {})

        self._atomic_json_dump(root / "rules.json", rule_memory if isinstance(rule_memory, dict) else {})
        self._atomic_json_dump(root / "question_classes" / "index.json", qcm if isinstance(qcm, dict) else {})
        self._atomic_json_dump(root / "skills" / "state.json", skill_mem if isinstance(skill_mem, dict) else {})

        skills = [skill for skill in (skill_mem.get("skills", []) if isinstance(skill_mem, dict) else []) if isinstance(skill, dict)]
        keep_skill_files = set()
        for skill in skills:
            skill_id = str(skill.get("id") or self._hash_payload(skill))
            filename = self._slug(skill_id) + ".json"
            keep_skill_files.add(filename)
            self._atomic_json_dump(root / "skills" / "by_id" / filename, skill)
        self._prune_json_files(root / "skills" / "by_id", keep_skill_files)

        trajectories_by_class: Dict[str, Dict[str, Any]] = defaultdict(dict)
        trajectories = skill_mem.get("trajectories", {}) if isinstance(skill_mem, dict) else {}
        if isinstance(trajectories, dict):
            for key, node in trajectories.items():
                if not isinstance(node, dict):
                    continue
                class_id = str(node.get("pattern") or "unknown")
                trajectories_by_class[class_id][str(key)] = node
        keep_trajectory_files = set()
        for class_id, payload in trajectories_by_class.items():
            filename = self._slug(class_id) + ".json"
            keep_trajectory_files.add(filename)
            self._atomic_json_dump(root / "trajectories" / "by_class" / filename, payload)
        self._prune_json_files(root / "trajectories" / "by_class", keep_trajectory_files)

        episodes = [
            self._compact_for_memory(episode)
            for episode in (episode_mem.get("episodes", []) if isinstance(episode_mem, dict) else [])
            if isinstance(episode, dict)
        ]
        self._atomic_jsonl_dump(
            root / "episodes" / "recent.jsonl",
            episodes[-self._hierarchical_episode_limit:],
        )

        facts = [
            self._compact_for_memory(fact)
            for fact in (evidence_mem.get("facts", []) if isinstance(evidence_mem, dict) else [])
            if isinstance(fact, dict)
        ]
        self._atomic_jsonl_dump(
            root / "evidence" / "facts.jsonl",
            facts[-self._hierarchical_fact_limit:],
        )

        working_summary = self._working_memory_summary()
        self._atomic_json_dump(root / "working" / "summary.json", working_summary)

        manifest = {
            "storage_format": "skill3d_hierarchical_memory",
            "manifest_version": self._hierarchical_manifest_version,
            "version": int(self._memory.get("version", 3)),
            "global_version": int(self._memory.get("global_version", 0)),
            "root": os.path.relpath(root, self.storage_path.parent),
            "counts": {
                "question_classes": len(qcm.get("classes", {})) if isinstance(qcm, dict) else 0,
                "skills": len(skills),
                "trajectories": len(trajectories) if isinstance(trajectories, dict) else 0,
                "episodes": len(episodes),
                "facts": len(facts),
            },
            "files": {
                "rules": "rules.json",
                "question_classes": "question_classes/index.json",
                "skill_state": "skills/state.json",
                "skill_cards": "skills/by_id/*.json",
                "trajectories": "trajectories/by_class/*.json",
                "episodes": "episodes/recent.jsonl",
                "evidence": "evidence/facts.jsonl",
                "working_summary": "working/summary.json",
            },
            "notes": [
                "This file is a compact manifest. Runtime memory shards live under root.",
                "Heavy tool outputs such as image arrays are intentionally omitted from memory.",
            ],
        }
        self._atomic_json_dump(self.storage_path, manifest)

    def _prune_json_files(self, root: Path, keep_names: set) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for child in root.iterdir():
            if child.is_file() and child.suffix == ".json" and child.name not in keep_names:
                child.unlink()

    def _save_question_class_memory(self) -> None:
        try:
            qcm = self._memory.get("question_class_memory", {})
            payload = {
                "version": int(qcm.get("version", 1)),
                "global_version": int(qcm.get("global_version", 0)),
                "class_order": qcm.get("class_order", []),
                "classes": qcm.get("classes", {}),
            }
            self.question_memory_export_path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_json_dump(self.question_memory_export_path, payload)
        except Exception as exc:
            logger.warning(
                "Failed to export question class memory to %s: %s",
                self.question_memory_export_path,
                exc,
            )

    def _build_scene_signature(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: Dict[str, Any],
    ) -> str:
        names = [str(call.get("name", "")).strip() for call in tool_calls if isinstance(call, dict) and str(call.get("name", "")).strip()]
        outputs = sorted(str(key) for key in tool_results.keys())
        payload = {"tools": names, "outputs": outputs}
        return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    def _episode_error_pattern(self, tool_results: Dict[str, Any]) -> List[str]:
        patterns: List[str] = []
        for result in (tool_results or {}).values():
            if not isinstance(result, dict) or result.get("success"):
                continue
            failure_type = str(result.get("failure_type") or "").strip()
            if failure_type and failure_type not in patterns:
                patterns.append(failure_type)
        return patterns

    def _extract_object_labels(self, tool_results: Dict[str, Any]) -> List[str]:
        labels: List[str] = []
        stack: List[Any] = list((tool_results or {}).values())
        while stack:
            result = stack.pop()
            if not isinstance(result, dict):
                continue
            for label in result.get("labels", []) or []:
                label_text = str(label).strip()
                if label_text and label_text not in labels:
                    labels.append(label_text)
            for detection in result.get("detections", []) or []:
                if isinstance(detection, dict):
                    label_text = str(detection.get("label", "")).strip()
                    if label_text and label_text not in labels:
                        labels.append(label_text)
            nested = result.get("result")
            if isinstance(nested, dict):
                stack.append(nested)
            elif isinstance(nested, list):
                stack.extend(item for item in nested if isinstance(item, dict))
        return labels

    def _extract_relation_hints(self, question: str) -> List[str]:
        text = self.normalize_question(question).lower()
        relation_keywords = [
            "left", "right", "between", "front", "behind", "closer", "farther",
            "distance", "visible", "occluded", "touching", "inside", "outside",
        ]
        return [keyword for keyword in relation_keywords if keyword in text]

    def _build_episode_retrieval_text(
        self,
        question: str,
        effective_sequence: List[str],
        tool_results: Dict[str, Any],
        success: bool,
        final_answer: Optional[str],
    ) -> str:
        labels = self._extract_object_labels(tool_results)
        relations = self._extract_relation_hints(question)
        sequence_text = " -> ".join(effective_sequence) if effective_sequence else "no tool chain"
        return "\n".join(
            [
                f"Episode for question: {self.normalize_question(question)}",
                f"Tool chain: {sequence_text}",
                f"Observed objects: {', '.join(labels) if labels else 'unknown'}",
                f"Spatial relations: {', '.join(relations) if relations else 'none explicit'}",
                f"Outcome: {'success' if success else 'failure'}",
                f"Final answer: {str(final_answer or '').strip() or 'unknown'}",
            ]
        )

    def _hash_payload(self, payload: Any) -> str:
        compact = self._compact_for_memory(payload)
        return hashlib.sha1(json.dumps(compact, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    def _working_memory_summary(self) -> Dict[str, Any]:
        sessions = self._working_memory().get("sessions", {})
        summary: Dict[str, Any] = {"version": 1, "sessions": {}}
        if not isinstance(sessions, dict):
            return summary
        for session_id, session in list(sessions.items())[-MAX_WORKING_SESSIONS:]:
            if not isinstance(session, dict):
                continue
            notes = session.get("notes", []) if isinstance(session.get("notes"), list) else []
            summary["sessions"][str(session_id)] = {
                "session_id": str(session.get("session_id") or session_id),
                "note_count": len(notes),
                "recent_note_types": [
                    str(note.get("type", "generic"))
                    for note in notes[-5:]
                    if isinstance(note, dict)
                ],
            }
        return summary

    def _is_large_array_like(self, value: Any) -> bool:
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                size = 1
                for dim in shape:
                    size *= int(dim)
                return size > MAX_MEMORY_LIST_ITEMS
            except Exception:
                return True
        if isinstance(value, list) and len(value) > MAX_MEMORY_LIST_ITEMS:
            sample = value[:8]
            if all(isinstance(item, (int, float, bool)) for item in sample):
                return True
            if any(isinstance(item, list) for item in sample):
                return True
        return False

    def _summarize_heavy_value(self, value: Any, reason: str) -> Dict[str, Any]:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            try:
                shape_payload = [int(dim) for dim in shape]
            except Exception:
                shape_payload = str(shape)
            return {
                "omitted": True,
                "reason": reason,
                "type": type(value).__name__,
                "shape": shape_payload,
                "dtype": str(dtype) if dtype is not None else None,
            }
        if isinstance(value, list):
            return {
                "omitted": True,
                "reason": reason,
                "type": "list",
                "length": len(value),
                "sample_type": type(value[0]).__name__ if value else None,
            }
        return {
            "omitted": True,
            "reason": reason,
            "type": type(value).__name__,
        }

    def _compact_for_memory(self, value: Any, key_hint: str = "", depth: int = 0) -> Any:
        key_lower = str(key_hint or "").lower()
        if key_lower in HEAVY_MEMORY_KEYS:
            return self._summarize_heavy_value(value, f"heavy field `{key_hint}` is not persisted in memory")
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if self._is_large_array_like(value):
            return self._summarize_heavy_value(value, "large array-like payload is not persisted in memory")
        if isinstance(value, dict):
            compact: Dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                compact[key_text] = self._compact_for_memory(item, key_text, depth + 1)
            return compact
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            if len(items) > MAX_MEMORY_LIST_ITEMS and depth > 0:
                return {
                    "omitted": True,
                    "reason": "long list truncated for memory persistence",
                    "length": len(items),
                    "preview": [self._compact_for_memory(item, key_hint, depth + 1) for item in items[:8]],
                }
            return [self._compact_for_memory(item, key_hint, depth + 1) for item in items]
        if hasattr(value, "tolist"):
            try:
                return self._compact_for_memory(value.tolist(), key_hint, depth + 1)
            except Exception:
                return self._summarize_heavy_value(value, "array-like payload could not be compacted safely")
        if hasattr(value, "item"):
            try:
                return self._compact_for_memory(value.item(), key_hint, depth + 1)
            except Exception:
                pass
        return str(value)

    def _atomic_json_dump(self, path: Path, payload: Dict[str, Any]) -> None:
        safe_payload = self._json_safe(_strip_time_fields_for_output(payload))
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            json.dump(safe_payload, tmp_file, indent=2, ensure_ascii=False)
            tmp_name = tmp_file.name
        os.replace(tmp_name, path)

    def _atomic_jsonl_dump(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        safe_rows = [self._json_safe(_strip_time_fields_for_output(row)) for row in rows]
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            for row in safe_rows:
                tmp_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp_name = tmp_file.name
        os.replace(tmp_name, path)

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if self._is_large_array_like(value):
            return self._summarize_heavy_value(value, "large array-like payload is not JSON-serialized into memory")
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            if self._is_large_array_like(items):
                return self._summarize_heavy_value(items, "large list payload is not JSON-serialized into memory")
            return [self._json_safe(item) for item in items]
        if hasattr(value, "tolist"):
            try:
                listed = value.tolist()
                if self._is_large_array_like(listed):
                    return self._summarize_heavy_value(value, "large array-like payload is not JSON-serialized into memory")
                return self._json_safe(listed)
            except Exception:
                pass
        if hasattr(value, "item"):
            try:
                return self._json_safe(value.item())
            except Exception:
                pass
        return str(value)
