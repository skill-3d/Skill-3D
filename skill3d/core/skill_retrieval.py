"""Hybrid retrieval over structured skill memory units.

This module is intentionally scoped to *skill memory*, not general-purpose RAG:
it indexes static/dynamic skill cards and their structured trajectory/failure
metadata, then returns scored skill units for prompt-time augmentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .skill_normalization import build_default_skills, skills_to_json


logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIM = 384
DEFAULT_MONOLITH_LOAD_LIMIT_MB = 256
DEFAULT_RETRIEVAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_LOCAL_RETRIEVAL_MODEL = Path(__file__).resolve().parents[2] / "Qwen3-Embedding-0.6B"
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


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else [] if value is None else [value]


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _strip_time_fields_for_output(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_time_fields_for_output(item)
            for key, item in value.items()
            if str(key).strip().lower() not in TIME_OUTPUT_KEYS
        }
    if isinstance(value, list):
        return [_strip_time_fields_for_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_time_fields_for_output(item) for item in value)
    return value


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _env_flag(name: str, default: bool = True) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


@dataclass
class SkillMemoryUnit:
    skill_id: str
    skill_name: str
    source_type: str
    task_pattern: str
    question_class: str
    trigger: str
    strategy: List[str] = field(default_factory=list)
    tool_usage: str = ""
    tool_candidates: List[str] = field(default_factory=list)
    fallbacks: List[Dict[str, Any]] = field(default_factory=list)
    failure_memory: Dict[str, Any] = field(default_factory=dict)
    view_policy: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    correct_rate: float = 0.0
    reward_avg: float = 0.0
    maturity: str = "static"
    last_updated: str = ""
    retrieval_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _strip_time_fields_for_output(asdict(self))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SkillMemoryUnit":
        data = dict(payload)
        data["strategy"] = [str(x) for x in _as_list(data.get("strategy"))]
        data["tool_candidates"] = [str(x) for x in _as_list(data.get("tool_candidates"))]
        data["fallbacks"] = [x for x in _as_list(data.get("fallbacks")) if isinstance(x, dict)]
        data["failure_memory"] = data.get("failure_memory") if isinstance(data.get("failure_memory"), dict) else {}
        data["view_policy"] = data.get("view_policy") if isinstance(data.get("view_policy"), dict) else {}
        data["metadata"] = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        return cls(**data)


def build_retrieval_text(unit: SkillMemoryUnit) -> str:
    strategy_text = "; ".join(_clean_text(item) for item in unit.strategy if _clean_text(item))
    tool_chain = ", ".join(unit.tool_candidates) or _clean_text(unit.tool_usage) or "unspecified tools"
    skip_conditions = unit.metadata.get("skip_conditions") or []
    preconditions = unit.metadata.get("preconditions") or []
    fallback_text = "; ".join(
        f"if {fb.get('failed_tool')} fails with {fb.get('failure_type')}, fallback to {fb.get('next_best_tool')}"
        for fb in unit.fallbacks
        if isinstance(fb, dict)
    )
    failure_text = "; ".join(
        f"{key}: {payload.get('last_error', '')}"
        for key, payload in unit.failure_memory.items()
        if isinstance(payload, dict)
    )
    view_text = ", ".join(f"{k}={v}" for k, v in unit.view_policy.items())

    return "\n".join(
        [
            f"This skill is useful for {unit.trigger or unit.task_pattern or unit.question_class}.",
            f"Typical tool chain is {tool_chain}.",
            f"Works best when {'; '.join(map(_clean_text, preconditions)) or strategy_text or unit.trigger}.",
            f"Avoid {'; '.join(map(_clean_text, skip_conditions)) or 'using it when the current evidence is already sufficient'}.",
            f"Fallback to {fallback_text or 'the closest matching static seed skill or a complementary tool'}." ,
            f"Historical performance: success_rate={unit.success_rate:.3f}, correct_rate={unit.correct_rate:.3f}, reward_avg={unit.reward_avg:.3f}, maturity={unit.maturity}.",
            f"Failure memory: {failure_text or 'none recorded'}.",
            f"View policy: {view_text or 'none'}.",
        ]
    )


def _extract_tools_from_text(text: str, known_tools: Iterable[str]) -> List[str]:
    found: List[str] = []
    for tool in known_tools:
        if tool in str(text or "") and tool not in found:
            found.append(tool)
    return found


def _static_units(memory: Dict[str, Any]) -> List[SkillMemoryUnit]:
    seed_skills = skills_to_json(build_default_skills())
    seed_patterns = [
        "spatial_localization",
        "depth_distance",
        "multiview_occlusion",
        "detection_counting",
        "segmentation_boundary",
        "fine_grained_pointing",
        "custom_category",
        "comprehensive_scene_understanding",
    ]
    known_tools = {
        "depth_estimation_tool", "segment_image_tool", "detect_objects_tool", "supervision_tool",
        "yoloe_detection_tool", "moondream_tool", "swinir_tool", "orient_anything_tool", "pi3_tool",
    }
    units: List[SkillMemoryUnit] = []
    for idx, skill in enumerate(seed_skills):
        pattern = seed_patterns[idx] if idx < len(seed_patterns) else f"static_{idx}"
        tool_usage = _clean_text(skill.get("tool_usage"))
        tool_candidates = _extract_tools_from_text(tool_usage, known_tools)
        unit = SkillMemoryUnit(
            skill_id=f"seed::{pattern}",
            skill_name=_clean_text(skill.get("name")) or pattern,
            source_type="static",
            task_pattern=pattern,
            question_class=pattern,
            trigger=_clean_text(skill.get("when")) or pattern,
            strategy=[_clean_text(item) for item in _as_list(skill.get("strategy"))],
            tool_usage=tool_usage,
            tool_candidates=tool_candidates,
            maturity="static",
            metadata={"preconditions": [], "skip_conditions": []},
        )
        unit.retrieval_text = build_retrieval_text(unit)
        units.append(unit)
    return units


def _dynamic_units(memory: Dict[str, Any]) -> List[SkillMemoryUnit]:
    skills = (((memory or {}).get("skill_memory") or {}).get("skills") or [])
    units: List[SkillMemoryUnit] = []
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        meta = skill.get("meta") if isinstance(skill.get("meta"), dict) else {}
        pattern = _clean_text(meta.get("pattern")) or "unknown"
        seed_parent = _clean_text(meta.get("seed_parent")) or pattern
        unit = SkillMemoryUnit(
            skill_id=_clean_text(skill.get("id")) or f"dynamic::{len(units)}",
            skill_name=_clean_text(skill.get("name")) or "learned skill",
            source_type="dynamic",
            task_pattern=pattern,
            question_class=pattern,
            trigger=_clean_text(skill.get("trigger") or skill.get("when")),
            strategy=[_clean_text(item) for item in _as_list(skill.get("strategy"))],
            tool_usage=_clean_text(skill.get("tool_usage")),
            tool_candidates=[_clean_text(item) for item in _as_list(skill.get("tool_candidates") or meta.get("sequence")) if _clean_text(item)],
            fallbacks=[item for item in _as_list(skill.get("fallbacks")) if isinstance(item, dict)],
            failure_memory=meta.get("failure_memory") if isinstance(meta.get("failure_memory"), dict) else {},
            view_policy=skill.get("view_policy") if isinstance(skill.get("view_policy"), dict) else {},
            success_rate=_safe_float(meta.get("success_rate")),
            correct_rate=_safe_float(meta.get("correct_rate")),
            reward_avg=_safe_float(meta.get("reward_avg")),
            maturity=_clean_text(meta.get("maturity")) or "dynamic",
            metadata={
                "seed_parent": seed_parent,
                "blocker_type": meta.get("blocker_type", ""),
                "preconditions": [str(x) for x in _as_list(skill.get("preconditions"))],
                "skip_conditions": [str(x) for x in _as_list(skill.get("skip_conditions"))],
                "stop_conditions": [str(x) for x in _as_list(skill.get("stop_conditions"))],
                "observed_support": {
                    "total": meta.get("total", 0),
                    "success": meta.get("success", 0),
                    "correct": meta.get("correct", 0),
                },
            },
        )
        unit.retrieval_text = build_retrieval_text(unit)
        units.append(unit)
    return units


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_manifest_root(memory_path: Path, manifest: Dict[str, Any]) -> Path:
    root_value = manifest.get("root") or manifest.get("memory_root")
    if not root_value:
        return memory_path.with_suffix("")
    root = Path(str(root_value))
    return root if root.is_absolute() else memory_path.parent / root


def load_skill_memory(memory_path: str | Path) -> Dict[str, Any]:
    path = Path(memory_path)
    if not path.exists():
        manifest_root = path.with_suffix("")
        skill_state = manifest_root / "skills" / "state.json"
        if skill_state.exists():
            payload = _load_json(skill_state)
            return {"skill_memory": payload} if isinstance(payload, dict) else {}
        return {}

    max_bytes = int(os.environ.get("SPAGENT_MONOLITH_LOAD_LIMIT_MB", str(DEFAULT_MONOLITH_LOAD_LIMIT_MB))) * 1024 * 1024
    if path.stat().st_size > max_bytes:
        manifest_root = path.with_suffix("")
        skill_state = manifest_root / "skills" / "state.json"
        if skill_state.exists():
            payload = _load_json(skill_state)
            return {"skill_memory": payload} if isinstance(payload, dict) else {}
        return {}

    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {}
    if payload.get("storage_format") in {"skill3d_hierarchical_memory", "spagent_hierarchical_memory"}:
        root = _resolve_manifest_root(path, payload)
        skill_state = root / "skills" / "state.json"
        if skill_state.exists():
            skill_payload = _load_json(skill_state)
            return {"skill_memory": skill_payload} if isinstance(skill_payload, dict) else {}
        return {}
    return payload


def load_skill_units(memory_path: str | Path) -> List[SkillMemoryUnit]:
    memory = load_skill_memory(memory_path)
    return _static_units(memory) + _dynamic_units(memory)


class EmbeddingBackend:
    name = "hashing"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEmbeddingBackend(EmbeddingBackend):
    name = "hashing"

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM):
        self.dim = int(dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = str(text or "").lower().split()
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, idx] += sign
        return _normalize_rows(vectors)


class SentenceTransformerBackend(EmbeddingBackend):
    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name
        trust_remote_code = _env_flag("SPAGENT_RETRIEVAL_TRUST_REMOTE_CODE", True)
        device = str(os.environ.get("SPAGENT_RETRIEVAL_DEVICE", "")).strip() or None
        try:
            kwargs = {"trust_remote_code": trust_remote_code}
            if device:
                kwargs["device"] = device
            self.model = SentenceTransformer(model_name, **kwargs)
        except TypeError:
            kwargs = {}
            if device:
                kwargs["device"] = device
            self.model = SentenceTransformer(model_name, **kwargs)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def make_embedding_backend(model_name: Optional[str] = None) -> EmbeddingBackend:
    model_name = (
        model_name
        or os.environ.get("SKILL3D_RETRIEVAL_MODEL")
        or os.environ.get("SPAGENT_RETRIEVAL_MODEL")
        or os.environ.get("SPAGENT_EMBEDDING_MODEL")
        or str(DEFAULT_LOCAL_RETRIEVAL_MODEL if DEFAULT_LOCAL_RETRIEVAL_MODEL.exists() else DEFAULT_RETRIEVAL_MODEL)
    )
    if str(model_name).strip().lower() in {"hash", "hashing", "none", "off"}:
        return HashingEmbeddingBackend()
    try:
        return SentenceTransformerBackend(model_name)
    except Exception as exc:
        if os.environ.get("SPAGENT_RETRIEVAL_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError(f"Failed to load retrieval embedding model {model_name}") from exc
        logger.warning("Failed to load retrieval embedding model %s; falling back to hashing embeddings: %s", model_name, exc)
        return HashingEmbeddingBackend()


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class SkillRetrievalIndex:
    def __init__(
        self,
        units: Optional[List[SkillMemoryUnit]] = None,
        embeddings: Optional[np.ndarray] = None,
        backend: Optional[EmbeddingBackend] = None,
        output_dir: Optional[str | Path] = None,
        faiss_index: Any = None,
    ):
        self.units = units or []
        self.embeddings = _normalize_rows(embeddings) if embeddings is not None and len(embeddings) else np.zeros((0, DEFAULT_EMBEDDING_DIM), dtype=np.float32)
        self.backend = backend or make_embedding_backend()
        self.output_dir = Path(output_dir) if output_dir else None
        self.faiss_index = faiss_index

    @staticmethod
    def from_memory(memory_path: str | Path, backend: Optional[EmbeddingBackend] = None) -> "SkillRetrievalIndex":
        backend = backend or make_embedding_backend()
        units = load_skill_units(memory_path)
        embeddings = backend.encode([unit.retrieval_text for unit in units]) if units else np.zeros((0, DEFAULT_EMBEDDING_DIM), dtype=np.float32)
        return SkillRetrievalIndex(units=units, embeddings=embeddings, backend=backend)

    def persist(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "units.jsonl").write_text(
            "\n".join(json.dumps(unit.to_dict(), ensure_ascii=False) for unit in self.units) + ("\n" if self.units else ""),
            encoding="utf-8",
        )
        np.save(out / "embeddings.npy", self.embeddings)
        meta = {
            "version": 1,
            "backend": getattr(self.backend, "name", "unknown"),
            "num_units": len(self.units),
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 and self.embeddings.size else 0,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self.output_dir = out

    @classmethod
    def load(cls, output_dir: str | Path, backend: Optional[EmbeddingBackend] = None) -> "SkillRetrievalIndex":
        out = Path(output_dir)
        units: List[SkillMemoryUnit] = []
        units_path = out / "units.jsonl"
        if units_path.exists():
            with units_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        units.append(SkillMemoryUnit.from_dict(json.loads(line)))
        embeddings_path = out / "embeddings.npy"
        embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
        return cls(units=units, embeddings=embeddings, backend=backend or make_embedding_backend(), output_dir=out)

    def update_index(self, new_skill_units: List[SkillMemoryUnit]) -> None:
        if not new_skill_units:
            return
        by_id = {unit.skill_id: unit for unit in self.units}
        for unit in new_skill_units:
            by_id[unit.skill_id] = unit
        self.units = list(by_id.values())
        self.embeddings = self.backend.encode([unit.retrieval_text for unit in self.units])

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        class_filter: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        start = time.perf_counter()
        if not self.units:
            return []

        allowed = set(str(item) for item in class_filter or [] if str(item))
        candidate_indices = [
            idx for idx, unit in enumerate(self.units)
            if not allowed or unit.question_class in allowed or unit.task_pattern in allowed or unit.metadata.get("seed_parent") in allowed
        ]
        if not candidate_indices:
            candidate_indices = list(range(len(self.units)))

        query_vec = self.backend.encode([query])[0]
        matrix = self.embeddings[candidate_indices]
        scores = np.dot(matrix, query_vec)
        order = np.argsort(-scores)[: max(1, int(top_k))]
        latency_ms = (time.perf_counter() - start) * 1000.0
        results: List[Dict[str, Any]] = []
        for rank, local_idx in enumerate(order, start=1):
            global_idx = candidate_indices[int(local_idx)]
            unit = self.units[global_idx]
            results.append(
                {
                    "rank": rank,
                    "semantic_score": float(scores[int(local_idx)]),
                    "latency_ms": latency_ms,
                    "unit": unit,
                }
            )
        return results


def _query_tool_signals(query: str) -> List[str]:
    text = str(query or "").lower()
    mapping = [
        ("depth_estimation_tool", ["closer", "farther", "distance", "meters", "nearest", "farthest", "front", "behind"]),
        ("detect_objects_tool", ["detect", "how many", "count", "object", "category"]),
        ("segment_image_tool", ["boundary", "mask", "outline", "closest point", "touch", "contact"]),
        ("moondream_tool", ["point", "exact", "pixel", "where exactly"]),
        ("swinir_tool", ["tiny", "small", "blurry", "low resolution", "unclear"]),
        ("orient_anything_tool", ["facing", "orientation", "pose", "front side", "back side", "rotation"]),
        ("pi3_tool", ["view", "viewpoint", "occluded", "hidden", "layout", "room size", "another view"]),
    ]
    hits: List[str] = []
    for tool, keywords in mapping:
        if any(keyword in text for keyword in keywords):
            hits.append(tool)
    return hits


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", str(text or "").lower())
        if token not in {"the", "and", "for", "with", "when", "from", "this", "that", "skill", "tool"}
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _unit_text_for_similarity(unit: SkillMemoryUnit) -> str:
    return " ".join(
        [
            unit.skill_name,
            unit.trigger,
            unit.task_pattern,
            unit.question_class,
            " ".join(unit.strategy),
            unit.tool_usage,
        ]
    )


def _unit_seed_parent(unit: SkillMemoryUnit) -> str:
    return str(unit.metadata.get("seed_parent") or unit.question_class or unit.task_pattern or "").strip()


def _unit_blocker_type(unit: SkillMemoryUnit) -> str:
    return str(unit.metadata.get("blocker_type") or "").strip()


def _unit_tool_chain(unit: SkillMemoryUnit) -> Tuple[str, ...]:
    return tuple(str(tool).strip() for tool in (unit.tool_candidates or []) if str(tool).strip())


def _same_skill_signature(left: SkillMemoryUnit, right: SkillMemoryUnit) -> bool:
    left_chain = _unit_tool_chain(left)
    right_chain = _unit_tool_chain(right)
    if not left_chain or not right_chain or left_chain != right_chain:
        return False
    left_class = _unit_seed_parent(left) or left.question_class or left.task_pattern
    right_class = _unit_seed_parent(right) or right.question_class or right.task_pattern
    if left_class != right_class:
        return False
    left_blocker = _unit_blocker_type(left)
    right_blocker = _unit_blocker_type(right)
    return not left_blocker or not right_blocker or left_blocker == right_blocker


def skill_similarity(left: SkillMemoryUnit, right: SkillMemoryUnit) -> float:
    """Estimate whether two skill cards are redundant for prompt injection."""
    if left.skill_id == right.skill_id:
        return 1.0

    left_tokens = _token_set(_unit_text_for_similarity(left))
    right_tokens = _token_set(_unit_text_for_similarity(right))
    lexical = _jaccard(left_tokens, right_tokens)

    left_chain = _unit_tool_chain(left)
    right_chain = _unit_tool_chain(right)
    left_tools = set(left_chain)
    right_tools = set(right_chain)
    tool_overlap = _jaccard(left_tools, right_tools)

    same_class = 1.0 if _unit_seed_parent(left) and _unit_seed_parent(left) == _unit_seed_parent(right) else 0.0
    same_chain = 1.0 if left_chain and left_chain == right_chain else 0.0
    same_blocker = 1.0 if _unit_blocker_type(left) and _unit_blocker_type(left) == _unit_blocker_type(right) else 0.0

    score = (
        0.45 * lexical
        + 0.25 * tool_overlap
        + 0.15 * same_class
        + 0.10 * same_chain
        + 0.05 * same_blocker
    )
    if _same_skill_signature(left, right) and lexical >= 0.35:
        score = max(score, 0.92)
    return max(0.0, min(1.0, score))


def filter_similar_skill_candidates(
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    similarity_threshold: Optional[float] = None,
    diversity_lambda: Optional[float] = None,
    enabled: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """SkillRouter-style post-rerank filter for near-duplicate skills.

    The paper uses a learned router and false-negative filtering to avoid
    confusing near-miss skills. Here we keep the same retrieval shape but add a
    deterministic diversity pass so similar dynamic skills do not crowd the
    prompt when better alternatives are available.
    """
    limit = max(1, int(top_k))
    if enabled is None:
        enabled = _env_flag("SPAGENT_SKILL_FILTER_ENABLED", True)
    threshold = float(
        similarity_threshold
        if similarity_threshold is not None
        else os.environ.get("SPAGENT_SKILL_FILTER_SIM_THRESHOLD", "0.88")
    )
    lam = float(
        diversity_lambda
        if diversity_lambda is not None
        else os.environ.get("SPAGENT_SKILL_FILTER_MMR_LAMBDA", "0.72")
    )
    lam = max(0.0, min(1.0, lam))

    diagnostics: Dict[str, Any] = {
        "enabled": bool(enabled),
        "input_count": len(candidates),
        "output_count": 0,
        "similarity_threshold": threshold,
        "diversity_lambda": lam,
        "deferred_duplicates": [],
        "selected": [],
    }

    ranked = [dict(item) for item in candidates if isinstance(item, dict) and item.get("unit")]
    ranked.sort(key=lambda row: float(row.get("score", row.get("semantic_score", 0.0)) or 0.0), reverse=True)
    if not enabled or len(ranked) <= 1:
        output = ranked[:limit]
        diagnostics["output_count"] = len(output)
        diagnostics["selected"] = [row["unit"].skill_id for row in output]
        return output, diagnostics

    scores = [float(row.get("score", row.get("semantic_score", 0.0)) or 0.0) for row in ranked]
    min_score = min(scores)
    max_score = max(scores)

    def normalized_score(row: Dict[str, Any]) -> float:
        score = float(row.get("score", row.get("semantic_score", 0.0)) or 0.0)
        if max_score <= min_score:
            return 1.0
        return (score - min_score) / (max_score - min_score)

    selected: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    remaining = list(ranked)

    while remaining and len(selected) < limit:
        best_idx = 0
        best_value = -float("inf")
        for idx, row in enumerate(remaining):
            unit = row["unit"]
            max_sim = max((skill_similarity(unit, chosen["unit"]) for chosen in selected), default=0.0)
            value = lam * normalized_score(row) - (1.0 - lam) * max_sim
            if value > best_value:
                best_idx = idx
                best_value = value

        row = remaining.pop(best_idx)
        unit = row["unit"]
        duplicate_of = None
        duplicate_similarity = 0.0
        for chosen in selected:
            sim = skill_similarity(unit, chosen["unit"])
            if sim >= threshold and _same_skill_signature(unit, chosen["unit"]):
                duplicate_of = chosen["unit"].skill_id
                duplicate_similarity = sim
                break

        if duplicate_of:
            row["skill_filter"] = {
                "status": "deferred_near_duplicate",
                "duplicate_of": duplicate_of,
                "similarity": duplicate_similarity,
            }
            deferred.append(row)
            diagnostics["deferred_duplicates"].append(
                {
                    "skill_id": unit.skill_id,
                    "duplicate_of": duplicate_of,
                    "similarity": duplicate_similarity,
                }
            )
            continue

        row["skill_filter"] = {"status": "selected", "mmr_score": best_value}
        selected.append(row)

    if len(selected) < limit:
        for row in deferred:
            if len(selected) >= limit:
                break
            row["skill_filter"]["status"] = "selected_fallback_duplicate"
            selected.append(row)

    diagnostics["output_count"] = len(selected)
    diagnostics["selected"] = [row["unit"].skill_id for row in selected]
    return selected, diagnostics


def rerank_skill_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    class_id: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    query_tools = set(_query_tool_signals(query))
    now = datetime.now()
    reranked: List[Dict[str, Any]] = []
    for item in candidates:
        unit: SkillMemoryUnit = item["unit"]
        semantic = float(item.get("semantic_score", 0.0))
        class_match = 1.0 if class_id and (unit.question_class == class_id or unit.task_pattern == class_id or unit.metadata.get("seed_parent") == class_id) else 0.0
        if not class_id:
            class_match = 0.5
        unit_tools = set(unit.tool_candidates or _extract_tools_from_text(unit.tool_usage, query_tools))
        tool_compat = len(query_tools & unit_tools) / max(len(query_tools), 1) if query_tools else 0.5
        success = max(0.0, min(1.0, unit.success_rate if unit.source_type == "dynamic" else 0.6))
        reward = max(0.0, min(1.0, unit.reward_avg if unit.source_type == "dynamic" else 0.5))
        last_dt = _parse_iso_datetime(unit.last_updated)
        freshness = 0.0
        if last_dt:
            days = max(0.0, (now - last_dt).total_seconds() / 86400.0)
            freshness = math.exp(-days / 30.0)
        failure_penalty = min(1.0, sum(_safe_float(v.get("count"), 0.0) for v in unit.failure_memory.values() if isinstance(v, dict)) / 20.0)
        view_policy_match = 0.0
        if "pi3_tool" in query_tools:
            view_policy_match = 1.0 if unit.view_policy.get("require_need_view_change") or "pi3_tool" in unit_tools else 0.0
        elif unit.view_policy.get("require_need_view_change"):
            view_policy_match = -0.2
        else:
            view_policy_match = 0.2

        final_score = (
            0.45 * semantic
            + 0.20 * class_match
            + 0.15 * tool_compat
            + 0.10 * success
            + 0.05 * reward
            + 0.05 * freshness
            + 0.05 * view_policy_match
            - 0.15 * failure_penalty
        )
        breakdown = {
            "semantic": semantic,
            "class_match_score": class_match,
            "tool_compatibility_score": tool_compat,
            "success_rate_score": success,
            "reward_avg_score": reward,
            "freshness_score": freshness,
            "failure_conflict_penalty": failure_penalty,
            "view_policy_match_score": view_policy_match,
            "final_score": final_score,
        }
        reranked.append({"unit": unit, "score": final_score, "score_breakdown": breakdown, "semantic_rank": item.get("rank")})

    reranked.sort(key=lambda row: row["score"], reverse=True)
    return reranked[: max(1, int(top_k))]


def build_index(memory_path: str | Path, output_dir: str | Path, model_name: Optional[str] = None) -> SkillRetrievalIndex:
    index = SkillRetrievalIndex.from_memory(memory_path, backend=make_embedding_backend(model_name))
    index.persist(output_dir)
    return index


def load_index(output_dir: str | Path, model_name: Optional[str] = None) -> SkillRetrievalIndex:
    return SkillRetrievalIndex.load(output_dir, backend=make_embedding_backend(model_name))


def update_index(output_dir: str | Path, new_skill_units: List[SkillMemoryUnit], model_name: Optional[str] = None) -> SkillRetrievalIndex:
    index = load_index(output_dir, model_name=model_name)
    index.update_index(new_skill_units)
    index.persist(output_dir)
    return index


def retrieve(query: str, output_dir: str | Path, top_k: int = 5, class_filter: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    return load_index(output_dir).retrieve(query=query, top_k=top_k, class_filter=class_filter)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=None)
    args = parser.parse_args()
    index = build_index(args.memory_path, args.output_dir, model_name=args.model_name)
    print(json.dumps({"num_units": len(index.units), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    _main()
