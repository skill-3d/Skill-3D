from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

_WEAVE = None
_WEAVE_ENABLED = False
_WEAVE_INIT_ATTEMPTED = False
_WEAVE_OPS: Dict[str, Any] = {}


def _env_flag(name: str, default: str = "0") -> bool:
    value = str(os.environ.get(name, default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _local_rank_is_primary() -> bool:
    local_rank = str(os.environ.get("LOCAL_RANK", "0")).strip()
    return local_rank in {"", "0"}


def _project_name() -> Optional[str]:
    explicit = str(os.environ.get("WEAVE_PROJECT", "")).strip()
    if explicit:
        return explicit

    entity = str(os.environ.get("WANDB_ENTITY", "")).strip()
    project = str(os.environ.get("WANDB_PROJECT", "")).strip()
    if entity and project:
        return f"{entity}/{project}"
    return None


def weave_enabled() -> bool:
    init_weave_once()
    return _WEAVE_ENABLED


def init_weave_once() -> bool:
    global _WEAVE, _WEAVE_ENABLED, _WEAVE_INIT_ATTEMPTED
    if _WEAVE_INIT_ATTEMPTED:
        return _WEAVE_ENABLED

    _WEAVE_INIT_ATTEMPTED = True

    if not _env_flag("SPAGENT_ENABLE_WEAVE", "0"):
        logger.info("Weave integration disabled. Set SPAGENT_ENABLE_WEAVE=1 to enable tracing.")
        return False

    if not _local_rank_is_primary():
        logger.info("Skipping Weave init on non-primary rank.")
        return False

    project = _project_name()
    if not project:
        logger.warning("Weave requested but WEAVE_PROJECT (or WANDB_ENTITY/WANDB_PROJECT) is missing.")
        return False

    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("Weave requested but WANDB_API_KEY is not set.")
        return False

    try:
        import weave  # type: ignore
    except Exception as exc:
        logger.warning("Failed to import weave: %s", exc)
        return False

    try:
        weave.init(project)
        _WEAVE = weave
        _WEAVE_ENABLED = True
        logger.info("Weave initialized for project: %s", project)
        return True
    except Exception as exc:
        logger.warning("Failed to initialize Weave for project %s: %s", project, exc)
        return False


def _truncate_text(text: str, limit: int = 240) -> str:
    raw = str(text or "")
    if len(raw) <= limit:
        return raw
    return raw[: limit - 3] + "..."


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return _truncate_text(repr(value), 200)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value, 500)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        output: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 32:
                output["__truncated__"] = f"{len(value) - 32} more fields"
                break
            output[str(key)] = _json_safe(item, depth=depth + 1)
        return output
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if len(items) > 32:
            items = items[:32] + [f"... ({len(value) - 32} more items)"]
        return [_json_safe(item, depth=depth + 1) for item in items]
    return _truncate_text(repr(value), 200)


def _get_op(name: str):
    if name in _WEAVE_OPS:
        return _WEAVE_OPS[name]

    if _WEAVE is None:
        return None

    @_WEAVE.op(name=name)
    def _event(payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

    _WEAVE_OPS[name] = _event
    return _event


def log_weave_event(name: str, payload: Dict[str, Any]) -> None:
    if not init_weave_once():
        return

    op = _get_op(name)
    if op is None:
        return

    safe_payload = _json_safe(payload)
    try:
        op(safe_payload)
    except Exception as exc:
        logger.warning("Weave event logging failed for %s: %s", name, exc)


def completion_preview(text: str, limit: int = 240) -> str:
    return _truncate_text(str(text or "").replace("\n", " "), limit)


def reward_summary_payload(
    reward_name: str,
    completions: Any,
    rewards: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reward_values = list(rewards or [])
    size = len(reward_values)
    avg_reward = float(sum(float(item) for item in reward_values) / size) if size else 0.0
    preview = [completion_preview(item) for item in list(completions or [])[:3]]
    payload: Dict[str, Any] = {
        "reward_name": reward_name,
        "batch_size": size,
        "avg_reward": avg_reward,
        "reward_preview": reward_values[:8],
        "completion_preview": preview,
    }
    if extra:
        payload.update(extra)
    return payload
