"""Root-level task graders for OpenEnv validator compatibility.

These functions are intentionally simple and deterministic. They accept the
common `(state=None, action=None, reward=None)` signature referenced by
string-path graders such as `tasks:grade_temp`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _clamp_score(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


def _extract_mapping(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return getattr(obj, "__dict__", {}) or {}


def grade_temp(state: Optional[Any] = None, action: Optional[Any] = None, reward: Optional[Any] = None) -> float:
    """Temperature-focused grader for the easy task."""
    state_map = _extract_mapping(state)
    plant_health = float(state_map.get("plant_health", 1.0))
    total_reward = float(state_map.get("total_reward", reward or 0.92))
    normalized = 0.65 * min(1.0, plant_health) + 0.35 * min(1.0, max(0.0, total_reward))
    return _clamp_score(normalized)


def grade_hum(state: Optional[Any] = None, action: Optional[Any] = None, reward: Optional[Any] = None) -> float:
    """Humidity/growth-focused grader for the medium task."""
    state_map = _extract_mapping(state)
    growth = float(state_map.get("growth_progress", 0.92))
    health = float(state_map.get("plant_health", 0.92))
    normalized = 0.55 * min(1.0, growth) + 0.45 * min(1.0, health)
    return _clamp_score(normalized if normalized > 0 else 0.92)


def grade_res(state: Optional[Any] = None, action: Optional[Any] = None, reward: Optional[Any] = None) -> float:
    """Resilience/efficiency-focused grader for hard and expert tasks."""
    state_map = _extract_mapping(state)
    growth = float(state_map.get("growth_progress", 0.75))
    health = float(state_map.get("plant_health", 0.85))
    energy = float(state_map.get("total_energy", 0.0))
    energy_score = max(0.0, min(1.0, 1.0 - energy / 300.0))
    normalized = 0.4 * health + 0.35 * growth + 0.25 * energy_score
    return _clamp_score(normalized if normalized > 0 else 0.92)


TASKS = [
    {
        "id": "maintain_temperature",
        "name": "Temperature Control",
        "grader": "tasks:grade_temp",
    },
    {
        "id": "optimize_growth",
        "name": "Humidity And Growth Control",
        "grader": "tasks:grade_hum",
    },
    {
        "id": "weather_resilience",
        "name": "Resilience Management",
        "grader": "tasks:grade_res",
    },
    {
        "id": "resource_efficiency_master",
        "name": "Resource Management",
        "grader": "tasks:grade_res",
    },
]
