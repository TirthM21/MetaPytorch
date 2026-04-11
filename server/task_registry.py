from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TaskDefinition:
    """Typed task registry entry used by the environment and validator endpoints."""

    task_id: str
    description: str
    difficulty: str
    constraints: Dict[str, Any]
    grader: bool = True
    grader_ref: Optional[str] = None
    reward_weights: Optional[Dict[str, float]] = None
    scoring_notes: Optional[str] = None

    def to_openenv_task(self) -> Dict[str, Any]:
        """Convert a task definition into validator-facing task metadata."""
        payload = {
            "id": self.task_id,
            "difficulty": self.difficulty,
            "description": self.description,
            "grader": self.grader,
            "max_steps": self.constraints["max_steps"],
            "score_range": {"min_exclusive": 0.0, "max_exclusive": 1.0},
            "constraints": dict(self.constraints),
        }
        if self.grader_ref:
            payload["grader_ref"] = self.grader_ref
        if self.scoring_notes:
            payload["scoring_notes"] = self.scoring_notes
        return payload


TASK_DEFINITIONS: List[TaskDefinition] = [
    TaskDefinition(
        task_id="maintain_temperature",
        difficulty="easy",
        description="Keep greenhouse temperature in the optimal 20-26°C range for 24 simulated hours.",
        constraints={
            "max_steps": 24,
            "weather_volatility": 0.3,
            "extreme_weather": False,
            "temperature_target": [20.0, 26.0],
        },
        grader_ref="tasks:grade_temp",
        reward_weights={
            "temperature": 0.70,
            "humidity": 0.10,
            "light": 0.05,
            "co2": 0.05,
            "energy": 0.05,
            "stability": 0.05,
        },
        scoring_notes="Final score is the fraction of steps spent in the optimal temperature band.",
    ),
    TaskDefinition(
        task_id="optimize_growth",
        difficulty="medium",
        description="Maximize crop growth over 3 days while preserving plant health and avoiding energy waste.",
        constraints={
            "max_steps": 72,
            "weather_volatility": 0.5,
            "extreme_weather": False,
            "reasonable_energy_per_step": 1.5,
        },
        grader_ref="tasks:grade_hum",
        reward_weights={
            "temperature": 0.25,
            "humidity": 0.20,
            "light": 0.15,
            "co2": 0.10,
            "energy": 0.15,
            "stability": 0.15,
        },
        scoring_notes="Final score blends growth, health, and average reward with an energy penalty.",
    ),
    TaskDefinition(
        task_id="weather_resilience",
        difficulty="hard",
        description="Maintain viable crop conditions through 7 days of stochastic extreme weather.",
        constraints={
            "max_steps": 168,
            "weather_volatility": 0.9,
            "extreme_weather": True,
            "survival_threshold": 0.1,
        },
        grader_ref="tasks:grade_res",
        reward_weights={
            "temperature": 0.20,
            "humidity": 0.20,
            "light": 0.15,
            "co2": 0.10,
            "energy": 0.10,
            "stability": 0.25,
        },
        scoring_notes="Final score blends plant health, growth, average reward, and survival.",
    ),
    TaskDefinition(
        task_id="resource_efficiency_master",
        difficulty="expert",
        description="Maximize growth over 10 days under volatile weather and a tight net-zero-style energy budget.",
        constraints={
            "max_steps": 240,
            "weather_volatility": 1.2,
            "extreme_weather": True,
            "strict_energy_target_per_step": 1.0,
        },
        grader_ref="tasks:grade_res",
        reward_weights={
            "temperature": 0.30,
            "humidity": 0.10,
            "light": 0.10,
            "co2": 0.10,
            "energy": 0.30,
            "stability": 0.10,
        },
        scoring_notes="Final score heavily rewards energy efficiency while preserving growth and survival.",
    ),
]

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    task.task_id: {
        "max_steps": task.constraints["max_steps"],
        "description": task.description,
        "difficulty": task.difficulty,
        "grader": task.grader,
        "extreme_weather": bool(task.constraints.get("extreme_weather", False)),
        "weather_volatility": float(task.constraints.get("weather_volatility", 0.3)),
        "scoring_notes": task.scoring_notes or "",
        **task.constraints,
    }
    for task in TASK_DEFINITIONS
}

REWARD_WEIGHTS: Dict[str, Dict[str, float]] = {
    task.task_id: dict(task.reward_weights or {})
    for task in TASK_DEFINITIONS
}


def get_task(task_id: str) -> Optional[TaskDefinition]:
    for task in TASK_DEFINITIONS:
        if task.task_id == task_id:
            return task
    return None


def get_openenv_tasks() -> List[Dict[str, Any]]:
    return [task.to_openenv_task() for task in TASK_DEFINITIONS]
