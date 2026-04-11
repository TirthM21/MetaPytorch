#!/usr/bin/env python3
"""
GRPO training scaffold for the Greenhouse environment.

This script prepares a task-conditioned dataset and reward functions that can
be used with TRL's GRPOTrainer. It is intentionally lightweight: the
environment and task logic remain local to this repo, while model training is
optional and only runs when TRL/transformers/datasets are installed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from server.task_registry import TASK_DEFINITIONS


@dataclass(frozen=True)
class GreenhousePrompt:
    task_id: str
    difficulty: str
    system_prompt: str
    user_prompt: str


SYSTEM_PROMPT = (
    "You are a greenhouse climate control agent. "
    "Return only a JSON object with heater_power, ventilation_rate, "
    "humidifier_level, and artificial_lighting in [0.0, 1.0]."
)


def build_training_prompts() -> List[GreenhousePrompt]:
    prompts: List[GreenhousePrompt] = []
    for task in TASK_DEFINITIONS:
        constraints = json.dumps(task.constraints, sort_keys=True)
        prompt = (
            f"Task: {task.task_id}\n"
            f"Difficulty: {task.difficulty}\n"
            f"Goal: {task.description}\n"
            f"Constraints: {constraints}\n"
            "Given an observation, output the next greenhouse control action."
        )
        prompts.append(
            GreenhousePrompt(
                task_id=task.task_id,
                difficulty=task.difficulty,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
            )
        )
    return prompts


def reward_json_validity(completions: List[str], **_: Any) -> List[float]:
    rewards: List[float] = []
    for text in completions:
        try:
            data = json.loads(text)
            keys = {"heater_power", "ventilation_rate", "humidifier_level", "artificial_lighting"}
            rewards.append(0.99 if keys.issubset(data.keys()) else 0.25)
        except Exception:
            rewards.append(0.01)
    return rewards


def reward_action_bounds(completions: List[str], **_: Any) -> List[float]:
    rewards: List[float] = []
    for text in completions:
        try:
            data = json.loads(text)
            values = [
                float(data["heater_power"]),
                float(data["ventilation_rate"]),
                float(data["humidifier_level"]),
                float(data["artificial_lighting"]),
            ]
            valid = all(0.0 <= value <= 1.0 for value in values)
            rewards.append(0.99 if valid else 0.10)
        except Exception:
            rewards.append(0.01)
    return rewards


def export_dataset(path: str) -> None:
    prompts = [asdict(prompt) for prompt in build_training_prompts()]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(prompts, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GRPO prompts for greenhouse control.")
    parser.add_argument("--export-json", type=str, default="grpo_prompts.json")
    args = parser.parse_args()
    export_dataset(args.export_json)
    print(f"Exported {len(TASK_DEFINITIONS)} GRPO prompts to {args.export_json}")


if __name__ == "__main__":
    main()
