# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task definitions for the Soil Moisture Irrigation Environment.
Each task returns a config dict consumed by the environment.
"""

from typing import Dict, Any

# Shared seed so graders are deterministic
TASK_SEED = 42


def get_task_config(task_name: str) -> Dict[str, Any]:
    """Return configuration dict for the named task."""
    if task_name == "single_field_timing":
        return _single_field_timing()
    elif task_name == "noisy_sensor":
        return _noisy_sensor()
    elif task_name == "multi_field_allocation":
        return _multi_field_allocation()
    else:
        raise ValueError(f"Unknown task: {task_name}")


def _single_field_timing() -> Dict[str, Any]:
    """
    Easy — 1 field (corn), 7 days, clean sensor.
    Random agent expected score: ~0.35
    """
    return {
        "task_name": "single_field_timing",
        "difficulty": "easy",
        "num_fields": 1,
        "crop_types": ["corn"],
        "initial_moisture": [45.0],
        "num_days": 7,
        "noisy": False,
        "water_budget": None,
        "task_seed": TASK_SEED,
    }


def _noisy_sensor() -> Dict[str, Any]:
    """
    Medium — 1 field (corn), 7 days, noisy sensor.
    Random agent expected score: ~0.25
    """
    return {
        "task_name": "noisy_sensor",
        "difficulty": "medium",
        "num_fields": 1,
        "crop_types": ["corn"],
        "initial_moisture": [45.0],
        "num_days": 7,
        "noisy": True,
        "water_budget": None,
        "task_seed": TASK_SEED,
    }


def _multi_field_allocation() -> Dict[str, Any]:
    """
    Hard — 3 fields (wheat, corn, tomatoes), 10 days, water budget = 100.
    Random agent expected score: ~0.20
    """
    return {
        "task_name": "multi_field_allocation",
        "difficulty": "hard",
        "num_fields": 3,
        "crop_types": ["wheat", "corn", "tomatoes"],
        "initial_moisture": [50.0, 45.0, 40.0],
        "num_days": 10,
        "noisy": True,
        "water_budget": 100.0,
        "task_seed": TASK_SEED,
    }


def compute_score(task_name: str, crop_health: list, water_used: float,
                  water_budget: float, wasteful_irrigations: int,
                  num_days: int, tomato_failed: bool) -> float:
    """
    Compute final episode score strictly within (0.0, 1.0) — exclusive.
    """
    if task_name == "single_field_timing":
        score = crop_health[0] / 100.0

    elif task_name == "noisy_sensor":
        health_score = crop_health[0] / 100.0
        waste_ratio = wasteful_irrigations / max(num_days, 1)
        score = (health_score * 0.7) + ((1.0 - waste_ratio) * 0.3)

    elif task_name == "multi_field_allocation":
        wheat_h, corn_h, tomato_h = crop_health[0], crop_health[1], crop_health[2]
        weighted = (tomato_h * 0.5 + corn_h * 0.3 + wheat_h * 0.2) / 100.0
        unused = max(0.0, water_budget - water_used)
        budget_eff = (unused / water_budget) * 0.1
        tomato_penalty = 0.4 if tomato_failed else 0.0
        score = weighted + budget_eff - tomato_penalty

    else:
        score = 0.5

    # Strictly within (0.0, 1.0) — validator rejects 0.0 and 1.0 exactly
    return max(0.001, min(0.999, score))
