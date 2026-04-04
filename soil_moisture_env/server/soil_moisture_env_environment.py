# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Soil Moisture Irrigation Environment Implementation.

A real-world agricultural RL environment where an AI agent manages
irrigation decisions based on IoT soil moisture sensor readings.
"""

from uuid import uuid4
from typing import Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import IrrigationAction, IrrigationObservation, IrrigationState, SensorReading
    from ..simulator import SoilMoistureSimulator, CROP_CONFIG, IRRIGATION_COST
    from ..tasks import get_task_config, compute_score
except (ImportError, ModuleNotFoundError):
    from models import IrrigationAction, IrrigationObservation, IrrigationState, SensorReading
    from simulator import SoilMoistureSimulator, CROP_CONFIG, IRRIGATION_COST
    from tasks import get_task_config, compute_score


class SoilMoistureEnvironment(Environment):
    """
    Agricultural irrigation decision environment.

    The agent observes soil moisture sensor readings and decides whether
    to irrigate each day. Rewards are shaped around crop health maintenance.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._sim: Optional[SoilMoistureSimulator] = None
        self._task_cfg: dict = {}
        self._days_remaining: int = 0
        self._num_days: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._wasteful_irrigations: int = 0
        self._tomato_failed: bool = False
        self._last_action_result: Optional[str] = None
        self._episode_id: str = str(uuid4())
        self._step_count: int = 0
        # snapshot of health before each step for delta calculation
        self._prev_health: list = []

    def reset(self, task_name: str = "single_field_timing", **kwargs) -> IrrigationObservation:
        """
        Reset the environment for a new episode.
        Zero state leakage between episodes.
        """
        self._task_cfg = get_task_config(task_name)
        cfg = self._task_cfg

        self._sim = SoilMoistureSimulator(
            task_seed=cfg["task_seed"],
            num_fields=cfg["num_fields"],
            crop_types=cfg["crop_types"],
            initial_moisture=cfg["initial_moisture"],
            noisy=cfg["noisy"],
            water_budget=cfg.get("water_budget"),
        )
        self._sim.reset(cfg["initial_moisture"], cfg["num_days"])

        self._num_days = cfg["num_days"]
        self._days_remaining = cfg["num_days"]
        self._cumulative_reward = 0.0
        self._done = False
        self._wasteful_irrigations = 0
        self._tomato_failed = False
        self._last_action_result = None
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._prev_health = list(self._sim.crop_health)

        return self._build_observation()

    def step(self, action: IrrigationAction, timeout_s=None, **kwargs) -> IrrigationObservation:
        """
        Apply action, compute reward, advance day.
        Returns IrrigationObservation with reward and done embedded.
        """
        if self._done:
            obs = self._build_observation()
            obs.reward = 0.0
            obs.done = True
            return obs

        self._step_count += 1
        cfg = self._task_cfg
        sim = self._sim

        prev_health = list(sim.crop_health)
        over_irrigation_penalty = 0.0
        stress_penalty = 0.0
        budget_waste = 0.0
        action_result = "waited"

        if action.action == "irrigate":
            # Determine which field to irrigate
            field_id = action.field_id if action.field_id is not None else 0
            field_id = max(0, min(field_id, cfg["num_fields"] - 1))

            # Check wasteful irrigation (medium task: irrigate when real moisture > 70)
            if sim.real_moisture[field_id] > 70.0:
                self._wasteful_irrigations += 1

            success, msg = sim.apply_irrigation(field_id)
            action_result = msg if success else f"failed_{msg}"

            # Over-irrigation penalty: moisture went above rot threshold
            crop = cfg["crop_types"][field_id]
            rot_above = CROP_CONFIG[crop]["rot_above"]
            if sim.real_moisture[field_id] > rot_above:
                over_irrigation_penalty = 1.0

        # Advance day (natural drop + rain)
        sim.advance_day()

        # Apply health penalties based on post-advance moisture
        sim.apply_health_penalties(over_irrigation_penalty > 0, field_id=0)

        # Stress penalty: any field below stress threshold
        for i, crop in enumerate(cfg["crop_types"]):
            if sim.real_moisture[i] < CROP_CONFIG[crop]["stress_below"]:
                stress_penalty = 1.0
                break

        # Budget waste (hard task only)
        if cfg.get("water_budget") is not None:
            remaining = cfg["water_budget"] - sim.water_used
            if remaining < 0:
                budget_waste = abs(remaining) / cfg["water_budget"]

        # Crop health delta (average across fields)
        curr_health = list(sim.crop_health)
        health_delta = sum(
            (curr_health[i] - prev_health[i]) / 100.0
            for i in range(cfg["num_fields"])
        ) / cfg["num_fields"]

        # Step reward formula
        raw_reward = (
            + 0.4 * health_delta
            - 0.3 * over_irrigation_penalty
            - 0.2 * stress_penalty
            - 0.1 * budget_waste
        )
        step_reward = max(0.0, min(1.0, (raw_reward + 1.0) / 2.0))

        self._cumulative_reward += step_reward
        self._days_remaining -= 1
        self._last_action_result = action_result
        self._prev_health = curr_health

        # Tomato failure check (hard task)
        if cfg["num_fields"] == 3:
            tomato_idx = cfg["crop_types"].index("tomatoes")
            if sim.crop_health[tomato_idx] < 20.0:
                self._tomato_failed = True

        self._done = self._days_remaining <= 0

        obs = self._build_observation()
        obs.reward = step_reward
        obs.done = self._done
        return obs

    @property
    def state(self) -> IrrigationState:
        """Full internal state for debugging — never sent to agent."""
        sim = self._sim
        if sim is None:
            return IrrigationState(
                task_name="none", day=0, real_moisture=[], crop_health=[],
                water_used=0.0, done=True, cumulative_reward=0.0,
                episode_id=self._episode_id, step_count=self._step_count,
            )
        return IrrigationState(
            task_name=self._task_cfg.get("task_name", ""),
            day=sim.day,
            real_moisture=list(sim.real_moisture),
            crop_health=list(sim.crop_health),
            water_used=sim.water_used,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            episode_id=self._episode_id,
            step_count=self._step_count,
        )

    def get_final_score(self) -> float:
        """Compute the task grader score after episode ends."""
        sim = self._sim
        if sim is None:
            return 0.0
        return compute_score(
            task_name=self._task_cfg.get("task_name", ""),
            crop_health=list(sim.crop_health),
            water_used=sim.water_used,
            water_budget=self._task_cfg.get("water_budget") or 0.0,
            wasteful_irrigations=self._wasteful_irrigations,
            num_days=self._num_days,
            tomato_failed=self._tomato_failed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> IrrigationObservation:
        sim = self._sim
        cfg = self._task_cfg

        sensor_readings = []
        raw_readings = sim.get_sensor_readings()
        for i, reading in enumerate(raw_readings):
            sensor_readings.append(SensorReading(
                raw_moisture=reading,
                temperature=20.0 + sim.rng.uniform(-5.0, 5.0),
                day=sim.day,
                field_id=i,
                done=self._done,
                reward=None,
            ))

        water_budget_remaining = None
        if cfg.get("water_budget") is not None:
            water_budget_remaining = cfg["water_budget"] - sim.water_used

        crop_type = cfg["crop_types"][0] if cfg["num_fields"] == 1 else None

        return IrrigationObservation(
            sensor_readings=sensor_readings,
            weather_forecast=sim.get_weather_forecast(),
            water_budget_remaining=water_budget_remaining,
            crop_type=crop_type,
            days_remaining=self._days_remaining,
            last_action_result=self._last_action_result,
            done=self._done,
            reward=None,
        )
