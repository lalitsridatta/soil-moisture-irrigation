# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Synthetic soil moisture simulator.
All randomness is seeded for deterministic grading.
"""

import random
import math
from typing import List, Optional, Tuple

# Crop thresholds: stress_below, rot_above, value_weight
CROP_CONFIG = {
    "wheat":    {"stress_below": 25, "rot_above": 80, "value_weight": 0.2},
    "corn":     {"stress_below": 30, "rot_above": 85, "value_weight": 0.3},
    "tomatoes": {"stress_below": 35, "rot_above": 80, "value_weight": 0.5},
}

IRRIGATION_AMOUNT = 25.0   # moisture % added per irrigation
IRRIGATION_COST = 20.0     # water units per irrigation
DAILY_DROP_MIN = 8.0
DAILY_DROP_MAX = 12.0
RAIN_ADD_MIN = 15.0
RAIN_ADD_MAX = 20.0
SENSOR_NOISE_STD = 15.0
WEATHER_ACCURACY = 0.70    # forecast is correct 70% of the time


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _gauss(rng: random.Random, mean: float, std: float) -> float:
    """Box-Muller transform using the seeded rng."""
    u1 = rng.random()
    u2 = rng.random()
    z = math.sqrt(-2.0 * math.log(max(u1, 1e-10))) * math.cos(2.0 * math.pi * u2)
    return mean + std * z


class SoilMoistureSimulator:
    """
    Simulates soil moisture dynamics for one episode.

    Parameters
    ----------
    task_seed : int
        Seed for reproducibility.
    num_fields : int
        Number of fields (1 for easy/medium, 3 for hard).
    crop_types : list[str]
        Crop type per field.
    initial_moisture : list[float]
        Starting moisture per field.
    noisy : bool
        Whether sensor readings include Gaussian noise.
    water_budget : float | None
        Total water budget (hard task only).
    """

    def __init__(
        self,
        task_seed: int,
        num_fields: int,
        crop_types: List[str],
        initial_moisture: List[float],
        noisy: bool = False,
        water_budget: Optional[float] = None,
    ):
        self.rng = random.Random(task_seed)
        self.num_fields = num_fields
        self.crop_types = crop_types
        self.noisy = noisy
        self.water_budget = water_budget

        self.real_moisture: List[float] = list(initial_moisture)
        self.crop_health: List[float] = [100.0] * num_fields
        self.water_used: float = 0.0
        self.day: int = 0

        # Pre-generate weather schedule so it's deterministic
        # Each day: True = rain actually happens
        self._rain_schedule: List[bool] = []
        self._forecast_schedule: List[str] = []

    def _generate_weather(self, num_days: int) -> None:
        """Pre-generate weather and forecasts for the episode."""
        self._rain_schedule = []
        self._forecast_schedule = []
        for _ in range(num_days):
            rain = self.rng.random() < 0.25  # 25% chance of rain each day
            # Forecast is 70% accurate
            if self.rng.random() < WEATHER_ACCURACY:
                forecast = "rain" if rain else "dry"
            else:
                forecast = "dry" if rain else "rain"
            self._rain_schedule.append(rain)
            self._forecast_schedule.append(forecast)

    def reset(self, initial_moisture: List[float], num_days: int) -> None:
        """Reset simulator state for a new episode."""
        self.real_moisture = list(initial_moisture)
        self.crop_health = [100.0] * self.num_fields
        self.water_used = 0.0
        self.day = 0
        self._generate_weather(num_days)

    def get_sensor_readings(self) -> List[float]:
        """Return noisy (or clean) moisture readings for all fields."""
        readings = []
        for m in self.real_moisture:
            if self.noisy:
                noisy = m + _gauss(self.rng, 0.0, SENSOR_NOISE_STD)
                readings.append(_clamp(noisy))
            else:
                readings.append(_clamp(m))
        return readings

    def get_weather_forecast(self) -> str:
        """Return today's weather forecast string."""
        if self.day < len(self._forecast_schedule):
            return self._forecast_schedule[self.day]
        return "unknown"

    def apply_irrigation(self, field_id: int) -> Tuple[bool, str]:
        """
        Irrigate a specific field.
        Returns (success, message).
        """
        if self.water_budget is not None:
            if self.water_used + IRRIGATION_COST > self.water_budget:
                return False, "insufficient_budget"
        self.real_moisture[field_id] = _clamp(
            self.real_moisture[field_id] + IRRIGATION_AMOUNT
        )
        self.water_used += IRRIGATION_COST
        if self.water_budget is not None:
            remaining = self.water_budget - self.water_used
            return True, f"irrigated_field_{field_id}_budget_remaining_{remaining:.0f}"
        return True, f"irrigated_field_{field_id}"

    def advance_day(self) -> None:
        """
        Advance one day: apply natural moisture drop and rain events.
        Also update crop health based on moisture levels.
        """
        rain = self._rain_schedule[self.day] if self.day < len(self._rain_schedule) else False

        for i in range(self.num_fields):
            # Natural daily drop
            drop = self.rng.uniform(DAILY_DROP_MIN, DAILY_DROP_MAX)
            self.real_moisture[i] = _clamp(self.real_moisture[i] - drop)

            # Rain event
            if rain:
                rain_add = self.rng.uniform(RAIN_ADD_MIN, RAIN_ADD_MAX)
                self.real_moisture[i] = _clamp(self.real_moisture[i] + rain_add)

        self.day += 1

    def compute_health_penalties(self) -> Tuple[List[float], List[float]]:
        """
        Compute per-field health penalties based on current moisture.
        Returns (stress_flags, rot_flags) — each 1.0 if triggered, else 0.0.
        """
        stress_flags = []
        rot_flags = []
        for i, crop in enumerate(self.crop_types):
            cfg = CROP_CONFIG[crop]
            m = self.real_moisture[i]
            stress = 1.0 if m < cfg["stress_below"] else 0.0
            rot = 1.0 if m > cfg["rot_above"] else 0.0
            stress_flags.append(stress)
            rot_flags.append(rot)
        return stress_flags, rot_flags

    def apply_health_penalties(self, over_irrigated: bool, field_id: int = 0) -> None:
        """Apply health changes based on current moisture state."""
        for i, crop in enumerate(self.crop_types):
            cfg = CROP_CONFIG[crop]
            m = self.real_moisture[i]
            if m < cfg["stress_below"]:
                self.crop_health[i] = max(0.0, self.crop_health[i] - 15.0)
            if m > cfg["rot_above"]:
                self.crop_health[i] = max(0.0, self.crop_health[i] - 10.0)
        self.crop_health = [_clamp(h) for h in self.crop_health]
