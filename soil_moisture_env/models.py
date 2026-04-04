# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Soil Moisture Irrigation Environment.
"""

from typing import Literal, Optional, List
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class IrrigationAction(Action):
    """Action for the irrigation environment."""
    action: Literal["irrigate", "wait"] = Field(..., description="irrigate or wait")
    amount: Optional[float] = Field(None, description="water units (hard task only)")
    field_id: Optional[int] = Field(None, description="which field to irrigate (hard task: 0-2)")


class SensorReading(Observation):
    """A single IoT sensor reading from one field."""
    raw_moisture: float = Field(..., description="0-100%, may include noise")
    temperature: float = Field(..., description="celsius")
    day: int = Field(..., description="current day")
    field_id: int = Field(default=0, description="0 for easy/medium, 0-2 for hard")

    # satisfy Observation base
    done: bool = Field(default=False)
    reward: Optional[float] = Field(default=None)


class IrrigationObservation(Observation):
    """What the agent sees each step."""
    sensor_readings: List[SensorReading] = Field(..., description="per-field sensor readings")
    weather_forecast: Optional[str] = Field(None, description="rain, dry, or unknown")
    water_budget_remaining: Optional[float] = Field(None, description="units left (hard task)")
    crop_type: Optional[str] = Field(None, description="crop type for single-field tasks")
    days_remaining: int = Field(..., description="steps left in episode")
    last_action_result: Optional[str] = Field(None, description="feedback from last action")


class IrrigationState(State):
    """Full internal state — for debugging only, never sent to agent."""
    task_name: str = Field(..., description="which task is running")
    day: int = Field(..., description="current day index")
    real_moisture: List[float] = Field(..., description="ground truth moisture per field")
    crop_health: List[float] = Field(..., description="0-100 per field")
    water_used: float = Field(..., description="total water consumed")
    done: bool = Field(default=False)
    cumulative_reward: float = Field(default=0.0)
