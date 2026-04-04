# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Soil Moisture Irrigation Environment Client."""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import IrrigationAction, IrrigationObservation, SensorReading


class SoilMoistureEnv(
    EnvClient[IrrigationAction, IrrigationObservation, State]
):
    """
    Client for the Soil Moisture Irrigation Environment.

    Example:
        >>> with SoilMoistureEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_name="single_field_timing")
        ...     result = client.step(IrrigationAction(action="irrigate"))
    """

    def _step_payload(self, action: IrrigationAction) -> Dict:
        payload: Dict = {"action": action.action}
        if action.amount is not None:
            payload["amount"] = action.amount
        if action.field_id is not None:
            payload["field_id"] = action.field_id
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[IrrigationObservation]:
        obs_data = payload.get("observation", {})
        raw_readings = obs_data.get("sensor_readings", [])
        sensor_readings = [
            SensorReading(
                raw_moisture=r.get("raw_moisture", 50.0),
                temperature=r.get("temperature", 20.0),
                day=r.get("day", 0),
                field_id=r.get("field_id", 0),
                done=r.get("done", False),
                reward=r.get("reward"),
            )
            for r in raw_readings
        ]
        observation = IrrigationObservation(
            sensor_readings=sensor_readings,
            weather_forecast=obs_data.get("weather_forecast"),
            water_budget_remaining=obs_data.get("water_budget_remaining"),
            crop_type=obs_data.get("crop_type"),
            days_remaining=obs_data.get("days_remaining", 0),
            last_action_result=obs_data.get("last_action_result"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
