"""
Inference Script — Soil Moisture Irrigation Environment
=======================================================
MANDATORY env vars:
    HF_TOKEN         Your Hugging Face / API key
    API_BASE_URL     The API endpoint for the LLM
    MODEL_NAME       The model identifier to use for inference
    IMAGE_NAME       Docker image name (if using from_docker_image)

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards formatted to 2 decimal places.
    - score formatted to 3 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string, or null if none.
    - All fields on a single line with no newlines within a line.
"""

import os
import sys
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

from soil_moisture_env.models import IrrigationAction
from soil_moisture_env.server.soil_moisture_env_environment import SoilMoistureEnvironment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("IMAGE_NAME")          # if using docker image
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional, for from_docker_image()
HF_TOKEN = os.getenv("HF_TOKEN")              # no default — mandatory per spec
API_KEY = HF_TOKEN or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "soil-moisture-irrigation"

TASKS = ["single_field_timing", "noisy_sensor", "multi_field_allocation"]
MAX_STEPS_PER_TASK = {"single_field_timing": 7, "noisy_sensor": 7, "multi_field_allocation": 10}
SUCCESS_SCORE_THRESHOLD = 0.4  # normalized score in [0, 1]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent managing crop irrigation.
    You receive soil moisture sensor readings each day and must decide:
    - "irrigate" — water the crop (uses 20 water units)
    - "wait" — do nothing today

    For the hard task (multi_field_allocation) also choose which field (0, 1, or 2) to irrigate.

    Respond ONLY with valid JSON. Examples:
      {"action": "irrigate", "amount": 20}
      {"action": "irrigate", "amount": 20, "field_id": 2}
      {"action": "wait"}

    Think about:
    - Is moisture too low? Irrigate.
    - Is moisture already high? Wait.
    - Is rain forecast? Maybe wait and save water.
    - Running low on water budget? Prioritize tomatoes > corn > wheat.
""").strip()


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, obs_text: str) -> dict:
    """Call the LLM and parse JSON action. Falls back to wait on any error."""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text},
        ],
        temperature=0.3,
        max_tokens=80,
        stream=False,
    )
    raw = (completion.choices[0].message.content or "").strip()
    # Strip markdown code fences if present
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def obs_to_text(obs_dict: dict, task_name: str) -> str:
    """Convert observation dict to a human-readable string for the LLM."""
    readings = obs_dict.get("sensor_readings", [])
    lines = [
        f"Task: {task_name}",
        f"Days remaining: {obs_dict.get('days_remaining', 0)}",
    ]
    for r in readings:
        lines.append(
            f"  Field {r.get('field_id', 0)}: moisture={r.get('raw_moisture', 0):.1f}%"
            f"  temp={r.get('temperature', 20):.1f}C"
        )
    forecast = obs_dict.get("weather_forecast")
    if forecast:
        lines.append(f"Weather forecast: {forecast}")
    budget = obs_dict.get("water_budget_remaining")
    if budget is not None:
        lines.append(f"Water budget remaining: {budget:.0f} units")
    crop = obs_dict.get("crop_type")
    if crop:
        lines.append(f"Crop: {crop}")
    last = obs_dict.get("last_action_result")
    if last:
        lines.append(f"Last action result: {last}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, env: SoilMoistureEnvironment, task_name: str) -> None:
    """
    Run a single task episode.
    [END] is ALWAYS emitted inside finally — even on exception.
    """
    max_steps = MAX_STEPS_PER_TASK.get(task_name, 10)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_name=task_name)
        obs_dict = obs.model_dump()
        done = False

        for step in range(1, max_steps + 1):
            if done:
                break

            # --- get LLM action ---
            error = None
            try:
                action_dict = call_llm(client, obs_to_text(obs_dict, task_name))
            except Exception as exc:
                action_dict = {"action": "wait"}
                error = str(exc).replace("\n", " ")[:120]

            action_str = action_dict.get("action", "wait")
            if action_str not in ("irrigate", "wait"):
                action_str = "wait"

            action = IrrigationAction(
                action=action_str,
                amount=action_dict.get("amount"),
                field_id=action_dict.get("field_id"),
            )

            # --- step environment ---
            try:
                obs = env.step(action)
                reward = float(obs.reward or 0.0)
                done = obs.done
                obs_dict = obs.model_dump()
            except Exception as exc:
                reward = 0.0
                done = True
                error = (error or "") + str(exc).replace("\n", " ")[:80]

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Normalized cumulative score
        final_score = env.get_final_score()
        score = min(max(final_score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        error_msg = str(exc).replace("\n", " ")[:120]
        if steps_taken == 0:
            log_step(step=1, action="wait", reward=0.0, done=True, error=error_msg)
            rewards.append(0.0)
            steps_taken = 1

    finally:
        try:
            env.close() if hasattr(env, "close") else None
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable", file=sys.stderr)
        for task_name in TASKS:
            log_start(task_name, BENCHMARK, MODEL_NAME)
            log_end(False, 0, 0.0, [])
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SoilMoistureEnvironment()

    for task_name in TASKS:
        run_task(client, env, task_name)


if __name__ == "__main__":
    main()
