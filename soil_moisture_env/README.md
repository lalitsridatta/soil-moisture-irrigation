---
title: Soil Moisture Irrigation
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - agriculture
  - iot
  - irrigation
---

# 🌱 Soil Moisture Irrigation Environment

A real-world agricultural RL environment where an AI agent manages irrigation decisions based on IoT soil moisture sensor readings — simulating what a smart farming system actually does every day.

---

## The Key Idea

Crops need moisture to survive. Too little → the crop stresses and loses health. Too much → roots rot and the crop loses health. The agent must read noisy sensor data and decide each day: **irrigate or wait?**

This mirrors a real problem: automated irrigation systems on farms use exactly this kind of decision loop. Getting it right saves water and maximizes yield.

---

## How It Works

Each episode runs for 7–10 days. Every day:

1. The agent receives **sensor readings** (soil moisture %, temperature, weather forecast)
2. The agent picks an action: `irrigate` or `wait`
3. The environment advances one day — moisture drops naturally, rain may fall
4. The agent gets a **reward** based on crop health change

### What the numbers mean

| Field | What it tells you |
|---|---|
| `raw_moisture` | Soil moisture % from the IoT sensor (may be noisy) |
| `real_moisture` | Ground truth moisture (hidden from agent, shown in state for debugging) |
| `crop_health` | 0–100. Starts at 100. Drops when moisture is too low or too high |
| `water_used` | Total water units spent so far |
| `days_remaining` | Steps left in the episode |
| `done: true` | Episode over — calculate final score |
| `weather_forecast` | "rain" or "dry" — 70% accurate |

### Moisture dynamics

```
Every day:  moisture drops 8–12% naturally
Irrigation: moisture +25%, costs 20 water units
Rain event: moisture +15–20% (random, 25% chance per day)
Cap:        moisture always stays between 0–100%
```

### Crop stress thresholds

| Crop | Stress below | Rot above | Value weight |
|---|---|---|---|
| Corn | 30% | 85% | 0.3 |
| Wheat | 25% | 80% | 0.2 |
| Tomatoes | 35% | 80% | 0.5 |

If moisture drops below the stress threshold → `crop_health -= 15` that step  
If moisture goes above the rot threshold → `crop_health -= 10` that step

---

## The 3 Tasks

### Task 1 — `single_field_timing` (Easy)
- 1 field of corn, 7 days
- Clean sensor (no noise) — you see exactly what the moisture is
- Goal: keep moisture between 30–85%
- Random agent score: ~0.35 | Good agent score: ~1.0

### Task 2 — `noisy_sensor` (Medium)
- 1 field of corn, 7 days
- Sensor has Gaussian noise (std=15) — reading might say 60% when real is 45%
- Extra penalty for irrigating when real moisture > 70% (wasteful)
- The agent must reason under uncertainty
- Random agent score: ~0.25

### Task 3 — `multi_field_allocation` (Hard)
- 3 fields: wheat, corn, tomatoes
- 10 days, total water budget = 100 units (5 irrigations max)
- Agent chooses WHICH field to irrigate each day
- Tomato failure (health < 20) = immediate -0.4 penalty
- Weather forecast provided — save water if rain is coming
- Random agent score: ~0.20

---

## Reading the Playground

Click **Reset** to start a new episode. You'll see the first observation.  
Pick an action and click **Step** to advance one day.  
Click **Get State** to see the hidden ground truth (real moisture, crop health).

**Example of a bad run** (what you saw above):
```json
{
  "day": 7,
  "real_moisture": [22.09],   // critically low — below 30% stress threshold
  "crop_health": [10.0],      // nearly dead — lost 90 health points over 7 days
  "done": true,
  "cumulative_reward": 2.72   // low score because crop was stressed most days
}
```
The agent waited too many days without irrigating. Moisture fell to 22% (below corn's 30% threshold) and the crop took -15 health every step.

**Example of a good run:**
```json
{
  "day": 7,
  "real_moisture": [55.0],    // healthy range
  "crop_health": [100.0],     // perfect health maintained
  "done": true,
  "cumulative_reward": 3.5    // high score
}
```

---

## Step Reward Formula

```python
step_reward = (
    + 0.4 * crop_health_delta        # reward for improving/maintaining health
    - 0.3 * over_irrigation_penalty  # penalty if moisture exceeded rot threshold
    - 0.2 * stress_penalty           # penalty if moisture dropped below stress threshold
    - 0.1 * budget_waste             # penalty for overspending (hard task only)
)
# Normalized to [0, 1]
step_reward = max(0.0, min(1.0, (step_reward + 1) / 2))
```

---

## Quick Start (Python)

```python
from soil_moisture_env import IrrigationAction, IrrigationEnv

async with IrrigationEnv.from_env("Lsd45/soil-moisture-irrigation") as env:
    result = await env.reset(task_name="single_field_timing")
    obs = result.observation

    for day in range(7):
        moisture = obs.sensor_readings[0].raw_moisture
        # Simple rule: irrigate if moisture is low
        action = "irrigate" if moisture < 40 else "wait"
        result = await env.step(IrrigationAction(action=action))
        obs = result.observation
        print(f"Day {day+1}: moisture={moisture:.1f}% action={action} reward={result.reward:.2f}")
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode. Body: `{"task_name": "single_field_timing"}` |
| `/step` | POST | Take action. Body: `{"action": {"action": "irrigate", "field_id": 0}}` |
| `/state` | GET | Get full debug state (real moisture, crop health) |
| `/health` | GET | Health check — returns `{"status": "healthy"}` |
| `/docs` | GET | Interactive Swagger UI |

---

## Baseline Scores (Qwen2.5-72B-Instruct)

| Task | Score |
|---|---|
| single_field_timing | 1.000 |
| noisy_sensor | 0.895 |
| multi_field_allocation | 0.820 |
