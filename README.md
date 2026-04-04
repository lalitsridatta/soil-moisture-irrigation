# 🌱 Soil Moisture Irrigation Environment

A real-world agricultural reinforcement learning environment built for the [Meta PyTorch OpenEnv Hackathon](https://huggingface.co/spaces/Lsd45/soil-moisture-irrigation). An AI agent manages crop irrigation decisions based on IoT soil moisture sensor readings — the same problem smart farming systems solve every day.

**Live demo:** https://huggingface.co/spaces/Lsd45/soil-moisture-irrigation

---

## Why This Was Built

Most RL environments are games or toys. This one simulates a real problem:

> A farm has soil moisture sensors in each field. Every day, an automated system must decide — irrigate or wait? Too little water and crops stress. Too much and roots rot. The agent must learn to read sensor data, account for weather forecasts, and manage a limited water budget across multiple fields.

This is exactly what precision agriculture systems do. Training an AI agent on this environment produces a model that can make better irrigation decisions than simple threshold-based rules.

---

## How It Works

Each episode runs for 7–10 days. Every day:

1. Agent receives **sensor readings** — soil moisture %, temperature, weather forecast
2. Agent picks an action: `irrigate` or `wait` (hard task: also picks which field)
3. Environment advances one day — moisture drops naturally, rain may fall
4. Agent gets a **reward** based on crop health change

### Moisture Dynamics

```
Every day:    moisture drops 8–12% naturally (random, seeded)
Irrigation:   moisture +25%, costs 20 water units
Rain event:   moisture +15–20% (25% chance/day, forecast 70% accurate)
Range:        moisture always clamped to 0–100%
```

### Crop Thresholds

| Crop | Stress below | Rot above | Value weight |
|---|---|---|---|
| Corn | 30% | 85% | 0.3 |
| Wheat | 25% | 80% | 0.2 |
| Tomatoes | 35% | 80% | 0.5 |

- Moisture below stress threshold → `crop_health -= 15` that step
- Moisture above rot threshold → `crop_health -= 10` that step

### Step Reward Formula

```python
step_reward = (
    + 0.4 * crop_health_delta        # reward for maintaining/improving health
    - 0.3 * over_irrigation_penalty  # penalty if moisture exceeded rot threshold
    - 0.2 * stress_penalty           # penalty if moisture dropped below stress threshold
    - 0.1 * budget_waste             # penalty for overspending (hard task only)
)
# Normalized to [0, 1]
step_reward = max(0.0, min(1.0, (step_reward + 1) / 2))
```

---

## The 3 Tasks

### Task 1 — `single_field_timing` (Easy)
- 1 field of corn, 7 days, clean sensor (no noise)
- You see exactly what the moisture is
- Goal: keep moisture between 30–85%
- Random agent score: ~0.35

### Task 2 — `noisy_sensor` (Medium)
- 1 field of corn, 7 days, sensor noise = Gaussian(0, std=15)
- Reading might say 60% when real moisture is 45%
- Extra penalty for irrigating when real moisture > 70% (wasteful)
- Agent must reason under uncertainty
- Random agent score: ~0.25

### Task 3 — `multi_field_allocation` (Hard)
- 3 fields: wheat, corn, tomatoes
- 10 days, total water budget = 100 units (5 irrigations max)
- Agent chooses which field to irrigate each day
- Tomato failure (health < 20) = immediate −0.4 penalty
- Weather forecast provided — save water if rain is coming
- Random agent score: ~0.20

---

## Project Structure

```
soil-moisture-irrigation/
├── inference.py                          ← Run the AI agent against all 3 tasks
├── soil_moisture_env/
│   ├── models.py                         ← IrrigationAction, IrrigationObservation, IrrigationState
│   ├── client.py                         ← WebSocket client for connecting to the server
│   ├── simulator.py                      ← Synthetic soil moisture dynamics (seeded, deterministic)
│   ├── tasks.py                          ← Task configs + grader score functions
│   ├── openenv.yaml                      ← OpenEnv manifest
│   ├── pyproject.toml                    ← Package metadata
│   └── server/
│       ├── soil_moisture_env_environment.py  ← reset(), step(), state() logic
│       ├── app.py                        ← FastAPI server
│       └── Dockerfile                    ← Container (exposes port 7860)
```

---

## How to Test It

### Option 1 — Live Playground (no setup needed)

Go to https://huggingface.co/spaces/Lsd45/soil-moisture-irrigation

1. Click **Reset** — starts a new episode, returns first sensor reading
2. Pick action (`irrigate` or `wait`) + optional Field Id → click **Step**
3. Watch `crop_health`, `real_moisture`, `reward` change each day
4. Click **Get State** to see the hidden ground truth
5. Episode ends when `done: true` (after 7 or 10 steps)

### Option 2 — Run Locally

```bash
# Clone and install
git clone https://github.com/lalitsridatta/soil-moisture-irrigation
cd soil-moisture-irrigation
pip install openenv-core fastapi uvicorn openai

# Start the server
python -m uvicorn soil_moisture_env.server.app:app --host 0.0.0.0 --port 7860

# Open Swagger UI in browser
# http://localhost:7860/docs
```

### Option 3 — Run the AI Agent

```bash
# Set your HuggingFace token
export HF_TOKEN="hf_your_token_here"   # Linux/Mac
$env:HF_TOKEN="hf_your_token_here"     # Windows PowerShell

python inference.py
```

Expected output:
```
[START] task=single_field_timing env=soil-moisture-irrigation model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=irrigate reward=0.50 done=false error=null
[STEP] step=2 action=wait reward=0.50 done=false error=null
...
[END] success=true steps=7 score=1.000 rewards=0.50,0.50,...
```

### Option 4 — Python Client

```python
import asyncio
from soil_moisture_env import IrrigationAction, SoilMoistureEnv

async def main():
    async with SoilMoistureEnv(base_url="http://localhost:7860") as env:
        result = await env.reset(task_name="single_field_timing")
        obs = result.observation

        for day in range(7):
            moisture = obs.sensor_readings[0].raw_moisture
            action = "irrigate" if moisture < 40 else "wait"
            result = await env.step(IrrigationAction(action=action))
            obs = result.observation
            print(f"Day {day+1}: moisture={moisture:.1f}% action={action} reward={result.reward:.2f} done={result.done}")

asyncio.run(main())
```

---

## Understanding the Response

When you call `/state` or click **Get State**, you see:

```json
{
  "task_name": "single_field_timing",
  "day": 3,
  "real_moisture": [42.5],    // ground truth — hidden from agent in noisy tasks
  "crop_health": [85.0],      // starts at 100, drops when moisture is wrong
  "water_used": 20.0,         // units spent so far
  "done": false,              // true when episode ends
  "cumulative_reward": 1.5    // total reward accumulated
}
```

A **good agent** keeps `crop_health` close to 100 by irrigating before moisture drops below the stress threshold, and avoids over-irrigating above the rot threshold.

---

## API Endpoints

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/reset` | POST | `{"task_name": "single_field_timing"}` | Start new episode |
| `/step` | POST | `{"action": {"action": "irrigate", "field_id": 0}}` | Take one action |
| `/state` | GET | — | Full debug state |
| `/health` | GET | — | Returns `{"status": "healthy"}` |
| `/docs` | GET | — | Interactive Swagger UI |

---

## Baseline Scores (Qwen/Qwen2.5-72B-Instruct)

| Task | Difficulty | Score |
|---|---|---|
| single_field_timing | Easy | 1.000 |
| noisy_sensor | Medium | 0.895 |
| multi_field_allocation | Hard | 0.820 |

---

## Built With

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — RL environment framework
- [FastAPI](https://fastapi.tiangolo.com/) — HTTP + WebSocket server
- [Pydantic](https://docs.pydantic.dev/) — typed models
- [HuggingFace Spaces](https://huggingface.co/spaces) — deployment
