# 🌱 Soil Moisture Irrigation Environment

> A real-world agricultural RL environment where an AI agent manages crop irrigation using IoT soil moisture sensors — the same decision loop used in precision farming systems today.

**Live demo:** https://huggingface.co/spaces/Lsd45/soil-moisture-irrigation  
**HF Space:** https://huggingface.co/spaces/Lsd45/soil-moisture-irrigation  
**GitHub:** https://github.com/lalitsridatta/soil-moisture-irrigation

---

## The Real-World Problem

Every day on a modern farm, automated systems ask: *should we irrigate this field today?*

Too little water → crops stress and lose yield. Too much → roots rot. Weather is uncertain. Sensors are noisy. Water budgets are limited. Multiple crops compete for the same resource.

This environment simulates exactly that. An AI agent trained here learns to:
- Read noisy IoT sensor data and infer true soil conditions
- Time irrigation decisions across a multi-day episode
- Allocate scarce water across fields with different crop priorities
- Use weather forecasts (which are only 70% accurate) to plan ahead

This is not a toy. Precision irrigation is a $4B+ industry and a direct application of sequential decision-making under uncertainty.

---

## How It Works

Each episode runs for 7–10 days. Every day:

1. Agent receives **sensor readings** — soil moisture %, temperature, weather forecast
2. Agent picks: `irrigate` or `wait` (hard task: also picks which field 0/1/2)
3. Environment advances one day — moisture drops naturally, rain may fall
4. Agent receives a **step reward** based on crop health change

### Moisture Dynamics

```
Every day:    moisture drops 8-12% naturally (seeded random)
Irrigation:   moisture +25%, costs 20 water units
Rain event:   moisture +15-20% (25% chance/day)
Forecast:     70% accurate — sometimes says rain but none comes
Range:        moisture clamped to 0-100%
```

### Crop Health Rules

| Crop | Stress below | Rot above | Value weight |
|---|---|---|---|
| Corn | 30% | 85% | 0.3 |
| Wheat | 25% | 80% | 0.2 |
| Tomatoes | 35% | 80% | 0.5 |

- Moisture drops below stress threshold → `crop_health -= 15` that step
- Moisture exceeds rot threshold → `crop_health -= 10` that step
- Crop health starts at 100, never recovers once lost

### Step Reward Formula

```python
step_reward = (
    + 0.4 * crop_health_delta        # reward for maintaining/improving health
    - 0.3 * over_irrigation_penalty  # 1.0 if moisture exceeded rot threshold
    - 0.2 * stress_penalty           # 1.0 if moisture dropped below stress threshold
    - 0.1 * budget_waste             # overspend fraction (hard task only)
)
# Normalized strictly within (0.01, 0.99)
step_reward = max(0.01, min(0.99, (step_reward + 1) / 2))
```

---

## The 3 Tasks

### Task 1 — `single_field_timing` (Easy)
- 1 field of corn, 7 days, **clean sensor** (no noise)
- Agent sees exact moisture readings
- Goal: keep moisture between 30–85% across all 7 days
- Score: `crop_health / 100` at episode end
- A good agent should score 0.7+

### Task 2 — `noisy_sensor` (Medium)
- 1 field of corn, 7 days, **sensor noise = Gaussian(0, std=15)**
- Sensor might read 65% when real moisture is 45% — agent must reason under uncertainty
- Extra penalty: irrigating when real moisture > 70% counts as wasteful
- Score: `(health * 0.7) + (1 - waste_ratio) * 0.3`
- Requires uncertainty-aware decision making

### Task 3 — `multi_field_allocation` (Hard)
- 3 fields: wheat, corn, **tomatoes** (highest value crop)
- 10 days, **water budget = 60 units** (only 3 irrigations total)
- Agent must choose which field to irrigate — can't water all of them
- Tomato crop failure (health < 20) = immediate −0.4 score penalty
- Weather forecast provided — smart agents save water when rain is likely
- Score: `(tomato*0.5 + corn*0.3 + wheat*0.2)/100 + budget_efficiency - tomato_penalty`
- Frontier models score ~0.2–0.4 on this task

---

## Observation Space

```json
{
  "sensor_readings": [
    {"field_id": 0, "raw_moisture": 42.3, "temperature": 22.1, "day": 3}
  ],
  "weather_forecast": "rain",          // "rain", "dry", or "unknown"
  "water_budget_remaining": 40.0,      // null for easy/medium tasks
  "crop_type": "corn",                 // null for hard task (3 fields)
  "days_remaining": 4,
  "last_action_result": "irrigated_field_0"
}
```

## Action Space

```json
// Easy / Medium:
{"action": "irrigate"}
{"action": "wait"}

// Hard task (specify which field):
{"action": "irrigate", "field_id": 2}
{"action": "wait"}
```

---

## Project Structure

```
soil-moisture-irrigation/
├── inference.py                              <- AI agent runner (all 3 tasks)
├── soil_moisture_env/
│   ├── models.py                             <- Typed Pydantic models
│   ├── simulator.py                          <- Deterministic moisture dynamics
│   ├── tasks.py                              <- Task configs + grader functions
│   ├── client.py                             <- WebSocket client
│   ├── openenv.yaml                          <- OpenEnv manifest
│   └── server/
│       ├── soil_moisture_env_environment.py  <- reset() / step() / state()
│       ├── app.py                            <- FastAPI server
│       └── Dockerfile                        <- Port 7860
```

---

## How to Test

### Option 1 — Live Playground (zero setup)

**https://huggingface.co/spaces/Lsd45/soil-moisture-irrigation**

1. Click **Reset** → choose task via body `{"task_name": "single_field_timing"}`
2. Click **Step** → body `{"action": {"action": "irrigate", "field_id": 0}}`
3. Click **Get State** → see hidden ground truth (real moisture, crop health)
4. Repeat until `done: true`

### Option 2 — Run Locally

```bash
git clone https://github.com/lalitsridatta/soil-moisture-irrigation
cd soil-moisture-irrigation
pip install openenv-core fastapi uvicorn openai

# Start server
python -m uvicorn soil_moisture_env.server.app:app --host 0.0.0.0 --port 7860

# Swagger UI at http://localhost:7860/docs
```

### Option 3 — Run the AI Agent

```bash
export HF_TOKEN="hf_your_token_here"    # Linux/Mac
$env:HF_TOKEN="hf_your_token_here"      # Windows PowerShell

python inference.py
```

Expected output:
```
[START] task=single_field_timing env=soil-moisture-irrigation model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=irrigate reward=0.50 done=false error=null
[STEP] step=2 action=wait reward=0.50 done=false error=null
...
[END] success=true steps=7 score=0.990 rewards=0.50,0.50,...
```

---

## API Reference

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/reset` | POST | `{"task_name": "single_field_timing"}` | Start new episode |
| `/step` | POST | `{"action": {"action": "irrigate", "field_id": 0}}` | Take action |
| `/state` | GET | — | Full debug state (real moisture, crop health) |
| `/health` | GET | — | `{"status": "healthy"}` |
| `/docs` | GET | — | Interactive Swagger UI |

---

## Baseline Scores

Evaluated with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Difficulty | Score |
|---|---|---|
| `single_field_timing` | Easy | 0.990 |
| `noisy_sensor` | Medium | 0.895 |
| `multi_field_allocation` | Hard | 0.350 |

---

## Built With

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — RL environment framework
- [FastAPI](https://fastapi.tiangolo.com/) — HTTP + WebSocket server
- [Pydantic](https://docs.pydantic.dev/) — typed models
- [HuggingFace Spaces](https://huggingface.co/spaces) — deployment
