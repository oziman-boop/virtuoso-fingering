# 🎹 Virtuoso Architect — Automatic Piano Fingering Estimator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://virtuoso-fingering-u9dcrxe6kyeg8ndejv7f37.streamlit.app)

> **Live Demo →** [virtuoso-fingering-u9dcrxe6kyeg8ndejv7f37.streamlit.app](https://virtuoso-fingering-u9dcrxe6kyeg8ndejv7f37.streamlit.app)

A production-grade system that assigns **hand** (`L` / `R`) and **finger number** (1–5) to every note in a piano MIDI file — optimised via dynamic programming to minimise awkward stretches, crossings, and hand switches.

---

## ✨ Features

- **Upload any piano MIDI** and get instant fingering annotations
- **Dynamic-programming solver** finds globally optimal finger assignments
- **Configurable cost model** — all penalty weights live in YAML, not code
- **Learned cost weights** via coordinate-descent optimisation against expert data
- **AI-powered practice advisor** using OpenAI GPT-4o-mini for pedagogical explanations
- **Export** annotations as JSON or annotated MIDI files
- **Interactive Streamlit UI** deployed live on the cloud

---

## 🏗️ Architecture Overview

The system is composed of three modular engines, each with a single responsibility:

```
┌──────────────────────────────────────────────────────────────┐
│                      Streamlit UI                            │
│              (app/streamlit_app.py)                           │
└──────────────┬────────────────────────────┬──────────────────┘
               │                            │
    ┌──────────▼──────────┐      ┌──────────▼──────────┐
    │  Fingering Engine   │      │    RAG Engine        │
    │  (rule-based DP)    │      │  (OpenAI advisor)    │
    └──────────┬──────────┘      └─────────────────────┘
               │
    ┌──────────▼──────────┐
    │    ML Engine         │
    │  (learned weights)   │
    └─────────────────────┘
```

### 1. Fingering Engine (`src/fingering_engine/`)

The core of the project. It models fingering as a **shortest-path problem** over a state graph and solves it with Viterbi-style dynamic programming.

| Module | Purpose |
|---|---|
| `midi_parser.py` | Loads MIDI via `pretty_midi`, extracts sorted note events |
| `feature_builder.py` | Computes deterministic features (intervals, chord sizes, hand regions) |
| `cost_model.py` | YAML-driven cost functions — stretch, crossing, repetition, hand-switch, chord, weak-finger penalties |
| `solver.py` | DP graph-search optimiser with backtracking |
| `annotate.py` | Pipeline orchestrator — wires parser → features → solver → export |

**How the solver works:**

1. Each note is assigned a **state** `(note_index, hand, finger)` where hand ∈ {L, R} and finger ∈ {1, 2, 3, 4, 5}
2. **Transition costs** are computed between consecutive states using the cost model
3. The DP **forward pass** evaluates all 10×10 state transitions per note pair
4. **Backtracking** recovers the minimum-cost path through the entire piece
5. Result: a deterministic, globally optimal fingering assignment

### 2. ML Engine (`src/ml_engine/`)

Learns optimal cost weights by evaluating the DP solver against expert ground-truth annotations, replacing hand-tuned values with data-driven ones.

| Module | Purpose |
|---|---|
| `dataset.py` | Loads and validates ground-truth `*_ground_truth.json` files paired with MIDI |
| `trainer.py` | Coordinate-descent optimiser — sweeps each weight to maximise note accuracy |
| `evaluator.py` | Scores solver output against expert annotations |

**Learnable vs Frozen Parameters:**

| Type | Parameters |
|---|---|
| **Learnable** | `stretch_weight`, `crossing_weight`, `repetition_weight`, `hand_switch_weight`, `chord_penalty_weight`, `weak_finger_weight` |
| **Frozen** | `max_comfortable_span` (finger pair limits), `split_pitch` (hand assignment heuristic) |

### 3. RAG Engine (`src/rag_engine/`)

Uses OpenAI's GPT-4o-mini to generate structured, pedagogically-informed explanations of fingering decisions.

| Feature | Detail |
|---|---|
| **Structured prompting** | System prompt requests four-section analysis: Technical Analysis, Ergonomic Considerations, Practice Strategy, Alternative Fingering |
| **Deterministic** | `temperature=0` for reproducible outputs |
| **Lazy initialisation** | OpenAI client created only when called; no side effects on import |
| **Secure** | API key loaded from `.env` via `python-dotenv`, never logged or committed |

---

## 📂 Project Structure

```
virtuoso-fingering/
├── app/
│   └── streamlit_app.py         ← Interactive web UI
├── configs/
│   └── fingering_costs.yaml     ← All penalty weights (no hardcoded constants)
├── src/
│   ├── config.py                ← Hardware/environment detection
│   ├── main.py                  ← CLI entry point
│   ├── fingering_engine/
│   │   ├── midi_parser.py       ← MIDI loading & note extraction
│   │   ├── feature_builder.py   ← Deterministic feature computation
│   │   ├── cost_model.py        ← YAML-driven cost functions
│   │   ├── solver.py            ← DP graph-search optimiser
│   │   └── annotate.py          ← Pipeline orchestrator & exporter
│   ├── ml_engine/
│   │   ├── dataset.py           ← Ground-truth loading & validation
│   │   ├── evaluator.py         ← Accuracy scoring vs expert data
│   │   └── trainer.py           ← Coordinate-descent weight optimiser
│   └── rag_engine/
│       └── advisor.py           ← OpenAI-based practice advisor
├── data/
│   ├── raw/                     ← Place MIDI files here
│   └── annotations/             ← Generated & ground-truth annotations
├── .streamlit/
│   └── config.toml              ← Streamlit theme & deployment config
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/oziman-boop/virtuoso-fingering.git
cd virtuoso-fingering

# 2. Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit UI
streamlit run app/streamlit_app.py
```

### Use the Live App

Simply visit the deployed app — no installation required:

👉 **[virtuoso-fingering-u9dcrxe6kyeg8ndejv7f37.streamlit.app](https://virtuoso-fingering-u9dcrxe6kyeg8ndejv7f37.streamlit.app)**

1. Upload a `.mid` or `.midi` file (piano solo works best)
2. Click **▶ Run Fingering Estimation**
3. View summary stats and the full annotation table
4. Download `annotations.json` and/or the annotated MIDI file

---

## 📊 Cost Model Configuration

All penalty weights are externalised in `configs/fingering_costs.yaml` — the cost model validates every key on startup and raises explicit errors for missing entries.

| Cost Component | Weight | Description |
|---|---|---|
| **Stretch** | `2.0` | Penalty per semitone exceeding comfortable span for a finger pair |
| **Crossing** | `4.0` | Higher finger on lower pitch (or vice versa) within the same hand |
| **Repetition** | `5.0` | Same finger on consecutive different pitches |
| **Hand switch** | `3.0` | Switching between L ↔ R between consecutive notes |
| **Chord penalty** | `1.5` | Per-note penalty when a chord exceeds 5 simultaneous notes |
| **Weak finger** | `1.0` | Small bias against ring (4) and pinky (5) fingers |

Comfortable span limits are defined per finger pair — for example, thumb-to-pinky (1→5) allows up to 10 semitones before penalties apply.

---

## 📤 Output Format

Each note produces a structured annotation:

```json
{
  "onset_time": 0.5,
  "pitch": 60,
  "hand": "R",
  "finger": 1
}
```

- **onset_time** — note start time in seconds
- **pitch** — MIDI pitch number (60 = Middle C)
- **hand** — `"L"` (left) or `"R"` (right)
- **finger** — 1 (thumb) through 5 (pinky)

---

## 🧠 Training Learned Weights

To optimise cost weights against expert-annotated data:

1. Place expert-annotated `*_ground_truth.json` files in `data/annotations/` with matching MIDI files in `data/raw/`
2. Run the trainer:

```bash
python -c "
from src.ml_engine.dataset import load_training_set
from src.ml_engine.trainer import train
pairs = load_training_set()
result = train(pairs)
print(f'Accuracy: {result[\"baseline_accuracy\"]:.2%} → {result[\"learned_accuracy\"]:.2%}')
"
```

3. Optimised weights are written to `configs/fingering_costs_learned.yaml`

---

## 🔮 Roadmap

| Phase | Feature | Status |
|---|---|---|
| **Phase 1** | Rule-based DP fingering engine | ✅ Complete |
| **Phase 2** | Learned cost weights (coordinate descent) | ✅ Complete |
| **Phase 3** | Neural sequence models (Transformer / GRU) | 🔜 Planned |
| **Phase 4** | RAG-based explanations with pedagogy corpus | 🔜 Planned |
| **Phase 5** | Fingering-aware score generation | 🔜 Planned |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.12** | Core language |
| **pretty_midi** | MIDI file parsing and manipulation |
| **NumPy / Pandas** | Numerical computation and data handling |
| **PyYAML** | Cost configuration loading |
| **Streamlit** | Interactive web UI and cloud deployment |
| **OpenAI API** | GPT-4o-mini for fingering explanations |
| **python-dotenv** | Secure API key management |

---

## 🎯 Design Philosophy

- **Determinism over cleverness** — reproducible results on every run, no randomness
- **Configuration over code** — all weights in YAML, validated at startup
- **CPU-first** — no GPU dependencies; runs anywhere
- **Extensibility** — clean module boundaries for ML and RAG integration
- **Production-grade** — explicit error messages, no silent failures

---

## 📄 License

This project is for educational and research purposes.

---

*Virtuoso Architect — where music meets engineering.* 🎹
