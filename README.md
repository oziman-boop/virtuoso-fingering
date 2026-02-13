# Virtuoso Architect — Automatic Fingering Estimator

A production-grade, rule-based system that assigns **hand** (`L` / `R`) and
**finger number** (1–5) to every NOTE_ON event in a piano MIDI file.

---

## Project Goal

Given a solo-piano MIDI file, Virtuoso Architect produces a deterministic
fingering annotation for every note — optimised via dynamic programming to
minimise awkward stretches, crossings, and hand switches.

## Approach

### Rule-Based Dynamic Programming (Phase 1)

The solver models fingering as a shortest-path problem over a state graph:

| Concept | Detail |
|---|---|
| **State** | `(note_index, hand, finger)` |
| **Transition cost** | Weighted sum of stretch, crossing, repetition, hand-switch, chord, and weak-finger penalties |
| **Optimisation** | Viterbi-style forward DP with backtracking |
| **Determinism** | No randomness, no multiprocessing |

### Cost Model

All penalty weights are externalised in `configs/fingering_costs.yaml` —
no hardcoded constants. Missing keys raise explicit `ValueError` messages.

Key cost components:

- **Stretch** — penalty per semitone exceeding comfortable span
- **Crossing** — penalises higher finger on lower pitch (or vice versa)
- **Repetition** — same finger on a different pitch
- **Hand switch** — switching L ↔ R between consecutive notes
- **Chord penalty** — chords wider than one hand span
- **Weak finger** — small bias against ring / pinky fingers

---

## Quick Start

```bash
# 1. Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run environment check
python -m src.main

# 4. Launch the Streamlit UI
streamlit run app/streamlit_app.py
```

## Project Structure

```
project_root/
├── data/
│   ├── raw/               ← place MIDI files here
│   └── annotations/       ← generated annotations + ground truth
├── src/
│   ├── config.py           ← hardware/env detection
│   ├── main.py             ← CLI entry point
│   ├── fingering_engine/
│   │   ├── midi_parser.py      ← MIDI loading & note extraction
│   │   ├── feature_builder.py  ← deterministic feature computation
│   │   ├── cost_model.py       ← YAML-driven cost functions
│   │   ├── solver.py           ← DP graph-search optimiser
│   │   └── annotate.py         ← pipeline orchestrator & exporter
│   ├── ml_engine/
│   │   ├── dataset.py          ← ground-truth loading & validation
│   │   ├── evaluator.py        ← accuracy scoring vs expert data
│   │   └── trainer.py          ← coordinate-descent weight optimiser
│   └── rag_engine/
│       └── advisor.py          ← OpenAI-based practice advisor
├── app/
│   └── streamlit_app.py   ← interactive UI
├── configs/
│   └── fingering_costs.yaml
├── requirements.txt
└── README.md
```

## Using the Streamlit App

1. Launch with `streamlit run app/streamlit_app.py`
2. Upload a `.mid` or `.midi` file
3. Click **▶ Run Fingering Estimation**
4. View the summary statistics and annotation table
5. Download `annotations.json` and/or the annotated MIDI file

## Output Format

Each note produces an annotation:

```json
{
  "onset_time": 0.5,
  "pitch": 60,
  "hand": "R",
  "finger": 1
}
```

## MLOps Philosophy

- **Determinism over cleverness** — reproducible results on every run
- **Configuration over code** — all weights in YAML, not Python
- **CPU-first** — no GPU dependencies; runs on macOS Intel natively
- **Extensibility** — clear module boundaries for future ML integration

## Phase 2 — Learned Cost Weights ✅

The `src/ml_engine/` package learns optimal cost weights by evaluating the
DP solver against expert ground-truth annotations.

### How It Works

1. Place expert-annotated `*_ground_truth.json` files in `data/annotations/`
   (same schema as solver output) with matching MIDI in `data/raw/`.
2. **Coordinate descent** sweeps each of the 6 scalar weights over candidate
   multipliers, keeping the value that maximises mean note accuracy.
3. Writes the optimised weights to `configs/fingering_costs_learned.yaml`.

```bash
# Train from CLI
python -c "
from src.ml_engine.dataset import load_training_set
from src.ml_engine.trainer import train
pairs = load_training_set()
result = train(pairs)
print(f'Accuracy: {result[\"baseline_accuracy\"]:.2%} → {result[\"learned_accuracy\"]:.2%}')
"
```

### Learnable vs Frozen Parameters

| Type | Parameters |
|---|---|
| **Learnable** | `stretch_weight`, `crossing_weight`, `repetition_weight`, `hand_switch_weight`, `chord_penalty_weight`, `weak_finger_weight` |
| **Frozen** | `max_comfortable_span`, `split_pitch` |

## Future Roadmap

| Phase | Feature |
|---|---|
| **Phase 2** | ✅ Learned cost weights (coordinate descent) |
| **Phase 3** | Neural sequence models (Transformer / GRU) for fingering |
| **Phase 4** | RAG-based explanations — embed pedagogy corpus, retrieve context |
| **Phase 5** | Generation integration — compose fingering-aware scores |

## Target Environment

- macOS (Intel / x86_64)
- Python 3.12
- CPU only
- No Apple Silicon assumptions
- No Linux / WSL assumptions

## Dataset Assumptions

- **GiantMIDI-Piano** or **MAESTRO** uploaded manually
- Piano solo only
- No audio required
- No hardcoded dataset paths

---

## Fingering Advisor (RAG Phase 1)

The advisor module (`src/rag_engine/advisor.py`) uses the OpenAI API to
generate structured, pedagogically-informed explanations of fingering
decisions.

### How It Works

- **Structured contextual prompting** — a carefully designed system
  prompt requests analysis in four sections: Technical Analysis,
  Ergonomic Considerations, Practice Strategy, and Alternative Fingering.
- **No embeddings or retrieval** — Phase 1 sends the fingering context
  directly to the chat completion endpoint.
- **Deterministic** — `temperature=0` for reproducible outputs.
- **Lazy initialisation** — the OpenAI client is created only when
  `explain_fingering()` is called; no side effects on import.

### API Key Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

The key is loaded via `python-dotenv` at call time. It is never printed,
logged, or committed to version control.

### Future RAG Roadmap

| Phase | Feature |
|---|---|
| **Current** | Structured prompt-based explanation (no retrieval) |
| **Next** | Embedding-based retrieval from pedagogy corpus |
| **Planned** | Composer-specific fingering corpora |
| **Planned** | Local knowledge base (offline-capable) |
| **Planned** | Hybrid rule + LLM analysis pipeline |

---

*Virtuoso Architect — where music meets engineering.*

