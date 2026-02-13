"""Dataset — load and validate ground-truth fingering annotations.

Ground-truth files follow the same JSON schema as the solver output::

    [
      {"onset_time": 0.0, "pitch": 60, "hand": "R", "finger": 1},
      {"onset_time": 0.5, "pitch": 64, "hand": "R", "finger": 3},
      ...
    ]

Files must be named ``*_ground_truth.json`` and placed in the
annotations directory (default ``data/annotations/``).
Each ground-truth file is paired with a MIDI file of the same stem
in ``data/raw/`` (e.g. ``sonata_ground_truth.json`` ↔ ``sonata.mid``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ── Validation constants ──────────────────────────────────────
_VALID_HANDS: set[str] = {"L", "R"}
_VALID_FINGERS: set[int] = {1, 2, 3, 4, 5}


def load_ground_truth(json_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a single ground-truth annotation file.

    Args:
        json_path: Path to a ``*_ground_truth.json`` file.

    Returns:
        List of validated annotation dicts, each containing:
            ``onset_time`` (float), ``pitch`` (int),
            ``hand`` (str), ``finger`` (int).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any annotation entry fails validation.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(
            f"Ground-truth file must contain a JSON array, got {type(data).__name__}: {path}"
        )

    validated: list[dict[str, Any]] = []
    for i, entry in enumerate(data):
        # ── Required keys ─────────────────────────────────────
        for key in ("onset_time", "pitch", "hand", "finger"):
            if key not in entry:
                raise ValueError(
                    f"Entry {i} in '{path.name}' is missing required key '{key}'"
                )

        hand = str(entry["hand"]).upper()
        finger = int(entry["finger"])

        if hand not in _VALID_HANDS:
            raise ValueError(
                f"Entry {i} in '{path.name}': hand must be 'L' or 'R', got '{hand}'"
            )
        if finger not in _VALID_FINGERS:
            raise ValueError(
                f"Entry {i} in '{path.name}': finger must be 1–5, got {finger}"
            )

        validated.append(
            {
                "onset_time": float(entry["onset_time"]),
                "pitch": int(entry["pitch"]),
                "hand": hand,
                "finger": finger,
            }
        )

    return validated


def load_training_set(
    annotations_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Discover all ground-truth / MIDI pairs for training.

    Scans ``annotations_dir`` for files matching ``*_ground_truth.json``
    and pairs each with the corresponding MIDI in ``raw_dir``.

    Args:
        annotations_dir: Directory containing ground-truth JSON files.
            Defaults to ``data/annotations/`` relative to project root.
        raw_dir: Directory containing source MIDI files.
            Defaults to ``data/raw/`` relative to project root.

    Returns:
        A list of dicts, each with:
            - ``midi_path``       (Path): absolute path to the MIDI file
            - ``ground_truth``    (list): validated annotation list
            - ``stem``            (str):  base filename stem

    Raises:
        FileNotFoundError: If the annotations or raw directory does not exist,
            or a matching MIDI file cannot be found.
    """
    project_root = Path(__file__).resolve().parents[2]

    if annotations_dir is None:
        annotations_dir = project_root / "data" / "annotations"
    else:
        annotations_dir = Path(annotations_dir)

    if raw_dir is None:
        raw_dir = project_root / "data" / "raw"
    else:
        raw_dir = Path(raw_dir)

    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw MIDI directory not found: {raw_dir}")

    # Discover ground-truth files
    gt_files = sorted(annotations_dir.glob("*_ground_truth.json"))
    if not gt_files:
        raise FileNotFoundError(
            f"No *_ground_truth.json files found in: {annotations_dir}"
        )

    pairs: list[dict[str, Any]] = []
    for gt_path in gt_files:
        # Derive stem: "sonata_ground_truth.json" → "sonata"
        stem = gt_path.name.replace("_ground_truth.json", "")

        # Find matching MIDI (try .mid and .midi)
        midi_path: Path | None = None
        for ext in (".mid", ".midi"):
            candidate = raw_dir / f"{stem}{ext}"
            if candidate.exists():
                midi_path = candidate
                break

        if midi_path is None:
            raise FileNotFoundError(
                f"No matching MIDI file for '{gt_path.name}' "
                f"in {raw_dir} (tried {stem}.mid / {stem}.midi)"
            )

        ground_truth = load_ground_truth(gt_path)

        pairs.append(
            {
                "midi_path": midi_path,
                "ground_truth": ground_truth,
                "stem": stem,
            }
        )

    return pairs
