"""Annotator — orchestrate the full fingering pipeline and export results.

Responsibilities:
    1. Call the MIDI parser to load and extract notes.
    2. Call the feature builder to enrich notes with features.
    3. Call the DP solver to compute optimal fingering.
    4. Save ``annotations.json`` (default: ``data/annotations/``).
    5. Optionally export an annotated MIDI file (same folder).
    6. Return the annotation data structure for programmatic use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pretty_midi

from .midi_parser import load_midi, extract_notes
from .feature_builder import build_features
from .solver import solve


def annotate(
    midi_path: str | Path,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    export_midi: bool = False,
) -> list[dict[str, Any]]:
    """Run the full fingering-estimation pipeline on a MIDI file.

    Args:
        midi_path: Path to the input ``.mid`` / ``.midi`` file.
        output_dir: Directory for output files.
            Defaults to ``data/annotations/`` relative to the project root.
        config_path: Path to the cost-config YAML.
            Defaults to ``configs/fingering_costs.yaml``.
        export_midi: If ``True``, also save an annotated MIDI file.

    Returns:
        List of annotation dicts, each containing:
            ``onset_time``, ``pitch``, ``hand``, ``finger``.
    """
    midi_path = Path(midi_path)

    # Resolve output directory
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "annotations"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Pipeline ──────────────────────────────────────────────
    midi_data = load_midi(midi_path)
    notes = extract_notes(midi_data)
    features = build_features(notes)
    annotations = solve(features, config_path=config_path)

    # ── Save annotations.json ─────────────────────────────────
    stem = midi_path.stem  # filename without extension (safe with spaces/commas)
    json_path = output_dir / f"{stem}_annotations.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(annotations, fh, indent=2, ensure_ascii=False)

    # ── Optional: export annotated MIDI ───────────────────────
    if export_midi:
        _export_annotated_midi(midi_data, annotations, output_dir, stem)

    return annotations


def _export_annotated_midi(
    midi_data: pretty_midi.PrettyMIDI,
    annotations: list[dict[str, Any]],
    output_dir: Path,
    stem: str,
) -> Path:
    """Write an annotated MIDI file with fingering encoded as lyrics.

    Each annotation is stored as a ``pretty_midi.Lyric`` event at the
    note's onset with the text ``H<hand>F<finger>`` (e.g. ``HRF2``).

    Args:
        midi_data: The original PrettyMIDI object.
        annotations: Solved fingering annotations.
        output_dir: Where to save the MIDI file.
        stem: Base filename (without extension).

    Returns:
        Path to the saved annotated MIDI file.
    """
    for ann in annotations:
        label = f"H{ann['hand']}F{ann['finger']}"
        midi_data.lyrics.append(
            pretty_midi.Lyric(text=label, time=ann["onset_time"])
        )

    midi_out_path = output_dir / f"{stem}_annotated.mid"
    midi_data.write(str(midi_out_path))
    return midi_out_path


def annotations_to_json_bytes(annotations: list[dict[str, Any]]) -> bytes:
    """Serialise annotations to UTF-8 JSON bytes (for download buttons).

    Args:
        annotations: The list of annotation dicts.

    Returns:
        UTF-8 encoded JSON bytes.
    """
    return json.dumps(annotations, indent=2, ensure_ascii=False).encode("utf-8")
