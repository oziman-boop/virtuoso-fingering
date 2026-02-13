"""Evaluator — score solver output against expert ground-truth annotations.

Provides three granularity levels:
    - ``note_accuracy``   : both hand and finger must match
    - ``hand_accuracy``   : only the hand assignment is checked
    - ``finger_accuracy`` : finger checked only where the hand is correct

Plus a convenience function ``evaluate_config`` that runs the full
feature → solver pipeline with a given YAML config and returns all metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def note_accuracy(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> float:
    """Fraction of notes where *both* hand and finger match.

    Args:
        predicted: Solver output (list of annotation dicts).
        ground_truth: Expert annotations (same schema).

    Returns:
        Accuracy in [0.0, 1.0]. Returns 0.0 on empty input.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if not predicted and not ground_truth:
        return 0.0
    if len(predicted) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted)}, "
            f"ground_truth={len(ground_truth)}"
        )

    correct = sum(
        1
        for p, g in zip(predicted, ground_truth)
        if p["hand"] == g["hand"] and p["finger"] == g["finger"]
    )
    return correct / len(ground_truth)


def hand_accuracy(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> float:
    """Fraction of notes where the hand assignment is correct.

    Args:
        predicted: Solver output.
        ground_truth: Expert annotations.

    Returns:
        Accuracy in [0.0, 1.0].
    """
    if not predicted and not ground_truth:
        return 0.0
    if len(predicted) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted)}, "
            f"ground_truth={len(ground_truth)}"
        )

    correct = sum(
        1 for p, g in zip(predicted, ground_truth) if p["hand"] == g["hand"]
    )
    return correct / len(ground_truth)


def finger_accuracy(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> float:
    """Fraction of correct finger assignments *among notes with correct hand*.

    Notes where the hand is wrong are excluded from the denominator,
    since finger comparison is meaningless across hands.

    Args:
        predicted: Solver output.
        ground_truth: Expert annotations.

    Returns:
        Accuracy in [0.0, 1.0]. Returns 0.0 if no notes have the
        correct hand assignment.
    """
    if not predicted and not ground_truth:
        return 0.0
    if len(predicted) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted)}, "
            f"ground_truth={len(ground_truth)}"
        )

    hand_correct_pairs = [
        (p, g)
        for p, g in zip(predicted, ground_truth)
        if p["hand"] == g["hand"]
    ]

    if not hand_correct_pairs:
        return 0.0

    finger_correct = sum(
        1 for p, g in hand_correct_pairs if p["finger"] == g["finger"]
    )
    return finger_correct / len(hand_correct_pairs)


def evaluate_config(
    midi_path: str | Path,
    ground_truth: list[dict[str, Any]],
    config_path: str | Path,
) -> dict[str, float]:
    """Run the full pipeline with a YAML config and score against ground truth.

    This function imports the fingering engine lazily to avoid circular
    dependencies.

    Args:
        midi_path: Path to the source MIDI file.
        ground_truth: Validated expert annotations (list of dicts).
        config_path: Path to the cost-config YAML to evaluate.

    Returns:
        A dict with keys:
            - ``note_accuracy``   (float)
            - ``hand_accuracy``   (float)
            - ``finger_accuracy`` (float)
    """
    # Lazy import to keep engine and ml_engine loosely coupled
    from src.fingering_engine.midi_parser import load_midi, extract_notes
    from src.fingering_engine.feature_builder import build_features
    from src.fingering_engine.solver import solve

    midi_data = load_midi(midi_path)
    notes = extract_notes(midi_data)
    features = build_features(notes)
    predicted = solve(features, config_path=config_path)

    # Align lengths — ground truth may be a subset (first N notes)
    # or same length. Truncate to the shorter list.
    min_len = min(len(predicted), len(ground_truth))
    if min_len == 0:
        return {
            "note_accuracy": 0.0,
            "hand_accuracy": 0.0,
            "finger_accuracy": 0.0,
        }

    pred_trimmed = predicted[:min_len]
    gt_trimmed = ground_truth[:min_len]

    return {
        "note_accuracy": note_accuracy(pred_trimmed, gt_trimmed),
        "hand_accuracy": hand_accuracy(pred_trimmed, gt_trimmed),
        "finger_accuracy": finger_accuracy(pred_trimmed, gt_trimmed),
    }
