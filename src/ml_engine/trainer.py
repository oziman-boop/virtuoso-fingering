"""Trainer — coordinate-descent optimiser for fingering cost weights.

Learns the optimal values for the six scalar weights in
``fingering_costs.yaml`` by evaluating the DP solver against
expert ground-truth annotations.

Algorithm:
    1. Start from the baseline YAML weights.
    2. For each weight, sweep over candidate multipliers.
    3. Keep the value that maximises mean note accuracy across
       all training examples.
    4. Repeat for ``max_rounds`` (default 3) until convergence.

Design:
    - Deterministic, single-threaded, CPU-only.
    - Numpy-only (no scipy required).
    - Frozen parameters: ``max_comfortable_span``, ``split_pitch``.
    - Learnable parameters: 6 scalar weights.
    - Writes the optimised config to a new YAML file.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .evaluator import evaluate_config


# ── Learnable weight keys (order matters for reproducibility) ─
_WEIGHT_KEYS: list[str] = [
    "stretch_weight",
    "crossing_weight",
    "repetition_weight",
    "hand_switch_weight",
    "chord_penalty_weight",
    "weak_finger_weight",
]

# ── Candidate multipliers for coordinate descent ─────────────
_MULTIPLIERS: list[float] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


def _write_temp_config(cfg: dict[str, Any]) -> Path:
    """Write a config dict to a temporary YAML file and return its path.

    The caller is responsible for cleanup (or relies on OS temp cleanup).

    Args:
        cfg: Full cost-config dictionary.

    Returns:
        Path to the temporary YAML file.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return Path(tmp.name)


def _mean_accuracy(
    training_pairs: list[dict[str, Any]],
    config_path: Path,
) -> float:
    """Compute the mean note accuracy across all training pairs.

    Args:
        training_pairs: Output of :func:`dataset.load_training_set`.
        config_path: Path to the YAML config being evaluated.

    Returns:
        Mean note accuracy in [0.0, 1.0].
    """
    accuracies: list[float] = []
    for pair in training_pairs:
        metrics = evaluate_config(
            midi_path=pair["midi_path"],
            ground_truth=pair["ground_truth"],
            config_path=config_path,
        )
        accuracies.append(metrics["note_accuracy"])

    if not accuracies:
        return 0.0
    return float(np.mean(accuracies))


def train(
    training_pairs: list[dict[str, Any]],
    base_config_path: str | Path | None = None,
    output_config_path: str | Path | None = None,
    max_rounds: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """Optimise cost weights via coordinate descent.

    For each of the 6 learnable weights, tries each candidate multiplier
    and keeps the value that yields the highest mean note accuracy.
    Repeats for ``max_rounds``.

    Args:
        training_pairs: List of training dicts from
            :func:`dataset.load_training_set`. Each must have
            ``midi_path`` (Path) and ``ground_truth`` (list).
        base_config_path: Path to the baseline YAML config.
            Defaults to ``configs/fingering_costs.yaml``.
        output_config_path: Where to write the optimised YAML.
            Defaults to ``configs/fingering_costs_learned.yaml``.
        max_rounds: Number of full sweeps over all weights.
        verbose: If True, print progress to stdout.

    Returns:
        A dict with:
            - ``config``           (dict): optimised config
            - ``baseline_accuracy`` (float): accuracy with original weights
            - ``learned_accuracy``  (float): accuracy after optimisation
            - ``output_path``       (Path): where the config was saved

    Raises:
        FileNotFoundError: If ``base_config_path`` does not exist.
        ValueError: If ``training_pairs`` is empty.
    """
    if not training_pairs:
        raise ValueError(
            "No training pairs provided. "
            "Place *_ground_truth.json files in data/annotations/ "
            "with matching MIDI files in data/raw/."
        )

    # ── Resolve paths ─────────────────────────────────────────
    project_root = Path(__file__).resolve().parents[2]

    if base_config_path is None:
        base_config_path = project_root / "configs" / "fingering_costs.yaml"
    else:
        base_config_path = Path(base_config_path)

    if output_config_path is None:
        output_config_path = project_root / "configs" / "fingering_costs_learned.yaml"
    else:
        output_config_path = Path(output_config_path)

    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    # ── Load baseline ─────────────────────────────────────────
    with open(base_config_path, "r", encoding="utf-8") as fh:
        base_cfg: dict[str, Any] = yaml.safe_load(fh)

    # Score baseline
    baseline_accuracy = _mean_accuracy(training_pairs, base_config_path)
    if verbose:
        print(f"Baseline note accuracy: {baseline_accuracy:.4f}")

    # ── Coordinate descent ────────────────────────────────────
    best_cfg = copy.deepcopy(base_cfg)
    best_accuracy = baseline_accuracy

    for round_idx in range(max_rounds):
        improved_this_round = False

        for key in _WEIGHT_KEYS:
            original_value = float(best_cfg[key])
            best_value_for_key = original_value
            best_acc_for_key = best_accuracy

            if verbose:
                print(
                    f"  Round {round_idx + 1}/{max_rounds} | "
                    f"Tuning {key} (current={original_value:.3f})"
                )

            for mult in _MULTIPLIERS:
                candidate_value = original_value * mult

                # Skip zero or negative weights
                if candidate_value <= 0:
                    continue

                # Build trial config
                trial_cfg = copy.deepcopy(best_cfg)
                trial_cfg[key] = round(candidate_value, 6)
                trial_path = _write_temp_config(trial_cfg)

                try:
                    acc = _mean_accuracy(training_pairs, trial_path)
                finally:
                    # Clean up temp file
                    trial_path.unlink(missing_ok=True)

                if acc > best_acc_for_key:
                    best_acc_for_key = acc
                    best_value_for_key = candidate_value

            # Update if improvement found
            if best_value_for_key != original_value:
                best_cfg[key] = round(best_value_for_key, 6)
                best_accuracy = best_acc_for_key
                improved_this_round = True
                if verbose:
                    print(
                        f"    → {key}: {original_value:.3f} → "
                        f"{best_value_for_key:.3f} (acc={best_accuracy:.4f})"
                    )

        if verbose:
            print(f"  Round {round_idx + 1} done — accuracy: {best_accuracy:.4f}")

        if not improved_this_round:
            if verbose:
                print("  No improvement — stopping early.")
            break

    # ── Write optimised config ────────────────────────────────
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w", encoding="utf-8") as fh:
        fh.write(
            "# ────────────────────────────────────────────────────────────────\n"
            "# Virtuoso Architect — Learned Fingering Cost Configuration\n"
            "# ────────────────────────────────────────────────────────────────\n"
            "# Generated by src/ml_engine/trainer.py (coordinate descent).\n"
            f"# Baseline accuracy : {baseline_accuracy:.4f}\n"
            f"# Learned accuracy  : {best_accuracy:.4f}\n"
            "# ────────────────────────────────────────────────────────────────\n\n"
        )
        yaml.dump(best_cfg, fh, default_flow_style=False, allow_unicode=True)

    if verbose:
        print(f"\nOptimised config saved to: {output_config_path}")
        print(f"  Baseline accuracy: {baseline_accuracy:.4f}")
        print(f"  Learned accuracy:  {best_accuracy:.4f}")
        print(f"  Improvement:       {best_accuracy - baseline_accuracy:+.4f}")

    return {
        "config": best_cfg,
        "baseline_accuracy": baseline_accuracy,
        "learned_accuracy": best_accuracy,
        "output_path": output_config_path,
    }
