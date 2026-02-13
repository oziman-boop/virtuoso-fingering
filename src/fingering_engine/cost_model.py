"""Cost Model — configurable transition-cost functions for piano fingering.

All weights are loaded from ``configs/fingering_costs.yaml``.
No hardcoded constants: if a required key is missing the YAML,
a ``ValueError`` is raised with a clear message.

Methods:
    stretch_cost      – penalises intervals beyond comfortable span
    crossing_cost     – penalises finger crossings
    repetition_cost   – penalises same-finger repetition (different pitch)
    hand_switch_cost  – penalises changing hands between consecutive notes
    chord_penalty     – penalises chords wider than one hand span
    total_cost        – aggregated transition cost
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class FingeringCostModel:
    """Rule-based cost model for evaluating finger transitions.

    Args:
        config_path: Path to the YAML configuration file.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            # Default: configs/fingering_costs.yaml relative to project root
            config_path = Path(__file__).resolve().parents[2] / "configs" / "fingering_costs.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Cost config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as fh:
            self._cfg: dict[str, Any] = yaml.safe_load(fh)

        # Validate required top-level keys
        required_keys = [
            "max_comfortable_span",
            "stretch_weight",
            "crossing_weight",
            "repetition_weight",
            "hand_switch_weight",
            "chord_penalty_weight",
            "weak_finger_weight",
            "split_pitch",
        ]
        for key in required_keys:
            if key not in self._cfg:
                raise ValueError(
                    f"Missing required key '{key}' in cost config: {config_path}"
                )

        self.max_comfortable_span: dict[str, int] = self._cfg["max_comfortable_span"]
        self.stretch_weight: float = float(self._cfg["stretch_weight"])
        self.crossing_weight: float = float(self._cfg["crossing_weight"])
        self.repetition_weight: float = float(self._cfg["repetition_weight"])
        self.hand_switch_weight: float = float(self._cfg["hand_switch_weight"])
        self.chord_penalty_weight: float = float(self._cfg["chord_penalty_weight"])
        self.weak_finger_weight: float = float(self._cfg["weak_finger_weight"])
        self.split_pitch: int = int(self._cfg["split_pitch"])

    # ── Individual cost components ────────────────────────────

    def stretch_cost(self, finger_a: int, finger_b: int, interval: int) -> float:
        """Penalise intervals that exceed the comfortable span for a finger pair.

        Args:
            finger_a: Previous finger (1–5).
            finger_b: Current finger (1–5).
            interval: Absolute semitone distance between the two pitches.

        Returns:
            Non-negative cost.
        """
        # Same finger → no stretch (handled by repetition_cost)
        if finger_a == finger_b:
            return 0.0
        lo, hi = sorted([finger_a, finger_b])
        key = f"{lo}_{hi}"
        max_span = self.max_comfortable_span.get(key)
        if max_span is None:
            raise ValueError(
                f"Missing max_comfortable_span entry for finger pair '{key}'"
            )
        excess = interval - max_span
        if excess <= 0:
            return 0.0
        return excess * self.stretch_weight

    def crossing_cost(
        self, finger_a: int, finger_b: int, pitch_a: int, pitch_b: int
    ) -> float:
        """Penalise finger crossings (higher finger on lower pitch or vice versa).

        A crossing occurs when the finger ordering disagrees with the pitch
        ordering within the same hand.

        Args:
            finger_a: Previous finger (1–5).
            finger_b: Current finger (1–5).
            pitch_a: Previous MIDI pitch.
            pitch_b: Current MIDI pitch.

        Returns:
            Non-negative cost.
        """
        pitch_up = pitch_b > pitch_a
        finger_up = finger_b > finger_a

        # Same finger or same pitch → no crossing
        if finger_a == finger_b or pitch_a == pitch_b:
            return 0.0

        if pitch_up != finger_up:
            return self.crossing_weight
        return 0.0

    def repetition_cost(self, finger_a: int, finger_b: int, pitch_a: int, pitch_b: int) -> float:
        """Penalise using the same finger on consecutive *different* pitches.

        Repeating the same finger on the same pitch (repeated note) is free.

        Args:
            finger_a: Previous finger (1–5).
            finger_b: Current finger (1–5).
            pitch_a: Previous MIDI pitch.
            pitch_b: Current MIDI pitch.

        Returns:
            Non-negative cost.
        """
        if finger_a == finger_b and pitch_a != pitch_b:
            return self.repetition_weight
        return 0.0

    def hand_switch_cost(self, hand_a: str, hand_b: str) -> float:
        """Penalise switching between left and right hand.

        Args:
            hand_a: Previous hand (``"L"`` or ``"R"``).
            hand_b: Current hand (``"L"`` or ``"R"``).

        Returns:
            Non-negative cost.
        """
        if hand_a != hand_b:
            return self.hand_switch_weight
        return 0.0

    def chord_penalty(self, chord_size: int) -> float:
        """Penalise chords that exceed a single hand's span (> 5 notes).

        Args:
            chord_size: Number of simultaneous notes.

        Returns:
            Non-negative cost proportional to excess notes.
        """
        excess = chord_size - 5
        if excess <= 0:
            return 0.0
        return excess * self.chord_penalty_weight

    def weak_finger_cost(self, finger: int) -> float:
        """Small extra cost for using weak fingers (ring=4, pinky=5).

        Args:
            finger: Finger number (1–5).

        Returns:
            Non-negative cost.
        """
        if finger in (4, 5):
            return self.weak_finger_weight
        return 0.0

    # ── Aggregate ─────────────────────────────────────────────

    def total_cost(
        self,
        finger_a: int,
        finger_b: int,
        pitch_a: int,
        pitch_b: int,
        hand_a: str,
        hand_b: str,
        chord_size: int = 1,
    ) -> float:
        """Compute the total transition cost between two note assignments.

        This is the sum of all individual cost components.

        Args:
            finger_a: Previous finger (1–5).
            finger_b: Current finger (1–5).
            pitch_a: Previous MIDI pitch.
            pitch_b: Current MIDI pitch.
            hand_a: Previous hand (``"L"`` or ``"R"``).
            hand_b: Current hand (``"L"`` or ``"R"``).
            chord_size: Number of simultaneous notes at the current onset.

        Returns:
            Aggregated non-negative cost.
        """
        interval = abs(pitch_b - pitch_a)

        cost = 0.0

        # Only apply within-hand costs when the hand stays the same
        if hand_a == hand_b:
            cost += self.stretch_cost(finger_a, finger_b, interval)
            cost += self.crossing_cost(finger_a, finger_b, pitch_a, pitch_b)
            cost += self.repetition_cost(finger_a, finger_b, pitch_a, pitch_b)
        else:
            cost += self.hand_switch_cost(hand_a, hand_b)

        cost += self.chord_penalty(chord_size)
        cost += self.weak_finger_cost(finger_b)

        return cost
