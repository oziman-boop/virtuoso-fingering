"""Solver — dynamic-programming graph search for optimal fingering assignment.

State:  ``(note_index, hand, finger)``
Transition: applies :class:`FingeringCostModel.total_cost` to evaluate
            every feasible (hand, finger) assignment for the next note.
Output: the minimum-cost deterministic fingering path.

Design choices:
    - No randomness, no multiprocessing.
    - ``hand`` is ``"L"`` or ``"R"``; ``finger`` is 1–5.
    - For the right hand, finger 1 = thumb (lowest pitch side).
    - For the left hand, finger 1 = thumb (highest pitch side).
    - The split heuristic (left below split_pitch, right above) supplies the
      initial hand preference; DP is free to override it when cheaper.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .cost_model import FingeringCostModel


# ── Public types ──────────────────────────────────────────────
HANDS: list[str] = ["L", "R"]
FINGERS: list[int] = [1, 2, 3, 4, 5]

# State key: (hand, finger) tuple
StateKey = tuple[str, int]


def solve(
    notes: list[dict[str, Any]],
    config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Find the optimal fingering for a sequence of notes via DP.

    Args:
        notes: Sorted list of note dicts (must contain at least
            ``pitch``, ``start``, ``chord_size``).
            Typically the output of :func:`feature_builder.build_features`.
        config_path: Optional path to ``fingering_costs.yaml``.
            Defaults to the project-relative config.

    Returns:
        A list of dicts (same length as *notes*), each containing:
            - ``onset_time`` (float)
            - ``pitch``      (int)
            - ``hand``       (``"L"`` | ``"R"``)
            - ``finger``     (1–5)
    """
    if not notes:
        return []

    cost_model = FingeringCostModel(config_path)
    n = len(notes)

    # ── DP tables ─────────────────────────────────────────────
    # dp[i][(hand, finger)] = minimum cumulative cost up to note i
    # bp[i][(hand, finger)] = predecessor state at note i-1
    dp: list[dict[StateKey, float]] = [{} for _ in range(n)]
    bp: list[dict[StateKey, StateKey | None]] = [{} for _ in range(n)]

    # ── Initialise first note ─────────────────────────────────
    first_pitch: int = notes[0]["pitch"]
    preferred_hand: str = "L" if first_pitch <= cost_model.split_pitch else "R"

    for hand in HANDS:
        for finger in FINGERS:
            # Small bias toward the "natural" hand
            bias: float = 0.0 if hand == preferred_hand else cost_model.hand_switch_weight * 0.5
            init_cost = cost_model.weak_finger_cost(finger) + bias
            dp[0][(hand, finger)] = init_cost
            bp[0][(hand, finger)] = None

    # ── Forward pass ──────────────────────────────────────────
    for i in range(1, n):
        cur_note = notes[i]
        prev_note = notes[i - 1]
        chord_size: int = cur_note.get("chord_size", 1)

        for hand_b in HANDS:
            for finger_b in FINGERS:
                best_cost: float = math.inf
                best_prev: StateKey | None = None

                for hand_a in HANDS:
                    for finger_a in FINGERS:
                        prev_cost = dp[i - 1].get((hand_a, finger_a))
                        if prev_cost is None:
                            continue

                        trans = cost_model.total_cost(
                            finger_a=finger_a,
                            finger_b=finger_b,
                            pitch_a=prev_note["pitch"],
                            pitch_b=cur_note["pitch"],
                            hand_a=hand_a,
                            hand_b=hand_b,
                            chord_size=chord_size,
                        )
                        total = prev_cost + trans
                        if total < best_cost:
                            best_cost = total
                            best_prev = (hand_a, finger_a)

                dp[i][(hand_b, finger_b)] = best_cost
                bp[i][(hand_b, finger_b)] = best_prev

    # ── Backtrack ─────────────────────────────────────────────
    # Find best final state
    best_final_cost: float = math.inf
    best_final_state: StateKey = ("R", 1)
    for state, cost in dp[n - 1].items():
        if cost < best_final_cost:
            best_final_cost = cost
            best_final_state = state

    # Trace path
    path: list[StateKey] = [best_final_state]
    for i in range(n - 1, 0, -1):
        prev_state = bp[i][path[-1]]
        if prev_state is None:
            # Should not happen after index 0
            prev_state = ("R", 1)
        path.append(prev_state)

    path.reverse()

    # ── Build result ──────────────────────────────────────────
    result: list[dict[str, Any]] = []
    for i, note in enumerate(notes):
        hand, finger = path[i]
        result.append(
            {
                "onset_time": note["start"],
                "pitch": note["pitch"],
                "hand": hand,
                "finger": finger,
            }
        )

    return result
