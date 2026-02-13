"""Feature Builder — deterministic feature computation for note sequences.

Computes per-note and per-pair features consumed by the cost model / solver:
    - delta_pitch       : signed interval from previous note (semitones)
    - delta_time        : time gap from previous note onset (seconds)
    - chord_size        : number of simultaneous notes at this onset
    - overlap_count     : notes still sounding when this note starts
    - pitch_register    : "low" / "mid" / "high" region hint

All computations are deterministic — no randomness, no learned parameters.
"""

from __future__ import annotations

from typing import Any


# ── Pitch register boundaries ──────────────────────────────────
# These are advisory hints only; the solver may override them.
_LOW_THRESHOLD: int = 48   # C3 and below → "low"
_HIGH_THRESHOLD: int = 72  # C5 and above → "high"


def _pitch_register(pitch: int) -> str:
    """Classify a MIDI pitch into a register region.

    Args:
        pitch: MIDI note number (0–127).

    Returns:
        ``"low"``, ``"mid"``, or ``"high"``.
    """
    if pitch <= _LOW_THRESHOLD:
        return "low"
    if pitch >= _HIGH_THRESHOLD:
        return "high"
    return "mid"


def build_features(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute deterministic features for a sequence of notes.

    The input ``notes`` list **must** already be sorted by ``(start, pitch)``
    (as returned by :func:`midi_parser.extract_notes`).

    Args:
        notes: Sorted list of note dicts with at least
            ``pitch``, ``start``, ``end``, ``duration``, ``velocity``.

    Returns:
        A new list where each element is a copy of the original note dict
        augmented with additional feature keys:
            - ``index``          (int)
            - ``delta_pitch``    (int | None)    — None for the first note
            - ``delta_time``     (float | None)  — None for the first note
            - ``chord_size``     (int)           — count of notes sharing this onset
            - ``overlap_count``  (int)           — notes still sounding at this onset
            - ``pitch_register`` (str)           — "low" / "mid" / "high"
    """
    if not notes:
        return []

    # ── Pre-compute chord groups ───────────────────────────────
    # Two notes belong to the same chord when their onsets are
    # essentially identical (< 30 ms tolerance for MIDI quantisation).
    chord_tolerance: float = 0.03  # seconds

    chord_sizes: list[int] = []
    group_start: int = 0
    for i, note in enumerate(notes):
        # Detect group boundary
        if i > 0 and abs(note["start"] - notes[group_start]["start"]) > chord_tolerance:
            group_start = i
        # We'll fill actual sizes in a second pass
        chord_sizes.append(group_start)

    # Second pass: compute group size for each note
    # group_start_indices maps group_start → count
    from collections import Counter

    group_counts = Counter(chord_sizes)
    chord_size_list: list[int] = [group_counts[gs] for gs in chord_sizes]

    # ── Build feature list ────────────────────────────────────
    enriched: list[dict[str, Any]] = []

    for i, note in enumerate(notes):
        feat: dict[str, Any] = dict(note)  # shallow copy
        feat["index"] = i

        # Interval / time delta from previous note
        if i == 0:
            feat["delta_pitch"] = None
            feat["delta_time"] = None
        else:
            feat["delta_pitch"] = note["pitch"] - notes[i - 1]["pitch"]
            feat["delta_time"] = round(note["start"] - notes[i - 1]["start"], 6)

        # Chord size
        feat["chord_size"] = chord_size_list[i]

        # Overlap count: how many earlier notes are still sounding?
        overlap: int = 0
        for j in range(i):
            if notes[j]["end"] > note["start"]:
                overlap += 1
        feat["overlap_count"] = overlap

        # Pitch register hint
        feat["pitch_register"] = _pitch_register(note["pitch"])

        enriched.append(feat)

    return enriched
