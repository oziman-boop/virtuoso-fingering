"""MIDI Parser — load and extract structured note data from MIDI files.

Responsibilities:
    - Load a MIDI file via *pretty_midi*.
    - Extract **all** NOTE_ON events across all instruments.
    - Return a sorted list of note dictionaries with:
        pitch, start, end, duration, velocity

No fingering logic lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pretty_midi


def load_midi(midi_path: str | Path) -> pretty_midi.PrettyMIDI:
    """Load a MIDI file and return a PrettyMIDI object.

    Args:
        midi_path: Path to the ``.mid`` / ``.midi`` file.
            May contain spaces or commas — handled via *pathlib.Path*.

    Returns:
        A ``pretty_midi.PrettyMIDI`` instance.

    Raises:
        FileNotFoundError: If *midi_path* does not exist.
        ValueError: If the file cannot be parsed as MIDI.
    """
    path = Path(midi_path)
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    try:
        midi_data = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to parse MIDI file '{path.name}': {exc}") from exc

    return midi_data


def extract_notes(midi_data: pretty_midi.PrettyMIDI) -> list[dict[str, Any]]:
    """Extract all NOTE_ON events from every instrument.

    Notes are sorted **deterministically** by ``(start, pitch)`` so downstream
    processing always sees the same ordering.

    Args:
        midi_data: A loaded ``PrettyMIDI`` object.

    Returns:
        A list of dicts, each containing:
            - ``pitch``    (int):   MIDI note number 0-127
            - ``start``    (float): onset time in seconds
            - ``end``      (float): offset time in seconds
            - ``duration`` (float): note length in seconds
            - ``velocity`` (int):   MIDI velocity 0-127
    """
    notes: list[dict[str, Any]] = []

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue  # skip percussion tracks
        for note in instrument.notes:
            notes.append(
                {
                    "pitch": note.pitch,
                    "start": round(note.start, 6),
                    "end": round(note.end, 6),
                    "duration": round(note.end - note.start, 6),
                    "velocity": note.velocity,
                }
            )

    # Deterministic sort: onset time first, then pitch (ascending)
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def parse_midi(midi_path: str | Path) -> list[dict[str, Any]]:
    """Convenience wrapper: load MIDI → extract notes.

    Args:
        midi_path: Path to the MIDI file.

    Returns:
        Sorted list of note dictionaries.
    """
    midi_data = load_midi(midi_path)
    return extract_notes(midi_data)
