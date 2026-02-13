"""Virtuoso Architect — Streamlit Fingering Estimator UI.

Minimal interactive application:
    1. Upload a MIDI file (.mid / .midi)
    2. Run the fingering estimation pipeline
    3. View summary statistics and an annotation table
    4. Download annotations.json and/or annotated MIDI

Constraints:
    - No plotting libraries
    - No audio playback
    - Simple, readable code
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.fingering_engine.annotate import annotate, annotations_to_json_bytes  # noqa: E402
from src.fingering_engine.midi_parser import load_midi  # noqa: E402

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Virtuoso Architect — Fingering Estimator",
    page_icon="🎹",
    layout="wide",
)

st.title("🎹 Virtuoso Architect — Fingering Estimator")
st.markdown(
    "Upload a MIDI file, run the rule-based dynamic-programming engine, "
    "and download annotated fingerings."
)
st.divider()

# ── File uploader ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose a MIDI file",
    type=["mid", "midi"],
    help="Piano solo MIDI files work best.",
)

if uploaded_file is not None:
    st.success(f"Loaded: **{uploaded_file.name}**")

    # Write the uploaded bytes to a temp file so pretty_midi can open it
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    # ── Run button ────────────────────────────────────────────
    col_run, col_midi = st.columns([1, 1])
    export_midi: bool = col_midi.checkbox("Also export annotated MIDI", value=False)
    run_clicked: bool = col_run.button("▶  Run Fingering Estimation", type="primary")

    if run_clicked:
        with st.spinner("Running DP solver …"):
            with tempfile.TemporaryDirectory() as out_dir:
                annotations = annotate(
                    midi_path=tmp_path,
                    output_dir=out_dir,
                    export_midi=export_midi,
                )

                # ── Summary stats ─────────────────────────────
                st.subheader("Summary")
                total_notes = len(annotations)
                left_count = sum(1 for a in annotations if a["hand"] == "L")
                right_count = total_notes - left_count

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Notes", total_notes)
                c2.metric("Left Hand", left_count)
                c3.metric("Right Hand", right_count)

                # ── Annotation table ──────────────────────────
                st.subheader("Annotation Table")
                df = pd.DataFrame(annotations)
                df.columns = ["Onset (s)", "Pitch", "Hand", "Finger"]
                st.dataframe(df, use_container_width=True, height=400)

                # ── Downloads ─────────────────────────────────
                st.subheader("Downloads")
                json_bytes = annotations_to_json_bytes(annotations)
                st.download_button(
                    label="⬇  Download annotations.json",
                    data=json_bytes,
                    file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_annotations.json",
                    mime="application/json",
                )

                if export_midi:
                    midi_out = Path(out_dir) / f"{tmp_path.stem}_annotated.mid"
                    if midi_out.exists():
                        st.download_button(
                            label="⬇  Download annotated MIDI",
                            data=midi_out.read_bytes(),
                            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_annotated.mid",
                            mime="audio/midi",
                        )

    # Clean up temp file reference (OS will remove on reboot if not sooner)
    # We intentionally do NOT delete it here because Streamlit may re-run.
