"""Virtuoso Architect — Fingering Estimator entry point.

Initialises the hardware environment and prints the startup banner.
This module is the CLI entry point; the Streamlit app lives in
``app/streamlit_app.py``.
"""

from src.config import setup_hardware


def main() -> None:
    """Run hardware detection and print the initialisation banner."""
    setup_hardware()
    print("Virtuoso Architect – Fingering Estimator Initialized")


if __name__ == "__main__":
    main()
