"""RAG Advisor — structured contextual prompting for fingering explanations.

Phase 1 of the RAG engine: uses the OpenAI Chat Completions API to generate
deterministic, pedagogically-informed explanations of fingering decisions
and practice suggestions.

Architecture:
    - **No embeddings** or vector retrieval yet (Phase 1 = structured prompting).
    - **Lazy initialisation** — the OpenAI client is created inside
      ``explain_fingering()``, never at import time.
    - **No global state** — every call builds its own client instance.
    - **No side effects on import** — ``python-dotenv`` is loaded lazily.
    - **Deterministic** — ``temperature=0`` for reproducible outputs.

Environment:
    A ``.env`` file in the project root must contain::

        OPENAI_API_KEY=sk-...

    The key is loaded via ``python-dotenv`` at call time.
    It is never printed or logged.

No network calls are made unless ``explain_fingering()`` is explicitly invoked.
"""

from __future__ import annotations

from pathlib import Path


# ── Project root (used to locate .env) ────────────────────────
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def explain_fingering(context: str) -> str:
    """Generate a structured explanation of piano fingering decisions.

    Loads the API key from ``.env`` via ``python-dotenv``, initialises
    the OpenAI client, and sends a deterministic (temperature=0)
    chat completion request with a pedagogically-structured prompt.

    Args:
        context: A string containing the passage description,
            fingering assignments, and summary statistics.
            Example::

                "Passage: C4→E4→G4→C5 (ascending C major arpeggio)
                 Fingerings: R1→R2→R3→R5
                 Hand switches: 0 | Stretches: 1"

    Returns:
        A multi-section explanation string with:
            - Technical Analysis
            - Ergonomic Considerations
            - Practice Strategy
            - Alternative Fingering (optional)

        If the API call fails, returns a human-readable error message
        instead of raising an exception.
    """
    # ── 1. Load environment lazily ────────────────────────────
    try:
        from dotenv import load_dotenv  # noqa: E402
    except ImportError:
        return (
            "Error: python-dotenv is not installed. "
            "Install it with: pip install python-dotenv"
        )

    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

    import os

    api_key: str | None = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Create a .env file in the project root with:\n"
            "  OPENAI_API_KEY=sk-..."
        )

    # ── 2. Initialise client lazily ───────────────────────────
    try:
        import openai  # noqa: E402
    except ImportError:
        return (
            "Error: openai package is not installed. "
            "Install it with: pip install openai"
        )

    client = openai.OpenAI(api_key=api_key)

    # ── 3. Construct structured prompt ────────────────────────
    system_prompt = (
        "You are an expert classical piano pedagogue specializing in "
        "ergonomic fingering. You have deep knowledge of hand biomechanics, "
        "traditional conservatory fingering conventions, and modern "
        "performance practice.\n\n"
        "When analysing a fingering assignment, structure your response "
        "into EXACTLY the following sections:\n\n"
        "## Technical Analysis\n"
        "Explain the musical and technical rationale behind the assigned "
        "fingering. Reference specific intervals, finger transitions, "
        "and positional shifts.\n\n"
        "## Ergonomic Considerations\n"
        "Discuss hand position, wrist alignment, tension risks, and "
        "physical comfort. Flag any potentially injurious patterns.\n\n"
        "## Practice Strategy\n"
        "Provide concrete, actionable practice steps. Include tempo "
        "recommendations, isolation exercises, and repetition strategies.\n\n"
        "## Alternative Fingering (optional)\n"
        "If a plausible alternative exists that may suit certain hand "
        "sizes or musical interpretations, briefly describe it. "
        "If the assigned fingering is clearly optimal, state so and "
        "omit alternatives.\n\n"
        "Be precise, concise, and pedagogically rigorous. "
        "Do not speculate beyond the provided context."
    )

    # ── 4. Call the API (no streaming) ────────────────────────
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            temperature=0,  # fully deterministic
            max_tokens=1024,
        )
    except Exception as exc:
        return (
            f"Fingering Advisor API error: {type(exc).__name__}: {exc}\n"
            "Please check your API key and network connection."
        )

    # ── 5. Extract and return the response ────────────────────
    if not response.choices:
        return "Fingering Advisor returned an empty response. Please try again."

    message = response.choices[0].message
    if message and message.content:
        return message.content.strip()

    return "Fingering Advisor returned an empty response. Please try again."
