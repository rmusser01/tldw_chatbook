# tldw_chatbook/Event_Handlers/ingest_utils.py
#
# Common utilities and constants for ingestion event handlers
#
# Imports
from typing import Optional

# 3rd-party Libraries
from ..Third_Party.textual_fspicker import Filters

# --- Notes Ingest Constants ---
MAX_NOTE_PREVIEWS = 10
NOTE_FILE_FILTERS = Filters(
    (
        "All Supported",
        lambda p: (
            p.suffix.lower()
            in (".json", ".yaml", ".yml", ".txt", ".md", ".markdown", ".rst", ".csv")
        ),
    ),
    ("JSON (*.json)", lambda p: p.suffix.lower() == ".json"),
    ("YAML (*.yaml, *.yml)", lambda p: p.suffix.lower() in (".yaml", ".yml")),
    ("Markdown (*.md)", lambda p: p.suffix.lower() in (".md", ".markdown")),
    ("Text (*.txt, *.rst)", lambda p: p.suffix.lower() in (".txt", ".text", ".rst")),
    ("CSV (*.csv)", lambda p: p.suffix.lower() == ".csv"),
    ("All Files", lambda _: True),
)


def _truncate_text(text: Optional[str], max_len: int) -> str:
    """
    Truncates a string to a maximum length, adding ellipsis if truncated.
    Returns 'N/A' if the input text is None or empty.
    """
    if not text:  # Handles None or empty string
        return "N/A"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text
