# tldw_chatbook/Event_Handlers/ingest_utils.py
#
# Common utilities and constants for ingestion event handlers
#
# Imports
from typing import Optional
from pathlib import Path

# 3rd-party Libraries
from ..Third_Party.textual_fspicker import Filters

# --- Prompt Ingest Constants ---
MAX_PROMPT_PREVIEWS = 10
PROMPT_FILE_FILTERS = Filters(
    ("Markdown", lambda p: p.suffix.lower() == ".md"),
    ("JSON", lambda p: p.suffix.lower() == ".json"),
    ("YAML", lambda p: p.suffix.lower() in (".yaml", ".yml")),
    ("Text", lambda p: p.suffix.lower() == ".txt"),
    ("All Supported", lambda p: p.suffix.lower() in (".md", ".json", ".yaml", ".yml", ".txt")),
    ("All Files", lambda _: True),
)

# --- Character Ingest Constants ---
MAX_CHARACTER_PREVIEWS = 5  # Show fewer character previews as they can be larger
CHARACTER_FILE_FILTERS = Filters(
    ("Character Cards (JSON, YAML, PNG, WebP, MD)",
     lambda p: p.suffix.lower() in (".json", ".yaml", ".yml", ".png", ".webp", ".md")),
    ("JSON (*.json)", lambda p: p.suffix.lower() == ".json"),
    ("YAML (*.yaml, *.yml)", lambda p: p.suffix.lower() in (".yaml", ".yml")),
    ("PNG (*.png)", lambda p: p.suffix.lower() == ".png"),
    ("WebP (*.webp)", lambda p: p.suffix.lower() == ".webp"),
    ("Markdown (*.md)", lambda p: p.suffix.lower() == ".md"),
    ("All Files", lambda _: True),
)

# --- Notes Ingest Constants ---
MAX_NOTE_PREVIEWS = 10
NOTE_FILE_FILTERS = Filters(
    ("All Supported", lambda p: p.suffix.lower() in (".json", ".yaml", ".yml", ".txt", ".md", ".markdown", ".rst", ".csv")),
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
        return text[:max_len - 3] + "..."
    return text