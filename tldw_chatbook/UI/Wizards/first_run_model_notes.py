"""The Model step's note on what else the chosen model is used for.

TASK-34100.2 AC#8 (coverage-17): document analysis (Library's
analyze-after-import and the Reader's Generate; dev's TASK-38605 deleted the
Media analysis panel) follows the chat model while ``[analysis_defaults]`` still equals the shipped
template (``Library.ingest_analysis.effective_analysis_defaults``); a value
the user set is kept. Setup says which applies, so picking a model here never
silently decides -- or fails to decide -- what analysis spends on.

Kept beside the Model step, whose module is pinned at its exact size
(``Tests/Architecture/test_module_size_ratchet.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from textual.widgets import Static

#: The note's widget id (tests and CSS find it by this).
ANALYSIS_NOTE_ID = "setup-model-analysis-note"


def analysis_note(app_config: Mapping[str, Any]) -> str:
    """One line on whether document analysis follows this model.

    Args:
        app_config: The app's loaded configuration.

    Returns:
        The note, or ``""`` when analysis has neither a template nor a user
        value to describe.
    """
    from tldw_chatbook.Library.ingest_analysis import effective_analysis_defaults

    probe = dict(app_config) if isinstance(app_config, Mapping) else {}
    # Whatever setup picks becomes [chat_defaults]; a stand-in asks only
    # whether [analysis_defaults] would follow it.
    probe["chat_defaults"] = {"provider": "setup-choice"}
    defaults, follows = effective_analysis_defaults(probe)
    if follows:
        return (
            "Document analysis and summaries (Library import, Reader) "
            "use this model too."
        )
    provider = str(defaults.get("provider") or "").strip()
    model = str(defaults.get("model") or "").strip()
    if not provider:
        return ""
    named = f"{provider} · {model}" if model else provider
    return f"Document analysis keeps its own model: {named}."


def analysis_note_widget(wizard: Any) -> Static:
    """The note as the Model step shows it, under the model choice."""
    app_config = getattr(getattr(wizard, "app_instance", None), "app_config", None)
    return Static(
        analysis_note(app_config if isinstance(app_config, Mapping) else {}),
        id=ANALYSIS_NOTE_ID,
        classes="setup-subtitle",
    )
