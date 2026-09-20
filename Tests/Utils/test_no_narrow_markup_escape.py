"""No Textual surface may go back to the narrow ``rich.markup.escape``.

TASK-32802.1 replaced it across the tree. It escapes only tags matching
``\\[[a-z#/@]...]``, so Textual 8 deletes every other bracketed token a user
types -- ``[TODO] Q3 plan`` rendered as `` Q3 plan``. The swap is mechanical
and the old import is the easy thing to reach for, so this census is what
stops the class coming back one new screen at a time.

Every module below keeps the narrow escape on purpose, because its value
reaches a markup-OFF sink where a backslash is shown to the reader (or is
sent to a model). Escaping there at all is a separate defect, owned by
TASK-32802.4 -- the entries come off this list as that task closes them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"

#: module path -> why it still imports the narrow escape.
MARKUP_OFF_SINKS: dict[str, str] = {
    "Chat/console_provider_gateway.py":
        "the escaped model id becomes a persisted error message and renders "
        "through Content.assemble as a literal segment",
    "Event_Handlers/STTS_Events/stts_events.py":
        "writes into RichLog(markup=False) (#tts-generation-log)",
    "Library/library_rag_state.py":
        "titles and snippets double as the LLM answer prompt payload",
    "TTS/audio_cpp_supervisor.py":
        "diagnostic lines land in RichLog(markup=False)",
    "UI/Library_Modules/library_export_controller.py":
        "the export error line renders in Static(markup=False)",
    "UI/Screens/chat_screen.py":
        "workbench help rows render in Static(markup=False) (help.py)",
    "UI/Screens/evals_screen.py":
        "the blocked-action label renders in Static(markup=False)",
    "UI/Screens/home_screen.py":
        "the canvas title renders in Static(markup=False)",
    "UI/Screens/study_screen.py":
        "the scope summary is carried in a chat handoff payload",
}


def _modules_importing_rich_escape() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "rich.markup"
            and any(alias.name == "escape" for alias in node.names)
        ]
        if lines:
            found[path.relative_to(PACKAGE_ROOT).as_posix()] = lines
    return found


def test_only_markup_off_sinks_still_use_the_narrow_escape():
    found = _modules_importing_rich_escape()
    unexpected = sorted(set(found) - set(MARKUP_OFF_SINKS))
    assert not unexpected, (
        "these modules import rich.markup.escape, which Textual 8 does not "
        "honour for an uppercase bracket -- use "
        "tldw_chatbook.Utils.input_validation.escape_markup instead, or add the module "
        f"here with the markup-off sink that justifies it: {unexpected}"
    )


def test_the_allowlist_has_no_stale_entries():
    """A closed TASK-32802.4 entry must be deleted, not left to rot."""
    found = _modules_importing_rich_escape()
    stale = sorted(set(MARKUP_OFF_SINKS) - set(found))
    assert not stale, (
        f"these no longer import the narrow escape; drop them here: {stale}"
    )


@pytest.mark.parametrize("module", sorted(MARKUP_OFF_SINKS))
def test_every_exception_names_its_sink(module):
    assert MARKUP_OFF_SINKS[module].strip(), f"{module} needs a reason"
