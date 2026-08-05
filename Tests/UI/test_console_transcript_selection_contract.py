"""Contract test for uniform Console transcript message selection styling.

TASK-385: selecting a user/assistant message shows the focus treatment (focus
colours + bold underline), but a selected Tool (or System) message kept its muted
``dim italic`` styling -- the single-class ``.console-transcript-message-tool`` /
``-system`` rules follow ``.console-transcript-message-selected`` in source order
with equal specificity, so they win the cascade for a row carrying both classes
and strip the selection treatment. A selected message of any kind must read the
same.
"""

import re
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

ROOT = Path(__file__).resolve().parents[2]
AGENTIC = ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
BUNDLE = ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"

_SELECTED = "console-transcript-message-selected"


def _rules(css_text: str) -> list[tuple[str, str]]:
    """Return (selector, body) pairs for every rule in the stylesheet.

    Comments are stripped first (mirroring ``test_non_obscuring_focus_contract``)
    so braces inside ``/* ... */`` -- common in the generated bundle -- are not
    mistaken for rule delimiters.
    """
    uncommented = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)
    return re.findall(r"([^{}]+)\{([^}]*)\}", uncommented)


def _selected_treatment_for(css_text: str, kind: str) -> str:
    """Return the merged body of every rule that targets a selected <kind> row.

    A rule counts if its selector combines the kind class and the selected class
    on the same compound token (higher specificity than the bare kind rule).
    """
    bodies: list[str] = []
    kind_class = f"console-transcript-message-{kind}"
    for selector, body in _rules(css_text):
        for token in selector.split(","):
            token = token.strip()
            if kind_class in token and _SELECTED in token and "." + kind_class + "." in token + ".":
                # both classes on one element (no descendant space between them)
                compound = token.replace(" ", "")
                if f".{kind_class}" in compound and f".{_SELECTED}" in compound:
                    bodies.append(body)
    return "\n".join(bodies)


def test_selected_tool_and_system_messages_share_the_selected_treatment():
    """A selected transcript row of any kind reads the same in source and bundle.

    Guards TASK-385: the muted tool/system role rules must not out-cascade the
    selection treatment, so selected tool/system rows re-assert the focus colour
    and bold underline.
    """
    for css_path in (AGENTIC, BUNDLE):
        css = css_path.read_text(encoding="utf-8")

        # Baseline: the canonical selected treatment the other kinds must match.
        # Match the standalone `.console-transcript-message-selected` rule exactly
        # (a selector-list token), never a compound `-tool…-selected` selector.
        selected = next(
            (
                b
                for s, b in _rules(css)
                if any(tok.strip() == f".{_SELECTED}" for tok in s.split(","))
            ),
            "",
        )
        assert "bold underline" in selected, f"{css_path.name}: baseline selected rule missing"

        for kind in ("tool", "system"):
            treatment = _selected_treatment_for(css, kind)
            assert treatment, (
                f"{css_path.name}: a selected {kind} message must re-assert the "
                f"selection treatment with higher specificity than the muted "
                f".console-transcript-message-{kind} rule"
            )
            assert "$ds-focus-fg" in treatment, (
                f"{css_path.name}: selected {kind} message must use the focus colour"
            )
            assert "bold" in treatment and "underline" in treatment, (
                f"{css_path.name}: selected {kind} message must be bold underline"
            )


def _transcript_messages() -> list[ConsoleChatMessage]:
    return [
        ConsoleChatMessage(
            id="user-native-id",
            role=ConsoleMessageRole.USER,
            content="Question",
        ),
        ConsoleChatMessage(
            id="assistant-native-id",
            role=ConsoleMessageRole.ASSISTANT,
            content="Answer [S1].",
            persisted_message_id="persisted-assistant-id",
        ),
        ConsoleChatMessage(
            id="other-native-id",
            role=ConsoleMessageRole.ASSISTANT,
            content="Another answer",
            persisted_message_id="persisted-other-id",
        ),
    ]


def test_citation_row_is_focusable_and_immediately_follows_owning_message() -> None:
    transcript = ConsoleTranscript()
    transcript.set_messages(_transcript_messages())
    transcript.set_citation_counts({"assistant-native-id": 2})

    rows = transcript._transcript_rows()
    message_index = next(
        index for index, row in enumerate(rows) if row.key == "message:assistant-native-id"
    )
    citation_row = rows[message_index + 1]
    button = transcript._build_row_widget(citation_row, track=False)

    assert citation_row.kind == "citations"
    assert citation_row.key == "citations:assistant-native-id"
    assert isinstance(button, Button)
    assert button.label.plain == "Sources (2)"
    assert button.id == "console-citation-sources-assistant-native-id"
    assert button.has_class("console-transcript-citation-sources")
    assert button.native_message_id == "assistant-native-id"
    assert button.can_focus


@pytest.mark.parametrize("counts", [{}, {"assistant-native-id": 0}])
def test_zero_or_absent_citation_count_adds_no_row(counts: dict[str, int]) -> None:
    transcript = ConsoleTranscript()
    transcript.set_messages(_transcript_messages())
    transcript.set_citation_counts(counts)

    assert all(row.kind != "citations" for row in transcript._transcript_rows())


class _CitationTranscriptHarness(App):
    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(_transcript_messages())
        transcript.set_citation_counts({"assistant-native-id": 1})
        transcript.selected_message_id = "user-native-id"
        yield transcript


@pytest.mark.asyncio
async def test_citation_button_click_preserves_existing_message_selection() -> None:
    app = _CitationTranscriptHarness()

    async with app.run_test() as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        await pilot.click("#console-citation-sources-assistant-native-id")

        assert transcript.selected_message_id == "user-native-id"
        assert (
            "console-transcript-citation-sources"
            in transcript.PROTECTED_CLICK_CLASSES
        )


@pytest.mark.asyncio
async def test_count_only_change_reconciles_footer_without_rebuilding_messages() -> None:
    app = _CitationTranscriptHarness()

    async with app.run_test() as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        before = transcript.row_build_counts()

        transcript.set_citation_counts({"assistant-native-id": 3})
        await transcript.refresh_messages()
        await pilot.pause()

        after = transcript.row_build_counts()
        button = transcript.query_one(
            "#console-citation-sources-assistant-native-id", Button
        )

    assert button.label.plain == "Sources (3)"
    assert after["citations:assistant-native-id"] > before[
        "citations:assistant-native-id"
    ]
    for message in _transcript_messages():
        assert after[f"message:{message.id}"] == before[f"message:{message.id}"]
