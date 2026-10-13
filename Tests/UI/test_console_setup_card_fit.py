"""TASK-389/441: the first-run setup card must fit its step-3 line.

The honest step-3 line "3. ○ Send your first message  Composer unlocks after
setup" is wider than the old 62-cell card gave in content, so its last word
wrapped to a hidden second line with no ellipsis. TASK-389 first fixed this by
widening the card (AC#1 pin below) and ellipsizing any remaining overflow on a
single ``height: 1`` line. TASK-441 replaces the ellipsis strategy: at default
terminal sizes the full sentence already fits on one line (AC#1), but on a
narrow terminal that caps ``.console-setup-modal-card`` below its declared
width, the copy must now *wrap* onto additional rows instead of truncating
(AC#2) -- ``.console-setup-step`` moved from a fixed ``height: 1`` +
``text-wrap: nowrap`` + ``text-overflow: ellipsis`` to ``height: auto`` with
Static's default wrapping restored.
"""

import re
from pathlib import Path

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_SETUP_STEP_THREE_DETAIL,
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal
from Tests.UI.consolidated_css import APP_STYLESHEETS

ROOT = Path(__file__).resolve().parents[2]
CONSOLE_CSS = ROOT / "tldw_chatbook/css/features/_console.tcss"
VARIABLES = ROOT / "tldw_chatbook/css/core/_variables.tcss"
_BUNDLED_STYLESHEET = ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


def _rule_body(css: str, selector: str) -> str:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    return match.group(1) if match else ""


def _cells(value: str) -> int:
    """A cell count, or a ``$token`` read from core/_variables.tcss."""
    if value.startswith("$"):
        tokens = VARIABLES.read_text(encoding="utf-8")
        value = re.search(rf"^\{value}\s*:\s*(\d+)\s*;", tokens, re.MULTILINE).group(1)
    return int(value)


def _card_content_width() -> int:
    """The setup card's content cells, from its rule in features/_console.tcss:
    width minus left and right padding minus a border cell each side."""
    card = _rule_body(
        CONSOLE_CSS.read_text(encoding="utf-8"), ".console-setup-modal-card"
    )
    width = _cells(re.search(r"(?<![-\w])width\s*:\s*(\S+)\s*;", card).group(1))
    side = _cells(re.search(r"padding\s*:\s*\S+\s+(\S+)\s*;", card).group(1))
    assert re.search(r"(?<![-\w])border\s*:\s*solid\b", card), card
    return width - 2 * side - 2


def _card_state_with_step_three() -> ConsoleSetupCardState:
    """Card state carrying the real production step-3 label + detail."""
    return ConsoleSetupCardState(
        mode="card",
        steps=(
            ConsoleSetupStep(state="active", label="Add an API key"),
            ConsoleSetupStep(state="done", label="Pick a model"),
            ConsoleSetupStep(
                state="pending",
                label="Send your first message",
                detail=CONSOLE_SETUP_STEP_THREE_DETAIL,
            ),
        ),
    )


def test_the_active_steps_readiness_word_gets_its_own_line():
    """TASK-33005.3: a wrap must not split "Not ready · refused :9199" (live at
    211x44 the port wrapped alone onto the next row), so the active step's
    word sits on its own line under the label, which the card fits whole."""
    step = ConsoleSetupStep(
        state="active",
        label="Reconnect the provider server",
        detail="Not ready · refused :9199",
    )
    line = ConsoleSetupModal._step_text(1, step)

    assert line == "1. ● Reconnect the provider server\n     Not ready · refused :9199"
    assert max(cell_len(row) for row in line.split("\n")) <= _card_content_width()


@pytest.mark.parametrize(
    ("provider_key", "blocker", "category", "url", "word"),
    [
        (
            "oobabooga",
            "endpoint_unreachable",
            "connection_refused",
            "http://127.0.0.1:5000",
            "Not ready · refused :5000",
        ),
        (
            "oobabooga",
            "endpoint_unreachable",
            "connection_error",
            None,
            "Not ready · unreachable",
        ),
        ("local_llm", "endpoint_not_saved", None, None, "Not ready · endpoint unsaved"),
    ],
)
def test_a_long_provider_name_never_wraps_the_readiness_word(
    provider_key, blocker, category, url, word
):
    """TASK-33008 capture fix round 1: step 1 names the provider before its
    word, and "Text Generation WebUI (Oobabooga): Not ready · refused :5000"
    is 65 cells, so ":5000" wrapped alone onto a row, the TASK-33005.3
    symptom. The name is cut so the word's line fits the card whole."""
    from tldw_chatbook.Chat.console_onboarding_state import (
        build_console_setup_card_state,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsReadiness
    from tldw_chatbook.Chat.provider_catalog import provider_display_name
    from tldw_chatbook.Chat.provider_endpoint_contract import (
        canonical_connection_identity,
    )
    from tldw_chatbook.Chat.provider_test_evidence import ProviderDraftIdentity

    connection = (
        ProviderDraftIdentity(
            provider_key=provider_key,
            connection_identity=canonical_connection_identity(provider_key, url),
            credential_source="none",
            credential_revision=0,
            draft_generation=0,
        )
        if url
        else None
    )
    readiness = ConsoleSettingsReadiness(
        "Not ready",
        "",
        False,
        operability="not_ready",
        blocker=blocker,
        recovery_action=(
            "retry_connection" if blocker == "endpoint_unreachable" else "save_endpoint"
        ),
        configuration="configured",
        credential="not_required",
        endpoint="unreachable" if blocker == "endpoint_unreachable" else "not_tested",
        endpoint_category=category,
        model="unconfirmed",
        connection=connection,
    )
    name = provider_display_name(provider_key)
    state = build_console_setup_card_state(
        readiness=readiness,
        provider_label=name,
        has_model=True,
        first_send_completed=False,
        has_messages=False,
        guidance_dismissed=False,
    )
    rows = ConsoleSetupModal._step_text(1, state.steps[0]).split("\n")

    assert rows[1].endswith(f": {word}")
    assert rows[1].lstrip().startswith(name[:12])
    # Capture fix round 2: measured against the card's rule, not a copied 60,
    # so narrowing the card or widening its padding turns this red.
    assert max(cell_len(row) for row in rows) <= _card_content_width()


def test_setup_card_is_wide_enough_for_the_step_three_line():
    """AC#1: the full step-3 sentence fits inside the card's content width."""
    step_three = ConsoleSetupStep(
        state="pending",
        label="Send your first message",
        detail=CONSOLE_SETUP_STEP_THREE_DETAIL,
    )
    line = ConsoleSetupModal._step_text(3, step_three)
    content_width = _card_content_width()

    assert cell_len(line) <= content_width, (
        f"step-3 line is {cell_len(line)} cells but the card content is only "
        f"{content_width}: {line!r}"
    )


def test_setup_step_wraps_instead_of_ellipsizing_on_overflow():
    """AC#2: overflow wraps onto another row instead of being truncated.

    TASK-389's ``text-wrap: nowrap`` + ``text-overflow: ellipsis`` guaranteed
    a *marked* truncation on unavoidable overflow, but still dropped words off
    the end of the sentence. TASK-441 asks for the words to survive by
    wrapping, so the step rule must no longer force a single line, and must
    grow (``height: auto``) rather than stay pinned at ``height: 1`` (which is
    what silently hid the wrapped second row before TASK-389's ellipsis fix
    existed at all).
    """
    step = _rule_body(CONSOLE_CSS.read_text(encoding="utf-8"), ".console-setup-step")
    assert "nowrap" not in step
    assert "ellipsis" not in step
    assert "height: auto" in step


class SetupModalGeometryApp(App[None]):
    """Mount the setup modal with the production stylesheet for geometry checks."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal(id="console-setup-modal")

    async def on_mount(self) -> None:
        modal = self.query_one("#console-setup-modal", ConsoleSetupModal)
        modal.sync_card_state(
            _card_state_with_step_three(),
            action_label="Configure API",
            action_tooltip="Open provider settings.",
        )


@pytest.mark.asyncio
async def test_step_three_is_one_complete_line_at_default_terminal_size():
    """AC#1, rendered: an 80-column default terminal shows the full sentence
    on a single row -- not clipped, and not wrapped (the card is wide enough)."""
    app = SetupModalGeometryApp()
    async with app.run_test(size=(80, 24)):
        step3 = app.query_one("#console-setup-step-3", Static)
        assert step3.size.height == 1
        rendered = step3.render_line(0).text
        assert rendered.strip() == (
            "3. ○ Send your first message  Composer unlocks after setup"
        )


class StagedNoticeGeometryApp(App[None]):
    """Mount the setup modal and sync a staged-evidence receipt (task-2852)."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, *, blocking: bool, notice: str) -> None:
        super().__init__()
        self._blocking = blocking
        self._notice = notice

    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal(id="console-setup-modal")

    async def on_mount(self) -> None:
        modal = self.query_one("#console-setup-modal", ConsoleSetupModal)
        card_state = (
            _card_state_with_step_three()
            if self._blocking
            else ConsoleSetupCardState(mode="quiet")
        )
        modal.sync_card_state(
            card_state,
            action_label="Configure API",
            action_tooltip="Open provider settings.",
            staged_evidence_notice=self._notice,
        )


@pytest.mark.asyncio
async def test_staged_evidence_notice_shows_on_a_locked_console():
    """Task-2852 AC#2: a locked Console names what's staged instead of
    landing the handoff with zero receipt."""
    notice = "Library Search/RAG evidence staged — finish provider setup to use it."
    app = StagedNoticeGeometryApp(blocking=True, notice=notice)
    async with app.run_test(size=(80, 24)):
        widget = app.query_one("#console-setup-modal-staged-notice", Static)
        assert widget.display is True
        assert str(widget.renderable).strip() == notice


@pytest.mark.asyncio
async def test_staged_evidence_notice_hidden_when_nothing_staged():
    """A locked Console with no pending launch shows the plain setup card,
    not a stray empty receipt line."""
    app = StagedNoticeGeometryApp(blocking=True, notice="")
    async with app.run_test(size=(80, 24)):
        widget = app.query_one("#console-setup-modal-staged-notice", Static)
        assert widget.display is False


@pytest.mark.asyncio
async def test_staged_evidence_notice_hidden_once_console_is_configured():
    """AC#3 regression guard at the widget level: once Console is no longer
    blocking, the receipt must not leak onto the (now-hidden) card even if
    the caller is slow to clear a stale notice string."""
    app = StagedNoticeGeometryApp(
        blocking=False,
        notice="Library Search/RAG evidence staged — finish provider setup to use it.",
    )
    async with app.run_test(size=(80, 24)):
        widget = app.query_one("#console-setup-modal-staged-notice", Static)
        assert widget.display is False


@pytest.mark.asyncio
async def test_step_three_wraps_onto_a_second_row_on_a_narrow_terminal():
    """AC#2, rendered: once the card's max-width caps it below the full
    sentence's width, the widget grows to a second row and every word --
    including the "setup" that TASK-389's evidence showed getting dropped --
    is still painted onscreen instead of being cut."""
    app = SetupModalGeometryApp()
    async with app.run_test(size=(50, 30)):
        step3 = app.query_one("#console-setup-step-3", Static)
        # A fixed height: 1 (the pre-fix rule) would clamp this at 1 no matter
        # how much text wraps -- the second row growing in is the geometry
        # proof that overflow is now handled by wrapping, not by clipping.
        assert step3.size.height >= 2
        rendered_rows = [step3.render_line(y).text for y in range(step3.size.height)]
        combined = " ".join(row.strip() for row in rendered_rows if row.strip())
        for word in (
            "Send",
            "your",
            "first",
            "message",
            "Composer",
            "unlocks",
            "after",
            "setup",
        ):
            assert word in combined, (
                f"{word!r} missing from wrapped rows {rendered_rows!r}"
            )
