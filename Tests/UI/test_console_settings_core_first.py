"""Chat settings' Model view opens core-first with one row grammar (TASK-33006.1).

Every test mounts the real modal under the production stylesheets
(``APP_STYLESHEETS``) at the full-screen sizes the redesign targets, and reads
what the compositor painted, not only widget values (AC#14).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.containers import ScrollableContainer
from textual.widgets import Button, Collapsible, Input, Select, Static

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Chat.console_provider_support import (
    MODEL_CONFIG_FIELDS,
    MODEL_FIELD_LABELS,
)
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_VALUE_SOURCE_WORDS,
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleValueLayer,
    resolve_console_value_layers,
)
from tldw_chatbook.Widgets.Console.console_settings_field_row import (
    BLANK_CHOICE_PROMPT,
    BLANK_FIELD_HELP,
    CONNECTION_DISCLOSURE_ID,
    CORE_FIELDS,
    FIELD_ROW_FIELDS,
    MODEL_CHANGE_ID,
    SAMPLING_DISCLOSURE_ID,
    SAMPLING_FIELDS,
    field_control_id,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal

# Census-gated (scripts/ui_pr_gate_census.txt): Tests/UI/conftest.py imports
# tldw_chatbook.app per test, which fails closed with
# RecoveryRequired("raw_source_selection_changed") under the per-test sandbox.
pytestmark = pytest.mark.bootstrap_profile

FULL_SCREEN_SIZES = ((211, 44), (235, 52))
CSS_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"


class CoreFirstHarness(ConsolidatedCSSApp):
    """Mounts the modal alone under the stylesheets the real app loads."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self) -> None:
        super().__init__()
        self.app_config: dict = {
            "api_settings": {
                "llama_cpp": {
                    "api_url": "http://127.0.0.1:9099",
                    "model_defaults": {"model-a": {"temperature": 0.4}},
                },
                "openai": {"api_key": "test-key"},
                "anthropic": {"api_key": "test-key"},
            },
            "chat_defaults": {"max_tokens": 2048},
            # A registry entry is decided as its family (llama.cpp here):
            # Reasoning effort and Thinking budget show, the rest hide.
            "custom_endpoints": {
                "gpu-box": {
                    "display_name": "GPU box",
                    "family": "llama_cpp",
                    "base_url": "http://192.168.1.9:8080",
                    "models": ["model-a"],
                }
            },
        }


def _settings(provider: str = "llama_cpp", model: str | None = "model-a", **values):
    base_url = "http://127.0.0.1:9099" if provider == "llama_cpp" else None
    values.setdefault("temperature", 0.4)
    values.setdefault("max_tokens", 2048)
    return ConsoleSessionSettings(
        provider=provider, model=model, base_url=base_url, **values
    )


def _modal(app: CoreFirstHarness, settings: ConsoleSessionSettings, **kwargs):
    providers_models = {"llama_cpp": ["model-a", "model-b"]}
    if settings.model:
        providers_models.setdefault(settings.provider, [settings.model])
    return ConsoleSettingsModal(
        settings=settings,
        app_config=app.app_config,
        providers_models=providers_models,
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
        **kwargs,
    )


async def _open(pilot, app, modal) -> None:
    await app.push_screen(modal)
    await _settle(pilot)


async def _settle(pilot) -> None:
    for _ in range(4):
        await pilot.pause()


def _painted(screen) -> list[str]:
    return [strip.text for strip in screen._compositor.render_strips()]


def _painted_cell(screen, x: int, y: int):
    """(glyph, fg, bg) the compositor paints at screen cell (x, y)."""
    position = 0
    for segment in screen._compositor.render_strips()[y]:
        if position + len(segment.text) > x:
            return segment.text[x - position], segment.style.color, segment.style.bgcolor
        position += len(segment.text)
    raise AssertionError(f"({x}, {y}) is off screen")


def _ratio(first, second) -> float:
    from textual.color import Color

    from tldw_chatbook.css.Themes.themes import _contrast_ratio

    return _contrast_ratio(Color.from_rich_color(first), Color.from_rich_color(second))


async def _painted_line(pilot, app, widget) -> str:
    """The painted screen row that holds ``widget``, scrolled into view."""
    # Centred, so the fold hint's row cannot clip it once it re-syncs.
    app.screen.query_one("#console-settings-body").scroll_to_center(
        widget, animate=False, immediate=True
    )
    for _ in range(3):
        await pilot.pause()
    return _painted(app.screen)[widget.region.y]


def _shown_rows(modal) -> list[str]:
    return [
        name
        for name in FIELD_ROW_FIELDS
        if modal.query_one(f"#{field_control_id(name)}-row").region.area
    ]


@pytest.mark.asyncio
async def test_model_view_opens_core_first_then_one_row_disclosures() -> None:
    """AC#1/#7: scope, MODEL, CORE, then Sampling, Connection, Request
    estimate and name, each disclosure one painted row; no Advanced
    generation, and every field keeps its widget id."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert not modal.query("#console-settings-generation-advanced")
        for name in FIELD_ROW_FIELDS:
            assert modal.query_one(f"#{field_control_id(name)}").id

        def top(selector: str) -> int:
            return modal.query_one(selector).region.y

        order = [
            top("#console-settings-scope"),
            top("#console-settings-model-row"),
            *(
                top(f"#{field_control_id(name)}-row")
                for name in ("temperature", "max_tokens", "streaming")
            ),
            top(f"#{SAMPLING_DISCLOSURE_ID}"),
            top(f"#{CONNECTION_DISCLOSURE_ID}"),
            top("#console-settings-request-estimate"),
            top("#console-settings-identity-advanced"),
        ]
        assert order == sorted(order) and len(set(order)) == len(order), order
        titles = (
            (SAMPLING_DISCLOSURE_ID, "Sampling"),
            (CONNECTION_DISCLOSURE_ID, "Connection"),
            ("console-settings-request-estimate", "Request estimate"),
            ("console-settings-identity-advanced", "Your name in this chat"),
        )
        painted = _painted(app.screen)
        for disclosure_id, title in titles:
            disclosure = modal.query_one(f"#{disclosure_id}", Collapsible)
            assert disclosure.collapsed is True, disclosure_id
            assert disclosure.region.height == 1, disclosure_id
            assert title in painted[disclosure.region.y], (disclosure_id, title)
        for name in SAMPLING_FIELDS:  # behind the closed Sampling disclosure
            assert not modal.query_one(f"#{field_control_id(name)}-row").region.area


@pytest.mark.parametrize("size", FULL_SCREEN_SIZES)
@pytest.mark.asyncio
async def test_every_row_paints_label_value_source_and_help(size) -> None:
    """AC#2/#3/#14: each row paints the shared label, its value, the
    resolver's Source word and the shared help line, all on one row. A blank
    row whose value no layer holds reads "provider", not "built-in": it sends
    nothing, so the provider decides (final review I6)."""
    app = CoreFirstHarness()
    settings = _settings()
    modal = _modal(app, settings)
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)
        modal.query_one(f"#{SAMPLING_DISCLOSURE_ID}", Collapsible).collapsed = False
        for _ in range(3):
            await pilot.pause()
        shown = _shown_rows(modal)
        assert {"temperature", "max_tokens", "streaming", *SAMPLING_FIELDS} <= set(
            shown
        )
        words = resolve_console_value_layers(
            app.app_config, "llama_cpp", "model-a", shown, chat_settings=settings
        )
        for name in shown:
            row = modal.query_one(f"#{field_control_id(name)}-row")
            assert row.region.height == 1, name
            line = await _painted_line(pilot, app, row)
            word = CONSOLE_VALUE_SOURCE_WORDS[words[name]]
            if words[name] is ConsoleValueLayer.BUILT_IN and getattr(settings, name) is None:
                word = CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.PROVIDER_SCALARS]
            source = modal.query_one(f"#{field_control_id(name)}-source", Static)
            assert str(source.content) == word, name
            assert MODEL_FIELD_LABELS[name] in line, (name, line)
            assert word in line, (name, line)
            help_text = MODEL_CONFIG_FIELDS[name].help
            assert help_text[:20] in line, (name, line)
            control = modal.query_one(f"#{field_control_id(name)}")
            value = (
                {"on": "On", "off": "Off"}[control.value]
                if name == "streaming"
                else getattr(control, "value", "")
            )
            if isinstance(value, str) and value:
                assert value in line, (name, value, line)
        temperature = await _painted_line(
            pilot, app, modal.query_one("#console-settings-temperature-row")
        )
        assert "0.4" in temperature and "model default" in temperature
        max_tokens = await _painted_line(
            pilot, app, modal.query_one("#console-settings-max-tokens-row")
        )
        assert "2048" in max_tokens and "Console Behavior" in max_tokens


@pytest.mark.asyncio
async def test_keyboard_edits_paint_the_new_value_and_edited_source() -> None:
    """AC#3/#6/#14: real keypresses repaint the value and mark it edited."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert app.focused is modal.query_one("#console-settings-temperature", Input)
        await pilot.press("ctrl+a", "0", ".", "9")
        await pilot.press("tab")
        await pilot.press("ctrl+a", "8", "1", "9", "2")
        await pilot.press("tab")
        streaming = modal.query_one("#console-settings-streaming", Select)
        assert app.focused is streaming
        before = streaming.value
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("down" if before == "on" else "up", "enter")
        for _ in range(4):
            await pilot.pause()
        assert streaming.value != before
        assert modal._streaming_draft is (streaming.value == "on")
        painted = _painted(app.screen)
        for name, value in (
            ("temperature", "0.9"),
            ("max_tokens", "8192"),
            ("streaming", {"on": "On", "off": "Off"}[streaming.value]),
        ):
            line = painted[modal.query_one(f"#{field_control_id(name)}-row").region.y]
            assert value in line and "edited *" in line, (name, line)


@pytest.mark.asyncio
async def test_untouched_streaming_shows_its_effective_value_and_stays_inherited() -> None:
    """AC#6 (R15): an On/Off Select shows the effective value; the draft
    stays Inherit (None) until the user picks, so Save as model default
    keeps deleting the override."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings(streaming=False))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        streaming = modal.query_one("#console-settings-streaming", Select)
        assert [str(prompt) for prompt, _ in streaming._options] == ["On", "Off"]
        assert streaming.value == "off"
        assert modal._build_draft().streaming is False
        # A transferred Inherit (None) draft shows its effective value and
        # stays None until the user picks.
        modal._streaming_draft = None
        modal._streaming_effective_fallback = True
        modal._show_streaming_value()
        await pilot.pause()
        assert streaming.value == "on"
        assert modal._streaming_draft is None


@pytest.mark.asyncio
async def test_blank_field_says_what_it_sends_and_no_placeholder_shows() -> None:
    """AC#4: a blank field shows what it inherits and where from; no
    Model view Input carries a placeholder that could read as a value, and a
    blank choice Select names what it inherits instead of "Select" (final
    review I6). A cleared required field names its range (T1 review item 5)."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        name_input = modal.query_one("#console-settings-user-display-name", Input)
        for widget in (*modal.query(".console-settings-field-row Input"), name_input):
            assert widget.placeholder == "", widget.id
        modal.query_one(f"#{SAMPLING_DISCLOSURE_ID}", Collapsible).collapsed = False
        for _ in range(3):
            await pilot.pause()
        top_k = modal.query_one("#console-settings-top-k", Input)
        assert top_k.value == ""
        help_line = modal.query_one("#console-settings-top-k-help", Static)
        assert str(help_line.content).startswith(BLANK_FIELD_HELP)
        line = await _painted_line(pilot, app, top_k)
        assert BLANK_FIELD_HELP in line and " provider " in line, line
        assert "built-in" not in line, line
        effort = modal.query_one("#console-settings-reasoning-effort", Select)
        assert effort.value is Select.NULL
        line = await _painted_line(pilot, app, effort)
        assert BLANK_CHOICE_PROMPT in line and "Select" not in line, line
        temperature = modal.query_one("#console-settings-temperature", Input)
        temperature.focus()
        temperature.clear()
        for _ in range(3):
            await pilot.pause()
        help_line = modal.query_one("#console-settings-temperature-help", Static)
        valid = MODEL_CONFIG_FIELDS["temperature"].valid_range
        assert str(help_line.content) == f"Required: {valid}."


@pytest.mark.parametrize("theme", ["agentic_terminal", "textual-light"])
@pytest.mark.asyncio
async def test_help_and_source_text_clear_aa_against_the_modal(theme) -> None:
    """AC#5: Source words and help lines paint at >= 4.5:1 on the modal."""
    from tldw_chatbook.css.Themes.themes import agentic_terminal_theme

    app = CoreFirstHarness()
    app.register_theme(agentic_terminal_theme)
    modal = _modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        app.theme = theme
        await _open(pilot, app, modal)
        checked = 0
        for name in ("temperature", "max_tokens", "streaming"):
            for part in ("source", "help"):
                widget = modal.query_one(f"#{field_control_id(name)}-{part}", Static)
                text = str(widget.content)
                offset = len(text) - len(text.lstrip())
                x = widget.content_region.x + offset
                glyph, fg, bg = _painted_cell(app.screen, x, widget.region.y)
                assert glyph == text.strip()[0], (name, part, glyph)
                assert _ratio(fg, bg) >= 4.5, (theme, name, part, fg, bg)
                checked += 1
        assert checked == 6


@pytest.mark.asyncio
async def test_focus_opens_on_temperature_and_tab_reaches_apply() -> None:
    """AC#8: Temperature has focus on open; Tab walks the shown rows and
    disclosure titles to Apply, never into collapsed contents or hidden
    (unsupported) rows."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings())
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert app.focused is modal.query_one("#console-settings-temperature", Input)
        hidden = [
            name for name in CORE_FIELDS
            if not modal.query_one(f"#{field_control_id(name)}-row").display
        ]
        assert hidden, "llama.cpp hides the controls it does not take"
        visited: list[object] = []
        for _ in range(30):
            await pilot.press("tab")
            visited.append(app.focused)
            if app.focused is modal.query_one("#console-settings-save", Button):
                break
        else:
            raise AssertionError("Tab never reached Apply")
        for widget in visited:
            for ancestor in widget.ancestors:
                if isinstance(ancestor, Collapsible):
                    assert widget.parent is ancestor, (widget, ancestor.id)
            assert widget.id not in {field_control_id(name) for name in hidden}
        titles = [
            widget.parent.id for widget in visited if isinstance(widget.parent, Collapsible)
        ]
        assert titles == [
            SAMPLING_DISCLOSURE_ID,
            CONNECTION_DISCLOSURE_ID,
            "console-settings-request-estimate",
            "console-settings-identity-advanced",
        ]
        await pilot.press("shift+tab")
        assert app.focused is not modal.query_one("#console-settings-save", Button)


@pytest.mark.asyncio
async def test_blocked_chat_opens_connection_on_its_recovery_action() -> None:
    """AC#8 with R6/R13: a missing key opens Connection and focuses the
    recovery action, so setup never starts inside the tuning fields."""
    app = CoreFirstHarness()
    app.app_config["api_settings"]["openai"] = {}
    modal = _modal(app, _settings("openai", "gpt-5", temperature=0.7))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        assert modal.query_one(f"#{CONNECTION_DISCLOSURE_ID}", Collapsible).collapsed is False
        assert app.focused is modal.query_one(
            "#console-settings-configure-credential", Button
        )


_FIT_CHATS = (
    _settings(),
    _settings("openai", "gpt-5", temperature=0.7),
    _settings("anthropic", "claude-opus-4-8", temperature=0.7),
    _settings("custom-ep:gpu-box", "model-a", temperature=0.7),
)


def _most_core_rows(app_config) -> int:
    """The most CORE rows any mapped provider shows, for any fit-set model.

    TASK-33006.2's review fix decides a registry entry as its family, so the
    worst case is no longer the registry entry's 8 rows but OpenAI's 6.
    """
    from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP
    from tldw_chatbook.Chat.console_provider_support import (
        console_generation_control_support,
        supported_generation_fields,
    )
    from tldw_chatbook.Widgets.Console.console_settings_field_row import (
        _SUPPORT_CONTROL_FIELDS,
    )

    def rows(provider: str, model: str) -> int:
        supported = supported_generation_fields(provider, model, app_config)
        return sum(
            console_generation_control_support(provider, model, name, app_config)
            != "unsupported"
            if name in _SUPPORT_CONTROL_FIELDS
            else name in supported
            for name in CORE_FIELDS
        )

    return max(
        rows(provider, model)
        for provider in PROVIDER_PARAM_MAP
        for model in ("model-a", *(chat.model for chat in _FIT_CHATS))
    )


@pytest.mark.parametrize("size", FULL_SCREEN_SIZES)
@pytest.mark.parametrize("settings", _FIT_CHATS, ids=lambda s: s.provider)
@pytest.mark.parametrize("edited", [False, True], ids=["fresh", "edited"])
@pytest.mark.asyncio
async def test_model_view_fits_through_the_footer_without_scrolling(
    size, settings, edited
) -> None:
    """AC#9: with disclosures collapsed, nothing scrolls and Apply paints
    inside the modal, before and after an edit shows the defaults line."""
    app = CoreFirstHarness()
    modal = _modal(app, settings)
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)
        if edited:
            await pilot.press("ctrl+a", "1")
            for _ in range(4):
                await pilot.pause()
        body = modal.query_one("#console-settings-body", ScrollableContainer)
        if settings.provider == "openai":  # the worst case is in the set
            assert len(
                [name for name in CORE_FIELDS if modal.query_one(
                    f"#{field_control_id(name)}-row").display]
            ) == _most_core_rows(app.app_config)
        assert body.max_scroll_y == 0, (size, settings.provider, body.virtual_size)
        assert not modal.query_one("#console-settings-fold-hint").display
        container = modal.query_one("#console-settings-modal")
        apply = modal.query_one("#console-settings-save", Button)
        assert apply.display and apply.region.area
        assert container.region.contains_region(apply.region)
        assert "Apply to this chat" in _painted(app.screen)[apply.region.y] or (
            str(apply.label) in _painted(app.screen)[apply.region.y]
        )


@pytest.mark.parametrize("size", FULL_SCREEN_SIZES)
@pytest.mark.asyncio
async def test_modal_size_comes_from_the_width_and_height_tokens(size) -> None:
    """AC#10: 150x22 from $ds-size-150 / $ds-size-22, capped to the viewport."""
    sheet = (CSS_ROOT / "features" / "_console_panels.tcss").read_text("utf-8")
    block = sheet.split("#console-settings-modal {", 1)[1].split("}", 1)[0]
    assert "width: $ds-size-150;" in block
    assert "height: $ds-size-22;" in block
    app = CoreFirstHarness()
    modal = _modal(app, _settings())
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)
        container = modal.query_one("#console-settings-modal")
        assert (container.region.width, container.region.height) == (150, 22)
        assert not container.has_class("-conversation-settings-wide")


@pytest.mark.parametrize("size", FULL_SCREEN_SIZES)
@pytest.mark.asyncio
async def test_context_view_labels_leave_a_blank_cell_before_their_controls(
    size,
) -> None:
    """TASK-33006.6 AC#1: every Context and memory label paints whole with at
    least one blank cell before its control. 'Conversation max tokens' is
    exactly as wide as the shared 23-cell label column, so it used to run
    into its input."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings(), focus_context=True)
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)
        labels = list(
            modal.query("#console-settings-context-view .console-settings-modal-label")
        )
        assert len(labels) == 10
        for label in labels:
            siblings = list(label.parent.children)
            control = siblings[siblings.index(label) + 1]
            row = await _painted_line(pilot, app, control)
            text = str(label.render())
            before = row[label.region.x : control.region.x]
            assert before.startswith(text), (text, row)
            assert len(before) > len(text) and not before[len(text) :].strip(), (
                text,
                row,
            )


@pytest.mark.parametrize(
    ("size", "model", "context"), (((211, 44), 23, 24), ((90, 40), 16, 16))
)
@pytest.mark.asyncio
async def test_label_column_comes_from_css_per_view_and_tier(size, model, context) -> None:
    """TASK-33006.6: the label column is CSS, not inline styles, so it can
    follow the view: 23 cells in the Model view, 24 in the Context view,
    and 16 in the compact tier in both."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings(model=None))  # a blocked chat opens Connection
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)

        def widths() -> set[int]:
            labels = modal.query(".console-settings-modal-label")
            return {label.region.width for label in labels if label.region.area}

        assert widths() == {model}
        modal.query_one("#console-settings-view-context", Button).press()
        for _ in range(4):
            await pilot.pause()
        assert widths() == {context}


_TO_CONTEXT = ("model", "context", "Model capacity", "console-context-budget-mode")


@pytest.mark.parametrize(
    ("size", "scrolled", "opened", "first_section", "first_control"),
    (
        ((211, 44), *_TO_CONTEXT),
        ((235, 52), *_TO_CONTEXT),
        # The Context view fits at 235x52, so only 211x44 can scroll it.
        ((211, 44), "context", "model", "model-a", "console-settings-temperature"),
    ),
)
@pytest.mark.asyncio
async def test_a_view_switch_opens_the_new_view_at_its_top(
    size, scrolled, opened, first_section, first_control
) -> None:
    """TASK-33006.7 AC#1-#3: both views share one scroll body, so a switch
    kept the other view's offset and Model capacity sat above the fold. The
    new view opens at its top: its first section paints, and the control it
    opens on (the shared open-focus rule) has focus and is visible. Sampling
    and Connection are open so the Model view scrolls at both sizes."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings())
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)
        for disclosure_id in (SAMPLING_DISCLOSURE_ID, CONNECTION_DISCLOSURE_ID):
            modal.query_one(f"#{disclosure_id}", Collapsible).collapsed = False
        body = modal.query_one("#console-settings-body", ScrollableContainer)
        if scrolled == "context":
            await pilot.click("#console-settings-view-context")
        await _settle(pilot)
        body.scroll_end(animate=False, immediate=True)
        await _settle(pilot)
        assert body.scroll_y > 0, "precondition: the view scrolled"

        await pilot.click(f"#console-settings-view-{opened}")
        await pilot.pause(0.2)  # the modal's focus reveal settles on a timer
        await _settle(pilot)

        viewport = body.content_region
        section = (
            modal.query("#console-settings-context-view .destination-section").first()
            if opened == "context"
            else modal.query_one("#console-settings-model-row")
        )
        assert viewport.contains_region(section.region), (body.scroll_y, section.region)
        assert first_section in _painted(app.screen)[section.region.y]
        focused = app.focused
        assert focused is not None and focused.id == first_control
        assert viewport.contains_region(focused.region), (focused.region, viewport)


@pytest.mark.asyncio
async def test_a_view_switch_to_a_blocked_model_view_focuses_its_fix() -> None:
    """TASK-33006.7 AC#2: the switch shares the open-focus rule, so a chat
    with no model lands on Change, as it does when Chat settings opens."""
    app = CoreFirstHarness()
    modal = _modal(app, _settings(model=None), focus_context=True)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        await pilot.click("#console-settings-view-model")
        await pilot.pause(0.2)
        await _settle(pilot)
        focused = app.focused
        assert focused is not None and focused.id == MODEL_CHANGE_ID
        body = modal.query_one("#console-settings-body", ScrollableContainer)
        assert body.content_region.contains_region(focused.region)


def _missing_key_modal(app: CoreFirstHarness, **kwargs) -> ConsoleSettingsModal:
    app.app_config["api_settings"]["openai"] = {}
    return _modal(app, _settings("openai", "gpt-5", temperature=0.7), **kwargs)


@pytest.mark.parametrize("size", FULL_SCREEN_SIZES)
@pytest.mark.asyncio
async def test_a_view_switch_to_a_missing_key_chat_lands_as_open_does(size) -> None:
    """TASK-33006.7 review: a missing key's Model view opens with the Model
    row first and Configure credential focused below the tuning rows. A
    switch used to leave focus on the tab while the new-chat default block
    (shown for a blocked chat) pulled the body to its end; the fix then
    scrolled to the top of the viewport, with the Model row above the fold
    at both sizes. The Context view scrolls only at 211x44."""
    app = CoreFirstHarness()
    modal = _missing_key_modal(app, focus_context=True)
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)
        body = modal.query_one("#console-settings-body", ScrollableContainer)
        body.scroll_end(animate=False, immediate=True)
        await _settle(pilot)

        await pilot.click("#console-settings-view-model")
        await pilot.pause(0.2)  # the modal's focus reveal settles on a timer
        await _settle(pilot)

        assert modal.query_one(f"#{CONNECTION_DISCLOSURE_ID}", Collapsible).collapsed is False
        credential = modal.query_one("#console-settings-configure-credential", Button)
        assert app.focused is credential
        viewport = body.content_region
        row = modal.query_one("#console-settings-model-row")
        assert body.scroll_y == 0, (body.scroll_y, row.region)
        assert viewport.contains_region(row.region)
        assert "gpt-5 · OpenAI" in _painted(app.screen)[row.region.y]
        assert viewport.contains_region(credential.region)


@pytest.mark.asyncio
async def test_pressing_the_shown_views_tab_does_nothing() -> None:
    """TASK-33006.7 review: re-pressing the shown view's tab is not a switch,
    so it neither moves focus into the view nor re-opens a Connection
    disclosure the user closed."""
    app = CoreFirstHarness()
    modal = _missing_key_modal(app)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        connection = modal.query_one(f"#{CONNECTION_DISCLOSURE_ID}", Collapsible)
        await pilot.click(f"#{CONNECTION_DISCLOSURE_ID} CollapsibleTitle")
        await _settle(pilot)
        assert connection.collapsed is True, "precondition: the user closed it"

        await pilot.click("#console-settings-view-model")
        await pilot.pause(0.2)
        await _settle(pilot)

        assert connection.collapsed is True
        focused = app.focused
        assert focused is not None and focused.id == "console-settings-view-model"
