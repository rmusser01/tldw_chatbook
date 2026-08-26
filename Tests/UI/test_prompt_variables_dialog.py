"""Shared Prompt-variable dialog contracts (TASK-199)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, Static

import tldw_chatbook
import tldw_chatbook.Widgets.Console.prompt_variables_dialog as dialog_module
from tldw_chatbook.Prompt_Management.prompt_variables import (
    PromptVariableApplication,
    fingerprint_system_text,
)
from tldw_chatbook.Widgets.Console.prompt_variables_dialog import (
    APPLY_BUTTON_ID,
    CANCEL_BUTTON_ID,
    DESTINATION_COPY_ID,
    ORIGINAL_BUTTON_ID,
    STATUS_ID,
    SYSTEM_CHECKBOX_ID,
    SYSTEM_COPY_ID,
    SYSTEM_STATE_ID,
    VARIABLE_INPUT_CLASS,
    VARIABLE_ROW_CLASS,
    VARIABLES_SCROLL_ID,
    PromptVariablesDialog,
    PromptVariablesDialogRequest,
)


COMPOSER_FINGERPRINT = "a" * 64
SESSION_ID = "console-session-7"


class DialogHarness(App[None]):
    """Minimal host that records the dialog dismissal value."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[PromptVariableApplication | None] = []

    def compose(self) -> ComposeResult:
        yield Static("Console")

    def show(self, request: PromptVariablesDialogRequest) -> None:
        self.push_screen(PromptVariablesDialog(request), callback=self.results.append)


class BundledDialogHarness(DialogHarness):
    """Dialog harness using the same generated stylesheet as production."""

    CSS_PATH = str(
        Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
    )


def _request(
    *,
    system_text: str | None = None,
    user_text: str | None = "Hello {customer}",
    destination: str = "replace_snapshot",
) -> PromptVariablesDialogRequest:
    return PromptVariablesDialogRequest(
        system_text=system_text,
        user_text=user_text,
        destination=destination,
        target_session_id=SESSION_ID,
        composer_fingerprint=(
            COMPOSER_FINGERPRINT if destination == "replace_snapshot" else None
        ),
        system_fingerprint=(
            fingerprint_system_text("current system")
            if system_text is not None
            else None
        ),
    )


def _row_state(dialog: PromptVariablesDialog) -> list[tuple[str, str, Input]]:
    state: list[tuple[str, str, Input]] = []
    for row in dialog.query(f".{VARIABLE_ROW_CLASS}"):
        label = str(row.query_one(Static).renderable)
        name = label.partition(" — ")[0]
        value_input = row.query_one(Input)
        state.append((name, label, value_input))
    return state


def _input_for(dialog: PromptVariablesDialog, name: str) -> Input:
    return next(
        value_input
        for row_name, _label, value_input in _row_state(dialog)
        if row_name == name
    )


def _painted_region_text(widget: Widget) -> str:
    lines = [
        "".join(segment.text for segment in strip)
        for strip in widget.screen._compositor.render_strips()
    ]
    region = widget.region
    return " ".join(
        lines[y][region.x : region.right].strip()
        for y in range(region.y, region.bottom)
    ).strip()


def _painted_relative_luminance(color) -> float:
    triplet = color.get_truecolor()

    def channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(triplet.red)
        + 0.7152 * channel(triplet.green)
        + 0.0722 * channel(triplet.blue)
    )


def _painted_contrast(first, second) -> float:
    lighter, darker = sorted(
        (_painted_relative_luminance(first), _painted_relative_luminance(second)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _painted_style_of_text(app: App, region, needle: str):
    strips = list(app.screen._compositor.render_strips())
    for y in range(region.y, region.bottom):
        segments = list(strips[y]._segments)
        row_text = "".join(segment.text for segment in segments)
        index = row_text.find(needle)
        if index == -1:
            continue
        x = 0
        for segment in segments:
            if x + len(segment.text) > index:
                return segment.style
            x += len(segment.text)
    return None


def test_request_is_frozen_validated_and_repr_safe() -> None:
    secret_system = "System {customer} secret"
    secret_user = "User {customer} secret"
    request = _request(system_text=secret_system, user_text=secret_user)

    assert secret_system not in repr(request)
    assert secret_user not in repr(request)
    assert COMPOSER_FINGERPRINT not in repr(request)
    assert "sha256:" not in repr(request)
    with pytest.raises(FrozenInstanceError):
        request.user_text = "changed"  # type: ignore[misc]

    with pytest.raises(ValueError, match="at least one source lane"):
        _request(system_text=None, user_text=None)
    with pytest.raises(ValueError, match="System fingerprint"):
        PromptVariablesDialogRequest(
            system_text="System",
            user_text="User",
            destination="replace_snapshot",
            target_session_id=SESSION_ID,
            composer_fingerprint=COMPOSER_FINGERPRINT,
            system_fingerprint=None,
        )


def test_request_validation_does_not_construct_an_expiring_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_constructed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("application TTL must start only at user confirmation")

    monkeypatch.setattr(dialog_module, "PromptVariableApplication", fail_if_constructed)

    request = _request(system_text="System {customer}")

    assert isinstance(request, PromptVariablesDialogRequest)
    with pytest.raises(ValueError, match="composer fingerprint"):
        PromptVariablesDialogRequest(
            system_text=None,
            user_text="User",
            destination="replace_snapshot",
            target_session_id=SESSION_ID,
            composer_fingerprint=None,
            system_fingerprint=None,
        )


@pytest.mark.asyncio
async def test_dialog_shows_truthful_destination_and_system_authorization_copy() -> (
    None
):
    app = DialogHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        app.show(_request(system_text="System {tone}"))
        await pilot.pause()
        dialog = app.screen

        assert isinstance(dialog, PromptVariablesDialog)
        assert (
            str(dialog.query_one(f"#{DESTINATION_COPY_ID}", Static).renderable)
            == "Replace the current Console draft"
        )
        checkbox = dialog.query_one(f"#{SYSTEM_CHECKBOX_ID}", Checkbox)
        assert str(checkbox.label) == (
            "Replace the current session System prompt with this System lane"
        )
        assert checkbox.value is False
        assert len(dialog.query(VerticalScroll)) == 1


@pytest.mark.asyncio
async def test_unique_case_sensitive_variables_show_once_in_first_occurrence_order() -> (
    None
):
    app = DialogHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        app.show(
            _request(
                user_text="{customer}/{customer}/{Customer}/{region}",
            )
        )
        await pilot.pause()
        dialog = app.screen

        labels = [
            str(row.query_one(Static).renderable)
            for row in dialog.query(f".{VARIABLE_ROW_CLASS}")
        ]
        assert labels == ["customer — User", "Customer — User", "region — User"]
        inputs = list(dialog.query(f".{VARIABLE_INPUT_CLASS}"))
        assert len(inputs) == 3
        assert all(
            isinstance(widget, Input) and widget.value == "" for widget in inputs
        )
        assert all(
            widget.id is not None
            and "customer" not in widget.id.lower()
            and "region" not in widget.id.lower()
            for widget in inputs
        )
        assert "{customer}" not in repr(dialog)
        assert not dialog.query(f"#{SYSTEM_CHECKBOX_ID}")


@pytest.mark.asyncio
async def test_blank_value_is_valid_and_apply_returns_only_rendered_active_lane() -> (
    None
):
    app = DialogHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        app.show(_request(user_text="Dear {customer}"))
        await pilot.pause()
        dialog = app.screen
        apply_button = dialog.query_one(f"#{APPLY_BUTTON_ID}", Button)

        assert apply_button.disabled is False
        assert dialog.query_one(f".{VARIABLE_INPUT_CLASS}", Input).value == ""
        await pilot.click(f"#{APPLY_BUTTON_ID}")
        await pilot.pause()

    assert len(app.results) == 1
    result = app.results[0]
    assert isinstance(result, PromptVariableApplication)
    assert result.system_text is None
    assert result.user_text == "Dear "
    assert result.apply_system is False
    assert result.apply_user is True
    assert "Dear" not in repr(result)


@pytest.mark.asyncio
async def test_library_destination_copy_is_exact() -> None:
    app = DialogHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        app.show(_request(destination="append_active"))
        await pilot.pause()

        assert (
            str(app.screen.query_one(f"#{DESTINATION_COPY_ID}", Static).renderable)
            == "Append to the current Console draft"
        )


@pytest.mark.asyncio
async def test_system_toggle_recomputes_lanes_and_restores_hidden_values_and_focus() -> (
    None
):
    app = DialogHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        app.show(
            _request(
                system_text="System {system_only} {shared}",
                user_text="User {shared} {user_only}",
            )
        )
        await pilot.pause()
        dialog = app.screen

        assert [label for _name, label, _input in _row_state(dialog)] == [
            "shared — User",
            "user_only — User",
        ]
        _input_for(dialog, "shared").value = "COMMON"
        _input_for(dialog, "user_only").value = "USER"
        _input_for(dialog, "shared").focus()

        await pilot.click(f"#{SYSTEM_CHECKBOX_ID}")
        await pilot.pause()
        assert [label for _name, label, _input in _row_state(dialog)] == [
            "system_only — System",
            "shared — System + User",
            "user_only — User",
        ]
        assert _input_for(dialog, "shared").value == "COMMON"
        assert _input_for(dialog, "user_only").value == "USER"
        assert _input_for(dialog, "shared").has_focus

        _input_for(dialog, "system_only").value = "SYSTEM"
        await pilot.click(f"#{SYSTEM_CHECKBOX_ID}")
        await pilot.pause()
        assert [name for name, _label, _input in _row_state(dialog)] == [
            "shared",
            "user_only",
        ]

        await pilot.click(f"#{SYSTEM_CHECKBOX_ID}")
        await pilot.pause()
        assert _input_for(dialog, "system_only").value == "SYSTEM"
        assert _input_for(dialog, "shared").value == "COMMON"

        await pilot.click(f"#{APPLY_BUTTON_ID}")
        await pilot.pause()

    result = app.results[0]
    assert isinstance(result, PromptVariableApplication)
    assert result.system_text == "System SYSTEM COMMON"
    assert result.user_text == "User COMMON USER"
    assert result.apply_system is True
    assert result.apply_user is True


@pytest.mark.asyncio
async def test_system_only_defaults_to_no_active_lane_until_explicitly_selected() -> (
    None
):
    app = DialogHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        app.show(_request(system_text="System {tone}", user_text=None))
        await pilot.pause()
        dialog = app.screen

        assert dialog.query_one(f"#{APPLY_BUTTON_ID}", Button).disabled is True
        assert dialog.query_one(f"#{ORIGINAL_BUTTON_ID}", Button).disabled is True
        assert dialog.query_one(f"#{CANCEL_BUTTON_ID}", Button).disabled is False
        assert str(dialog.query_one(f"#{STATUS_ID}", Static).renderable) == (
            "Select a lane to apply"
        )

        await pilot.click(f"#{SYSTEM_CHECKBOX_ID}")
        await pilot.pause()
        assert dialog.query_one(f"#{APPLY_BUTTON_ID}", Button).disabled is False
        assert dialog.query_one(f"#{ORIGINAL_BUTTON_ID}", Button).disabled is False
        assert str(dialog.query_one(f"#{STATUS_ID}", Static).renderable) == ""


@pytest.mark.asyncio
async def test_system_authorization_remains_keyboard_accessible() -> None:
    app = DialogHarness()
    async with app.run_test(size=(64, 24)) as pilot:
        app.show(_request(system_text="System {tone}"))
        await pilot.pause()
        checkbox = app.screen.query_one(f"#{SYSTEM_CHECKBOX_ID}", Checkbox)
        checkbox.focus()

        await pilot.press("space")
        await pilot.pause()

        assert checkbox.value is True
        assert (
            str(app.screen.query_one(f"#{SYSTEM_STATE_ID}", Static).renderable) == "On"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "expected_status"),
    [
        (
            "{" + "A" * 65 + "}",
            "A Prompt variable name exceeds 64 characters.",
        ),
        (
            " ".join(f"{{v{index}}}" for index in range(65)),
            "This Prompt has more than 64 variables.",
        ),
    ],
)
async def test_limit_error_blocks_rendering_but_keeps_original_source_available(
    source: str,
    expected_status: str,
) -> None:
    app = DialogHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        app.show(_request(user_text=source))
        await pilot.pause()
        dialog = app.screen

        status = dialog.query_one(f"#{STATUS_ID}", Static)
        assert str(status.renderable) == expected_status
        assert dialog.query_one(f"#{APPLY_BUTTON_ID}", Button).disabled is True
        assert dialog.query_one(f"#{ORIGINAL_BUTTON_ID}", Button).disabled is False
        assert dialog.query_one(f"#{CANCEL_BUTTON_ID}", Button).disabled is False
        assert len(str(status.renderable)) < 80

        await pilot.click(f"#{ORIGINAL_BUTTON_ID}")
        await pilot.pause()

    result = app.results[0]
    assert isinstance(result, PromptVariableApplication)
    assert result.user_text == source
    assert source not in expected_status


@pytest.mark.asyncio
async def test_use_original_returns_selected_source_without_interpolation() -> None:
    source = "[bold]Dear {customer}; keep {{literal}}[/bold]"
    app = DialogHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        app.show(_request(user_text=source))
        await pilot.pause()
        value_input = app.screen.query_one(f".{VARIABLE_INPUT_CLASS}", Input)
        value_input.value = "[italic]SECRET[/italic]"

        await pilot.click(f"#{ORIGINAL_BUTTON_ID}")
        await pilot.pause()

    result = app.results[0]
    assert isinstance(result, PromptVariableApplication)
    assert result.user_text == source
    assert "SECRET" not in repr(result)
    assert source not in repr(result)


@pytest.mark.asyncio
async def test_use_original_with_system_selected_returns_both_exact_sources() -> None:
    system_source = "System {customer} {{literal}}"
    user_source = "User {customer} {{literal}}"
    request = _request(system_text=system_source, user_text=user_source)
    app = DialogHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        app.show(request)
        await pilot.pause()
        await pilot.click(f"#{SYSTEM_CHECKBOX_ID}")
        await pilot.pause()
        _input_for(app.screen, "customer").value = "INTERPOLATED"

        await pilot.click(f"#{ORIGINAL_BUTTON_ID}")
        await pilot.pause()

    result = app.results[0]
    assert isinstance(result, PromptVariableApplication)
    assert result.system_text == system_source
    assert result.user_text == user_source
    assert result.apply_system is True
    assert result.apply_user is True
    assert result.system_fingerprint == request.system_fingerprint
    assert "INTERPOLATED" not in (result.system_text, result.user_text)


@pytest.mark.asyncio
async def test_disabled_application_actions_paint_at_three_to_one_contrast() -> None:
    app = BundledDialogHarness()
    async with app.run_test(size=(64, 24)) as pilot:
        app.show(_request(system_text="System {customer}", user_text=None))
        await pilot.pause()

        for button_id, label in (
            (ORIGINAL_BUTTON_ID, "Use original placeholders"),
            (APPLY_BUTTON_ID, "Apply"),
        ):
            button = app.screen.query_one(f"#{button_id}", Button)
            assert button.disabled is True
            style = _painted_style_of_text(app, button.region, label)
            assert style is not None
            assert style.color is not None and style.bgcolor is not None
            ratio = _painted_contrast(style.color, style.bgcolor)
            assert ratio >= 3.0, f"{button_id} paints at only {ratio:.2f}:1"


@pytest.mark.asyncio
async def test_markup_looking_value_remains_literal_when_applied() -> None:
    app = DialogHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        app.show(_request(user_text="Hello {customer}"))
        await pilot.pause()
        value_input = app.screen.query_one(f".{VARIABLE_INPUT_CLASS}", Input)
        value_input.value = "[bold magenta]ACME[/bold magenta]"
        assert value_input.value == "[bold magenta]ACME[/bold magenta]"

        await pilot.click(f"#{APPLY_BUTTON_ID}")
        await pilot.pause()

    result = app.results[0]
    assert isinstance(result, PromptVariableApplication)
    assert result.user_text == "Hello [bold magenta]ACME[/bold magenta]"


@pytest.mark.asyncio
async def test_zero_variables_supports_rendered_escapes_and_original_source() -> None:
    source = "Keep {{customer}} and [bold]literal[/bold]"
    rendered_app = DialogHarness()
    async with rendered_app.run_test(size=(120, 40)) as pilot:
        rendered_app.show(_request(user_text=source))
        await pilot.pause()
        assert len(rendered_app.screen.query(f".{VARIABLE_ROW_CLASS}")) == 0
        assert (
            rendered_app.screen.query_one(f"#{APPLY_BUTTON_ID}", Button).disabled
            is False
        )
        await pilot.click(f"#{APPLY_BUTTON_ID}")
        await pilot.pause()
    rendered = rendered_app.results[0]
    assert isinstance(rendered, PromptVariableApplication)
    assert rendered.user_text == "Keep {customer} and [bold]literal[/bold]"

    original_app = DialogHarness()
    async with original_app.run_test(size=(120, 40)) as pilot:
        original_app.show(_request(user_text=source))
        await pilot.pause()
        await pilot.click(f"#{ORIGINAL_BUTTON_ID}")
        await pilot.pause()
    original = original_app.results[0]
    assert isinstance(original, PromptVariableApplication)
    assert original.user_text == source


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_control", [CANCEL_BUTTON_ID, "escape"])
async def test_cancel_and_escape_dismiss_without_application(exit_control: str) -> None:
    app = DialogHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        app.show(_request())
        await pilot.pause()
        if exit_control == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(f"#{exit_control}")
        await pilot.pause()

    assert app.results == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(64, 24), (120, 40)])
async def test_real_bundle_keeps_one_scroll_owner_and_fixed_actions_reachable(
    size: tuple[int, int],
) -> None:
    variables = " ".join(f"{{v{index}}}" for index in range(64))
    app = BundledDialogHarness()
    async with app.run_test(size=size) as pilot:
        app.show(_request(system_text="System {shared}", user_text=variables))
        await pilot.pause()
        dialog = app.screen
        shell = dialog.query_one("#prompt-variables-dialog")
        scroll = dialog.query_one(f"#{VARIABLES_SCROLL_ID}", VerticalScroll)
        actions = dialog.query_one("#prompt-variables-actions")
        action_y = actions.region.y

        assert len(dialog.query(VerticalScroll)) == 1
        assert scroll.max_scroll_y > 0
        assert scroll.region.bottom <= actions.region.y
        assert shell.content_region.contains_region(actions.region)
        for button_id in (CANCEL_BUTTON_ID, ORIGINAL_BUTTON_ID, APPLY_BUTTON_ID):
            button = dialog.query_one(f"#{button_id}", Button)
            assert 0 <= button.region.y < size[1]
            assert cell_len(str(button.label)) <= button.content_region.width
        painted = "\n".join(
            strip.text for strip in app.screen._compositor.render_strips()
        )
        normalized_paint = " ".join(painted.split())
        assert "Replace the current Console draft" in normalized_paint
        system_copy = dialog.query_one(f"#{SYSTEM_COPY_ID}", Static)
        assert _painted_region_text(system_copy) == (
            "Replace the current session System prompt with this System lane"
        )
        assert "Use original placeholders" in normalized_paint
        assert "Cancel" in normalized_paint
        assert "Apply" in normalized_paint
        state = dialog.query_one(f"#{SYSTEM_STATE_ID}", Static)
        checkbox = dialog.query_one(f"#{SYSTEM_CHECKBOX_ID}", Checkbox)
        off_control = _painted_region_text(checkbox)
        assert off_control
        assert "…" not in off_control
        assert str(state.renderable) == "Off"
        assert _painted_region_text(state) == "Off"

        await pilot.click(f"#{SYSTEM_CHECKBOX_ID}")
        await pilot.pause()
        state = dialog.query_one(f"#{SYSTEM_STATE_ID}", Static)
        on_control = _painted_region_text(checkbox)
        assert on_control
        assert "…" not in on_control
        assert str(state.renderable) == "On"
        assert _painted_region_text(state) == "On"
        assert (off_control, "Off") != (on_control, "On")
        assert dialog.query_one(f"#{ORIGINAL_BUTTON_ID}", Button).disabled is False

        scroll.scroll_to(y=scroll.max_scroll_y, animate=False)
        await pilot.pause()
        assert actions.region.y == action_y
        assert scroll.scroll_y == scroll.max_scroll_y

        await pilot.click(f"#{CANCEL_BUTTON_ID}")
        await pilot.pause()

    assert app.results == [None]


@pytest.mark.asyncio
async def test_max_length_variable_label_wraps_instead_of_clipping_at_narrow_size() -> (
    None
):
    variable_name = "A" * 64
    app = BundledDialogHarness()
    async with app.run_test(size=(64, 24)) as pilot:
        app.show(_request(user_text=f"{{{variable_name}}}"))
        await pilot.pause()
        row = app.screen.query_one(f".{VARIABLE_ROW_CLASS}")
        label = row.query_one(Static)

        assert str(label.renderable) == f"{variable_name} — User"
        assert cell_len(str(label.renderable)) > label.content_region.width
        assert label.region.height >= 2
