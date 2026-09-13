"""Regression controls for widget-local CSS subject keys.

These selectors keep their existing owner ancestry and presentation while
moving the rightmost key off Textual's globally common bare widget types.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.geometry import Spacing
from textual.widgets import Button

from Tests.UI.consolidated_css import APP_STYLESHEETS, CSS_DIR, ConsolidatedCSSApp
from tldw_chatbook.css import build_css

from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    ProviderContinuationRecoveryCallout,
    ProviderContinuationRecoveryState,
)
from tldw_chatbook.UI.Library_Modules.skill_import_choice_modal import (
    SkillImportChoiceModal,
)
from tldw_chatbook.UI.Screens.scheduling.forms.automation_definition_form import (
    AutomationDefinitionForm,
)
from tldw_chatbook.UI.Screens.scheduling.forms.new_task_choice_modal import (
    NewTaskChoiceModal,
)
from tldw_chatbook.Widgets.Settings_Widgets.tool_profiles_panel import (
    ToolProfilesPanel,
)


@pytest.mark.parametrize(
    ("owner", "css_attribute", "old_subject", "new_subject"),
    (
        (
            AutomationDefinitionForm,
            "BUNDLED_CSS",
            "AutomationDefinitionForm > VerticalScroll",
            "AutomationDefinitionForm > VerticalScroll.automation-form-scroll",
        ),
        (
            AutomationDefinitionForm,
            "BUNDLED_CSS",
            ".button-container Button",
            ".button-container Button.automation-form-action",
        ),
        (
            NewTaskChoiceModal,
            "BUNDLED_CSS",
            "NewTaskChoiceModal .new-task-choice-actions Button",
            "NewTaskChoiceModal .new-task-choice-actions Button.new-task-choice-action",
        ),
        (
            ProviderContinuationRecoveryCallout,
            "DEFAULT_CSS",
            "ProviderContinuationRecoveryCallout Button",
            "ProviderContinuationRecoveryCallout Button.provider-continuation-action",
        ),
        (
            SkillImportChoiceModal,
            "BUNDLED_CSS",
            "#skill-import-choice-actions Button",
            "#skill-import-choice-actions Button.skill-import-choice-action",
        ),
        (
            ToolProfilesPanel,
            "BUNDLED_CSS",
            "ToolProfilesPanel Button",
            "ToolProfilesPanel Button.tool-profile-button",
        ),
        (
            ToolProfilesPanel,
            "BUNDLED_CSS",
            "ToolProfilesPanel .tool-profile-actions Button",
            "ToolProfilesPanel .tool-profile-actions Button.tool-profile-button",
        ),
    ),
)
def test_owner_css_has_no_ancestor_scoped_bare_type_subject(
    owner: type[object],
    css_attribute: str,
    old_subject: str,
    new_subject: str,
) -> None:
    """The seven paid-down subjects retain owner scope under a class key."""
    css = getattr(owner, css_attribute)
    assert f"{old_subject} {{" not in css
    assert f"{new_subject} {{" in css


class _FullCSSApp(ConsolidatedCSSApp):
    """Load the complete production CSS union for cascade-parity checks."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]


_BASELINE_SELECTOR_REPLACEMENTS = (
    (
        "AutomationDefinitionForm > VerticalScroll.automation-form-scroll",
        "AutomationDefinitionForm > VerticalScroll",
    ),
    (
        ".button-container Button.automation-form-action",
        ".button-container Button",
    ),
    (
        "NewTaskChoiceModal .new-task-choice-actions Button.new-task-choice-action",
        "NewTaskChoiceModal .new-task-choice-actions Button",
    ),
    (
        "#skill-import-choice-actions Button.skill-import-choice-action",
        "#skill-import-choice-actions Button",
    ),
    (
        "ToolProfilesPanel Button.tool-profile-button",
        "ToolProfilesPanel Button",
    ),
    (
        "ToolProfilesPanel .tool-profile-actions Button.tool-profile-button",
        "ToolProfilesPanel .tool-profile-actions Button",
    ),
)


class _BaselineFullCSSApp(_FullCSSApp):
    """Pair current widgets with the pre-paydown generated selectors."""

    def _get_default_css(self):  # noqa: D102 - diagnostic cascade mirror
        sources = []
        for location, css, tie_breaker, scope in build_css.widget_defaults_sources(
            CSS_DIR
        ):
            for current, baseline in _BASELINE_SELECTOR_REPLACEMENTS:
                css = css.replace(current, baseline)
            sources.append((location, css, tie_breaker, scope))
        return sources + super(ConsolidatedCSSApp, self)._get_default_css()


class _Host(_FullCSSApp):
    """Mount one non-screen widget under the complete production CSS union."""

    def __init__(self, child: object) -> None:
        super().__init__()
        self._child = child

    def compose(self) -> ComposeResult:
        yield self._child


class _BaselineHost(_BaselineFullCSSApp):
    """Mount one widget under the paired pre-paydown CSS source."""

    def __init__(self, child: object) -> None:
        super().__init__()
        self._child = child

    def compose(self) -> ComposeResult:
        yield self._child


def _painted(app: ConsolidatedCSSApp) -> str:
    strips = app.screen._compositor.render_strips()
    return "\n".join("".join(segment.text for segment in strip) for strip in strips)


def _hit_widget(app: ConsolidatedCSSApp, button: Button) -> object:
    x = button.region.x + button.region.width // 2
    y = button.region.y + button.region.height // 2
    hit, _region = app.screen._compositor.get_widget_at(x, y)
    return hit


def _computed_signature(widget: object) -> tuple[object, ...]:
    styles = widget.styles
    return (
        repr(styles.width),
        repr(styles.min_width),
        repr(styles.max_width),
        repr(styles.height),
        repr(styles.min_height),
        repr(styles.max_height),
        repr(styles.margin),
        repr(styles.padding),
        repr(styles.border),
        repr(styles.outline),
        repr(styles.background),
        repr(styles.color),
        repr(styles.text_style),
        tuple(widget.region),
    )


async def _state_signature(
    app: ConsolidatedCSSApp, pilot: object, button: Button
) -> tuple[object, ...]:
    """Capture computed, painted, geometry, and hit state for comparison."""
    states = []
    for disabled, focused in ((False, False), (True, False), (False, True)):
        app.screen.set_focus(None)
        button.disabled = disabled
        if focused:
            button.focus()
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        await pilot.pause()
        x = button.region.x + button.region.width // 2
        y = button.region.y + button.region.height // 2
        hit = _hit_widget(app, button)
        states.append(
            (
                _computed_signature(button),
                repr(button.visual_style),
                repr(app.screen._compositor.get_style_at(x, y)),
                type(hit).__name__,
                getattr(hit, "id", None),
                _painted(app),
            )
        )
    return tuple(states)


async def _assert_state_paint_and_geometry_parity(
    app: ConsolidatedCSSApp,
    pilot: object,
    button: Button,
    label: str,
) -> None:
    """Pin baseline geometry/hit ownership across normal, disabled, focus."""
    app.screen.set_focus(None)
    await pilot.pause()
    baseline_region = button.region
    normal_visual = repr(button.visual_style)
    assert label in _painted(app)
    assert _hit_widget(app, button) is button

    button.disabled = True
    await pilot.pause()
    assert button.region == baseline_region
    assert label in _painted(app)
    assert _hit_widget(app, button) is button
    disabled_visual = repr(button.visual_style)
    assert button.disabled

    button.disabled = False
    button.focus()
    await pilot.pause()
    assert button.region == baseline_region
    assert label in _painted(app)
    assert _hit_widget(app, button) is button
    focused_visual = repr(button.visual_style)

    assert normal_visual
    assert disabled_visual
    assert focused_visual != disabled_visual


def _profile_panel() -> ToolProfilesPanel:
    profile = SimpleNamespace(
        profile_id="research",
        origin="imported",
        lifecycle_valid=True,
        binding_state="unbound",
        first_bind_confirmation_required=True,
        reference_counts=(0, 0),
        posture_counts=(4, 3, 2),
        receipt_health="available",
        removal_eligible=True,
        removal_blocker=None,
        revision=3,
        policy_digest="a" * 64,
    )
    return ToolProfilesPanel(
        SimpleNamespace(profiles=(profile,), unavailable_category=None)
    )


async def _capture_surface(
    app_type: type[_FullCSSApp],
    host_type: type[_Host],
    surface: str,
    size: tuple[int, int],
) -> tuple[object, ...]:
    if surface == "provider":
        callout = ProviderContinuationRecoveryCallout(
            state=ProviderContinuationRecoveryState(
                "assistant-owner", 1, "local", "A tool may not have finished.", True
            ),
            on_action=lambda *_args: True,
        )
        app = host_type(callout)
        selector = "#console-continuation-resume"
        extra = ()
    elif surface == "profiles":
        app = host_type(_profile_panel())
        selector = "#tool-profile-export-0"
        extra = ()
    else:
        app = app_type()
        modal = {
            "automation": lambda: AutomationDefinitionForm(object()),
            "new-task": NewTaskChoiceModal,
            "skill": lambda: SkillImportChoiceModal(("skill-a", "skill-b")),
        }[surface]()
        selector = {
            "automation": "#automation-preview-btn",
            "new-task": "#new-task-choice-reminder",
            "skill": "#skill-import-choice-import",
        }[surface]
        extra = ()

    async with app.run_test(size=size) as pilot:
        if surface not in {"provider", "profiles"}:
            await app.push_screen(modal)
        await pilot.pause()
        if surface == "automation":
            extra = (
                _computed_signature(
                    app.screen.query_one("#automation-form-box", VerticalScroll)
                ),
            )
        return (*extra, await _state_signature(app, pilot, app.screen.query_one(selector, Button)))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("surface", "size"),
    (
        ("automation", (80, 24)),
        ("automation", (120, 50)),
        ("new-task", (80, 24)),
        ("new-task", (120, 40)),
        ("skill", (80, 24)),
        ("skill", (120, 40)),
        ("provider", (38, 16)),
        ("provider", (160, 45)),
        ("profiles", (80, 24)),
        ("profiles", (160, 45)),
    ),
)
async def test_class_key_preserves_pre_paydown_full_cascade_and_paint(
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    size: tuple[int, int],
) -> None:
    """Pair old and new subjects under the complete production CSS union."""
    current_provider_css = ProviderContinuationRecoveryCallout.DEFAULT_CSS
    if surface == "provider":
        monkeypatch.setattr(
            ProviderContinuationRecoveryCallout,
            "DEFAULT_CSS",
            current_provider_css.replace(
                "ProviderContinuationRecoveryCallout Button.provider-continuation-action",
                "ProviderContinuationRecoveryCallout Button",
            ),
        )
    baseline = await _capture_surface(
        _BaselineFullCSSApp, _BaselineHost, surface, size
    )
    monkeypatch.setattr(
        ProviderContinuationRecoveryCallout, "DEFAULT_CSS", current_provider_css
    )
    current = await _capture_surface(_FullCSSApp, _Host, surface, size)
    assert current == baseline


@pytest.mark.asyncio
@pytest.mark.parametrize(("size", "expected_width"), (((80, 24), 80), ((120, 50), 84)))
async def test_automation_form_subjects_keep_baseline_styles_and_paint(
    size: tuple[int, int], expected_width: int
) -> None:
    app = _FullCSSApp()
    async with app.run_test(size=size) as pilot:
        await app.push_screen(AutomationDefinitionForm(object()))
        await pilot.pause()

        scroll = app.screen.query_one("#automation-form-box", VerticalScroll)
        assert scroll.has_class("automation-form-scroll")
        assert scroll.styles.width.value == 84
        assert scroll.styles.max_width.value == 100
        assert scroll.styles.padding == Spacing(1, 2, 1, 2)
        assert scroll.styles.border.top[0] == "thick"
        assert scroll.region.width == expected_width

        button = app.screen.query_one("#automation-preview-btn", Button)
        assert button.styles.margin == Spacing(0, 1, 0, 1)
        await _assert_state_paint_and_geometry_parity(app, pilot, button, "Preview")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((80, 24), (120, 40)))
@pytest.mark.parametrize(
    ("modal_factory", "selector", "label", "margin", "min_width", "height"),
    (
        (
            NewTaskChoiceModal,
            "#new-task-choice-reminder",
            "Scheduled task",
            Spacing(0, 0, 0, 1),
            16,
            1,
        ),
        (
            lambda: SkillImportChoiceModal(("skill-a", "skill-b")),
            "#skill-import-choice-import",
            "Import skill",
            Spacing(0, 1, 0, 0),
            10,
            3,
        ),
    ),
)
async def test_choice_modal_actions_keep_baseline_styles_and_paint(
    size: tuple[int, int],
    modal_factory: object,
    selector: str,
    label: str,
    margin: Spacing,
    min_width: int,
    height: int,
) -> None:
    app = _FullCSSApp()
    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal_factory())
        await pilot.pause()
        button = app.screen.query_one(selector, Button)
        assert button.styles.margin == margin
        assert button.styles.min_width.value == min_width
        assert button.region.height == height
        await _assert_state_paint_and_geometry_parity(app, pilot, button, label)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((38, 16), (160, 45)))
async def test_provider_recovery_actions_keep_baseline_styles_and_paint(
    size: tuple[int, int]
) -> None:
    callout = ProviderContinuationRecoveryCallout(
        state=ProviderContinuationRecoveryState(
            "assistant-owner", 1, "local", "A tool may not have finished.", True
        ),
        on_action=lambda *_args: True,
    )
    app = _Host(callout)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        button = app.screen.query_one("#console-continuation-resume", Button)
        assert button.styles.margin == Spacing(0, 1, 0, 0)
        assert button.styles.min_width.value == 10
        assert button.region.height == 1
        await _assert_state_paint_and_geometry_parity(app, pilot, button, "Resume")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((80, 24), (160, 45)))
async def test_tool_profile_actions_keep_baseline_styles_and_paint(
    size: tuple[int, int]
) -> None:
    profile = SimpleNamespace(
        profile_id="research",
        origin="imported",
        lifecycle_valid=True,
        binding_state="unbound",
        first_bind_confirmation_required=True,
        reference_counts=(0, 0),
        posture_counts=(4, 3, 2),
        receipt_health="available",
        removal_eligible=True,
        removal_blocker=None,
        revision=3,
        policy_digest="a" * 64,
    )
    panel = ToolProfilesPanel(
        SimpleNamespace(profiles=(profile,), unavailable_category=None)
    )
    app = _Host(panel)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        toolbar = app.screen.query_one("#tool-profiles-import", Button)
        action = app.screen.query_one("#tool-profile-export-0", Button)

        assert toolbar.styles.margin == Spacing(0, 1, 0, 0)
        assert action.styles.margin == Spacing(0, 0, 0, 0)
        for button in (toolbar, action):
            assert button.styles.min_width.value == 8
            assert button.styles.height.value == 1
            assert button.styles.min_height.value == 1
        await _assert_state_paint_and_geometry_parity(
            app, pilot, action, "Export"
        )
