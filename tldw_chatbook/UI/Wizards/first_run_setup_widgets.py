"""Legacy hints, Textual bindings and best-effort recovery are preserved verbatim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional  # noqa: UP035

from loguru import logger
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.compose import compose as _drain_compose_result
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Checkbox, RadioButton, RadioSet, Static
from textual.widgets.option_list import Option

from ...Library.library_shell_state import (
    LIBRARY_GLYPH_RADIO_SELECTED,
    LIBRARY_GLYPH_RADIO_UNSELECTED,
)
from . import first_run_setup_state as wizard_state
from . import first_run_step_guard as step_guard
from .BaseWizard import WizardStep

if TYPE_CHECKING:
    from textual.geometry import Region


class SetupRadioButton(RadioButton):
    """RadioButton whose selected state is structural, not color-only.

    TASK-1497: stock ToggleButton renders one constant BUTTON_INNER glyph and
    conveys on/off purely through the glyph's color, which is invisible in a
    monochrome capture and fails WCAG 1.4.1 (use of color). The inner glyph
    itself switches here — the Library legend's radio pair — so state survives
    any palette; a bold text-style on the selected row (see _wizards.tcss) is
    the second cue. BUTTON_INNER is set as an instance attribute right before
    the parent property renders, shadowing the class attribute per-state.

    task-32464 fix round 1: the pair used to be two literals here, the same
    duplication that surface's twin (``ConsoleAccessRadioButton``) carried.
    Both now read the one definition in ``Library.library_shell_state``, so a
    legend change cannot pass either of them by.
    """

    @property
    def _button(self):
        # BUTTON_INNER is ToggleButton's documented per-instance glyph seam;
        # super() resolves the parent property without importing Textual's
        # private module or touching .fget. The remaining coupling (that a
        # ``_button`` property renders the glyph at all) is pinned by
        # test_selected_and_unselected_glyphs_differ_structurally, so a
        # Textual upgrade that changes the mechanism fails loudly in CI
        # instead of silently regressing to color-only state.
        self.BUTTON_INNER = (
            LIBRARY_GLYPH_RADIO_SELECTED
            if self.value
            else LIBRARY_GLYPH_RADIO_UNSELECTED
        )
        return super()._button


def _radio_model_id(button) -> str:
    """The raw embedding model id a `SetupRadioButton` was built from.

    Falls back to the rendered label for buttons built before the id was
    carried on `name` (tier-2 review S21 P3).
    """
    return str(getattr(button, "name", None) or button.label)


class SetupCheckbox(Checkbox):
    """Checkbox whose checked state is structural, not color-only.

    TASK-21146 follow-on to TASK-1497 (SetupRadioButton): stock
    ToggleButton renders a constant "X" glyph and conveys on/off purely
    through color — in live UAT the UNCHECKED consent box read as checked
    (▐X▌). The inner glyph itself switches: ✓ checked, blank unchecked.
    """

    @property
    def _button(self):
        self.BUTTON_INNER = "✓" if self.value else " "
        return super()._button


class SetupRadioSet(RadioSet):
    """Wizard radio group with WAI-ARIA radio semantics (TASK-21142).

    UAT N-8: stock RadioSet separates highlight from selection, so a user
    who arrows to "Full setup" and presses Next silently proceeds on the
    Quick track — the highlight glyph is far subtler than the ● selection
    glyph and reads as a no-op. Here selection follows the highlight, the
    way OS radio groups and the WAI-ARIA radio pattern behave.

    UAT N-1: stock RadioSet consumes Enter as a redundant re-toggle. With
    selection following the highlight there is nothing left for Enter to
    toggle, so it requests a wizard advance instead — the "form + Enter =
    continue" reflex.
    """

    class AdvanceRequested(Message):
        """Enter on a settled radio group asks the wizard to advance."""

    BINDINGS = [Binding("enter", "request_advance", "Next", show=False)]  # noqa: RUF012

    def action_next_button(self) -> None:
        """Move the highlight down and select it (selection follows focus)."""
        super().action_next_button()
        self._select_highlighted()

    def action_previous_button(self) -> None:
        """Move the highlight up and select it (selection follows focus)."""
        super().action_previous_button()
        self._select_highlighted()

    def _select_highlighted(self) -> None:
        # Follow the highlight only during USER navigation: RadioSet's own
        # _on_mount calls action_next_button() to seat the initial
        # highlight, and following that call would auto-select the first
        # option on every mount — clobbering deliberately-unselected
        # groups (AppearanceStep's fresh-run theme radio commits nothing
        # precisely because nothing is pressed). Focus is the discriminator:
        # key bindings only fire on the focused set.
        if not self.has_focus:
            return
        # ``_selected`` is RadioSet's highlight index — private, but this
        # repo pins textual >=8,<9 and the coupling fails loudly in the
        # keyboard contract tests on any upgrade that changes it.
        index = self._selected
        buttons = list(self.query(RadioButton))
        if index is None or not (0 <= index < len(buttons)):
            return
        button = buttons[index]
        if not button.value and not button.disabled:
            button.value = True

    def action_request_advance(self) -> None:
        self._select_highlighted()
        self.post_message(self.AdvanceRequested())


_SETUP_STEP_FAILURE_REASONS = frozenset({"compose_failed", "render_failed"})


REQUIRED_STEP_MANUAL_SETTINGS_CATEGORIES: Mapping[str, str] = {
    wizard_state.STEP_WELCOME: "diagnostics",
    wizard_state.STEP_PROVIDER: "providers-models",
    wizard_state.STEP_MODEL: "providers-models",
    wizard_state.STEP_VOICE: "speech-tts",
    wizard_state.STEP_SPEECH: "speech-tts",
    wizard_state.STEP_TOOLS: "advanced-config",
    wizard_state.STEP_NOTES: "advanced-config",
    wizard_state.STEP_APPEARANCE: "appearance",
    wizard_state.STEP_PROTECT: "privacy-security",
    wizard_state.STEP_SUMMARY: "diagnostics",
}


def manual_settings_context_for_required_step(
    step_id: str,
) -> dict[str, str] | None:
    """Return the actionable Settings context for a known required step."""

    category = REQUIRED_STEP_MANUAL_SETTINGS_CATEGORIES.get(step_id)
    if category is None:
        return None
    return {"category": category}


@dataclass(frozen=True, slots=True)
class SetupStepFailure:
    """Secret-free failure state shared by a step and its container."""

    step_id: str
    required: bool
    reason_code: str

    def __post_init__(self) -> None:
        if self.reason_code not in _SETUP_STEP_FAILURE_REASONS:
            raise ValueError("unsupported setup step failure reason")


class SetupStep(step_guard.WizardErrorGuard, WizardStep):
    """Base step: adds an awaitable commit hook and an inline error line.

    TASK-1495: also tags every setup step with its own ``setup-step`` CSS
    class. BaseWizard.py's shared ``.wizard-step`` rule (never modified --
    see this module's own docstring) is ``height: 100%`` with no overflow,
    which silently clips any step whose natural content is taller than the
    surrounding ``.wizard-steps-container`` -- Provider's ~27-row provider
    list plus its API-key field is the case that motivated this fix. Scoping
    the scroll-region CSS to ``.setup-step`` (added here, in this module
    only) rather than touching ``.wizard-step`` itself keeps the Chatbook
    wizards -- whose steps carry no ``setup-step`` class -- byte-for-byte
    unaffected; see ``_wizards.tcss``'s "First-run setup wizard" section for
    the actual scroll/height rules keyed off this class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_class("setup-step")
        self.compose_failed = False
        self.compose_failure: SetupStepFailure | None = None

    @property
    def required(self) -> bool:
        """Whether this step may be omitted, derived from its real config."""

        return self.config is None or not self.config.can_skip

    #: Compatibility flag set when compose_step() raises. Failure policy is
    #: carried by compose_failure, which distinguishes required from optional.
    compose_failed: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Guard subclass lifecycle hooks against a failed compose.

        TASK-1266: a step whose compose_step() raised has none of its usual
        widgets, so its own on_mount/on_show (which query them) would crash
        the mount. Rather than asking every step to re-check the flag, wrap
        the hooks here once. All current hooks are sync (asserted by the
        wrapper returning None on skip).

        Args:
            cls: The subclass being defined; supplied automatically by
                Python whenever a ``SetupStep`` subclass is created.
            **kwargs: Forwarded to ``super().__init_subclass__()``; unused
                by this hook itself.
        """
        super().__init_subclass__(**kwargs)
        import functools

        for hook_name in ("on_mount", "on_show"):
            hook = cls.__dict__.get(hook_name)
            if hook is None:
                continue

            def _make(wrapped):
                @functools.wraps(wrapped)
                def _guarded(self, *args: Any, **kw: Any):
                    if getattr(self, "compose_failed", False):
                        return None
                    return wrapped(self, *args, **kw)

                return _guarded

            setattr(cls, hook_name, _make(hook))

    def compose(self) -> ComposeResult:
        """Final wrapper: render compose_step(), degrading on failure.

        A step whose composition raises must never crash the wizard screen.
        Required steps render recovery actions in place; optional steps render
        a bounded skip notice and are reported by Summary. Subclasses implement
        ``compose_step``.

        Finding A fix: ``compose_step()`` is fully drained into a list
        BEFORE anything is yielded to Textual. The original ``yield from
        self.compose_step()`` streamed each widget straight through as it
        was produced, so a step that yielded some widgets and THEN raised
        left those already-yielded widgets mounted -- rendering a
        half-built form ABOVE the "couldn't be shown" notice, which then
        lied about the step having been skipped. Buffering means either
        ALL of ``compose_step()``'s widgets are yielded (success) or NONE
        are (failure -- notice only).

        Returns:
            Yields ``compose_step()``'s widgets on success. On a raised
            exception, yields the bounded required-recovery or optional-skip
            surface and records a ``SetupStepFailure``.
        """
        if self.compose_failure is not None:
            self.compose_failed = True
            yield from self._failure_widgets(self.compose_failure)
            return

        try:
            # Finding A: drain compose_step() through Textual's OWN
            # textual.compose.compose() helper -- NOT a plain list(...) --
            # because plain list() steals every yielded value away from
            # Textual's per-item "attach to the enclosing with-block
            # container" step (compose_add_child), which normally runs
            # inside the SAME loop that calls next() on this generator.
            # Nested containers (``with SetupRadioSet(): yield SetupRadioButton``)
            # would silently end up childless -- their leaves float as
            # stray top-level siblings instead -- if drained with a bare
            # list(). textual.compose.compose() reproduces that per-item
            # attach step itself, so it is safe to fully exhaust up front.
            buffered = _drain_compose_result(self, self.compose_step())
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Wizard step composition failed (category=compose, error_type={})",
                type(exc).__name__,
            )
            self.compose_failed = True
            self.compose_failure = SetupStepFailure(
                step_id=self.config.id if self.config else "unknown",
                required=self.required,
                reason_code="compose_failed",
            )
            yield from self._failure_widgets(self.compose_failure)
            return
        yield from buffered

    def _failure_widgets(self, failure: SetupStepFailure) -> list[Widget]:
        if not failure.required:
            return [
                Static(
                    "This optional step couldn't be shown and was skipped; "
                    "its settings remain available in Settings.",
                    classes="setup-step-error",
                )
            ]
        return [
            Static(
                "This required step couldn't be shown. Retry here, continue "
                "in Settings, or exit setup and return later.",
                classes="setup-step-error",
            ),
            Horizontal(
                Button("Retry", id="setup-step-retry", variant="primary"),
                Button(
                    "Use manual setup",
                    id="setup-step-manual",
                    disabled=manual_settings_context_for_required_step(failure.step_id)
                    is None,
                ),
                Button("Exit setup", id="setup-step-later"),
                classes="setup-step-recovery-actions",
            ),
        ]

    def compose_step(self) -> ComposeResult:
        """Step content; override in subclasses (default: framework empty).

        Returns:
            Yields this step's content widgets. The default (unoverridden)
            body yields whatever ``WizardStep.compose()`` yields -- a single
            empty ``Container()``; concrete steps override this to yield
            their own field layout.
        """
        yield from super().compose()

    async def commit(self) -> tuple[bool, str]:
        """Persist this step's data. Return (ok, error_message)."""
        return True, ""

    def preferred_focus(self) -> Optional[Widget]:  # noqa: UP045
        """The widget this step wants focused on entry, or None.

        Returns:
            A displayed, focusable descendant to focus when the step is
            shown, or None to fall back to the container's first-displayed-
            focusable heuristic. Steps whose DOM order puts a conditional
            affordance ahead of their primary control (ProviderStep's pinned
            discovery button) override this so re-entry cannot land focus on
            the secondary control.
        """
        return None

    def confirm_before_advance(self) -> Optional[str]:  # noqa: UP045
        """A question the user must answer before Next commits this step.

        TASK-21143 (UAT M-2): steps that KNOW their state is broken (a
        failed credential probe) return the question here; the container
        shows it as a confirmation dialog and only advances on an explicit
        "Continue anyway". None (the default) advances normally.
        """
        return None

    def show_step_error(self, message: str) -> None:
        """Render a step error on the wizard's pinned error strip.

        TASK-21140 (UAT W-1/G-3 and the F-1 investigation): the previous
        per-step ``.setup-step-error`` tail Static sat at the BOTTOM of an
        overflowing scroll region -- a refused Next rendered its reason
        below the fold and the wizard just looked stuck. The pinned strip
        lives in the container chrome between the step body and the nav
        bar, so it is visible at any terminal size. It is cleared on every
        step change (``show_step``).
        """
        try:
            strip = self.screen.query_one("#setup-step-error-pinned", Static)
        except Exception:  # noqa: BLE001
            logger.warning("Setup step error had nowhere to render: {}", message)
            return
        strip.update(message)
        strip.remove_class("hidden")

    def refresh(
        self,
        *regions: "Region",  # noqa: UP037
        repaint: bool = True,
        layout: bool = False,
        recompose: bool = False,
    ) -> "SetupStep":  # noqa: UP037
        """TASK-22281 (UAT F-1): recompose must never orphan keyboard focus.

        A recompose rebuilds this step's children; if ``app.focused`` is one
        of them, Textual 8.2.8 leaves it pointing at the DETACHED widget --
        every subsequent key event then dispatches into a dead message pump,
        so no binding anywhere (container ctrl+n/ctrl+b, screen escape, even
        the app palette) ever resolves and the wizard soft-locks. Confirmed
        live: SpeechSetupStep's first ``on_show`` schedules exactly this
        recompose an instant before ``show_step``'s focus fix targets a
        pre-recompose child. The heal runs after every recompose (first-show
        load, load-completion, discovery updates alike) because a one-shot
        fix at the focus site would be re-orphaned by the next recompose.
        """
        if recompose:
            try:
                self.call_after_refresh(self._heal_orphaned_focus)
            except Exception:  # noqa: BLE001, S110
                # Not mounted yet (compose-time refresh): nothing to heal.
                pass
        return super().refresh(
            *regions, repaint=repaint, layout=layout, recompose=recompose
        )

    def _heal_orphaned_focus(self) -> None:
        """Re-anchor focus if the recompose just detached the focused widget.

        No-ops when focus is alive (attached and displayed) or when this
        step is hidden -- a background recompose on a non-visible step must
        never steal focus from the step the user is on. Restore priority:
        the same-id widget in the rebuilt tree (so focus appears not to
        move), then this step's preferred/first focusable, then the wizard
        nav bar -- mirroring show_step()'s F-B focus fix so the container
        stays in the focused widget's ancestry and ctrl+n/ctrl+b resolve.
        """
        try:
            app = self.app
        except Exception:  # noqa: BLE001
            return
        focused = app.focused
        if focused is not None and focused.is_attached and focused.display:
            return
        if not self.display or not self.is_attached:
            return
        target: Optional[Widget] = None  # noqa: UP045
        prior_id = getattr(focused, "id", None) if focused is not None else None
        if prior_id:
            try:
                candidate = self.query_one(f"#{prior_id}", Widget)
                if (
                    candidate.focusable
                    and candidate.display
                    and not candidate.has_class("hidden")
                ):
                    target = candidate
            except Exception:  # noqa: BLE001
                target = None
        if target is None:
            preferred = self.preferred_focus()
            if (
                preferred is not None
                and preferred.focusable
                and preferred.display
                and not preferred.has_class("hidden")
            ):
                target = preferred
        if target is None:
            target = next(
                (
                    widget
                    for widget in self.walk_children(Widget)
                    if widget.focusable
                    and widget.display
                    and not widget.has_class("hidden")
                ),
                None,
            )
        if target is None:
            try:
                target = self.screen.query_one("#wizard-next", Button)
            except Exception:  # noqa: BLE001
                return
        target.focus()


class ProviderChoiceOption(Option):
    """A provider row or a disabled group heading in the provider list."""

    def __init__(
        self,
        prompt: Text,
        *,
        option_id: str,
        provider_key: str | None,
    ) -> None:
        super().__init__(prompt, id=option_id, disabled=provider_key is None)
        self.provider_key: str | None = provider_key
