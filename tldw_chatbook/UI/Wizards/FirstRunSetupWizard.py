"""First-run setup wizard: hermes-agent's setup process in chatbook chrome.

Screen + container subclass over BaseWizard (which is never modified).
All decisions and config mutations are built by first_run_setup_state;
this module renders them and owns persistence via one exclusive worker.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
)

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import (
    Button,
    Input,
    Label,
    OptionList,
    RadioButton,
    RadioSet,
    Static,
)

from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.config import get_runtime_config_snapshot
from tldw_chatbook.UI.Wizards import first_run_model_discovery as model_discovery
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards import first_run_step_guard as step_guard
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupRadioButton as SetupRadioButton,  # re-exported: old import path
    _radio_model_id,
    SetupCheckbox as SetupCheckbox,  # re-exported: old import path
    SetupRadioSet,
    REQUIRED_STEP_MANUAL_SETTINGS_CATEGORIES as REQUIRED_STEP_MANUAL_SETTINGS_CATEGORIES,
    manual_settings_context_for_required_step,
    SetupStepFailure,
    SetupStep,
    ProviderChoiceOption as ProviderChoiceOption,  # re-exported: old import path
)
from tldw_chatbook.UI.Wizards.first_run_appearance_step import AppearanceStep
from tldw_chatbook.UI.Wizards.first_run_busy_status import (
    SetupBusyStatus,
    busy_label_for,
)
from tldw_chatbook.UI.Wizards.first_run_model_step import ModelStep
from tldw_chatbook.UI.Wizards.first_run_notes_step import NotesSyncStep
from tldw_chatbook.UI.Wizards.first_run_protect_step import ProtectKeysStep
from tldw_chatbook.UI.Wizards.first_run_provider_step import ProviderStep
from tldw_chatbook.UI.Wizards.first_run_rag_step import RagStep
from tldw_chatbook.UI.Wizards.first_run_speech_step import SpeechSetupStep
from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker, wizard_work
from tldw_chatbook.UI.Wizards.first_run_summary_step import SummaryStep
from tldw_chatbook.UI.Wizards.first_run_tools_step import ToolsStep
from tldw_chatbook.UI.Wizards.first_run_welcome_step import WelcomeStep
from tldw_chatbook.UI.Wizards.BaseWizard import (
    WizardContainer,
    WizardNavigation,
    WizardProgress,
    WizardScreen,
    WizardStep,
    WizardStepConfig,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

if TYPE_CHECKING:
    from tldw_chatbook.UI.Wizards.first_run_voice_step import VoiceSetupStep


#: TASK-25821: steps 1-5 teach "Esc skip setup" / "Esc exit setup" and Esc
#: works. On the summary the cancel button is hidden and Esc goes inert, but
#: the hint line simply dropped the exit vocabulary -- so the key the wizard
#: spent five screens teaching stopped working with no explanation, and
#: nothing said how setup actually ends. Name the finish route instead of
#: leaving a gap. Deliberately does NOT mention Esc: it does not exit here,
#: and the footer must only advertise keys that work (same rule as the
#: Console footer's setup-blocked variant).
SUMMARY_KEY_HINTS = "Ctrl+B back · choose an action below to finish"


class SetupWizardProgress(WizardProgress):
    #: TASK-21148 (UAT F-2/F-3): the stacked number+title layout. Declared
    #: as BUNDLED_CSS so build_css.py lifts it into the widget-defaults
    #: tier of the app bundle — a class-level DEFAULT_CSS would register
    #: another stylesheet source against Textual's 64-entry parse cache
    #: (see Tests/UI/test_widget_css_consolidation.py for the rule).
    BUNDLED_CSS = """
    SetupWizardProgress .setup-progress-item {
        height: auto;
    }
    SetupWizardProgress .step-indicator-stack {
        layout: vertical;
        align: center top;
        width: auto;
        height: auto;
    }
    SetupWizardProgress .step-number {
        width: auto;
        min-width: 4;
        margin-right: 0;
    }
    SetupWizardProgress .step-title {
        margin-right: 0;
        text-align: center;
    }
    """

    """Progress indicator rendered from the resolved first-run track."""

    _NUMBER_WIDTH = 4
    _TITLE_HORIZONTAL_MARGIN = 2
    _ITEM_HORIZONTAL_MARGIN = 2
    _CONNECTOR_WIDTH = 4
    _ITEM_SAFETY_WIDTH = 1

    def __init__(
        self,
        items: tuple[wizard_state.SetupProgressItem, ...],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.add_class("w-full")
        self.items = items
        self._sync_compatibility_state()

    def _sync_compatibility_state(self) -> None:
        self.total_steps = len(self.items)
        self.current_step = next(
            (
                index + 1
                for index, item in enumerate(self.items)
                if item.state == "active"
            ),
            1,
        )
        self.step_titles = [item.title for item in self.items]

    def set_items(self, items: tuple[wizard_state.SetupProgressItem, ...]) -> None:
        if items == self.items:
            return
        self.items = items
        self._sync_compatibility_state()
        if self.is_attached:
            self._sync_compact_mode()
            self.refresh(recompose=True)
            self.call_after_refresh(self._sync_compact_mode)

    def on_mount(self) -> None:
        self.call_after_refresh(self._sync_compact_mode)

    def on_resize(self) -> None:
        self._sync_compact_mode()

    def _titled_track_width(self) -> int:
        """Return the minimum safe width for the fully titled tracker."""

        # TASK-21148 (UAT F-2): titles render UNDER the number boxes, so an
        # item costs max(number, title) width — the full 10-step track keeps
        # its titles at 140 columns instead of collapsing to anonymous boxes.
        item_width = sum(
            max(self._NUMBER_WIDTH, len(item.title))
            + self._ITEM_HORIZONTAL_MARGIN
            + self._ITEM_SAFETY_WIDTH
            for item in self.items
        )
        connector_width = self._CONNECTOR_WIDTH * max(len(self.items) - 1, 0)
        return item_width + connector_width

    def _sync_compact_mode(self) -> None:
        compact = self.size.width < self._titled_track_width()
        self.set_class(compact, "-compact")
        for title in self.query(".step-title"):
            title.display = not compact
        for connector in self.query(".step-connector"):
            connector.display = not compact

    def compose(self) -> ComposeResult:
        compact = self.has_class("-compact")
        for index, item in enumerate(self.items):
            state_class = f"-{item.state}"
            with Container(
                id=f"setup-progress-{item.step_id}",
                classes=f"step-indicator-container setup-progress-item {state_class}",
            ):
                # TASK-21148 (UAT F-2/F-3): number box and title stack
                # vertically — titles survive at full-track widths, and the
                # box grows for two-digit step numbers instead of clipping
                # "10" down to "1".
                with Vertical(classes="step-indicator-stack"):
                    number_classes = f"step-number {item.state}"
                    # TASK-21143 (UAT N-7): "attention" = visited, but its
                    # probe failed — "!" instead of the ✓ users read as "OK".
                    yield Static(
                        "✓"
                        if item.state == "complete"
                        else "!"
                        if item.state == "attention"
                        else str(index + 1),
                        classes=number_classes,
                    )
                    title = Label(
                        item.title,
                        classes=f"step-title {item.state}",
                    )
                    title.display = not compact
                    yield title
                if index < len(self.items) - 1:
                    connector_classes = "step-connector"
                    if item.state in ("complete", "attention"):
                        connector_classes += " complete"
                    connector = Static("", classes=connector_classes)
                    connector.display = not compact
                    yield connector


@dataclass(frozen=True, slots=True)
class _SetupFailureAction:
    """Identity captured when one required-failure recovery action starts."""

    screen: "FirstRunSetupWizard"
    index: int
    step: SetupStep
    step_id: str
    failure: SetupStepFailure


class _ProviderSaveStatus(Static):
    """Focusable live status for an irreversible provider save."""

    can_focus = True


class SetupWizardNavigation(WizardNavigation):
    """Footer where Tab reaches Next before the abandon action (UAT N-2).

    TASK-21142: Textual's focus order is VISUAL order — siblings sort by
    ``_focus_sort_key`` = (y, x), not DOM order — so with the stock
    layout (Cancel far left) Tab from step content always landed on the
    abandon action first, and a web-form "Tab, Enter" reflex opened the
    exit dialog. A DOM reorder alone changes nothing (measured: the chain
    stayed Cancel-first). The fix is the Windows-wizard footer
    convention: progress text docked left, and a right-aligned
    [← Back] [Next →] [Exit] cluster — visually conventional, and the
    (y, x) sort then yields Back → Next → Exit, with Next as the first
    enabled stop after step content. Layout lives in the
    ``.setup-navigation`` rules in _wizards.tcss; BaseWizard stays
    unmodified per this module's house rule.
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="wizard-progress", classes="wizard-progress-text")
        yield Button("← Back", id="wizard-back", variant="default", disabled=True)
        yield Button("Next →", id="wizard-next", variant="default", disabled=True)
        yield Button("Cancel", id="wizard-cancel", variant="error")


class SetupWizardContainer(step_guard.WizardErrorGuard, WizardContainer):
    """Navigates over the active-step subset; commits on Next via one worker."""

    # TASK-21142 (UAT N-1): Enter advances whenever the focused widget does
    # not consume it (Buttons press, OptionLists select, Inputs submit —
    # see _advance_on_input_submit; SetupRadioSet requests an advance).
    # Merges with WizardContainer's escape/ctrl+b/ctrl+n bindings.
    BINDINGS = [Binding("enter", "next", "Next step", show=False)]

    @on(SetupRadioSet.AdvanceRequested)
    def _on_radio_advance_requested(
        self, event: SetupRadioSet.AdvanceRequested
    ) -> None:
        event.stop()
        self.action_next()

    @on(Input.Submitted)
    def _advance_on_input_submit(self, event: Input.Submitted) -> None:
        """Enter in a step Input means "continue" (UAT N-1) — with one
        exception: the provider key field, where Enter with a key launches
        the credential probe (TASK-1506's live-but-never-blocking check).

        task-32555 AC#3: Enter with NO key used to do nothing at all — no
        probe, no advance, no message. It now skips the provider (the field's
        own hint says so): the choice is cleared so the step's commit takes
        its skip-safe path, and the Summary reports the provider as not
        configured.
        """
        if event.input.id == "setup-provider-api-key":
            if event.value.strip():
                return
            step = self.steps[self.current_step]
            if isinstance(step, ProviderStep):
                step.skip_without_key()
        event.stop()
        self.action_next()

    def __init__(
        self,
        app_instance,
        rerun: bool = False,
        resume_draft: wizard_state.SetupDraft | None = None,
        provider_dismiss_warning_seconds: float = 2.0,
        **kwargs,
    ):
        self.rerun = rerun
        self.resume_draft = resume_draft
        self.key_entered = False
        self._staged_provider_draft: wizard_state.FirstRunProviderDraft | None = None
        self._provider_setup_committed = False
        self._committed_provider_model = ""
        self._committed_provider_expected_state: object | None = None
        self._provider_stage_generation = 0
        self._provider_commit_generation = 0
        self._provider_commit_lock = asyncio.Lock()
        self._provider_commit_task: asyncio.Task[bool] | None = None
        self._provider_commit_identity: (
            tuple[
                int,
                str,
                wizard_state.FirstRunModelDiscoveryKey,
                Literal["discovered", "manual"],
                object,
            ]
            | None
        ) = None
        from tldw_chatbook.Chat.provider_setup_persistence import (
            ProviderSetupWriteGuard,
        )

        self._provider_write_guard = ProviderSetupWriteGuard()
        self._provider_last_config_result: object | None = None
        self._provider_commit_write_started = False
        self._provider_cleanup_requested = False
        self._provider_dismiss_pending = False
        self._provider_ui_detached = False
        self._first_run_selected_provider_models: dict[
            wizard_state.FirstRunModelDiscoveryKey, tuple[str, ...]
        ] = {}
        self._first_run_selected_provider_outcomes: dict[
            wizard_state.FirstRunModelDiscoveryKey, object
        ] = {}
        self._first_run_provider_config_preconditions: dict[
            wizard_state.FirstRunModelDiscoveryKey, object
        ] = {}
        self._provider_dismiss_warning_seconds = max(
            0.0, float(provider_dismiss_warning_seconds)
        )
        self._draft_mutation_lock = asyncio.Lock()
        self._draft_mutations_terminal = False
        # (task-2040) MUST be set before ``_create_steps()``: step
        # constructors read ``self.wizard.app_instance`` (SpeechSetupStep
        # reads ``app_config`` through it at __init__ time), and the base
        # ``WizardContainer.__init__`` that normally assigns it runs only
        # AFTER the steps exist -- every fresh-profile first boot crashed
        # with AttributeError before this line existed. The base class
        # re-assigns the same value harmlessly.
        self.app_instance = app_instance
        # TASK-1499: default to the QUICK track — it is the preselected
        # (recommended) Welcome option, so the progress row anchors at
        # "Step 1 of 4" instead of front-loading all nine steps before
        # the user has chosen anything. Picking Full expands it.
        self.track = (
            resume_draft.track if resume_draft is not None else wizard_state.TRACK_QUICK
        )
        steps = self._create_steps()
        super().__init__(
            app_instance=app_instance,
            steps=steps,
            title="Set up tldw chatbook",
            on_complete=self._handle_complete,
            **kwargs,
        )
        self.active_ids: tuple[str, ...] = wizard_state.active_step_ids(
            self.track, key_entered=self._effective_key_entered()
        )
        self.skipped_step_reasons: dict[str, str] = {}
        self._advancing = False
        self._finishing = False  # _finalize, not _advance, lifts the fence
        self._advance_confirmed = False
        self._failure_action_running = False
        self._failure_action: _SetupFailureAction | None = None
        # F3 hardening: guards _dismiss_screen/_finalize against ever
        # dismissing the screen twice -- see those methods' docstrings.
        self._finalized = False
        if resume_draft is not None:
            self.wizard_data = {
                step_id: dict(step_values)
                for step_id, step_values in resume_draft.values.items()
            }

    @property
    def staged_provider_draft(self) -> wizard_state.FirstRunProviderDraft | None:
        """Return the in-memory provider connection staged by Provider."""

        return self._staged_provider_draft

    @property
    def provider_setup_committed(self) -> bool:
        """Whether the staged provider/model pair fully reached runtime config."""

        return self._provider_setup_committed

    @property
    def committed_provider_model(self) -> str:
        """Return the model committed with the current staged provider."""

        return self._committed_provider_model

    def invalidate_provider_model_handoff(self) -> None:
        """Clear model state derived from a superseded provider credential."""

        self.invalidate_provider_write_expectation()
        self._first_run_selected_provider_models = {}
        self._first_run_selected_provider_outcomes = {}
        self._first_run_provider_config_preconditions = {}
        self.wizard_data.pop(wizard_state.STEP_MODEL, None)
        model_index = self._step_index_for_id(wizard_state.STEP_MODEL)
        if model_index is None:
            return
        model_step = self.steps[model_index]
        if isinstance(model_step, ModelStep):
            model_step.invalidate_credential_bound_selection()

    def invalidate_provider_write_expectation(self) -> None:
        """Fence a queued provider writer without retaining credential material."""

        self._provider_write_guard.invalidate()

    def _refresh_changed_provider_identity(
        self,
        owner: ProviderStep,
        provider_draft: wizard_state.FirstRunProviderDraft,
    ) -> None:
        """Fence old model state and start discovery for the current draft."""

        current_key = owner._model_discovery_key(provider_draft)
        current_models = self._first_run_selected_provider_models.get(current_key)
        current_outcome = self._first_run_selected_provider_outcomes.get(current_key)
        current_precondition = self._first_run_provider_config_preconditions.get(
            current_key
        )
        self._first_run_selected_provider_models = (
            {current_key: current_models}
            if current_key is not None and current_models is not None
            else {}
        )
        self._first_run_selected_provider_outcomes = (
            {current_key: current_outcome}
            if current_key is not None and current_outcome is not None
            else {}
        )
        self._first_run_provider_config_preconditions = (
            {current_key: current_precondition}
            if current_key is not None and current_precondition is not None
            else {}
        )
        self.wizard_data.pop(wizard_state.STEP_MODEL, None)
        if (
            owner.is_attached
            and current_key is not None
            and not model_discovery.discovery_is_reusable(owner, current_key)
        ):
            owner._begin_selected_provider_discovery(
                provider_draft,
                sync_live_credential=False,
            )
        model_index = self._step_index_for_id(wizard_state.STEP_MODEL)
        if model_index is None:
            return
        model_step = self.steps[model_index]
        if isinstance(model_step, ModelStep) and model_step.is_attached:
            model_step.invalidate_discovery_bound_selection()

    def clear_provider_setup_sensitive_state(
        self, *, clear_widgets: bool = True
    ) -> None:
        """Fence provider work and release raw state at the valid boundary."""

        self.invalidate_provider_write_expectation()
        self._provider_stage_generation += 1
        task = self._provider_commit_task
        irreversible_write = bool(
            task is not None and not task.done() and self._provider_commit_write_started
        )
        self._provider_cleanup_requested = True
        if not irreversible_write:
            self._provider_commit_generation += 1
            if task is not None and not task.done():
                task.cancel()
            self._provider_commit_task = None
            self._provider_commit_identity = None
            self._provider_commit_write_started = False
        self._staged_provider_draft = None
        self._provider_setup_committed = False
        self._committed_provider_model = ""
        self._committed_provider_expected_state = None
        self._first_run_selected_provider_models = {}
        self._first_run_selected_provider_outcomes = {}
        self._first_run_provider_config_preconditions = {}
        owner = getattr(self, "_first_run_provider_discovery_owner", None)
        if isinstance(owner, ProviderStep):
            if not self._provider_ui_detached:
                owner._cancel_discovery_workers(publish_status=False)
            if clear_widgets and not self._provider_ui_detached:
                owner.clear_sensitive_widgets()
            owner.clear_sensitive_state()

    def on_unmount(self) -> None:
        self._provider_ui_detached = True
        self._provider_dismiss_pending = False
        self.clear_provider_setup_sensitive_state(clear_widgets=False)

    def finish_later_message(self) -> str:
        """Describe provider persistence accurately for the current step."""

        if (
            self._staged_provider_draft is not None
            and not self._provider_setup_committed
        ):
            return (
                "This provider connection is staged only in this wizard and has "
                "not been saved. Your non-secret setup progress will resume at "
                "Provider."
            )
        if self._provider_setup_committed:
            return (
                "Your provider and model are saved. Other completed setup steps "
                "are also saved, and you can continue from Settings ▸ Diagnostics."
            )
        return (
            "Steps you've already completed are saved. You can continue setup any "
            "time from Settings ▸ Diagnostics."
        )

    def stage_provider_setup(
        self, provider_draft: wizard_state.FirstRunProviderDraft
    ) -> bool:
        """Hold a provider connection in wizard memory without writing config."""

        if type(provider_draft) is not wizard_state.FirstRunProviderDraft:
            return False
        if self._provider_commit_write_started:
            return False
        self._provider_cleanup_requested = False
        if self._provider_drafts_match(self._staged_provider_draft, provider_draft):
            return True
        self.invalidate_provider_write_expectation()
        self._provider_stage_generation += 1
        self._provider_commit_generation += 1
        self._staged_provider_draft = provider_draft
        self._provider_setup_committed = False
        self._committed_provider_model = ""
        self._committed_provider_expected_state = None
        return True

    def can_validate_committed_provider_setup(
        self,
        model_id: str,
        *,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None,
        model_provenance: Literal["discovered", "manual"],
    ) -> bool:
        """Return whether Next can validate an already committed model decision."""

        from tldw_chatbook.Chat.provider_setup_persistence import (
            ExpectedProviderSetupState,
        )

        expected_state = self._committed_provider_expected_state
        if (
            type(model_id) is not str
            or type(discovery_key) is not wizard_state.FirstRunModelDiscoveryKey
            or model_provenance not in {"discovered", "manual"}
            or type(expected_state) is not ExpectedProviderSetupState
            or not self._provider_setup_committed
            or self._committed_provider_model != model_id.strip()
        ):
            return False
        identity = expected_state.identity
        return bool(
            identity.provider_key == discovery_key.provider_key
            and identity.connection_identity == discovery_key.connection_identity
            and identity.credential_source == discovery_key.credential_source
            and identity.credential_revision == discovery_key.credential_revision
            and identity.model_id == model_id.strip()
            and identity.model_provenance == model_provenance
        )

    @staticmethod
    def capture_provider_config_precondition(
        discovery_key: wizard_state.FirstRunModelDiscoveryKey,
    ) -> object | None:
        """Capture authoritative provider config for discovery or manual input."""

        if type(discovery_key) is not wizard_state.FirstRunModelDiscoveryKey:
            return None
        try:
            from tldw_chatbook.Chat.provider_setup_persistence import (
                capture_provider_setup_precondition,
            )
            from tldw_chatbook.config import get_atomic_config_snapshot

            return capture_provider_setup_precondition(
                get_atomic_config_snapshot(),
                provider=discovery_key.provider_key,
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _provider_drafts_match(
        left: wizard_state.FirstRunProviderDraft | None,
        right: wizard_state.FirstRunProviderDraft,
    ) -> bool:
        """Compare exact transient drafts without retaining a secret-derived key."""

        import hmac

        if type(left) is not wizard_state.FirstRunProviderDraft:
            return False
        if (
            left.provider != right.provider
            or left.endpoint != right.endpoint
            or left.discovery_endpoint != right.discovery_endpoint
        ):
            return False
        left_credential = left.credential
        right_credential = right.credential
        if (
            left_credential.source != right_credential.source
            or left_credential.revision != right_credential.revision
        ):
            return False
        return hmac.compare_digest(
            wizard_state._credential_value_for_boundary(left_credential),
            wizard_state._credential_value_for_boundary(right_credential),
        )

    async def commit_staged_provider_setup(
        self,
        model_id: str,
        *,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = None,
        model_provenance: Literal["discovered", "manual"] = "manual",
        config_precondition: object | None = None,
    ) -> bool:
        """Persist the staged connection and model through one atomic mutation."""

        if type(model_id) is not str:
            return False
        if discovery_key is not None and (
            type(discovery_key) is not wizard_state.FirstRunModelDiscoveryKey
        ):
            return False
        if model_provenance not in {"discovered", "manual"}:
            return False
        if config_precondition is not None:
            from tldw_chatbook.Chat.provider_setup_persistence import (
                ProviderSetupConfigPrecondition,
            )

            if type(config_precondition) is not ProviderSetupConfigPrecondition:
                return False
        normalized_model = model_id.strip()
        owner = getattr(self, "_first_run_provider_discovery_owner", None)
        changed_draft: wizard_state.FirstRunProviderDraft | None = None
        committed_validation: tuple[object, int] | None = None
        async with self._provider_commit_lock:
            if isinstance(owner, ProviderStep) and owner.is_attached:
                owner._sync_live_credential_revision()
                current_draft = owner._effective_provider_draft()
                current_key = owner._model_discovery_key(current_draft)
                expected_key = discovery_key
                if expected_key is None and self._staged_provider_draft is not None:
                    expected_key = owner._model_discovery_key(
                        self._staged_provider_draft
                    )
                if current_draft is None or current_key is None:
                    logger.debug("First-run provider save rejected (identity=invalid)")
                    return False
                if expected_key != current_key:
                    logger.debug("First-run provider save rejected (identity=changed)")
                    if not self.stage_provider_setup(current_draft):
                        return False
                    changed_draft = current_draft
                elif not self.stage_provider_setup(current_draft):
                    return False
            if changed_draft is not None:
                operation = None
            else:
                provider_draft = self._staged_provider_draft
                if provider_draft is None:
                    return False
                expected_key = discovery_key
                if expected_key is None:
                    try:
                        expected_key = wizard_state.build_first_run_model_discovery_key(
                            provider_draft
                        )
                    except ValueError:
                        return False
                if (
                    self._provider_setup_committed
                    and self._committed_provider_model == normalized_model
                ):
                    if not self.can_validate_committed_provider_setup(
                        normalized_model,
                        discovery_key=expected_key,
                        model_provenance=model_provenance,
                    ):
                        return False
                    committed_validation = (
                        self._committed_provider_expected_state,
                        self._provider_stage_generation,
                    )
                    operation = None
                else:
                    identity = (
                        self._provider_stage_generation,
                        normalized_model,
                        expected_key,
                        model_provenance,
                        config_precondition,
                    )
                    active_task = self._provider_commit_task
                    if (
                        active_task is not None
                        and not active_task.done()
                        and identity == self._provider_commit_identity
                    ):
                        operation = active_task
                    else:
                        if self._provider_commit_write_started:
                            return False
                        self._provider_commit_generation += 1
                        lease = self._provider_commit_generation
                        operation = asyncio.create_task(
                            self._run_provider_setup_commit(
                                provider_draft,
                                normalized_model,
                                expected_key,
                                model_provenance,
                                config_precondition,
                                self._provider_stage_generation,
                                lease,
                            )
                        )
                        operation.add_done_callback(self._provider_commit_finished)
                        self._provider_commit_task = operation
                        self._provider_commit_identity = identity
        if changed_draft is not None and isinstance(owner, ProviderStep):
            self._refresh_changed_provider_identity(owner, changed_draft)
            return False
        if committed_validation is not None:
            expected_state, stage_generation = committed_validation
            return await self._validate_committed_provider_setup(
                expected_state,
                stage_generation=stage_generation,
                owner=owner,
            )
        assert operation is not None
        return await asyncio.shield(operation)

    async def _validate_committed_provider_setup(
        self,
        expected_state: object,
        *,
        stage_generation: int,
        owner: object,
    ) -> bool:
        """Validate a committed no-op against one authoritative config read."""

        from tldw_chatbook.Chat.provider_setup_persistence import (
            ExpectedProviderSetupState,
            provider_setup_expected_state_matches_snapshot,
        )
        from tldw_chatbook.config import (
            ConfigMutationResult,
            get_atomic_config_snapshot,
        )

        if type(expected_state) is not ExpectedProviderSetupState:
            return False
        try:
            snapshot = await asyncio.to_thread(get_atomic_config_snapshot)
            matches = provider_setup_expected_state_matches_snapshot(
                expected_state,
                snapshot,
            )
        except (TypeError, ValueError):
            self._provider_last_config_result = ConfigMutationResult(
                False,
                False,
                "before_replace",
            )
            return False

        async with self._provider_commit_lock:
            if (
                self._provider_ui_detached
                or stage_generation != self._provider_stage_generation
                or expected_state is not self._committed_provider_expected_state
                or not self._provider_setup_committed
            ):
                return False
            if matches:
                return True
            self._provider_last_config_result = ConfigMutationResult(
                False,
                False,
                None,
                conflict=True,
                conflict_reason="identity_changed",
            )
            self._provider_setup_committed = False
            self._committed_provider_model = ""
            self._committed_provider_expected_state = None

        if isinstance(owner, ProviderStep) and owner.is_attached:
            current_draft = owner._effective_provider_draft()
            if current_draft is not None:
                self._refresh_changed_provider_identity(owner, current_draft)
        return False

    def _provider_commit_finished(self, task: asyncio.Task[bool]) -> None:
        """Consume a detached result when its awaiting caller was cancelled."""

        if not task.cancelled():
            task.exception()
        if self._provider_cleanup_requested:
            self.clear_provider_setup_sensitive_state(clear_widgets=False)

    async def _run_provider_setup_commit(
        self,
        provider_draft: wizard_state.FirstRunProviderDraft,
        model_id: str,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey,
        model_provenance: Literal["discovered", "manual"],
        config_precondition: object | None,
        stage_generation: int,
        lease: int,
    ) -> bool:
        """Own one secret-free lease from validation through atomic persistence."""

        await asyncio.sleep(0)
        current_task = asyncio.current_task()
        write_started = False
        changed_draft: wizard_state.FirstRunProviderDraft | None = None
        owner = getattr(self, "_first_run_provider_discovery_owner", None)
        try:
            async with self._provider_commit_lock:
                if (
                    lease != self._provider_commit_generation
                    or stage_generation != self._provider_stage_generation
                    or self._staged_provider_draft is not provider_draft
                ):
                    return False
                if isinstance(owner, ProviderStep) and owner.is_attached:
                    owner._sync_live_credential_revision()
                    current_draft = owner._effective_provider_draft()
                    current_key = owner._model_discovery_key(current_draft)
                    if current_draft is None or current_key is None:
                        logger.debug(
                            "First-run provider save lease rejected (identity=invalid)"
                        )
                        return False
                    if current_key != discovery_key:
                        logger.debug(
                            "First-run provider save lease rejected (identity=changed)"
                        )
                        if not self.stage_provider_setup(current_draft):
                            return False
                        changed_draft = current_draft
                    elif not self._provider_drafts_match(provider_draft, current_draft):
                        logger.debug(
                            "First-run provider save lease rejected (draft=changed)"
                        )
                        return False
                if changed_draft is not None:
                    mutation = None
                else:
                    try:
                        from tldw_chatbook.config import get_atomic_config_snapshot

                        config_snapshot = get_atomic_config_snapshot()
                        mutation = wizard_state.build_first_run_provider_commit(
                            provider_draft,
                            model_id,
                            config_snapshot.values,
                        )
                        committed_expected_state = (
                            self._bind_provider_write_expectation(
                                mutation,
                                config_snapshot=config_snapshot,
                                discovery_key=discovery_key,
                                model_id=model_id,
                                model_provenance=model_provenance,
                                config_precondition=config_precondition,
                            )
                        )
                    except (TypeError, ValueError):
                        logger.warning(
                            "First-run provider commit rejected (category=validation)"
                        )
                        return False
                    self._provider_commit_write_started = True
                    write_started = True

            if changed_draft is not None and isinstance(owner, ProviderStep):
                self._refresh_changed_provider_identity(owner, changed_draft)
                return False
            assert mutation is not None

            self._provider_last_config_result = None
            saved = await self.commit_config(
                mutation.section_values,
                delete_keys=mutation.delete_keys,
                provider_setup_mutation=mutation,
            )
            if not saved:
                result = self._provider_last_config_result
                if (
                    getattr(result, "conflict_reason", None) == "identity_changed"
                    and isinstance(owner, ProviderStep)
                    and owner.is_attached
                ):
                    self._provider_commit_write_started = False
                    write_started = False
                    current_draft = owner._effective_provider_draft()
                    if current_draft is not None and self.stage_provider_setup(
                        current_draft
                    ):
                        self._refresh_changed_provider_identity(owner, current_draft)
                return False
            if not self._provider_cleanup_requested:
                self._provider_setup_committed = True
                self._committed_provider_model = model_id
                self._committed_provider_expected_state = committed_expected_state
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("First-run provider commit failed (category=writer)")
            return False
        finally:
            async with self._provider_commit_lock:
                if write_started and self._provider_commit_task is current_task:
                    self._provider_commit_write_started = False
                if self._provider_commit_task is current_task:
                    self._provider_commit_task = None
                    self._provider_commit_identity = None

    def _bind_provider_write_expectation(
        self,
        mutation: object,
        *,
        config_snapshot: object,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey,
        model_id: str,
        model_provenance: Literal["discovered", "manual"],
        config_precondition: object | None,
    ) -> object:
        """Bind a secret-free CAS token to the issued atomic setup mutation."""

        from tldw_chatbook.Chat.provider_setup_persistence import (
            ProviderSetupConfigPrecondition,
            ProviderSetupWriteIdentity,
            bind_provider_setup_precondition,
            bind_provider_setup_write_expectation,
            capture_expected_provider_setup_state,
            project_provider_setup_expected_state,
        )

        expected_identity = ProviderSetupWriteIdentity(
            provider_key=discovery_key.provider_key,
            connection_identity=discovery_key.connection_identity,
            credential_source=discovery_key.credential_source,
            credential_revision=discovery_key.credential_revision,
            model_id=model_id,
            model_provenance=model_provenance,
        )
        expectation = self._provider_write_guard.arm(expected_identity)
        if type(config_precondition) is ProviderSetupConfigPrecondition:
            expected_state = bind_provider_setup_precondition(
                config_precondition,
                identity=expected_identity,
            )
        else:
            expected_state = capture_expected_provider_setup_state(
                config_snapshot,
                identity=expected_identity,
            )

        bind_provider_setup_write_expectation(
            mutation,
            guard=self._provider_write_guard,
            expectation=expectation,
            expected_state=expected_state,
        )
        return project_provider_setup_expected_state(
            config_snapshot,
            mutation=mutation,
            identity=expected_identity,
        )

    def compose(self) -> ComposeResult:
        """Compose with progress derived from the resolved setup track."""

        yield Label(self.title, classes="wizard-title")
        yield SetupWizardProgress(
            wizard_state.build_setup_progress(self.active_ids, 0),
            classes="wizard-progress",
        )
        with Container(classes="wizard-steps-container"):
            yield from self.steps
        yield _ProviderSaveStatus(
            "",
            id="setup-provider-save-status",
            classes="setup-step-error hidden",
            markup=False,
        )
        # TASK-34100.1 (cross-cutting-14): a slow Next names its work here.
        yield SetupBusyStatus(id="setup-busy-status", classes="setup-busy-status")
        # TASK-21140 (UAT W/G findings): the one step-error surface, pinned
        # between the scrollable step body and the nav bar so a refused Next
        # is always explained on screen. Steps' show_step_error() renders
        # here; the old per-step tail Statics sat below the fold of every
        # overflowing step and made commit failures invisible.
        yield Static(
            "",
            id="setup-step-error-pinned",
            classes="setup-step-error hidden",
            markup=False,
        )
        yield SetupWizardNavigation(classes="wizard-navigation setup-navigation")

    def _post_mount_hook(self) -> None:
        """Refresh the initial active track after all steps have composed.

        Failure-policy follow-up: ``self.active_ids`` is first computed in
        ``__init__``, before any step has actually composed -- a step's
        ``compose_failed`` flag can only be known once its own compose()
        has actually run, which Textual does while mounting this
        container's children, i.e. by the time ``WizardContainer.on_mount``
        calls this hook. ``_refresh_active_ids()`` re-derives the projection
        against the now-accurate ``compose_failed`` flags, so optional failures
        leave progress/navigation while required failures remain represented.

        TASK-2710: overrides ``WizardContainer._post_mount_hook`` instead of
        defining its own ``on_mount()`` that calls ``super().on_mount()`` --
        Textual's dispatcher already invokes ``WizardContainer.on_mount``
        separately for this Mount event, so the old ``super().on_mount()``
        call ran ``show_step(0)`` (and the duplicate validation timer) a
        second time, sandwiched around this method's own work. The hook
        preserves the intended ordering (this logic runs once, strictly
        after the base's initialization) without the duplicate execution.
        """
        self._refresh_active_ids()
        self.update_progress()
        self._sync_exit_controls()
        self._restore_resume_target()

    def _sync_exit_controls(self) -> None:
        """Keep global navigation distinct from Summary destination actions."""

        try:
            step = self.steps[self.current_step]
            step_id = step.config.id if step.config is not None else ""
            back = self.query_one("#wizard-back", Button)
            next_button = self.query_one("#wizard-next", Button)
            cancel = self.query_one("#wizard-cancel", Button)
            hints = self.screen.query_one("#setup-key-hints", Static)
        except (IndexError, NoMatches):
            return

        on_summary = step_id == wizard_state.STEP_SUMMARY
        back.display = True
        next_button.display = not on_summary
        cancel.display = not on_summary
        cancel.variant = "default"
        if on_summary:
            hints.update(SUMMARY_KEY_HINTS)
        elif step_id == wizard_state.STEP_WELCOME:
            cancel.label = "Skip setup"
            cancel.tooltip = (
                "Close setup and stop showing it at launch. You can rerun it "
                "from Settings ▸ Diagnostics."
            )
            hints.update("Enter / Ctrl+N next · Ctrl+B back · Esc skip setup")
        else:
            cancel.label = "Exit setup"
            cancel.tooltip = (
                "Save completed steps and continue later from Settings ▸ Diagnostics."
            )
            hints.update(
                # TASK-34100.8: a step whose Enter differs says so (Voice).
                getattr(step, "KEY_HINTS", "")
                or "Enter / Ctrl+N next · Ctrl+B back · Esc exit setup"
            )

    def _restore_resume_target(self) -> None:
        """Show a validated resume target and clear its marker after paint."""

        draft = self.resume_draft
        if draft is None or draft.active_step_id not in self.active_ids:
            return
        if not self._restore_resume_controls(draft):
            return
        target_index = self._step_index_for_id(draft.active_step_id)
        if target_index is None:
            return
        target = self.steps[target_index]
        if getattr(target, "compose_failed", False):
            return
        try:
            self.show_step(target_index)
        except Exception:
            logger.warning("Setup resume target mount failed (category=mount)")
            return
        current = self.steps[self.current_step]
        if (
            current.config is None
            or current.config.id != draft.active_step_id
            or not current.is_attached
        ):
            return
        screen = self.screen
        if isinstance(screen, FirstRunSetupWizard):
            screen.call_after_refresh(
                screen._clear_resume_attempt_after_target_mount,
                self,
                target,
                draft.active_step_id,
            )

    def _restore_resume_controls(self, draft: wizard_state.SetupDraft) -> bool:
        """Apply allowlisted checkpoint values to mounted step controls/state."""

        try:
            track_choice = self.query_one("#setup-track-choice", RadioSet)
            self._restore_radio_selection(
                track_choice,
                lambda button: (
                    button.id
                    == (
                        "setup-track-full"
                        if draft.track == wizard_state.TRACK_FULL
                        else "setup-track-quick"
                    )
                ),
            )

            provider_values = draft.values.get(wizard_state.STEP_PROVIDER, {})
            provider_step = self.steps[
                self._step_index_for_id(wizard_state.STEP_PROVIDER)
            ]
            if isinstance(provider_step, ProviderStep):
                if "provider_key" in provider_values:
                    provider_key = str(provider_values["provider_key"])
                    provider_step.selected_provider_key = provider_key
                    choices = provider_step.query_one(
                        "#setup-provider-choice", OptionList
                    )
                    for index in range(choices.option_count):
                        option = choices.get_option_at_index(index)
                        if getattr(option, "provider_key", None) == provider_key:
                            choices.highlighted = index
                            break
                if "provider_value" in provider_values:
                    provider_step.provider_value_for_chat_defaults = str(
                        provider_values["provider_value"]
                    )

            model_values = draft.values.get(wizard_state.STEP_MODEL, {})
            model_step = self.steps[self._step_index_for_id(wizard_state.STEP_MODEL)]
            if isinstance(model_step, ModelStep) and "model_id" in model_values:
                model_id = str(model_values["model_id"])
                model_step.selected_model_id = model_id
                model_step._model_id_from_custom_input = bool(model_id)
                model_step.query_one("#setup-model-custom", Input).value = model_id

            voice_values = draft.values.get(wizard_state.STEP_VOICE, {})
            voice_step = self.steps[self._step_index_for_id(wizard_state.STEP_VOICE)]
            from tldw_chatbook.UI.Wizards.first_run_voice_step import VoiceSetupStep

            if isinstance(voice_step, VoiceSetupStep) and voice_values:
                voice_step.restore_checkpoint(voice_values)

            rag_values = draft.values.get(wizard_state.STEP_RAG, {})
            rag_step = self.steps[self._step_index_for_id(wizard_state.STEP_RAG)]
            if isinstance(rag_step, RagStep) and "embedding_model" in rag_values:
                embedding_model = str(rag_values["embedding_model"])
                rag_step.selected_embedding_model = embedding_model
                self._restore_radio_selection(
                    rag_step.query_one("#setup-rag-model-choice", RadioSet),
                    lambda button: _radio_model_id(button) == embedding_model,
                )

            appearance_values = draft.values.get(wizard_state.STEP_APPEARANCE, {})
            appearance_step = self.steps[
                self._step_index_for_id(wizard_state.STEP_APPEARANCE)
            ]
            if isinstance(appearance_step, AppearanceStep):
                if "theme" in appearance_values:
                    theme = str(appearance_values["theme"])
                    appearance_step.selected_theme = theme
                    self._restore_radio_selection(
                        appearance_step.query_one("#setup-theme-choice", RadioSet),
                        lambda button: getattr(button, "_theme_name", "") == theme,
                    )
                if "splash_card" in appearance_values:
                    splash_card = str(appearance_values["splash_card"])
                    appearance_step.selected_splash_card = splash_card
                    appearance_step._picked_surprise_me = False
                    self._restore_radio_selection(
                        appearance_step.query_one("#setup-splash-choice", RadioSet),
                        # Qodo review fix: labels are humanized (TASK-21149);
                        # match on the raw-id rider, mirroring _theme_name.
                        lambda button: (
                            getattr(button, "_card_name", "") == splash_card
                            if splash_card
                            else str(button.label).startswith("Surprise me")
                        ),
                    )

            protect_values = draft.values.get(wizard_state.STEP_PROTECT, {})
            protect_step = self.steps[
                self._step_index_for_id(wizard_state.STEP_PROTECT)
            ]
            if (
                isinstance(protect_step, ProtectKeysStep)
                and "encryption_enabled" in protect_values
            ):
                encryption_enabled = protect_values["encryption_enabled"]
                protect_step.encryption_enabled = encryption_enabled
                if encryption_enabled:
                    protect_step.query_one("#setup-protect-status", Static).update(
                        "Encryption enabled."
                    )
        except Exception:
            logger.warning("Setup resume control restore failed (category=runtime)")
            return False
        return True

    @staticmethod
    def _restore_radio_selection(
        radio_set: RadioSet,
        matches: Callable[[RadioButton], bool],
    ) -> None:
        """Restore one RadioSet selection without emitting user-change events."""

        buttons = list(radio_set.query(RadioButton))
        selected = next((button for button in buttons if matches(button)), None)
        with radio_set.prevent(RadioButton.Changed):
            for button in buttons:
                button.value = button is selected
        radio_set._pressed_button = selected
        radio_set._selected = buttons.index(selected) if selected is not None else None

    # -- step construction -------------------------------------------------
    def _build_step(self, config: WizardStepConfig) -> SetupStep:
        """Construct one setup step from the canonical config-backed factory."""

        from tldw_chatbook.UI.Wizards.first_run_voice_step import VoiceSetupStep

        step_types: dict[str, type[SetupStep]] = {
            wizard_state.STEP_WELCOME: WelcomeStep,
            wizard_state.STEP_PROVIDER: ProviderStep,
            wizard_state.STEP_MODEL: ModelStep,
            wizard_state.STEP_VOICE: VoiceSetupStep,
            wizard_state.STEP_RAG: RagStep,
            wizard_state.STEP_SPEECH: SpeechSetupStep,
            wizard_state.STEP_TOOLS: ToolsStep,
            wizard_state.STEP_NOTES: NotesSyncStep,
            wizard_state.STEP_APPEARANCE: AppearanceStep,
            wizard_state.STEP_PROTECT: ProtectKeysStep,
            wizard_state.STEP_SUMMARY: SummaryStep,
        }
        step_type = step_types[config.id]
        if step_type is ProviderStep:
            return ProviderStep(wizard=self, config=config, environ=os.environ)
        return step_type(wizard=self, config=config)

    def _create_steps(self) -> List[WizardStep]:
        # Later tasks append real steps here; the skeleton ships Welcome +
        # placeholder SetupSteps so navigation is testable end to end.
        def cfg(
            step_id: str,
            title: str,
            number: int,
            *,
            required: bool = True,
        ) -> WizardStepConfig:
            return WizardStepConfig(
                id=step_id,
                title=title,
                step_number=number,
                can_skip=not required,
            )

        titles = wizard_state.STEP_TITLES
        configs = (
            cfg(wizard_state.STEP_WELCOME, titles[wizard_state.STEP_WELCOME], 1),
            cfg(wizard_state.STEP_PROVIDER, titles[wizard_state.STEP_PROVIDER], 2),
            cfg(wizard_state.STEP_MODEL, titles[wizard_state.STEP_MODEL], 3),
            cfg(wizard_state.STEP_VOICE, titles[wizard_state.STEP_VOICE], 4),
            cfg(
                wizard_state.STEP_RAG,
                titles[wizard_state.STEP_RAG],
                5,
                required=False,
            ),
            cfg(wizard_state.STEP_SPEECH, titles[wizard_state.STEP_SPEECH], 6),
            cfg(wizard_state.STEP_TOOLS, titles[wizard_state.STEP_TOOLS], 7),
            cfg(wizard_state.STEP_NOTES, titles[wizard_state.STEP_NOTES], 8),
            cfg(
                wizard_state.STEP_APPEARANCE,
                titles[wizard_state.STEP_APPEARANCE],
                9,
            ),
            cfg(wizard_state.STEP_PROTECT, titles[wizard_state.STEP_PROTECT], 10),
            cfg(wizard_state.STEP_SUMMARY, titles[wizard_state.STEP_SUMMARY], 11),
        )
        return [self._build_step(config) for config in configs]

    # -- active-step navigation --------------------------------------------
    def select_track(self, track: str) -> None:
        """Recompute the active subset after the Welcome choice."""
        self.track = track
        self._refresh_active_ids()

    def note_key_entered(self) -> None:
        if not self.key_entered:
            self.key_entered = True
            self._refresh_active_ids()

    def _effective_key_entered(self) -> bool:
        """Bug-4 fix: config-derived fallback for the Protect-keys gate.

        ``self.key_entered`` only flips true when a secret is TYPED this
        run, so a rerun over a config that already has a plaintext key on
        disk (hand-edited config.toml, or a prior completed run) could
        never reach Protect Keys without retyping a credential -- even
        though ``check_encryption_needed``'s own intent is config-derived.
        """
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        return self.key_entered or wizard_state.stored_plaintext_key_present(app_config)

    def _refresh_active_ids(self) -> None:
        ids = wizard_state.active_step_ids(
            self.track, key_entered=self._effective_key_entered()
        )
        optional_failures = {
            step.config.id: step.compose_failure.reason_code
            for step in self.steps
            if (
                isinstance(step, SetupStep)
                and step.config
                and step.compose_failure is not None
                and not step.compose_failure.required
            )
        }
        self.skipped_step_reasons = optional_failures
        self.active_ids = tuple(sid for sid in ids if sid not in optional_failures)
        self._rebuild_progress()
        # TASK-2154.9 (FR-02): keep the "Step X of Y" text in sync with the
        # rebuilt dots -- note_key_entered() reaches here while the user is
        # still on the Provider step, and without this the text total lagged
        # one navigation behind the conditional protect-keys step joining.
        self.update_progress()
        # Finding B: a step's compose_failed flag can only be known once its
        # own compose() has actually run -- which may land after this
        # container already displayed it (WelcomeStep is index 0 and
        # BaseWizard.on_mount unconditionally shows it first). If the page
        # Redirect only optional failures. Required failures intentionally stay
        # visible on their own recovery surface.
        if (
            0 <= self.current_step < len(self.steps)
            and isinstance(self.steps[self.current_step], SetupStep)
            and self.steps[self.current_step].compose_failure is not None
            and not self.steps[self.current_step].compose_failure.required
        ):
            resolved = self._resolve_visible_index(self.current_step)
            if resolved != self.current_step:
                self.show_step(resolved)

    def compose_failed_steps(self) -> list[str]:
        """Titles of optional steps dropped by the compose-crash policy.

        Returns:
            Display titles of steps whose composition failed this session.
        """
        return [
            step.config.title
            for step in self.steps
            if step.config and step.config.id in self.skipped_step_reasons
        ]

    def _step_index_for_id(self, step_id: str) -> Optional[int]:
        for index, step in enumerate(self.steps):
            if step.config and step.config.id == step_id:
                return index
        return None

    def _active_position(self, absolute_index: int) -> int:
        step = self.steps[absolute_index]
        step_id = step.config.id if step.config else ""
        return self.active_ids.index(step_id) if step_id in self.active_ids else 0

    def _next_active_index(self, absolute_index: int) -> Optional[int]:
        position = self._active_position(absolute_index)
        if position + 1 >= len(self.active_ids):
            return None
        return self._step_index_for_id(self.active_ids[position + 1])

    def _previous_active_index(self, absolute_index: int) -> Optional[int]:
        position = self._active_position(absolute_index)
        if position <= 0:
            return None
        return self._step_index_for_id(self.active_ids[position - 1])

    def _resolve_visible_index(self, step_index: int) -> int:
        """Redirect optional failed steps while retaining required failures.

        ``_refresh_active_ids()`` already drops a compose-failed step from
        navigation/progress, but nothing stopped the container from still
        SHOWING it as the current page -- WelcomeStep sits at absolute
        index 0, and BaseWizard.on_mount (never modified) unconditionally
        calls ``show_step(0)`` on first mount, before this container has
        had a chance to refresh ``active_ids``. A step's own
        ``compose_failed`` flag is already final by the time ANY
        ``show_step`` call happens (Textual composes the whole step
        subtree before this container's on_mount fires at all), so
        re-derive the active set fresh here -- rather than trusting
        ``self.active_ids``, which may still be the pre-refresh value on
        this very first call -- and redirect to its first non-failed
        member instead of trusting the caller's index.

        Args:
            step_index: The absolute step index the caller wants to show.

        Returns:
            ``step_index`` for successful or required-failed steps; otherwise
            the first active index that can be shown.
        """
        if not (0 <= step_index < len(self.steps)):
            return step_index
        failed_step = self.steps[step_index]
        if not getattr(failed_step, "compose_failed", False):
            return step_index
        if isinstance(failed_step, SetupStep) and failed_step.required:
            return step_index
        ids = wizard_state.active_step_ids(
            self.track, key_entered=self._effective_key_entered()
        )
        for step_id in ids:
            index = self._step_index_for_id(step_id)
            if index is None:
                continue
            candidate = self.steps[index]
            if not getattr(candidate, "compose_failed", False) or (
                isinstance(candidate, SetupStep) and candidate.required
            ):
                return index
        return step_index

    def show_step(self, step_index: int) -> None:
        """F-B root cause fix: BaseWizard.show_step() (never modified --
        this overrides it in the subclass, same pattern as update_progress/
        handle_next/handle_back/action_next/action_back below) hides the
        OUTGOING step via ``current.add_class("hidden")``, which sets
        ``display: none`` on it. Textual clears focus to None once the
        widget that held it is no longer displayed -- confirmed live via
        diagnostic instrumentation across a real tmux session: a user whose
        last interaction was with a control INSIDE a step's own content (a
        RadioButton, an Input -- not the persistent WizardNavigation bar,
        which is never hidden) loses ALL focus the instant that step is
        hidden. With ``app.focused`` None, ctrl+n/ctrl+b (bound on THIS
        container, several ancestors up from wherever the user last
        interacted) have no focus chain left to resolve bindings through
        and go silently inert -- the wizard "stays open" with no error or
        indication anything happened.

        Round-2 regression fix: the first cut of this fix always refocused
        the persistent nav bar's Next/Cancel button. That broke direct
        keyboard interaction with the NEW step's own content -- landing on
        Provider with focus already parked on "Next" meant Down/Space (which
        only act on a FOCUSED RadioSet) silently did nothing, and a user who
        never thinks to Tab away from the nav bar gets the exact "selection
        doesn't commit" symptom F-A already fixed at the commit layer, one
        level up in the UI. Prefer the incoming step's own first focusable
        descendant (DOM order, matching compose()'s visual top-to-bottom
        order -- e.g. the RadioSet on Provider/Model, the first exit Button
        on Summary) so arrow/space/typing keep working with no Tab-hunting
        required; fall back to the nav bar only when the step truly has no
        focusable widget of its own. Either way the container remains in
        the focused widget's ancestry, so ctrl+n/ctrl+b still resolve.
        """
        if self._failure_action_running and step_index != self.current_step:
            return
        step_index = self._resolve_visible_index(step_index)
        super().show_step(step_index)
        self._clear_pinned_step_error()
        # TASK-21143 (UAT P-5): the step that owns the fix must show the
        # failure — returning to Provider after a failed probe explains
        # what went wrong right where the key/endpoint is edited.
        try:
            shown = self.steps[self.current_step]
        except IndexError:
            shown = None
        if isinstance(shown, ProviderStep):
            failure = self.provider_probe_failure()
            if failure == wizard_state.PROVIDER_PROBE_AUTH:
                shown.show_step_error(
                    "The last connection check failed: this API key was "
                    "rejected. Update it, then continue."
                )
            elif failure == wizard_state.PROVIDER_PROBE_CONNECTION:
                shown.show_step_error(
                    "The last connection check couldn't reach the server. "
                    "Check it's running, then continue."
                )
        self._sync_exit_controls()
        try:
            current_step = self.steps[self.current_step]
            # TASK-1496/1498: "focusable" alone is not enough — a widget
            # hidden via display:none (e.g. the pinned "Use this server"
            # button before discovery finds anything) must never be the
            # focus target, or keyboard input lands on an invisible control.
            target = None
            if isinstance(current_step, SetupStep):
                preferred = current_step.preferred_focus()
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
                        for widget in current_step.walk_children(Widget)
                        if widget.focusable
                        and widget.display
                        and not widget.has_class("hidden")
                    ),
                    None,
                )
            if target is None:
                next_button = self.query_one("#wizard-next", Button)
                target = (
                    next_button
                    if not next_button.disabled
                    else self.query_one("#wizard-cancel", Button)
                )
            target.focus()
        except Exception:
            logger.debug("Wizard step-change focus fix skipped", exc_info=True)

    def update_progress(self) -> None:
        """Recount against the ACTIVE subset, not the full step list."""
        try:
            position = self._active_position(self.current_step or 0)
            nav = self.query_one(".wizard-navigation", WizardNavigation)
            nav.total_steps = len(self.active_ids)
            nav.can_go_back = position > 0  # before current_step: its watcher reads it
            nav.current_step = position + 1
            nav.can_go_forward = self.can_proceed
            self._rebuild_progress()
        except Exception:
            pass

    def _rebuild_progress(self) -> None:
        """Refresh the setup-specific tracker from the active-track projection."""
        try:
            # TASK-21143 (UAT N-7): a visited Provider/Model pair whose
            # probe failed shows "!" instead of the ✓ users read as "OK".
            # TASK-25818 widens that to the step the user simply walked
            # through without configuring: the summary already reports it as
            # unconfigured, and the tracker must not disagree.
            attention = wizard_state.setup_attention_ids(
                self.wizard_data,
                probe_failed=bool(self.provider_probe_failure()),
            )
            items = wizard_state.build_setup_progress(
                self.active_ids,
                self._active_position(self.current_step or 0),
                attention_ids=attention,
            )
            self.query_one(".wizard-progress", SetupWizardProgress).set_items(items)
        except Exception:
            logger.debug("Wizard progress rebuild skipped", exc_info=True)

    # -- required-step failure recovery -----------------------------------
    @on(Button.Pressed, "#setup-step-retry")
    def handle_step_retry(self) -> None:
        action = self._begin_failure_action()
        if action is None:
            return
        try:
            run_wizard_worker(
                self,
                self._retry_failed_step(action),
                exclusive=True,
                group="setup-step-recovery",
            )
        except Exception:
            self._release_failure_action(action)
            raise

    @on(Button.Pressed, "#setup-step-manual")
    def handle_step_manual(self) -> None:
        action = self._begin_failure_action()
        if action is None:
            return
        try:
            run_wizard_worker(
                self,
                self._use_manual_setup(action),
                exclusive=True,
                group="setup-step-recovery",
            )
        except Exception:
            self._release_failure_action(action)
            raise

    @on(Button.Pressed, "#setup-step-later")
    def handle_step_later(self) -> None:
        action = self._begin_failure_action()
        if action is None:
            return
        try:
            run_wizard_worker(
                self,
                self._finish_later_from_failure(action),
                exclusive=True,
                group="setup-step-recovery",
            )
        except Exception:
            self._release_failure_action(action)
            raise

    def _active_required_failure(self) -> SetupStep | None:
        try:
            step = self.steps[self.current_step]
        except IndexError:
            return None
        if (
            isinstance(step, SetupStep)
            and step.compose_failure is not None
            and step.required
        ):
            return step
        return None

    def _begin_failure_action(self) -> _SetupFailureAction | None:
        if self._failure_action_running or self._advancing or self._finalized:
            return None
        step = self._active_required_failure()
        if (
            step is None
            or step.config is None
            or step.compose_failure is None
            or not self.is_attached
            or not step.is_attached
        ):
            return None
        try:
            screen = self.screen
        except Exception:
            return None
        if not isinstance(screen, FirstRunSetupWizard):
            return None
        action = _SetupFailureAction(
            screen=screen,
            index=self.current_step,
            step=step,
            step_id=step.config.id,
            failure=step.compose_failure,
        )
        self._failure_action = action
        self._failure_action_running = True
        self._sync_action_controls()
        return action

    def _failure_action_is_current(
        self,
        action: _SetupFailureAction,
        *,
        require_step_mounted: bool = True,
    ) -> bool:
        try:
            same_screen = (
                self.screen is action.screen
                and action.screen.app.screen is action.screen
                and action.screen.query_one(SetupWizardContainer) is self
            )
            same_step = (
                self.current_step == action.index
                and self.steps[action.index] is action.step
                and action.step.config is not None
                and action.step.config.id == action.step_id
                and action.step.compose_failure is action.failure
            )
        except Exception:
            return False
        return (
            self._failure_action_running
            and self._failure_action is action
            and not self._finalized
            and self.is_attached
            and action.screen.is_attached
            and same_screen
            and same_step
            and (not require_step_mounted or action.step.is_attached)
        )

    def _retry_replacement_is_current(
        self,
        action: _SetupFailureAction,
        replacement: SetupStep,
    ) -> bool:
        try:
            return (
                self._failure_action_running
                and self._failure_action is action
                and not self._finalized
                and self.is_attached
                and action.screen.is_attached
                and self.screen is action.screen
                and action.screen.app.screen is action.screen
                and action.screen.query_one(SetupWizardContainer) is self
                and self.current_step == action.index
                and self.steps[action.index] is replacement
                and replacement.is_attached
                and replacement.config is not None
                and replacement.config.id == action.step_id
            )
        except Exception:
            return False

    def _release_failure_action(
        self,
        action: _SetupFailureAction,
    ) -> None:
        if self._failure_action is not action:
            return
        self._failure_action = None
        self._failure_action_running = False
        try:
            current = self.steps[self.current_step]
            same_screen = (
                self.is_attached
                and action.screen.is_attached
                and self.screen is action.screen
                and action.screen.app.screen is action.screen
                and action.screen.query_one(SetupWizardContainer) is self
            )
        except Exception:
            return
        if same_screen and current.is_attached:
            self._sync_action_controls()

    def _sync_action_controls(self) -> None:
        blocked = (
            self._advancing
            or self._failure_action_running
            or self._provider_dismiss_pending
        )
        try:
            if blocked:  # disabling blurs: hold focus for when the fence lifts
                step_guard.hold_fenced_focus(self)
                for selector in ("#wizard-back", "#wizard-next", "#wizard-cancel"):
                    self.query_one(selector, Button).disabled = True
            else:
                self.update_progress()
                nav = self.query_one(".wizard-navigation", WizardNavigation)
                nav.update_button_states()
                self.query_one("#wizard-cancel", Button).disabled = False
        except NoMatches:
            pass

        failure = self._active_required_failure()
        for selector in (
            "#setup-step-retry",
            "#setup-step-manual",
            "#setup-step-later",
        ):
            try:
                button = self.query_one(selector, Button)
                if blocked:
                    button.disabled = True
                elif selector == "#setup-step-manual" and failure is not None:
                    button.disabled = (
                        failure.config is None
                        or manual_settings_context_for_required_step(failure.config.id)
                        is None
                    )
                else:
                    button.disabled = False
            except NoMatches:
                pass
        self._sync_exit_controls()
        if not blocked:
            step_guard.restore_fenced_focus(self)

    async def _checkpoint_required_failure(
        self, action: _SetupFailureAction
    ) -> bool | None:
        if not self._failure_action_is_current(action):
            return None
        try:
            saved = await self.persist_current_checkpoint()
        except Exception as exc:
            logger.error(
                "Wizard failure checkpoint failed "
                "(category=persistence, error_type={})",
                type(exc).__name__,
            )
            saved = False
        if not self._failure_action_is_current(action):
            return None
        if not saved:
            action.step.show_step_error(
                "Setup progress could not be saved. Retry this action."
            )
        return saved

    async def _use_manual_setup(self, action: _SetupFailureAction) -> None:
        try:
            context = manual_settings_context_for_required_step(action.step_id)
            if context is None:
                if self._failure_action_is_current(action):
                    action.step.show_step_error(
                        "Manual setup is unavailable for this step. "
                        "Retry or exit setup and return later."
                    )
                return
            if await self._checkpoint_required_failure(action) is not True:
                return
            result = {
                "completed": False,
                "exit_route": "settings",
                "exit_context": context,
            }
            self._dismiss_screen(result)
        finally:
            self._release_failure_action(action)

    async def _finish_later_from_failure(self, action: _SetupFailureAction) -> None:
        try:
            if await self._checkpoint_required_failure(action) is not True:
                return
            self._dismiss_screen(None)
        finally:
            self._release_failure_action(action)

    async def _retry_failed_step(self, action: _SetupFailureAction) -> None:
        """Replace only the active failed step with a clean factory instance."""

        replacement: SetupStep | None = None
        parent: Widget | None = None
        next_sibling: Widget | None = None
        try:
            if not self._failure_action_is_current(action):
                return
            index = action.index
            failed_step = action.step
            parent = failed_step.parent
            if parent is None:
                return
            siblings = list(parent.children)
            sibling_index = siblings.index(failed_step)
            next_sibling = (
                siblings[sibling_index + 1]
                if sibling_index + 1 < len(siblings)
                else None
            )
            replacement = self._build_step(failed_step.config)
            replacement.step_number = failed_step.step_number
            replacement.add_class("hidden")

            await failed_step.remove()
            if not self._failure_action_is_current(action, require_step_mounted=False):
                return
            self.steps[index] = replacement
            if next_sibling is None:
                await parent.mount(replacement)
            else:
                await parent.mount(replacement, before=next_sibling)
            if not self._retry_replacement_is_current(action, replacement):
                return
            self._refresh_active_ids()
            self.show_step(index)
        except Exception as exc:
            logger.error(
                "Wizard step retry failed (category=recovery, error_type={})",
                type(exc).__name__,
            )
            if parent is not None:
                await self._rollback_failed_step_retry(
                    action,
                    replacement,
                    parent,
                    next_sibling,
                )
        finally:
            self._release_failure_action(action)

    async def _rollback_failed_step_retry(
        self,
        action: _SetupFailureAction,
        replacement: SetupStep | None,
        parent: Widget,
        next_sibling: Widget | None,
    ) -> None:
        """Restore one coherent failed step after a partial retry replacement."""

        try:
            same_screen = (
                self._failure_action is action
                and not self._finalized
                and self.is_attached
                and action.screen.is_attached
                and self.screen is action.screen
                and action.screen.app.screen is action.screen
                and action.screen.query_one(SetupWizardContainer) is self
                and self.current_step == action.index
            )
        except Exception:
            return
        if not same_screen:
            return

        recovery_step: SetupStep | None = None
        try:
            if replacement is not None and replacement.parent is not None:
                await replacement.remove()
            recovery_step = self._build_step(action.step.config)
            recovery_step.step_number = action.step.step_number
            recovery_step.compose_failed = True
            recovery_step.compose_failure = action.failure
            recovery_step.add_class("hidden")
            mount_before = (
                next_sibling
                if next_sibling is not None and next_sibling.parent is parent
                else None
            )
            if mount_before is None:
                await parent.mount(recovery_step)
            else:
                await parent.mount(recovery_step, before=mount_before)
            self.steps[action.index] = recovery_step
            self._refresh_active_ids()
            self.show_step(action.index)
        except Exception as exc:
            if recovery_step is not None and recovery_step.parent is parent:
                self.steps[action.index] = recovery_step
            logger.error(
                "Wizard step retry rollback failed (category=recovery, error_type={})",
                type(exc).__name__,
            )

    # -- commit-on-Next ----------------------------------------------------
    @on(Button.Pressed, "#wizard-next")
    def handle_next(self, event: Button.Pressed) -> None:
        # Textual's @on dispatch walks the WHOLE MRO and invokes every
        # matching decorated handler on every class, not just the closest
        # override (see textual.message_pump.MessagePump._get_dispatch_methods).
        # Without prevent_default(), WizardContainer.handle_next() ALSO fires
        # on this same click, flat-advancing current_step by one (ignoring
        # the active-id subset) before our worker even starts — silently
        # breaking track branching and double-firing on_complete on the last
        # step. prevent_default() is the documented way to suppress handlers
        # in base classes for this exact message.
        event.prevent_default()
        self.advance_programmatically()

    def advance_programmatically(self) -> None:
        """Same commit-and-advance path as clicking Next, without an event.

        SummaryStep's three destination buttons are not the "#wizard-next"
        button, so they have no Button.Pressed event to hand to handle_next()
        above -- which requires one to call event.prevent_default() (see that
        method's docstring for why). This is the extracted guard + worker
        dispatch body shared by both callers; the real Next button's dispatch
        semantics (the prevent_default() suppression) are unchanged.

        TASK-21143 (UAT M-2): a step that knows its state is broken can gate
        the advance behind an explicit confirmation (confirm_before_advance).
        The one-shot ``_advance_confirmed`` flag lets the dialog's "Continue
        anyway" re-enter this method exactly once without re-asking.
        """
        # Consume the one-shot confirmation UP FRONT (review TASK-21143
        # follow-up): if the dialog's resolution re-enters while an early
        # guard trips, a surviving flag would let the NEXT press bypass the
        # gate silently. Losing a confirmation to a blocked re-entry only
        # re-asks — the safe direction for a trust gate.
        confirmed = self._advance_confirmed
        self._advance_confirmed = False
        if self._advancing or self._failure_action_running or not self.can_proceed:
            return
        if not confirmed:
            try:
                step = self.steps[self.current_step]
            except IndexError:
                step = None
            if isinstance(step, SetupStep):
                prompt = step.confirm_before_advance()
                if prompt:
                    self._push_advance_confirmation(prompt)
                    return
        try:
            label = busy_label_for(self.steps[self.current_step])
        except IndexError:
            label = ""
        self._set_advancing(True, label)
        run_wizard_worker(
            self, self._advance(), exclusive=True, group="setup-wizard-advance"
        )

    def _push_advance_confirmation(self, prompt: str) -> None:
        """Ask before committing a step that reports itself broken."""

        dialog = _SettlingGuardedConfirmationDialog(
            title="Continue anyway?",
            message=prompt,
            confirm_label="Continue anyway",
            cancel_label="Keep editing",
        )

        def _resolve(confirmed: bool | None) -> None:
            if confirmed:
                self._advance_confirmed = True
                self.advance_programmatically()

        self.app.push_screen(dialog, _resolve)

    def request_model_catalog_refresh(self) -> None:
        """Ask the app to refresh provider model catalogs now (TASK-21150).

        Mirrors ``_handle_model_catalog_consent``'s allow path so a Summary
        "yes" and a Console-modal "yes" have the same effect. Silent when
        the app exposes no refresher (bare test hosts).
        """
        app_instance = getattr(self, "app_instance", None)
        refresh = getattr(app_instance, "refresh_model_catalogs_now", None)
        if callable(refresh):
            refresh()

    def provider_probe_failure(self) -> str:
        """The Model step's classified probe failure for the live identity.

        TASK-21143: single source for the trust chain — the tracker's "!"
        state, the Provider step's returned-to notice, and the Summary's
        row/action override all read this. Returns "" whenever the Model
        step can't vouch for a CURRENT failure (never probed, superseded
        identity, step missing).
        """
        index = self._step_index_for_id(wizard_state.STEP_MODEL)
        if index is None:
            return ""
        step = self.steps[index]
        if not isinstance(step, ModelStep):
            return ""
        try:
            return step.current_probe_failure()
        except Exception:
            return ""

    def _clear_pinned_step_error(self) -> None:
        """Empty and hide the pinned error strip (on every step change)."""

        try:
            strip = self.query_one("#setup-step-error-pinned", Static)
        except NoMatches:
            return
        strip.update("")
        strip.add_class("hidden")

    def _set_advancing(self, active: bool, label: str = "") -> None:
        """Fence navigation while a step's config handoff settles, and run the
        busy line, which names ``label`` after about 400 ms (TASK-34100.1)."""

        self._advancing = active
        self._sync_action_controls()
        try:
            busy = self.query_one("#setup-busy-status", SetupBusyStatus)
        except NoMatches:
            return
        if active:
            busy.start(label)
        else:
            busy.stop()

    async def _advance(self) -> None:
        started_at = self.current_step  # TASK-33621.14: did a failed Next move?
        try:
            step = self.steps[self.current_step]
            if isinstance(step, SetupStep):
                if step.compose_failure is not None and step.required:
                    return
                ok, error = await step.commit()
                if not ok:
                    # TASK-21140 (UAT F-1 follow-on): the old suffix offered
                    # "Skip this step", a control that does not exist. Name
                    # only affordances that are on screen.
                    step.show_step_error(f"{error}  Retry with Next, or go Back.")
                    return
            if isinstance(step, WelcomeStep):
                self.select_track(step.chosen_track())
            step_id = step.config.id if step.config else f"step_{self.current_step}"
            self.wizard_data[step_id] = step.get_step_data()
            step.is_complete = True
            next_index = self._next_active_index(self.current_step)
            if step_id != wizard_state.STEP_SUMMARY:
                next_step_id = (
                    self.steps[next_index].config.id
                    if next_index is not None
                    and self.steps[next_index].config is not None
                    else step_id
                )
                if not await self.persist_setup_checkpoint(next_step_id):
                    step.show_step_error(
                        "Saving setup progress failed. Retry before continuing."
                    )
                    return
            if next_index is None:
                self.complete_wizard()
            else:  # review round 2: a nearly due busy line paints before the mount
                for busy in self.query(SetupBusyStatus):
                    await busy.reveal_before_step_change()
                self.show_step(next_index)
        except Exception as error:  # TASK-33621.14: Next must not exit the app.
            step_guard.report_advance_error(self, error, started_at)
        finally:
            if not self._finishing:
                self._set_advancing(False)

    @on(Button.Pressed, "#wizard-back")
    def handle_back(self, event: Button.Pressed) -> None:
        # Same base-class double-dispatch as handle_next; see the comment
        # there. WizardContainer.handle_back() would otherwise also fire and
        # flat-decrement current_step, ignoring the active-id subset.
        event.prevent_default()
        if self._advancing or self._failure_action_running:
            return
        previous = self._previous_active_index(self.current_step)
        if previous is not None:
            self.show_step(previous)

    # -- keyboard shortcuts (BINDINGS ctrl+n / ctrl+b are inherited from
    # BaseWizard, which this module's own docstring above documents as
    # never modified -- these actions are overridden here instead) --------
    def action_next(self) -> None:
        """ctrl+n: same guarded, commit-and-advance path as clicking Next.

        BaseWizard.action_next() calls self.handle_next() with NO
        arguments, but this class's handle_next() override above requires a
        Button.Pressed event (to call event.prevent_default() -- see its
        docstring). Left un-overridden, pressing ctrl+n on a mounted
        SetupWizardContainer raises TypeError. advance_programmatically() is
        the same event-free body handle_next() and SummaryStep's exit
        buttons already share; routing the action there keeps active-id
        navigation, per-step commit, and the on-Welcome track selection
        (self.select_track(...) inside _advance()) all working from the
        keyboard exactly as they do from the mouse.
        """
        try:
            current = self.steps[self.current_step]
            step_id = current.config.id if current.config is not None else ""
        except IndexError:
            return
        if step_id == wizard_state.STEP_SUMMARY:
            try:
                self.query_one("#setup-exit-chat", Button).focus()
            except NoMatches:
                pass
            return
        self.advance_programmatically()

    def action_back(self) -> None:
        """ctrl+b: same active-subset Back navigation as clicking Back.

        BaseWizard.action_back() calls self.handle_back() with NO
        arguments, which likewise crashes against this class's
        handle_back(event) override. This mirrors that override's body
        exactly, minus the event.prevent_default() call action dispatch has
        no event for.
        """
        if self._advancing or self._failure_action_running:
            return
        previous = self._previous_active_index(self.current_step)
        if previous is not None:
            self.show_step(previous)

    def review_provider_setup(self) -> None:
        """Return an incomplete Summary to the provider step without mutation."""

        provider_index = self._step_index_for_id(wizard_state.STEP_PROVIDER)
        if provider_index is not None:
            self.show_step(provider_index)

    def open_provider_settings(self) -> None:
        """Checkpoint the wizard before routing to provider settings."""

        run_wizard_worker(
            self,
            self._open_provider_settings(),
            exclusive=True,
            group="setup-wizard-review-settings",
        )

    async def _open_provider_settings(self) -> None:
        if not await self.persist_current_checkpoint():
            self._show_completion_save_error()
            return
        self._dismiss_screen(
            {
                "completed": False,
                "exit_route": "settings",
                "exit_context": {"category": "providers-models"},
            }
        )

    # -- explicit whole-wizard skip ---------------------------------------
    @on(Button.Pressed, "#setup-skip-entirely")
    def handle_skip_entirely(self) -> None:
        run_wizard_worker(
            self, self._skip_entirely(), exclusive=True, group="setup-wizard-advance"
        )

    async def _skip_entirely(self) -> None:
        async with self._draft_mutation_lock:
            saved = await self._complete_setup_locked()
        if not saved:
            self._show_completion_save_error()
            return
        self._dismiss_screen({"completed": True, "exit_route": None})

    async def _complete_setup_locked(self) -> bool:
        """Persist completion and draft deletion while the mutation lock is held."""

        if self._draft_mutations_terminal:
            return True
        _, delete_keys = wizard_state.build_setup_draft_mutation(None)
        saved = await self.commit_config(
            wizard_state.build_wizard_state_commit(completed=True),
            delete_keys=delete_keys,
        )
        if saved:
            self._draft_mutations_terminal = True
        return saved

    def _show_completion_save_error(self) -> None:
        """Keep completion failures visible without exposing config values."""

        try:
            self.steps[self.current_step].show_step_error(
                "Setup completion could not be saved. Retry before closing."
            )
        except (IndexError, AttributeError):
            logger.warning("Setup completion error could not render (category=ui)")

    async def persist_setup_checkpoint(self, active_step_id: str) -> bool:
        """Persist one allowlisted checkpoint after a successful step commit."""

        async with self._draft_mutation_lock:
            return await self._persist_setup_checkpoint_locked(active_step_id)

    async def _persist_setup_checkpoint_locked(self, active_step_id: str) -> bool:
        """Persist a checkpoint while the caller holds the mutation lock."""

        if self._draft_mutations_terminal:
            return False
        checkpoint_step_id = active_step_id
        if (
            self._staged_provider_draft is not None
            and not self._provider_setup_committed
            and active_step_id != wizard_state.STEP_PROVIDER
        ):
            # The endpoint and credential are intentionally memory-only. A
            # restart cannot safely reconstruct this staged connection, so
            # recovery returns to Provider until Model commits it atomically.
            checkpoint_step_id = wizard_state.STEP_PROVIDER
        try:
            draft = wizard_state.setup_draft_checkpoint(
                track=self.track,
                active_step_id=checkpoint_step_id,
                values=self.wizard_data,
            )
            settings, delete_keys = wizard_state.build_setup_draft_mutation(draft)
        except (TypeError, ValueError):
            logger.warning("Setup checkpoint rejected (category=validation)")
            return False
        if delete_keys:
            saved = await self.commit_config(settings, delete_keys=delete_keys)
        else:
            saved = await self.commit_config(settings)
        if saved:
            self.resume_draft = draft
            return True
        return False

    async def clear_resume_attempt(self, expected_target_id: str) -> bool:
        """Narrowly clear the marker against authoritative state under lock."""

        async with self._draft_mutation_lock:
            return await self._clear_resume_attempt_locked(expected_target_id)

    async def _clear_resume_attempt_locked(self, expected_target_id: str) -> bool:
        if self._draft_mutations_terminal:
            return False
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        first_run = app_config.get(wizard_state.WIZARD_STATE_SECTION)
        if not isinstance(first_run, Mapping):
            return False
        if wizard_state.coerce_wizard_flag(
            first_run.get(wizard_state.SETUP_COMPLETED_KEY)
        ):
            return False
        draft = wizard_state.read_setup_draft(app_config)
        if (
            draft is None
            or not draft.resume_attempted
            or draft.active_step_id != expected_target_id
        ):
            return False
        saved = await self.commit_config(
            {
                wizard_state.WIZARD_STATE_SECTION: {
                    wizard_state.DRAFT_RESUME_ATTEMPTED_KEY: False
                }
            }
        )
        if not saved:
            return False
        cleared = wizard_state.SetupDraft(
            version=draft.version,
            track=draft.track,
            active_step_id=draft.active_step_id,
            values=draft.values,
            resume_attempted=False,
        )
        self.resume_draft = cleared
        try:
            screen = self.screen
        except Exception:
            screen = None
        if isinstance(screen, FirstRunSetupWizard):
            screen.resume_draft = cleared
        return True

    async def persist_current_checkpoint(self) -> bool:
        """Persist the latest completed values with the currently visible target."""

        step = self.steps[self.current_step]
        if step.config is None:
            return False
        return await self.persist_setup_checkpoint(step.config.id)

    async def open_voice_api_key_settings(self, step: VoiceSetupStep) -> bool:
        """Checkpoint the current Voice draft, then route to Speech & TTS."""

        try:
            current = self.steps[self.current_step]
        except IndexError:
            return False
        if (
            self._finalized
            or current is not step
            or step.config is None
            or step.config.id != wizard_state.STEP_VOICE
        ):
            return False
        self.wizard_data[wizard_state.STEP_VOICE] = step.get_step_data()
        if not await self.persist_current_checkpoint():
            step.query_one("#setup-voice-status", Static).update(
                "Setup progress could not be saved. Retry opening Settings."
            )
            return False
        self._dismiss_screen(
            {
                "completed": False,
                "exit_route": "settings",
                "exit_context": {"category": "speech-tts"},
            }
        )
        return True

    # -- persistence (the only write path for steps) -----------------------
    async def commit_config(
        self,
        section_values: Mapping[str, Mapping[str, object]],
        *,
        delete_keys: Mapping[str, tuple[str, ...]] | None = None,
        after_write: Callable[[], None] | None = None,
        provider_setup_mutation: object | None = None,
    ) -> bool:
        """Serialize every config write through one worker-side call."""
        requested_deletes = {} if delete_keys is None else dict(delete_keys)
        if not section_values and not requested_deletes:
            return True
        if not wizard_state.commit_sections_allowed(section_values):
            logger.error("Wizard commit rejected non-owned sections")
            return False
        if not wizard_state.commit_sections_allowed(
            {section: {} for section in requested_deletes}
        ):
            logger.error("Wizard delete rejected non-owned sections")
            return False
        import asyncio

        if provider_setup_mutation is not None:
            from tldw_chatbook.Chat.provider_setup_persistence import (
                ProviderSetupMutation,
                persist_provider_setup,
            )

            if (
                type(provider_setup_mutation) is not ProviderSetupMutation
                or section_values != provider_setup_mutation.section_values
                or requested_deletes != provider_setup_mutation.delete_keys
                or after_write is not None
            ):
                logger.error("Wizard provider commit rejected (category=validation)")
                return False
            owner = getattr(self, "_first_run_provider_discovery_owner", None)
            evidence_save = (
                owner._begin_provider_evidence_save(provider_setup_mutation)
                if isinstance(owner, ProviderStep)
                else None
            )
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    persist_provider_setup,
                    provider_setup_mutation,
                )
            except BaseException:
                self._provider_last_config_result = None
                if isinstance(owner, ProviderStep):
                    owner._finish_provider_evidence_save(evidence_save, None)
                raise
            self._provider_last_config_result = result
            if result.fully_applied:
                self._mirror_into_app_config(section_values, requested_deletes)
                if isinstance(owner, ProviderStep):
                    owner._finish_provider_evidence_save(evidence_save, result)
                return True
            if isinstance(owner, ProviderStep):
                owner._finish_provider_evidence_save(evidence_save, None)
            return False

        from tldw_chatbook.config import save_settings_to_cli_config

        def _write() -> tuple[bool, Exception | None]:
            if requested_deletes:
                ok = save_settings_to_cli_config(
                    section_values, delete_keys=requested_deletes
                )
            else:
                ok = save_settings_to_cli_config(section_values)
            callback_error: Exception | None = None
            if ok and after_write is not None:
                try:
                    after_write()
                except Exception as exc:
                    callback_error = exc
            return ok, callback_error

        ok, callback_error = await asyncio.get_running_loop().run_in_executor(
            None, _write
        )
        if ok:
            self._mirror_into_app_config(section_values, requested_deletes)
        if callback_error is not None:
            raise callback_error
        return ok

    def _mirror_into_app_config(
        self,
        section_values: Mapping[str, Mapping[str, object]],
        delete_keys: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Keep the in-memory app_config consistent (chat_screen.py pattern)."""
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return
        for dotted_section, values in section_values.items():
            target = app_config
            for part in dotted_section.split("."):
                nxt = target.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    target[part] = nxt
                target = nxt
            target.update(values)
        for dotted_section, keys in (delete_keys or {}).items():
            target = app_config
            for part in dotted_section.split("."):
                target = target.get(part)
                if not isinstance(target, dict):
                    break
            if isinstance(target, dict):
                for key in keys:
                    target.pop(key, None)

    # -- completion / cancel ----------------------------------------------
    def _handle_complete(self, wizard_data: Dict[str, Any]) -> None:
        summary_data = wizard_data.get(wizard_state.STEP_SUMMARY, {})
        exit_route = summary_data.get("exit_route")
        offer_profile_interview = summary_data.get("offer_profile_interview") is True
        # F-B fix: complete_wizard() runs this synchronously inside _advance(),
        # the running "setup-wizard-advance" worker. Scheduling _finalize into
        # that same exclusive group would cancel_group() its own in-flight
        # worker (CPython forces a task whose coro returns while _must_cancel
        # is set into CANCELLED), so it gets a dedicated group, as
        # ProtectKeysStep's _on_password_result does. TASK-34100.1 review
        # round 2: the fence and busy line stay up until _finalize settles
        # (the completion write), so a second press cannot start another.
        run_wizard_worker(
            self,
            self._finalize(exit_route, offer_profile_interview),
            exclusive=True,
            group="setup-wizard-finalize",
        )
        self._finishing = True  # set once scheduled: a failed start lifts it

    async def _finalize(
        self,
        exit_route: Optional[str],
        offer_profile_interview: bool = False,
    ) -> None:
        """F3 hardening: a second entry is a clean no-op.

        Checked here (not just inside ``_dismiss_screen``) so a duplicate
        call -- e.g. a stray extra Finish click/ctrl+n racing the exclusive
        "setup-wizard-finalize" worker -- also skips re-committing
        ``first_run.setup_completed``, not merely the redundant dismiss.
        Deliberately does NOT set ``self._finalized`` itself:
        ``_dismiss_screen`` is the sole setter (see its docstring) -- if
        this method set the flag before calling ``_dismiss_screen``, that
        call would see it already True and skip the real dismiss on the
        very FIRST, intended run.
        """
        try:
            if self._finalized:
                return
            async with self._draft_mutation_lock:
                saved = await self._complete_setup_locked()
            if not saved:
                self._show_completion_save_error()
                return
            from tldw_chatbook.Constants import TAB_CHAT

            if exit_route == TAB_CHAT and not self._stage_console_first_chat_handoff():
                self._show_first_chat_handoff_error()
                return
            result = {"completed": True, "exit_route": exit_route}
            if offer_profile_interview:
                result["offer_profile_interview"] = True
            self._dismiss_screen(result)
        finally:
            self._finishing = False
            if not self._finalized and self.is_attached:  # setup stays open
                self._set_advancing(False)

    def _stage_console_first_chat_handoff(self) -> bool:
        """Stage a revision-fenced, secret-free target after setup commits."""

        from uuid import uuid4

        from tldw_chatbook.Chat.console_session_settings import (
            build_default_console_session_settings,
        )
        from tldw_chatbook.UI.Navigation.pending_handoff_store import (
            ConsoleFirstChatIntent,
            HandoffChannel,
        )

        try:
            snapshot = get_runtime_config_snapshot()
            defaults = build_default_console_session_settings(snapshot.values)
            provider = provider_config_key(defaults.provider)
            model = str(defaults.model or "").strip()
            if not provider or not model:
                return False

            session_id: str | None = None
            for screen in reversed(tuple(self.app_instance.screen_stack)):
                session_owner = getattr(screen, "_session", None)
                eligible_session = getattr(
                    session_owner,
                    "eligible_console_first_chat_session_id",
                    None,
                )
                if not callable(eligible_session):
                    continue
                session_id = eligible_session()
                break
            reserves_new_session = session_id is None
            if session_id is None:
                session_id = str(uuid4())
            intent = ConsoleFirstChatIntent(
                session_id=session_id,
                provider=provider,
                model=model,
                config_revision=snapshot.generation,
            )
            if reserves_new_session:
                self.app_instance.pending_handoffs.stage_reserved_console_first_chat(
                    intent
                )
            else:
                self.app_instance.pending_handoffs.stage(
                    HandoffChannel.CONSOLE_FIRST_CHAT,
                    intent,
                )
        except Exception as exc:  # noqa: BLE001 - keep the UI boundary retryable
            logger.warning(
                "First-chat handoff could not be staged (error_type={})",
                type(exc).__name__,
            )
            return False
        return True

    def _show_first_chat_handoff_error(self) -> None:
        """Keep a failed handoff retry attached to the mounted Summary."""

        try:
            self.steps[self.current_step].show_step_error(
                "Console could not open this setup yet. Review the provider and try again."
            )
        except (IndexError, AttributeError):
            logger.warning("First-chat handoff error could not render (category=ui)")

    def _dismiss_screen(self, result: Optional[dict]) -> None:
        """F3 hardening: the single choke point both ``_finalize`` (Finish)
        and ``_skip_entirely`` (the whole-wizard Skip button) funnel
        through to actually pop the screen -- idempotent no-op on a second
        entry, from either caller. Textual's ``Screen.dismiss()`` is not
        designed to tolerate being called twice on the same screen; without
        this guard, a duplicate call (Skip arriving after Finish already
        completed, or any other double-entry into either caller) would
        attempt a second dismiss.
        """
        if self._finalized or self._provider_dismiss_pending:
            return
        task = self._provider_commit_task
        if task is not None and not task.done() and self._provider_commit_write_started:
            self._provider_dismiss_pending = True
            self._show_provider_save_status(
                "Finishing save…",
                focus=True,
                announce=True,
            )
            self._sync_action_controls()
            run_wizard_worker(
                self,
                self._settle_provider_write_then_dismiss(task, result),
                exclusive=True,
                group="setup-wizard-provider-dismiss",
            )
            return
        self._complete_dismiss_screen(result)

    async def _settle_provider_write_then_dismiss(
        self,
        task: asyncio.Task[bool],
        result: dict | None,
    ) -> None:
        """Wait for an irreversible executor write before releasing its draft."""

        try:
            try:
                saved = await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=self._provider_dismiss_warning_seconds,
                )
            except TimeoutError:
                self._show_provider_save_status(
                    "Saving is taking longer than expected. Keep this setup "
                    "screen open; it will finish automatically, or let you "
                    "retry here if it fails.",
                    focus=True,
                    announce=True,
                )
                saved = await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                return
        except Exception:  # noqa: BLE001 - task boundary must recover any writer error.
            saved = False
        self._provider_dismiss_pending = False
        if self._provider_ui_detached:
            self.clear_provider_setup_sensitive_state(clear_widgets=False)
            return
        if saved:
            self._complete_dismiss_screen(result)
            return
        self._recover_from_provider_save_failure()

    def _show_provider_save_status(
        self,
        message: str,
        *,
        focus: bool = False,
        announce: bool = False,
    ) -> None:
        """Publish bounded save state only while this container is mounted."""

        if self._provider_ui_detached or not self.is_attached:
            return
        try:
            status = self.query_one("#setup-provider-save-status", _ProviderSaveStatus)
        except NoMatches:
            return
        status.update(message)
        status.set_class(not message, "hidden")
        if focus:
            status.focus()
        if announce:
            self.notify(message, severity="information")

    def hold_provider_save_settlement(self) -> bool:
        """Keep cancel actions on the active irreversible-save status."""

        if not self._provider_dismiss_pending:
            return False
        if self._provider_ui_detached or not self.is_attached:
            return True
        try:
            status = self.query_one("#setup-provider-save-status", _ProviderSaveStatus)
        except NoMatches:
            return True
        status.focus()
        self.notify(str(status.renderable), severity="information")
        return True

    def _recover_from_provider_save_failure(self) -> None:
        """Release failed-save secrets and return to an enabled Provider step."""

        self.clear_provider_setup_sensitive_state(
            clear_widgets=not self._provider_ui_detached
        )
        if self._provider_ui_detached:
            return
        owner = getattr(self, "_first_run_provider_discovery_owner", None)
        if isinstance(owner, ProviderStep):
            owner.prepare_retry_after_failed_save()
        if not self.is_attached:
            return
        provider_index = self._step_index_for_id(wizard_state.STEP_PROVIDER)
        if provider_index is not None:
            self.show_step(provider_index)
        self._show_provider_save_status(
            "Couldn't finish saving the provider. Review the endpoint and "
            "credential, then retry.",
            announce=True,
        )
        self._sync_action_controls()
        if isinstance(owner, ProviderStep) and owner.is_attached:
            try:
                owner.query_one("#setup-provider-endpoint", Input).focus()
            except NoMatches:
                return

    def _complete_dismiss_screen(self, result: Optional[dict]) -> None:
        """Clear provider state and dismiss after all irreversible work settles."""

        if self._finalized or self._provider_ui_detached:
            if self._provider_ui_detached:
                self.clear_provider_setup_sensitive_state(clear_widgets=False)
            return
        self._finalized = True
        self.clear_provider_setup_sensitive_state()
        screen = self.screen
        if isinstance(screen, FirstRunSetupWizard):
            screen.dismiss(result)

    def action_cancel(self) -> None:
        if self.hold_provider_save_settlement():
            return
        if self._advancing or self._failure_action_running:
            return
        screen = self.screen
        if isinstance(screen, FirstRunSetupWizard):
            screen.action_cancel()


class _SettlingGuardedConfirmationDialog(ConfirmationDialog):
    """TASK-2314: absorb a reflexive double-tap of the finish-later Escape.

    UAT live reproduction: the wizard is pushed while several heavy steps
    (10 composed steps, the full provider catalog, discovery workers) are
    still settling, so a user who presses Escape once and perceives no
    immediate feedback over that render lag reflexively presses it again.
    ``ConfirmationDialog``'s own binding -- Escape mirrors the Cancel
    button everywhere else in the app, "dismissing is always the safe
    outcome" (see that module's docstring), which is the right default for
    every OTHER use of the widget -- means that second press lands
    directly on THIS dialog (it is now the top of the screen stack) and
    silently snaps the wizard back open with no visible sign anything
    happened: exactly the "silently ignores Escape ... feels frozen" UAT
    finding, confirmed live by sending two Escape presses within
    milliseconds of the wizard's first paint (see task-2314's
    Implementation Notes for the reproduction).

    The fix is scoped to the Escape BINDING only, via a distinct action
    name -- never ``action_cancel_dialog`` itself, which the Cancel BUTTON
    also calls (``on_button_pressed``); a deliberate mouse click must stay
    instant regardless of timing. Escape still opens this dialog on the
    very first press (the "Escape -> confirm" asymmetry task-2314 asks to
    preserve is untouched: this only guards a SECOND press arriving too
    soon after the dialog itself appeared).

    task-31820 extended the guard's clock: it now starts at the dialog's
    first delivered frame (``call_after_refresh`` in ``on_mount``), not at
    mount. On a machine choked by a concurrent pytest sweep the paint
    lagged whole seconds behind the push, so a second Escape sent well
    after the 0.5s wall-clock grace still dismissed a dialog that had
    never been on screen -- the wizard read as "Escape is dead" while the
    footer's Exit button (mouse path) worked. Until the first frame lands,
    Escape is absorbed unconditionally; nothing else changed.
    """

    #: Absorbs a reflexive double-tap (typically well under 300ms apart);
    #: comfortably shorter than the time it takes to actually read
    #: "Steps you've already completed are saved...".
    _ESCAPE_GRACE_SECONDS = 0.5

    BINDINGS = [
        Binding("escape", "cancel_dialog_if_settled", "Cancel", show=False),
    ]

    def __init__(
        self,
        *args: Any,
        escape_grace_seconds: float = _ESCAPE_GRACE_SECONDS,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._escape_grace_seconds = escape_grace_seconds
        self._opened_at: Optional[float] = None

    def on_mount(self) -> None:
        # task-31820: anchor the settle clock to the first delivered FRAME,
        # not to mount. Live release-UAT walkthrough on a loaded machine
        # (full pytest sweep running): Escape on the Provider step opened
        # this dialog, but the paint lagged for seconds — a second
        # "is this thing on?" Escape landed after the 0.5s wall-clock grace
        # and dismissed a dialog nobody had ever seen. To the user, Escape
        # did nothing, twice, while the mouse path worked — the exact
        # "silently ignores Escape ... feels frozen" failure this class
        # exists to prevent. A dialog that has never painted can never be
        # deliberately re-Escaped, so the grace must start at first paint.
        self.call_after_refresh(self._mark_settled)

    def _mark_settled(self) -> None:
        if self._opened_at is None:
            self._opened_at = time.monotonic()

    async def action_cancel_dialog_if_settled(self) -> None:
        if self._opened_at is None:
            return  # never painted yet -- a second press cannot be deliberate
        if (time.monotonic() - self._opened_at) < self._escape_grace_seconds:
            return  # too soon to be a deliberate second press -- swallow it
        await self.action_cancel_dialog()


class FirstRunSetupWizard(WizardScreen):
    """Full-screen first-run setup wizard. Dismisses dict | None."""

    #: TASK-31807: this is a modal onboarding gate pushed over the initial
    #: screen at startup. A stray navigation -- e.g. a shell-destination key
    #: (F9/F10/ctrl+N ...) that leaks in during splash teardown, while the
    #: app's global bindings are live on the just-mounted initial screen and
    #: this wizard's own push is still a `call_after_refresh` behind it --
    #: would otherwise reach `_dismiss_navigation_overlays` and `dismiss(None)`
    #: the wizard, discarding onboarding with ZERO user input (and leaving
    #: `setup_started` persisted from `on_mount`, so it never re-offers
    #: cleanly). The wizard is only ever meant to be left through its own
    #: controls (Next / Back / Skip / the Esc confirm dialog), which dismiss it
    #: directly, so `TldwCli._handle_screen_navigation_locked` treats any
    #: navigation arriving while this screen is on the stack as spurious and
    #: ignores it rather than tearing the wizard down.
    blocks_stray_navigation: bool = True

    def __init__(
        self,
        app_instance,
        rerun: bool = False,
        resume_draft: wizard_state.SetupDraft | None = None,
        provider_dismiss_warning_seconds: float = 2.0,
    ):
        super().__init__(app_instance)
        self.rerun = rerun
        self.resume_draft = resume_draft
        self.provider_dismiss_warning_seconds = provider_dismiss_warning_seconds

    def compose(self) -> ComposeResult:
        yield SetupWizardContainer(
            self.app_instance,
            rerun=self.rerun,
            resume_draft=self.resume_draft,
            provider_dismiss_warning_seconds=self.provider_dismiss_warning_seconds,
        )
        # TASK-1505: the wizard's keys are otherwise undiscoverable — one
        # quiet, always-visible line names them.
        yield Static(
            "Enter / Ctrl+N next · Ctrl+B back · Esc skip setup",
            id="setup-key-hints",
            classes="setup-key-hints",
        )
        # TASK-21148 (UAT Z-1/Z-2): stock macOS Terminal is 80x24 — every
        # step still works there, but content scrolls hard and nothing used
        # to say so. One quiet line names the fix.
        yield Static("", id="setup-size-hint", classes="setup-size-hint hidden")

    def on_mount(self) -> None:
        if not self.rerun:
            self._persist_started_flag()
        self._sync_size_hint()

    def on_resize(self, event: object = None) -> None:
        self._sync_size_hint()

    def _sync_size_hint(self) -> None:
        """Show the enlarge-terminal nudge below ~100x30 (UAT Z-2)."""
        try:
            hint = self.query_one("#setup-size-hint", Static)
        except NoMatches:
            return
        width, height = self.size.width, self.size.height
        small = 0 < width < 100 or 0 < height < 30
        if small:
            hint.update(
                f"Small terminal ({width}×{height}) — setup works best at "
                "100×30 or larger. Everything still works; steps may scroll."
            )
        hint.set_class(not small, "hidden")
        # Styles-level too: hosts without the app stylesheet (bare test
        # harnesses) have no `.hidden` rule, and a docked row that only
        # PRETENDS to hide shifts every geometry below it.
        hint.display = small

    @wizard_work(thread=True, group="setup-wizard-started-flag")
    def _persist_started_flag(self) -> None:
        from tldw_chatbook.config import save_settings_to_cli_config

        try:
            saved = save_settings_to_cli_config(
                wizard_state.build_wizard_state_commit(started=True)
            )
        except Exception as exc:
            logger.warning(
                "Failed to persist wizard started flag "
                "(category=persistence, error_type={})",
                type(exc).__name__,
            )
            return
        if not saved:
            logger.warning(
                "Failed to persist wizard started flag "
                "(category=persistence, error_type=save_returned_false)"
            )
            return
        app_config = getattr(self.app_instance, "app_config", None)
        if isinstance(app_config, dict):
            app_config.setdefault(wizard_state.WIZARD_STATE_SECTION, {})[
                wizard_state.SETUP_STARTED_KEY
            ] = True

    def action_cancel(self) -> None:
        mode = "exit"
        message = (
            "Steps you've already completed are saved. You can continue "
            "setup any time from Settings ▸ Diagnostics."
        )
        try:
            container = self.query_one(SetupWizardContainer)
            if container.hold_provider_save_settlement():
                return
            if container._advancing or container._failure_action_running:
                return
            step = container.steps[container.current_step]
            step_id = step.config.id if step.config is not None else ""
            if step_id == wizard_state.STEP_WELCOME:
                mode = "skip"
                message = (
                    "Skip setup and stop showing it at launch? You can rerun "
                    "setup from Settings ▸ Diagnostics."
                )
            else:
                message = container.finish_later_message()
        except NoMatches:
            pass
        self._pending_cancel_mode = mode
        dialog = _SettlingGuardedConfirmationDialog(
            title="Skip setup?" if mode == "skip" else "Exit setup?",
            message=message,
            confirm_label="Skip setup" if mode == "skip" else "Exit setup",
            cancel_label="Keep going",
        )
        self.app.push_screen(dialog, self._handle_cancel_confirm)

    def _handle_cancel_confirm(self, confirmed: bool | None) -> None:
        if confirmed:
            try:
                if self.query_one(SetupWizardContainer).hold_provider_save_settlement():
                    return
            except NoMatches:
                return
            # TASK-1500: an uncommitted theme preview must not outlive the
            # wizard — finish-later restores whatever the user had before.
            try:
                self.query_one(AppearanceStep).revert_preview()
            except Exception:
                pass
            if getattr(self, "_pending_cancel_mode", "exit") == "skip":
                try:
                    container = self.query_one(SetupWizardContainer)
                except NoMatches:
                    return
                run_wizard_worker(
                    container,
                    container._skip_entirely(),
                    exclusive=True,
                    group="setup-wizard-advance",
                )
                return
            run_wizard_worker(
                self,
                self._finish_later(),
                exclusive=True,
                group="setup-wizard-finish-later",
            )

    async def _finish_later(self) -> None:
        try:
            container = self.query_one(SetupWizardContainer)
            if container.hold_provider_save_settlement():
                return
            saved = await container.persist_current_checkpoint()
        except Exception:
            logger.warning("Setup finish-later checkpoint failed (category=runtime)")
            saved = False
        if not saved:
            self.notify(
                "Setup progress could not be saved. Retry Exit setup.",
                severity="error",
            )
            return
        container._dismiss_screen(None)

    def _clear_resume_attempt_after_target_mount(
        self,
        container: SetupWizardContainer,
        target: WizardStep,
        target_step_id: str,
    ) -> None:
        """Fence marker clearing against navigation or screen replacement."""

        try:
            current = container.steps[container.current_step]
            same_screen = self.app.screen is self and container.screen is self
            same_container = self.query_one(SetupWizardContainer) is container
        except (IndexError, NoMatches):
            return
        if (
            not same_screen
            or not same_container
            or self.resume_draft is None
            or container.resume_draft is None
            or self.resume_draft.active_step_id != target_step_id
            or container.resume_draft.active_step_id != target_step_id
            or current is not target
            or current.config is None
            or current.config.id != target_step_id
            or not current.is_attached
            or not current.display
            or not current.visible
        ):
            return
        run_wizard_worker(
            self,
            container.clear_resume_attempt(target_step_id),
            exclusive=True,
            group="setup-wizard-resume-clear",
        )


#: Public names that moved out (TASK-33921, TASK-34100.1) and that this module
#: does not use itself, by owning module; ``__getattr__`` serves them so old
#: imports keep resolving. Private helpers and the two tunable timeouts are
#: absent on purpose: a test that patches one here must fail with
#: AttributeError, not silently patch a name the moved step no longer reads.
_MOVED_PUBLIC_NAMES: dict[str, str] = {
    "VoiceSetupStep": "first_run_voice_step",
    "EXIT_ROUTE_LIBRARY_NOTES": "first_run_summary_step",
    "ProviderChoiceList": "first_run_provider_step",
    "ProviderEndpointCandidateList": "first_run_provider_step",
    "ProviderEndpointCandidateOption": "first_run_provider_step",
    "GENERIC_DISCOVERY_FAILURE_CATEGORY": "first_run_model_discovery",
}


def __getattr__(name: str) -> object:
    """Re-export a public name that moved to its own module.

    Lazy: the owning module is imported on the first lookup.

    Args:
        name: The attribute looked up on this module.

    Returns:
        The moved object, looked up on its owning module.

    Raises:
        AttributeError: For any other missing name.
    """
    module_name = _MOVED_PUBLIC_NAMES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__package__}.{module_name}"), name)
