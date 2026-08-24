"""First-run setup wizard: hermes-agent's setup process in chatbook chrome.

Screen + container subclass over BaseWizard (which is never modified).
All decisions and config mutations are built by first_run_setup_state;
this module renders them and owns persistence via one exclusive worker.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
import os
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
)

from loguru import logger
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.compose import compose as _drain_compose_result
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Label,
    OptionList,
    RadioButton,
    RadioSet,
    Static,
    Switch,
)
from textual.widgets.option_list import Option
from textual.worker import Worker, get_current_worker

from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.config import get_runtime_config_snapshot
from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
    PARAKEET_PRECISIONS,
    active_managed_parakeet_dir,
    parakeet_descriptor,
    parakeet_v2_managed_service,
    parakeet_reference,
    parakeet_vad_descriptor,
    parakeet_vad_reference,
    run_parakeet_preflight,
    run_parakeet_provision,
    run_parakeet_vad_preflight,
    run_parakeet_vad_provision,
)
from tldw_chatbook.Model_Artifacts.store import managed_model_artifact_root
from tldw_chatbook.STT.parakeet_external import (
    ExternalParakeetVerificationError,
    format_external_parakeet_recovery,
)
from tldw_chatbook.STT.parakeet_sources import (
    ParakeetSourceError,
    ParakeetSourceErrorCode,
    ParakeetSourceKey,
    PreparedExternalSelection,
)
from tldw_chatbook.STT.transcribe_cpp_config import (
    configure_model_path as configure_transcribe_cpp_model_path,
    is_gguf_file,
)
from tldw_chatbook.Third_Party.textual_fspicker import (
    FileOpen,
    Filters,
    SelectDirectory,
)
from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message
from tldw_chatbook.UI.Screens.model_installed_view import lifecycle_failure_message
from tldw_chatbook.UI.Wizards import first_run_speech_step_state as speech_state
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state
from tldw_chatbook.UI.Wizards.BaseWizard import (
    WizardContainer,
    WizardNavigation,
    WizardProgress,
    WizardScreen,
    WizardStep,
    WizardStepConfig,
)
from tldw_chatbook.Widgets.ModelArtifacts import (
    ActivationRequested,
    DeletionRequested,
    InstallProgressed,
    ModelActivationControls,
    ModelInstallModal,
    ModelInstallProgress,
    make_progress_callback,
)
from tldw_chatbook.Widgets.delete_confirmation_dialog import DeleteConfirmationDialog
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_provider_support import (
        ConsoleProviderCatalogEntry,
    )


class SetupRadioButton(RadioButton):
    """RadioButton whose selected state is structural, not color-only.

    TASK-1497: stock ToggleButton renders one constant BUTTON_INNER glyph and
    conveys on/off purely through the glyph's color, which is invisible in a
    monochrome capture and fails WCAG 1.4.1 (use of color). The inner glyph
    itself switches here — ● selected, ○ unselected — so state survives any
    palette; a bold text-style on the selected row (see _wizards.tcss) is the
    second cue. BUTTON_INNER is set as an instance attribute right before the
    parent property renders, shadowing the class attribute per-state.
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
        self.BUTTON_INNER = "●" if self.value else "○"
        return super()._button


class SetupWizardProgress(WizardProgress):
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
        self.styles.width = "100%"
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
        if self.is_mounted:
            self._sync_compact_mode()
            self.refresh(recompose=True)
            self.call_after_refresh(self._sync_compact_mode)

    def on_mount(self) -> None:
        self.call_after_refresh(self._sync_compact_mode)

    def on_resize(self) -> None:
        self._sync_compact_mode()

    def _titled_track_width(self) -> int:
        """Return the minimum safe width for the fully titled tracker."""

        item_width = sum(
            self._NUMBER_WIDTH
            + self._TITLE_HORIZONTAL_MARGIN
            + self._ITEM_HORIZONTAL_MARGIN
            + self._ITEM_SAFETY_WIDTH
            + len(item.title)
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
                number_classes = f"step-number {item.state}"
                yield Static(
                    "✓" if item.state == "complete" else str(index + 1),
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
                    if item.state == "complete":
                        connector_classes += " complete"
                    connector = Static("", classes=connector_classes)
                    connector.display = not compact
                    yield connector


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


class SetupStep(WizardStep):
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
            # Nested containers (``with RadioSet(): yield SetupRadioButton``)
            # would silently end up childless -- their leaves float as
            # stray top-level siblings instead -- if drained with a bare
            # list(). textual.compose.compose() reproduces that per-item
            # attach step itself, so it is safe to fully exhaust up front.
            buffered = _drain_compose_result(self, self.compose_step())
        except Exception as exc:
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

    def preferred_focus(self) -> Optional[Widget]:
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

    def show_step_error(self, message: str) -> None:
        try:
            self.query_one(".setup-step-error", Static).update(message)
        except Exception:
            logger.warning("Setup step error had nowhere to render: {}", message)


@dataclass(frozen=True, slots=True)
class _SetupFailureAction:
    """Identity captured when one required-failure recovery action starts."""

    screen: "FirstRunSetupWizard"
    index: int
    step: SetupStep
    step_id: str
    failure: SetupStepFailure


CLOUD_PROBE_TIMEOUT_SECONDS = 8.0


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


class ProviderChoiceList(OptionList):
    """Provider options with Space activation and post-navigation signaling."""

    BINDINGS = [Binding("space", "select", "Select", show=False)]

    class Interacted(Message):
        """A keyboard navigation action has resolved against the list."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._provider_interaction_ready = False
        super().__init__(*args, **kwargs)
        self._provider_interaction_ready = True

    def _post_navigation_interaction(self) -> None:
        if self._provider_interaction_ready:
            self.post_message(self.Interacted())

    def action_cursor_up(self) -> None:
        super().action_cursor_up()
        self._post_navigation_interaction()

    def action_cursor_down(self) -> None:
        super().action_cursor_down()
        self._post_navigation_interaction()

    def action_first(self) -> None:
        super().action_first()
        self._post_navigation_interaction()

    def action_last(self) -> None:
        super().action_last()
        self._post_navigation_interaction()

    def action_page_up(self) -> None:
        previous_highlight = self.highlighted
        previous_option = self.highlighted_option
        super().action_page_up()
        if (
            self.highlighted is None
            and previous_highlight is not None
            and previous_option is not None
            and not previous_option.disabled
        ):
            self.highlighted = previous_highlight
        self._post_navigation_interaction()

    def action_page_down(self) -> None:
        previous_highlight = self.highlighted
        previous_option = self.highlighted_option
        super().action_page_down()
        if (
            self.highlighted is None
            and previous_highlight is not None
            and previous_option is not None
            and not previous_option.disabled
        ):
            self.highlighted = previous_highlight
        self._post_navigation_interaction()


class ProviderEndpointCandidateOption(Option):
    """One detected endpoint or a disabled result-list heading/status row."""

    def __init__(
        self,
        prompt: Text,
        *,
        option_id: str,
        server: object | None = None,
    ) -> None:
        super().__init__(prompt, id=option_id, disabled=server is None)
        self.server = server


class ProviderEndpointCandidateList(OptionList):
    """Keyboard-selectable detected endpoints with nonselectable status rows."""

    BINDINGS = [Binding("space", "select", "Select", show=False)]


@dataclass(slots=True)
class _ProviderConnectionUiDraft:
    """Memory-only, provider-owned controls that never render credential values."""

    endpoint: str = ""
    api_key: str = ""
    clear_requested: bool = False
    key_input_visible: bool = True
    auth_collapsed: bool = True
    detected_servers: tuple[object, ...] = ()
    detected_server: object | None = None
    credential_revision: int = 0

    def __repr__(self) -> str:
        return (
            "_ProviderConnectionUiDraft("
            f"endpoint_present={bool(self.endpoint)!r}, "
            f"credential_present={bool(self.api_key)!r}, "
            f"clear_requested={self.clear_requested!r}, "
            f"key_input_visible={self.key_input_visible!r}, "
            f"auth_collapsed={self.auth_collapsed!r}, "
            f"detected_count={len(self.detected_servers)!r}, "
            f"credential_revision={self.credential_revision!r})"
        )

    def __copy__(self) -> object:
        raise TypeError("Provider credentials are memory-only.")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("Provider credentials are memory-only.")

    def clear_secret(self) -> None:
        self.api_key = ""


def _provider_group_option_id(title: str) -> str:
    """Return the deterministic option ID for a provider group heading."""
    return "group-" + "-".join(title.casefold().split())


def _provider_options(
    entries: Sequence[ConsoleProviderCatalogEntry],
) -> list[Option]:
    """Build grouped provider options with non-selectable heading rows."""
    options: list[Option] = []
    for group_title, group in ProviderStep._grouped_sections(entries):
        options.append(
            ProviderChoiceOption(
                Text(group_title, style="bold"),
                option_id=_provider_group_option_id(group_title),
                provider_key=None,
            )
        )
        options.extend(
            ProviderChoiceOption(
                Text(entry.display_name),
                option_id=f"provider-{entry.readiness_key}",
                provider_key=entry.readiness_key,
            )
            for entry in group
        )
    return options


async def _probe_first_run_provider_connection(
    endpoint: str,
    *,
    provider: str,
    credential_source: str,
    credential_value: str | None,
):
    """Send one exact draft to the shared probe without retaining its secret."""

    import httpx

    from tldw_chatbook.Chat.local_server_discovery import (
        DISCOVERY_PROBE_TIMEOUT_SECONDS,
    )
    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbeOutcome,
        probe_settings_endpoint,
    )

    del credential_source
    client: httpx.AsyncClient | None = None
    try:
        if credential_value:
            try:
                client = httpx.AsyncClient(
                    headers={"Authorization": f"Bearer {credential_value}"}
                )
            except Exception:  # noqa: BLE001 - return a bounded connection state.
                return SettingsEndpointProbeOutcome(
                    state="unreachable",
                    category="connection_error",
                    summary="unreachable: connection error",
                )
        return await probe_settings_endpoint(
            endpoint,
            provider=provider,
            timeout=(
                CLOUD_PROBE_TIMEOUT_SECONDS
                if credential_value
                else DISCOVERY_PROBE_TIMEOUT_SECONDS
            ),
            http_client=client,
        )
    finally:
        if client is not None:
            try:
                await client.aclose()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - cleanup detail is not user-actionable.
                pass


def _model_ids_from_discovery_result(result: object) -> tuple[str, ...]:
    """Extract exact typed catalog IDs without accepting duck-typed payloads."""

    from tldw_chatbook.Chat.local_server_discovery import MODEL_IDS_MAX_COUNT
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        DiscoveredModel,
        ModelDiscoveryResult,
    )

    if type(result) is not ModelDiscoveryResult:
        raise ValueError("Model discovery result is invalid.")
    if result.status != "success":
        return ()
    if type(result.models) is not tuple or len(result.models) > MODEL_IDS_MAX_COUNT:
        raise ValueError("Model discovery result is invalid.")
    model_ids: list[str] = []
    seen: set[str] = set()
    for discovered in result.models:
        if type(discovered) is not DiscoveredModel:
            raise ValueError("Model discovery result is invalid.")
        try:
            model_id = wizard_state.validate_first_run_model_id(discovered.model_id)
        except ValueError as exc:
            raise ValueError("Model discovery result is invalid.") from exc
        if model_id in seen:
            continue
        seen.add(model_id)
        model_ids.append(model_id)
    return tuple(model_ids)


def _legacy_model_ids(values: object) -> tuple[str, ...]:
    """Validate the intentionally retained injected string-list test seam."""

    from tldw_chatbook.Chat.local_server_discovery import MODEL_IDS_MAX_COUNT

    if type(values) not in {list, tuple} or len(values) > MODEL_IDS_MAX_COUNT:
        raise ValueError("Legacy model discovery result is invalid.")
    model_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        model_id = wizard_state.validate_first_run_model_id(value)
        if model_id in seen:
            continue
        seen.add(model_id)
        model_ids.append(model_id)
    return tuple(model_ids)


def _model_discovery_ui_outcome(result: object) -> tuple[list[str], str, str]:
    """Interpret one typed discovery result into bounded Model-step state."""

    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        ModelDiscoveryResult,
    )

    if type(result) is not ModelDiscoveryResult:
        raise ValueError("Model discovery result is invalid.")
    if result.status == "success":
        models = list(_model_ids_from_discovery_result(result))
        return models, "available" if models else "empty", ""
    if result.status == "unsupported" or (
        result.error is not None and result.error.kind == "unsupported_endpoint"
    ):
        return [], "listing_unavailable", ""
    error_kind = result.error.kind if result.error is not None else ""
    category = {
        "invalid_response": "invalid response",
        "missing_credentials": "authentication",
    }.get(error_kind, "request failed")
    return [], "connection_failed", category


def _first_run_discovery_staged_settings(
    provider_draft: wizard_state.FirstRunProviderDraft,
    discovery_key: wizard_state.FirstRunModelDiscoveryKey,
) -> dict[str, dict[str, dict[str, str]]]:
    """Build transient settings for one exact typed discovery boundary."""

    from tldw_chatbook.Chat.provider_setup_persistence import provider_endpoint_key

    endpoint = (
        provider_draft.discovery_endpoint
        or provider_draft.endpoint
        or discovery_key.connection_identity[1]
    )
    settings = {provider_endpoint_key(discovery_key.provider_key): endpoint}
    credential = provider_draft.credential
    credential_value = wizard_state._credential_value_for_boundary(credential)
    boundary_source = credential.source
    if boundary_source == "draft" and not credential_value:
        boundary_source = "none"
    settings["credential_source"] = boundary_source
    if credential.source == "draft":
        settings["api_key"] = credential_value
    elif credential.source == "environment":
        settings["api_key_env_var"] = credential_value
    elif credential.source == "none":
        settings["api_key"] = ""
    return {"api_settings": {discovery_key.provider_key: settings}}


def _process_environment() -> Mapping[str, str]:
    """Return the current process environment without retaining it on a widget."""

    return os.environ


def _empty_environment() -> Mapping[str, str]:
    """Return an empty environment after provider state has been disposed."""

    return {}


@dataclass(frozen=True, slots=True)
class _CredentialObservation:
    """Private credential version marker whose digest is never represented."""

    source: str
    digest: bytes = field(repr=False)

    def matches(self, source: str, digest: bytes) -> bool:
        return self.source == source and hmac.compare_digest(self.digest, digest)


class ProviderStep(SetupStep):
    """Choose a provider, supply credentials, verify without blocking."""

    _MAX_PROVIDER_DRAFTS = 64
    _OPENAI_COMPATIBLE_PROBE_PROVIDERS = frozenset(
        {
            "aphrodite",
            "custom",
            "custom_2",
            "deepseek",
            "groq",
            "koboldcpp",
            "llama_cpp",
            "local_llamacpp",
            "local_llamafile",
            "local_ollama",
            "local_vllm",
            "mistral",
            "mistralai",
            "ollama",
            "oobabooga",
            "openai",
            "openrouter",
            "qwencloud",
            "tabbyapi",
            "vllm",
        }
    )

    def __init__(
        self,
        wizard: Optional["SetupWizardContainer"] = None,
        config: Optional[WizardStepConfig] = None,
        *,
        discover: Optional[Callable[..., Any]] = None,
        probe: Optional[Callable[..., Any]] = None,
        local_discover: Optional[Callable[..., Any]] = None,
        environ: Optional[Mapping[str, str] | Callable[[], Mapping[str, str]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(wizard=wizard, config=config, **kwargs)
        from tldw_chatbook.Chat.local_server_discovery import discover_local_servers

        # ``discover`` is the selected-provider seam. The provider-neutral
        # localhost scan stays separate so an untouched mount can never pass
        # an application config mapping to a provider/catalog service.
        self._discover = discover
        self._local_discover = local_discover or discover_local_servers
        self._probe = probe or _probe_first_run_provider_connection
        # Resolve environment credentials from a provider at each boundary.
        # Keeping the mapping itself on the widget leaks rotated values through
        # retained object state after dismissal.
        if environ is None:
            self._environment_provider = _process_environment
        elif callable(environ):
            self._environment_provider = environ
        else:
            self._environment_provider = lambda source=environ: source
        self._credential_observation_key = os.urandom(32)
        self._sensitive_key_input: Input | None = None
        self._sensitive_endpoint_input: Input | None = None
        self._credential_observations: dict[str, _CredentialObservation] = {}
        self.probe_generation = 0
        self._discovery_visible = False
        self._local_discovery_generation = 0
        self._local_discovery_state = "idle"
        self._selected_discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = (
            None
        )
        self._selected_discovery_generation = 0
        self._selected_discovery_state = "idle"
        self._selected_discovery_credential_decision: tuple[str, str | int] | None = (
            None
        )
        self._selected_provider_models: dict[
            wizard_state.FirstRunModelDiscoveryKey, tuple[str, ...]
        ] = {}
        self._selected_provider_outcomes: dict[
            wizard_state.FirstRunModelDiscoveryKey, object
        ] = {}
        self._selected_discovery_done: asyncio.Event | None = None
        self.selected_provider_key: str = ""
        self.provider_value_for_chat_defaults: str = ""
        self._last_committed_provider_value: Optional[str] = None
        self._entered_key = False
        self._clear_requested = False
        self._credential_revision = 0
        self._credential_decision_generation = 0
        self._last_credential_decision: tuple[str, str | int] | None = None
        self._detected_endpoint_provider_key = ""
        self._detected_servers: tuple[object, ...] = ()
        self._local_discovery_provider_key = ""
        self._provider_choice_interacted = False
        self._updating_connection_controls = False
        self._pending_programmatic_endpoint_changes: list[tuple[str, str]] = []
        self._provider_drafts: OrderedDict[str, _ProviderConnectionUiDraft] = (
            OrderedDict()
        )
        self._provider_draft_generation = 0
        from tldw_chatbook.Chat.provider_test_evidence import (
            ProviderTestEvidenceStore,
        )

        self._provider_test_evidence = ProviderTestEvidenceStore()
        self._active_probe_token: object | None = None
        self._last_tested_provider_identity: object | None = None
        if wizard is not None:
            setattr(wizard, "_first_run_provider_discovery_owner", self)

    def compose_step(self) -> ComposeResult:
        from tldw_chatbook.Chat.console_provider_support import (
            supported_console_provider_catalog,
        )

        entries = supported_console_provider_catalog()
        with Vertical(classes="setup-provider"):
            yield Static("Connect a provider", classes="setup-title")
            yield Static(
                "Cloud providers need an API key. Local servers just need to "
                "be running — we'll look for them.",
                classes="setup-subtitle",
            )
            # TASK-1498: the discovery payoff is PINNED above the list — the
            # subtitle promises "we'll look for them", so the found-server
            # banner must appear where that promise was made, not below a
            # scrolling list. Being before the OptionList in DOM also keeps the
            # TASK-1496 Tab order intact (provider list → key input; the button is
            # only reachable backwards or by click, and it is hidden until a
            # server is actually found).
            # Hidden until discovery finds something — an empty banner would
            # otherwise burn two rows of the tight 120x40 budget that keeps
            # the API-key input on screen (TASK-1495's row accounting).
            yield Static(
                "",
                id="setup-provider-detected",
                classes="setup-probe-status setup-detected-banner hidden",
            )
            yield Button(
                "Use this server",
                id="setup-provider-use-detected",
                classes="hidden",
                variant="primary",
            )
            yield ProviderChoiceList(
                *_provider_options(entries),
                id="setup-provider-choice",
                classes="setup-choice-list",
            )
            with Vertical(id="setup-provider-connection", classes="hidden"):
                yield Label("Endpoint", classes="setup-field-label")
                yield Input(
                    id="setup-provider-endpoint",
                    placeholder="http://127.0.0.1:8080 or full chat URL",
                )
                yield Static(
                    "",
                    id="setup-provider-effective-chat",
                    classes="setup-endpoint-value",
                )
                yield Static(
                    "",
                    id="setup-provider-endpoint-status",
                    classes="setup-probe-status",
                )
                with Horizontal(classes="setup-provider-connection-actions"):
                    yield Button("Detect", id="setup-provider-detect")
                    yield Button("Test", id="setup-provider-test", variant="primary")
                yield ProviderEndpointCandidateList(
                    ProviderEndpointCandidateOption(
                        Text("Detected endpoints", style="bold"),
                        option_id="detected-endpoints-heading",
                    ),
                    ProviderEndpointCandidateOption(
                        Text("Not checked yet"),
                        option_id="detected-endpoints-status",
                    ),
                    id="setup-provider-detection-results",
                    classes="setup-detection-results hidden",
                )
            with Collapsible(
                title="Authentication (optional)",
                collapsed=True,
                id="setup-provider-auth-toggle",
                classes="hidden",
            ):
                yield Label("API key", classes="setup-field-label")
                yield Input(
                    password=True,
                    id="setup-provider-api-key",
                    placeholder="Paste your API key",
                )
                yield Static(
                    "", id="setup-provider-key-status", classes="setup-probe-status"
                )
                with Horizontal(id="setup-provider-key-actions", classes="hidden"):
                    yield Button("Keep current", id="setup-provider-key-keep")
                    yield Button("Replace", id="setup-provider-key-replace")
                    yield Button("Clear", id="setup-provider-key-clear")
            yield Static(
                "", id="setup-provider-probe-status", classes="setup-probe-status"
            )
            yield Static("", classes="setup-step-error")

    # TASK-1498: providers most first-time users are actually looking for, in
    # display order. Filtered against the live catalog, so a missing key
    # simply doesn't render.
    _POPULAR_PROVIDER_KEYS = ("openai", "anthropic", "ollama", "llama_cpp")

    @classmethod
    def _grouped_sections(cls, entries):
        """Sectioned provider list: Popular, then Cloud, Local, Other.

        Args:
            entries: ConsoleProviderCatalogEntry sequence from the catalog.

        Returns:
            List of (section_title, entries) pairs, empty sections dropped.
        """
        from tldw_chatbook.Chat.provider_catalog import (
            PROVIDER_CUSTOM_GROUP_KEYS,
        )

        by_key = {e.readiness_key: e for e in entries}
        popular = [by_key[key] for key in cls._POPULAR_PROVIDER_KEYS if key in by_key]
        popular_keys = {e.readiness_key for e in popular}
        rest = [e for e in entries if e.readiness_key not in popular_keys]
        alpha = lambda e: e.display_name.lower()  # noqa: E731
        cloud = sorted(
            (
                e
                for e in rest
                if e.requires_api_key
                and e.readiness_key not in PROVIDER_CUSTOM_GROUP_KEYS
            ),
            key=alpha,
        )
        local = sorted(
            (
                e
                for e in rest
                if not e.requires_api_key
                and e.readiness_key not in PROVIDER_CUSTOM_GROUP_KEYS
            ),
            key=alpha,
        )
        other = sorted(
            (e for e in rest if e.readiness_key in PROVIDER_CUSTOM_GROUP_KEYS),
            key=alpha,
        )
        sections = [
            ("Popular", popular),
            ("Cloud", cloud),
            ("Local", local),
            ("Other", other),
        ]
        return [(title, group) for title, group in sections if group]

    def preferred_focus(self) -> Optional[Widget]:
        """Focus the provider list on entry, even when the pinned discovery
        button is visible (it precedes the list in DOM order).

        Returns:
            The provider OptionList, or None if it is not queryable yet.
        """
        try:
            return self.query_one("#setup-provider-choice", OptionList)
        except Exception:
            return None

    def on_show(self) -> None:
        super().on_show()
        if self._discovery_visible:
            return
        self._discovery_visible = True
        if self.selected_provider_key:
            credential_rotated = self._sync_live_credential_revision()
            if credential_rotated:
                return
            provider_draft = self._effective_provider_draft()
            discovery_key = self._model_discovery_key(provider_draft)
            if (
                discovery_key != self._selected_discovery_key
                or self._selected_discovery_state
                in {
                    "idle",
                    "cancelled",
                }
            ):
                self._begin_selected_provider_discovery(provider_draft)
        elif self._local_discovery_state in {"idle", "cancelled"}:
            self._start_discovery()

    def on_hide(self) -> None:
        super().on_hide()
        if not self._discovery_visible:
            return
        self._discovery_visible = False
        if self._can_handoff_selected_discovery():
            if self._local_discovery_state == "in_progress":
                self._local_discovery_state = "cancelled"
            self._local_discovery_generation += 1
            self._cancel_worker_groups(
                "setup-provider-local-discovery", "setup-provider-probe"
            )
        else:
            self._cancel_discovery_workers()

    def on_unmount(self) -> None:
        self.clear_sensitive_widgets(release_references=True)
        self._cancel_discovery_workers(publish_status=False)
        self.clear_sensitive_state()

    def clear_sensitive_state(self) -> None:
        """Drop provider-owned state without touching mounted UI controls."""

        self._discovery_visible = False
        self._provider_test_evidence.invalidate()
        self._active_probe_token = None
        self._last_tested_provider_identity = None
        self._selected_discovery_credential_decision = None
        self._selected_discovery_key = None
        self._selected_provider_models.clear()
        self._selected_provider_outcomes.clear()
        self._credential_observations.clear()
        self._credential_observation_key = b""
        self._environment_provider = _empty_environment
        self._pending_programmatic_endpoint_changes.clear()
        self._detected_servers = ()
        self._detected_endpoint_provider_key = ""
        for attribute in ("detected_server", "detected_base_url"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        for draft in self._provider_drafts.values():
            draft.clear_secret()
        self._provider_drafts.clear()

    def clear_sensitive_widgets(self, *, release_references: bool = False) -> None:
        """Clear provider inputs only while their widget tree is attached."""

        try:
            key_input = self._sensitive_key_input
            endpoint_input = self._sensitive_endpoint_input
            if (key_input is None or endpoint_input is None) and self.is_mounted:
                key_input = self.query_one("#setup-provider-api-key", Input)
                endpoint_input = self.query_one("#setup-provider-endpoint", Input)
            if key_input is None or endpoint_input is None:
                return
            with (
                key_input.prevent(Input.Changed),
                endpoint_input.prevent(Input.Changed),
            ):
                key_input.value = ""
                endpoint_input.value = ""
            self.query_one("#setup-provider-effective-chat", Static).update("")
            self.query_one("#setup-provider-endpoint-status", Static).update("")
            self.query_one("#setup-provider-probe-status", Static).update("")
            self.query_one("#setup-provider-detected", Static).update("")
            self.query_one(
                "#setup-provider-detection-results",
                ProviderEndpointCandidateList,
            ).clear_options()
        except Exception:
            pass
        finally:
            if release_references:
                self._sensitive_key_input = None
                self._sensitive_endpoint_input = None

    def on_mount(self) -> None:
        self._sensitive_key_input = self.query_one("#setup-provider-api-key", Input)
        self._sensitive_endpoint_input = self.query_one(
            "#setup-provider-endpoint", Input
        )

    def prepare_retry_after_failed_save(self) -> None:
        """Release the failed draft, then restore live boundary resolvers."""

        self._cancel_discovery_workers(publish_status=False)
        self.clear_sensitive_state()
        self._environment_provider = _process_environment
        self._credential_observation_key = os.urandom(32)

    def _environment(self) -> Mapping[str, str]:
        """Read the current environment through the injected live provider."""

        try:
            environment = self._environment_provider()
        except Exception:
            return {}
        return environment if isinstance(environment, Mapping) else {}

    def _cancel_discovery_workers(self, *, publish_status: bool = True) -> None:
        """Invalidate and cancel setup-owned network work without publishing."""

        selected_was_in_progress = self._selected_discovery_state == "in_progress"
        self._cancel_active_probe()
        self._obsolete_provider_generation(
            "setup-provider-discovery",
            "setup-provider-probe",
        )
        if selected_was_in_progress and publish_status and self.is_mounted:
            try:
                self.query_one("#setup-provider-probe-status", Static).update(
                    "Check paused; returning will retry."
                )
            except Exception:
                pass
        if self._local_discovery_state == "in_progress":
            self._local_discovery_state = "cancelled"
        self._local_discovery_generation += 1
        self._cancel_worker_groups("setup-provider-local-discovery")

    def cancel_selected_discovery_handoff(self) -> None:
        """Fence Provider-owned discovery once Model no longer consumes it."""

        self._obsolete_provider_generation("setup-provider-discovery")
        self._selected_provider_models.clear()
        self._selected_provider_outcomes.clear()
        self.wizard._first_run_selected_provider_models = {}
        self.wizard._first_run_selected_provider_outcomes = {}

    def _cancel_worker_groups(self, *groups: str) -> None:
        try:
            for group in groups:
                self.workers.cancel_group(self, group)
        except Exception:
            pass

    def _obsolete_provider_generation(self, *groups: str) -> int:
        """Invalidate shared provider work and cancel the requested groups."""

        if self._selected_discovery_done is not None:
            self._selected_discovery_done.set()
        if self._selected_discovery_state == "in_progress":
            self._selected_discovery_state = "cancelled"
        self.probe_generation += 1
        self._cancel_worker_groups(*groups)
        return self.probe_generation

    def _start_discovery(self, *, user_requested: bool = False) -> None:
        self._local_discovery_generation += 1
        generation = self._local_discovery_generation
        provider_key = self.selected_provider_key
        self._local_discovery_provider_key = provider_key
        self._local_discovery_state = "in_progress"
        if user_requested:
            self._render_detection_results((), status="Searching local endpoints…")
        self.run_worker(
            partial(self._discover_servers, generation, provider_key),
            exclusive=True,
            group="setup-provider-local-discovery",
        )

    async def _discover_servers(self, generation: int, provider_key: str) -> None:
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        state = "complete"
        try:
            servers = tuple(await self._local_discover(app_config) or ())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state = "failed"
            logger.debug(
                "Wizard local discovery failed (error_type={})",
                type(exc).__name__,
            )
            servers = ()
        if (
            generation != self._local_discovery_generation
            or provider_key != self._local_discovery_provider_key
            or provider_key != self.selected_provider_key
        ):
            return
        self._local_discovery_state = state
        if not self.is_mounted or not self.is_active:
            return
        self._detected_servers = servers
        if servers:
            self._render_detection_results(servers)
            self._apply_discovered_server(servers[0])
        else:
            status = (
                "Detection failed. Try again."
                if state == "failed"
                else "No local endpoints found."
            )
            self._render_detection_results((), status=status)
        self._capture_provider_ui_draft(provider_key)

    def _render_detection_results(
        self,
        servers: Sequence[object],
        *,
        status: str = "Select an endpoint to use.",
    ) -> None:
        """Render bounded, secret-free detection options without editing input."""

        from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
        from tldw_chatbook.Chat.provider_catalog import provider_display_name
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            resolve_provider_endpoint,
        )

        options: list[Option] = [
            ProviderEndpointCandidateOption(
                Text("Detected endpoints", style="bold"),
                option_id="detected-endpoints-heading",
            ),
            ProviderEndpointCandidateOption(
                Text(status),
                option_id="detected-endpoints-status",
            ),
        ]
        for index, server in enumerate(servers):
            if type(server) is not DiscoveredLocalServer:
                continue
            try:
                provider_key = self._canonical_provider_key(server.provider_key)
            except (TypeError, ValueError):
                provider_key = ""
            resolution = resolve_provider_endpoint(provider_key, server.base_url)
            if resolution.persisted_endpoint is None:
                label = f"Candidate {index + 1}: invalid endpoint"
                selectable_server = None
            else:
                label = (
                    f"{provider_display_name(provider_key)} · "
                    f"{resolution.persisted_display}"
                )
                selectable_server = server
            options.append(
                ProviderEndpointCandidateOption(
                    Text(label),
                    option_id=f"detected-endpoint-{index}",
                    server=selectable_server,
                )
            )
        results = self.query_one(
            "#setup-provider-detection-results", ProviderEndpointCandidateList
        )
        results.clear_options()
        results.add_options(options)
        results.remove_class("hidden")

    def _highlight_discovered_server(self, server: object) -> None:
        """Restore the exact candidate row without equating duplicate URLs."""

        try:
            results = self.query_one(
                "#setup-provider-detection-results",
                ProviderEndpointCandidateList,
            )
        except NoMatches:
            return
        for index in range(results.option_count):
            option = results.get_option_at_index(index)
            if getattr(option, "server", None) is server:
                results.highlighted = index
                return

    def _apply_discovered_server(self, server: Any) -> None:
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            resolve_provider_endpoint,
        )

        try:
            provider_key = self._canonical_provider_key(server.provider_key)
        except (TypeError, ValueError):
            provider_key = ""
        resolution = resolve_provider_endpoint(provider_key, server.base_url)
        banner = self.query_one("#setup-provider-detected", Static)
        use_button = self.query_one("#setup-provider-use-detected", Button)
        if resolution.persisted_endpoint is None:
            for attribute in ("detected_server", "detected_base_url"):
                if hasattr(self, attribute):
                    delattr(self, attribute)
            self._detected_endpoint_provider_key = ""
            banner.update("")
            banner.add_class("hidden")
            use_button.add_class("hidden")
            return
        self.detected_server = server
        self.detected_base_url = server.base_url
        banner.update(f"Found a local endpoint: {resolution.persisted_display}.")
        banner.remove_class("hidden")
        use_button.remove_class("hidden")
        self._highlight_discovered_server(server)

    @staticmethod
    def _canonical_provider_key(provider_key: str) -> str:
        from tldw_chatbook.Chat.console_provider_support import (
            resolve_console_provider_identity,
        )

        return resolve_console_provider_identity(provider_key).readiness_key

    def _provider_evidence_store(self):
        """Return the exact shared evidence owner used by this mounted step."""

        return self._provider_test_evidence

    def _provider_ui_draft(self, provider_key: str) -> _ProviderConnectionUiDraft:
        draft = self._provider_drafts.get(provider_key)
        if draft is not None:
            self._provider_drafts.move_to_end(provider_key)
            return draft
        draft = _ProviderConnectionUiDraft()
        self._provider_drafts[provider_key] = draft
        while len(self._provider_drafts) > self._MAX_PROVIDER_DRAFTS:
            _, evicted = self._provider_drafts.popitem(last=False)
            evicted.clear_secret()
        return draft

    def _capture_provider_ui_draft(self, provider_key: str | None = None) -> None:
        """Capture only the active provider's controls before replacing them."""

        owner = provider_key or self.selected_provider_key
        if not owner or not self.is_mounted:
            return
        draft = self._provider_ui_draft(owner)
        try:
            draft.endpoint = self.query_one("#setup-provider-endpoint", Input).value
            key_input = self.query_one("#setup-provider-api-key", Input)
            draft.api_key = key_input.value
            draft.key_input_visible = bool(key_input.display)
            draft.auth_collapsed = self.query_one(
                "#setup-provider-auth-toggle", Collapsible
            ).collapsed
        except NoMatches:
            return
        draft.clear_requested = self._clear_requested
        draft.detected_servers = self._detected_servers
        draft.detected_server = getattr(self, "detected_server", None)
        draft.credential_revision = self._credential_revision

    def _provider_requires_api_key(self, provider_key: str) -> bool:
        from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        return get_provider_readiness(
            provider_key, app_config, environ=self._environment()
        ).requires_api_key

    def _credential_at_request_boundary(self) -> tuple[str, str | None, object]:
        """Resolve a credential once, immediately before a live request."""

        from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
        from tldw_chatbook.config import is_valid_provider_api_key

        provider_key = self.selected_provider_key
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        key_input = self.query_one("#setup-provider-api-key", Input)
        ui_draft = self._provider_drafts.get(provider_key)
        typed = (
            key_input.value.strip()
            if key_input.display
            else (ui_draft.api_key.strip() if ui_draft is not None else "")
        )
        base_readiness = get_provider_readiness(
            provider_key, app_config, environ=self._environment()
        )
        if typed:
            if is_valid_provider_api_key(typed):
                return "draft", typed, base_readiness
            return "none", None, base_readiness
        if self._clear_requested:
            return "none", None, base_readiness
        if base_readiness.ready and base_readiness.api_key is not None:
            source = (
                "environment"
                if str(base_readiness.api_key_source).startswith("env:")
                else "stored"
            )
            return source, base_readiness.api_key, base_readiness
        return "none", None, base_readiness

    def _sync_live_credential_revision(self) -> bool:
        """Invalidate exact evidence when a request-boundary credential rotates."""

        provider_key = self.selected_provider_key
        if not provider_key or not self._credential_observation_key:
            return False
        source, value, _ = self._credential_at_request_boundary()
        previous = self._credential_observations.get(provider_key)
        digest = hmac.new(
            self._credential_observation_key,
            f"{source}\0{value or ''}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
        observation = _CredentialObservation(source, digest)
        self._credential_observations[provider_key] = observation
        if previous is None or previous.matches(source, digest):
            return False
        self._credential_decision_generation += 1
        self._credential_revision += 1
        self._invalidate_provider_test()
        if self._selected_discovery_done is not None:
            self._selected_discovery_done.set()
        self._selected_discovery_key = None
        self._selected_discovery_credential_decision = None
        self._selected_discovery_state = "cancelled"
        self._selected_provider_models.clear()
        self._selected_provider_outcomes.clear()
        self._capture_provider_ui_draft()
        provider_draft = self._effective_provider_draft()
        stage_provider = getattr(self.wizard, "stage_provider_setup", None)
        if provider_draft is not None and callable(stage_provider):
            stage_provider(provider_draft)
        invalidate_handoff = getattr(
            self.wizard, "invalidate_provider_model_handoff", None
        )
        if callable(invalidate_handoff):
            invalidate_handoff()
        if self.is_mounted and provider_draft is not None:
            self._begin_selected_provider_discovery(
                provider_draft, sync_live_credential=False
            )
        return True

    def _remember_current_credential(self) -> None:
        """Rebase private rotation tracking after an explicit UI decision."""

        provider_key = self.selected_provider_key
        if not provider_key or not self._credential_observation_key:
            return
        source, value, _ = self._credential_at_request_boundary()
        digest = hmac.new(
            self._credential_observation_key,
            f"{source}\0{value or ''}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
        self._credential_observations[provider_key] = _CredentialObservation(
            source, digest
        )

    def _current_provider_readiness(self):
        """Return shared readiness after applying the current transient decision."""

        from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

        provider_key = self.selected_provider_key
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        source, value, base = self._credential_at_request_boundary()
        if source in {"stored", "environment"}:
            return base
        api_settings = app_config.get("api_settings", {})
        staged_api_settings = (
            dict(api_settings) if isinstance(api_settings, Mapping) else {}
        )
        settings = dict(self._provider_settings(provider_key))
        settings.pop("api_key", None)
        typed_value = self.query_one("#setup-provider-api-key", Input).value.strip()
        if self._clear_requested or (typed_value and source == "none"):
            settings.pop("api_key_env_var", None)
        if source == "draft" and value is not None:
            settings["api_key"] = value
        staged_api_settings[provider_key] = settings
        staged_config = dict(app_config)
        staged_config["api_settings"] = staged_api_settings
        return get_provider_readiness(
            provider_key, staged_config, environ=self._environment()
        )

    def _probe_target(self) -> str:
        provider_key = self.selected_provider_key
        if provider_key not in self._OPENAI_COMPATIBLE_PROBE_PROVIDERS:
            return ""
        candidate = ""
        try:
            connection = self.query_one("#setup-provider-connection", Vertical)
            if connection.display:
                candidate = self.query_one("#setup-provider-endpoint", Input).value
            else:
                candidate = self._cloud_probe_base_url(provider_key)
        except Exception:
            return ""
        if not candidate.strip():
            return ""
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            resolve_provider_endpoint,
        )

        resolution = resolve_provider_endpoint(provider_key, candidate)
        if resolution.errors or resolution.models_url is None:
            return ""
        return candidate

    def _provider_current_draft_identity(self):
        """Build a secret-free identity for the exact controls now on screen."""

        from tldw_chatbook.Chat.provider_endpoint_contract import (
            canonical_connection_identity,
        )
        from tldw_chatbook.Chat.provider_test_evidence import ProviderDraftIdentity

        provider_key = self.selected_provider_key
        target = self._probe_target()
        connection_identity = canonical_connection_identity(provider_key, target)
        if connection_identity is None:
            return None
        credential_source, _, _ = self._credential_at_request_boundary()
        return ProviderDraftIdentity(
            provider_key=provider_key,
            connection_identity=connection_identity,
            credential_source=credential_source,
            credential_revision=self._credential_revision,
            draft_generation=self._provider_draft_generation,
        )

    def _cancel_active_probe(self) -> bool:
        token = self._active_probe_token
        if token is None:
            return False
        cancelled = self._provider_test_evidence.cancel_probe(token)
        if cancelled and self._active_probe_token is token:
            self._active_probe_token = None
        return cancelled

    def _invalidate_provider_test(self, *, changed: bool = True) -> None:
        """Invalidate exact evidence and only clear status owned by that probe."""

        invalidate_save = getattr(
            self.wizard, "invalidate_provider_write_expectation", None
        )
        if callable(invalidate_save):
            invalidate_save()
        self._provider_draft_generation += 1
        cancelled = self._cancel_active_probe()
        invalidated = self._provider_test_evidence.invalidate()
        self._obsolete_provider_generation(
            "setup-provider-discovery", "setup-provider-probe"
        )
        if (cancelled or invalidated or changed) and self.is_mounted:
            self.query_one("#setup-provider-probe-status", Static).update(
                "Provider settings changed since test; test again." if changed else ""
            )

    def _credential_semantics_changed(self) -> None:
        self._credential_decision_generation += 1
        self._credential_revision += 1
        self._remember_current_credential()
        self._invalidate_provider_test()
        self._capture_provider_ui_draft()
        self._refresh_auth_readiness()

    def _model_semantics_changed(
        self,
        *,
        model_id: str = "",
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = None,
    ) -> None:
        """Keep evidence only for a model returned by its exact settled probe."""

        from tldw_chatbook.Chat.provider_test_evidence import ProviderDraftIdentity

        if self._sync_live_credential_revision():
            return
        tested = self._last_tested_provider_identity
        current_credential_source, _, _ = self._credential_at_request_boundary()
        credential_source_matches = (
            type(discovery_key) is wizard_state.FirstRunModelDiscoveryKey
            and type(tested) is ProviderDraftIdentity
            and (
                discovery_key.credential_source == tested.credential_source
                or (
                    discovery_key.credential_source == "none"
                    and tested.credential_source == "stored"
                    and current_credential_source == "stored"
                )
            )
        )
        if (
            type(discovery_key) is wizard_state.FirstRunModelDiscoveryKey
            and type(tested) is ProviderDraftIdentity
            and discovery_key.provider_key == tested.provider_key
            and discovery_key.connection_identity == tested.connection_identity
            and credential_source_matches
            and discovery_key.credential_revision == tested.credential_revision
        ):
            evidence = self._provider_test_evidence.evidence_for(tested)
            if (
                evidence is not None
                and evidence.endpoint == "reachable"
                and model_id in evidence.model_ids
            ):
                return
        self._invalidate_provider_test()

    def _begin_provider_evidence_save(self, mutation: object):
        """Lease settled evidence for an equivalent atomic provider save."""

        from tldw_chatbook.Chat.provider_setup_persistence import ProviderSetupMutation
        from tldw_chatbook.Chat.provider_test_evidence import (
            ProviderDraftIdentity,
        )

        tested = self._last_tested_provider_identity
        if (
            type(mutation) is not ProviderSetupMutation
            or type(tested) is not ProviderDraftIdentity
            or mutation.semantic_identity is None
        ):
            return None
        lease = self._provider_test_evidence.begin_save(tested)
        if lease is None:
            return None
        semantic = mutation.semantic_identity
        saved = ProviderDraftIdentity(
            provider_key=semantic.provider_key,
            connection_identity=semantic.connection_identity,
            credential_source=semantic.credential_source,
            credential_revision=semantic.credential_revision,
            draft_generation=max(semantic.draft_generation, tested.draft_generation),
        )
        return tested, saved, lease

    def _finish_provider_evidence_save(
        self, save: object, result: object | None
    ) -> None:
        from tldw_chatbook.config import ConfigMutationResult

        if type(save) is not tuple or len(save) != 3:
            return
        tested, saved, lease = save
        if type(result) is not ConfigMutationResult or not result.fully_applied:
            self._provider_test_evidence.cancel_save(lease)
            return
        if self._provider_test_evidence.rebase_after_save(
            tested, saved, result, lease=lease
        ):
            self._last_tested_provider_identity = saved

    def _refresh_auth_readiness(self) -> None:
        if not self.selected_provider_key or not self.is_mounted:
            return
        readiness = self._current_provider_readiness()
        auth = self.query_one("#setup-provider-auth-toggle", Collapsible)
        auth.title = (
            "Authentication"
            if readiness.requires_api_key
            else "Authentication (optional)"
        )
        test_button = self.query_one("#setup-provider-test", Button)
        target = self._probe_target()
        identity = self._provider_current_draft_identity() if target else None
        test_available = bool(target and identity is not None)
        test_button.disabled = not readiness.ready or not test_available
        status = self.query_one("#setup-provider-key-status", Static)
        if not readiness.ready:
            recovery = readiness.recovery or "Add a provider credential."
            status.update(f"API key required. {recovery}")
            return
        if self._clear_requested:
            status.update(
                "No API key will be used for chat. The stored key will be removed "
                "when you continue."
            )
            return
        credential_source, _, _ = self._credential_at_request_boundary()
        if test_available:
            unavailable = ""
        elif self.selected_provider_key in self._OPENAI_COMPATIBLE_PROBE_PROVIDERS:
            unavailable = " Enter a valid endpoint to enable connection testing."
        else:
            unavailable = " Connection testing is unavailable for this provider."
        if credential_source == "stored":
            status.update(
                f"An API key is already configured for this provider.{unavailable}"
            )
        elif credential_source == "environment":
            env_var = readiness.env_var or "the configured environment variable"
            status.update(
                f"Found {env_var} in your environment; nothing to store.{unavailable}"
            )
        elif credential_source == "draft":
            status.update(
                f"A replacement API key is ready for this provider.{unavailable}"
            )
        else:
            status.update(unavailable.strip())

    def _credential_draft(
        self, *, revision: int | None = None
    ) -> wizard_state.ProviderCredentialDraft:
        """Return the current credential decision without exposing its value."""

        provider_key = self.selected_provider_key
        key_input = self.query_one("#setup-provider-api-key", Input)
        ui_draft = self._provider_drafts.get(provider_key)
        typed_key = (
            key_input.value.strip()
            if key_input.display and key_input.value
            else (ui_draft.api_key.strip() if ui_draft is not None else "")
        )
        if typed_key:
            source, value = "draft", typed_key
        elif self._clear_requested:
            source, value = "draft", ""
        else:
            source, _, readiness = self._credential_at_request_boundary()
            value = (readiness.env_var or "") if source == "environment" else ""
        return wizard_state.ProviderCredentialDraft(
            source, value, self._credential_revision if revision is None else revision
        )

    def _credential_decision(self) -> tuple[str, str | int]:
        credential = self._credential_draft(revision=self._credential_revision)
        return (
            credential.source,
            (
                wizard_state._credential_value_for_boundary(credential)
                if credential.source == "environment"
                else self._credential_decision_generation
            ),
        )

    def _provider_settings(self, provider_key: str) -> Mapping[str, object]:
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        try:
            return wizard_state._first_run_provider_settings(  # noqa: SLF001
                app_config, provider_key
            )
        except (TypeError, ValueError):
            return {}

    def _initial_endpoint_for(self, provider_key: str) -> str:
        from tldw_chatbook.Chat.console_provider_endpoints import (
            builtin_provider_endpoint,
            first_configured_endpoint,
        )

        configured = first_configured_endpoint(self._provider_settings(provider_key))
        if configured:
            return configured
        local_defaults = {
            "llama_cpp": "http://127.0.0.1:8080",
            "local_llamacpp": "http://127.0.0.1:8080",
            "ollama": "http://127.0.0.1:11434",
            "local_ollama": "http://127.0.0.1:11434",
        }
        return local_defaults.get(provider_key) or (
            builtin_provider_endpoint(
                provider_key, self._provider_settings(provider_key)
            )
            or ""
        )

    def _provider_exposes_endpoint(self, provider_key: str) -> bool:
        from tldw_chatbook.Chat.console_provider_endpoints import (
            URL_BASED_PROVIDER_KEYS,
            provider_uses_endpoint,
        )

        settings = self._provider_settings(provider_key)
        return provider_key in URL_BASED_PROVIDER_KEYS or provider_uses_endpoint(
            provider_key, settings
        )

    def _refresh_endpoint_resolution(self) -> None:
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            resolve_provider_endpoint,
        )

        provider_key = self.selected_provider_key
        effective = self.query_one("#setup-provider-effective-chat", Static)
        status = self.query_one("#setup-provider-endpoint-status", Static)
        if not provider_key:
            effective.update("")
            status.update("")
            return
        endpoint = self.query_one("#setup-provider-endpoint", Input).value
        resolution = resolve_provider_endpoint(provider_key, endpoint)
        if resolution.chat_url is None:
            effective.update("")
            status.update(
                resolution.errors[0] if resolution.errors else "Invalid endpoint."
            )
            return
        effective.update(f"Chat URL: {resolution.chat_display}")
        status.update(" ".join(resolution.warnings))

    def _effective_provider_draft(
        self, *, revision: int | None = None
    ) -> wizard_state.FirstRunProviderDraft | None:
        """Resolve the exact staged connection used for discovery and commit."""

        provider_key = self.selected_provider_key
        if not provider_key:
            return None
        endpoint = ""
        try:
            connection = self.query_one("#setup-provider-connection", Vertical)
            if connection.display:
                endpoint = self.query_one("#setup-provider-endpoint", Input).value
                if (
                    not endpoint.strip()
                    and self._detected_endpoint_provider_key == provider_key
                ):
                    endpoint = str(getattr(self, "detected_base_url", "") or "")
        except Exception:
            if self._detected_endpoint_provider_key == provider_key:
                endpoint = str(getattr(self, "detected_base_url", "") or "")
        try:
            draft = wizard_state.FirstRunProviderDraft(
                provider=provider_key,
                endpoint=endpoint,
                credential=self._credential_draft(revision=revision),
            )
            return wizard_state.resolve_first_run_provider_draft(
                draft, getattr(self.wizard.app_instance, "app_config", {}) or {}
            )
        except (TypeError, ValueError):
            return None

    def _model_discovery_key(
        self,
        provider_draft: wizard_state.FirstRunProviderDraft | None,
    ) -> wizard_state.FirstRunModelDiscoveryKey | None:
        if provider_draft is None:
            return None
        try:
            return wizard_state.build_first_run_model_discovery_key(provider_draft)
        except ValueError:
            return None

    def _discovery_staged_settings(
        self,
        provider_draft: wizard_state.FirstRunProviderDraft,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey,
    ) -> dict[str, dict[str, dict[str, str]]]:
        """Build transient exact settings; callers must never persist or cache it."""

        return _first_run_discovery_staged_settings(provider_draft, discovery_key)

    def _begin_selected_provider_discovery(
        self,
        provider_draft: wizard_state.FirstRunProviderDraft | str | None,
        *,
        sync_live_credential: bool = True,
    ) -> None:
        """Start one selected-provider probe/catalog request generation."""

        credential_rotated = (
            self._sync_live_credential_revision() if sync_live_credential else False
        )
        if credential_rotated:
            return
        elif isinstance(provider_draft, str):
            canonical_key = self._canonical_provider_key(provider_draft)
            if canonical_key != self.selected_provider_key:
                return
            provider_draft = self._effective_provider_draft()
        discovery_key = self._model_discovery_key(provider_draft)
        if provider_draft is None or discovery_key is None:
            self._obsolete_provider_generation(
                "setup-provider-discovery", "setup-provider-probe"
            )
            self._selected_discovery_key = None
            self._selected_discovery_state = "idle"
            self._selected_provider_models.clear()
            self._selected_provider_outcomes.clear()
            self.wizard._first_run_selected_provider_models = {}
            self.wizard._first_run_selected_provider_outcomes = {}
            self.wizard._first_run_provider_config_preconditions = {}
            return
        capture_precondition = getattr(
            self.wizard, "capture_provider_config_precondition", None
        )
        config_precondition = (
            capture_precondition(discovery_key)
            if callable(capture_precondition)
            else None
        )
        generation = self._obsolete_provider_generation(
            "setup-provider-discovery",
            "setup-provider-probe",
        )
        self._selected_discovery_key = discovery_key
        self._selected_discovery_credential_decision = self._credential_decision()
        self._selected_discovery_generation = generation
        self._selected_discovery_state = "in_progress"
        self._selected_provider_models.clear()
        self._selected_provider_outcomes.clear()
        self.wizard._first_run_selected_provider_models = {}
        self.wizard._first_run_selected_provider_outcomes = {}
        self.wizard._first_run_provider_config_preconditions = (
            {discovery_key: config_precondition}
            if config_precondition is not None
            else {}
        )
        self._selected_discovery_done = asyncio.Event()
        self.query_one("#setup-provider-probe-status", Static).update(
            "Checking the selected provider…"
        )
        self.run_worker(
            partial(
                self._discover_selected_provider,
                provider_draft,
                discovery_key,
                generation,
            ),
            exclusive=True,
            group="setup-provider-discovery",
        )

    def _owns_selected_discovery(
        self,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey,
        generation: int,
    ) -> bool:
        if (
            not self.is_attached
            or generation != self.probe_generation
            or generation != self._selected_discovery_generation
            or discovery_key != self._selected_discovery_key
            or discovery_key.provider_key != self.selected_provider_key
        ):
            return False
        current_key = self._model_discovery_key(self._effective_provider_draft())
        staged_key = self._model_discovery_key(
            getattr(self.wizard, "staged_provider_draft", None)
        )
        return discovery_key == current_key and (
            self.is_active or discovery_key == staged_key
        )

    def _can_handoff_selected_discovery(self) -> bool:
        if self._selected_discovery_state not in {"in_progress", "complete"}:
            return False
        staged_key = self._model_discovery_key(
            getattr(self.wizard, "staged_provider_draft", None)
        )
        return staged_key is not None and staged_key == self._selected_discovery_key

    async def _discover_selected_provider(
        self,
        provider_draft: wizard_state.FirstRunProviderDraft,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey,
        generation: int,
    ) -> None:
        provider_key = discovery_key.provider_key
        provider_result: tuple[Any, ...] = ()
        models: tuple[str, ...] = ()
        model_outcome: object | None = None
        attempted = False
        failed = False
        try:
            if self._discover is not None:
                attempted = True
                try:
                    provider_result = tuple(
                        await asyncio.wait_for(
                            self._discover(provider_key),
                            timeout=MODEL_DISCOVERY_TIMEOUT_SECONDS,
                        )
                        or ()
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    failed = True
                    logger.debug(
                        "Wizard selected provider discovery failed (error_type={})",
                        type(exc).__name__,
                    )
            if not self._owns_selected_discovery(discovery_key, generation):
                return

            scope_service = getattr(
                self.wizard.app_instance,
                "llm_provider_catalog_scope_service",
                None,
            )
            discover_models = getattr(scope_service, "discover_models", None)
            if callable(discover_models):
                attempted = True
                try:
                    result = await asyncio.wait_for(
                        discover_models(
                            mode="local",
                            provider=provider_key,
                            staged_settings=self._discovery_staged_settings(
                                provider_draft, discovery_key
                            ),
                            use_shared_cache=False,
                        ),
                        timeout=MODEL_DISCOVERY_TIMEOUT_SECONDS,
                    )
                    models = _model_ids_from_discovery_result(result)
                    model_outcome = result
                    if result.status != "success":
                        failed = True
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    failed = True
                    logger.debug(
                        "Wizard selected model discovery failed (error_type={})",
                        type(exc).__name__,
                    )
            if not self._owns_selected_discovery(discovery_key, generation):
                return

            self._selected_provider_models[discovery_key] = models
            if model_outcome is not None:
                self._selected_provider_outcomes[discovery_key] = model_outcome
            setattr(
                self.wizard,
                "_first_run_selected_provider_models",
                dict(self._selected_provider_models),
            )
            self.wizard._first_run_selected_provider_outcomes = dict(
                self._selected_provider_outcomes
            )
            discovered_server = next(
                (
                    item
                    for item in provider_result
                    if getattr(item, "provider_key", None) == provider_key
                    and getattr(item, "base_url", None)
                ),
                None,
            )
            if discovered_server is not None:
                self._apply_discovered_server(discovered_server)

            self._selected_discovery_state = "failed" if failed else "complete"
            from tldw_chatbook.Chat.provider_catalog import provider_display_name

            display = provider_display_name(provider_key)
            status = self.query_one("#setup-provider-probe-status", Static)
            if models:
                status.update(f"Found {len(models)} model(s) for {display}.")
            elif failed:
                status.update(
                    f"Couldn't discover models for {display}. You can continue anyway."
                )
            elif attempted:
                status.update(f"Checked {display}; no models were reported.")
            else:
                status.update("")
        finally:
            done = self._selected_discovery_done
            if done is not None and generation == self.probe_generation:
                done.set()

    async def _models_from_selected_discovery(
        self,
        provider_key: str,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = None,
    ) -> tuple[str, ...] | None:
        """Return this generation's models without starting another request."""

        canonical_key = self._canonical_provider_key(provider_key)
        if canonical_key != self.selected_provider_key:
            return None
        provider_draft = getattr(self.wizard, "staged_provider_draft", None)
        if type(provider_draft) is not wizard_state.FirstRunProviderDraft:
            provider_draft = self._effective_provider_draft()
        staged_discovery_key = self._model_discovery_key(provider_draft)
        if discovery_key is not None and discovery_key != staged_discovery_key:
            return None
        discovery_key = staged_discovery_key
        if discovery_key is None:
            return None
        if discovery_key in self._selected_provider_models:
            return self._selected_provider_models[discovery_key]
        done = self._selected_discovery_done
        if done is None:
            return None
        await done.wait()
        return self._selected_provider_models.get(discovery_key)

    async def _outcome_from_selected_discovery(
        self,
        provider_key: str,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey,
    ) -> object | None:
        """Return this generation's typed result without starting a request."""

        canonical_key = self._canonical_provider_key(provider_key)
        if canonical_key != self.selected_provider_key:
            return None
        staged_key = self._model_discovery_key(
            getattr(self.wizard, "staged_provider_draft", None)
        )
        if discovery_key != staged_key:
            return None
        if discovery_key in self._selected_provider_outcomes:
            return self._selected_provider_outcomes[discovery_key]
        done = self._selected_discovery_done
        if done is None:
            return None
        await done.wait()
        return self._selected_provider_outcomes.get(discovery_key)

    def _test_evidence_for_discovery_key(
        self, discovery_key: wizard_state.FirstRunModelDiscoveryKey
    ):
        """Return settled probe evidence only for the same secret-free identity."""

        from tldw_chatbook.Chat.provider_test_evidence import ProviderDraftIdentity

        tested = self._last_tested_provider_identity
        if type(tested) is not ProviderDraftIdentity:
            return None
        source_matches = (
            discovery_key.credential_source == tested.credential_source
            or (
                discovery_key.credential_source == "none"
                and tested.credential_source == "stored"
            )
        )
        if not (
            discovery_key.provider_key == tested.provider_key
            and discovery_key.connection_identity == tested.connection_identity
            and source_matches
            and discovery_key.credential_revision == tested.credential_revision
        ):
            return None
        return self._provider_test_evidence.evidence_for(tested)

    @on(Input.Changed, "#setup-provider-endpoint")
    def _on_endpoint_changed(self, event: Input.Changed) -> None:
        self._refresh_endpoint_resolution()
        pending = next(
            (
                item
                for item in self._pending_programmatic_endpoint_changes
                if item[1] == event.value
            ),
            None,
        )
        if pending is not None:
            self._pending_programmatic_endpoint_changes.remove(pending)
            return
        if self._updating_connection_controls:
            return
        for attribute in ("detected_base_url", "detected_server"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        self._detected_endpoint_provider_key = ""
        self._invalidate_provider_test()
        self._selected_discovery_key = None
        self._selected_provider_models.clear()
        self._selected_provider_outcomes.clear()
        self._capture_provider_ui_draft()
        self._refresh_auth_readiness()

    @on(Button.Pressed, "#setup-provider-detect")
    def _on_detect_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._cancel_worker_groups("setup-provider-local-discovery")
        self._start_discovery(user_requested=True)

    @on(OptionList.OptionSelected, "#setup-provider-detection-results")
    def _on_detected_endpoint_selected(self, event: OptionList.OptionSelected) -> None:
        from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            resolve_provider_endpoint,
        )

        server = getattr(event.option, "server", None)
        if type(server) is not DiscoveredLocalServer:
            return
        try:
            provider_key = self._canonical_provider_key(server.provider_key)
        except (TypeError, ValueError):
            provider_key = ""
        resolution = resolve_provider_endpoint(provider_key, server.base_url)
        if resolution.persisted_endpoint is None:
            self.query_one("#setup-provider-endpoint-status", Static).update(
                "The selected endpoint is invalid."
            )
            return
        detected_servers = self._detected_servers
        self.select_provider(provider_key)
        self._detected_servers = detected_servers
        self._render_detection_results(detected_servers)
        self._apply_discovered_server(server)
        self._detected_endpoint_provider_key = provider_key
        self._updating_connection_controls = True
        try:
            endpoint_input = self.query_one("#setup-provider-endpoint", Input)
            if endpoint_input.value != server.base_url:
                self._pending_programmatic_endpoint_changes.append(
                    (provider_key, server.base_url)
                )
                endpoint_input.value = server.base_url
        finally:
            self._updating_connection_controls = False
        self._refresh_endpoint_resolution()
        self._begin_selected_provider_discovery(self._effective_provider_draft())
        self._capture_provider_ui_draft()

    @on(Button.Pressed, "#setup-provider-use-detected")
    def _on_use_detected(self) -> None:
        """One-click connect: adopt the discovered server as the provider."""
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            resolve_provider_endpoint,
        )

        server = getattr(self, "detected_server", None)
        if server is None:
            return
        try:
            provider_key = self._canonical_provider_key(server.provider_key)
        except (TypeError, ValueError):
            provider_key = ""
        resolution = resolve_provider_endpoint(provider_key, server.base_url)
        if resolution.persisted_endpoint is None:
            self.query_one("#setup-provider-endpoint-status", Static).update(
                "The selected endpoint is invalid."
            )
            return
        detected_servers = self._detected_servers or (server,)
        self.select_provider(provider_key)
        self._detected_servers = detected_servers
        self._render_detection_results(detected_servers)
        self._apply_discovered_server(server)
        self._detected_endpoint_provider_key = self.selected_provider_key
        self._updating_connection_controls = True
        try:
            endpoint_input = self.query_one("#setup-provider-endpoint", Input)
            if endpoint_input.value != server.base_url:
                self._pending_programmatic_endpoint_changes.append(
                    (self.selected_provider_key, server.base_url)
                )
                endpoint_input.value = server.base_url
        finally:
            self._updating_connection_controls = False
        self._refresh_endpoint_resolution()
        self.query_one("#setup-provider-detected", Static).update(
            f"✓ Using {resolution.persisted_display}."
        )
        self.query_one("#setup-provider-detected", Static).remove_class("hidden")
        self.query_one("#setup-provider-use-detected", Button).remove_class("hidden")
        self._begin_selected_provider_discovery(self._effective_provider_draft())
        self._capture_provider_ui_draft()

    def _clear_detected_provider_state(self) -> None:
        """Drop an adopted endpoint and its provider-owned discovery results."""

        for attribute in ("detected_base_url", "detected_server"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        self._detected_endpoint_provider_key = ""
        self._detected_servers = ()
        self._selected_provider_models.clear()
        self._selected_provider_outcomes.clear()
        try:
            banner = self.query_one("#setup-provider-detected", Static)
            banner.update("")
            banner.add_class("hidden")
            self.query_one("#setup-provider-use-detected", Button).add_class("hidden")
            results = self.query_one(
                "#setup-provider-detection-results", ProviderEndpointCandidateList
            )
            results.clear_options()
            results.add_options(
                [
                    ProviderEndpointCandidateOption(
                        Text("Detected endpoints", style="bold"),
                        option_id="detected-endpoints-heading",
                    ),
                    ProviderEndpointCandidateOption(
                        Text("Not checked yet"),
                        option_id="detected-endpoints-status",
                    ),
                ]
            )
            results.add_class("hidden")
        except Exception:
            pass

    def select_provider(self, provider_key: str) -> None:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            read_provider_secret_presence,
        )

        provider_key = self._canonical_provider_key(provider_key)
        previous_provider = self.selected_provider_key
        provider_changed = provider_key != previous_provider
        had_saved_draft = provider_key in self._provider_drafts
        if provider_changed:
            self._capture_provider_ui_draft(previous_provider)
            self._invalidate_provider_test(changed=False)
            self._local_discovery_generation += 1
            self._local_discovery_provider_key = ""
            self._cancel_worker_groups("setup-provider-local-discovery")
            self._clear_detected_provider_state()
        self.selected_provider_key = provider_key
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        presence = read_provider_secret_presence(
            app_config, self._environment(), provider_key=provider_key
        )
        ui_draft = self._provider_ui_draft(provider_key)
        if not had_saved_draft:
            ui_draft.endpoint = self._initial_endpoint_for(provider_key)
            ui_draft.key_input_visible = not (
                presence.inline_configured or presence.env_var_set
            )
            ui_draft.auth_collapsed = not self._provider_requires_api_key(provider_key)
        self._clear_requested = ui_draft.clear_requested
        self._credential_revision = ui_draft.credential_revision
        status = self.query_one("#setup-provider-key-status", Static)
        actions = self.query_one("#setup-provider-key-actions", Horizontal)
        key_input = self.query_one("#setup-provider-api-key", Input)
        connection = self.query_one("#setup-provider-connection", Vertical)
        auth = self.query_one("#setup-provider-auth-toggle", Collapsible)
        if provider_changed:
            self._updating_connection_controls = True
            try:
                key_input.value = ui_draft.api_key
                key_input.display = ui_draft.key_input_visible
                endpoint_visible = self._provider_exposes_endpoint(provider_key)
                connection.display = endpoint_visible
                connection.set_class(not endpoint_visible, "hidden")
                endpoint_input = self.query_one("#setup-provider-endpoint", Input)
                restored_endpoint = ui_draft.endpoint if endpoint_visible else ""
                if endpoint_input.value != restored_endpoint:
                    self._pending_programmatic_endpoint_changes.append(
                        (provider_key, restored_endpoint)
                    )
                    endpoint_input.value = restored_endpoint
                auth.display = True
                auth.remove_class("hidden")
                optional_auth = not self._provider_requires_api_key(provider_key)
                auth.title = (
                    "Authentication (optional)" if optional_auth else "Authentication"
                )
                auth.collapsed = ui_draft.auth_collapsed
            finally:
                self._updating_connection_controls = False
            self._refresh_endpoint_resolution()
        if ui_draft.clear_requested:
            status.update(
                "No API key will be used for chat. The stored key will be removed "
                "when you continue."
            )
            actions.remove_class("hidden")
        elif ui_draft.api_key:
            status.update("A replacement API key is ready for this provider.")
            actions.remove_class("hidden")
        elif presence.inline_configured:
            status.update("An API key is already configured for this provider.")
            actions.remove_class("hidden")
        elif presence.env_var_set:
            status.update(
                f"Found {presence.env_var} in your environment ✓ — nothing to store."
            )
            actions.add_class("hidden")
        else:
            status.update("")
            actions.add_class("hidden")
        self.query_one("#setup-provider-probe-status", Static).update("")
        self._detected_servers = ui_draft.detected_servers
        if ui_draft.detected_servers:
            self._render_detection_results(ui_draft.detected_servers)
        if ui_draft.detected_server is not None:
            self._apply_discovered_server(ui_draft.detected_server)
        self._refresh_auth_readiness()
        if provider_changed:
            self._begin_selected_provider_discovery(self._effective_provider_draft())

    def _select_provider_option(self, option: Option) -> None:
        provider_key = getattr(option, "provider_key", None)
        if (
            provider_key is not None
            and not option.disabled
            and provider_key != self.selected_provider_key
        ):
            self.select_provider(provider_key)

    @on(OptionList.OptionHighlighted, "#setup-provider-choice")
    def _on_provider_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if self._provider_choice_interacted:
            self._select_provider_option(event.option)

    @on(ProviderChoiceList.Interacted)
    def _on_provider_list_interacted(self) -> None:
        self._provider_choice_interacted = True
        choices = self.query_one("#setup-provider-choice", ProviderChoiceList)
        highlighted = choices.highlighted_option
        if highlighted is not None:
            self._select_provider_option(highlighted)

    @on(OptionList.OptionSelected, "#setup-provider-choice")
    def _on_provider_chosen(self, event: OptionList.OptionSelected) -> None:
        self._provider_choice_interacted = True
        self._select_provider_option(event.option)

    def _effective_provider_key(self) -> str:
        """Return the selected key, falling back to the highlighted option."""
        if self.selected_provider_key:
            return self.selected_provider_key
        if not self._provider_choice_interacted:
            return ""
        try:
            highlighted = self.query_one(
                "#setup-provider-choice", OptionList
            ).highlighted_option
        except Exception:
            return ""
        if highlighted is None or highlighted.disabled:
            return ""
        return getattr(highlighted, "provider_key", None) or ""

    @on(Button.Pressed, "#setup-provider-key-replace")
    def _on_replace(self) -> None:
        """Reveal the masked input so the user can type a new key.

        Leaving it blank after Replace is a cancel: commit() only persists a
        typed, non-empty value, so the currently-configured secret is left
        untouched (never re-shown).
        """
        key_input = self.query_one("#setup-provider-api-key", Input)
        changed = self._clear_requested or not key_input.display
        self._clear_requested = False
        key_input.display = True
        if changed:
            self._credential_semantics_changed()

    @on(Button.Pressed, "#setup-provider-key-keep")
    def _on_keep(self) -> None:
        """Abandon any in-progress Replace/Clear; the stored secret is untouched."""
        key_input = self.query_one("#setup-provider-api-key", Input)
        changed = self._clear_requested or bool(key_input.value) or key_input.display
        self._clear_requested = False
        with key_input.prevent(Input.Changed):
            key_input.value = ""
        key_input.display = False
        if changed:
            self._credential_semantics_changed()

    @on(Button.Pressed, "#setup-provider-key-clear")
    def _on_clear(self) -> None:
        """Mark the configured secret for removal on commit.

        Unlike Replace, leaving the field blank here is the whole point: it
        signals commit() to persist an explicit empty api_key rather than
        leaving the existing one in place (build_provider_commit's truthiness
        check would otherwise treat "" exactly like "nothing to write").
        """
        key_input = self.query_one("#setup-provider-api-key", Input)
        changed = (
            not self._clear_requested or bool(key_input.value) or not key_input.display
        )
        self._clear_requested = True
        with key_input.prevent(Input.Changed):
            key_input.value = ""
        key_input.display = True
        self.query_one("#setup-provider-key-status", Static).update(
            "No API key will be used for chat. The stored key will be removed "
            "when you continue."
        )
        if changed:
            self._credential_semantics_changed()

    @on(Input.Changed, "#setup-provider-api-key")
    def _on_key_changed(self, event: Input.Changed) -> None:
        if self._updating_connection_controls:
            return
        captured = self._provider_drafts.get(self.selected_provider_key)
        if (
            captured is not None
            and captured.api_key == event.value
            and captured.credential_revision == self._credential_revision
        ):
            return
        self._clear_requested = False
        self._credential_semantics_changed()

    @on(Input.Submitted, "#setup-provider-api-key")
    def _on_key_submitted(self, event: Input.Submitted) -> None:
        """Live-but-never-blocking verification: probe on Enter in the key field."""
        if event.value.strip():
            self._launch_probe()

    @on(Button.Pressed, "#setup-provider-test")
    def _on_test_pressed(self, event: Button.Pressed) -> None:
        """TASK-1506: same probe as Enter-in-field, behind a visible control."""
        event.stop()
        self._launch_probe()

    def _launch_probe(self, *, api_key: str | None = None) -> None:
        del api_key
        self._sync_live_credential_revision()
        generation = self._obsolete_provider_generation(
            "setup-provider-discovery",
            "setup-provider-probe",
        )
        provider_key = self.selected_provider_key
        readiness = self._current_provider_readiness()
        if not readiness.ready:
            self.query_one("#setup-provider-probe-status", Static).update(
                readiness.recovery or "An API key is required before testing."
            )
            return
        target = self._probe_target()
        identity = self._provider_current_draft_identity()
        if not target or identity is None:
            self.query_one("#setup-provider-probe-status", Static).update(
                "Enter a valid endpoint before testing."
            )
            return
        credential_source, credential_value, _ = self._credential_at_request_boundary()
        token = self._provider_test_evidence.begin(identity)
        self._active_probe_token = token
        self.query_one("#setup-provider-probe-status", Static).update("Testing…")
        self.run_worker(
            partial(
                self._run_probe,
                generation,
                token,
                identity,
                provider_key=provider_key,
                endpoint=target,
                credential_source=credential_source,
                credential_value=credential_value,
            ),
            exclusive=True,
            group="setup-provider-probe",
        )

    async def _run_probe(
        self,
        generation: int,
        token: object,
        identity: object,
        *,
        provider_key: str,
        endpoint: str,
        credential_source: str,
        credential_value: str | None,
    ) -> None:
        from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
            SettingsEndpointProbeOutcome,
        )

        try:
            outcome = await self._probe(
                endpoint,
                provider=provider_key,
                credential_source=credential_source,
                credential_value=credential_value,
            )
            result = self._provider_probe_result_from_outcome(outcome)
        except asyncio.CancelledError:
            if self._provider_test_evidence.cancel_probe(token):
                if self._active_probe_token is token:
                    self._active_probe_token = None
                if generation == self.probe_generation and self.is_mounted:
                    self.query_one("#setup-provider-probe-status", Static).update("")
            raise
        except Exception as exc:
            logger.debug(
                "Wizard provider probe failed (error_type={})",
                type(exc).__name__,
            )
            outcome = SettingsEndpointProbeOutcome(
                state="unreachable",
                category="connection_error",
                summary="Probe errored.",
            )
            result = self._provider_probe_result_from_outcome(outcome)
        settled = self._provider_test_evidence.settle(token, result)
        if not settled:
            return
        if self._active_probe_token is token:
            self._active_probe_token = None
        self._last_tested_provider_identity = identity
        if (
            generation == self.probe_generation
            and provider_key == self.selected_provider_key
            and self.is_mounted
        ):
            self._render_provider_evidence(identity)

    @staticmethod
    def _provider_probe_result_from_outcome(outcome: object):
        """Convert only the exact shared Settings outcome to exact evidence."""

        from tldw_chatbook.Chat.provider_test_evidence import ProviderProbeResult
        from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
            SettingsEndpointProbeOutcome,
        )

        if type(outcome) is not SettingsEndpointProbeOutcome:
            raise ValueError("Provider probe outcome is invalid.")
        state = str(outcome.state)
        if state == "reachable":
            return ProviderProbeResult("reachable", outcome.model_ids)
        if state == "model_listing_unavailable":
            return ProviderProbeResult(
                "model_listing_unavailable", (), outcome.category
            )
        if state == "unreachable":
            return ProviderProbeResult("unreachable", (), outcome.category)
        raise ValueError("Provider probe outcome is invalid.")

    def _render_provider_evidence(self, identity: object) -> None:
        from tldw_chatbook.Chat.provider_test_evidence import (
            ProviderDraftIdentity,
            ProviderReadinessSnapshot,
            provider_readiness_verdict,
        )

        if type(identity) is not ProviderDraftIdentity:
            return
        evidence = self._provider_test_evidence.evidence_for(identity)
        if evidence is None:
            return
        snapshot = ProviderReadinessSnapshot(
            configuration="configured",
            endpoint=evidence.endpoint,
            model="unconfirmed",
            category=evidence.category,
        )
        verdict = provider_readiness_verdict(snapshot)
        prefix = (
            "✓ "
            if evidence.endpoint == "reachable"
            else ("✗ " if evidence.endpoint == "unreachable" else "")
        )
        self.query_one("#setup-provider-probe-status", Static).update(
            f"{prefix}{verdict.detail}"
        )

    @staticmethod
    def _cloud_probe_base_url(provider_key: str) -> str:
        """OpenAI-compatible base URLs for cloud-key verification (v1 fence:
        only providers with a known compatible /v1/models endpoint)."""
        return {
            "openai": "https://api.openai.com",
            "openrouter": "https://openrouter.ai/api",
            "groq": "https://api.groq.com/openai",
            "deepseek": "https://api.deepseek.com",
            "mistral": "https://api.mistral.ai",
            "mistralai": "https://api.mistral.ai",
        }.get(provider_key, "")

    def apply_probe_result(
        self,
        generation: int,
        *,
        reachable: bool,
        summary: str,
        provider_key: str | None = None,
    ) -> None:
        """Render a probe outcome only if it is still current (no stale ✓)."""
        if generation != self.probe_generation or (
            provider_key is not None and provider_key != self.selected_provider_key
        ):
            return
        del summary
        prefix = "✓ " if reachable else "✗ "
        detail = (
            "Model listing reached."
            if reachable
            else "The model listing endpoint could not be reached."
        )
        self.query_one("#setup-provider-probe-status", Static).update(
            f"{prefix}{detail}"
        )

    async def commit(self) -> tuple[bool, str]:
        provider_key = self._effective_provider_key()
        if not provider_key:
            return True, ""  # legitimately nothing pressed -- skip is correct
        self.selected_provider_key = provider_key
        key_input = self.query_one("#setup-provider-api-key", Input)
        captured = self._provider_drafts.get(provider_key)
        if captured is None or captured.api_key != key_input.value:
            self._clear_requested = False
            self._credential_semantics_changed()
        readiness = self._current_provider_readiness()
        if not readiness.ready:
            recovery = readiness.recovery or "Add a provider credential."
            return False, f"API key required. {recovery}"
        typed_key = bool(
            key_input.display and key_input.value and key_input.value.strip()
        )
        self.provider_value_for_chat_defaults = self._display_value_for(
            self.selected_provider_key
        )
        credential_decision = self._credential_decision()
        revision = self._credential_revision
        provider_draft = self._effective_provider_draft(revision=revision)
        if provider_draft is None:
            return False, "The provider settings are invalid."
        stage = getattr(self.wizard, "stage_provider_setup", None)
        if not callable(stage) or not stage(provider_draft):
            return False, "Staging the provider settings failed."
        self._credential_revision = revision
        self._last_credential_decision = credential_decision
        discovery_key = self._model_discovery_key(provider_draft)
        if discovery_key != self._selected_discovery_key or (
            self._selected_discovery_state not in {"in_progress", "complete"}
        ):
            self._begin_selected_provider_discovery(provider_draft)
        self._last_committed_provider_value = self.provider_value_for_chat_defaults
        if typed_key:
            self._entered_key = True
            self.wizard.note_key_entered()
        return True, ""

    @staticmethod
    def _display_value_for(provider_key: str) -> str:
        # chat_screen._apply_detected_local_server (line ~9137) persists the
        # RAW provider_key into chat_defaults["provider"] (e.g. "llama_cpp",
        # "openai") — not a human display name. Mirror that exact string
        # form here so this step's commit and the live Console apply path
        # never disagree about what chat_defaults.provider means.
        return provider_key

    def get_step_data(self) -> Dict[str, Any]:
        return {
            "provider_key": self.selected_provider_key,
            "provider_value": self.provider_value_for_chat_defaults,
            "entered_key": self._entered_key,
        }


MODEL_DISCOVERY_TIMEOUT_SECONDS = 8.0


class ModelStep(SetupStep):
    """Pick a default model for the chosen provider.

    Model discovery tries the injectable scope service first (an 8s guard
    keeps a hanging/slow provider from blocking Next), then falls back to
    the curated ``[providers]`` table from config.toml. Whichever provider
    key form ProviderStep handed us (raw key or display name; see
    ``ProviderStep._display_value_for``), the curated lookup bridges both
    forms via ``first_run_setup_state.curated_models_for_provider`` so a
    case/format mismatch never silently empties the list.
    """

    def __init__(
        self,
        wizard=None,
        config=None,
        *,
        discover_models=None,
        provider_draft: wizard_state.FirstRunProviderDraft | None = None,
        **kwargs,
    ):
        super().__init__(wizard=wizard, config=config, **kwargs)
        if provider_draft is not None and (
            type(provider_draft) is not wizard_state.FirstRunProviderDraft
        ):
            raise TypeError("Model discovery requires FirstRunProviderDraft.")
        self._discover_models = discover_models
        self._explicit_provider_draft = provider_draft
        self._shown_for_provider: Optional[str] = None
        self._shown_for_discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = (
            None
        )
        self._selection_discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = (
            None
        )
        self._rendered_discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = (
            None
        )
        self._selection_config_precondition: object | None = None
        self._manual_decision_active = False
        self.selected_model_id: str = ""
        # Bug-5: tracks whether selected_model_id's current value came from
        # the free-text custom Input (as opposed to the RadioSet) -- lets
        # clearing that Input fall back to any active radio selection
        # instead of leaving a stale custom value in place.
        self._model_id_from_custom_input: bool = False
        self._model_load_generation = 0

    def invalidate_credential_bound_selection(self) -> None:
        """Drop model state derived under a credential that has rotated."""

        self.invalidate_discovery_bound_selection()

    def invalidate_discovery_bound_selection(self) -> None:
        """Drop a selection whose exact provider discovery identity changed."""

        self._model_load_generation += 1
        self._shown_for_discovery_key = None
        self._selection_discovery_key = None
        self._rendered_discovery_key = None
        self._selection_config_precondition = None
        self._manual_decision_active = False
        self.selected_model_id = ""
        self._model_id_from_custom_input = False
        if not self.is_mounted:
            return
        try:
            custom = self.query_one("#setup-model-custom", Input)
            with custom.prevent(Input.Changed):
                custom.value = ""
            self._clear_model_radio_selection()
        except Exception:
            return
        if self.is_active:
            self.on_show()

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-model"):
            yield Static("Pick a default model", classes="setup-title")
            yield Static("", id="setup-model-provider-line", classes="setup-subtitle")
            with RadioSet(id="setup-model-choice", classes="setup-choice-list"):
                # disabled=True: an un-disabled placeholder is a real,
                # toggleable RadioButton -- pressing Enter/Space while it is
                # the only/highlighted option (e.g. an impatient user, or
                # discovery that never resolves) would fire RadioSet.Changed
                # and commit the literal placeholder text as the model id
                # (see _on_model_chosen). Same reasoning applies to the two
                # other placeholders this step ever mounts, below.
                yield SetupRadioButton(
                    "(loading models…)", id="setup-model-loading", disabled=True
                )
            yield Label("Or enter a model name", classes="setup-field-label")
            yield Input(id="setup-model-custom", placeholder="model-id")
            yield Button(
                "Retry", id="setup-model-retry", variant="default", classes="hidden"
            )
            yield Static("", classes="setup-step-error")

    def _current_provider(self) -> tuple[str, str]:
        provider_draft = self._current_provider_draft()
        if provider_draft is not None:
            return provider_draft.provider, provider_draft.provider
        data = (self.wizard.wizard_data or {}).get(wizard_state.STEP_PROVIDER, {})
        provider_key = str(data.get("provider_key", ""))
        provider_value = str(data.get("provider_value", ""))
        if provider_key:
            return provider_key, provider_value
        return "", ""

    def _current_discovery_key(
        self,
    ) -> wizard_state.FirstRunModelDiscoveryKey | None:
        provider_draft = self._current_provider_draft()
        if provider_draft is None:
            return None
        try:
            return wizard_state.build_first_run_model_discovery_key(provider_draft)
        except ValueError:
            return None

    def _current_provider_draft(
        self,
    ) -> wizard_state.FirstRunProviderDraft | None:
        provider_draft = getattr(self.wizard, "staged_provider_draft", None)
        if type(provider_draft) is wizard_state.FirstRunProviderDraft:
            return provider_draft
        return self._explicit_provider_draft

    def on_show(self) -> None:
        super().on_show()
        self._model_load_generation += 1
        load_generation = self._model_load_generation
        provider_key, provider_value = self._current_provider()
        discovery_key = self._current_discovery_key()
        exact_key_changed = (
            discovery_key is not None and discovery_key != self._shown_for_discovery_key
        )
        provider_changed = provider_key != self._shown_for_provider
        if exact_key_changed or (discovery_key is None and provider_changed):
            # UI half of dependency invalidation: the config half (clearing
            # chat_defaults.model) already happened in ProviderStep.commit()
            # via invalidate_model_for_provider_change. This just keeps the
            # step's own in-memory selection from surviving a Back-and-switch.
            #
            # TASK-1374: re-run prefill from a genuinely reachable condition.
            # The old guard keyed on wizard_data lacking a provider entry --
            # unreachable, since _advance() always records one before Model
            # can be shown. The real re-run signal is the session provider
            # MATCHING the persisted chat_defaults.provider: same provider ->
            # surface the saved model; changed provider -> blank (the config
            # half of that invalidation already happened in ProviderStep).
            first_identity = (
                self._shown_for_provider is None
                and self._shown_for_discovery_key is None
            )
            prefill_model_id = (
                wizard_state.rerun_model_prefill(
                    getattr(self.wizard.app_instance, "app_config", {}) or {},
                    provider_value=provider_value,
                )
                if first_identity
                else ""
            )
            self.selected_model_id = prefill_model_id
            self._model_id_from_custom_input = False
            self._shown_for_provider = provider_key
            self._shown_for_discovery_key = discovery_key
            self._selection_discovery_key = discovery_key if prefill_model_id else None
            self._selection_config_precondition = (
                self._config_precondition_for_discovery(discovery_key)
                if prefill_model_id
                else None
            )
            self._manual_decision_active = False
            try:
                self.query_one("#setup-model-custom", Input).value = prefill_model_id
            except Exception:
                pass
        try:
            # TASK-1503: display-case the provider in user copy — raw keys
            # like "anthropic"/"llama_cpp" are internals, not UI language.
            from tldw_chatbook.Chat.provider_catalog import provider_display_name

            display = provider_display_name(provider_key) if provider_key else ""
            self.query_one("#setup-model-provider-line", Static).update(
                f"Models for {display or 'your provider'}."
            )
        except Exception:
            pass
        live_radios = tuple(self.query("#setup-model-choice RadioButton"))
        rendered_is_current = (
            discovery_key is not None
            and discovery_key == self._rendered_discovery_key
            and any(getattr(button, "_model_id", "") for button in live_radios)
        )
        if rendered_is_current:
            self._restore_model_radio_selection(discovery_key)
        if provider_key and not rendered_is_current:
            self.run_worker(
                partial(
                    self._load_models,
                    provider_key,
                    provider_value,
                    discovery_key,
                    load_generation,
                ),
                exclusive=True,
                group="setup-model-load",
            )
        elif not provider_key:
            # F-F fix: with no provider chosen yet there is nothing to
            # discover against, so the old code simply skipped this branch
            # and left the initial "(loading models…)" RadioButton in place
            # forever -- a permanently-stuck loading indicator for a state
            # that was never actually loading. Replace it with copy that
            # tells the user what to do instead.
            self.run_worker(
                partial(
                    self._render_models,
                    [],
                    no_provider=True,
                    discovery_key=discovery_key,
                    load_generation=load_generation,
                ),
                exclusive=True,
                group="setup-model-load",
            )

    def _cancel_model_discovery(self) -> None:
        self._model_load_generation += 1
        if self.is_attached:
            self.workers.cancel_group(self, "setup-model-load")

    def on_hide(self) -> None:
        super().on_hide()
        self._cancel_model_discovery()
        owner = getattr(self.wizard, "_first_run_provider_discovery_owner", None)
        if isinstance(owner, ProviderStep):
            owner.cancel_selected_discovery_handoff()
        self._explicit_provider_draft = None

    def on_unmount(self) -> None:
        self._cancel_model_discovery()
        owner = getattr(self.wizard, "_first_run_provider_discovery_owner", None)
        if isinstance(owner, ProviderStep):
            owner.cancel_selected_discovery_handoff()
        self._explicit_provider_draft = None
        self._selection_config_precondition = None
        self._manual_decision_active = False

    async def _load_models(
        self,
        provider_key: str,
        provider_value: str,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = None,
        load_generation: int | None = None,
    ) -> None:
        import asyncio

        models: list[str] = []
        discovery_state = "available"
        failure_category = ""
        await self._render_models(
            [],
            discovery_state="loading",
            discovery_key=discovery_key,
            load_generation=load_generation,
        )
        provider_draft = self._current_provider_draft()
        if provider_draft is None or discovery_key is None:
            await self._render_models(
                [],
                discovery_key=discovery_key,
                load_generation=load_generation,
            )
            return
        discover = self._discover_models
        owner = getattr(self.wizard, "_first_run_provider_discovery_owner", None)
        handed_off = getattr(self.wizard, "_first_run_selected_provider_models", {})
        handed_outcomes = getattr(
            self.wizard, "_first_run_selected_provider_outcomes", {}
        )
        if isinstance(handed_outcomes, Mapping) and discovery_key in handed_outcomes:
            models, discovery_state, failure_category = _model_discovery_ui_outcome(
                handed_outcomes[discovery_key]
            )
            discover = None
        elif isinstance(handed_off, Mapping) and discovery_key in handed_off:
            models = list(handed_off[discovery_key])
            if (
                not models
                and isinstance(owner, ProviderStep)
                and owner._selected_discovery_key == discovery_key
                and owner._selected_discovery_state == "failed"
            ):
                discovery_state = "connection_failed"
                failure_category = "request failed"
            discover = None
        elif (
            isinstance(owner, ProviderStep)
            and owner.is_mounted
            and owner.app is self.app
        ):
            try:
                selected_outcome = await asyncio.wait_for(
                    owner._outcome_from_selected_discovery(provider_key, discovery_key),
                    timeout=MODEL_DISCOVERY_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                owner.cancel_selected_discovery_handoff()
                selected_outcome = None
                discovery_state = "connection_failed"
                failure_category = "timeout"
            except Exception:
                selected_outcome = None
            if selected_outcome is not None:
                models, discovery_state, failure_category = _model_discovery_ui_outcome(
                    selected_outcome
                )
            elif (
                owner._selected_discovery_key == discovery_key
                and owner._selected_discovery_state == "failed"
            ):
                discovery_state = "connection_failed"
                failure_category = "request failed"
            # ProviderStep owns setup network work for this selection. If the
            # user advances before it finishes, use curated fallback rather
            # than issuing the same provider catalog request from ModelStep.
            discover = None
        elif isinstance(owner, ProviderStep):
            discover = None
        if isinstance(owner, ProviderStep):
            evidence = owner._test_evidence_for_discovery_key(discovery_key)
            if evidence is not None:
                if evidence.endpoint == "model_listing_unavailable":
                    discovery_state = "listing_unavailable"
                    models = []
                elif evidence.endpoint == "unreachable":
                    discovery_state = "connection_failed"
                    failure_category = (
                        evidence.category or "connection_error"
                    ).replace("_", " ")
                    models = []
                elif evidence.endpoint == "reachable" and not models:
                    models = list(evidence.model_ids)
        if discover is None:
            service = (
                None
                if isinstance(owner, ProviderStep)
                else getattr(
                    self.wizard.app_instance,
                    "llm_provider_catalog_scope_service",
                    None,
                )
            )
            if service is not None:

                async def discover(*, provider=provider_key, svc=service, **_identity):
                    return await svc.discover_models(
                        mode="local",
                        provider=provider,
                        staged_settings=_first_run_discovery_staged_settings(
                            provider_draft, discovery_key
                        ),
                        use_shared_cache=False,
                    )

        if discover is not None:
            try:
                result = await asyncio.wait_for(
                    discover(
                        provider=provider_key,
                        endpoint=discovery_key.connection_identity[1],
                        credential_source=discovery_key.credential_source,
                        credential_revision=discovery_key.credential_revision,
                    ),
                    timeout=MODEL_DISCOVERY_TIMEOUT_SECONDS,
                )
                try:
                    models, discovery_state, failure_category = (
                        _model_discovery_ui_outcome(result)
                    )
                except ValueError:
                    models = list(_legacy_model_ids(result))
            except TimeoutError:
                discovery_state = "connection_failed"
                failure_category = "timeout"
            except Exception:
                discovery_state = "connection_failed"
                failure_category = "connection error"
                logger.debug("Wizard model discovery failed", exc_info=True)
        if not models and discovery_state == "available":
            from tldw_chatbook.config import get_cli_providers_and_models

            models = wizard_state.curated_models_for_provider(
                get_cli_providers_and_models(), provider_value
            )
        await self._render_models(
            models[:20],
            discovery_state=discovery_state,
            failure_category=failure_category,
            discovery_key=discovery_key,
            load_generation=load_generation,
        )

    async def _render_models(
        self,
        models: list[str],
        *,
        no_provider: bool = False,
        discovery_state: str = "available",
        failure_category: str = "",
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = None,
        load_generation: int | None = None,
    ) -> None:
        if (
            load_generation is not None
            and load_generation != self._model_load_generation
        ):
            return
        if discovery_key != self._current_discovery_key():
            return
        try:
            radio_set = self.query_one("#setup-model-choice", RadioSet)
        except Exception:
            return
        # Textual does not clear these owner pointers when children are
        # removed. Reset them before rebuilding so a restored selection can
        # only reference one of the newly mounted rows.
        radio_set._pressed_button = None
        radio_set._selected = None
        # remove_children()/mount() are message-queue operations -- both
        # return awaitables that must be awaited before the DOM change is
        # actually applied. Without awaiting the removal, a second call (e.g.
        # a provider switch that fires before the first discovery settles)
        # can try to mount fresh "setup-model-option-N" ids while the stale
        # ones are still present, raising DuplicateIds.
        await radio_set.remove_children()
        if (
            load_generation is not None
            and load_generation != self._model_load_generation
        ):
            return
        if discovery_key != self._current_discovery_key():
            return
        if models:
            # TASK-1503: the first entry (curated-default / top discovery hit)
            # carries a "recommended" tag in its LABEL only; the clean model
            # id lives on the button as `_model_id` so selection and commits
            # never round-trip display decoration into config.
            def _button(index: int, model_id: str) -> SetupRadioButton:
                label = f"{model_id}   (recommended)" if index == 0 else model_id
                button = SetupRadioButton(label, id=f"setup-model-option-{index}")
                button._model_id = model_id
                button._discovery_key = discovery_key
                return button

            await radio_set.mount_all(
                _button(index, model_id) for index, model_id in enumerate(models)
            )
            selected = (
                self.selected_model_id
                if not self._model_id_from_custom_input
                and self._selection_discovery_key == discovery_key
                else ""
            )
            if selected:
                self._restore_model_radio_selection(discovery_key)
        elif discovery_state == "listing_unavailable":
            await radio_set.mount(
                SetupRadioButton(
                    "Model listing unavailable; enter the model ID used by this endpoint.",
                    id="setup-model-listing-unavailable",
                    disabled=True,
                )
            )
        elif discovery_state == "connection_failed":
            category = failure_category or "connection error"
            await radio_set.mount(
                SetupRadioButton(
                    f"Connection failed ({category}). Retry or enter a model ID below.",
                    id="setup-model-connection-failed",
                    disabled=True,
                )
            )
        elif discovery_state == "loading":
            await radio_set.mount(
                SetupRadioButton(
                    "(loading models…)",
                    id="setup-model-loading",
                    disabled=True,
                )
            )
        elif no_provider:
            await radio_set.mount(
                SetupRadioButton(
                    "Pick a provider first — or type a model name below",
                    id="setup-model-no-provider",
                    disabled=True,
                )
            )
        else:
            await radio_set.mount(
                SetupRadioButton(
                    "(no models found — enter one below)",
                    id="setup-model-empty",
                    disabled=True,
                )
            )
        if not self.is_attached:
            return
        try:
            retry = self.query_one("#setup-model-retry", Button)
            retry.set_class(discovery_state != "connection_failed", "hidden")
        except NoMatches:
            return
        self._rendered_discovery_key = discovery_key

    @on(Button.Pressed, "#setup-model-retry")
    def _retry_model_discovery(self, event: Button.Pressed) -> None:
        event.stop()
        if not self.is_active or self._current_discovery_key() is None:
            return
        event.button.add_class("hidden")
        self._rendered_discovery_key = None
        self._manual_decision_active = False
        if self._model_id_from_custom_input:
            self._selection_config_precondition = None
        owner = getattr(self.wizard, "_first_run_provider_discovery_owner", None)
        provider_draft = self._current_provider_draft()
        if isinstance(owner, ProviderStep) and provider_draft is not None:
            owner._begin_selected_provider_discovery(
                provider_draft,
                sync_live_credential=False,
            )
        self.on_show()

    @on(RadioSet.Changed, "#setup-model-choice")
    def _on_model_chosen(self, event: RadioSet.Changed) -> None:
        if event.pressed is not None:
            self.set_selected_model_from_button(event.pressed)

    def set_selected_model_from_button(self, button: RadioButton) -> None:
        """Select via a radio row, reading the clean id, not the label.

        TASK-1503: labels may carry display decoration ("(recommended)");
        the undecorated model id is stored on the button as ``_model_id``.

        Args:
            button: The pressed radio row. Only its clean ``_model_id``
                attribute can supply a model id; status labels are ignored.
        """
        model_id = getattr(button, "_model_id", None)
        if not isinstance(model_id, str) or not model_id:
            return
        try:
            custom_input = self.query_one("#setup-model-custom", Input)
            with custom_input.prevent(Input.Changed):
                custom_input.value = ""
        except Exception:
            pass
        self.set_selected_model(
            model_id,
            discovery_key=getattr(button, "_discovery_key", None),
        )

    def _clear_model_radio_selection(self) -> None:
        """Clear Textual's radio value and owner pointer without event races."""

        try:
            radio_set = self.query_one("#setup-model-choice", RadioSet)
        except Exception:
            return
        pressed = radio_set.pressed_button
        if pressed is None:
            return
        with (
            radio_set.prevent(RadioButton.Changed),
            pressed.prevent(RadioButton.Changed),
        ):
            radio_set._pressed_button = None
            pressed.value = False

    def _restore_model_radio_selection(
        self,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None,
    ) -> None:
        """Point RadioSet state only at the live row for this selection."""

        if (
            self._model_id_from_custom_input
            or self._selection_discovery_key != discovery_key
        ):
            return
        try:
            radio_set = self.query_one("#setup-model-choice", RadioSet)
        except Exception:
            return
        buttons = list(radio_set.query(RadioButton))
        selected = next(
            (
                button
                for button in buttons
                if getattr(button, "_model_id", "") == self.selected_model_id
            ),
            None,
        )
        with radio_set.prevent(RadioButton.Changed):
            for button in buttons:
                button.value = button is selected
        radio_set._pressed_button = selected
        radio_set._selected = buttons.index(selected) if selected is not None else None

    @on(Input.Changed, "#setup-model-custom")
    def _on_custom_model(self, event: Input.Changed) -> None:
        """Bug-5 fix: clearing the custom Input must clear the selection too.

        The old handler only ever ASSIGNED on a non-empty value, so
        clearing a previously-typed custom model left ``selected_model_id``
        stuck at the last typed value -- a "skip-safe" commit would then
        silently persist a model the input no longer shows. On empty, fall
        back to whatever radio button is currently pressed (or "" if none),
        rather than just blanking unconditionally.
        """
        previous_model = self.selected_model_id
        value = event.value.strip()
        if value:
            current_key = self._current_discovery_key()
            self._clear_model_radio_selection()
            self.selected_model_id = value
            self._model_id_from_custom_input = True
            if (
                not self._manual_decision_active
                or self._selection_discovery_key != current_key
            ):
                self._manual_decision_active = True
                self._selection_config_precondition = (
                    self._capture_current_config_precondition()
                )
            self._selection_discovery_key = current_key
        elif self._model_id_from_custom_input:
            self._model_id_from_custom_input = False
            self._manual_decision_active = False
            pressed = self._live_pressed_radio()
            # TASK-1503: clean id, never the (possibly decorated) label.
            self.selected_model_id = (
                str(getattr(pressed, "_model_id", pressed.label))
                if pressed is not None
                else ""
            )
            self._selection_discovery_key = (
                self._current_discovery_key() if self.selected_model_id else None
            )
            self._selection_config_precondition = (
                self._config_precondition_for_discovery(self._selection_discovery_key)
                if self.selected_model_id
                else None
            )
        if self.selected_model_id != previous_model:
            self._notify_provider_model_changed()

    def set_selected_model(
        self,
        model_id: str,
        *,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = None,
    ) -> None:
        changed = model_id != self.selected_model_id
        self.selected_model_id = model_id
        self._model_id_from_custom_input = False
        self._manual_decision_active = False
        current_key = self._current_discovery_key()
        self._selection_discovery_key = (
            discovery_key
            if model_id and discovery_key == current_key
            else current_key
            if model_id
            else None
        )
        self._selection_config_precondition = (
            self._config_precondition_for_discovery(self._selection_discovery_key)
            if model_id
            else None
        )
        if changed:
            self._notify_provider_model_changed(
                model_id=model_id,
                discovery_key=discovery_key,
            )

    def _notify_provider_model_changed(
        self,
        *,
        model_id: str = "",
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = None,
    ) -> None:
        invalidate_save = getattr(
            self.wizard, "invalidate_provider_write_expectation", None
        )
        if callable(invalidate_save):
            invalidate_save()
        owner = getattr(self.wizard, "_first_run_provider_discovery_owner", None)
        if isinstance(owner, ProviderStep):
            owner._model_semantics_changed(
                model_id=model_id,
                discovery_key=discovery_key,
            )

    def _config_precondition_for_discovery(
        self,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None,
    ) -> object | None:
        preconditions = getattr(
            self.wizard, "_first_run_provider_config_preconditions", {}
        )
        if discovery_key is None or not isinstance(preconditions, Mapping):
            return None
        return preconditions.get(discovery_key)

    def _capture_current_config_precondition(self) -> object | None:
        discovery_key = self._current_discovery_key()
        capture = getattr(self.wizard, "capture_provider_config_precondition", None)
        if discovery_key is None or not callable(capture):
            return None
        return capture(discovery_key)

    def _live_pressed_radio(self) -> Optional[RadioButton]:
        """F1 fix: read ``#setup-model-choice``'s ``pressed_button``, but only
        if it is still one of the RadioSet's *current* children.

        Textual's ``RadioSet._pressed_button`` (``textual/widgets/_radio_set.py``)
        is a plain instance attribute; ``remove_children()`` prunes DOM
        children but never touches it. ``_render_models`` calls
        ``remove_children()``/``mount_all()`` on every provider switch to
        swap in the new provider's models, so a RadioButton pressed under
        the OLD provider stays referenced by ``_pressed_button`` -- now
        pointing at a detached, no-longer-mounted widget -- until the user
        presses something in the NEW list. Reading ``pressed_button``
        unguarded after a provider switch (Back -> switch provider -> Next)
        therefore resurrects the previous provider's model id even though
        nothing in the currently-visible list was ever pressed. Guarding
        with membership in ``radio_set.query(RadioButton)`` (the set's
        live, currently-mounted children) closes that window without
        reaching into ``_pressed_button`` from application code.
        """
        try:
            radio_set = self.query_one("#setup-model-choice", RadioSet)
        except Exception:
            return None
        pressed = radio_set.pressed_button
        if pressed is None or pressed not in radio_set.query(RadioButton):
            return None
        discovery_key = self._current_discovery_key()
        if discovery_key is not None and (
            getattr(pressed, "_discovery_key", None) != discovery_key
        ):
            return None
        return pressed

    def _effective_model_id(self) -> str:
        """F-A fix: fall back to the RadioSet's own ``pressed_button`` when
        this step's own bookkeeping (``selected_model_id``, updated only by
        ``_on_model_chosen``/``_on_custom_model``) has nothing -- same
        reasoning as ``ProviderStep._effective_provider_key``. The three
        placeholder rows this step ever mounts (loading / no-provider /
        no-models-found) are all ``disabled=True`` and so can never actually
        become ``pressed_button``.

        F1 fix: the fallback goes through ``_live_pressed_radio()`` rather
        than reading ``pressed_button`` directly, so a stale press left over
        from a provider switch (see ``_live_pressed_radio``'s docstring)
        cannot resurrect the previous provider's model at commit time.
        """
        discovery_key = self._current_discovery_key()
        if self.selected_model_id and (
            discovery_key is None or self._selection_discovery_key == discovery_key
        ):
            return self.selected_model_id
        pressed = self._live_pressed_radio()
        if pressed is None:
            return ""
        # TASK-1503: read the clean id, never the (possibly decorated) label.
        return str(getattr(pressed, "_model_id", pressed.label))

    async def commit(self) -> tuple[bool, str]:
        _, provider_value = self._current_provider()
        model_id = self._effective_model_id()
        if not (provider_value and model_id):
            return True, ""  # skip-safe
        commit_staged = getattr(self.wizard, "commit_staged_provider_setup", None)
        if not callable(commit_staged):
            return False, "Return to Provider and review the connection."
        selection_key = self._selection_discovery_key
        if selection_key is None:
            pressed = self._live_pressed_radio()
            selection_key = getattr(pressed, "_discovery_key", None)
        provenance: Literal["discovered", "manual"] = (
            "manual" if self._model_id_from_custom_input else "discovered"
        )
        can_validate = getattr(
            self.wizard,
            "can_validate_committed_provider_setup",
            None,
        )
        if (
            callable(getattr(self.wizard, "capture_provider_config_precondition", None))
            and self._selection_config_precondition is None
            and not (
                callable(can_validate)
                and can_validate(
                    model_id,
                    discovery_key=selection_key,
                    model_provenance=provenance,
                )
            )
        ):
            return (
                False,
                (
                    "Connection settings changed. Refresh models or re-enter the "
                    "model ID."
                ),
            )
        commit_kwargs = {
            "discovery_key": selection_key,
            "model_provenance": provenance,
        }
        if self._selection_config_precondition is not None:
            commit_kwargs["config_precondition"] = self._selection_config_precondition
        ok = await commit_staged(model_id, **commit_kwargs)
        if ok:
            self.selected_model_id = model_id
            self._selection_discovery_key = self._current_discovery_key()
            if provenance == "manual":
                self._manual_decision_active = False
                self._selection_config_precondition = None
        elif (
            getattr(
                getattr(self.wizard, "_provider_last_config_result", None),
                "conflict_reason",
                None,
            )
            == "identity_changed"
            or selection_key != self._current_discovery_key()
        ):
            return (
                False,
                "Connection settings changed. Models were refreshed; select a "
                "model again or re-enter its ID.",
            )
        return (
            (True, "") if ok else (False, "Saving the provider and model setup failed.")
        )

    def get_step_data(self) -> Dict[str, Any]:
        return {"model_id": self._effective_model_id()}


class VoiceSetupStep(SetupStep):
    """Compact OpenAI-compatible TTS setup shared by Quick and Full tracks."""

    _SAVE_TIMEOUT_SECONDS = 30.0

    def __init__(self, wizard=None, config=None, **kwargs: Any) -> None:
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._preset = voice_state.VOICE_PRESET_POCKET_TTS
        self._custom_draft: voice_state.VoiceSetupDraft | None = None
        self._verified_draft: voice_state.VoiceSetupDraft | None = None
        self._next_save_request_id = 1
        self._save_request_id: int | None = None
        self._save_draft: voice_state.VoiceSetupDraft | None = None
        self._save_future: asyncio.Future[tuple[bool, str]] | None = None
        self._test_generation = 0
        self._test_in_progress_generation: int | None = None
        self._sample_audio_path: Path | None = None

    @staticmethod
    def _initial_draft() -> voice_state.VoiceSetupDraft:
        return voice_state.VoiceSetupDraft(
            endpoint=voice_state.POCKET_TTS_ENDPOINT,
            authentication_mode="none",
            model_id=voice_state.POCKET_TTS_MODEL,
            voice_id=voice_state.POCKET_TTS_VOICE,
            response_format="wav",
            speed=1.0,
            sample_text="Hello from Chatbook.",
            use_as_default=False,
        )

    def compose_step(self) -> ComposeResult:
        draft = self._initial_draft()
        with Vertical(classes="setup-voice"):
            yield Static("Set up a voice", classes="setup-title")
            yield Label("Service", classes="setup-field-label")
            with RadioSet(id="setup-voice-preset", classes="setup-voice-segmented"):
                yield SetupRadioButton(
                    "PocketTTS",
                    id="setup-voice-preset-pocket",
                    value=True,
                )
                yield SetupRadioButton(
                    "Official OpenAI",
                    id="setup-voice-preset-official",
                )
                yield SetupRadioButton(
                    "Custom compatible",
                    id="setup-voice-preset-custom",
                )
            yield Label("Endpoint", classes="setup-field-label")
            yield Input(
                value=draft.endpoint,
                id="setup-voice-endpoint",
                placeholder="http://127.0.0.1:8765/v1/audio/speech",
            )
            yield Label("Authentication", classes="setup-field-label")
            with RadioSet(id="setup-voice-auth", classes="setup-voice-segmented"):
                yield SetupRadioButton(
                    "None",
                    id="setup-voice-auth-none",
                    value=True,
                )
                yield SetupRadioButton("API key", id="setup-voice-auth-key")
            yield Label("Model", classes="setup-field-label")
            yield Input(value=draft.model_id, id="setup-voice-model")
            yield Label("Voice", classes="setup-field-label")
            yield Input(value=draft.voice_id, id="setup-voice-voice")
            with Horizontal(classes="setup-voice-output-row"):
                with Vertical():
                    yield Label("Format", classes="setup-field-label")
                    yield Input(value=draft.response_format, id="setup-voice-format")
                with Vertical():
                    yield Label("Speed", classes="setup-field-label")
                    yield Input(value=str(draft.speed), id="setup-voice-speed")
            yield Label("Sample text", classes="setup-field-label")
            yield Input(
                value=draft.sample_text,
                id="setup-voice-sample",
                max_length=500,
            )
            yield Static(
                f"{len(draft.sample_text)} / 500",
                id="setup-voice-sample-count",
                classes="setup-field-help",
            )
            yield Button(
                "Test and Hear",
                id="setup-voice-test",
                variant="primary",
            )
            yield Static(
                "Needs test. You can save this configuration while offline.",
                id="setup-voice-status",
                classes="setup-subtitle",
            )
            add_key = Button(
                "Add API key in Settings",
                id="setup-voice-add-key",
            )
            add_key.display = False
            yield add_key
            yield Checkbox(
                "Use as default",
                id="setup-voice-default",
                value=False,
            )
            yield Static("", classes="setup-step-error")

    def _selected_authentication(self) -> str:
        pressed = self.query_one("#setup-voice-auth", RadioSet).pressed_button
        return (
            "api_key"
            if pressed is not None and pressed.id == "setup-voice-auth-key"
            else "none"
        )

    def _draft_from_controls(self) -> voice_state.VoiceSetupDraft:
        try:
            speed = float(self.query_one("#setup-voice-speed", Input).value)
        except ValueError as error:
            raise ValueError("Speed must be a number between 0.25 and 4.0.") from error
        return voice_state.VoiceSetupDraft(
            endpoint=self.query_one("#setup-voice-endpoint", Input).value,
            authentication_mode=self._selected_authentication(),
            model_id=self.query_one("#setup-voice-model", Input).value,
            voice_id=self.query_one("#setup-voice-voice", Input).value,
            response_format=self.query_one("#setup-voice-format", Input)
            .value.strip()
            .lower(),
            speed=speed,
            sample_text=self.query_one("#setup-voice-sample", Input).value,
            use_as_default=self.query_one("#setup-voice-default", Checkbox).value,
        )

    def _apply_draft_to_controls(self, draft: voice_state.VoiceSetupDraft) -> None:
        self.query_one("#setup-voice-endpoint", Input).value = draft.endpoint
        self.query_one("#setup-voice-model", Input).value = draft.model_id
        self.query_one("#setup-voice-voice", Input).value = draft.voice_id
        self.query_one("#setup-voice-format", Input).value = draft.response_format
        self.query_one("#setup-voice-speed", Input).value = str(draft.speed)
        self.query_one("#setup-voice-sample", Input).value = draft.sample_text
        self.query_one("#setup-voice-default", Checkbox).value = draft.use_as_default
        auth_id = (
            "setup-voice-auth-none"
            if draft.authentication_mode == "none"
            else "setup-voice-auth-key"
        )
        restore_selection = getattr(self.wizard, "_restore_radio_selection", None)
        if callable(restore_selection):
            restore_selection(
                self.query_one("#setup-voice-auth", RadioSet),
                lambda button: button.id == auth_id,
            )
        else:
            self._set_radio(auth_id)
        self._refresh_sample_state()

    def _set_radio(self, button_id: str) -> None:
        radio_set = self.query_one(f"#{button_id}", RadioButton).parent
        if not isinstance(radio_set, RadioSet):
            return
        buttons = list(radio_set.query(RadioButton))
        selected = next((button for button in buttons if button.id == button_id), None)
        with radio_set.prevent(RadioButton.Changed):
            for button in buttons:
                button.value = button is selected
        radio_set._pressed_button = selected
        radio_set._selected = buttons.index(selected) if selected is not None else None

    @on(RadioSet.Changed, "#setup-voice-preset")
    def _on_preset(self, event: RadioSet.Changed) -> None:
        if event.pressed is None:
            return
        preset = {
            "setup-voice-preset-pocket": voice_state.VOICE_PRESET_POCKET_TTS,
            "setup-voice-preset-official": voice_state.VOICE_PRESET_OFFICIAL_OPENAI,
            "setup-voice-preset-custom": voice_state.VOICE_PRESET_CUSTOM,
        }.get(event.pressed.id)
        if preset is None or preset == self._preset:
            return
        try:
            current = self._draft_from_controls()
        except (TypeError, ValueError):
            self.query_one(".setup-step-error", Static).update(
                "Enter a valid speed before changing the service preset."
            )
            return
        if self._preset == voice_state.VOICE_PRESET_CUSTOM:
            self._custom_draft = current
        self._preset = preset
        base = (
            self._custom_draft
            if preset == voice_state.VOICE_PRESET_CUSTOM
            and self._custom_draft is not None
            else current
        )
        self._apply_draft_to_controls(voice_state.apply_voice_preset(base, preset))

    @on(Input.Changed, "#setup-voice-sample")
    def _on_sample_changed(self) -> None:
        self._invalidate_sample_evidence()
        self._refresh_sample_state()

    @on(Input.Changed)
    def _on_voice_input_changed(self, event: Input.Changed) -> None:
        if (
            event.input.id
            and event.input.id.startswith("setup-voice-")
            and event.input.id != "setup-voice-sample"
        ):
            self._invalidate_sample_evidence()

    @on(RadioSet.Changed, "#setup-voice-auth")
    def _on_authentication_changed(self) -> None:
        self._invalidate_sample_evidence()

    def _invalidate_sample_evidence(self) -> None:
        self._test_generation += 1
        self._test_in_progress_generation = None
        self._verified_draft = None
        try:
            self.workers.cancel_group(self, "setup-voice-sample")
        except Exception:
            pass
        try:
            self.query_one("#setup-voice-status", Static).update(
                "Needs test. You can save this configuration while offline."
            )
        except Exception:
            pass
        self._refresh_sample_state()

    @staticmethod
    def _sample_identity(draft: voice_state.VoiceSetupDraft) -> tuple[object, ...]:
        """Return only fields that affect the sample request."""

        return (
            draft.endpoint,
            draft.authentication_mode,
            draft.model_id,
            draft.voice_id,
            draft.response_format,
            draft.speed,
            draft.sample_text,
        )

    def _refresh_sample_state(self) -> None:
        try:
            sample = self.query_one("#setup-voice-sample", Input).value
            trimmed_count = len(sample.strip())
            self.query_one("#setup-voice-sample-count", Static).update(
                f"{trimmed_count} / 500"
            )
            try:
                draft = self._draft_from_controls()
                valid = voice_state.validate_voice_setup_draft(
                    draft
                ).configuration_valid
            except (TypeError, ValueError):
                valid = False
                draft = None
            missing_key = (
                draft is not None
                and draft.authentication_mode == "api_key"
                and self._existing_openai_credential() is None
            )
            self.query_one("#setup-voice-add-key", Button).display = missing_key
            self.query_one("#setup-voice-test", Button).disabled = (
                self._test_in_progress_generation is not None
                or not valid
                or missing_key
            )
            status = self.query_one("#setup-voice-status", Static)
            status_text = str(status.renderable)
            if missing_key and self._test_in_progress_generation is None:
                self._verified_draft = None
                status.update(
                    "API key required. Add an API key in Settings to test or save."
                )
            elif (
                not missing_key
                and status_text.startswith("API key required")
                and self._test_in_progress_generation is None
            ):
                status.update(
                    "Needs test. You can save this configuration while offline."
                )
        except Exception:
            return

    def on_show(self) -> None:
        super().on_show()
        self._refresh_sample_state()

    def _cancel_active_sample(self) -> None:
        if self._test_in_progress_generation is None:
            self._refresh_sample_state()
            return
        self._test_generation += 1
        self._test_in_progress_generation = None
        self._verified_draft = None
        try:
            self.workers.cancel_group(self, "setup-voice-sample")
        except Exception:
            pass
        try:
            self.query_one("#setup-voice-status", Static).update(
                "Needs test. The sample was cancelled; retry when ready."
            )
        except Exception:
            pass
        self._refresh_sample_state()

    def on_hide(self) -> None:
        super().on_hide()
        self._cancel_active_sample()

    def on_unmount(self) -> None:
        self._cancel_active_sample()
        if self._sample_audio_path is not None:
            self._sample_audio_path.unlink(missing_ok=True)
            self._sample_audio_path = None

    @on(Button.Pressed, "#setup-voice-test")
    def _on_test_and_hear(self) -> None:
        try:
            draft = self._draft_from_controls()
        except (TypeError, ValueError):
            return
        if not voice_state.validate_voice_setup_draft(draft).configuration_valid:
            return
        self._test_generation += 1
        generation = self._test_generation
        self._test_in_progress_generation = generation
        self.query_one("#setup-voice-status", Static).update("Testing voice…")
        self._refresh_sample_state()
        self.run_worker(
            self._run_voice_sample(generation, draft),
            exclusive=True,
            group="setup-voice-sample",
            exit_on_error=False,
        )

    @on(Button.Pressed, "#setup-voice-add-key")
    def _on_add_api_key(self) -> None:
        callback = getattr(self.wizard, "open_voice_api_key_settings", None)
        if not callable(callback):
            self.query_one("#setup-voice-status", Static).update(
                "Open Settings, then Speech & TTS, to add the OpenAI API key."
            )
            return
        try:
            route = callback(self)
        except Exception:
            self.query_one("#setup-voice-status", Static).update(
                "Could not open Settings. Use Speech & TTS to add the API key."
            )
            return
        if not asyncio.iscoroutine(route):
            self.query_one("#setup-voice-status", Static).update(
                "Could not open Settings. Use Speech & TTS to add the API key."
            )
            return
        self.run_worker(
            route,
            exclusive=True,
            group="setup-voice-api-key-settings",
            exit_on_error=False,
        )

    def _existing_openai_credential(self) -> str | None:
        if self._selected_authentication() != "api_key":
            return None
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        if isinstance(app_config, Mapping):
            persisted = app_config.get("COMPREHENSIVE_CONFIG_RAW")
            source = persisted if isinstance(persisted, Mapping) else app_config
            locations = (
                ("api_settings", "openai", "api_key"),
                ("openai_api", "api_key"),
                ("API", "openai_api_key"),
            )
            for location in locations:
                current: object = source
                for part in location:
                    if not isinstance(current, Mapping):
                        current = None
                        break
                    current = current.get(part)
                if isinstance(current, str) and current:
                    return current
            api_settings = source.get("api_settings")
            if isinstance(api_settings, Mapping):
                openai = api_settings.get("openai")
                if isinstance(openai, Mapping):
                    environment_name = openai.get("api_key_env_var")
                    if isinstance(environment_name, str) and environment_name:
                        environment_value = os.environ.get(environment_name)
                        if environment_value:
                            return environment_value
            projected = app_config.get("OPENAI_API_KEY")
            if isinstance(projected, str) and projected:
                return projected
        value = os.environ.get("OPENAI_API_KEY")
        return value if value else None

    async def _run_voice_sample(
        self,
        generation: int,
        draft: voice_state.VoiceSetupDraft,
    ) -> None:
        try:
            result = await voice_state.run_voice_sample(
                draft,
                credential=self._existing_openai_credential(),
            )
        except asyncio.CancelledError:
            if generation == self._test_generation:
                self.query_one("#setup-voice-status", Static).update(
                    "Needs test. The sample was cancelled; retry when ready."
                )
            raise
        except Exception:
            if generation == self._test_generation:
                self.query_one("#setup-voice-status", Static).update(
                    "Needs test. The sample failed; review the service and retry."
                )
            return
        else:
            if generation != self._test_generation:
                return
            try:
                current = self._draft_from_controls()
            except (TypeError, ValueError):
                return
            if self._sample_identity(current) != self._sample_identity(draft):
                return
            self._verified_draft = draft
            try:
                played = await self._play_sample(result)
            except asyncio.CancelledError:
                if generation == self._test_generation:
                    self.query_one("#setup-voice-status", Static).update(
                        "Verified, playback failed. Retry playback/test."
                    )
                raise
            if generation != self._test_generation:
                return
            self.query_one("#setup-voice-status", Static).update(
                "Verified. The sample is ready to hear."
                if played
                else "Verified, playback failed. Retry playback/test."
            )
        finally:
            if self._test_in_progress_generation == generation:
                self._test_in_progress_generation = None
                self._refresh_sample_state()

    async def _play_sample(self, result: voice_state.VoiceSampleResult) -> bool:
        audio_player = getattr(self.app, "audio_player", None)
        play = getattr(audio_player, "play", None)
        if not callable(play):
            return False
        suffix = "." + result.response_format
        sample_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="chatbook-voice-sample-",
                suffix=suffix,
                delete=False,
            ) as handle:
                handle.write(result.body)
                sample_path = Path(handle.name)
            prior_path = self._sample_audio_path
            played = play(sample_path)
            if asyncio.iscoroutine(played):
                await played
            if prior_path is not None:
                prior_path.unlink(missing_ok=True)
            self._sample_audio_path = sample_path
            return True
        except asyncio.CancelledError:
            if sample_path is not None:
                sample_path.unlink(missing_ok=True)
            raise
        except Exception:
            if sample_path is not None:
                sample_path.unlink(missing_ok=True)
            logger.debug("Voice sample playback failed (category=playback)")
            return False

    async def commit(self) -> tuple[bool, str]:
        try:
            draft = self._draft_from_controls()
        except (TypeError, ValueError) as error:
            return False, str(error) or "Review the Voice setup fields."
        validation = voice_state.validate_voice_setup_draft(draft)
        if not validation.configuration_valid:
            return False, validation.errors[
                0
            ] if validation.errors else "Review the Voice setup fields."
        if (
            draft.authentication_mode == "api_key"
            and self._existing_openai_credential() is None
        ):
            return False, "Add an API key in Settings before saving this voice."
        request_id = self._next_save_request_id
        self._next_save_request_id += 1
        self._save_request_id = request_id
        self._save_draft = draft
        self._save_future = asyncio.get_running_loop().create_future()
        self.app.post_message(
            voice_state.build_voice_setup_save_event(
                draft,
                request_id=request_id,
                reply_to=self,
            )
        )
        try:
            return await asyncio.wait_for(
                asyncio.shield(self._save_future),
                timeout=self._SAVE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return False, "Voice settings are still applying. Retry to continue."
        finally:
            self._save_request_id = None
            self._save_draft = None
            self._save_future = None

    def _receive_save_result(self, result: object) -> None:
        from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
            STTSSettingsSaveResult,
        )

        future = self._save_future
        if (
            type(result) is not STTSSettingsSaveResult
            or result.request_id != self._save_request_id
            or future is None
            or future.done()
        ):
            return
        if not result.persisted:
            future.set_result((False, "Saving the Voice settings failed. Retry."))
            return
        provider_status = result.provider_statuses.get("openai")
        if provider_status == "pending":
            return
        runtime_ready = (
            provider_status in {"applied", "unchanged"}
            and "openai" in result.provider_configuration_revisions
            and "openai" in result.provider_runtime_revisions
        )
        if not runtime_ready:
            future.set_result(
                (False, "The Voice settings were saved, but are not active. Retry.")
            )
            return
        draft = self._save_draft
        if draft is None:
            return
        if not draft.use_as_default:
            future.set_result((True, ""))
            return
        if result.defaults_activated is True:
            future.set_result((True, ""))
            return
        future.set_result(
            (
                False,
                "The Voice settings were saved, but the default was not activated. Retry.",
            )
        )

    def receive_stts_settings_save_result(self, result: object) -> None:
        self._receive_save_result(result)

    def receive_stts_settings_runtime_result(self, result: object) -> None:
        self._receive_save_result(result)

    def get_step_data(self) -> Dict[str, Any]:
        values: Dict[str, Any] = {
            "endpoint": self.query_one("#setup-voice-endpoint", Input).value,
            "authentication_mode": self._selected_authentication(),
            "model_id": self.query_one("#setup-voice-model", Input).value,
            "voice_id": self.query_one("#setup-voice-voice", Input).value,
            "response_format": self.query_one("#setup-voice-format", Input)
            .value.strip()
            .lower(),
            "sample_text": self.query_one("#setup-voice-sample", Input).value,
            "use_as_default": self.query_one("#setup-voice-default", Checkbox).value,
        }
        try:
            speed = float(self.query_one("#setup-voice-speed", Input).value)
            if not math.isfinite(speed):
                raise ValueError
        except ValueError:
            return values
        values["speed"] = speed
        return values


class RagStep(SetupStep):
    """RAG/embeddings: report dep status; pick a default embedding model."""

    def __init__(self, wizard=None, config=None, *, deps_installed=None, **kwargs):
        super().__init__(wizard=wizard, config=config, **kwargs)
        if deps_installed is None:
            from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

            deps_installed = embeddings_rag_deps_installed
        self._deps_installed = deps_installed
        self.selected_embedding_model: str = ""

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-rag"):
            yield Static("Search & RAG", classes="setup-title")
            yield Static("", id="setup-rag-status", classes="setup-subtitle")
            with RadioSet(id="setup-rag-model-choice", classes="setup-choice-list"):
                for model_id in self._embedding_model_ids():
                    yield SetupRadioButton(model_id)
            yield Static("", classes="setup-step-error")

    def _embedding_model_ids(self) -> list[str]:
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        embedding_config = app_config.get("embedding_config", {})
        models = (
            embedding_config.get("models", {})
            if isinstance(embedding_config, dict)
            else {}
        )
        return sorted(models) if isinstance(models, dict) else []

    def on_mount(self) -> None:
        status = self.query_one("#setup-rag-status", Static)
        if self._deps_installed():
            status.update(
                "Embedding dependencies are installed. Pick a default model, or skip."
            )
        else:
            status.update(
                # Static.update() treats [..] as Rich markup by default, so the
                # extras-package brackets must be escaped or "[embeddings_rag]"
                # silently vanishes from the rendered text instead of showing.
                # TASK-1502: quoted plainly — backticks are markdown idiom and
                # render literally in a TUI.
                "RAG needs optional dependencies that aren't installed. Install the "
                'extras package "tldw_chatbook\\[embeddings_rag]" with your package '
                "manager, then revisit Settings ▸ RAG. Skipping for now is fine."
            )
            try:
                # TASK-1502: hide the model list outright — a wall of disabled
                # options under a "not installed" message reads as breakage
                # and adds nothing the user can act on.
                self.query_one("#setup-rag-model-choice", RadioSet).display = False
            except Exception:
                pass

    @on(RadioSet.Changed, "#setup-rag-model-choice")
    def _on_model(self, event: RadioSet.Changed) -> None:
        self.selected_embedding_model = str(event.pressed.label)

    def _effective_embedding_model(self) -> str:
        """F-A fix: same pressed-radio fallback as ProviderStep/ModelStep."""
        if self.selected_embedding_model:
            return self.selected_embedding_model
        try:
            pressed = self.query_one("#setup-rag-model-choice", RadioSet).pressed_button
        except Exception:
            return ""
        return str(pressed.label) if pressed is not None else ""

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import build_rag_commit

        model_id = self._effective_embedding_model()
        if not (self._deps_installed() and model_id):
            return True, ""
        ok = await self.wizard.commit_config(
            build_rag_commit(default_model_id=model_id)
        )
        if ok:
            self.selected_embedding_model = model_id
        return (True, "") if ok else (False, "Saving the embedding model failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {"embedding_model": self.selected_embedding_model}


class SpeechSetupStep(SetupStep):
    """Optional speech setup for exact managed Parakeet ONNX artifacts.

    TASK-1301: reuses the TASK-596 shared model-artifact controls
    (ModelInstallModal, ModelInstallProgress, ModelActivationControls) and
    the TASK-595 ModelArtifactService via the SAME
    Local_Ingestion.parakeet_v2_artifact convenience wrappers LibraryScreen's
    own Parakeet install surface already uses -- no duplicate artifact or
    network logic (AC#4). Language/precision options are enumerated from the
    canonical STT policy/catalog (first_run_speech_step_state, backed by
    tldw_chatbook.STT.routing) and gated to the exact managed Parakeet model
    and precision combinations (AC#2).

    Runtime gate (review Important 4): the `onnx-asr` extra is optional --
    missing it means a downloaded Parakeet artifact could never actually run.
    Gated exactly like RagStep gates on ``embeddings_rag_deps_installed()``:
    when the extra is absent, the install action stays visible for orientation
    but disabled so no unusable download can start.

    Persistence gate (AC#5 / review Important 3): commit() re-verifies --
    off the event loop -- that the exact selected artifact is active,
    AND requires that the user actually engaged this step THIS run
    (installed or activated it here) before writing anything to
    [transcription]. An artifact that merely happens to be active from an
    earlier session (e.g. installed via the Library screen) is not enough
    on its own -- a re-run that just presses Next through this step leaves
    whatever is already persisted completely untouched, however different
    (``remote-whisper``, ``default_language="auto"``, ...). The step also
    shows what is currently persisted before the user acts (the AC#5
    "prefill" clause) via ``first_run_speech_step_state.speech_prefill_status``.

    Skip and failures never trap the user (AC#6): Next/commit never blocks
    on install state, and a failed download still refreshes the step's own
    installed-state read so it never gets stuck showing a stale
    "installing…" affordance. A broken/not-ready installed item still shows
    ModelActivationControls(ready=False) so Delete (recovery) stays
    reachable (review Important 5).
    """

    def __init__(
        self,
        wizard: Optional["SetupWizardContainer"] = None,
        config: Optional[WizardStepConfig] = None,
        *,
        service_factory: Optional[Callable[[], Any]] = None,
        runtime_installed: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._service_factory = service_factory or parakeet_v2_managed_service
        if runtime_installed is None:
            from tldw_chatbook.Utils.optional_deps import parakeet_onnx_deps_installed

            runtime_installed = parakeet_onnx_deps_installed
        self._runtime_installed = runtime_installed
        recommended = speech_state.recommended_speech_selection()
        self._selected_language = recommended.language
        self._selected_precision = recommended.precision
        self._service: Any = None
        self._loading = False
        self._loaded = False
        self._reload_after_load = False
        self._load_error: Optional[str] = None
        self._installed_item: Any = None
        self._operation: Optional[str] = None
        self._pending_report: Any = None
        self._progress: Any = None
        self._external_selection_generation = 0
        self._external_selection_token: tuple[int, int] | None = None
        self._external_scope_ids: dict[tuple[int, int], str] = {}
        self._external_selection_worker: Worker | None = None
        self._external_busy = False
        self._external_status = ""
        self._pending_external_selection: PreparedExternalSelection | None = None
        self._external_commit_handoff: asyncio.Task[bool] | None = None
        self._external_commit_detached = False
        self._external_commit_pending = False
        # Review Important 3: set only by a SUCCESSFUL install/activation
        # made THROUGH THIS STEP during this run -- see commit()'s use via
        # first_run_speech_step_state.should_persist_speech_config.
        self._acted_this_run = False
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        transcription = (
            app_config.get("transcription", {})
            if isinstance(app_config, Mapping)
            else {}
        )
        prefill = speech_state.read_speech_prefill(app_config)
        if prefill.provider_id == speech_state.routing_policy().parakeet_provider_id:
            selected = speech_state.resolve_speech_selection(
                selected_language=prefill.language,
                selected_precision=prefill.precision or "int8",
                curated_selections=self._curated_selections(),
            )
            if selected is not None and selected.model_id == prefill.model_id:
                self._selected_language = selected.language
                self._selected_precision = selected.precision
        direct_config = (
            transcription.get("transcribe_cpp", {})
            if isinstance(transcription, Mapping)
            else {}
        )
        self._transcribe_cpp_configured = bool(
            direct_config.get("model_path")
            if isinstance(direct_config, Mapping)
            else False
        )

    def compose_step(self) -> ComposeResult:
        """Render the title/prefill/status/action block, then the language
        and precision catalogs.

        Entirely pure/I/O-free: it builds from ``self`` bookkeeping already
        in memory and the canonical Parakeet routing policy/precision values.
        Any I/O (checking installed state) is deferred to ``on_show``.

        Returns:
            The composed widgets: title, optional prefill line, status line
            plus its action control, an optional "use as default" affordance,
            then the language and precision ``RadioSet`` catalogs (see the
            Review Important 2 note below for the ordering rationale).
        """
        # Review Important 2: the primary action must be visible at the
        # wizard's own tested 120x40 budget -- title/subtitle/prefill/
        # status/action come FIRST (typically <=6 rows); the informational
        # language/precision catalog (up to 27 disabled rows, already
        # capped by ".setup-choice-list") comes after, reachable by
        # scrolling like any other step's overflow content.
        with Vertical(classes="setup-speech"):
            yield Static("Speech transcription (optional)", classes="setup-title")
            yield Static(
                f"Selected: {self._model_label()} — on-device speech-to-text. "
                "Skip and set this up later from Lab ▸ Models.",
                classes="setup-subtitle",
            )
            prefill_text = self._prefill_status_text()
            if prefill_text:
                # NEW-1 (review): persisted values (and, below, the runtime-
                # missing message's bracketed extras names) may contain
                # literal "[...]" -- markup=False, same fix already applied
                # to "#setup-summary-rows" for the identical trap.
                yield Static(
                    prefill_text,
                    id="setup-speech-prefill",
                    classes="setup-subtitle",
                    markup=False,
                )
            status_text, action_widget = self._status_and_action()
            yield Static(
                status_text,
                id="setup-speech-status",
                classes="setup-subtitle",
                markup=False,
            )
            progress = ModelInstallProgress(
                self._progress, id="setup-speech-install-progress"
            )
            progress.display = (
                self._operation == "install" and self._progress is not None
            )
            yield progress
            yield Button(
                "Use model from disk…",
                id="setup-speech-use-from-disk",
                variant=(
                    "primary"
                    if action_widget is None
                    or getattr(action_widget, "disabled", False)
                    else "default"
                ),
                disabled=self._external_busy or self._lifecycle_pending,
            )
            external_status = Static(
                self._external_status,
                id="setup-speech-external-status",
                classes="setup-subtitle",
                markup=False,
            )
            external_status.display = bool(self._external_status)
            yield external_status
            if self._external_busy:
                yield Button(
                    "Cancel external setup",
                    id="setup-speech-cancel-external",
                    variant="default",
                )
            if action_widget is not None:
                yield action_widget
            if self._use_as_default_offer():
                # Review NEW-2: installed + active + configured elsewhere is
                # the one state where neither "install" nor "activate" is a
                # real action -- offer the affordance the prefill sentence
                # actually promises instead of leaving it undeliverable.
                yield Button(
                    f"Use {self._model_label()} as my default",
                    id="setup-speech-use-as-default",
                    variant="primary",
                )
            yield Static(
                "Existing local transcribe.cpp GGUF configured."
                if self._transcribe_cpp_configured
                else "No existing local transcribe.cpp GGUF configured.",
                id="setup-speech-transcribe-cpp-status",
                classes="setup-subtitle",
                markup=False,
            )
            yield Button(
                "Choose another GGUF…"
                if self._transcribe_cpp_configured
                else "Use an existing transcribe.cpp GGUF…",
                id="setup-speech-choose-transcribe-cpp-gguf",
                disabled=self._external_commit_pending,
            )
            yield Static("", classes="setup-step-error")
            yield Label("Language", classes="setup-field-label")
            with RadioSet(
                id="setup-speech-language-choice", classes="setup-choice-list"
            ):
                for option in speech_state.speech_language_options(
                    curated_model_ids=self._curated_model_ids()
                ):
                    label = option.display_name + (
                        " (recommended)"
                        if option.code == "en"
                        else " — not yet available for managed install"
                        if not option.selectable
                        else ""
                    )
                    yield SetupRadioButton(
                        label,
                        id=f"setup-speech-language-{option.code}",
                        value=option.selectable
                        and option.code == self._selected_language,
                        disabled=not option.selectable or self._lifecycle_pending,
                    )
            yield Label("Precision", classes="setup-field-label")
            with RadioSet(
                id="setup-speech-precision-choice", classes="setup-choice-list"
            ):
                for option in speech_state.speech_precision_options(
                    model_id=self._selection().model_id,
                    curated_selections=self._curated_selections(),
                ):
                    label = option.display_name + (
                        " (recommended)"
                        if option.value == "int8"
                        else " — not yet available for managed install"
                        if not option.selectable
                        else ""
                    )
                    # Minor 8: pre-press ONLY the one recommended option --
                    # "selectable" alone would pre-press every selectable
                    # precision the moment a second one is ever curated.
                    yield SetupRadioButton(
                        label,
                        id=f"setup-speech-precision-{option.value}",
                        value=(
                            option.selectable
                            and option.value == self._selected_precision
                        ),
                        disabled=not option.selectable or self._lifecycle_pending,
                    )

    # -- pure, I/O-free helpers ------------------------------------------
    @staticmethod
    def _curated_model_ids() -> frozenset[str]:
        policy = speech_state.routing_policy()
        return frozenset({policy.parakeet_v2_model_id, policy.parakeet_v3_model_id})

    @staticmethod
    def _curated_selections() -> frozenset[tuple[str, str]]:
        return frozenset(
            (model_id, precision)
            for model_id in SpeechSetupStep._curated_model_ids()
            for precision in PARAKEET_PRECISIONS
        )

    def _selection(self) -> speech_state.SpeechSelection:
        selection = speech_state.resolve_speech_selection(
            selected_language=self._selected_language,
            selected_precision=self._selected_precision,
            curated_selections=self._curated_selections(),
        )
        # The stored selection is initialized from, and only changed through,
        # selectable radios. This guard keeps a later registry change skip-safe.
        return selection or speech_state.recommended_speech_selection()

    @property
    def _reference(self) -> Any:
        selection = self._selection()
        return parakeet_reference(selection.model_id, selection.precision)

    def _model_label(self) -> str:
        selection = self._selection()
        descriptor = parakeet_descriptor(selection.model_id, selection.precision)
        policy = speech_state.routing_policy()
        version = "v2" if descriptor.model_id == policy.parakeet_v2_model_id else "v3"
        language = speech_state.LANGUAGE_DISPLAY_NAMES.get(
            selection.language, selection.language
        )
        return f"Parakeet {version} ({language}, {descriptor.precision.upper()})"

    # -- review finding 2: read the PRESSED radio, never a hardcoded default --
    _LANGUAGE_RADIO_ID_PREFIX = "setup-speech-language-"
    _PRECISION_RADIO_ID_PREFIX = "setup-speech-precision-"

    @on(RadioSet.Changed, "#setup-speech-language-choice")
    def _on_speech_language_changed(self, event: RadioSet.Changed) -> None:
        if self._lifecycle_pending or event.pressed is None:
            return
        button_id = event.pressed.id or ""
        language = button_id.removeprefix(self._LANGUAGE_RADIO_ID_PREFIX)
        self._set_exact_selection(language, self._selected_precision)

    @on(RadioSet.Changed, "#setup-speech-precision-choice")
    def _on_speech_precision_changed(self, event: RadioSet.Changed) -> None:
        if self._lifecycle_pending or event.pressed is None:
            return
        button_id = event.pressed.id or ""
        precision = button_id.removeprefix(self._PRECISION_RADIO_ID_PREFIX)
        self._set_exact_selection(self._selected_language, precision)

    def _set_exact_selection(self, language: str, precision: str) -> None:
        if self._external_commit_pending:
            return
        selection = speech_state.resolve_speech_selection(
            selected_language=language,
            selected_precision=precision,
            curated_selections=self._curated_selections(),
        )
        if selection is None:
            return
        if (
            selection.language == self._selected_language
            and selection.precision == self._selected_precision
        ):
            return
        self._discard_external_selection()
        self._selected_language = selection.language
        self._selected_precision = selection.precision
        self._installed_item = None
        self._loaded = False
        self._load_error = None
        self._pending_report = None
        self._ensure_loaded(force=True)

    def _effective_language(self) -> str:
        """The code of the currently pressed language radio, or "" for none.

        Mirrors ``ModelSetupStep._live_pressed_radio``'s guard: reading
        ``RadioSet.pressed_button`` unguarded can resurrect a stale press
        left over from before a ``recompose``, so membership in the set's
        CURRENT children is required too. "" (never pressed / step
        unmounted, e.g. ``commit()`` called before ``on_show()``) is a
        valid, skip-safe result -- ``resolve_speech_selection`` falls back
        to the recommended default for it.
        """
        return (
            self._pressed_radio_code(
                "#setup-speech-language-choice", self._LANGUAGE_RADIO_ID_PREFIX
            )
            or self._selected_language
        )

    def _effective_precision(self) -> str:
        """The value of the currently pressed precision radio, or "" for none."""
        return (
            self._pressed_radio_code(
                "#setup-speech-precision-choice", self._PRECISION_RADIO_ID_PREFIX
            )
            or self._selected_precision
        )

    def _pressed_radio_code(self, selector: str, id_prefix: str) -> str:
        try:
            radio_set = self.query_one(selector, RadioSet)
        except Exception:
            return ""
        pressed = radio_set.pressed_button
        if pressed is None or pressed not in radio_set.query(RadioButton):
            return ""
        button_id = pressed.id or ""
        if not button_id.startswith(id_prefix):
            return ""
        return button_id[len(id_prefix) :]

    def _prefill(self) -> Any:
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        return speech_state.read_speech_prefill(app_config)

    def _installed_active(self) -> bool:
        item = self._installed_item
        return bool(item is not None and item.active)

    def _prefill_status_text(self) -> str:
        """AC#5's "re-run prefills" clause: show what is already persisted.

        Reads ``self.wizard.app_instance.app_config`` directly -- the same
        in-memory, already-loaded dict other steps read synchronously in
        compose_step() (e.g. RagStep._embedding_model_ids()) -- so this
        needs no worker. ``installed_active``/``acted_this_run`` make the
        copy state-aware (review NEW-2): the "installing or activating"
        promise is only shown when one of those is still a real action.
        """
        return speech_state.speech_prefill_status(
            self._prefill(),
            installed_active=self._installed_active(),
            acted_this_run=self._acted_this_run,
            runtime_installed=self._runtime_installed(),
            selected_label=self._model_label(),
        )

    def _use_as_default_offer(self) -> bool:
        """Review NEW-2: offer the real affordance the prefill sentence
        promises -- installed AND active (so neither Install nor Activate
        is available), a DIFFERENT provider is currently persisted, the
        runtime can actually run Parakeet, and the user has not already
        opted in this run (once acted, commit() already persists on Next).
        """
        if self._acted_this_run:
            return False
        if not self._runtime_installed():
            return False
        if not self._installed_active():
            return False
        prefill = self._prefill()
        return prefill.provider_id != speech_state.routing_policy().parakeet_provider_id

    @property
    def _lifecycle_pending(self) -> bool:
        """Review NEW-3: a forced reload in flight must ALSO disable the
        install/activation controls, not just an explicit operation --
        otherwise a just-deleted (or just-installed) artifact's stale
        ``_installed_item`` briefly re-renders with enabled controls before
        the reload's own callback replaces it (InstalledView's own pending
        computation includes its loading flag for the identical reason).
        """
        return (
            self._operation is not None
            or self._loading
            or self._external_busy
            or self._external_commit_pending
        )

    def _status_and_action(self) -> tuple[str, Optional[Widget]]:
        # Review Important 4: gate BEFORE the installed-state load so a
        # minimal install sees the real reason immediately. The action stays
        # visible for orientation but cannot start an unusable download.
        if not self._runtime_installed():
            return (
                'The "onnx-asr" runtime is not installed, so a downloaded '
                "model could not run. Install the extras package "
                '"tldw_chatbook[transcription_parakeet_onnx]" (or run '
                "pip install 'onnx-asr[cpu]==0.12.0'), then revisit this "
                "step. Skipping is safe — set this up later from Lab ▸ "
                "Models.",
                Button(
                    "Review and install…",
                    id="setup-speech-install",
                    variant="primary",
                    disabled=True,
                ),
            )
        if not self._loaded:
            if self._load_error:
                return self._load_error, Button("Retry", id="setup-speech-retry")
            return "Checking installed models…", None
        item = self._installed_item
        if item is None:
            return "Not installed.", Button(
                "Review and install…",
                id="setup-speech-install",
                variant="primary",
                disabled=self._lifecycle_pending,
            )
        if item.error is not None or not item.ready:
            # Review Important 5: reuse the SAME 596 control instead of a
            # dead end -- ready=False already keeps Delete enabled (the
            # only real recovery path) while disabling Activate.
            return (
                "This model needs attention — delete it below and install "
                "again, or manage it from Lab ▸ Models ▸ Installed.",
                ModelActivationControls(
                    self._reference,
                    active=item.active,
                    ready=item.ready,
                    pending=self._lifecycle_pending,
                ),
            )
        status = (
            "Installed and active." if item.active else "Installed, not yet active."
        )
        return status, ModelActivationControls(
            self._reference,
            active=item.active,
            ready=item.ready,
            pending=self._lifecycle_pending,
        )

    # -- user-owned external roots ---------------------------------------
    def _source_service(self) -> Any:
        """Return the app-owned source service shared with Lab and Library."""

        return self.wizard.app_instance._ensure_parakeet_source_service()

    def _source_key(self) -> ParakeetSourceKey:
        selection = self._selection()
        return ParakeetSourceKey.from_values(selection.model_id, selection.precision)

    def _next_external_token(self) -> tuple[int, int]:
        """Fence picker and worker callbacks to one exact mounted generation."""

        prior = self._external_selection_token
        if prior is not None:
            self._release_external_scope(prior)
        worker = self._external_selection_worker
        if worker is not None and not worker.is_finished:
            worker.cancel()
        self._external_selection_generation += 1
        token = (self._external_selection_generation, id(self))
        self._external_selection_token = token
        self._external_scope_ids[token] = f"setup-speech-{token[1]}-{token[0]}"
        self._external_selection_worker = None
        self._pending_external_selection = None
        return token

    def _owns_external_token(self, token: tuple[int, int]) -> bool:
        return (
            token == self._external_selection_token
            and token[1] == id(self)
            and self.is_mounted
        )

    def _release_external_scope(self, token: tuple[int, int]) -> None:
        scope_id = self._external_scope_ids.pop(token, None)
        if scope_id is None:
            return
        service = getattr(
            self.wizard.app_instance,
            "_parakeet_source_service",
            None,
        )
        if service is not None:
            service.release_scope(scope_id)

    def _discard_external_selection(self) -> None:
        """Cancel pending external work without changing persisted source state."""

        token = self._external_selection_token
        handoff_active = (
            self._external_commit_handoff is not None
            and not self._external_commit_handoff.done()
        )
        worker = self._external_selection_worker
        if worker is not None and not worker.is_finished:
            worker.cancel()
        self._external_selection_generation += 1
        self._external_selection_token = None
        self._external_selection_worker = None
        self._external_busy = False
        self._external_status = ""
        self._pending_external_selection = None
        if handoff_active:
            self._external_commit_detached = True
        elif token is not None:
            self._release_external_scope(token)

    def _set_external_status(
        self,
        text: str,
        *,
        busy: bool | None = None,
    ) -> None:
        self._external_status = text
        if busy is not None:
            self._external_busy = busy
        self.refresh(recompose=True)

    @on(Button.Pressed, "#setup-speech-use-from-disk")
    def _use_external_pressed(self) -> None:
        if self._lifecycle_pending:
            return
        token = self._next_external_token()
        key = self._source_key()
        self.app.push_screen(
            SelectDirectory(
                str(Path.home()),
                title=f"Choose {key.model_id} {key.precision.upper()} directory",
            ),
            lambda selected: self._external_directory_selected(
                token,
                key,
                selected,
            ),
        )

    @on(Button.Pressed, "#setup-speech-cancel-external")
    def _cancel_external_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._discard_external_selection()
        self._set_external_status(
            "External setup cancelled. The prior source is unchanged.",
            busy=False,
        )
        self.call_after_refresh(self._focus_external_disk_action)

    def _focus_external_disk_action(self) -> None:
        """Keep Enter on the external action after Cancel recomposes the step."""

        try:
            self.query_one("#setup-speech-use-from-disk", Button).focus()
        except NoMatches:
            pass

    def _external_directory_selected(
        self,
        token: tuple[int, int],
        key: ParakeetSourceKey,
        selected: Path | None,
    ) -> None:
        if not self._owns_external_token(token):
            self._release_external_scope(token)
            return
        if selected is None:
            self._discard_external_selection()
            return
        scope_id = self._external_scope_ids.get(token)
        if scope_id is None:
            return
        self._set_external_status("Verifying model files…", busy=True)
        self._external_selection_worker = self._verify_external_source(
            token,
            key,
            Path(selected),
            scope_id,
        )

    @work(
        thread=True,
        group="setup-speech-external-verify",
        exclusive=True,
        exit_on_error=False,
        description="Verify external Parakeet source",
    )
    def _verify_external_source(
        self,
        token: tuple[int, int],
        key: ParakeetSourceKey,
        directory: Path,
        scope_id: str,
    ) -> None:
        """Hash one exact external root outside the Textual event loop."""

        worker = get_current_worker()

        def cancelled() -> bool:
            return worker.is_cancelled

        def progress(done: int, total: int) -> None:
            self.app.call_from_thread(
                self._apply_external_hash_progress,
                token,
                done,
                total,
            )

        try:
            prepared = self._source_service().prepare_external(
                key,
                directory,
                owner=("scope", scope_id),
                cancelled=cancelled,
                progress=progress,
            )
        except ExternalParakeetVerificationError as exc:
            message, is_error = format_external_parakeet_recovery(exc.code)
            if is_error:
                logger.warning(
                    "External Parakeet verification failed; error_type={}",
                    type(exc).__name__,
                )
            self.app.call_from_thread(
                self._apply_external_verification_result,
                token,
                None,
                message,
                is_error,
            )
            return
        except Exception as exc:
            logger.warning(
                "External Parakeet verification failed; error_type={}",
                type(exc).__name__,
            )
            self.app.call_from_thread(
                self._apply_external_verification_result,
                token,
                None,
                "The selected model could not be verified. Choose the directory again.",
                True,
            )
            return
        self.app.call_from_thread(
            self._apply_external_verification_result,
            token,
            prepared,
            None,
            False,
        )

    def _apply_external_hash_progress(
        self,
        token: tuple[int, int],
        done: int,
        total: int,
    ) -> None:
        if self._owns_external_token(token):
            self._set_external_status(
                f"Verifying model files · {done:,} / {total:,} bytes",
                busy=True,
            )

    def _apply_external_verification_result(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection | None,
        error: str | None,
        error_is_failure: bool = True,
    ) -> None:
        if not self._owns_external_token(token):
            self._release_external_scope(token)
            return
        self._external_selection_worker = None
        if error is not None or prepared is None:
            self._release_external_scope(token)
            message = error or "The selected model could not be verified."
            self._set_external_status(
                message,
                busy=False,
            )
            self.notify(
                message,
                severity="error" if error_is_failure else "information",
            )
            return
        self._set_external_status(
            "Checking the managed VAD dependency…",
            busy=True,
        )
        self._external_selection_worker = self._prepare_external_readiness(
            token,
            prepared,
        )

    @work(
        thread=True,
        group="setup-speech-external-ready",
        exclusive=True,
        exit_on_error=False,
        description="Prepare external Parakeet configuration",
    )
    def _prepare_external_readiness(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
    ) -> None:
        """Recheck root/VAD and prepare a write-free config patch off-loop."""

        worker = get_current_worker()
        if worker.is_cancelled:
            return
        try:
            self._source_service().prepare_config_commit(prepared)
        except ParakeetSourceError as exc:
            outcome = (
                "vad"
                if exc.code is ParakeetSourceErrorCode.VAD_UNAVAILABLE
                else "error"
            )
        except Exception as exc:
            logger.warning(
                "External Parakeet readiness failed; error_type={}",
                type(exc).__name__,
            )
            outcome = "error"
        else:
            outcome = "ready"
        self.app.call_from_thread(
            self._apply_external_readiness_result,
            token,
            prepared,
            outcome,
        )

    def _apply_external_readiness_result(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        outcome: str,
    ) -> None:
        if not self._owns_external_token(token):
            self._release_external_scope(token)
            return
        self._external_selection_worker = None
        if outcome == "vad":
            self._set_external_status(
                "Preparing the managed VAD dependency…",
                busy=True,
            )
            self._external_selection_worker = self._preflight_external_vad(
                token,
                prepared,
            )
            return
        if outcome != "ready":
            self._release_external_scope(token)
            message = (
                "The external source could not be prepared. "
                "The prior source is unchanged."
            )
            self._set_external_status(message, busy=False)
            self.notify(message, severity="error")
            return
        self._pending_external_selection = prepared
        message = (
            "External model verified. Continue to save."
            if self._runtime_installed()
            else "Runtime required"
        )
        self._set_external_status(message, busy=False)

    @work(
        thread=True,
        group="setup-speech-external-vad-preflight",
        exclusive=True,
        exit_on_error=False,
        description="Check managed VAD dependency",
    )
    def _preflight_external_vad(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
    ) -> None:
        try:
            report = asyncio.run(run_parakeet_vad_preflight())
        except Exception as exc:
            logger.warning(
                "Managed VAD preflight failed; error_type={}",
                type(exc).__name__,
            )
            self.app.call_from_thread(
                self._apply_external_vad_preflight_result,
                token,
                prepared,
                None,
                "The managed VAD dependency could not be prepared.",
            )
            return
        self.app.call_from_thread(
            self._apply_external_vad_preflight_result,
            token,
            prepared,
            report,
            None,
        )

    def _apply_external_vad_preflight_result(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        report: Any,
        error: str | None,
    ) -> None:
        if not self._owns_external_token(token):
            self._release_external_scope(token)
            return
        self._external_selection_worker = None
        vad_reference = parakeet_vad_reference()
        vad_source_url = parakeet_vad_descriptor().source_url
        if (
            error is not None
            or report is None
            or report.root != vad_reference
            or not report.entries
            or any(
                entry.ref != vad_reference or entry.source_url != vad_source_url
                for entry in report.entries
            )
        ):
            self._release_external_scope(token)
            message = error or "The managed VAD plan changed. Choose the model again."
            self._set_external_status(message, busy=False)
            self.notify(message, severity="error")
            return
        self.app.push_screen(
            ModelInstallModal(report, model_label="Silero VAD dependency"),
            lambda confirmed: self._confirm_external_vad(
                bool(confirmed),
                token,
                prepared,
                report,
            ),
        )

    def _confirm_external_vad(
        self,
        confirmed: bool,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        report: Any,
    ) -> None:
        if not self._owns_external_token(token):
            return
        if not confirmed:
            self._discard_external_selection()
            self._set_external_status(
                "VAD install cancelled. The prior source is unchanged.",
                busy=False,
            )
            return
        self._set_external_status(
            "Installing the managed VAD dependency…",
            busy=True,
        )
        self._external_selection_worker = self._provision_external_vad(
            token,
            prepared,
            report,
        )

    @work(
        group="setup-speech-external-vad-install",
        exclusive=True,
        exit_on_error=False,
        description="Install managed VAD dependency",
    )
    async def _provision_external_vad(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        report: Any,
    ) -> None:
        def progress(event: Any) -> None:
            self._apply_external_vad_progress(
                token,
                event.bytes_done,
                event.bytes_total,
            )

        try:
            await run_parakeet_vad_provision(report, progress=progress)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Managed VAD installation failed; error_type={}",
                type(exc).__name__,
            )
            self._apply_external_vad_provision_result(
                token,
                prepared,
                "The managed VAD dependency could not be installed.",
            )
            return
        self._apply_external_vad_provision_result(
            token,
            prepared,
            None,
        )

    def _apply_external_vad_progress(
        self,
        token: tuple[int, int],
        done: int,
        total: int,
    ) -> None:
        if self._owns_external_token(token):
            self._set_external_status(
                f"Installing managed VAD dependency · {done:,} / {total:,} bytes",
                busy=True,
            )

    def _apply_external_vad_provision_result(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        error: str | None,
    ) -> None:
        if not self._owns_external_token(token):
            self._release_external_scope(token)
            return
        self._external_selection_worker = None
        if error is not None:
            self._release_external_scope(token)
            self._set_external_status(error, busy=False)
            self.notify(error, severity="error")
            return
        self._set_external_status(
            "Rechecking model files and managed VAD…",
            busy=True,
        )
        self._external_selection_worker = self._prepare_external_readiness(
            token,
            prepared,
        )

    def on_unmount(self) -> None:
        self._discard_external_selection()

    # -- lazy installed-state load ----------------------------------------
    def on_show(self) -> None:
        """Trigger the lazy installed-state read for the selected artifact.

        ``compose_step`` renders synchronously from in-memory state only;
        this is where the step first asks the artifact service (via
        ``_ensure_loaded`` -> ``_load_installed_state``, an exclusive
        background worker) whether the selected managed artifact already
        exists, so the status line can move past "Checking installed
        models…". Idempotent: a step already re-shown (rerun navigation)
        does not re-trigger a redundant load (see ``_ensure_loaded``).

        Returns:
            None.
        """
        super().on_show()
        self._ensure_loaded()

    def _ensure_loaded(self, *, force: bool = False) -> None:
        # Minor 11: a forced reload requested while a load is already in
        # flight must not be silently dropped -- remember it and honor it
        # when the in-flight load's own callback runs (InstalledView's own
        # _reload_after_load pattern).
        if self._loading:
            if force:
                self._reload_after_load = True
            return
        if self._loaded and not force:
            return
        self._loading = True
        self._load_error = None
        self.refresh(recompose=True)
        self._load_installed_state()

    def _service_for_worker(self) -> Any:
        if self._service is None:
            self._service = self._service_factory()
        return self._service

    @work(thread=True, group="setup-speech-load", exclusive=True, exit_on_error=False)
    def _load_installed_state(self) -> None:
        try:
            service = self._service_for_worker()
            item = next(
                (
                    candidate
                    for candidate in service.list_installed()
                    if candidate.descriptor is not None
                    and candidate.descriptor.reference == self._reference
                ),
                None,
            )
        except Exception:
            logger.opt(exception=True).error(
                "Speech setup step could not read installed models"
            )
            self.app.call_from_thread(
                self._apply_installed_state,
                None,
                "Could not check installed speech models.",
            )
            return
        self.app.call_from_thread(self._apply_installed_state, item, None)

    def _apply_installed_state(self, item: Any, error: Optional[str]) -> None:
        self._installed_item = item
        self._loading = False
        self._loaded = error is None
        self._load_error = error
        reload_after_load = self._reload_after_load
        self._reload_after_load = False
        if reload_after_load:
            self._ensure_loaded(force=True)
        else:
            self.refresh(recompose=True)

    # -- install: preflight -> consent modal -> provision ------------------
    @on(Button.Pressed, "#setup-speech-install")
    def _install_pressed(self) -> None:
        if self._lifecycle_pending:  # review NEW-3: also refuse during a reload
            return
        self._operation = "install"
        self.refresh(recompose=True)
        self._preflight_install()

    @on(Button.Pressed, "#setup-speech-retry")
    def _retry_pressed(self) -> None:
        self._ensure_loaded(force=True)

    @on(Button.Pressed, "#setup-speech-use-as-default")
    def _use_as_default_pressed(self) -> None:
        """Review NEW-2: make the affordance real. Sets the SAME
        ``_acted_this_run`` flag install/activate success sets -- nothing
        is written to disk here (matches every other step: only commit()
        on Next writes), but the pending choice is now genuine, and the
        prefill sentence updates to say so."""
        if self._lifecycle_pending or not self._use_as_default_offer():
            return
        self._acted_this_run = True
        self.notify(
            f"{self._model_label()} will become your default when you continue.",
            severity="information",
        )
        self.refresh(recompose=True)

    @on(Button.Pressed, "#setup-speech-choose-transcribe-cpp-gguf")
    def _choose_transcribe_cpp_gguf_pressed(self) -> None:
        """Open a GGUF-only picker for optional direct-local transcription."""

        if self._external_commit_pending:
            return

        async def picker_callback(selected_path: Path | None) -> None:
            if selected_path is not None and not self._external_commit_pending:
                self._configure_transcribe_cpp_gguf(selected_path)

        self.app.push_screen(
            FileOpen(
                location=Path.home(),
                title="Choose transcribe.cpp GGUF",
                filters=Filters(("GGUF models", is_gguf_file)),
            ),
            picker_callback,
        )

    @work(
        thread=True,
        group="setup-speech-transcribe-cpp-gguf",
        exclusive=True,
        exit_on_error=False,
    )
    def _configure_transcribe_cpp_gguf(self, selected_path: Path) -> None:
        """Admit and persist a selected GGUF off the Textual event loop."""
        try:
            configure_transcribe_cpp_model_path(selected_path)
        except Exception:
            self.app.call_from_thread(
                self._apply_transcribe_cpp_gguf_result,
                False,
            )
            return
        self.app.call_from_thread(
            self._apply_transcribe_cpp_gguf_result,
            True,
        )

    def _apply_transcribe_cpp_gguf_result(self, configured: bool) -> None:
        """Apply a path-free direct-local GGUF configuration result."""
        if not configured:
            self.notify(
                "That GGUF cannot be used by transcribe.cpp. Choose another GGUF.",
                severity="warning",
            )
            return
        self._transcribe_cpp_configured = True
        self.notify(
            "Local GGUF configured for transcribe.cpp.",
            severity="information",
        )
        self.refresh(recompose=True)

    @work(
        thread=True, group="setup-speech-install", exclusive=True, exit_on_error=False
    )
    def _preflight_install(self) -> None:
        import asyncio

        selection = self._selection()
        try:
            report = asyncio.run(  # policy-exception: worker-thread loop
                run_parakeet_preflight(selection.model_id, selection.precision)
            )
        except Exception as exc:
            logger.opt(exception=True).error("Speech transcription preflight failed")
            self.app.call_from_thread(
                self._apply_preflight_result,
                None,
                install_failure_message(exc, model_label=self._model_label()),
            )
            return
        self.app.call_from_thread(self._apply_preflight_result, report, None)

    def _apply_preflight_result(self, report: Any, error: Optional[str]) -> None:
        if error is not None or report is None:
            self._operation = None
            self.notify(error or "Speech model preflight failed.", severity="error")
            self.refresh(recompose=True)
            return
        self._pending_report = report
        self.app.push_screen(
            ModelInstallModal(
                report,
                model_label=self._model_label(),
                container_id="setup-speech-install-modal",
                confirm_id="setup-speech-install-confirm",
                cancel_id="setup-speech-install-cancel",
            ),
            self._confirm_install,
        )

    def _confirm_install(self, confirmed: bool) -> None:
        if self._external_commit_pending:
            return
        if not confirmed:
            self._pending_report = None
            self._operation = None
            self.refresh(recompose=True)
            return
        self._provision_install()

    @work(
        thread=True, group="setup-speech-install", exclusive=True, exit_on_error=False
    )
    def _provision_install(self) -> None:
        import asyncio

        report = self._pending_report
        if report is None:
            self.app.call_from_thread(
                self._apply_provision_result,
                "No install plan is available; review the model again.",
            )
            return
        try:
            selection = self._selection()
            asyncio.run(  # policy-exception: worker-thread loop
                run_parakeet_provision(
                    selection.model_id,
                    selection.precision,
                    report,
                    progress=make_progress_callback(self.post_message),
                )
            )
        except Exception as exc:
            logger.opt(exception=True).error("Speech model installation failed")
            self.app.call_from_thread(
                self._apply_provision_result,
                install_failure_message(exc, model_label=self._model_label()),
            )
            return
        try:
            self._source_service().prefer_managed(self._source_key())
        except Exception as exc:
            logger.warning(
                "Speech model source preference failed after installation; "
                "error_type={}",
                type(exc).__name__,
            )
            self.app.call_from_thread(
                self._apply_provision_result,
                None,
                "Speech model installed and activated, but its source "
                "preference could not be saved. Activate it again to retry "
                "the preference.",
            )
            return
        self.app.call_from_thread(self._apply_provision_result, None)

    @on(InstallProgressed)
    def _install_progressed(self, event: InstallProgressed) -> None:
        event.stop()  # Minor 9: LibraryScreen's equivalent handler stops it too
        self._progress = event.progress
        try:
            progress = self.query_one(
                "#setup-speech-install-progress", ModelInstallProgress
            )
        except NoMatches:
            self.refresh(recompose=True)
            return
        progress.display = True
        progress.update_progress(event.progress)

    def _apply_provision_result(
        self,
        error: Optional[str],
        preference_error: Optional[str] = None,
    ) -> None:
        self._pending_report = None
        self._operation = None
        self._progress = None
        if error is not None:
            self.notify(error, severity="error")
        else:
            # Review Important 3: a SUCCESSFUL install made through this
            # step this run is the engagement commit()'s no-clobber gate
            # requires -- see should_persist_speech_config.
            self._acted_this_run = True
            self._discard_external_selection()
            self.notify(
                preference_error or "Speech model installed and activated.",
                severity="warning" if preference_error else "information",
            )
        # AC#6: failures never trap -- always refresh installed state so the
        # step reflects reality (and drops the disabled "installing…" affordance)
        # whether provisioning succeeded or failed.
        self._ensure_loaded(force=True)

    # -- activation / deletion: the 596 controls, reused verbatim ----------
    @on(ActivationRequested)
    def _activation_requested(self, event: ActivationRequested) -> None:
        event.stop()
        if self._lifecycle_pending:  # review NEW-3: also refuse during a reload
            return
        self._operation = "activate"
        self.refresh(recompose=True)
        self._activate_model()

    @work(
        thread=True, group="setup-speech-lifecycle", exclusive=True, exit_on_error=False
    )
    def _activate_model(self) -> None:
        try:
            self._service_for_worker().activate(self._reference)
        except Exception as exc:
            logger.opt(exception=True).error("Speech model activation failed")
            self.app.call_from_thread(
                self._apply_lifecycle_result,
                lifecycle_failure_message(exc, operation="activation"),
            )
            return
        try:
            self._source_service().prefer_managed(self._source_key())
        except Exception as exc:
            logger.warning(
                "Speech model source preference failed after activation; error_type={}",
                type(exc).__name__,
            )
            self.app.call_from_thread(
                self._apply_lifecycle_result,
                None,
                "Speech model activated, but its source preference could not "
                "be saved. Activate it again to retry the preference.",
            )
            return
        self.app.call_from_thread(self._apply_lifecycle_result, None)

    @on(DeletionRequested)
    def _deletion_requested(self, event: DeletionRequested) -> None:
        event.stop()
        if self._lifecycle_pending:  # review NEW-3: also refuse during a reload
            return
        self.app.push_screen(
            DeleteConfirmationDialog(
                item_type="Model",
                item_name=self._model_label(),
                additional_warning=(
                    "The managed model files will be removed from this device."
                ),
                permanent=True,
            ),
            self._confirm_deletion,
        )

    def _confirm_deletion(self, confirmed: bool) -> None:
        if not confirmed or self._lifecycle_pending:
            return
        self._operation = "delete"
        self.refresh(recompose=True)
        self._delete_model()

    @work(
        thread=True, group="setup-speech-lifecycle", exclusive=True, exit_on_error=False
    )
    def _delete_model(self) -> None:
        try:
            self._service_for_worker().delete(self._reference)
        except Exception as exc:
            logger.opt(exception=True).error("Speech model deletion failed")
            self.app.call_from_thread(
                self._apply_lifecycle_result,
                lifecycle_failure_message(exc, operation="deletion"),
            )
            return
        self.app.call_from_thread(self._apply_lifecycle_result, None)

    def _apply_lifecycle_result(
        self,
        error: Optional[str],
        preference_error: Optional[str] = None,
    ) -> None:
        # Capture BEFORE clearing: only a successful ACTIVATE counts as
        # engagement (review Important 3) -- deleting is not "opting in",
        # and the artifact will not be active afterwards anyway, so
        # commit()'s active-check already keeps that case skip-safe.
        operation = self._operation
        self._operation = None
        if error is not None:
            self.notify(error, severity="error")
        else:
            if operation == "activate":
                self._acted_this_run = True
                self._discard_external_selection()
            self.notify(
                preference_error or "Speech model updated.",
                severity="warning" if preference_error else "information",
            )
        self._ensure_loaded(force=True)

    # -- persistence gate (AC#5 / review Important 3 & 4) -------------------
    async def commit(self) -> tuple[bool, str]:
        """Persist ``[transcription]`` defaults, but only when it is safe to.

        A verified external selection is the exception to the managed-runtime
        gate: its source and speech defaults are written atomically even when
        the optional runtime is absent, so setup can finish before installation.

        For managed selections, writes nothing -- returns the
        ok-but-skip result ``(True, "")`` -- unless ALL of the following
        hold, each freshly re-verified rather than trusted from stale
        widget state:

        * the ``onnx-asr`` runtime extra is importable (Important 4) --
          persisting a provider the runtime cannot execute is worse than no
          config change at all;
        * the exact selected Parakeet artifact is verified ACTIVE right now
          (``_check_active``, run off the event loop in an executor, never
          the possibly-stale ``self._installed_item``);
        * the user engaged this step THIS wizard run -- installed,
          activated, or used "use as default" (``self._acted_this_run``) --
          see ``first_run_speech_step_state.should_persist_speech_config``.

        That last condition is Important 3's core no-clobber guarantee: an
        artifact that merely happens to be active from an earlier session
        (for example, installed via the Library screen) is not, on its
        own, reason to overwrite whatever is already configured in
        ``[transcription]`` (``remote-whisper``, ``default_language="auto"``,
        ...) just because the user pressed Next through a re-run without
        touching this step.

        When it does write, provider/model/language/precision come
        from ``first_run_speech_step_state.resolve_speech_selection`` --
        the PRESSED language/precision radios (read via
        ``_effective_language``/``_effective_precision``), never a
        hardcoded recommendation (review finding 2; see that function's
        docstring for the fallback rules).

        Returns:
            ``(True, "")`` when nothing needed writing, or the write
            succeeded; ``(False, <message>)`` when work is still pending or
            preparing, writing, or accepting an external selection failed.
        """
        if self._external_busy:
            return False, "Wait for external model verification to finish."
        if self._pending_external_selection is not None:
            return await self._commit_external_selection()

        # Important 4: never persist a provider the runtime cannot execute,
        # even if somehow both active and acted (belt-and-suspenders; the
        # UI-side gate in _status_and_action is the primary defense --
        # mirrors RagStep's own commit() re-check of deps_installed()).
        if not self._runtime_installed():
            return True, ""
        selection = speech_state.resolve_speech_selection(
            selected_language=self._effective_language(),
            selected_precision=self._effective_precision(),
            curated_selections=self._curated_selections(),
        )
        if selection is None:
            return True, ""
        active_dir = await asyncio.get_running_loop().run_in_executor(
            None, self._check_active, selection
        )
        if not speech_state.should_persist_speech_config(
            active=active_dir is not None, acted_this_run=self._acted_this_run
        ):
            # Skip-safe: nothing verified active, OR the user never engaged
            # this step this run -- either way, leave [transcription] byte-
            # identical to whatever is already persisted (review Important 3).
            return True, ""
        ok = await self.wizard.commit_config(
            speech_state.build_speech_transcription_commit(
                provider_id=selection.provider_id,
                model_id=selection.model_id,
                language=selection.language,
                precision=selection.precision,
            )
        )
        return (
            (True, "")
            if ok
            else (False, "Saving the speech transcription choice failed.")
        )

    async def _commit_external_selection(self) -> tuple[bool, str]:
        """Atomically write speech defaults plus one prepared external source."""

        prepared = self._pending_external_selection
        if prepared is None:
            return True, ""
        selection = speech_state.resolve_speech_selection(
            selected_language=self._effective_language(),
            selected_precision=self._effective_precision(),
            curated_selections=self._curated_selections(),
        )
        if selection is None:
            return False, "The selected speech model is no longer available."
        try:
            expected_key = ParakeetSourceKey.from_values(
                selection.model_id,
                selection.precision,
            )
        except ValueError:
            return False, "The selected speech model cannot use an external directory."
        if prepared.key is not expected_key:
            return (
                False,
                "The external model selection changed. Choose the directory again.",
            )

        loop = asyncio.get_running_loop()
        service = self._source_service()
        token = self._external_selection_token
        self._external_commit_detached = False
        self._external_commit_pending = True
        self.refresh(recompose=True)
        try:
            try:
                source_commit = await loop.run_in_executor(
                    None,
                    service.prepare_config_commit,
                    prepared,
                )
            except Exception as exc:
                logger.warning(
                    "External Parakeet config preparation failed; error_type={}",
                    type(exc).__name__,
                )
                return (
                    False,
                    "The external model or managed VAD changed. Choose the directory again.",
                )
            if token is not None and token != self._external_selection_token:
                self._release_external_scope(token)
                return False, "The external model selection changed. Choose it again."

            patch = speech_state.speech_config_patch(selection, source_commit)
            handoff = asyncio.create_task(
                self.wizard.commit_config(
                    patch,
                    after_write=lambda: service.accept_committed(source_commit),
                )
            )
            self._external_commit_handoff = handoff
            cancelled = False

            async def settle(operation: asyncio.Future[Any]) -> Any:
                nonlocal cancelled
                while True:
                    try:
                        return await asyncio.shield(operation)
                    except asyncio.CancelledError:
                        if operation.cancelled():
                            raise
                        cancelled = True
                        self._external_commit_detached = True
                        task = asyncio.current_task()
                        if task is not None:
                            task.uncancel()

            try:
                ok = await settle(handoff)
            except Exception as handoff_error:
                logger.warning(
                    "External Parakeet commit handoff failed; error_type={}",
                    type(handoff_error).__name__,
                )
                retry = loop.run_in_executor(
                    None,
                    service.accept_committed,
                    source_commit,
                )
                try:
                    await settle(retry)
                except Exception as retry_error:
                    logger.warning(
                        "External Parakeet commit reconciliation failed; error_type={}",
                        type(retry_error).__name__,
                    )
                    message = (
                        "The external source was saved, but it could not be activated "
                        "in this session. Restart the app, then retry setup."
                    )
                    self._external_status = message
                    return False, message
                ok = True

            if not ok:
                return False, "Saving the speech transcription choice failed."

            if token is not None:
                self._release_external_scope(token)
            if self._external_selection_token == token:
                self._external_selection_token = None
            self._pending_external_selection = None
            if cancelled:
                raise asyncio.CancelledError
            self._external_status = (
                "External source ready."
                if self._runtime_installed()
                else "Runtime required"
            )
            return True, ""
        finally:
            self._external_commit_handoff = None
            self._external_commit_pending = False
            if self.is_mounted:
                self.refresh(recompose=True)

    def _check_active(self, selection: speech_state.SpeechSelection) -> Any:
        try:
            return active_managed_parakeet_dir(
                selection.model_id,
                selection.precision,
                service=self._service_for_worker(),
            )
        except Exception:
            logger.opt(exception=True).error("Speech setup active-check failed")
            return None


class ToolsStep(SetupStep):
    """Enable built-in tools (all default OFF; risk-tagged ones still ask per call)."""

    def compose_step(self) -> ComposeResult:
        from tldw_chatbook.Agents.tool_catalog import gateable_builtin_tools

        self._entries = list(gateable_builtin_tools())
        # Re-run prefill: resurface whatever gates are already on instead of
        # always showing OFF. First-run behavior is unchanged, since a fresh
        # app_config has no "tools" section and tool_gates comes back empty.
        prefill = wizard_state.read_wizard_prefill(
            getattr(self.wizard.app_instance, "app_config", {}) or {}
        )
        gate_values = dict(prefill.tool_gates)
        with Vertical(classes="setup-tools"):
            yield Static("Built-in tools", classes="setup-title")
            yield Static(
                "Everything is off by default. Tools that read or change your "
                "files still show an approval card every time they run.",
                classes="setup-subtitle",
            )
            for entry in self._entries:
                title, desc = self._TOOL_COPY.get(
                    entry.tool_name,
                    (entry.tool_name.replace("_", " ").capitalize(), ""),
                )
                with Horizontal(classes="setup-tool-row"):
                    yield Switch(
                        value=gate_values.get(entry.gate_key, False),
                        id=f"setup-tool-{entry.tool_name}",
                    )
                    with Vertical(classes="setup-tool-text"):
                        yield Label(title, classes="setup-tool-name")
                        yield Static(
                            desc,
                            id=f"setup-tool-desc-{entry.tool_name}",
                            classes="setup-tool-desc",
                            markup=False,
                        )
            yield Static("", classes="setup-step-error")

    # TASK-1501: plain-language names and one-line descriptions per built-in
    # tool. The ⚠ marks tools that create or change data on disk — a static
    # judgment mirroring each tool's risk_tags without importing the tool
    # modules at compose time. An unknown (future) tool degrades to its
    # capitalized name with no description rather than breaking the step.
    _TOOL_COPY = {
        "read_file": ("Read file", "Read a file you point the assistant at."),
        "list_directory": ("List directory", "Browse the contents of a folder."),
        "write_file": ("Write file", "⚠ Creates or overwrites files on disk."),
        "create_note": ("Create note", "⚠ Adds new notes to your notebook."),
        "update_note": ("Update note", "⚠ Edits your existing notes."),
        "glob_files": ("Find files", "Match file names by pattern (like *.md)."),
        "grep_files": ("Search in files", "Search inside files for text."),
        "expand_document": (
            "Expand document",
            "Read the whole document behind a search result.",
        ),
    }

    def gate_key_for(self, switch: Switch) -> str:
        tool_name = (switch.id or "").removeprefix("setup-tool-")
        for entry in self._entries:
            if entry.tool_name == tool_name:
                return entry.gate_key
        return ""

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            build_tools_commit,
            read_wizard_prefill,
            tools_commit_delta,
        )

        # Every switch's current value, on or off -- delta-aware commit
        # needs to see OFF switches too, to catch an ON->OFF transition
        # against a re-run's prefilled config (Task 11 prefills these
        # switches from persisted gates; a bare "only persist enables"
        # filter can never write a disable, so re-run could not turn a
        # gate back off).
        gate_values: dict[str, bool] = {}
        for switch in self.query(Switch):
            gate_key = self.gate_key_for(switch)
            if gate_key:
                gate_values[gate_key] = bool(switch.value)
        current_gates = dict(
            read_wizard_prefill(
                getattr(self.wizard.app_instance, "app_config", {}) or {}
            ).tool_gates
        )
        delta = tools_commit_delta(gate_values=gate_values, current_gates=current_gates)
        if not delta:
            return True, ""
        ok = await self.wizard.commit_config(build_tools_commit(gate_values=delta))
        return (True, "") if ok else (False, "Saving tool settings failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {
            "enabled_gates": [
                self.gate_key_for(sw) for sw in self.query(Switch) if sw.value
            ]
        }


class NotesSyncStep(SetupStep):
    """Explain where reviewed lasting folder sync is configured."""

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-notes"):
            yield Static("Notes folder sync", classes="setup-title")
            yield Static(
                "After setup, use Library → Notes → Add from files… to review a folder before activating sync.",
                classes="setup-subtitle",
            )
            yield Static(
                "Nothing is activated during first-run setup.",
                classes="setup-step-error",
            )

    async def commit(self) -> tuple[bool, str]:
        return True, ""

    def get_step_data(self) -> Dict[str, Any]:
        return {}


class AppearanceStep(SetupStep):
    """Theme and splash card. Applies the theme live on commit (best effort)."""

    selected_theme: str = ""
    selected_splash_card: str = ""
    # Bug-2 fix: True only when the user EXPLICITLY re-picked "Surprise me"
    # this run (see _on_card) -- distinct from selected_splash_card=="",
    # which is ALSO true on a fresh mount where nothing was ever chosen
    # (RadioSet does not fire Changed for its own initial pre-selection).
    _picked_surprise_me: bool = False

    def compose_step(self) -> ComposeResult:
        # Re-run prefill: pre-select the theme RadioButton matching the
        # persisted default_theme, when it's in the rendered list. First-run
        # has no general.default_theme, so prefill.default_theme is "" and
        # nothing matches -- identical to the old always-unselected render.
        prefill = wizard_state.read_wizard_prefill(
            getattr(self.wizard.app_instance, "app_config", {}) or {}
        )
        # Bug-2a fix: initialize selected_theme from the persisted value.
        # RadioSet does not emit Changed for its own initial pre-selection
        # (only _on_theme below updates selected_theme), so without this a
        # rerun that never touches the theme radio left selected_theme=="",
        # and commit()'s old "fall back to textual-dark" default would
        # clobber the persisted theme just because some OTHER field (e.g.
        # only the splash card) changed on this step.
        self.selected_theme = prefill.default_theme
        with Vertical(classes="setup-appearance"):
            yield Static("Appearance", classes="setup-title")
            yield Label("Theme", classes="setup-field-label")
            with RadioSet(id="setup-theme-choice", classes="setup-choice-list"):
                yield from self._theme_buttons(self._theme_shortlist())
            yield Button(
                "Show all themes…",
                id="setup-theme-show-all",
                classes="setup-tertiary-button",
            )
            yield Label("Splash screen card", classes="setup-field-label")
            with RadioSet(id="setup-splash-choice", classes="setup-choice-list"):
                yield SetupRadioButton("Surprise me (random)", value=True)
                for card_name in self._card_names()[:10]:
                    yield SetupRadioButton(card_name)
            yield Static("", classes="setup-step-error")

    def _theme_buttons(self, names: list[str]):
        """Radio rows for theme names, marking the persisted one "(current)".

        TASK-1500: like the model rows, the label may carry decoration; the
        clean theme name rides on the button as ``_theme_name`` so previews
        and commits never see display text.
        """
        for theme_name in names:
            label = (
                f"{theme_name}   (current)"
                if theme_name == self.selected_theme and theme_name
                else theme_name
            )
            button = SetupRadioButton(label, value=(theme_name == self.selected_theme))
            button._theme_name = theme_name
            yield button

    # TASK-1500: flagship candidates for the shortlist, in preference order.
    # Filtered against what this Textual build actually registers; the two
    # stock themes are always present.
    _FLAGSHIP_THEMES = ("nord", "gruvbox", "tokyo-night", "catppuccin-mocha")

    def _theme_names(self) -> list[str]:
        try:
            return sorted(self.app.available_themes)
        except Exception:
            return ["textual-dark", "textual-light"]

    def _theme_shortlist(self) -> list[str]:
        """Curated first screen: current + stock defaults + a few flagships.

        The full alphabetical wall (novelty themes first) buried the sane
        choices; "Show all themes…" swaps in the complete list on demand.
        """
        available = self._theme_names()
        shortlist: list[str] = []
        for name in (
            self.selected_theme,
            "textual-dark",
            "textual-light",
            *self._FLAGSHIP_THEMES,
        ):
            if name and name in available and name not in shortlist:
                shortlist.append(name)
        return shortlist or available[:6]

    @on(Button.Pressed, "#setup-theme-show-all")
    async def _on_show_all_themes(self, event: Button.Pressed) -> None:
        event.stop()
        radio_set = self.query_one("#setup-theme-choice", RadioSet)
        await radio_set.remove_children()
        await radio_set.mount_all(self._theme_buttons(self._theme_names()))
        self.query_one("#setup-theme-show-all", Button).display = False

    @staticmethod
    def _card_names() -> list[str]:
        try:
            from tldw_chatbook.Utils.Splash_Screens.card_definitions import (
                get_all_card_definitions,
            )

            return sorted(get_all_card_definitions())
        except Exception:
            return []

    #: Theme active before the first preview; None = nothing to revert.
    _preview_original: Optional[str] = None

    @on(RadioSet.Changed, "#setup-theme-choice")
    def _on_theme(self, event: RadioSet.Changed) -> None:
        if event.pressed is None:
            return
        # Clean value, never the "(current)"-decorated label.
        self.selected_theme = str(
            getattr(event.pressed, "_theme_name", event.pressed.label)
        )
        self._preview_theme(self.selected_theme)

    def _preview_theme(self, theme_name: str) -> None:
        """TASK-1500: selecting a theme applies it immediately as a preview.

        The pre-preview theme is remembered once so `revert_preview` can
        restore it if the user backs out (finish-later) without committing.
        A successful commit clears the revert obligation — the new theme is
        then the persisted one.
        """
        if not theme_name:
            return
        try:
            if self._preview_original is None:
                self._preview_original = str(self.app.theme)
            self.app.theme = theme_name
        except Exception:
            logger.debug("Theme preview failed for %s", theme_name, exc_info=True)

    def revert_preview(self) -> None:
        """Restore the pre-preview theme (no-op when nothing was previewed)."""
        if self._preview_original is not None:
            try:
                self.app.theme = self._preview_original
            except Exception:
                logger.debug("Theme preview revert failed", exc_info=True)
            self._preview_original = None

    @on(RadioSet.Changed, "#setup-splash-choice")
    def _on_card(self, event: RadioSet.Changed) -> None:
        label = str(event.pressed.label)
        if label.startswith("Surprise me"):
            self.selected_splash_card = ""
            self._picked_surprise_me = True
        else:
            self.selected_splash_card = label
            self._picked_surprise_me = False

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            build_appearance_commit,
            read_wizard_prefill,
        )

        prefill = read_wizard_prefill(
            getattr(self.wizard.app_instance, "app_config", {}) or {}
        )
        # Bug-2c fix: only reset to "random" when the user EXPLICITLY
        # re-picked Surprise-me this run over a config that currently names
        # a specific card -- a fresh/no-op run (nothing pressed, or already
        # "random") must not write anything.
        reset_to_random = (
            self._picked_surprise_me
            and bool(prefill.card_selection)
            and prefill.card_selection != "random"
        )
        if (
            not self.selected_theme
            and not self.selected_splash_card
            and not reset_to_random
        ):
            return True, ""
        # Bug-2b fix: delta-aware theme write -- only persist default_theme
        # when the chosen theme actually differs from what's already on
        # disk, so a rerun that only changes the splash card (theme radio
        # left at its prefilled, already-persisted position) leaves the
        # persisted theme untouched instead of rewriting it (or a stale
        # "textual-dark" fallback) back over itself.
        chosen_theme = self.selected_theme or "textual-dark"
        theme_to_persist = (
            chosen_theme if chosen_theme != prefill.default_theme else None
        )
        ok = await self.wizard.commit_config(
            build_appearance_commit(
                default_theme=theme_to_persist,
                splash_card=self.selected_splash_card or None,
                reset_splash_to_random=reset_to_random,
            )
        )
        if ok and self.selected_theme:
            try:
                self.app.theme = self.selected_theme
            except Exception:
                logger.debug("Live theme apply failed; persisted value still wins")
            # TASK-1500: the commit made the previewed theme real — nothing
            # to revert on cancel any more.
            self._preview_original = None
        return (True, "") if ok else (False, "Saving appearance settings failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {"theme": self.selected_theme, "splash_card": self.selected_splash_card}


class WelcomeStep(SetupStep):
    """Track choice: Quick / Full / Skip."""

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-welcome"):
            yield Static("Welcome to tldw chatbook", classes="setup-title")
            yield Static(
                "Let's get you set up. Pick a path — everything here can be "
                "changed later in Settings, and every step can be skipped.",
                classes="setup-subtitle",
            )
            with RadioSet(id="setup-track-choice", classes="setup-choice-list"):
                # TASK-2154.9 (FR-02): name the steps the tracker will show
                # (Welcome is this one; Provider, Model and Summary follow)
                # so the "Step 1 of 4" count is not a surprise after picking
                # what read as a two-item "provider & model" track.
                yield SetupRadioButton(
                    "Quick setup — provider, model, voice & summary (recommended)",
                    value=True,
                    id="setup-track-quick",
                )
                yield SetupRadioButton(
                    "Full setup — configure everything", id="setup-track-full"
                )
            yield Static("", classes="setup-step-error")

    def get_step_data(self) -> Dict[str, Any]:
        return {"track": self.chosen_track()}

    def chosen_track(self) -> str:
        try:
            full = self.query_one("#setup-track-full", RadioButton).value
        except Exception:
            full = False
        return wizard_state.TRACK_FULL if full else wizard_state.TRACK_QUICK


class ProtectKeysStep(SetupStep):
    """Offer config encryption for any keys entered this run.

    Encryption goes only through the existing mechanism: PasswordDialog
    (setup mode) collects the password, enable_config_encryption(password)
    does the actual rewrite under the config RLock. This step never rolls
    its own crypto.
    """

    def __init__(self, wizard=None, config=None, *, enable_encryption=None, **kwargs):
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._enable_encryption = enable_encryption
        self.encryption_enabled = False

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-protect"):
            yield Static("Protect your keys", classes="setup-title")
            yield Static(
                "Encrypt the API keys in your config file with a password. "
                "You'll be asked for this password each time chatbook starts. "
                "Skip to leave keys as plain text (you can enable this later "
                "in Settings ▸ Privacy & Security).",
                classes="setup-subtitle",
            )
            yield Button(
                "Set a password", id="setup-protect-set-password", variant="primary"
            )
            yield Static("", id="setup-protect-status", classes="setup-probe-status")
            yield Static("", classes="setup-step-error")

    @on(Button.Pressed, "#setup-protect-set-password")
    def _on_set_password(self) -> None:
        from tldw_chatbook.Widgets.password_dialog import PasswordDialog

        # Mirrors the only other setup-mode caller,
        # Tools_Settings_Window.py's _setup_encryption (~line 7309):
        #   PasswordDialog(mode="setup", on_submit=lambda p: None,
        #                  on_cancel=lambda: None)
        # That caller does not override title/message -- it relies on
        # PasswordDialog's own mode="setup" defaults ("Setup Master
        # Password" / "Create a master password to encrypt your API keys
        # and sensitive configuration data."). Its on_submit/on_cancel are
        # no-ops (the real work happens after dismiss, same as here), so
        # they add nothing; this uses the push_screen(dialog, callback)
        # idiom already established in this module (see
        # FirstRunSetupWizard.action_cancel's ConfirmationDialog) instead of
        # that caller's await/wait_for_dismiss=True style -- both dispatch
        # through the same ModalScreen.dismiss(password), so the two forms
        # are behaviorally identical here.
        dialog = PasswordDialog(mode="setup")
        self.app.push_screen(dialog, self._on_password_result)

    def _on_password_result(self, password: str | None) -> None:
        if not password:
            return
        # Deviation from the task brief: the brief's pseudocode runs this
        # worker in group "setup-wizard-advance", but that group name is
        # SetupWizardContainer's OWN commit-on-Next / finalize worker
        # (handle_next, _skip_entirely, _finalize all use it, each
        # exclusive=True). Reusing it here would let this step's worker
        # collide with the container's -- exclusive=True workers in the same
        # group cancel/replace each other, so a password-apply in flight
        # could be cancelled by a Next click, or vice versa. A dedicated
        # group avoids that; the actual serialization guarantee against
        # concurrent config writes is enable_config_encryption's own config
        # RLock, not the worker group name.
        self.run_worker(
            self._apply_password_worker(password),
            exclusive=True,
            group="setup-protect-encrypt",
        )

    async def _apply_password_worker(self, password: str) -> None:
        ok = await self.apply_password(password)
        status = self.query_one("#setup-protect-status", Static)
        if ok:
            status.update("✓ Encryption enabled.")
        else:
            self.show_step_error(
                "Enabling encryption failed — your keys are unchanged (plain text)."
            )

    async def apply_password(self, password: str) -> bool:
        import asyncio

        enable = self._enable_encryption
        if enable is None:
            from tldw_chatbook.config import enable_config_encryption

            enable = enable_config_encryption
        ok = bool(
            await asyncio.get_running_loop().run_in_executor(None, enable, password)
        )
        self.encryption_enabled = ok
        return ok

    def get_step_data(self) -> Dict[str, Any]:
        return {"encryption_enabled": self.encryption_enabled}


class SummaryStep(SetupStep):
    """Read-back ✓/✗ matrix plus mode-dependent exits.

    Always re-reads the persisted config (never step memory) so the summary
    reflects what actually landed on disk, not what the in-memory steps
    think they committed.
    """

    def __init__(
        self,
        wizard=None,
        config=None,
        *,
        load_config=None,
        rag_deps_installed=None,
        speech_installed=None,
        speech_runtime_installed=None,
        **kwargs,
    ):
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._load_config = load_config
        self._rag_deps_installed = rag_deps_installed
        # TASK-1301 AC#6: same injectable-callable shape as rag_deps_installed
        # -- defaults to a real, off-loop-safe check of the configured exact
        # managed Parakeet artifact's installed/active state.
        self._speech_installed = speech_installed
        # Review Important 4 residual: same shape again -- defaults to the
        # real onnx-asr runtime probe so Summary agrees with the Speech
        # step's own runtime gate instead of only checking files-on-disk.
        self._speech_runtime_installed = speech_runtime_installed
        self.exit_route: Optional[str] = None
        self.provider_model_complete = False

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-summary"):
            yield Static("Setup summary", classes="setup-title")
            yield Static("", id="setup-summary-defaults-note", classes="setup-subtitle")
            # markup=False: row labels/details come from persisted config data
            # (embedding model ids, notes directories, ...) which may contain
            # literal "[...]" -- Static.update() otherwise parses that as Rich
            # markup and silently drops it from the rendered text.
            yield Static("", id="setup-summary-rows", markup=False)
            yield Static(
                "", id="setup-summary-footer", classes="setup-subtitle", markup=False
            )
        # The exit actions are a DIRECT child of the step (the .setup-step
        # scroll container), not of the scrolling .setup-summary Vertical:
        # Textual docks position against the container's visible frame and
        # never scroll with content, which is what keeps the wizard's final
        # CTAs on screen no matter how tall the read-back matrix gets
        # (TASK-1495 AC #3 -- full-track content previously pushed them
        # below the fold at 120x40).
        with Horizontal(classes="setup-summary-actions"):
            yield Button(
                "Review provider setup", id="setup-exit-chat", variant="primary"
            )
            yield Button("Explore Home", id="setup-exit-home")
            yield Button("Review settings", id="setup-exit-settings")

    def on_show(self) -> None:
        super().on_show()
        track = (
            (self.wizard.wizard_data or {})
            .get(wizard_state.STEP_WELCOME, {})
            .get("track")
        )
        if track == wizard_state.TRACK_QUICK:
            self.query_one("#setup-summary-defaults-note", Static).update(
                "Left at recommended defaults: tools off, RAG off, default theme, "
                "notes sync off — each lives in Settings when you want it."
            )
        self.run_worker(self._render_rows(), exclusive=True, group="setup-summary-load")

    async def _render_rows(self) -> None:
        import asyncio

        load = self._load_config
        if load is None:
            from tldw_chatbook.config import load_cli_config_and_ensure_existence

            def load():
                return load_cli_config_and_ensure_existence(force_reload=True)

        config = await asyncio.get_running_loop().run_in_executor(None, load)

        deps = self._rag_deps_installed
        if deps is None:
            from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

            deps = embeddings_rag_deps_installed
        speech_installed_check = self._speech_installed
        if speech_installed_check is None:
            prefill = speech_state.read_speech_prefill(config)
            selection = speech_state.recommended_speech_selection()
            if (
                prefill.provider_id
                == speech_state.routing_policy().parakeet_provider_id
            ):
                resolved = speech_state.resolve_speech_selection(
                    selected_language=prefill.language,
                    selected_precision=prefill.precision or "int8",
                    curated_selections=SpeechSetupStep._curated_selections(),
                )
                if resolved is not None and resolved.model_id == prefill.model_id:
                    selection = resolved

            def speech_installed_check() -> bool:
                # Minor 12: a Quick-track user (who never saw the Speech
                # step) reaching Summary must not cause the managed
                # artifact store's directories to be created on disk --
                # constructing a real ModelArtifactService (what
                # active_managed_parakeet_dir() does internally) mkdirs
                # unconditionally. A read-only existence check first means
                # "nothing was ever installed by anyone" costs zero
                # filesystem writes; only go on to the real check once
                # something has legitimately created the root already.
                if not managed_model_artifact_root().exists():
                    return False
                return (
                    active_managed_parakeet_dir(
                        selection.model_id,
                        selection.precision,
                    )
                    is not None
                )

        speech_runtime_check = self._speech_runtime_installed
        if speech_runtime_check is None:
            from tldw_chatbook.Utils.optional_deps import parakeet_onnx_deps_installed

            speech_runtime_check = parakeet_onnx_deps_installed

        speech_installed = await asyncio.get_running_loop().run_in_executor(
            None, speech_installed_check
        )
        speech_runtime_installed = await asyncio.get_running_loop().run_in_executor(
            None, speech_runtime_check
        )
        from tldw_chatbook.UI.Wizards.first_run_setup_state import build_summary_rows

        rows = build_summary_rows(
            config,
            dict(os.environ),
            rag_deps_installed=deps(),
            speech_installed=speech_installed,
            speech_runtime_installed=speech_runtime_installed,
        )
        row_states = {row.label: row.state for row in rows}
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            ROW_CONFIGURED,
            build_first_run_summary_actions,
        )

        primary, _, _ = build_first_run_summary_actions(
            provider_configured=row_states.get("Provider") == ROW_CONFIGURED,
            model_configured=row_states.get("Default model") == ROW_CONFIGURED,
        )
        self.provider_model_complete = primary == "start_chatting"
        primary_button = self.query_one("#setup-exit-chat", Button)
        primary_button.label = (
            "Start chatting"
            if self.provider_model_complete
            else "Review provider setup"
        )
        primary_button.tooltip = (
            "Open Console with this provider and model."
            if self.provider_model_complete
            else "Return to Provider and finish the connection and model setup."
        )
        # Static.update() parses "[...]" as Rich markup by default, so any
        # bracketed literal in a label/detail (e.g. a package extra name)
        # must be escaped or it silently vanishes from the rendered text.
        lines = [
            f"{row.glyph} {row.label}" + (f" — {row.detail}" if row.detail else "")
            for row in rows
        ]
        # TASK-1266: steps dropped by the compose-crash policy get a reasoned
        # row — the matrix must reflect that an area was never presented, not
        # silently omit it.
        failed_titles = []
        try:
            failed_titles = self.wizard.compose_failed_steps()
        except Exception:
            logger.debug("compose_failed_steps unavailable", exc_info=True)
        lines.extend(
            f"✗ {title} — step couldn't be shown (skipped); configure in Settings"
            for title in failed_titles
        )
        self.query_one("#setup-summary-rows", Static).update("\n".join(lines))
        from tldw_chatbook.config import get_cli_config_path

        # F-D fix: resolving the path and updating the widget were one bare
        # try/except Exception: pass -- ANY failure in either half (a
        # get_cli_config_path() error, or the query_one below) left the
        # footer exactly as compose() first rendered it (""), so the label
        # itself never even appeared, and any real failure vanished with no
        # trace. Resolve the path in its own guarded step with a visible
        # fallback string, so the footer's "Config file:" line always shows
        # SOMETHING and a genuine resolution failure is at least logged
        # instead of silently producing an empty-looking row.
        try:
            config_path_text = str(get_cli_config_path())
        except Exception:
            logger.warning(
                "Summary footer could not resolve the config path", exc_info=True
            )
            config_path_text = "(unknown — see Settings ▸ Diagnostics)"
        try:
            self.query_one("#setup-summary-footer", Static).update(
                f"Config file: {config_path_text}\n"
                "Re-run setup any time: Settings ▸ Diagnostics ▸ Run setup wizard."
            )
        except Exception:
            logger.debug("Summary footer widget unavailable to update", exc_info=True)

    @on(Button.Pressed, "#setup-exit-chat")
    def _exit_chat(self) -> None:
        if not self.provider_model_complete:
            self.wizard.review_provider_setup()
            return
        from tldw_chatbook.Constants import TAB_CHAT

        self._finish(TAB_CHAT)

    @on(Button.Pressed, "#setup-exit-home")
    def _exit_home(self) -> None:
        from tldw_chatbook.Constants import TAB_HOME

        self._finish(TAB_HOME)

    @on(Button.Pressed, "#setup-exit-settings")
    def _exit_settings(self) -> None:
        self.wizard.open_provider_settings()

    def _finish(self, exit_route: Optional[str]) -> None:
        self.exit_route = exit_route
        # Deviation from the task brief: the brief calls
        # self.wizard.handle_next() directly, but SetupWizardContainer's
        # handle_next is the @on(Button.Pressed, "#wizard-next") override
        # documented above -- it takes the Button.Pressed event and calls
        # event.prevent_default() on it (required so the base class's own
        # handle_next() doesn't ALSO fire per Textual's whole-MRO @on
        # dispatch; see that method's docstring/comment). Calling
        # handle_next() with no event, or with None, would raise on
        # event.prevent_default(). advance_programmatically() is the
        # extracted body (guard + worker dispatch) with no event
        # dependency, used by both the real button handler and this
        # programmatic exit path, so the dispatch semantics for the actual
        # Next button are unchanged.
        self.wizard.advance_programmatically()

    def get_step_data(self) -> Dict[str, Any]:
        return {"exit_route": self.exit_route}


class _ProviderSaveStatus(Static):
    """Focusable live status for an irreversible provider save."""

    can_focus = True


class SetupWizardContainer(WizardContainer):
    """Navigates over the active-step subset; commits on Next via one worker."""

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
            owner.is_mounted
            and current_key is not None
            and (
                owner._selected_discovery_key != current_key
                or owner._selected_discovery_state not in {"in_progress", "complete"}
            )
        ):
            owner._begin_selected_provider_discovery(
                provider_draft,
                sync_live_credential=False,
            )
        model_index = self._step_index_for_id(wizard_state.STEP_MODEL)
        if model_index is None:
            return
        model_step = self.steps[model_index]
        if isinstance(model_step, ModelStep) and model_step.is_mounted:
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
            if isinstance(owner, ProviderStep) and owner.is_mounted:
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

        if isinstance(owner, ProviderStep) and owner.is_mounted:
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
                if isinstance(owner, ProviderStep) and owner.is_mounted:
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
                    and owner.is_mounted
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
        yield WizardNavigation(classes="wizard-navigation")

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
            hints.update("Ctrl+B back")
        elif step_id == wizard_state.STEP_WELCOME:
            cancel.label = "Skip setup"
            cancel.tooltip = (
                "Close setup and stop showing it at launch. You can rerun it "
                "from Settings ▸ Diagnostics."
            )
            hints.update("Ctrl+N next · Ctrl+B back · Esc skip setup")
        else:
            cancel.label = "Exit setup"
            cancel.tooltip = (
                "Save completed steps and continue later from Settings ▸ Diagnostics."
            )
            hints.update("Ctrl+N next · Ctrl+B back · Esc exit setup")

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
            or not current.is_mounted
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
            if isinstance(voice_step, VoiceSetupStep) and voice_values:
                initial = voice_step._initial_draft()
                restored_voice = voice_state.VoiceSetupDraft(
                    endpoint=str(voice_values.get("endpoint", initial.endpoint)),
                    authentication_mode=str(
                        voice_values.get(
                            "authentication_mode",
                            initial.authentication_mode,
                        )
                    ),
                    model_id=str(voice_values.get("model_id", initial.model_id)),
                    voice_id=str(voice_values.get("voice_id", initial.voice_id)),
                    response_format=str(
                        voice_values.get("response_format", initial.response_format)
                    ),
                    speed=float(voice_values.get("speed", initial.speed)),
                    sample_text=str(
                        voice_values.get("sample_text", initial.sample_text)
                    ),
                    use_as_default=bool(
                        voice_values.get("use_as_default", initial.use_as_default)
                    ),
                )
                voice_step._custom_draft = restored_voice
                voice_step._preset = voice_state.VOICE_PRESET_CUSTOM
                voice_step._apply_draft_to_controls(restored_voice)
                self._restore_radio_selection(
                    voice_step.query_one("#setup-voice-preset", RadioSet),
                    lambda button: button.id == "setup-voice-preset-custom",
                )

            rag_values = draft.values.get(wizard_state.STEP_RAG, {})
            rag_step = self.steps[self._step_index_for_id(wizard_state.STEP_RAG)]
            if isinstance(rag_step, RagStep) and "embedding_model" in rag_values:
                embedding_model = str(rag_values["embedding_model"])
                rag_step.selected_embedding_model = embedding_model
                self._restore_radio_selection(
                    rag_step.query_one("#setup-rag-model-choice", RadioSet),
                    lambda button: str(button.label) == embedding_model,
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
                        lambda button: (
                            str(button.label) == splash_card
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
            nav.current_step = position + 1
            nav.can_go_back = position > 0
            nav.can_go_forward = self.can_proceed
            self._rebuild_progress()
        except Exception:
            pass

    def _rebuild_progress(self) -> None:
        """Refresh the setup-specific tracker from the active-track projection."""
        try:
            items = wizard_state.build_setup_progress(
                self.active_ids,
                self._active_position(self.current_step or 0),
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
            self.run_worker(
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
            self.run_worker(
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
            self.run_worker(
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
            or not self.is_mounted
            or not step.is_mounted
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
            and self.is_mounted
            and action.screen.is_mounted
            and same_screen
            and same_step
            and (not require_step_mounted or action.step.is_mounted)
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
                and self.is_mounted
                and action.screen.is_mounted
                and self.screen is action.screen
                and action.screen.app.screen is action.screen
                and action.screen.query_one(SetupWizardContainer) is self
                and self.current_step == action.index
                and self.steps[action.index] is replacement
                and replacement.is_mounted
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
                self.is_mounted
                and action.screen.is_mounted
                and self.screen is action.screen
                and action.screen.app.screen is action.screen
                and action.screen.query_one(SetupWizardContainer) is self
            )
        except Exception:
            return
        if same_screen and current.is_mounted:
            self._sync_action_controls()

    def _sync_action_controls(self) -> None:
        blocked = (
            self._advancing
            or self._failure_action_running
            or self._provider_dismiss_pending
        )
        try:
            if blocked:
                for selector in ("#wizard-back", "#wizard-next", "#wizard-cancel"):
                    self.query_one(selector, Button).disabled = True
            else:
                self.update_progress()
                self.query_one(
                    ".wizard-navigation", WizardNavigation
                ).update_button_states()
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
                and self.is_mounted
                and action.screen.is_mounted
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
        """
        if self._advancing or self._failure_action_running or not self.can_proceed:
            return
        self._set_advancing(True)
        self.run_worker(self._advance(), exclusive=True, group="setup-wizard-advance")

    def _set_advancing(self, active: bool) -> None:
        """Fence navigation while a step's config handoff is settling."""

        self._advancing = active
        self._sync_action_controls()

    async def _advance(self) -> None:
        try:
            step = self.steps[self.current_step]
            if isinstance(step, SetupStep):
                if step.compose_failure is not None and step.required:
                    return
                ok, error = await step.commit()
                if not ok:
                    step.show_step_error(f"{error}  (Retry, or Skip this step.)")
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
            else:
                self.show_step(next_index)
        finally:
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

        self.run_worker(
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
        self.run_worker(
            self._skip_entirely(), exclusive=True, group="setup-wizard-advance"
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
        # F-B fix: BaseWizard.complete_wizard() calls this callback
        # SYNCHRONOUSLY (self.on_complete(self.wizard_data)), and it is
        # itself invoked synchronously from _advance() -- which is the body
        # of the currently-RUNNING "setup-wizard-advance" worker (Summary's
        # own commit() has no real await, so nothing yields control back to
        # the event loop between _advance() starting and reaching here).
        # Scheduling _finalize into that SAME exclusive group from inside it
        # asks Textual's WorkerManager.add_worker to cancel_group() the
        # group it is currently executing -- i.e. cancel its own in-flight
        # worker (confirmed via CPython's Task.__step_run_and_handle_result:
        # a task whose coro returns normally while _must_cancel is set gets
        # forced into the CANCELLED state anyway, "Task is cancelled right
        # before coro stops"). A separately-created task happens to survive
        # that regardless, which is why this was not visibly broken in
        # testing -- but it is the same "worker schedules another worker
        # into its own exclusive group" hazard ProtectKeysStep's
        # _on_password_result already reasons about avoiding (see its
        # comment) by using a dedicated group; do the same here rather than
        # relying on a scheduling accident.
        self.run_worker(
            self._finalize(exit_route), exclusive=True, group="setup-wizard-finalize"
        )

    async def _finalize(self, exit_route: Optional[str]) -> None:
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
        self._dismiss_screen({"completed": True, "exit_route": exit_route})

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
            self.run_worker(
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

        if self._provider_ui_detached or not self.is_mounted:
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
        if self._provider_ui_detached or not self.is_mounted:
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
        if not self.is_mounted:
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
        if isinstance(owner, ProviderStep) and owner.is_mounted:
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
        self._opened_at = time.monotonic()

    async def action_cancel_dialog_if_settled(self) -> None:
        if (
            self._opened_at is not None
            and (time.monotonic() - self._opened_at) < self._escape_grace_seconds
        ):
            return  # too soon to be a deliberate second press -- swallow it
        await self.action_cancel_dialog()


class FirstRunSetupWizard(WizardScreen):
    """Full-screen first-run setup wizard. Dismisses dict | None."""

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
            "Ctrl+N next · Ctrl+B back · Esc skip setup",
            id="setup-key-hints",
            classes="setup-key-hints",
        )

    def on_mount(self) -> None:
        if not self.rerun:
            self._persist_started_flag()

    @work(thread=True, group="setup-wizard-started-flag")
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
                container.run_worker(
                    container._skip_entirely(),
                    exclusive=True,
                    group="setup-wizard-advance",
                )
                return
            self.run_worker(
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
            or not current.is_mounted
            or not current.display
            or not current.visible
        ):
            return
        self.run_worker(
            container.clear_resume_attempt(target_step_id),
            exclusive=True,
            group="setup-wizard-resume-clear",
        )
