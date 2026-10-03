"""The first-run wizard's Provider step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
from collections import OrderedDict
from dataclasses import (
    dataclass,
    field,
)
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
)

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import (
    Horizontal,
    Vertical,
)
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    Label,
    OptionList,
    Static,
)
from textual.widgets.option_list import Option

from tldw_chatbook.UI.Wizards import first_run_model_discovery as model_discovery
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards import first_run_step_guard as step_guard
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.first_run_model_discovery import (
    _first_run_discovery_staged_settings,
    _model_ids_from_discovery_result,
)
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    ProviderChoiceOption,
    SetupStep,
)
from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_provider_support import ConsoleProviderCatalogEntry
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardContainer


CLOUD_PROBE_TIMEOUT_SECONDS = 8.0


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

    def __reduce__(self) -> object:
        raise TypeError("Provider credentials are memory-only.")

    def __reduce_ex__(self, protocol: int) -> object:
        # `pickle` consults `__reduce_ex__` first, so sealing only
        # `__reduce__` still let `pickle.dumps` emit the plaintext key --
        # the gap between this class and its `ProviderCredentialDraft`
        # sibling, which seals all four (tier-2 review S21 P3).
        del protocol
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


#: TASK-21149 (UAT P-3): where a first-time user gets a key, per provider.
_PROVIDER_KEY_URLS = {
    "openai": "platform.openai.com/api-keys",
    "anthropic": "console.anthropic.com",
    "groq": "console.groq.com/keys",
    "openrouter": "openrouter.ai/keys",
    "mistralai": "console.mistral.ai",
    "deepseek": "platform.deepseek.com",
    "cohere": "dashboard.cohere.com",
    "google": "aistudio.google.com/apikey",
}


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

        # ``discover`` is the selected-provider seam. The localhost scan stays
        # separate and runs off the UI loop (its admission and TLS setup block).
        self._discover = discover
        self._local_discover = local_discover or step_guard.off_loop(discover_local_servers)
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
        self._subscription_readiness_status: str | None = None
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
        entries = step_guard.first_run_provider_catalog()  # TASK-33621.14
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
                # TASK-21144 (UAT P-8): labels name the outcome, not the
                # mechanism — "Detect" vs "Test" was an unexplained pair.
                with Horizontal(classes="setup-provider-connection-actions"):
                    yield Button("Find local servers", id="setup-provider-detect")
                    yield Button(
                        "Test connection",
                        id="setup-provider-test",
                        variant="primary",
                    )
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
            # TASK-21144 (UAT P-6): the probe/discovery status renders
            # ABOVE the Authentication collapsible, directly under the
            # connection controls it reports on — at its old panel-bottom
            # position it fell below the fold at 40-row terminals, which
            # (compounded by the F-1 focus soft-lock eating the button
            # presses) is how UAT experienced "silent" probes.
            yield Static(
                "", id="setup-provider-probe-status", classes="setup-probe-status"
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
            if (key_input is None or endpoint_input is None) and self.is_attached:
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
        self.set_interval(0.25, self._refresh_subscription_readiness)

    def _refresh_subscription_readiness(self) -> None:
        """Refresh only the visible selection from the bounded credential cache."""
        if (
            not self.is_attached
            or not self._discovery_visible
            or self.screen not in self.app.screen_stack
            or self.selected_provider_key != "anthropic"
        ):
            return
        status = self._current_provider_readiness().subscription_status
        # Compare state, not just completion revision: credentials may expire
        # during the cache lifetime without another file or Keychain read.
        if status != self._subscription_readiness_status:
            self._refresh_auth_readiness()

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
        if selected_was_in_progress and publish_status and self.is_attached:
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
        run_wizard_worker(
            self,
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
        listed = self.query("#setup-provider-detection-results")  # gone in teardown
        if not self.is_attached or not self.is_active or not listed:
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
        if not owner or not self.is_attached:
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
            provider_key,
            app_config,
            environ=self._environment(),
            background_credentials=True,
        ).requires_api_key

    def _credential_at_request_boundary(self) -> tuple[str, str | None, object]:
        """Resolve the current API key and nonblocking readiness snapshot.

        Borrowed subscription tokens remain owned by the actual send path;
        they never enter a first-run credential draft or saved config.
        """

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
            provider_key,
            app_config,
            environ=self._environment(),
            background_credentials=True,
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
        if self.is_attached and provider_draft is not None:
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
            provider_key,
            staged_config,
            environ=self._environment(),
            background_credentials=True,
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
        if (cancelled or invalidated or changed) and self.is_attached:
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
        if not self.selected_provider_key or not self.is_attached:
            return
        readiness = self._current_provider_readiness()
        self._subscription_readiness_status = readiness.subscription_status
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
        if readiness.subscription_status is not None:
            status.update(
                readiness.reason if readiness.ready else readiness.user_message
            )
            return
        if not readiness.ready:
            # TASK-21149 (UAT P-3): the input right above is the primary
            # path — lead with it and where to get a key; the env-var route
            # is the expert aside, not the headline.
            pointer = _PROVIDER_KEY_URLS.get(self.selected_provider_key, "")
            parts = ["An API key is needed — paste it above."]
            if pointer:
                parts.append(f"New keys: {pointer}.")
            env_var = getattr(readiness, "env_var", "") or ""
            if env_var:
                parts.append(
                    f"(Already exported {env_var}? It's picked up "
                    "automatically.)"
                )
            # task-32555 AC#3: the skip is visible where the key goes.
            parts.append(
                "No key yet? Enter skips this step — you can add a provider "
                "later in Settings."
            )
            status.update(" ".join(parts))
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
                f"Key staged — it will be checked when you continue.{unavailable}"
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
        run_wizard_worker(
            self,
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
                            timeout=model_discovery.MODEL_DISCOVERY_TIMEOUT_SECONDS,
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
                        timeout=model_discovery.MODEL_DISCOVERY_TIMEOUT_SECONDS,
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
                status.update(self._discovery_failure_status(display))
            elif attempted:
                status.update(f"Checked {display}; no models were reported.")
            else:
                status.update("")
        finally:
            done = self._selected_discovery_done
            if done is not None and generation == self.probe_generation:
                done.set()

    def _discovery_failure_status(self, display: str) -> str:
        """Failure copy that never promises what Next will refuse.

        task-31820 (release UAT): with a keyed cloud provider and no
        credential, this status said "You can continue anyway." while
        commit() was simultaneously hard-blocking Next with "API key
        required." -- both on screen at once. Promise continuation only
        when the readiness gate would actually allow it; when it wouldn't,
        name the unblock instead (the key input sits directly below, and
        "go Back" matches the pinned refusal's vocabulary).
        """
        try:
            readiness = self._current_provider_readiness()
            ready = bool(readiness.ready)
        except Exception:
            return f"Couldn't discover models for {display}."
        if ready:
            return f"Couldn't discover models for {display}. You can continue anyway."
        # Qodo (PR #2445): name the provider's OWN unblock -- "add an API
        # key" is wrong for auth modes that need a login instead (e.g. a
        # Claude subscription). readiness.recovery is the same string
        # commit()'s refusal footer shows, so both surfaces speak with one
        # vocabulary; fall back to the key-input hint only if it is empty.
        recovery = (getattr(readiness, "recovery", None) or "").strip()
        if not recovery:
            recovery = "Add an API key below to continue."
        return f"Couldn't discover models for {display}. {recovery} Or go Back."

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
        provider_key = self._canonical_provider_key(provider_key)
        previous_provider = self.selected_provider_key
        provider_changed = provider_key != previous_provider
        had_saved_draft = provider_key in self._provider_drafts
        # TASK-33621.14: the reads that can raise run before any state moves, so a
        # failed read leaves the old provider whole (never its key under this one).
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        presence = wizard_state.read_provider_secret_presence(
            app_config, self._environment(), provider_key=provider_key
        )
        initial_endpoint = (
            "" if had_saved_draft else self._initial_endpoint_for(provider_key)
        )
        optional_auth = not self._provider_requires_api_key(provider_key)
        endpoint_visible = self._provider_exposes_endpoint(provider_key)
        status = self.query_one("#setup-provider-key-status", Static)
        actions = self.query_one("#setup-provider-key-actions", Horizontal)
        key_input = self.query_one("#setup-provider-api-key", Input)
        connection = self.query_one("#setup-provider-connection", Vertical)
        auth = self.query_one("#setup-provider-auth-toggle", Collapsible)
        endpoint_input = self.query_one("#setup-provider-endpoint", Input)
        if provider_changed:
            self._capture_provider_ui_draft(previous_provider)
            self._invalidate_provider_test(changed=False)
            self._local_discovery_generation += 1
            self._local_discovery_provider_key = ""
            self._cancel_worker_groups("setup-provider-local-discovery")
            self._clear_detected_provider_state()
        ui_draft = self._provider_ui_draft(provider_key)
        if not had_saved_draft:
            ui_draft.endpoint = initial_endpoint
            ui_draft.key_input_visible = not (
                presence.inline_configured or presence.env_var_set
            )
            ui_draft.auth_collapsed = optional_auth
        self.selected_provider_key = provider_key
        self._clear_requested = ui_draft.clear_requested
        self._credential_revision = ui_draft.credential_revision
        if provider_changed:
            self._updating_connection_controls = True
            try:
                key_input.value = ui_draft.api_key
                key_input.display = ui_draft.key_input_visible
                connection.display = endpoint_visible
                connection.set_class(not endpoint_visible, "hidden")
                restored_endpoint = ui_draft.endpoint if endpoint_visible else ""
                if endpoint_input.value != restored_endpoint:
                    self._pending_programmatic_endpoint_changes.append(
                        (provider_key, restored_endpoint)
                    )
                    endpoint_input.value = restored_endpoint
                auth.display = True
                auth.remove_class("hidden")
                auth.title = "Authentication" + (" (optional)" if optional_auth else "")
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
            status.update("Key staged — it will be checked when you continue.")
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
        if provider_key is None or option.disabled:
            return
        # TASK-33621.14: a failed pick is reported with the provider that stays.
        with step_guard.provider_switch(self, provider_key):
            if provider_key != self.selected_provider_key:
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

    def skip_without_key(self) -> None:
        """Forget the provider choice so ``commit`` takes its skip path.

        task-32555 AC#3: Enter in the empty key field. Nothing was staged
        for this provider (a stage needs a ready credential), so clearing
        the choice loses nothing; the highlighted list row re-selects on
        the next interaction if the user comes Back.

        A provider that is ALREADY ready without a typed key (an exported
        env var, a local server) is the exception -- commit() would stage
        it, and the key field's hint never offered the skip in that state
        (``_refresh_auth_readiness`` adds it only under "not ready").
        """
        if self.selected_provider_key and self._current_provider_readiness().ready:
            return
        self.selected_provider_key = ""
        self._provider_choice_interacted = False

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
        # Review TASK-21143 follow-up (P-5): the returned-to-Provider notice
        # ("this API key was rejected") must not outlive the edit that
        # addresses it — a stale rejection over a fresh key reads as "still
        # broken".
        try:
            wizard = self.wizard
            if wizard is not None and hasattr(wizard, "_clear_pinned_step_error"):
                wizard._clear_pinned_step_error()
        except Exception:
            pass
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
        run_wizard_worker(
            self,
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
                if generation == self.probe_generation and self.is_attached:
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
            and self.is_attached
        ):
            self._render_provider_evidence(identity)

    @staticmethod
    def _provider_probe_result_from_outcome(outcome: object):
        """Convert only the exact shared Settings outcome to exact evidence."""

        from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
            SettingsEndpointProbeOutcome,
            provider_probe_result_from_settings_outcome,
        )

        # The one mapping every listing probe uses (a 403 never blocks).
        if type(outcome) is not SettingsEndpointProbeOutcome or str(
            outcome.state
        ) not in {"reachable", "model_listing_unavailable", "unreachable"}:
            raise ValueError("Provider probe outcome is invalid.")
        return provider_probe_result_from_settings_outcome(outcome)

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
            if readiness.subscription_status is not None:
                return False, readiness.user_message
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
        # TASK-34100.1 (cross-cutting-14): a settled discovery is reused for
        # the same provider identity, whatever it returned. A failure used to
        # restart here on every Next, each one up to the 8 s guard. Changing
        # the key or endpoint makes a new identity; Model's Retry asks again.
        if discovery_key != self._selected_discovery_key or (
            self._selected_discovery_state
            not in {"in_progress", "complete", "failed"}
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

    def busy_label(self) -> str:
        """What a slow Next from Provider is doing (TASK-34100.1)."""
        from tldw_chatbook.Chat.provider_catalog import provider_display_name

        key = self.selected_provider_key
        display = provider_display_name(key) if key else ""
        return f"Saving the {display} connection…" if display else ""

    def get_step_data(self) -> Dict[str, Any]:
        return {
            "provider_key": self.selected_provider_key,
            "provider_value": self.provider_value_for_chat_defaults,
            "entered_key": self._entered_key,
        }
