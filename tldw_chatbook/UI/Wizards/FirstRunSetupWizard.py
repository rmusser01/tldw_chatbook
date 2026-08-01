"""First-run setup wizard: hermes-agent's setup process in chatbook chrome.

Screen + container subclass over BaseWizard (which is never modified).
All decisions and config mutations are built by first_run_setup_state;
this module renders them and owns persistence via one exclusive worker.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Mapping, Optional

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.compose import compose as _drain_compose_result
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static, Switch

from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
    active_managed_parakeet_v2_dir,
    parakeet_v2_managed_service,
    parakeet_v2_reference,
    run_parakeet_v2_preflight,
    run_parakeet_v2_provision,
)
from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message
from tldw_chatbook.UI.Screens.model_installed_view import lifecycle_failure_message
from tldw_chatbook.UI.Wizards import first_run_speech_step_state as speech_state
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

from tldw_chatbook.UI.Wizards.BaseWizard import (
    WizardContainer,
    WizardNavigation,
    WizardProgress,
    WizardScreen,
    WizardStep,
    WizardStepConfig,
)
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


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

    #: TASK-1266: set when compose_step() raised — the container drops the
    #: step from navigation and the Summary reports it.
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

        TASK-1266 (spec §5): a step whose composition raises must never
        crash the wizard screen. The step renders a one-line notice in its
        place, flags itself, and the container auto-skips it; the Summary
        adds a reasoned row. Subclasses implement ``compose_step``.

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
            exception, yields a single ``Static`` notice explaining the
            step was skipped instead (and sets ``compose_failed = True``
            so the container drops the step and the Summary reports it).
        """
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
        except Exception:
            logger.exception(
                "Wizard step %s failed to compose; auto-skipping",
                self.config.id if self.config else type(self).__name__,
            )
            self.compose_failed = True
            yield Static(
                "This step couldn't be shown and was skipped — its settings "
                "are still available in Settings.",
                classes="setup-step-error",
            )
            return
        yield from buffered

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


CLOUD_PROBE_TIMEOUT_SECONDS = 8.0


class ProviderStep(SetupStep):
    """Choose a provider, supply credentials, verify without blocking."""

    def __init__(
        self,
        wizard: Optional["SetupWizardContainer"] = None,
        config: Optional[WizardStepConfig] = None,
        *,
        discover: Optional[Callable[..., Any]] = None,
        probe: Optional[Callable[..., Any]] = None,
        environ: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(wizard=wizard, config=config, **kwargs)
        from tldw_chatbook.Chat.local_server_discovery import discover_local_servers
        from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
            probe_settings_endpoint,
        )

        self._discover = discover or discover_local_servers
        self._probe = probe or probe_settings_endpoint
        self._environ = dict(environ) if environ is not None else dict(os.environ)
        self.probe_generation = 0
        self.selected_provider_key: str = ""
        self.provider_value_for_chat_defaults: str = ""
        self._last_committed_provider_value: Optional[str] = None
        self._entered_key = False
        self._clear_requested = False

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
            # scrolling list. Being before the RadioSet in DOM also keeps the
            # TASK-1496 Tab order intact (radio → key input; the button is
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
                "Use this server", id="setup-provider-use-detected",
                classes="hidden", variant="primary",
            )
            with RadioSet(id="setup-provider-choice", classes="setup-choice-list"):
                # TASK-1498: visible group headers; popular providers first so
                # the capped list's initial window is the useful one.
                for group_title, group in self._grouped_sections(entries):
                    yield Static(group_title, classes="setup-choice-header")
                    for entry in group:
                        yield SetupRadioButton(
                            entry.display_name,
                            id=f"setup-provider-{entry.readiness_key}",
                        )
            # TASK-1496: key Input, its status line, and the Keep/Replace/
            # Clear affordances sit BEFORE the discovered-server banner/button
            # in both DOM and visual order now -- Tab from the RadioSet above
            # must reach the key Input next, not a below-the-fold "Use this
            # server" Button (Textual's focus chain follows DOM order, not
            # rendered position). Before this reorder, a discovered local
            # server unhid that Button ahead of the Input in DOM order, so
            # Tab landed on it instead; a typed API key then went into a
            # Button (which does not accept text input) and vanished
            # silently, and Protect-keys could never activate since
            # note_key_entered() never fired. Putting the detected-server
            # affordance LAST (see below) makes Tab order match visual
            # top-to-bottom order exactly: radio list -> key input ->
            # Keep/Replace/Clear -> detected-server button.
            yield Label("API key", classes="setup-field-label")
            # TASK-1506: the probe used to fire only on Enter inside the
            # field — undiscoverable. A visible Test button shares the
            # input's row so the 1495 row budget is unchanged.
            with Horizontal(classes="setup-key-row"):
                yield Input(password=True, id="setup-provider-key-input",
                            placeholder="Paste your API key")
                yield Button("Test", id="setup-provider-test")
            yield Static("", id="setup-provider-key-status", classes="setup-probe-status")
            with Horizontal(id="setup-provider-key-actions", classes="hidden"):
                yield Button("Keep current", id="setup-provider-key-keep")
                yield Button("Replace", id="setup-provider-key-replace")
                yield Button("Clear", id="setup-provider-key-clear")
            yield Static("", id="setup-provider-probe-status", classes="setup-probe-status")
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
        popular = [
            by_key[key] for key in cls._POPULAR_PROVIDER_KEYS if key in by_key
        ]
        popular_keys = {e.readiness_key for e in popular}
        rest = [e for e in entries if e.readiness_key not in popular_keys]
        alpha = lambda e: e.display_name.lower()  # noqa: E731
        cloud = sorted(
            (e for e in rest
             if e.requires_api_key
             and e.readiness_key not in PROVIDER_CUSTOM_GROUP_KEYS),
            key=alpha,
        )
        local = sorted(
            (e for e in rest
             if not e.requires_api_key
             and e.readiness_key not in PROVIDER_CUSTOM_GROUP_KEYS),
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
            The provider RadioSet, or None if it is not queryable yet.
        """
        try:
            return self.query_one("#setup-provider-choice", RadioSet)
        except Exception:
            return None

    def on_show(self) -> None:
        super().on_show()
        self._start_discovery()

    def _start_discovery(self) -> None:
        self.run_worker(self._discover_servers(), exclusive=True,
                        group="setup-provider-discovery")

    async def _discover_servers(self) -> None:
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        try:
            servers = tuple(await self._discover(app_config) or ())
        except Exception:
            logger.debug("Wizard local discovery failed", exc_info=True)
            return
        if not servers:
            return
        self.detected_server = servers[0]
        banner = self.query_one("#setup-provider-detected", Static)
        banner.update(
            f"Found a local server at {self.detected_server.base_url} "
            f"({self.detected_server.provider_key})."
        )
        banner.remove_class("hidden")
        use_button = self.query_one("#setup-provider-use-detected", Button)
        use_button.remove_class("hidden")

    @on(Button.Pressed, "#setup-provider-use-detected")
    def _on_use_detected(self) -> None:
        """One-click connect: adopt the discovered server as the provider."""
        server = getattr(self, "detected_server", None)
        if server is None:
            return
        self.select_provider(server.provider_key)
        self.detected_base_url = server.base_url
        self.query_one("#setup-provider-detected", Static).update(
            f"✓ Using {server.base_url} ({server.provider_key})."
        )

    def select_provider(self, provider_key: str) -> None:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            read_provider_secret_presence,
        )

        provider_changed = provider_key != self.selected_provider_key
        self.selected_provider_key = provider_key
        self.probe_generation += 1
        self._clear_requested = False
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        presence = read_provider_secret_presence(
            app_config, self._environ, provider_key=provider_key
        )
        status = self.query_one("#setup-provider-key-status", Static)
        actions = self.query_one("#setup-provider-key-actions", Horizontal)
        key_input = self.query_one("#setup-provider-key-input", Input)
        if provider_changed:
            # Bug-1 fix: the key Input is shared across every provider this
            # step renders. Without clearing it here, a key typed for
            # provider A silently survived a switch to provider B and would
            # commit under B's api_settings section on Next.
            key_input.value = ""
        if presence.env_var_set:
            status.update(f"Found {presence.env_var} in your environment ✓ — nothing to store.")
            key_input.display = False
            actions.add_class("hidden")
        elif presence.configured:
            status.update("An API key is already configured for this provider.")
            key_input.display = False
            actions.remove_class("hidden")
        else:
            status.update("")
            key_input.display = True
            actions.add_class("hidden")
        self.query_one("#setup-provider-probe-status", Static).update("")

    @on(RadioSet.Changed, "#setup-provider-choice")
    def _on_provider_chosen(self, event: RadioSet.Changed) -> None:
        pressed_id = event.pressed.id or ""
        self.select_provider(pressed_id.removeprefix("setup-provider-"))

    def _effective_provider_key(self) -> str:
        """F-A fix: the RadioSet's own ``pressed_button`` is the source of
        truth at commit time, not just the ``Changed``-driven instance
        attribute.

        ``self.selected_provider_key`` is set by ``select_provider()``, which
        runs from two places: the ``RadioSet.Changed`` handler above (a real
        keyboard-toggle/click), and ``_on_use_detected`` (the one-click
        "Use this server" path, which never presses a RadioButton in
        ``#setup-provider-choice`` at all). Textual's RadioSet distinguishes
        the merely-*highlighted* button (arrow-key navigation moves this,
        see ``RadioSet.action_next_button``/``action_previous_button``) from
        the *pressed* one (``RadioSet.pressed_button``, only set by an
        explicit toggle -- Enter/Space/click, or an initial ``value=True`` at
        mount, see ``RadioSet._on_mount``'s ``switched_on`` handling) --
        which never fires ``Changed``. No button in this step's catalog sets
        ``value=True`` today, so this fallback is currently a no-op guard
        against a future default selection (mirroring how WelcomeStep reads
        its RadioButton's live ``.value`` instead of trusting an instance
        attribute) or any other path that presses a radio without routing
        through ``select_provider()``.

        Preferring ``self.selected_provider_key`` when it is already set
        keeps the "Use this server" one-click path correct even if an
        earlier, different radio press left a stale ``pressed_button``
        behind; the RadioSet is consulted only when this step's own
        bookkeeping has nothing.
        """
        if self.selected_provider_key:
            return self.selected_provider_key
        try:
            pressed = self.query_one("#setup-provider-choice", RadioSet).pressed_button
        except Exception:
            return ""
        if pressed is None:
            return ""
        return (pressed.id or "").removeprefix("setup-provider-")

    @on(Button.Pressed, "#setup-provider-key-replace")
    def _on_replace(self) -> None:
        """Reveal the masked input so the user can type a new key.

        Leaving it blank after Replace is a cancel: commit() only persists a
        typed, non-empty value, so the currently-configured secret is left
        untouched (never re-shown).
        """
        self._clear_requested = False
        self.query_one("#setup-provider-key-input", Input).display = True

    @on(Button.Pressed, "#setup-provider-key-keep")
    def _on_keep(self) -> None:
        """Abandon any in-progress Replace/Clear; the stored secret is untouched."""
        self._clear_requested = False
        key_input = self.query_one("#setup-provider-key-input", Input)
        key_input.value = ""
        key_input.display = False

    @on(Button.Pressed, "#setup-provider-key-clear")
    def _on_clear(self) -> None:
        """Mark the configured secret for removal on commit.

        Unlike Replace, leaving the field blank here is the whole point: it
        signals commit() to persist an explicit empty api_key rather than
        leaving the existing one in place (build_provider_commit's truthiness
        check would otherwise treat "" exactly like "nothing to write").
        """
        self._clear_requested = True
        key_input = self.query_one("#setup-provider-key-input", Input)
        key_input.value = ""
        key_input.display = True
        self.query_one("#setup-provider-key-status", Static).update(
            "The stored key will be removed when you continue."
        )

    @on(Input.Submitted, "#setup-provider-key-input")
    def _on_key_submitted(self, event: Input.Submitted) -> None:
        """Live-but-never-blocking verification: probe on Enter in the key field."""
        if event.value.strip():
            self._launch_probe(api_key=event.value.strip())

    @on(Button.Pressed, "#setup-provider-test")
    def _on_test_pressed(self, event: Button.Pressed) -> None:
        """TASK-1506: same probe as Enter-in-field, behind a visible control."""
        event.stop()
        typed = self.query_one("#setup-provider-key-input", Input).value.strip()
        self._launch_probe(api_key=typed or None)

    def _launch_probe(self, *, api_key: str | None = None) -> None:
        self.probe_generation += 1
        generation = self.probe_generation
        base_url = getattr(self, "detected_base_url", None)
        self.query_one("#setup-provider-probe-status", Static).update("Testing…")
        self.run_worker(
            self._run_probe(generation, base_url=base_url, api_key=api_key),
            exclusive=True,
            group="setup-provider-probe",
        )

    async def _run_probe(
        self, generation: int, *, base_url: str | None, api_key: str | None
    ) -> None:
        import httpx

        from tldw_chatbook.Chat.local_server_discovery import (
            DISCOVERY_PROBE_TIMEOUT_SECONDS,
        )

        # Local servers probe their own base URL; cloud keys probe the
        # provider's OpenAI-compatible endpoint with the key as a bearer
        # header via the http_client seam (probe_settings_endpoint has no
        # auth parameter by design). Providers without a known compatible
        # endpoint resolve to "couldn't verify — save anyway".
        target = base_url or self._cloud_probe_base_url(self.selected_provider_key)
        if not target:
            self.apply_probe_result(
                generation, reachable=False, summary="No test endpoint for this provider."
            )
            return
        client = None
        try:
            if api_key:
                client = httpx.AsyncClient(
                    headers={"Authorization": f"Bearer {api_key}"}
                )
            outcome = await self._probe(
                target,
                timeout=(
                    CLOUD_PROBE_TIMEOUT_SECONDS
                    if api_key
                    else DISCOVERY_PROBE_TIMEOUT_SECONDS
                ),
                http_client=client,
            )
            self.apply_probe_result(
                generation, reachable=outcome.reachable, summary=outcome.summary
            )
        except Exception:
            logger.debug("Wizard provider probe failed", exc_info=True)
            self.apply_probe_result(
                generation, reachable=False, summary="Probe errored."
            )
        finally:
            if client is not None:
                await client.aclose()

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
        }.get(provider_key, "")

    def apply_probe_result(self, generation: int, *, reachable: bool, summary: str) -> None:
        """Render a probe outcome only if it is still current (no stale ✓)."""
        if generation != self.probe_generation:
            return
        prefix = "✓ " if reachable else "✗ "
        suffix = "" if reachable else "  Couldn't verify — you can save anyway."
        self.query_one("#setup-provider-probe-status", Static).update(
            f"{prefix}{summary}{suffix}"
        )

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            build_provider_commit,
            invalidate_model_for_provider_change,
            read_wizard_prefill,
        )

        provider_key = self._effective_provider_key()
        if not provider_key:
            return True, ""  # legitimately nothing pressed -- skip is correct
        self.selected_provider_key = provider_key
        key_input = self.query_one("#setup-provider-key-input", Input)
        typed_key = (
            key_input.value.strip() if key_input.display and key_input.value else None
        )
        api_url = getattr(self, "detected_base_url", None)
        if not typed_key and self._clear_requested:
            # build_provider_commit's truthiness check treats "" exactly like
            # "nothing to write" -- that's right for every other path (an
            # untouched/blank field must never clobber a stored key) but
            # wrong for an explicit Clear, which must persist an empty
            # string. Build that one section directly instead.
            section = f"api_settings.{self.selected_provider_key}"
            values: Dict[str, Any] = {"api_key": ""}
            if api_url:
                values["api_url"] = api_url
            commit: Dict[str, Dict[str, Any]] = {section: values}
        else:
            commit = build_provider_commit(
                provider_key=self.selected_provider_key,
                api_key=typed_key,
                api_url=api_url,
            )
        # Resolve the exact chat_defaults.provider value form the same way
        # chat_screen._apply_detected_local_server does (read it once; mirror it).
        self.provider_value_for_chat_defaults = self._display_value_for(
            self.selected_provider_key
        )
        # Bug-3 fix: on the very first commit this session,
        # self._last_committed_provider_value is still None -- fall back to
        # the PERSISTED chat_defaults.provider so a first-ever provider
        # selection (with Model skipped) still syncs chat_defaults instead
        # of silently leaving it at whatever the template/previous run had,
        # even though credentials just landed under api_settings. A later
        # commit this same session (Back-and-switch) still prefers the
        # in-session value over a possibly-stale persisted one.
        if self._last_committed_provider_value is not None:
            effective_previous_provider = self._last_committed_provider_value
        else:
            app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
            effective_previous_provider = read_wizard_prefill(app_config).provider_value
        commit = invalidate_model_for_provider_change(
            commit,
            previous_provider_value=effective_previous_provider,
            new_provider_value=self.provider_value_for_chat_defaults,
        )
        ok = await self.wizard.commit_config(commit)
        if not ok:
            return False, "Saving the provider settings failed."
        self._last_committed_provider_value = self.provider_value_for_chat_defaults
        self._clear_requested = False
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

    def __init__(self, wizard=None, config=None, *, discover_models=None, **kwargs):
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._discover_models = discover_models
        self._shown_for_provider: Optional[str] = None
        self.selected_model_id: str = ""
        # Bug-5: tracks whether selected_model_id's current value came from
        # the free-text custom Input (as opposed to the RadioSet) -- lets
        # clearing that Input fall back to any active radio selection
        # instead of leaving a stale custom value in place.
        self._model_id_from_custom_input: bool = False

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
            yield Static("", classes="setup-step-error")

    def _current_provider(self) -> tuple[str, str]:
        data = (self.wizard.wizard_data or {}).get(wizard_state.STEP_PROVIDER, {})
        return str(data.get("provider_key", "")), str(data.get("provider_value", ""))

    def on_show(self) -> None:
        super().on_show()
        provider_key, provider_value = self._current_provider()
        if provider_key != self._shown_for_provider:
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
            prefill_model_id = wizard_state.rerun_model_prefill(
                getattr(self.wizard.app_instance, "app_config", {}) or {},
                provider_value=provider_value,
            )
            self.selected_model_id = prefill_model_id
            self._model_id_from_custom_input = False
            self._shown_for_provider = provider_key
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
        if provider_key:
            self.run_worker(
                self._load_models(provider_key, provider_value),
                exclusive=True,
                group="setup-model-load",
            )
        else:
            # F-F fix: with no provider chosen yet there is nothing to
            # discover against, so the old code simply skipped this branch
            # and left the initial "(loading models…)" RadioButton in place
            # forever -- a permanently-stuck loading indicator for a state
            # that was never actually loading. Replace it with copy that
            # tells the user what to do instead.
            self.run_worker(
                self._render_models([], no_provider=True),
                exclusive=True,
                group="setup-model-load",
            )

    async def _load_models(self, provider_key: str, provider_value: str) -> None:
        import asyncio

        models: list[str] = []
        discover = self._discover_models
        if discover is None:
            service = getattr(
                self.wizard.app_instance, "llm_provider_catalog_scope_service", None
            )
            if service is not None:

                async def discover(pk=provider_key, svc=service):
                    result = await svc.discover_models(
                        mode="local", provider=pk, staged_settings=None
                    )
                    if str(getattr(result, "status", "")) == "success":
                        return list(getattr(result, "models", ()) or ())
                    return []

        if discover is not None:
            try:
                models = list(
                    await asyncio.wait_for(
                        discover(provider_key), timeout=MODEL_DISCOVERY_TIMEOUT_SECONDS
                    )
                )
            except Exception:
                logger.debug("Wizard model discovery failed", exc_info=True)
        if not models:
            from tldw_chatbook.config import get_cli_providers_and_models

            models = wizard_state.curated_models_for_provider(
                get_cli_providers_and_models(), provider_value
            )
        await self._render_models(models[:20])

    async def _render_models(
        self, models: list[str], *, no_provider: bool = False
    ) -> None:
        try:
            radio_set = self.query_one("#setup-model-choice", RadioSet)
        except Exception:
            return
        # remove_children()/mount() are message-queue operations -- both
        # return awaitables that must be awaited before the DOM change is
        # actually applied. Without awaiting the removal, a second call (e.g.
        # a provider switch that fires before the first discovery settles)
        # can try to mount fresh "setup-model-option-N" ids while the stale
        # ones are still present, raising DuplicateIds.
        await radio_set.remove_children()
        if models:
            # TASK-1503: the first entry (curated-default / top discovery hit)
            # carries a "recommended" tag in its LABEL only; the clean model
            # id lives on the button as `_model_id` so selection and commits
            # never round-trip display decoration into config.
            def _button(index: int, model_id: str) -> SetupRadioButton:
                label = f"{model_id}   (recommended)" if index == 0 else model_id
                button = SetupRadioButton(label, id=f"setup-model-option-{index}")
                button._model_id = model_id
                return button

            await radio_set.mount_all(
                _button(index, model_id) for index, model_id in enumerate(models)
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
                SetupRadioButton("(no models found — enter one below)", disabled=True)
            )

    @on(RadioSet.Changed, "#setup-model-choice")
    def _on_model_chosen(self, event: RadioSet.Changed) -> None:
        if event.pressed is not None:
            self.set_selected_model_from_button(event.pressed)

    def set_selected_model_from_button(self, button: RadioButton) -> None:
        """Select via a radio row, reading the clean id, not the label.

        TASK-1503: labels may carry display decoration ("(recommended)");
        the undecorated model id is stored on the button as ``_model_id``.

        Args:
            button: The pressed radio row; its ``_model_id`` attribute (or
                label when absent) supplies the model id to select.
        """
        self.set_selected_model(getattr(button, "_model_id", str(button.label)))

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
        value = event.value.strip()
        if value:
            self.selected_model_id = value
            self._model_id_from_custom_input = True
        elif self._model_id_from_custom_input:
            self._model_id_from_custom_input = False
            pressed = self._live_pressed_radio()
            # TASK-1503: clean id, never the (possibly decorated) label.
            self.selected_model_id = (
                str(getattr(pressed, "_model_id", pressed.label))
                if pressed is not None
                else ""
            )

    def set_selected_model(self, model_id: str) -> None:
        self.selected_model_id = model_id
        self._model_id_from_custom_input = False

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
        if self.selected_model_id:
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
        ok = await self.wizard.commit_config(
            wizard_state.build_model_commit(
                provider_value=provider_value, model_id=model_id
            )
        )
        if ok:
            self.selected_model_id = model_id
        return (True, "") if ok else (False, "Saving the model choice failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {"model_id": self.selected_model_id}


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
        models = embedding_config.get("models", {}) if isinstance(embedding_config, dict) else {}
        return sorted(models) if isinstance(models, dict) else []

    def on_mount(self) -> None:
        status = self.query_one("#setup-rag-status", Static)
        if self._deps_installed():
            status.update("Embedding dependencies are installed. Pick a default model, or skip.")
        else:
            status.update(
                # Static.update() treats [..] as Rich markup by default, so the
                # extras-package brackets must be escaped or "[embeddings_rag]"
                # silently vanishes from the rendered text instead of showing.
                # TASK-1502: quoted plainly — backticks are markdown idiom and
                # render literally in a TUI.
                "RAG needs optional dependencies that aren't installed. Install the "
                "extras package \"tldw_chatbook\\[embeddings_rag]\" with your package "
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
    """Optional Speech transcription setup: install/activate Parakeet v2.

    TASK-1301: reuses the TASK-596 shared model-artifact controls
    (ModelInstallModal, ModelInstallProgress, ModelActivationControls) and
    the TASK-595 ModelArtifactService via the SAME
    Local_Ingestion.parakeet_v2_artifact convenience wrappers LibraryScreen's
    own Parakeet v2 install surface already uses -- no duplicate artifact or
    network logic (AC#4). Language/precision options are enumerated from the
    canonical STT policy/catalog (first_run_speech_step_state, backed by
    tldw_chatbook.STT.routing) and gated to what a curated descriptor can
    actually download today (AC#2).

    Persistence gate (AC#5): commit() re-verifies -- off the event loop --
    that the managed Parakeet v2 artifact is installed AND active before
    writing anything to [transcription]; an install that never completed, or
    one a user later deleted, leaves this step skip-safe. Skip and failures
    never trap the user (AC#6): Next/commit never blocks on install state,
    and a failed download still refreshes the step's own installed-state
    read so it never gets stuck showing a stale "installing…" affordance.
    """

    def __init__(
        self,
        wizard: Optional["SetupWizardContainer"] = None,
        config: Optional[WizardStepConfig] = None,
        *,
        service_factory: Optional[Callable[[], Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._service_factory = service_factory or parakeet_v2_managed_service
        self._reference = parakeet_v2_reference()
        self._service: Any = None
        self._loading = False
        self._loaded = False
        self._load_error: Optional[str] = None
        self._installed_item: Any = None
        self._operation: Optional[str] = None
        self._pending_report: Any = None
        self._progress: Any = None

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-speech"):
            yield Static("Speech transcription (optional)", classes="setup-title")
            yield Static(
                "Recommended: Parakeet v2 — English, INT8, runs fully on this "
                "device. Used for dictation and audio/video transcripts. "
                "Skip and set this up later in Settings ▸ Speech any time.",
                classes="setup-subtitle",
            )
            yield Label("Language", classes="setup-field-label")
            with RadioSet(id="setup-speech-language-choice", classes="setup-choice-list"):
                for option in speech_state.speech_language_options(
                    curated_model_ids=self._curated_model_ids()
                ):
                    label = option.display_name + (
                        " (recommended)"
                        if option.selectable
                        else " — not yet available for managed install"
                    )
                    yield SetupRadioButton(
                        label,
                        id=f"setup-speech-language-{option.code}",
                        value=option.selectable and option.code == "en",
                        disabled=not option.selectable,
                    )
            yield Label("Precision", classes="setup-field-label")
            with RadioSet(id="setup-speech-precision-choice", classes="setup-choice-list"):
                for option in speech_state.speech_precision_options(
                    curated_precisions=self._curated_precisions()
                ):
                    label = option.display_name + (
                        " (recommended)"
                        if option.selectable
                        else " — not yet available for managed install"
                    )
                    yield SetupRadioButton(
                        label,
                        id=f"setup-speech-precision-{option.value}",
                        value=option.selectable,
                        disabled=not option.selectable,
                    )
            status_text, action_widget = self._status_and_action()
            yield Static(status_text, id="setup-speech-status", classes="setup-subtitle")
            progress = ModelInstallProgress(self._progress, id="setup-speech-install-progress")
            progress.display = self._operation == "install" and self._progress is not None
            yield progress
            if action_widget is not None:
                yield action_widget
            yield Static("", classes="setup-step-error")

    # -- pure, I/O-free helpers (curated_registry() only builds descriptors
    # in memory -- see its own module docstring) -------------------------
    @staticmethod
    def _curated_model_ids() -> frozenset[str]:
        from tldw_chatbook.Model_Artifacts.curated_registry import curated_registry

        return frozenset(descriptor.model_id for descriptor in curated_registry().list())

    @staticmethod
    def _curated_precisions() -> frozenset[str]:
        from tldw_chatbook.Model_Artifacts.curated_registry import curated_registry

        policy = speech_state.routing_policy()
        return frozenset(
            descriptor.precision
            for descriptor in curated_registry().list()
            if descriptor.model_id == policy.parakeet_v2_model_id
        )

    def _status_and_action(self) -> tuple[str, Optional[Widget]]:
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
                disabled=self._operation is not None,
            )
        if item.error is not None or not item.ready:
            return (
                "This model needs attention — reinstall from Settings ▸ Speech.",
                None,
            )
        status = "Installed and active." if item.active else "Installed, not yet active."
        return status, ModelActivationControls(
            self._reference,
            active=item.active,
            ready=item.ready,
            pending=self._operation is not None,
        )

    # -- lazy installed-state load ----------------------------------------
    def on_show(self) -> None:
        super().on_show()
        self._ensure_loaded()

    def _ensure_loaded(self, *, force: bool = False) -> None:
        if self._loading or (self._loaded and not force):
            return
        self._loading = True
        if force:
            self._loaded = False
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
        self.refresh(recompose=True)

    # -- install: preflight -> consent modal -> provision ------------------
    @on(Button.Pressed, "#setup-speech-install")
    def _install_pressed(self) -> None:
        if self._operation is not None:
            return
        self._operation = "install"
        self.refresh(recompose=True)
        self._preflight_install()

    @on(Button.Pressed, "#setup-speech-retry")
    def _retry_pressed(self) -> None:
        self._ensure_loaded(force=True)

    @work(thread=True, group="setup-speech-install", exclusive=True, exit_on_error=False)
    def _preflight_install(self) -> None:
        import asyncio

        try:
            report = asyncio.run(run_parakeet_v2_preflight())  # policy-exception: worker-thread loop
        except Exception as exc:
            logger.opt(exception=True).error("Speech transcription preflight failed")
            self.app.call_from_thread(
                self._apply_preflight_result,
                None,
                install_failure_message(exc, model_label="Parakeet v2"),
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
                model_label="Parakeet v2 (English, INT8)",
                container_id="setup-speech-install-modal",
                confirm_id="setup-speech-install-confirm",
                cancel_id="setup-speech-install-cancel",
            ),
            self._confirm_install,
        )

    def _confirm_install(self, confirmed: bool) -> None:
        if not confirmed:
            self._pending_report = None
            self._operation = None
            self.refresh(recompose=True)
            return
        self._provision_install()

    @work(thread=True, group="setup-speech-install", exclusive=True, exit_on_error=False)
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
            asyncio.run(  # policy-exception: worker-thread loop
                run_parakeet_v2_provision(
                    report, progress=make_progress_callback(self.post_message)
                )
            )
        except Exception as exc:
            logger.opt(exception=True).error("Speech model installation failed")
            self.app.call_from_thread(
                self._apply_provision_result,
                install_failure_message(exc, model_label="Parakeet v2"),
            )
            return
        self.app.call_from_thread(self._apply_provision_result, None)

    @on(InstallProgressed)
    def _install_progressed(self, event: InstallProgressed) -> None:
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

    def _apply_provision_result(self, error: Optional[str]) -> None:
        self._pending_report = None
        self._operation = None
        self._progress = None
        if error is not None:
            self.notify(error, severity="error")
        else:
            self.notify("Speech model installed and activated.", severity="information")
        # AC#6: failures never trap -- always refresh installed state so the
        # step reflects reality (and drops the disabled "installing…" affordance)
        # whether provisioning succeeded or failed.
        self._ensure_loaded(force=True)

    # -- activation / deletion: the 596 controls, reused verbatim ----------
    @on(ActivationRequested)
    def _activation_requested(self, event: ActivationRequested) -> None:
        event.stop()
        if self._operation is not None:
            return
        self._operation = "activate"
        self.refresh(recompose=True)
        self._activate_model()

    @work(thread=True, group="setup-speech-lifecycle", exclusive=True, exit_on_error=False)
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
        self.app.call_from_thread(self._apply_lifecycle_result, None)

    @on(DeletionRequested)
    def _deletion_requested(self, event: DeletionRequested) -> None:
        event.stop()
        if self._operation is not None:
            return
        self.app.push_screen(
            DeleteConfirmationDialog(
                item_type="Model",
                item_name="Parakeet v2 (English, INT8)",
                additional_warning=(
                    "The managed model files will be removed from this device."
                ),
                permanent=True,
            ),
            self._confirm_deletion,
        )

    def _confirm_deletion(self, confirmed: bool) -> None:
        if not confirmed or self._operation is not None:
            return
        self._operation = "delete"
        self.refresh(recompose=True)
        self._delete_model()

    @work(thread=True, group="setup-speech-lifecycle", exclusive=True, exit_on_error=False)
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

    def _apply_lifecycle_result(self, error: Optional[str]) -> None:
        self._operation = None
        if error is not None:
            self.notify(error, severity="error")
        else:
            self.notify("Speech model updated.", severity="information")
        self._ensure_loaded(force=True)

    # -- persistence gate (AC#5) --------------------------------------------
    async def commit(self) -> tuple[bool, str]:
        import asyncio

        active_dir = await asyncio.get_running_loop().run_in_executor(
            None, self._check_active
        )
        if active_dir is None:
            return True, ""  # skip-safe: nothing verified active, nothing to persist
        provider_id, model_id, language = speech_state.recommended_speech_selection()
        ok = await self.wizard.commit_config(
            speech_state.build_speech_transcription_commit(
                provider_id=provider_id, model_id=model_id, language=language,
            )
        )
        return (True, "") if ok else (False, "Saving the speech transcription choice failed.")

    def _check_active(self) -> Any:
        try:
            return active_managed_parakeet_v2_dir(self._service_for_worker())
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
        return {"enabled_gates": [
            self.gate_key_for(sw) for sw in self.query(Switch) if sw.value
        ]}


class NotesSyncStep(SetupStep):
    """Optional bidirectional notes sync: a directory and a toggle."""

    def compose_step(self) -> ComposeResult:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import read_wizard_prefill

        prefill = read_wizard_prefill(
            getattr(self.wizard.app_instance, "app_config", {}) or {}
        )
        with Vertical(classes="setup-notes"):
            yield Static("Notes sync", classes="setup-title")
            yield Static(
                "Keep a folder of Markdown files in sync with your notes. "
                "Skip if you only want in-app notes.",
                classes="setup-subtitle",
            )
            with Horizontal(classes="setup-tool-row"):
                yield Switch(value=prefill.auto_sync_enabled, id="setup-notes-enable")
                yield Label("Enable notes sync")
            yield Label("Notes directory", classes="setup-field-label")
            yield Input(
                value=prefill.sync_directory or "~/Documents/Notes",
                id="setup-notes-directory",
            )
            yield Static("", classes="setup-step-error")

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            build_notes_commit,
            read_wizard_prefill,
        )

        enabled = self.query_one("#setup-notes-enable", Switch).value
        directory = self.query_one("#setup-notes-directory", Input).value.strip()
        if enabled:
            if not directory:
                return False, "Pick a directory or turn sync off."
            ok = await self.wizard.commit_config(
                build_notes_commit(sync_directory=directory, auto_sync_enabled=True)
            )
            return (True, "") if ok else (False, "Saving notes sync settings failed.")
        # Toggle is off. Task 11's prefill can start this switch ON on
        # re-run, so an OFF-transition is reachable here -- only write the
        # disable when the persisted config currently says ON (fresh config
        # stays a true no-op); sync_directory is deliberately left out of
        # the commit so it survives untouched (see build_notes_commit).
        prefill = read_wizard_prefill(
            getattr(self.wizard.app_instance, "app_config", {}) or {}
        )
        if not prefill.auto_sync_enabled:
            return True, ""
        ok = await self.wizard.commit_config(build_notes_commit(auto_sync_enabled=False))
        return (True, "") if ok else (False, "Saving notes sync settings failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {"auto_sync_enabled": self.query_one("#setup-notes-enable", Switch).value}


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
        if not self.selected_theme and not self.selected_splash_card and not reset_to_random:
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
                yield SetupRadioButton(
                    "Quick setup — provider & model (recommended)",
                    value=True,
                    id="setup-track-quick",
                )
                yield SetupRadioButton("Full setup — configure everything", id="setup-track-full")
            # TASK-1507: tertiary treatment — quiet, link-like, clearly a
            # control but visually subordinate to the track choice above.
            yield Button(
                "Skip — explore on my own",
                id="setup-skip-entirely",
                variant="default",
                classes="setup-tertiary-button",
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
            yield Button("Set a password", id="setup-protect-set-password",
                        variant="primary")
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

    def __init__(self, wizard=None, config=None, *, load_config=None,
                 rag_deps_installed=None, speech_installed=None, **kwargs):
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._load_config = load_config
        self._rag_deps_installed = rag_deps_installed
        # TASK-1301 AC#6: same injectable-callable shape as rag_deps_installed
        # -- defaults to a real, off-loop-safe check of the managed Parakeet
        # v2 artifact's installed/active state.
        self._speech_installed = speech_installed
        self.exit_route: Optional[str] = None

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-summary"):
            yield Static("Setup summary", classes="setup-title")
            yield Static("", id="setup-summary-defaults-note", classes="setup-subtitle")
            # markup=False: row labels/details come from persisted config data
            # (embedding model ids, notes directories, ...) which may contain
            # literal "[...]" -- Static.update() otherwise parses that as Rich
            # markup and silently drops it from the rendered text.
            yield Static("", id="setup-summary-rows", markup=False)
            yield Static("", id="setup-summary-footer", classes="setup-subtitle",
                        markup=False)
        # The exit actions are a DIRECT child of the step (the .setup-step
        # scroll container), not of the scrolling .setup-summary Vertical:
        # Textual docks position against the container's visible frame and
        # never scroll with content, which is what keeps the wizard's final
        # CTAs on screen no matter how tall the read-back matrix gets
        # (TASK-1495 AC #3 -- full-track content previously pushed them
        # below the fold at 120x40).
        with Horizontal(classes="setup-summary-actions"):
            if getattr(self.wizard, "rerun", False):
                yield Button("Done", id="setup-exit-done", variant="primary")
                yield Button("Go to Chat", id="setup-exit-chat")
            else:
                yield Button("Start chatting", id="setup-exit-chat", variant="primary")
                yield Button("Explore on my own", id="setup-exit-home")

    def on_show(self) -> None:
        super().on_show()
        track = (self.wizard.wizard_data or {}).get(
            wizard_state.STEP_WELCOME, {}
        ).get("track")
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

        deps = self._rag_deps_installed
        if deps is None:
            from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

            deps = embeddings_rag_deps_installed
        speech_installed_check = self._speech_installed
        if speech_installed_check is None:

            def speech_installed_check() -> bool:
                return active_managed_parakeet_v2_dir() is not None

        config = await asyncio.get_running_loop().run_in_executor(None, load)
        speech_installed = await asyncio.get_running_loop().run_in_executor(
            None, speech_installed_check
        )
        from tldw_chatbook.UI.Wizards.first_run_setup_state import build_summary_rows

        rows = build_summary_rows(
            config,
            dict(os.environ),
            rag_deps_installed=deps(),
            speech_installed=speech_installed,
        )
        # Static.update() parses "[...]" as Rich markup by default, so any
        # bracketed literal in a label/detail (e.g. a package extra name)
        # must be escaped or it silently vanishes from the rendered text.
        lines = [
            f"{row.glyph} {row.label}"
            + (f" — {row.detail}" if row.detail else "")
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
        from tldw_chatbook.Constants import TAB_CHAT

        self._finish(TAB_CHAT)

    @on(Button.Pressed, "#setup-exit-home")
    def _exit_home(self) -> None:
        from tldw_chatbook.Constants import TAB_HOME

        self._finish(TAB_HOME)

    @on(Button.Pressed, "#setup-exit-done")
    def _exit_done(self) -> None:
        self._finish(None)

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


class SetupWizardContainer(WizardContainer):
    """Navigates over the active-step subset; commits on Next via one worker."""

    def __init__(self, app_instance, rerun: bool = False, **kwargs):
        self.rerun = rerun
        self.key_entered = False
        # TASK-1499: default to the QUICK track — it is the preselected
        # (recommended) Welcome option, so the progress row anchors at
        # "Step 1 of 4" instead of front-loading all nine steps before
        # the user has chosen anything. Picking Full expands it.
        self.track = wizard_state.TRACK_QUICK
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
        self._advancing = False
        # F3 hardening: guards _dismiss_screen/_finalize against ever
        # dismissing the screen twice -- see those methods' docstrings.
        self._finalized = False

    def on_mount(self) -> None:
        """TASK-1499: base on_mount renders the progress row from the FULL
        step list; rebuild it immediately so the initial render honors the
        quick-track default (4 dots, "Step 1 of 4") instead of front-loading
        all nine steps before the user has chosen anything.

        TASK-1266 follow-up: ``self.active_ids`` is first computed in
        ``__init__``, before any step has actually composed -- a step's
        ``compose_failed`` flag can only be known once its own compose()
        has actually run, which Textual does while mounting this
        container's children, i.e. by the time ``super().on_mount()``
        (BaseWizard.on_mount, which calls ``show_step(0)`` and therefore
        forces the children through their mount/compose pipeline) returns
        here. Calling ``_refresh_active_ids()`` -- rather than
        ``_rebuild_progress()`` directly -- re-derives ``active_ids``
        against the now-accurate ``compose_failed`` flags before the very
        first progress/nav render, instead of leaving a step that failed to
        compose counted and shown until some later event (track selection,
        a key being entered) happens to trigger a refresh.
        """
        super().on_mount()
        self._refresh_active_ids()
        self.update_progress()

    # -- step construction -------------------------------------------------
    def _create_steps(self) -> List[WizardStep]:
        # Later tasks append real steps here; the skeleton ships Welcome +
        # placeholder SetupSteps so navigation is testable end to end.
        def cfg(step_id: str, title: str, number: int) -> WizardStepConfig:
            return WizardStepConfig(id=step_id, title=title, step_number=number)

        return [
            WelcomeStep(wizard=self, config=cfg(wizard_state.STEP_WELCOME, "Welcome", 1)),
            ProviderStep(
                wizard=self,
                config=cfg(wizard_state.STEP_PROVIDER, "Provider", 2),
                environ=os.environ,
            ),
            ModelStep(wizard=self, config=cfg(wizard_state.STEP_MODEL, "Model", 3)),
            RagStep(wizard=self, config=cfg(wizard_state.STEP_RAG, "RAG", 4)),
            SpeechSetupStep(
                wizard=self, config=cfg(wizard_state.STEP_SPEECH, "Speech", 5)
            ),
            ToolsStep(wizard=self, config=cfg(wizard_state.STEP_TOOLS, "Tools", 6)),
            NotesSyncStep(wizard=self, config=cfg(wizard_state.STEP_NOTES, "Notes", 7)),
            AppearanceStep(
                wizard=self, config=cfg(wizard_state.STEP_APPEARANCE, "Style", 8)
            ),
            ProtectKeysStep(
                wizard=self, config=cfg(wizard_state.STEP_PROTECT, "Protect", 9)
            ),
            SummaryStep(wizard=self, config=cfg(wizard_state.STEP_SUMMARY, "Summary", 10)),
        ]

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
        # TASK-1266: steps whose compose failed are auto-skipped — they have
        # no usable surface, and the Summary reports them (see
        # compose_failed_steps / SummaryStep._render_rows).
        failed = {
            step.config.id
            for step in self.steps
            if step.config and getattr(step, "compose_failed", False)
        }
        self.active_ids = tuple(sid for sid in ids if sid not in failed)
        self._rebuild_progress()
        # Finding B: a step's compose_failed flag can only be known once its
        # own compose() has actually run -- which may land after this
        # container already displayed it (WelcomeStep is index 0 and
        # BaseWizard.on_mount unconditionally shows it first). If the page
        # currently on screen has since turned out to be a casualty, advance
        # to the next viable active step instead of leaving its "couldn't be
        # shown" notice as the visible page.
        if 0 <= self.current_step < len(self.steps) and getattr(
            self.steps[self.current_step], "compose_failed", False
        ):
            resolved = self._resolve_visible_index(self.current_step)
            if resolved != self.current_step:
                self.show_step(resolved)

    def compose_failed_steps(self) -> list[str]:
        """Titles of steps dropped by the TASK-1266 compose-crash policy.

        Returns:
            Display titles of steps whose composition failed this session.
        """
        return [
            step.config.title
            for step in self.steps
            if step.config and getattr(step, "compose_failed", False)
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
        """Finding B: never show a step whose own compose_step() raised.

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
            ``step_index`` unchanged if that step's compose_step() did not
            fail; otherwise the absolute index of the first active step
            (in active-id order) whose compose_step() succeeded, or
            ``step_index`` itself if every active step has failed.
        """
        if not (0 <= step_index < len(self.steps)):
            return step_index
        if not getattr(self.steps[step_index], "compose_failed", False):
            return step_index
        ids = wizard_state.active_step_ids(
            self.track, key_entered=self._effective_key_entered()
        )
        for step_id in ids:
            index = self._step_index_for_id(step_id)
            if index is None:
                continue
            if not getattr(self.steps[index], "compose_failed", False):
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
        step_index = self._resolve_visible_index(step_index)
        super().show_step(step_index)
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
        except Exception:
            pass

    def _rebuild_progress(self) -> None:
        # WizardProgress has no watchers; replace it wholesale on track change.
        #
        # F-C fix: mount() with no before=/after= appends at the PARENT's
        # END. BaseWizard.compose() yields WizardProgress as the container's
        # SECOND child (right after the title, before the steps container
        # and WizardNavigation) -- a plain parent.mount(fresh) re-inserted
        # the replacement AFTER WizardNavigation instead, rendering the
        # whole progress bar below the Back/Next buttons on every track
        # change (live-verified via tmux screenshot). Capture the sibling
        # that immediately followed the old widget and mount the
        # replacement in that exact slot instead of just appending.
        try:
            old = self.query_one(".wizard-progress", WizardProgress)
            parent = old.parent
            siblings = list(parent.children) if parent is not None else []
            old_index = siblings.index(old) if old in siblings else None
            next_sibling = (
                siblings[old_index + 1]
                if old_index is not None and old_index + 1 < len(siblings)
                else None
            )
            old.remove()
            fresh = WizardProgress(classes="wizard-progress")
            fresh.total_steps = len(self.active_ids)
            fresh.current_step = self._active_position(self.current_step or 0) + 1
            fresh.step_titles = [
                self.steps[self._step_index_for_id(step_id)].config.title
                for step_id in self.active_ids
                if self._step_index_for_id(step_id) is not None
            ]
            if parent is not None:
                if next_sibling is not None:
                    parent.mount(fresh, before=next_sibling)
                else:
                    parent.mount(fresh)
        except Exception:
            logger.debug("Wizard progress rebuild skipped", exc_info=True)

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

        SummaryStep's own exit buttons ("Start chatting", "Explore on my
        own", "Done", "Go to Chat") are not the "#wizard-next" button, so
        they have no Button.Pressed event to hand to handle_next() above --
        which requires one, to call event.prevent_default() (see that
        method's docstring for why). This is the extracted guard + worker
        dispatch body, shared by both callers; the real Next button's
        dispatch semantics (the prevent_default() suppression) are
        unchanged.
        """
        if self._advancing or not self.can_proceed:
            return
        self._advancing = True
        self.run_worker(self._advance(), exclusive=True, group="setup-wizard-advance")

    async def _advance(self) -> None:
        try:
            step = self.steps[self.current_step]
            if isinstance(step, SetupStep):
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
            if next_index is None:
                self.complete_wizard()
            else:
                self.show_step(next_index)
        finally:
            self._advancing = False

    @on(Button.Pressed, "#wizard-back")
    def handle_back(self, event: Button.Pressed) -> None:
        # Same base-class double-dispatch as handle_next; see the comment
        # there. WizardContainer.handle_back() would otherwise also fire and
        # flat-decrement current_step, ignoring the active-id subset.
        event.prevent_default()
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
        self.advance_programmatically()

    def action_back(self) -> None:
        """ctrl+b: same active-subset Back navigation as clicking Back.

        BaseWizard.action_back() calls self.handle_back() with NO
        arguments, which likewise crashes against this class's
        handle_back(event) override. This mirrors that override's body
        exactly, minus the event.prevent_default() call action dispatch has
        no event for.
        """
        previous = self._previous_active_index(self.current_step)
        if previous is not None:
            self.show_step(previous)

    # -- explicit whole-wizard skip ---------------------------------------
    @on(Button.Pressed, "#setup-skip-entirely")
    def handle_skip_entirely(self) -> None:
        self.run_worker(self._skip_entirely(), exclusive=True, group="setup-wizard-advance")

    async def _skip_entirely(self) -> None:
        await self.commit_config(
            wizard_state.build_wizard_state_commit(completed=True)
        )
        self._dismiss_screen({"completed": True, "exit_route": None})

    # -- persistence (the only write path for steps) -----------------------
    async def commit_config(self, section_values: dict) -> bool:
        """Serialize every config write through one worker-side call."""
        if not section_values:
            return True
        if not wizard_state.commit_sections_allowed(section_values):
            logger.error("Wizard commit rejected non-owned sections: {}", list(section_values))
            return False
        import asyncio

        from tldw_chatbook.config import save_settings_to_cli_config

        def _write() -> bool:
            return save_settings_to_cli_config(section_values)

        ok = await asyncio.get_running_loop().run_in_executor(None, _write)
        if ok:
            self._mirror_into_app_config(section_values)
        return ok

    def _mirror_into_app_config(self, section_values: dict) -> None:
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
        await self.commit_config(wizard_state.build_wizard_state_commit(completed=True))
        self._dismiss_screen({"completed": True, "exit_route": exit_route})

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
        if self._finalized:
            return
        self._finalized = True
        screen = self.screen
        if isinstance(screen, FirstRunSetupWizard):
            screen.dismiss(result)

    def action_cancel(self) -> None:
        screen = self.screen
        if isinstance(screen, FirstRunSetupWizard):
            screen.action_cancel()


class FirstRunSetupWizard(WizardScreen):
    """Full-screen first-run setup wizard. Dismisses dict | None."""

    def __init__(self, app_instance, rerun: bool = False):
        super().__init__(app_instance)
        self.rerun = rerun

    def compose(self) -> ComposeResult:
        yield SetupWizardContainer(self.app_instance, rerun=self.rerun)
        # TASK-1505: the wizard's keys are otherwise undiscoverable — one
        # quiet, always-visible line names them.
        yield Static(
            "Ctrl+N next · Ctrl+B back · Esc finish later",
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
            save_settings_to_cli_config(
                wizard_state.build_wizard_state_commit(started=True)
            )
        except Exception as exc:
            logger.warning("Failed to persist wizard started flag: {}", exc)
        app_config = getattr(self.app_instance, "app_config", None)
        if isinstance(app_config, dict):
            app_config.setdefault(wizard_state.WIZARD_STATE_SECTION, {})[
                wizard_state.SETUP_STARTED_KEY
            ] = True

    def action_cancel(self) -> None:
        dialog = ConfirmationDialog(
            title="Finish setup later?",
            message=(
                "Steps you've already completed are saved. You can finish "
                "setup any time from Settings ▸ Diagnostics."
            ),
            confirm_label="Finish later",
            cancel_label="Keep going",
        )
        self.app.push_screen(dialog, self._handle_cancel_confirm)

    def _handle_cancel_confirm(self, confirmed: bool | None) -> None:
        if confirmed:
            # TASK-1500: an uncommitted theme preview must not outlive the
            # wizard — finish-later restores whatever the user had before.
            try:
                self.query_one(AppearanceStep).revert_preview()
            except Exception:
                pass
            self.dismiss(None)
