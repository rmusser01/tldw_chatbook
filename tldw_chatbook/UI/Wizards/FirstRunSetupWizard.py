"""First-run setup wizard: hermes-agent's setup process in chatbook chrome.

Screen + container subclass over BaseWizard (which is never modified).
All decisions and config mutations are built by first_run_setup_state;
this module renders them and owns persistence via one exclusive worker.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static, Switch

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
    """Base step: adds an awaitable commit hook and an inline error line."""

    async def commit(self) -> tuple[bool, str]:
        """Persist this step's data. Return (ok, error_message)."""
        return True, ""

    def show_step_error(self, message: str) -> None:
        try:
            self.query_one(".setup-step-error", Static).update(message)
        except Exception:
            logger.warning("Setup step error had nowhere to render: {}", message)


CLOUD_PROBE_TIMEOUT_SECONDS = 8.0


class ProviderStep(SetupStep):
    """Choose a provider, supply credentials, verify without blocking."""

    def __init__(self, wizard=None, config=None, *, discover=None, probe=None,
                 environ=None, **kwargs):
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

    def compose(self) -> ComposeResult:
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
            with RadioSet(id="setup-provider-choice"):
                for entry in self._grouped(entries):
                    yield RadioButton(
                        entry.display_name, id=f"setup-provider-{entry.readiness_key}"
                    )
            yield Static("", id="setup-provider-detected", classes="setup-probe-status")
            yield Button(
                "Use this server", id="setup-provider-use-detected",
                classes="hidden", variant="primary",
            )
            yield Label("API key", classes="setup-field-label")
            yield Input(password=True, id="setup-provider-key-input",
                        placeholder="Paste your API key")
            yield Static("", id="setup-provider-key-status", classes="setup-probe-status")
            with Horizontal(id="setup-provider-key-actions", classes="hidden"):
                yield Button("Keep current", id="setup-provider-key-keep")
                yield Button("Replace", id="setup-provider-key-replace")
                yield Button("Clear", id="setup-provider-key-clear")
            yield Static("", id="setup-provider-probe-status", classes="setup-probe-status")
            yield Static("", classes="setup-step-error")

    @staticmethod
    def _grouped(entries):
        from tldw_chatbook.Chat.provider_catalog import (
            PROVIDER_CUSTOM_GROUP_KEYS,
        )

        def group_rank(entry):
            if entry.readiness_key in PROVIDER_CUSTOM_GROUP_KEYS:
                return 2
            return 0 if entry.requires_api_key else 1

        return sorted(entries, key=lambda e: (group_rank(e), e.display_name.lower()))

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
        self.query_one("#setup-provider-detected", Static).update(
            f"Found a local server at {self.detected_server.base_url} "
            f"({self.detected_server.provider_key})."
        )
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
                timeout=CLOUD_PROBE_TIMEOUT_SECONDS if api_key else 2.5,
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
        )

        if not self.selected_provider_key:
            return True, ""  # skipping the step entirely is legal
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
        commit = invalidate_model_for_provider_change(
            commit,
            previous_provider_value=self._last_committed_provider_value,
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


class WelcomeStep(SetupStep):
    """Track choice: Quick / Full / Skip."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="setup-welcome"):
            yield Static("Welcome to tldw chatbook", classes="setup-title")
            yield Static(
                "Let's get you set up. Pick a path — everything here can be "
                "changed later in Settings, and every step can be skipped.",
                classes="setup-subtitle",
            )
            with RadioSet(id="setup-track-choice"):
                yield RadioButton(
                    "Quick setup — provider & model (recommended)",
                    value=True,
                    id="setup-track-quick",
                )
                yield RadioButton("Full setup — configure everything", id="setup-track-full")
            yield Button(
                "Skip — explore on my own", id="setup-skip-entirely", variant="default"
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


class SetupWizardContainer(WizardContainer):
    """Navigates over the active-step subset; commits on Next via one worker."""

    def __init__(self, app_instance, rerun: bool = False, **kwargs):
        self.rerun = rerun
        self.key_entered = False
        self.track = wizard_state.TRACK_FULL
        steps = self._create_steps()
        super().__init__(
            app_instance=app_instance,
            steps=steps,
            title="Set up tldw chatbook",
            on_complete=self._handle_complete,
            **kwargs,
        )
        self.active_ids: tuple[str, ...] = wizard_state.active_step_ids(
            self.track, key_entered=self.key_entered
        )
        self._advancing = False

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
            SetupStep(wizard=self, config=cfg(wizard_state.STEP_MODEL, "Model", 3)),
            SetupStep(wizard=self, config=cfg(wizard_state.STEP_RAG, "RAG", 4)),
            SetupStep(wizard=self, config=cfg(wizard_state.STEP_TOOLS, "Tools", 5)),
            SetupStep(wizard=self, config=cfg(wizard_state.STEP_NOTES, "Notes sync", 6)),
            SetupStep(
                wizard=self, config=cfg(wizard_state.STEP_APPEARANCE, "Appearance", 7)
            ),
            SetupStep(
                wizard=self, config=cfg(wizard_state.STEP_PROTECT, "Protect keys", 8)
            ),
            SetupStep(wizard=self, config=cfg(wizard_state.STEP_SUMMARY, "Summary", 9)),
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

    def _refresh_active_ids(self) -> None:
        self.active_ids = wizard_state.active_step_ids(
            self.track, key_entered=self.key_entered
        )
        self._rebuild_progress()

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
        try:
            old = self.query_one(".wizard-progress", WizardProgress)
            parent = old.parent
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
        self.run_worker(
            self._finalize(exit_route), exclusive=True, group="setup-wizard-advance"
        )

    async def _finalize(self, exit_route: Optional[str]) -> None:
        await self.commit_config(wizard_state.build_wizard_state_commit(completed=True))
        self._dismiss_screen({"completed": True, "exit_route": exit_route})

    def _dismiss_screen(self, result: Optional[dict]) -> None:
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
            self.dismiss(None)
