"""Focused llama.cpp tuning, readiness, and verified navigation controls."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING
from uuid import uuid4

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.widgets import Button, Collapsible, Input, Label, Select, Static

from tldw_chatbook.LLM_Management.llamacpp_connection import (
    canonical_base_url,
    connection_owner,
    local_launch_url,
    probe_llamacpp_target,
)
from tldw_chatbook.LLM_Management.llamacpp_profiles import (
    LlamaCppLaunchProfileV1,
    LlamaCppProfileDocumentV1,
    LlamaCppProfileError,
    LlamaCppProfileRepository,
    LlamaCppTuning,
)

if TYPE_CHECKING:
    from tldw_chatbook.LLM_Management.llamacpp_launch_preview import (
        LlamaCppLaunchOptions,
    )


_FIELDS = {
    "context_size": "Context size",
    "gpu_layers": "GPU layers (-1 = all)",
    "threads": "CPU threads",
    "parallel": "Parallel slots",
    "flash_attention": "Flash attention (on / off / auto)",
    "cache_type_k": "K cache type",
    "cache_type_v": "V cache type",
    "batch_size": "Logical batch size",
    "ubatch_size": "Physical batch size",
}
_TEXT_FIELDS = {"flash_attention", "cache_type_k", "cache_type_v"}
_STATUS = {
    "not_configured": "Connection unchecked.",
    "target_changed": "Target changed. Check again.",
    "checking": "Checking API health and model identity…",
    "ready": "API and model verified. Ready to use in Console.",
    "loading_model": "Model is loading. Check again after loading completes.",
    "choose_model": "Select a model, then check again.",
    "process_unavailable": "Local process is stopped or unavailable.",
    "credential_required": "Authentication required. Configure credentials for this exact endpoint in Settings, then check again.",
    "unsafe_model_id": "Server model ID is unsafe. Configure a non-path --alias and check again.",
    "model_missing": "Expected model is unavailable. Check the server alias and model.",
    "timeout": "Connection check timed out. Check the endpoint and try again.",
    "connection_failed": "Connection failed. Check the endpoint and server.",
    "health_failed": "Health check failed. Check the server configuration.",
    "invalid_models_response": "Server returned an unsupported model list.",
    "invalid_endpoint": "Enter an HTTP(S) endpoint without credentials, query, or fragment.",
    "screen_detached": "Connection unchecked. Check again before use.",
}


class LlamaCppSetupView(Vertical):
    """Session-local launch draft with device-local tuning profiles."""

    DEFAULT_CSS = """
    LlamaCppSetupView { height: auto; }
    LlamaCppSetupView Horizontal { height: auto; }
    LlamaCppSetupView .llamacpp-status { height: auto; }
    """

    def __init__(
        self,
        app_instance,
        *,
        repository=None,
        launch_options: Callable[[], LlamaCppLaunchOptions] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(id="llamacpp-setup-view", **kwargs)
        self.app_instance = app_instance
        self._launch_options = launch_options
        self._preview_text: str | None = None
        self._current_preview_text: str | None = None
        self.owner = connection_owner(app_instance)
        self.repository = repository or LlamaCppProfileRepository()
        self._profiles = LlamaCppProfileDocumentV1(version=1, revision=0, profiles=())
        self._profiles_loaded = False
        self._profile_busy = False
        self._hydrating = True
        self._local_claim = None
        self._local_url = None
        self._departing = None
        self._staged = None
        self._last_models = ()
        self._last_diagnostics = None
        self._draft = getattr(app_instance, "_llamacpp_lab_draft", {})
        if hasattr(app_instance, "_llm_server_lifecycle_lock"):
            from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
                current_server_claim,
            )

            claim = current_server_claim(app_instance, "llamacpp")
            if claim is not None and claim._connection_url is not None:
                self._local_claim = claim
                self._local_url = claim._connection_url

    def compose(self) -> ComposeResult:
        with Collapsible(title="Tuning and launch profiles", collapsed=True):
            yield Select(
                [("Runtime defaults", "defaults")],
                value="defaults",
                allow_blank=False,
                id="llamacpp-profile-select",
            )
            yield Input(
                value=self._draft.get("name", ""),
                placeholder="Profile name",
                id="llamacpp-profile-name",
            )
            with Horizontal():
                for action in ("save", "delete", "reload"):
                    yield Button(
                        action.title(), id=f"llamacpp-profile-{action}", disabled=True
                    )
            yield Static(
                "Loading profiles…",
                id="llamacpp-profile-status",
                classes="llamacpp-status",
            )
            for field, label in _FIELDS.items():
                yield Label(label)
                yield Input(
                    value=self._draft.get(field, ""),
                    placeholder="Runtime default",
                    id="llamacpp-" + field.replace("_", "-"),
                )
        with Collapsible(
            title="Launch preview", collapsed=True, id="llamacpp-launch-preview"
        ):
            yield Static(
                "",
                id="llamacpp-current-launch",
                classes="llamacpp-status",
                markup=False,
            )
            yield Static(
                "Launch draft", id="llamacpp-preview-title", classes="section-title"
            )
            yield Static(
                "Preview the current launch settings before starting.",
                id="llamacpp-preview-command",
                classes="llamacpp-status",
                markup=False,
            )
            with Horizontal(classes="llamacpp-preview-actions"):
                yield Button("Preview launch", id="llamacpp-preview-launch")
                yield Button("Copy redacted", id="llamacpp-copy-launch", disabled=True)
        yield Label("Verified connection", classes="section-title")
        yield Input(
            value=self._draft.get("endpoint", ""),
            placeholder="Existing server URL, e.g. http://127.0.0.1:8080",
            id="llamacpp-existing-url",
        )
        yield Select(
            [],
            prompt="Verified server model",
            id="llamacpp-connection-model",
            disabled=True,
        )
        with Horizontal():
            yield Button("Check connection", id="llamacpp-check")
            yield Button("Use in Console", id="llamacpp-use-console", disabled=True)
            yield Button("Make default…", id="llamacpp-make-default", disabled=True)
        yield Static(
            "Connection unchecked.",
            id="llamacpp-connection-status",
            classes="llamacpp-status",
        )
        yield Static("", id="llamacpp-bind-status", classes="llamacpp-status")
        yield Static(
            "Make default opens Settings for review and Save.",
            classes="llamacpp-status",
        )

    def on_mount(self) -> None:
        self.run_worker(
            self.profile_action("reload"), group="llamacpp-profiles", exclusive=True
        )
        self.call_after_refresh(self._finish_hydration)
        self.set_interval(0.5, self.refresh_state)
        self.refresh_state()

    def _finish_hydration(self) -> None:
        self._hydrating = False

    def _remember(self) -> None:
        if not self.is_mounted:
            return
        draft = {
            field: self.query_one("#llamacpp-" + field.replace("_", "-"), Input).value
            for field in _FIELDS
        }
        draft["name"] = self.query_one("#llamacpp-profile-name", Input).value
        try:
            draft["endpoint"] = canonical_base_url(
                self.query_one("#llamacpp-existing-url", Input).value
            )
        except ValueError:
            draft["endpoint"] = ""
        draft["profile_id"] = str(
            self.query_one("#llamacpp-profile-select", Select).value
        )
        self.app_instance._llamacpp_lab_draft = draft

    def tuning(self) -> LlamaCppTuning:
        values = {}
        for field in _FIELDS:
            value = self.query_one(
                "#llamacpp-" + field.replace("_", "-"), Input
            ).value.strip()
            values[field] = (
                (value if field in _TEXT_FIELDS else int(value)) if value else None
            )
        return LlamaCppTuning(**values)

    def _apply_tuning(self, tuning: LlamaCppTuning) -> None:
        for field in _FIELDS:
            widget = self.query_one("#llamacpp-" + field.replace("_", "-"), Input)
            value = getattr(tuning, field)
            with widget.prevent(Input.Changed):
                widget.value = "" if value is None else str(value)
        self.launch_draft_changed()
        self._remember()

    def _active(self) -> bool:
        from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
            server_is_active,
        )

        if not hasattr(self.app_instance, "_llm_server_lifecycle_lock"):
            return False
        return server_is_active(self.app_instance, "llamacpp")

    async def profile_action(self, action: str) -> None:
        if self._profile_busy or (action != "reload" and not self._profiles_loaded):
            return
        self._profile_busy = True
        self.refresh_state()
        try:
            selected = self.query_one("#llamacpp-profile-select", Select).value
            if not self._profiles_loaded:
                selected = self._draft.get("profile_id", selected)
            if action == "save":
                profile = LlamaCppLaunchProfileV1(
                    profile_id=str(uuid4())
                    if selected == "defaults"
                    else str(selected),
                    name=self.query_one("#llamacpp-profile-name", Input).value,
                    tuning=self.tuning(),
                )
                document = await asyncio.to_thread(
                    self.repository.save,
                    profile,
                    expected_revision=self._profiles.revision,
                )
                selected = profile.profile_id
            elif action == "delete":
                if selected == "defaults":
                    return
                document = await asyncio.to_thread(
                    self.repository.delete,
                    str(selected),
                    expected_revision=self._profiles.revision,
                )
                selected = "defaults"
            else:
                document = await asyncio.to_thread(self.repository.load)
            if not self.is_mounted:
                return
            self._profiles = document
            self._profiles_loaded = True
            selector = self.query_one("#llamacpp-profile-select", Select)
            options = [
                ("Runtime defaults", "defaults"),
                *((p.name, p.profile_id) for p in document.profiles),
            ]
            with selector.prevent(Select.Changed):
                selector.set_options(options)
                selector.value = (
                    selected
                    if selected in {value for _, value in options}
                    else "defaults"
                )
            self._remember()
            self.query_one("#llamacpp-profile-status", Static).update(
                "Profiles loaded."
                if action == "reload"
                else "Profile saved."
                if action == "save"
                else "Profile deleted; current tuning kept as a draft."
            )
        except (LlamaCppProfileError, ValueError, OSError):
            if self.is_mounted:
                self.query_one("#llamacpp-profile-status", Static).update(
                    "Profile action failed. Check values and name, then Reload before retrying. Existing profiles were preserved."
                )
        finally:
            self._profile_busy = False
            if self.is_mounted:
                self.refresh_state()

    @on(Select.Changed, "#llamacpp-profile-select")
    def _profile_selected(self, event: Select.Changed) -> None:
        event.stop()
        if self._hydrating or not self._profiles_loaded:
            return
        profile = next(
            (p for p in self._profiles.profiles if p.profile_id == event.value), None
        )
        self._apply_tuning(profile.tuning if profile else LlamaCppTuning())
        name = self.query_one("#llamacpp-profile-name", Input)
        with name.prevent(Input.Changed):
            name.value = profile.name if profile else ""
        self._remember()

    @on(Input.Changed)
    def _input_changed(self, event: Input.Changed) -> None:
        if event.input not in self.query(Input):
            return
        event.stop()
        if self._hydrating:
            return
        self._remember()
        if event.input.id != "llamacpp-profile-name":
            if event.input.id != "llamacpp-existing-url":
                self.launch_draft_changed()
            else:
                self.invalidate()
            if event.input.id == "llamacpp-existing-url":
                self._local_claim = None
                self._local_url = None
                self._last_models = ()
                selector = self.query_one("#llamacpp-connection-model", Select)
                with selector.prevent(Select.Changed):
                    selector.set_options([])
                    selector.value = Select.NULL
                self.refresh_state()

    @on(Select.Changed, "#llamacpp-connection-model")
    def _model_selected(self, event: Select.Changed) -> None:
        event.stop()
        self.invalidate()

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        action = event.button.id or ""
        if event.button not in self.query(Button):
            return
        event.stop()
        if action in {"llamacpp-preview-launch", "llamacpp-copy-launch"}:
            if self.preview_launch() and action == "llamacpp-copy-launch":
                self.app_instance.copy_to_clipboard(self._preview_text)
                self.app_instance.notify("Redacted launch preview copied.")
            # The summary may grow above the invoking action. Scroll only after
            # its new layout exists so keyboard focus remains on screen.
            self.call_after_refresh(event.button.scroll_visible, animate=False)
        elif action.startswith("llamacpp-profile-"):
            self.run_worker(
                self.profile_action(action.removeprefix("llamacpp-profile-")),
                group="llamacpp-profiles",
                exclusive=True,
            )
        elif action == "llamacpp-check":
            self.run_worker(
                self.check_connection(), group="llamacpp-readiness", exclusive=True
            )
        elif action in {
            "llamacpp-use-console",
            "llamacpp-make-default",
        } and not self.stage_handoff(default=action == "llamacpp-make-default"):
            self.app_instance.notify(
                "Connection changed. Check again before use.", severity="warning"
            )

    def launch_draft_changed(self) -> None:
        """Fence a stale preview without invalidating a different, live launch."""
        self._preview_text = None
        if self.is_mounted:
            self.query_one("#llamacpp-preview-command", Static).update(
                "Launch settings changed. Preview again before copying."
            )
            self.query_one("#llamacpp-copy-launch", Button).disabled = True
        if self._active():
            self.refresh_state()
        else:
            self.invalidate()

    def preview_launch(self) -> bool:
        """Render a fresh side-effect-free, redacted launch draft."""
        from tldw_chatbook.LLM_Management.llamacpp_launch_preview import (
            INVALID_TUNING_MESSAGE,
            LlamaCppLaunchValidationError,
        )

        self._preview_text = None
        try:
            if self._launch_options is None:
                raise LlamaCppLaunchValidationError(
                    "Launch controls are unavailable. Reopen Models and try again."
                )
            self._preview_text = self._launch_options().preview
            message = self._preview_text
        except LlamaCppLaunchValidationError as error:
            message = str(error)
        except (ValueError, TypeError):
            message = INVALID_TUNING_MESSAGE
        self.query_one("#llamacpp-preview-command", Static).update(message)
        self.query_one("#llamacpp-copy-launch", Button).disabled = (
            self._preview_text is None
        )
        self._render_launch_state()
        return self._preview_text is not None

    def _render_launch_state(self) -> None:
        """Read the exact claim's sanitized summary; never rebuild it from edits."""
        from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
            current_server_claim,
        )

        active = self._active()
        claim = current_server_claim(self.app_instance, "llamacpp") if active else None
        current = getattr(claim, "_launch_preview", None)
        message = (
            "Current launch\n" + (current or "Preview unavailable for this launch.")
            if active
            else ""
        )
        widget = self.query_one("#llamacpp-current-launch", Static)
        widget.display = active
        if message != self._current_preview_text:
            self._current_preview_text = message
            widget.update(message)
        self.query_one("#llamacpp-preview-title", Static).update(
            "Next launch — current server unchanged" if active else "Launch draft"
        )

    def invalidate(self) -> None:
        self._departing = None
        self.owner.invalidate()
        self.refresh_state()

    def launch_started(self, host: str, port: str, claim) -> None:
        self._local_claim = claim
        self._local_url = local_launch_url(host, port)
        claim._connection_url = self._local_url
        widget = self.query_one("#llamacpp-existing-url", Input)
        with widget.prevent(Input.Changed):
            widget.value = self._local_url
        self.invalidate()
        self._remember()
        self.run_worker(
            self.check_connection(local=True),
            group="llamacpp-readiness",
            exclusive=True,
        )

    def _credential(self, base_url: str) -> str | None:
        # Never forward a configured secret to a newly entered endpoint.
        from tldw_chatbook.Chat.provider_readiness import resolve_provider_credential

        config = getattr(self.app_instance, "app_config", {})
        api_settings = (
            config.get("api_settings", {}) if isinstance(config, Mapping) else {}
        )
        settings = (
            api_settings.get("llama_cpp", {})
            if isinstance(api_settings, Mapping)
            else {}
        )
        if not isinstance(settings, Mapping):
            return None
        configured = (
            settings.get("api_base")
            or settings.get("api_url")
            or settings.get("base_url")
        )
        if not isinstance(configured, str):
            return None
        try:
            if canonical_base_url(configured) != base_url:
                return None
        except ValueError:
            return None
        return resolve_provider_credential("llama_cpp", settings, environ=os.environ)[0]

    async def check_connection(self, *, local: bool = False) -> None:
        from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
            current_server_claim,
            snapshot_claim_is_live,
        )

        local = local or self._local_claim is not None
        claim = self._local_claim
        try:
            endpoint = (
                self._local_url
                if local
                else self.query_one("#llamacpp-existing-url", Input).value
            )
            selected = self.query_one("#llamacpp-connection-model", Select).value
            request = self.owner.begin(
                endpoint,
                runtime_owner="lab_process" if local else "external_server",
                model_id=None if local or selected is Select.NULL else str(selected),
                live_check=(lambda: snapshot_claim_is_live(self.app_instance, claim))
                if local
                else None,
            )
        except (TypeError, ValueError):
            self.owner.invalidate("invalid_endpoint")
            self.query_one("#llamacpp-connection-status", Static).update(
                "Enter a valid HTTP(S) endpoint without credentials, query, or fragment."
            )
            return
        self._departing = None
        self.refresh_state()
        for attempt in range(60 if local else 1):
            if not self.is_mounted or self.owner.snapshot().request is not request:
                return
            result = await probe_llamacpp_target(
                request, credential=self._credential(request.base_url)
            )
            if not self.is_mounted or not self.owner.accept(result):
                return
            self.refresh_state()
            if result.code == "ready" or not local:
                return
            if (
                claim.cancel_event.is_set()
                or current_server_claim(self.app_instance, "llamacpp") is not claim
            ):
                return
            if result.code not in {
                "loading_model",
                "connection_failed",
                "process_unavailable",
                "timeout",
            }:
                return
            if attempt < 59:
                await asyncio.sleep(1)

    def refresh_state(self) -> None:
        if not self.is_mounted:
            return
        active = self._active()
        self._render_launch_state()
        self.query_one("#llamacpp-profile-select", Select).disabled = (
            self._profile_busy or not self._profiles_loaded
        )
        for action in ("save", "delete", "reload"):
            self.query_one("#llamacpp-profile-" + action, Button).disabled = (
                self._profile_busy or (action != "reload" and not self._profiles_loaded)
            )
        snapshot = self.owner.snapshot()
        status = _STATUS.get(snapshot.code, "Connection unavailable. Check again.")
        if (
            snapshot.code == "process_unavailable"
            and active
            and self._local_claim is not None
            and not self._local_claim.cancel_event.is_set()
        ):
            status = "Starting process. Waiting for API readiness…"
        self.query_one("#llamacpp-connection-status", Static).update(status)
        bind_status = self.query_one("#llamacpp-bind-status", Static)
        exposed = active and bool(
            getattr(self._local_claim, "_connection_exposed", False)
        )
        bind_status.display = exposed
        bind_status.update(
            "Server listens beyond loopback. Review server authentication and network access."
            if exposed
            else ""
        )
        for action in ("use-console", "make-default"):
            self.query_one("#llamacpp-" + action, Button).disabled = (
                snapshot.target is None
            )
        self.query_one("#llamacpp-check", Button).disabled = (
            snapshot.state == "checking"
        )
        self.query_one("#llamacpp-existing-url", Input).disabled = active
        model = self.query_one("#llamacpp-connection-model", Select)
        if snapshot.model_ids and snapshot.model_ids != self._last_models:
            self._last_models = snapshot.model_ids
            with model.prevent(Select.Changed):
                model.set_options([(value, value) for value in snapshot.model_ids])
                model.value = (
                    snapshot.target.model_id if snapshot.target else Select.NULL
                )
        model.disabled = active or not self._last_models
        self._render_diagnostics()

    def _render_diagnostics(self) -> None:
        from textual.widgets import RichLog

        claim = getattr(self.app_instance, "_llamacpp_diagnostics_claim", None)
        sink = getattr(claim, "_diagnostics", None)
        entries = sink.snapshot() if sink is not None else ()
        identity = (claim, entries)
        if identity == self._last_diagnostics:
            return
        self._last_diagnostics = identity
        try:
            log = self.parent.query_one("#llamacpp-log-output", RichLog)
        except (QueryError, AttributeError):
            return
        log.clear()
        for entry in entries:
            log.write(entry)

    def stage_handoff(self, *, default: bool) -> bool:
        from tldw_chatbook.Constants import TAB_CHAT, TAB_SETTINGS
        from tldw_chatbook.UI.Navigation.llamacpp_handoff import (
            LlamaCppConsoleIntent,
            LlamaCppDefaultIntent,
        )
        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
        from tldw_chatbook.UI.Navigation.pending_handoff_store import (
            HandoffChannel,
            PendingHandoffStore,
        )

        target = self.owner.snapshot().target
        store = getattr(self.app_instance, "pending_handoffs", None)
        if (
            not self.is_attached
            or target is None
            or type(store) is not PendingHandoffStore
        ):
            return False
        intent = (
            LlamaCppDefaultIntent if default else LlamaCppConsoleIntent
        ).from_target(target)
        channel = (
            HandoffChannel.LLAMACPP_DEFAULT
            if default
            else HandoffChannel.LLAMACPP_CONSOLE
        )
        try:
            revision = store.stage(channel, intent)
        except (ValueError, TypeError, RuntimeError):
            return False
        self._departing = target
        self._staged = (channel, revision, intent)
        try:
            posted = self.owner.is_current(target) and self.post_message(
                NavigateToScreen(
                    TAB_SETTINGS if default else TAB_CHAT,
                    {"category": "providers-models"} if default else None,
                )
            )
        except Exception:  # noqa: BLE001 - failed UI dispatch must discard its exact intent
            posted = False
        if posted:
            return True
        store.discard_pending_exact(channel, revision, intent)
        self._departing = None
        self._staged = None
        return False

    def deactivate(self) -> None:
        if self.query(Input):
            self._remember()
        store = getattr(self.app_instance, "pending_handoffs", None)
        preserve = False
        if (
            self._staged is not None
            and self._departing is not None
            and self.owner.is_current(self._departing)
        ):
            channel, revision, _intent = self._staged
            preserve = store.exact_revision_status(channel, revision) in {
                "pending",
                "in_flight",
            }
        if not preserve:
            self.owner.invalidate("screen_detached")
        self.workers.cancel_group(self, "llamacpp-readiness")

    def on_unmount(self) -> None:
        self.deactivate()
