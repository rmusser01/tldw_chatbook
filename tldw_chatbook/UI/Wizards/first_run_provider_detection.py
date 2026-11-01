"""Local-server detection for the first-run Provider step (TASK-34100.6).

Moved out of ``first_run_provider_step.py`` so the step's detection rework has
room under the size ratchet. ``ProviderDetectionStep`` is a ``SetupStep``
subclass (not a plain mixin) so Textual registers its ``@on`` handlers;
``ProviderStep`` inherits from it. Patch this module to replace what the
detection code calls.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Sequence

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Collapsible, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.UI.Wizards.first_run_provider_skip import missing_url_reason
from tldw_chatbook.UI.Wizards.first_run_setup_state import first_run_endpoint_refusal
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import SetupStep
from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker


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


#: A failed check's categories where nothing answered: only then does the
#: no-server how-to apply (review round 2, G6-VR2-09).
NOTHING_ANSWERED = frozenset({"connection_refused", "connection_error", "timeout"})
#: Seconds between quiet loopback re-checks while Provider is visible and
#: nothing answered (review E3): a server started mid-setup appears unasked.
RECHECK_SECONDS = 15.0
#: 'Don't have a local server yet?' (review provider-14): short, and pointing
#: at the User Guide rather than install commands that go stale.
NO_SERVER_HELP = (
    "How to run Ollama, LM Studio or llama.cpp: see 'Local servers' in the "
    "User Guide's First-Run Setup page.",
    "Set it up now, start it later: pick your engine, press Next, and type the "
    "model you'll run.",
    "Popular models to run: llama3.2 (ollama pull llama3.2).",
)
#: E3: the 'start it later' path for the popular model above.
LATER_ENGINE, LATER_MODEL = "ollama", "llama3.2"
LATER_LABEL = "Use llama3.2 with Ollama — I'll start it later"


def tls_failure_line(display: str, category: object, control: str) -> str:
    """A certificate or plain-HTTP failure's own line, or "" (coverage-19).

    Shared by every check on the step -- the selected-provider check and
    Check key (review round 2, R2-03) -- and names only a control that is on
    screen (R2-04).
    """
    if category == "tls_untrusted":
        then = f", then press {control}." if control else "."
        return (
            f"{display} answered, but its certificate isn't trusted. Add your CA "
            f"bundle in Settings ▸ Network{then}"
        )
    if category == "tls_plain_http":
        return "This server speaks plain HTTP — use http:// instead of https://."
    return ""


def checked_addresses(app_config: Any) -> str:
    """'127.0.0.1:8080, :9099, :11434, :1234, :1337 and :5001' (provider-14)."""
    from urllib.parse import urlsplit

    from tldw_chatbook.Chat.local_server_discovery import build_local_server_candidates

    parts: list[str] = []
    last_host = ""
    for candidate in build_local_server_candidates(app_config):
        address = urlsplit(candidate.base_url)
        host, port = address.hostname or "", address.port
        if host in {"localhost", "127.0.0.1"} and any(
            p.endswith(f":{port}") for p in parts
        ):
            continue  # both loopback spellings of one port read as one
        parts.append(f":{port}" if host == last_host else f"{host}:{port}")
        last_host = host
    if len(parts) < 2:
        return "".join(parts)
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


class QuietStatic(Static):
    """A status line that takes no rows while it says nothing.

    Review round 2, G6-VR2-06 / G6-VR2-11: empty lines (the chat URL, the
    endpoint's status, the step's status) each held a row plus a margin, so
    a Test result landed below the fold at 120x40 and 8-10 blank rows sat
    between a per-account URL field and Authentication.
    """

    def __init__(self, content: Any = "", **kwargs: Any) -> None:
        # Review v3r1, R1-10: these lines carry what the user typed (an
        # endpoint's path), so they print it literally (CLAUDE.md, TASK-1513).
        kwargs.setdefault("markup", False)
        super().__init__(content, **kwargs)
        self.display = bool(str(content).strip())

    def update(self, content: Any = "", *, layout: bool = True) -> None:
        super().update(content, layout=layout)
        self.display = bool(str(content).strip())


class ProviderDetectionStep(SetupStep):
    """Detection half of the Provider step: the localhost scan and its rows."""

    _recheck_timer: Any = None
    #: The next settled status line answers a click: bring it on screen.
    _reveal_status: bool = False
    _found_ever_scanned = False

    def compose_no_server_block(self) -> ComposeResult:
        """What was checked and how to go on when nothing answered (provider-14)."""
        with Vertical(id="setup-provider-no-server", classes="setup-no-server hidden"):
            yield Static("", id="setup-provider-no-server-text", classes="setup-probe-status")
            yield Button("Check again", id="setup-provider-check-again")
            with Collapsible(
                title="Don't have a local server yet?",
                collapsed=True,
                id="setup-provider-no-server-help",
            ):
                for line in NO_SERVER_HELP:
                    yield Static(line, classes="setup-no-server-line setup-subtitle")
                yield Button(LATER_LABEL, id="setup-provider-use-later")

    def _start_recheck(self) -> None:
        """E3: re-check loopback on a slow cadence while the step is visible."""
        if self._recheck_timer is None:
            self._recheck_timer = self.set_interval(RECHECK_SECONDS, self._quiet_recheck)

    def _stop_recheck(self) -> None:
        timer, self._recheck_timer = self._recheck_timer, None
        if timer is not None:
            timer.stop()

    def _quiet_recheck(self) -> None:
        """One quiet loopback scan while the step shows; it only adds rows.

        Review v3r1, G6-BV3-04: it ran only while nothing had been found, so a
        server started mid-setup beside one already listed never appeared.
        """
        if (
            not getattr(self, "_discovery_visible", False)
            or self._local_discovery_state == "in_progress"
        ):
            return
        self._start_discovery(quiet=True)

    def _render_no_server_state(self, servers: Sequence[object]) -> None:
        """'Nothing answered' while it is the relevant news (provider-14)."""
        try:
            text = self.query_one("#setup-provider-no-server-text", Static)
        except NoMatches:
            return
        key = getattr(self, "selected_provider_key", "")
        relevant = not key or self._setup_facts(key).locality == "local"
        self._no_server_shown = bool(self._found_ever_scanned and not servers and relevant)
        if self._no_server_shown:
            app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
            text.update(
                f"Looked on this computer at {checked_addresses(app_config)} — "
                "nothing answered."
            )
        text.display = self._no_server_shown
        self._sync_check_again()

    def _sync_check_again(self, *, failed: bool | None = None) -> None:
        """The block with 'Check again' shows only where checking again helps."""
        try:
            block = self.query_one("#setup-provider-no-server")
        except NoMatches:
            return
        if failed is not None:
            key = getattr(self, "selected_provider_key", "")
            self._selected_check_failed = bool(
                failed and key and self._setup_facts(key).locality == "local"
            )
        visible = bool(
            getattr(self, "_no_server_shown", False)
            or getattr(self, "_selected_check_failed", False)
        )
        block.set_class(not visible, "hidden")
        block.display = visible
        # G6-VR2-09: a server that answered (plain HTTP, a certificate, an
        # HTTP error) is no case for 'nothing answered' or the how-to.
        answered = visible and self._selected_server_answered()
        for selector, shown in (
            ("#setup-provider-no-server-help", not answered),
            ("#setup-provider-no-server-text", getattr(self, "_no_server_shown", False) and not answered),
        ):
            for node in self.query(selector):
                node.display = shown

    def _selected_server_answered(self) -> bool:
        """Whether the selected server answered its last check or Test."""
        categories: list[object] = []
        outcomes = getattr(self, "_selected_provider_outcomes", {})
        outcome = outcomes.get(getattr(self, "_selected_discovery_key", None))
        if outcome is not None:
            error = getattr(outcome, "error", None)
            if error is None or getattr(error, "kind", None) == "invalid_response":
                categories.append("answered")
            else:
                categories.append(getattr(error, "category", None))
        tested = getattr(self, "_last_tested_provider_identity", None)
        store = getattr(self, "_provider_test_evidence", None)
        evidence = store.evidence_for(tested) if store is not None and tested is not None else None
        if evidence is not None:
            unreachable = evidence.endpoint == "unreachable"
            categories.append(evidence.category if unreachable else "answered")
        return any(c is not None and c not in NOTHING_ANSWERED for c in categories)

    @on(Button.Pressed, "#setup-provider-check-again")
    def _on_check_again(self, event: Button.Pressed) -> None:
        event.stop()
        self._reveal_status = True
        self._cancel_worker_groups("setup-provider-local-discovery")
        self._start_discovery()
        if getattr(self, "selected_provider_key", ""):
            self._begin_selected_provider_discovery(self._effective_provider_draft())

    @on(Button.Pressed, "#setup-provider-use-later")
    def _on_use_later(self, event: Button.Pressed) -> None:
        """E3: pick the engine now; Model starts with the model typed in."""
        event.stop()
        from tldw_chatbook.Chat.provider_catalog import provider_display_name

        setattr(self.wizard, "first_run_model_hint", (LATER_ENGINE, LATER_MODEL))
        self.select_provider(LATER_ENGINE)
        engine = provider_display_name(LATER_ENGINE)  # G6-BV3-06: say it took
        self._set_status(
            f"{engine} with {LATER_MODEL} is set — press Next, and start {engine} "
            "before your first chat."
        )

    def _set_status(self, text: str) -> None:
        """The step's one status line (review cross-cutting-07, provider-16).

        Every writer goes through here; the leading glyph picks the outcome
        style: ✓ success, ✗ error, ! warning, anything else neutral.
        """
        self._stale_line_owed = False  # any line replaces an owed 'test again'
        try:
            status = self.query_one("#setup-provider-probe-status", Static)
        except NoMatches:
            return
        status.update(text)
        for glyph, style in (("✓", "-ok"), ("✗", "-error"), ("!", "-warn")):
            status.set_class(text.startswith(glyph), style)
        if text and self._reveal_status:  # G6-VR2-06: a click's answer shows
            self._reveal_status = text.endswith("…")  # 'Testing…' is interim
            # Immediate, from the offset it runs at: a deferred scroll kept a
            # target computed before a refusal's Authentication reveal and
            # undid it (TASK-33008's test, flaky after G6-VR2-06).
            self.call_after_refresh(status.scroll_visible, animate=False, immediate=True)

    def _checking_text(self) -> str:
        """The status line while the selected provider's check runs."""
        from tldw_chatbook.Chat.provider_catalog import provider_display_name

        key = getattr(self, "selected_provider_key", "")
        return f"Checking {provider_display_name(key)}…" if key else "Checking…"

    @staticmethod
    def _setup_facts(provider_key: str):
        from tldw_chatbook.Chat.provider_setup_facts import provider_setup_facts

        return provider_setup_facts(provider_key)

    def _typed_base_url(self) -> str:
        """The per-account base URL typed for Azure/Databricks/Cloudflare, or ""."""
        key = getattr(self, "selected_provider_key", "")
        if not key or not self._setup_facts(key).base_url_required:
            return ""
        try:
            return self.query_one("#setup-provider-endpoint", Input).value.strip()
        except NoMatches:
            return ""

    def _configure_endpoint_field(self, provider_key: str) -> None:
        """Label, example and buttons of the endpoint panel, from the setup facts.

        TASK-34100.6 (review provider-05, provider-06, provider-21 (c)): a
        per-account URL is labelled per provider; 'Find local servers' shows
        only for local rows and 'Test connection' only where it can enable.
        """
        from tldw_chatbook.Chat.provider_setup_facts import endpoint_example

        facts = self._setup_facts(provider_key)
        if facts.base_url_label:
            label = facts.base_url_label
        else:
            label = "Server address" if facts.locality == "local" else "Endpoint"
        example = endpoint_example(facts)
        try:
            self.query_one("#setup-provider-endpoint-label", Label).update(label)
            self.query_one("#setup-provider-endpoint", Input).placeholder = (
                f"e.g. {example}" if example else "Base URL"
            )
            self.query_one("#setup-provider-detect", Button).display = (
                facts.locality == "local"
            )
            testable = provider_key in getattr(
                self, "_OPENAI_COMPATIBLE_PROBE_PROVIDERS", ()
            )
            self.query_one("#setup-provider-test", Button).display = testable
            self.query_one(".setup-provider-connection-actions").display = (
                testable or facts.locality == "local"
            )  # no empty button row under a cloud URL field (G6-VR2-11)
        except NoMatches:
            return

    def _refresh_endpoint_resolution(self, readiness: Any = None) -> None:
        """Chat URL preview and the endpoint's own status line."""
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            resolve_provider_endpoint,
        )

        provider_key = getattr(self, "selected_provider_key", "")
        effective = self.query_one("#setup-provider-effective-chat", Static)
        if not provider_key:
            effective.update("")
            self.query_one("#setup-provider-endpoint-status", Static).update("")
            return
        endpoint = self.query_one("#setup-provider-endpoint", Input).value
        resolution = resolve_provider_endpoint(provider_key, endpoint)
        effective.update(
            f"Chat URL: {resolution.chat_display}" if resolution.chat_url else ""
        )
        self._refresh_endpoint_status(readiness, resolution=resolution)

    def _refresh_endpoint_status(self, readiness: Any = None, *, resolution: Any = None) -> None:
        """One line under the endpoint: its error or warning, and why Test is off.

        Nothing the user has not typed is an error (review cross-cutting-07):
        an empty field reads neutral. A disabled 'Test connection' says what
        enables it (review provider-15).
        """
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            resolve_provider_endpoint,
        )

        provider_key = getattr(self, "selected_provider_key", "")
        try:
            status = self.query_one("#setup-provider-endpoint-status", Static)
            endpoint = self.query_one("#setup-provider-endpoint", Input).value
            test = self.query_one("#setup-provider-test", Button)
        except NoMatches:
            return
        if not provider_key:
            status.update("")
            return
        if resolution is None:
            resolution = resolve_provider_endpoint(provider_key, endpoint)
        parts: list[str] = []
        native_refusal = first_run_endpoint_refusal(provider_key, resolution.chat_url)
        invalid = bool(endpoint.strip()) and (
            resolution.chat_url is None or bool(native_refusal)
        )
        if invalid:
            parts.append("✗ " + (
                native_refusal or (resolution.errors[0] if resolution.errors else "Invalid endpoint.")
            ))
            if test.display:
                parts.append("Fix it to enable Test connection.")
        elif endpoint.strip():
            parts.extend(resolution.warnings)
            if test.display and test.disabled and readiness is not None and not readiness.ready:
                parts.append("Add the API key below to enable Test connection.")
        elif readiness is not None and (what := missing_url_reason(readiness)):
            # Review round 1, live F8: a missing per-account URL is this
            # field's own line (readiness names it once the key is found).
            from tldw_chatbook.Chat.provider_catalog import provider_display_name

            parts.append(f"Add your {provider_display_name(provider_key)} {what} here to continue.")
        status.update(" ".join(parts))
        status.set_class(invalid, "-error")  # provider-16: the field's own error

    def _invalid_draft_refusal(self) -> str:
        """Commit's refusal when the draft won't resolve: name the field."""
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            resolve_provider_endpoint,
        )

        try:
            connection = self.query_one("#setup-provider-connection")
            endpoint = self.query_one("#setup-provider-endpoint", Input).value
        except NoMatches:
            return "The provider settings are invalid. Pick another provider."
        key = getattr(self, "selected_provider_key", "")
        resolution = resolve_provider_endpoint(key, endpoint)
        if connection.display and (
            refusal := first_run_endpoint_refusal(key, resolution.chat_url)
        ):
            return refusal
        if connection.display and resolution.errors:
            return (
                "The endpoint isn't a valid address — fix it above or pick another "
                "provider."
            )
        return "The provider settings are invalid — check them above or pick another provider."

    def _endpoint_edited(self) -> None:
        """A user edit of the endpoint: drop what described the old value.

        The 'Using the … server found at' line, and a pinned refusal that
        named this field: on base it stayed through edits and a successful
        Test, beside '✓ Connected' (review round 1, live F2).
        """
        banner = self.query_one("#setup-provider-detected", Static)
        banner.update("")
        banner.add_class("hidden")
        getattr(self.wizard, "_clear_pinned_step_error", lambda: None)()

    def _drop_checking_status(self) -> None:
        """Clear a 'Checking …' line whose check was just cancelled.

        TASK-34100.6 folded defect (a): Model's hide cancels the check it
        borrowed from Provider; the line kept claiming a check was running.
        """
        if not self.is_attached:
            return
        try:
            status = self.query_one("#setup-provider-probe-status", Static)
        except NoMatches:
            return
        if str(status.render()).startswith("Checking"):
            self._set_status("")

    def _start_discovery(self, *, user_requested: bool = False, quiet: bool = False) -> None:
        self._local_discovery_generation += 1
        generation = self._local_discovery_generation
        provider_key = self.selected_provider_key
        self._local_discovery_provider_key = provider_key
        self._local_discovery_state = "in_progress"
        if user_requested:
            self._render_detection_results((), status="Searching local endpoints…")
        run_wizard_worker(
            self,
            partial(
                self._discover_servers,
                generation,
                provider_key,
                user_requested=user_requested,
                quiet=quiet,
            ),
            exclusive=True,
            group="setup-provider-local-discovery",
        )

    async def _discover_servers(
        self,
        generation: int,
        provider_key: str,
        *,
        user_requested: bool = False,
        quiet: bool = False,
    ) -> None:
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
        if generation != self._local_discovery_generation:
            return
        self._local_discovery_state = state
        listed = self.query("#setup-provider-detection-results")  # gone in teardown
        if not self.is_attached or not listed:
            return
        # TASK-34100.6 (review provider-02): every scan updates the list's
        # sticky 'Ready on this machine' rows, whatever is selected; a quiet
        # re-check only adds (G6-BV3-04: a slow answer never drops a row).
        self._found_ever_scanned = True
        if quiet:
            servers = (*getattr(self, "_found_servers", ()), *servers)
        self._show_found_servers(servers)
        self._render_no_server_state(servers)
        if (
            not user_requested  # review round 1, F7: quiet scans own only the rows
            or provider_key != self._local_discovery_provider_key
            or provider_key != self.selected_provider_key
            or not provider_key
            or not self.is_active
        ):
            return
        self._detected_servers = servers
        if servers:
            self._render_detection_results(servers)
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
        """Say which found server the selection uses (review provider-21 (b))."""
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            resolve_provider_endpoint,
        )
        from tldw_chatbook.UI.Wizards.first_run_provider_list import (
            server_address,
            server_engine,
        )

        try:
            provider_key = self._canonical_provider_key(server.provider_key)
        except (TypeError, ValueError):
            provider_key = ""
        resolution = resolve_provider_endpoint(provider_key, server.base_url)
        banner = self.query_one("#setup-provider-detected", Static)
        if resolution.persisted_endpoint is None:
            for attribute in ("detected_server", "detected_base_url"):
                if hasattr(self, attribute):
                    delattr(self, attribute)
            self._detected_endpoint_provider_key = ""
            banner.update("")
            banner.add_class("hidden")
            return
        self.detected_server = server
        self.detected_base_url = server.base_url
        count = len(tuple(getattr(server, "model_ids", ()) or ()))
        banner.update(
            f"✓ Using the {server_engine(server)} server found at "
            f"{server_address(server)} — {count} model{'' if count == 1 else 's'}."
        )
        banner.remove_class("hidden")
        # Review v3r1, G6-BV3-01: the line lands above the list being browsed.
        getattr(self, "keep_browsed_row_in_view", lambda: False)()
        self._highlight_discovered_server(server)

    def _use_found_server(self, server: Any) -> None:
        """Adopt a detected server: its row, its port, its models (provider-02)."""
        from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            resolve_provider_endpoint,
        )

        if type(server) is not DiscoveredLocalServer:
            return
        try:
            provider_key = self._canonical_provider_key(server.provider_key)
        except (TypeError, ValueError):
            provider_key = ""
        endpoint_input = self.query_one("#setup-provider-endpoint", Input)
        if (
            provider_key == self.selected_provider_key
            and getattr(self, "detected_server", None) is server
            and endpoint_input.value == server.base_url
        ):
            return  # a no-op re-pick changes nothing (review provider-26)
        if resolve_provider_endpoint(provider_key, server.base_url).persisted_endpoint is None:
            self.query_one("#setup-provider-endpoint-status", Static).update(
                "The selected endpoint is invalid."
            )
            return
        detected_servers = self._detected_servers
        if provider_key != self.selected_provider_key:
            self.select_provider(provider_key)
        self._detected_servers = detected_servers
        if detected_servers:
            self._render_detection_results(detected_servers)
        self._apply_discovered_server(server)
        self._detected_endpoint_provider_key = provider_key
        self._updating_connection_controls = True
        try:
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

    @on(Button.Pressed, "#setup-provider-detect")
    def _on_detect_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._cancel_worker_groups("setup-provider-local-discovery")
        self._start_discovery(user_requested=True)

    @on(OptionList.OptionSelected, "#setup-provider-detection-results")
    def _on_detected_endpoint_selected(self, event: OptionList.OptionSelected) -> None:
        self._use_found_server(getattr(event.option, "server", None))

    def _retry_control(self) -> str:
        """The control on screen that checks again, or "" (review round 2, R2-04).

        'Check again' shows only for local rows (``_sync_check_again``);
        a keyed cloud row's control is 'Check key'. A line never names a
        control the user cannot see.
        """
        key = getattr(self, "selected_provider_key", "")
        if key and self._setup_facts(key).locality == "local":
            return "Check again"
        try:
            check_key = self.query_one("#setup-provider-check-key", Button)
        except NoMatches:
            return ""
        return "Check key" if check_key.display else ""

    def _transport_failure_line(self, display: str) -> str:
        """The selected engine's own reason a check failed (provider-14), or ""."""
        from tldw_chatbook.UI.Wizards.first_run_model_discovery import (
            answered_http_status,
            http_error_hint,
        )

        outcomes = getattr(self, "_selected_provider_outcomes", {})
        outcome = outcomes.get(getattr(self, "_selected_discovery_key", None))
        error = getattr(outcome, "error", None)
        category = getattr(error, "category", None)
        key = getattr(self, "selected_provider_key", "")
        control = self._retry_control()
        code = answered_http_status(error)
        if code and code not in {404, 405}:  # it answered (review round 2, R2-02)
            what, todo = http_error_hint(code)
            meaning = f" — {what}" if what else ""
            then = f"; {todo}, then press {control}." if todo and control else (
                f"; press {control} to try again." if control else "."
            )
            return f"{display} answered HTTP {code}{meaning}{then}"
        try:
            from urllib.parse import urlsplit

            parts = urlsplit(self.query_one("#setup-provider-endpoint", Input).value.strip())
            endpoint = f"{parts.scheme}://{parts.netloc}" if parts.netloc else ""
        except (NoMatches, ValueError):
            endpoint = ""
        at = f" at {endpoint}" if endpoint else ""
        if tls := tls_failure_line(display, category, control):
            return tls
        if not key or self._setup_facts(key).locality != "local":
            return ""
        if category in {"connection_refused", "connection_error"}:
            return f"Nothing is answering{at} — start {display}, then Check again."
        if category == "timeout":
            return f"{display} didn't answer{at} in time — check it's running, then Check again."
        return ""

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
