"""Checks on the first-run Provider step (TASK-34100.6 slices E and F).

Review findings provider-06 (no visible way to check a cloud key; a Test
button that never enabled), provider-03 (a rejected key advanced with ✓),
provider-15 (endpoint errors hid their HTTP status and the '/api' slip),
provider-21 (a hard-coded 'your chosen model was not in its list') and
cross-cutting-07 (stale 'test again' lines with no control on screen).

'Check key' does not make a request of its own: it runs the step's
selected-provider discovery -- the same non-billable ``discover_models``
listing Settings' key check sends (TASK-33005.4) -- and reads the settled
outcome through Settings' own ``key_check_probe_result``, so the wizard and
Settings give one verdict and Next can reuse the listing.

A provider whose model list answers without a valid key (OpenRouter,
Hugging Face, Novita) is never reported as 'Key works' from that listing:
only ``ModelDiscoveryResult.key_verified`` -- set by TASK-34100.2's
authenticated checks (OpenRouter /api/v1/key, Hugging Face whoami-v2) --
can confirm it. Until that lands, those providers say plainly that the key
is checked on the first message.

The localhost connection probe ('Test connection') moved here whole from
``first_run_provider_step.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
from functools import partial
from typing import Any
from urllib.parse import urlsplit

from loguru import logger
from textual import on
from textual.css.query import NoMatches
from textual.widgets import Button, Input

from tldw_chatbook.UI.Wizards import first_run_model_discovery as model_discovery
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.first_run_provider_detection import tls_failure_line
from tldw_chatbook.UI.Wizards.first_run_provider_list import (  # noqa: F401 - re-exported
    ProviderListStep,
    grouped_sections,
)
from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker

CLOUD_PROBE_TIMEOUT_SECONDS = 8.0
CHECK_KEY_LABEL = "Check key"
_HTTP_STATUS = re.compile(r"HTTP (\d{3})")


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
    from tldw_chatbook.Utils.tls_trust import build_httpx_async_client

    del credential_source
    client: httpx.AsyncClient | None = None
    try:
        if credential_value:
            try:  # the app's TLS trust policy, like the keyless probe (34100.6)
                client = build_httpx_async_client(
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


def _models(count: int) -> str:
    return f"{count} model" if count == 1 else f"{count} models"


def key_check_line(
    display: str, result: object, *, public_listing: bool, control: str = CHECK_KEY_LABEL
) -> tuple[str, bool]:
    """The one line a key check ends in, and whether the key was rejected.

    Args:
        display: The provider's display name.
        result: The settled ``ModelDiscoveryResult`` (or None).
        public_listing: The provider lists models without a valid key.
        control: The on-screen control a certificate fix is retried with.

    Returns:
        (line, rejected): '✓ Key works — …', '✗ … rejected this key.' or a
        plain '!' line saying the key is checked on the first message. Only
        a listing route that answered 404/405 is 'without a paid request'.
    """
    from tldw_chatbook.UI.Screens.settings_endpoint_probe import key_check_probe_result

    later = "it's checked when you send your first message."
    paid = f"! {display} can't be checked without a paid request — {later}"
    probe = key_check_probe_result(result, key_sent=True) if result is not None else None
    if probe is None:  # no request was sent: claim nothing about the provider
        # Review round 2, G6-VR2-07: Azure's check sent nothing (no listing
        # route the wizard can use) and still said "without a paid request".
        return f"! {display}'s key isn't checked here — {later}", False
    if probe.endpoint == "unreachable":
        if probe.category == "unauthorized":
            return f"✗ {display} rejected this key.", True
        if tls := tls_failure_line(display, probe.category, control):  # R2-03
            return f"! {tls}", False
        return f"! Couldn't check the key — {display} didn't answer; {later}", False
    if probe.key_accepted:
        if public_listing and not getattr(result, "key_verified", False):
            return (
                f"! {display}'s model list answers without a key, so it can't "
                f"confirm this one here — {later}",
                False,
            )
        # The listing's own length: the evidence keeps at most 100 ids.
        listed = len(getattr(result, "models", ()) or ()) or len(probe.model_ids)
        return f"✓ Key works — {display} returned {_models(listed)}.", False
    # The provider answered without a usable list (review round 1, F6): say
    # what it answered; only a listing route that isn't there is 'paid'.
    error = getattr(result, "error", None)
    code = _HTTP_STATUS.search(str(getattr(error, "message", "") or ""))
    if getattr(error, "category", None) == "forbidden":
        return f"! {display} answered, but won't list models for this key (HTTP 403) — {later}", False
    if getattr(result, "status", None) == "success":
        what = "an empty model list"
    elif getattr(error, "kind", None) == "unsupported_endpoint" or (
        code is not None and code.group(1) in {"404", "405"}
    ):
        return paid, False  # live F7: Cloudflare's listing route answers 405
    else:
        what = f"an error (HTTP {code.group(1)})" if code else "a model list it couldn't read"
    return f"! {display} answered with {what}, so the key couldn't be checked — {later}", False


def http_status_line(code: int) -> str:
    """'✗ HTTP 500 — the server hit an error; check its logs, then Test again.'"""
    from tldw_chatbook.UI.Wizards.first_run_model_discovery import http_error_hint

    what, todo = http_error_hint(code)  # 503: still loading (review round 2)
    if what:
        return f"✗ HTTP {code} — {what}; {todo}, then Test again."
    return f"✗ HTTP {code} — the server refused the request; check the endpoint."


class ProviderCheckStep(ProviderListStep):
    """Check key, Test connection and the status line they report to."""

    _key_check_requested = False
    _unsaved_key_owner = ""
    _rejected_discovery_key: Any = None
    #: (provider, connection identity, source, digest) of every credential a
    #: provider rejected. Unlike the exact discovery key these survive a
    #: revision bump: choosing 'Store a different key' or Replace with the
    #: field still empty leaves the same rejected key in use (found live,
    #: g6-r2fix/55-56). The digest is an HMAC under the step's private key.
    _rejected_credentials: frozenset = frozenset()
    _last_probe_status_code: int | None = None

    def _sync_check_key(self, readiness: Any) -> None:
        """'Check key' beside the key, for keyed providers only (provider-06)."""
        try:
            button = self.query_one("#setup-provider-check-key", Button)
        except NoMatches:
            return
        button.display = bool(getattr(readiness, "requires_api_key", False))
        self._sync_key_companions()  # review round 2, G6-VR2-11 (1)

    @on(Button.Pressed, "#setup-provider-check-key")
    def _on_check_key_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._launch_key_check()

    def _launch_key_check(self) -> None:
        """List models with the draft key: the check Settings uses."""
        from tldw_chatbook.Chat.provider_catalog import provider_display_name

        display = provider_display_name(self.selected_provider_key)
        self._reveal_status = True  # review round 2, G6-VR2-06
        if not self._current_provider_readiness().ready:
            self._set_status(f"Paste a {display} API key above, then Check key.")
            return
        self._key_check_requested = True
        self._begin_selected_provider_discovery(self._effective_provider_draft())

    def _render_discovery_outcome(
        self,
        display: str,
        models: tuple[str, ...],
        *,
        failed: bool,
        attempted: bool,
        outcome: object = None,
        draft: object = None,
    ) -> None:
        """The settled selected-provider check, as one status line.

        TASK-33008: a failed listing is told as the Model step tells it
        (``failed_listing``), and every settled listing moves the key mark
        the Authentication title reads (``note_probe_failure`` /
        ``note_probe_reached``).
        """
        key = self._selected_discovery_key
        source = getattr(key, "credential_source", "none")
        requested, self._key_check_requested = self._key_check_requested, False
        failed_line = ""
        if failed and not models:
            failed_line, failure = model_discovery.failed_listing(self, display, outcome, draft)
            model_discovery.note_probe_failure(self, failure, key)
        elif models or (attempted and outcome is not None):  # the listing answered
            model_discovery.note_probe_reached(self, key)
        outcome = self._selected_provider_outcomes.get(key, outcome)
        keyed = self._current_provider_readiness().requires_api_key
        if keyed and source != "none" and outcome is not None:
            facts = self._setup_facts(self.selected_provider_key)
            line, rejected = key_check_line(
                display,
                outcome,
                public_listing=not facts.listing_requires_auth,
                control=self._retry_control(),
            )
            self._rejected_discovery_key = key if rejected else None
            if key == self._model_discovery_key(self._effective_provider_draft()):
                mark = self._credential_mark(key)
                if mark is not None:
                    self._rejected_credentials = (
                        self._rejected_credentials | {mark}
                        if rejected
                        else self._rejected_credentials - {mark}
                    )
            if rejected:  # review provider-03: the error sits under the field
                line, note, _refusal = self._rejected_key_copy()
                self._set_key_note(note, error=True)
            # Drop 'Key staged' / 'Press Check key' once a check has run
            # (cross-cutting-07; review round 1, live F6).
            self._refresh_auth_readiness()
            if requested or rejected or line.startswith("✓"):
                self._set_status(line)
                return
        if models:
            self._set_status(f"✓ Found {_models(len(models))} for {display}.")
        elif failed:
            self._set_status(failed_line)
        elif attempted:
            self._set_status(f"! Checked {display}; it reported no models.")
        else:
            self._set_status("")

    def _rejected_key_copy(self) -> tuple[str, str, str]:
        """(status line, field note, Next's refusal) for the rejected key.

        Review round 1, F1: the copy names fixes that work for where the key
        came from. An exported or saved key has no field to clear, so Ctrl+U
        is never offered for it; the source choice, Replace and the skip are.
        """
        from tldw_chatbook.Chat.provider_catalog import provider_display_name
        from tldw_chatbook.UI.Wizards.first_run_provider_skip import SKIP_LABEL

        display = provider_display_name(self.selected_provider_key)
        source = getattr(self._rejected_discovery_key, "credential_source", None) or (
            self._credential_at_request_boundary()[0]  # the one still rejected
        )
        # The source choice / Replace already used: the field is the fix.
        chosen = source in {"environment", "stored"} and bool(self._key_field().display)
        if source == "environment":
            env_var = self._current_provider_readiness().env_var or "your environment"
            line = f"✗ {display} rejected the key in {env_var}."
            fix = (
                "paste a working key above"
                if chosen
                else "choose 'Store a different key for this app' and paste a working key"
            )
            return line, line, (
                f"{display} rejected the key in {env_var} — {fix}, pick another "
                f"provider, or choose {SKIP_LABEL}."
            )
        if source == "stored":
            line = f"✗ {display} rejected the saved key."
            fix = "paste a working key above" if chosen else "press Replace and paste a working key"
            return line, line, (
                f"{display} rejected the saved key — {fix}, pick another provider, "
                f"or choose {SKIP_LABEL}."
            )
        line = f"✗ {display} rejected this key."
        note = f"{line} Check it was copied completely, or paste a new one."
        # Review round 2, R2-05: with an exported or saved key behind the
        # field, clearing it makes Next use that key -- never a skip.
        base = self._credential_at_request_boundary()[2]
        if base.ready and base.api_key is not None:
            other = (
                f"choose 'Use {base.env_var} from your environment'"
                if str(base.api_key_source).startswith("env:")
                else "press Keep current"
            )
            if self._base_key_rejected(base):  # never offer a key turned down
                other = "pick another provider"
            else:
                other = f"{other}, pick another provider"
            return line, note, (
                f"{display} rejected this key — paste a working key above, {other}, "
                f"or choose {SKIP_LABEL}."
            )
        return line, note, (
            f"{display} rejected this key — paste a working key above, or clear it "
            "(Ctrl+U) and press Next to skip."
        )

    def key_check_refusal(self) -> str:
        """Commit's refusal when this exact key was rejected (review provider-03).

        A 401 is definitive, so Next stays here with the error under the key
        and no 'continue' wording. A 403 is not: the owner ruled on
        2026-10-02 that it only means the key may not list models
        (``key_check_probe_result``), so it never blocks.
        """
        current = self._model_discovery_key(self._effective_provider_draft())
        if current is None:
            return ""
        if current != self._rejected_discovery_key and not self._still_rejected(current):
            return ""
        return self._rejected_key_copy()[2]

    def _digest(self, source: str, value: str | None) -> bytes | None:
        """An HMAC of one credential under the step's private key, or None."""
        secret = getattr(self, "_credential_observation_key", None)
        if not secret or not value:
            return None
        return hmac.new(secret, f"{source}\0{value}".encode("utf-8"), hashlib.sha256).digest()

    def _credential_mark(self, key: Any, source: str = "", value: str | None = None) -> Any:
        """(provider, identity, source, digest) for ``key``'s credential, or None.

        Without ``source`` it marks the credential a request would carry now.
        """
        if not source:
            source, value, _base = self._credential_at_request_boundary()
        digest = self._digest(source, value)
        if digest is None:
            return None
        return (key.provider_key, key.connection_identity, source, digest)

    def _still_rejected(self, current: Any) -> bool:
        """Whether ``current`` carries a credential the provider rejected."""
        return self._credential_mark(current) in self._rejected_credentials

    def _base_key_rejected(self, base: Any) -> bool:
        """Whether the exported or saved key behind the field was rejected."""
        current = self._model_discovery_key(self._effective_provider_draft())
        if current is None:
            return False
        source = "environment" if str(base.api_key_source).startswith("env:") else "stored"
        return self._credential_mark(current, source, base.api_key) in self._rejected_credentials

    def refuse_rejected_key(self) -> str:
        """Commit's refusal, with the error kept under the key field."""
        refusal = self.key_check_refusal()
        if refusal:
            self._set_key_note(self._rejected_key_copy()[1], error=True)
            if (field := self._rejected_key_field()) is not None:
                self.call_after_refresh(field.focus)  # the Ctrl+U it names works
        return refusal

    def rejected_key_outcome(self, skip: bool) -> tuple[bool, str] | None:
        """Commit's answer while this exact key stands rejected, else None.

        A one-shot skip still skips (review round 1, F1: an exported key the
        provider rejects left no way forward); anything else is refused.
        """
        if not self.key_check_refusal():
            return None
        if skip:
            from tldw_chatbook.UI.Wizards.first_run_provider_skip import (
                skip_provider_step,
            )

            return skip_provider_step(self)
        return False, self.refuse_rejected_key()

    def _key_checked(self) -> bool:
        """Whether a check of the key now on screen has already settled."""
        key = self._selected_discovery_key
        if key is None or key not in self._selected_provider_outcomes:
            return False
        return key == self._model_discovery_key(self._effective_provider_draft())

    def _env_key_present(self, env_var: str | None) -> bool:
        from tldw_chatbook.config import is_valid_provider_api_key

        return bool(env_var) and is_valid_provider_api_key(
            self._environment().get(env_var)
        )

    def key_status_text(self, readiness: Any, test_available: bool) -> str:
        """The line under the key while the provider is ready.

        Never repeats an instruction already carried out (review round 1,
        live F6), never claims 'testing is unavailable' beside Check key, and
        says when a saved key outranks an exported one (F5).
        """
        source, _value, _base = self._credential_at_request_boundary()
        keyed = bool(readiness.requires_api_key)
        check = " Press Check key to try it." if keyed and not self._key_checked() else ""
        if test_available:
            unavailable = check
        elif self.selected_provider_key in self._OPENAI_COMPATIBLE_PROBE_PROVIDERS:
            unavailable = " Enter a valid endpoint to enable connection testing."
        else:  # provider-06: a keyed row's check is Check key, never 'unavailable'
            unavailable = check if keyed else " Connection testing is unavailable for this provider."
        env_name = readiness.env_var or self._setup_facts(self.selected_provider_key).env_var
        env_var = env_name or "the configured environment variable"
        if source == "stored":
            outranks = f" It's used instead of {env_var}." if self._env_key_present(env_name) else ""
            return f"An API key is already configured for this provider.{outranks}{unavailable}"
        if source == "environment":
            if self._key_field().display:  # 'Store a different key' was chosen
                return f"Paste the key to store above — until you do, {env_var} is used."
            return (
                f"Found {env_var} in your environment — nothing is stored, so it "
                f"must be set in the shell that starts chatbook.{unavailable}"
            )
        if source == "draft" and keyed:  # Check key's line is the verdict (F6)
            return "" if self.key_check_refusal() else f"Key staged.{unavailable}"
        if source == "draft":  # TASK-33008: what the last check of the key found
            return model_discovery.staged_key_line(self) + unavailable
        return unavailable.strip()

    def _rejected_key_field(self) -> Input | None:
        """The key field while its key stands rejected, if it can take focus."""
        if not self.key_check_refusal():
            return None
        try:
            field = self._key_field()
        except NoMatches:
            return None
        return field if field.display else None

    def _key_field_after_skip(self) -> Input | None:
        """After a skip, the key field of a provider that still needs one."""
        if not getattr(self, "_skipped", False) or not self.selected_provider_key:
            return None
        try:
            readiness = self._current_provider_readiness()
            field = self._key_field()
        except Exception:  # noqa: BLE001 - focus is a convenience, never fatal.
            return None
        needs_key = readiness.requires_api_key and not readiness.ready
        return field if needs_key and field.display else None

    def entry_focus(self) -> Any:
        """Focus on entry: the key after a rejected key or a skip, else the list.

        Live 2026-10-04: after Back from a rejected key the list had focus, so
        the 'clear it (Ctrl+U)' the refusal names did nothing. Review round
        2, G6-VR2-03: after Back from a skip, a paste was dropped the same way.
        """
        if (field := self._rejected_key_field() or self._key_field_after_skip()) is not None:
            return field
        try:
            return self.query_one("#setup-provider-choice")
        except NoMatches:
            return None

    def _note_unsaved_key(self, left: str, picked: str) -> None:
        """Say a typed key stays unsaved unless its provider is chosen (coverage-08).

        Drafts are kept per provider while setup is open, but only the chosen
        provider is saved; moving away from a typed key used to drop it
        silently.
        """
        from tldw_chatbook.Chat.provider_catalog import provider_display_name

        typed = getattr(self._provider_drafts.get(left), "api_key", "") if left else ""
        try:
            note = self.query_one("#setup-provider-unsaved-note")
        except NoMatches:
            return
        if typed.strip() and left != picked:
            display = provider_display_name(left)
            note.update(f"! Your {display} key won't be saved unless you choose {display}.")
            self._unsaved_key_owner = left
        elif picked == self._unsaved_key_owner:
            note.update("")
            self._unsaved_key_owner = ""  # back to it: the key is in use again
        note.set_class(not self._unsaved_key_owner, "hidden")

    def _stale_check_line(self) -> str:
        """After an edit: name the check again only if its control is on screen."""
        try:
            testable = self.query_one("#setup-provider-test", Button)
            connection = self.query_one("#setup-provider-connection")
        except NoMatches:
            return ""
        if testable.display and connection.display:
            return "Provider settings changed since test; test again."
        return ""

    def _discovery_failure_status(self, display: str, failure: str = "") -> str:
        """Failure copy that never promises what Next will refuse.

        task-31820 / TASK-34100.6 (review cross-cutting-07): nothing is an
        error before the user acts -- a provider still waiting for its key
        says so under the key field, so this line stays empty. A ready
        provider that failed names its own reason (the address that didn't
        answer, an untrusted certificate) before the generic warning.
        TASK-33008: ``failure`` (a ``PROVIDER_PROBE_*``) names a refused key
        or a missing one, as the Model step does.
        """
        try:
            readiness = self._current_provider_readiness()
            ready = bool(readiness.ready)
        except Exception:
            return f"! Couldn't discover models for {display}."
        if not ready:
            return ""
        reason = self._transport_failure_line(display)  # provider-14
        if reason:
            return f"! {reason}"
        line = wizard_state.probe_failure_copy(failure).provider_line.format(display=display)
        key_failures = {wizard_state.PROVIDER_PROBE_AUTH, wizard_state.PROVIDER_PROBE_KEY_REQUIRED}
        return f"{'✗' if failure in key_failures else '!'} {line}"

    def return_notice(self, failure: str) -> str:
        """The notice pinned on Back after a failed check (TASK-21143 P-5).

        TASK-33008 offers to continue, as the step's line does; a key this
        step refuses (review provider-03: a definitive 401 on a keyed
        provider) is never offered that way, since Next will not take it.
        """
        if failure == wizard_state.PROVIDER_PROBE_AUTH and self.key_check_refusal():
            return wizard_state.PROVIDER_REFUSED_KEY_NOTICE
        return wizard_state.probe_failure_return_notice(failure)

    @on(Input.Submitted, "#setup-provider-api-key")
    def _on_key_submitted(self, event: Input.Submitted) -> None:
        """Enter in a filled key field checks it (live, never blocking)."""
        if not event.value.strip():
            return
        if self._current_provider_readiness().requires_api_key:
            self._launch_key_check()
        else:
            self._launch_probe()

    @on(Button.Pressed, "#setup-provider-test")
    def _on_test_pressed(self, event: Button.Pressed) -> None:
        """TASK-1506: same probe as Enter-in-field, behind a visible control."""
        event.stop()
        self._launch_probe()

    def _launch_probe(self, *, api_key: str | None = None) -> None:
        del api_key
        self._reveal_status = True  # review round 2, G6-VR2-06
        self._sync_live_credential_revision()
        generation = self._obsolete_provider_generation(
            "setup-provider-discovery",
            "setup-provider-probe",
        )
        provider_key = self.selected_provider_key
        readiness = self._current_provider_readiness()
        if not readiness.ready:
            self._set_status(readiness.recovery or "An API key is required before testing.")
            return
        target = self._probe_target()
        identity = self._provider_current_draft_identity()
        if not target or identity is None:
            self._set_status("Enter a valid endpoint above before testing.")
            return
        credential_source, credential_value, _ = self._credential_at_request_boundary()
        token = self._provider_test_evidence.begin(identity)
        self._active_probe_token = token
        self._set_status("Testing…")
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
                    self._set_status("")
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
        self._listed_after_test = False
        code = _HTTP_STATUS.search(str(getattr(outcome, "summary", "")))
        self._last_probe_status_code = int(code.group(1)) if code else None
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

    def _extra_path_hint(self) -> str:
        """'remove /api?' when the endpoint carries a path past its host (15)."""
        try:
            endpoint = self.query_one("#setup-provider-endpoint", Input).value.strip()
            path = urlsplit(endpoint if "://" in endpoint else f"http://{endpoint}").path
        except (NoMatches, ValueError):
            return ""
        segments = [s for s in path.split("/") if s and s not in {"v1", "chat", "completions", "models"}]
        if not segments:
            return ""
        extra = "/" + "/".join(segments)
        return (
            f"! The server answered, but there's no model list under {extra} — "
            f"remove {extra} from the endpoint?"
        )

    def _render_provider_evidence(self, identity: object) -> None:
        from tldw_chatbook.Chat.provider_catalog import provider_display_name
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
        code = self._last_probe_status_code
        verdict = provider_readiness_verdict(
            ProviderReadinessSnapshot(
                configuration="configured",
                endpoint=evidence.endpoint,
                model="unconfirmed",
                category=evidence.category,
            )
        )
        # TASK-33008: a 401 and an unavailable listing are told as the
        # listing tells them (a keyless 401 asks for a key -- review round 1,
        # live F13 -- and the Authentication title says so), and every
        # settled Test moves the key mark.
        detail = model_discovery.tested_evidence_detail(self, identity, evidence, verdict.detail)
        glyph = {"reachable": "✓ ", "unreachable": "✗ "}.get(evidence.endpoint, "")
        line = detail if detail[:1] in {"✗", "!"} else glyph + detail
        if evidence.endpoint == "reachable":  # review provider-21 (a)
            display = provider_display_name(self.selected_provider_key)
            line = (
                f"✓ Connected — {display} returned "
                f"{_models(len(evidence.model_ids))}. Pick one on the next step."
            )
        elif evidence.category == "http_status":
            if evidence.endpoint == "model_listing_unavailable":
                # Only a 404 on the models route lands here (settings_endpoint_probe).
                line = self._extra_path_hint() or line
            elif code is not None:
                line = http_status_line(code)  # review provider-15
        self._set_status(line)
        self._sync_check_again()  # G6-VR2-09: what the Test found decides the how-to

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
        self._set_status(
            "✓ Model listing reached."
            if reachable
            else "✗ The model listing endpoint could not be reached."
        )
