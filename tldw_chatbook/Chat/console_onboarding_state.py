"""Pure first-run setup-card state contracts for the native Console."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any
from urllib.parse import urlparse

from rich.cells import cell_len, set_cell_size

from tldw_chatbook.Chat.console_provider_endpoints import safe_endpoint_display
from tldw_chatbook.Chat.console_session_settings import (
    SEND_OUTCOME_BLOCKERS,
    ConsoleSettingsReadiness,
    readiness_words,
)
from tldw_chatbook.Chat.provider_catalog import provider_display_name

CONSOLE_SETUP_CARD_TITLE = "Get started"
#: TASK-2154.8 (FR-10): one plain-language sentence under the card title so a
#: first-time user knows what a "provider" even is before the steps ask them
#: to connect one. Kept provider-agnostic (no settings jargon). TASK-33008.4:
#: its second line says where cloud setup goes, and Enter is the only key it
#: names, matching the blocked footer's Enter hint (ADR-299 rule 5).
#: TASK-33008.2: the line promises no return for a custom endpoint, because
#: Settings offers none there (no return can name a ``custom-ep:`` id).
CONSOLE_SETUP_CARD_SUBTITLE = (
    "A provider is the AI service that answers your messages — "
    "for example OpenAI, Anthropic, or a server running on this computer.\n"
    "Cloud setup opens Settings and comes back here, except for custom "
    "endpoints. Press Enter to use the highlighted action."
)
CONSOLE_READY_EMPTY_COPY = "Ready — type a message to begin."
CONSOLE_QUIET_EMPTY_COPY = "No messages yet."
CONSOLE_SETUP_STEP_GLYPHS = {"done": "✓", "active": "●", "pending": "○"}
#: The card's content cells: .console-setup-modal-card (features/_console.tcss)
#: is 66 wide, padding 1 2, solid border. test_console_setup_card_fit.py
#: measures the step lines against that rule.
CONSOLE_SETUP_CARD_CONTENT_CELLS = 60
#: An active step's readiness word sits on its own line, this far in.
CONSOLE_SETUP_STEP_DETAIL_INDENT = "     "

# TASK-2154.8 (FR-10): endpoint variants name the thing in plain terms first
# ("server address") with the jargon glossed after it; "send-capable" becomes
# an ordinary sentence. CONSOLE_SETUP_CARD_SUBTITLE explains "provider" itself.
_STEP_ONE_LABELS = {
    "credential_missing": "Connect a provider (API key or local server)",
    "credential_rejected": "Reconnect the provider credential",
    "endpoint_invalid": "Save the provider's server address (endpoint)",
    "endpoint_not_saved": "Save the provider's server address (endpoint)",
    "endpoint_unreachable": "Reconnect the provider server",
    "provider_missing": "Choose a supported provider",
    "provider_unsupported": "Choose a provider that works in the Console",
    "provider_configuration_invalid": "Finish provider setup",
    "readiness_unknown": "Finish provider setup",
}
CONSOLE_SETUP_STEP_THREE_DETAIL = "Composer unlocks after setup"
#: TASK-34100.6: a local server answered, so 'no key' would point the wrong way.
CONSOLE_SETUP_NO_MODEL_CONNECTED = "No model connected"
# task-32140: the numbered steps above are all provider setup -- a
# local-first user who came for notes was told the only available action
# needed an API key. This secondary action needs no provider and routes
# straight to Library's New note view (same destination as the
# command-palette "new_note" quick action).
CONSOLE_SETUP_NOTES_ACTION_LABEL = "Write a note in Library"
CONSOLE_SETUP_NOTES_ACTION_TOOLTIP = (
    "Opens Library's New note view. No provider needed."
)
_TRUE_STRINGS = {"true", "yes", "1", "on"}


@dataclass(frozen=True)
class ConsoleSetupStep:
    """One numbered step in the Console first-run setup card."""

    state: str
    label: str
    detail: str = ""

    @property
    def glyph(self) -> str:
        return CONSOLE_SETUP_STEP_GLYPHS.get(self.state, "○")


@dataclass(frozen=True)
class ConsoleSetupCardState:
    """Display state for the Console empty-transcript onboarding surface."""

    mode: str
    steps: tuple[ConsoleSetupStep, ...] = ()
    body_copy: str = ""


@dataclass(frozen=True)
class ConsoleDetectedServerAction:
    """Secondary setup-card affordance for an auto-detected local server."""

    label: str
    tooltip: str
    provider_key: str
    base_url: str
    model_id: str | None = None


def _detected_server_host_display(base_url: str) -> str:
    """Return a compact credential-free ``host:port`` label for card copy.

    Args:
        base_url: Detected server base URL.

    Returns:
        The endpoint display with any http(s) scheme prefix removed.
    """
    display = safe_endpoint_display(base_url)
    for prefix in ("http://", "https://"):
        if display.startswith(prefix):
            return display[len(prefix) :]
    return display


def build_console_detected_server_action(
    server: Any,
    *,
    card_mode: str,
) -> ConsoleDetectedServerAction | None:
    """Build the detected-local-server card affordance, if one applies.

    Args:
        server: Discovery result with ``provider_key``, ``base_url``, and
            ``model_ids`` attributes (``DiscoveredLocalServer``), or ``None``.
        card_mode: Current ``ConsoleSetupCardState.mode``; the affordance only
            exists while the blocking ``card`` is showing.

    Returns:
        The action (label, tooltip, and the detected provider, endpoint and
        first listed model) or ``None`` when no detected server should be
        offered.
    """
    if server is None or card_mode != "card":
        return None
    provider_key = str(getattr(server, "provider_key", "") or "").strip()
    base_url = str(getattr(server, "base_url", "") or "").strip()
    if not provider_key or not base_url:
        return None
    try:
        hostname = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return None
    if hostname not in {"127.0.0.1", "localhost"}:
        # The card only ever offers loopback servers; anything else means the
        # discovery filter was bypassed and the offer is dropped, not shown.
        return None
    host_display = _detected_server_host_display(base_url)
    if not host_display or host_display == "invalid endpoint":
        return None
    display_name = provider_display_name(provider_key)
    model_ids = tuple(
        model_id
        for model_id in (getattr(server, "model_ids", ()) or ())
        if isinstance(model_id, str) and model_id.strip()
    )
    model_id = model_ids[0].strip() if model_ids else None
    return ConsoleDetectedServerAction(
        label=f"Use detected {display_name} ({host_display})",
        # TASK-33008.3: the action opens Switch model on this provider; it no
        # longer adopts the first listed model, so the tooltip names none.
        tooltip=(
            f"Opens Switch model on {display_name} at {host_display}. "
            "The model you pick becomes the default for new chats."
        ),
        provider_key=provider_key,
        base_url=base_url,
        model_id=model_id,
    )


def coerce_console_first_send_completed(raw: Any) -> bool:
    """Normalize the persisted first-send flag from arbitrary config input.

    Config values may arrive as a native bool, a legacy int/string form, or
    anything else if the config was hand-edited or corrupted. Anything
    unrecognized is treated as not-yet-completed rather than raising.

    Args:
        raw: The raw ``first_send_completed`` value read from config (or
            ``None`` if absent).

    Returns:
        ``True`` if ``raw`` unambiguously represents completion (``True``,
        a nonzero int, or a truthy string like ``"true"``/``"yes"``/``"1"``/
        ``"on"``); ``False`` for everything else, including ``None`` and
        unrecognized types.
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw != 0
    if isinstance(raw, str):
        return raw.strip().lower() in _TRUE_STRINGS
    return False


def _ready_line_copy(
    provider_label: str, model: str, readiness: ConsoleSettingsReadiness
) -> str:
    """The one arrival line: what setup connected, then the ready prompt.

    TASK-34100.5 AC#11 (E10): first-run arrival raised a stack of toasts;
    the empty transcript now names the provider and model instead. A model
    saved as a file path (llama.cpp) shows its file name. TASK-33005.7: the
    prompt leads with the qualified readiness word every surface shows
    ("Ready · not tested", "Ready · reachable HH:MM", ...).
    """
    ready = CONSOLE_READY_EMPTY_COPY
    if readiness.operability == "ready_to_send":
        ready = ready.replace("Ready", readiness_words(readiness), 1)
    provider = str(provider_label or "").strip()
    name = str(model or "").strip()
    if name.startswith(("/", "~")):
        name = PurePath(name).name
    if not provider or not name:
        return ready
    return f"Setup complete — {provider} · {name}. {ready}"


def env_keyed_provider_for_recovery(
    app_config: Mapping[str, object],
    current_provider: str,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """A provider whose key is exported, when the current one still needs one.

    TASK-34100.6 (review new-coverage-power-03): with only ANTHROPIC_API_KEY
    exported, 'Set up provider' opened Settings on the template OpenAI with
    'API key source: missing'. Settings now opens on the provider whose key
    is there. Only the variable's presence is read; nothing is written.

    Args:
        app_config: The loaded configuration.
        current_provider: The Console's provider, in any saved spelling.
        environ: The environment to read (the process environment by default).

    Returns:
        That provider's readiness key, or None when the current provider is
        ready, needs no key, or no other provider's key is exported.
    """
    import os

    from tldw_chatbook.Chat.console_provider_support import (
        resolve_console_provider_identity,
    )
    from tldw_chatbook.Chat.console_session_settings import settings_provider_catalog
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
    from tldw_chatbook.config import is_valid_provider_api_key

    environ = os.environ if environ is None else environ
    try:
        current = resolve_console_provider_identity(current_provider).readiness_key
        readiness = get_provider_readiness(  # review F8: never a sync Keychain read
            current, app_config, environ=environ, background_credentials=True
        )
    except (TypeError, ValueError):
        return None
    if readiness.ready or not readiness.requires_api_key:
        return None
    for entry in settings_provider_catalog():
        facts = entry.setup_facts
        if entry.readiness_key == current or facts.legacy or not entry.requires_api_key:
            continue
        if facts.env_var and is_valid_provider_api_key(environ.get(facts.env_var)):
            return entry.readiness_key
    return None


def build_console_setup_card_state(
    *,
    readiness: ConsoleSettingsReadiness,
    provider_label: str,
    has_model: bool,
    first_send_completed: bool,
    has_messages: bool,
    guidance_dismissed: bool,
    model: str = "",
    local_server_found: bool = False,
    held_copy: str = "",
) -> ConsoleSetupCardState:
    """Derive the onboarding surface state from the readiness single source.

    Args:
        readiness: Current settings readiness from
            ``build_console_settings_readiness``.
        provider_label: User-facing provider name for the done-step detail.
        has_model: Whether a model is selected for the session.
        first_send_completed: Persisted global flag; once True the card never
            returns, including new tabs and workspaces.
        has_messages: Whether the active transcript has any messages.
        guidance_dismissed: In-session dismissal (user started composing).
        model: The selected model, named in the ready line (TASK-34100.5).
        local_server_found: A local server answered the card's scan, so the
            missing piece is connecting it, not a key (TASK-34100.6).
        held_copy: The emergency stop's copy while it holds new runs, else ""
            (TASK-33620.1). Once setup is done the empty transcript says the
            hold instead of "Ready — type a message to begin.".

    Returns:
        Card state: full ``card`` while setup is incomplete, one ``ready_line``
        when setup is complete but nothing was ever sent, ``quiet`` otherwise.
    """
    provider_done = readiness.operability == "ready_to_send" or readiness.blocker in {
        "model_missing",
        "active_run",
    }
    held = str(held_copy or "").strip()
    if held and (has_messages or first_send_completed or (provider_done and has_model)):
        return ConsoleSetupCardState(mode="quiet", body_copy=held)
    if has_messages or first_send_completed:
        return ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)

    # TASK-34100.2 review round 2 (R2-2): a past send's 401 or 404 never
    # refuses the next send (``console_send_gate_setup_copy``), so it must never cover
    # the composer either. The one line says why the last send failed; the
    # header and the empty chat's recovery action name the fix. Pass 2: a
    # reply reservation that fills the window is the same kind of model-level
    # fact -- the gate refuses that send with copy naming its fix, so the
    # card explains it on this line instead of locking the composer.
    if (
        readiness.blocker in SEND_OUTCOME_BLOCKERS | {"capacity_exhausted"}
        and has_model
    ):
        if guidance_dismissed:
            return ConsoleSetupCardState(
                mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY
            )
        return ConsoleSetupCardState(
            mode="ready_line",
            body_copy=" — ".join(
                part
                for part in (readiness_words(readiness), readiness.detail.strip())
                if part
            ),
        )

    if provider_done and has_model:
        if guidance_dismissed:
            return ConsoleSetupCardState(
                mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY
            )
        return ConsoleSetupCardState(
            mode="ready_line",
            body_copy=_ready_line_copy(provider_label, model, readiness),
        )

    provider_name = str(provider_label or "Provider").strip() or "Provider"
    # TASK-33005.3: the active step carries the one readiness word.
    word = readiness_words(readiness)
    if local_server_found and readiness.blocker == "credential_missing":
        word = CONSOLE_SETUP_NO_MODEL_CONNECTED  # entry-exit-handoff-21
    # Capture finding 12: the word is this chat's provider's, so the step
    # names it; beside "Use detected <server>" an unnamed "Not ready · no
    # key" read as the server's. "Provider" is the callers' fallback, no name.
    # The card prints this on its own line (console_setup_modal._step_text):
    # a long name is cut so the word is never split by a wrap (TASK-33005.3).
    budget = (
        CONSOLE_SETUP_CARD_CONTENT_CELLS
        - len(CONSOLE_SETUP_STEP_DETAIL_INDENT)
        - 2
        - cell_len(word)
    )
    name = provider_name
    if cell_len(name) > budget:
        name = set_cell_size(name, budget - 1).rstrip() + "…"
    named_word = word if provider_name == "Provider" else f"{name}: {word}"
    step_one = ConsoleSetupStep(
        state="done" if provider_done else "active",
        label=(
            "Provider ready"
            if provider_done
            else _STEP_ONE_LABELS.get(readiness.blocker, "Finish provider setup")
        ),
        detail=provider_name if provider_done else named_word,
    )
    # Step two only counts as done once the provider is actually ready AND a
    # model is set: template defaults (e.g. gpt-4o) must not pre-check the
    # step on a virgin profile where the provider is still blocked.
    step_two = ConsoleSetupStep(
        state=(
            "done"
            if provider_done and has_model
            else ("active" if provider_done else "pending")
        ),
        label="Pick a model",
        detail=word if provider_done and not has_model else "",
    )
    step_three = ConsoleSetupStep(
        state="pending",
        label="Send your first message",
        detail=CONSOLE_SETUP_STEP_THREE_DETAIL,
    )
    return ConsoleSetupCardState(
        mode="card",
        steps=(step_one, step_two, step_three),
    )


def console_setup_is_blocking(
    *,
    readiness: ConsoleSettingsReadiness,
    has_model: bool,
    first_send_completed: bool,
) -> bool:
    """Return whether a fresh Console navigation would show the blocking setup card.

    Task-2852: Library's "Use in Console" pre-navigation check needs to know
    this from OUTSIDE an active Console session -- there is no mounted
    ``ChatScreen`` to ask, and no in-session transcript to read. Rather than
    hand-rolling a second copy of ``build_console_setup_card_state``'s
    branch logic, this delegates to it directly with the two fields that
    are genuinely per-session-only (``has_messages``, ``guidance_dismissed``)
    fixed at their "nothing has happened in this session yet" values -- the
    persisted global ``first_send_completed`` flag already captures "has
    Console setup ever been completed", which is what a caller outside an
    active session actually needs.

    Args:
        readiness: Current settings readiness from
            ``build_console_settings_readiness``.
        has_model: Whether a model is selected for the (would-be) session.
        first_send_completed: Persisted global first-send flag.

    Returns:
        True when the first-run setup card (``mode == "card"``) would cover
        the Console workbench right now.
    """
    state = build_console_setup_card_state(
        readiness=readiness,
        provider_label="",
        has_model=has_model,
        first_send_completed=first_send_completed,
        has_messages=False,
        guidance_dismissed=False,
    )
    return state.mode == "card"


# --- TASK-34100.2 AC#7 / AC#12 (coverage-04, E2): env-key first launch ---
#: At most this many "Found X · Use Y" offers on the Get started card; the
#: rest stay reachable from Set up provider.
CONSOLE_SETUP_ENV_KEY_ACTION_LIMIT = 3
CONSOLE_SETUP_ENV_KEY_WORKBENCH_ACTION_PREFIX = "use-env-key:"
_RUN_SETUP_HINT = "Run setup any time: Settings ▸ Diagnostics ▸ Run setup wizard."


@dataclass(frozen=True)
class EnvKeyedProvider:
    """A provider whose API key was found in the environment."""

    provider_key: str
    env_var: str
    display_name: str
    model: str
    #: The provider's table already holds a stored key (plain or still
    #: encrypted). That key outranks the env var, and committing the env
    #: offer would delete it (review round 1, F3), so it is never offered.
    has_stored_key: bool = False


def _holds_stored_key(table: Mapping[str, Any]) -> bool:
    """Whether ``api_key`` holds a value a save could lose.

    Any non-blank, non-placeholder string counts -- including ``enc:``
    ciphertext, which the resolver reads as absent but persistence would
    still delete.
    """
    from tldw_chatbook.config import PROVIDER_API_KEY_PLACEHOLDERS

    raw = table.get("api_key")
    if not isinstance(raw, str):
        return False
    value = raw.strip()
    return bool(value) and value not in PROVIDER_API_KEY_PLACEHOLDERS


@dataclass(frozen=True)
class EnvKeyFirstLaunchNotice:
    """The one first-launch notice for an install that env keys set up."""

    title: str
    message: str
    ready: bool


@dataclass(frozen=True)
class ConsoleEnvKeyAction:
    """Get started card offer: make an env-keyed provider the chat default."""

    label: str
    detail: str
    tooltip: str
    provider_key: str
    env_var: str
    model: str

    @property
    def workbench_action(self) -> str:
        return f"{CONSOLE_SETUP_ENV_KEY_WORKBENCH_ACTION_PREFIX}{self.provider_key}"


def env_keyed_providers(
    app_config: Any, environ: Any, *, key_required_only: bool = True
) -> tuple[EnvKeyedProvider, ...]:
    """Name each provider whose declared env var holds a real key.

    Config order; one entry per provider and per variable. A placeholder or
    blank value is not a found key (the loader's own validity check).

    Args:
        app_config: The live configuration snapshot.
        environ: The environment mapping to read (never written).
        key_required_only: Only cloud providers that need a key (the ones a
            card offer can make the chat default); ``False`` also names a
            local server's optional key, so a launch it set up is not silent.

    Returns:
        The env-keyed providers, each with its curated default model.
    """
    from tldw_chatbook.Chat.provider_readiness import PROVIDERS_REQUIRING_API_KEY_KEYS
    from tldw_chatbook.Chat.provider_setup_persistence import canonical_provider_key
    from tldw_chatbook.config import is_valid_provider_api_key

    api_settings = (
        app_config.get("api_settings") if isinstance(app_config, Mapping) else None
    )
    if not isinstance(api_settings, Mapping) or not isinstance(environ, Mapping):
        return ()
    found: list[EnvKeyedProvider] = []
    seen: set[str] = set()
    for section, table in api_settings.items():
        if not isinstance(table, Mapping):
            continue
        env_var = table.get("api_key_env_var")
        if not isinstance(env_var, str) or not env_var.strip():
            continue
        env_var = env_var.strip()
        try:
            provider_key = canonical_provider_key(section)
        except (TypeError, ValueError):
            continue
        if (
            (key_required_only and provider_key not in PROVIDERS_REQUIRING_API_KEY_KEYS)
            or provider_key in seen
            or env_var in seen
            or not is_valid_provider_api_key(environ.get(env_var))
        ):
            continue
        seen.update((provider_key, env_var))
        model = table.get("model")
        found.append(
            EnvKeyedProvider(
                provider_key=provider_key,
                env_var=env_var,
                display_name=provider_display_name(provider_key),
                model=model.strip() if isinstance(model, str) else "",
                has_stored_key=_holds_stored_key(table),
            )
        )
    return tuple(found)


def _env_key_offers(
    app_config: Any, environ: Any, current: str
) -> tuple[EnvKeyedProvider, ...]:
    """The env-keyed providers the Get started card offers (one rule for the
    card and the notice that names its offers): a curated model, no stored
    key a commit would delete, and not the provider new chats already use."""
    return tuple(
        entry
        for entry in env_keyed_providers(app_config, environ)
        if entry.provider_key != current and entry.model and not entry.has_stored_key
    )


def _new_chat_verdict(app_config: Any, environ: Any) -> tuple[str, str, str, Any]:
    """The provider, display name, model and shared readiness of a new chat."""
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
        build_default_console_session_settings,
        provider_config_key,
    )

    settings = build_default_console_session_settings(app_config)
    readiness = build_console_settings_readiness(
        settings, app_config=app_config, environ=environ
    )
    provider_key = provider_config_key(settings.provider)
    model = str(settings.model or "").strip()
    return provider_key, provider_display_name(provider_key), model, readiness


def build_env_key_first_launch_notice(
    app_config: Any, environ: Any
) -> EnvKeyFirstLaunchNotice | None:
    """Build the honest first-launch notice for an install env keys set up.

    'Ready to chat' is said only when the shared readiness verdict --
    credentials, connection and context capacity -- is ready for the
    provider and model new chats use. Otherwise the notice names the key it
    found and why new chats cannot use it, pointing at the Get started
    card's actions. Pure: nothing is written.

    Args:
        app_config: The live configuration snapshot.
        environ: The environment mapping to read.

    Returns:
        The notice, or ``None`` when no env key was found.
    """
    named = env_keyed_providers(app_config, environ, key_required_only=False)
    if not named:
        return None
    names = [entry.env_var for entry in named]
    shown = ", ".join(names[:2]) + (" (and more)" if len(names) > 2 else "")
    provider_key, display, model, readiness = _new_chat_verdict(app_config, environ)
    pair = f"{display} · {model}" if model else display
    if readiness.operability == "ready_to_send" and model:
        return EnvKeyFirstLaunchNotice(
            title="Provider key detected",
            message=f"Found {shown} — you're ready to chat with {pair}. {_RUN_SETUP_HINT}",
            ready=True,
        )
    own = next((entry for entry in named if entry.provider_key == provider_key), None)
    if own is not None:
        reason = readiness_words(readiness)
        # Review round 1 (F10): these blockers render the empty chat's one
        # line with its own action, not the Get started card.
        if readiness.blocker == "capacity_exhausted":
            fix = "Use Fix model limits on Console's empty chat."
        elif readiness.blocker in SEND_OUTCOME_BLOCKERS:
            fix = "Console's empty chat says what to change."
        else:
            fix = "Fix it from Console's Get started card."
        message = (
            f"Found {own.env_var}, but {pair} can't take a message yet "
            f"({reason}). {fix}"
        )
    else:
        reason = (
            "which has no key"
            if readiness.blocker in {"credential_missing", "credential_rejected"}
            else f"which isn't ready ({readiness_words(readiness)})"
        )
        offered = _env_key_offers(app_config, environ, provider_key)
        offers = "".join(f"Use {entry.display_name} or " for entry in offered[:2])
        message = (
            f"Found {shown}, but new chats use {display}, {reason}. "
            f"On Console's Get started card choose {offers}Set up provider."
        )
    return EnvKeyFirstLaunchNotice(
        title="Provider key found — setup needed", message=message, ready=False
    )


def build_console_env_key_actions(
    app_config: Any, environ: Any, *, card_mode: str
) -> tuple[ConsoleEnvKeyAction, ...]:
    """Offer each env-keyed provider other than the one new chats use.

    Only while the blocking card shows; the provider new chats already use
    needs no offer (its key is already in use).

    Args:
        app_config: The live configuration snapshot.
        environ: The environment mapping to read.
        card_mode: The current ``ConsoleSetupCardState.mode``.

    Returns:
        Up to ``CONSOLE_SETUP_ENV_KEY_ACTION_LIMIT`` offers, config order.
    """
    if card_mode != "card":
        return ()
    if not env_keyed_providers(app_config, environ):
        return ()
    from tldw_chatbook.Chat.console_session_settings import (
        build_default_console_session_settings,
        provider_config_key,
    )

    current = provider_config_key(build_default_console_session_settings(app_config).provider)
    actions = [
        ConsoleEnvKeyAction(
            label=f"Use {entry.display_name}",
            detail=f"Found {entry.env_var}",
            tooltip=(
                f"Makes {entry.display_name}"
                + (f" · {entry.model}" if entry.model else "")
                + f" the chat default, using {entry.env_var} from your environment."
            ),
            provider_key=entry.provider_key,
            env_var=entry.env_var,
            model=entry.model,
        )
        for entry in _env_key_offers(app_config, environ, current)
    ]
    return tuple(actions[:CONSOLE_SETUP_ENV_KEY_ACTION_LIMIT])


def build_env_key_provider_mutation(action: ConsoleEnvKeyAction, app_config: Any):
    """One atomic setup mutation: the provider and model as the chat default,
    ``credential_source = environment`` naming the variable (never its value).

    Built by the shared provider-setup owner, so it commits through
    ``persist_provider_setup`` exactly as Settings and setup do.

    Args:
        action: The card offer the user chose.
        app_config: The live configuration snapshot.

    Returns:
        The ``ProviderSetupMutation`` to persist.
    """
    from tldw_chatbook.Chat.console_provider_endpoints import first_configured_endpoint
    from tldw_chatbook.Provider_Contract.provider_endpoint_contract import resolve_provider_endpoint
    from tldw_chatbook.Chat.provider_setup_persistence import (
        ProviderSetupDraft,
        build_provider_setup_mutation,
    )

    if type(action) is not ConsoleEnvKeyAction:
        raise ValueError("Env-key action is invalid.")
    api_settings = (
        app_config.get("api_settings") if isinstance(app_config, Mapping) else None
    )
    table = (
        api_settings.get(action.provider_key)
        if isinstance(api_settings, Mapping)
        else None
    )
    endpoint = first_configured_endpoint(table) if isinstance(table, Mapping) else None
    if endpoint:
        resolution = resolve_provider_endpoint(action.provider_key, endpoint)
        endpoint = None if resolution.errors else resolution.persisted_endpoint
    return build_provider_setup_mutation(
        ProviderSetupDraft(
            provider=action.provider_key,
            model=action.model,
            endpoint=endpoint or "",
            credential_source="environment",
            credential_revision=0,
            draft_generation=0,
            credential_env_var=action.env_var,
        ),
        app_config,
    )
