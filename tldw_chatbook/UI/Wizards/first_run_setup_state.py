"""Pure state contracts for the first-run setup wizard.

No Textual imports, no I/O — every function is a pure transform over the
in-memory app config, mirroring Chat/console_onboarding_state.py. The wizard
Screen owns rendering and persistence; this module owns every decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

WIZARD_STATE_SECTION = "first_run"
SETUP_STARTED_KEY = "setup_started"
SETUP_COMPLETED_KEY = "setup_completed"

# Endpoint keys a local provider may use (mirrors
# Chat/local_server_discovery._ENDPOINT_CONFIG_KEYS).
_ENDPOINT_KEYS = ("api_url", "api_base_url", "api_base", "base_url", "api_endpoint", "endpoint")

_PLACEHOLDER_MARKERS = ("<", ">")


def coerce_wizard_flag(raw: Any) -> bool:
    """Tolerantly parse a persisted wizard flag.

    Args:
        raw: Whatever the TOML loader produced for the key.

    Returns:
        True only for bool True, int 1, or the string "true" (case-insensitive).
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw == 1
    if isinstance(raw, str):
        return raw.strip().lower() == "true"
    return False


def _is_real_secret(value: Any) -> bool:
    """A non-empty string that is not a <PLACEHOLDER> template value."""
    if not isinstance(value, str) or not value.strip():
        return False
    stripped = value.strip()
    return not (stripped.startswith(_PLACEHOLDER_MARKERS[0]) and stripped.endswith(_PLACEHOLDER_MARKERS[1]))


def any_provider_configured(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Return True when any provider has usable credentials or an endpoint.

    Walks the NESTED ``app_config["api_settings"]`` dict. Do not replace this
    with config.get_detected_api_providers(): that helper matches
    "api_settings.<p>" as a top-level key and always returns [].
    """
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return False
    for settings in api_settings.values():
        if not isinstance(settings, Mapping):
            continue
        if _is_real_secret(settings.get("api_key")):
            return True
        env_var = settings.get("api_key_env_var")
        if isinstance(env_var, str) and env_var.strip() and environ.get(env_var.strip()):
            return True
        for endpoint_key in _ENDPOINT_KEYS:
            if _is_real_secret(settings.get(endpoint_key)):
                return True
    return False


def _wizard_flag(app_config: Mapping[str, object], key: str) -> bool:
    section = app_config.get(WIZARD_STATE_SECTION)
    if not isinstance(section, Mapping):
        return False
    return coerce_wizard_flag(section.get(key))


def should_offer_wizard(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Auto-offer once: no wizard state keys AND nothing configured."""
    if _wizard_flag(app_config, SETUP_STARTED_KEY):
        return False
    if _wizard_flag(app_config, SETUP_COMPLETED_KEY):
        return False
    return not any_provider_configured(app_config, environ)


def should_show_resume_toast(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Started but never finished: point at Settings, never re-push."""
    return _wizard_flag(app_config, SETUP_STARTED_KEY) and not _wizard_flag(
        app_config, SETUP_COMPLETED_KEY
    )


TRACK_QUICK = "quick"
TRACK_FULL = "full"

STEP_WELCOME = "welcome"
STEP_PROVIDER = "provider"
STEP_MODEL = "model"
STEP_RAG = "rag"
STEP_TOOLS = "tools"
STEP_NOTES = "notes"
STEP_APPEARANCE = "appearance"
STEP_PROTECT = "protect-keys"
STEP_SUMMARY = "summary"

_FULL_TRACK = (
    STEP_WELCOME, STEP_PROVIDER, STEP_MODEL, STEP_RAG, STEP_TOOLS,
    STEP_NOTES, STEP_APPEARANCE, STEP_PROTECT, STEP_SUMMARY,
)
_QUICK_TRACK = (STEP_WELCOME, STEP_PROVIDER, STEP_MODEL, STEP_PROTECT, STEP_SUMMARY)

WIZARD_OWNED_SECTIONS = frozenset(
    {"chat_defaults", "embedding_config", "tools", "notes", "general",
     "splash_screen", WIZARD_STATE_SECTION}
)
_API_SETTINGS_PREFIX = "api_settings."


def active_step_ids(track: str, *, key_entered: bool) -> tuple[str, ...]:
    """Resolve the ordered active step ids for a track.

    Args:
        track: TRACK_QUICK or TRACK_FULL (anything else falls back to full).
        key_entered: Whether any secret was entered this run; gates STEP_PROTECT.
    """
    base = _QUICK_TRACK if track == TRACK_QUICK else _FULL_TRACK
    if key_entered:
        return base
    return tuple(step for step in base if step != STEP_PROTECT)


def stored_plaintext_key_present(app_config: Mapping[str, object]) -> bool:
    """Whether a real, unencrypted provider secret already sits on disk.

    Bug-4 fix: ``active_step_ids``'s ``key_entered`` gate previously only
    reflected secrets typed THIS run, so a rerun over a config that already
    has a plaintext key (e.g. hand-edited config.toml, or a completed prior
    run) could never reach Protect Keys without retyping a credential. This
    is the config-derived signal the spec actually wants: any configured
    key, regardless of when it was entered, as long as it is not already
    protected by encryption.

    Args:
        app_config: Loaded app configuration.

    Returns:
        True when at least one ``api_settings.<provider>.api_key`` holds a
        real (non-placeholder) secret AND config encryption is not enabled.
    """
    if coerce_wizard_flag(_section(app_config, "encryption").get("enabled")):
        return False
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return False
    for settings in api_settings.values():
        if isinstance(settings, Mapping) and _is_real_secret(settings.get("api_key")):
            return True
    return False


def build_provider_commit(
    *, provider_key: str, api_key: str | None, api_url: str | None
) -> dict[str, dict[str, Any]]:
    """Mutation for the provider step. Empty when the key lives in the env."""
    values: dict[str, Any] = {}
    if api_key:
        values["api_key"] = api_key
    if api_url:
        values["api_url"] = api_url
    if not values:
        return {}
    return {f"{_API_SETTINGS_PREFIX}{provider_key}": values}


def build_model_commit(*, provider_value: str, model_id: str) -> dict[str, dict[str, Any]]:
    """Mutation for the model step.

    Args:
        provider_value: The ``chat_defaults.provider`` value the model is
            paired with (whatever form ProviderStep last committed).
        model_id: The chosen default model id.

    Returns:
        The section/value mapping to persist under ``chat_defaults``.
    """
    return {"chat_defaults": {"provider": provider_value, "model": model_id}}


def curated_models_for_provider(
    catalog: Mapping[str, Any], provider_value: str
) -> list[str]:
    """Look up curated fallback models for a provider, in ANY key form.

    ProviderStep persists ``chat_defaults.provider`` as the RAW provider_key
    (e.g. "llama_cpp", "openai") -- matching chat_screen's detected-server
    path -- while the curated ``[providers]`` table in config.toml (surfaced
    via ``config.get_cli_providers_and_models()``) is keyed by human display
    names (e.g. "OpenAI"). A plain ``catalog.get(provider_value)`` would
    silently return [] whenever the two forms disagree even though a
    matching entry exists, so normalize both sides via
    ``provider_readiness.provider_config_key`` before comparing.
    """
    if not provider_value:
        return []
    direct = catalog.get(provider_value)
    if direct:
        return list(direct)
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    target_key = provider_config_key(provider_value)
    if not target_key:
        return []
    for name, models in catalog.items():
        if provider_config_key(str(name)) == target_key:
            return list(models)
    return []


def build_rag_commit(*, default_model_id: str) -> dict[str, dict[str, Any]]:
    """Mutation for the RAG step.

    Args:
        default_model_id: The chosen default embedding model id.

    Returns:
        The section/value mapping to persist under ``embedding_config``.
    """
    return {"embedding_config": {"default_model_id": default_model_id}}


def build_tools_commit(*, gate_values: Mapping[str, bool]) -> dict[str, dict[str, Any]]:
    """Mutation for the tools step.

    Args:
        gate_values: Mapping of tool gate key to the desired enabled state.
            Callers typically pass only the delta (see
            ``tools_commit_delta``), not every gate.

    Returns:
        The section/value mapping to persist under ``tools``.
    """
    return {"tools": {key: bool(value) for key, value in gate_values.items()}}


def tools_commit_delta(
    *, gate_values: Mapping[str, bool], current_gates: Mapping[str, bool]
) -> dict[str, bool]:
    """Keys whose desired value differs from the currently persisted gate.

    ``current_gates`` should hold the EFFECTIVE value for every key present
    in ``gate_values`` (False when the ``[tools]`` section has no entry for
    it -- the same default ``BuiltinToolProvider``'s own gate check uses).
    A fresh config where every switch starts and stays False therefore
    produces an empty delta (no write), while flipping a previously-True
    gate back to False is still reported -- unlike a naive "only ever
    persist True" filter, which silently drops OFF-transitions on re-run.
    """
    return {
        key: bool(value)
        for key, value in gate_values.items()
        if bool(value) != bool(current_gates.get(key, False))
    }


def build_notes_commit(
    *, sync_directory: str | None = None, auto_sync_enabled: bool
) -> dict[str, dict[str, Any]]:
    """Mutation for the notes-sync step.

    ``sync_directory`` is optional: omit it (leave as None) to commit only
    the enabled flag -- e.g. an OFF-transition on re-run, where the
    directory should survive untouched. ``save_settings_to_cli_config``
    merges per-key within a section, so a dict missing "sync_directory"
    never clobbers the persisted value; passing an empty string here would.

    Args:
        sync_directory: The notes-sync directory to persist, or None to
            omit the key entirely (see above).
        auto_sync_enabled: Whether notes sync should be turned on.

    Returns:
        The section/value mapping to persist under ``notes``.
    """
    values: dict[str, Any] = {"auto_sync_enabled": auto_sync_enabled}
    if sync_directory is not None:
        values["sync_directory"] = sync_directory
    return {"notes": values}


def build_appearance_commit(
    *,
    default_theme: str | None,
    splash_card: str | None = None,
    reset_splash_to_random: bool = False,
) -> dict[str, dict[str, Any]]:
    """Mutation for the appearance step.

    Bug-2 fix: both ``default_theme`` and the splash-card write are now
    each individually optional, so the caller (``AppearanceStep.commit()``)
    can build a delta-aware commit -- e.g. a rerun that only changes the
    splash card omits ``general.default_theme`` entirely instead of
    rewriting it back to a (possibly stale) fallback value.

    Args:
        default_theme: Theme name to persist under
            ``general.default_theme``, or falsy (``None``/``""``) to omit
            that key entirely -- e.g. the chosen theme matches what is
            already persisted, so nothing needs to change.
        splash_card: A specific card name to persist under
            ``splash_screen.card_selection``, or falsy to leave that key
            out (e.g. nothing was chosen, or "Surprise me" was picked --
            see ``reset_splash_to_random``).
        reset_splash_to_random: When True and ``splash_card`` is falsy,
            write ``splash_screen.card_selection = "random"``. This is the
            explicit "Surprise me" choice over a previously persisted
            specific card, which otherwise has no truthy value of its own
            to signal that a write is even needed.

    Returns:
        The section/value mapping to persist; empty where nothing changed.
    """
    commit: dict[str, dict[str, Any]] = {}
    if default_theme:
        commit["general"] = {"default_theme": default_theme}
    if splash_card:
        commit["splash_screen"] = {"card_selection": splash_card}
    elif reset_splash_to_random:
        commit["splash_screen"] = {"card_selection": "random"}
    return commit


def build_wizard_state_commit(
    *, started: bool | None = None, completed: bool | None = None
) -> dict[str, dict[str, Any]]:
    """Mutation for the wizard's own progress flags.

    Args:
        started: Set ``first_run.setup_started`` to this value, or omit
            (leave as None) to leave that key out of the commit.
        completed: Set ``first_run.setup_completed`` to this value, or omit
            (leave as None) to leave that key out of the commit.

    Returns:
        The section/value mapping to persist under ``first_run``; empty
        when neither flag was passed.
    """
    values: dict[str, Any] = {}
    if started is not None:
        values[SETUP_STARTED_KEY] = started
    if completed is not None:
        values[SETUP_COMPLETED_KEY] = completed
    return {WIZARD_STATE_SECTION: values} if values else {}


def invalidate_model_for_provider_change(
    commit: dict[str, dict[str, Any]],
    *,
    previous_provider_value: str | None,
    new_provider_value: str,
) -> dict[str, dict[str, Any]]:
    """Supersede a stale model when the committed provider changes.

    Without this, Back-and-switch leaves chat_defaults pairing the new
    provider with the old provider's model.

    Bug-3 fix: ``previous_provider_value`` is compared with ``!=`` against
    an empty-string default rather than gated behind a truthiness check.
    The old ``if previous_provider_value and ...`` guard treated a falsy
    "no previous provider" (``None`` or ``""``) as "nothing to compare",
    which silently skipped the chat_defaults write on a genuinely first-ever
    provider selection (an empty/absent persisted ``chat_defaults.provider``
    is exactly as "different" from a real new provider as any other prior
    provider would be). Callers resolve ``previous_provider_value`` as the
    in-session previous commit if one exists this run, else the PERSISTED
    ``chat_defaults.provider`` (see ``read_wizard_prefill``), so this
    function no longer needs to special-case "no information at all".

    Args:
        commit: The commit built so far (e.g. from ``build_provider_commit``).
        previous_provider_value: The provider ``chat_defaults.provider`` was
            associated with before this commit -- the in-session value if
            this step has already committed once this run, else the
            persisted value (``""`` on a fresh/absent config). ``None`` is
            treated the same as ``""``.
        new_provider_value: The provider value this commit is selecting.

    Returns:
        ``commit`` unchanged when the provider is the same as before;
        otherwise a copy with ``chat_defaults`` set to the new provider and
        an emptied model.
    """
    effective_previous = previous_provider_value or ""
    if effective_previous != new_provider_value:
        merged = dict(commit)
        merged["chat_defaults"] = {"provider": new_provider_value, "model": ""}
        return merged
    return commit


def commit_sections_allowed(section_values: Mapping[str, Mapping[Any, Any]]) -> bool:
    """The invariant oracle: wizard commits touch only wizard-owned sections."""
    for section in section_values:
        if section in WIZARD_OWNED_SECTIONS:
            continue
        if section.startswith(_API_SETTINGS_PREFIX) and len(section) > len(_API_SETTINGS_PREFIX):
            continue
        return False
    return True


@dataclass(frozen=True)
class SecretPresence:
    """Whether a provider secret exists — never the secret itself."""

    configured: bool
    env_var: str | None = None
    env_var_set: bool = False


@dataclass(frozen=True)
class WizardPrefill:
    """Current config values for re-run prefill (no secrets)."""

    provider_value: str = ""
    model_id: str = ""
    sync_directory: str = ""
    auto_sync_enabled: bool = False
    default_theme: str = ""
    tool_gates: tuple[tuple[str, bool], ...] = ()
    card_selection: str = ""
    """Persisted ``splash_screen.card_selection`` -- "" when never set.

    Bug-2 fix: AppearanceStep needs this to tell "the user just explicitly
    re-picked Surprise-me over a persisted specific card" (a real reset)
    apart from "nothing was ever configured" (a no-op).
    """


@dataclass(frozen=True)
class SummaryRow:
    """One ✓/✗ line of the final summary matrix."""

    label: str
    ok: bool
    detail: str = ""


def _section(app_config: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = app_config.get(name)
    return section if isinstance(section, Mapping) else {}


def read_provider_secret_presence(
    app_config: Mapping[str, object],
    environ: Mapping[str, str],
    *,
    provider_key: str,
) -> SecretPresence:
    """Resolve whether a provider secret is configured, without ever reading it.

    Falls back to the conventional ``<PROVIDER>_API_KEY`` environment variable
    name (via ``provider_readiness.default_api_key_env_var``, the same
    resolution Chat's own readiness check uses) when ``api_key_env_var`` is
    not explicitly present in ``app_config``. Without this fallback, a wizard
    run before the packaged default config.toml's ``api_key_env_var`` entries
    have been persisted to disk (or any app_config that omits them) could
    never detect an already-exported key -- exactly the first-run scenario
    this step exists for.
    """
    from tldw_chatbook.Chat.provider_readiness import default_api_key_env_var

    settings = _section(_section(app_config, "api_settings"), provider_key)
    env_var_raw = settings.get("api_key_env_var")
    env_var = (
        env_var_raw.strip()
        if isinstance(env_var_raw, str) and env_var_raw.strip()
        else default_api_key_env_var(provider_key)
    )
    env_var_set = bool(env_var and environ.get(env_var))
    inline = _is_real_secret(settings.get("api_key"))
    return SecretPresence(
        configured=inline or env_var_set, env_var=env_var, env_var_set=env_var_set
    )


def read_wizard_prefill(app_config: Mapping[str, object]) -> WizardPrefill:
    chat_defaults = _section(app_config, "chat_defaults")
    notes = _section(app_config, "notes")
    general = _section(app_config, "general")
    tools = _section(app_config, "tools")
    splash_screen = _section(app_config, "splash_screen")
    return WizardPrefill(
        provider_value=str(chat_defaults.get("provider") or ""),
        model_id=str(chat_defaults.get("model") or ""),
        sync_directory=str(notes.get("sync_directory") or ""),
        auto_sync_enabled=coerce_wizard_flag(notes.get("auto_sync_enabled")),
        default_theme=str(general.get("default_theme") or ""),
        tool_gates=tuple(
            (str(key), coerce_wizard_flag(value)) for key, value in tools.items()
        ),
        card_selection=str(splash_screen.get("card_selection") or ""),
    )


def build_summary_rows(
    app_config: Mapping[str, object],
    environ: Mapping[str, str],
    *,
    rag_deps_installed: bool,
) -> tuple[SummaryRow, ...]:
    """Build the ✓/✗ matrix strictly from persisted config (never step memory)."""
    prefill = read_wizard_prefill(app_config)
    provider_ok = any_provider_configured(app_config, environ)
    tools_on = [key for key, value in prefill.tool_gates if value]
    notes_on = prefill.auto_sync_enabled and bool(prefill.sync_directory)
    encryption_on = coerce_wizard_flag(_section(app_config, "encryption").get("enabled"))
    rag_model = str(_section(app_config, "embedding_config").get("default_model_id") or "")
    if not rag_deps_installed:
        rag_row = SummaryRow("RAG", False, "embeddings deps not installed")
    elif rag_model:
        rag_row = SummaryRow("RAG", True, f"embedding model: {rag_model}")
    else:
        rag_row = SummaryRow("RAG", False, "no embedding model selected")
    return (
        SummaryRow("Provider", provider_ok, "" if provider_ok else "no credentials or endpoint"),
        SummaryRow(
            "Default model",
            bool(prefill.model_id),
            prefill.model_id or "not selected",
        ),
        rag_row,
        SummaryRow(
            "Tools",
            bool(tools_on),
            f"{len(tools_on)} enabled" if tools_on else "all off (default)",
        ),
        SummaryRow(
            "Notes sync",
            notes_on,
            prefill.sync_directory if notes_on else "off",
        ),
        SummaryRow(
            "Theme", bool(prefill.default_theme), prefill.default_theme or "default"
        ),
        SummaryRow(
            "Key encryption", encryption_on, "" if encryption_on else "off"
        ),
    )
