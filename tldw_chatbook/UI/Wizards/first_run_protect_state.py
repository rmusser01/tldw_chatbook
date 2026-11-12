"""What the first-run Protect step says, derived from the persisted config.

TASK-34100.4 (protect-summary-04, -06, -07). The step used to model two states
(keys or no keys), decided by whether a key was TYPED this run or a plaintext
key sat in the in-memory app config. That in-memory view has no
``[encryption]`` table at all, so an encrypted, unlocked config read as "plain
text" and was offered a second password, while a keyless or env-key user read
the encryption pitch next to "No API keys saved yet", and a key typed on
Provider but never saved (Model skipped) was offered a password for a key that
was not on disk.

Here the state comes from the config FILE (the same merged read-back Summary
uses) plus the environment and the provider a key was typed for:

* ``encrypted`` -- ``[encryption] enabled``; offer "Change password…".
* ``plaintext`` -- a secret is saved as plain text; name each provider (or
  the config key, for a secret no provider owns) and the file, and offer two
  explicit buttons. "A secret" means exactly what enabling encryption would
  encrypt (review round 1, R1-F1): a legacy ``[API] <provider>_api_key`` and a
  custom endpoint's ``api_key`` count, not just ``api_settings``.
* ``unsaved`` -- a key was typed this run but is not on disk; offer
  "Back to Model".
* ``env_only`` -- keys come only from environment variables; name them.
* ``nothing`` -- no key anywhere.
* ``unreadable`` -- config.toml could not be read and its ``[encryption]``
  state is unknown: no offer, and no guess (review round 1, R1-F5).
* ``stranded`` -- encryption is off, but a saved value is still ``enc:``
  ciphertext under an earlier password: name it and where to re-enter or
  clear it. Encrypting is refused over it, so there is no offer (review
  v5r1, R1-4: this read "No API key is saved in config.toml").

A key typed this run but not on disk is called out in every state that has
saved keys too, encrypted included (review v5r1, R1-3).

Pure: no I/O, no Textual. The step reads the file off the UI thread and
renders what :func:`protect_copy` returns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

PROTECT_ENCRYPTED = "encrypted"
PROTECT_PLAINTEXT = "plaintext"
PROTECT_UNSAVED = "unsaved"
PROTECT_ENV_ONLY = "env_only"
PROTECT_NOTHING = "nothing"
PROTECT_UNREADABLE = "unreadable"
PROTECT_STRANDED = "stranded"

#: States where config.toml held no key to protect: the tracker shows '–'
#: for Protect instead of the ✓ that reads as "your keys are protected".
NOTHING_TO_PROTECT = frozenset({PROTECT_UNSAVED, PROTECT_ENV_ONLY, PROTECT_NOTHING})
#: Protect wears ✓ only when the keys are encrypted (review round 1, R1-F7).
#: Every other recorded state shows '–': nothing was there to protect, the
#: keys were kept as plain text (Summary flags that; the tracker does not cry
#: wolf over a skippable step, TASK-25818), or the file could not be read.
PROTECT_UNTICKED = NOTHING_TO_PROTECT | {
    PROTECT_PLAINTEXT,
    PROTECT_UNREADABLE,
    PROTECT_STRANDED,
}

#: The step's two primary-button labels; the button keeps one id.
ENCRYPT_LABEL = "Encrypt with a password…"
CHANGE_LABEL = "Change password…"
KEEP_PLAIN_LABEL = "Keep as plain text"
BACK_TO_MODEL_LABEL = "Back to Model"

ENCRYPTED_STATUS = (
    "✓ Your saved API keys are encrypted. You'll enter your master password "
    "when chatbook starts."
)
#: The outcome line under the buttons; the body above already says what
#: encryption means for startup.
ENABLED_STATUS = "✓ Encryption enabled."
#: Where a still-encrypted value is re-entered or cleared. Mirrors the
#: Settings card's refusal (settings_encryption): Providers & Models edits only
#: `api_settings.<provider>.api_key`; Advanced Config edits any value.
WHERE_PROVIDERS_PAGE = "Settings ▸ Providers & Models"
WHERE_ADVANCED_PAGE = "Settings ▸ Advanced Config"
CHANGED_STATUS = (
    "✓ Master password changed. Use the new one the next time chatbook starts."
)


@dataclass(frozen=True, slots=True)
class ProtectState:
    """One Protect state and the names its copy needs.

    Attributes:
        kind: One of the ``PROTECT_*`` constants.
        providers: Display names of the providers (and custom endpoints)
            whose key is saved in config.toml (``plaintext``), in config
            order.
        env_vars: Environment variables that supply a key (``env_only``).
        unsaved_provider: Display name of the provider whose typed key is not
            on disk (``unsaved``, or ``plaintext`` beside other stored keys).
        other_secrets: Dotted config keys of saved plaintext secrets that no
            provider owns (``plaintext``), in config order.
        stranded: What is still encrypted under an earlier password while
            encryption is off ("OpenAI key", or a dotted config key), in
            config order (``stranded``).
        stranded_where: The Settings page that can re-enter or clear every
            stranded value (``stranded``).
    """

    kind: str
    providers: tuple[str, ...] = ()
    env_vars: tuple[str, ...] = ()
    unsaved_provider: str = ""
    other_secrets: tuple[str, ...] = ()
    stranded: tuple[str, ...] = ()
    stranded_where: str = ""

    @property
    def nothing_to_protect(self) -> bool:
        """Whether config.toml held no key, so the tracker shows '–'."""
        return self.kind in NOTHING_TO_PROTECT


@dataclass(frozen=True, slots=True)
class ProtectCopy:
    """Every line the step renders for one state.

    Attributes:
        title: The step heading.
        body: The explanation (may be empty).
        path_line: The config file line, shown dim (may be empty).
        later_line: Where to change this later -- always names the Settings
            Encryption card.
        primary_label: The primary button's label, or "" to hide it.
        show_keep_plain: Whether "Keep as plain text" is shown.
        show_back_to_model: Whether "Back to Model" is shown.
    """

    title: str
    body: str
    path_line: str
    later_line: str
    primary_label: str
    show_keep_plain: bool = False
    show_back_to_model: bool = False


def encryption_card_path() -> str:
    """Where encryption lives after setup: the Settings Encryption card."""
    from tldw_chatbook.UI.Screens.settings_encryption import ENCRYPTION_CARD_PATH

    return ENCRYPTION_CARD_PATH


def _flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _display_name(provider_key: str) -> str:
    """The provider's name for Protect's copy.

    The catalog's picker asides ("Mistral AI (legacy alias)") are dropped:
    "Your Mistral AI (legacy alias) key is saved…" names the slot, not the
    provider.
    """
    import re

    from tldw_chatbook.Chat.provider_catalog import provider_display_name

    name = provider_display_name(provider_key) or provider_key
    return re.sub(r"\s*\(legacy [^)]*\)\s*$", "", name) or provider_key


#: Top-level tables that hold no user secret: the encryption metadata and
#: setup's own checkpoint (its draft values are allowlisted non-secrets).
_NEVER_SECRET_TABLES = frozenset({"encryption", "first_run"})


def plaintext_secret_paths(config: Mapping[str, object]) -> tuple[tuple[str, ...], ...]:
    """Config paths of the real secrets that encrypting would encrypt.

    Mirrors ``config.encrypt_api_keys_in_config``: every table (arrays are
    not walked, as there), every key ``is_sensitive_config_key`` names, a
    non-blank string that is not already ``enc:`` and not a ``<placeholder>``.
    On top of that, a template value is never the user's secret: one equal to
    the shipped default at the same path (the merged read-back always carries
    ``YOUR_OPENAI_API_KEY…`` and the tldw_api placeholder token), a known
    provider-key placeholder, or the tldw_api placeholder token. Nor is
    anything in setup's own ``[first_run]`` checkpoint, which never holds a
    credential: its ``provider_key = "llama_cpp"`` matches the ``_key``
    suffix, and a keyless user was offered encryption for it (seen live).

    Args:
        config: The merged config read back from config.toml.

    Returns:
        Each secret's path as a tuple of keys, in config order.
    """
    from tldw_chatbook.config import (
        DEFAULT_CONFIG_FROM_TOML,
        TLDW_API_PLACEHOLDER_AUTH_TOKEN,
        resolve_provider_api_key,
    )
    from tldw_chatbook.Utils.sensitive_config_keys import is_sensitive_config_key

    found: list[tuple[str, ...]] = []

    def _walk(table: Mapping, defaults: object, prefix: tuple[str, ...]) -> None:
        for key, value in table.items():
            if not prefix and key in _NEVER_SECRET_TABLES:
                continue
            default = defaults.get(key) if isinstance(defaults, Mapping) else None
            if isinstance(value, Mapping):
                _walk(value, default, prefix + (str(key),))
                continue
            if not (isinstance(value, str) and is_sensitive_config_key(key)):
                continue
            # None for blanks, provider placeholders and `enc:` ciphertext.
            text = resolve_provider_api_key(value)
            if (
                text is None
                or (text.startswith("<") and text.endswith(">"))
                or text == TLDW_API_PLACEHOLDER_AUTH_TOKEN
                or (isinstance(default, str) and text == default.strip())
            ):
                continue
            found.append(prefix + (str(key),))

    _walk(config, DEFAULT_CONFIG_FROM_TOML, ())
    return tuple(found)


def stranded_secret_paths(
    config: Mapping[str, object],
) -> tuple[tuple[str, ...], ...]:
    """Config paths of every value still ``enc:`` ciphertext.

    Mirrors ``config._encrypted_value_paths`` (what the Settings card names
    when it refuses to encrypt): every table but ``[encryption]`` -- and
    setup's own ``[first_run]`` checkpoint -- arrays not walked. With
    encryption off, each one is stranded under an earlier password.

    Args:
        config: The merged config read back from config.toml.

    Returns:
        Each value's path as a tuple of keys, in config order.
    """
    from tldw_chatbook.config import is_encrypted_config_value

    found: list[tuple[str, ...]] = []

    def _walk(table: Mapping, prefix: tuple[str, ...]) -> None:
        for key, value in table.items():
            if not prefix and key in _NEVER_SECRET_TABLES:
                continue
            if isinstance(value, Mapping):
                _walk(value, prefix + (str(key),))
            elif is_encrypted_config_value(value):
                found.append(prefix + (str(key),))

    _walk(config, ())
    return tuple(found)


def stranded_fix_page(paths: tuple[tuple[str, ...], ...]) -> str:
    """The Settings page that can re-enter or clear every path in ``paths``."""
    if all(
        len(path) == 3 and path[0] == "api_settings" and path[2] == "api_key"
        for path in paths
    ):
        return WHERE_PROVIDERS_PAGE
    return WHERE_ADVANCED_PAGE


def _secret_owner(
    path: tuple[str, ...], config: Mapping[str, object]
) -> tuple[str, str]:
    """``(provider key, display name)`` owning a secret, or ``("", "")``.

    ``api_settings.<p>.*`` and a legacy ``API.<p>_api_key`` for a known
    provider belong to ``p``; ``custom_endpoints.<slug>.api_key`` to that
    endpoint, by its display name. Anything else has no provider owner.
    """
    from tldw_chatbook.Chat.provider_catalog import (
        PROVIDER_DISPLAY_NAMES,
        provider_display_name,
    )
    from tldw_chatbook.config import normalize_provider_config_key

    if len(path) >= 3 and path[0] == "api_settings":
        key = normalize_provider_config_key(path[1])
        return key, _display_name(path[1])
    if len(path) == 2 and path[0] == "API" and path[1].lower().endswith("_api_key"):
        key = normalize_provider_config_key(path[1][: -len("_api_key")])
        if key in PROVIDER_DISPLAY_NAMES:
            return key, _display_name(key)
    if len(path) == 3 and path[0] == "custom_endpoints" and path[2] == "api_key":
        endpoint_id = f"custom-ep:{path[1]}"
        try:
            name = provider_display_name(endpoint_id, config)
        except Exception:
            name = ""
        return endpoint_id, (name if name and name != endpoint_id else path[1])
    return "", ""


def read_protect_state(
    config: Mapping[str, object],
    environ: Mapping[str, str],
    *,
    typed_provider_key: str = "",
    typed_key_committed: bool = False,
) -> ProtectState:
    """Derive the Protect state from the persisted config.

    Args:
        config: The merged config read back from config.toml (decrypted with
            the session password when one is held).
        environ: Process environment, for keys supplied by env vars.
        typed_provider_key: The provider a key was typed for on the Provider
            step this run, or "" when none was typed.
        typed_key_committed: Whether the wizard owner successfully committed
            this exact staged draft. An older saved key is not a receipt.

    Returns:
        The state, with the provider names or env vars its copy names.
    """
    from tldw_chatbook.config import (
        is_valid_provider_api_key,
        normalize_provider_config_key,
    )

    stored_keys: set[str] = set()
    stored: list[str] = []
    others: list[str] = []
    for path in plaintext_secret_paths(config):
        owner_key, owner_name = _secret_owner(path, config)
        if owner_key:
            stored_keys.add(owner_key)
            stored.append(owner_name)
        else:
            others.append(".".join(path))
    still_encrypted = stranded_secret_paths(config)
    typed = (
        normalize_provider_config_key(typed_provider_key) if typed_provider_key else ""
    )

    def _unsaved(saved: set[str]) -> str:
        return (
            _display_name(typed_provider_key)
            if typed and (not typed_key_committed or typed not in saved)
            else ""
        )

    encryption = config.get("encryption")
    if isinstance(encryption, Mapping) and _flag(encryption.get("enabled")):
        # Review v5r1 (R1-3): a key typed this run but never saved is called
        # out here too. A stored value (locked or decrypted) belongs to this
        # run only when the wizard owner committed the exact staged draft.
        saved = stored_keys | {_secret_owner(p, config)[0] for p in still_encrypted}
        return ProtectState(PROTECT_ENCRYPTED, unsaved_provider=_unsaved(saved))
    # With encryption off, a saved key still `enc:` is the user's typed key
    # only after Model saves the new one over it, so it does not count.
    unsaved = _unsaved(stored_keys)
    if still_encrypted:
        # A provider's key first: encrypting also encrypts template values
        # (the tldw_api token, embedding models' keys), which strand with it.
        owned: list[str] = []
        unowned: list[str] = []
        for path in still_encrypted:
            owner_key, owner_name = _secret_owner(path, config)
            if owner_key:
                owned.append(f"{owner_name} key")
            else:
                unowned.append(".".join(path))
        labels = owned + unowned
        return ProtectState(
            PROTECT_STRANDED,
            providers=tuple(dict.fromkeys(stored)),
            unsaved_provider=unsaved,
            other_secrets=tuple(dict.fromkeys(others)),
            stranded=tuple(dict.fromkeys(labels)),
            stranded_where=stranded_fix_page(still_encrypted),
        )
    if stored or others:
        return ProtectState(
            PROTECT_PLAINTEXT,
            providers=tuple(dict.fromkeys(stored)),
            unsaved_provider=unsaved,
            other_secrets=tuple(dict.fromkeys(others)),
        )
    if unsaved:
        return ProtectState(PROTECT_UNSAVED, unsaved_provider=unsaved)
    api_settings = config.get("api_settings")
    env_vars: list[str] = []
    for settings in (
        api_settings if isinstance(api_settings, Mapping) else {}
    ).values():
        env_var = (
            settings.get("api_key_env_var") if isinstance(settings, Mapping) else None
        )
        if (
            isinstance(env_var, str)
            and env_var.strip()
            and is_valid_provider_api_key(environ.get(env_var.strip()))
        ):
            env_vars.append(env_var.strip())
    if env_vars:
        return ProtectState(PROTECT_ENV_ONLY, env_vars=tuple(dict.fromkeys(env_vars)))
    return ProtectState(PROTECT_NOTHING)


def _join(names: tuple[str, ...]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


#: How many other secrets' config keys the plaintext copy names.
_NAMED_OTHER_SECRETS = 3


def _plaintext_subject(state: ProtectState) -> tuple[str, bool]:
    """The plaintext body's subject, and whether it is plural.

    "Your OpenAI key", "Your OpenAI and Anthropic keys", "Your OpenAI key and
    1 other secret (SearchEngines.serper_search_api_key)", or, with no
    provider, "2 secrets (…)".
    """
    others = state.other_secrets
    named = ", ".join(others[:_NAMED_OTHER_SECRETS])
    if len(others) > _NAMED_OTHER_SECRETS:
        named += f" and {len(others) - _NAMED_OTHER_SECRETS} more"
    other_phrase = f"{len(others)} {{}}secret{'s' if len(others) > 1 else ''} ({named})"
    if not state.providers:
        return other_phrase.format(""), len(others) > 1
    keys = f"{_join(state.providers)} key{'s' if len(state.providers) > 1 else ''}"
    if not others:
        return f"Your {keys}", len(state.providers) > 1
    return f"Your {keys} and {other_phrase.format('other ')}", True


def _unsaved_sentence(provider: str) -> str:
    """The callout for a key typed this run that is not on disk."""
    return (
        f" The {provider} key you typed isn't saved: it is saved together with "
        "the model you choose on Model."
    )


def config_path_line(config_path: str, width: int = 0) -> str:
    """Return "Config file: <path>", keeping every character of the path.

    A path too long for ``width`` cells goes on its own lines, broken only
    after a separator, so no directory is dropped and no name is split.

    Args:
        config_path: The config file path ("" for none).
        width: Cells available per line; 0 or less never folds.

    Returns:
        The dim path line, or "" when there is no path.
    """
    from tldw_chatbook.Utils.Utils import fold_path_lines

    if not config_path:
        return ""
    line = f"Config file: {config_path}"
    if width <= 0 or len(line) <= width:
        return line
    return "Config file:\n" + fold_path_lines(config_path, width)


def protect_copy(
    state: ProtectState, *, config_path: str, width: int = 0
) -> ProtectCopy:
    """The lines and buttons the step shows for ``state``.

    No line names a "Skip" control: the footer has Back, Next and Exit
    setup, and the plaintext offer has its own "Keep as plain text" button.

    Args:
        state: The derived Protect state.
        config_path: The config file path, for the dim path line.
        width: Cells per line for that path line (see `config_path_line`).

    Returns:
        The step's copy and which buttons it shows.
    """
    card = encryption_card_path()
    path_line = config_path_line(config_path, width)
    if state.kind == PROTECT_ENCRYPTED:
        body = ENCRYPTED_STATUS
        if state.unsaved_provider:
            body += _unsaved_sentence(state.unsaved_provider)
        return ProtectCopy(
            title="Your keys are protected",
            body=body,
            path_line=path_line,
            later_line=(
                f"Change the password or turn encryption off any time in {card}."
            ),
            primary_label=CHANGE_LABEL,
            show_back_to_model=bool(state.unsaved_provider),
        )
    if state.kind == PROTECT_STRANDED:
        many = len(state.stranded) > 1
        it = "them" if many else "it"
        named = state.stranded[:_NAMED_OTHER_SECRETS]
        if len(state.stranded) > len(named):
            more = len(state.stranded) - len(named)
            names = f"{', '.join(named)} and {more} more"
        else:
            names = _join(named)
        body = (
            f"! Your saved {names} {'are' if many else 'is'} "
            "still encrypted with an earlier master password, and encryption "
            f"is now off, so chatbook can't read {it}. Re-enter or clear {it} "
            f"in {state.stranded_where}."
        )
        if state.providers or state.other_secrets:
            subject, plural = _plaintext_subject(state)
            body += f" {subject} {'are' if plural else 'is'} saved as plain text."
        if state.unsaved_provider:
            body += _unsaved_sentence(state.unsaved_provider)
        # No offer: encrypting is refused while a value is stranded.
        return ProtectCopy(
            title="Saved keys can't be read" if many else "A saved key can't be read",
            body=body,
            path_line=path_line,
            later_line=f"Once that's fixed, you can encrypt your saved keys in {card}.",
            primary_label="",
            show_back_to_model=bool(state.unsaved_provider),
        )
    if state.kind == PROTECT_PLAINTEXT:
        subject, many = _plaintext_subject(state)
        it = "them" if many else "it"
        body = (
            f"{subject} {'are' if many else 'is'} saved as plain text in "
            f"config.toml, where any program running as you can read {it}. "
            f"Encrypt {it} with a master password, and chatbook will ask for "
            "that password each time it starts."
        )
        if state.unsaved_provider:
            body += _unsaved_sentence(state.unsaved_provider)
        if state.providers:
            title = "Protect your API keys" if many else "Protect your API key"
        else:
            title = (
                "Protect your saved secrets" if many else "Protect your saved secret"
            )
        return ProtectCopy(
            title=title,
            body=body,
            path_line=path_line,
            later_line=f"You can change this later in {card}.",
            primary_label=ENCRYPT_LABEL,
            show_keep_plain=True,
            show_back_to_model=bool(state.unsaved_provider),
        )
    if state.kind == PROTECT_UNREADABLE:
        return ProtectCopy(
            title="Couldn't read config.toml",
            body=(
                "! chatbook couldn't read config.toml, so it can't tell whether "
                "your saved keys are encrypted. Nothing was changed."
            ),
            path_line=path_line,
            later_line=f"Check encryption any time in {card}.",
            primary_label="",
        )
    later = f"If you save a key later, you can encrypt it in {card}."
    if state.kind == PROTECT_UNSAVED:
        return ProtectCopy(
            title="Nothing to protect yet",
            body=(
                f"! The {state.unsaved_provider} key you typed isn't saved: it is "
                "saved together with the model you choose. Go back to Model to "
                "save it, or continue without it."
            ),
            path_line="",
            later_line=later,
            primary_label="",
            show_back_to_model=True,
        )
    if state.kind == PROTECT_ENV_ONLY:
        return ProtectCopy(
            title="Nothing to protect",
            body=(
                "Your keys come from environment variables "
                f"({', '.join(state.env_vars)}), so config.toml holds no key "
                "to encrypt."
            ),
            path_line="",
            later_line=later,
            primary_label="",
        )
    return ProtectCopy(
        title="Nothing to protect",
        body=(
            "No API key is saved in config.toml, so there is nothing to "
            "encrypt. Local servers usually don't need one."
        ),
        path_line="",
        later_line=later,
        primary_label="",
    )


def typed_provider_key(wizard_data: Mapping[str, object] | None) -> str:
    """The provider a key was typed for on the Provider step, or "".

    Reads the Provider step's committed data (``entered_key`` and
    ``provider_key``), which the container records on Next.
    """
    from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_PROVIDER

    if not isinstance(wizard_data, Mapping):
        return ""
    provider = wizard_data.get(STEP_PROVIDER)
    if not isinstance(provider, Mapping) or provider.get("entered_key") is not True:
        return ""
    key = provider.get("provider_key")
    return key.strip() if isinstance(key, str) else ""
