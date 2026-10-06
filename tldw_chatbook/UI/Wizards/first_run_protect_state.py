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
* ``plaintext`` -- a provider key is saved as plain text; name each provider
  and the file, and offer two explicit buttons.
* ``unsaved`` -- a key was typed this run but is not on disk; offer
  "Back to Model".
* ``env_only`` -- keys come only from environment variables; name them.
* ``nothing`` -- no key anywhere.

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

#: States where config.toml held no key to protect: the tracker shows '–'
#: for Protect instead of the ✓ that reads as "your keys are protected".
NOTHING_TO_PROTECT = frozenset({PROTECT_UNSAVED, PROTECT_ENV_ONLY, PROTECT_NOTHING})

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
CHANGED_STATUS = (
    "✓ Master password changed. Use the new one the next time chatbook starts."
)


@dataclass(frozen=True, slots=True)
class ProtectState:
    """One Protect state and the names its copy needs.

    Attributes:
        kind: One of the ``PROTECT_*`` constants.
        providers: Display names of the providers whose key is saved in
            config.toml (``plaintext``), in config order.
        env_vars: Environment variables that supply a key (``env_only``).
        unsaved_provider: Display name of the provider whose typed key is not
            on disk (``unsaved``).
    """

    kind: str
    providers: tuple[str, ...] = ()
    env_vars: tuple[str, ...] = ()
    unsaved_provider: str = ""

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
    from tldw_chatbook.Chat.provider_catalog import provider_display_name

    return provider_display_name(provider_key) or provider_key


def read_protect_state(
    config: Mapping[str, object],
    environ: Mapping[str, str],
    *,
    typed_provider_key: str = "",
) -> ProtectState:
    """Derive the Protect state from the persisted config.

    Args:
        config: The merged config read back from config.toml (decrypted with
            the session password when one is held).
        environ: Process environment, for keys supplied by env vars.
        typed_provider_key: The provider a key was typed for on the Provider
            step this run, or "" when none was typed.

    Returns:
        The state, with the provider names or env vars its copy names.
    """
    from tldw_chatbook.config import is_valid_provider_api_key

    encryption = config.get("encryption")
    if isinstance(encryption, Mapping) and _flag(encryption.get("enabled")):
        return ProtectState(PROTECT_ENCRYPTED)
    api_settings = config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        api_settings = {}
    stored: list[str] = []
    env_vars: list[str] = []
    for provider_key, settings in api_settings.items():
        if not isinstance(settings, Mapping):
            continue
        # is_valid_provider_api_key rejects blanks, template placeholders and
        # `enc:` ciphertext, so only a real plaintext key counts.
        if is_valid_provider_api_key(settings.get("api_key")):
            stored.append(_display_name(str(provider_key)))
            continue
        env_var = settings.get("api_key_env_var")
        if (
            isinstance(env_var, str)
            and env_var.strip()
            and is_valid_provider_api_key(environ.get(env_var.strip()))
        ):
            env_vars.append(env_var.strip())
    if stored:
        return ProtectState(PROTECT_PLAINTEXT, providers=tuple(dict.fromkeys(stored)))
    if typed_provider_key:
        return ProtectState(
            PROTECT_UNSAVED, unsaved_provider=_display_name(typed_provider_key)
        )
    if env_vars:
        return ProtectState(PROTECT_ENV_ONLY, env_vars=tuple(dict.fromkeys(env_vars)))
    return ProtectState(PROTECT_NOTHING)


def _join(names: tuple[str, ...]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def protect_copy(state: ProtectState, *, config_path: str) -> ProtectCopy:
    """The lines and buttons the step shows for ``state``.

    No line names a "Skip" control: the footer has Back, Next and Exit
    setup, and the plaintext offer has its own "Keep as plain text" button.

    Args:
        state: The derived Protect state.
        config_path: The config file path, for the dim path line.

    Returns:
        The step's copy and which buttons it shows.
    """
    card = encryption_card_path()
    path_line = f"Config file: {config_path}" if config_path else ""
    if state.kind == PROTECT_ENCRYPTED:
        return ProtectCopy(
            title="Your keys are protected",
            body=ENCRYPTED_STATUS,
            path_line=path_line,
            later_line=(
                f"Change the password or turn encryption off any time in {card}."
            ),
            primary_label=CHANGE_LABEL,
        )
    if state.kind == PROTECT_PLAINTEXT:
        many = len(state.providers) > 1
        names = _join(state.providers)
        return ProtectCopy(
            title="Protect your API keys" if many else "Protect your API key",
            body=(
                f"Your {names} {'keys are' if many else 'key is'} saved as plain "
                "text in config.toml, where any program running as you can read "
                f"{'them' if many else 'it'}. Encrypt {'them' if many else 'it'} "
                "with a master password, and chatbook will ask for that password "
                "each time it starts."
            ),
            path_line=path_line,
            later_line=f"You can change this later in {card}.",
            primary_label=ENCRYPT_LABEL,
            show_keep_plain=True,
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
