"""Console "New endpoint from template" creation modal (ADR-146, Task 6).

A small creation flow for ``[custom_endpoints.<slug>]`` entries: pick a
template (blank OpenAI-compatible, a built-in provider, or an existing
registry entry), adjust the prefilled form, and Create persists the entry
through the atomic config writer. The modal dismisses with the new
``custom-ep:<slug>`` provider id (``None`` on cancel) and posts
``EndpointCreated`` to the opener screen so it can switch selection without
re-reading config from disk.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import ClassVar, Literal

from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Chat.console_provider_endpoints import first_configured_endpoint
from tldw_chatbook.Chat.console_session_settings import (
    DEFAULT_LLAMACPP_BASE_URL,
    build_console_provider_options,
    normalize_llamacpp_base_url,
)
from tldw_chatbook.Chat.custom_endpoint_registry import (
    CUSTOM_ENDPOINT_ID_PREFIX,
    SLUG_PATTERN,
    CustomEndpointEntry,
    CustomEndpointSlugError,
    build_entry_mutation,
    derive_slug,
    load_custom_endpoints,
    split_custom_endpoint_id,
    validate_entry,
)
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.config import (
    AtomicConfigSnapshot,
    apply_settings_mutation_to_cli_config,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

MODAL_ID = "console-endpoint-template-modal"
TEMPLATE_PICKER_ID = "endpoint-template-picker"
NAME_INPUT_ID = "endpoint-template-name"
FAMILY_SELECT_ID = "endpoint-template-family"
URL_INPUT_ID = "endpoint-template-url"
MODELS_INPUT_ID = "endpoint-template-models"
CREATE_BUTTON_ID = "endpoint-template-create"
CANCEL_BUTTON_ID = "endpoint-template-cancel"
ERROR_STATIC_ID = "endpoint-template-error"

BLANK_TEMPLATE_LABEL = "OpenAI-compatible (blank)"
#: The three execution families a registry entry can run as (ADR-146);
#: values are the exact registry family strings ``validate_entry`` accepts.
FAMILY_SELECT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("llama.cpp", "llama_cpp"),
    ("OpenAI-compatible", "openai_compatible"),
    ("Ollama", "ollama"),
)
OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
MODELS_INPUT_PLACEHOLDER = "Comma-separated model ids"
#: H4: the llama prefill is the configured Chatbook default (:9099), while
#: stock llama-server listens on :8080 -- the placeholder explains both so
#: the prefill/placeholder pair stops contradicting itself.
DEFAULT_URL_PLACEHOLDER = "http://127.0.0.1:8080"
_LLAMA_URL_PLACEHOLDER = "llama-server default :8080 · Chatbook default :9099"
#: Display names for registry families, reused by the same-family starter
#: label (H5) so the picker names the family exactly like the Family select.
_FAMILY_DISPLAY_NAMES = {
    family: label for label, family in FAMILY_SELECT_OPTIONS
}
SAVE_FAILED_COPY = "Could not write the endpoint to the config file."
INVALID_SLUG_COPY = (
    "Enter a display name containing letters or numbers to derive an id."
)
MODAL_CONTROL_HEIGHT = 3
#: Bounded attempts to (re-)derive a slug when a concurrent create takes
#: the derived slug before this modal's create-only write commits.
_CREATE_DERIVE_ATTEMPTS = 3
#: Suffix appended to a duplicated entry's display name.
_DUPLICATE_NAME_SUFFIX = " (copy)"
#: Source-name budget so the duplicate prefill fits the registry's
#: 80-character display-name validation (``validate_entry``) without
#: manual editing for names already at the maximum.
_DUPLICATE_NAME_SOURCE_LIMIT = 80 - len(_DUPLICATE_NAME_SUFFIX)
_LLAMA_FAMILY_PROVIDER_KEYS = frozenset({"llama_cpp", "local_llamacpp"})
_OLLAMA_FAMILY_PROVIDER_KEYS = frozenset({"ollama", "local_ollama"})

#: Outcome of the create-only config write (see
#: :func:`_create_entry_only_if_absent`).
_CreateWriteOutcome = Literal["saved", "collision", "failed"]


@dataclass(frozen=True)
class _EndpointTemplate:
    """One template-picker choice: what the form prefills from.

    Attributes:
        label: Picker label (registry entries render as ``name (duplicate)``).
        provider_id: Template provider id recorded as ``created_from``
            (built-in key or ``custom-ep:<slug>``); None for the blank start.
        family: Registry family the template executes as.
        base_url: Prefill base URL (template's configured endpoint or the
            family default).
        models: Prefill model ids (template's configured models).
        duplicate_name: Display-name prefill for a registry-entry template
            (``<name> (copy)``); None for templates without a name to copy.
        api_key_env: Credential *reference* carried over when duplicating a
            registry entry -- the stored ``api_key`` is never copied (same
            security posture as the F9 convert path).
    """

    label: str
    provider_id: str | None
    family: str
    base_url: str
    models: tuple[str, ...] = ()
    duplicate_name: str | None = None
    api_key_env: str | None = None


def _family_for_provider_key(provider_key: str) -> str:
    """Map a built-in provider key onto its registry endpoint family."""
    if provider_key in _LLAMA_FAMILY_PROVIDER_KEYS:
        return "llama_cpp"
    if provider_key in _OLLAMA_FAMILY_PROVIDER_KEYS:
        return "ollama"
    return "openai_compatible"


def _family_default_base_url(family: str) -> str:
    """Return the family's stock base URL (blank for OpenAI-compatible)."""
    if family == "llama_cpp":
        return DEFAULT_LLAMACPP_BASE_URL
    if family == "ollama":
        return OLLAMA_DEFAULT_BASE_URL
    return ""


def _normalized_base_url(family: str, base_url: str) -> str:
    """Normalize a candidate URL per family, mirroring the registry loader."""
    raw = str(base_url or "").strip()
    if family == "llama_cpp":
        return normalize_llamacpp_base_url(raw)
    return raw.rstrip("/")


def _provider_settings(
    app_config: Mapping[str, object], provider_key: str
) -> Mapping[str, object]:
    """Return the ``api_settings`` table whose key normalizes to ``provider_key``.

    Mirrors ``ConsoleSettingsModal._provider_settings`` (aliasing tables such
    as ``OpenAI-Compatible`` resolve through ``provider_config_key``).
    """
    api_settings = app_config.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        return {}
    for configured_provider, configured_settings in api_settings.items():
        if (
            provider_config_key(str(configured_provider)) == provider_key
            and isinstance(configured_settings, Mapping)
        ):
            return configured_settings
    return {}


def _configured_models_for(
    providers_models: Mapping[str, list[str]], provider: str
) -> tuple[str, ...]:
    """Return the configured model list for ``provider`` (aliasing-aware)."""
    provider_key = provider_config_key(provider)
    for configured_provider, configured_models in providers_models.items():
        if provider_config_key(configured_provider) != provider_key:
            continue
        return tuple(
            model.strip()
            for model in configured_models
            if isinstance(model, str) and model.strip()
        )
    return ()


def _duplicate_display_name(source: str) -> str:
    """Build the duplicate prefill: ``<source> (copy)``, capped at 80 chars.

    The registry validates display names at 80 characters
    (``validate_entry``), so the source is truncated to reserve the
    seven-character suffix; sources that already fit with the suffix pass
    through unchanged apart from the suffix.

    Args:
        source: The duplicated entry's display name.

    Returns:
        The prefilled duplicate display name, always within the limit.
    """
    trimmed = source[:_DUPLICATE_NAME_SOURCE_LIMIT].rstrip()
    return f"{trimmed}{_DUPLICATE_NAME_SUFFIX}"


def _create_entry_only_if_absent(
    entry: CustomEndpointEntry,
) -> _CreateWriteOutcome:
    """Persist ``entry`` only when its slug's config section is absent.

    One :func:`apply_settings_mutation_to_cli_config` transaction holds the
    config writer lock while the locked-snapshot precondition checks the
    authoritative raw config and the mutation applies, so a same-process
    create that committed this slug after the modal's derivation aborts
    here as ``"collision"`` instead of being overwritten by the later
    write. The success/failure interpretation matches
    ``save_settings_to_cli_config``.

    Args:
        entry: The entry to persist (from ``build_entry_mutation``).

    Returns:
        ``"saved"`` when the section was created; ``"collision"`` when the
        section already exists under the lock; ``"failed"`` when the
        transaction itself failed.
    """

    def _slug_absent(snapshot: AtomicConfigSnapshot) -> bool:
        section = snapshot.values.get("custom_endpoints")
        return not (isinstance(section, Mapping) and entry.slug in section)

    result = apply_settings_mutation_to_cli_config(
        build_entry_mutation(entry),
        locked_snapshot_precondition=_slug_absent,
    )
    if result.conflict:
        return "collision"
    if result.failure_phase is None and not result.file_replaced:
        return "saved"
    return "saved" if result.fully_applied else "failed"


class ConsoleEndpointTemplateModal(
    SafeModalDismissMixin, ModalScreen[str | None]
):
    """Create a custom endpoint entry; dismisses with the new 'custom-ep:<slug>'
    provider id (or None on cancel)."""

    SAFE_MODAL_CONTENT = f"#{MODAL_ID}"
    BINDINGS: ClassVar = [("escape", "request_safe_cancel", "Cancel")]

    DEFAULT_CSS = f"""
    ConsoleEndpointTemplateModal {{
        align: center middle;
    }}

    ConsoleEndpointTemplateModal #{MODAL_ID} {{
        width: 76;
        max-width: 95%;
        height: auto;
        max-height: 90%;
        border: round $panel;
        background: $surface;
        padding: 1 2;
    }}

    ConsoleEndpointTemplateModal .console-settings-error {{
        background: $error 25%;
        color: $text-error;
        text-style: bold;
        border-left: thick $error;
        height: auto;
        min-height: 1;
        margin: 1 0;
        padding: 0 1;
    }}

    ConsoleEndpointTemplateModal #{TEMPLATE_PICKER_ID} {{
        height: auto;
        max-height: 8;
        margin: 0 0 1 0;
    }}

    ConsoleEndpointTemplateModal .console-endpoint-template-label {{
        width: 16;
        min-width: 16;
        max-width: 16;
        height: 1;
        min-height: 1;
        content-align: left middle;
    }}

    ConsoleEndpointTemplateModal .console-settings-modal-row {{
        height: auto;
        min-height: {MODAL_CONTROL_HEIGHT};
    }}

    ConsoleEndpointTemplateModal .console-settings-control {{
        width: 1fr;
        min-width: 0;
    }}

    ConsoleEndpointTemplateModal Input,
    ConsoleEndpointTemplateModal Select,
    ConsoleEndpointTemplateModal Button {{
        height: {MODAL_CONTROL_HEIGHT};
        min-height: {MODAL_CONTROL_HEIGHT};
    }}

    ConsoleEndpointTemplateModal .console-endpoint-template-actions {{
        height: auto;
        min-height: {MODAL_CONTROL_HEIGHT};
        align-horizontal: right;
    }}
    """

    def __init__(
        self,
        *,
        app_config: Mapping[str, object],
        providers_models: Mapping[str, list[str]],
        template_provider: str | None = None,
    ) -> None:
        """Initialize the template creation flow.

        Args:
            app_config: The full CLI config mapping (shared with the opener,
                so the persisted entry is mirrored back into it on Create).
            providers_models: Configured model lists keyed by provider.
            template_provider: Provider id whose template the picker starts
                on (built-in key, ``custom-ep:<slug>``, or None for blank).
        """
        super().__init__()
        self._app_config = app_config
        self._providers_models = providers_models
        self._template_provider = template_provider
        self._same_family_starter: _EndpointTemplate | None = None
        self._templates = self._build_templates()
        self._active_template_index = 0
        if self._same_family_starter is not None:
            # H5: opened from a registry entry -- start on the same-family
            # starter (a second server), with the entry's "(duplicate)" one
            # row below for a true copy.
            self._active_template_index = self._templates.index(
                self._same_family_starter
            )
        elif template_provider is not None:
            for index, template in enumerate(self._templates):
                if template.provider_id == template_provider:
                    self._active_template_index = index
                    break
        self._create_in_flight = False
        # H8 (calm dialog): Create is gated from the first frame, but the
        # error banner only renders once the user has actually touched the
        # form -- an untouched blank template is not an error to scold.
        self._form_touched = False
        self._untouched_form_values: tuple[str, str, str, str] | None = None

    class EndpointCreated(Message):
        """Posted to the opener screen after an entry was persisted."""

        def __init__(self, provider_id: str) -> None:
            super().__init__()
            self.provider_id = provider_id

    def _build_templates(self) -> list[_EndpointTemplate]:
        """Build the picker's template list: blank first, then providers.

        Built-in provider options come from the shared option builder
        (Cloud / Local / Custom & legacy order); their registry entries follow
        as ``<display name> (duplicate)`` duplicates of themselves. When the
        opener names a registry entry as ``template_provider`` (H5), a
        synthetic same-family starter is inserted immediately before that
        entry's duplicate option and becomes the active template.
        """
        templates = [
            _EndpointTemplate(
                label=BLANK_TEMPLATE_LABEL,
                provider_id=None,
                family="openai_compatible",
                base_url="",
                models=(),
            )
        ]
        entries = load_custom_endpoints(self._app_config)
        starter_slug = split_custom_endpoint_id(self._template_provider)
        for option in build_console_provider_options(
            self._providers_models, app_config=self._app_config
        ):
            if option.value.startswith(CUSTOM_ENDPOINT_ID_PREFIX):
                slug = option.value[len(CUSTOM_ENDPOINT_ID_PREFIX) :]
                entry = entries.get(slug)
                if entry is None:
                    continue
                if slug == starter_slug and self._same_family_starter is None:
                    # H5: adding "from" an existing entry usually means a
                    # second server of the same kind, so the starter preselect
                    # prefills only the family -- blank name and URL, never
                    # the family default (that is a different server's port)
                    # and never the entry's own URL/models.
                    self._same_family_starter = _EndpointTemplate(
                        label=(
                            "Same family "
                            f"({_FAMILY_DISPLAY_NAMES.get(entry.family, entry.family)})"
                            " — new URL"
                        ),
                        provider_id=None,
                        family=entry.family,
                        base_url="",
                        models=(),
                    )
                    templates.append(self._same_family_starter)
                templates.append(
                    _EndpointTemplate(
                        label=f"{entry.display_name} (duplicate)",
                        provider_id=option.value,
                        family=entry.family,
                        base_url=entry.base_url,
                        models=entry.models,
                        duplicate_name=_duplicate_display_name(entry.display_name),
                        api_key_env=entry.api_key_env,
                    )
                )
                continue
            provider_key = provider_config_key(option.value)
            family = _family_for_provider_key(provider_key)
            configured = first_configured_endpoint(
                _provider_settings(self._app_config, provider_key)
            )
            base_url = _normalized_base_url(
                family, configured or _family_default_base_url(family)
            )
            templates.append(
                _EndpointTemplate(
                    label=option.label,
                    provider_id=option.value,
                    family=family,
                    base_url=base_url,
                    models=_configured_models_for(
                        self._providers_models, option.value
                    ),
                )
            )
        return templates

    def compose(self) -> ComposeResult:
        """Build the template picker, prefilled form, and actions."""
        # Lazy import: this module is imported (module-level) by
        # console_settings_modal for the EndpointCreated wiring, so importing
        # ConsoleSettingsInput eagerly would cycle. By compose time that
        # module is fully loaded.
        from tldw_chatbook.Widgets.Console.console_settings_modal import (
            ConsoleSettingsInput,
        )

        template = self._templates[self._active_template_index]
        with Vertical(id=MODAL_ID):
            yield Static(
                "New endpoint from template", classes="console-modal-header"
            )
            yield Static("Template", classes="console-endpoint-template-label")
            # User-authored registry display names reach the prompt text, so
            # escape Rich markup the way the model options list does.
            picker = OptionList(
                *[
                    Option(escape_markup(candidate.label))
                    for candidate in self._templates
                ],
                id=TEMPLATE_PICKER_ID,
            )
            picker.highlighted = self._active_template_index
            yield picker
            with Horizontal(classes="console-settings-modal-row"):
                yield Static("Display name", classes="console-endpoint-template-label")
                yield ConsoleSettingsInput(
                    # A registry-entry template pre-fills "<name> (copy)" so
                    # the duplicate starts from the source's display name.
                    value=template.duplicate_name or "",
                    placeholder="Endpoint name",
                    id=NAME_INPUT_ID,
                    classes="console-settings-control",
                )
            with Horizontal(classes="console-settings-modal-row"):
                yield Static("Family", classes="console-endpoint-template-label")
                yield Select(
                    FAMILY_SELECT_OPTIONS,
                    value=template.family,
                    allow_blank=False,
                    id=FAMILY_SELECT_ID,
                    classes="console-settings-control",
                )
            with Horizontal(classes="console-settings-modal-row"):
                yield Static("Base URL", classes="console-endpoint-template-label")
                yield ConsoleSettingsInput(
                    value=template.base_url,
                    placeholder=self._url_placeholder_for_family(template.family),
                    id=URL_INPUT_ID,
                    classes="console-settings-control",
                )
            with Horizontal(classes="console-settings-modal-row"):
                yield Static("Models", classes="console-endpoint-template-label")
                yield ConsoleSettingsInput(
                    value=", ".join(template.models),
                    placeholder=MODELS_INPUT_PLACEHOLDER,
                    id=MODELS_INPUT_ID,
                    classes="console-settings-control",
                )
            yield Static(
                "",
                id=ERROR_STATIC_ID,
                classes="console-settings-error",
                markup=False,
            )
            with Horizontal(classes="console-endpoint-template-actions"):
                yield Button("Cancel", id=CANCEL_BUTTON_ID)
                yield Button("Create", id=CREATE_BUTTON_ID, variant="primary")

    def on_mount(self) -> None:
        """Validate the prefilled form so Create starts in a truthful state."""
        super().on_mount()
        # Reset the touch flag BEFORE validating: the family Select announces
        # its composed value as a Select.Changed around mount, and that
        # programmatic echo is not interaction. An echo landing after this
        # point is filtered by the value comparison in
        # ``_form_field_changed``.
        self._form_touched = False
        self._untouched_form_values = self._current_form_values()
        self._sync_validation()
        try:
            self.query_one(f"#{NAME_INPUT_ID}", Input).focus()
        except (NoMatches, QueryError):
            pass

    @on(OptionList.OptionSelected, f"#{TEMPLATE_PICKER_ID}")
    def _template_selected(self, event: OptionList.OptionSelected) -> None:
        """Apply the chosen template's prefill to the form."""
        event.stop()
        picker = self.query_one(f"#{TEMPLATE_PICKER_ID}", OptionList)
        index = picker.highlighted
        if index is None or not 0 <= index < len(self._templates):
            return
        self._active_template_index = index
        template = self._templates[index]
        self._form_touched = True
        if template.duplicate_name is not None:
            # Duplicating carries the source display name (suffixed
            # "(copy)"); templates without a name leave a typed name alone.
            self.query_one(f"#{NAME_INPUT_ID}", Input).value = template.duplicate_name
        self.query_one(f"#{FAMILY_SELECT_ID}", Select).value = template.family
        self.query_one(f"#{URL_INPUT_ID}", Input).value = template.base_url
        self.query_one(f"#{MODELS_INPUT_ID}", Input).value = ", ".join(
            template.models
        )
        self._sync_validation()

    @on(Input.Changed)
    @on(Select.Changed)
    def _form_field_changed(self, _event) -> None:
        """Re-validate on every edit so Create reflects the current draft."""
        if (
            not self._form_touched
            and self._current_form_values() == self._untouched_form_values
        ):
            # The composed prefill re-announcing itself (the family Select's
            # mount echo): not a user edit, so the banner stays calm.
            self._sync_validation()
            return
        self._form_touched = True
        # Keep the URL placeholder aligned with the chosen family (H4): the
        # explanatory llama copy only applies while the llama family is (or
        # becomes) active.
        self._sync_url_placeholder()
        self._sync_validation()

    def _sync_url_placeholder(self) -> None:
        """Render the family-appropriate URL placeholder (presentation only)."""
        try:
            url_input = self.query_one(f"#{URL_INPUT_ID}", Input)
        except (NoMatches, QueryError):
            return
        url_input.placeholder = self._url_placeholder_for_family(self._family_value())

    @staticmethod
    def _url_placeholder_for_family(family: str) -> str:
        """Return the URL-input placeholder explaining the family's defaults."""
        if family == "llama_cpp":
            return _LLAMA_URL_PLACEHOLDER
        return DEFAULT_URL_PLACEHOLDER

    def _current_form_values(self) -> tuple[str, str, str, str]:
        """Return the four editable form values (used for echo filtering)."""
        return (
            self.query_one(f"#{NAME_INPUT_ID}", Input).value,
            self._family_value(),
            self.query_one(f"#{URL_INPUT_ID}", Input).value,
            self.query_one(f"#{MODELS_INPUT_ID}", Input).value,
        )

    @on(Button.Pressed, f"#{CANCEL_BUTTON_ID}")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, f"#{CREATE_BUTTON_ID}")
    async def _create(self, event: Button.Pressed) -> None:
        """Persist the validated entry, announce it, and dismiss with its id.

        Registry loading, slug derivation, and the write all run off the
        event loop via ``asyncio.to_thread``. The write is a create-only
        transaction sharing the config writer lock
        (:func:`_create_entry_only_if_absent`): when a concurrent
        same-process create already committed the derived slug, the
        mutation aborts with a collision and the slug is re-derived
        avoiding the collided slug (bounded to
        :data:`_CREATE_DERIVE_ATTEMPTS`). Exhaustion surfaces the same
        inline name-in-use error as ``derive_slug`` itself, not the
        generic save-failure copy.
        """
        event.stop()
        if self._create_in_flight:
            return
        # A Create press is interaction: any validation failure it surfaces
        # belongs in the banner even on an otherwise-untouched form.
        self._form_touched = True
        name = self.query_one(f"#{NAME_INPUT_ID}", Input).value.strip()
        family = self._family_value()
        base_url = self.query_one(f"#{URL_INPUT_ID}", Input).value
        errors = validate_entry(name, family, base_url)
        if errors:
            self._show_errors(errors)
            return
        self._create_in_flight = True
        create_button = self.query_one(f"#{CREATE_BUTTON_ID}", Button)
        create_button.disabled = True
        entry: CustomEndpointEntry | None = None
        error_copy: str | None = None
        saved = False
        collided = False
        # Slugs whose create-only writes collided under the config writer
        # lock; each retry derives around them so a stale in-memory
        # registry cannot re-derive the same occupied slug forever.
        collided_slugs: set[str] = set()
        try:
            for _attempt in range(_CREATE_DERIVE_ATTEMPTS):
                try:
                    slug = await asyncio.to_thread(
                        self._derive_fresh_slug, name, frozenset(collided_slugs)
                    )
                except CustomEndpointSlugError as error:
                    # Every derivable slug is taken: surface the collision
                    # inline instead of persisting an entry that would
                    # overwrite an existing slug's config section.
                    error_copy = str(error)
                    break
                if not SLUG_PATTERN.fullmatch(slug):
                    # A name with no alphanumeric characters derives an empty slug.
                    error_copy = INVALID_SLUG_COPY
                    break
                entry = CustomEndpointEntry(
                    slug=slug,
                    display_name=name,
                    family=family,
                    base_url=_normalized_base_url(family, base_url),
                    # Duplicating a registry entry carries the credential
                    # reference only; a stored api_key is never copied.
                    api_key_env=self._templates[
                        self._active_template_index
                    ].api_key_env,
                    models=self._parsed_models(),
                    created_from=self._templates[self._active_template_index].provider_id,
                )
                try:
                    outcome = await asyncio.to_thread(
                        _create_entry_only_if_absent, entry
                    )
                except Exception:
                    outcome = "failed"
                if outcome == "collision":
                    # A concurrent create committed this slug while holding
                    # the writer lock: re-derive around it on the next
                    # bounded attempt (or surface the in-use error).
                    collided = True
                    collided_slugs.add(slug)
                    continue
                collided = False
                saved = outcome == "saved"
                break
        finally:
            self._create_in_flight = False
        if not saved:
            create_button.disabled = False
            if entry is None or collided:
                # Derivation never survived to a write (a slug error or an
                # underivable name) or every bounded attempt collided with
                # a concurrent create: the name-in-use copy is the truthful
                # error, not a generic save failure.
                self._show_errors([error_copy or str(CustomEndpointSlugError())])
            else:
                self._show_errors([SAVE_FAILED_COPY])
            return
        self._mirror_entry_into_app_config(entry)
        provider_id = f"{CUSTOM_ENDPOINT_ID_PREFIX}{entry.slug}"
        self._announce_created(provider_id)
        self.dismiss(provider_id)

    def _derive_fresh_slug(self, name: str, avoid: frozenset[str]) -> str:
        """Derive a slug against the current in-memory registry plus ``avoid``.

        Runs in a worker thread via ``asyncio.to_thread`` from ``_create``:
        the registry load (Pydantic validation per entry) and the bounded
        suffix search must never block the UI event loop, even for a
        registry with thousands of occupied slugs.

        Args:
            name: Display name typed into the form.
            avoid: Extra slugs to steer derivation around (collided
                create-only writes from earlier attempts).

        Returns:
            The derived slug, unique against the loaded registry + ``avoid``.

        Raises:
            CustomEndpointSlugError: Every candidate collides.
        """
        existing = load_custom_endpoints(self._app_config).keys()
        return derive_slug(name, set(existing) | set(avoid))

    def _family_value(self) -> str:
        """Return the selected family string (one of the three fixed ones)."""
        return str(self.query_one(f"#{FAMILY_SELECT_ID}", Select).value)

    def _parsed_models(self) -> tuple[str, ...]:
        """Parse the comma-separated models input, dropping blanks/dupes."""
        raw = self.query_one(f"#{MODELS_INPUT_ID}", Input).value
        parsed: list[str] = []
        for part in raw.split(","):
            model_id = part.strip()
            if model_id and model_id not in parsed:
                parsed.append(model_id)
        return tuple(parsed)

    def _sync_validation(self) -> None:
        """Show inline errors and gate Create on ``validate_entry``."""
        errors = validate_entry(
            self.query_one(f"#{NAME_INPUT_ID}", Input).value.strip(),
            self._family_value(),
            self.query_one(f"#{URL_INPUT_ID}", Input).value,
        )
        self._show_errors(errors)
        try:
            create = self.query_one(f"#{CREATE_BUTTON_ID}", Button)
        except (NoMatches, QueryError):
            return
        create.disabled = bool(errors) or self._create_in_flight

    def _show_errors(self, errors: list[str]) -> None:
        """Render the inline error banner, hiding it when there is no error.

        The banner also stays hidden while the form is untouched (H8): the
        prefilled/blank state at mount is not user error. ``_sync_validation``
        still gates Create from the first frame, so nothing invalid is
        submittable while the banner is calm.
        """
        try:
            error = self.query_one(f"#{ERROR_STATIC_ID}", Static)
        except (NoMatches, QueryError):
            return
        error.update(" ".join(errors))
        error.display = bool(errors) and self._form_touched

    def _mirror_entry_into_app_config(self, entry: CustomEndpointEntry) -> None:
        """Mirror the persisted entry into the shared in-memory app_config.

        The disk file is the source of truth (written by the atomic mutator);
        this keeps the opener's in-memory option builders -- which share this
        exact mapping -- able to surface the new provider without a restart.
        Read-only configs simply skip the mirror.
        """
        if not isinstance(self._app_config, MutableMapping):
            return
        raw_section = self._app_config.get("custom_endpoints")
        section = raw_section if isinstance(raw_section, dict) else {}
        values: dict[str, object] = {
            "display_name": entry.display_name,
            "family": entry.family,
            "base_url": entry.base_url,
            "models": list(entry.models),
        }
        if entry.api_key_env is not None:
            values["api_key_env"] = entry.api_key_env
        if entry.created_from is not None:
            values["created_from"] = entry.created_from
        section[entry.slug] = values
        if not isinstance(raw_section, dict):
            self._app_config["custom_endpoints"] = section

    def _announce_created(self, provider_id: str) -> None:
        """Deliver ``EndpointCreated`` to the opener screen.

        A message posted on this modal bubbles to the App, not sideways to
        the suspended opener, so it is posted to the opener directly (and
        still bubbles from there when nothing consumes it).
        """
        message = self.EndpointCreated(provider_id)
        screen_stack = self.app.screen_stack
        opener = screen_stack[-2] if len(screen_stack) >= 2 else None
        if opener is not None:
            opener.post_message(message)
        else:
            self.post_message(message)
