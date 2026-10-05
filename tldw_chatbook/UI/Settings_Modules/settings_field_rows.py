"""Settings' field rows: label | one-row control | Source word | help (TASK-33007.5).

Spec §6 gives every editor the same row. Labels and help lines come from the
shared field table (``MODEL_CONFIG_FIELDS``); Source words come from the
resolver Chat settings uses (``resolve_console_value_layers`` mapped through
``CONSOLE_VALUE_SOURCE_WORDS``); the core and Sampling orders and the
hidden-field line are Chat settings' own (``console_settings_field_row``), so
both surfaces name the same hidden fields in the same words.

Model defaults uses the rows first: it edits the default model's
``model_defaults`` profile, so a blank field deletes that one override and
says what it inherits instead ("inherits 1.0 · Console Behavior"); a
placeholder only states a range or unit.

Console Behavior's global fallbacks (``chat_defaults``) use the same rows and
orders (TASK-33007.7): a value ``[chat_defaults]`` holds says "Console
Behavior", one it does not hold says "built-in", and streaming is an On/Off
Select from the model default's Inherit/On/Off family.

The card module imports this one, and ``settings_screen`` imports both inside
the functions that use them, so the Settings route's pre-import payload does
not grow (ADR-097).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from rich.cells import cell_len
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.css.query import QueryError
from textual.widget import Widget
from textual.widgets import Collapsible, Input, Select, Static

from ...Chat.console_provider_support import MODEL_CONFIG_FIELDS, MODEL_FIELD_LABELS
from ...Chat.console_session_settings import (
    CONSOLE_VALUE_SOURCE_WORDS,
    ConsoleValueLayer,
    build_default_console_session_settings,
    chat_defaults_held_fields,
    resolve_console_value_layers,
)
from ...config import coerce_bool_setting
from ...Widgets.Console.console_settings_field_row import (
    BLANK_CHOICE_PROMPT,
    BLANK_FIELD_HELP,
    CORE_FIELDS,
    REQUIRED_FIELDS,
    SAMPLING_FIELDS,
    SAMPLING_TITLE,
    hidden_fields_line,
    hidden_fields_list,
)
from ..Screens.settings_config_models import SettingsCategoryId
from ..Screens.settings_screen import (
    CLOSED_ENUM_SELECT_OPTIONS,
    MODEL_PROFILE_INPUT_PLACEHOLDERS,
    MODEL_PROFILE_SELECT_FIELD_KEYS,
    MODEL_PROFILE_STREAMING_SELECT_OPTIONS,
)

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from ..Screens.settings_screen import SettingsScreen

#: Model defaults keeps the id tests, '/' search and the registry lock know.
MODEL_DEFAULTS_ID = "settings-generation-defaults"
MODEL_DEFAULTS_TITLE = "Model defaults"
SAMPLING_ID = "settings-model-sampling"
#: Inside the opened Sampling disclosure: every hidden field, by its label.
#: The id is the one the old one-line summary had, so its readers still find
#: the names.
HIDDEN_LIST_ID = "settings-provider-generation-support"
#: Model defaults' rows: core first, then the Sampling disclosure's.
MODEL_DEFAULT_FIELDS = CORE_FIELDS + SAMPLING_FIELDS
#: A blank Select row inherits (ADR-095:75-81 "Inherit / On / Off").
INHERIT_PROMPT = "Inherit"
#: The Sampling title's one-row budget before the card is laid out: its
#: 117-cell title row at 211x44 less the "▶ " and padding. Once laid out, the
#: measured width replaces it (131 cells at 235x52).
SAMPLING_TITLE_CELLS = 113
_TITLE_CHROME_CELLS = 4
_INTEGER_FIELDS = frozenset({"top_k", "max_tokens", "seed", "thinking_budget_tokens"})
_EDITED = CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.EDITED_DRAFT]
_MODEL_DEFAULT = CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.MODEL_DEFAULT]
_PROVIDER = CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.PROVIDER_SCALARS]
_CONSOLE_BEHAVIOR = CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.CHAT_DEFAULTS]
_BUILT_IN = CONSOLE_VALUE_SOURCE_WORDS[ConsoleValueLayer.BUILT_IN]
#: Console Behavior's global fallbacks (TASK-33007.7): their rows' container
#: and their closed Sampling disclosure.
CONSOLE_FALLBACKS_ID = "settings-console-fallbacks"
CONSOLE_SAMPLING_ID = "settings-console-sampling"
#: A fallback ``[chat_defaults]`` does not hold shows tldw's own value, which
#: a provider's table outranks (the default chain's order).
UNSET_FALLBACK_HELP = "not set here · a provider's own setting comes first"
#: The global fallbacks' Sampling title when no sampler holds a value;
#: nothing inherits on this surface, so it is not Model defaults' word.
NO_SAMPLER_SET = "none set"


def draft_key(name: str) -> str:
    """Return a field's Providers & Models draft key.

    Args:
        name: A field-table name, e.g. ``"max_tokens"``.

    Returns:
        E.g. ``"model_profile_max_tokens"``.
    """
    return f"model_profile_{name}"


def control_id(name: str) -> str:
    """Return the widget id of a Model defaults control.

    Args:
        name: A field-table name, e.g. ``"top_p"``.

    Returns:
        E.g. ``"settings-model-profile-top-p"``.
    """
    return "settings-model-profile-" + name.replace("_", "-")


def field_row(label: str, control: Widget, *, row_id: str, classes: str) -> Horizontal:
    """Build one row: label, control, Source word and help line (spec §6).

    The Source word and help Statics are empty until the owner's refresh
    fills them; their ids are the control's id plus ``-source`` / ``-help``.

    Args:
        label: The field table's label.
        control: The row's one-row Input or Select, which carries an id.
        row_id: The row's id.
        classes: The row's classes (input row, select row, hidden).

    Returns:
        The row.
    """
    return Horizontal(
        Static(label, classes="settings-input-label"),
        control,
        Static(
            "", id=f"{control.id}-source", classes="settings-source-word", markup=False
        ),
        Static("", id=f"{control.id}-help", classes="settings-row-help", markup=False),
        id=row_id,
        classes=classes,
    )


def format_value(value: object) -> str:
    """Spell a resolved value the way its row shows it.

    Args:
        value: A resolved setting value.

    Returns:
        "On" / "Off" for a boolean, else the value as text.
    """
    if isinstance(value, bool):
        return "On" if value else "Off"
    return str(value)


def inherited_values(
    app_config: Mapping[str, object], provider: str, model: str
) -> dict[str, tuple[object, str]]:
    """Resolve what each blank model default would inherit, and from where.

    The model's own profile is left out, so the answer is the next layer
    down: Console Behavior's fallbacks, the provider's settings or the
    built-in value. A field no layer sets sends nothing, so the provider
    decides ("provider").

    Args:
        app_config: The saved configuration.
        provider: The provider the card holds.
        model: The default model the card holds.

    Returns:
        ``{name: (value, Source word)}``; the value is ``None`` when the
        provider decides.
    """
    names = MODEL_DEFAULT_FIELDS
    skipped = frozenset(names)
    layers = resolve_console_value_layers(
        app_config, provider, model, names, excluded_model_profile_fields=skipped
    )
    values = build_default_console_session_settings(
        app_config, provider, model, excluded_model_profile_fields=skipped
    )
    resolved: dict[str, tuple[object, str]] = {}
    for name in names:
        value = getattr(values, name)
        word = _PROVIDER if value is None else CONSOLE_VALUE_SOURCE_WORDS[layers[name]]
        resolved[name] = (value, word)
    return resolved


def row_copy(
    name: str,
    shown: str,
    edited: bool,
    inherited: tuple[object, str],
    *,
    saved_word: str = _MODEL_DEFAULT,
) -> tuple[str, str]:
    """Return one row's Source word and help line.

    Args:
        name: The row's field-table name.
        shown: The control's value as text; blank when it holds none.
        edited: Whether the draft changed this field.
        inherited: ``(value, Source word)`` from ``inherited_values``.
        saved_word: The layer a set field comes from: "model default" on
            Model defaults, "Console Behavior" on the global fallbacks.

    Returns:
        A set field: ``saved_word`` (or "edited *") and the field's help.
        A blank one: the layer it inherits from (or "edited *") and
        "inherits <value> · <layer>", or "blank = provider default".
    """
    if shown:
        return (_EDITED if edited else saved_word), MODEL_CONFIG_FIELDS[name].help
    value, word = inherited
    help_line = (
        BLANK_FIELD_HELP
        if value is None
        else f"inherits {format_value(value)} · {word}"
    )
    return (_EDITED if edited else word), help_line


def control_text(control: Widget) -> str:
    """Return a row control's value as text, blank when it holds none.

    Args:
        control: The row's Input or Select.

    Returns:
        The text, e.g. ``"0.7"`` or ``"true"``; ``""`` for a blank control.
    """
    if isinstance(control, Select):
        return "" if control.value is Select.NULL else str(control.value)
    if isinstance(control, Input):
        return control.value.strip()
    return ""


def model_defaults_title(screen: SettingsScreen, provider: str, model: str) -> Content:
    """Name the provider·model pair Model defaults edits.

    Args:
        screen: The Settings screen (it owns the display names).
        provider: The provider the card holds.
        model: The default model the card holds.

    Returns:
        A literal title, e.g. "Model defaults · Anthropic · claude-sonnet-4-5";
        a registry name is user text, never markup.
    """
    from .providers_models_card import provider_model_pair

    return Content(
        f"{MODEL_DEFAULTS_TITLE} · {provider_model_pair(screen, provider, model)}"
    )


def hidden_model_default_fields(
    screen: SettingsScreen, provider: str, model: str
) -> tuple[str, ...]:
    """Name the Model defaults fields the provider·model request does not carry.

    Args:
        screen: The Settings screen (it owns the field-support decision).
        provider: The provider the card holds.
        model: The default model the card holds.

    Returns:
        Field-table names in the Sampling line's order (Sampling fields,
        then core), as Chat settings orders them.
    """
    return tuple(
        name
        for name in SAMPLING_FIELDS + CORE_FIELDS
        if not screen._model_profile_field_supported(provider, draft_key(name), model)
    )


def sampling_state(
    values: Mapping[str, str], *, cells: int | None = None, unset: str = "all inherit"
) -> str:
    """Summarise the shown Sampling fields for the closed title.

    Args:
        values: ``{name: control text}`` for the Sampling rows the provider
            accepts.
        cells: The title's one-row budget. Given, every set field is named
            while "Sampling · <names>" fits it; omitted, at most two are
            (Model defaults' title also has hidden fields to name).
        unset: The word for no set field.

    Returns:
        "Top P 0.95 · Seed 7" (the set fields), "3 set" when they do not
        fit, ``unset`` when none is set, or ``""`` when the provider accepts
        none of them.
    """
    if not values:
        return ""
    chosen = [
        f"{MODEL_FIELD_LABELS[name]} {text}" for name, text in values.items() if text
    ]
    if not chosen:
        return unset
    named = " · ".join(chosen)
    fits = (
        len(chosen) <= 2
        if cells is None
        else cell_len(f"{SAMPLING_TITLE} · {named}") <= cells
    )
    return named if fits else f"{len(chosen)} set"


class ModelDefaultsDisclosure(Collapsible):
    """The Model defaults disclosure, which says its rows' sources once laid out.

    A card rebuild can mount it after the screen's own refresh has passed,
    so it fills its Source words, help lines and Sampling title itself: on
    mount, again after the first layout, and on a resize, when the Sampling
    title's one-row width changes. It re-says them for the pair the last
    refresh named (``refresh_model_defaults`` keeps it current), never for a
    pair read back from widgets mid-change.
    """

    def __init__(
        self, *children: Widget, pair: tuple[str, str], **kwargs: object
    ) -> None:
        """Keep the provider·model pair the rows describe.

        Args:
            *children: The disclosure's rows.
            pair: ``(provider, model)`` at compose time.
            **kwargs: ``Collapsible`` options (title, collapsed, id, ...).
        """
        super().__init__(*children, **kwargs)
        self.pair = pair

    def _refresh_rows(self) -> None:
        """Re-say the rows for the pair they describe."""
        refresh_model_defaults(self.screen, *self.pair)

    def on_mount(self) -> None:
        """Say the rows' sources now and once the layout is known."""
        self._refresh_rows()
        self.call_after_refresh(self._refresh_rows)

    def on_resize(self) -> None:
        """Re-fit the Sampling title to the new width."""
        self._refresh_rows()


def _model_default_control(
    screen: SettingsScreen,
    provider: str,
    model: str,
    name: str,
    values: Mapping[str, object],
    supported: bool,
) -> Widget:
    """Build one Model defaults control from the staged values.

    Args:
        screen: The Settings screen that owns the draft.
        provider: The provider the card holds.
        model: The default model the card holds.
        name: The field-table name.
        values: The card's staged values, keyed by draft key.
        supported: Whether the provider·model request carries the field.

    Returns:
        A one-row Select (streaming and the closed enums) or Input.
    """
    key = draft_key(name)
    if name == "streaming":
        return Select(
            list(MODEL_PROFILE_STREAMING_SELECT_OPTIONS),
            value=screen._streaming_select_value(values[key]),
            id=control_id(name),
            classes="settings-compact-select",
            allow_blank=True,
            prompt=INHERIT_PROMPT,
            compact=True,
            disabled=not supported,
        )
    if key in MODEL_PROFILE_SELECT_FIELD_KEYS:
        return screen._model_profile_enum_select(provider, key, dict(values))
    value = (
        screen._model_profile_input_value(provider, key, model, values[key])
        if name == "thinking_budget_tokens"
        else screen._profile_input_value(values[key])
    )
    return Input(
        value=value,
        id=control_id(name),
        classes="settings-compact-input",
        placeholder=MODEL_PROFILE_INPUT_PLACEHOLDERS[key],
        restrict=r"^[0-9]*$" if name in _INTEGER_FIELDS else None,
        disabled=not supported,
    )


def compose_model_defaults(
    screen: SettingsScreen,
    provider: str,
    model: str,
    values: Mapping[str, object],
    *,
    registry_locked: bool,
) -> ComposeResult:
    """Compose Model defaults: core rows, then the closed Sampling disclosure.

    Open by default and titled with the pair it edits. A row the provider·model
    request does not carry is hidden and disabled (never a focus stop) and
    named in the Sampling title (TASK-33001.2, spec §4 rule 2).

    Args:
        screen: The Settings screen that owns the draft and the handlers.
        provider: The provider the card holds.
        model: The default model the card holds.
        values: The card's staged values, keyed by draft key.
        registry_locked: A registry default is edited in Custom endpoints.

    Yields:
        The Model defaults disclosure.
    """
    supported = {
        name: screen._model_profile_field_supported(provider, draft_key(name), model)
        for name in MODEL_DEFAULT_FIELDS
    }

    def row(name: str) -> Horizontal:
        control = _model_default_control(
            screen, provider, model, name, values, supported[name]
        )
        classes = screen._gated_profile_row_classes(supported[name])
        if isinstance(control, Select):
            classes += " settings-select-row"
        return field_row(
            MODEL_FIELD_LABELS[name],
            control,
            row_id=f"{control_id(name)}-row",
            classes=classes,
        )

    with ModelDefaultsDisclosure(
        title=model_defaults_title(screen, provider, model),
        collapsed=screen._generation_defaults_collapsed,
        id=MODEL_DEFAULTS_ID,
        disabled=registry_locked,
        pair=(provider, model),
    ):
        for name in CORE_FIELDS:
            yield row(name)
        with Collapsible(
            title=Content(hidden_fields_line("", (), state="")),
            collapsed=screen._sampling_defaults_collapsed,
            id=SAMPLING_ID,
        ):
            hidden_list = Static(
                "", id=HIDDEN_LIST_ID, classes="settings-detail-row", markup=False
            )
            hidden_list.set_class(True, "settings-gated-profile-hidden")
            yield hidden_list
            for name in SAMPLING_FIELDS:
                yield row(name)


def refresh_model_defaults(screen: SettingsScreen, provider: str, model: str) -> None:
    """Re-say Model defaults: its title, every Source word and help, and Sampling.

    Runs after any field edit, a provider or model change and the first
    layout. Rows are already shown or hidden by the screen's support sync.

    Args:
        screen: The Settings screen that owns the card.
        provider: The provider the card holds.
        model: The default model the card holds.
    """
    try:
        defaults = screen.query_one(f"#{MODEL_DEFAULTS_ID}", Collapsible)
        sampling = screen.query_one(f"#{SAMPLING_ID}", Collapsible)
        hidden_list = screen.query_one(f"#{HIDDEN_LIST_ID}", Static)
    except QueryError:
        return
    if isinstance(defaults, ModelDefaultsDisclosure):
        defaults.pair = (provider, model)
    defaults.title = model_defaults_title(screen, provider, model)
    draft = screen._provider_draft()
    dirty = draft.dirty_keys if draft is not None else frozenset()
    inherited = inherited_values(screen._app_config_mapping(), provider, model)
    hidden = hidden_model_default_fields(screen, provider, model)
    shown_sampling: dict[str, str] = {}
    for name in SAMPLING_FIELDS + CORE_FIELDS:
        key = draft_key(name)
        if name in hidden:
            continue
        try:
            control = screen.query_one(f"#{control_id(name)}")
        except QueryError:
            continue
        text = control_text(control)
        if name in SAMPLING_FIELDS:
            shown_sampling[name] = text
        word, help_line = row_copy(name, text, key in dirty, inherited[name])
        screen._set_static_text(f"#{control_id(name)}-source", word)
        screen._set_static_text(f"#{control_id(name)}-help", help_line)
    display = screen._provider_display_label(provider) or "this provider"
    width = sampling.size.width - _TITLE_CHROME_CELLS
    sampling.title = Content(
        hidden_fields_line(
            display,
            hidden,
            cells=width if width > 0 else SAMPLING_TITLE_CELLS,
            state=sampling_state(shown_sampling),
        )
    )
    hidden_list.update(hidden_fields_list(display, hidden))
    hidden_list.set_class(not hidden, "settings-gated-profile-hidden")


def console_fallback_id(name: str) -> str:
    """Return the widget id of a Console Behavior global fallback control.

    Args:
        name: A field-table name, e.g. ``"top_p"``.

    Returns:
        E.g. ``"settings-console-default-top-p"``.
    """
    return "settings-console-default-" + name.replace("_", "-")


def _console_fallback_control(screen: SettingsScreen, name: str) -> Widget:
    """Build one global fallback control from its saved-or-draft value.

    Args:
        screen: The Settings screen that owns the Console Behavior draft.
        name: The field-table name.

    Returns:
        Streaming's On/Off Select (the model default's family less Inherit:
        a global fallback has nothing to inherit from), a closed-enum Select,
        or a one-row Input.
    """
    value = screen._console_behavior_value(name)
    if name == "streaming":
        return Select(
            list(MODEL_PROFILE_STREAMING_SELECT_OPTIONS),
            value="true" if coerce_bool_setting(value, True) else "false",
            id=console_fallback_id(name),
            classes="settings-compact-select",
            allow_blank=False,
            compact=True,
        )
    if draft_key(name) in MODEL_PROFILE_SELECT_FIELD_KEYS:
        options = CLOSED_ENUM_SELECT_OPTIONS[name]
        return Select(
            [(option, option) for option in options],
            value=screen._select_option_value(value, options),
            id=console_fallback_id(name),
            classes="settings-compact-select",
            allow_blank=True,
            # Chat settings' word for the same state: nothing is sent.
            prompt=BLANK_CHOICE_PROMPT,
            compact=True,
        )
    return Input(
        value=screen._console_input_value(value),
        id=console_fallback_id(name),
        classes="settings-compact-input",
        placeholder=MODEL_PROFILE_INPUT_PLACEHOLDERS[draft_key(name)],
        restrict=r"^[0-9]*$" if name in _INTEGER_FIELDS else None,
    )


class ConsoleFallbacks(Vertical):
    """Console Behavior's global fallback rows; they say their sources once mounted."""

    def on_mount(self) -> None:
        """Fill the Source words, help lines and the Sampling title."""
        refresh_console_fallbacks(self.screen)

    def on_resize(self) -> None:
        """Re-fit the Sampling title once the width is known, and when it changes."""
        refresh_console_fallbacks(self.screen)


def compose_console_fallbacks(screen: SettingsScreen) -> ComposeResult:
    """Compose the global fallbacks with Model defaults' rows (TASK-33007.7).

    Core rows first (the reasoning note heads the reasoning and thinking
    rows), then the six samplers in one closed one-row Sampling disclosure.
    The fallbacks apply to every provider, so no row is hidden.

    Args:
        screen: The Settings screen that owns the Console Behavior draft.

    Yields:
        The fallback rows' container.
    """

    def row(name: str) -> Horizontal:
        control = _console_fallback_control(screen, name)
        classes = "settings-input-row"
        if isinstance(control, Select):
            classes += " settings-select-row"
        return field_row(
            MODEL_FIELD_LABELS[name],
            control,
            row_id=f"{control.id}-row",
            classes=classes,
        )

    with ConsoleFallbacks(id=CONSOLE_FALLBACKS_ID):
        for name in CORE_FIELDS:
            yield row(name)
            if name == "streaming":
                # Heads the reasoning and thinking rows that follow.
                yield Static(
                    "Reasoning and thinking controls are sent only to providers "
                    "that support them.",
                    id="settings-console-reasoning-help",
                    classes="settings-detail-row",
                )
        with Collapsible(
            title=Content(hidden_fields_line("", ())),
            collapsed=True,
            id=CONSOLE_SAMPLING_ID,
        ):
            for name in SAMPLING_FIELDS:
                yield row(name)


def refresh_console_fallbacks(screen: SettingsScreen) -> None:
    """Re-say the global fallbacks' Source words, help lines and Sampling title.

    Runs once mounted, on a resize and after every Console Behavior draft
    change, save or revert. An edited row reads "edited *". Otherwise a value
    ``[chat_defaults]`` holds reads "Console Behavior"; one it does not hold
    (tldw's own Streaming, Temperature or Top P) reads "built-in" and says a
    provider's own setting comes first, as the default chain orders them; a
    blank optional row leaves the choice to the provider. A cleared
    Temperature or Top P says it is required, because Save refuses a blank.
    The Sampling title names only samplers ``[chat_defaults]`` holds or the
    draft edited, never a built-in value shown at rest.

    Args:
        screen: The Settings screen that owns the card.
    """
    try:
        fallbacks = screen.query_one(f"#{CONSOLE_FALLBACKS_ID}")
        sampling = fallbacks.query_one(f"#{CONSOLE_SAMPLING_ID}", Collapsible)
    except QueryError:
        return
    draft = screen._settings_drafts.get(SettingsCategoryId.CONSOLE_BEHAVIOR)
    dirty = draft.dirty_keys if draft is not None else frozenset()
    held = chat_defaults_held_fields(screen._app_config_mapping(), MODEL_DEFAULT_FIELDS)
    names = {console_fallback_id(name): name for name in MODEL_DEFAULT_FIELDS}
    shown_sampling: dict[str, str] = {}
    for row in fallbacks.query(".settings-input-row"):
        _label, control, source, help_line = row.children
        name = names[str(control.id)]
        text = control_text(control)
        if name in SAMPLING_FIELDS:
            # A built-in Top P is shown, not set: the title leaves it out.
            shown_sampling[name] = text if name in held or name in dirty else ""
        word, help_text = row_copy(
            name,
            text,
            name in dirty,
            (None, _PROVIDER),
            saved_word=_CONSOLE_BEHAVIOR if name in held else _BUILT_IN,
        )
        if not text and name in REQUIRED_FIELDS:
            # Chat settings' copy for the same state, with the table's range.
            help_text = f"Required: {MODEL_CONFIG_FIELDS[name].valid_range}."
        elif word == _BUILT_IN:
            help_text = UNSET_FALLBACK_HELP
        source.update(word)
        help_line.update(help_text)
    width = sampling.size.width - _TITLE_CHROME_CELLS
    state = sampling_state(
        shown_sampling,
        cells=width if width > 0 else SAMPLING_TITLE_CELLS,
        unset=NO_SAMPLER_SET,
    )
    sampling.title = Content(hidden_fields_line("", (), state=state))
