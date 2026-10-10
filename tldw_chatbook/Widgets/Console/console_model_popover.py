"""Console Switch model (Alt+M): provider·model pairs plus the quick values.

TASK-33004.4 replaced the quick settings form with the pair list of the
model-configuration spec (§2, mockup (a)). The class, its constructor seams
and the ids ``console-popover-apply``, ``-temperature``, ``-streaming``,
``-save-model-default`` and ``-make-new-chat-default`` stay, so the ADR-095
Apply path is unchanged. Every selectable row is a provider·model pair (spec
rule 1). Readiness comes only from the screen-injected configuration
resolver, once per provider per open and off the UI thread; catalogs and
recents come from injected loaders, so this widget calls no provider service
(ADR-011).

TASK-33004.5 added the value row: exactly the quick mask (Temperature, Max
tokens, Streaming), each one row with its spec §6 Source word, and the keys
Enter / Tab / Ctrl+N / Ctrl+O / Esc, with Esc asking before it drops edits.

TASK-33004.6 added pick-only mode (spec §4 rule 1, for Chat settings'
Change): the same pairs, no value row and no default action; Enter returns
the highlighted ``(provider, model)`` and NEEDS SETUP rows cannot be picked.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import isfinite
from typing import Any, ClassVar, Literal, Protocol
from uuid import uuid4

from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Chat.console_context_policy import ContextCompactionMode
from tldw_chatbook.Chat.console_provider_support import MODEL_FIELD_LABELS
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_VALUE_SOURCE_WORDS,
    ConsoleSessionSettings,
    ConsoleSettingsReadiness,
    build_console_provider_options,
    provider_left_at_shipped_default,
    readiness_words,
    resolve_console_value_layers,
)
from tldw_chatbook.Chat.console_settings_apply import (
    QUICK_MODEL_DEFAULT_FIELDS,
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
    ConsoleSettingsSurface,
    ConsoleSettingsTransfer,
    remember_model_draft,
)
from tldw_chatbook.Chat.provider_catalog import (
    PROVIDER_LEGACY_ALIAS_KEYS,
    provider_display_name,
)
from tldw_chatbook.Chat.provider_endpoint_contract import URL_BASED_PROVIDER_KEYS
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.Chat.sampling_params import MIN_MAX_TOKENS
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Widgets.Console.console_settings_unsaved import unsaved_prompt_copy
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
from tldw_chatbook.Widgets.model_search_picker import CURRENT_MARK, normalize_model_id

CONSOLE_POPOVER_OPEN_FULL_SETTINGS = "open-full-settings"


@dataclass(frozen=True, slots=True)
class ConsoleModelPopoverResult:
    """Deprecated pre-ADR-095 result retained for downstream import stability.

    The rebuilt popover never emits this value. Callers receive a committed typed
    submission, a full-settings transfer, or ``None``.
    """

    settings: ConsoleSessionSettings
    compaction_mode: ContextCompactionMode


_CONSOLE_POPOVER_TEMPERATURE_MIN = 0.0
_CONSOLE_POPOVER_TEMPERATURE_MAX = 2.0
_FULL_SETTINGS_ACTION = "full_settings"
#: The value row, in order: exactly the quick default mask (ADR-095 D3), so
#: Save as model default and Ctrl+N save every value this surface shows.
VALUE_FIELDS = ("temperature", "max_tokens", "streaming")
_INVALID = object()
_VALUE_ERRORS = {
    "temperature": "Temperature must be a finite number from 0 to 2.",
    "max_tokens": "Max tokens must be a whole number of at least 1 (blank: no cap).",
}

#: Spec §2 / mockup (a): READY PROVIDERS shows this many models per provider.
TOP_MODELS_PER_PROVIDER = 3
#: Row caps that keep one open to a screenful; typing searches everything.
_RECENT_ROWS = 6
_SETUP_ROWS = 8
_MATCH_ROWS = 30
HIGHLIGHT_GLYPH = "▶"
#: Column widths of one pair row (mockup (a)). The model column grows to the
#: longest id shown, so ids render whole (spec: never truncated); a row too
#: long for the list ends in an ellipsis (the list is ``nowrap``), so the
#: note, then readiness, give way before any id does.
_MODEL_COLUMNS = 28
_PROVIDER_COLUMNS = 20
_CONTEXT_COLUMNS = 5
_READINESS_COLUMNS = 28

_SETUP_HINTS = {
    "configure_credential": "Enter: add key in Settings",
    "configure_endpoint": "Enter: endpoint in Settings",
    "save_endpoint": "Enter: endpoint in Settings",
    # TASK-33005.5: a server that refused or timed out is fixed outside the
    # app, so Enter explains in place instead of opening Settings.
    "retry_connection": "start it; rechecked on open",
}


def _retries_in_place(readiness: ConsoleSettingsReadiness) -> bool:
    """A refused or timed-out server the user starts: Enter explains in place.

    A built-in cloud's failed key check is re-run only by Settings 't' (D2),
    so its row opens Settings instead (TASK-33005 final review I-4), as the
    Console's Retry connection does.
    """
    connection = readiness.connection
    return readiness.recovery_action == "retry_connection" and (
        connection is None
        or connection.custom_endpoint_id is not None
        or connection.provider_key in URL_BASED_PROVIDER_KEYS
    )


#: Blockers a connection test produced; such a row leads NEEDS SETUP, so the
#: row cap never hides a stopped local server behind keyless cloud rows.
_TEST_FAILURES = frozenset({"endpoint_unreachable", "credential_rejected"})
#: A local probe can change only these rows: Ready, or a failed test.
_PROBED_BLOCKERS = _TEST_FAILURES | {None}


class DraftRebaser(Protocol):
    """Injected provider/model rebase seam owned by ConsoleChatController."""

    def __call__(
        self,
        state: ConsoleSettingsDraftState,
        *,
        provider: str,
        model: str | None,
        app_config: Mapping[str, object],
        exposed_fields: frozenset[str],
    ) -> ConsoleSettingsDraftState: ...


LiveCommitter = Callable[[ConsoleSettingsSubmission], ConsoleSettingsLiveCommit]
PopoverSubmitAction = ConsoleSettingsAction | Literal["full_settings"]


class DefaultReadinessResolver(Protocol):
    """Injected configuration-owned readiness seam."""

    def __call__(
        self,
        provider: str,
        model: str | None,
    ) -> ConsoleSettingsReadiness: ...


class PairUse(Protocol):
    """One recent provider·model pair (``model_switcher.ModelPairUse``)."""

    provider: str
    model: str

    def used_label(self, now: datetime) -> str: ...


RecentPairsLoader = Callable[[], Awaitable[Sequence[PairUse]]]
PreviousPairResolver = Callable[[Sequence[PairUse]], "PairUse | None"]
CatalogLoader = Callable[[str], Awaitable[Sequence[str]]]
SetupOpener = Callable[[str, "str | None"], None]
#: Probes listed providers' local servers (provider -> row model) and calls
#: back with each provider whose shared evidence settled (TASK-33005.5).
ConnectionProber = Callable[
    [Mapping[str, "str | None"], Callable[[str], None]], Awaitable[None]
]

RowKind = Literal["header", "info", "pair", "setup", "more", "typed"]


@dataclass(frozen=True, slots=True)
class SwitcherRow:
    """One line of the pair list.

    ``pair`` and ``typed`` rows apply their pair; ``setup`` rows open that
    provider's Settings fix; ``more`` fills Find with the provider's name;
    ``header`` and ``info`` rows cannot be highlighted.
    """

    kind: RowKind
    text: str = ""
    provider: str = ""
    model: str | None = None
    note: str = ""
    score: int = 99

    @property
    def key(self) -> tuple[str, str, str | None]:
        return (self.kind, self.provider, self.model)


def switcher_readiness_words(readiness: ConsoleSettingsReadiness | None) -> str:
    """A row's spec §5 word (TASK-33005.3: the one shared mapping)."""
    return "checking…" if readiness is None else readiness_words(readiness)


def _is_ready(readiness: ConsoleSettingsReadiness | None) -> bool:
    return readiness is not None and readiness.operability == "ready_to_send"


def provider_key(provider: object) -> str:
    """Canonical provider key; ``custom-ep:`` registry ids stay verbatim."""
    text = str(provider or "").strip()
    return text if text.startswith("custom-ep:") else provider_config_key(text)


def _model_column_width(rows: Sequence[SwitcherRow]) -> int:
    """Return the model column width that shows every row's id whole.

    Args:
        rows: The switcher rows about to be rendered.

    Returns:
        The longest model id's length, never below the column's floor.
    """
    longest = max((len(row.model) for row in rows if row.model), default=0)
    return max(_MODEL_COLUMNS, longest)


def _fit(text: str, width: int) -> str:
    """Shorten in the middle: ids and names differ most at their ends."""
    return text if len(text) <= width else f"{text[: width - 4]}…{text[-3:]}"


def context_copy(tokens: int, verified: bool) -> str:
    """Return a context window's short size, e.g. ``"200k"``, or ``"?"``.

    Args:
        tokens: The window size in tokens.
        verified: Whether the size is known. A provider or application
            fallback is a guess, so it reads ``"?"`` (unknown), never a size
            (TASK-33007 #12).

    Returns:
        The size in ``k`` or ``M`` units, or ``"?"``.
    """
    if not verified:
        return "?"
    if tokens >= 1_000_000:
        size = f"{round(tokens / 1_000_000, 1):g}M"
    elif tokens >= 1_000:
        size = f"{tokens // 1_000}k"
    else:
        size = str(tokens)
    return size


def _temperature_in_range(value: float) -> bool:
    """Return whether a parsed temperature is finite and within modal bounds.

    Args:
        value: Parsed temperature candidate.

    Returns:
        True if ``value`` is within ``[0.0, 2.0]``. NaN and infinite values
        always return False, since any comparison against them is False.
    """
    return (
        isfinite(value)
        and _CONSOLE_POPOVER_TEMPERATURE_MIN
        <= value
        <= _CONSOLE_POPOVER_TEMPERATURE_MAX
    )


def _widget_screen_region(widget: Widget) -> Any:
    """Return one mounted widget's region in screen coordinates."""

    return getattr(widget, "screen_region", None) or widget.region


class ConsolePopoverInput(Input):
    """Input that releases Textual Web mouse capture before action clicks."""

    def on_click(self, event: events.Click | None = None) -> None:
        self.release_mouse()
        if event is None:
            return
        recover = getattr(self.screen, "_recover_redirected_control_click", None)
        if callable(recover):
            recover(event)

    def on_blur(self) -> None:
        self.release_mouse()


class UnsavedEditsGuard(Static, can_focus=True):
    """Esc with edits: 'Enter apply · d discard · Esc keep editing' (spec §4).

    It takes focus while shown, so Enter and ``d`` reach it instead of Find;
    Esc stays the switcher's own binding and keeps editing.
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "screen.guard_apply", "Apply", show=False),
        Binding("d", "screen.guard_discard", "Discard", show=False),
    ]


class ConsoleModelPopover(
    SafeModalDismissMixin,
    ModalScreen[
        "ConsoleSettingsCommittedSubmission | ConsoleSettingsTransfer"
        " | tuple[str, str] | None"
    ],
):
    """Switch model: choose this chat's provider·model pair and quick values."""

    # The 140-column width and 80% height cap are tokens in the app tier
    # (features/_console_panels.tcss); the highlighted-row bar lives there too
    # (components/_lists.tcss), because app CSS outranks DEFAULT_CSS. The
    # 100% clamps here only keep a harness without the app CSS on screen.
    DEFAULT_CSS = """
    ConsoleModelPopover {
        align: center middle;
    }

    #console-model-popover {
        max-width: 100%;
        max-height: 100%;
        height: auto;
        border: round $primary;
        border-title-color: $text-primary;
        border-title-style: bold;
        background: $panel;
        padding: 0 1;
    }

    #console-popover-pairs {
        height: auto;
        max-height: 100%;
        background: $panel;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    #console-popover-pairs.-fill {
        height: 1fr;
    }

    #console-popover-pairs > .option-list--option-disabled {
        color: $text-primary;
        text-style: bold;
    }

    /* An unpickable pair (pick-only NEEDS SETUP): $text-muted is AA-gated on
       every shipped theme (test_theme_contrast); the panel is its base. */
    ConsoleModelPopover > .console-popover--unpickable {
        color: $text-muted;
        background: $panel;
    }

    .console-popover-strip {
        height: auto;
    }

    .console-popover-strip > Static {
        width: auto;
        padding: 0 1 0 0;
        color: $text-muted;
    }

    .console-popover-strip > Button {
        width: auto;
        min-width: 0;
        margin: 0 1 0 0;
    }

    #console-popover-find-label,
    #console-popover-values-label {
        color: $text;
    }

    #console-popover-find {
        width: 1fr;
    }

    .console-popover-value {
        margin: 0 1 0 0;
    }

    /* Focused while shown: the app's focus outline paints its blank first and
       last lines. Not padding rows: Textual 8.2.8 caches the top padding row,
       outline edge included, and repeats it as the bottom one. */
    #console-popover-guard {
        height: auto;
        background: $warning 25%;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }

    #console-popover-scope {
        width: 1fr;
        content-align: right middle;
    }

    #console-popover-error {
        height: auto;
        background: $error 25%;
        color: $text-error;
        text-style: bold;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "request_safe_cancel", "Cancel"),
        Binding("up", "pairs('cursor_up')", show=False),
        Binding("down", "pairs('cursor_down')", show=False),
        Binding("pageup", "pairs('page_up')", show=False),
        Binding("pagedown", "pairs('page_down')", show=False),
        Binding("ctrl+n", "make_new_chat_default", "Default for new chats", show=False),
        Binding("ctrl+o", "chat_settings", "Chat settings", show=False),
    ]
    SAFE_MODAL_CONTENT = "#console-model-popover"
    COMPONENT_CLASSES = {"console-popover--unpickable"}

    def __init__(
        self,
        *,
        origin: ConsoleSettingsOrigin,
        app_config: Mapping[str, object],
        initial_draft: ConsoleSettingsDraftState,
        providers_models: Mapping[str, Sequence[str]],
        scope_copy: str,
        durability_copy: str,
        draft_rebaser: DraftRebaser,
        live_committer: LiveCommitter,
        default_readiness_resolver: DefaultReadinessResolver,
        recent_pairs_loader: RecentPairsLoader | None = None,
        previous_pair: PreviousPairResolver | None = None,
        catalog_loader: CatalogLoader | None = None,
        setup_opener: SetupOpener | None = None,
        pick_only: bool = False,
        query: str = "",
        connection_prober: ConnectionProber | None = None,
        served_models: Mapping[str, Sequence[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize one exact-origin Switch model transaction.

        Args:
            origin: Stable session/conversation binding captured before opening.
            app_config: Configuration snapshot for the rebaser and the
                provider listing (registry entries, chat defaults).
            initial_draft: Complete typed draft shared with full settings.
            providers_models: Saved model list per provider key; the in-memory
                catalog until ``catalog_loader`` resolves a provider.
            scope_copy: Exact conversation scope label.
            durability_copy: Exact unsaved or temporary durability label.
            draft_rebaser: Controller-owned provider/model rebase callback.
            live_committer: Synchronous exact-origin live commit callback.
            default_readiness_resolver: Configuration-only readiness for one
                provider/model target; called once per provider per open, in
                a worker thread.
            recent_pairs_loader: Reads RECENT after the switcher opens.
            previous_pair: Picks PREVIOUS from RECENT (and process memory).
            catalog_loader: Resolves one provider's cached catalog.
            setup_opener: Opens Settings for a NEEDS SETUP provider and one
                of its own models (so Settings never keeps another provider's).
            pick_only: Return the chosen ``(provider, model)`` instead of
                applying it. Lists the same pairs, but shows no values or
                default actions, never calls ``draft_rebaser``,
                ``live_committer`` or ``setup_opener``, and NEEDS SETUP rows
                cannot be picked.
            query: Text Find opens with (``/model <query>``); the best match
                is highlighted, and nothing applies until Enter.
            connection_prober: The screen's local-server probe, run once per
                listed provider per open; this widget calls no network.
            served_models: Models a Chat settings listing found per provider,
                listed beside the saved ones (a new entry's, TASK-33006.4).
            **kwargs: Forwarded to ``ModalScreen``.
        """
        super().__init__(**kwargs)
        if not isinstance(origin, ConsoleSettingsOrigin):
            raise TypeError("origin must be ConsoleSettingsOrigin")
        if not isinstance(initial_draft, ConsoleSettingsDraftState):
            raise TypeError("initial_draft must be ConsoleSettingsDraftState")
        self._origin = origin
        self._app_config = app_config
        self._draft = initial_draft
        self._providers_models = providers_models
        self._scope_copy = scope_copy
        self._durability_copy = durability_copy
        self._draft_rebaser = draft_rebaser
        self._live_committer = live_committer
        self._default_readiness_resolver = default_readiness_resolver
        self._recent_pairs_loader = recent_pairs_loader
        self._previous_pair = previous_pair
        self._catalog_loader = catalog_loader
        self._setup_opener = setup_opener
        self._pick_only = pick_only
        self._connection_prober = connection_prober
        self._served = {provider_key(p): m for p, m in (served_models or {}).items()}
        self._probes_offered: set[str] = set()
        settings = initial_draft.settings
        self._chat_settings = settings
        self._streaming = bool(settings.streaming)
        # Input and Select post Changed once at mount, after on_mount, with
        # the value they were composed with; that echo is not an edit. A
        # blank Input posts none ("" is its default), so none is awaited:
        # a stale "" would swallow the user clearing that value.
        self._mount_echo: dict[str, object] = {
            name: text
            for name in ("temperature", "max_tokens")
            if (text := self._input_text(getattr(settings, name)))
        }
        self._mount_echo["streaming"] = self._streaming
        self._guard_focus: Widget | None = None
        self._updating_controls = False
        self._submit_pending = False
        self._current: tuple[str, str | None] = (
            str(settings.provider or "").strip(),
            settings.model or None,
        )
        self._query = query
        self._rows: list[SwitcherRow] = []
        self._model_columns = _MODEL_COLUMNS
        self._painted_index: int | None = None
        self._highlight_key: tuple[str, str, str | None] | None = None
        self._user_moved = False
        self._recent: tuple[PairUse, ...] = ()
        self._previous: PairUse | None = None
        self._readiness: dict[str, ConsoleSettingsReadiness] = {}
        self._readiness_pending: set[str] = set()
        self._first_readiness_done = False
        self._saved: dict[str, tuple[str, ...]] = {}
        self._catalogs: dict[str, tuple[str, ...]] = {}
        self._catalog_state: dict[str, str] = {}
        self._context_labels: dict[tuple[str, str], str] = {}
        self._display_names: dict[str, str] = {}
        self._provider_order = tuple(
            dict.fromkeys(
                provider_key(option.value)
                for option in build_console_provider_options(
                    providers_models, app_config=app_config
                )
                if provider_key(option.value)
            )
        )

    # -- labels ---------------------------------------------------------

    def _display(self, provider: str) -> str:
        key = provider_key(provider)
        name = self._display_names.get(key)
        if name is None:
            name = (
                provider_display_name(key, self._app_config) if key else "No provider"
            )
            self._display_names[key] = name
        return name

    @staticmethod
    def _input_text(value: object) -> str:
        return "" if value is None else str(value)

    def _values_label(self) -> str:
        settings = self._draft.settings
        return (
            f"Values for {settings.model or 'no model'} · "
            f"{self._display(settings.provider)}"
        )

    @staticmethod
    def _saved_fields_copy() -> str:
        return "saves " + ", ".join(MODEL_FIELD_LABELS[name] for name in VALUE_FIELDS)

    def _source_words(self) -> dict[str, str]:
        """Spec §6 Source word per value, from the one shared resolver."""
        settings = self._draft.settings
        chat_pair = (provider_key(settings.provider), settings.model) == (
            provider_key(self._current[0]),
            self._current[1],
        )
        layers = resolve_console_value_layers(
            self._app_config,
            settings.provider,
            settings.model,
            VALUE_FIELDS,
            edited=frozenset(
                field.name for field in self._draft.field_drafts if field.dirty
            ),
            chat_settings=self._chat_settings if chat_pair else None,
        )
        return {
            name: CONSOLE_VALUE_SOURCE_WORDS[layer] for name, layer in layers.items()
        }

    def _edited_labels(self) -> tuple[str, ...]:
        """Labels of every value edited in this open, for any pair."""
        edited = {
            field.name
            for fields in (
                self._draft.field_drafts,
                *(remembered.field_drafts for remembered in self._draft.model_drafts),
            )
            for field in fields
            if field.dirty
        }
        return tuple(
            MODEL_FIELD_LABELS.get(name, name)
            for name in VALUE_FIELDS
            if name in edited
        )

    def _title(self) -> str:
        provider, model = self._current
        now = f"{self._display(provider)} · {model}" if model else "no model"
        return f"Switch model · now: {now}"

    def _find_placeholder(self) -> str:
        ready = [
            key for key in self._provider_order if _is_ready(self._readiness.get(key))
        ]
        enter = "Enter picks" if self._pick_only else "Enter applies"
        if not ready:
            return f"type to search models · {enter}"
        count = sum(len(self._models_for(key)) for key in ready)
        return f"type to search {count} models in {len(ready)} providers · {enter}"

    # -- compose --------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Build the Find row, the pair list, the value strip and the keys."""
        settings = self._draft.settings
        with Vertical(id="console-model-popover"):
            with Horizontal(
                id="console-popover-find-row", classes="console-popover-strip"
            ):
                yield Static("Find", id="console-popover-find-label")
                yield ConsolePopoverInput(
                    self._query,
                    placeholder=self._find_placeholder(),
                    id="console-popover-find",
                    compact=True,
                )
            error = Static("", id="console-popover-error", markup=False)
            error.display = False
            yield error
            pairs = OptionList(id="console-popover-pairs", compact=True)
            pairs.can_focus = False
            yield pairs
            if self._pick_only:
                with Horizontal(
                    id="console-popover-keys", classes="console-popover-strip"
                ):
                    yield Static("Enter picks · Esc cancel", markup=False)
                return
            yield Static(
                self._values_label(), id="console-popover-values-label", markup=False
            )
            # One row: label, a one-row control, its Source word (spec §6).
            with Horizontal(
                id="console-popover-values", classes="console-popover-strip"
            ):
                for name in VALUE_FIELDS:
                    yield Static(
                        MODEL_FIELD_LABELS[name],
                        classes="console-popover-field-label",
                        markup=False,
                    )
                    yield self._value_control(name, settings)
                    yield Static(
                        "",
                        id=f"console-popover-{name.replace('_', '-')}-source",
                        classes="console-popover-source",
                        markup=False,
                    )
            guard = UnsavedEditsGuard("", id="console-popover-guard", markup=False)
            guard.display = False
            yield guard
            with Horizontal(id="console-popover-keys", classes="console-popover-strip"):
                yield Button(
                    "Enter apply to this chat",
                    id="console-popover-apply",
                    variant="primary",
                    compact=True,
                )
                yield Static("· Tab edit values ·", markup=False)
                yield Button(
                    "Ctrl+N default for new chats",
                    id="console-popover-make-new-chat-default",
                    compact=True,
                )
                yield Static("·", markup=False)
                yield Button(
                    "Ctrl+O chat settings",
                    id="console-popover-full-settings",
                    compact=True,
                )
                yield Static("· Esc cancel", id="console-popover-esc-key", markup=False)
            with Horizontal(
                id="console-popover-defaults-row", classes="console-popover-strip"
            ):
                yield Button(
                    "Save as model default",
                    id="console-popover-save-model-default",
                    compact=True,
                )
                yield Static(
                    self._saved_fields_copy(),
                    id="console-popover-save-model-default-copy",
                    markup=False,
                )
                scope = Static(
                    self._scope_copy, id="console-popover-scope", markup=False
                )
                scope.tooltip = self._durability_copy
                yield scope

    def _value_control(self, name: str, settings: ConsoleSessionSettings) -> Widget:
        """The one-row control for one value: two Inputs and an On/Off Select."""
        if name == "streaming":
            # ADR-095:79: Streaming is On or Off at chat scope, never Inherit.
            return Select(
                (("On", True), ("Off", False)),
                value=self._streaming,
                allow_blank=False,
                compact=True,
                id="console-popover-streaming",
                classes="console-popover-value",
            )
        return ConsolePopoverInput(
            value=self._input_text(getattr(settings, name)),
            placeholder="no cap" if name == "max_tokens" else "",
            restrict=r"[0-9]*" if name == "max_tokens" else None,
            id=f"console-popover-{name.replace('_', '-')}",
            classes="console-popover-value",
            compact=True,
        )

    def on_mount(self) -> None:
        """Highlight PREVIOUS, focus Find, then resolve readiness and recents."""
        # Text, not str: a str title is parsed as markup, and model ids and
        # endpoint display names may hold brackets ("foo[/]" raised).
        self.query_one("#console-model-popover").border_title = Text(self._title())
        if self._previous_pair is not None:
            self._previous = self._previous_pair(())
        self._rebuild_rows()
        self.query_one("#console-popover-find", Input).focus()
        if not self._pick_only:
            # The rows' first highlight may not rebase (the chat's own pair).
            self.call_after_refresh(self._sync_source_words)
        self._request_readiness(self._readiness_targets())
        if self._recent_pairs_loader is not None:
            self.run_worker(
                self._load_recent_pairs(),
                group="console-switcher-recents",
                exclusive=True,
                exit_on_error=False,
            )

    def on_resize(self, _event: events.Resize) -> None:
        """Re-check whether the list must fill the box after a resize."""
        self.call_after_refresh(self._sync_list_height)

    def _sync_list_height(self) -> None:
        """Let the pair list fill the box only once its rows outgrow it.

        The box is auto height up to its max-height, and an auto-height list
        taller than that would push the key rows out of view.
        """
        if not self.is_mounted:
            return
        try:
            box = self.query_one("#console-model-popover", Vertical)
            pairs = self.query_one("#console-popover-pairs", OptionList)
        except NoMatches:
            return
        limit = box.styles.max_height
        if limit is None:
            return
        rows = int(limit.resolve(self.size, self.app.size))
        chrome = box.outer_size.height - pairs.outer_size.height
        pairs.set_class(pairs.option_count + chrome > rows, "-fill")

    # -- readiness, recents and catalogs ---------------------------------

    def _used_provider_keys(self) -> set[str]:
        keys = {provider_key(self._current[0])}
        if self._previous is not None:
            keys.add(provider_key(self._previous.provider))
        keys.update(provider_key(use.provider) for use in self._recent)
        defaults = self._app_config.get("chat_defaults")
        if isinstance(defaults, Mapping):
            keys.add(provider_key(defaults.get("provider")))
        keys.discard("")
        return keys

    def _readiness_targets(self) -> tuple[str, ...]:
        used = self._used_provider_keys()
        return tuple(
            dict.fromkeys(
                key
                for key in (*self._provider_order, *sorted(used))
                if key and (key not in PROVIDER_LEGACY_ALIAS_KEYS or key in used)
            )
        )

    def _representative_model(self, key: str) -> str | None:
        if key == provider_key(self._current[0]) and self._current[1]:
            return self._current[1]
        models = self._models_for(key)
        return models[0] if models else None

    def _request_readiness(
        self, providers: Sequence[str], *, refresh: bool = False
    ) -> None:
        """Resolve readiness for providers not yet asked (or ``refresh``), in one worker."""
        missing = tuple(
            key
            for key in providers
            if key
            and (refresh or key not in self._readiness)
            and key not in self._readiness_pending
        )
        if not missing:
            return
        self._readiness_pending.update(missing)
        self.run_worker(
            self._resolve_readiness(missing),
            group="console-switcher-readiness",
            exit_on_error=False,
        )

    async def _resolve_readiness(self, providers: tuple[str, ...]) -> None:
        resolver = self._default_readiness_resolver
        targets = {key: self._representative_model(key) for key in providers}

        def resolve() -> dict[str, ConsoleSettingsReadiness]:
            resolved: dict[str, ConsoleSettingsReadiness] = {}
            for key, model in targets.items():
                try:
                    resolved[key] = resolver(key, model)
                except Exception:  # noqa: BLE001 - one bad provider must not hide the rest
                    resolved[key] = ConsoleSettingsReadiness(
                        "Not ready", "Review provider settings", False
                    )
            return resolved

        resolved = await asyncio.to_thread(resolve)
        self._readiness.update(resolved)
        self._readiness_pending.difference_update(providers)
        self._first_readiness_done |= not self._readiness_pending
        if not self.is_mounted:
            return
        self._rebuild_rows()
        self._sync_find_placeholder()
        self._load_ready_catalogs(tuple(resolved))
        self._offer_probes(resolved)

    def _offer_probes(self, resolved: Mapping[str, ConsoleSettingsReadiness]) -> None:
        """Hand rows a probe can change to the screen's probe, once per open."""
        targets = {
            key: self._representative_model(key)
            for key, readiness in resolved.items()
            if key not in self._probes_offered
            and readiness.blocker in _PROBED_BLOCKERS
        }
        if self._connection_prober is None or not targets or not self.is_attached:
            return
        self._probes_offered.update(targets)
        self.run_worker(
            self._connection_prober(targets, self.refresh_readiness),
            group="console-switcher-probes",
            exit_on_error=False,
        )

    def refresh_readiness(self, provider: str) -> None:
        """Re-read one provider's readiness after a probe settled; no network."""
        if self.is_attached:  # A popped screen stays "mounted".
            self._request_readiness((provider_key(provider),), refresh=True)

    def _sync_find_placeholder(self) -> None:
        for find in self.query("#console-popover-find").results(Input):
            find.placeholder = self._find_placeholder()

    async def _load_recent_pairs(self) -> None:
        loader = self._recent_pairs_loader
        if loader is None:
            return
        recent = tuple(await loader())
        if not self.is_mounted:
            return
        self._recent = recent
        if self._previous_pair is not None:
            self._previous = self._previous_pair(recent)
        self._request_readiness(sorted(self._used_provider_keys()))
        self._rebuild_rows()

    def _load_ready_catalogs(self, providers: Sequence[str]) -> None:
        if self._catalog_loader is None:
            return
        targets = tuple(
            key
            for key in providers
            if _is_ready(self._readiness.get(key))
            and not key.startswith("custom-ep:")
            and key not in self._catalog_state
        )
        if not targets:
            return
        for key in targets:
            self._catalog_state[key] = "loading"
        self._rebuild_rows()
        self.run_worker(
            self._load_catalogs(targets),
            group="console-switcher-catalogs",
            exit_on_error=False,
        )

    async def _load_catalogs(self, providers: tuple[str, ...]) -> None:
        loader = self._catalog_loader
        if loader is None:
            return
        for key in providers:
            try:
                loaded = await loader(key)
            except Exception:  # noqa: BLE001 - AC#9: say "unavailable", keep the saved list
                self._catalog_state[key] = "unavailable"
                continue
            models = tuple(
                dict.fromkeys(
                    model
                    for model in (normalize_model_id(item) for item in loaded)
                    if model
                )
            )
            self._catalog_state[key] = "ready" if models else "empty"
            if models:
                self._catalogs[key] = models
        if not self.is_mounted:
            return
        self._rebuild_rows()
        self._sync_find_placeholder()

    def _saved_models(self, key: str) -> tuple[str, ...]:
        """The provider's saved models (a registry entry's own list), then
        the ones a Chat settings listing found it serving (``served_models``).
        """
        cached = self._saved.get(key)
        if cached is not None:
            return cached
        if key.startswith("custom-ep:"):
            from tldw_chatbook.Chat.custom_endpoint_registry import entry_for

            entry = entry_for(self._app_config, key)
            listed: list[object] = list(entry.models) if entry is not None else []
        else:
            listed = [
                model
                for provider, models in self._providers_models.items()
                if provider_key(provider) == key
                and not isinstance(models, (str, bytes))
                for model in models
            ]
        listed += self._served.get(key, ())
        cached = tuple(
            dict.fromkeys(model for model in map(normalize_model_id, listed) if model)
        )
        self._saved[key] = cached
        return cached

    def _models_for(self, key: str) -> tuple[str, ...]:
        catalog = self._catalogs.get(key)
        return catalog if catalog is not None else self._saved_models(key)

    # -- rows -----------------------------------------------------------

    def _pair_score(
        self, query: str, tokens: Sequence[str], provider: str, model: str
    ) -> int | None:
        """Match rank, lower is better; None when a token is missing.

        Every token must appear in the model id or the provider's name.
        """
        if not tokens:
            return 99
        folded = model.casefold()
        text = f"{folded} {self._display(provider).casefold()}"
        if not all(token in text for token in tokens):
            return None
        if folded == query:
            return 0
        if folded.startswith(query):
            return 1
        return 2 if query in folded else 3

    def _is_current(self, provider: str, model: str | None) -> bool:
        current_provider, current_model = self._current
        return bool(model) and (provider_key(provider), model) == (
            provider_key(current_provider),
            current_model,
        )

    def _pair_row(
        self,
        kind: RowKind,
        provider: str,
        model: str | None,
        note: str = "",
        score: int = 99,
    ) -> SwitcherRow:
        if self._is_current(provider, model):
            note = CURRENT_MARK
        if provider_key(provider) in PROVIDER_LEGACY_ALIAS_KEYS:
            note = f"legacy alias · {note}" if note else "legacy alias"
        return SwitcherRow(kind, provider=provider, model=model, note=note, score=score)

    def _visible_providers(self) -> list[str]:
        """Current provider first, then failed tests, then listing order.

        Legacy aliases never get readiness unless used (``_readiness_targets``),
        so they drop out below with every other unresolved provider.
        """
        current = provider_key(self._current[0])
        order = [current] if current else []
        order += [key for key in self._provider_order if key != current]
        order += sorted(self._used_provider_keys() - set(order))
        blockers = {key: getattr(self._readiness.get(key), "blocker", None) for key in order}
        return sorted(
            (
                key
                for key in order
                if key == current or blockers[key] != "provider_unsupported"
            ),
            key=lambda key: key != current and blockers[key] not in _TEST_FAILURES,
        )

    def _build_rows(self) -> list[SwitcherRow]:
        query = self._query.strip().casefold()
        tokens = query.split()
        now = datetime.now(UTC)
        rows: list[SwitcherRow] = []
        shown: set[tuple[str, str]] = set()

        def group(header: str, members: list[SwitcherRow]) -> None:
            if members:
                rows.append(SwitcherRow("header", header))
                rows.extend(members)

        def claim(provider: str, model: str) -> None:
            shown.add((provider_key(provider), model))

        previous = self._previous
        members: list[SwitcherRow] = []
        if previous is not None and not self._is_current(
            previous.provider, previous.model
        ):
            score = self._pair_score(query, tokens, previous.provider, previous.model)
            if score is not None:
                members.append(
                    self._pair_row(
                        "pair",
                        previous.provider,
                        previous.model,
                        previous.used_label(now),
                        score,
                    )
                )
                claim(previous.provider, previous.model)
        group(
            "PREVIOUS" if self._pick_only else "PREVIOUS · Alt+M, Enter swaps back",
            members,
        )

        members = []
        for use in self._recent:
            if (provider_key(use.provider), use.model) in shown:
                continue
            score = self._pair_score(query, tokens, use.provider, use.model)
            if score is None:
                continue
            members.append(
                self._pair_row(
                    "pair", use.provider, use.model, use.used_label(now), score
                )
            )
            claim(use.provider, use.model)
            if not tokens and len(members) >= _RECENT_ROWS:
                break
        group("RECENT · your last 50 chats", members)

        current_key, current_model = provider_key(self._current[0]), self._current[1]
        current_shown = not current_model or (current_key, current_model) in shown

        if not self._first_readiness_done:
            members = []
            if not current_shown:
                members.append(self._pair_row("pair", current_key, current_model))
            members.append(SwitcherRow("info", "  checking providers…"))
            group("READY PROVIDERS", members)
            rows.extend(self._typed_rows(tokens))
            return rows

        ready_rows: list[SwitcherRow] = []
        matches: list[tuple[int, int, int, SwitcherRow]] = []
        setup_rows: list[SwitcherRow] = []
        setup_extra = 0
        not_running: list[str] = []
        used = self._used_provider_keys()
        for rank, key in enumerate(self._visible_providers()):
            readiness = self._readiness.get(key)
            if readiness is None:
                continue
            if (
                # TASK-33005.6 AC#3 (owner ruling): a shipped localhost default
                # the user never set up or used is still probed, so a running
                # one is found, but its refusal is no setup task.
                readiness.endpoint_category == "connection_refused"
                and _retries_in_place(readiness)
                and key not in used
                and provider_left_at_shipped_default(self._app_config, key)
            ):
                if self._pair_score(query, tokens, key, "") is not None:
                    not_running.append(self._display(key))
                continue
            models = list(self._models_for(key))
            if key == current_key and current_model:
                # ADR-020: the active model stays listed, first.
                models = [current_model, *(m for m in models if m != current_model)]
            if _is_ready(readiness):
                if tokens:
                    for index, model in enumerate(models):
                        score = self._pair_score(query, tokens, key, model)
                        if score is not None and (key, model) not in shown:
                            row = self._pair_row("pair", key, model, score=score)
                            matches.append((score, rank, index, row))
                    continue
                visible = [model for model in models if (key, model) not in shown]
                ready_rows.extend(
                    self._pair_row("pair", key, model)
                    for model in visible[:TOP_MODELS_PER_PROVIDER]
                )
                more = len(visible) - TOP_MODELS_PER_PROVIDER
                if more > 0:
                    ready_rows.append(
                        SwitcherRow(
                            "more",
                            f"    … {more} more {self._display(key)} models",
                            provider=key,
                        )
                    )
                ready_rows.extend(self._catalog_status_rows(key, bool(models)))
                continue
            setup = self._setup_rows(key, readiness, query, tokens, current_shown)
            if setup and len(setup_rows) >= _SETUP_ROWS:
                setup_extra += 1
                continue
            setup_rows.extend(setup)
        if tokens:
            matches.sort(key=lambda match: match[:3])
            ready_rows = [match[3] for match in matches[:_MATCH_ROWS]]
            if len(matches) > _MATCH_ROWS:
                ready_rows.append(
                    SwitcherRow(
                        "info",
                        f"    … {len(matches) - _MATCH_ROWS} more matches · keep typing",
                    )
                )
            ready_rows.extend(
                row
                for key in self._catalog_state
                if self._catalog_state[key] == "loading"
                for row in self._catalog_status_rows(key, False)
            )
        if setup_extra:
            setup_rows.append(
                SwitcherRow(
                    "info",
                    f"    … {setup_extra} more providers need setup · type a name to find one",
                )
            )
        group(
            # Typed rows are the best matches, not three per provider.
            "READY PROVIDERS · matches from every provider's catalog"
            if tokens
            else "READY PROVIDERS · top 3 each · typing searches every provider's catalog",
            ready_rows,
        )
        group(
            "NEEDS SETUP · set up in Settings first"
            if self._pick_only
            else "NEEDS SETUP · Enter opens the fix or explains it",
            setup_rows,
        )
        group(
            "NOT RUNNING · local servers never set up · rechecked on open",
            [SwitcherRow("info", "  " + " · ".join(not_running))] if not_running else [],
        )
        rows.extend(self._typed_rows(tokens))
        return rows

    def _catalog_status_rows(self, key: str, has_rows: bool) -> list[SwitcherRow]:
        """AC#9: a loading, empty or unavailable catalog says so in a row."""
        state = self._catalog_state.get(key)
        name = self._display(key)
        if state == "loading" and not has_rows:
            return [SwitcherRow("info", f"    {name} · loading models…")]
        if state == "unavailable":
            copy = "showing saved models" if has_rows else "type its name and an id"
            return [SwitcherRow("info", f"    {name} · catalog unavailable · {copy}")]
        if not has_rows and state != "loading":
            return [
                SwitcherRow(
                    "info", f"    {name} · no models reported · type its name and an id"
                )
            ]
        return []

    def _setup_rows(
        self,
        key: str,
        readiness: ConsoleSettingsReadiness,
        query: str,
        tokens: Sequence[str],
        current_shown: bool,
    ) -> list[SwitcherRow]:
        # Pick-only Enter cannot open the fix, so no row promises it.
        action = str(readiness.recovery_action or "")
        if action == "retry_connection" and not _retries_in_place(readiness):
            action = ""  # Settings 't' re-tests it: "Enter: open Settings".
        hint = "" if self._pick_only else _SETUP_HINTS.get(action, "Enter: open Settings")
        current_provider, current_model = self._current
        is_current_provider = key == provider_key(current_provider)
        if not tokens:
            model = current_model if is_current_provider and not current_shown else None
            return [self._pair_row("setup", key, model, hint, 99)]
        rows = [
            self._pair_row("setup", key, model, hint, score)
            for model in self._models_for(key)
            if (score := self._pair_score(query, tokens, key, model)) is not None
        ]
        if not rows:
            score = self._pair_score(query, tokens, key, "")
            if score is not None:
                rows.append(self._pair_row("setup", key, None, hint, score))
        return rows[:3]

    def _typed_rows(self, tokens: Sequence[str]) -> list[SwitcherRow]:
        """The escape hatch: a model id no catalog lists (TASK-14812 AC#5).

        The id pairs with the chat's provider, or with the provider whose name
        the query starts with ("Vale endpoint my-model").
        """
        query = self._query.strip()
        if not tokens:
            return []
        folded = query.casefold()
        providers = (*self._provider_order, *sorted(self._used_provider_keys()))
        if any(self._display(key).casefold() == folded for key in providers):
            return []  # a provider's name is a search, not a model id
        named = [
            key
            for key in providers
            if folded.startswith(f"{self._display(key).casefold()} ")
        ]
        if named:
            provider = max(named, key=lambda key: len(self._display(key)))
            text = query[len(self._display(provider)) :]
        else:
            # The chat's provider, not the draft's: the draft follows the
            # highlight, which moves as the user types.
            provider = provider_key(self._current[0])
            text = query
        typed = normalize_model_id(text)
        if (
            not provider
            or typed is None
            # ponytail: Console surfaces (control bar, workbench mode labels)
            # still parse a model id as markup and crash on "foo[/]"; refuse
            # "[" here until those render sites stop parsing markup.
            or "[" in typed
            or typed in self._models_for(provider)
            or self._is_current(provider, typed)
        ):
            return []
        return [
            SwitcherRow("header", "TYPED MODEL ID · not in any catalog"),
            SwitcherRow(
                "typed",
                provider=provider,
                model=typed,
                note="typed id · in no catalog",
                score=98,
            ),
        ]

    def _context_label(self, provider: str, model: str) -> str:
        cache_key = (provider, model)
        label = self._context_labels.get(cache_key)
        if label is None:
            from tldw_chatbook.Utils.token_counter import resolve_context_window

            try:
                window = resolve_context_window(provider, model)
                label = context_copy(window.tokens, window.verified)
            except Exception:  # noqa: BLE001 - an unknown size is shown, not raised
                label = "?"
            self._context_labels[cache_key] = label
        return label

    def _prompt(self, row: SwitcherRow, highlighted: bool) -> Text:
        if row.kind == "header":
            return Text(row.text, style="bold")
        if row.kind in {"info", "more"}:
            glyph = HIGHLIGHT_GLYPH if highlighted else " "
            return Text(f"{glyph}{row.text[1:]}" if row.kind == "more" else row.text)
        glyph = HIGHLIGHT_GLYPH if highlighted else " "
        model = row.model or "(any model)"
        context = self._context_label(row.provider, row.model) if row.model else ""
        readiness = self._readiness.get(provider_key(row.provider))
        style = Style()
        if not self._selectable(row):
            # Disabled options paint like headers; an unpickable pair is muted.
            muted = self.get_component_rich_style("console-popover--unpickable")
            style = Style(color=muted.color, bold=False)
        return Text(
            f"{glyph} {model:<{self._model_columns}} "
            f"{_fit(self._display(row.provider), _PROVIDER_COLUMNS):<{_PROVIDER_COLUMNS}} "
            f"{context:>{_CONTEXT_COLUMNS}}  "
            f"{_fit(switcher_readiness_words(readiness), _READINESS_COLUMNS):<{_READINESS_COLUMNS}} "
            f"{row.note}",
            style=style,
        )

    def _rebuild_rows(self) -> None:
        """Re-render the list and keep (or choose) the highlighted row."""
        if not self.is_mounted:
            return
        pairs = self.query_one("#console-popover-pairs", OptionList)
        self._rows = self._build_rows()
        self._model_columns = _model_column_width(self._rows)
        self._painted_index = None
        with pairs.prevent(OptionList.OptionHighlighted):
            pairs.clear_options()
            pairs.add_options(
                Option(self._prompt(row, False), disabled=not self._selectable(row))
                for row in self._rows
            )
        self._set_highlight(self._highlight_target())
        self._sync_list_height()
        self.call_after_refresh(self._sync_list_height)

    def _selectable(self, row: SwitcherRow) -> bool:
        """Headers and info rows never; NEEDS SETUP rows not while picking."""
        return row.kind not in {"header", "info"} and not (
            self._pick_only and row.kind == "setup"
        )

    def _highlight_target(self) -> int | None:
        selectable = [
            index for index, row in enumerate(self._rows) if self._selectable(row)
        ]
        if not selectable:
            return None
        # A row the user chose (Up/Down, a click, Tab into its values, an
        # edit) stays highlighted through late fills; typing re-ranks.
        if self._user_moved and self._highlight_key is not None:
            for index in selectable:
                if self._rows[index].key == self._highlight_key:
                    return index
        if self._query.strip():
            ranked = [index for index in selectable if self._rows[index].kind != "more"]
            if ranked:
                return min(ranked, key=lambda index: (self._rows[index].score, index))
        previous = self._previous
        if previous is not None and not self._query.strip():
            for index in selectable:
                row = self._rows[index]
                if (provider_key(row.provider), row.model) == (
                    provider_key(previous.provider),
                    previous.model,
                ):
                    return index
        for index in selectable:
            if self._rows[index].note == CURRENT_MARK:
                return index
        return selectable[0]

    def _set_highlight(self, index: int | None) -> None:
        pairs = self.query_one("#console-popover-pairs", OptionList)
        with pairs.prevent(OptionList.OptionHighlighted):
            pairs.highlighted = index
        self._paint_highlight(index)

    def _paint_highlight(self, index: int | None) -> None:
        """Mark the highlighted row with the glyph, not colour alone.

        Every highlight change also rebases the draft to the row's pair, so
        the value strip always names and shows the pair that Enter, Ctrl+N,
        Save and Ctrl+O act on.
        """
        pairs = self.query_one("#console-popover-pairs", OptionList)
        old = self._painted_index
        if old is not None and old != index and old < len(self._rows):
            pairs.replace_option_prompt_at_index(
                old, self._prompt(self._rows[old], False)
            )
        row = (
            self._rows[index] if index is not None and index < len(self._rows) else None
        )
        if row is not None:
            pairs.replace_option_prompt_at_index(index, self._prompt(row, True))
            self._highlight_key = row.key
        self._painted_index = index
        # A row with no pair of its own (NEEDS SETUP with no model, "… more",
        # or no row at all) shows this chat's pair, never one passed while typing.
        provider, model = (
            (row.provider, row.model)
            if row is not None and row.kind in {"pair", "typed", "setup"} and row.model
            else self._current
        )
        if model and not self._pick_only:
            self._rebase_to(provider, model)

    def highlighted_row(self) -> SwitcherRow | None:
        """The row Enter acts on, or None when nothing is selectable."""
        index = self.query_one("#console-popover-pairs", OptionList).highlighted
        if index is None or index >= len(self._rows):
            return None
        return self._rows[index]

    @on(OptionList.OptionHighlighted, "#console-popover-pairs")
    def _pair_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        self._paint_highlight(event.option_index)

    def action_pairs(self, action: str) -> None:
        """Move the list highlight while focus stays in Find."""
        if action not in {"cursor_up", "cursor_down", "page_up", "page_down"}:
            return
        self._user_moved = True
        pairs = self.query_one("#console-popover-pairs", OptionList)
        getattr(pairs, f"action_{action}")()
        if pairs.highlighted is None:
            # Textual's page move onto a trailing or leading disabled row (a
            # header or an info line) highlights nothing; keep a pair.
            if action == "page_down":
                pairs.action_last()
            else:
                pairs.action_first()

    @on(Input.Changed, "#console-popover-find")
    def _find_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._query = event.value
        self._user_moved = False  # a new query picks its own best match
        self._rebuild_rows()

    @on(Input.Submitted, "#console-popover-find")
    @on(Input.Submitted, "#console-popover-temperature")
    @on(Input.Submitted, "#console-popover-max-tokens")
    def _enter_pressed(self, event: Input.Submitted) -> None:
        event.stop()
        self._activate(self.highlighted_row())

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Tab into the values pins the highlighted pair they belong to."""
        if "console-popover-value" in event.widget.classes:
            self._pin_highlight()

    def _pin_highlight(self) -> None:
        """Keep the highlighted pair through late fills (Tab in, or an edit).

        Not a TYPED MODEL ID row: typed ahead of readiness it is the only
        match, and the catalog row readiness brings must still win; the
        rebaser carries the edits to it.
        """
        row = self.highlighted_row()
        if row is None or row.kind != "typed":
            self._user_moved = True

    @on(OptionList.OptionSelected, "#console-popover-pairs")
    def _pair_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self._user_moved = True
        if event.option_index < len(self._rows):
            self._activate(self._rows[event.option_index])

    def _activate(
        self,
        row: SwitcherRow | None,
        action: ConsoleSettingsAction = ConsoleSettingsAction.APPLY_TO_CHAT,
    ) -> None:
        """Enter: apply the highlighted pair, open its fix, or expand a provider."""
        if row is None:
            # Spec rule 1: with no row highlighted there is no pair to act on.
            self._set_error(
                "Choose a model: type to search, then Enter.",
                focus=self.query_one("#console-popover-find", Input),
            )
            return
        if row.kind == "more":
            find = self.query_one("#console-popover-find", Input)
            find.value = f"{self._display(row.provider)} "
            find.cursor_position = len(find.value)
            find.focus()
            return
        if self._pick_only:
            # Hand the pair back, provider canonical; the opener decides.
            picked = self._selectable(row) and row.model
            if picked and action is ConsoleSettingsAction.APPLY_TO_CHAT:
                self._release_mouse_capture()
                self.dismiss_safe_once((provider_key(row.provider), row.model))
            return
        if row.kind == "setup" and action is ConsoleSettingsAction.APPLY_TO_CHAT:
            key = provider_key(row.provider)
            readiness = self._readiness.get(key)
            if readiness is not None and _retries_in_place(readiness):
                self._set_error(
                    f"{self._display(key)} did not answer: start it; rechecked on open."
                )
                return
            self._open_setup(key, row.model or self._representative_model(key))
            return
        if row.kind in {"pair", "typed", "setup"} and row.model:
            # Save/Ctrl+N on a not-ready pair: Ctrl+N's readiness check says why.
            self._rebase_to(row.provider, row.model)
            self._submit(action)
            return
        if row.kind == "setup":
            self._set_error(
                f"{self._display(row.provider)} needs setup: Enter opens the fix."
            )

    def _open_setup(self, provider: str, model: str | None) -> None:
        """D4: credentials stay in Settings; close and open that provider's fix."""
        self._release_mouse_capture()
        if not self.dismiss_safe_once(None):
            return
        if self._setup_opener is not None:
            self._setup_opener(provider, model)

    # -- draft editing --------------------------------------------------

    def _set_error(self, message: str, *, focus: Widget | None = None) -> None:
        try:
            error = self.query_one("#console-popover-error", Static)
        except NoMatches:
            return
        error.update(message)
        error.display = bool(message)
        if focus is not None:
            focus.focus()

    @staticmethod
    def _parse_value(name: str, raw: str) -> object:
        """One value control's text as its value; blank is None (no value)."""
        text = raw.strip()
        if not text:
            return None
        if not validate_text_input(text, max_length=32):
            return _INVALID
        try:
            value = float(text) if name == "temperature" else int(text)
        except ValueError:
            return _INVALID
        if name == "temperature":
            return value if _temperature_in_range(value) else _INVALID
        return value if value >= MIN_MAX_TOKENS else _INVALID

    def _value_input(self, name: str) -> Input:
        return self.query_one(f"#console-popover-{name.replace('_', '-')}", Input)

    def _replace_quick_field(
        self,
        name: str,
        value: object,
        *,
        direct_edit: bool,
    ) -> None:
        fields = list(self._draft.field_drafts)
        existing_index = next(
            (index for index, field in enumerate(fields) if field.name == name),
            None,
        )
        field = ConsoleSettingsFieldDraft(
            name=name,
            effective_value=value,
            profile_override=value,
            provenance=(
                ConsoleSettingsFieldProvenance.EXPLICIT
                if direct_edit
                else ConsoleSettingsFieldProvenance.INHERITED
            ),
            dirty=direct_edit,
        )
        if existing_index is None:
            fields.append(field)
        else:
            existing = fields[existing_index]
            field = replace(
                existing,
                effective_value=value,
                profile_override=value,
                provenance=(
                    ConsoleSettingsFieldProvenance.EXPLICIT
                    if direct_edit
                    else existing.provenance
                ),
                dirty=direct_edit or existing.dirty,
            )
            fields[existing_index] = field
        self._draft = replace(
            self._draft,
            settings=replace(self._draft.settings, **{name: value}),
            field_drafts=tuple(fields),
        )

    @on(Input.Changed, "#console-popover-temperature")
    @on(Input.Changed, "#console-popover-max-tokens")
    def _value_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        name = (
            "temperature"
            if event.input.id == "console-popover-temperature"
            else "max_tokens"
        )
        self._value_edited(name, event.value, self._parse_value(name, event.value))

    @on(Select.Changed, "#console-popover-streaming")
    def _streaming_changed(self, event: Select.Changed) -> None:
        event.stop()
        if isinstance(event.value, bool):
            self._value_edited("streaming", event.value, event.value)

    def _value_edited(self, name: str, raw: object, value: object) -> None:
        """Record one user edit of a value; syncs and the mount echo are not edits."""
        if self._updating_controls:
            return
        if name in self._mount_echo and self._mount_echo.pop(name) == raw:
            return
        self._pin_highlight()  # a late fill must not retarget an edit
        if name == "streaming":
            self._streaming = bool(value)
        if value is _INVALID:
            # Still an edit: Esc asks, and Apply names the bad value.
            self._draft = replace(
                self._draft,
                field_drafts=tuple(
                    replace(field, dirty=True) if field.name == name else field
                    for field in self._draft.field_drafts
                ),
            )
        else:
            self._replace_quick_field(name, value, direct_edit=True)
        self._sync_source_words()

    def _remember_current_draft(self) -> ConsoleSettingsDraftState:
        for name in ("temperature", "max_tokens"):
            value = self._parse_value(name, self._value_input(name).value)
            if value is not _INVALID and value != getattr(self._draft.settings, name):
                self._replace_quick_field(name, value, direct_edit=False)
        self._draft = replace(
            self._draft,
            settings=replace(self._draft.settings, streaming=self._streaming),
        )
        self._draft = remember_model_draft(self._draft)
        return self._draft

    def _rebase_to(self, provider: str, model: str | None) -> None:
        """Rebase the draft to one pair through the controller seam.

        The typed compaction override rides along unchanged (ADR-095), and
        remembered drafts bring back a pair's edits (A→B→A).
        """
        if (self._draft.settings.provider, self._draft.settings.model) == (
            provider,
            model,
        ):
            return
        self._draft = self._draft_rebaser(
            self._remember_current_draft(),
            provider=provider,
            model=model,
            app_config=self._app_config,
            exposed_fields=QUICK_MODEL_DEFAULT_FIELDS,
        )
        self._streaming = bool(self._draft.settings.streaming)
        self._sync_controls_from_draft()

    def _sync_controls_from_draft(self) -> None:
        if not self.is_mounted:
            return
        settings = self._draft.settings
        self._updating_controls = True
        try:
            for name in ("temperature", "max_tokens"):
                control = self._value_input(name)
                value = getattr(settings, name)
                if self._parse_value(name, control.value) == value:
                    continue  # never rewrite text being typed ("0" -> "0.0")
                with control.prevent(Input.Changed):
                    control.value = self._input_text(value)
            streaming = self.query_one("#console-popover-streaming", Select)
            with streaming.prevent(Select.Changed):
                streaming.value = self._streaming
            self.query_one("#console-popover-values-label", Static).update(
                self._values_label()
            )
        finally:
            self._updating_controls = False
        self._sync_source_words()

    def _sync_source_words(self) -> None:
        if not self.is_mounted:
            return
        for name, word in self._source_words().items():
            selector = f"#console-popover-{name.replace('_', '-')}-source"
            for source in self.query(selector).results(Static):
                source.update(word)

    @on(Button.Pressed, "#console-popover-full-settings")
    def _full_settings(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_chat_settings()

    def action_chat_settings(self) -> None:
        """Ctrl+O: carry the highlighted pair and its edits to Chat settings.

        Nothing is applied or discarded; the full modal opens on the draft.
        """
        self._submit(_FULL_SETTINGS_ACTION)

    @on(Button.Pressed, "#console-popover-apply")
    def _apply(self, event: Button.Pressed) -> None:
        """Apply the highlighted pair to the exact originating conversation."""
        event.stop()
        self._activate(self.highlighted_row())

    @on(Button.Pressed, "#console-popover-save-model-default")
    def _save_model_default(self, event: Button.Pressed) -> None:
        event.stop()
        self._activate(self.highlighted_row(), ConsoleSettingsAction.SAVE_MODEL_DEFAULT)

    @on(Button.Pressed, "#console-popover-make-new-chat-default")
    def _make_new_chat_default(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_make_new_chat_default()

    def action_make_new_chat_default(self) -> None:
        """Ctrl+N: make the highlighted pair the default for new chats."""
        self._activate(
            self.highlighted_row(), ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT
        )

    def _validated_draft(self) -> ConsoleSettingsDraftState | None:
        settings = self._draft.settings
        if not str(settings.provider or "").strip() or not settings.model:
            # Spec rule 1: nothing applies a provider without a model.
            self._set_error(
                "Choose a model: type to search, then Enter.",
                focus=self.query_one("#console-popover-find", Input),
            )
            return None
        values: dict[str, object] = {}
        for name in ("temperature", "max_tokens"):
            control = self._value_input(name)
            value = self._parse_value(name, control.value)
            if value is _INVALID:
                self._set_error(_VALUE_ERRORS[name], focus=control)
                return None
            values[name] = value
        for name, value in values.items():
            # The Changed handlers keep the draft in step; this catches only
            # text the handlers never saw. An unsupported field gains no draft.
            if value != getattr(self._draft.settings, name):
                self._replace_quick_field(name, value, direct_edit=False)
        self._draft = replace(
            self._draft,
            settings=replace(self._draft.settings, streaming=self._streaming),
        )
        self._draft = remember_model_draft(self._draft)
        self._set_error("")
        return self._draft

    def _release_mouse_capture(self) -> None:
        captured = self.app.mouse_captured
        if captured is not None:
            captured.release_mouse()
        if self.app.mouse_captured is not None:
            self.app.capture_mouse(None)

    @staticmethod
    def _without_endpoint_intent(
        draft: ConsoleSettingsDraftState,
    ) -> ConsoleSettingsDraftState:
        """Return a quick-submission draft with no endpoint save intent."""

        return replace(
            draft,
            model_drafts=tuple(
                replace(remembered, endpoint_draft=None)
                for remembered in draft.model_drafts
            ),
            endpoint_draft=None,
        )

    def on_click(self, event: events.Click) -> None:  # type: ignore[override]
        """Recover control clicks redirected through a captured input."""
        self._recover_redirected_control_click(event)

    def _recover_redirected_control_click(self, event: events.Click) -> None:
        captured = self.app.mouse_captured
        click_origin = getattr(event, "widget", None)
        focused = self.app.focused
        screen_routed = click_origin is self and isinstance(
            focused, ConsolePopoverInput
        )
        if (
            not isinstance(captured, ConsolePopoverInput)
            and not isinstance(click_origin, ConsolePopoverInput)
            and not screen_routed
        ):
            return
        if isinstance(captured, ConsolePopoverInput):
            captured.release_mouse()
        if event.button != 1 or event.screen_x is None or event.screen_y is None:
            return
        for control in self.query(Button):
            if control.disabled or not control.display:
                continue
            if _widget_screen_region(control).contains(event.screen_x, event.screen_y):
                control.focus()
                control.press()
                event.stop()
                return

    def _submit(self, action: PopoverSubmitAction) -> None:
        if self._pick_only or self._submit_pending or self._safe_dismiss_committed:
            return
        draft = self._validated_draft()
        if draft is None:
            return
        if action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT:
            readiness = self._default_readiness_resolver(
                draft.settings.provider, draft.settings.model
            )
            if not readiness.native_send_supported:
                self._set_error(f"Unavailable: {readiness.detail}")
                return

        self._release_mouse_capture()
        if action == _FULL_SETTINGS_ACTION:
            self.dismiss_safe_once(ConsoleSettingsTransfer(self._origin, draft))
            return

        draft = self._without_endpoint_intent(draft)
        submission = ConsoleSettingsSubmission(
            submission_id=uuid4().hex,
            action=action,
            surface=ConsoleSettingsSurface.QUICK_POPOVER,
            origin=self._origin,
            draft=draft,
            user_display_name_override=None,
            default_field_mask=(
                frozenset()
                if action is ConsoleSettingsAction.APPLY_TO_CHAT
                else QUICK_MODEL_DEFAULT_FIELDS
            ),
        )
        self._submit_pending = True
        try:
            live_commit = self._live_committer(submission)
        except ValueError as error:
            message = str(error).strip().rstrip(".")
            if message == "Chat closed; nothing applied":
                self.notify("Chat closed; nothing applied", severity="warning")
                self.dismiss_safe_once(None)
                return
            self._set_error(str(error) or "Settings could not be applied.")
            self._submit_pending = False
            return
        except Exception:
            self._set_error("Settings could not be applied; nothing changed.")
            self._submit_pending = False
            return
        if not isinstance(live_commit, ConsoleSettingsLiveCommit):
            self._set_error("Settings could not be applied; nothing changed.")
            self._submit_pending = False
            return
        delivered = False
        try:
            delivered = self.dismiss_safe_once(
                ConsoleSettingsCommittedSubmission(submission, live_commit)
            )
        finally:
            if not delivered and live_commit.durability_admission is not None:
                live_commit.durability_admission.release()

    async def action_dismiss_popover(self) -> None:
        """Dismiss the popover with no result (Escape)."""
        await self.action_request_safe_cancel()

    # -- Esc with edits (spec §4 rule 4) ---------------------------------

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Esc/backdrop: close if nothing is edited, else ask; in the ask, keep editing."""
        del source
        if self._pick_only:
            self.dismiss_safe_once(None)  # nothing was edited here to ask about
            return
        guard = self.query_one("#console-popover-guard", UnsavedEditsGuard)
        if guard.display:
            self._hide_guard()
            return
        labels = self._edited_labels()
        if not labels:
            self.dismiss_safe_once(None)
            return
        self._guard_focus = self.focused
        guard.update(f"\n{unsaved_prompt_copy(labels)}\n")  # blank edge lines (CSS)
        guard.display = True
        # While it asks, Esc keeps editing; the key row must not say cancel.
        self.query_one("#console-popover-esc-key", Static).update("· Esc keep editing")
        # Focus leaves Find, so Enter and d reach the guard, not the query.
        guard.focus()
        self.call_after_refresh(self._sync_list_height)

    async def confirm_quit(self) -> bool:
        """Ask before Ctrl+Q drops edits that Esc would ask about (TASK-33622.15).

        Returns:
            True to let the quit proceed; False to keep editing.
        """
        labels = () if self._pick_only else self._edited_labels()
        if not labels:
            return True
        from tldw_chatbook.Widgets.confirmation_dialog import (
            confirm_quit_discarding_edits,
        )
        from tldw_chatbook.Widgets.Console.console_settings_unsaved import (
            unsaved_summary_copy,
        )

        return await confirm_quit_discarding_edits(self, unsaved_summary_copy(labels))

    def _hide_guard(self) -> None:
        self.query_one("#console-popover-guard", UnsavedEditsGuard).display = False
        self.query_one("#console-popover-esc-key", Static).update("· Esc cancel")
        focus, self._guard_focus = self._guard_focus, None
        if focus is None or not focus.is_attached:
            focus = self.query_one("#console-popover-find", Input)
        focus.focus()
        self.call_after_refresh(self._sync_list_height)

    def action_guard_apply(self) -> None:
        """Guard Enter: apply the highlighted pair with its edits to this chat."""
        self._hide_guard()
        self._activate(self.highlighted_row())

    def action_guard_discard(self) -> None:
        """Guard d: close and drop every edit; nothing is applied or saved."""
        self._release_mouse_capture()
        self.dismiss_safe_once(None)
