"""Settings ▸ Providers & Models ▸ Advanced ▸ Saved model list (TASK-35011).

The block's controls and its title, Add to model list's receipts, and how a
discovered model chosen in Default model joins the provider's saved list on
``s`` (the owner's 2026-10-14 "Pick = use as default" ruling; ADR-002's
2026-10-14 amendment).

The pick is remembered, not derived at save time: ``save_state`` keeps no
discovery listing, and a kept draft outlives the visit. It counts only while
the Providers & Models draft stages exactly that provider and model, so a
Revert, a Discard or a successful save ends it with the draft.

``providers_models_card`` imports this module at load (an Advanced title
names its summary); ``settings_screen`` imports it only inside functions
(ADR-097), by when that module is loaded, so its constants import safely.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.containers import Horizontal
from textual.css.query import QueryError
from textual.widgets import Button, Input, SelectionList, Static

from ...Chat.provider_readiness import provider_config_key
from ...config import SHIPPED_SAMPLE_MODEL_IDS
from ..Screens.provider_model_resolution import MODEL_CAPABILITY_UNKNOWN_WARNING
from ..Screens.settings_screen import INSTANT_APPLY_BEHAVIOR_COPY

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from ..Screens.settings_screen import SettingsScreen

DISCOVER_ID = "settings-discover-provider-models"
LIST_ID = "settings-discovered-models-list"
ADD_ID = "settings-save-discovered-provider-models"  # the id Save selected had
CLEAR_ID = "settings-clear-discovered-provider-models"
SAVED_NOT_LISTED_ID = "settings-saved-models-not-listed"
#: A2: the button no longer shares its save verb with s.
ADD_LABEL = "Add to model list"
ADD_TOOLTIP = (
    "Add the checked models to this provider's saved model list now; "
    "no s needed. Choose one in Default model to make it the default."
)
#: The Saved model list title while a chosen model waits for s (AC#4).
JOINS_ON_SAVE = "1 joins on save"
#: ``save_state``'s key for the pending addition (view state).
PICK_STATE_KEY = "discovered_pick"
#: The Inspector's Save row for the controls that write no config (B19).
SAVES_NOTHING_COPY = "saves nothing · only Add to model list writes the list"
#: AC#7: the list changed on disk between s's read and its locked write.
LIST_CHANGED_COPY = (
    "The saved model list changed while saving; nothing was written. Press s again."
)
#: The Inspector's Save row per control (B19).
SAVE_ROW_COPY = {
    ADD_ID: INSTANT_APPLY_BEHAVIOR_COPY,
    DISCOVER_ID: SAVES_NOTHING_COPY,
    LIST_ID: SAVES_NOTHING_COPY,
    CLEAR_ID: SAVES_NOTHING_COPY,
}
#: The Inspector's content rows per control (B19): what it is, what it
#: does, where it saves, and what it needs.
_GUIDES = {
    DISCOVER_ID: (
        ("Focused setting", "Discover models"),
        (
            "Purpose",
            (
                "Lists the models this provider's endpoint serves now; choose "
                "one in Default model, or check some and press Add to model list."
            ),
        ),
        ("Saved as", "not saved; the listing lasts while Settings is open"),
        ("Validation", "needs the provider's OpenAI-compatible endpoint"),
    ),
    LIST_ID: (
        ("Focused setting", "Discovered models"),
        (
            "Purpose",
            (
                "Space checks a model for Add to model list; each row says "
                "whether it is already saved."
            ),
        ),
        ("Saved as", "not saved; checked rows wait for Add to model list"),
        ("Validation", "check one or more listed models"),
    ),
    ADD_ID: (
        ("Focused setting", ADD_LABEL),
        (
            "Purpose",
            (
                "Adds the checked models to this provider's saved model list, "
                "which Default model and Switch model list."
            ),
        ),
        ("Saved as", "the provider's [providers] model list"),
        ("Validation", "check at least one discovered model"),
    ),
    CLEAR_ID: (
        ("Focused setting", "Clear"),
        (
            "Purpose",
            "Drops the discovered listing; the saved model list stays as it is.",
        ),
        ("Saved as", "not saved; clears the listing only"),
        ("Validation", "available after Discover models"),
    ),
}


def saved_list_guide(field_id: str | None) -> tuple[tuple[str, str], ...] | None:
    """The Inspector's content rows for one of this block's controls.

    Args:
        field_id: The focused widget's id.

    Returns:
        Four (label, value) rows, or None for any other field.
    """
    return _GUIDES.get(field_id or "")


def _model_id(model: object) -> str:
    return str(getattr(model, "model_id", "") or "").strip()


def note_pick(screen: SettingsScreen, model_id: str, *, custom: bool = False) -> None:
    """Stage a Default model choice; remember it when it is a discovered one.

    A pick is remembered only when the current listing holds it and the
    saved list does not (the ruling's "choosing a discovered model"); a
    Custom ID or any other pick forgets the earlier one. A shipped sample
    id is never remembered: no saved list keeps one (TASK-35010, OD-1), so
    it is saved as the default alone (review round 1).

    Args:
        screen: The Settings screen.
        model_id: The chosen id ("" for none).
        custom: The id was typed as a Custom ID.
    """
    screen._set_model_field_value(model_id, quiet=False)
    provider = screen._provider_widget_value()
    saved = screen._provider_saved_model_ids(provider)
    listed = (m for m in screen._model_discovery_models if _model_id(m) == model_id)
    skip = custom or not model_id or model_id in SHIPPED_SAMPLE_MODEL_IDS
    model = None if skip else next(listed, None)
    screen._discovered_pick = (
        (provider_config_key(provider), model_id)
        if model is not None and not screen._discovered_model_is_saved(model, saved)
        else None
    )


def pending_pick(screen: SettingsScreen) -> str:
    """The discovered model the next ``s`` adds to the saved list, or "".

    Only while the draft still stages exactly that provider and model.
    """
    pick = screen._discovered_pick
    draft = screen._provider_draft()
    if pick is None or draft is None or "model" not in draft.dirty_keys:
        return ""
    values = screen._provider_setting_values_mapping()
    staged = (
        provider_config_key(str(values.get("provider") or "")),
        str(values.get("model") or "").strip(),
    )
    return pick[1] if staged == tuple(pick) else ""


def kept_pick(screen: SettingsScreen) -> tuple[str, str] | None:
    """The pending addition ``save_state`` keeps, while it still counts."""
    return screen._discovered_pick if pending_pick(screen) else None


def restore_pick(screen: SettingsScreen, state: Mapping[str, object]) -> None:
    """Take back a kept pending addition (``restore_state``)."""
    pick = state.get(PICK_STATE_KEY)
    if type(pick) is tuple and len(pick) == 2 and all(type(p) is str for p in pick):
        screen._discovered_pick = pick


def saved_models_summary(screen: SettingsScreen) -> str:
    """Count the provider's saved models against the discovered ones.

    Args:
        screen: The Settings screen that owns the discovery state.

    Returns:
        E.g. "12 saved in config · 41 discovered, 29 not saved · 2 selected",
        ending "· 1 joins on save" while a chosen model waits for s.
    """
    saved_ids = set(screen._provider_saved_model_ids(screen._provider_widget_value()))
    parts = [f"{len(saved_ids)} saved in config"]
    rows = [model for model in screen._model_discovery_models if _model_id(model)]
    if not rows:
        parts.append("none discovered")
    else:
        unsaved = sum(
            not screen._discovered_model_is_saved(model, saved_ids) for model in rows
        )
        parts.append(f"{len(rows)} discovered, {unsaved} not saved")
        if screen._model_discovery_selected_model_ids:
            parts.append(f"{len(screen._model_discovery_selected_model_ids)} selected")
    # In memory only: this runs on every keystroke (refresh_advanced).
    if (pick := pending_pick(screen)) and pick not in saved_ids:
        parts.append(JOINS_ON_SAVE)
    return " · ".join(parts)


def saved_models_not_listed(screen: SettingsScreen) -> str:
    """Name the saved models the open list does not show (capture round 2, 5).

    The list holds discovered models only, so a saved model the server does
    not serve appeared only in the picker, under "Saved fallback".

    Args:
        screen: The Settings screen that owns the discovery state.

    Returns:
        "Saved: a, b" before a discovery, "Saved, not served now: b" after
        one, or "" when there is nothing more to name.
    """
    saved = screen._provider_saved_model_ids(screen._provider_widget_value())
    if not screen._model_discovery_models:
        return f"Saved: {', '.join(saved)}" if saved else ""
    served = {_model_id(model) for model in screen._model_discovery_models}
    rest = [model for model in saved if model not in served]
    return f"Saved, not served now: {', '.join(rest)}" if rest else ""


def compose_saved_model_list(screen: SettingsScreen, provider: str) -> ComposeResult:
    """Compose the Saved model list's rows, in their Tab order (B6).

    Discover models, then the list it fills, then Add to model list and
    Clear, which act on that list.

    Args:
        screen: The Settings screen that owns the discovery state.
        provider: The provider the form holds.

    Yields:
        The block's widgets.
    """
    status = Static(
        screen._model_discovery_status,
        id="settings-model-discovery-status",
        classes="settings-status-row",
        markup=False,  # it can name a provider key or quote the server
    )
    status.display = bool(screen._model_discovery_status)  # "" after a find
    yield status
    yield Static(
        MODEL_CAPABILITY_UNKNOWN_WARNING,
        id="settings-model-discovery-capability-warning",
        classes="settings-status-row",
    )
    with Horizontal(classes="settings-input-row"):
        yield Button(
            "Discover models",
            id=DISCOVER_ID,
            disabled=not screen._model_discovery_available(provider),
            tooltip=(
                "Query the configured OpenAI-compatible provider endpoint "
                "for available models."
            ),
        )
    discovered_list = SelectionList(
        *screen._model_discovery_selection_options(),
        id=LIST_ID,
        classes="settings-discovered-models-list",
        disabled=not screen._model_discovery_models,
    )
    # Without its box an empty list is zero rows tall; the status
    # line above says why (capture finding 8: it alone says so).
    discovered_list.display = bool(screen._model_discovery_models)
    yield discovered_list
    with Horizontal(classes="settings-input-row"):
        yield Button(
            ADD_LABEL,
            id=ADD_ID,
            disabled=not screen._model_discovery_models,
            tooltip=ADD_TOOLTIP,
        )
        yield Button(
            "Clear",
            id=CLEAR_ID,
            disabled=not screen._model_discovery_models,
            tooltip="Clear runtime-discovered models for this provider.",
        )
        # ADR-033's labelled exception: Add writes now, not on s.
        yield Static(INSTANT_APPLY_BEHAVIOR_COPY, classes="settings-instant-apply-hint")
    not_listed = saved_models_not_listed(screen)
    line = Static(
        not_listed,
        id=SAVED_NOT_LISTED_ID,
        classes="settings-status-row",
        markup=False,  # model ids may hold "[" and "]"
    )
    line.display = bool(not_listed)
    yield line


def _named(model_ids: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c', or 'a, b, c and 2 more'."""
    shown = list(model_ids[:3])
    if len(model_ids) > 3:
        return f"{', '.join(shown)} and {len(model_ids) - 3} more"
    return " and ".join(filter(None, (", ".join(shown[:-1]), shown[-1])))


def add_receipt(label: str, added: Sequence[str], filled: str = "") -> str:
    """Add to model list's receipt (C4 residual, AC#9, AC#10).

    Args:
        label: The provider's display name.
        added: The ids the write added.
        filled: The id that filled an empty Default model, if any.

    Returns:
        The one line that reports the write.
    """
    if not added:
        return (
            f"The selected models are already in {label}'s model list. To make "
            "one the default, choose it in Default model."
        )
    receipt = f"Added {_named(added)} to {label}'s model list."
    if filled:
        receipt += f" Default model is now {filled}; press s to save it."
    return receipt


def fill_default(screen: SettingsScreen, added: tuple[str, ...]) -> str:
    """Populate an empty Default model with a just-added model (TASK-369).

    Returns:
        The id staged as an unsaved draft, or "" when the field was set.
    """
    try:
        model_input = screen.query_one("#settings-model-value", Input)
    except QueryError:
        return ""
    next_value = screen._model_to_activate_after_save(model_input.value, added)
    if not next_value or next_value == model_input.value:
        return ""
    screen._set_model_field_value(next_value, quiet=False)  # stages the draft
    return next_value


def retry_step(screen: SettingsScreen, provider: str, selected: Sequence[str]) -> str:
    """Name the true next step after a raised Add to model list.

    The step follows what is on screen now: Discover, a Clear that worked
    or a provider change since the press drop the checked rows; a Clear
    that failed keeps them. A Clear still running may yet drop them, so it
    counts as dropping them.
    """
    if provider_config_key(screen._provider_widget_value()) != provider_config_key(
        provider
    ):
        return (
            f"Switch back to {screen._provider_display_label(provider)}, Discover "
            f"and select them again, then press {ADD_LABEL}"
        )
    if not screen._model_discovery_clears_in_flight and set(selected) <= (
        screen._model_discovery_selected_model_ids
    ):
        return f"Press {ADD_LABEL} again"
    return (
        "Select them again (Discover first if they are no longer listed), then "
        f"press {ADD_LABEL}"
    )


@dataclass(frozen=True)
class ListJoin:
    """What one ``s`` save adds to the saved list, and what it says.

    Attributes:
        pending: A pick was pending for this save.
        list_key: The exact ``[providers]`` key written, or "" for none.
        read: That list as read before the write.
        receipt: The toast's sentence about the list ("" when silent).
    """

    pending: bool = False
    list_key: str = ""
    read: object = None
    receipt: str = ""

    @property
    def precondition(self) -> Callable[[object], bool] | None:
        """The locked write's check that the list is still the one read."""
        if not self.list_key:
            return None

        def unchanged(snapshot) -> bool:
            # TASK-33641.6: pure. The locked snapshot only; no config call,
            # no task, no await (the write lock refuses one). The same
            # effective view as the read, never the loader-filtered list.
            providers = snapshot.values.get("providers")
            return isinstance(providers, Mapping) and (
                providers.get(self.list_key) == self.read
            )

        return unchanged

    def applied(self, screen: SettingsScreen, provider: str) -> None:
        """After a save that fully applied: the lists follow the file.

        Whenever a pick was pending, also when nothing was written for it:
        the file may have gained the model elsewhere (review round 1).
        """
        screen._discovered_pick = None
        if not self.pending:
            return
        from ...LLM_Provider_Catalog.model_discovery_persistence import (
            refresh_providers_models,
        )

        refresh_providers_models(screen.app_instance)
        screen._sync_model_picker(provider, reload=True)  # its row reads "saved"
        screen._refresh_model_discovery_widgets()


def join_pick(
    screen: SettingsScreen,
    provider: str,
    model: str,
    section_values: dict[str, dict[str, object]],
) -> ListJoin:
    """Add the pending pick to this save, starting from the file's list.

    Reads the list under the config lock (no cache), appends the model (the
    append also drops sample ids, TASK-35010) into ``section_values`` and
    returns the precondition that refuses the write if the list moved.

    Args:
        screen: The Settings screen.
        provider: The provider being saved.
        model: The model being saved as the default.
        section_values: The save's sections; gains ``providers`` when the
            model joins.

    Returns:
        The join; an empty one when nothing is pending for this save.
    """
    if not model or pending_pick(screen) != model:
        return ListJoin()
    from ... import config
    from ...Chat.provider_setup_persistence import (
        canonical_provider_key,
        saved_model_list_admits,
    )
    from ...LLM_Provider_Catalog.model_discovery_persistence import (
        append_models_to_provider_list,
    )
    from ...LLM_Provider_Catalog.model_discovery_provider_identity import (
        resolve_provider_list_key,
    )

    label = screen._provider_display_label(provider)

    def not_added(why: str) -> ListJoin:
        return ListJoin(
            pending=True,
            receipt=f" {model} was not added to {label}'s model list: {why}.",
        )

    try:
        providers = config.get_atomic_config_snapshot().values.get("providers")
    except Exception:  # noqa: BLE001 - any read failure: save the default alone
        providers = None
    if not isinstance(providers, Mapping):
        return not_added("the config file could not be read")
    resolution = resolve_provider_list_key(provider, providers)
    if resolution.status == "ambiguous":
        return not_added(f"more than one [providers] list matches {label}")
    if resolution.status != "resolved":
        return not_added(f"[providers] has no list for {label}")
    list_key = str(resolution.provider_list_key)
    read = providers[list_key]
    if isinstance(read, list) and model in read:
        return ListJoin(pending=True)  # added elsewhere since: nothing to write
    updated = append_models_to_provider_list({list_key: read}, list_key, [model])
    try:
        owner = canonical_provider_key(provider)
    except ValueError:
        owner = ""  # the save itself refuses this provider
    # The combined save's own check (review rounds 1 and 2): a list it would
    # refuse would refuse the default with it.
    if not saved_model_list_admits(list_key, updated[list_key], model, owner):
        # Each cause on its own (review round 3): "m" is an id the check
        # admits, so it tests the key alone; then the model alone.
        cannot = "Settings cannot write (empty, over 120 characters, or not plain text)"
        if model not in updated[list_key]:
            why = "it is a sample id older versions shipped, which saved lists ignore"
        elif not saved_model_list_admits(list_key, ["m"], "m", owner):
            why = f'the [providers] key "{list_key}" is not one Settings can write'
        elif not saved_model_list_admits(list_key, [model], model, owner):
            why = f"it is an id {cannot}"
        else:
            why = f"the list holds an id {cannot}"
        return not_added(why)
    section_values["providers"] = {list_key: updated[list_key]}
    return ListJoin(
        pending=True,
        list_key=list_key,
        read=read,
        receipt=f" Added {model} to {label}'s model list.",
    )
