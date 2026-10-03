"""Chat settings' footer actions: Use saved defaults and Ctrl+N (TASK-33006.5).

ADR-095's 2026-09-26 amendment gives a chat that holds work one explicit way
to adopt newly saved defaults. **Use saved defaults** stages, for the draft's
own provider·model, exactly what a new chat on that pair resolves
(``build_target_default_console_session_settings``: the model profile, the
saved ``[console.provider_defaults.<provider>]``, ``chat_defaults``, then the
provider settings). It replaces every unapplied edit, keeps the pair, and
applies nothing: **Apply to this chat** commits it and writes no
configuration.

The defaults reach the draft through the controller's rebaser, as a pick
does. They arrive as the snapshot of a chat opened on that pair, with no
edits, so the rebaser's same-pair branch keeps them and carries nothing over
(handed the chat's own snapshot, that branch would keep the chat's values and
the button would do nothing). While the draft already equals them, the
button is disabled and its label gives the reason, so the footer gains no
row.

The logic lives here, not in ``console_settings_modal.py``, because that
module sits at its ADR-097 size ceiling; the modal keeps only the wiring.
"""

from __future__ import annotations

from dataclasses import replace

from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, Input

from tldw_chatbook.Chat.console_provider_support import supported_generation_fields
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_target_default_console_session_settings,
)
from tldw_chatbook.Chat.console_settings_apply import FULL_MODEL_DEFAULT_FIELDS

USE_SAVED_DEFAULTS_ID = "console-settings-use-saved-defaults"
USE_SAVED_DEFAULTS_LABEL = "Use saved defaults"
#: The disabled button's label is its reason (AC#6 of TASK-33006.5).
SAVED_DEFAULTS_MATCH_LABEL = "Matches saved defaults"
SAVE_MODEL_DEFAULT_LABEL = "Save as model default"
NEW_CHAT_DEFAULT_LABEL = "Default for new chats (Ctrl+N)"
APPLY_LABEL = "Apply to this chat (Ctrl+Enter)"
_USE_TOOLTIP = (
    "Replace this draft with the saved defaults for this provider and model. "
    "Nothing changes until Apply to this chat."
)
_MATCH_TOOLTIP = "This draft already equals the saved defaults for this model."
_NO_MODEL_TOOLTIP = "Choose a model first."
#: Fields whose blank the draft replaces with the opened value.
_REQUIRED = frozenset({"temperature", "top_p"})


class ConsoleSettingsSavedDefaultsMixin:
    """Use saved defaults and Ctrl+N for ``ConsoleSettingsModal``.

    It uses the modal's draft, rebaser and controls directly. Named ``on_*``
    handlers compose with the modal's own (Textual walks the MRO); a plain
    mixin cannot carry ``@on`` handlers.
    """

    #: ``((provider, model), values)`` for the draft's pair: the opening
    #: configuration snapshot never changes while the modal is open.
    _saved_defaults_cache: tuple[tuple[str, str | None], dict[str, object]] | None = None

    def _saved_defaults(self) -> ConsoleSessionSettings:
        """Return the draft with its generation values at the saved defaults.

        Fields the provider does not accept are blank, as the rebaser leaves
        them; the endpoint is the one a new chat on the pair takes; every
        other value (the pair, system prompt, Persona) stays the draft's.

        Returns:
            The settings Use saved defaults stages.
        """
        pair = (self._active_provider, self._current_model_value())
        if self._saved_defaults_cache is None or self._saved_defaults_cache[0] != pair:
            defaults = build_target_default_console_session_settings(
                self._app_config, *pair
            )
            accepted = supported_generation_fields(*pair, self._app_config)
            values: dict[str, object] = {
                name: getattr(defaults, name) if name in accepted else None
                for name in FULL_MODEL_DEFAULT_FIELDS
            }
            self._saved_defaults_cache = (pair, {**values, "base_url": defaults.base_url})
        return replace(self._draft.settings, **self._saved_defaults_cache[1])

    def _saved_defaults_differ(self) -> bool:
        """Return whether Use saved defaults would change a shown value.

        A hidden field is left out: Apply commits it blank either way. A
        cleared Temperature or Top P counts as blank, as the unsaved guard
        counts it, although the draft falls back to the opened value.

        Returns:
            True when a shown field or the endpoint differs from the defaults.
        """
        defaults = self._saved_defaults()
        draft = self._build_draft()
        for row in self.query(".console-settings-field-row"):
            if not row.display:
                continue
            control_id = str(row.id).removesuffix("-row")
            name = control_id.removeprefix("console-settings-").replace("-", "_")
            control = row.get_child_by_id(control_id)
            cleared = isinstance(control, Input) and not control.value.strip()
            value = None if cleared and name in _REQUIRED else getattr(draft, name)
            if value != getattr(defaults, name):
                return True
        provider = self._active_provider
        shown = self._current_base_url_value(provider)
        if shown is None:
            return False
        default_url = self._initial_base_url_for_provider(provider, defaults.base_url)
        return shown != ((default_url or "").strip() or None)

    def _sync_saved_defaults_action(self) -> None:
        """Enable Use saved defaults only while it would change the draft."""
        try:  # a deferred sync can land while a dismissed modal unmounts
            button = self.query_one(f"#{USE_SAVED_DEFAULTS_ID}", Button)
            model = self._current_model_value()
            available = self._draft_rebaser is not None and model is not None
            matches = available and not self._saved_defaults_differ()
        except (NoMatches, QueryError):
            return
        button.disabled = matches or not available
        button.tooltip = (
            _NO_MODEL_TOOLTIP if model is None
            else _MATCH_TOOLTIP if matches
            else _USE_TOOLTIP if available
            else None
        )
        label = SAVED_DEFAULTS_MATCH_LABEL if matches else USE_SAVED_DEFAULTS_LABEL
        if str(button.label) != label:
            button.label = label
            modal = self.query_one("#console-settings-modal")
            self._sync_action_copy((modal.size.width or self.size.width) < 100)

    def _use_saved_defaults(self) -> None:
        """Stage the saved defaults for the draft's pair; Apply commits them."""
        settings = self._saved_defaults()
        source = replace(
            self._initial_full_draft(settings),
            context_policy_overrides=self._draft.context_policy_overrides,
            model_drafts=self._draft.model_drafts,
        )
        self._cancel_connection_probe()
        if self._rebase_to(settings.provider, settings.model, source):
            self._advance_model_generation_preserving_current_listing()
        self.call_after_refresh(self._focus_after_saved_defaults)

    def _focus_after_saved_defaults(self) -> None:
        """Move focus off the now-disabled button, to Apply when it can."""
        apply = self.query_one("#console-settings-save", Button)
        if self._is_effectively_focusable(apply):
            apply.focus()
        else:
            self._focus_highest_priority_connection()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Run Use saved defaults.

        Args:
            event: The pressed button; only Use saved defaults is handled.
        """
        if event.button.id == USE_SAVED_DEFAULTS_ID:
            event.stop()
            self._use_saved_defaults()

    def action_make_new_chat_default(self) -> None:
        """Ctrl+N: the footer's Default for new chats, while it is offered."""
        if not self.query_one("#console-settings-close-guard").display:
            self.query_one("#console-settings-make-default", Button).press()
