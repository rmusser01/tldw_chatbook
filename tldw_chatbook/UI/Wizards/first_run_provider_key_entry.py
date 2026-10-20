"""API-key entry for the first-run Provider step (TASK-34100.6 slice B).

Review findings gap-04, provider-17 and enhancement E11 (2026-10-02):

* A pasted key with an invisible character crashed the step (a contained
  ``ValueError`` from the credential draft), and one wrapped in smart quotes
  was reported as 'couldn't reach the server'. ``clean_api_key`` strips what a
  key never contains -- format/control/separator characters, one pair of
  wrapping quotes, a ``Bearer `` or ``NAME=`` prefix -- and says what it
  removed. Any non-ASCII left over is a field-level message, never a crash.
* Textual's ``Input`` keeps only the FIRST line of a multi-line paste, so a
  key wrapped across lines lost its tail. ``ProviderKeyInput`` cleans the
  whole pasted text instead, then reports a non-secret echo (length and last
  four characters) that the step shows under the field.
* ``suggest_provider_for_key`` recognises well-known key prefixes locally, so
  a key pasted under the wrong provider can be moved. It never logs, sends or
  blocks anything.

Nothing in this module logs or stores a key value.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from typing import Any

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Input, Label, RadioSet, Static

from tldw_chatbook.Utils.input_validation import escape_markup
from tldw_chatbook.UI.Wizards.first_run_provider_detection import (
    ProviderDetectionStep,
)
from tldw_chatbook.UI.Wizards.first_run_provider_skip import (
    SKIP_LABEL,
    missing_url_reason,
)
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupRadioButton,
    SetupRadioSet,
)

#: Refreshes a pick waits for the expanded Authentication section's layout
#: before scrolling to it anyway (live F5: usually one or two).
_LAYOUT_WAIT_REFRESHES = 10

#: Opening -> closing quote pairs a copied key arrives wrapped in.
_QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "`": "`",
    "“": "”",  # “ ”
    "‘": "’",  # ‘ ’
    "„": "“",  # „ “
    "«": "»",  # « »
}
#: Characters a key never contains: format (zero-width, BOM, bidi marks),
#: control (tabs, newlines), separators (spaces, line/paragraph separators).
_STRIPPED_CATEGORIES = frozenset({"Cf", "Cc", "Zs", "Zl", "Zp"})
#: ``OPENAI_API_KEY=`` or ``export OPENAI_API_KEY=``: an all-caps shell name,
#: which a mixed-case key body is never mistaken for.
_ASSIGNMENT_PREFIX = re.compile(r"^(?:export\s+)?[A-Z][A-Z0-9_]*\s*=\s*")
_BEARER_PREFIX = re.compile(r"^bearer\s+", re.IGNORECASE)

#: Well-known key prefixes -> the provider that issues them, longest first so
#: ``sk-ant-`` / ``sk-or-`` / ``sk-proj-`` win over a bare ``sk-``.
KEY_PREFIX_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("sk-ant-", "anthropic"),
    ("sk-or-", "openrouter"),
    ("sk-proj-", "openai"),
    ("gsk_", "groq"),
    ("AIza", "google"),
)


@dataclass(frozen=True, slots=True)
class CleanedKey:
    """A cleaned key and what cleaning removed (no secret in the repr)."""

    value: str
    note: str
    non_ascii: int

    def __repr__(self) -> str:
        return (
            f"CleanedKey(length={len(self.value)}, note={self.note!r}, "
            f"non_ascii={self.non_ascii})"
        )


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _strip_quotes(text: str) -> tuple[str, bool]:
    if len(text) >= 2 and _QUOTE_PAIRS.get(text[0]) == text[-1]:
        return text[1:-1], True
    return text, False


def clean_api_key(raw: str) -> CleanedKey:
    """Remove what a pasted API key never contains, and say what went.

    In order: invisible characters (format, control and non-ASCII separator
    characters -- zero-width spaces, BOMs, newlines, tabs); a shell
    ``NAME=`` / ``export NAME=`` prefix; one pair of wrapping quotes
    (straight or smart); a ``Bearer `` prefix; then any spaces left.

    Args:
        raw: The pasted (or typed) text.

    Returns:
        The cleaned value, a note naming what was removed ("" when nothing
        was), and how many non-ASCII characters remain.
    """
    text = raw if isinstance(raw, str) else ""
    invisible = 0
    visible: list[str] = []
    for character in text:
        if character != " " and unicodedata.category(character) in _STRIPPED_CATEGORIES:
            invisible += 1
        else:
            visible.append(character)
    work = "".join(visible).strip(" ")
    prefix = ""
    match = _ASSIGNMENT_PREFIX.match(work)
    if match and work[match.end() :].strip(" "):
        prefix = match.group(0).split("=", 1)[0].split()[-1] + "="
        work = work[match.end() :].strip(" ")
    work, quoted = _strip_quotes(work)
    match = _BEARER_PREFIX.match(work.strip(" "))
    bearer = bool(match) and bool(work.strip(" ")[match.end() :].strip(" "))
    if bearer:
        work = work.strip(" ")[match.end() :]
    if not quoted:
        work, quoted = _strip_quotes(work.strip(" "))
    value = work.replace(" ", "")
    spaces = text.count(" ") - (value.count(" "))
    spaces -= sum(part.count(" ") for part in (prefix,) if part)
    if bearer:
        spaces -= 1
    if prefix and text.lstrip().startswith("export"):
        spaces -= 1
    parts: list[str] = []
    if prefix:
        parts.append(f"the {prefix} prefix")
    if bearer:
        parts.append("a 'Bearer ' prefix")
    if invisible:
        parts.append(_plural(invisible, "invisible character"))
    if spaces > 0:
        parts.append(_plural(spaces, "space"))
    if quoted:
        parts.append("surrounding quotes")
    note = ""
    if value != text:
        if not parts:
            parts.append("extra characters")
        joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        note = f"Removed {joined} from the pasted key."
    non_ascii = sum(1 for character in value if not character.isascii())
    return CleanedKey(value=value, note=note, non_ascii=non_ascii)


def non_ascii_message(count: int) -> str:
    """Field-level copy for a key that still holds non-ASCII characters."""
    return (
        f"The key contains {_plural(count, 'character')} that API keys never use "
        "(they are plain letters, digits and symbols). Copy it again from the "
        "provider's page and paste it here."
    )


def paste_echo(value: str) -> str:
    """A non-secret echo of a pasted key: its length and last four characters.

    Shown only on screen, cleared on blur and Next, never logged.
    """
    if not value:
        return ""
    tail = value[-4:] if len(value) > 8 else ""
    ending = f", ending …{tail}" if tail else ""
    return f"Pasted {_plural(len(value), 'character')}{ending}."


def suggest_provider_for_key(value: str, selected_provider: str) -> str:
    """The provider a key's well-known prefix belongs to, if not the selected one.

    Computed locally from the prefix alone; the key is never logged or sent.

    Args:
        value: The cleaned key.
        selected_provider: The provider the key was typed under.

    Returns:
        The other provider's key, or "" when no hint applies.
    """
    for prefix, provider in KEY_PREFIX_PROVIDERS:
        if value.startswith(prefix):
            return "" if provider == selected_provider else provider
    return ""


class BurstSafeInput(Input):
    """An Input whose own editing keys apply in the order they were typed.

    TASK-34332: Textual's Input inserts a printable key in its own handler,
    but resolves Backspace, End, Delete, Ctrl+U... through non-priority
    bindings only after the key has bubbled up to the App, one message hop per
    ancestor. When keys arrive in a burst (a paste-like typist, or a busy UI
    loop) the characters typed after a Backspace land first and the late
    Backspaces then delete them: 'http://127.0.0.1:9099' became 'http://127.0'.
    Running the field's own binding here keeps every key in arrival order.
    """

    async def _on_key(self, event: events.Key) -> None:
        if not event.is_printable:
            for binding in self._bindings.key_to_bindings.get(event.key, ()):
                if binding.priority or "." in binding.action:
                    continue
                event.stop()
                event.prevent_default()
                self._restart_blink()
                await self.run_action(binding.action)
                return
        await super()._on_key(event)


class ProviderKeyInput(BurstSafeInput):
    """The masked key field. A paste is cleaned whole, then echoed safely."""

    BINDINGS = [
        # TASK-34332 lesson (lessons-textual: Screen's Tab binding is not
        # priority): a burst typed after Tab must land in the next field.
        Binding("tab", "app.focus_next", show=False, priority=True),
        Binding("shift+tab", "app.focus_previous", show=False, priority=True),
        # Review round 2, G6-VR2-10: the help line promises 'Ctrl+U clears
        # the field'; Input's own Ctrl+U keeps what is right of the cursor,
        # unseen behind the mask (a click lands mid-key).
        Binding("ctrl+u", "clear_key", show=False),
    ]

    def action_clear_key(self) -> None:
        """Clear the whole key, wherever the cursor is."""
        self.value = ""

    class Pasted(Message):
        """A paste landed in the key field; carries only non-secret facts."""

        def __init__(self, note: str, echo: str) -> None:
            super().__init__()
            self.note = note
            self.echo = echo

    def _on_paste(self, event: events.Paste) -> None:
        # prevent_default: Textual's MRO dispatch would also run Input's own
        # _on_paste, which inserts the raw first line a second time.
        event.prevent_default()
        event.stop()
        if not event.text:
            return
        cleaned = clean_api_key(event.text)
        selection = self.selection
        if selection.is_empty:
            self.insert_text_at_cursor(cleaned.value)
        else:
            self.replace(cleaned.value, *selection)
        self.post_message(self.Pasted(cleaned.note, paste_echo(self.value)))


class AuthCollapsible(Collapsible):
    """The Authentication section: a provider switch never scrolls to it.

    TASK-34100.6 (live finding): Textual's Collapsible scrolls itself into
    view whenever ``collapsed`` changes, programmatic changes included.
    Arrowing through the provider list switches providers, and each switch
    between a keyed row (expanded) and a keyless one (collapsed) scrolled the
    list being browsed out of view.
    """

    _hold_scroll = False

    def set_collapsed_quietly(self, collapsed: bool) -> None:
        """Expand or collapse without scrolling the step."""
        self._hold_scroll = True
        self.collapsed = collapsed
        self.call_after_refresh(self._release_scroll)  # after Collapsible's own

    def _release_scroll(self) -> None:
        self._hold_scroll = False

    def scroll_visible(self, *args: Any, **kwargs: Any) -> None:
        if not self._hold_scroll:
            super().scroll_visible(*args, **kwargs)


def usable_typed_key(typed: str) -> str:
    """The typed key if it can go on the wire, else "" (non-ASCII never can)."""
    value = typed.strip()
    if not value.isascii() or any(
        unicodedata.category(character) in _STRIPPED_CATEGORIES for character in value
    ):
        return ""
    return value


class KeySourceRadioButton(SetupRadioButton):
    """A key-source option whose label wraps instead of being cut short.

    TASK-34100.6 pass 2 (live, g6b-env80/02): at 80x24 the default option read
    'Use ANTHROPIC_API_KEY from your environment (found; no…'. Textual's
    ToggleButton pins its content height to one row, so ``text-wrap: wrap``
    alone only moves the cut; this measures the wrapped label instead.
    """

    def get_content_height(self, container: Any, viewport: Any, width: int) -> int:
        """The wrapped label's rows (one row unless the stylesheet wraps it)."""
        if self.styles.text_wrap != "wrap":
            return super().get_content_height(container, viewport, width)
        return max(1, Widget.get_content_height(self, container, viewport, width))


def compose_key_entry() -> ComposeResult:
    """The Authentication section's contents, in visual (Tab) order."""
    with SetupRadioSet(id="setup-provider-key-source", classes="hidden"):
        yield KeySourceRadioButton("", id="setup-provider-key-source-env", value=True)
        yield KeySourceRadioButton(
            "Store a different key for this app", id="setup-provider-key-source-store"
        )
    yield Label("API key", id="setup-provider-key-label", classes="setup-field-label")
    with Horizontal(classes="setup-key-row"):
        yield ProviderKeyInput(
            password=True,
            id="setup-provider-api-key",
            placeholder="Paste the key",  # 13 columns at 80x24 (G6-VR2-11)
        )
        yield Button("Show", id="setup-provider-key-reveal")
        yield Button("Check key", id="setup-provider-check-key")  # provider-06
        yield Button(SKIP_LABEL, id="setup-provider-skip")
    with Horizontal(id="setup-provider-key-hint", classes="setup-key-hint hidden"):
        yield Static("", id="setup-provider-key-hint-text")
        yield Button("Switch", id="setup-provider-key-switch")
    yield Static(  # literal text, never markup (review v3r1, R1-10)
        "", id="setup-provider-key-note", classes="setup-probe-status hidden", markup=False
    )
    yield Static("", id="setup-provider-key-status", classes="setup-probe-status", markup=False)
    with Horizontal(id="setup-provider-key-actions", classes="hidden"):
        yield Button("Keep current", id="setup-provider-key-keep")
        yield Button("Replace", id="setup-provider-key-replace")
        yield Button("Clear", id="setup-provider-key-clear")


def key_help_line(provider_key: str, readiness: Any) -> Text:
    """The not-ready help under the key field: the on-screen fix first.

    Where to get a key comes from the shared catalog facts (review
    provider-25), the same source Settings reads. The expert routes (an
    exported variable, config.toml) sit on a quieter second line (review
    cross-cutting-07), set apart in italics: Rich 'dim' over the muted line
    measured 3.72:1 (review round 1, live F8).

    Args:
        provider_key: The selected provider.
        readiness: Its ``ProviderReadiness``.

    Returns:
        Paste the key, where to get one and the skip; then, in italics, the
        environment and config.toml routes ("" while a URL is what's missing).
    """
    from tldw_chatbook.Chat.provider_catalog import provider_display_name
    from tldw_chatbook.Chat.provider_setup_facts import (
        key_help_text,
        provider_setup_facts,
    )

    display = provider_display_name(provider_key)
    if missing_url_reason(readiness):  # the URL field's own line names it (F8)
        return Text("")
    article = "an" if display[:1].lower() in "aeiou" else "a"
    parts = [f"Needs {article} {display} API key — paste it above (Ctrl+U clears the field)."]
    parts.append(key_help_text(provider_setup_facts(provider_key), display))
    parts.append(f"No key yet? Choose {SKIP_LABEL} or press Next.")
    line = Text(" ".join(parts))
    env_var = getattr(readiness, "env_var", "") or ""
    if env_var:
        line.append(f"\nOr export {env_var}, or add api_key to config.toml.", style="italic")
    return line


class ProviderKeyEntryStep(ProviderDetectionStep):
    """Key-entry half of the Provider step: cleaning, echo, reveal, hints.

    A ``SetupStep`` subclass (via the detection step) so Textual registers
    these ``@on`` handlers; ``ProviderStep`` inherits from it.
    """

    _key_hint_provider = ""
    _key_note_echo = False  # the note under the key carries the paste echo

    def _key_field(self) -> Input:
        return self.query_one("#setup-provider-api-key", Input)

    def _set_key_note(
        self, text: str, *, error: bool = False, hide: bool = False, echo: bool = False
    ) -> None:
        """Write the line under the key field.

        Once shown, the row keeps its height when emptied (until ``hide``):
        clearing it on blur must not shift the buttons below it between a
        click's press and release (lessons-live-verification: a control that
        moves when the form changes produces phantom dead clicks). ``echo``
        marks a line carrying the paste echo, the one blur clears.
        """
        try:
            note = self.query_one("#setup-provider-key-note", Static)
        except NoMatches:
            return
        self._key_note_echo = echo and bool(text)
        note.update(text)
        note.set_class(error, "-error")
        note.display = bool(text) or (note.display and not hide)

    def clear_key_echo(self) -> None:
        """Drop the paste echo (on blur and on Next): it shows key characters."""
        self._set_key_note("")

    @on(ProviderKeyInput.Pasted)
    def _key_pasted(self, event: ProviderKeyInput.Pasted) -> None:
        event.stop()
        cleaned = clean_api_key(self._key_field().value)
        if cleaned.non_ascii:
            self._set_key_note(non_ascii_message(cleaned.non_ascii), error=True)
            return
        self._set_key_note(
            " ".join(part for part in (event.echo, event.note) if part), echo=bool(event.echo)
        )

    @on(Input.Changed, "#setup-provider-api-key")
    def _key_field_cleaned(self, event: Input.Changed) -> None:
        """Strip invisible characters as they arrive; flag non-ASCII at once."""
        value = event.value
        stripped = "".join(
            character
            for character in value
            if character == " "
            or unicodedata.category(character) not in _STRIPPED_CATEGORIES
        )
        if stripped != value:
            removed = len(value) - len(stripped)
            event.input.value = stripped
            self._set_key_note(
                f"Removed {_plural(removed, 'invisible character')} from the key."
            )
            return
        non_ascii = sum(1 for character in value if not character.isascii())
        if non_ascii:
            self._set_key_note(non_ascii_message(non_ascii), error=True)
        elif not value and not event.input.value:  # still empty: not a stale event
            self.clear_key_echo()
        self._sync_key_hint(value)

    @on(Input.Blurred, "#setup-provider-api-key")
    def _key_field_blurred(self, event: Input.Blurred) -> None:
        # Only the paste echo (it shows key characters) clears on blur; a
        # notice such as the switch's 'it replaced the key you had typed'
        # stays (review round 1, live F14).
        if self._key_note_echo:
            self.clear_key_echo()

    def _sync_key_hint(self, value: str) -> None:
        """Offer to move a key whose prefix belongs to another provider (E11)."""
        from tldw_chatbook.Chat.provider_catalog import provider_display_name

        selected = getattr(self, "selected_provider_key", "")
        target = suggest_provider_for_key(value.strip(), selected) if selected else ""
        self._key_hint_provider = target
        try:
            hint = self.query_one("#setup-provider-key-hint", Horizontal)
        except NoMatches:
            return
        hint.display = bool(target)
        if not target:
            return
        display = provider_display_name(target)
        self.query_one("#setup-provider-key-hint-text", Static).update(
            f"This looks like an {display} key — switch to {display}?"
            if display[:1].lower() in "aeiou"
            else f"This looks like a {display} key — switch to {display}?"
        )
        self.query_one("#setup-provider-key-switch", Button).label = (
            f"Switch to {escape_markup(display)}"  # TASK-1513: labels parse markup
        )

    @on(Button.Pressed, "#setup-provider-key-switch")
    def _key_switch_pressed(self, event: Button.Pressed) -> None:
        """Move the typed key to the provider its prefix belongs to."""
        event.stop()
        target = self._key_hint_provider
        if not target:
            return
        field = self._key_field()
        value = field.value
        # Live F14: a key typed for the target earlier was replaced silently.
        earlier = getattr(getattr(self, "_provider_drafts", {}).get(target), "api_key", "")
        field.value = ""
        self._pick_provider_key(target)
        if getattr(self, "selected_provider_key", "") != target:
            field.value = value  # the switch failed: the key stays put
            return
        field = self._key_field()
        field.display = True
        self._sync_key_companions()
        field.value = value
        field.focus(scroll_visible=False)
        # Live (G6-VR2-08's fix, g6-r2fix/06): a cleared filter grows the list
        # and pushes the field down; scroll to it once that is laid out.
        self.call_after_refresh(self._reveal_after_layout, field, False)
        from tldw_chatbook.Chat.provider_catalog import provider_display_name

        # Review round 2, G6-VR2-08: say where the key went (the list may not).
        display = provider_display_name(target)
        if earlier and earlier != value:
            self._set_key_note(
                f"Moved the key to {display}, replacing the key you had typed there."
            )
        else:
            self._set_key_note(f"Moved the key to {display}.")

    def _pick_provider_key(self, provider_key: str) -> None:
        """Select ``provider_key`` through the step's guarded pick path."""
        from tldw_chatbook.UI.Wizards import first_run_step_guard as step_guard

        with step_guard.provider_switch(self, provider_key):
            if provider_key != getattr(self, "selected_provider_key", ""):
                self.select_provider(provider_key)  # type: ignore[attr-defined]
        try:
            choices = self.query_one("#setup-provider-choice")
            option_id = f"provider-{provider_key}"
            if getattr(self, "_list_query", "") and not any(
                choices.get_option_at_index(index).id == option_id
                for index in range(choices.option_count)
            ):  # G6-VR2-08: a filter that hides the row moved to is cleared
                field = self.query_one("#setup-provider-filter", Input)
                with field.prevent(Input.Changed):
                    field.value = ""
                self._list_query = ""
                self._rebuild_provider_list()  # type: ignore[attr-defined]
            choices.highlighted = choices.get_option_index(option_id)
        except Exception:  # noqa: BLE001 - a hidden row keeps its highlight.
            pass

    def _sync_key_companions(self) -> None:
        """The 'API key' label and Show come and go with the field.

        Review round 2, G6-VR2-11 (1): with the environment source chosen the
        field was hidden but its label and Show stayed (Show flipped only its
        own label).
        """
        try:
            shown = bool(self._key_field().display)
            for selector in ("#setup-provider-key-label", "#setup-provider-key-reveal"):
                self.query_one(selector).display = shown
        except NoMatches:
            return

    @on(Button.Pressed, "#setup-provider-key-reveal")
    def _key_reveal_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        field = self._key_field()
        field.password = not field.password
        event.button.label = "Show" if field.password else "Hide"

    def _sync_env_source_choice(self, env_var: str) -> None:
        """Offer 'Use <ENV> from your environment' or 'Store a different key'.

        Review provider-19: with an env key exported the key field used to
        vanish, so a rejected env key could not be replaced in the wizard.
        """
        try:
            choice = self.query_one("#setup-provider-key-source", RadioSet)
        except NoMatches:
            return
        choice.display = bool(env_var)
        if not env_var:
            return
        # Short enough for 120 columns (live F12); the line under the key
        # says it must be set in the shell that starts chatbook.
        self.query_one("#setup-provider-key-source-env", SetupRadioButton).label = (
            f"Use {escape_markup(env_var)} from your environment (found; not stored)"
        )
        # Review round 2, G6-VR2-04: only ever turn the wanted button ON.
        # RadioSet turns the other off itself; a pressed button set False
        # directly is reverted ('clicked off'), which left both radios on and
        # 'Store a different key' dead after a switch between exported keys.
        storing = bool(self._key_field().display)
        wanted = "store" if storing else "env"
        target = self.query_one(f"#setup-provider-key-source-{wanted}", SetupRadioButton)
        if not target.value:
            target.value = True  # its Changed finds the choice already applied

    @on(RadioSet.Changed, "#setup-provider-key-source")
    def _env_source_changed(self, event: RadioSet.Changed) -> None:
        event.stop()
        storing = event.pressed.id == "setup-provider-key-source-store"
        if storing == bool(self._key_field().display):
            return  # a sync of what is already on screen, not a choice
        if storing:
            self._on_replace()  # type: ignore[attr-defined]
            self._key_field().focus()
        else:
            self._on_keep()  # type: ignore[attr-defined]

    def _focus_key_after_pick(self) -> None:
        """An explicit pick (Enter, Space, click) of a keyed provider goes to its key.

        A provider whose exported key is used shows its environment choice
        instead (focus stays on the list). Review round 1, live F5: a click
        picks in one go, and the focus landed before the expanded section was
        laid out -- Textual's focus scroll aimed at an empty region and the
        user typed into a field below the fold. The target is scrolled into
        view once it has a place on screen.
        """
        try:
            field = self._key_field()
            readiness = self._current_provider_readiness()  # type: ignore[attr-defined]
            choice = self.query_one("#setup-provider-key-source", RadioSet)
        except Exception:  # noqa: BLE001 - focus is a convenience, never fatal.
            return
        if field.display and readiness.requires_api_key and not readiness.ready:
            self.call_after_refresh(self._reveal_after_layout, field, True)
        elif choice.display and not field.display:
            self.call_after_refresh(self._reveal_after_layout, choice, False)

    def _reveal_after_layout(self, widget: Widget, focus: bool, tries: int = 0) -> None:
        """Focus ``widget`` (if asked), then scroll it on screen once laid out."""
        if focus and tries == 0:
            widget.focus(scroll_visible=False)
        if not widget.region and tries < _LAYOUT_WAIT_REFRESHES:  # not laid out yet
            self.call_after_refresh(self._reveal_after_layout, widget, focus, tries + 1)
            return
        widget.scroll_visible(animate=False)

    def _sync_auth_title_focus(self, required: bool) -> None:
        """The 'Authentication' header is no Tab stop while a key is required."""
        try:
            auth = self.query_one("#setup-provider-auth-toggle", Collapsible)
        except NoMatches:
            return
        for title in auth.query("CollapsibleTitle"):
            title.can_focus = not (required and not auth.collapsed)


def replace_key(step: Any) -> None:
    """Reveal the masked input so the user can type a new key (Replace).

    Leaving it blank after Replace is a cancel: commit() only persists a
    typed, non-empty value, so the currently-configured secret is left
    untouched (never re-shown). 'Store a different key for this app' runs
    this too. Review round 2, G6-VR2-11 (2): the pinned refusal that named
    this control is taken down once it has been used.
    """
    key_input = step.query_one("#setup-provider-api-key", Input)
    changed = step._clear_requested or not key_input.display
    step._clear_requested = False
    key_input.display = True
    getattr(step.wizard, "_clear_pinned_step_error", lambda: None)()
    if changed:
        step._credential_semantics_changed()


def clean_key_for_commit(step: Any) -> str:
    """Drop the paste echo and clean a typed key before Next reads it.

    A key typed (not pasted) with quotes or a ``Bearer `` prefix is cleaned
    here, so what is staged is what the provider expects. A key that still
    can't go on the wire (a smart dash) is refused: review round 2, R2-06
    found it dropped silently, so the credential fell back to an exported
    or saved key and Next advanced with that one instead.

    Returns:
        Commit's refusal for a typed key that can't be sent, else "".
    """
    step.clear_key_echo()  # then the cleaning note, if any (review F10)
    try:
        field = step._key_field()
    except NoMatches:
        return ""
    cleaned = clean_api_key(field.value)
    if cleaned.value != field.value and not cleaned.non_ascii:
        field.value = cleaned.value
        # Typed, not pasted (review round 1, F10): say 'the key'.
        step._set_key_note(cleaned.note.replace("the pasted key", "the key"))
    typed = field.value
    if not field.display or not typed.strip() or usable_typed_key(typed):
        return ""
    message = non_ascii_message(sum(1 for character in typed if not character.isascii()))
    step._set_key_note(message, error=True)
    return message
