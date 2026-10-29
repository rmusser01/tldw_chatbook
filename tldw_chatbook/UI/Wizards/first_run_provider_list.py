"""The first-run Provider step's list (TASK-34100.6 slices D and E).

Review findings provider-11 (60 providers, unsearchable, in a 5-row box with
no descriptions), provider-12 (legacy aliases and '#2' looked like duplicates),
provider-02 (a detected server was lost on any pick) and enhancement E2 ('Ready
on this machine' first). The list is built from the shared catalog record and
its setup facts:

* a 'Ready on this machine' group first: each detected local server (one row
  per port, pre-highlighted so Enter uses it) and each provider whose key is
  already exported or saved;
* then Popular, Cloud, Local, Other, each row saying where it runs;
* legacy aliases and the '#2' slot behind 'Show legacy entries (N)' unless the
  config already uses them -- the filter still finds them;
* a filter that matches like Settings' provider search; letters typed on the
  list go to it while the list keeps focus, so a burst is never split between
  two widgets (lessons-textual: a focus move mid-burst misroutes keys).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import HorizontalGroup
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.UI.Wizards.first_run_provider_key_entry import (
    BurstSafeInput,
    ProviderKeyEntryStep,
)
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import ProviderChoiceOption

READY_GROUP = "Ready on this machine"
SHOW_LEGACY_ID = "show-legacy"
FILTER_PLACEHOLDER = "Filter providers — name or ID"
#: Providers most first-time users are looking for, in display order (TASK-1498).
POPULAR_PROVIDER_KEYS = ("openai", "anthropic", "ollama", "llama_cpp")


def provider_group_option_id(title: str) -> str:
    """Return the deterministic option ID for a provider group heading."""
    return "group-" + "-".join(title.casefold().split())


class ProviderChoiceList(OptionList):
    """Provider options with Space activation and post-navigation signaling.

    Letters typed on the list are posted as ``FilterKey`` (review provider-11)
    and Backspace edits the filter; Space stays the select binding.
    """

    BINDINGS = [
        Binding("space", "select", "Select", show=False),
        # TASK-34332: Screen's Tab is not priority, so keys typed right after
        # Tab still reached this list (End jumped it to the last row).
        Binding("tab", "app.focus_next", show=False, priority=True),
        Binding("shift+tab", "app.focus_previous", show=False, priority=True),
    ]

    class Interacted(Message):
        """A keyboard navigation action has resolved against the list."""

    class FilterKey(Message):
        """A letter (or ``"\\b"`` for Backspace) typed on the list."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class KeyPasted(Message):
        """A paste on the list: it is a key, never filter text (G6-VR2-03)."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self._text = text

        def __repr__(self) -> str:  # never the pasted text
            return f"KeyPasted(length={len(self._text)})"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._provider_interaction_ready = False
        super().__init__(*args, **kwargs)
        self._provider_interaction_ready = True

    def _on_paste(self, event: events.Paste) -> None:
        event.stop()
        event.prevent_default()
        if event.text:
            self.post_message(self.KeyPasted(event.text))

    def on_key(self, event: events.Key) -> None:
        """Send printable keys (never Space) and Backspace to the filter."""
        if event.key == "backspace":
            text = "\b"
        elif event.character and event.is_printable and not event.character.isspace():
            text = event.character
        else:
            return
        event.stop()
        event.prevent_default()
        self.post_message(self.FilterKey(text))

    def _post_navigation_interaction(self) -> None:
        if self._provider_interaction_ready:
            self.post_message(self.Interacted())

    def action_cursor_up(self) -> None:
        super().action_cursor_up()
        self._post_navigation_interaction()

    def action_cursor_down(self) -> None:
        super().action_cursor_down()
        self._post_navigation_interaction()

    def action_first(self) -> None:
        super().action_first()
        self._post_navigation_interaction()

    def action_last(self) -> None:
        super().action_last()
        self._post_navigation_interaction()

    def _keep_highlight(self, page_action: Any) -> None:
        previous_highlight = self.highlighted
        previous_option = self.highlighted_option
        page_action()
        if (
            self.highlighted is None
            and previous_highlight is not None
            and previous_option is not None
            and not previous_option.disabled
        ):
            self.highlighted = previous_highlight
        self._post_navigation_interaction()

    def action_page_up(self) -> None:
        self._keep_highlight(super().action_page_up)

    def action_page_down(self) -> None:
        self._keep_highlight(super().action_page_down)

    def watch_highlighted(self, highlighted: int | None) -> None:
        """Keep the row being browsed on screen while the list has focus.

        Review round 1, live F10 (reproduced: g6-r1fix/02): with the step
        scrolled down -- a mouse wheel, or a section a pick scrolled to --
        the arrow keys went on browsing (selecting rows, rewriting the
        endpoint) while every row was above the step's viewport.

        Review round 2 (R2-01): Textual resolves the watcher with getattr, so
        this override replaces OptionList's own -- which scrolls the list to
        the row and posts OptionHighlighted. It must run first.
        """
        super().watch_highlighted(highlighted)
        if highlighted is not None and self.has_focus:
            self.call_after_refresh(self._keep_highlighted_row_on_screen)

    def _keep_highlighted_row_on_screen(self, tries: int = 0) -> None:
        if self.highlighted is None or not self.has_focus:
            return
        try:
            line = self._index_to_line[self.highlighted]
            clip = self.screen.find_widget(self).visible_region
        except Exception:  # noqa: BLE001 - not laid out yet: nothing to keep.
            return
        y = self.region.y + self.gutter.top + line - int(self.scroll_offset.y)
        if clip.y <= y < clip.bottom:
            return
        # Review v3r1, G6-BV3-01: at 80x24 the step shows ~5 of the list's 8
        # rows, so bringing the list's top on screen left a lower row hidden.
        # Scroll the step by what the row itself is outside the view.
        scroller = next((a for a in self.ancestors_with_self[1:] if getattr(a, "allow_vertical_scroll", False)), None)
        if scroller is None or not clip.height:
            self.scroll_visible(animate=False)  # wholly off screen: list first
            if tries < 2:
                self.call_after_refresh(self._keep_highlighted_row_on_screen, tries + 1)
            return
        delta = y - clip.bottom + 1 if y >= clip.bottom else y - clip.y
        scroller.scroll_relative(y=delta, animate=False, immediate=True)


class FoundServerOption(ProviderChoiceOption):
    """A detected local server: picking it uses that server (review provider-02)."""

    def __init__(self, prompt: Text, *, option_id: str, provider_key: str, server: object):
        super().__init__(prompt, option_id=option_id, provider_key=provider_key)
        self.server = server


class LegacyRevealOption(Option):
    """'Show legacy entries (N)': selecting it lists the hidden rows."""

    provider_key = None

    def __init__(self, count: int) -> None:
        super().__init__(
            Text(f"Show legacy entries ({count})", style="italic"), id=SHOW_LEGACY_ID
        )


@dataclass(frozen=True, slots=True)
class ListBuild:
    """The options to show and the count line above them."""

    options: list[Option]
    count_line: str
    hidden_legacy: int


def grouped_sections(entries: Sequence[Any]) -> list[tuple[str, list[Any]]]:
    """Popular, then Cloud, Local, Other (empty sections dropped)."""
    from tldw_chatbook.Chat.provider_catalog import PROVIDER_CUSTOM_GROUP_KEYS

    by_key = {entry.readiness_key: entry for entry in entries}
    popular = [by_key[key] for key in POPULAR_PROVIDER_KEYS if key in by_key]
    popular_keys = {entry.readiness_key for entry in popular}
    rest = [entry for entry in entries if entry.readiness_key not in popular_keys]

    def alpha(entry: Any) -> str:
        return entry.display_name.lower()

    def other(entry: Any) -> bool:
        return entry.readiness_key in PROVIDER_CUSTOM_GROUP_KEYS

    sections = [
        ("Popular", popular),
        ("Cloud", sorted((e for e in rest if e.requires_api_key and not other(e)), key=alpha)),
        ("Local", sorted((e for e in rest if not e.requires_api_key and not other(e)), key=alpha)),
        ("Other", sorted((e for e in rest if other(e)), key=alpha)),
    ]
    return [(title, group) for title, group in sections if group]


def where_it_runs(entry: Any) -> str:
    """The dim second column: whether the row runs here or needs a cloud key."""
    if entry.readiness_key in {"custom", "custom_2"}:
        return "This computer or your server"
    facts = getattr(entry, "setup_facts", None)
    if (facts is not None and facts.locality == "local") or not entry.requires_api_key:
        return "This computer"
    return "Cloud · API key"


def _row_prompt(label: str, detail: str) -> Text:
    return Text.assemble(label, (f"  · {detail}" if detail else "", "dim"))


def _pluralise(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def server_address(server: Any) -> str:
    """'127.0.0.1:9099' for a detected server's base URL."""
    try:
        parts = urlsplit(str(getattr(server, "base_url", "")))
        host = parts.hostname or "this computer"
        return f"{host}:{parts.port}" if parts.port else host
    except ValueError:
        return "this computer"


def server_engine(server: Any) -> str:
    """The engine a detected server runs, e.g. 'llama.cpp' or 'LM Studio'.

    Only an engine name discovery itself assigns is shown; anything else
    falls back to the row's catalog name, never raw server-side text.
    """
    from tldw_chatbook.Chat.local_server_discovery import ENGINE_DISCOVERY_DEFAULTS
    from tldw_chatbook.Chat.provider_catalog import provider_display_name

    engine = getattr(server, "engine", "")
    if engine in {name for _key, _url, name in ENGINE_DISCOVERY_DEFAULTS}:
        return engine
    return provider_display_name(getattr(server, "provider_key", ""))


def usable_found_server(server: Any) -> bool:
    """A detected server the list may show: a known row and a valid endpoint."""
    from tldw_chatbook.Chat.console_provider_support import (
        resolve_console_provider_identity,
    )
    from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
    from tldw_chatbook.Chat.provider_endpoint_contract import resolve_provider_endpoint
    from tldw_chatbook.Chat.provider_setup_facts import provider_setup_facts

    if type(server) is not DiscoveredLocalServer:
        return False
    try:
        key = resolve_console_provider_identity(server.provider_key).readiness_key
    except (TypeError, ValueError):
        return False
    if key != server.provider_key or provider_setup_facts(key).locality != "local":
        return False
    return resolve_provider_endpoint(key, server.base_url).persisted_endpoint is not None


def found_server_label(server: Any) -> str:
    """'llama.cpp on this computer · 127.0.0.1:9099 · 3 models'."""
    models = len(tuple(getattr(server, "model_ids", ()) or ()))
    return (
        f"{server_engine(server)} on this computer · {server_address(server)}"
        f" · {_pluralise(models, 'model')}"
    )


def matches(entry_label: str, provider_id: str, query: str) -> bool:
    """Settings' provider-search rule (label or ID, case-insensitive)."""
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        provider_matches_query,
    )

    return provider_matches_query(entry_label, provider_id, query)


def ready_badges(
    entries: Sequence[Any],
    app_config: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Providers whose key is already exported or saved, with a text badge.

    Cheap, secret-free reads (an env-var presence check and a config lookup
    per keyed row), run after the step's first paint so entry never waits
    (review provider-11: its own increment, within the ADR-097 budgets).

    Args:
        entries: The catalog entries listed.
        app_config: The loaded configuration.
        environ: The environment to read; the process environment by default.

    Returns:
        provider key -> badge such as 'key in OPENAI_API_KEY'.
    """
    from tldw_chatbook.config import is_valid_provider_api_key

    environ = os.environ if environ is None else environ
    api_settings = app_config.get("api_settings") if isinstance(app_config, Mapping) else {}
    api_settings = api_settings if isinstance(api_settings, Mapping) else {}
    badges: dict[str, str] = {}
    for entry in entries:
        facts = getattr(entry, "setup_facts", None)
        if not entry.requires_api_key or facts is None or facts.legacy:
            continue
        table = api_settings.get(entry.readiness_key)
        if isinstance(table, Mapping) and is_valid_provider_api_key(table.get("api_key")):
            badges[entry.readiness_key] = "key saved"
            continue
        env_var = facts.env_var
        if env_var and is_valid_provider_api_key(environ.get(env_var)):
            badges[entry.readiness_key] = f"key in {env_var}"
    return badges


def _visible_entries(
    entries: Sequence[Any], query: str, show_legacy: bool, in_use: frozenset[str]
) -> tuple[list[Any], int]:
    hidden_legacy = 0
    visible: list[Any] = []
    for entry in entries:
        facts = getattr(entry, "setup_facts", None)
        legacy = bool(facts and facts.legacy) and entry.readiness_key not in in_use
        if query:
            if matches(entry.display_name, entry.readiness_key, query):
                visible.append(entry)
        elif legacy and not show_legacy:
            hidden_legacy += 1
        else:
            visible.append(entry)
    return visible, hidden_legacy


def _ready_rows(
    entries: Sequence[Any], query: str, ready: Mapping[str, str], found: Sequence[Any]
) -> list[Option]:
    by_key = {entry.readiness_key: entry for entry in entries}
    rows: list[Option] = []
    for index, server in enumerate(found):
        label = found_server_label(server)
        key = getattr(server, "provider_key", "")
        if query and not matches(label, key, query):
            continue
        rows.append(
            FoundServerOption(Text(label), option_id=f"found-{index}", provider_key=key, server=server)
        )
    for key, badge in ready.items():
        entry = by_key.get(key)
        if entry is None or (query and not matches(entry.display_name, key, query)):
            continue
        rows.append(
            ProviderChoiceOption(
                _row_prompt(entry.display_name, badge), option_id=f"ready-{key}", provider_key=key
            )
        )
    return rows


def _heading(title: str) -> ProviderChoiceOption:
    return ProviderChoiceOption(
        Text(title, style="bold"), option_id=provider_group_option_id(title), provider_key=None
    )


def build_provider_options(
    entries: Sequence[Any],
    *,
    query: str = "",
    show_legacy: bool = False,
    in_use: frozenset[str] = frozenset(),
    ready: Mapping[str, str] | None = None,
    found: Sequence[Any] = (),
) -> ListBuild:
    """Build the provider list's options and count line.

    Args:
        entries: Catalog entries (``ConsoleProviderCatalogEntry``).
        query: The filter text.
        show_legacy: Whether hidden legacy rows were revealed.
        in_use: Legacy keys the config already uses (always listed).
        ready: provider key -> badge, for the Ready group.
        found: Detected local servers, one per port.

    Returns:
        The options, the count line and how many legacy rows are hidden.
    """
    query = query.strip()
    visible, hidden_legacy = _visible_entries(entries, query, show_legacy, in_use)
    options: list[Option] = []
    ready_rows = _ready_rows(entries, query, ready or {}, found)
    if ready_rows:
        options.append(_heading(READY_GROUP))
        options.extend(ready_rows)
    for title, group in grouped_sections(visible):
        options.append(_heading(title))
        options.extend(
            ProviderChoiceOption(
                _row_prompt(entry.display_name, where_it_runs(entry)),
                option_id=f"provider-{entry.readiness_key}",
                provider_key=entry.readiness_key,
            )
            for entry in group
        )
    if hidden_legacy:
        options.append(LegacyRevealOption(hidden_legacy))
    total = len(entries)
    if not query:
        count_line = f"{total} providers · type to filter"
    elif visible:
        count_line = f"{len(visible)} of {total} match “{query}”"
    elif ready_rows:  # only a found server matched (review round 1, live F9)
        servers = len(ready_rows)
        noun = "local server matches" if servers == 1 else "local servers match"
        count_line = f"{servers} {noun} “{query}” · clear to see all {total}"
    else:
        count_line = f"No match · clear to see all {total}"
    return ListBuild(options, count_line, hidden_legacy)


def provider_options(entries: Sequence[Any]) -> list[Option]:
    """The unfiltered list with every row (legacy included), no Ready group."""
    return build_provider_options(entries, show_legacy=True).options


def legacy_keys_in_use(app_config: Mapping[str, Any], resumed_provider: str = "") -> frozenset[str]:
    """Legacy keys the user already relies on, which stay listed (provider-12)."""
    from tldw_chatbook.Chat.provider_setup_facts import LEGACY_PROVIDER_KEYS
    from tldw_chatbook.config import normalize_provider_config_key

    defaults = app_config.get("chat_defaults") if isinstance(app_config, Mapping) else {}
    provider = defaults.get("provider") if isinstance(defaults, Mapping) else ""
    keys = {normalize_provider_config_key(str(provider or "")), resumed_provider}
    return frozenset(key for key in keys if key in LEGACY_PROVIDER_KEYS)


def one_row_per_port(servers: Sequence[Any]) -> tuple[Any, ...]:
    """Merge 127.0.0.1/localhost/::1 spellings of one server (review provider-22).

    Both spellings are probed (a server bound to only one answers on only
    one); the first that answered, in probe order, is kept.
    """
    kept: list[Any] = []
    seen: set[tuple[str, int | None, str]] = set()
    for server in servers:
        try:
            parts = urlsplit(str(getattr(server, "base_url", "")))
            host = parts.hostname or ""
            port = parts.port
        except ValueError:
            continue
        if host in {"localhost", "::1"}:
            host = "127.0.0.1"
        identity = (host, port, parts.path.rstrip("/"))
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(server)
    return tuple(kept)


class ProviderListStep(ProviderKeyEntryStep):
    """The list half of the Provider step: filter, groups, Ready rows."""

    _list_entries: tuple[Any, ...] = ()
    _list_query = ""
    _show_legacy = False
    _legacy_in_use: frozenset[str] = frozenset()
    _ready_badges: Mapping[str, str] = MappingProxyType({})
    _found_servers: tuple[Any, ...] = ()

    def _list_app_config(self) -> Mapping[str, Any]:
        app_instance = getattr(getattr(self, "wizard", None), "app_instance", None)
        config = getattr(app_instance, "app_config", None)
        return config if isinstance(config, Mapping) else {}

    def _build_list(self) -> ListBuild:
        # Pre-merge review (2026-10-29), minor 2: the row Next would commit is
        # in use, so clearing the filter that found it never hides it.
        selected = legacy_keys_in_use({}, getattr(self, "selected_provider_key", ""))
        return build_provider_options(
            self._list_entries,
            query=self._list_query,
            show_legacy=self._show_legacy,
            in_use=self._legacy_in_use | selected,
            ready=self._ready_badges,
            found=self._found_servers,
        )

    def compose_provider_list(self) -> ComposeResult:
        """The filter, its count line and the list (review provider-11)."""
        from tldw_chatbook.UI.Wizards import first_run_step_guard as step_guard

        self._list_entries = tuple(step_guard.first_run_provider_catalog())  # 33621.14
        self._legacy_in_use = legacy_keys_in_use(
            self._list_app_config(), self._resumed_provider_key()
        )
        build = self._build_list()
        with HorizontalGroup(classes="setup-provider-filter-row"):  # one row at 80x24
            yield BurstSafeInput(
                placeholder=FILTER_PLACEHOLDER,
                id="setup-provider-filter",
                classes="setup-provider-filter",
                compact=True,
            )
            yield Static(  # echoes the typed filter: never markup (TASK-1513)
                build.count_line,
                id="setup-provider-count",
                classes="setup-provider-count setup-subtitle",
                markup=False,
            )
        yield ProviderChoiceList(
            *build.options, id="setup-provider-choice", classes="setup-choice-list"
        )
        # coverage-08: a typed key left behind is named here, under the list.
        yield Static("", id="setup-provider-unsaved-note", classes="setup-probe-status -warn hidden")

    def _resumed_provider_key(self) -> str:
        """The provider a resumed run restores; it stays listed (minor 2)."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_PROVIDER

        draft = getattr(getattr(self, "wizard", None), "resume_draft", None)
        values = getattr(draft, "values", None)
        provider = values.get(STEP_PROVIDER) if isinstance(values, Mapping) else None
        if not isinstance(provider, Mapping):
            return ""
        return str(provider.get("provider_key") or "")

    @on(events.Mount)
    def _list_mounted(self) -> None:
        # E2: readiness badges are their own increment, after first paint.
        self.call_after_refresh(self._refresh_ready_badges)

    def _refresh_ready_badges(self) -> None:
        environ = self._environment() if hasattr(self, "_environment") else None
        try:
            badges = ready_badges(self._list_entries, self._list_app_config(), environ)
        except Exception:  # noqa: BLE001 - badges are a hint; the list stands without them.
            return
        if badges != dict(self._ready_badges):
            self._ready_badges = MappingProxyType(badges)
            self._rebuild_provider_list()

    def _list_highlight_target(self, options: Sequence[Option], previous: str | None, prefer_found: bool) -> int | None:
        ids = [str(option.id) for option in options]
        enabled = [index for index, option in enumerate(options) if not option.disabled]
        if not enabled:
            return None
        wanted: list[str] = []
        if prefer_found and not self.selected_provider_key:
            wanted.append("found-0")
        if not self._list_query.strip():
            if previous:
                wanted.append(previous)
            if self.selected_provider_key:
                wanted.append(f"provider-{self.selected_provider_key}")
        for option_id in wanted:
            if option_id in ids and not options[ids.index(option_id)].disabled:
                return ids.index(option_id)
        return enabled[0]

    def _rebuild_provider_list(self, *, prefer_found: bool = False) -> None:
        """Rebuild the options without selecting anything (filtering browses)."""
        try:
            choices = self.query_one("#setup-provider-choice", ProviderChoiceList)
            count = self.query_one("#setup-provider-count", Static)
        except NoMatches:
            return
        build = self._build_list()
        previous = choices.highlighted_option
        previous_id = str(previous.id) if previous is not None else None
        with choices.prevent(OptionList.OptionHighlighted):
            choices.clear_options()
            choices.add_options(build.options)
            choices.highlighted = self._list_highlight_target(
                build.options, previous_id, prefer_found
            )
        count.update(build.count_line)

    def _show_found_servers(self, servers: Sequence[Any]) -> None:
        """Keep detected servers as the list's first rows (sticky, provider-02)."""
        merged = one_row_per_port(tuple(s for s in servers if usable_found_server(s)))
        if merged == self._found_servers:
            return
        self._found_servers = merged
        untouched = not getattr(self, "_provider_choice_interacted", False)
        self._rebuild_provider_list(prefer_found=untouched)
        try:
            choices = self.query_one("#setup-provider-choice", ProviderChoiceList)
        except NoMatches:
            return
        if untouched and merged and choices.has_focus:
            # Live at 80x24 the rows arrived after focus; keep them on screen.
            self.call_after_refresh(choices.scroll_visible, animate=False)

    def keep_browsed_row_in_view(self) -> bool:
        """While the list or its filter has focus, keep the browsed row on screen.

        Review v3r1, G6-BV3-01 (live at 120x40 and 80x24): browsing selects,
        and what a browsed row's check or fill does afterwards -- a keyless
        401 scrolling Authentication into view, the 'Using the … server
        found' line appearing above the list -- moved the row the user was
        on off screen while the list kept focus. Callers that would scroll
        the step ask this first.

        Browsing means the user has moved through the list: a provider picked
        before that (the step's first selection, ``select_provider``) still
        brings Authentication into view when its check asks for a key
        (TASK-33008 capture 06b).

        Returns:
            True while the user is browsing: nothing else may scroll the step.
        """
        if not getattr(self, "_provider_choice_interacted", False):
            return False
        try:
            choices = self.query_one("#setup-provider-choice", ProviderChoiceList)
            query = self.query_one("#setup-provider-filter", Input)
        except NoMatches:
            return False
        if not (choices.has_focus or query.has_focus):
            return False
        choices.call_after_refresh(choices._keep_highlighted_row_on_screen)
        return True

    def adopt_highlighted_found_server(self) -> str:
        """At Next with nothing picked, the highlighted found server is the pick.

        Pre-merge review (2026-10-29): a found server is the list's first row,
        "already highlighted" (User Guide), yet Next with nothing touched read
        the untouched list as nothing picked and skipped the step. Only
        commit asks this, so highlighting alone still never picks, and an
        explicit skip is never turned into a pick.

        Returns:
            The provider key now selected, or "" when the highlighted row is
            not a found server.
        """
        if getattr(self, "_skip_requested", False):
            return ""
        try:
            option = self.query_one("#setup-provider-choice", ProviderChoiceList).highlighted_option
        except NoMatches:
            return ""
        if not isinstance(option, FoundServerOption) or option.disabled:
            return ""
        self._select_provider_option(option)  # type: ignore[attr-defined]
        return getattr(self, "selected_provider_key", "")

    def found_server_for(self, provider_key: str) -> Any:
        """The first detected server for ``provider_key``, or None."""
        return next(
            (s for s in self._found_servers if getattr(s, "provider_key", "") == provider_key),
            None,
        )

    def found_server_to_fill(self, provider_key: str) -> Any:
        """The found server a hand-pick of ``provider_key`` may fill in, or None.

        Review round 1, F3: only where nothing the user chose is lost -- no
        endpoint configured, the shipped template's own default, or that
        found server itself (compared as connection identities, like
        Console's task-16476 rule: a different user endpoint is kept).
        """
        server = self.found_server_for(provider_key)
        if server is None:
            return None
        from tldw_chatbook.Chat.console_provider_endpoints import (
            first_configured_endpoint,
        )
        from tldw_chatbook.Chat.provider_endpoint_contract import (
            canonical_connection_identity,
        )
        from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

        configured = first_configured_endpoint(self._provider_settings(provider_key))  # type: ignore[attr-defined]
        if not configured:
            return server
        template_settings = (DEFAULT_CONFIG_FROM_TOML.get("api_settings") or {}).get(provider_key)
        template = first_configured_endpoint(template_settings or {})

        def identity(value: object) -> object:
            return canonical_connection_identity(provider_key, value)

        mine = identity(configured)
        if mine is None or mine in {identity(template), identity(server.base_url)}:
            return server
        return None

    def _handle_list_choice(self, option: Option) -> bool:
        """'Show legacy entries' reveals the hidden rows; True when handled."""
        if not isinstance(option, LegacyRevealOption):
            return False
        try:
            choices = self.query_one("#setup-provider-choice", ProviderChoiceList)
        except NoMatches:
            return True
        before = {str(choices.get_option_at_index(i).id) for i in range(choices.option_count)}
        self._show_legacy = True
        self._rebuild_provider_list()
        # Live: the rebuild fell back to the top row, so the rows just revealed
        # (at the list's end) were off screen. Land on the first of them.
        revealed = next(
            (
                index
                for index in range(choices.option_count)
                if str(choices.get_option_at_index(index).id) not in before
                and not choices.get_option_at_index(index).disabled
            ),
            None,
        )
        if revealed is not None:
            with choices.prevent(OptionList.OptionHighlighted):
                choices.highlighted = revealed
            choices.scroll_to_highlight()
        return True

    @on(Input.Changed, "#setup-provider-filter")
    def _on_filter_changed(self, event: Input.Changed) -> None:
        event.stop()
        if event.value != self._list_query:
            self._list_query = event.value
            self._rebuild_provider_list()

    @on(Input.Submitted, "#setup-provider-filter")
    def _on_filter_submitted(self, event: Input.Submitted) -> None:
        """Enter in the filter picks the highlighted match; it never advances.

        The wizard reads Enter in a step's Input as 'continue'; here it means
        'this one', through the list's own selection (as Enter on the list).
        """
        event.stop()
        choices = self.query_one("#setup-provider-choice", ProviderChoiceList)
        option = choices.highlighted_option
        if option is not None and not option.disabled:
            choices.focus()
            choices.action_select()

    @on(ProviderChoiceList.KeyPasted)
    def _on_list_paste(self, event: ProviderChoiceList.KeyPasted) -> None:
        """A paste on the list goes to the key field, never the filter.

        Review round 2, G6-VR2-03: with the list focused a terminal paste was
        dropped (a key typed there went to the filter, unmasked).
        """
        event.stop()
        field = self._key_field()
        try:
            keyed = bool(self._current_provider_readiness().requires_api_key)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - nothing picked yet.
            keyed = False
        if not (getattr(self, "selected_provider_key", "") and keyed and field.display):
            self._set_status("Pick a provider first, then paste its key.")
            return
        field.focus()
        field.post_message(events.Paste(event._text))

    @on(ProviderChoiceList.FilterKey)
    def _on_filter_key(self, event: ProviderChoiceList.FilterKey) -> None:
        event.stop()
        field = self.query_one("#setup-provider-filter", Input)
        value = field.value[:-1] if event.text == "\b" else field.value + event.text
        field.value = value
        field.cursor_position = len(value)
        self._list_query = value
        self._rebuild_provider_list()
