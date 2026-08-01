"""Export a briefing as a self-contained markdown document (spec #2 phase 3,
Task 1).

The UI's `#artifacts-toolbar` Export button (`UI/Watchlists_Modules/
artifacts_pane.py`) lets a user save the SELECTED briefing's body to a file
of their own choosing, through a `FileSave` dialog (`WatchlistsCollections
Screen.handle_export_briefing_requested`). This module is the pure half of
that flow: turning a `briefings` row into the document text, and turning
arbitrary (user- or model-authored) text into a filesystem-safe name. It
does no I/O of its own -- the screen validates the chosen destination
(`Utils.path_validation.validate_path_simple`) and writes the file, off the
event loop, itself.

`safe_export_stem` is reused verbatim by Task 4's feed-directory writer for
episode filenames, so its contract (alnum/space/-/_ only, a caller-supplied
fallback when nothing survives) is load-bearing for both callers, not just
this one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class BriefingExportError(RuntimeError):
    """Raised when a briefing has nothing worth exporting.

    The only raise site today is a `complete` row whose `body_markdown` is
    NULL or blank -- `briefing_service.generate_briefing` never records an
    empty body as `complete` (an empty provider response is recorded
    `failed` instead), so reaching this is a hand-edited or otherwise
    corrupted row, not a code path the service can produce. An empty file
    is not an export, so this refuses rather than writing nothing and
    calling it done.
    """


def safe_export_stem(text: str, *, fallback: str) -> str:
    """A filesystem-safe filename stem from arbitrary text.

    Whitelists alnum, space, `-` and `_` and drops everything else --
    including path separators, `..`, and any markup-shaped punctuation
    (`[`, `]`, `<`, `>`, `:`, quotes) a watchlist name or briefing title
    might contain. This is a whitelist, not a blacklist of "dangerous"
    characters, for the same reason `content_pane.render_article` gives for
    never hand-escaping model/source text: a blacklist fails open on
    whatever it did not anticipate, while a whitelist cannot.

    `text` may be empty, whitespace-only, or entirely made of characters
    outside the whitelist (e.g. `"###???"`, or a lone `".."`) -- any of
    which leaves nothing to build a name from, so `fallback` is returned
    verbatim in that case. `fallback` is trusted, caller-supplied plain
    text (every caller passes a short literal), so it is not itself run
    through the whitelist.

    Args:
        text: The candidate text -- a watchlist name, a briefing title, or
            similar user/model-authored free text.
        fallback: Returned verbatim when nothing in `text` survives the
            whitelist.

    Returns:
        A non-empty stem containing only alnum, space, `-` and `_`, with
        leading/trailing whitespace stripped.
    """
    cleaned = "".join(
        char for char in (text or "") if char.isalnum() or char in (" ", "-", "_")
    ).strip()
    return cleaned or fallback


def _coverage_window(briefing: Mapping[str, Any]) -> str:
    """What the briefing says it covers, in one line.

    Mirrors `artifacts_pane._window_text`'s own two-part shape (the
    `covers_from_ts` floor and the `covers_through_item_id` watermark), but
    is not that function: this module is beneath the UI layer (Subscriptions
    imports nothing from `UI/`), so the string is rebuilt here from the same
    two columns rather than importing a private UI helper.
    """
    parts: list[str] = []
    covers_from = briefing.get("covers_from_ts")
    if covers_from:
        parts.append(f"since {covers_from}")
    covers_through = briefing.get("covers_through_item_id")
    if covers_through not in (None, ""):
        parts.append(f"through item {covers_through}")
    return " · ".join(parts) if parts else "unknown"


def briefing_markdown_document(briefing: Mapping[str, Any]) -> str:
    """One briefing's body, as a standalone markdown document.

    A short front-matter header precedes the body verbatim, naming the
    four things a reader needs to place the document without the app
    around it: which watchlist it is from (`briefing["watchlist_name"]` --
    a `briefings` row itself only carries `watchlist_id`, so the caller is
    responsible for resolving and merging in the display name before
    calling this, exactly as `WatchlistsCollectionsScreen._watchlist_
    display_name` already does for every other briefing-scoped toast),
    its status, the window it covers, and when it was created.

    Args:
        briefing: A `briefings` row (as `dict(sqlite3.Row)`, or an
            equivalent mapping in tests), with `watchlist_name` merged in.

    Returns:
        The full document text, ready to write to a `.md` file verbatim.

    Raises:
        BriefingExportError: `briefing["body_markdown"]` is NULL, missing,
            or blank. Named error message includes the briefing's id so a
            toast built from `str(exc)` tells the user which row failed.
    """
    body = str(briefing.get("body_markdown") or "").strip()
    if not body:
        raise BriefingExportError(
            f"Briefing {briefing.get('id', 'unknown')} has no body to export."
        )

    watchlist_name = str(briefing.get("watchlist_name") or "").strip() or (
        "this watchlist"
    )
    status = str(briefing.get("status") or "").strip() or "unknown"
    created_at = str(briefing.get("created_at") or "").strip() or "unknown time"
    coverage = _coverage_window(briefing)

    front_matter = (
        "---\n"
        f"watchlist: {watchlist_name}\n"
        f"status: {status}\n"
        f"covers: {coverage}\n"
        f"created: {created_at}\n"
        "---\n\n"
    )
    return f"{front_matter}{body}\n"


def default_briefing_filename(
    briefing: Mapping[str, Any], *, watchlist_name: str
) -> str:
    """The filename a `FileSave` dialog opens with for this briefing.

    Built from the watchlist's name and the briefing's own timestamp, run
    through `safe_export_stem` so a watchlist named with path-shaped or
    markup-shaped text (see that function's own docstring) cannot escape
    the destination directory the user picks in the dialog, nor produce a
    stem with no visible characters at all.

    Args:
        briefing: A `briefings` row -- only `id`/`created_at` are read.
        watchlist_name: The watchlist's display name (resolved by the
            caller; a `briefings` row itself only carries `watchlist_id`).

    Returns:
        `"<stem>.md"`. The stem never contains `/` or `\\` (excluded by
        `safe_export_stem`'s whitelist), so this is always a bare filename,
        never a path.
    """
    created_at = str(briefing.get("created_at") or "").strip()
    stem_source = f"{watchlist_name} {created_at}".strip() if created_at else (
        watchlist_name
    )
    briefing_id = briefing.get("id")
    fallback = f"briefing-{briefing_id}" if briefing_id is not None else "briefing"
    stem = safe_export_stem(stem_source, fallback=fallback)
    return f"{stem}.md"
