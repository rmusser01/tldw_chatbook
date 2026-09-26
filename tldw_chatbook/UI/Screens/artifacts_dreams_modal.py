"""Dreams story detail modal (Dreams Phase 1, Task 7; ingest in Task 8).

One story's full surface on the Artifacts screen: title, source URL,
body, provenance, a "what we'll look for" preview of the NEXT cycle's
queries, and the Phase-1 action set -- keep/unkeep, dive deeper (stage a
``ChatHandoffPayload`` into Chat), export to Markdown, ingest into the
read-it-later capture queue (Task 8, via
``Dreams.ingest_action.ingest_story_url`` bound to the app's capture
service), more/less feedback, close.

Modal structure follows this stream's own modal idioms:

- Dismiss pattern and literal-text discipline from
  ``UI/Watchlists_Modules/kept_briefings_modal.py``: every story-derived
  string (title, URL, snippet, body) was written by an LLM from feed/web
  material neither this app nor the user chose, so it renders through
  ``rich.text.Text`` (never a bare ``str``, never markup-parsed) or
  ``rich.markdown.Markdown(..., hyperlinks=False)`` -- a link must never
  carry an invisible destination.
- Panel styling from the Artifacts screen's own sibling modal
  (``artifact_share_dialog.ArtifactShareDialog``): a small
  ``DEFAULT_CSS`` composed of existing theme variables (``$background``/
  ``$surface``/``$primary``/``$warning``) rather than any new tcss sheet,
  token, or hex literal (ADR-150: nothing new to govern).
- Handoff staging follows the ``skills_screen.attach_to_console`` idiom
  exactly: resolve ``open_chat_with_handoff`` off the running app with
  ``getattr``, warn-and-stay when the runtime lacks it, never invent
  navigation -- that app method owns staging AND the switch.

**Synthetic rows are a status view, not a story.** Task 6's
``dreams_view.list_recent_dreams`` also returns one synthetic row per
failed collection (``{"label": "Cycle <date>: failed", "synthetic": True}``
with NO story fields). The modal still opens for them -- a failed cycle
stays inspectable -- but offers Close only: every story action is a
silent no-op, the hint line advertises Close only, and nothing is ever
written for a row with no ``dream_stories`` id behind it.

**Ingest is http(s)-only.** LLM-knowledge stories (``dreams://llm/…`` URLs)
have no web destination for the read-it-later queue, so their modal hides
the Ingest button, drops it from the hint line, and the ``i`` keybinding
degrades to a gentle notice instead of submitting a doomed capture.

**All writes are single-row instant SQLite calls** made directly in the
action handlers (controller ruling, mirroring the kept-briefings modal):
``DreamsDB.set_story_kept``/``record_feedback`` are one-statement
transactions against a thread-local held connection, so no worker and no
``asyncio.to_thread`` hop is warranted. After ANY mutating action the
modal calls ``on_changed()`` (the Artifacts screen re-reads its rows) and
posts a dismissible notice; export/path errors surface as notices, never
crashes.
"""

from __future__ import annotations

import inspect
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Utils.path_validation import validate_filename, validate_path_simple
from ...Utils.timestamps import utc_now_iso

#: Same refusal as ``kept_briefings_modal._MARKDOWN_HYPERLINKS`` and the
#: Artifacts screen's own report preview: story bodies are LLM-written
#: from web material, so a link must paint literally, never navigate.
_MARKDOWN_HYPERLINKS = False

#: Export filename slug cap: generous for real headlines, bounded so a
#: pathologically long title cannot produce an unusable filename.
_SLUG_MAX_LENGTH = 40

#: The preview section's honest label: Phase 1 renders the fallback
#: (deterministic) queries for the CURRENT snapshot -- what the next
#: cycle searches if synthesis degrades -- and says so, rather than
#: calling an LLM from the modal.
_PREVIEW_LABEL = "preview (fallback queries until next cycle)"

_NO_DB_NOTICE = "Dreams is unavailable in this runtime: no dreams database."
_NO_CAPTURE_BACKEND_NOTICE = (
    "Read-it-later capture is unavailable in this runtime."
)
_HANDOFF_UNAVAILABLE_NOTICE = (
    "Console handoff is unavailable for Dreams in this runtime."
)
_SYNTHETIC_NOTICE = (
    "This discovery cycle failed and recorded no story to act on."
)
_UNCONFIRMED_INGEST_NOTICE = (
    "The server did not confirm this save; check read-it-later before "
    "retrying."
)
_NO_HTTP_URL_NOTICE = (
    "This story has no web URL to ingest."
)


def _ingestable(story: Mapping[str, Any]) -> bool:
    """Whether the story URL may enter the read-it-later queue.

    Ingest captures a WEB destination, so only http(s) URLs qualify: the
    synthetic ``dreams://llm/…`` rows (search failed; the story came
    straight from model knowledge) have nothing for the capture backend to
    fetch, and the ingest seam's own ``validate_url`` would refuse them
    anyway. Such rows hide the button, the hint, and the keybinding acts as
    a gentle notice.
    """
    try:
        parsed = urlparse(str(story.get("url") or "").strip())
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _dreams_export_dir() -> Path:
    """The sanctioned export root (module seam so tests can sink it)."""
    return Path.home() / "Documents" / "tldw_exports" / "dreams"


def _slugify(title: str) -> str:
    """Filename-safe slug for one story title (``A B -- C!`` -> ``a-b-c``)."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(title or "").strip().lower()).strip("-")
    return slug[:_SLUG_MAX_LENGTH] or "dream-story"


def _provenance_line(story: Mapping[str, Any]) -> Text:
    """The provenance line: matched topics, query, kind, event date if set."""
    line = Text()
    topics = story.get("matched_topics") or []
    if topics:
        line.append("topics: ", style="dim")
        line.append(", ".join(str(topic) for topic in topics))
    query = story.get("query")
    if query:
        if len(line):
            line.append(" · ")
        line.append("query: ", style="dim")
        line.append(str(query))
    kind = story.get("kind")
    if kind:
        if len(line):
            line.append(" · ")
        line.append("kind: ", style="dim")
        line.append(str(kind))
    event_date = story.get("event_date")
    if event_date:
        if len(line):
            line.append(" · ")
        line.append("event: ", style="dim")
        line.append(str(event_date))
    return line if len(line) else Text("no provenance recorded", style="dim")


def _detail_renderable(story: Mapping[str, Any]) -> RenderableType:
    """Header + body for one story (literal `Text`, hyperlink-free Markdown)."""
    header = Text()
    header.append(str(story.get("title") or "Untitled dream story"), style="bold")
    header.append("\n")
    header.append(str(story.get("url") or "no source URL"), style="dim")
    header.append("\n")
    header.append(_provenance_line(story))
    header.append("\n")
    body = str(story.get("body") or "").strip()
    if not body:
        return Group(header, Text("This story recorded no body."))
    return Group(header, Markdown(body, hyperlinks=_MARKDOWN_HYPERLINKS))


def _synthetic_renderable(story: Mapping[str, Any]) -> RenderableType:
    """The failed-cycle status view (synthetic rows carry no story fields)."""
    text = Text()
    text.append(str(story.get("label") or "Cycle: failed"), style="bold")
    text.append("\n")
    text.append(
        "This discovery cycle failed and recorded no story. "
        "The next cycle retries automatically.",
        style="dim",
    )
    return text


def _export_markdown(story: Mapping[str, Any]) -> str:
    """The export stub: title, source URL, provenance, body, generated date."""
    lines = [f"# {story.get('title') or 'Untitled dream story'!s}", ""]
    lines.append(f"- Source: {story.get('url') or 'unknown'}")
    topics = ", ".join(str(topic) for topic in story.get("matched_topics") or [])
    lines.append(f"- Matched topics: {topics or 'none'}")
    lines.append(f"- Query: {story.get('query') or 'unknown'}")
    lines.append(f"- Kind: {story.get('kind') or 'unknown'}")
    if story.get("event_date"):
        lines.append(f"- Event date: {story['event_date']}")
    lines.append(f"- Dreams story id: {story.get('id')}")
    generated = utc_now_iso()
    lines.append(f"- Generated: {generated}")
    lines.extend(["", str(story.get("body") or "").strip(), ""])
    return "\n".join(lines)


class DreamsStoryModal(ModalScreen[None]):
    """Inspect one Dreams story and act on it; always dismisses ``None``.

    Args:
        story: One ``dreams_view.list_recent_dreams`` row -- a full
            ``dream_stories`` row plus ``label``/``collection_date``, or
            one synthetic failed-cycle row (``synthetic`` truthy), for
            which the modal is a Close-only status view.
        dreams_db_getter: Zero-arg callable returning the app's
            ``DreamsDB`` or ``None``; called lazily at action time so a
            runtime without the database still renders the modal.
        capture_backend_getter: Zero-arg callable returning the app's
            read-it-later capture backend (``local_collections_capture_service``,
            a ``CollectionsCaptureBackend``) or ``None``; called lazily at
            action time so the Ingest action degrades to a notice rather
            than crashing a runtime without the capture service.
        on_changed: Zero-arg callback fired after every mutating action;
            the Artifacts screen re-reads its Dreams rows.
    """

    BINDINGS = (
        ("k", "keep", "Keep"),
        ("d", "dive", "Dive deeper"),
        ("e", "export", "Export"),
        ("i", "ingest", "Ingest"),
        ("m", "more", "More like this"),
        ("l", "less", "Less like this"),
        ("q", "close", "Close"),
        # Hidden: Escape is the safe-dismissal grammar (task-16211), not a
        # footer-advertised action -- the hint line stays exactly the seven.
        Binding("escape", "close", "Close", show=False),
    )

    # Mirrors ArtifactShareDialog's placement ruling: DEFAULT_CSS parses
    # at mount (runtime), not into the boot bundle, and uses only
    # existing theme variables -- zero new tokens or literals (ADR-150).
    DEFAULT_CSS = """
    DreamsStoryModal { align: center middle; background: $background 70%; }
    DreamsStoryModal > VerticalScroll {
        width: 76; max-width: 96%; height: auto; max-height: 90%;
        background: $surface; border: solid $primary; padding: 1 2;
    }
    DreamsStoryModal #dsm-hints { color: $warning; margin-top: 1; }
    """

    def __init__(
        self,
        story: dict[str, Any],
        *,
        dreams_db_getter: Callable[[], Any],
        capture_backend_getter: Callable[[], Any],
        on_changed: Callable[[], None],
    ) -> None:
        super().__init__()
        self._story = dict(story)
        self._synthetic = bool(story.get("synthetic"))
        self._dreams_db_getter = dreams_db_getter
        self._capture_backend_getter = capture_backend_getter
        self._on_changed = on_changed
        self._ingest_in_flight = False

    # --- Compose ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dsm-dialog"):
            if self._synthetic:
                yield Static(_synthetic_renderable(self._story), id="dsm-detail")
            else:
                yield Static(_detail_renderable(self._story), id="dsm-detail")
                yield Static("", id="dsm-preview")
            with Horizontal(id="dsm-actions"):
                if not self._synthetic:
                    yield Button(
                        "Unkeep (k)" if self._story.get("kept") else "Keep (k)",
                        id="dsm-keep-button",
                        compact=True,
                    )
                    yield Button(
                        "Dive deeper (d)", id="dsm-dive-button", compact=True
                    )
                    yield Button(
                        "Export (e)", id="dsm-export-button", compact=True
                    )
                    if _ingestable(self._story):
                        yield Button(
                            "Ingest (i)", id="dsm-ingest-button", compact=True
                        )
                    yield Button(
                        "More like this (m)", id="dsm-more-button", compact=True
                    )
                    yield Button(
                        "Less like this (l)", id="dsm-less-button", compact=True
                    )
                yield Button("Close (q)", id="dsm-close-button", compact=True)
            # ADR-031 rule 4: the hint line advertises EXACTLY the
            # implemented actions -- all seven for a story with an http(s)
            # URL, Close only for a synthetic failed-cycle row, and no
            # Ingest for a dreams://llm row whose action is gated off.
            yield Static(self._hints_text(), id="dsm-hints")

    def _hints_text(self) -> Text:
        hints = Text(style="dim")
        if not self._synthetic:
            hints.append("k Keep · d Dive deeper · e Export")
            if _ingestable(self._story):
                hints.append(" · i Ingest")
            hints.append(" · m More like this · l Less like this · ")
        hints.append("q Close")
        return hints

    def on_mount(self) -> None:
        if self._synthetic:
            return
        self._render_preview()

    def _render_preview(self) -> None:
        """Paint the next-cycle query preview (fallback, never an LLM call)."""
        queries = self._preview_queries()
        text = Text()
        text.append("What we'll look for ", style="bold")
        text.append(f"({_PREVIEW_LABEL})", style="dim")
        text.append("\n")
        if queries:
            for query in queries:
                text.append(f"- {query}\n")
        else:
            text.append("No interest profile yet; nothing to preview.", style="dim")
        self.query_one("#dsm-preview", Static).update(text)

    def _preview_queries(self) -> list[str]:
        """Fallback queries for the current snapshot, via the public seam."""
        db = self._db()
        if db is None:
            return []
        try:
            from ...Dreams.interest_profile import snapshot
            from ...Dreams.query_synthesis import preview_queries
            from ...Dreams.settings import dreams_setting

            snap = snapshot(db, now_epoch=time.time())
            topics = [str(topic["text"]) for topic in snap.get("topics", [])]
            try:
                count = int(dreams_setting("queries_per_cycle") or 3)
            except (TypeError, ValueError):
                count = 3
            return preview_queries(topics, max(1, count))
        except Exception as exc:  # noqa: BLE001 - degrade the section, not the modal
            logger.warning(
                f"Dreams query preview failed: {type(exc).__name__}"
            )
            return []

    # --- Helpers -----------------------------------------------------------

    def _db(self) -> Any:
        try:
            return self._dreams_db_getter()
        except Exception:  # noqa: BLE001 - a broken getter is a missing DB
            return None

    def _capture_backend(self) -> Any:
        try:
            return self._capture_backend_getter()
        except Exception:  # noqa: BLE001 - a broken getter is a missing backend
            return None

    def _story_id(self) -> int | None:
        story_id = self._story.get("id")
        if story_id is None:
            return None
        try:
            return int(story_id)
        except (TypeError, ValueError):
            return None

    def _record_feedback(self, kind: str) -> bool:
        """One feedback row; False when no DB/story or the write failed."""
        db = self._db()
        story_id = self._story_id()
        if db is None or story_id is None:
            self.notify(_NO_DB_NOTICE, severity="warning", markup=False)
            return False
        try:
            db.record_feedback(story_id, kind)
        except Exception as exc:  # noqa: BLE001 - a failed write is a notice
            logger.warning(f"Dreams feedback ({kind}) failed: {type(exc).__name__}")
            self.notify(
                f"Could not record {kind} feedback: {type(exc).__name__}",
                severity="error",
                markup=False,
            )
            return False
        return True

    def _changed(self) -> None:
        try:
            self._on_changed()
        except Exception as exc:  # noqa: BLE001 - a broken refresh must not crash
            logger.warning(f"Dreams on_changed callback failed: {type(exc).__name__}")

    # --- Actions (ADR-031 single letters; synthetic rows are no-ops) -------

    def action_keep(self) -> None:
        """Toggle kept + refresh the screen rows.

        ``kept`` feedback (a positive topic signal) is recorded only on the
        transition TO kept: unkeeping must never boost the story's topics.
        """
        if self._synthetic:
            self.notify(_SYNTHETIC_NOTICE, severity="warning", markup=False)
            return
        db = self._db()
        story_id = self._story_id()
        if db is None or story_id is None:
            self.notify(_NO_DB_NOTICE, severity="warning", markup=False)
            return
        kept = not bool(self._story.get("kept"))
        try:
            db.set_story_kept(story_id, kept)
        except Exception as exc:  # noqa: BLE001 - a failed write is a notice
            logger.warning(f"Dreams keep toggle failed: {type(exc).__name__}")
            self.notify(
                f"Could not update this story: {type(exc).__name__}",
                severity="error",
                markup=False,
            )
            return
        if kept and not self._record_feedback("kept"):
            return
        self._story["kept"] = kept
        self.notify(
            "Kept -- this story survives future cycles."
            if kept
            else "Unkept.",
            markup=False,
        )
        self._changed()
        if self.is_attached:
            self.refresh(recompose=True)

    def action_dive(self) -> None:
        """Stage the story into Chat (skills_screen idiom), then dismiss."""
        if self._synthetic:
            self.notify(_SYNTHETIC_NOTICE, severity="warning", markup=False)
            return
        open_chat_with_handoff = getattr(self.app, "open_chat_with_handoff", None)
        if not callable(open_chat_with_handoff):
            self.notify(_HANDOFF_UNAVAILABLE_NOTICE, severity="warning", markup=False)
            return
        if not self._record_feedback("dived"):
            return
        title = str(self._story.get("title") or "Untitled dream story")
        body = str(self._story.get("body") or "")
        url = str(self._story.get("url") or "")
        if _ingestable(self._story):
            # "Dive deeper" is about the source: carry its link too.
            body = f"{body}\n\nSource: {url}".strip()
        open_chat_with_handoff(
            ChatHandoffPayload(
                source="dreams",
                item_type="dream_story",
                title=title,
                body=body,
                display_summary=f"Dream story staged: {title}",
                suggested_prompt=(
                    "Tell me more about this story and what it means for "
                    "my interests."
                ),
            )
        )
        self._changed()
        self.dismiss(None)

    def action_export(self) -> None:
        """Write the Markdown stub to the sanctioned export folder."""
        if self._synthetic:
            self.notify(_SYNTHETIC_NOTICE, severity="warning", markup=False)
            return
        story_id = self._story_id()
        if story_id is None:
            self.notify(_NO_DB_NOTICE, severity="warning", markup=False)
            return
        try:
            export_dir = _dreams_export_dir()
            export_dir.mkdir(parents=True, exist_ok=True)
            filename = validate_filename(
                f"{_slugify(str(self._story.get('title') or ''))}-{story_id}.md"
            )
            path = validate_path_simple(
                export_dir / filename, require_exists=False
            )
            path.write_text(_export_markdown(self._story), encoding="utf-8")
        except (OSError, ValueError) as exc:
            logger.warning(f"Dreams export failed: {type(exc).__name__}")
            self.notify(
                f"Could not export this story: {type(exc).__name__}",
                severity="error",
                markup=False,
            )
            return
        if not self._record_feedback("exported"):
            return
        # The filename embeds a title-derived slug (web-derived text), so
        # this notice never parses markup.
        self.notify(f"Exported to {path.name}", markup=False)
        self._changed()

    async def action_ingest(self) -> None:
        """Submit the story URL to read-it-later, then record feedback.

        The real save entry is async (``CollectionsCaptureBackend.
        save_capture``), so this action awaits it; every failure --
        missing backend, bad URL shape, backend error -- is a dismissible
        notice, never a crash, and feedback/``on_changed`` fire only
        after a CONFIRMED capture; an outcome-unknown save is a warning.
        """
        if self._synthetic:
            self.notify(_SYNTHETIC_NOTICE, severity="warning", markup=False)
            return
        if not _ingestable(self._story):
            # dreams://llm rows have no web destination to capture; the
            # button and hint are already hidden for them, this guard
            # covers the "i" keybinding.
            self.notify(_NO_HTTP_URL_NOTICE, severity="information",
                        markup=False)
            return
        if self._ingest_in_flight:
            # The save is async: a second press while it is awaited would
            # submit the same URL twice.
            return
        backend = self._capture_backend()
        if backend is None:
            self.notify(
                _NO_CAPTURE_BACKEND_NOTICE, severity="warning", markup=False
            )
            return
        # Lazy Dreams import (same pattern as ``_preview_queries``).
        from ...Dreams.ingest_action import (
            UNKNOWN_OUTCOME_IDENTIFIER,
            ingest_story_url,
        )

        self._ingest_in_flight = True
        try:
            identifier = await ingest_story_url(
                backend,
                url=str(self._story.get("url") or ""),
                title=str(self._story.get("title") or "Untitled dream story"),
                source_note=(
                    f"via Dreams {datetime.now(UTC).date().isoformat()}"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a failed capture is a notice
            logger.warning(f"Dreams ingest failed: {type(exc).__name__}")
            self.notify(
                f"Could not ingest this story: {type(exc).__name__}",
                severity="error",
                markup=False,
            )
            return
        finally:
            self._ingest_in_flight = False
        if identifier == UNKNOWN_OUTCOME_IDENTIFIER:
            # The server may or may not have saved it: no success receipt,
            # and no ``ingested`` signal for the interest profile.
            self.notify(_UNCONFIRMED_INGEST_NOTICE, severity="warning",
                        markup=False)
            return
        if not self._record_feedback("ingested"):
            return
        # The identifier is a backend-generated id string, never markup.
        self.notify(f"Ingested to read-it-later ({identifier}).", markup=False)
        self._changed()

    def action_more(self) -> None:
        """Record ``more`` feedback, notify, dismiss."""
        if self._synthetic:
            self.notify(_SYNTHETIC_NOTICE, severity="warning", markup=False)
            return
        if not self._record_feedback("more"):
            return
        self.notify(
            "Noted -- future cycles will lean toward stories like this.",
            markup=False,
        )
        self._changed()
        self.dismiss(None)

    def action_less(self) -> None:
        """Record ``less`` feedback, notify, dismiss."""
        if self._synthetic:
            self.notify(_SYNTHETIC_NOTICE, severity="warning", markup=False)
            return
        if not self._record_feedback("less"):
            return
        self.notify(
            "Noted -- future cycles will lean away from stories like this.",
            markup=False,
        )
        self._changed()
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)

    # --- Event routing ---------------------------------------------------

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = str(event.button.id or "")
        dispatch = {
            "dsm-keep-button": self.action_keep,
            "dsm-dive-button": self.action_dive,
            "dsm-export-button": self.action_export,
            "dsm-ingest-button": self.action_ingest,
            "dsm-more-button": self.action_more,
            "dsm-less-button": self.action_less,
            "dsm-close-button": self.action_close,
        }
        handler = dispatch.get(button_id)
        if handler is not None:
            result = handler()
            # ``action_ingest`` is async (the capture entry is); the rest
            # return None immediately.
            if inspect.isawaitable(result):
                await result
