"""Detail-pane content for a selected dataset: the snippet table and import.

Mounted by ``evals_screen.py``'s ``_compose_detail_pane`` in place of the
inline ``Static`` fields it used to yield directly (Task 3's placeholder
dataset branch) -- see ``evals_screen.py``'s own module docstring for why no
``Screen`` subclass is mounted anywhere here.

**Whitespace validation is this editor's headline feature, not a nicety.**
``"The protestors were"`` and ``"The protestors were "`` measure entirely
different next-token distributions: with the trailing space, the
leading-space token variants that dominate the first case become
impossible and the whole distribution shifts to bare-word tokens. A user
comparing two snippets where one has a stray trailing space would read a
large divergence as a finding about the model -- exactly the false
conclusion this benchmark exists to prevent. Anomalous whitespace (leading,
trailing, or interior runs) therefore renders a highlighted ``␣`` and raises
a warning; normal text carries no marker at all, so the marker means
something wherever it appears (see ``render_snippet_cell``).

**Only exact duplicates are flagged, after whitespace normalization.**
Minimal pairs differing by one word (``"The protestors were"`` /
``"The rioters were"``) ARE the instrument a word bench measures with --
near-duplicate detection would warn on every well-constructed bench and
train users to ignore the warning strip where the whitespace warning also
lives (see ``find_exact_duplicate_labels``).

The count column is characters, never tokens: there is no client-side
tokenizer in this codebase, and tokenization is per-model, so a token count
here would be a guess rendered as a fact in a tool whose entire purpose is
measuring token-level behaviour.

Datasets are stored inline via the existing convention this module reuses
rather than duplicates: ``EvalsDB`` keeps authored samples in
``metadata[RESERVED_LOCAL_DATASET_SAMPLES_KEY]`` (the same reserved key
``LocalEvaluationsService.create_dataset`` writes) with
``source_path = "inline:<name>"``. This module talks to ``EvalsDB``
directly -- the same layer ``evals_state.py`` and ``bench_editor.py``
already use -- rather than routing through the Interop service.

Neither this module nor ``bench_editor.py``/``inspector.py`` imports the
HTTP capture client or the runner that drives it: importing a dataset is
local file I/O plus a database write, never a provider call. A source-scan
test in ``Tests/UI/test_evals_snippet_editor.py`` pins that guarantee.
"""

from __future__ import annotations

import csv
import io
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Optional

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from ...DB.Evals_DB import EvalsDB
from ...Evaluations_Interop.evaluation_normalizers import (
    RESERVED_LOCAL_DATASET_SAMPLES_KEY,
)
from ...Third_Party.textual_fspicker import FileOpen, Filters
from ...Utils.path_validation import validate_path_simple
from .evals_state import EvalsViewModel
from .notify_mixin import NotifyMixin

#: The glyph anomalous whitespace is replaced with when rendering a
#: snippet's text. Reverse video (not a design-token colour) is used
#: deliberately: a Rich ``Text`` object embedded in a Static's content
#: cannot reference Textual CSS variables, and reverse video reads clearly
#: against either a light or a dark terminal theme without hardcoding a
#: colour that might clash in one of them.
WHITESPACE_MARKER_GLYPH = "␣"
_WHITESPACE_MARKER_STYLE = "reverse bold"

_WHITESPACE_KIND_LABELS = {
    "leading": "leading ␣",
    "interior": "interior ␣",
    "trailing": "trailing ␣",
}
_WHITESPACE_KIND_ORDER = ("leading", "interior", "trailing")

# Leading/trailing whitespace is anomalous at ANY length -- the design
# mockup's own example flags a single trailing space. A single interior
# space between words is the normal, expected case for running prose;
# only a RUN of 2+ interior whitespace characters is anomalous there.
_LEADING_WS_RE = re.compile(r"\A\s+")
_TRAILING_WS_RE = re.compile(r"\s+\Z")
_INTERIOR_RUN_RE = re.compile(r"(?<=\S)\s{2,}(?=\S)")


def _classify_whitespace(text: str) -> list[tuple[int, int, str]]:
    """Non-overlapping ``(start, end, kind)`` spans of anomalous whitespace.

    ``kind`` is one of ``"leading"``, ``"trailing"``, ``"interior"``.
    """
    if not text:
        return []
    spans: list[tuple[int, int, str]] = []
    leading = _LEADING_WS_RE.match(text)
    trailing = _TRAILING_WS_RE.search(text)
    if leading and trailing and leading.span() == trailing.span():
        # The whole string is whitespace -- one span, not two overlapping
        # claims on the same characters.
        trailing = None
    if leading:
        spans.append((leading.start(), leading.end(), "leading"))
    if trailing:
        spans.append((trailing.start(), trailing.end(), "trailing"))
    for match in _INTERIOR_RUN_RE.finditer(text):
        if leading and match.start() < leading.end():
            continue
        if trailing and match.end() > trailing.start():
            continue
        spans.append((match.start(), match.end(), "interior"))
    spans.sort(key=lambda item: item[0])
    return spans


def whitespace_warning_kinds(text: str) -> frozenset[str]:
    """Which anomaly kinds (a subset of leading/interior/trailing) a
    snippet's raw text carries. Empty for clean text -- callers use this
    (rather than a boolean) so a row's flag text can name what is wrong."""
    return frozenset(kind for _, _, kind in _classify_whitespace(text))


def snippet_whitespace_flag_label(text: str) -> Optional[str]:
    """A short, ordered label naming every whitespace anomaly kind present,
    or ``None`` for clean text -- clean text must carry no marker at all,
    since the marker only means something because it is not decorating
    every row (see the module docstring)."""
    kinds = whitespace_warning_kinds(text)
    if not kinds:
        return None
    return ", ".join(
        _WHITESPACE_KIND_LABELS[kind] for kind in _WHITESPACE_KIND_ORDER if kind in kinds
    )


def render_snippet_cell(text: str) -> Text:
    """A Rich ``Text`` rendering of ``text`` with every anomalous
    whitespace character replaced by a styled marker glyph, and every other
    character rendered literally with no styling at all -- a clean
    snippet's rendering is indistinguishable from its plain text."""
    spans = _classify_whitespace(text)
    rich_text = Text()
    cursor = 0
    for start, end, _kind in spans:
        if start > cursor:
            rich_text.append(text[cursor:start])
        rich_text.append(
            WHITESPACE_MARKER_GLYPH * (end - start), style=_WHITESPACE_MARKER_STYLE
        )
        cursor = end
    if cursor < len(text):
        rich_text.append(text[cursor:])
    return rich_text


def normalize_snippet_whitespace(text: str) -> str:
    """Collapse whitespace runs to a single space and strip both ends --
    the equality this module's duplicate detector compares on. Two
    snippets identical except for a stray leading/trailing/doubled space
    are the SAME measurement condition and must be caught even though
    their raw text differs by one character."""
    return " ".join(text.split())


def find_exact_duplicate_labels(
    snippets: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Maps a duplicate snippet's id to a label naming the row it
    duplicates. Only EXACT duplicates (after whitespace normalization) are
    ever flagged here -- never near-duplicates. A minimal pair differing by
    one word (the instrument a word bench measures with) normalizes to two
    different strings and is never flagged; see the module docstring.

    The first occurrence in each normalized-text group is left unflagged;
    every later occurrence names the first by its 1-based row number,
    mirroring the design mockup's ``"exact dup of 4"``.
    """
    first_seen: dict[str, int] = {}
    labels: dict[str, str] = {}
    for index, snippet in enumerate(snippets):
        normalized = normalize_snippet_whitespace(str(snippet.get("text") or ""))
        if normalized in first_seen:
            first_row = first_seen[normalized] + 1
            snippet_id = str(snippet.get("id") or index)
            labels[snippet_id] = f"exact dup of {first_row}"
        else:
            first_seen[normalized] = index
    return labels


def count_warnings(snippets: Sequence[Mapping[str, Any]]) -> int:
    """Total warning count for the editor's footer strip: every snippet
    with anomalous whitespace, plus every snippet flagged as an exact
    duplicate (a row carrying both counts twice, matching the design
    mockup's additive footer)."""
    whitespace_count = sum(
        1 for snippet in snippets if whitespace_warning_kinds(str(snippet.get("text") or ""))
    )
    duplicate_count = len(find_exact_duplicate_labels(snippets))
    return whitespace_count + duplicate_count


def dataset_snippets(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The dataset's authored samples from the existing inline-storage
    convention (``metadata[RESERVED_LOCAL_DATASET_SAMPLES_KEY]``), or an
    empty list for a dataset with none yet."""
    metadata = dataset.get("metadata") or {}
    samples = metadata.get(RESERVED_LOCAL_DATASET_SAMPLES_KEY)
    if not isinstance(samples, list):
        return []
    return [dict(sample) for sample in samples if isinstance(sample, Mapping)]


def import_snippets_into_dataset(
    db: EvalsDB, dataset_id: str, new_snippets: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Append ``new_snippets`` to a dataset's inline samples and persist,
    reusing the existing ``RESERVED_LOCAL_DATASET_SAMPLES_KEY`` /
    ``inline:<name>`` convention rather than a second storage path. Returns
    the full, post-import snippet list so a caller can re-render without a
    second database round trip.

    De-duplicates ids against every id already in the dataset (and against
    earlier entries in this same batch) before appending, minting a fresh
    UUID for any collision -- so a snippet's id stays a usable identity
    (e.g. the ``find_exact_duplicate_labels`` lookup key below) even after
    re-importing the same export twice (the round-trip
    ``parse_json_snippets``'s own docstring advertises: an id is preserved
    verbatim when present). This is write-side hygiene, not what keeps
    ``_compose_row`` from crashing -- that widget derives every id it mounts
    from a row's INDEX, not its snippet id (see ``_compose_row``'s own
    comment), specifically so a dataset written before this de-dup existed,
    or by anything other than this importer, still renders. A minted-
    fresh-id re-import is not silently dropped: its text is identical to
    the row it re-imports, so ``find_exact_duplicate_labels`` (text-based,
    not id-based) flags it as an exact duplicate the same way any other
    repeated snippet would be -- the user sees it, rather than the import
    silently doing nothing.
    """
    dataset = db.get_dataset(dataset_id)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_id}' was not found.")
    metadata = dict(dataset.get("metadata") or {})
    existing = dataset_snippets(dataset)
    seen_ids = {str(snippet.get("id")) for snippet in existing}
    deduped_new: list[dict[str, Any]] = []
    for snippet in new_snippets:
        snippet = dict(snippet)
        snippet_id = str(snippet.get("id") or "")
        if not snippet_id or snippet_id in seen_ids:
            snippet_id = str(uuid.uuid4())
            snippet["id"] = snippet_id
        seen_ids.add(snippet_id)
        deduped_new.append(snippet)
    combined = existing + deduped_new
    metadata[RESERVED_LOCAL_DATASET_SAMPLES_KEY] = combined
    metadata["sample_count"] = len(combined)
    metadata["inline_samples"] = True
    db.update_dataset(
        dataset_id,
        metadata=metadata,
        source_path=dataset.get("source_path")
        or f"inline:{dataset.get('name') or dataset_id}",
    )
    return combined


# ---------------------------------------------------------------------------
# Import parsers -- plain text, CSV, JSON. Every parsed snippet gets a UUID
# at authoring time (see the module docstring); positional identifiers
# would silently remap historical results when a dataset is reordered,
# since eval_results is keyed on (run_id, sample_id).
# ---------------------------------------------------------------------------


class ParsedSnippets(NamedTuple):
    """A parser's result: the snippets it kept, plus how many input rows it
    dropped. All three parsers share this shape (even
    ``parse_plain_text_snippets``, whose only skippable "row" is a blank
    line) so ``_handle_import_file_selected`` can thread a skipped count
    into its notification without caring which parser produced it.

    The skipped count exists because the skip-invalid-entries policy (see
    ``parse_json_snippets``'s docstring) would otherwise be invisible: a
    500-row export with 40 malformed rows used to report only "Imported
    460 snippet(s)", with nothing naming the other 40 -- silently shrinking
    the instrument a word bench measures with. See
    ``_handle_import_file_selected`` for where this becomes user-visible.
    """

    snippets: list[dict[str, Any]]
    skipped: int


def parse_plain_text_snippets(content: str) -> ParsedSnippets:
    """One snippet per non-empty line. The low-friction import path most
    sets will use, and which therefore cannot express multi-line snippets.

    A line's whitespace is preserved verbatim (never stripped) -- silently
    cleaning up a trailing space on import would erase exactly the
    condition this editor's marker exists to surface.
    """
    snippets: list[dict[str, Any]] = []
    skipped = 0
    for line in content.splitlines():
        if not line:
            skipped += 1
            continue
        snippets.append(
            {"id": str(uuid.uuid4()), "text": line, "group": None, "note": None}
        )
    return ParsedSnippets(snippets, skipped)


def parse_csv_snippets(content: str) -> ParsedSnippets:
    """CSV with a ``text`` column (required, case-insensitive header
    match) and an optional ``group`` column. Cell content is preserved
    verbatim, same reasoning as ``parse_plain_text_snippets``."""
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        raise ValueError("CSV import requires a header row with a 'text' column.")
    lowered_fieldnames = {
        name.strip().lower(): name for name in reader.fieldnames if name
    }
    if "text" not in lowered_fieldnames:
        raise ValueError("CSV import requires a 'text' column.")
    text_key = lowered_fieldnames["text"]
    group_key = lowered_fieldnames.get("group")

    snippets: list[dict[str, Any]] = []
    skipped = 0
    for row in reader:
        text = row.get(text_key) or ""
        if text == "":
            skipped += 1
            continue
        group = row.get(group_key) if group_key else None
        group = group if group not in (None, "") else None
        snippets.append(
            {"id": str(uuid.uuid4()), "text": text, "group": group, "note": None}
        )
    return ParsedSnippets(snippets, skipped)


#: A snippet's ``id`` is never interpolated into a Textual widget id --
#: ``_compose_row`` below derives every widget id in a row from that row's
#: INDEX instead, precisely so this sanitizer is not the only thing
#: standing between a stored id and a mount-time crash (see
#: ``_compose_row``'s own comment). This regex still matters at write time:
#: an id that fails it is not safe to treat as a stable identity (round-
#: tripping an export, matching a duplicate-label lookup key) even though
#: it can no longer crash the renderer either way. Textual's own identifier
#: grammar (``textual.css.tokenize.IDENTIFIER``, ``[a-zA-Z_-][a-zA-Z0-9_-]*``)
#: restricts only the FIRST character of a widget id to a non-digit; this
#: regex is deliberately looser than that (permits a leading digit)
#: because the majority of real ``uuid.uuid4()`` values start with one --
#: confirmed the hard way: an earlier version of this pattern broke the
#: "preserve an existing id verbatim" round-trip for most legitimate
#: UUIDs. Every character (not just the first) must still be a legal
#: identifier character -- e.g. a space is not -- so this is not simply
#: "any non-empty string".
_SNIPPET_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _sanitize_snippet_id(raw_id: Any) -> str:
    """``raw_id`` if it is already a legal widget-id string, else a fresh
    UUID -- the same fallback a missing ``id`` already gets. An id that
    fails this check is not a usable identity (it cannot become a widget
    id), so minting a replacement is preferable to round-tripping a value
    that will crash the next compose()."""
    if isinstance(raw_id, str) and _SNIPPET_ID_RE.match(raw_id):
        return raw_id
    return str(uuid.uuid4())


def parse_json_snippets(content: str) -> ParsedSnippets:
    """A JSON list of snippet objects, or an object with a ``"snippets"``
    list, for round-tripping an exported set.

    An existing ``id`` is preserved verbatim when it is a legal widget-id
    string (so re-importing a genuine export does not mint new identities
    for snippets that already have stable ones); a missing, blank, or
    invalid ``id`` gets a fresh UUID, same as the other two import shapes
    (see ``_sanitize_snippet_id``).

    **Per-entry policy matches ``parse_csv_snippets``**: an entry that
    isn't an object, or whose ``text`` is missing/blank, is skipped rather
    than aborting the whole import -- the same tolerance
    ``parse_csv_snippets`` already gives a CSV's blank rows. A large,
    otherwise-valid export must not be rejected wholesale over one bad
    entry. Only a structurally invalid payload (unparseable JSON, or a
    shape that isn't a list/``{"snippets": [...]}``) raises -- that is a
    "this is not the expected file" signal, not a single skippable row.
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON: {exc}") from exc

    if isinstance(parsed, Mapping):
        parsed = parsed.get("snippets")
    if not isinstance(parsed, list):
        raise ValueError(
            "JSON snippet import expects a list of snippets, or an object "
            "with a 'snippets' list."
        )

    snippets: list[dict[str, Any]] = []
    skipped = 0
    for entry in parsed:
        if not isinstance(entry, Mapping):
            skipped += 1
            continue
        text = entry.get("text")
        if not isinstance(text, str) or text == "":
            skipped += 1
            continue
        snippet_id = _sanitize_snippet_id(entry.get("id"))
        group = entry.get("group")
        group = group if isinstance(group, str) and group != "" else None
        note = entry.get("note")
        note = note if isinstance(note, str) and note != "" else None
        snippets.append({"id": snippet_id, "text": text, "group": group, "note": note})
    return ParsedSnippets(snippets, skipped)


_IMPORT_PARSERS = {
    ".csv": parse_csv_snippets,
    ".json": parse_json_snippets,
}


class SnippetEditor(NotifyMixin, Vertical):
    """Detail-pane content for a selected dataset: a read-only snippet
    table (character count, whitespace flag, exact-duplicate flag) and an
    import control (``#evals-import-snippets``)."""

    def __init__(self, view_model: EvalsViewModel, dataset_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._view_model = view_model
        self._dataset_id = dataset_id

    def compose(self) -> ComposeResult:
        db = self._view_model.db
        if db is None:
            yield Static(
                "The evaluation service is unavailable.",
                id="evals-snippet-editor-unavailable",
            )
            return
        dataset = self._view_model.dataset_by_id(self._dataset_id)
        if dataset is None:
            # Re-checked here (not only by evals_screen.py's own guard)
            # against the same deleted-between-selection-and-compose race
            # bench_editor.py's widgets guard against.
            yield Static(
                "This dataset's data could not be read.",
                id="evals-snippet-editor-error",
            )
            return

        snippets = dataset_snippets(dataset)
        groups = {snippet.get("group") for snippet in snippets if snippet.get("group")}
        snippet_word = "snippet" if len(snippets) == 1 else "snippets"
        group_word = "group" if len(groups) == 1 else "groups"

        # markup=False: the dataset name is user-authored free text -- a
        # bare `[/]` would raise `MarkupError` the instant this Static
        # lays out (task-1482 Task 1; mirrors bench_editor.py's identical
        # fix for `config.name`).
        yield Static(
            str(dataset.get("name") or "Untitled dataset"),
            id="evals-detail-dataset-name",
            classes="evals-pane-heading",
            markup=False,
        )
        yield Static(
            f"inline · {len(snippets)} {snippet_word} · {len(groups)} {group_word}",
            id="evals-snippet-editor-summary",
        )

        if not snippets:
            yield Static(
                "No snippets yet. Use Import to add plain text, CSV, or JSON.",
                id="evals-snippet-empty",
            )
        else:
            duplicate_labels = find_exact_duplicate_labels(snippets)
            yield Static(
                "#   Snippet   Group · Chars · Flags",
                id="evals-snippet-table-header",
                classes="evals-snippet-table-header",
                markup=False,
            )
            with Vertical(id="evals-snippet-table"):
                for index, snippet in enumerate(snippets):
                    yield from self._compose_row(index, snippet, duplicate_labels)

            total_warnings = count_warnings(snippets)
            warning_word = "warning" if total_warnings == 1 else "warnings"
            yield Static(
                f"{total_warnings} {warning_word}"
                if total_warnings
                else "No warnings",
                id="evals-snippet-warnings-summary",
            )

        yield Button("Import…", id="evals-import-snippets")

    @staticmethod
    def _compose_row(
        index: int, snippet: dict[str, Any], duplicate_labels: dict[str, str]
    ) -> ComposeResult:
        # `snippet_id` is used ONLY as a `duplicate_labels` lookup key below
        # -- never as (or as part of) a widget id. Every widget id in this
        # row is derived from `index` instead (see the `Horizontal(id=...)`
        # below): a rendering path must never trust stored data to be
        # widget-id-safe, no matter how it got into the dataset. Validating
        # at the import boundary (`_sanitize_snippet_id`) only protects
        # data THIS importer wrote; a dataset written before that
        # validation existed, or by anything else that touches
        # `RESERVED_LOCAL_DATASET_SAMPLES_KEY` directly, can still contain
        # a duplicate or non-identifier-safe id, and display of that
        # dataset must not depend on it being clean. An index is always a
        # legal, unique widget id regardless of what a snippet's own id
        # turns out to be.
        snippet_id = str(snippet.get("id") or index)
        text = str(snippet.get("text") or "")
        group = snippet.get("group") or "—"
        char_count = len(text)
        ws_label = snippet_whitespace_flag_label(text)
        dup_label = duplicate_labels.get(snippet_id)
        flags = ", ".join(label for label in (ws_label, dup_label) if label) or "—"

        with Horizontal(
            id=f"evals-snippet-row-{index}", classes="evals-snippet-row"
        ):
            yield Static(f"{index + 1}.", classes="evals-snippet-index", markup=False)
            yield Static(
                render_snippet_cell(text),
                id=f"evals-snippet-text-{index}",
                classes="evals-snippet-text",
                markup=False,
            )
            yield Static(
                f"group: {group} · {char_count} chars · {flags}",
                id=f"evals-snippet-meta-{index}",
                classes="evals-snippet-meta",
                markup=False,
            )

    @on(Button.Pressed, "#evals-import-snippets")
    def _on_import_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._open_import_dialog()

    def _open_import_dialog(self) -> None:
        filters = Filters(
            ("Text (one snippet per line)", lambda p: p.suffix.lower() == ".txt"),
            ("CSV", lambda p: p.suffix.lower() == ".csv"),
            ("JSON", lambda p: p.suffix.lower() == ".json"),
            ("All files", lambda p: True),
        )
        self.app.push_screen(
            FileOpen(title="Import snippets", filters=filters),
            self._handle_import_file_selected,
        )

    # _notify lives on NotifyMixin (see .notify_mixin) -- shared with
    # LibraryRail and ResultsGrid, which carried byte-identical copies of
    # this same method (TASK-861).

    def _handle_import_file_selected(self, path: Optional[Any]) -> None:
        """The ``FileOpen`` dialog's selection callback. Public-shaped
        (not name-mangled) so tests can drive an import directly with a
        real temp file, bypassing the modal picker itself -- the same
        approach ``STTS_Window._handle_file_selection`` established for
        this codebase's other file-based import flows.
        """
        if not path:
            return
        try:
            # CLAUDE.md's security requirements call for `path_validation.py`
            # on every file path; `validate_path_simple` (not `validate_path`)
            # is the right tool here -- there is no single base directory to
            # confine to, since a user can legitimately import a snippet set
            # from anywhere on disk (the same reasoning `Character_Chat_Lib.
            # parse_character_card`'s identical call documents). `path` was
            # chosen through a `FileOpen` dialog the user themselves drove,
            # not typed as an untrusted string, so this is defense in depth
            # rather than a live traversal hole -- but the rule is explicit
            # and the helper is cheap, so it applies here too rather than
            # reading the file directly.
            file_path = validate_path_simple(path, require_exists=True)
        except ValueError as exc:
            self._notify(f"Could not read {Path(path).name}: {exc}", severity="error")
            return
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # UnicodeDecodeError is a ValueError, not an OSError -- a file
            # that can be opened but isn't valid UTF-8 (e.g. a CSV Excel
            # exported as cp1252/Latin-1, the most likely non-UTF-8 file a
            # user actually picks) used to propagate straight out of this
            # push_screen callback and crash the app instead of producing
            # the same "could not read" notification an unreadable path
            # already gets.
            self._notify(f"Could not read {file_path.name}: {exc}", severity="error")
            return

        parser = _IMPORT_PARSERS.get(file_path.suffix.lower(), parse_plain_text_snippets)
        try:
            new_snippets, skipped_count = parser(content)
        except ValueError as exc:
            self._notify(f"Import failed: {exc}", severity="error")
            return

        if not new_snippets:
            message = "No snippets found to import"
            message += self._skipped_suffix(skipped_count)
            self._notify(message, severity="warning")
            return

        db = self._view_model.db
        if db is None:
            self._notify("The evaluation service is unavailable.", severity="error")
            return
        try:
            import_snippets_into_dataset(db, self._dataset_id, new_snippets)
        except ValueError as exc:
            self._notify(str(exc), severity="error")
            return

        # The skip-invalid-entries policy (see parse_json_snippets's
        # docstring) would otherwise be invisible here: a large export with
        # some malformed rows used to report only the survivor count, with
        # nothing naming what was dropped -- silently shrinking a word
        # bench's own instrument. `_skipped_suffix` adds nothing when
        # skipped_count is 0, so the common case stays clean.
        message = f"Imported {len(new_snippets)} snippet(s)"
        message += self._skipped_suffix(skipped_count)
        self._notify(message, severity="information")
        self.refresh(recompose=True)

    @staticmethod
    def _skipped_suffix(skipped_count: int) -> str:
        """The sentence-closing suffix for an import notification: a bare
        ``"."`` when nothing was dropped, else ``"; skipped N invalid
        entries."`` (singular for exactly one) -- appended directly to a
        base message with no trailing punctuation of its own, so the two
        clauses read as one sentence (``"Imported 460 snippet(s); skipped
        40 invalid entries."``) rather than two."""
        if not skipped_count:
            return "."
        entry_word = "entry" if skipped_count == 1 else "entries"
        return f"; skipped {skipped_count} invalid {entry_word}."
