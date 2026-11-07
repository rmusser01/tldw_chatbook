"""Bounded, read-only parsers for discovered one-time note import sources."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any
from unicodedata import normalize

import yaml

from tldw_chatbook.Notes.note_folder_models import (
    FolderValidationError,
    normalize_folder_name,
)
from tldw_chatbook.Notes.note_import_discovery import (
    DiscoveredImportSource,
    ImportDiscovery,
    ImportSelectionError,
    VerifiedSourceReadError,
    read_discovered_source,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    MAX_IMPORT_FILE_BYTES,
    MAX_IMPORT_KEYWORD_LENGTH,
    MAX_IMPORT_REASON_LENGTH,
    MAX_IMPORT_TEMPLATE_NAME_LENGTH,
    MAX_IMPORT_TITLE_LENGTH,
    NON_IMPORTABLE_CLASSIFICATIONS,
    WIKILINK_SCAN,
    ImportBounds,
    ImportClassification,
    ImportSourceKind,
    ParsedNotePayload,
    ProposedFolderMembership,
    wikilink_target,
)

SUPPORTED_NOTE_EXTENSIONS = frozenset(
    {".txt", ".text", ".md", ".markdown", ".rst", ".json", ".yaml", ".yml", ".csv"}
)

_TITLE_ALIASES = ("title", "name")
_CONTENT_ALIASES = ("content", "body")
_KEYWORD_ALIASES = ("keywords", "tags")
_CSV_RESERVED_HEADERS = frozenset(
    (*_TITLE_ALIASES, *_CONTENT_ALIASES, *_KEYWORD_ALIASES, "template")
)
_MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})
_FRONTMATTER_KEYS = ("tags", "keywords", "aliases")
"""Front matter properties read as keywords, in precedence order.

``keywords`` is a read alias for ``tags`` (TASK-34000.31): the single-note
**Export Markdown** file wrote ``keywords: a, b`` before that fix, so the files
users already exported still restore their keywords.
"""
_TAG_KEYS = frozenset({"tags", "keywords"})
_OWN_EXPORT_KEYS = frozenset({"date", "note_id", "id", "created_at", "updated_at"})
"""Provenance keys the app's own exporters write (``Export Markdown``, bundles).

They are never imported -- the Library has its own ids and stamps -- and, in a
block that carries an export's own marker (``_carries_export_provenance``), never
reported as "not imported" either (TASK-34000.31 fix round 1, M3): dropping
them there loses nothing a user wrote.
"""
_EXPORT_MARKDOWN_PROVENANCE_KEY = "note_id"
"""**Export Markdown** writes ``title``, ``date``, ``tags`` (``keywords`` before
TASK-34000.31) and this key (``build_note_export_content``)."""
_BUNDLE_PROVENANCE_KEYS = frozenset({"id", "created_at", "updated_at"})
"""A bundle's note file always carries all three (``ChatbookCreator``)."""
_ALIAS_KEYWORD_PREFIX = "alias: "
"""Marks a keyword that came from `aliases:` rather than from `tags:`."""
_MAX_FRONTMATTER_LINES = 200
_MAX_FRONTMATTER_BYTES = 64 * 1024
"""A larger block is read as no front matter (final review M-11): the line cap
does not bound a line's length, and the YAML reader takes ~0.25 s per MB of a
file that may be up to 128 MB (a bundle member)."""
_MAX_WIKILINKS_PER_NOTE = 200
_TEMPLATE_PLACEHOLDER = re.compile(r"\{\{.*?\}\}")
"""A `{{date}}`-style template placeholder, which never becomes a note title."""


# ``csv.field_size_limit`` is process-global. Every CSV parse in this module holds
# this lock while raising and restoring the limit so overlapping imports cannot
# restore one another's value out of order.
_CSV_FIELD_SIZE_LIMIT_LOCK = RLock()


_MESSAGES = {
    "unsupported_extension": "This file type is not supported.",
    # task-32262: a vault's own file types deserve the vault-aware copy
    # `.obsidian` and `.trash` already get, not one generic sentence for
    # everything from a canvas to a PDF.
    "unsupported_canvas": "Obsidian canvas — not a note.",
    "unsupported_base": "Obsidian base — not a note.",
    "unsupported_image": "Image — not a note. Add it in Library ▸ Media.",
    "unsupported_document": "Document — not a note. Add it in Library ▸ Media.",
    "unsupported_media": "Audio or video — not a note. Add it in Library ▸ Media.",
    "source_changed": "This source changed during the import preview.",
    "source_unavailable": "This source could not be read safely.",
    "secure_read_unavailable": "Secure source reading is unavailable.",
    "max_file_bytes_exceeded": "This source is too large to import.",
    "max_total_bytes_exceeded": "The selected sources are too large in total.",
    "invalid_utf8": "This source is not valid UTF-8 text.",
    "invalid_content": "This source could not be parsed as notes.",
    "empty_source": "Empty file — nothing to import.",
    "not_a_note": "Not a note file (app configuration).",
    "empty_structured_source": "This source does not contain any notes.",
    "too_many_notes": "This source contains too many notes.",
    "too_many_keywords": "A note in this source contains too many keywords.",
    "destination_required": "Choose a destination folder for selected files.",
    "destination_not_allowed": "A folder import already defines its destination.",
    "invalid_destination": "The destination folder path is not valid.",
    "selection_changed": "The discovered source set changed before parsing.",
}

_UNSUPPORTED_EXTENSION_REASONS = {
    ".canvas": "unsupported_canvas",
    ".base": "unsupported_base",
    ".png": "unsupported_image",
    ".jpg": "unsupported_image",
    ".jpeg": "unsupported_image",
    ".gif": "unsupported_image",
    ".webp": "unsupported_image",
    ".svg": "unsupported_image",
    ".pdf": "unsupported_document",
    ".docx": "unsupported_document",
    ".epub": "unsupported_document",
    ".mp3": "unsupported_media",
    ".m4a": "unsupported_media",
    ".wav": "unsupported_media",
    ".mp4": "unsupported_media",
    ".mov": "unsupported_media",
    ".webm": "unsupported_media",
}
"""Reason code per non-note extension, keyed casefolded (task-32262).

Anything absent keeps the generic ``unsupported_extension`` sentence: an
honest "this file type is not supported" beats a confident wrong noun.
"""


# task-32130: every other parse failure stays FAILED.
_FAILURE_CLASSIFICATIONS = {
    "empty_source": ImportClassification.EMPTY,
    "not_a_note": ImportClassification.SKIPPED,
    # A well-formed document that simply holds no note is not broken either.
    "empty_structured_source": ImportClassification.SKIPPED,
}


class _ParseFailure(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        # task-32176: `detail` names the record that failed inside an otherwise
        # valid structured document, so a 200-note export does not report one
        # bad row as one unreadable file.
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(reason_code)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    if not isinstance(node, yaml.nodes.MappingNode):
        raise _ParseFailure("invalid_content")
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise _ParseFailure("invalid_content") from error
        if duplicate:
            raise _ParseFailure("invalid_content")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class ParsedImportSource:
    """Successfully parsed payloads and their manual folder placement."""

    candidate: DiscoveredImportSource
    payloads: tuple[ParsedNotePayload, ...] = field(repr=False)
    memberships: tuple[ProposedFolderMembership, ...]

    def __post_init__(self) -> None:
        payloads = tuple(self.payloads)
        memberships = tuple(self.memberships)
        if not isinstance(self.candidate, DiscoveredImportSource):
            raise TypeError("candidate must be a DiscoveredImportSource.")
        if not payloads or not all(
            isinstance(payload, ParsedNotePayload) for payload in payloads
        ):
            raise ValueError("payloads must contain parsed notes.")
        if not all(
            isinstance(membership, ProposedFolderMembership)
            for membership in memberships
        ):
            raise ValueError("memberships must contain proposed placements.")
        if len(memberships) != len(payloads) or {
            membership.payload_index for membership in memberships
        } != set(range(len(payloads))):
            raise ValueError("memberships must cover every parsed payload.")
        object.__setattr__(self, "payloads", payloads)
        object.__setattr__(self, "memberships", memberships)


@dataclass(frozen=True, slots=True)
class ImportParseIssue:
    """One unsupported or failed source with private execution path hidden."""

    display_path: str
    source_path: Path = field(repr=False, compare=False)
    classification: ImportClassification
    reason_code: str
    user_message: str

    def __post_init__(self) -> None:
        if not isinstance(self.display_path, str):
            raise TypeError("display_path must be text.")
        display_path = PurePosixPath(self.display_path)
        if (
            not self.display_path
            or display_path.is_absolute()
            or display_path == PurePosixPath(".")
            or ".." in display_path.parts
            or "\\" in self.display_path
            or "\x00" in self.display_path
        ):
            raise ValueError("display_path must be a safe relative path.")
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a Path.")
        if (
            not isinstance(self.classification, ImportClassification)
            or self.classification not in NON_IMPORTABLE_CLASSIFICATIONS
        ):
            raise ValueError("classification must be a non-importable outcome.")
        if (
            not isinstance(self.reason_code, str)
            or not self.reason_code
            or len(self.reason_code) > 64
            or not all(
                character.isascii()
                and (character.islower() or character.isdigit() or character == "_")
                for character in self.reason_code
            )
        ):
            raise ValueError("reason_code must be a stable safe code.")
        if not isinstance(self.user_message, str):
            raise TypeError("user_message must be text.")
        if (
            not self.user_message.strip()
            or len(self.user_message) > MAX_IMPORT_REASON_LENGTH
            or "\x00" in self.user_message
        ):
            raise ValueError("user_message must be bounded safe text.")


@dataclass(frozen=True, slots=True)
class ParsedImportBatch:
    """Immutable parse result preceding Task 4 classification."""

    parsed: tuple[ParsedImportSource, ...] = field(repr=False)
    issues: tuple[ImportParseIssue, ...]
    proposed_folder_paths: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        parsed = tuple(self.parsed)
        issues = tuple(self.issues)
        if isinstance(self.proposed_folder_paths, (str, bytes)):
            raise TypeError("proposed_folder_paths must be a collection, not text.")
        try:
            raw_paths = tuple(self.proposed_folder_paths)
        except TypeError as error:
            raise ValueError("proposed_folder_paths must be a collection.") from error
        paths_list: list[tuple[str, ...]] = []
        for path in raw_paths:
            if isinstance(path, (str, bytes)):
                raise TypeError(
                    "Each proposed folder path must be a collection, not text."
                )
            try:
                paths_list.append(tuple(path))
            except TypeError as error:
                raise ValueError(
                    "Each proposed folder path must be a collection."
                ) from error
        paths = tuple(paths_list)
        if not all(isinstance(item, ParsedImportSource) for item in parsed):
            raise ValueError("parsed must contain parsed sources.")
        if not all(isinstance(issue, ImportParseIssue) for issue in issues):
            raise ValueError("issues must contain parse issues.")
        if len(set(paths)) != len(paths):
            raise ValueError("proposed_folder_paths must be unique.")
        for path in paths:
            if not path:
                raise ValueError("proposed folder paths cannot be empty.")
            for segment in path:
                try:
                    normalized = normalize_folder_name(segment)
                except FolderValidationError as error:
                    raise ValueError("proposed folder paths must be valid.") from error
                if normalized.display != segment:
                    raise ValueError("proposed folder paths must be canonical.")
        object.__setattr__(self, "parsed", parsed)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "proposed_folder_paths", paths)


def parse_import_sources(
    discovery: ImportDiscovery,
    bounds: ImportBounds,
    *,
    destination_folder_segments: Iterable[str] | None = None,
    obsidian_mode: bool = False,
) -> ParsedImportBatch:
    """Parse a discovered selection without writes or durable side effects.

    Args:
        discovery: Previously admitted sources and safe discovery failures.
        bounds: Resource and diagnostic limits for parsing.
        destination_folder_segments: Optional manual destination for selected files.
        obsidian_mode: Whether a detected vault's Markdown is read as Obsidian
            notes: leading YAML frontmatter supplies the title and keywords and
            leaves the body, and `[[wikilinks]]` are recorded for the executor.
            It has no effect unless discovery detected a vault.

    Returns:
        Parsed note payloads, safe issues, and proposed folder paths.

    Raises:
        TypeError: An argument has an invalid type.
        ValueError: A destination folder segment is invalid.
        ImportSelectionError: Discovery totals changed or exceed the bounds.
    """
    if not isinstance(discovery, ImportDiscovery):
        raise TypeError("discovery must be an ImportDiscovery.")
    if not isinstance(bounds, ImportBounds):
        raise TypeError("bounds must be an ImportBounds.")
    if type(obsidian_mode) is not bool:
        raise TypeError("obsidian_mode must be a boolean.")
    obsidian = obsidian_mode and discovery.vault_detected

    destination = _validate_destination(
        discovery,
        destination_folder_segments,
        bounds,
    )
    expected_total = sum(candidate.size_bytes for candidate in discovery.candidates)
    if (
        expected_total != discovery.total_bytes
        or expected_total > bounds.max_total_bytes
        or len(discovery.candidates) > bounds.max_files
    ):
        _reject_selection(bounds, "selection_changed")

    parsed: list[ParsedImportSource] = []
    issues = [
        ImportParseIssue(
            display_path=entry.display_path,
            source_path=entry.source_path,
            classification=classification,
            reason_code=entry.reason_code,
            user_message=entry.user_message[: bounds.max_reason_length],
        )
        for entries, classification in (
            (discovery.failures, ImportClassification.FAILED),
            (discovery.skips, ImportClassification.SKIPPED),
        )
        for entry in entries
    ]
    bytes_read = 0
    for candidate in discovery.candidates:
        extension = PurePosixPath(candidate.source.display_path).suffix.casefold()
        if extension not in SUPPORTED_NOTE_EXTENSIONS:
            issues.append(
                _issue(
                    candidate,
                    bounds,
                    ImportClassification.UNSUPPORTED,
                    _UNSUPPORTED_EXTENSION_REASONS.get(
                        extension, "unsupported_extension"
                    ),
                )
            )
            continue
        try:
            raw_content = read_discovered_source(candidate, bounds)
            bytes_read += len(raw_content)
            if bytes_read > bounds.max_total_bytes:
                raise _ParseFailure("max_total_bytes_exceeded")
            text = raw_content.decode("utf-8-sig")
            payloads = _parse_text(
                candidate,
                extension,
                text,
                bounds,
                obsidian_mode=obsidian,
            )
            folder_segments = _folder_segments(candidate, destination)
            memberships = tuple(
                ProposedFolderMembership(
                    payload_index=index,
                    folder_segments=folder_segments,
                )
                for index in range(len(payloads))
            )
            parsed.append(
                ParsedImportSource(
                    candidate=candidate,
                    payloads=payloads,
                    memberships=memberships,
                )
            )
        except UnicodeDecodeError:
            issues.append(
                _issue(candidate, bounds, ImportClassification.FAILED, "invalid_utf8")
            )
        except VerifiedSourceReadError as error:
            issues.append(
                _issue(
                    candidate, bounds, ImportClassification.FAILED, error.reason_code
                )
            )
        except _ParseFailure as error:
            issues.append(
                _issue(
                    candidate,
                    bounds,
                    _FAILURE_CLASSIFICATIONS.get(
                        error.reason_code, ImportClassification.FAILED
                    ),
                    error.reason_code,
                    detail=error.detail,
                )
            )
        except (
            csv.Error,
            json.JSONDecodeError,
            yaml.YAMLError,
            RecursionError,
            ValueError,
            TypeError,
        ):
            issues.append(
                _issue(
                    candidate, bounds, ImportClassification.FAILED, "invalid_content"
                )
            )

    proposed_paths = _proposed_folder_paths(parsed)
    return ParsedImportBatch(
        parsed=tuple(parsed),
        issues=tuple(issues),
        proposed_folder_paths=proposed_paths,
    )


def _validate_destination(
    discovery: ImportDiscovery,
    raw_segments: Iterable[str] | None,
    bounds: ImportBounds,
) -> tuple[str, ...] | None:
    if discovery.root_label is not None:
        if raw_segments is not None:
            _reject_selection(bounds, "destination_not_allowed")
        return None
    if raw_segments is None:
        _reject_selection(bounds, "destination_required")
    if isinstance(raw_segments, (str, bytes)):
        _reject_selection(bounds, "invalid_destination")
    try:
        segments = tuple(raw_segments)
    except TypeError:
        _reject_selection(bounds, "invalid_destination")
    if not segments or len(segments) > bounds.max_depth + 1:
        _reject_selection(bounds, "invalid_destination")
    for segment in segments:
        try:
            normalized = normalize_folder_name(segment)
        except FolderValidationError:
            _reject_selection(bounds, "invalid_destination")
        if normalized.display != segment:
            _reject_selection(bounds, "invalid_destination")
    return segments


def _parse_text(
    candidate: DiscoveredImportSource,
    extension: str,
    text: str,
    bounds: ImportBounds,
    *,
    obsidian_mode: bool = False,
) -> tuple[ParsedNotePayload, ...]:
    # task-32130: an empty source is empty whatever its extension.
    if not text.strip():
        raise _ParseFailure("empty_source")
    if extension in {".txt", ".text", ".rst", ".md", ".markdown"}:
        markdown = extension in _MARKDOWN_EXTENSIONS
        metadata: Mapping[Any, Any] | None = None
        # Links come from the note body only. For a frontmatter-only file the
        # body is empty while the content stays the original YAML, so a
        # `source: "[[README]]"` property is metadata and is never rewritten.
        body = text
        unimported_keys: tuple[str, ...] = ()
        # TASK-34000.31: the YAML block is read outside a vault too, so a note
        # exported with Export Markdown and picked through Import once (the
        # toggle is offered for a vault alone) comes back with its title and
        # keywords instead of its own ``---`` block as body text (the review's
        # N-20 symptom, reproduced live). Fix round 1 ruling: outside a vault
        # the block is stripped ONLY when every key is one the import honours,
        # or the block is the app's own export (ruling F-1: it carries the
        # exporters' provenance) -- a user's ``Summary:`` or ``Yes:`` block, or
        # a daily note's ``date:`` beside ``tags:``, stays in the body exactly
        # as before, nothing of theirs is dropped. Obsidian mode still governs
        # what is Obsidian's: the vault's private folders, ``[[wikilinks]]``,
        # and stripping any block.
        if markdown:
            metadata, body = split_frontmatter(text)
            if (
                metadata is not None
                and not obsidian_mode
                and not _is_own_export_block(metadata)
            ):
                metadata, body = None, text
            # A note that is only frontmatter (an Obsidian Properties-only file,
            # a templated daily note) still has to import: keep the original
            # text as the body rather than turning a note this mode understands
            # BETTER into a failure it did not have before.
            if metadata is not None and body.strip():
                text = body
                # Only a stripped block loses anything (task-32262); a
                # frontmatter-only note keeps every property in its content.
                unimported_keys = _unimported_frontmatter_keys(metadata, bounds)
        stem = PurePosixPath(candidate.source.display_path).stem
        title = _frontmatter_title(metadata) or _text_title(text, stem, markdown)
        if len(title) > MAX_IMPORT_TITLE_LENGTH:
            raise _ParseFailure("invalid_content")
        return (
            ParsedNotePayload(
                title=title,
                content=text,
                keywords=_frontmatter_keywords(metadata, bounds),
                wikilinks=_wikilinks(body) if obsidian_mode and markdown else (),
                unimported_frontmatter_keys=unimported_keys,
            ),
        )
    if extension == ".json":
        value = json.loads(text, object_pairs_hook=_unique_json_object)
        return _structured_payloads(value, bounds)
    if extension in {".yaml", ".yml"}:
        for token in yaml.scan(text):
            if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)):
                raise _ParseFailure("invalid_content")
        value = yaml.load(text, Loader=_UniqueKeySafeLoader)
        return _structured_payloads(value, bounds)
    if extension == ".csv":
        return _csv_payloads(text, bounds)
    raise _ParseFailure("invalid_content")


def _text_title(text: str, stem: str, markdown: bool) -> str:
    """Return the first usable `# ` heading, else the file stem.

    A heading that is only a template placeholder (`# {{date:YYYY-MM-DD}}`) is
    never a note title, so the stem is used instead.
    """
    if not markdown:
        return stem
    for line in text.splitlines()[:10]:
        if line.startswith("# ") and line[2:].strip():
            heading = line[2:].strip()
            return stem if _TEMPLATE_PLACEHOLDER.search(heading) else heading
    return stem


def split_frontmatter(text: str) -> tuple[Mapping[Any, Any] | None, str]:
    """Split one leading `---` YAML block from the note body.

    Anything that is not a complete, safely loadable mapping is left in the body
    exactly as written, so an odd block degrades to today's behaviour.

    Public since TASK-34000.31: the chatbook importer and the exporters' tests
    read a note file the same way **Add from files…** does, so the one reader
    is shared rather than re-implemented with a naive ``split("---")``.

    Args:
        text: The note file's text.

    Returns:
        ``(metadata, body)`` -- the parsed mapping and the text after the
        closing fence, or ``(None, text)`` when there is no readable block.
    """
    lines = text.split("\n")
    if not lines or lines[0].rstrip() != "---":
        return None, text
    for index, line in enumerate(lines[1 : _MAX_FRONTMATTER_LINES + 1], start=1):
        # Only a fence at column 0 closes the block (TASK-34000.31 fix round
        # 1, M1): the dumper folds a multi-line title so a continuation line
        # can be an indented ``  ---``, which ``strip()`` mistook for the end.
        if line.rstrip() not in {"---", "..."}:
            continue
        block = "\n".join(lines[1:index])
        if len(block.encode("utf-8")) > _MAX_FRONTMATTER_BYTES:
            return None, text
        try:
            for token in yaml.scan(block):
                if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)):
                    return None, text
            metadata = yaml.load(block, Loader=_UniqueKeySafeLoader)
        except (yaml.YAMLError, _ParseFailure, RecursionError, ValueError, TypeError):
            return None, text
        if not isinstance(metadata, Mapping):
            return None, text
        metadata = _with_raw_scalar_title(metadata, block)
        return metadata, re.sub(r"^(?:\r?\n)+", "", "\n".join(lines[index + 1 :]))
    return None, text


def _with_raw_scalar_title(metadata: Mapping[Any, Any], block: str) -> Mapping[Any, Any]:
    """Give ``title`` back the text the author wrote when YAML typed it.

    YAML 1.1 reads ``title: 3.10`` as 3.1, ``12:30`` as 750 (base 60), ``0755``
    as 493 and ``2026-10-24`` as a date. A title is a name, so when the typed
    value is not a string the block is re-read with ``yaml.BaseLoader`` (every
    scalar a string) and that key's raw text is used (TASK-34000.31 fix round
    2, R2-4). A list or a mapping stays what it is and is reported instead.
    """
    value = metadata.get("title")
    if value is None or isinstance(value, str):
        return metadata
    try:
        raw = yaml.load(block, Loader=yaml.BaseLoader)
    except (yaml.YAMLError, RecursionError, ValueError, TypeError):
        return metadata
    if not isinstance(raw, Mapping) or not isinstance(raw.get("title"), str):
        return metadata
    patched = dict(metadata)
    patched["title"] = raw["title"]
    return patched


_split_frontmatter = split_frontmatter
"""Pre-TASK-34000.31 name; kept for one release for out-of-tree callers."""


def _frontmatter_title(metadata: Mapping[Any, Any] | None) -> str | None:
    """Return the frontmatter title, unless it is blank or a placeholder.

    ``split_frontmatter`` already restored a typed scalar (``title: 2026``,
    ``12:30``) to the text the author wrote; a list or a mapping is not a title
    and is reported instead.
    """
    if metadata is None:
        return None
    value = metadata.get("title")
    if not isinstance(value, str):
        return None
    title = value.strip()
    if not title or _TEMPLATE_PLACEHOLDER.search(title):
        return None
    return title


_IMPORTED_FRONTMATTER_KEYS = frozenset({"title", *_FRONTMATTER_KEYS})
"""Frontmatter properties the import actually keeps."""

_STRIPPABLE_OUTSIDE_A_VAULT = _IMPORTED_FRONTMATTER_KEYS | _OWN_EXPORT_KEYS

_MAX_REPORTED_FRONTMATTER_KEYS = 12


def _carries_export_provenance(keys: set[Any]) -> bool:
    """Does the block name the exporter that wrote it?

    ``note_id`` (Export Markdown) or ``id`` + ``created_at`` + ``updated_at``
    (a bundle's note file) -- the two markers ruling F-1 names.
    """
    return _EXPORT_MARKDOWN_PROVENANCE_KEY in keys or _BUNDLE_PROVENANCE_KEYS <= keys


def _is_own_export_block(metadata: Mapping[Any, Any]) -> bool:
    """Can this block leave the body without losing anything the user wrote?

    Outside a vault that is the only block worth removing from the body: it
    round-trips the app's own ``Export Markdown`` file and bundles while a
    block holding anything else -- a user's ``Summary:``, a ``Yes:`` key the
    YAML reader typed as ``True`` -- stays in the note exactly as written.

    Every key must be one the import honours or one the exporters write, and
    an exporter's key (``date``, ``id``, ...) counts as noise only when the
    block carries that exporter's provenance (final review I-2, ruling F-1):
    an Obsidian daily note's ``{date, tags}`` or a Hugo post's ``{title, date,
    tags}`` holds nothing but such keys, yet its ``date`` is the user's. A
    block of only honoured keys (``title``, ``tags``, ``keywords``,
    ``aliases``) qualifies on its own: all of it is read back.
    """
    # Exact match, the way the readers match (``title``, never ``Title``):
    # a key the readers would not read is a key the user would lose.
    keys = set(metadata)
    if not keys <= _STRIPPABLE_OUTSIDE_A_VAULT:
        return False
    return not keys & _OWN_EXPORT_KEYS or _carries_export_provenance(keys)


def _unimported_frontmatter_keys(
    metadata: Mapping[Any, Any] | None,
    bounds: ImportBounds | None = None,
) -> tuple[str, ...]:
    """Return what this import drops from a stripped frontmatter block, in file order.

    Property names that reach neither the title, the keywords nor the stored
    body; a non-string key as ``str(key)`` (``Yes:`` reads as ``True``); a
    ``title`` that is not text; and, when ``bounds`` is given, a ``tags`` /
    ``keywords`` / ``aliases`` value the import could not keep -- too many
    entries, or not readable as keywords -- named with its reason. Nothing a
    stripped block held disappears silently (TASK-34000.31 fix round 1).

    The app's own provenance keys (``date``, ``id``, ...) are left out ONLY
    when the whole block is the app's own export, provenance marker included
    (``_is_own_export_block``); beside a ``mood:`` -- or beside only ``tags:``
    -- a ``date:`` is the user's Obsidian property and is reported like any
    other (fix round 2, R2-3; final review I-2). Keyword reports come first
    and are never cut by the cap -- they are the only sign the keywords were
    lost (R2-5).

    Args:
        metadata: One parsed frontmatter mapping, or None when there was none.
        bounds: The import bounds the keywords are checked against; None skips
            the keyword report (callers that never keep keywords).

    Returns:
        Bounded, deduplicated entries for the review row's "not imported".
    """
    if metadata is None:
        return ()
    own_export = _is_own_export_block(metadata)
    names: list[str] = []
    for key in metadata:
        if not isinstance(key, str):
            name = str(key).strip()
            if name:
                names.append(name[:MAX_IMPORT_KEYWORD_LENGTH])
            continue
        name = key.strip()
        if not name:
            continue
        if key == "title":
            if metadata[key] is not None and _frontmatter_title(metadata) is None:
                names.append(name[:MAX_IMPORT_KEYWORD_LENGTH])
            continue
        if key in _FRONTMATTER_KEYS or (own_export and key in _OWN_EXPORT_KEYS):
            continue
        names.append(name[:MAX_IMPORT_KEYWORD_LENGTH])
    reports = _unkept_keyword_reports(metadata, bounds) if bounds is not None else []
    capped = tuple(dict.fromkeys(names))[: max(0, _MAX_REPORTED_FRONTMATTER_KEYS - len(reports))]
    return tuple(dict.fromkeys([*reports, *capped]))


def _unkept_keyword_reports(metadata: Mapping[Any, Any], bounds: ImportBounds) -> list[str]:
    """Name each keyword property ``_frontmatter_keywords`` had to leave out."""
    reports: list[str] = []
    for key in _FRONTMATTER_KEYS:
        if key not in metadata or metadata[key] is None:
            continue
        try:
            _keywords(metadata[key], bounds)
        except _ParseFailure as error:
            if error.reason_code == "too_many_keywords":
                reports.append(f"{key} (more than {bounds.max_keywords_per_note})")
            else:
                reports.append(f"{key} (not readable as keywords)")
    # The combined cap is applied to the DEDUPLICATED list (R2-6): 60 tags
    # repeated as 60 keywords are 60 keywords, all kept.
    if len(_collect_frontmatter_keywords(metadata, bounds)) > bounds.max_keywords_per_note:
        reports.append(f"keywords (more than {bounds.max_keywords_per_note} in total)")
    return reports


def _frontmatter_keywords(
    metadata: Mapping[Any, Any] | None,
    bounds: ImportBounds,
) -> tuple[str, ...]:
    """Return `tags`, then `keywords`, then `aliases` as deduplicated keywords.

    Aliases are alternate titles in Obsidian and there is no separate note
    metadata store here, so they are kept as keywords: the note stays findable by
    them and they are visible in Info. They carry an ``alias:`` prefix so a
    reader can still tell an alternate name from a tag the author chose
    (task-32178). Unusable metadata yields no keywords rather than failing the
    whole note.
    """
    if metadata is None:
        return ()
    return tuple(_collect_frontmatter_keywords(metadata, bounds)[: bounds.max_keywords_per_note])


def _collect_frontmatter_keywords(
    metadata: Mapping[Any, Any], bounds: ImportBounds
) -> list[str]:
    """The deduplicated keyword list before the per-note cap (shared by the report)."""
    keywords: list[str] = []
    seen: set[str] = set()
    for key in _FRONTMATTER_KEYS:
        try:
            values = _keywords(metadata.get(key), bounds)
        except _ParseFailure:
            continue
        for value in values:
            keyword = value if key in _TAG_KEYS else f"{_ALIAS_KEYWORD_PREFIX}{value}"
            if len(keyword) > MAX_IMPORT_KEYWORD_LENGTH:
                # `_keywords` already bounded `value`, so only our display
                # prefix pushed it over. Keep the alternate name findable
                # un-prefixed rather than dropping it (PR #2556 review).
                keyword = value
            if keyword.casefold() in seen:
                continue
            seen.add(keyword.casefold())
            keywords.append(keyword)
    return keywords


def _wikilinks(text: str) -> tuple[str, ...]:
    """Return the note's non-embedded `[[targets]]` in first-use order.

    Anything inside a code span is sample text, not a link, and is skipped here
    exactly as the executor skips it when rewriting.
    """
    targets: list[str] = []
    seen: set[str] = set()
    for match in WIKILINK_SCAN.finditer(text):
        target = wikilink_target(match)
        if not target or target in seen or len(target) > MAX_IMPORT_TITLE_LENGTH:
            continue
        seen.add(target)
        targets.append(target)
        if len(targets) >= _MAX_WIKILINKS_PER_NOTE:
            break
    return tuple(targets)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _ParseFailure("invalid_content")
        result[key] = value
    return result


def _structured_payloads(
    value: Any,
    bounds: ImportBounds,
) -> tuple[ParsedNotePayload, ...]:
    records = value if isinstance(value, list) else [value]
    if not records:
        raise _ParseFailure("empty_structured_source")
    if len(records) > bounds.max_notes_per_file:
        raise _ParseFailure("too_many_notes")
    note_shaped = tuple(_is_note_record(record) for record in records)
    if not any(note_shaped):
        # A well-formed document that holds no note record is configuration.
        raise _ParseFailure("not_a_note")
    if not all(note_shaped):
        # Part note, part something else: a damaged export, not configuration.
        raise _ParseFailure(
            "invalid_content",
            f"Record {note_shaped.index(False) + 1} of {len(records)} "
            "has no note content.",
        )
    payloads: list[ParsedNotePayload] = []
    for position, record in enumerate(records, start=1):
        try:
            payloads.append(_payload_from_mapping(record, bounds))
        except _ParseFailure as error:
            if error.reason_code != "invalid_content":
                raise
            raise _ParseFailure(
                "invalid_content",
                f"Record {position} of {len(records)} could not be read as a note.",
            ) from error
    return tuple(payloads)


def _is_note_record(record: Any) -> bool:
    """Report whether one structured record carries a note body at all."""
    return isinstance(record, Mapping) and any(
        alias in record for alias in _CONTENT_ALIASES
    )


def _payload_from_mapping(
    record: Mapping[Any, Any],
    bounds: ImportBounds,
) -> ParsedNotePayload:
    # task-32619: `detail` also names which role failed ("row" for a
    # shape-level problem, else the mapping key CSV filled from a column) so
    # a caller that knows the source's own column labels -- only the CSV
    # path currently does -- can name the exact cell instead of the row.
    if not all(isinstance(key, str) for key in record):
        raise _ParseFailure("invalid_content", "row")
    if any(
        sum(alias in record for alias in aliases) > 1
        for aliases in (_TITLE_ALIASES, _CONTENT_ALIASES, _KEYWORD_ALIASES)
    ):
        raise _ParseFailure("invalid_content", "row")
    # Whether a record is note-shaped at all is decided for the whole document
    # in _structured_payloads; a body-less record only reaches here from CSV,
    # which always supplies one.
    content_value = record.get("content", record.get("body"))
    if not isinstance(content_value, str) or not content_value.strip():
        raise _ParseFailure("invalid_content", "content")
    title_value = record.get("title", record.get("name", "Untitled"))
    if not isinstance(title_value, str):
        raise _ParseFailure("invalid_content", "title")
    if not title_value.strip():
        title_value = "Untitled"
    if len(title_value) > MAX_IMPORT_TITLE_LENGTH:
        raise _ParseFailure("invalid_content", "title")
    keywords_value = record.get("keywords", record.get("tags"))
    keywords = _keywords(keywords_value, bounds)
    template = record.get("template")
    if template is not None and not isinstance(template, str):
        raise _ParseFailure("invalid_content", "template")
    if template is not None and len(template) > MAX_IMPORT_TEMPLATE_NAME_LENGTH:
        raise _ParseFailure("invalid_content", "template")
    return ParsedNotePayload(
        title=title_value,
        content=content_value,
        keywords=keywords,
        template_name=template,
    )


def _keywords(value: Any, bounds: ImportBounds) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        if not value.strip():
            return ()
        raw_keywords = value.split(",")
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        raw_keywords = value
    else:
        raise _ParseFailure("invalid_content", "keywords")
    keywords = tuple(keyword.strip() for keyword in raw_keywords)
    if any(
        not keyword or len(keyword) > MAX_IMPORT_KEYWORD_LENGTH for keyword in keywords
    ):
        raise _ParseFailure("invalid_content", "keywords")
    if len(keywords) > bounds.max_keywords_per_note:
        raise _ParseFailure("too_many_keywords")
    return keywords


@contextmanager
def _bounded_csv_field_size_limit(bounds: ImportBounds) -> Iterator[None]:
    """Temporarily align CSV's global field limit with bounded source bytes."""
    with _CSV_FIELD_SIZE_LIMIT_LOCK:
        previous_limit = csv.field_size_limit()
        bounded_limit = min(bounds.max_file_bytes, MAX_IMPORT_FILE_BYTES)
        parse_limit = max(previous_limit, bounded_limit)
        csv.field_size_limit(parse_limit)
        try:
            yield
        finally:
            csv.field_size_limit(previous_limit)


def _csv_payloads(text: str, bounds: ImportBounds) -> tuple[ParsedNotePayload, ...]:
    with _bounded_csv_field_size_limit(bounds):
        reader = csv.reader(io.StringIO(text), strict=True)
        try:
            headers = next(reader)
        except StopIteration as error:
            raise _ParseFailure("empty_structured_source") from error
        normalized_headers = tuple(
            normalize("NFKC", header.strip()).casefold() for header in headers
        )
        if (
            len(headers) < 2
            or any(not header for header in normalized_headers)
            or len(set(normalized_headers)) != len(normalized_headers)
        ):
            raise _ParseFailure("invalid_content")

        title_index = _role_header_index(normalized_headers, _TITLE_ALIASES)
        content_index = _role_header_index(normalized_headers, _CONTENT_ALIASES)
        keyword_index = _role_header_index(normalized_headers, _KEYWORD_ALIASES)
        if title_index is None and content_index is None:
            title_index, content_index = _generic_header_indexes(normalized_headers)
        elif title_index is None:
            title_index = _fallback_header_index(normalized_headers)
        elif content_index is None:
            content_index = _fallback_header_index(normalized_headers)
        template_index = _first_header(normalized_headers, ("template",))
        # task-32619: the raw (un-normalized) header text a failure names,
        # so "fix the X cell" points at the column exactly as the user
        # spelled it in their own file.
        column_labels = {"title": headers[title_index], "content": headers[content_index]}
        if keyword_index is not None:
            column_labels["keywords"] = headers[keyword_index]
        if template_index is not None:
            column_labels["template"] = headers[template_index]

        def _row_failure(payloads: list[ParsedNotePayload], field: str) -> _ParseFailure:
            where = (
                f'the "{column_labels[field]}" cell'
                if field in column_labels
                else "that row"
            )
            count = len(payloads)
            # task-32619 fix round 1 (F1): the CSV path is atomic -- a failed
            # row rejects the whole file, so no row is ever actually
            # imported. "N row(s) imported so far" said otherwise. Names how
            # many rows READ cleanly before the bad one, and says plainly
            # that none of them were imported -- both halves have to fit
            # inside bounds.max_reason_length (as low as 120 in this test
            # suite), hence the terse "Read N, imported none" shape.
            read = "no rows" if count == 0 else "1 row" if count == 1 else f"{count} rows"
            return _ParseFailure(
                "invalid_content",
                f"Read {read}, imported none. Row {reader.line_num}: "
                f"fix {where}, then import again.",
            )

        payloads: list[ParsedNotePayload] = []
        for row in reader:
            if len(payloads) >= bounds.max_notes_per_file:
                raise _ParseFailure("too_many_notes")
            if len(row) != len(headers) or not any(cell for cell in row):
                raise _row_failure(payloads, "row")
            mapping: dict[str, Any] = {
                "title": row[title_index],
                "content": row[content_index],
            }
            if keyword_index is not None:
                mapping["keywords"] = row[keyword_index]
            if template_index is not None:
                mapping["template"] = row[template_index] or None
            try:
                payloads.append(_payload_from_mapping(mapping, bounds))
            except _ParseFailure as error:
                if error.reason_code != "invalid_content":
                    raise
                raise _row_failure(payloads, error.detail) from error
        if not payloads:
            raise _ParseFailure("empty_structured_source")
        return tuple(payloads)


def _first_header(headers: tuple[str, ...], names: tuple[str, ...]) -> int | None:
    for name in names:
        if name in headers:
            return headers.index(name)
    return None


def _role_header_index(
    headers: tuple[str, ...],
    aliases: tuple[str, ...],
) -> int | None:
    indexes = [index for index, header in enumerate(headers) if header in aliases]
    if len(indexes) > 1:
        raise _ParseFailure("invalid_content")
    return indexes[0] if indexes else None


def _fallback_header_index(headers: tuple[str, ...]) -> int:
    for index, header in enumerate(headers):
        if header not in _CSV_RESERVED_HEADERS:
            return index
    raise _ParseFailure("invalid_content")


def _generic_header_indexes(headers: tuple[str, ...]) -> tuple[int, int]:
    indexes = tuple(
        index
        for index, header in enumerate(headers)
        if header not in _CSV_RESERVED_HEADERS
    )
    if len(indexes) < 2:
        raise _ParseFailure("invalid_content")
    return indexes[0], indexes[1]


def _folder_segments(
    candidate: DiscoveredImportSource,
    destination: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if candidate.source.kind is ImportSourceKind.SELECTED_FILE:
        if destination is None:
            raise _ParseFailure("invalid_content")
        return destination
    parts = PurePosixPath(candidate.source.display_path).parts[:-1]
    if not parts:
        raise _ParseFailure("invalid_content")
    return parts


def _proposed_folder_paths(
    parsed: list[ParsedImportSource],
) -> tuple[tuple[str, ...], ...]:
    paths: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for source in parsed:
        for membership in source.memberships:
            for depth in range(1, len(membership.folder_segments) + 1):
                path = membership.folder_segments[:depth]
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
    return tuple(paths)


def _issue(
    candidate: DiscoveredImportSource,
    bounds: ImportBounds,
    classification: ImportClassification,
    reason_code: str,
    *,
    detail: str = "",
) -> ImportParseIssue:
    return ImportParseIssue(
        display_path=candidate.source.display_path,
        source_path=candidate.source.source_path,
        classification=classification,
        reason_code=reason_code,
        user_message=_message(bounds, reason_code, detail=detail),
    )


def _message(bounds: ImportBounds, reason_code: str, *, detail: str = "") -> str:
    """Return one bounded reason, followed by any record-level detail."""
    message = _MESSAGES.get(reason_code, _MESSAGES["source_unavailable"])
    if detail:
        message = f"{message} {detail}"
    return message[: bounds.max_reason_length]


def _reject_selection(bounds: ImportBounds, reason_code: str) -> None:
    raise ImportSelectionError(reason_code, _message(bounds, reason_code))
