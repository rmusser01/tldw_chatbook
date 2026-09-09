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
_FRONTMATTER_KEYS = ("tags", "aliases")
_ALIAS_KEYWORD_PREFIX = "alias: "
"""Marks a keyword that came from `aliases:` rather than from `tags:`."""
_MAX_FRONTMATTER_LINES = 200
_MAX_WIKILINKS_PER_NOTE = 200
_TEMPLATE_PLACEHOLDER = re.compile(r"\{\{.*?\}\}")
"""A `{{date}}`-style template placeholder, which never becomes a note title."""


# ``csv.field_size_limit`` is process-global. Every CSV parse in this module holds
# this lock while raising and restoring the limit so overlapping imports cannot
# restore one another's value out of order.
_CSV_FIELD_SIZE_LIMIT_LOCK = RLock()


_MESSAGES = {
    "unsupported_extension": "This file type is not supported.",
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
                    "unsupported_extension",
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
        if obsidian_mode and markdown:
            metadata, body = _split_frontmatter(text)
            # A note that is only frontmatter (an Obsidian Properties-only file,
            # a templated daily note) still has to import: keep the original
            # text as the body rather than turning a note this mode understands
            # BETTER into a failure it did not have before.
            if body.strip():
                text = body
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


def _split_frontmatter(text: str) -> tuple[Mapping[Any, Any] | None, str]:
    """Split one leading `---` YAML block from the note body.

    Anything that is not a complete, safely loadable mapping is left in the body
    exactly as written, so an odd block degrades to today's behaviour.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, text
    for index, line in enumerate(lines[1 : _MAX_FRONTMATTER_LINES + 1], start=1):
        if line.strip() not in {"---", "..."}:
            continue
        block = "\n".join(lines[1:index])
        try:
            for token in yaml.scan(block):
                if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)):
                    return None, text
            metadata = yaml.load(block, Loader=_UniqueKeySafeLoader)
        except (yaml.YAMLError, _ParseFailure, RecursionError, ValueError, TypeError):
            return None, text
        if not isinstance(metadata, Mapping):
            return None, text
        return metadata, "\n".join(lines[index + 1 :]).lstrip("\n")
    return None, text


def _frontmatter_title(metadata: Mapping[Any, Any] | None) -> str | None:
    """Return the frontmatter title, unless it is blank or a placeholder."""
    if metadata is None:
        return None
    value = metadata.get("title")
    if not isinstance(value, str):
        return None
    title = value.strip()
    if not title or _TEMPLATE_PLACEHOLDER.search(title):
        return None
    return title


def _frontmatter_keywords(
    metadata: Mapping[Any, Any] | None,
    bounds: ImportBounds,
) -> tuple[str, ...]:
    """Return `tags` then `aliases` as deduplicated keywords.

    Aliases are alternate titles in Obsidian and there is no separate note
    metadata store here, so they are kept as keywords: the note stays findable by
    them and they are visible in Info. They carry an ``alias:`` prefix so a
    reader can still tell an alternate name from a tag the author chose
    (task-32178). Unusable metadata yields no keywords rather than failing the
    whole note.
    """
    if metadata is None:
        return ()
    keywords: list[str] = []
    seen: set[str] = set()
    for key in _FRONTMATTER_KEYS:
        try:
            values = _keywords(metadata.get(key), bounds)
        except _ParseFailure:
            continue
        for value in values:
            keyword = value if key == "tags" else f"{_ALIAS_KEYWORD_PREFIX}{value}"
            if (
                len(keyword) > MAX_IMPORT_KEYWORD_LENGTH
                or keyword.casefold() in seen
            ):
                continue
            seen.add(keyword.casefold())
            keywords.append(keyword)
    return tuple(keywords[: bounds.max_keywords_per_note])


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
    if not all(isinstance(key, str) for key in record):
        raise _ParseFailure("invalid_content")
    if any(
        sum(alias in record for alias in aliases) > 1
        for aliases in (_TITLE_ALIASES, _CONTENT_ALIASES, _KEYWORD_ALIASES)
    ):
        raise _ParseFailure("invalid_content")
    # Whether a record is note-shaped at all is decided for the whole document
    # in _structured_payloads; a body-less record only reaches here from CSV,
    # which always supplies one.
    content_value = record.get("content", record.get("body"))
    if not isinstance(content_value, str) or not content_value.strip():
        raise _ParseFailure("invalid_content")
    title_value = record.get("title", record.get("name", "Untitled"))
    if not isinstance(title_value, str):
        raise _ParseFailure("invalid_content")
    if not title_value.strip():
        title_value = "Untitled"
    if len(title_value) > MAX_IMPORT_TITLE_LENGTH:
        raise _ParseFailure("invalid_content")
    keywords_value = record.get("keywords", record.get("tags"))
    keywords = _keywords(keywords_value, bounds)
    template = record.get("template")
    if template is not None and not isinstance(template, str):
        raise _ParseFailure("invalid_content")
    if template is not None and len(template) > MAX_IMPORT_TEMPLATE_NAME_LENGTH:
        raise _ParseFailure("invalid_content")
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
        raise _ParseFailure("invalid_content")
    keywords = tuple(keyword.strip() for keyword in raw_keywords)
    if any(
        not keyword or len(keyword) > MAX_IMPORT_KEYWORD_LENGTH for keyword in keywords
    ):
        raise _ParseFailure("invalid_content")
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

        payloads: list[ParsedNotePayload] = []
        for row in reader:
            if len(payloads) >= bounds.max_notes_per_file:
                raise _ParseFailure("too_many_notes")
            if len(row) != len(headers) or not any(cell for cell in row):
                raise _ParseFailure("invalid_content")
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
                raise _ParseFailure(
                    "invalid_content",
                    f"Row {reader.line_num} could not be read as a note.",
                ) from error
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
