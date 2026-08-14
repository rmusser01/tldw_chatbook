"""Public orchestration facade for one-time Database Notes import planning.

Planning is a pure, read-only transformation over an already parsed batch. Prior
observations are caller-owned, device-private inputs and are never copied into the
returned plan or its diagnostic projection.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from unicodedata import normalize

from tldw_chatbook.Notes.note_import_discovery import (
    DiscoveredImportSource,
    ImportDiscovery,
    ImportDiscoveryFailure,
    ImportSelectionError,
    SourceIdentity,
    discover_import_sources,
)
from tldw_chatbook.Notes.note_import_parsers import (
    SUPPORTED_NOTE_EXTENSIONS,
    ImportParseIssue,
    ParsedImportBatch,
    ParsedImportSource,
    parse_import_sources,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportAction,
    ImportBounds,
    ImportClassification,
    ImportMatch,
    ImportMatchKind,
    ImportPreviewItem,
    ImportSource,
    ImportSourceKind,
    NoteImportPlan,
    ParsedNotePayload,
)

_NEW_REASON = "Ready to import as a new note."
_UNCHANGED_REASON = "This source matches an unchanged existing note."
_CHANGED_REASON = "This source differs from an existing note."
_UNCERTAIN_REASON = "This source may match an existing note; review before updating."
_UNSUPPORTED_REASON = "This file type is not supported."
_FAILED_REASON = "This source could not be imported safely."


@dataclass(frozen=True, slots=True)
class PriorImportObservation:
    """Caller-supplied private evidence about one prior source-level import.

    Exact observations require a lowercase SHA-256 payload fingerprint. Uncertain
    observations intentionally carry no fingerprint. A source that parses into
    multiple notes cannot safely map this single-note observation to every payload;
    :func:`classify_import_batch` therefore degrades it to an uncertain match.
    """

    display_path: str
    match_kind: ImportMatchKind
    note_id: str
    note_version: int | None = None
    payload_fingerprint: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.display_path, str):
            raise TypeError("observation display_path must be text.")
        display_path = PurePosixPath(self.display_path)
        if (
            not self.display_path
            or display_path.is_absolute()
            or display_path == PurePosixPath(".")
            or ".." in display_path.parts
            or "\\" in self.display_path
            or "\x00" in self.display_path
        ):
            raise ValueError("observation display_path must be a safe relative path.")
        if not isinstance(self.match_kind, ImportMatchKind) or self.match_kind not in {
            ImportMatchKind.EXACT,
            ImportMatchKind.UNCERTAIN,
        }:
            raise ValueError("observation match_kind must be exact or uncertain.")
        if (
            not isinstance(self.note_id, str)
            or not self.note_id
            or len(self.note_id) > 256
            or not self.note_id.isascii()
            or any(
                not (character.isalnum() or character in "-_.:")
                for character in self.note_id
            )
        ):
            raise ValueError("observation note_id must be a safe opaque identifier.")
        if self.note_version is not None:
            if type(self.note_version) is not int:
                raise TypeError("observation note_version must be an integer.")
            if self.note_version < 0:
                raise ValueError("observation note_version must be non-negative.")
        if self.match_kind is ImportMatchKind.EXACT:
            if (
                not isinstance(self.payload_fingerprint, str)
                or len(self.payload_fingerprint) != 64
                or self.payload_fingerprint != self.payload_fingerprint.casefold()
                or any(
                    character not in "0123456789abcdef"
                    for character in self.payload_fingerprint
                )
            ):
                raise ValueError(
                    "An exact observation requires a lowercase SHA-256 fingerprint."
                )
        elif self.payload_fingerprint is not None:
            raise ValueError("An uncertain observation cannot carry a fingerprint.")


def _private_payload_fingerprint(
    payloads: Iterable[ParsedNotePayload],
) -> str:
    """Return a deterministic, device-private fingerprint of parsed payloads.

    The return value is matching material, not a public identifier. Callers must not
    place it in diagnostics or logs.
    """
    if isinstance(payloads, (str, bytes)):
        raise TypeError("payloads must be a collection of parsed note payloads.")
    try:
        copied = tuple(payloads)
    except TypeError as error:
        raise TypeError(
            "payloads must be a collection of parsed note payloads."
        ) from error
    if not copied or not all(
        isinstance(payload, ParsedNotePayload) for payload in copied
    ):
        raise ValueError("payloads must contain at least one parsed note payload.")

    canonical_payloads = [
        {
            "type": "parsed_note_payload",
            "content": normalize("NFC", payload.content),
            "keywords": [normalize("NFC", keyword) for keyword in payload.keywords],
            "template_name": (
                normalize("NFC", payload.template_name)
                if payload.template_name is not None
                else None
            ),
            "title": normalize("NFC", payload.title),
        }
        for payload in copied
    ]
    canonical_bytes = json.dumps(
        {
            "payloads": canonical_payloads,
            "type": "tldw_note_import_payload_set",
            "version": 1,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def classify_import_batch(
    batch: ParsedImportBatch,
    bounds: ImportBounds,
    *,
    prior_observations: Iterable[PriorImportObservation] = (),
) -> NoteImportPlan:
    """Build an immutable preview without persistence or filesystem mutation.

    Observations are cardinality-one per relative source path. Unknown, duplicate,
    or ambiguous batch paths are rejected rather than guessed. A multi-note parsed
    source always degrades a source-level prior observation to ``UNCERTAIN_MATCH``.
    """
    if not isinstance(batch, ParsedImportBatch):
        raise TypeError("batch must be a ParsedImportBatch.")
    if not isinstance(bounds, ImportBounds):
        raise TypeError("bounds must be an ImportBounds.")
    observations = _validated_observations(
        prior_observations,
        max_observations=len(batch.parsed),
    )

    parsed_by_path = {
        _source_key(source.candidate.source.display_path): source
        for source in batch.parsed
    }
    issue_by_path = {_source_key(issue.display_path): issue for issue in batch.issues}
    source_count = len(batch.parsed) + len(batch.issues)
    all_paths = set(parsed_by_path) | set(issue_by_path)
    if len(all_paths) != source_count:
        raise ValueError("Import batch source display paths must be unique.")
    unknown_observations = set(observations) - set(parsed_by_path)
    if unknown_observations:
        raise ValueError("A prior observation refers to an unknown import source.")

    ordered_entries = sorted(
        (
            *(
                (source.candidate.source.display_path, "parsed", path)
                for path, source in parsed_by_path.items()
            ),
            *(
                (issue.display_path, "issue", path)
                for path, issue in issue_by_path.items()
            ),
        ),
        key=lambda entry: _display_sort_key(entry[0]),
    )
    items: list[ImportPreviewItem] = []
    for index, (_, entry_kind, source_key) in enumerate(ordered_entries, start=1):
        item_id = f"item-{index:06d}"
        if entry_kind == "parsed":
            items.append(
                _classify_parsed_source(
                    parsed_by_path[source_key],
                    observations.get(source_key),
                    item_id,
                    bounds,
                )
            )
        else:
            items.append(_issue_item(issue_by_path[source_key], item_id, bounds))

    return NoteImportPlan(
        bounds=bounds,
        items=tuple(items),
        proposed_folder_paths=batch.proposed_folder_paths,
    )


def _validated_observations(
    raw_observations: Iterable[PriorImportObservation],
    *,
    max_observations: int,
) -> dict[str, PriorImportObservation]:
    if isinstance(raw_observations, (str, bytes)):
        raise TypeError("prior observations must be a collection.")
    try:
        iterator = iter(raw_observations)
    except TypeError:
        raise TypeError("prior observations must be a collection.") from None
    except Exception:  # noqa: BLE001 - sanitize caller iterator failures
        raise ValueError("prior observations could not be read safely.") from None
    by_path: dict[str, PriorImportObservation] = {}
    index = 0
    while True:
        try:
            observation = next(iterator)
        except StopIteration:
            break
        except Exception:  # noqa: BLE001 - sanitize caller iterator failures
            raise ValueError("prior observations could not be read safely.") from None
        index += 1
        if not isinstance(observation, PriorImportObservation):
            raise TypeError(
                "prior observations must contain PriorImportObservation values."
            )
        source_key = _source_key(observation.display_path)
        if source_key in by_path:
            raise ValueError("prior observations contain a duplicate source path.")
        if index > max_observations:
            raise ValueError("prior observations contain too many source records.")
        by_path[source_key] = observation
    return by_path


def _classify_parsed_source(
    parsed: ParsedImportSource,
    observation: PriorImportObservation | None,
    item_id: str,
    bounds: ImportBounds,
) -> ImportPreviewItem:
    classification = ImportClassification.NEW
    reason = _NEW_REASON
    match: ImportMatch | None = None
    allowed_actions = (ImportAction.SKIP, ImportAction.CREATE_NEW)

    if observation is not None:
        match_kind = observation.match_kind
        if len(parsed.payloads) != 1:
            match_kind = ImportMatchKind.UNCERTAIN
        match = ImportMatch(
            kind=match_kind,
            note_id=observation.note_id,
            note_version=observation.note_version,
        )
        if match_kind is ImportMatchKind.UNCERTAIN:
            classification = ImportClassification.UNCERTAIN_MATCH
            reason = _UNCERTAIN_REASON
        else:
            current_fingerprint = _private_payload_fingerprint(parsed.payloads)
            if hmac.compare_digest(
                current_fingerprint,
                observation.payload_fingerprint or "",
            ):
                classification = ImportClassification.UNCHANGED_REPEAT
                reason = _UNCHANGED_REASON
            else:
                classification = ImportClassification.CHANGED_REPEAT
                reason = _CHANGED_REASON
            allowed_actions = (
                ImportAction.SKIP,
                ImportAction.CREATE_NEW,
                ImportAction.UPDATE_EXISTING,
            )

    default_action = (
        ImportAction.SKIP
        if classification is ImportClassification.UNCHANGED_REPEAT
        else ImportAction.CREATE_NEW
    )
    selected_action = default_action
    add_membership = selected_action is ImportAction.CREATE_NEW
    return ImportPreviewItem(
        item_id=item_id,
        source=parsed.candidate.source,
        payloads=parsed.payloads,
        memberships=parsed.memberships,
        classification=classification,
        reason=_bounded_reason(reason, bounds),
        default_action=default_action,
        selected_action=selected_action,
        allowed_actions=allowed_actions,
        match=match,
        replace_content=False,
        add_membership=add_membership,
    )


def _issue_item(
    issue: ImportParseIssue,
    item_id: str,
    bounds: ImportBounds,
) -> ImportPreviewItem:
    source_kind = (
        ImportSourceKind.SELECTED_FILE
        if issue.display_path == issue.source_path.name
        else ImportSourceKind.DIRECTORY_MEMBER
    )
    return ImportPreviewItem(
        item_id=item_id,
        source=ImportSource(
            kind=source_kind,
            display_path=issue.display_path,
            source_path=issue.source_path,
        ),
        payloads=(),
        memberships=(),
        classification=issue.classification,
        reason=_bounded_reason(
            _UNSUPPORTED_REASON
            if issue.classification is ImportClassification.UNSUPPORTED
            else _FAILED_REASON,
            bounds,
        ),
        default_action=ImportAction.SKIP,
        selected_action=ImportAction.SKIP,
        allowed_actions=(ImportAction.SKIP,),
        match=None,
        replace_content=False,
        add_membership=False,
    )


def _display_sort_key(value: str) -> tuple[str, str, str]:
    normalized = normalize("NFC", value)
    return (normalized.casefold(), normalized, value)


def _source_key(value: str) -> str:
    return normalize("NFC", value)


def _bounded_reason(value: str, bounds: ImportBounds) -> str:
    return value[: bounds.max_reason_length]


__all__ = [
    "SUPPORTED_NOTE_EXTENSIONS",
    "DiscoveredImportSource",
    "ImportDiscovery",
    "ImportDiscoveryFailure",
    "ImportParseIssue",
    "ImportSelectionError",
    "ParsedImportBatch",
    "ParsedImportSource",
    "PriorImportObservation",
    "SourceIdentity",
    "classify_import_batch",
    "discover_import_sources",
    "parse_import_sources",
]
