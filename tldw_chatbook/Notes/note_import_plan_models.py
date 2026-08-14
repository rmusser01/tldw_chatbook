"""Immutable domain vocabulary for one-time Database Notes import plans.

The records in this module describe a read-only preview. They deliberately carry
no receipt fingerprints, persistence services, or execution behavior. Generic
dataclass serialization such as :func:`dataclasses.asdict` is not safe for logs;
use :meth:`NoteImportPlan.to_diagnostic` for the supported redacted projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TypeVar


class ImportClassification(str, Enum):
    """Planner classification for one discovered source."""

    NEW = "new"
    UNCHANGED_REPEAT = "unchanged_repeat"
    CHANGED_REPEAT = "changed_repeat"
    UNCERTAIN_MATCH = "uncertain_match"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class ImportAction(str, Enum):
    """Action an approved preview item may request from a later executor."""

    SKIP = "skip"
    CREATE_NEW = "create_new"
    UPDATE_EXISTING = "update_existing"


class ImportSourceKind(str, Enum):
    """How a discovered file entered the import selection."""

    SELECTED_FILE = "selected_file"
    DIRECTORY_MEMBER = "directory_member"


class ImportMatchKind(str, Enum):
    """Confidence attached to a best-effort existing-note match."""

    EXACT = "exact"
    UNCERTAIN = "uncertain"
    USER_CONFIRMED = "user_confirmed"


class RootCollisionChoice(str, Enum):
    """Explicit resolution for a colliding imported root label."""

    USE_EXISTING = "use_existing"
    UNIQUE_SIBLING = "unique_sibling"
    RENAMED_ROOT = "renamed_root"


MAX_IMPORT_REASON_LENGTH = 1_024
"""Absolute safety ceiling for a public import-preview reason."""

MAX_IMPORT_DEPTH = 64
"""Absolute recursion-safe ceiling for one import discovery walk."""

MAX_IMPORT_FILES = 10_000
MAX_IMPORT_FILE_BYTES = 64 * 1024 * 1024
MAX_IMPORT_TOTAL_BYTES = 512 * 1024 * 1024
MAX_IMPORT_ENTRIES = 100_000
MAX_IMPORT_NOTES_PER_FILE = 10_000
MAX_IMPORT_KEYWORDS_PER_NOTE = 1_000
"""Absolute resource ceilings for one bounded import preview."""

MAX_IMPORT_TITLE_LENGTH = 4_096
MAX_IMPORT_TEMPLATE_NAME_LENGTH = 1_024
MAX_IMPORT_KEYWORD_LENGTH = 512
"""Absolute scalar lengths for parsed note metadata."""

MAX_IMPORT_ITEM_ID_LENGTH = 256
"""Absolute length ceiling for opaque preview item identifiers."""


_EnumT = TypeVar("_EnumT", bound=Enum)


def _as_tuple(values: object, *, field_name: str) -> tuple[object, ...]:
    """Copy a caller-owned collection into a tuple."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection, not text.")
    try:
        return tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field_name} must be a collection.") from error


def _enum_tuple(
    values: object,
    enum_type: type[_EnumT],
    *,
    field_name: str,
) -> tuple[_EnumT, ...]:
    copied = _as_tuple(values, field_name=field_name)
    if not all(isinstance(value, enum_type) for value in copied):
        raise ValueError(f"{field_name} contains an invalid value.")
    return copied  # type: ignore[return-value]


def _validate_folder_segment(segment: str, *, field_name: str) -> None:
    if (
        not isinstance(segment, str)
        or not segment.strip()
        or segment != segment.strip()
        or segment in {".", ".."}
        or "/" in segment
        or "\\" in segment
        or "\x00" in segment
    ):
        raise ValueError(f"{field_name} contains an invalid folder segment.")


def _validate_import_item_id(value: object) -> str:
    """Return one bounded opaque preview identifier or fail closed."""
    if not isinstance(value, str):
        raise TypeError("item_id must be text.")
    if (
        not value
        or len(value) > MAX_IMPORT_ITEM_ID_LENGTH
        or not value.isascii()
        or any(not (character.isalnum() or character in "-_.:") for character in value)
    ):
        raise ValueError("item_id must be a safe opaque item identifier.")
    return value


@dataclass(frozen=True, slots=True)
class ParsedNotePayload:
    """One parsed note produced by a source file."""

    title: str = field(repr=False)
    content: str = field(repr=False)
    keywords: tuple[str, ...] = field(default=(), repr=False)
    template_name: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not isinstance(self.content, str):
            raise TypeError("Parsed note title and content must be text.")
        if not self.title.strip() or not self.content.strip():
            raise ValueError("Parsed note title and content must be non-blank.")
        if len(self.title) > MAX_IMPORT_TITLE_LENGTH:
            raise ValueError("Parsed note title exceeds its absolute safety ceiling.")
        keywords = _as_tuple(self.keywords, field_name="keywords")
        if not all(isinstance(keyword, str) for keyword in keywords):
            raise ValueError("keywords must contain only text values.")
        if any(len(keyword) > MAX_IMPORT_KEYWORD_LENGTH for keyword in keywords):
            raise ValueError("A keyword exceeds its absolute safety ceiling.")
        if self.template_name is not None and not isinstance(self.template_name, str):
            raise ValueError("template_name must be text when provided.")
        if (
            self.template_name is not None
            and len(self.template_name) > MAX_IMPORT_TEMPLATE_NAME_LENGTH
        ):
            raise ValueError("template_name exceeds its absolute safety ceiling.")
        object.__setattr__(self, "keywords", keywords)


@dataclass(frozen=True, slots=True)
class ImportSource:
    """A source reference with a relative display path and private execution path."""

    kind: ImportSourceKind
    display_path: str
    source_path: Path = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ImportSourceKind):
            raise TypeError("kind must be an ImportSourceKind.")
        if not isinstance(self.display_path, str):
            raise TypeError("display_path must be text.")
        if not self.display_path:
            raise ValueError("display_path must be a non-empty relative path.")
        if "\\" in self.display_path or "\x00" in self.display_path:
            raise ValueError("display_path must be a safe relative POSIX path.")
        display_path = PurePosixPath(self.display_path)
        if (
            display_path == PurePosixPath(".")
            or display_path.is_absolute()
            or ".." in display_path.parts
        ):
            raise ValueError("display_path must be a safe relative path.")
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a Path.")


@dataclass(frozen=True, slots=True)
class ProposedFolderMembership:
    """A proposed manual placement for one payload from a source item."""

    payload_index: int
    folder_segments: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.payload_index) is not int:
            raise TypeError("payload_index must be an integer.")
        if self.payload_index < 0:
            raise ValueError("payload_index must be a non-negative integer.")
        segments = _as_tuple(self.folder_segments, field_name="folder_segments")
        if not segments:
            raise ValueError("folder_segments must identify a folder.")
        for segment in segments:
            _validate_folder_segment(segment, field_name="folder_segments")
        object.__setattr__(self, "folder_segments", segments)


@dataclass(frozen=True, slots=True)
class ImportMatch:
    """Public reference to an existing note without private match fingerprints."""

    kind: ImportMatchKind
    note_id: str
    note_version: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ImportMatchKind):
            raise TypeError("kind must be an ImportMatchKind.")
        if not isinstance(self.note_id, str) or not self.note_id:
            raise ValueError("note_id must be non-empty text.")
        if self.note_version is not None:
            if type(self.note_version) is not int:
                raise TypeError("note_version must be an integer.")
            if self.note_version < 0:
                raise ValueError("note_version must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class RootCollisionState:
    """Collision status and, when supplied, its explicit user resolution."""

    proposed_label: str
    collides: bool
    choice: RootCollisionChoice | None = None
    resolved_label: str | None = None

    def __post_init__(self) -> None:
        _validate_folder_segment(self.proposed_label, field_name="proposed_label")
        if not isinstance(self.collides, bool):
            raise TypeError("collides must be a boolean.")
        if not self.collides:
            if self.choice is not None or self.resolved_label is not None:
                raise ValueError("A non-colliding root cannot have a resolution.")
            return
        if self.choice is None:
            if self.resolved_label is not None:
                raise ValueError(
                    "An unresolved collision cannot have a resolved label."
                )
            return
        if not isinstance(self.choice, RootCollisionChoice):
            raise TypeError("choice must be a RootCollisionChoice.")
        if self.choice in {
            RootCollisionChoice.UNIQUE_SIBLING,
            RootCollisionChoice.RENAMED_ROOT,
        }:
            if self.resolved_label is None:
                raise ValueError("This collision choice requires a resolved label.")
            _validate_folder_segment(
                self.resolved_label,
                field_name="resolved_label",
            )
        elif self.resolved_label is not None:
            raise ValueError("Use-existing does not accept a different resolved label.")


@dataclass(frozen=True, slots=True)
class ImportBounds:
    """Finite resource and diagnostic limits for one planning operation."""

    max_files: int
    max_file_bytes: int
    max_total_bytes: int
    max_depth: int
    max_reason_length: int = 240
    max_entries: int = 10_000
    max_notes_per_file: int = 1_000
    max_keywords_per_note: int = 100

    def __post_init__(self) -> None:
        positive_fields = (
            "max_files",
            "max_file_bytes",
            "max_total_bytes",
            "max_reason_length",
            "max_entries",
            "max_notes_per_file",
            "max_keywords_per_note",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        absolute_ceilings = {
            "max_files": MAX_IMPORT_FILES,
            "max_file_bytes": MAX_IMPORT_FILE_BYTES,
            "max_total_bytes": MAX_IMPORT_TOTAL_BYTES,
            "max_entries": MAX_IMPORT_ENTRIES,
            "max_notes_per_file": MAX_IMPORT_NOTES_PER_FILE,
            "max_keywords_per_note": MAX_IMPORT_KEYWORDS_PER_NOTE,
        }
        for field_name, ceiling in absolute_ceilings.items():
            if getattr(self, field_name) > ceiling:
                raise ValueError(f"{field_name} exceeds its absolute safety ceiling.")
        if type(self.max_depth) is not int:
            raise TypeError("max_depth must be an integer.")
        if self.max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer.")
        if self.max_depth > MAX_IMPORT_DEPTH:
            raise ValueError("max_depth exceeds the absolute safety ceiling.")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("max_file_bytes cannot exceed max_total_bytes.")
        if self.max_reason_length > MAX_IMPORT_REASON_LENGTH:
            raise ValueError(
                "max_reason_length cannot exceed the absolute safety ceiling."
            )


_DEFAULT_ACTIONS = {
    ImportClassification.NEW: ImportAction.CREATE_NEW,
    ImportClassification.UNCHANGED_REPEAT: ImportAction.SKIP,
    ImportClassification.CHANGED_REPEAT: ImportAction.CREATE_NEW,
    ImportClassification.UNCERTAIN_MATCH: ImportAction.CREATE_NEW,
    ImportClassification.UNSUPPORTED: ImportAction.SKIP,
    ImportClassification.FAILED: ImportAction.SKIP,
}


@dataclass(frozen=True, slots=True)
class ImportPreviewItem:
    """One immutable source-level preview with proposed note outcomes."""

    item_id: str
    source: ImportSource
    payloads: tuple[ParsedNotePayload, ...] = field(repr=False)
    memberships: tuple[ProposedFolderMembership, ...]
    classification: ImportClassification
    reason: str
    default_action: ImportAction
    selected_action: ImportAction
    allowed_actions: tuple[ImportAction, ...]
    match: ImportMatch | None
    replace_content: bool
    add_membership: bool

    def __post_init__(self) -> None:
        payloads = _as_tuple(self.payloads, field_name="payloads")
        memberships = _as_tuple(self.memberships, field_name="memberships")
        allowed_actions = _enum_tuple(
            self.allowed_actions,
            ImportAction,
            field_name="allowed_actions",
        )
        object.__setattr__(self, "payloads", payloads)
        object.__setattr__(self, "memberships", memberships)
        object.__setattr__(self, "allowed_actions", allowed_actions)

        _validate_import_item_id(self.item_id)
        if not isinstance(self.source, ImportSource):
            raise TypeError("source must be an ImportSource.")
        if self.match is not None and not isinstance(self.match, ImportMatch):
            raise TypeError("match must be an ImportMatch when provided.")
        if not all(isinstance(payload, ParsedNotePayload) for payload in payloads):
            raise ValueError("payloads must contain ParsedNotePayload values.")
        if not all(
            isinstance(membership, ProposedFolderMembership)
            for membership in memberships
        ):
            raise ValueError(
                "memberships must contain ProposedFolderMembership values."
            )
        if not isinstance(self.classification, ImportClassification):
            raise TypeError("classification must be an ImportClassification.")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be text.")
        if (
            not self.reason.strip()
            or len(self.reason) > MAX_IMPORT_REASON_LENGTH
            or "\x00" in self.reason
        ):
            raise ValueError("reason must be bounded, safe text.")
        if not isinstance(self.default_action, ImportAction) or not isinstance(
            self.selected_action,
            ImportAction,
        ):
            raise TypeError("default and selected actions must be ImportAction values.")
        if not allowed_actions:
            raise ValueError("allowed_actions cannot be empty.")
        if len(set(allowed_actions)) != len(allowed_actions):
            raise ValueError("allowed_actions cannot contain duplicates.")
        if not isinstance(self.replace_content, bool) or not isinstance(
            self.add_membership,
            bool,
        ):
            raise TypeError("replace_content and add_membership must be booleans.")

        if self.selected_action is ImportAction.UPDATE_EXISTING and (
            self.match is None
            or self.match.kind
            not in {
                ImportMatchKind.EXACT,
                ImportMatchKind.USER_CONFIRMED,
            }
        ):
            raise ValueError("Update requires an exact or user-confirmed match.")

        self._validate_classification_contract(allowed_actions)

        if self.default_action not in allowed_actions:
            raise ValueError("default_action must be present in allowed_actions.")
        if self.selected_action not in allowed_actions:
            raise ValueError("selected_action must be present in allowed_actions.")
        if self.default_action is not _DEFAULT_ACTIONS[self.classification]:
            raise ValueError("default_action does not match the classification.")
        importable = self.classification not in {
            ImportClassification.UNSUPPORTED,
            ImportClassification.FAILED,
        }
        if importable and not payloads:
            raise ValueError("Importable items require at least one payload.")

        covered_payload_indexes: set[int] = set()
        for membership in memberships:
            if membership.payload_index >= len(payloads):
                raise ValueError("A membership payload index is outside payloads.")
            covered_payload_indexes.add(membership.payload_index)
        expected_payload_indexes = set(range(len(payloads)))

        if (
            self.replace_content
            and self.selected_action is not ImportAction.UPDATE_EXISTING
        ):
            raise ValueError("replace_content requires Update existing.")
        if self.selected_action is ImportAction.CREATE_NEW and not self.add_membership:
            raise ValueError("Create new requires membership approval.")
        if self.add_membership and covered_payload_indexes != expected_payload_indexes:
            raise ValueError("membership must cover every payload.")
        if ImportAction.UPDATE_EXISTING in allowed_actions and (
            len(payloads) != 1 or self.match is None or self.match.note_version is None
        ):
            raise ValueError(
                "Update authorization requires one payload and a current note version."
            )
        if self.selected_action is ImportAction.UPDATE_EXISTING and not (
            self.replace_content or self.add_membership
        ):
            raise ValueError("Update must replace content or add membership.")
        if self.selected_action is ImportAction.SKIP and (
            self.replace_content or self.add_membership
        ):
            raise ValueError("Skip cannot replace content or add membership.")

    def _validate_classification_contract(
        self,
        allowed_actions: tuple[ImportAction, ...],
    ) -> None:
        """Reject classification, match, and action combinations that cannot run."""
        if self.classification in {
            ImportClassification.UNSUPPORTED,
            ImportClassification.FAILED,
        }:
            if allowed_actions != (ImportAction.SKIP,):
                raise ValueError("Unsupported and failed items must only allow Skip.")
            if self.match is not None:
                raise ValueError("Unsupported and failed items cannot carry a match.")
            return

        if self.classification is ImportClassification.NEW:
            if self.match is not None:
                raise ValueError("New items cannot carry an existing-note match.")
            expected_actions = {ImportAction.SKIP, ImportAction.CREATE_NEW}
        elif self.classification in {
            ImportClassification.UNCHANGED_REPEAT,
            ImportClassification.CHANGED_REPEAT,
        }:
            if self.match is None or self.match.kind is not ImportMatchKind.EXACT:
                raise ValueError("Repeat classifications require an exact match.")
            expected_actions = {
                ImportAction.SKIP,
                ImportAction.CREATE_NEW,
                ImportAction.UPDATE_EXISTING,
            }
        else:
            if self.match is None or self.match.kind not in {
                ImportMatchKind.UNCERTAIN,
                ImportMatchKind.USER_CONFIRMED,
            }:
                raise ValueError(
                    "Uncertain classifications require an uncertain or user-confirmed match."
                )
            expected_actions = {ImportAction.SKIP, ImportAction.CREATE_NEW}
            if self.match.kind is ImportMatchKind.USER_CONFIRMED:
                expected_actions.add(ImportAction.UPDATE_EXISTING)

        if set(allowed_actions) != expected_actions:
            raise ValueError(
                "allowed_actions do not match the classification and match."
            )


@dataclass(frozen=True, slots=True)
class ImportItemDiagnostic:
    """Redacted, content-free diagnostic view of one preview item."""

    source_display_path: str
    classification: ImportClassification
    selected_action: ImportAction
    payload_count: int
    membership_count: int


@dataclass(frozen=True, slots=True)
class ImportPlanDiagnostic:
    """Immutable diagnostic projection safe for structured logging."""

    item_count: int
    proposed_folder_count: int
    items: tuple[ImportItemDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class NoteImportPlan:
    """Aggregate immutable preview returned by the read-only planner."""

    bounds: ImportBounds
    items: tuple[ImportPreviewItem, ...]
    proposed_folder_paths: tuple[tuple[str, ...], ...]
    root_collision: RootCollisionState | None = None

    def __post_init__(self) -> None:
        items = _as_tuple(self.items, field_name="items")
        raw_paths = _as_tuple(
            self.proposed_folder_paths,
            field_name="proposed_folder_paths",
        )
        paths: list[tuple[str, ...]] = []
        for raw_path in raw_paths:
            segments = _as_tuple(raw_path, field_name="proposed_folder_paths")
            if not segments:
                raise ValueError("A proposed folder path cannot be empty.")
            for segment in segments:
                _validate_folder_segment(
                    segment,
                    field_name="proposed_folder_paths",
                )
            paths.append(segments)

        if not isinstance(self.bounds, ImportBounds):
            raise TypeError("bounds must be ImportBounds.")
        if not all(isinstance(item, ImportPreviewItem) for item in items):
            raise ValueError("items must contain ImportPreviewItem values.")
        if len({item.item_id for item in items}) != len(items):
            raise ValueError("Preview item identifiers must be unique.")
        selected_update_targets = tuple(
            item.match.note_id
            for item in items
            if item.selected_action is ImportAction.UPDATE_EXISTING
            and item.match is not None
        )
        if len(set(selected_update_targets)) != len(selected_update_targets):
            raise ValueError("The plan contains a duplicate update target.")
        if len(set(paths)) != len(paths):
            raise ValueError("Proposed folder paths cannot contain duplicates.")
        if any(len(item.reason) > self.bounds.max_reason_length for item in items):
            raise ValueError("An item reason exceeds bounds.max_reason_length.")
        if self.root_collision is not None and not isinstance(
            self.root_collision,
            RootCollisionState,
        ):
            raise ValueError("root_collision must be RootCollisionState when provided.")

        object.__setattr__(self, "items", items)
        object.__setattr__(self, "proposed_folder_paths", tuple(paths))

    def to_diagnostic(self) -> ImportPlanDiagnostic:
        """Return the only supported content-free serialization for logging."""
        diagnostic_items = tuple(
            ImportItemDiagnostic(
                source_display_path=item.source.display_path,
                classification=item.classification,
                selected_action=item.selected_action,
                payload_count=len(item.payloads),
                membership_count=len(item.memberships),
            )
            for item in self.items
        )
        return ImportPlanDiagnostic(
            item_count=len(self.items),
            proposed_folder_count=len(self.proposed_folder_paths),
            items=diagnostic_items,
        )
