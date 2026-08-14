"""Immutable domain vocabulary for one-time Database Notes import plans.

The records in this module describe a read-only preview. They deliberately carry
no receipt fingerprints, persistence services, or execution behavior.
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


@dataclass(frozen=True, slots=True)
class ParsedNotePayload:
    """One parsed note produced by a source file."""

    title: str
    content: str
    keywords: tuple[str, ...] = ()
    template_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not isinstance(self.content, str):
            raise TypeError("Parsed note title and content must be text.")
        keywords = _as_tuple(self.keywords, field_name="keywords")
        if not all(isinstance(keyword, str) for keyword in keywords):
            raise ValueError("keywords must contain only text values.")
        if self.template_name is not None and not isinstance(self.template_name, str):
            raise ValueError("template_name must be text when provided.")
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
        if display_path.is_absolute() or ".." in display_path.parts:
            raise ValueError("display_path must be a safe relative path.")
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a Path.")


@dataclass(frozen=True, slots=True)
class ProposedFolderMembership:
    """A proposed manual placement for one payload from a source item."""

    payload_index: int
    folder_segments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.payload_index, int) or self.payload_index < 0:
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
        if self.note_version is not None and (
            not isinstance(self.note_version, int) or self.note_version < 0
        ):
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

    def __post_init__(self) -> None:
        positive_fields = (
            "max_files",
            "max_file_bytes",
            "max_total_bytes",
            "max_reason_length",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if not isinstance(self.max_depth, int) or self.max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer.")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("max_file_bytes cannot exceed max_total_bytes.")


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
    payloads: tuple[ParsedNotePayload, ...]
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

        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValueError("item_id must be non-empty text.")
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
        if len(self.reason) > 240 or "\x00" in self.reason:
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
        if (
            self.replace_content
            and self.selected_action is not ImportAction.UPDATE_EXISTING
        ):
            raise ValueError("replace_content requires Update existing.")
        if self.selected_action is ImportAction.SKIP and (
            self.replace_content or self.add_membership
        ):
            raise ValueError("Skip cannot replace content or add membership.")
        if not isinstance(self.replace_content, bool) or not isinstance(
            self.add_membership,
            bool,
        ):
            raise TypeError("replace_content and add_membership must be booleans.")
        for membership in memberships:
            if membership.payload_index >= len(payloads):
                raise ValueError("A membership payload index is outside payloads.")

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
        if self.root_collision is not None and not isinstance(
            self.root_collision,
            RootCollisionState,
        ):
            raise ValueError("root_collision must be RootCollisionState when provided.")

        object.__setattr__(self, "items", items)
        object.__setattr__(self, "proposed_folder_paths", tuple(paths))
