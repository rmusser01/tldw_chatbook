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
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from unicodedata import normalize

from tldw_chatbook.Notes.note_folder_models import (
    FolderValidationError,
    NormalizedFolderName,
    normalize_folder_name,
)
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
    ProposedFolderMembership,
    RootCollisionChoice,
    RootCollisionState,
    _validate_import_item_id,
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
            if self.note_version is None:
                raise ValueError(
                    "An exact observation requires a current note version."
                )
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
            "content": payload.content,
            "keywords": list(payload.keywords),
            "template_name": payload.template_name,
            "title": payload.title,
        }
        for payload in copied
    ]
    canonical_bytes = json.dumps(
        {
            "payloads": canonical_payloads,
            "type": "tldw_note_import_payload_set",
            "version": 1,
        },
        ensure_ascii=True,
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

    Args:
        batch: Parsed sources, issues, and proposed folder hierarchy.
        bounds: Resource and diagnostic limits governing the preview.
        prior_observations: Device-local prior import observations to compare.

    Returns:
        An immutable preview with classifications and safe default actions.

    Raises:
        TypeError: An argument has an invalid type.
        ValueError: Batch paths or prior observations are inconsistent.
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
    exact_note_ids: set[str] = set()
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
        if observation.match_kind is ImportMatchKind.EXACT:
            if observation.note_id in exact_note_ids:
                raise ValueError(
                    "prior observations contain a duplicate exact note target."
                )
            exact_note_ids.add(observation.note_id)
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


def analyze_root_collision(
    plan: NoteImportPlan,
    existing_top_level_names: Iterable[str],
) -> NoteImportPlan:
    """Return ``plan`` with immutable directory-root collision state.

    A manual destination used for selected files is not an imported directory root.
    Empty plans and plans whose selected actions create no memberships therefore
    carry no collision state.

    Args:
        plan: Preview whose proposed directory root should be analyzed.
        existing_top_level_names: Existing top-level folder labels.

    Returns:
        A new immutable plan carrying the applicable collision state.

    Raises:
        TypeError: ``plan`` or an existing folder name has an invalid type.
        ValueError: An existing folder name violates folder constraints.
    """
    _require_plan(plan)
    root_label = _meaningful_directory_root(plan)
    if root_label is None:
        if plan.root_collision is None:
            return plan
        return replace(plan, root_collision=None)

    existing_keys = _existing_folder_keys(existing_top_level_names, plan.bounds)
    root = _normalized_folder_name(root_label)
    return replace(
        plan,
        root_collision=RootCollisionState(
            proposed_label=root.display,
            collides=root.key in existing_keys,
        ),
    )


def resolve_root_collision(
    plan: NoteImportPlan,
    choice: RootCollisionChoice,
    *,
    existing_top_level_names: Iterable[str],
    renamed_root: str | None = None,
) -> NoteImportPlan:
    """Resolve one previously detected directory-root collision explicitly.

    Args:
        plan: Preview with one unresolved root collision.
        choice: Explicit resolution selected by the user.
        existing_top_level_names: Existing top-level folder labels.
        renamed_root: Replacement root label for ``RENAMED_ROOT``.

    Returns:
        A new immutable plan with the root collision resolved.

    Raises:
        TypeError: An argument has an invalid type.
        ValueError: The collision state or requested resolution is invalid.
    """
    _require_plan(plan)
    if not isinstance(choice, RootCollisionChoice):
        raise TypeError("choice must be a RootCollisionChoice.")
    root_label = _meaningful_directory_root(plan)
    if root_label is None:
        raise ValueError("The plan does not contain a meaningful directory root.")
    collision = plan.root_collision
    if collision is None or not collision.collides or collision.choice is not None:
        raise ValueError("Resolution requires one unresolved colliding root.")
    root = _normalized_folder_name(root_label)
    proposed = _normalized_folder_name(collision.proposed_label)
    if root.key != proposed.key:
        raise ValueError("The collision state does not match the proposed root.")

    existing_keys = _existing_folder_keys(existing_top_level_names, plan.bounds)
    if proposed.key not in existing_keys:
        raise ValueError("Resolution requires one unresolved colliding root.")

    if choice is RootCollisionChoice.USE_EXISTING:
        if renamed_root is not None:
            raise ValueError("Use-existing does not accept a replacement root.")
        return replace(
            plan,
            root_collision=RootCollisionState(
                proposed_label=collision.proposed_label,
                collides=True,
                choice=choice,
            ),
        )

    if choice is RootCollisionChoice.UNIQUE_SIBLING:
        if renamed_root is not None:
            raise ValueError("Unique-sibling chooses its replacement root.")
        resolved_label = _unique_sibling_label(
            collision.proposed_label,
            existing_keys,
            plan.bounds,
        )
    else:
        if renamed_root is None:
            raise ValueError("Renamed-root requires a replacement root.")
        resolved = _normalized_folder_name(renamed_root)
        if resolved.key in existing_keys:
            raise ValueError("The replacement root collides with an existing folder.")
        resolved_label = resolved.display

    return _rebase_root(plan, collision, choice, resolved_label)


def confirm_uncertain_match(plan: NoteImportPlan, item_id: str) -> NoteImportPlan:
    """Confirm one uncertain match without changing its classification.

    Args:
        plan: Preview containing the uncertain item.
        item_id: Stable identifier of the item to confirm.

    Returns:
        A new plan whose item permits an explicit update action.

    Raises:
        TypeError: ``plan`` or ``item_id`` has an invalid type.
        ValueError: The item is missing or cannot be confirmed for update.
    """
    _require_plan(plan)
    item_index, item = _find_item(plan, item_id)
    if (
        item.classification is not ImportClassification.UNCERTAIN_MATCH
        or item.match is None
        or item.match.kind is not ImportMatchKind.UNCERTAIN
    ):
        raise ValueError("Only an uncertain match can be explicitly confirmed.")
    if len(item.payloads) != 1 or item.match.note_version is None:
        raise ValueError("This uncertain match cannot be confirmed for update.")
    _require_unique_update_authorization(
        plan,
        item_index,
        item.match.note_id,
    )
    confirmed = replace(
        item,
        match=replace(item.match, kind=ImportMatchKind.USER_CONFIRMED),
        allowed_actions=(
            ImportAction.SKIP,
            ImportAction.CREATE_NEW,
            ImportAction.UPDATE_EXISTING,
        ),
    )
    return _replace_item(plan, item_index, confirmed)


def apply_item_override(
    plan: NoteImportPlan,
    item_id: str,
    action: ImportAction,
    *,
    replace_content: bool = False,
    add_membership: bool = False,
) -> NoteImportPlan:
    """Apply one validated action and effect choice to a frozen preview item.

    Args:
        plan: Preview containing the item to override.
        item_id: Stable identifier of the item to override.
        action: Import action selected by the user.
        replace_content: Whether an update replaces note content.
        add_membership: Whether an update adds proposed folder membership.

    Returns:
        A new immutable plan with the validated override applied.

    Raises:
        TypeError: An argument has an invalid type.
        ValueError: The item or requested action/effects are invalid.
    """
    _require_plan(plan)
    if not isinstance(action, ImportAction):
        raise TypeError("action must be an ImportAction.")
    if type(replace_content) is not bool or type(add_membership) is not bool:
        raise TypeError("override effects must be booleans.")
    item_index, item = _find_item(plan, item_id)
    if action not in item.allowed_actions:
        raise ValueError("The requested action is not allowed for this item.")

    if action is ImportAction.SKIP:
        requested_replace = False
        requested_membership = False
    elif action is ImportAction.CREATE_NEW:
        if replace_content:
            raise ValueError("Create new cannot replace existing content.")
        requested_replace = False
        requested_membership = True
    else:
        if not (replace_content or add_membership):
            raise ValueError("Update must replace content or add membership.")
        if item.match is None:
            raise ValueError("Update requires one authorized note target.")
        _require_unique_selected_update_target(
            plan,
            item_index,
            item.match.note_id,
        )
        requested_replace = replace_content
        requested_membership = add_membership

    updated = replace(
        item,
        selected_action=action,
        replace_content=requested_replace,
        add_membership=requested_membership,
    )
    return _replace_item_and_recompute_folders(plan, item_index, updated)


def _require_plan(plan: NoteImportPlan) -> None:
    if not isinstance(plan, NoteImportPlan):
        raise TypeError("plan must be a NoteImportPlan.")


def _require_unique_update_authorization(
    plan: NoteImportPlan,
    item_index: int,
    note_id: str,
) -> None:
    if any(
        index != item_index
        and ImportAction.UPDATE_EXISTING in other.allowed_actions
        and other.match is not None
        and other.match.note_id == note_id
        for index, other in enumerate(plan.items)
    ):
        raise ValueError("The plan contains a duplicate update target.")


def _require_unique_selected_update_target(
    plan: NoteImportPlan,
    item_index: int,
    note_id: str,
) -> None:
    if any(
        index != item_index
        and other.selected_action is ImportAction.UPDATE_EXISTING
        and other.match is not None
        and other.match.note_id == note_id
        for index, other in enumerate(plan.items)
    ):
        raise ValueError("The plan contains a duplicate update target.")


def _normalized_folder_name(value: str) -> NormalizedFolderName:
    try:
        return normalize_folder_name(value)
    except FolderValidationError:
        raise ValueError("Folder inputs must contain valid folder names.") from None
    except Exception:  # noqa: BLE001 - sanitize foreign validation failures
        raise ValueError("Folder inputs could not be validated safely.") from None


def _existing_folder_keys(
    raw_names: Iterable[str],
    bounds: ImportBounds,
) -> frozenset[str]:
    if isinstance(raw_names, (str, bytes)):
        raise TypeError("existing folder names must be a collection.")
    try:
        iterator = iter(raw_names)
    except TypeError:
        raise TypeError("existing folder names must be a collection.") from None
    except Exception:  # noqa: BLE001 - sanitize caller iterator failures
        raise ValueError("existing folder names could not be read safely.") from None

    keys: set[str] = set()
    count = 0
    while True:
        try:
            value = next(iterator)
        except StopIteration:
            break
        except Exception:  # noqa: BLE001 - sanitize caller iterator failures
            raise ValueError(
                "existing folder names could not be read safely."
            ) from None
        count += 1
        if count > bounds.max_entries:
            raise ValueError("existing folder names contain too many values.")
        if not isinstance(value, str):
            raise TypeError("existing folder names must contain text values.")
        keys.add(_normalized_folder_name(value).key)
    return frozenset(keys)


def _meaningful_directory_root(plan: NoteImportPlan) -> str | None:
    if not plan.proposed_folder_paths:
        return None
    relevant_items = tuple(
        item
        for item in plan.items
        if item.source.kind is ImportSourceKind.DIRECTORY_MEMBER and item.add_membership
    )
    if not relevant_items:
        return None

    root_label = plan.proposed_folder_paths[0][0]
    if any(path[0] != root_label for path in plan.proposed_folder_paths):
        raise ValueError("Proposed folders do not share one directory root.")
    for item in plan.items:
        for membership in item.memberships:
            if membership.folder_segments[0] != root_label:
                raise ValueError(
                    "Proposed memberships do not share the directory root."
                )
    return root_label


def _unique_sibling_label(
    proposed_label: str,
    existing_keys: frozenset[str],
    bounds: ImportBounds,
) -> str:
    for sequence in range(2, bounds.max_entries + 3):
        suffix = f" ({sequence})"
        available = 255 - len(suffix)
        base = proposed_label[:available].rstrip()
        if not base:
            raise ValueError("A unique sibling root could not be generated safely.")
        candidate = _normalized_folder_name(f"{base}{suffix}")
        if candidate.key not in existing_keys:
            return candidate.display
    raise ValueError("A unique sibling root could not be generated safely.")


def _rebase_root(
    plan: NoteImportPlan,
    collision: RootCollisionState,
    choice: RootCollisionChoice,
    resolved_label: str,
) -> NoteImportPlan:
    original_label = plan.proposed_folder_paths[0][0]
    paths = tuple((resolved_label, *path[1:]) for path in plan.proposed_folder_paths)
    items: list[ImportPreviewItem] = []
    for item in plan.items:
        memberships: list[ProposedFolderMembership] = []
        for membership in item.memberships:
            if membership.folder_segments[0] != original_label:
                raise ValueError(
                    "Proposed memberships do not share the directory root."
                )
            memberships.append(
                replace(
                    membership,
                    folder_segments=(
                        resolved_label,
                        *membership.folder_segments[1:],
                    ),
                )
            )
        items.append(replace(item, memberships=tuple(memberships)))
    return replace(
        plan,
        items=tuple(items),
        proposed_folder_paths=paths,
        root_collision=RootCollisionState(
            proposed_label=collision.proposed_label,
            collides=True,
            choice=choice,
            resolved_label=resolved_label,
        ),
    )


def _find_item(
    plan: NoteImportPlan,
    item_id: str,
) -> tuple[int, ImportPreviewItem]:
    validated_item_id = _validate_import_item_id(item_id)
    matches = tuple(
        (index, item)
        for index, item in enumerate(plan.items)
        if item.item_id == validated_item_id
    )
    if len(matches) != 1:
        raise ValueError("The item identifier must match exactly one preview item.")
    return matches[0]


def _replace_item(
    plan: NoteImportPlan,
    item_index: int,
    item: ImportPreviewItem,
) -> NoteImportPlan:
    items = list(plan.items)
    items[item_index] = item
    return replace(plan, items=tuple(items))


def _replace_item_and_recompute_folders(
    plan: NoteImportPlan,
    item_index: int,
    item: ImportPreviewItem,
) -> NoteImportPlan:
    items = list(plan.items)
    items[item_index] = item
    frozen_items = tuple(items)
    proposed_folder_paths = _selected_folder_ancestor_closure(frozen_items)
    return replace(
        plan,
        items=frozen_items,
        proposed_folder_paths=proposed_folder_paths,
        root_collision=_collision_after_item_override(
            plan.root_collision,
            frozen_items,
        ),
    )


def _selected_folder_ancestor_closure(
    items: tuple[ImportPreviewItem, ...],
) -> tuple[tuple[str, ...], ...]:
    paths: set[tuple[str, ...]] = set()
    for item in items:
        if not item.add_membership:
            continue
        for membership in item.memberships:
            for depth in range(1, len(membership.folder_segments) + 1):
                paths.add(membership.folder_segments[:depth])
    return tuple(
        sorted(
            paths,
            key=lambda path: tuple(_display_sort_key(segment) for segment in path),
        )
    )


def _collision_after_item_override(
    collision: RootCollisionState | None,
    items: tuple[ImportPreviewItem, ...],
) -> RootCollisionState | None:
    if collision is None:
        return None
    active_roots = {
        membership.folder_segments[0]
        for item in items
        if item.add_membership and item.source.kind is ImportSourceKind.DIRECTORY_MEMBER
        for membership in item.memberships
    }
    if len(active_roots) != 1:
        return None
    active_root = _normalized_folder_name(next(iter(active_roots)))
    collision_root_label = collision.resolved_label or collision.proposed_label
    collision_root = _normalized_folder_name(collision_root_label)
    if active_root.key != collision_root.key:
        return None
    return collision


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
    "analyze_root_collision",
    "apply_item_override",
    "classify_import_batch",
    "confirm_uncertain_match",
    "discover_import_sources",
    "parse_import_sources",
    "resolve_root_collision",
]
