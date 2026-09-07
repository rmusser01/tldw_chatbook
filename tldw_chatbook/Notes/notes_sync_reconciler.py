"""Pure reconciliation planning for lasting Database Notes sync."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import StrEnum

from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncAction,
    NotesSyncActionKind,
    NotesSyncDirection,
    NotesSyncSerializationProfile,
    normalize_notes_sync_relative_path,
    validate_notes_sync_digest,
    validate_notes_sync_opaque_id,
    validate_notes_sync_reason_code,
)

RECONCILIATION_PAGE_SIZE = 100
DELETION_GROUP_THRESHOLD = 100


class ReconciliationAttentionKind(StrEnum):
    """Review-required planner result."""

    CONFLICT = "conflict"
    DELETION_REVIEW = "deletion_review"
    PAUSE = "pause"


class ReconciliationSkipKind(StrEnum):
    """Root-level reason automatic planning is unavailable."""

    OFFLINE = "offline"
    UNSAFE_ROOT = "unsafe_root"
    CAPABILITY = "capability"


class ManagedPlacementEffectKind(StrEnum):
    """Managed-placement change classified without performing it."""

    FILE_MOVE = "file_move"
    REPRESENTATION_REFRESH = "representation_refresh"


@dataclass(frozen=True, slots=True, repr=False)
class BindingObservation:
    """Private baseline and current observations for one possible binding."""

    binding_id: str
    baseline_file_digest: str
    baseline_note_digest: str
    baseline_identity_digest: str
    baseline_relative_path: str
    file_digest: str | None
    note_digest: str | None
    file_identity_digest: str | None
    relative_path: str
    note_scope_id: str
    note_id: str
    note_version: int
    note_implied_relative_path: str | None = None
    duplicate_authority: bool = False
    bound: bool = True
    baseline_serialization: NotesSyncSerializationProfile | None = None
    serialization: NotesSyncSerializationProfile | None = None

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.binding_id, field_name="binding_id")
        validate_notes_sync_opaque_id(self.note_scope_id, field_name="note_scope_id")
        validate_notes_sync_opaque_id(self.note_id, field_name="note_id")
        if type(self.note_version) is not int or self.note_version < 0:
            raise ValueError("note_version must be non-negative.")
        for name in (
            "baseline_file_digest",
            "baseline_note_digest",
            "baseline_identity_digest",
        ):
            validate_notes_sync_digest(getattr(self, name), field_name=name)
        for name in (
            "file_digest",
            "note_digest",
            "file_identity_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_notes_sync_digest(value, field_name=name)
        for name in ("baseline_serialization", "serialization"):
            value = getattr(self, name)
            if value is not None and type(value) is not NotesSyncSerializationProfile:
                raise TypeError(f"{name} must be a NotesSyncSerializationProfile.")
        object.__setattr__(
            self,
            "baseline_relative_path",
            normalize_notes_sync_relative_path(self.baseline_relative_path),
        )
        object.__setattr__(
            self,
            "relative_path",
            normalize_notes_sync_relative_path(self.relative_path),
        )
        if self.note_implied_relative_path is not None:
            object.__setattr__(
                self,
                "note_implied_relative_path",
                normalize_notes_sync_relative_path(self.note_implied_relative_path),
            )
        if type(self.duplicate_authority) is not bool or type(self.bound) is not bool:
            raise TypeError("binding flags must be booleans.")

    def __repr__(self) -> str:
        return f"BindingObservation(binding_id={self.binding_id!r}, <private>)"

    def as_unbound(self) -> "BindingObservation":
        """Return the same immutable observations without an existing binding."""

        return replace(self, bound=False)


@dataclass(frozen=True, slots=True, repr=False)
class ReconciliationInput:
    """Complete immutable input to one pure planning pass."""

    root_id: str
    direction: NotesSyncDirection
    bindings: tuple[BindingObservation, ...]
    observation_generation: int
    expected_generation: int
    root_available: bool = True
    root_overlap: bool = False
    write_capable: bool = True
    capability_generation: int = 0

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.root_id, field_name="root_id")
        if type(self.direction) is not NotesSyncDirection:
            raise TypeError("direction must be a NotesSyncDirection.")
        if type(self.bindings) is not tuple or any(
            type(item) is not BindingObservation for item in self.bindings
        ):
            raise TypeError("bindings must be a tuple of BindingObservation values.")
        if len({item.binding_id for item in self.bindings}) != len(self.bindings):
            raise ValueError("binding identifiers must be unique.")
        for name in ("observation_generation", "expected_generation"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if any(
            type(value) is not bool
            for value in (self.root_available, self.root_overlap, self.write_capable)
        ):
            raise TypeError("root capability flags must be booleans.")
        if (
            type(self.capability_generation) is not int
            or self.capability_generation < 0
        ):
            raise ValueError("capability_generation must be non-negative.")

    def __repr__(self) -> str:
        return f"ReconciliationInput(root_id={self.root_id!r}, <private>)"


@dataclass(frozen=True, slots=True)
class ReconciliationAttention:
    kind: ReconciliationAttentionKind
    reason_code: str
    binding_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not ReconciliationAttentionKind:
            raise TypeError("kind must be a ReconciliationAttentionKind.")
        validate_notes_sync_reason_code(self.reason_code)
        if self.binding_id is not None:
            validate_notes_sync_opaque_id(self.binding_id, field_name="binding_id")


@dataclass(frozen=True, slots=True)
class ReconciliationSkip:
    kind: ReconciliationSkipKind
    reason_code: str

    def __post_init__(self) -> None:
        if type(self.kind) is not ReconciliationSkipKind:
            raise TypeError("kind must be a ReconciliationSkipKind.")
        validate_notes_sync_reason_code(self.reason_code)


@dataclass(frozen=True, slots=True)
class ManagedPlacementEffect:
    kind: ManagedPlacementEffectKind
    binding_id: str

    def __post_init__(self) -> None:
        if type(self.kind) is not ManagedPlacementEffectKind:
            raise TypeError("kind must be a ManagedPlacementEffectKind.")
        validate_notes_sync_opaque_id(self.binding_id, field_name="binding_id")


@dataclass(frozen=True, slots=True)
class DeletionGroup:
    items: tuple[ReconciliationAttention, ...]

    def __post_init__(self) -> None:
        if (
            type(self.items) is not tuple
            or not self.items
            or any(
                type(item) is not ReconciliationAttention
                or item.kind is not ReconciliationAttentionKind.DELETION_REVIEW
                or item.binding_id is None
                for item in self.items
            )
        ):
            raise ValueError("items must be deletion-review attention values.")

    @property
    def binding_ids(self) -> tuple[str, ...]:
        return tuple(
            item.binding_id for item in self.items if item.binding_id is not None
        )


@dataclass(frozen=True, slots=True, repr=False)
class ReconciliationPlan:
    root_id: str
    observation_token: str
    safe_actions: tuple[NotesSyncAction, ...]
    attention: tuple[ReconciliationAttention, ...]
    skips: tuple[ReconciliationSkip, ...]
    managed_placement_effects: tuple[ManagedPlacementEffect, ...]
    deletion_groups: tuple[DeletionGroup, ...]
    page_size: int = RECONCILIATION_PAGE_SIZE

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.root_id, field_name="root_id")
        validate_notes_sync_digest(
            self.observation_token,
            field_name="observation_token",
        )
        for name, item_type in (
            ("safe_actions", NotesSyncAction),
            ("attention", ReconciliationAttention),
            ("skips", ReconciliationSkip),
            ("managed_placement_effects", ManagedPlacementEffect),
            ("deletion_groups", DeletionGroup),
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not item_type for value in values
            ):
                raise TypeError(
                    f"{name} must be a tuple of {item_type.__name__} values."
                )
        if type(self.page_size) is not int or self.page_size <= 0:
            raise ValueError("page_size must be positive.")

    def __repr__(self) -> str:
        return (
            "ReconciliationPlan("
            f"safe={len(self.safe_actions)}, attention={len(self.attention)}, "
            f"skips={len(self.skips)}, deletion_groups={len(self.deletion_groups)})"
        )


def _observation_token(request: ReconciliationInput) -> str:
    payload = {
        "root_id": request.root_id,
        "direction": request.direction.value,
        "generation": request.observation_generation,
        "expected_generation": request.expected_generation,
        "root_available": request.root_available,
        "root_overlap": request.root_overlap,
        "write_capable": request.write_capable,
        "capability_generation": request.capability_generation,
        "bindings": [
            {
                "binding_id": binding.binding_id,
                "baseline_file_digest": binding.baseline_file_digest,
                "baseline_note_digest": binding.baseline_note_digest,
                "baseline_identity_digest": binding.baseline_identity_digest,
                "baseline_relative_path": binding.baseline_relative_path,
                "file_digest": binding.file_digest,
                "note_digest": binding.note_digest,
                "file_identity_digest": binding.file_identity_digest,
                "relative_path": binding.relative_path,
                "note_scope_id": binding.note_scope_id,
                "note_id": binding.note_id,
                "note_version": binding.note_version,
                "note_implied_relative_path": binding.note_implied_relative_path,
                "duplicate_authority": binding.duplicate_authority,
                "bound": binding.bound,
                "baseline_serialization": (
                    None
                    if binding.baseline_serialization is None
                    else {
                        "utf8_bom": binding.baseline_serialization.utf8_bom,
                        "newline": binding.baseline_serialization.newline,
                        "final_newline": binding.baseline_serialization.final_newline,
                        "mode": binding.baseline_serialization.mode,
                    }
                ),
                "serialization": (
                    None
                    if binding.serialization is None
                    else {
                        "utf8_bom": binding.serialization.utf8_bom,
                        "newline": binding.serialization.newline,
                        "final_newline": binding.serialization.final_newline,
                        "mode": binding.serialization.mode,
                    }
                ),
            }
            for binding in sorted(request.bindings, key=lambda item: item.binding_id)
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _action(
    root_id: str,
    observation_token: str,
    binding_id: str,
    kind: NotesSyncActionKind,
    reason_code: str | None,
) -> NotesSyncAction:
    identity = "\0".join(
        (root_id, observation_token, binding_id, kind.value, reason_code or "")
    )
    return NotesSyncAction(
        action_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        kind=kind,
        binding_id=binding_id,
        reason_code=reason_code,
    )


def _empty_plan(
    request: ReconciliationInput,
    token: str,
    *,
    skip: ReconciliationSkip | None = None,
    attention: ReconciliationAttention | None = None,
) -> ReconciliationPlan:
    return ReconciliationPlan(
        root_id=request.root_id,
        observation_token=token,
        safe_actions=(),
        attention=(() if attention is None else (attention,)),
        skips=(() if skip is None else (skip,)),
        managed_placement_effects=(),
        deletion_groups=(),
    )


def _plan_unbound(
    request: ReconciliationInput,
    observation_token: str,
    binding: BindingObservation,
) -> tuple[NotesSyncAction | None, ReconciliationAttention | None]:
    file_exists = binding.file_digest is not None
    note_exists = binding.note_digest is not None
    if file_exists and note_exists:
        return None, ReconciliationAttention(
            ReconciliationAttentionKind.CONFLICT,
            "duplicate_authority",
            binding.binding_id,
        )
    if not file_exists and not note_exists:
        return None, None
    if file_exists:
        if request.direction is NotesSyncDirection.NOTES_TO_FOLDER:
            return None, ReconciliationAttention(
                ReconciliationAttentionKind.CONFLICT,
                "out_of_direction_create",
                binding.binding_id,
            )
        return _action(
            request.root_id,
            observation_token,
            binding.binding_id,
            NotesSyncActionKind.CREATE_NOTE,
            "file_discovered",
        ), None
    if request.direction is NotesSyncDirection.FOLDER_TO_NOTES:
        return None, ReconciliationAttention(
            ReconciliationAttentionKind.CONFLICT,
            "out_of_direction_create",
            binding.binding_id,
        )
    return _action(
        request.root_id,
        observation_token,
        binding.binding_id,
        NotesSyncActionKind.CREATE_FILE,
        "note_selected",
    ), None


def _plan_bound(
    request: ReconciliationInput,
    observation_token: str,
    binding: BindingObservation,
) -> tuple[
    NotesSyncAction | None,
    ReconciliationAttention | None,
    ManagedPlacementEffect | None,
]:
    if binding.duplicate_authority:
        return (
            None,
            ReconciliationAttention(
                ReconciliationAttentionKind.PAUSE,
                "duplicate_authority",
                binding.binding_id,
            ),
            None,
        )
    if binding.note_implied_relative_path is not None:
        return (
            None,
            ReconciliationAttention(
                ReconciliationAttentionKind.CONFLICT,
                "note_implied_filesystem_move",
                binding.binding_id,
            ),
            None,
        )

    placement: ManagedPlacementEffect | None = None
    if binding.relative_path != binding.baseline_relative_path:
        if binding.file_identity_digest != binding.baseline_identity_digest:
            return (
                None,
                ReconciliationAttention(
                    ReconciliationAttentionKind.CONFLICT,
                    "ambiguous_identity",
                    binding.binding_id,
                ),
                None,
            )
        if request.direction is NotesSyncDirection.NOTES_TO_FOLDER:
            return (
                None,
                ReconciliationAttention(
                    ReconciliationAttentionKind.CONFLICT,
                    "out_of_direction_move",
                    binding.binding_id,
                ),
                None,
            )
        placement = ManagedPlacementEffect(
            ManagedPlacementEffectKind.FILE_MOVE,
            binding.binding_id,
        )

    if binding.file_digest is None or binding.note_digest is None:
        if binding.file_digest is None and binding.note_digest is None:
            reason = "both_missing"
        elif binding.file_digest is None:
            reason = "file_missing"
        else:
            reason = "note_missing"
        return (
            None,
            ReconciliationAttention(
                ReconciliationAttentionKind.DELETION_REVIEW,
                reason,
                binding.binding_id,
            ),
            placement,
        )

    representation_changed = (
        binding.baseline_serialization is not None
        and binding.serialization is not None
        and binding.baseline_serialization != binding.serialization
    )
    if representation_changed:
        if request.direction is NotesSyncDirection.NOTES_TO_FOLDER:
            return (
                None,
                ReconciliationAttention(
                    ReconciliationAttentionKind.CONFLICT,
                    "out_of_direction_representation",
                    binding.binding_id,
                ),
                placement,
            )
        placement = ManagedPlacementEffect(
            ManagedPlacementEffectKind.REPRESENTATION_REFRESH,
            binding.binding_id,
        )

    file_changed = binding.file_digest != binding.baseline_file_digest
    note_changed = binding.note_digest != binding.baseline_note_digest
    if file_changed and note_changed:
        return (
            None,
            ReconciliationAttention(
                ReconciliationAttentionKind.CONFLICT,
                "both_sides_changed",
                binding.binding_id,
            ),
            placement,
        )
    if file_changed:
        if request.direction is NotesSyncDirection.NOTES_TO_FOLDER:
            return (
                None,
                ReconciliationAttention(
                    ReconciliationAttentionKind.CONFLICT,
                    "out_of_direction_change",
                    binding.binding_id,
                ),
                placement,
            )
        return (
            _action(
                request.root_id,
                observation_token,
                binding.binding_id,
                NotesSyncActionKind.UPDATE_NOTE,
                "file_changed",
            ),
            None,
            placement,
        )
    if note_changed:
        if request.direction is NotesSyncDirection.FOLDER_TO_NOTES:
            return (
                None,
                ReconciliationAttention(
                    ReconciliationAttentionKind.CONFLICT,
                    "out_of_direction_change",
                    binding.binding_id,
                ),
                placement,
            )
        return (
            _action(
                request.root_id,
                observation_token,
                binding.binding_id,
                NotesSyncActionKind.UPDATE_FILE,
                "note_changed",
            ),
            None,
            placement,
        )
    return (
        _action(
            request.root_id,
            observation_token,
            binding.binding_id,
            NotesSyncActionKind.NO_CHANGE,
            None,
        ),
        None,
        placement,
    )


def plan_reconciliation(request: ReconciliationInput) -> ReconciliationPlan:
    """Classify immutable observations without selecting conflict/deletion winners."""

    if type(request) is not ReconciliationInput:
        raise TypeError("request must be a ReconciliationInput.")
    token = _observation_token(request)
    if not request.root_available:
        return _empty_plan(
            request,
            token,
            skip=ReconciliationSkip(ReconciliationSkipKind.OFFLINE, "root_offline"),
        )
    if request.root_overlap:
        return _empty_plan(
            request,
            token,
            skip=ReconciliationSkip(
                ReconciliationSkipKind.UNSAFE_ROOT,
                "root_overlap",
            ),
        )
    if (
        not request.write_capable
        and request.direction is not NotesSyncDirection.FOLDER_TO_NOTES
    ):
        return _empty_plan(
            request,
            token,
            skip=ReconciliationSkip(
                ReconciliationSkipKind.CAPABILITY,
                "capability_loss",
            ),
        )
    if request.observation_generation != request.expected_generation:
        return _empty_plan(
            request,
            token,
            attention=ReconciliationAttention(
                ReconciliationAttentionKind.PAUSE,
                "stale_observation",
            ),
        )

    actions: list[NotesSyncAction] = []
    attention: list[ReconciliationAttention] = []
    effects: list[ManagedPlacementEffect] = []
    for binding in sorted(request.bindings, key=lambda item: item.binding_id):
        if not binding.bound:
            action, issue = _plan_unbound(request, token, binding)
            effect = None
        else:
            action, issue, effect = _plan_bound(request, token, binding)
        if action is not None:
            actions.append(action)
        if issue is not None:
            attention.append(issue)
        if effect is not None:
            effects.append(effect)

    deletions = tuple(
        item
        for item in attention
        if item.kind is ReconciliationAttentionKind.DELETION_REVIEW
    )
    deletion_groups: tuple[DeletionGroup, ...] = ()
    if len(deletions) >= DELETION_GROUP_THRESHOLD:
        deletion_groups = (DeletionGroup(deletions),)
        attention = [
            item
            for item in attention
            if item.kind is not ReconciliationAttentionKind.DELETION_REVIEW
        ]
    return ReconciliationPlan(
        root_id=request.root_id,
        observation_token=token,
        safe_actions=tuple(actions),
        attention=tuple(attention),
        skips=(),
        managed_placement_effects=tuple(effects),
        deletion_groups=deletion_groups,
    )


def assert_review_token(plan: ReconciliationPlan, observed_token: str) -> None:
    """Reject a reviewed plan after its observations have changed."""

    if type(plan) is not ReconciliationPlan:
        raise TypeError("plan must be a ReconciliationPlan.")
    validate_notes_sync_digest(observed_token, field_name="observed_token")
    if plan.observation_token != observed_token:
        raise ValueError("stale_review")


def assert_review_current(
    plan: ReconciliationPlan,
    fresh_observations: ReconciliationInput,
) -> None:
    """Recompute the token from fresh authority observations before apply."""

    if type(fresh_observations) is not ReconciliationInput:
        raise TypeError("fresh_observations must be a ReconciliationInput.")
    assert_review_token(plan, _observation_token(fresh_observations))


__all__ = [
    "BindingObservation",
    "DELETION_GROUP_THRESHOLD",
    "DeletionGroup",
    "ManagedPlacementEffect",
    "ManagedPlacementEffectKind",
    "RECONCILIATION_PAGE_SIZE",
    "ReconciliationAttention",
    "ReconciliationAttentionKind",
    "ReconciliationInput",
    "ReconciliationPlan",
    "ReconciliationSkip",
    "ReconciliationSkipKind",
    "assert_review_token",
    "assert_review_current",
    "plan_reconciliation",
]
