"""Temporary memory and durable SQLite rules share one scoped service."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from threading import RLock

from tldw_chatbook.DB.transaction_observer import (
    current_managed_transaction,
    register_transaction_completion,
)

from .models import (
    MAX_EFFECTIVE_RULES,
    RuleAssessment,
    RuleBinding,
    RuleLearningResult,
    RuleRevision,
    RuleScope,
    RuleSource,
    RuleValidation,
    canonical_json,
)
from .repository import (
    ResponseRuleRepository,
    RuleBindingConflict,
    body_free_validation,
    validate_activation,
)
from .resolution import resolve_effective_rules


@dataclass
class _TemporaryRules:
    bindings: dict[str, RuleBinding] = field(default_factory=dict, repr=False)
    revisions: dict[tuple[str, int], RuleRevision] = field(
        default_factory=dict, repr=False
    )
    validations: dict[tuple[str, int], RuleValidation] = field(
        default_factory=dict, repr=False
    )
    drafts: list[tuple[RuleSource, RuleLearningResult, str]] = field(
        default_factory=list, repr=False
    )
    assessments: list[RuleAssessment] = field(default_factory=list, repr=False)
    unavailable_sources: set[str] = field(default_factory=set, repr=False)


class ResponseRuleStore:
    """Own local bindings; temporary publication happens only after commit."""

    def __init__(self, repository: ResponseRuleRepository) -> None:
        self.repository = repository
        self._temporary: dict[str, _TemporaryRules] = {}
        self._adopting: set[str] = set()
        self._listeners: list[Callable[[RuleScope, str], None]] = []
        self._lock = RLock()
        self._contexts: dict[str, tuple[RuleScope, RuleScope | None, RuleScope]] = {}

    def register_context(
        self, chat: RuleScope, workspace: RuleScope | None, global_scope: RuleScope
    ) -> None:
        """Retain body-free live scope combinations for precommit capacity checks."""
        with self._lock:
            self._contexts[chat.owner_id] = (chat, workspace, global_scope)

    def _assert_capacity(
        self, proposed: RuleBinding, global_scope: RuleScope, cursor: sqlite3.Cursor
    ) -> None:
        """Resolve known target Chats under the same serialized write boundary."""
        contexts = dict(self._contexts)
        for row in cursor.execute(
            "SELECT id,workspace_id FROM conversations WHERE deleted=0"
        ).fetchall():
            contexts.setdefault(
                row[0],
                (
                    RuleScope("chat", row[0]),
                    RuleScope("workspace", row[1]) if row[1] else None,
                    global_scope,
                ),
            )
        # A scope with no open/saved Chats still cannot grow beyond its own limit.
        contexts.setdefault(
            proposed.scope.owner_id,
            (
                (
                    proposed.scope
                    if proposed.scope.kind == "chat"
                    else RuleScope("chat", "capacity-preview")
                ),
                proposed.scope if proposed.scope.kind == "workspace" else None,
                global_scope,
            ),
        )
        bindings = [
            RuleBinding(RuleScope(r[0], r[1]), r[2], r[3], r[4], r[5])
            for r in cursor.execute(
                "SELECT scope_kind,scope_id,rule_id,revision,state,binding_revision FROM console_response_rule_bindings"
            ).fetchall()
        ]
        bindings.extend(
            b for state in self._temporary.values() for b in state.bindings.values()
        )
        bindings = [
            b
            for b in bindings
            if (b.scope, b.rule_id) != (proposed.scope, proposed.rule_id)
        ] + [proposed]
        for scopes in contexts.values():
            if proposed.scope not in scopes:
                continue
            selected: dict[str, RuleBinding] = {}
            for scope in scopes:
                for binding in bindings:
                    if binding.scope == scope:
                        selected.setdefault(binding.rule_id, binding)
            if (
                sum(b.state == "enabled" for b in selected.values())
                > MAX_EFFECTIVE_RULES
            ):
                raise ValueError("too_many_effective_rules")

    def register_temporary(self, session_id: str) -> None:
        """Declare a live Chat's memory ownership before any binding mutation."""
        with self._lock:
            self._temporary.setdefault(session_id, _TemporaryRules())

    def discard_temporary(self, session_id: str) -> None:
        """Release a closed temporary Chat without writing private history."""
        with self._lock:
            if session_id in self._adopting:
                raise RuntimeError("rule_adoption_pending")
            scope = RuleScope("chat", session_id)
            state = self._temporary.pop(session_id, None)
            self._contexts.pop(session_id, None)
            if state:
                for rule_id in state.bindings:
                    self._invalidate(scope, rule_id)

    def add_invalidation_listener(
        self, listener: Callable[[RuleScope, str], None]
    ) -> Callable[[], None]:
        """Fence pending work before a destructive binding write, even on failure."""
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _invalidate(self, scope: RuleScope, rule_id: str) -> None:
        for listener in tuple(self._listeners):
            listener(scope, rule_id)

    def _memory(self, scope: RuleScope) -> _TemporaryRules | None:
        return self._temporary.get(scope.owner_id) if scope.kind == "chat" else None

    def _assert_not_adopting(self, scope: RuleScope) -> None:
        if scope.kind == "chat" and scope.owner_id in self._adopting:
            raise RuntimeError("rule_adoption_pending")

    def list_bindings(self, scope: RuleScope) -> tuple[RuleBinding, ...]:
        """Read only bindings owned by the requested scope."""
        with self._lock:
            memory = self._memory(scope)
            if memory is not None:
                return tuple(memory.bindings[key] for key in sorted(memory.bindings))
            return self.repository.list_bindings(scope)

    def get_revision(self, rule_id: str, revision: int) -> RuleRevision:
        """Read an exact immutable definition; never silently select the latest."""
        with self._lock:
            for memory in self._temporary.values():
                if rule := memory.revisions.get((rule_id, revision)):
                    return rule
            return self.repository.get_revision(rule_id, revision)

    def get_validation(self, rule_id: str, revision: int) -> RuleValidation | None:
        """Read historical calibration; it never grants live execution authority."""
        with self._lock:
            for memory in self._temporary.values():
                if validation := memory.validations.get((rule_id, revision)):
                    return validation
            return self.repository.get_validation(rule_id, revision)

    def next_revision(self, rule_id: str) -> int:
        """Allocate above every retained immutable draft and binding revision."""
        with self._lock:
            revisions = [
                revision
                for state in self._temporary.values()
                for identity, revision in state.revisions
                if identity == rule_id
            ]
            with self.repository.db.transaction() as cursor:
                row = cursor.execute(
                    "SELECT MAX(revision) FROM console_response_rule_revisions WHERE rule_id=?",
                    (rule_id,),
                ).fetchone()
            return max([0, row[0] or 0, *revisions]) + 1

    def effective_rules(
        self, chat: RuleScope, workspace: RuleScope | None, global_scope: RuleScope
    ) -> tuple[RuleRevision, ...]:
        """Resolve exact revisions with Chat exclusions and scope precedence."""
        with self._lock:
            self.register_context(chat, workspace, global_scope)
            bindings = tuple(
                b
                for scope in (chat, workspace, global_scope)
                if scope is not None
                for b in self.list_bindings(scope)
            )
            revisions = []
            for binding in bindings:
                if binding.revision is not None:
                    try:
                        revisions.append(
                            self.get_revision(binding.rule_id, binding.revision)
                        )
                    except KeyError:
                        pass
            return resolve_effective_rules(
                bindings,
                tuple(revisions),
                chat=chat,
                workspace=workspace,
                global_scope=global_scope,
            )

    def activate(
        self,
        rule: RuleRevision,
        validation: RuleValidation,
        scope: RuleScope,
        *,
        expected_binding_revision: int,
        current: Callable[[], bool] | None = None,
        admission_lock=None,
    ) -> RuleBinding:
        """Publish a calibrated revision only after one successful local commit."""
        validate_activation(rule, validation)
        with self._lock, admission_lock or nullcontext():
            if current is not None and not current():
                raise RuleBindingConflict("rule_source_changed")
            self._assert_not_adopting(scope)
            if (
                scope.kind == "chat"
                and scope.owner_id == rule.origin.session_id
                and rule.origin.conversation_id is None
            ):
                self.register_temporary(scope.owner_id)
            memory = self._memory(scope)
            if memory is not None:
                with self.repository.db.transaction(immediate=True) as cursor:
                    self._assert_capacity(
                        RuleBinding(scope, rule.rule_id, rule.revision, "enabled", 1),
                        RuleScope("global", rule.origin.profile_id),
                        cursor,
                    )
                previous = memory.bindings.get(rule.rule_id)
                if (
                    previous.binding_revision if previous else 0
                ) != expected_binding_revision:
                    raise RuleBindingConflict("rule_binding_changed")
                key = (rule.rule_id, rule.revision)
                if key in memory.revisions and memory.revisions[key] != rule:
                    raise ValueError("immutable_rule_revision")
                binding = RuleBinding(
                    scope,
                    rule.rule_id,
                    rule.revision,
                    "enabled",
                    expected_binding_revision + 1,
                )
                memory.revisions[key] = rule
                memory.validations[key] = validation
                memory.bindings[rule.rule_id] = binding
            else:
                with self.repository.db.transaction(immediate=True) as cursor:
                    self._assert_capacity(
                        RuleBinding(scope, rule.rule_id, rule.revision, "enabled", 1),
                        RuleScope("global", rule.origin.profile_id),
                        cursor,
                    )
                    self.repository._assert_source(cursor, rule.origin)
                    self.repository._put_revision(cursor, rule)
                    self.repository._put_validation(cursor, rule, validation)
                    binding = self.repository._set_binding(
                        cursor,
                        RuleBinding(scope, rule.rule_id, rule.revision, "enabled", 1),
                        expected_binding_revision,
                    )
            self._invalidate(scope, rule.rule_id)
            return binding

    def set_binding(
        self,
        binding: RuleBinding,
        *,
        expected_binding_revision: int,
        current: Callable[[], bool] | None = None,
    ) -> RuleBinding:
        """Change a reviewed pin/state using compare-and-swap."""
        with self._lock:
            if current is not None and not current():
                raise RuleBindingConflict("rule_scope_changed")
            self._assert_not_adopting(binding.scope)
            self._invalidate(binding.scope, binding.rule_id)
            if binding.state == "enabled" and binding.revision is not None:
                rule = self.get_revision(binding.rule_id, binding.revision)
                validation = self.get_validation(binding.rule_id, binding.revision)
                if validation is None:
                    raise ValueError("validation_required")
                validate_activation(rule, validation)
            memory = self._memory(binding.scope)
            if memory is not None:
                if binding.state == "enabled":
                    with self.repository.db.transaction(immediate=True) as cursor:
                        self._assert_capacity(
                            binding, RuleScope("global", rule.origin.profile_id), cursor
                        )
                previous = memory.bindings.get(binding.rule_id)
                if (
                    previous.binding_revision if previous else 0
                ) != expected_binding_revision:
                    raise RuleBindingConflict("rule_binding_changed")
                if binding.revision is not None:
                    self.get_revision(binding.rule_id, binding.revision)
                updated = replace(
                    binding, binding_revision=expected_binding_revision + 1
                )
                memory.bindings[binding.rule_id] = updated
                return updated
            with self.repository.db.transaction(immediate=True) as cursor:
                if binding.state == "enabled":
                    self._assert_capacity(
                        binding, RuleScope("global", rule.origin.profile_id), cursor
                    )
                return self.repository._set_binding(
                    cursor, binding, expected_binding_revision
                )

    def promote(
        self,
        rule_id: str,
        revision: int,
        destination: RuleScope,
        *,
        expected_binding_revision: int,
        current: Callable[[], bool] | None = None,
    ) -> RuleBinding:
        """Pin the reviewed definition more broadly without copying fixtures."""
        if destination.kind == "chat":
            raise ValueError("promotion_scope_invalid")
        with self._lock:
            if current is not None and not current():
                raise RuleBindingConflict("rule_scope_changed")
            rule = self.get_revision(rule_id, revision)
            validation = self.get_validation(rule_id, revision)
            if validation is None:
                raise ValueError("validation_required")
            validate_activation(rule, validation)
            with self.repository.db.transaction(immediate=True) as cursor:
                self._assert_capacity(
                    RuleBinding(destination, rule_id, revision, "enabled", 1),
                    (
                        destination
                        if destination.kind == "global"
                        else RuleScope("global", rule.origin.profile_id)
                    ),
                    cursor,
                )
                self.repository._put_revision(cursor, rule)
                self.repository._retain_validation_provenance(cursor, rule, validation)
                binding = self.repository._set_binding(
                    cursor,
                    RuleBinding(destination, rule_id, revision, "enabled", 1),
                    expected_binding_revision,
                )
            self._invalidate(destination, rule_id)
            return binding

    def save_draft(
        self,
        scope: RuleScope,
        source: RuleSource,
        result: RuleLearningResult,
        *,
        complaint: str,
        current: Callable[[], bool] | None = None,
        admission_lock=None,
    ) -> None:
        """Keep failed or tested inactive drafts in their explicit source scope."""
        if len(complaint.encode("utf-8")) > 8192:
            raise ValueError("complaint_too_large")
        with self._lock, admission_lock or nullcontext():
            if current is not None and not current():
                raise RuleBindingConflict("rule_source_changed")
            self._assert_not_adopting(scope)
            if (
                scope.kind == "chat"
                and scope.owner_id == source.session_id
                and source.conversation_id is None
            ):
                self.register_temporary(scope.owner_id)
            memory = self._memory(scope)
            if memory is not None:
                if result.rule:
                    key = (result.rule.rule_id, result.rule.revision)
                    if key in memory.revisions and memory.revisions[key] != result.rule:
                        raise ValueError("immutable_rule_revision")
                    memory.revisions[key] = result.rule
                    if result.validation:
                        memory.validations[key] = result.validation
                memory.drafts.append(
                    (
                        source,
                        (
                            replace(result, state="inactive")
                            if result.state == "active"
                            else result
                        ),
                        complaint,
                    )
                )
            else:
                with self.repository.db.transaction(immediate=True) as cursor:
                    if current is not None:
                        self.repository._assert_source(cursor, source)
                    self.repository._save_draft(
                        cursor, scope, source, result, complaint
                    )

    def list_drafts(self, scope: RuleScope) -> tuple[RuleLearningResult, ...]:
        """Inspect drafts without crossing the requested scope."""
        with self._lock:
            memory = self._memory(scope)
            return (
                tuple(r for _, r, _ in memory.drafts)
                if memory is not None
                else self.repository.list_drafts(scope)
            )

    def delete_binding(
        self,
        scope: RuleScope,
        rule_id: str,
        *,
        expected_binding_revision: int,
        current: Callable[[], bool] | None = None,
    ) -> None:
        """Remove one binding; promoted or independently drafted revisions survive."""
        with self._lock:
            if current is not None and not current():
                raise RuleBindingConflict("rule_scope_changed")
            self._assert_not_adopting(scope)
            self._invalidate(scope, rule_id)
            memory = self._memory(scope)
            if memory is not None:
                old = memory.bindings.get(rule_id)
                if old is None or old.binding_revision != expected_binding_revision:
                    raise RuleBindingConflict("rule_binding_changed")
                del memory.bindings[rule_id]
                return
            with self.repository.db.transaction(immediate=True) as cursor:
                removed = cursor.execute(
                    "DELETE FROM console_response_rule_bindings WHERE scope_kind=? AND scope_id=? AND rule_id=? AND binding_revision=?",
                    (scope.kind, scope.owner_id, rule_id, expected_binding_revision),
                )
                if removed.rowcount != 1:
                    raise RuleBindingConflict("rule_binding_changed")
                self.repository._garbage_collect(cursor)

    def save_assessment(
        self,
        assessment: RuleAssessment,
        *,
        current: Callable[[], bool] | None = None,
        admission_lock=None,
    ) -> None:
        """Retain historical outcomes separately from admission authority."""
        with self._lock, admission_lock or nullcontext():
            if current is not None and not current():
                raise RuleBindingConflict("rule_source_changed")
            if assessment.source.conversation_id is None:
                self.register_temporary(assessment.source.session_id)
                self._temporary[assessment.source.session_id].assessments.append(
                    assessment
                )
                return
            with self.repository.db.transaction() as cursor:
                if current is not None:
                    self.repository._assert_source(cursor, assessment.source)
                cursor.execute(
                    "INSERT INTO console_response_rule_assessments VALUES (?,?,?,?,?)",
                    (
                        assessment.assessment_id,
                        assessment.source.conversation_id,
                        assessment.source.message_id,
                        assessment.state,
                        canonical_json(asdict(assessment)),
                    ),
                )

    def adopt_temporary(
        self,
        session_id: str,
        conversation_id: str,
        message_ids: Mapping[str, str],
        cursor: sqlite3.Cursor,
    ) -> None:
        """Write staged ownership on the caller's cursor; clear memory postcommit."""
        with self._lock:
            memory = self._temporary.get(session_id)
            if memory is None:
                return
            if session_id in self._adopting:
                raise RuntimeError("rule_adoption_pending")
            token = current_managed_transaction(cursor.connection)
            if token is None:
                raise RuntimeError("managed_transaction_required")
            self._adopting.add(session_id)

            def settled(committed: bool | None) -> None:
                with self._lock:
                    self._adopting.discard(session_id)
                    if committed is True:
                        self._temporary.pop(session_id, None)

            register_transaction_completion(cursor.connection, token, settled)

            def remap(origin: RuleSource) -> RuleSource:
                if origin.message_id in memory.unavailable_sources:
                    return origin
                if origin.message_id not in message_ids:
                    raise ValueError("rule_source_not_adopted")
                durable_id = message_ids[origin.message_id]
                row = cursor.execute(
                    "SELECT version FROM messages WHERE id=? AND conversation_id=? AND sender='assistant' AND deleted=0",
                    (durable_id, conversation_id),
                ).fetchone()
                if row is None:
                    raise ValueError("rule_source_not_adopted")
                return replace(
                    origin,
                    conversation_id=conversation_id,
                    message_id=durable_id,
                    message_version=row[0],
                    branch_id=message_ids.get(origin.branch_id, origin.branch_id),
                )

            adopted = {}
            for key, rule in memory.revisions.items():
                updated = replace(rule, origin=remap(rule.origin))
                existing = cursor.execute(
                    "SELECT origin_json FROM console_response_rule_revisions WHERE rule_id=? AND revision=?",
                    key,
                ).fetchone()
                if existing and json_source_conversation(existing[0]) is None:
                    cursor.execute(
                        "UPDATE console_response_rule_revisions SET origin_json=? WHERE rule_id=? AND revision=?",
                        (canonical_json(asdict(updated.origin)), *key),
                    )
                self.repository._put_revision(cursor, updated)
                adopted[key] = updated
            for key, validation in memory.validations.items():
                writer = (
                    self.repository._retain_validation_provenance
                    if validation.source.message_id in memory.unavailable_sources
                    else self.repository._put_validation
                )
                writer(
                    cursor,
                    adopted[key],
                    replace(validation, source=remap(validation.source)),
                )
            scope = RuleScope("chat", conversation_id)
            for binding in memory.bindings.values():
                self.repository._set_binding(cursor, replace(binding, scope=scope), 0)
            for origin, result, complaint in memory.drafts:
                updated_result = replace(
                    result,
                    rule=(
                        adopted.get((result.rule.rule_id, result.rule.revision))
                        if result.rule
                        else None
                    ),
                    validation=(
                        replace(
                            result.validation, source=remap(result.validation.source)
                        )
                        if result.validation
                        else None
                    ),
                )

                self.repository._save_draft(
                    cursor, scope, remap(origin), updated_result, complaint
                )
            for assessment in memory.assessments:
                updated_assessment = replace(
                    assessment, source=remap(assessment.source)
                )
                cursor.execute(
                    "INSERT INTO console_response_rule_assessments VALUES (?,?,?,?,?)",
                    (
                        assessment.assessment_id,
                        conversation_id,
                        updated_assessment.source.message_id,
                        assessment.state,
                        canonical_json(asdict(updated_assessment)),
                    ),
                )

    def remove_temporary_sources(self, session_id: str, message_ids: set[str]) -> None:
        """Remove deleted private examples while retaining independent rule pins."""
        with self._lock:
            self._assert_not_adopting(RuleScope("chat", session_id))
            memory = self._temporary.get(session_id)
            if memory is None:
                return
            memory.unavailable_sources.update(message_ids)
            memory.drafts = [
                (s, r, c)
                for s, r, c in memory.drafts
                if s.message_id not in message_ids
            ]
            memory.assessments = [
                a for a in memory.assessments if a.source.message_id not in message_ids
            ]
            for key, validation in tuple(memory.validations.items()):
                if validation.source.message_id in message_ids:
                    memory.validations[key] = body_free_validation(validation)

    def remove_source(
        self,
        conversation_id: str,
        message_id: str | None,
        *,
        permanent: bool,
        cursor: sqlite3.Cursor,
    ) -> None:
        """Delete source-owned private evidence; preserve independent definitions."""
        with self._lock:
            for binding in self.repository.list_bindings(
                RuleScope("chat", conversation_id)
            ):
                self._invalidate(binding.scope, binding.rule_id)
            self.repository.remove_source(
                conversation_id, message_id, permanent=permanent, cursor=cursor
            )


def json_source_conversation(encoded: str) -> str | None:
    import json

    return json.loads(encoded)["conversation_id"]
