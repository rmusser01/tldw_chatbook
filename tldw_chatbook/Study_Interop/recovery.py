"""Semantic study/quiz ownership of one shared ChaChaNotes physical payload."""

from contextlib import closing
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import sqlite3
from threading import Event
from typing import Mapping

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    SchemaPolicy,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.DB.recovery_core import core_adapters


def _core():
    return next(
        adapter
        for adapter in core_adapters()
        if adapter.owner_id == "db.chachanotes.primary"
    )


@dataclass(frozen=True)
class _SharedAdapter:
    owner_id: str
    activation_required: bool = True

    def discover(self, config: Mapping[str, object]) -> tuple[StorageItem, ...]:
        context = discovery_context(config)
        item = _core().discover(config)[0]
        return (
            replace(
                item,
                owner=self.owner_id,
                logical_id=storage_logical_id(context, self.owner_id),
                dependencies=(storage_logical_id(context, "db.chachanotes.primary"),),
            ),
        )

    def schema_policy(self) -> SchemaPolicy:
        return replace(_core().schema_policy(), owner=self.owner_id)

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        if item.owner != self.owner_id:
            raise ValueError("invalid_capture_item")
        # Executors archive one physical shared payload. Delegation always copies
        # the whole DB; semantic ownership never adds a selective second export.
        _core().capture(
            replace(item, owner="db.chachanotes.primary"), destination, cancel
        )
        issues = self.validate(destination)
        if issues:
            raise ValueError(issues[0])

    def validate(self, candidate: Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        issues = _core().validate(candidate)
        if issues:
            return issues
        try:
            with closing(
                connect_private_sqlite(
                    "recovery.domain.study", candidate, read_only=True
                )
            ) as conn:
                conn.execute("PRAGMA trusted_schema=OFF")
                if conn.execute(
                    "SELECT 1 FROM flashcard_assets WHERE typeof(content) != 'blob' OR byte_size != length(content) LIMIT 1"
                ).fetchone():
                    return ("missing_required_asset",)
                queries = (
                    (
                        "SELECT * FROM flashcards",
                        "SELECT * FROM flashcard_templates",
                        "SELECT * FROM decks",
                    )
                    if self.owner_id == "study.local"
                    else (
                        "SELECT * FROM quiz_questions",
                        "SELECT * FROM quiz_attempts",
                    )
                )
                for query in queries:
                    for row in conn.execute(query):
                        for text in row:
                            if not isinstance(text, str):
                                continue
                            identities = re.findall(
                                r"""flashcard-asset://([A-Za-z0-9_-]+)(?=$|[\s)"'])""",
                                text,
                            )
                            if text.count("flashcard-asset://") != len(identities):
                                return ("missing_required_asset",)
                            for identity in identities:
                                if (
                                    conn.execute(
                                        "SELECT 1 FROM flashcard_assets WHERE asset_uuid=?",
                                        (identity,),
                                    ).fetchone()
                                    is None
                                ):
                                    return ("missing_required_asset",)
                if self.owner_id == "study.local":
                    for kind, identity in conn.execute(
                        "SELECT entity_type,entity_id FROM study_sessions"
                    ):
                        query = {
                            "topic": "SELECT 1 FROM topics WHERE id=?",
                            "flashcard_deck": "SELECT 1 FROM decks WHERE id=?",
                            "mindmap": "SELECT 1 FROM mindmaps WHERE id=?",
                        }.get(kind)
                        if query is None:
                            return ("unsupported_domain_reference",)
                        if conn.execute(query, (identity,)).fetchone() is None:
                            return ("invalid_domain_reference",)
                else:
                    for snapshot, answers in conn.execute(
                        "SELECT questions_snapshot,answers FROM quiz_attempts"
                    ):
                        questions = json.loads(snapshot or "[]")
                        responses = json.loads(answers or "[]")
                        if (
                            not isinstance(questions, list)
                            or not isinstance(responses, list)
                            or any(
                                not isinstance(q, dict) for q in questions + responses
                            )
                        ):
                            return ("invalid_domain_reference",)
                        ids = {q.get("id") for q in questions}
                        if len(ids) != len(questions) or any(
                            a.get("question_id") not in ids for a in responses
                        ):
                            return ("invalid_domain_reference",)
                return ()
        except (OSError, ValueError, TypeError, sqlite3.Error):
            return ("domain_validation_unavailable",)

    def relocate(self, candidate: Path, mapping: Mapping[str, Path]) -> None:
        # flashcard-asset:// UUIDs address retained BLOB bytes inside the shared
        # DB; historical question snapshots are identities, never launch inputs.
        issues = self.validate(candidate)
        if issues:
            raise ValueError(issues[0])

    def validate_dependencies(
        self, item: StorageItem, candidate: Path, candidates: Mapping[str, Path]
    ) -> tuple[str, ...]:
        parts = item.logical_id.split(":")
        if (
            item.owner != self.owner_id
            or len(parts) != 3
            or parts[0] != "profile"
            or parts[2] != self.owner_id
        ):
            return ("invalid_dependency_context",)
        key = ":".join(parts[:2]) + ":db.chachanotes.primary"
        if key not in item.dependencies or key not in candidates:
            return ("dependency_unavailable",)
        try:
            if not candidate.samefile(candidates[key]):
                return ("shared_identity_mismatch",)
        except OSError:
            return ("dependency_unavailable",)
        return self.validate(candidate)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_SharedAdapter("study.local"), _SharedAdapter("quiz.local"))
