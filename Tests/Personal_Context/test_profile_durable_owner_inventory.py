from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime

import pytest
from tldw_profile_core import ActorType

from tldw_chatbook.Agents.profile_tool_provider import (
    ProfileToolProvider,
    ProfileToolRunScope,
)
from tldw_chatbook.Personal_Context.export_service import ExportRequest
from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
from tldw_chatbook.Personal_Context.proposal_service import ProfileProposalQuota
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.runtime_policy import AgentAuthority
from tldw_chatbook.Personal_Context.service import PersonalContextService


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


class _Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, label: str) -> str:
        self.value += 1
        return f"{label}-{self.value}"


def _proposal_envelope_bytes(
    repository: PersonalContextRepository, proposal_id: str
) -> tuple[bytes, bytes]:
    with closing(repository._connect()) as connection:
        row = connection.execute(
            "SELECT ciphertext, wrapped_dek FROM encrypted_objects "
            "WHERE object_type = 'proposal' AND object_id = ?",
            (proposal_id,),
        ).fetchone()
        assert row is not None
        return bytes(row["ciphertext"]), bytes(row["wrapped_dek"])


def _durable_profile_artifacts(repository: PersonalContextRepository) -> bytes:
    return b"".join(
        path.read_bytes()
        for path in (
            repository.db_path,
            *(
                repository.db_path.with_name(repository.db_path.name + suffix)
                for suffix in ("-journal", "-wal", "-shm")
            ),
        )
        if path.exists()
    )


@pytest.mark.parametrize("resolution", ["reject", "supersede", "expire"])
def test_unaccepted_terminal_proposal_content_is_absent_from_every_decoded_owner(
    tmp_path, resolution: str
) -> None:
    canary = "REJECTED-PROPOSAL-CANARY-41a927"
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db",
        key_protector=InMemoryProfileKeyProtector(),
    )
    with sqlite3.connect(repository.db_path) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
    service = PersonalContextService(repository, clock=lambda: NOW, id_factory=_Ids())
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    service.set_runtime_enabled(True)
    service.set_scope_authority(scope.scope_id, AgentAuthority.PROPOSE)
    view = service.authorized_context_view()
    provider = ProfileToolProvider(
        service,
        run_scope=ProfileToolRunScope(
            run_id="turn-1",
            session_id="session-1",
            profile_id=manifest.profile_id,
            scope_id=scope.scope_id,
            authority=AgentAuthority.PROPOSE,
            generation=view.generation,
            authority_revision=view.authority_revision,
            current_user_message_id="message-1",
            current_user_text=f"Remember {canary}.",
        ),
        quota=ProfileProposalQuota(),
    )

    tool_result = provider.invoke(
        "profile_propose",
        {
            "operation": "create",
            "evidence_span": f"Remember {canary}.",
            "proposed_payload": {
                "kind": "preference",
                "subject": "privacy.inventory",
                "polarity": "like",
                "value": canary,
            },
        },
    )
    proposals = service.proposal_service()
    pending = proposals.list_pending()
    assert len(pending) == 1
    old_ciphertext, old_wrapped_dek = _proposal_envelope_bytes(
        repository, pending[0].proposal_id
    )
    getattr(proposals, resolution)(pending[0].proposal_id)

    export_path = tmp_path / "profile.json"
    service.export_plaintext(
        ExportRequest(destination=export_path, confirm_plaintext=True)
    )
    manifest_snapshot, scopes, records, proposal_receipts = (
        service.snapshot_for_export()
    )
    decoded_owners = {
        "tool_result": tool_result.content,
        "canonical_export": json.dumps(
            {
                "manifest": manifest_snapshot.model_dump(mode="json"),
                "scopes": [scope.model_dump(mode="json") for scope in scopes],
                "records": [record.model_dump(mode="json") for record in records],
                "proposals": [
                    proposal.model_dump(mode="json") for proposal in proposal_receipts
                ],
            },
            sort_keys=True,
        ),
        "plaintext_export": export_path.read_text(encoding="utf-8"),
    }

    for owner, content in decoded_owners.items():
        assert canary not in content, owner
    receipt = repository.get_proposal(pending[0].proposal_id)
    assert receipt is not None
    assert receipt.proposed_record is None
    assert receipt.confidence is None
    assert repository.list_records() == []
    assert all(
        canary.encode() not in path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    )
    durable_bytes = _durable_profile_artifacts(repository)
    assert old_ciphertext not in durable_bytes
    assert old_wrapped_dek not in durable_bytes


def test_accepted_proposal_content_survives_only_as_the_user_approved_record(
    tmp_path,
) -> None:
    canary = "ACCEPTED-PROPOSAL-CANARY-7d196e"
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db",
        key_protector=InMemoryProfileKeyProtector(),
    )
    with sqlite3.connect(repository.db_path) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
    service = PersonalContextService(repository, clock=lambda: NOW, id_factory=_Ids())
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    service.set_runtime_enabled(True)
    service.set_scope_authority(scope.scope_id, AgentAuthority.PROPOSE)
    view = service.authorized_context_view()
    provider = ProfileToolProvider(
        service,
        run_scope=ProfileToolRunScope(
            run_id="turn-1",
            session_id="session-1",
            profile_id=manifest.profile_id,
            scope_id=scope.scope_id,
            authority=AgentAuthority.PROPOSE,
            generation=view.generation,
            authority_revision=view.authority_revision,
            current_user_message_id="message-1",
            current_user_text=f"Remember {canary}.",
        ),
        quota=ProfileProposalQuota(),
    )
    provider.invoke(
        "profile_propose",
        {
            "operation": "create",
            "evidence_span": f"Remember {canary}.",
            "proposed_payload": {
                "kind": "preference",
                "subject": "privacy.accepted",
                "polarity": "like",
                "value": canary,
            },
        },
    )
    proposals = service.proposal_service()
    pending = proposals.list_pending()[0]
    old_ciphertext, old_wrapped_dek = _proposal_envelope_bytes(
        repository, pending.proposal_id
    )

    record = proposals.accept(pending.proposal_id, user_actor=ActorType.USER)

    receipt = repository.get_proposal(pending.proposal_id)
    assert receipt is not None and receipt.proposed_record is None
    assert canary not in receipt.model_dump_json()
    assert record.payload is not None
    assert canary in record.payload.model_dump_json()
    assert canary.encode() not in repository.db_path.read_bytes()
    durable_bytes = _durable_profile_artifacts(repository)
    assert old_ciphertext not in durable_bytes
    assert old_wrapped_dek not in durable_bytes


def test_personal_context_connections_require_secure_delete_without_disabling_wal(
    tmp_path,
) -> None:
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db",
        key_protector=InMemoryProfileKeyProtector(),
    )
    with sqlite3.connect(repository.db_path) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)

    with closing(repository._connect()) as connection:
        assert connection.execute("PRAGMA secure_delete").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
