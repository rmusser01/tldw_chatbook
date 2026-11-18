"""Trace lifecycle writes retain their exact repository transactions across the Windows owner."""

import sys

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.skipif(sys.platform != "win32", reason="Windows profile writer")
@private_profile_test
@pytest.mark.asyncio
async def test_trace_response_start_terminal_error_and_expired_recovery_use_owner(request, monkeypatch):
    from Tests.Chat.test_console_trace_settlement import _call, _request
    from tldw_chatbook.Chat.console_trace_models import TraceCallState
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.Chat.console_trace_settlement import ConsoleTraceSettlementCoordinator
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.config import get_chachanotes_db_path

    db = CharactersRAGDB(get_chachanotes_db_path(), "trace-writer")
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    try:
        _, segment, first_id = _call(db, repository)
        _, _, stale_id = _call(db, repository)
        failure = _request(first_id, outcome=TraceCallState.ERROR, usage={"prompt_tokens": 2})

        def client_write(*args, **kwargs):
            raise AssertionError("client entered the trace write")

        with monkeypatch.context() as patch:
            patch.setattr(ConsoleTraceRepository, "advance_call_state", client_write)
            started = coordinator.mark_response_started(db, call_id=first_id, occurred_at=failure.response_started_at)
            assert started.state is TraceCallState.RESPONSE_STARTED
            assert coordinator.submit(db, failure)
            assert coordinator.settle(db, failure).state is TraceCallState.ERROR
            conflicting = _request(first_id, outcome=TraceCallState.ERROR, usage={"prompt_tokens": 3})
            assert not coordinator.submit(db, conflicting) and coordinator.pending_count == 0
            (recovered,) = coordinator.recover_open_calls(db, occurred_at="2026-08-31T00:00:00Z")
            assert recovered.call_id == stale_id and recovered.state is TraceCallState.DISPATCH_UNKNOWN
        events = repository.read_events(db.get_connection().cursor(), segment)
        assert [event.event_type for event in events] == ["call_outcome", "usage"]

        attempts = []

        def lost_receipt(_database, _profile, operation, _arguments):
            attempts.append(operation)
            raise profile_writer.ProfileWriterOutcomeUnknown("receipt unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(profile_writer, "call_bound_profile_writer", lost_receipt)
            assert not coordinator.submit(db, failure) and coordinator.retry_pending() == 0
        assert attempts == ["chacha.trace.write_prepared"]
    finally:
        db.close_connection()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows profile writer")
@private_profile_test
@pytest.mark.asyncio
async def test_verified_response_is_linked_without_transferring_original_proof_or_redacted_input(request, monkeypatch):
    import json

    from Tests.Chat.test_console_trace_settlement import _call, _request
    from Tests.Chat.test_console_trace_thinking_settlement import PROFILE
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.Chat.console_trace_settlement import ConsoleTraceSettlementCoordinator
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.config import get_chachanotes_db_path

    db = CharactersRAGDB(get_chachanotes_db_path(), "trace-verified-response")
    repo = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repo)
    try:
        chat, _, call = _call(db, repo, response_projection=PROFILE, provider=PROFILE["provider"], model=PROFILE["model"])
        message = db.add_message({"conversation_id": chat, "sender": "assistant", "content": "answer"})
        clean = _request(call, canonical_message_id=message)
        _, _, private_call = _call(db, repo)
        canary = "Bearer sk-proj-" + "x" * 48
        withheld = _request(private_call, response={"role": "assistant", "content": canary})
        transported = []
        send = profile_writer.call_bound_profile_writer

        def capture(source, profile, operation, arguments):
            if operation == "chacha.trace.write_prepared":
                transported.append(arguments)
            return send(source, profile, operation, arguments)

        monkeypatch.setattr(profile_writer, "call_bound_profile_writer", capture)
        assert coordinator.submit(db, clean) and coordinator.submit(db, clean)
        link = repo.get_response_link(db.get_connection().cursor(), call)
        assert link.link_kind == "revision" and link.verification_outcome == "verified_equal"
        assert len(transported) == 2
        assert coordinator.submit(db, withheld)
        assert len(transported) == 2
        wire = json.dumps(transported)
        assert canary not in wire and "_response_equality" not in wire and "_control_owner" not in wire
        withheld_link = repo.get_response_link(db.get_connection().cursor(), private_call)
        assert withheld_link.link_kind == "artifact"
        artifact = repo.get_artifact(db.get_connection().cursor(), withheld_link.artifact_id)
        assert canary.encode() not in artifact.sanitized_bytes
    finally:
        db.close_connection()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows profile writer")
@private_profile_test
@pytest.mark.asyncio
async def test_custom_or_changed_credential_filter_keeps_trace_in_its_original_owner(request, monkeypatch):
    from Tests.Chat.test_console_trace_settlement import _call, _request
    from tldw_chatbook.Chat.console_trace_redaction import CredentialSanitizationResult, CredentialSanitizer
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.Chat.console_trace_settlement import ConsoleTraceSettlementCoordinator
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.profile_writer_trace import settlement_request
    from tldw_chatbook.config import get_chachanotes_db_path

    db = CharactersRAGDB(get_chachanotes_db_path(), "trace-filter-source")
    repo = ConsoleTraceRepository()
    try:
        _, _, custom = _call(db, repo)
        _, _, changed = _call(db, repo)
        coordinator = ConsoleTraceSettlementCoordinator(repo)
        custom_filter = CredentialSanitizer(known_credentials=("private-test-filter",))
        original_owner = ConsoleTraceSettlementCoordinator(repo, sanitizer=custom_filter)

        def unexpected_transfer(*args, **kwargs):
            raise AssertionError("a custom credential policy crossed the writer boundary")

        with monkeypatch.context() as patch:
            patch.setattr(profile_writer, "call_bound_profile_writer", unexpected_transfer)
            assert original_owner.submit(db, _request(custom))
            original_sanitize = CredentialSanitizer.sanitize
            patch.setattr(CredentialSanitizer, "sanitize", lambda self, value: original_sanitize(self, value))
            assert coordinator.submit(db, _request(changed))

        private = _request(changed, response={"role": "assistant", "content": "Bearer private-test-canary"})
        with monkeypatch.context() as patch:
            patch.setattr(CredentialSanitizer, "sanitize", lambda self, value: CredentialSanitizationResult(
                available=True, value=value, omission_reason_code=None, redacted=False,
            ))
            previous = coordinator._prepare(private)
        # Replacing a historical filter with today's default does not release
        # a queued handoff that contains original unfiltered bytes or usage.
        assert previous.response_bytes is not None
        assert b"private-test-canary" in previous.response_bytes
        assert settlement_request(coordinator, previous) is None
    finally:
        db.close_connection()
