from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Agents.run_context import CurrentRunActor, use_run_actor
from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook


class _Gate:
    def begin_turn(self, _run_id):
        return None

    def resolve(self, _tool):
        return SimpleNamespace(state="allow", risk_floored=False)

    def is_session_approved(self, _name):
        return False

    def stamp(self, *_args):
        raise AssertionError("lesson approval must not stamp the builtin gate")


class _BuiltinProvider:
    def tool_for(self, _name):
        return None


class _LessonService:
    def __init__(self, note_payload=None, *, fail=False):
        self.note_payload = note_payload
        self.fail = fail
        self.calls = []

    def invoke(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if self.fail:
            raise RuntimeError("secret note body must not escape")
        if name == "library_get_note":
            return self.note_payload or {
                "error": {
                    "code": "not_found",
                    "message": "not found",
                    "retryable": False,
                    "details": {},
                }
            }
        return {"ok": True}

    def agent_lesson_preflight_snapshot(self, note_id):
        if self.fail:
            raise RuntimeError("secret note body must not escape")
        if self.note_payload is None:
            return None
        item = self.note_payload["item"]
        public_state = item.get("organization_state")
        receipt_state = (
            "pending_organization" if public_state == "pending" else public_state
        )
        if receipt_state == "ready":
            receipt_state = None
        return {
            "public_note_id": note_id,
            "note_id": note_id,
            "note_version": int(self.note_payload["revision"]),
            "keywords": tuple(
                row["name"] for row in item.get("keyword_metadata", ())
            ),
            "organization_version": item["organization_version"],
            "receipt_state": receipt_state,
            "receipt_note_version": (
                int(self.note_payload["revision"]) if receipt_state else None
            ),
            "receipt_organization_version": (
                item["organization_version"] if receipt_state else None
            ),
        }


def _note_payload(*, keywords=(), state="ready", version=7):
    return {
        "item": {
            "id": "note:bm90ZS0x",
            "title": "Existing",
            "keyword_metadata": [
                {"id": f"keyword:{index}", "name": keyword}
                for index, keyword in enumerate(keywords)
            ],
            "keyword_metadata_total": len(keywords),
            "keyword_metadata_truncated": False,
            "organization_state": state,
            "organization_version": "b" * 64,
        },
        "revision": str(version),
        "text": "not exposed",
    }


def _save_call(call_id="call-1", **overrides):
    arguments = {
        "title": "A reusable fix",
        "content": "# Evidence\n\nNo secrets.",
        "ensure_keywords": ["agent-lesson"],
    }
    arguments.update(overrides)
    return ToolCall("library_save_note", arguments, call_id)


def _hook(provider, request):
    return build_tool_review_hook(
        _Gate(),
        _BuiltinProvider(),
        None,
        request,
        library_provider=provider,
    )


def test_primary_lesson_save_uses_exact_per_call_rows_and_approve_once_only():
    provider = LibraryToolProvider(_LessonService())
    calls = [_save_call("call-a", title="Allowed"), _save_call("call-b", title="Denied")]
    seen = []

    def request(rows):
        seen.extend(rows)
        return {"call-a": "approve_once", "call-b": "deny"}

    with use_run_actor(CurrentRunActor("primary", "run-1", None)):
        verdicts = _hook(provider, request)(calls, "run-1")

    assert [(row.call_id, row.options) for row in seen] == [
        ("call-a", ("approve_once", "deny")),
        ("call-b", ("approve_once", "deny")),
    ]
    assert seen[0].arguments == {
        "operation": "create",
        "title": "Allowed",
        "classification": "requested_marker",
        "call_digest": provider.preflight_agent_lesson_save(
            calls[0].name, calls[0].args, calls[0].call_id
        ).call_digest,
    }
    assert "content" not in seen[0].arguments
    assert verdicts["call-a"] == "proceed"
    assert verdicts["call-b"].startswith("foreground approval denied")
    approved = provider.peek_agent_lesson_approval(
        "run-1", "call-a", seen[0].arguments["call_digest"]
    )
    assert approved is not None
    assert approved.preflight.classification.reason == "requested_marker"
    assert provider.peek_agent_lesson_approval(
        "run-1", "call-b", seen[1].arguments["call_digest"]
    ) is None


def test_rejected_exact_preview_never_dispatches_or_issues_authority():
    service = _LessonService()
    provider = LibraryToolProvider(service)
    seen = []

    with use_run_actor(CurrentRunActor("primary", "run-reject", None)):
        verdicts = _hook(
            provider,
            lambda rows: seen.extend(rows) or {"call-reject": "deny"},
        )([_save_call("call-reject")], "run-reject")

    assert len(seen) == 1
    assert seen[0].options == ("approve_once", "deny")
    assert verdicts["call-reject"].startswith("foreground approval denied")
    assert service.calls == []
    assert provider.agent_lesson_approval_count("run-reject") == 0


def test_lesson_call_id_collision_with_builtin_fails_lesson_closed():
    class AskGate(_Gate):
        def __init__(self):
            self.stamps = []

        def resolve(self, _tool):
            return SimpleNamespace(state="ask", risk_floored=False)

        def stamp(self, run_id, name, decision):
            self.stamps.append((run_id, name, decision))

    class CalculatorProvider(_BuiltinProvider):
        def tool_for(self, name):
            return object() if name == "calculator" else None

    provider = LibraryToolProvider(_LessonService())
    gate = AskGate()
    seen = []
    hook = build_tool_review_hook(
        gate,
        CalculatorProvider(),
        None,
        lambda rows: seen.extend(rows) or {"shared-call": "approve_once"},
        library_provider=provider,
    )

    with use_run_actor(CurrentRunActor("primary", "run-1", None)):
        verdicts = hook(
            [
                _save_call("shared-call"),
                ToolCall("calculator", {"expression": "2 + 2"}, "shared-call"),
            ],
            "run-1",
        )

    assert [(row.server_label, row.call_id) for row in seen] == [
        ("Built-in", "shared-call")
    ]
    assert verdicts["shared-call"] == "approval_required"
    assert provider.agent_lesson_approval_count("run-1") == 0


@pytest.mark.parametrize("kind", ["subagent"])
def test_non_foreground_actor_is_refused_without_card_or_stamp(kind):
    provider = LibraryToolProvider(_LessonService())
    requested = []

    with use_run_actor(CurrentRunActor(kind, "child-1", "run-1")):
        verdicts = _hook(provider, lambda rows: requested.append(rows) or {})(
            [_save_call("call-child")], "child-1"
        )

    assert verdicts == {"call-child": "foreground_required"}
    assert requested == []
    assert provider.agent_lesson_approval_count("child-1") == 0


def test_unbound_classified_save_is_refused_without_card_or_stamp():
    provider = LibraryToolProvider(_LessonService())
    requested = []

    verdicts = _hook(provider, lambda rows: requested.append(rows) or {})(
        [_save_call("call-direct")], "run-1"
    )

    assert verdicts == {"call-direct": "approval_required"}
    assert requested == []
    assert provider.agent_lesson_approval_count("run-1") == 0


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (_note_payload(keywords=("agent-lesson",)), "current_marker"),
        (_note_payload(state="pending"), "pending_organization"),
        (_note_payload(state="placement_review"), "placement_review"),
    ],
)
def test_current_marker_and_actual_receipt_states_force_review(payload, reason):
    provider = LibraryToolProvider(_LessonService(payload))
    call = _save_call(
        "call-update",
        note_id="note:bm90ZS0x",
        expected_version=7,
        expected_organization_version="b" * 64,
        ensure_keywords=[],
    )
    seen = []

    with use_run_actor(CurrentRunActor("primary", "run-1", None)):
        verdicts = _hook(provider, lambda rows: seen.extend(rows) or {"call-update": "approve_once"})(
            [call], "run-1"
        )

    assert verdicts["call-update"] == "proceed"
    assert seen[0].arguments["classification"] == reason


def test_real_pending_receipt_snapshot_keeps_actual_database_state(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Library.library_tool_contract import make_public_id
    from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService

    db = CharactersRAGDB(tmp_path / "notes.db", "lesson-preflight")
    notes = NotesInteropService(
        tmp_path,
        "lesson-preflight",
        global_db_to_use=db,
    )
    notes._db_instances["local_library"] = db
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO notes_organization_sync_checkpoints("
            "server_profile_id, dataset_id, local_state, server_state, "
            "inventory_phase, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "profile-a",
                "dataset-a",
                "initializing",
                "initializing",
                "not_started",
                "2026-08-30T00:00:00Z",
            ),
        )
    saved = notes.save_note_with_organization(
        "local_library",
        title="Pending",
        content="verified evidence",
        ensure_keywords=("agent-lesson",),
        server_profile_id="profile-a",
        dataset_id="dataset-a",
    )
    assert saved["receipt_state"] == "pending_organization"
    public_id = make_public_id("note", saved["id"])
    service = LocalLibraryToolService(notes_service=notes)

    snapshot = service.agent_lesson_preflight_snapshot(public_id)
    preflight = LibraryToolProvider(service).preflight_agent_lesson_save(
        "library_save_note",
        {
            "title": "Pending update",
            "content": "more verified evidence",
            "note_id": public_id,
            "expected_version": saved["version"],
            "expected_organization_version": saved["organization_version"],
        },
        "call-pending-real",
    )

    assert snapshot["receipt_state"] == "pending_organization"
    assert snapshot["receipt_note_version"] == saved["version"]
    assert preflight is not None
    assert preflight.classification.reason == "pending_organization"
    db.close_connection()


def test_case_variant_and_ordinary_note_do_not_enter_lesson_review():
    provider = LibraryToolProvider(_LessonService(_note_payload(keywords=("Agent-Lesson",))))
    call = _save_call(
        "call-ordinary",
        note_id="note:bm90ZS0x",
        expected_version=7,
        expected_organization_version="b" * 64,
        ensure_keywords=["Agent-Lesson"],
    )
    requested = []

    with use_run_actor(CurrentRunActor("primary", "run-1", None)):
        verdicts = _hook(provider, lambda rows: requested.append(rows) or {})(
            [call], "run-1"
        )

    assert verdicts == {}
    assert requested == []


def test_classification_failure_is_content_free_and_fails_closed():
    provider = LibraryToolProvider(_LessonService(fail=True))
    call = _save_call(
        "call-fail",
        note_id="note:bm90ZS0x",
        expected_version=7,
        expected_organization_version="b" * 64,
        ensure_keywords=[],
        content="PRIVATE CONTENT MUST NOT APPEAR",
    )
    requested = []

    with use_run_actor(CurrentRunActor("primary", "run-1", None)):
        verdicts = _hook(provider, lambda rows: requested.append(rows) or {})(
            [call], "run-1"
        )

    assert verdicts == {"call-fail": "approval_required"}
    assert requested == []
    assert "PRIVATE CONTENT" not in repr(verdicts)
    assert provider.agent_lesson_approval_count("run-1") == 0


def test_hook_entry_and_raising_approval_round_clear_this_runs_stamps():
    provider = LibraryToolProvider(_LessonService())
    call = _save_call("call-a")
    with use_run_actor(CurrentRunActor("primary", "run-1", None)):
        first = _hook(provider, lambda _rows: {"call-a": "approve_once"})
        assert first([call], "run-1")["call-a"] == "proceed"
        assert provider.agent_lesson_approval_count("run-1") == 1

        def fail(_rows):
            raise RuntimeError("approval bridge unavailable")

        with pytest.raises(RuntimeError, match="approval bridge unavailable"):
            _hook(provider, fail)([call], "run-1")

    assert provider.agent_lesson_approval_count("run-1") == 0
