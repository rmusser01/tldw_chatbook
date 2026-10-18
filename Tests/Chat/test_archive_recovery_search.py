"""Real-store archive recovery search contracts."""

import json

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


@pytest.fixture
def stores(tmp_path):
    db = CharactersRAGDB(tmp_path / "chat.sqlite", client_id="recovery")
    workspace_db = WorkspaceDB(tmp_path / "workspace.sqlite")
    registry = LocalWorkspaceRegistryService(workspace_db)
    yield db, ChatConversationService(db), registry
    db.close_connection()
    workspace_db.close()


def search(stores, **kwargs):
    from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveSearchRequest
    from tldw_chatbook.Chat.archive_recovery.search_service import ArchiveSearchService

    return ArchiveSearchService(stores[1], stores[2]).search(
        ArchiveSearchRequest(**kwargs)
    )


def add_chat(db, title="chat", workspace_id=None, archived=True):
    data = {"title": title}
    if workspace_id:
        data.update(scope_type="workspace", workspace_id=workspace_id)
    cid = db.add_conversation(data)
    if archived:
        db.set_conversations_archived([cid], archived=True, expected_versions={cid: 1})
    return cid


def test_workspace_only_archive_is_recoverable(stores):
    db, _, registry = stores
    registry.create_workspace(workspace_id="project", name="Project")
    cid = add_chat(db, workspace_id="project", archived=False)
    db.add_message(
        {"conversation_id": cid, "sender": "user", "content": "visible needle"}
    )
    registry.archive_workspace("project")
    page = search(stores, query="needle")
    assert [item.target.conversation_id for item in page.items] == [cid]
    assert page.items[0].allowed_scopes == ("workspace_only",)
    assert not page.items[0].target.conversation_archived


@pytest.mark.parametrize(
    "chat,workspace,lower,upper,expected",
    [
        (
            "2026-09-21T10:00:00+00:00",
            "2026-09-25T10:00:00+00:00",
            "2026-09-21T00:00:00+00:00",
            "2026-09-22T00:00:00+00:00",
            ("chat",),
        ),
        (
            "2026-09-21T00:00:00+00:00",
            "2026-09-21T01:00:00+00:00",
            "2026-09-21T00:00:00+00:00",
            "2026-09-22T00:00:00+00:00",
            ("chat", "workspace"),
        ),
        (
            "2026-09-22T00:00:00+00:00",
            None,
            "2026-09-21T00:00:00+00:00",
            "2026-09-22T00:00:00+00:00",
            (),
        ),
        (None, None, None, None, ()),
        (
            "2026-09-21T01:00:00+01:00",
            None,
            "2026-09-21T00:00:00+00:00",
            None,
            ("chat",),
        ),
    ],
)
def test_date_window_event_oracle(chat, workspace, lower, upper, expected):
    from tldw_chatbook.Chat.archive_recovery.search_service import (
        matching_archive_events,
    )

    assert (
        matching_archive_events(
            chat_at=chat, workspace_at=workspace, after=lower, before=upper
        )
        == expected
    )


@pytest.mark.parametrize(
    "role,content,extra",
    [
        ("system", "private needle", {}),
        ("tool", "private needle", {}),
        ("assistant", '{"reasoning_content":"private needle"}', {}),
        ("assistant", "<think>private needle</think>", {}),
        (
            "assistant",
            "visible answer",
            {"metadata_json": '{"control":"private needle"}'},
        ),
    ],
)
def test_private_only_needles_never_match(stores, role, content, extra):
    db = stores[0]
    cid = add_chat(db)
    db.add_message(
        {
            "conversation_id": cid,
            "sender": role,
            "role": role,
            "content": content,
            **extra,
        }
    )
    assert search(stores, query="needle").items == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": "x" * 513},
        {"query": 3},
        {"limit": 0},
        {"limit": 21},
        {"limit": True},
        {"limit": "10"},
        {"include_unknown_archive_times": 1},
        {"archived_after": "2026-09-21T00:00:00"},
        {"workspace_id": " "},
        {"cursor": "untrusted"},
        {
            "archived_after": "2026-09-22T00:00:00Z",
            "archived_before": "2026-09-21T00:00:00Z",
        },
    ],
)
def test_request_refuses_invalid_boundaries(kwargs):
    from pydantic import ValidationError

    from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveSearchRequest

    with pytest.raises(ValidationError):
        ArchiveSearchRequest(**kwargs)


def test_mixed_lifecycle_default_missing_workspace_and_deleted(stores):
    db, _, registry = stores
    registry.create_workspace(workspace_id="active", name="Active")
    registry.create_workspace(workspace_id="hidden", name="Hidden")
    registry.archive_workspace("hidden")
    default = add_chat(db)
    missing = add_chat(db, workspace_id="missing")
    hidden = add_chat(db, workspace_id="hidden", archived=False)
    add_chat(db, workspace_id="active", archived=False)
    add_chat(db, workspace_id="missing", archived=False)
    deleted = add_chat(db)
    db.get_connection().execute(
        "UPDATE conversations SET deleted = 1 WHERE id = ?", (deleted,)
    )
    page = search(stores)
    assert {i.target.conversation_id for i in page.items} == {default, missing, hidden}
    found = {i.target.conversation_id: i for i in page.items}
    assert found[default].target.workspace_id is None
    assert found[missing].target.workspace_id == "missing"
    assert found[missing].target.workspace is None
    assert found[missing].allowed_scopes == ()
    assert found[hidden].allowed_scopes == ("workspace_only",)


@pytest.mark.parametrize(
    "chat_at,workspace_at",
    [
        ("2026-09-21T10:00:00Z", "2026-09-25T10:00:00Z"),
        ("2026-09-25T10:00:00Z", "2026-09-21T10:00:00Z"),
        ("2026-09-21T00:00:00Z", "2026-09-21T11:00:00+02:00"),
        ("2026-09-22T00:00:00Z", None),
        (None, None),
    ],
)
def test_sql_event_matrix_matches_pure_oracle(stores, chat_at, workspace_at):
    from tldw_chatbook.Chat.archive_recovery.search_service import (
        matching_archive_events,
    )

    db, _, registry = stores
    registry.create_workspace(workspace_id="dates", name="Dates")
    registry.archive_workspace("dates")
    cid = add_chat(db, workspace_id="dates")
    db.get_connection().execute(
        "UPDATE conversations SET archived_at = ? WHERE id = ?", (chat_at, cid)
    )
    with registry.db.connection() as conn:
        conn.execute(
            "UPDATE workspace_records SET archived_at = ? WHERE workspace_id = ?",
            (workspace_at, "dates"),
        )
    lower, upper = "2026-09-21T00:00:00Z", "2026-09-22T00:00:00Z"
    events = matching_archive_events(
        chat_at=chat_at, workspace_at=workspace_at, after=lower, before=upper
    )
    page = search(stores, archived_after=lower, archived_before=upper)
    assert bool(page.items) == bool(events)
    if events:
        assert page.items[0].match_basis == events
        assert page.items[0].allowed_scopes == ("chat_only", "chat_and_workspace")
    include = search(
        stores,
        archived_after=lower,
        archived_before=upper,
        include_unknown_archive_times=True,
    )
    assert bool(include.items) == bool(
        events or chat_at is None or workspace_at is None
    )
    assert page.unknown_times_present == (chat_at is None or workspace_at is None)


def test_workspace_only_pages_have_stable_identity_ties(stores):
    from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveSearchRequest
    from tldw_chatbook.Chat.archive_recovery.search_service import ArchiveSearchService

    db, chat, registry = stores
    registry.create_workspace(workspace_id="many", name="Many")
    registry.archive_workspace("many")
    ids = [
        add_chat(db, title="duplicate", workspace_id="many", archived=False)
        for _ in range(27)
    ]
    service = ArchiveSearchService(chat, registry)
    request = ArchiveSearchRequest(limit=20)
    page = service.search(request)
    tail = service.search(
        request, after=page.next_anchor, expected_workspace_digest=page.workspace_digest
    )
    assert [i.target.conversation_id for i in page.items + tail.items] == sorted(ids)
    assert tail.next_anchor is None


def test_changed_workspace_digest_requires_refresh(stores):
    from tldw_chatbook.Chat.archive_recovery.contracts import (
        ArchiveRecoveryError,
        ArchiveSearchRequest,
    )
    from tldw_chatbook.Chat.archive_recovery.search_service import ArchiveSearchService

    service = ArchiveSearchService(stores[1], stores[2])
    page = service.search(ArchiveSearchRequest())
    stores[2].create_workspace(workspace_id="new", name="New")
    with pytest.raises(ArchiveRecoveryError) as error:
        service.search(
            ArchiveSearchRequest(), expected_workspace_digest=page.workspace_digest
        )
    assert error.value.code == "stale"
    with pytest.raises(ArchiveRecoveryError) as error:
        service.search(ArchiveSearchRequest(workspace_id="missing"))
    assert error.value.code == "unavailable"


@pytest.mark.parametrize("text", ["café 雪 😀", "OR NEAR NOT *", "literal % _ \\"])
def test_unicode_and_fts_operators_are_literal(stores, text):
    db = stores[0]
    cid = add_chat(db, title="Unrelated")
    db.add_message(
        {
            "conversation_id": cid,
            "sender": "assistant",
            "content": "prefix " + text + " suffix",
        }
    )
    other = add_chat(db, title="OR something NOT related")
    page = search(stores, query=text)
    assert [i.target.conversation_id for i in page.items] == [cid]
    assert text in page.items[0].excerpt
    assert other != cid


def test_thinking_sidecar_and_control_only_text_never_match(stores):
    from tldw_chatbook.Chat.thinking_blocks import (
        DisplayableThinkingBlock,
        ThinkingEnvelope,
        dump_thinking_blocks_json,
    )

    db = stores[0]
    cid = add_chat(db)
    raw = dump_thinking_blocks_json(
        ThinkingEnvelope(
            blocks=(
                DisplayableThinkingBlock(
                    block_id="round-0",
                    round_ordinal=0,
                    provider="llama_cpp",
                    model="qwen3",
                    protocol="openai_chat",
                    source_format="start_anchored_think",
                    status="complete",
                    text="thoughtneedle",
                ),
            )
        )
    )
    db.add_message(
        {
            "conversation_id": cid,
            "sender": "assistant",
            "content": "visible answer",
            "thinking_blocks_json": raw,
        }
    )
    assert search(stores, query="thoughtneedle").items == ()
    assert search(stores, query="visible").items[0].excerpt == "visible answer"


def test_inactive_message_retains_provenance_without_changing_leaf(stores):
    db = stores[0]
    cid = add_chat(db)
    mid = db.add_message(
        {"conversation_id": cid, "sender": "user", "content": "branch needle"}
    )
    leaf = db.add_message(
        {"conversation_id": cid, "sender": "assistant", "content": "other root"}
    )
    db.set_conversation_active_leaf(cid, leaf)
    page = search(stores, query="needle")
    assert page.items[0].match_message_id == mid
    assert "inactive_branch" in page.items[0].match_basis
    assert db.get_conversation_active_leaf(cid) == leaf


def test_unicode_byte_fitting_preserves_continuation(stores):
    from dataclasses import asdict

    from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveSearchRequest
    from tldw_chatbook.Chat.archive_recovery.search_service import (
        MAX_PAGE_BYTES,
        ArchiveSearchService,
    )

    db, chat, registry = stores
    ids = []
    for _ in range(20):
        cid = add_chat(db, title="😀" * 2000)
        db.add_message(
            {"conversation_id": cid, "sender": "user", "content": "雪" * 1000}
        )
        ids.append(cid)
    service = ArchiveSearchService(chat, registry)
    request = ArchiveSearchRequest(limit=20)
    page = service.search(request)
    assert (
        len(
            json.dumps(asdict(page), ensure_ascii=False, separators=(",", ":")).encode()
        )
        <= MAX_PAGE_BYTES
    )
    assert all(len(i.excerpt) <= 320 for i in page.items)
    assert any(i.target.title.endswith("…") for i in page.items)
    seen = [i.target.conversation_id for i in page.items]
    while page.next_anchor:
        page = service.search(
            request,
            after=page.next_anchor,
            expected_workspace_digest=page.workspace_digest,
        )
        seen.extend(i.target.conversation_id for i in page.items)
    assert set(seen) == set(ids)
    assert len(seen) == len(ids)


def test_failed_workspace_enumeration_is_unavailable(stores):
    from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveRecoveryError

    with stores[2].db.connection() as conn:
        conn.execute("ALTER TABLE workspace_records RENAME TO unavailable_records")
    with pytest.raises(ArchiveRecoveryError) as error:
        search(stores)
    assert error.value.code == "unavailable"


def test_deadline_covers_workspace_projection(stores):
    from tldw_chatbook.Chat.archive_recovery.contracts import (
        ArchiveRecoveryError,
        ArchiveSearchRequest,
    )
    from tldw_chatbook.Chat.archive_recovery.search_service import ArchiveSearchService

    values = iter([0, 0, 0, 0, 6])
    clock = lambda: next(values, 6)
    with pytest.raises(ArchiveRecoveryError) as error:
        ArchiveSearchService(stores[1], stores[2], monotonic=clock).search(
            ArchiveSearchRequest()
        )
    assert error.value.code == "search_incomplete"
    with stores[2].db.connection() as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_keywords_discover_legacy_content_without_leaking_body(stores):
    db, chat, _ = stores
    cid = add_chat(db, title="Legacy archive")
    db.add_message(
        {
            "conversation_id": cid,
            "sender": "assistant",
            "content": '{"reasoning":"private bodyneedle"}',
        }
    )
    chat.replace_conversation_keywords(cid, ["keywordneedle"])
    assert search(stores, query="bodyneedle").items == ()
    item = search(stores, query="keywordneedle").items[0]
    assert item.target.conversation_id == cid
    assert "keywords" in item.match_basis
    assert item.excerpt == ""


def test_unknown_candidates_follow_proven_dates_and_workspace_filter(stores):
    db, _, registry = stores
    registry.create_workspace(workspace_id="dates", name="Dates")
    known = add_chat(db, workspace_id="dates")
    unknown = add_chat(db, workspace_id="dates")
    unrelated = add_chat(db)
    db.get_connection().execute(
        "UPDATE conversations SET archived_at = NULL WHERE id = ?", (unknown,)
    )
    page = search(stores, workspace_id="dates")
    assert [i.target.conversation_id for i in page.items] == [known, unknown]
    assert page.items[-1].sort_key[0] == 1
    assert page.items[-1].chronology_incomplete
    assert unrelated not in {i.target.conversation_id for i in page.items}


def test_oversized_minimal_identity_fails_explicitly(stores):
    from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveRecoveryError

    db, _, registry = stores
    registry.create_workspace(workspace_id="oversized", name="Oversized")
    add_chat(db, workspace_id="oversized")
    with registry.db.connection() as conn:
        conn.execute(
            "UPDATE workspace_records SET name = ? WHERE workspace_id = ?",
            ("雪" * 20000, "oversized"),
        )
    with pytest.raises(ArchiveRecoveryError) as error:
        search(stores)
    assert error.value.code == "storage_error"


@pytest.mark.parametrize(
    "prefix",
    [
        "\n",
        "\t",
        "\r\n",
        "\v",
        "\f",
        "\x1c",
        "\x85",
        "\u00a0",
        "\u1680",
        "\u2000",
        "\u200a",
        "\u2028",
        "\u2029",
        "\u202f",
        "\u205f",
        "\u3000",
        "\ufeff",
        " \n\t\u00a0\u3000",
    ],
)
@pytest.mark.parametrize(
    "body", ['{"reasoning_content":"private needle"}', '[{"control":"private needle"}]']
)
def test_whitespace_prefixed_private_serialization_never_matches_or_projects(
    stores, prefix, body
):
    db = stores[0]
    cid = add_chat(db, title="Visible title")
    db.add_message(
        {"conversation_id": cid, "sender": "assistant", "content": prefix + body}
    )
    assert search(stores, query="needle").items == ()
    assert search(stores, query="Visible title").items[0].excerpt == ""
    assert search(stores).items[0].excerpt == ""
