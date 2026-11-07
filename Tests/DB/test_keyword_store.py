"""Direct pins for the shared KeywordStore spec seam (task-34903).

The two host suites cover the store's behavior end-to-end; these tests pin
the SEAM itself -- the per-host parameterization that preserves each host's
exact exception construction, normalization, and unlink scope -- so a future
spec edit cannot silently homogenize the two hosts (review round 1
recommendation).
"""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import (
    ConflictError as MediaConflictError,
    DatabaseError as MediaDatabaseError,
    InputError as MediaInputError,
    MediaDatabase,
)
from tldw_chatbook.DB.keyword_store import KeywordStore, KeywordStoreSpec
from tldw_chatbook.DB.Prompts_DB import (
    ConflictError as PromptsConflictError,
    DatabaseError as PromptsDatabaseError,
    InputError as PromptsInputError,
    PromptsDatabase,
)


def _media_store() -> KeywordStore:
    db = MediaDatabase(":memory:", "seam-client")
    return db._keywords


def _prompts_store() -> KeywordStore:
    db = PromptsDatabase(":memory:", "seam-client")
    return db._keywords


def test_media_spec_construction_is_exact():
    store = _media_store()
    spec = store.spec
    assert spec.keywords_table == "Keywords"
    assert spec.fts_table == "keyword_fts"
    assert spec.link_table == "MediaKeywords"
    assert spec.entity_fk == "media_id"
    assert spec.sync_entity == "Keywords"
    assert spec.link_sync_entity == "MediaKeywords"
    assert spec.entity_uuid_key == "media_uuid"
    assert spec.unlink_all_links is False
    # canonicalize reproduces the original inline normalization
    assert spec.canonicalize_keywords([" B ", "", "a", "A", "b"]) == ["a", "b"]
    assert spec.normalize_keyword("  MiXeD  ") == "mixed"


def test_prompts_spec_construction_is_exact():
    store = _prompts_store()
    spec = store.spec
    assert spec.keywords_table == "PromptKeywordsTable"
    assert spec.fts_table == "prompt_keywords_fts"
    assert spec.link_table == "PromptKeywordLinks"
    assert spec.entity_fk == "prompt_id"
    assert spec.sync_entity == "PromptKeywordsTable"
    assert spec.link_sync_entity == "PromptKeywordLinks"
    assert spec.entity_uuid_key == "prompt_uuid"
    assert spec.unlink_all_links is True
    # Prompts normalization collapses internal whitespace runs
    assert spec.normalize_keyword("  Two   Spaces ") == "two spaces"


def test_media_conflict_factory_preserves_the_two_positional_construction():
    err = _media_store().spec.make_conflict("undelete", 42)
    assert isinstance(err, MediaConflictError)
    # The original call was ConflictError("Keywords", kw_id): message first.
    assert err.args[0] == "Keywords"
    assert err.entity == 42


def test_prompts_conflict_factory_preserves_the_three_arg_construction():
    spec = _prompts_store().spec
    undelete = spec.make_conflict("undelete", 7)
    soft = spec.make_conflict("soft_delete", 8)
    for err in (undelete, soft):
        assert isinstance(err, PromptsConflictError)
        assert err.entity == "PromptKeywordsTable"
    assert undelete.args[0] == (
        "Failed to undelete keyword due to version mismatch or it was not found."
    )
    assert soft.args[0] == "Failed to soft-delete keyword due to a version mismatch."
    assert undelete.identifier == 7
    assert soft.identifier == 8


def test_error_factories_produce_host_types():
    media = _media_store().spec
    prompts = _prompts_store().spec
    assert isinstance(media.make_input_error("m"), MediaInputError)
    assert isinstance(media.make_database_error("m"), MediaDatabaseError)
    assert isinstance(prompts.make_input_error("m"), PromptsInputError)
    assert isinstance(prompts.make_database_error("m"), PromptsDatabaseError)


def test_fts_nul_scrub_covers_both_hosts():
    # The scrub is a strict improvement adopted from Media for both hosts.
    # Both FTS tables are EXTERNAL-CONTENT (SELECT projects the content
    # table's raw value), so the observable contract is INDEX health: a
    # NUL-bearing keyword still MATCHes its scrubbed terms.
    for store in (_media_store(), _prompts_store()):
        kw_id, _uuid = store.add_keyword_full("nul\u0000scrub")
        hits = store.db.execute_query(
            f"SELECT rowid FROM {store.spec.fts_table} "
            f"WHERE {store.spec.fts_table} MATCH 'scrub'"
        ).fetchall()
        assert [dict(h)["rowid"] for h in hits] == [kw_id]


def test_soft_delete_unlink_scope_matches_host_semantics():
    # With one ACTIVE-parent link and one deleted-parent leftover link:
    # Media removes only the active-parent link; Prompts
    # (unlink_all_links) removes both.
    media_db = MediaDatabase(":memory:", "seam-client")
    active_media, _, _ = media_db.add_media_with_keywords(
        title="A", media_type="text", content="c", keywords=["scope"]
    )
    dead_media, _, _ = media_db.add_media_with_keywords(
        title="B",
        media_type="text",
        content="c",
        keywords=["scope"],
        url="urn:keyword-scope:B",
    )
    assert active_media and dead_media
    media_db.soft_delete_media(dead_media, cascade=False)  # keeps its link
    assert media_db.soft_delete_keyword("scope") is True
    remaining = media_db.execute_query(
        "SELECT COUNT(*) AS n FROM MediaKeywords"
    ).fetchone()
    assert dict(remaining)["n"] == 1  # deleted-parent link retained

    prompts_db = PromptsDatabase(":memory:", "seam-client")
    prompt_id, _, _ = prompts_db.add_prompt(
        name="P", author="a", details=None, keywords=["scope"]
    )
    assert prompt_id is not None
    assert prompts_db.soft_delete_prompt(prompt_id) is True  # removes links
    # An active-parent link plus a legacy leftover link to the deleted
    # parent: the unlink_all_links scope removes BOTH.
    active_prompt, _, _ = prompts_db.add_prompt(
        name="A", author="a", details=None, keywords=["scope"]
    )
    assert active_prompt is not None
    kw_id, _ = prompts_db._keywords.add_keyword_full("scope")
    with prompts_db.transaction() as conn:
        conn.execute(
            "INSERT INTO PromptKeywordLinks (prompt_id, keyword_id) VALUES (?, ?)",
            (prompt_id, kw_id),
        )
    assert prompts_db.soft_delete_keyword("scope") is True
    remaining = prompts_db.execute_query(
        "SELECT COUNT(*) AS n FROM PromptKeywordLinks"
    ).fetchone()
    assert dict(remaining)["n"] == 0  # deleted-parent link removed too


@pytest.mark.parametrize("store_factory", [_media_store, _prompts_store])
def test_add_keyword_full_round_trip_via_store(store_factory):
    store = store_factory()
    kw_id, kw_uuid = store.add_keyword_full("RoundTrip")
    assert kw_id and kw_uuid
    # Idempotent: second add returns the same row without a new version.
    again_id, again_uuid = store.add_keyword_full("roundtrip   ")
    same = again_id == kw_id
    # Prompts' whitespace-collapse normalizes "roundtrip   " to the stored
    # form; Media's strip().lower() also matches. Both hosts find it.
    assert same
    assert again_uuid == kw_uuid
