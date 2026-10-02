"""Absolute census of the triggers on the ChaChaNotes schema (task-19565).

Why this exists: the index census (``test_index_census.py``, task-19045)
disclosed that most schema artifacts were verified by nothing, and triggers
were worse -- at the 2026-08-21 lane's census, 52 of 75 triggers were named
nowhere in ``Tests/`` and NO trigger had its body pinned anywhere. That is
not hypothetical: ``notes_au`` shipped without its ``deleted = 0`` guards,
and the repair lived for years in a runtime self-heal that DROP/reCREATEd
the trigger on EVERY database open (replaced by the v73->v74 migration in
this same task). A trigger whose body was wrong shipped to every user
precisely because nothing compared the live body to a declared expectation.

This module is the guard, mirroring ``test_index_census.py``:
``EXPECTED_CHACHANOTES_TRIGGERS`` is a hand-maintained literal asserted in
BOTH directions against a live fully-migrated DB -- fresh bootstrap AND a
chain-migrated-from-v4 DB, so a stop/resume divergence is caught too. The
literal is deliberately NOT derived from the schema code it checks; updating
it is a deliberate schema-review act, and each failure message says which
side to fix.

What is pinned per trigger: the table, the timing (BEFORE/AFTER), the event
(including the ``UPDATE OF`` column list), and a sha256 digest of the
whitespace-normalized CREATE TRIGGER text -- the digest makes ANY body edit
fail the census, while staying compact for the 184-trigger set. For the
load-bearing FTS soft-delete family (``notes_*``, ``messages_*`` -- where the
notes_au incident lived and where TASK-19566's soft-delete guard depends)
the FULL normalized bodies are pinned below as readable literals, so the
exact semantics are reviewable in this file rather than only as a hash.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from typing import NamedTuple

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import (
    MINIMUM_BOOTSTRAP_VERSION,
    chachanotes_db_at_version,
    open_current_chachanotes_from_legacy,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

_THIS_FILE = "Tests/ChaChaNotesDB/test_trigger_census.py"


class TriggerPin(NamedTuple):
    """The pinned shape: table, timing, event, body digest."""

    table: str
    timing: str
    event: str
    body_digest: str


#: Parses the head of a stored CREATE TRIGGER statement. ``IF NOT EXISTS``
#: never appears in sqlite_master (SQLite strips it on store), but the
#: optional group keeps the parser usable on authored SQL too. The
#: ``UPDATE OF`` column list is matched non-greedily up to `` ON `` so
#: comma-separated column lists parse.
_TRIGGER_HEADER_RE = re.compile(
    r'CREATE TRIGGER (?:IF NOT EXISTS )?(?:"[^"]*"|`[^`]*`|\[[^\]]*\]|[^\s]+)'
    r"\s+(BEFORE|AFTER|INSTEAD OF)?\s*"
    r"(DELETE|INSERT|UPDATE(?:\s+OF\b.*?)?)\s+ON\s+"
    r'(?:"[^"]*"|`[^`]*`|\[[^\]]*\]|[^\s(]+)'
)

_KEYWORD_RE = re.compile(r"^(DELETE|INSERT|UPDATE)", re.IGNORECASE)


def _census(conn: sqlite3.Connection) -> dict[str, TriggerPin]:
    """Read every trigger from a live connection as a ``TriggerPin`` set.

    The digest is taken over the whitespace-normalized CREATE TRIGGER text
    stored in ``sqlite_master``: formatting-only edits do not fail the
    census, any semantic edit does.
    """
    census: dict[str, TriggerPin] = {}
    for name, tbl, sql in conn.execute(
        "SELECT name, tbl_name, sql FROM sqlite_master "
        "WHERE type = 'trigger' ORDER BY name"
    ):
        normalized = " ".join((sql or "").split())
        match = _TRIGGER_HEADER_RE.match(normalized)
        assert match is not None, (
            f"trigger {name} has an unparseable header: {normalized[:120]!r}"
        )
        timing = (match.group(1) or "AFTER").upper()
        event = _KEYWORD_RE.sub(lambda m: m.group(1).upper(), match.group(2))
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        census[name] = TriggerPin(tbl, timing, event, digest)
    return census


@pytest.fixture(scope="module", params=["fresh_bootstrap", "chain_migrated_from_v4"])
def live_trigger_census(request, tmp_path_factory) -> dict[str, TriggerPin]:
    """Census of a live fully-migrated DB, built two independent ways."""
    if request.param == "fresh_bootstrap":
        db = CharactersRAGDB(":memory:", client_id="trigger-census-fresh")
        try:
            return _census(db.get_connection())
        finally:
            db.close_connection()
    db_path = tmp_path_factory.mktemp("trigger_census") / "chain_migrated.sqlite"
    with chachanotes_db_at_version(db_path, MINIMUM_BOOTSTRAP_VERSION):
        pass  # bootstrap a genuinely-v4 DB, then close it
    db = open_current_chachanotes_from_legacy(
        db_path, client_id="trigger-census-chain"
    )
    try:
        return _census(db.get_connection())
    finally:
        db.close_connection()


#: The full expected trigger set on a fully-migrated ChaChaNotes DB.
#: HAND-MAINTAINED ON PURPOSE (see module docstring): update it only as part
#: of a deliberate schema change, in the same commit as the migration that
#: adds, drops, renames, or rewrites a trigger. Sorted by name. Regenerate a
#: candidate entry with the helper at the bottom of this file.
EXPECTED_CHACHANOTES_TRIGGERS: dict[str, TriggerPin] = {
    "agent_lessons_seed_state_monotonic_update": TriggerPin("agent_lessons_seed_state", "BEFORE", "UPDATE OF scope_mode, state",
        "c9be6020820d9e57a78d4ae41c04a51c90c94236ab27b068102d9507528a8f9a"),
    "canvas_documents_ownership_immutable": TriggerPin("canvas_documents", "BEFORE", "UPDATE OF id, conversation_id, created_at",
        "27aca0f3b5ca5adb9dde2d640a0dd2650e899b8e3930d90a566bc9acec1c680e"),
    "canvas_origin_message_owner_guard": TriggerPin("messages", "BEFORE", "UPDATE OF conversation_id",
        "e9a4501d0d6d96b1a57fd76e5512bcbf9cee6ccc592f9d05895f0709832b82c9"),
    "canvas_revisions_no_delete": TriggerPin("canvas_revisions", "BEFORE", "DELETE",
        "ff483ab51720d12ffdd61932dd35e87ed47191697c4ced18d70c8eecae73f60d"),
    "canvas_revisions_no_update": TriggerPin("canvas_revisions", "BEFORE", "UPDATE",
        "0a08030acbfac3e3218cd682a0d354fea96003f3fb291b5b5643c01587466ca9"),
    "canvas_revisions_origin_owner_guard": TriggerPin("canvas_revisions", "BEFORE", "INSERT",
        "c9d0d99d06e2086d08b16b39dc5e13f6bbc99d1b8ca16a4f966bdd90b4cd9f7c"),
    "canvas_revisions_parent_guard": TriggerPin("canvas_revisions", "BEFORE", "INSERT",
        "eec7818a107421aceac436e7f826d40961743ca59571c5b7122a7c519d952a63"),
    "character_cards_ad": TriggerPin("character_cards", "AFTER", "DELETE",
        "5cd1cb3ae94106598f2ad16bf1e9cf2aec2a03f7721ec1c46a83a5453448b2a6"),
    "character_cards_ai": TriggerPin("character_cards", "AFTER", "INSERT",
        "82220e4f9242cad43a9855518086932d16f73f65adab0e01a0a1229d662435d1"),
    "character_cards_au": TriggerPin("character_cards", "AFTER", "UPDATE",
        "9d1e818202fa7cf802899d665bc063390c2598a67b4e027cc0aa3df2c422a648"),
    "character_cards_sync_create": TriggerPin("character_cards", "AFTER", "INSERT",
        "3984e708cfccec664646a03faa063fe77b4c9532d21ededb8efbec168e378ed2"),
    "character_cards_sync_delete": TriggerPin("character_cards", "AFTER", "UPDATE",
        "e4648eaaac553e0a3347dee96bc8a9a04071781a6696a6418afa47f300b0cd9e"),
    "character_cards_sync_undelete": TriggerPin("character_cards", "AFTER", "UPDATE",
        "273798ed595c8aa925a7df39ae539ecb6dfeed7f2940b7539eeaded6cb9d67f2"),
    "character_cards_sync_update": TriggerPin("character_cards", "AFTER", "UPDATE",
        "eff5e8c81508141c2077ccee2d73532a619a9ffa627cdcddbea8f6c5bb81d1cb"),
    "character_conversation_search_characters_ad": TriggerPin("character_cards", "AFTER", "DELETE",
        "0679dda340ea2b4fd422f4c6c3c6ed7bf47e22416dd7adfb55e4cc44d69f9f89"),
    "character_conversation_search_characters_ai": TriggerPin("character_cards", "AFTER", "INSERT",
        "77914a918f3d1a9bbab326d41f7307e8660a6b3f9d4b596b45e55823bb043322"),
    "character_conversation_search_characters_au": TriggerPin("character_cards", "AFTER", "UPDATE",
        "548949918bb30381a3bcaafc65f93be582ef58e0fd7fed2ab2c23a13b8d666c3"),
    "character_conversation_search_conversations_ad": TriggerPin("conversations", "AFTER", "DELETE",
        "2dc4f635a84353e5483854e9152485ee87766718d9ed50105126b3226328c6d8"),
    "character_conversation_search_conversations_ai": TriggerPin("conversations", "AFTER", "INSERT",
        "ded49060645b6f949199fa9e7445d4eb396263ce4c1e75c8a98751d49e578f16"),
    "character_conversation_search_conversations_au": TriggerPin("conversations", "AFTER", "UPDATE",
        "72164b8f61efd231be3f47730fcdd8c5a797890ed73ec8128e7aa441896fcf27"),
    "character_conversation_search_documents_ad": TriggerPin("character_conversation_search_documents", "AFTER", "DELETE",
        "52dbb35b57c284629e9a5327a620edc62aba43db8bb84914e6f64a291e358b9d"),
    "character_conversation_search_documents_ai": TriggerPin("character_conversation_search_documents", "AFTER", "INSERT",
        "a95e77f464cf99c6499b87558570a923ec9ac83a0ea5e8f779d1768a5cd46112"),
    "character_conversation_search_documents_au": TriggerPin("character_conversation_search_documents", "AFTER", "UPDATE",
        "f0185412c5a38bc5654f9e60c43a68ab7a7a90677c17f1adf143551c973087b4"),
    "character_conversation_search_messages_ad": TriggerPin("messages", "AFTER", "DELETE",
        "c3c512d962342e2291fdf631f8bc324bab03ecbabbe68b3f93173dfbadba6012"),
    "character_conversation_search_messages_ai": TriggerPin("messages", "AFTER", "INSERT",
        "eea1ddb876eab672b62da42cbd0b723da844708c9ee912ab9fee49f05a76df86"),
    "character_conversation_search_messages_au": TriggerPin("messages", "AFTER", "UPDATE",
        "3c4734e4b7c6487bb32a063b5f225e2356d0e5d99635736ba6db177239545342"),
    "character_conversation_search_revision_no_delete": TriggerPin("character_conversation_search_revision", "BEFORE", "DELETE",
        "890906a069a20c48b1d535323d6241d976c9b186e2cd0b15c5b475dfefde46b6"),
    "chat_dictionaries_ad": TriggerPin("chat_dictionaries", "AFTER", "DELETE",
        "8fc88f61d74b5c74e39ccb7c870676e906c874770dbef66b681f449921b83ef9"),
    "chat_dictionaries_ai": TriggerPin("chat_dictionaries", "AFTER", "INSERT",
        "e379c074633a86fba6d772508f67650fd813092ede758237e0754454366e8c25"),
    "chat_dictionaries_au": TriggerPin("chat_dictionaries", "AFTER", "UPDATE",
        "ea07922345dfce33e56e64e03075434360c64f5f4c478f5c4dabe0e6c7b9f5c1"),
    "chat_dictionaries_sync_create": TriggerPin("chat_dictionaries", "AFTER", "INSERT",
        "7308be9a3f44daca97f514743416ab28a499ee9809ab796f31252dee1440cb40"),
    "chat_dictionaries_sync_delete": TriggerPin("chat_dictionaries", "AFTER", "UPDATE",
        "bb14f5eb5e5bbc4eb77e019337fb8599df3dab52f3e9859fa9d8dd4598372eb4"),
    "chat_dictionaries_sync_undelete": TriggerPin("chat_dictionaries", "AFTER", "UPDATE",
        "4c13f1212fdc20d8900463ca8b6032fa58ffc7f0ac23f602693237f589e4ccb8"),
    "chat_dictionaries_sync_update": TriggerPin("chat_dictionaries", "AFTER", "UPDATE",
        "3c5b3eb6ee8ba131cd9586c1d5024a21debca5b0d60085f81d68d7b8abcbe95a"),
    "chat_dictionaries_update_timestamp": TriggerPin("chat_dictionaries", "AFTER", "UPDATE",
        "e4fffab4bfb37a7f8f9a7263c7a14b0f7a6da6dcae82c99e5318c545999d81d4"),
    "console_trace_artifacts_no_delete": TriggerPin("console_trace_artifacts", "BEFORE", "DELETE",
        "c665c10ceabc477ebeee23c21b919075d37952378750015e20d3b5a497f804af"),
    "console_trace_artifacts_no_update": TriggerPin("console_trace_artifacts", "BEFORE", "UPDATE",
        "0e3aafe83d33dd0d02ac582e619b8400c0ed47b1fec883b7418bb52393a781e1"),
    "console_trace_call_boundary_source_guard": TriggerPin("console_trace_events", "BEFORE", "INSERT",
        "36b5045fcf0541a45753145b571325de0240d76e8322139a08ba3e85fce8823f"),
    "console_trace_calls_binding_guard": TriggerPin("console_trace_calls", "BEFORE", "UPDATE",
        "fd7009acdb91b26a43babb116475ae340ad5493f85ab2780c8bef76de296a2e3"),
    "console_trace_calls_immutable_guard": TriggerPin("console_trace_calls", "BEFORE", "UPDATE",
        "e42e50d40186c8800cc2335868b65f1d54a1034633292de03cf6aed4781693cf"),
    "console_trace_calls_insert_reserved": TriggerPin("console_trace_calls", "BEFORE", "INSERT",
        "e8bc925e39c24a27d343f56180349be8d0e2398213ebef878d6ea209343cbb46"),
    "console_trace_calls_lifecycle_guard": TriggerPin("console_trace_calls", "BEFORE", "UPDATE",
        "4710ba2b786a1df77a6f6fa45ba31823d6d118428ab08355a846901611f0b60a"),
    "console_trace_calls_no_delete": TriggerPin("console_trace_calls", "BEFORE", "DELETE",
        "f1f98083aba58d154dd057d889a763caf1030d2553886849304081fabf21bfad"),
    "console_trace_calls_open_root_epoch": TriggerPin("console_trace_calls", "AFTER", "UPDATE OF state",
        "66622c5434059c0edd07f7668392fbdf5a826e13643b016d07ee3f741619099e"),
    "console_trace_calls_owner_lineage": TriggerPin("console_trace_calls", "BEFORE", "INSERT",
        "5bc59789ce0d726663b1f7421c27d9cd719741ff1a7581b1dbfacefb2d4031e9"),
    "console_trace_calls_provenance_update_guard": TriggerPin("console_trace_calls", "BEFORE", "UPDATE",
        "2c9be7593038d17d6a080daf27158fb0d6e759856b3c0363cb2036c4b4c84b1f"),
    "console_trace_calls_set_once_guard": TriggerPin("console_trace_calls", "BEFORE", "UPDATE",
        "95e78136135dd442914342ef897cd95f118ebf80ce4f9f62d289d05a09b1cd48"),
    "console_trace_calls_terminal_guard": TriggerPin("console_trace_calls", "BEFORE", "UPDATE",
        "5923402e214e9b45b25586cf2aa7d5f90bc88ba0143d55562bb915f3e6ed9ac1"),
    "console_trace_compaction_state_immutable_key": TriggerPin("console_trace_compaction_state", "BEFORE", "UPDATE",
        "818e6c1065569696adf55b5d2dc335f875bd2345d1a93e4254f2c3bc9fd6aab7"),
    "console_trace_compaction_state_no_delete": TriggerPin("console_trace_compaction_state", "BEFORE", "DELETE",
        "a8bf038669dc40aa8a23e51abac2e5b7266d377895a3c33f7145d6acf5191935"),
    "console_trace_conversations_detach_owner": TriggerPin("conversations", "BEFORE", "DELETE",
        "7deec059f3ff52c41fdaa4f788533e0b7609dc07607c1b010c28a87c74f85053"),
    "console_trace_events_append_order": TriggerPin("console_trace_events", "BEFORE", "INSERT",
        "8b9d51763dea1d103d8857daefcae799ef2a6d5117cc279718f31e43579f8925"),
    "console_trace_events_lineage_guard": TriggerPin("console_trace_events", "BEFORE", "INSERT",
        "880fd46d99b5e8aa88037f0d096fbf287c4fdb5b4ec01ea0636b28faf57e61f4"),
    "console_trace_events_no_delete": TriggerPin("console_trace_events", "BEFORE", "DELETE",
        "2ddff037280e7f37d260628091f85dabae2b915571d8762fd8a07ec692c41397"),
    "console_trace_events_no_update": TriggerPin("console_trace_events", "BEFORE", "UPDATE",
        "36748b15844b9facb3e1e518e3a111aa459daaf8af917e6af6d102ee9be9f408"),
    "console_trace_events_owner_guard": TriggerPin("console_trace_events", "BEFORE", "INSERT",
        "cd432089d4cfaab855d55a040f3c0769733ace42f0f8a0b76250e87a299e8333"),
    "console_trace_events_shape_guard": TriggerPin("console_trace_events", "BEFORE", "INSERT",
        "b7d85d6e7762b2493f8750e8ae7ef92e82d97127021df3cba50b84b355996725"),
    "console_trace_graph_epoch_monotonic": TriggerPin("console_trace_graph_epoch", "BEFORE", "UPDATE",
        "5a22c02a29d4eb86d23ad191cf35c4346b12a1374eb3e04dc8915baab78e0113"),
    "console_trace_graph_epoch_no_delete": TriggerPin("console_trace_graph_epoch", "BEFORE", "DELETE",
        "0a55ff409767499089979c463a5d1013b0ff724523868f8c197f956cb2f4ec84"),
    "console_trace_header_components_no_delete": TriggerPin("console_trace_header_components", "BEFORE", "DELETE",
        "467cbdfb30da6e2a93918d3125832d29ccefd6e3b1bc271aabc72eeb1b33999d"),
    "console_trace_header_components_no_update": TriggerPin("console_trace_header_components", "BEFORE", "UPDATE",
        "666b4722e8e2ec0783dc8e168d451543a4a9891117102dd3b7b181237cae4400"),
    "console_trace_maintenance_state_immutable_key": TriggerPin("console_trace_maintenance_state", "BEFORE", "UPDATE",
        "3a19892095ee40bc49acf261d9b3dc8aa34dab538e9769435f4d92cbbb64d9a1"),
    "console_trace_maintenance_state_no_delete": TriggerPin("console_trace_maintenance_state", "BEFORE", "DELETE",
        "53bb57513954915b1d32f42ae2f767e7e9ce24d232577abdd5c75cc7b0c892ee"),
    "console_trace_migration_root_epoch": TriggerPin("console_trace_migration_state", "AFTER", "UPDATE OF status",
        "d651b122bac33b6fe88dad7c370edd73dbd584ab4298dc2415b57adc128b9498"),
    "console_trace_migration_state_immutable_key": TriggerPin("console_trace_migration_state", "BEFORE", "UPDATE",
        "bd9896256d9a36b608b52f52931c8f99a2367623443a110aa41ddb3d42a17e7e"),
    "console_trace_migration_state_no_delete": TriggerPin("console_trace_migration_state", "BEFORE", "DELETE",
        "53321722a56e7f426ec9e640b507ceb4a08f0c1aecc909feca61535ec9bd5b24"),
    "console_trace_owners_active_prefix_guard": TriggerPin("console_trace_owners", "BEFORE", "INSERT",
        "ed7e06336c480bd7c812baf6aefb66ae24dc93c74c75ceb5c7b5695f6440db9c"),
    "console_trace_owners_detach_only": TriggerPin("console_trace_owners", "BEFORE", "UPDATE",
        "0cf7238aab35be58eb8735172ec6137cd826d6d27fdf1631fe7eb2023d8c4fe6"),
    "console_trace_owners_empty_root_guard": TriggerPin("console_trace_owners", "BEFORE", "INSERT",
        "cd67d3e4e67efebe332d4cf39ff8e53af02eef05a2632e17d16f93ab748b42b6"),
    "console_trace_owners_no_delete": TriggerPin("console_trace_owners", "BEFORE", "DELETE",
        "67a680c6c1e5152fbffa1e262040f421ab3e307a50d9717b0f6435ad5d060ad0"),
    "console_trace_policies_no_delete": TriggerPin("console_trace_policies", "BEFORE", "DELETE",
        "07562ca0d1c1fc54842dd9cef2418c777075b9909614346dae0eb6f88fb7ab51"),
    "console_trace_policies_no_update": TriggerPin("console_trace_policies", "BEFORE", "UPDATE",
        "3c2760373a2efb9e0f84df1f255b5d7d50025bc25eb62ae71052b26e02f413db"),
    "console_trace_redaction_spans_no_delete": TriggerPin("console_trace_redaction_spans", "BEFORE", "DELETE",
        "f47d7d1dfd6e2c9b4a50f5e54f8930c56a7668368ff9a9b15728c54ba572a71b"),
    "console_trace_redaction_spans_no_update": TriggerPin("console_trace_redaction_spans", "BEFORE", "UPDATE",
        "c9f79dc921626a89ab21571e9dcc764533e31eebebe242711d0c1e8add391d13"),
    "console_trace_request_headers_no_delete": TriggerPin("console_trace_request_headers", "BEFORE", "DELETE",
        "72bcbb7e7f9ffa03e10689a899c4ea9c6630380d60fa4d757dd70f4374cd09c8"),
    "console_trace_request_headers_no_update": TriggerPin("console_trace_request_headers", "BEFORE", "UPDATE",
        "afd70980b7f8a1b58c42eb88b6fc4da02e07b370fbca03cd27c9a45a851bcf25"),
    "console_trace_response_links_no_delete": TriggerPin("console_trace_response_links", "BEFORE", "DELETE",
        "a7ff297466141ddaf5548b35e848bd0c2d00bc11e45a43dded65932bb62a486c"),
    "console_trace_response_links_no_update": TriggerPin("console_trace_response_links", "BEFORE", "UPDATE",
        "c8f3d9ae0c4e2a30c9fa3753f1afa828eb4c85964ddeaea99f539f83ba02738e"),
    "console_trace_response_links_owner_guard": TriggerPin("console_trace_response_links", "BEFORE", "INSERT",
        "a1347987f05ef9464070ebe20c6765211495a1a9f549369910e67977f593fbc7"),
    "console_trace_retention_roots_delete_epoch": TriggerPin("console_trace_retention_roots", "AFTER", "DELETE",
        "8478964d1e4adfbd732458bab0f5b6798e433586dc0a17de9ca172428c25bcd9"),
    "console_trace_retention_roots_insert_epoch": TriggerPin("console_trace_retention_roots", "AFTER", "INSERT",
        "c8b60445f209311aa7651302c4e11ee56aecc9126beb8f08ba582ff88ed5128e"),
    "console_trace_retention_roots_no_delete": TriggerPin("console_trace_retention_roots", "BEFORE", "DELETE",
        "41239874bc4f47988543e9856ec625adf4876a4e190e57d4ea1370df21c1a4a4"),
    "console_trace_retention_roots_no_update": TriggerPin("console_trace_retention_roots", "BEFORE", "UPDATE",
        "48ab4b741ab598b90f7f7c7f601ac2cd3d00aadc293f62986b65091f09a5c23b"),
    "console_trace_revision_bindings_no_delete": TriggerPin("console_trace_revision_bindings", "BEFORE", "DELETE",
        "ad537af2ba7cd4724b3305070812481bee899d7de017251994463ed11fd8f965"),
    "console_trace_revision_bindings_no_update": TriggerPin("console_trace_revision_bindings", "BEFORE", "UPDATE",
        "cf9d231213b692db7b13ef616279e099208ed76c855a5fb08b0b1b0a2d1998cf"),
    "console_trace_segments_inherited_surface": TriggerPin("console_trace_segments", "BEFORE", "INSERT",
        "8b95b98174d473c4edc0b8c9680b8038bf320dddbfc2f80c7ac1d08cc7495e1e"),
    "console_trace_segments_no_delete": TriggerPin("console_trace_segments", "BEFORE", "DELETE",
        "0c3eed33694f939fd785da8268df70ce672207a2d53dcdee4347524378d5a843"),
    "console_trace_segments_no_update": TriggerPin("console_trace_segments", "BEFORE", "UPDATE",
        "ffaa0610dea1d5a7224ec861fde10d27f82200a6b408b72708cd2e43ac2200d3"),
    "console_trace_segments_parent_owner_guard": TriggerPin("console_trace_segments", "BEFORE", "INSERT",
        "d4450696c942d031df5faad60cd2dcf863ebc41997cc36675bf34dd22ccd812a"),
    "console_trace_semantic_revisions_lineage": TriggerPin("console_trace_semantic_revisions", "BEFORE", "INSERT",
        "542fd453138cd8341b77c33160dc01518ee637abb93d2ba2f06b1d27c03cf477"),
    "console_trace_semantic_revisions_locator_only": TriggerPin("console_trace_semantic_revisions", "BEFORE", "UPDATE",
        "87c788847b18352d3068f7d9a130f0745c818aeb3627c39905e1ce40545740f3"),
    "console_trace_semantic_revisions_no_delete": TriggerPin("console_trace_semantic_revisions", "BEFORE", "DELETE",
        "161ecd6ad30755da30e78186e8e9fbb92aa16c64ccb61616b506bbc82a26b28f"),
    "console_trace_semantic_revisions_retirement_guard": TriggerPin("console_trace_semantic_revisions", "BEFORE", "UPDATE OF live_message_id, live_locator_retired_at",
        "e7c4d78ce61c1aaf417f25bf81a21db097ab7cd20dcba9da5f2e587a034da9b7"),
    "console_trace_surface_nodes_contiguous": TriggerPin("console_trace_surface_nodes", "BEFORE", "INSERT",
        "48800f57ed5638d91a1e222365515134cc9ebce7c2e32de3424c6e8e12493955"),
    "console_trace_surface_nodes_no_delete": TriggerPin("console_trace_surface_nodes", "BEFORE", "DELETE",
        "b62b7a91ab01c85ab8705e72ef58f48aac9e3a8cad3cf5d1a45d640a2b1b1f3a"),
    "console_trace_surface_nodes_no_update": TriggerPin("console_trace_surface_nodes", "BEFORE", "UPDATE",
        "07a30ff85d502a607991e87457db7fc42aea404e0c19a77d0f5a9645515797df"),
    "console_trace_surface_nodes_owner_guard": TriggerPin("console_trace_surface_nodes", "BEFORE", "INSERT",
        "7a56ed486173a2497ac2b4e7f0e65d4faeebda28e244db30fe0871d2441f6fd9"),
    "console_trace_surface_replacements_lineage": TriggerPin("console_trace_surface_replacements", "BEFORE", "INSERT",
        "a565cd6e3824761a80dc16b8e305cda4d89680837f4a92cb03db91b54c465f24"),
    "console_trace_surface_replacements_no_delete": TriggerPin("console_trace_surface_replacements", "BEFORE", "DELETE",
        "fcf458248674c1223b1a58803867e176c7adcac4a36c0215907755960faad811"),
    "console_trace_surface_replacements_no_update": TriggerPin("console_trace_surface_replacements", "BEFORE", "UPDATE",
        "a7b9004bc5dc2c1a27acfa09404b729127fcdf0223e167562274e4c7ba51d604"),
    "console_trace_surface_replacements_owner_guard": TriggerPin("console_trace_surface_replacements", "BEFORE", "INSERT",
        "ee40c7be84373051e5f045a1d9fdc0d67c58198dcaf81c5f199645b31d711490"),
    "conversation_dictionary_index_ad": TriggerPin("conversations", "AFTER", "DELETE",
        "a82430856bb1c7d1b69b437d5eb851b8087e50aacdc9615d9e4f92f75f105c61"),
    "conversation_dictionary_index_ai": TriggerPin("conversations", "AFTER", "INSERT",
        "10c1a25778b0616bc5e59d45be9bdeaca956a6b1a004c353879c63e4c1f53f61"),
    "conversation_dictionary_index_au": TriggerPin("conversations", "AFTER", "UPDATE",
        "ed42acc9948e7126c716bc298651e90bb4bfeb96f3429bff656acf42625fac91"),
    "conversations_ad": TriggerPin("conversations", "AFTER", "DELETE",
        "ef7722bf7842c7baba6b39161fbad021ce71468add98bd72571bb549bfed9d76"),
    "conversations_ai": TriggerPin("conversations", "AFTER", "INSERT",
        "fccecbf9f50bc108123328e8990eaea65d5b151e93954952acb160ce9b985bb0"),
    "conversations_au": TriggerPin("conversations", "AFTER", "UPDATE",
        "254c622021af11553057789047e5e6701de13266f36a4cb1018c8fb964e9b569"),
    "conversations_sync_create": TriggerPin("conversations", "AFTER", "INSERT",
        "fa041b6058580f0e55b31cef40ed4360c48a4ae549c53686aba8b22c04ceb62d"),
    "conversations_sync_delete": TriggerPin("conversations", "AFTER", "UPDATE",
        "222a89fdf0fd339aff10bf8eda7346e54b0d5180cc6f2e9b0fd7efa6675248a3"),
    "conversations_sync_undelete": TriggerPin("conversations", "AFTER", "UPDATE",
        "d9c3cdfa9399e69bd36b28441001efcc9058e7ba50daf18d9a1456784ca9331e"),
    "conversations_sync_update": TriggerPin("conversations", "AFTER", "UPDATE",
        "32056dd37bd2bb56988f6ee245b3501007be3599b9eaa6c94741587141c057a5"),
    "flashcards_ad": TriggerPin("flashcards", "AFTER", "DELETE",
        "5f8712859238f5620a1452ca8b81b6709fe8fd7c3b4fba1f524f523e48324f97"),
    "flashcards_ai": TriggerPin("flashcards", "AFTER", "INSERT",
        "ec7ded02b913d7f61bc7800045c99be4e953f74c2f43b21b8e3c90fb87cd6451"),
    "flashcards_au": TriggerPin("flashcards", "AFTER", "UPDATE",
        "87af29f6d37e3faf81822387214d96f8db464ac84b8c299763b6bc192898fb1c"),
    "keyword_collections_ad": TriggerPin("keyword_collections", "AFTER", "DELETE",
        "3df6760d837c6de4f5a79f9c6d86d7d111a8b0c8d7401dd95d574d351cb956ca"),
    "keyword_collections_ai": TriggerPin("keyword_collections", "AFTER", "INSERT",
        "fc59f75f4b19a52776a7f27ad3e12cd71e7cb41736d389cc083033cac796857b"),
    "keyword_collections_au": TriggerPin("keyword_collections", "AFTER", "UPDATE",
        "babeb5ff83a986f12d7133e7338f9559fa9042db58c46b50628dec71d2c5d873"),
    "keyword_collections_sync_create": TriggerPin("keyword_collections", "AFTER", "INSERT",
        "49839bedc914a5d2d3b4914068a814c00cc0c62383ee7322dbd1d27b6462c49d"),
    "keyword_collections_sync_delete": TriggerPin("keyword_collections", "AFTER", "UPDATE",
        "8ca2b96da84482f25733292e432cbb158cb604654b24200d9511edbdb65c0903"),
    "keyword_collections_sync_undelete": TriggerPin("keyword_collections", "AFTER", "UPDATE",
        "96ed2eb1c1546ca1ad429bc7e2fd479c93c8ea07f189eaf9f3dea058cfac1278"),
    "keyword_collections_sync_update": TriggerPin("keyword_collections", "AFTER", "UPDATE",
        "2a2308f0c649261bf7afd6943923b7515ef3674f272c7e5d9c1045e93a1cbceb"),
    "keywords_ai": TriggerPin("keywords", "AFTER", "INSERT",
        "b0bc7a2138c5d05d8302fbe248639152cf36538fe9c9238cd387316cb79fb54e"),
    "keywords_au": TriggerPin("keywords", "AFTER", "UPDATE",
        "d203c741b98522418c9296438eb5cdd49327fbf3d82499a66038dbfef8c7e7a4"),
    "keywords_bd": TriggerPin("keywords", "BEFORE", "DELETE",
        "92225a9eee857afb07c0ca514ab2ec44bab9f7e887dc23197d1efc6cd41732b2"),
    "keywords_sync_create": TriggerPin("keywords", "AFTER", "INSERT",
        "81c647107dd012b61108e028acc5ea331382ff4a7505405ff233a6563615fc17"),
    "keywords_sync_delete": TriggerPin("keywords", "AFTER", "UPDATE",
        "aadd35a9f753bb1041f92597f20a472f7311f7f16e04145cdd176a9879df5b52"),
    "keywords_sync_undelete": TriggerPin("keywords", "AFTER", "UPDATE",
        "85c7de8df5b51b7b78e49ffe6e45918cb154f16f660fb289b3282913efc7552d"),
    "keywords_sync_update": TriggerPin("keywords", "AFTER", "UPDATE",
        "98264a148f85dd11ef4d71d999edef531ff7238edef7d5f1413f6b6e7c8132d6"),
    "message_attachments_semantic_delete_guard": TriggerPin("message_attachments", "BEFORE", "DELETE",
        "6723f68f2e9f7ce51d8a4019e628938acb811c85aeabeab950480192a6420a0b"),
    "message_attachments_semantic_insert_guard": TriggerPin("message_attachments", "BEFORE", "INSERT",
        "ef4e7381f453ccce9462d290b2e8d9acdc08f870bb25d92131980f19e6922812"),
    "message_attachments_semantic_update_guard": TriggerPin("message_attachments", "BEFORE", "UPDATE",
        "7a0fa412594761ca570a695f7144492f30a5924fb4e6104a5ce435aae0788978"),
    "messages_ad": TriggerPin("messages", "AFTER", "DELETE",
        "7d48fc7e1cec21fd0743fe00835fc77753ad2066fd3507f2ed2fbc9fa1e87d05"),
    "messages_ai": TriggerPin("messages", "AFTER", "INSERT",
        "ddbde056fc7034be012f91028a9be66cf1c90d3c92cf692d76bf4e178c6b0744"),
    "messages_au": TriggerPin("messages", "AFTER", "UPDATE OF content, deleted",
        "81f0680e14494bfc75ebbbbe9d0037c3d0a844fc3cdde38b0aaf353edd7698b1"),
    "messages_semantic_delete_guard": TriggerPin("messages", "BEFORE", "DELETE",
        "a5d0d78f26d0673dedfed7259858f8f04242a7d21d02fd9dcb9c0f273f928e1e"),
    "messages_semantic_update_guard": TriggerPin("messages", "BEFORE", "UPDATE OF id, conversation_id, parent_message_id, sender, role, content, image_data, image_mime_type, provider_continuation_json, thinking_blocks_json, assistant_generation_state",
        "15be15d1a98b924cbb3cd86ec89580a6599c87dda4d4136d7d11f6b0dab129ae"),
    "messages_sync_create": TriggerPin("messages", "AFTER", "INSERT",
        "a8eb0b2447385e64aeba8a9d56f9b3df485fef146793d386dc3358b4a4359ee0"),
    "messages_sync_delete": TriggerPin("messages", "AFTER", "UPDATE",
        "cd6b0ce9534096b7c70ef2c72b8c94db5d31cefd938f3f9789bcb2cf99fb5e7a"),
    "messages_sync_undelete": TriggerPin("messages", "AFTER", "UPDATE",
        "259cea539e1d10d21f0e742c3633c964e99ff897ce1f4dd683a9c184586f9f92"),
    "messages_sync_update": TriggerPin("messages", "AFTER", "UPDATE",
        "4fa0d65427fccfa1cbd0414b3f38973bff6873485573589a0e2705d984b375c4"),
    "mindmap_nodes_ad": TriggerPin("mindmap_nodes", "AFTER", "DELETE",
        "3bfe6c2c89fadafcc8996366d9fe95f69e9b80274090eeaa04aa084607ad9532"),
    "mindmap_nodes_ai": TriggerPin("mindmap_nodes", "AFTER", "INSERT",
        "c4afd77c393f442a54313f482887fbfb71c0e3dc5f749d2bf35e89d5eda1eb35"),
    "mindmap_nodes_au": TriggerPin("mindmap_nodes", "AFTER", "UPDATE",
        "7eb5fcf0fc7dbfaabc7c0988512b50032ee9bbf76a5acd0fe71b2641a57742f6"),
    "notes_ad": TriggerPin("notes", "AFTER", "DELETE",
        "f3b4b1fa5fcf8fad1cd103a4f89210b4a2706eefdd996375a6c5e78ba5ec22b8"),
    "notes_ai": TriggerPin("notes", "AFTER", "INSERT",
        "07ada62d03ebb8f76dca87a6800434d183b670e4bdb4a1849d54f7eeec0648d4"),
    "notes_au": TriggerPin("notes", "AFTER", "UPDATE",
        "e4b936dd71661c8746d0dd353fffac1980d646e989786905e82f0d13dccb6f6f"),
    "notes_sync_create": TriggerPin("notes", "AFTER", "INSERT",
        "d3d7d104893e25ffc49d20f75402549a061eee42583286f13b60bdc7383f1f99"),
    "notes_sync_delete": TriggerPin("notes", "AFTER", "UPDATE",
        "b7cbb86b20c84e599311cf7331966a3ad578c19ff70b4f788a4d7829f7f5ca56"),
    "notes_sync_undelete": TriggerPin("notes", "AFTER", "UPDATE",
        "ae4da843c48cb624b5161c4ce7d157d933b3a95ffb73e8def1b7b6507ff80482"),
    "notes_sync_update": TriggerPin("notes", "AFTER", "UPDATE",
        "5b99da7a81d3a42b67e7fa57b9aed6ce742879c58626031cdee56e8145989e3b"),
    "sync_log_prune_character_cards": TriggerPin("character_cards", "AFTER", "UPDATE",
        "93f97de18398a7bd8fc0c2aeca3b9eb47f53a1991d504a2bb495aaaade836675"),
    "sync_log_prune_character_cards_hard": TriggerPin("character_cards", "AFTER", "DELETE",
        "86dfc1b2da5bc3f7c9b61583536d55cb94f6bb25ec4f3bcc6740ad538ca577f8"),
    "sync_log_prune_chat_dictionaries": TriggerPin("sync_log", "AFTER", "INSERT",
        "42d1250c3624ba875f4b41ddbe38fe18b3d408fa10249d59e6eabd4796dfa9b2"),
    "sync_log_prune_chat_dictionaries_hard": TriggerPin("chat_dictionaries", "AFTER", "DELETE",
        "570e6dad589247dc4cecc5f3aa315936a1ebde88f665e49fd58188d7c2f85721"),
    "sync_log_prune_conversations": TriggerPin("sync_log", "AFTER", "INSERT",
        "28ebe17cd0605c1a6c834ea62d6653d2421c0a3428e244c022e71af5c06b5aca"),
    "sync_log_prune_conversations_hard": TriggerPin("conversations", "AFTER", "DELETE",
        "43f10bd2f10738f6ce03c12736de4a0842848f551f74816f30c394c3e362635e"),
    "sync_log_prune_keyword_collections": TriggerPin("keyword_collections", "AFTER", "UPDATE",
        "0b8b4d10836c295ad30d65170124145c2054de9cb32ef92d682601eaa8fe31d5"),
    "sync_log_prune_keyword_collections_hard": TriggerPin("keyword_collections", "AFTER", "DELETE",
        "1d7bac34c93bb1060c62404837d3fd9ab92a417a8cbb85c32e62f63bbe932fe8"),
    "sync_log_prune_keywords": TriggerPin("keywords", "AFTER", "UPDATE",
        "e576ceaf2b735f1f564132510427d6a8441707616081adf9421f15af9e41c7a0"),
    "sync_log_prune_keywords_hard": TriggerPin("keywords", "AFTER", "DELETE",
        "7bf09f738312bc6be6c655257058b0a9233701c1099a50a405456e6518dda2c7"),
    "sync_log_prune_messages": TriggerPin("messages", "AFTER", "UPDATE",
        "5e9c404b5203f2ce811b36aae742332760ca28da4372cf59a2b0b0ba80149797"),
    "sync_log_prune_messages_hard": TriggerPin("messages", "AFTER", "DELETE",
        "804a6fd942022fa5bf4af8fba8f6105952c65c084c0d75ed334c26eb1a2b5e47"),
    "sync_log_prune_notes": TriggerPin("notes", "AFTER", "UPDATE",
        "d57c117f3789c48ec8fea90157ffc7a07f2400e8362b6a74dddf086bca1c2c49"),
    "sync_log_prune_notes_hard": TriggerPin("notes", "AFTER", "DELETE",
        "8ecb9e0a72b2b51263a3f4aaa31d7943b019b511fd98e0b27a41959e9c984ea7"),
    "sync_log_prune_world_book_entries": TriggerPin("sync_log", "AFTER", "INSERT",
        "883dd021a01a7b794cac48ba5b91b2f213634cf04dc8671c5972259b9a32e01c"),
    "sync_log_prune_world_book_entries_hard": TriggerPin("world_book_entries", "AFTER", "DELETE",
        "533d7c4de69d8b8c6431e5aa44a5ada51e0bb6001763f5e2a49a17c4d392a3a7"),
    "sync_log_prune_world_books": TriggerPin("sync_log", "AFTER", "INSERT",
        "eef9fca8569e755523a53fb4add6a72130884d9d8727b44e2f16d8fae2056f6e"),
    "sync_log_prune_world_books_hard": TriggerPin("world_books", "AFTER", "DELETE",
        "f522074daeb5ebb6ac9c67a94599811c4e39e2595b41b6a7bfee31868387bbfa"),
    "topics_ad": TriggerPin("topics", "AFTER", "DELETE",
        "e2e6a916c2f89c27ef9ddf49489571d9376cf2c8bd65a970116c0b58117ffe08"),
    "topics_ai": TriggerPin("topics", "AFTER", "INSERT",
        "0b6cc525c7ddd84c048afed15d9aecec7ce889fd397ec66e75f951eee7142acc"),
    "topics_au": TriggerPin("topics", "AFTER", "UPDATE",
        "668f179d09e9705036516369a706d96dd3c036eca5c36f65c57a5154776a4c1a"),
    "world_book_entries_ad": TriggerPin("world_book_entries", "AFTER", "DELETE",
        "cda53054e8b433766ec773ce6c4267652e7c4eef94a621fd16a87c5798186bd2"),
    "world_book_entries_ai": TriggerPin("world_book_entries", "AFTER", "INSERT",
        "1cc2687209a59a88422ada404d9213b0cc25830839122a25c03cf2bc6c0a4fb8"),
    "world_book_entries_au": TriggerPin("world_book_entries", "AFTER", "UPDATE",
        "1c64a49d9b665c1d16275e6819f55d9fc2d716efdaa33d743b8e0427e2681b00"),
    "world_book_entries_sync_create": TriggerPin("world_book_entries", "AFTER", "INSERT",
        "10328ac946f441316d0d7a1b6eaa61c2217ac3e8d3ebdeba27e80fa28469431d"),
    "world_book_entries_sync_delete": TriggerPin("world_book_entries", "AFTER", "DELETE",
        "7e6bd36e795368e1dafeede8d84b5db464599f459fa5fea4e78c087bd3297658"),
    "world_book_entries_sync_update": TriggerPin("world_book_entries", "AFTER", "UPDATE",
        "12492c1ca32aa80dcdf9bf4cbad0bf737fef18778838453a36069509f6b76ab8"),
    "world_books_ad": TriggerPin("world_books", "AFTER", "DELETE",
        "4fcb751e1a91fc8ab0bc7afed49f049998b86c99b664008938850dd10cf109b2"),
    "world_books_ai": TriggerPin("world_books", "AFTER", "INSERT",
        "4ccedd916d5722445c5efa281b3ac6b1016419e104fd3bdd9369e3c102f28477"),
    "world_books_au": TriggerPin("world_books", "AFTER", "UPDATE",
        "61a98fd58d6271545097b59a68f8fcf64ceefb71372fc6ab18d20536a784674a"),
    "world_books_sync_create": TriggerPin("world_books", "AFTER", "INSERT",
        "28eb1e1da6c01661b782a3f506ac6969bfdeda5b6786b95f30bf390cac8aa181"),
    "world_books_sync_delete": TriggerPin("world_books", "AFTER", "UPDATE",
        "0f54e9c9ee823b03e35623d71dcd8a8eccf612bba6d9c6ce2d77d687f9c035e6"),
    "world_books_sync_undelete": TriggerPin("world_books", "AFTER", "UPDATE",
        "edcc04682e13cb096cc7f05c3e461dd4487479aa16acbb73a2873bd4a581c8db"),
    "world_books_sync_update": TriggerPin("world_books", "AFTER", "UPDATE",
        "d195452f51c5bb939457d48d64d2ba3a93db9ea5a694e52358f95dddc4eac2b7"),
}


class TestChachanotesTriggerCensusMatchesLiveSchema:
    """task-19565: the absolute trigger census, both directions."""

    def test_no_missing_triggers(self, live_trigger_census):
        """Every pinned trigger must exist on the live, fully-migrated DB."""
        missing = sorted(set(EXPECTED_CHACHANOTES_TRIGGERS) - set(live_trigger_census))
        assert not missing, (
            f"Pinned ChaChaNotes triggers are MISSING from the live schema: "
            f"{missing}. A migration dropped or renamed them, and no other "
            f"test turns red for that (task-19565: 52 of 75 triggers were "
            f"referenced by no test at all). If the drop/rename is "
            f"deliberate, update EXPECTED_CHACHANOTES_TRIGGERS in "
            f"{_THIS_FILE} in the same commit as the schema change; "
            f"otherwise restore the CREATE TRIGGER in "
            f"tldw_chatbook/DB/ChaChaNotes_DB.py (or the migrations/*.sql "
            f"file the step executes)."
        )

    def test_no_unexpected_triggers(self, live_trigger_census):
        """Every live trigger must be pinned in the expected literal."""
        unexpected = sorted(set(live_trigger_census) - set(EXPECTED_CHACHANOTES_TRIGGERS))
        paste_lines = "\n".join(
            f'    "{name}": TriggerPin("{pin.table}", "{pin.timing}", "{pin.event}",\n'
            f'        "{pin.body_digest}"),'
            for name, pin in sorted(live_trigger_census.items())
            if name in unexpected
        )
        assert not unexpected, (
            f"Live ChaChaNotes schema defines triggers not pinned in "
            f"EXPECTED_CHACHANOTES_TRIGGERS: {unexpected}. If your migration "
            f"deliberately adds them, pin them in {_THIS_FILE} (sorted by "
            f"name) in the same commit -- ready to paste:\n{paste_lines}"
        )

    def test_trigger_shapes_match(self, live_trigger_census):
        """Table, timing, event, and body digest must match per trigger.

        The body digest comparison is the reason this census exists: the
        notes_au incident shipped a wrong BODY with a perfectly ordinary
        name. A digest mismatch means the live trigger text changed; diff
        the live sqlite_master.sql against the defining migration (or the
        v4 base script) before touching the pin.
        """
        divergent = []
        for name in sorted(set(EXPECTED_CHACHANOTES_TRIGGERS) & set(live_trigger_census)):
            expected = EXPECTED_CHACHANOTES_TRIGGERS[name]
            live = live_trigger_census[name]
            if expected != live:
                divergent.append(f"{name}: expected {expected!r} != live {live!r}")
        assert not divergent, (
            "Pinned trigger shapes diverge from the live ChaChaNotes schema "
            "(update EXPECTED_CHACHANOTES_TRIGGERS in "
            f"{_THIS_FILE} only if the body change is deliberate -- a body "
            "digest change is a semantic change to what the trigger does at "
            "runtime, e.g. the notes_au deleted-guard incident):\n"
            + "\n".join(divergent)
        )


#: Full normalized bodies for the load-bearing FTS soft-delete family.
#: These are the triggers whose bodies decide whether soft-deleted rows stay
#: out of FTS search and whether restored rows come back (the notes_au
#: defect class) -- TASK-19566's FTS soft-delete guard lives here too. They
#: are pinned as readable text, not just digests, so a mechanical literal
#: update cannot hide what the body does.
LOAD_BEARING_TRIGGER_BODIES: dict[str, str] = {
    "messages_ad": (
        "CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN INSERT INTO messages_fts(messages_fts,rowid,content) SELECT 'delete',old.rowid,old.content WHERE EXISTS (SELECT 1 FROM messages_fts_docsize WHERE rowid = old.rowid); END"
    ),
    "messages_ai": (
        "CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN INSERT INTO messages_fts(rowid,content) SELECT new.rowid,new.content WHERE new.deleted = 0; END"
    ),
    "messages_au": (
        "CREATE TRIGGER messages_au AFTER UPDATE OF content, deleted ON messages BEGIN INSERT INTO messages_fts(messages_fts,rowid,content) SELECT 'delete',old.rowid,old.content WHERE old.deleted = 0 AND EXISTS (SELECT 1 FROM messages_fts_docsize WHERE rowid = old.rowid); INSERT INTO messages_fts(rowid,content) SELECT new.rowid,new.content WHERE new.deleted = 0; END"
    ),
    "notes_ad": (
        "CREATE TRIGGER notes_ad AFTER DELETE ON notes BEGIN INSERT INTO notes_fts(notes_fts,rowid,title,content) VALUES('delete',old.rowid,old.title,old.content); END"
    ),
    "notes_ai": (
        "CREATE TRIGGER notes_ai AFTER INSERT ON notes BEGIN INSERT INTO notes_fts(rowid,title,content) SELECT new.rowid,new.title,new.content WHERE new.deleted = 0; END"
    ),
    "notes_au": (
        "CREATE TRIGGER notes_au AFTER UPDATE ON notes BEGIN INSERT INTO notes_fts(notes_fts,rowid,title,content) SELECT 'delete',old.rowid,old.title,old.content WHERE old.deleted = 0; INSERT INTO notes_fts(rowid,title,content) SELECT new.rowid,new.title,new.content WHERE new.deleted = 0; END"
    ),
}


class TestLoadBearingFtsTriggerBodies:
    """The FTS soft-delete family is pinned as readable bodies, not hashes."""

    @pytest.mark.parametrize("name", sorted(LOAD_BEARING_TRIGGER_BODIES))
    def test_body_is_pinned_and_guarded(self, live_trigger_census, name):
        assert name in live_trigger_census, (
            f"{name} is pinned as load-bearing but missing from the live schema"
        )
        live = live_trigger_census[name]
        assert live.body_digest == hashlib.sha256(
            LOAD_BEARING_TRIGGER_BODIES[name].encode("utf-8")
        ).hexdigest(), (
            f"{name}: the live body diverges from the pinned load-bearing "
            f"literal. This family decides whether soft-deleted rows stay "
            f"out of FTS search (the notes_au incident); reconcile the "
            f"migration and the pin in the same commit."
        )

    def test_notes_au_carries_both_deleted_guards(self):
        """The exact incident shape: notes_au needs BOTH guards.

        The historical defect was the missing ``old.deleted = 0`` on the
        FTS 'delete' half; the pre-fix body used an unguarded
        ``VALUES('delete',...)``. Both guards are pinned so neither half
        can silently regress.
        """
        body = LOAD_BEARING_TRIGGER_BODIES["notes_au"]
        assert "WHERE old.deleted = 0" in body
        assert "WHERE new.deleted = 0" in body
        assert "VALUES('delete'" not in body


def _print_live_pins() -> None:
    """Print the current live census as pin literal lines (dev helper).

    Run under pytest config redirection is NOT needed; call from a REPL
    against a fresh in-memory DB and paste the output into
    EXPECTED_CHACHANOTES_TRIGGERS as part of a deliberate schema change.
    """
    db = CharactersRAGDB(":memory:", client_id="trigger-census-dev")
    try:
        for name, pin in sorted(_census(db.get_connection()).items()):
            print(
                f'    "{name}": TriggerPin("{pin.table}", "{pin.timing}", '
                f'"{pin.event}",\n        "{pin.body_digest}"),'
            )
    finally:
        db.close_connection()
