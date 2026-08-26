"""task-1337 (plan Task 9): the descriptor-backed direct Library tools on the
local MCP surface.

The Console gained the descriptor-backed ``library_*`` tools in Tasks 1-8;
this file pins their local MCP exposure, which is deliberately FastMCP-free
(owner directive, 2026-08-07) -- the in-app surface is the manifest plus the
direct runtime delegate:

- manifest: ``describe_local_mcp_capabilities()`` keeps the 9 implemented
  AST-derived legacy tools unchanged and appends exactly the descriptor
  tools, with names/descriptions/``inputSchema`` taken from
  ``LIBRARY_TOOL_DESCRIPTORS`` (never hand-duplicated literals);
- direct runtime: ``LocalMCPRuntimeDelegate.execute_tool`` dispatches
  descriptor names to the shared synchronous ``LocalLibraryToolService`` via
  ``asyncio.to_thread``, returns the service payload unchanged, keeps the
  existing ``KeyError`` for unknown names, and reports descriptor tools as
  ``implemented`` (not ``missing``) in protocol diagnostics;
- bootstrap: ``build_local_library_tool_service`` composes all six local
  backends with their real constructor signatures into one shared service,
  degrading any failing backend to ``feature_unavailable``, threads the
  runtime-policy enforcer into the chunk tool service (chunking-agent-tools
  Task 5), and the delegate builds it lazily exactly once.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS
from tldw_chatbook.MCP.local_runtime_delegate import (
    RAW_TOOL_CALL_REFUSED_MESSAGE,
    LocalMCPRuntimeDelegate,
    RawToolCallRefusedError,
)
from tldw_chatbook.MCP.server import describe_local_mcp_capabilities

LEGACY_TOOL_NAMES = [
    "chat_with_llm",
    "chat_with_character",
    "search_rag",
    "search_conversations",
    "create_note",
    "search_notes",
    "list_characters",
    "get_conversation_history",
    "export_conversation",
]

LIBRARY_TOOL_NAMES = list(LIBRARY_TOOL_DESCRIPTORS)


class FakeLibraryToolService:
    """Synchronous stand-in for ``LocalLibraryToolService`` (duck-typed invoke)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.thread_ids: list[int] = []

    def invoke(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        self.thread_ids.append(threading.get_ident())
        return {"echo": tool_name, "arguments": dict(arguments)}


# -- Manifest -----------------------------------------------------------------


def test_manifest_keeps_legacy_tools_then_appends_the_24_descriptor_tools():
    manifest = describe_local_mcp_capabilities()
    tools = manifest["tools"]
    names = [entry["name"] for entry in tools]

    assert names[: len(LEGACY_TOOL_NAMES)] == LEGACY_TOOL_NAMES
    library_entries = tools[len(LEGACY_TOOL_NAMES) :]
    assert [entry["name"] for entry in library_entries] == LIBRARY_TOOL_NAMES
    assert len(tools) == len(LEGACY_TOOL_NAMES) + 24


def test_manifest_does_not_advertise_unimplemented_ingest_media():
    """The local MCP manifest omits the retired placeholder ingest tool."""
    names = {
        entry["name"] for entry in describe_local_mcp_capabilities()["tools"]
    }

    assert "ingest_media" not in names


def test_manifest_library_entries_match_descriptors_exactly():
    manifest = describe_local_mcp_capabilities()
    by_name = {entry["name"]: entry for entry in manifest["tools"]}

    for name, descriptor in LIBRARY_TOOL_DESCRIPTORS.items():
        entry = by_name[name]
        assert entry["description"] == descriptor.description
        assert entry["inputSchema"] == descriptor.input_schema


def test_manifest_library_entries_do_not_alias_descriptor_schemas():
    """Mutating a manifest entry must never corrupt the shared descriptor table."""
    manifest = describe_local_mcp_capabilities()
    by_name = {entry["name"]: entry for entry in manifest["tools"]}
    entry = by_name["library_get_media"]
    entry["inputSchema"]["properties"]["id"]["type"] = "MUTATED"

    descriptor = LIBRARY_TOOL_DESCRIPTORS["library_get_media"]
    assert descriptor.input_schema["properties"]["id"]["type"] == "string"


@pytest.mark.asyncio
async def test_tools_list_request_exposes_library_schemas():
    delegate = LocalMCPRuntimeDelegate(library_service=FakeLibraryToolService())

    result = await delegate.request("tools/list")

    by_name = {entry["name"]: entry for entry in result["tools"]}
    for name, descriptor in LIBRARY_TOOL_DESCRIPTORS.items():
        assert by_name[name]["description"] == descriptor.description
        assert by_name[name]["inputSchema"] == descriptor.input_schema


# -- Direct-runtime dispatch ----------------------------------------------------


@pytest.mark.asyncio
async def test_delegate_refuses_retired_placeholder_ingest_media():
    """Direct runtime dispatch rejects the retired placeholder ingest tool."""
    delegate = LocalMCPRuntimeDelegate(library_service=FakeLibraryToolService())

    with pytest.raises(KeyError, match="Unsupported local MCP tool: ingest_media"):
        await delegate.execute_tool("ingest_media", {"url": "https://example.com"})


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", LIBRARY_TOOL_NAMES)
async def test_delegate_dispatches_library_tools_off_the_event_loop(tool_name):
    service = FakeLibraryToolService()
    delegate = LocalMCPRuntimeDelegate(library_service=service)
    loop_thread_id = threading.get_ident()

    result = await delegate.execute_tool(tool_name, {"probe": True})

    # The service's dict comes back unchanged.
    assert result == {"echo": tool_name, "arguments": {"probe": True}}
    # Exactly one dispatch, with the raw name and payload.
    assert service.calls == [(tool_name, {"probe": True})]
    # The synchronous service ran on a worker thread (asyncio.to_thread), not
    # the event loop's thread.
    assert service.thread_ids
    assert set(service.thread_ids).isdisjoint({loop_thread_id})


@pytest.mark.asyncio
async def test_delegate_keeps_keyerror_for_unknown_tool_names():
    service = FakeLibraryToolService()
    delegate = LocalMCPRuntimeDelegate(library_service=service)

    with pytest.raises(KeyError, match="Unsupported local MCP tool"):
        await delegate.execute_tool("library_not_a_real_tool", {})
    with pytest.raises(KeyError, match="Unsupported local MCP tool"):
        await delegate.execute_tool("definitely_unknown", {})
    assert service.calls == []


@pytest.mark.asyncio
async def test_delegate_raw_tools_call_request_refuses_library_tools():
    service = FakeLibraryToolService()
    delegate = LocalMCPRuntimeDelegate(library_service=service)

    with pytest.raises(RawToolCallRefusedError) as exc_info:
        await delegate.request(
            "tools/call",
            {"name": "library_search_notes", "arguments": {"query": "roadmap"}},
        )

    assert str(exc_info.value) == RAW_TOOL_CALL_REFUSED_MESSAGE
    assert service.calls == []


def test_delegate_diagnostics_report_library_tools_implemented_not_missing():
    delegate = LocalMCPRuntimeDelegate(library_service=FakeLibraryToolService())

    diagnostics = delegate.get_protocol_diagnostics()
    tools = diagnostics["implementation"]["tools"]

    for name in LIBRARY_TOOL_NAMES:
        assert name in tools["implemented"]
        assert name not in tools["missing"]
    # The legacy surface is unchanged: every legacy tool except the documented
    # unavailable one stays implemented, and nothing at all is missing.
    assert tools["unavailable"] == ["chat_with_llm"]
    for name in LEGACY_TOOL_NAMES:
        if name != "chat_with_llm":
            assert name in tools["implemented"]
    assert tools["missing"] == []


@pytest.mark.asyncio
async def test_delegate_lazily_constructs_one_shared_service(monkeypatch):
    import tldw_chatbook.MCP.local_runtime_delegate as delegate_module
    import tldw_chatbook.MCP.server as server_module

    fake_service = FakeLibraryToolService()
    factory_calls = []

    def fake_factory(**kwargs):
        factory_calls.append(kwargs)
        return fake_service

    monkeypatch.setattr(
        server_module, "build_local_library_tool_service", fake_factory, raising=False
    )
    monkeypatch.setattr(
        delegate_module, "get_chachanotes_db_lazy", lambda: object()
    )
    monkeypatch.setattr(delegate_module, "get_media_db_lazy", lambda: object())

    delegate = LocalMCPRuntimeDelegate()
    await delegate.execute_tool("library_list_media", {})
    await delegate.execute_tool("library_list_notes", {})

    assert len(factory_calls) == 1  # built once, then cached
    assert set(factory_calls[0]) == {"chachanotes_db", "media_db", "policy_enforcer"}
    assert [call[0] for call in fake_service.calls] == [
        "library_list_media",
        "library_list_notes",
    ]


@pytest.mark.asyncio
async def test_delegate_forwards_its_policy_enforcer_to_the_factory(monkeypatch):
    """Task 5 (spec §6): the delegate carries the enforcer handle into the
    shared-service factory so the chunk tools' writing operations are
    service-level gated on the local MCP surface too."""
    import tldw_chatbook.MCP.local_runtime_delegate as delegate_module
    import tldw_chatbook.MCP.server as server_module

    enforcer = object()
    seen = {}

    def fake_factory(**kwargs):
        seen.update(kwargs)
        return FakeLibraryToolService()

    monkeypatch.setattr(
        server_module, "build_local_library_tool_service", fake_factory, raising=False
    )
    monkeypatch.setattr(
        delegate_module, "get_chachanotes_db_lazy", lambda: object()
    )
    monkeypatch.setattr(delegate_module, "get_media_db_lazy", lambda: object())

    delegate = LocalMCPRuntimeDelegate(policy_enforcer=enforcer)
    await delegate.execute_tool("library_list_media", {})

    assert seen["policy_enforcer"] is enforcer


# -- Shared-service factory (bootstrap) -----------------------------------------


def _recording_class(record: list):
    class _RecordingFake:
        def __init__(self, *args, **kwargs):
            self.ctor_args = args
            self.ctor_kwargs = kwargs
            record.append(self)

    return _RecordingFake


def _patch_factory_backends(monkeypatch, tmp_path, *, raising: set[str] | None = None):
    """Patch every constructor the shared-service factory touches.

    Returns a SimpleNamespace of per-site instance records. Sites named in
    ``raising`` raise instead of constructing, to exercise the factory's
    per-backend degradation.
    """
    from tldw_chatbook import config as config_module
    import tldw_chatbook.Chat.chat_conversation_service as conversation_module
    import tldw_chatbook.DB.Library_Collections_DB as collections_db_module
    import tldw_chatbook.Library.library_collections_service as collections_module
    import tldw_chatbook.Media.local_media_reading_service as media_module
    import tldw_chatbook.Notes.Notes_Library as notes_module
    import tldw_chatbook.Prompt_Management.local_prompt_service as prompt_module
    import tldw_chatbook.Skills_Interop.local_skills_service as skills_module

    raising = raising or set()
    records = SimpleNamespace(
        notes=[],
        media=[],
        prompt=[],
        skills=[],
        conversation=[],
        collections_db=[],
        collections=[],
    )

    monkeypatch.setattr(
        config_module, "get_chachanotes_db_path", lambda: tmp_path / "chacha.db"
    )
    monkeypatch.setattr(
        config_module,
        "get_library_collections_db_path",
        lambda: tmp_path / "collections.db",
    )
    monkeypatch.setattr(config_module, "get_user_data_dir", lambda: tmp_path)

    def _site(key, module, attribute):
        if key in raising:

            def _raise(*args, **kwargs):
                raise RuntimeError(f"boom:{key}")

            monkeypatch.setattr(module, attribute, _raise)
        else:
            monkeypatch.setattr(
                module, attribute, _recording_class(getattr(records, key))
            )

    _site("notes", notes_module, "NotesInteropService")
    _site("media", media_module, "LocalMediaReadingService")
    _site("prompt", prompt_module, "LocalPromptService")
    _site("skills", skills_module, "LocalSkillsService")
    _site("conversation", conversation_module, "ChatConversationService")
    _site("collections_db", collections_db_module, "LibraryCollectionsDB")
    _site("collections", collections_module, "LocalLibraryCollectionsService")
    return records


def test_factory_builds_six_backends_with_real_signatures(monkeypatch, tmp_path):
    import tldw_chatbook.MCP.server as server_module
    from tldw_chatbook.config import CLI_APP_CLIENT_ID
    import tldw_chatbook.Skills_Interop.local_skills_service as skills_module

    records = _patch_factory_backends(monkeypatch, tmp_path)
    chachanotes_db, media_db = object(), object()

    service = server_module.build_local_library_tool_service(
        chachanotes_db=chachanotes_db, media_db=media_db
    )

    notes = records.notes[0]
    assert notes.ctor_kwargs == {
        "base_db_directory": tmp_path,
        "api_client_id": CLI_APP_CLIENT_ID,
        "global_db_to_use": chachanotes_db,
    }
    assert records.media[0].ctor_args == (media_db,)
    assert records.prompt[0].ctor_args == ()
    assert records.skills[0].ctor_kwargs == {
        "store_dir": skills_module.default_local_skills_store_dir(tmp_path)
    }
    assert records.conversation[0].ctor_args == (chachanotes_db,)
    assert records.collections_db[0].ctor_args == (
        tmp_path / "collections.db",
        CLI_APP_CLIENT_ID,
    )
    assert records.collections[0].ctor_args == (records.collections_db[0],)

    assert service._media is records.media[0]
    assert service._notes is notes
    assert service._prompts is records.prompt[0]
    assert service._skills is records.skills[0]
    assert service._conversations is records.conversation[0]
    assert service._collections is records.collections[0]


def test_factory_degrades_a_failing_backend_to_feature_unavailable(
    monkeypatch, tmp_path
):
    import tldw_chatbook.MCP.server as server_module

    records = _patch_factory_backends(monkeypatch, tmp_path, raising={"skills"})

    service = server_module.build_local_library_tool_service(
        chachanotes_db=object(), media_db=object()
    )

    assert service._skills is None
    assert service._media is records.media[0]
    payload = service.invoke("library_list_skills", {})
    assert payload["error"]["code"] == "feature_unavailable"


def test_factory_reuses_a_caller_supplied_notes_service(monkeypatch, tmp_path):
    import tldw_chatbook.MCP.server as server_module

    records = _patch_factory_backends(monkeypatch, tmp_path)
    supplied_notes = object()

    service = server_module.build_local_library_tool_service(
        chachanotes_db=object(), media_db=object(), notes_service=supplied_notes
    )

    assert records.notes == []  # not rebuilt
    assert service._notes is supplied_notes


def test_factory_wires_the_policy_enforcer_into_the_chunk_tool_service(
    monkeypatch, tmp_path
):
    """Task 5 (spec §6): the MCP construction site passes the runtime-policy
    enforcer into the chunk tool service, so the writing chunk tools
    (`library_save_chunk_spec`, `library_rechunk_media`) are service-level
    gated on the local MCP surface -- not only under the Console."""
    import tldw_chatbook.Library.local_media_chunk_tool_service as chunk_module
    import tldw_chatbook.MCP.server as server_module

    _patch_factory_backends(monkeypatch, tmp_path)
    built = []
    real_ctor = chunk_module.LocalMediaChunkToolService

    class _RecordingChunkService(real_ctor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(chunk_module, "LocalMediaChunkToolService", _RecordingChunkService)
    enforcer = object()

    service = server_module.build_local_library_tool_service(
        chachanotes_db=object(), media_db=object(), policy_enforcer=enforcer
    )

    assert service._media_chunk is built[0]
    assert built[0]._policy_enforcer is enforcer
