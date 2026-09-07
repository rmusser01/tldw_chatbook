# Direct Local Library Agent and MCP Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Console agents and local MCP clients safe, read-only list, get, and literal lexical-search tools for Media, Notes, Prompts, Skills, Conversations, and Collections, with exact totals, stable IDs, byte-bounded continuation, and no semantic/RAG dependency in the direct tools.

**Architecture:** Add one descriptor-backed synchronous `LocalLibraryToolService` over additive, text-only query methods in the six existing local Library services. Console exposes that service through a native `LibraryToolProvider` when `[console].direct_library_tools` is enabled and exposes a bounded Library RAG provider when it is disabled. FastMCP and the direct local MCP runtime derive the same 18 schemas and route calls to the same service with `asyncio.to_thread`; Console-specific MCP composition filters built-in overlaps so the setting cannot be bypassed.

**Tech Stack:** Python 3.11+, Textual, SQLite/FTS5, descriptor JSON schemas plus explicit runtime boundary validation, `hashlib`/`base64`/`json` from the standard library, pytest with temporary SQLite databases and managed temporary Skill directories.

**Spec:** `Docs/superpowers/specs/2026-08-02-local-library-agent-tools-design.md` (source of truth; read before implementation).

**Backlog:** `backlog/tasks/task-1337 - Add-direct-local-Library-tools-for-Console-agents-and-MCP.md`

**ADR required:** yes

**ADR path:** `backlog/decisions/030-local-library-agent-tool-boundary.md`

**Reason:** This introduces a durable cross-module Console/MCP service contract, a privacy-sensitive runtime boundary between direct reads and RAG, and stable-ID/continuation semantics.

**Worktree:** Execute in a dedicated git worktree using `superpowers:using-git-worktrees`. The current worktree contains unrelated user changes, including overlapping edits in `ChaChaNotes_DB.py`, Console settings/storage tests, and CSS. Do not edit, revert, stage, or copy those changes. Build from the current committed branch and resolve integration conflicts explicitly at handoff.

**Compatibility:** No schema migration. Existing unnamespaced MCP tools and their payloads remain unchanged. No direct tool may select binary columns, return filesystem paths, or call RAG/vector/embedding code.

---

### Task 0: Create the isolated implementation worktree and baseline

**Files:**
- Verify: `backlog/tasks/task-1337 - Add-direct-local-Library-tools-for-Console-agents-and-MCP.md`
- Verify: `backlog/decisions/030-local-library-agent-tool-boundary.md`
- Verify: `Docs/superpowers/specs/2026-08-02-local-library-agent-tools-design.md`
- Verify: `Docs/superpowers/plans/2026-08-02-local-library-agent-tools.md`

- [ ] **Step 1: Create a dedicated worktree**

Use `superpowers:using-git-worktrees` and create a `codex/`-prefixed feature branch from the commit containing this plan, ADR, task, and spec. Do not use the dirty primary worktree for implementation.

- [ ] **Step 2: Confirm governance links and task state**

Run:

```bash
backlog task 1337 --plain
rg -n "ADR-030|TASK-1337|2026-08-02-local-library-agent-tools" \
  backlog/decisions/030-local-library-agent-tool-boundary.md \
  Docs/superpowers/specs/2026-08-02-local-library-agent-tools-design.md \
  Docs/superpowers/plans/2026-08-02-local-library-agent-tools.md
```

Expected: TASK-1337 is `In Progress`; the task, ADR, spec, and plan link to one another.

- [ ] **Step 3: Capture the focused baseline**

Run:

```bash
python3 -m pytest \
  Tests/Agents/test_tool_catalog.py \
  Tests/Agents/test_mcp_tool_provider.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/MCP/test_hub_tool_catalog.py \
  Tests/MCP/test_local_control_service.py \
  Tests/UI/test_settings_library_rag_defaults.py \
  -q
```

Expected: PASS, or record any pre-existing failure before changing code. Do not treat optional ML/import aborts as feature failures; report them separately.

- [ ] **Step 4: Commit only if the execution worktree needed a governance-link correction**

```bash
git add backlog/decisions/030-local-library-agent-tool-boundary.md \
  "backlog/tasks/task-1337 - Add-direct-local-Library-tools-for-Console-agents-and-MCP.md" \
  Docs/superpowers/specs/2026-08-02-local-library-agent-tools-design.md \
  Docs/superpowers/plans/2026-08-02-local-library-agent-tools.md
git commit -m "docs: link local Library tools plan and ADR"
```

Skip the commit if there is no change.

---

### Task 1: Shared descriptors, IDs, cursors, validation, and byte fitting

**Files:**
- Create: `tldw_chatbook/Library/library_tool_contract.py`
- Create: `Tests/Library/test_library_tool_contract.py`
- Modify: `tldw_chatbook/Library/__init__.py`

- [ ] **Step 1: Write failing descriptor and schema tests**

Create `Tests/Library/test_library_tool_contract.py` with a canonical expected-name set and assertions that every descriptor has a unique name, route, description, and bounded input schema:

```python
EXPECTED_LIBRARY_TOOLS = {
    "library_list_media", "library_get_media", "library_search_media",
    "library_list_notes", "library_get_note", "library_search_notes",
    "library_list_prompts", "library_get_prompt", "library_search_prompts",
    "library_list_skills", "library_get_skill", "library_search_skills",
    "library_list_conversations", "library_get_conversation", "library_search_conversations",
    "library_list_collections", "library_get_collection", "library_search_collections",
}


def test_descriptor_table_has_exact_canonical_surface():
    assert set(LIBRARY_TOOL_DESCRIPTORS) == EXPECTED_LIBRARY_TOOLS
    assert len({d.route for d in LIBRARY_TOOL_DESCRIPTORS.values()}) == 18


def test_list_and_search_schemas_bound_pagination():
    for descriptor in LIBRARY_TOOL_DESCRIPTORS.values():
        props = descriptor.input_schema["properties"]
        if descriptor.operation in {"list", "search"}:
            assert props["limit"]["default"] == 20
            assert props["limit"]["maximum"] == 50
            assert props["offset"]["minimum"] == 0
        if descriptor.operation == "search":
            assert "query" in descriptor.input_schema["required"]
```

Also assert get schemas require `id`, never accept title/name/raw row IDs, cap `max_chars` at 16,000, and expose only the type-specific section/message/member arguments in the spec.

- [ ] **Step 2: Run the contract tests to verify they fail**

Run: `python3 -m pytest Tests/Library/test_library_tool_contract.py -x -q`

Expected: FAIL because `library_tool_contract.py` does not exist.

- [ ] **Step 3: Implement the descriptor table and public constants**

In `library_tool_contract.py`, define immutable `LibraryToolDescriptor` records plus:

```python
LIBRARY_ITEM_TYPES = ("media", "note", "prompt", "skill", "conversation", "collection")
DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 50
DEFAULT_MAX_CHARS = 8_000
MAX_MAX_CHARS = 16_000
MAX_RESULT_BYTES = 32 * 1024
PAGE_MANDATORY_RESERVE_BYTES = 24 * 1024
MAX_PUBLIC_ID_BYTES = 128
```

Descriptions must say returned text is untrusted Library data, search is literal/lexical, and the operation is read-only. Use the descriptor table as the only source of public tool names, descriptions, schemas, item type, operation, and service route.

- [ ] **Step 4: Add failing stable-ID and cursor tests**

Cover:

- round-trip IDs for all six prefixes;
- ASCII-only output and the 128-byte ceiling;
- malformed base64, wrong type, empty/raw path-like backing IDs, and oversized IDs;
- cursor round-trip for item ID, section, offset, revision, message/file token state;
- one-byte cursor mutation returning `invalid_argument` rather than decoding;
- revision mismatch mapping to `content_changed` with a fresh-start hint.

Use public helpers rather than inspecting private payload layout.

- [ ] **Step 5: Implement fail-closed codecs and structured errors**

Use URL-safe base64 without padding over a versioned UTF-8 payload and include a SHA-256 checksum over the canonical payload for cursor tamper detection. Public IDs use `type:<base64url(raw identity)>`; reject unexpected prefixes and any encoded result over 128 bytes. Cursors bind item ID, section/file/message state, character offset, and revision.

Define JSON-safe errors with only:

```python
{"error": {"code": code, "message": message, "retryable": retryable, "details": bounded_details}}
```

Allow only the spec codes: `invalid_argument`, `not_found`, `content_changed`, `index_unavailable`, `feature_unavailable`, and `storage_error`. Never serialize exception `repr`, SQL, secrets, or paths.

- [ ] **Step 6: Add failing serializer-bound tests**

Test `json.dumps(..., ensure_ascii=False, separators=(",", ":")).encode("utf-8")` length, not Python character length. Required cases:

- 50 rows with multibyte and JSON-escaped titles;
- preservation of item count, ID, type, display title/name, and exact keyword counts;
- deterministic title/name shortening no lower than the 32-byte display floor;
- optional trimming order: extra keyword values, preview, optional metadata;
- `response_truncated` and stable `omitted_fields` paths;
- get text containing emoji, quotes, backslashes, and newlines where byte fitting shortens below requested `max_chars` without skipping/repeating continuation characters.

- [ ] **Step 7: Implement normalization and byte fitting**

Implement small pure helpers:

```python
normalize_display_text(value, *, max_bytes=160, floor_bytes=32)
validate_page_args(limit, offset)
validate_search_query(query)
fit_page_payload(payload) -> dict[str, Any]
fit_text_segment(payload, canonical_text, requested_end) -> dict[str, Any]
serialized_size(payload) -> int
```

Normalize control characters to spaces, cut only at UTF-8/Unicode boundaries, and use binary search for the largest get-text character prefix whose complete JSON response is `< MAX_RESULT_BYTES`. Page fitting must retain the requested page's rows and all mandatory fields.

- [ ] **Step 8: Run tests and commit**

```bash
python3 -m pytest Tests/Library/test_library_tool_contract.py -q
git add tldw_chatbook/Library/library_tool_contract.py \
  tldw_chatbook/Library/__init__.py \
  Tests/Library/test_library_tool_contract.py
git commit -m "feat: define local Library tool contracts"
```

---

### Task 2: Add exact, text-only Media and Notes query seams

**Files:**
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py`
- Modify: `tldw_chatbook/Media/local_media_reading_service.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/Notes/Notes_Library.py`
- Modify: `tldw_chatbook/Notes/notes_scope_service.py`
- Modify: `Tests/Media/test_local_media_reading_service.py`
- Modify: `Tests/Notes/test_notes_library_unit.py`
- Modify: `Tests/Notes/test_notes_scope_service.py`

- [ ] **Step 1: Write failing Media query tests**

Add tests that seed titles, content, duplicate keyword matches, soft-deleted/trashed rows, large offsets, and a non-empty `vector_embedding`. Exercise new additive methods:

```python
page = service.list_library_media(limit=2, offset=1)
matches = service.search_library_media(query='100%_literal "OR"', limit=2, offset=0)
detail = service.get_library_media_text(media_uuid, start=0, max_chars=100)
```

Assert exact distinct totals, deterministic ID tie-break ordering, title/content/keyword `matched_fields`, literal `%`/`_`/quotes/FTS operators, and that the detail result has no embedding, binary value, or path. Use a SQLite trace callback or a narrow fake cursor assertion to prove the detail projection does not select `vector_embedding`; verify a large offset is expressed as SQL `LIMIT/OFFSET`, not prefix materialization.

- [ ] **Step 2: Run Media tests to verify failure**

Run: `python3 -m pytest Tests/Media/test_local_media_reading_service.py -x -q`

Expected: FAIL because the additive Library methods do not exist.

- [ ] **Step 3: Implement Media page/search/detail projection**

Add DB/service methods named consistently with the tests. Use one SQLite read transaction for `COUNT(DISTINCT Media.id)` and the final deduplicated page. Combine parameterized branches for exact/substring title, safe internally generated FTS tokens over title/content, and keyword relation matches. Escape `LIKE` wildcard characters with an explicit escape clause. Return UUID/version and only contract text/metadata columns; do not call semantic/RAG helpers. Detail SQL must return `length(content)`, a caller-bounded `substr(content, start, max_chars + 1)`, and reliable revision metadata rather than materializing the full Media body.

- [ ] **Step 4: Write failing Notes query tests**

Seed Notes that match title, body, multiple keywords, or more than one branch. Assert:

- local `NotesScopeService.list_notes(limit=..., offset=...)` forwards offset;
- exact totals precede slicing;
- a Note matched through two branches appears once;
- UUID/version are returned;
- literal `%`, `_`, quotes, `OR`, and `NEAR` do not broaden the query;
- deleted notes are excluded;
- detail returns only text and safe metadata.

- [ ] **Step 5: Run Notes tests to verify failure**

```bash
python3 -m pytest \
  Tests/Notes/test_notes_library_unit.py \
  Tests/Notes/test_notes_scope_service.py \
  -x -q
```

Expected: FAIL on missing exact-count search/offset behavior.

- [ ] **Step 6: Implement Notes page/search/detail projection**

Add a DB-level `search_library_notes_page(query, limit, offset)` returning `{items, total}` from one read transaction. Safely OR title/content FTS or escaped substring branches with keyword links and deduplicate before count/page. Fix the existing local scope adapter to forward `offset`. Add a text-detail projection that returns `length(content)`, a bounded `substr`, and the Note version without loading the full body. Preserve existing public Notes methods; the new Library seam is additive.

Before editing `ChaChaNotes_DB.py`, inspect the implementation worktree's version and the dirty primary worktree diff. Do not copy or revert the user's uncommitted DB changes; keep this feature's changes localized to new methods/query helpers so later integration can resolve cleanly.

- [ ] **Step 7: Run focused tests and commit**

```bash
python3 -m pytest \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Notes/test_notes_library_unit.py \
  Tests/Notes/test_notes_scope_service.py \
  -q
git add tldw_chatbook/DB/Client_Media_DB_v2.py \
  tldw_chatbook/Media/local_media_reading_service.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  tldw_chatbook/Notes/Notes_Library.py \
  tldw_chatbook/Notes/notes_scope_service.py \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Notes/test_notes_library_unit.py \
  Tests/Notes/test_notes_scope_service.py
git commit -m "feat: add exact Media and Notes Library queries"
```

---

### Task 3: Add exact Prompt queries and bounded, trust-aware Skill reads

**Files:**
- Modify: `tldw_chatbook/DB/Prompts_DB.py`
- Modify: `tldw_chatbook/Prompt_Management/Prompts_Interop.py`
- Modify: `tldw_chatbook/Prompt_Management/local_prompt_service.py`
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py`
- Modify: `Tests/Prompts_DB/test_prompts_db_pytest.py`
- Modify: `Tests/Prompt_Management/test_local_prompt_service.py`
- Modify: `Tests/Skills/test_local_skills_service.py`

- [ ] **Step 1: Write failing Prompt tests**

Seed active/deleted prompts whose query occurs in name, details, system prompt, user prompt, JSON prompt definition, or keywords. Assert the new page method preserves `total`, `search_fields`, UUID, version, and exact keyword matches through the interop/local adapter. Add get tests for:

- default bounded overview plus manifest;
- `section` values `details`, `system_prompt`, `user_prompt`, `prompt_definition`;
- no version-history expansion;
- structured definition continuation;
- invalid section rejection.

- [ ] **Step 2: Run Prompt tests to verify failure**

```bash
python3 -m pytest \
  Tests/Prompts_DB/test_prompts_db_pytest.py \
  Tests/Prompt_Management/test_local_prompt_service.py \
  -x -q
```

- [ ] **Step 3: Implement Prompt exact search and section projection**

Add an exact-count DB method using parameterized, case-insensitive literal matching over all specified text fields and `keywords`, with exact-name precedence and deterministic tie-breaking. Keep compiled system/user text and the structured definition as separate canonical sections. Each selected detail section returns its total character length, one bounded SQL substring, and the Prompt version; the default overview independently bounds every included section. Make `Prompts_Interop` and `LocalPromptService` forward totals and match fields rather than discarding them. Do not load complete unselected sections or version-history rows in Library detail.

- [ ] **Step 4: Write failing Skill tests**

Use `tmp_path` managed Skill roots with:

- trusted and blocked skills;
- mixed-case name/description/body/metadata keyword matches;
- a large `SKILL.md` and multiple supporting files;
- a supporting-file manifest and opaque file token;
- a file-content revision change between continuation calls;
- an attempted arbitrary/path-traversal token.

Assert one managed enumeration produces exact total before slicing. Blocked skills may match and return safe name/description/trust status, but never body, supporting-file snippets, or content previews. Prove list/search does not call the existing eager `_read_supporting_files` path.

- [ ] **Step 5: Run Skill tests to verify failure**

Run: `python3 -m pytest Tests/Skills/test_local_skills_service.py -x -q`

- [ ] **Step 6: Implement bounded Skill scan and read methods**

Add new Library-specific methods that enumerate managed records once, casefold-match name/description/body/metadata keywords, and then slice. Reuse the existing validated-root and trust checks. For detail, return a bounded supporting-file manifest with opaque file tokens; only read the selected main/supporting file. Bind continuation revisions to the selected file's SHA-256 content hash. Never accept a caller path and never use the eager all-supporting-file response builder.

- [ ] **Step 7: Run focused tests and commit**

```bash
python3 -m pytest \
  Tests/Prompts_DB/test_prompts_db_pytest.py \
  Tests/Prompt_Management/test_local_prompt_service.py \
  Tests/Skills/test_local_skills_service.py \
  -q
git add tldw_chatbook/DB/Prompts_DB.py \
  tldw_chatbook/Prompt_Management/Prompts_Interop.py \
  tldw_chatbook/Prompt_Management/local_prompt_service.py \
  tldw_chatbook/Skills_Interop/local_skills_service.py \
  Tests/Prompts_DB/test_prompts_db_pytest.py \
  Tests/Prompt_Management/test_local_prompt_service.py \
  Tests/Skills/test_local_skills_service.py
git commit -m "feat: add Prompt and Skill Library read seams"
```

---

### Task 4: Add exact Conversation and Collection queries with bounded child pages

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py`
- Modify: `tldw_chatbook/Library/library_collections_service.py`
- Modify: `Tests/DB/test_search_conversations_fts.py`
- Modify: `Tests/Chat/test_chat_conversation_service.py`
- Modify: `Tests/Library/test_library_collections_service.py`

- [ ] **Step 1: Write failing Conversation search/detail tests**

Seed title matches, message-body matches, keyword matches, duplicate multi-message matches, RAG-context messages, an image BLOB, and one message larger than 32 KiB. Assert:

- exact distinct conversation total before paging;
- UUID-based stable backing identity and deterministic order;
- title/message/keyword `matched_fields`;
- `include_rag_context=False` always;
- exact `message_total` and actual `returned_message_count`;
- structured message offset/stable message ID/within-message continuation;
- no `image_data` in SQL projection or output;
- message revision/hash invalidates stale continuation.

- [ ] **Step 2: Run Conversation tests to verify failure**

```bash
python3 -m pytest \
  Tests/DB/test_search_conversations_fts.py \
  Tests/Chat/test_chat_conversation_service.py \
  -x -q
```

- [ ] **Step 3: Implement Conversation exact search and text-only message page**

Add additive DB/service methods. Search must deduplicate conversation UUIDs across title, message FTS/literal, and keyword branches in one transaction. Message detail must select explicit text/metadata columns, never `SELECT *`, never `image_data`, and exclude RAG-context messages in SQL. Return a caller-bounded text substring plus total character length for each included message; never materialize all preceding rows or the complete body of a long message. Use stored message version only if it changes with content; otherwise compute a deterministic content hash through the narrowest existing DB seam. Preserve existing Conversation service APIs.

- [ ] **Step 4: Write failing Collection tests**

Seed Collections where query matches name, description, multiple direct member titles, and unsupported member types. Assert:

- exact distinct total and SQL-backed offset page;
- search does not inspect member content;
- exact `member_total`, deterministic membership page, `membership_id`, source type, and bounded title;
- supported Media/Note/Prompt/Skill/Conversation/Collection source records receive corresponding type-prefixed `item_id` values that round-trip through the ID codec;
- unsupported sources return `item_id=None` plus an opaque reference;
- member content is never inlined.

- [ ] **Step 5: Run Collection tests to verify failure**

Run: `python3 -m pytest Tests/Library/test_library_collections_service.py -x -q`

- [ ] **Step 6: Implement Collection exact search and membership page**

Extend `LibraryCollectionsService` and its local implementation with list/search/count/member-page operations. Count and page within one read transaction, deduplicate Collections before pagination, and restrict search to Collection name/description/direct stored member title. Map supported member source identities through the shared public-ID codec; never resolve or inline member content.

- [ ] **Step 7: Run focused tests and commit**

```bash
python3 -m pytest \
  Tests/DB/test_search_conversations_fts.py \
  Tests/Chat/test_chat_conversation_service.py \
  Tests/Library/test_library_collections_service.py \
  -q
git add tldw_chatbook/DB/ChaChaNotes_DB.py \
  tldw_chatbook/Chat/chat_conversation_service.py \
  tldw_chatbook/Library/library_collections_service.py \
  Tests/DB/test_search_conversations_fts.py \
  Tests/Chat/test_chat_conversation_service.py \
  Tests/Library/test_library_collections_service.py
git commit -m "feat: add Conversation and Collection Library queries"
```

---

### Task 5: Implement the shared 18-operation `LocalLibraryToolService`

**Files:**
- Create: `tldw_chatbook/Library/local_library_tool_service.py`
- Create: `Tests/Library/test_local_library_tool_service.py`
- Modify: `tldw_chatbook/Library/__init__.py`

- [ ] **Step 1: Write failing service dispatch and parity tests**

Build fakes for all six backend services and parametrize across every descriptor. Assert:

```python
result = service.invoke("library_list_notes", {"limit": 2, "offset": 0})
assert result == {
    "items": ANY,
    "total": 3,
    "limit": 2,
    "offset": 0,
    "has_more": True,
    "next_offset": 2,
    "response_truncated": False,
    "omitted_fields": [],
}
```

Cover operation-to-backend routing, normalized type-specific briefs, stable ID encoding, exact keyword counts, match evidence, empty terminal pages, and all required error codes. Assert no method name or fake call contains `rag`, `embedding`, `vector`, or semantic-search routing for the 18 descriptors.

- [ ] **Step 2: Run service tests to verify failure**

Run: `python3 -m pytest Tests/Library/test_local_library_tool_service.py -x -q`

- [ ] **Step 3: Implement constructor, descriptor dispatch, and validation**

Use explicit dependencies:

```python
class LocalLibraryToolService:
    def __init__(
        self,
        *,
        media_service,
        notes_service,
        prompt_service,
        skills_service,
        conversation_service,
        collections_service,
    ) -> None: ...

    def invoke(self, tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]: ...
```

Resolve only names in `LIBRARY_TOOL_DESCRIPTORS`; repeat numeric/type/query/ID/cursor validation at runtime. Catch expected backend absence as `feature_unavailable`, map missing records to `not_found`, SQLite/filesystem operational failures to a scrubbed `storage_error`, and never catch `BaseException`.

For local services whose current method is declared async but performs local work, bridge only inside the synchronous core/worker context. Keep this module free of Textual, MCP, and agent imports.

- [ ] **Step 4: Implement common list/search normalization**

Normalize backend pages into the common envelope, bound titles/names/keywords/previews, preserve exact totals, derive `has_more`/`next_offset` from actual returned count, and call `fit_page_payload` before returning. Require every backend brief to supply a real stable backing ID; do not fabricate IDs from row position or title.

- [ ] **Step 5: Implement get and continuation normalization**

Decode the expected type prefix before backend access. For text types, validate cursor identity/section/revision, request only a bounded backend slice plus `total_chars`/revision, and apply `fit_text_segment` to that slice; never fetch an unbounded body merely to truncate it in the serializer. Implement the Prompt manifest/section, Skill manifest/file token, Conversation message page/within-message, and Collection membership shapes from the spec. Every successful or error result must serialize below 32 KiB.

- [ ] **Step 6: Add cross-backend integration cases**

Using temporary real databases where practical, assert all six list/search/get round trips, wrong-type IDs, malformed IDs, not-found IDs, keyword matches, content-change continuation, and byte ceilings. Explicitly assert output contains no path key/value, bytes, embedding, vector, or image data.

- [ ] **Step 7: Run tests and commit**

```bash
python3 -m pytest \
  Tests/Library/test_library_tool_contract.py \
  Tests/Library/test_local_library_tool_service.py \
  -q
git add tldw_chatbook/Library/local_library_tool_service.py \
  tldw_chatbook/Library/__init__.py \
  Tests/Library/test_local_library_tool_service.py
git commit -m "feat: implement shared local Library tool service"
```

---

### Task 6: Add Console direct-Library and fallback-RAG providers

**Files:**
- Create: `tldw_chatbook/Agents/library_tool_provider.py`
- Create: `tldw_chatbook/Agents/library_rag_tool_provider.py`
- Create: `Tests/Agents/test_library_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`

- [ ] **Step 1: Write failing provider tests**

Assert `LibraryToolProvider` derives all 18 catalog entries and full schemas from descriptors, invokes the synchronous service, JSON-serializes success into `ToolResult.content`, and JSON-serializes the same structured error object into `ToolResult.error` with `ok=False`. Assert tool IDs use a separate source such as `library:<name>`.

Assert `LibraryRagToolProvider` exposes exactly `search_library_rag`, accepts a required query and optional bounded `top_k`/supported source types, limits scope to Notes/Media/Conversations, and maps unavailable/setup conditions to `index_unavailable` without enabling direct tools.

- [ ] **Step 2: Run provider tests to verify failure**

Run: `python3 -m pytest Tests/Agents/test_library_tool_provider.py -x -q`

- [ ] **Step 3: Implement the providers**

Keep both providers synchronous to satisfy `ToolProvider`; they run on the agent worker thread. `LibraryToolProvider` delegates to `LocalLibraryToolService`. `LibraryRagToolProvider` calls the existing app-owned Library RAG search service through a bounded adapter, caps excerpts/results so the final JSON is below 32 KiB, and never falls back to direct lexical reads.

- [ ] **Step 4: Write failing Console registry-order and inheritance tests**

Update `_compose_run_registry_and_allowed` tests to cover:

- enabled order: built-ins, Library, eligible Skills, eligible MCP, spawn;
- disabled order: built-ins, one RAG tool, eligible Skills, eligible MCP, spawn;
- collisions are won in that order;
- ordinary subagents inherit the parent's allowed tool set under existing rules;
- `_BridgeSkillRunner` receives only original built-in names, not Library/RAG/Skill/MCP names;
- per-run composition refreshes the selected retrieval provider without reconstructing unrelated state.

- [ ] **Step 5: Implement run-scoped registry composition**

Extend `_compose_run_registry_and_allowed` and `ConsoleAgentBridge.run_reply` with a single already-constructed `library_provider` argument. Register it after `BuiltinToolProvider` and before Skills. Keep `builtin_names` captured before Library registration. Rebuild the per-run registry whenever Skills, MCP, or a Library/RAG provider is present. Do not alter skill-triggered child narrowing.

Add a `library_provider_factory: Callable[[], ToolProvider | None]` seam to `ConsoleChatController`. In `_run_agent_reply`, call the factory once on the main loop for that run, then pass its result alongside the composed MCP provider into the bridge's `asyncio.to_thread` call. This is the actual run handoff seam; do not read Textual config from inside the bridge worker.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m pytest \
  Tests/Agents/test_library_tool_provider.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/Agents/test_tool_catalog.py \
  -q
git add tldw_chatbook/Agents/library_tool_provider.py \
  tldw_chatbook/Agents/library_rag_tool_provider.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  Tests/Agents/test_library_tool_provider.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/Chat/test_console_chat_controller.py
git commit -m "feat: expose local Library retrieval to Console agents"
```

---

### Task 7: Persist and surface the Console retrieval-mode setting

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_library_rag_defaults.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `Tests/UI/test_settings_library_rag_defaults.py`
- Create: `Tests/UI/test_console_library_tool_setting.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`

- [ ] **Step 1: Write failing settings model/persistence tests**

Assert `SettingsLibraryRagDefaults.direct_library_tools`:

- defaults to `True` when `[console]` or the key is missing/malformed;
- reads true/false string forms through existing coercion;
- deep-merges `{"console": {"direct_library_tools": value}}` without dropping unrelated Console or RAG settings;
- remains global application config and is absent from conversation/session settings serialization.

- [ ] **Step 2: Run settings tests to verify failure**

Run: `python3 -m pytest Tests/UI/test_settings_library_rag_defaults.py -x -q`

- [ ] **Step 3: Implement the config model and save-section merge**

Add the boolean to `SettingsLibraryRagDefaults`, load it from the top-level `[console]` section, and make `build_library_rag_save_sections` return deep-merged updates for both `AppRAGSearchConfig` and `console`. Do not modify `console_session_settings.py`; this setting is not session-scoped.

- [ ] **Step 4: Write failing visible-copy and bridge-injection tests**

Add UI/compose tests that find the toggle and exact visible concepts from the approved copy:

- On permits automatic local list/count/read/lexical search;
- Off removes direct list/count/view/search and makes Library RAG default;
- RAG covers Notes, Media, and Conversations and needs an index;
- cloud models receive retrieved Library data off-device;
- use a local model to keep data on-device;
- setting affects Console only and MCP is controlled separately.

Use the complete approved wording in spec Section 8 for the rendered control; do not paraphrase it down to these test keywords.

Add `ChatScreen` tests with fake app-owned services. Flip config between consecutive runs and assert the next run receives `LibraryToolProvider` vs. `LibraryRagToolProvider` without recreating the cached bridge. Assert the direct service is assembled only from local app service attributes and missing one backend yields per-tool `feature_unavailable`, not total bridge failure.

- [ ] **Step 5: Implement the Settings UI and Console injection**

Add a `Use direct Library tools` switch to the existing Settings > Library/RAG detail, with the full privacy and scope warning rendered below it, not in a tooltip. Wire draft/revert/validation/save behavior through existing Settings patterns; no new CSS is required unless an existing reusable class cannot express the layout.

In `chat_screen.py`, add a provider-factory callback that reads `[console].direct_library_tools` fresh for each Console run and constructs the appropriate provider. Direct mode assembles `LocalLibraryToolService` from `local_media_reading_service`, `notes_service` (or the local Notes scope seam), `local_prompt_service`, `local_skills_service`, `local_chat_conversation_service`, and `local_library_collections_service`. Off mode constructs the bounded RAG provider over `library_rag_search_service`. Inject that callback into the cached `ConsoleChatController`; the controller resolves it once per run and passes the provider to `ConsoleAgentBridge.run_reply`. Changing the toggle therefore applies to the next run without rebuilding the controller or bridge.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m pytest \
  Tests/UI/test_settings_library_rag_defaults.py \
  Tests/UI/test_console_library_tool_setting.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/Chat/test_console_chat_controller.py \
  -q
git add tldw_chatbook/UI/Screens/settings_library_rag_defaults.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  Tests/UI/test_settings_library_rag_defaults.py \
  Tests/UI/test_console_library_tool_setting.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/Chat/test_console_chat_controller.py
git commit -m "feat: add Console Library retrieval mode setting"
```

---

### Task 8: Prevent Console MCP bypass and preserve built-in schemas

**Files:**
- Modify: `tldw_chatbook/Agents/mcp_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/MCP/hub_tool_catalog.py`
- Modify: `Tests/Agents/test_mcp_tool_provider.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/MCP/test_hub_tool_catalog.py`

- [ ] **Step 1: Write failing source-aware exclusion tests**

Build a mixed inventory containing:

- all 18 raw descriptor names from `builtin:tldw_chatbook`;
- legacy `search_rag`, `search_notes`, `search_conversations`, `get_conversation_history`, `export_conversation` from the built-in source;
- an unrelated built-in tool;
- external/local MCP profiles with identical raw names.

Assert the Console-composed provider excludes only the 23 built-in raw names in both direct and RAG modes. External/local profile tools remain eligible and governed. Assert the local MCP inventory itself remains unchanged.

- [ ] **Step 2: Write failing built-in schema preservation tests**

Change the current expectation in `test_builtin_tools_have_no_schema_but_execute`: when inventory supplies `inputSchema`, `builtin_tools_from_inventory` must preserve the non-empty mapping; legacy entries with no schema still yield `None`.

- [ ] **Step 3: Run tests to verify failure**

```bash
python3 -m pytest \
  Tests/Agents/test_mcp_tool_provider.py \
  Tests/MCP/test_hub_tool_catalog.py \
  -x -q
```

- [ ] **Step 4: Implement Console-only raw-name filtering**

Add an optional immutable `builtin_raw_name_exclusions` constructor argument to `MCPToolProvider`, applied during `compose_catalog` only when `tool.server_key == "builtin:tldw_chatbook"`. In `console_chat_controller.py`, build the Console exclusion set from descriptor names plus the five explicit legacy names and pass it when `_compose_mcp_provider` constructs the per-run MCP provider. Do not put legacy names into the shared descriptor table. Default to no exclusions so all non-Console callers preserve current behavior.

- [ ] **Step 5: Preserve inventory input schemas**

Update `builtin_tools_from_inventory` to defensively copy a non-empty `inputSchema` mapping into `HubTool.input_schema`. Do not synthesize a schema for legacy AST entries lacking one.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m pytest \
  Tests/Agents/test_mcp_tool_provider.py \
  Tests/Chat/test_console_chat_controller.py \
  Tests/MCP/test_hub_tool_catalog.py \
  -q
git add tldw_chatbook/Agents/mcp_tool_provider.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/MCP/hub_tool_catalog.py \
  Tests/Agents/test_mcp_tool_provider.py \
  Tests/Chat/test_console_chat_controller.py \
  Tests/MCP/test_hub_tool_catalog.py
git commit -m "fix: prevent Console MCP Library tool bypass"
```

---

### Task 9: Register and execute the 18 tools through local MCP

**Files:**
- Modify: `tldw_chatbook/MCP/server.py`
- Modify: `tldw_chatbook/MCP/local_runtime_delegate.py`
- Modify: `tldw_chatbook/MCP/local_control_service.py`
- Modify: `tldw_chatbook/runtime_policy/registry.py`
- Create: `Tests/MCP/test_library_tools.py`
- Modify: `Tests/MCP/test_local_control_service.py`
- Modify: `Tests/MCP/test_builtin_tool_imports.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_core.py`

- [ ] **Step 1: Write failing manifest and direct-delegate tests**

Assert `describe_local_mcp_capabilities()` retains all AST-derived legacy tools and appends exactly 18 descriptor tools with identical names/descriptions/`inputSchema` mappings. Assert `tools/list` exposes those schemas.

Inject a fake synchronous `LocalLibraryToolService` into `LocalMCPRuntimeDelegate`. Parametrize all 18 names and assert `execute_tool`:

- dispatches only descriptor names;
- uses `asyncio.to_thread` rather than blocking the current event loop;
- returns the service's dict unchanged;
- returns/raises the existing unsupported-tool behavior for unknown names;
- is reported as `implemented`, not `missing`, by runtime diagnostics.

- [ ] **Step 2: Write failing policy tests**

Assert each list/get/search tool resolves to a read action owned by its Library type. Reuse the registered list/detail actions for Media, Notes, Prompts, Skills, and Conversations. Add a dedicated local `library.collections` list/detail resource to `CAPABILITY_REGISTRY` rather than mapping Library Collections to `collections.reading_list.*`. Verify both `tool.execute` previews and `tools/call` runtime requests use the same mapping and retain the generic MCP trigger as a secondary policy seam.

- [ ] **Step 3: Write failing standalone bootstrap tests**

Monkeypatch configured path helpers and current service constructors. Instantiate `TldwMCPServer` without opening real user databases and assert it uses `get_chachanotes_db_path`/`get_media_db_path`, constructs all six current local service types with their real signatures, creates one `LocalLibraryToolService`, and keeps legacy `MCPTools`/resources/prompts available. Regression-lock removal of the nonexistent `CharacterInteropService` import and obsolete Notes constructor.

- [ ] **Step 4: Run MCP tests to verify failure**

```bash
python3 -m pytest \
  Tests/MCP/test_library_tools.py \
  Tests/MCP/test_local_control_service.py \
  Tests/MCP/test_builtin_tool_imports.py \
  -x -q
```

- [ ] **Step 5: Implement descriptor-backed manifest and FastMCP registration**

Keep `_describe_local_tools()` for legacy AST entries, then append descriptor-derived entries with `inputSchema`. First inspect the installed FastMCP registration API used by this repository and regression-test its actual `list_tools()` output. Register the 18 handlers through one adapter that calls `await asyncio.to_thread(library_service.invoke, name, arguments)`. Prefer a supported explicit-schema registration hook; if this FastMCP version infers schemas only from callables, generate stable callables with descriptor-derived `__name__`, `__doc__`, annotations, and `inspect.Signature` so the real FastMCP inventory exactly matches each descriptor. Runtime validation remains mandatory. Do not duplicate 18 schema literals or expose one generic `arguments` object in place of the public per-tool schemas.

- [ ] **Step 6: Implement generic direct-runtime delegation and diagnostics**

Inject or lazily construct the shared service in `LocalMCPRuntimeDelegate`. In `execute_tool`, route names in `LIBRARY_TOOL_DESCRIPTORS` through `asyncio.to_thread`; otherwise retain existing `_tool_<name>` dispatch. Update diagnostics so descriptor-dispatched tools count as implemented. Keep legacy handlers byte-for-byte compatible unless the scoped standalone bootstrap fix requires a constructor change.

- [ ] **Step 7: Implement read-policy mapping and current standalone bootstrap**

Derive the 18 policy entries from descriptor item type/operation, mapping list/search to registered list actions and get to registered detail actions. Extend `runtime_policy/registry.py` with the local-only `library.collections` list/detail resource and regression-lock its generated action IDs in `Tests/RuntimePolicy/test_runtime_policy_core.py`; do not reuse the reading-list capability. Correct `_init_databases` to use canonical configured path helpers and real service signatures. Construct the shared service once and pass it to the FastMCP and direct local runtime adapters.

- [ ] **Step 8: Add compatibility assertions**

Call legacy `search_notes`, `search_conversations`, `get_conversation_history`, `export_conversation`, and `search_rag` with existing fixtures and assert their names and payload shapes remain unchanged. Assert the new `[console]` setting has no effect on direct MCP inventory or execution.

- [ ] **Step 9: Run tests and commit**

```bash
python3 -m pytest \
  Tests/MCP/test_library_tools.py \
  Tests/MCP/test_local_control_service.py \
  Tests/MCP/test_builtin_tool_imports.py \
  Tests/MCP/test_hub_tool_catalog.py \
  Tests/RuntimePolicy/test_runtime_policy_core.py \
  -q
git add tldw_chatbook/MCP/server.py \
  tldw_chatbook/MCP/local_runtime_delegate.py \
  tldw_chatbook/MCP/local_control_service.py \
  tldw_chatbook/runtime_policy/registry.py \
  Tests/MCP/test_library_tools.py \
  Tests/MCP/test_local_control_service.py \
  Tests/MCP/test_builtin_tool_imports.py \
  Tests/RuntimePolicy/test_runtime_policy_core.py
git commit -m "feat: expose local Library tools through MCP"
```

---

### Task 10: End-to-end parity, documentation, and completion hygiene

**Files:**
- Create: `Docs/Development/Agent-Tools/local-library-tools.md`
- Modify: `Docs/Design/MCP.md`
- Modify: `Docs/superpowers/specs/2026-08-02-local-library-agent-tools-design.md` only for implementation deviations
- Modify: `backlog/tasks/task-1337 - Add-direct-local-Library-tools-for-Console-agents-and-MCP.md`
- Test: all focused files from Tasks 1–9

- [ ] **Step 1: Add cross-runtime contract parity tests**

For representative list/search/get success and every error code, invoke the same shared service through `LibraryToolProvider` and `LocalMCPRuntimeDelegate`. JSON-decode Console `ToolResult.content`/`error` and assert equality with the MCP dict. Parametrize all 18 input schemas and assert Console and MCP manifest schemas equal descriptor schemas.

- [ ] **Step 2: Add maximum-bound security regressions**

Run real temporary backends with:

- 50 multibyte, control-character, quote/backslash-heavy titles;
- more than 20 long keywords;
- text larger than 32 KiB;
- long Conversation messages with image BLOBs;
- Media rows with vector embeddings and local source paths;
- blocked Skills and many supporting files;
- large SQL offsets.

Assert every serialized result is `< 32 * 1024` UTF-8 bytes, mandatory page rows/IDs survive, continuation does not skip/repeat, and no binary/path material appears. Assert monkeypatched RAG/vector/embedding entry points are never called by the 18 direct tools.

- [ ] **Step 3: Document the public behavior**

Document:

- the 18 exact names grouped by Library type;
- list/search exact-total and stable-ID semantics;
- literal/keyword-only search and explicit no-semantic boundary;
- get chunk/continuation and 32 KiB result ceiling;
- blocked-Skill and binary/path exclusions;
- Console toggle ON/OFF behavior, current RAG scope, and visible cloud privacy warning;
- MCP independence from the Console toggle and legacy MCP compatibility.

- [ ] **Step 4: Run formatter/static checks on touched files**

Use the repository's configured formatter/linter if present. At minimum:

```bash
python3 -m compileall -q tldw_chatbook/Library tldw_chatbook/Agents tldw_chatbook/MCP
git diff --check
```

If `ruff` is configured in `pyproject.toml`, run it only on touched Python files and fix feature-introduced findings.

- [ ] **Step 5: Run the focused verification suite**

```bash
python3 -m pytest \
  Tests/Library/test_library_tool_contract.py \
  Tests/Library/test_local_library_tool_service.py \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Notes/test_notes_library_unit.py \
  Tests/Notes/test_notes_scope_service.py \
  Tests/Prompts_DB/test_prompts_db_pytest.py \
  Tests/Prompt_Management/test_local_prompt_service.py \
  Tests/Skills/test_local_skills_service.py \
  Tests/DB/test_search_conversations_fts.py \
  Tests/Chat/test_chat_conversation_service.py \
  Tests/Library/test_library_collections_service.py \
  Tests/Agents/test_library_tool_provider.py \
  Tests/Agents/test_mcp_tool_provider.py \
  Tests/Agents/test_tool_catalog.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/MCP/test_library_tools.py \
  Tests/MCP/test_hub_tool_catalog.py \
  Tests/MCP/test_local_control_service.py \
  Tests/MCP/test_builtin_tool_imports.py \
  Tests/UI/test_settings_library_rag_defaults.py \
  Tests/UI/test_console_library_tool_setting.py \
  -q
```

- [ ] **Step 6: Run the broadest supported suite**

Run: `python3 -m pytest -q`

If collection aborts due to a known unavailable optional ML dependency (for example MLX/`parakeet_mlx`), rerun the broadest non-optional suite using the repository's existing optional-dependency exclusion mechanism. Record the original abort and the supported-suite result separately; do not misreport either.

- [ ] **Step 7: Self-review against the spec and ADR**

Use `superpowers:requesting-code-review`, fix actionable findings, then use `superpowers:verification-before-completion`. Confirm every acceptance criterion has direct test evidence and no existing MCP contract changed.

- [ ] **Step 8: Update TASK-1337 and mark Done only after the full DoD**

In the task file/CLI:

- check all eight acceptance criteria;
- add concise `## Implementation Notes` covering approach, decisions, modified areas, tests, and any optional-dependency limitation;
- link ADR-030, the spec, this plan, and updated user/MCP documentation;
- set status to Done only after tests, static checks, docs, review, and no-regression checks succeed.

```bash
backlog task edit 1337 -s Done --notes "Implemented the descriptor-backed local Library service, Console direct/RAG mode boundary, and MCP adapters; see ADR-030 and Docs/superpowers/plans/2026-08-02-local-library-agent-tools.md. Focused and supported-suite results are recorded in this task."
```

- [ ] **Step 9: Commit documentation and task completion**

```bash
git add Docs/Development/Agent-Tools/local-library-tools.md \
  Docs/Design/MCP.md \
  "backlog/tasks/task-1337 - Add-direct-local-Library-tools-for-Console-agents-and-MCP.md"
git commit -m "docs: document direct local Library tools"
```

---

## Acceptance-criterion traceability

| TASK-1337 criterion | Primary implementation tasks | Primary verification |
| --- | --- | --- |
| 1. 18 Console + MCP list/get/search tools | 1, 5, 6, 9 | descriptor surface, Console provider, MCP manifest/delegate tests |
| 2. bounded exact pages, stable IDs, keyword matches | 1–5 | backend exact-total tests and serializer-bound tests |
| 3. ID-only get, revision continuation, <32 KiB | 1, 3–5 | codec, prompt/skill/message/member continuation and byte tests |
| 4. no RAG/semantic/binary/path | 2–5, 10 | SQL projection spies and forbidden-call/output assertions |
| 5. default-on toggle, off-mode RAG, privacy copy | 6–7 | settings model/UI and next-run composition tests |
| 6. no Console MCP bypass; MCP compatibility | 8–9 | source-filter and legacy-contract tests |
| 7. automated contract/trust/bootstrap/integration coverage | 1–10 | focused suite plus supported broad suite |
| 8. ADR/design/implementation docs linked | 0, 10 | Backlog/ADR/spec/plan/user-guide link check |
