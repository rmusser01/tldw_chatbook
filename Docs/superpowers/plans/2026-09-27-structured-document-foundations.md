# Structured Document Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve source and language across existing save, recovery, sync, and export authorities before exposing structured editing.
**Architecture:** Extend existing note stores and versioned contracts. Keep file authority, portable note content, and device recovery separate; reject incompatible writes rather than silently dropping metadata.
**Tech Stack:** Existing Python/SQLite/PostgreSQL/FastAPI/Pydantic and TypeScript Notes storage; pytest and Vitest.
**Spec:** [Approved design](../specs/2026-09-27-structured-document-editing-design.md)

## Global Constraints

- All [programme constraints, roots, interfaces, and checks](2026-09-27-structured-document-editing.md) apply.
- "Syntax errors never add a save veto."
- "Run targeted checks only unless the user explicitly authorizes a full suite."
- ADR required: yes. ADR path: `C/backlog/decisions/194-structured-note-language-and-local-editor-validation.md`; server ADR-031 governs existing Notes sync and requires a repository-local extension before new-version implementation.
- Status: Not Started. Units affecting both repositories use separate repo-local Backlog tasks and commits; they are not one cross-repository transaction.

## F1: Repair exact-source transport and structured file admission

**Dependencies:** none. Independently useful lossless transport/file fix.
**Files:** modify C `tldw_chatbook/Sync_Interop/domain_adapters/notes_m1.py`, `tldw_chatbook/Notes/file_notes_service.py`; test C `Tests/Sync_Interop/test_notes_m1_adapters.py`, `Tests/Notes/test_file_notes_service.py`.
**Interfaces:** existing `_validated_note_payload(payload)` returns accepted text unchanged or rejects; `FileNotesService.open_file(relative_path)` continues returning `OpenedFileNote`, preserving representation facts. No new public save API.

- [ ] Add a failing exact-content regression beside the adapter's existing tests:

```python
@pytest.mark.parametrize("body", ['{"value":"a&b <tag>"}\n', "x" * 100_001])
def test_m1_payload_keeps_source(body):
    from tldw_chatbook.Sync_Interop.domain_adapters.notes_m1 import NotesM1SyncAdapter
    payload = NotesM1SyncAdapter._validated_note_payload({"title": "T", "content": body})
    assert payload is not None
    assert payload["content"] == body
```

- [ ] Run `python -m pytest Tests/Sync_Interop/test_notes_m1_adapters.py -q` from C; confirm the new assertions fail for transformation, not test setup.
- [ ] Replace sanitizing/truncating acceptance with typed length/value checks against the negotiated note contract. Align title behavior with server ADR-031 too. Reject over-limit/forbidden content without transforming it; preserve original source for recovery.
- [ ] In the file service, add `.yaml`, `.yml`, `.json`, `.jsonl` to supported extensions and guard frontmatter extraction by source format. Use `Path(relative_path).suffix.lower()`; keep existing Markdown/text compatibility while bypassing stripping for all new structured extensions.
- [ ] Add real temporary-file cases with `---\na: 1\n...\n---\nb: 2\n`, JSON BOM, CRLF, final newline, external hash conflict, and interactive excerpts. Assert full YAML body and exact bytes after a no-op save; keep existing Markdown frontmatter assertions.
- [ ] Run both targeted files, scoped lint, and diff checks. Commit the adapter and file-admission changes separately if their independent reviews warrant it; preserve their corresponding test evidence.

## F2: Canonical note language in each database and ordinary API

**Dependencies:** F1 for exact-source fixtures. Deliver as F2-C and F2-S repo-local changes.
**Files C:** modify `tldw_chatbook/DB/ChaChaNotes_DB.py`, `tldw_chatbook/Notes/Notes_Library.py`, `tldw_chatbook/Notes/notes_scope_service.py`, `tldw_chatbook/tldw_api/notes_workspace_schemas.py`, `tldw_chatbook/Notes/server_notes_workspace_service.py`; create `Tests/ChaChaNotesDB/test_note_content_language.py` and a numbered migration in `tldw_chatbook/DB/migrations/` using the next verified schema transition.
**Files S:** modify `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`, `tldw_Server_API/app/core/DB_Management/chacha/note_store.py`, `tldw_Server_API/app/core/Notes/Notes_Library.py`, `tldw_Server_API/app/api/v1/schemas/notes_schemas.py`, `tldw_Server_API/app/api/v1/endpoints/notes.py`; create `tldw_Server_API/tests/Notes_NEW/integration/test_note_content_language.py`.
**Interfaces:** `add_note(..., content_language: str | None = None)`; existing `update_note(..., update_data, expected_version)` carries the field only when present. Responses expose nullable `content_language`. Known selector values are the five spec modes; bounded unknown stored strings remain round-trippable.

- [ ] Add database tests using each repository's real temporary SQLite fixture:

```python
def test_body_update_preserves_language(db):
    note_id = db.add_note(title="Config", content="a: 1", content_language="yaml")
    before = db.get_note_by_id(note_id)
    db.update_note(note_id, {"content": "a: 2"}, expected_version=before["version"])
    assert db.get_note_by_id(note_id)["content_language"] == "yaml"
```

Use C's `CharactersRAGDB` constructor as shown in existing ChaChaNotes fixtures;
S's fixture uses `CharactersRAGDB(db_path=str(tmp_path / "notes.db"), client_id="test")`.
Also test explicit null, unknown future value, language-only version increments,
stale expected versions, create/get/list/duplicate and migration of existing rows.

- [ ] Run the new selections and establish red. At execution start, record actual SQLite/PostgreSQL schema heads; this planning snapshot observed C SQLite 73, S SQLite 73 and PostgreSQL 77. Allocate the next transitions from the execution base, not these stale numbers.
- [ ] Add nullable TEXT to fresh schemas and migrations; include it in exact note version snapshots and optimistic updates. Do not constrain storage to a closed enum that destroys future values. Keep FTS body/title triggers correct on language-only updates. Verify restore from a current backup retains language and an old backup migrates to null without changing source; portable archives remain F6's boundary.
- [ ] In Pydantic updates use `model_fields_set`/`model_dump(exclude_unset=True)` to distinguish omitted from explicit null; never default an omitted update to null. Add an authenticated `GET /api/v1/notes/capabilities` static route before dynamic note-ID routes, reporting `content_language: true` only with migrated storage and complete write support. A 404/old response means unsupported; the UI does not infer capability from health.
- [ ] Exercise POST, GET, list, PATCH, duplicate, and version conflict through real route serialization. Test SQLite and the existing PostgreSQL fixture/SQL compatibility guards. Regenerate API client types with the repository generator if applicable; no handwritten generated type edits.
- [ ] Keep language mutation gated off while active Sync lacks F3 support: a REST path that cannot publish the complete mutation must reject it before storage. Deploy the column additively; do not downgrade/erase stored values during rollback.
- [ ] Run the new tests plus C `Tests/Notes/test_notes_scope_service.py`, `Tests/tldw_api/test_notes_workspace_client.py`; S `tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py` and `tldw_Server_API/tests/DB_Management/test_chacha_postgres_notes_bootstrap_lifecycle.py`. Commit each repository independently with migration evidence.

## F3: Server versioned language publication

**Dependencies:** F2-S. Independently testable server contract extension.
**Files S:** modify `tldw_Server_API/app/core/Sync/v2/models.py`, `domain_adapters/notes.py`, `materializers/notes.py`, `adapters.py`, `service.py` under that same Sync v2 directory, and `tldw_Server_API/app/api/v1/endpoints/notes.py`; add `tldw_Server_API/tests/Sync/test_sync_v2_note_language.py`; extend existing `test_sync_v2_models.py` and `test_sync_v2_notes_materializer.py`.
**Interfaces:** retain v1 validator unchanged; add `validate_notes_note_upsert_payload_v2(payload)` requiring all v1 fields plus explicit `content_language` for a complete v2 snapshot. Plan adapter/schema pair `(2, 2)` for `notes.note`, rechecked against current registrations before allocation. Unsupported pairs are rejected before append.

- [ ] Record the new version in a server-local ADR extending ADR-031, with its allocated filename linked from that task and this plan. This is required because accepted ADR-031 names a fixed v1 contract; Chatbook's ADR is not a replacement for server governance.
- [ ] Add strict validator tests:

```python
def test_v2_requires_explicit_language():
    from tldw_Server_API.app.core.Sync.v2.models import validate_notes_note_upsert_payload_v2
    with pytest.raises(ValueError):
        validate_notes_note_upsert_payload_v2({"title": "T", "content": "{}"})
    value = validate_notes_note_upsert_payload_v2({
        "title": "T", "content": "{}", "conversation_id": None,
        "message_id": None, "content_language": "json",
    })
    assert value["content_language"] == "json"
```

- [ ] Run the new file red; implement explicit version dispatch, supported/writable adapter capability reporting, and current-version REST mutation publication through the existing durable envelope/materializer pipeline.
- [ ] For v1 updates, preserve the current language in the materialized row. Do not silently convert a v1 payload to v2 while retaining its old hash. A v1 producer attempting to overwrite a language-bearing current object without a compatible base/version receives an actionable conflict/upgrade result. V2 writers use complete current snapshots and correct hash lineage.
- [ ] Add real Sync store tests for v1→v2, omitted/reset, restored tombstones, idempotent retry, language-only conflict, and REST↔Sync projection equality. Include note-derived publication producers: enumerate every `notes.note` constructor with `rg -n 'notes.note' tldw_Server_API/app`, and route each through the same version/payload builder rather than updating only the Notes endpoint.
- [ ] Exercise two clients with different supported versions. New heads are never down-projected and rehashed for old consumers; old consumers see explicit unsupported-version state. Confirm an old client cannot erase stored language through a subsequent full snapshot.
- [ ] Run the three Sync selections and the F2-S API selection; scope Bandit/static checks to touched modules. Commit the server contract and ADR together.

## F4: Chatbook and browser clients preserve negotiated language

**Dependencies:** F2-C, F2-S, F3. Separate C and S client commits.
**Files C:** modify `tldw_chatbook/Sync_Interop/envelope_builder.py`, `envelope_applier.py`, `notes_outbox_producer.py`, `notes_local_store.py`, `domain_adapters/notes.py`, `domain_adapters/notes_m1.py`, `sync_state.py`; `tldw_chatbook/tldw_api/sync_schemas.py`, `client.py`; create `Tests/Sync_Interop/test_note_language_compatibility.py`.
**Files U:** modify `src/services/tldw/TldwApiClient.ts`, `src/services/tldw/__tests__/notes-client.test.ts`; add `src/services/note-language-capability.ts` for the capability type/response decoder and `src/services/__tests__/note-language-compatibility.test.ts`.
**Interfaces:** `NotesLanguageCapability` has `available: bool` and a bounded reason code; builders select an advertised schema/adapter pair. Add `TldwApiClient.getNotesLanguageCapability(): Promise<NotesLanguageCapability>` using F2's authenticated capability route. Pending portable language belongs to the note's scoped mutation state, not the validator.

- [ ] Test capability absence before adding optimistic UI behavior:

```ts
it("requires an explicit advertised language capability", async () => {
  mocks.bgRequest.mockResolvedValueOnce({});
  const client = new TldwApiClient();
  expect(await client.getNotesLanguageCapability()).toEqual({
    available: false, reason: "unsupported_server"
  });
  expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({
    path: "/api/v1/notes/capabilities", method: "GET"
  }));
  mocks.bgRequest.mockResolvedValueOnce({ id: "n1", content: "{}" });
  await client.createNote("{}", { title: "Config" });
  expect(mocks.bgRequest).toHaveBeenLastCalledWith(expect.objectContaining({
    body: { content: "{}", title: "Config" }
  }));
});
```

This case belongs in the existing `notes-client.test.ts`, which already defines
`mocks.bgRequest` and imports `TldwApiClient`. Add separate cases for explicit
support, authenticated 404, transport failure, and profile changes. F5 verifies
that a body-only acknowledgment does not clear pending language in caller state.

- [ ] Add red envelope tests using `SyncEnvelopeBuilder` and `SyncEnvelopeApplier` from the existing M1 fixture. Assert exact text and language after builder→serialization→receiver→real SQLite. Include encrypted domain `notes`, not just clear `notes.note`.
- [ ] Extend both builders with language and negotiated versions, receivers with omission preservation, and canonical payload hashing with language. Keep language encrypted in the private payload; do not add it to routing metadata. Retain pending intent when unsupported; do not dequeue it on a body-only acknowledgment.
- [ ] Support new receivers reading old payloads without clearing language. Pause unsafe old full-snapshot private publication rather than guessing that an old writer will preserve unknown fields. Persist the pending reason across restart.
- [ ] Verify authentication/profile switching cannot apply one account's metadata or capability cache to another note. Cache capabilities by server/profile, invalidate on reconnect/version change, and recheck before publication.
- [ ] Run C new selection plus `Tests/Sync_Interop/test_envelope_builder.py`, `test_notes_m1_adapters.py`, `test_notes_outbox_producer.py`; U run the two named client/compatibility files. Commit each client separately after its server compatibility control passes.

## F5: Canonical drafts carry language and survive empty-body failures

**Dependencies:** F2/F4. Separate C and S commits.
**Files C:** modify `tldw_chatbook/Library/library_notes_state.py`, `library_notes_session.py`, `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`, `tldw_chatbook/Notes/notes_device_state_schema.py`; create `tldw_chatbook/Notes/note_draft_recovery.py`, `Tests/Notes/test_note_draft_recovery.py`; extend `Tests/Library/test_library_notes_session.py`.
**Files U:** modify `src/components/Notes/hooks/useNotesEditorState.tsx`, `src/components/Notes/notes-manager-utils.ts`; add `src/components/Notes/__tests__/useNotesEditorState.structured-draft.test.tsx`.
**Interfaces:** extend `NormalizedDatabaseNote`, `DatabaseNoteDraft`, and `DatabaseNoteSavePayload` with `content_language: str | None`. Add an omission sentinel to coordinator mutation so null can explicitly reset. `NoteDraftRecovery.save(scope, draft, base_version)`, `.load(scope, note_id)`, `.acknowledge(scope, note_id, saved_revision)` persist only pending note drafts.

- [ ] Add a red session test using existing `FakeDatabaseNotePort`:

```python
@pytest.mark.asyncio
async def test_language_only_edit_is_a_real_save():
    port = FakeDatabaseNotePort(_detail())
    port.save_replies.append(DatabaseNotePortSaveReply.saved(version=2))
    owner = DatabaseNoteSessionCoordinator(port=port)
    await owner.open_session("n-1")
    owner.mutate(content_language="yaml")
    await owner.request_save(explicit=True)
    assert port.save_calls[-1][2].content_language == "yaml"
    assert port.save_calls[-1][2].body == "Original body"
    assert len(port.save_calls) == 1
```

This case belongs beside the existing `FakeDatabaseNotePort` and `_detail` helpers
in `test_library_notes_session.py`; extend their dataclass construction with the
default null language. Assert a single revision increment as well as body equality.

- [ ] Extend canonical state and save payloads, including conflict reload/overwrite, while leaving syntax checking out of `validate_database_note_draft`. Metadata and body must belong to the same captured draft revision.
- [ ] Reuse the device-state DB connection/migration ownership, adding a dedicated note-draft table rather than inserting fake filesystem sync journal entries. Scope by profile/principal/workspace/note. Store exact text and language with restrictive file permissions; exclude draft bodies from portable backups/sync. Do not persist diagnostics. Retain source until its matching save is acknowledged or the user explicitly discards it.
- [ ] Test empty draft: start from a saved nonempty note, delete all text, simulate server empty-body rejection, navigate away, reopen a new repository/session instance, and recover the empty text and language. A newer pending revision must survive an older save acknowledgment. Failed local recovery storage must block a discard-producing navigation and show a truthful error.
- [ ] On the web use the existing authority-scoped offline draft queue and storage mechanism. Extend its serialized version and snapshots with language; do not introduce a second raw browser storage mechanism. Gate WYSIWYG derivation until mode is known, including offline draft restoration.
- [ ] Run C recovery/session files and U structured-draft tests; include profile isolation, logout, storage quota failures, crash/restart, and matching-revision cleanup. Commit each app's recovery integration separately.

## F6: Portable envelopes and source import/export contracts

**Dependencies:** F2/F4. Separate C archive and S web export changes.
**Files C:** modify `tldw_chatbook/Chatbooks/chatbook_models.py`, `chatbook_creator.py`, `chatbook_importer.py`; `tldw_chatbook/Notes/note_import_parsers.py`, `note_import_plan_models.py`, `note_import_executor.py`; add `Tests/Chatbooks/test_chatbook_note_language_round_trip.py`, `Tests/Notes/test_structured_note_import.py`.
**Files U:** modify `src/components/Notes/export-utils.ts`, `hooks/useNotesImport.tsx`, `hooks/useNotesExport.tsx`; add `src/components/Notes/__tests__/structured-note-export.test.ts`.
**Interfaces:** raw source output is `(filename, exact text)`; a portable note envelope includes `format_version`, `title`, `content`, `content_language`, and existing metadata. Keep existing v1–v3 Chatbook readers; choose the next verified archive version for language-bearing notes (v4 in the inspected tree).

- [ ] Add red envelope/source tests:

```ts
it("distinguishes source from a portable note", () => {
  const note = { title: "Data", content: '{"x":9007199254740993}\n',
    keywords: [], content_language: "json" };
  expect(buildStructuredSource(note)).toEqual({ filename: "Data.json", text: note.content });
  const portable = JSON.parse(buildPortableNote(note));
  expect(portable.content).toBe(note.content);
  expect(portable.content_language).toBe("json");
});
```

`buildStructuredSource(note)` and `buildPortableNote(note)` are new exports from
`export-utils.ts`; the latter serializes an envelope containing source as a string,
never parses the user's document into application values.

- [ ] Implement versioned envelopes and archive note records without wrapping structured content in Markdown. Test exports containing both Canvas and structured Notes so archive version selection does not discard either extension. Validate import version before mutating storage; unknown versions fail atomically.
- [ ] Carry inferred extension mode through import planning and execution; never infer a language from note text. Existing raw source without a filename has no portable language promise. Preserve exact file representation on file-to-file export and exact note text in envelopes.
- [ ] Run C new files plus `Tests/Chatbooks/test_chatbook_canvas_round_trip.py`, `Tests/Chatbooks/test_import_transactions.py`; U structured export tests. Commit each repository's contracts before I4 exposes their controls.
