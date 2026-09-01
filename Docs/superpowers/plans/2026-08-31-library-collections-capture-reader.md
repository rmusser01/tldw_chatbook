# Library Collections Capture Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the obsolete generic-container Collections surface with an authority-safe Local/Server capture reader for saving, finding, reading, annotating, archiving, recovering, and permanently deleting Pocket/Instapaper-style captures.

**Architecture:** Land one independent `tldw_server` prerequisite that makes Reading List count and rows share a database snapshot and advertises that fact. In Chatbook, add an additive schema-v2 capture repository beside untouched v1 legacy tables, normalize Local and Server through one capture-specific scope service, and mount destination-owned Items/Work widgets inside the existing `LibraryAdaptiveReaderShell`; reuse the existing authenticated API client, secure URL extraction, inert HTML-to-text path, private-file primitives, and shared layout resolver.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite/FTS5, existing `TLDWAPIClient`/Pydantic schemas, stdlib `dataclasses`/`enum`/`hashlib`/`json`, pytest, Hypothesis, Textual Pilot, TCSS.

**ADR required:** yes

**ADR path:** `backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md`

**Reason:** TASK-18919 changes durable Collections storage, source authority, migration, service, and legacy-data boundaries.

---

## Source documents and delivery boundary

- Specification: `Docs/superpowers/specs/2026-08-31-library-collections-capture-reader-design.md`
- Architecture: `backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md`
- Paging contract: `backlog/decisions/067-library-top-level-pagination-contracts.md`
- Destructive actions: `backlog/decisions/055-library-destructive-action-reversibility-rule.md`
- Shared reader shell: `backlog/decisions/086-library-adaptive-reader-shell.md`
- Testing lessons: `backlog/docs/lessons-testing-evidence.md`
- Live verification lessons: `backlog/docs/lessons-live-verification.md`
- Backlog lessons: `backlog/docs/lessons-backlog-hygiene.md`

This specification crosses two repositories, so deliver it through two reviewable PRs:

| PR | Repository | Purpose | Safe intermediate state |
| --- | --- | --- | --- |
| A | `rmusser01/tldw_server` | Make the existing Reading List page coherent and advertise `hasReadingSnapshotPagesV1=true` | No public endpoint or response-shape change; older Chatbook clients ignore the capability |
| B | `rmusser01/tldw_chatbook` | Add Local captures, legacy recovery, Server adapter, and the adaptive reader | Local works independently; Server browse fails closed until PR A's exact capability is observed |

Do not mix commits from the two repositories. PR B may be developed while PR A is under review, but
the Server live walkthrough is blocked until PR A is deployed. Do not add a new endpoint, a new
database setting, a dependency, a Media record, a generic Library controller, or another pane
framework.

Before each PR, fetch the current `dev`, search open branches/PRs for TASK-18919 and the capability
name, and rebase the feature commits. Re-run the task/ADR id collision check after rebasing.

## File responsibility map

### PR A — `tldw_server` paging prerequisite

- Modify `tldw_Server_API/app/core/DB_Management/Collections_DB.py` — hold count, rows, and tag
  hydration under the existing pinned backend transaction.
- Modify `tldw_Server_API/app/api/v1/endpoints/config_info.py` — advertise the exact capability only
  after the database behavior is covered.
- Modify `tldw_Server_API/tests/Collections/test_reading_service.py` — controlled concurrent-writer
  snapshot regression.
- Modify `tldw_Server_API/tests/Config/test_docs_info_capabilities.py` — exact boolean attestation.

### PR B — Chatbook capture domain

- Modify `tldw_chatbook/DB/Library_Collections_DB.py` — atomic schema-v2 migration, future-version
  refusal, and v1 compatibility checks.
- Create `tldw_chatbook/Library/collections_capture_models.py` — immutable authority, scope, page,
  summary, detail, mutation, save-outcome, and tri-state capability contracts plus envelope
  validation.
- Create `tldw_chatbook/Library/collections_capture_repository.py` — synchronous Local capture CRUD,
  exact page snapshots, FTS, saved searches, highlights, linked-Note references, revisions, and
  extraction state.
- Create `tldw_chatbook/Library/collections_offline_store.py` — authority-rooted managed-file
  admission, publication, open/delete, reconciliation, tombstones, and resumable scavenging.
- Create `tldw_chatbook/Library/collections_legacy_recovery.py` — bounded v1 inspection and complete
  coherent-snapshot atomic JSON export.
- Create `tldw_chatbook/Library/collections_capture_service.py` — capture backend protocol, Local
  off-loop adapter, authority resolution, scope service, capability cache, and source replacement.
- Create `tldw_chatbook/Library/server_collections_capture_service.py` — focused async Reading API
  adapter using the existing authenticated client and exact docs-info evidence.
- Create `tldw_chatbook/UI/Library_Modules/library_collections_capture_controller.py` — immutable
  page/session state, settle delay, selected-versus-loaded identity, generation fencing, stale
  reconciliation, receipts, and mutation orchestration.
- Create `tldw_chatbook/Widgets/Library/library_collections_capture_reader.py` — contextual Library
  scopes, compact Items list/Quick Capture, permanent Read/Highlights/Notes/Info Work pane, legacy
  recovery surface, and disabled-action explanations.
- Modify `tldw_chatbook/Library/library_collections_service.py` — preserve the v1 compatibility type
  only as a read-only legacy seam; mutations return `legacy_read_only`.
- Modify `tldw_chatbook/UI/Screens/library_screen.py` — compose and drive the new reader without
  becoming its domain controller.
- Modify `tldw_chatbook/app.py`, `tldw_chatbook/Library/__init__.py`, and
  `tldw_chatbook/Widgets/Library/__init__.py` — wire and export capture services separately from
  legacy recovery.
- Modify `tldw_chatbook/config.py` and `tldw_chatbook/UI/Screens/settings_screen.py` — add only
  `[library.collections_reader]` Items preferences; shared Library preferences remain centralized.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` and regenerate
  `tldw_chatbook/css/tldw_cli_modular.tcss` — Collections-specific rows, Work modes, stale/error,
  overflow, and recovery styling; never hand-edit only the generated bundle.
- Delete `tldw_chatbook/Library/library_collections_state.py`,
  `tldw_chatbook/UI/Library_Modules/library_collections_browse_controller.py`, and
  `tldw_chatbook/Widgets/Library/library_collections_panel.py` after the new mounted reader owns the
  route; the new legacy recovery module replaces only their bounded recovery responsibility.
- Modify `tldw_chatbook/Library/library_tool_contract.py`,
  `tldw_chatbook/Library/local_library_tool_service.py`, `tldw_chatbook/MCP/server.py`, and
  `tldw_chatbook/UI/Console_Modules/library_activity.py` — remove generic Collections as a current
  Library item/tool backend; never redirect an old operation name to captures.
- Modify `tldw_chatbook/runtime_policy/registry.py` — remove the retired
  `library.collections` LIST/DETAIL resource and generic display label while retaining unrelated
  Library agent-tool policy resources.

## PR A — Server snapshot prerequisite

### Task 1: Pin Reading List count and rows to one server snapshot

**Files:**
- Modify: `tldw_Server_API/tests/Collections/test_reading_service.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Collections_DB.py`

- [ ] **Step 1: Reproduce the pre-fix mismatch with a controlled writer**

Add a real SQLite test that seeds 21 Reading items, pauses the reader after `COUNT(*)`, inserts one
matching item through a second connection, then releases the page query. Assert that the fixed
operation returns a page and total from one of the two legal snapshots, never a mixed pair:

```python
legal = {
    (21, tuple(before_ids[:20])),
    (22, tuple(after_ids[:20])),
}
rows, total = reader.list_items(page=1, size=20, sort="created_desc")
assert (total, tuple(row.id for row in rows)) in legal
```

Use `threading.Event` and a test-only interception at the backend execution boundary; do not use a
fixed sleep. First keep the production implementation unchanged so the test demonstrates the mixed
snapshot on the pre-fix path.

- [ ] **Step 2: Run the controlled regression and witness failure**

```bash
python -m pytest tldw_Server_API/tests/Collections/test_reading_service.py -k snapshot -q
```

Expected: FAIL because the writer can commit between count and page evaluation.

- [ ] **Step 3: Use the existing transaction and connection plumbing**

In `CollectionsDatabase.list_content_items`, keep current filtering and SQL construction, but execute
the count, page, and tag hydration inside the existing `transaction()` context and pass its
connection through every backend call:

```python
with self.transaction() as connection:
    total = self.backend.execute(
        count_query,
        tuple(params),
        connection=connection,
    ).scalar
    row_result = self.backend.execute(
        page_query,
        tuple([*params, page_size, page_offset]),
        connection=connection,
    )
    tag_map = self._fetch_tags_for_item_ids(
        item_ids,
        connection=connection,
    )
```

Add the optional `connection` parameter only to the existing focused tag helper and forward it to
its current backend call. Do not create a second transaction abstraction or change the endpoint
schema.

- [ ] **Step 4: Run focused server paging tests**

```bash
python -m pytest tldw_Server_API/tests/Collections/test_reading_service.py -k 'list or snapshot' -q
```

Expected: PASS, including the controlled writer.

- [ ] **Step 5: Commit the database correction**

```bash
git add tldw_Server_API/app/core/DB_Management/Collections_DB.py tldw_Server_API/tests/Collections/test_reading_service.py
git commit -m 'fix(reading): keep page count and rows in one snapshot'
```

### Task 2: Advertise and verify the paging attestation

**Files:**
- Modify: `tldw_Server_API/tests/Config/test_docs_info_capabilities.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/config_info.py`

- [ ] **Step 1: Write the exact capability test**

```python
def test_docs_info_attests_reading_snapshot_pages(monkeypatch, tmp_path):
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    safe_config = config_info.load_safe_config()

    assert safe_config["capabilities"]["hasReadingSnapshotPagesV1"] is True
    assert safe_config["supported_features"]["hasReadingSnapshotPagesV1"] is True

    response = asyncio.run(config_info.get_documentation_config())
    assert response["capabilities"]["hasReadingSnapshotPagesV1"] is True
```

- [ ] **Step 2: Run it and witness failure**

```bash
python -m pytest tldw_Server_API/tests/Config/test_docs_info_capabilities.py -k reading_snapshot -q
```

Expected: FAIL because the key is absent.

- [ ] **Step 3: Add one literal capability entry**

Add `"hasReadingSnapshotPagesV1": True` to the one `caps` dictionary already returned as both
`capabilities` and `supported_features`. Do not add an environment toggle: this attests shipped
behavior, not optional configuration.

- [ ] **Step 4: Verify the complete prerequisite**

```bash
python -m pytest tldw_Server_API/tests/Collections/test_reading_service.py -k 'list or snapshot' -q
python -m pytest tldw_Server_API/tests/Config/test_docs_info_capabilities.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit and open PR A**

```bash
git add tldw_Server_API/app/api/v1/endpoints/config_info.py tldw_Server_API/tests/Config/test_docs_info_capabilities.py
git commit -m 'feat(config): attest coherent Reading List pages'
git diff --check origin/dev...HEAD
```

Open a server PR that states the same endpoint is unchanged and that the new exact capability is the
Chatbook enablement gate. Do not proceed to a Server live walkthrough until this PR is deployed.

## PR B — Chatbook capture reader

### Task 3: Rebase, inventory cutover surfaces, and pin the task checkpoint

**Files:**
- Modify: `backlog/tasks/task-18919 - Build-the-Local-and-Server-Collections-capture-reader.md`
- Create: `Docs/superpowers/reviews/2026-08-31-library-collections-cutover-inventory.md`

- [ ] **Step 1: Rebase only the feature commits onto current `origin/dev`**

```bash
git fetch origin dev
git log --oneline --decorate origin/dev..HEAD
git rebase origin/dev
git diff --check origin/dev...HEAD
```

Expected: the three approved design commits and this planning commit remain; unrelated worktree
changes do not enter the branch.

- [ ] **Step 2: Search for duplicate work and id collisions**

```bash
git branch -a | rg '18919|collections-capture|reading-snapshot'
gh pr list --repo rmusser01/tldw_chatbook --state open --search '18919 Collections capture'
rg -n 'ADR-107|TASK-18919' backlog Docs
```

Expected: no competing implementation and one canonical ADR-107/TASK-18919 pair.

- [ ] **Step 3: Record every old generic-container surface before editing**

Create a table with columns `surface`, `current symbol/selector`, `cutover result`, and
`verification`. Include Library destination, app wiring, Local tool service, MCP server, Console
activity, Home/rail counts, RAG descriptions, help text, tests, generic create/rename/membership/
delete actions, and legacy recovery. Mark each old operation `retire`, `legacy_read_only`, or
`recovery-only`; no row may say “redirect to capture.”

- [ ] **Step 4: Add the Backlog implementation block**

Ensure the task includes this exact block before production code changes:

```text
ADR required: yes
ADR path: backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md
Reason: TASK-18919 changes durable Collections storage, source authority, migration, service, and legacy-data boundaries.
```

- [ ] **Step 5: Commit the implementation checkpoint**

```bash
git add 'backlog/tasks/task-18919 - Build-the-Local-and-Server-Collections-capture-reader.md' Docs/superpowers/reviews/2026-08-31-library-collections-cutover-inventory.md
git commit -m 'docs(collections): inventory capture reader cutover'
```

### Task 4: Add the atomic schema-v2 capture foundation

**Files:**
- Modify: `tldw_chatbook/DB/Library_Collections_DB.py`
- Create: `Tests/DB/test_library_collections_capture_migration.py`

- [ ] **Step 1: Write migration and future-version failures first**

Cover fresh creation, a real v1 fixture with active/deleted collections and memberships, injected
DDL failure rollback, two concurrent openers, v2 reopen, and synthetic version 3. Assert v1 rows and
their stored values are unchanged. Version 3 must raise a typed error whose reason is
`schema_too_new` and must not write any schema row.

- [ ] **Step 2: Run the focused migration file and witness failure**

```bash
../../.venv/bin/python -m pytest Tests/DB/test_library_collections_capture_migration.py -q
```

Expected: FAIL because `_CURRENT_SCHEMA_VERSION` is 1 and capture tables do not exist.

- [ ] **Step 3: Implement one bounded migration path**

Advance `_CURRENT_SCHEMA_VERSION` to 2. Replace startup `executescript` stamping with one
`BEGIN IMMEDIATE` path that reads `MAX(version)` inside the transaction, refuses versions above 2,
creates missing v1 objects for a fresh database, creates all capture-owned v2 objects, then inserts
version 2 and commits. Required capture-owned objects are:

```text
collection_capture_items
collection_capture_tags
collection_capture_item_tags
collection_capture_highlights
collection_capture_saved_searches
collection_capture_note_links
collection_capture_offline_files
collection_capture_scavenge_state
collection_capture_search (FTS5) and owned triggers
```

Use these durable shapes so later tasks do not invent incompatible storage:

```text
collection_capture_items:
  authority_key TEXT NOT NULL; capture_id TEXT NOT NULL; submitted_url TEXT NOT NULL;
  canonical_url TEXT NOT NULL; domain TEXT NOT NULL DEFAULT '';
  title/summary/freeform_note/text_content/clean_html/byline/published_at/read_at TEXT;
  content_hash TEXT; word_count INTEGER; status TEXT NOT NULL CHECK saved|reading|read|archived;
  favorite INTEGER NOT NULL CHECK 0|1; processing_state TEXT NOT NULL
  CHECK queued|processing|ready|failed|interrupted; last_fetch_error TEXT;
  media_authority_key/media_item_id TEXT; created_at/updated_at TEXT NOT NULL;
  revision INTEGER NOT NULL DEFAULT 1 CHECK revision > 0; purge_state TEXT;
  PRIMARY KEY (authority_key, capture_id); UNIQUE (authority_key, canonical_url)

collection_capture_tags:
  authority_key TEXT NOT NULL; tag_id INTEGER NOT NULL; normalized_name/display_name TEXT NOT NULL;
  PRIMARY KEY (authority_key, tag_id); UNIQUE (authority_key, normalized_name)
collection_capture_item_tags:
  authority_key/capture_id/tag_id composite primary key with same-authority cascading foreign keys
collection_capture_highlights:
  authority_key/highlight_id composite primary key; capture_id TEXT; quote/note/anchor_json TEXT;
  detached INTEGER; created_at/updated_at TEXT; revision INTEGER; same-authority item foreign key
collection_capture_saved_searches:
  authority_key/search_id composite primary key; name TEXT NOT NULL; query_json TEXT NOT NULL;
  created_at/updated_at TEXT; revision INTEGER; unique authority/name
collection_capture_note_links:
  authority_key/link_id composite primary key; capture_id TEXT; note_authority_key/note_id TEXT;
  created_at TEXT; unique same-authority capture/reference tuple; no cross-database foreign key
collection_capture_offline_files:
  authority_key/file_id composite primary key; capture_id TEXT; relative_path TEXT; content_hash TEXT;
  reserved_size/actual_size INTEGER; media_type TEXT; state TEXT CHECK staging|ready|failed|purging;
  failure_reason/temporary_name TEXT; created_at/updated_at TEXT; revision INTEGER
collection_capture_scavenge_state:
  authority_key TEXT PRIMARY KEY; authority_fingerprint/cursor_kind/cursor_value/updated_at TEXT
```

FTS indexes only title, summary, freeform note, readable text, and denormalized tag text. Owned
triggers update/delete the `(authority_key, capture_id)` row from capture/tag mutations. Every Local
query and mutation requires the active `authority_key`; no repository method has an all-authority
fallback. Add indexes for each allowed page sort,
status/favorite/domain filters, item tags, capture-owned children, and offline lifecycle state. Test
`PRAGMA foreign_key_check` and query plans for the fixed 20-row paths; do not add speculative fields
or generalized metadata blobs.

Use a bounded SQLite busy timeout already supported by `BaseDB`; do not loop indefinitely. Keep
`library_collections` and `library_collection_items` names and rows untouched. Expose
`require_capture_schema()` and `has_compatible_legacy_schema()` so capture callers and legacy
recovery fail independently.

- [ ] **Step 4: Prove the migration matrix**

```bash
../../.venv/bin/python -m pytest Tests/DB/test_library_collections_capture_migration.py -q
../../.venv/bin/python -m pytest Tests/DB/test_held_connections.py -k collections -q
```

Expected: PASS; held-connection behavior remains intact.

- [ ] **Step 5: Commit the storage foundation**

```bash
git add tldw_chatbook/DB/Library_Collections_DB.py Tests/DB/test_library_collections_capture_migration.py
git commit -m 'feat(collections): add atomic capture schema v2'
```

### Task 5: Define capture identities, scopes, envelopes, and capabilities

**Files:**
- Create: `tldw_chatbook/Library/collections_capture_models.py`
- Create: `Tests/Library/test_collections_capture_models.py`

- [ ] **Step 1: Write contract examples and malformed-envelope properties**

Pin fixed page size 20, allowed statuses/sorts, normalized exact filters, one-based pages,
authority-qualified identity, stable ids, exact applied scope, and tri-state capabilities. Property
tests reject duplicate ids, oversized pages, undersized non-final pages, impossible totals,
inconsistent scope/page metadata, invalid sort names, nested saved-search expressions, and unknown
keys.

- [ ] **Step 2: Run and witness the missing module**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_collections_capture_models.py -q
```

Expected: collection/import failure.

- [ ] **Step 3: Implement the complete public contracts as frozen dataclasses**

Use this public shape; source-specific payloads stay behind adapters:

```python
class CapabilityState(str, Enum):
    UNKNOWN = "unknown"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class CaptureAuthority:
    kind: Literal["local", "server"]
    key: str                 # opaque, non-loggable authority key
    fingerprint: str         # compact, non-reversible display-safe identity


@dataclass(frozen=True)
class CapturePageRequest:
    authority_key: str
    search: str = ""
    statuses: tuple[str, ...] = ()
    favorite: bool | None = None
    tags: tuple[str, ...] = ()
    domain: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    sort: str = "saved_desc"
    page: int = 1
    size: int = 20


@dataclass(frozen=True)
class CapturePage:
    applied: CapturePageRequest
    items: tuple[CaptureSummary, ...]
    total: int
    source_revision: str | None = None


@dataclass(frozen=True)
class CaptureCapability:
    state: CapabilityState
    reason: str | None = None
```

Also define `CaptureIdentity`, `CaptureSummary`, `CaptureDetail`, `CaptureSaveRequest`,
`CaptureSaveOutcome`, `CaptureConflict`, `SavedCaptureSearch`, `CaptureHighlight`,
`CaptureHighlightDraft`, `CaptureNoteLink`, `ExternalNoteReference`, `ExternalMediaReference`,
`ExternalReferenceAvailability`, `ResolvedCaptureDetail`, `CaptureOfflineCopy`,
`CaptureActionResult`, and typed `CollectionsCaptureError(reason, retryable=False)`. Define the
remaining aggregate contracts explicitly:

```python
CAPTURE_CAPABILITY_NAMES = (
    "browse", "capture", "update", "highlights", "linked_notes",
    "summarize", "listen", "archive", "offline_copy", "hard_delete",
    "retry_extraction", "legacy_recovery",
)


@dataclass(frozen=True)
class CaptureCapabilities:
    values: Mapping[str, CaptureCapability]

    def for_action(self, action: str) -> CaptureCapability: ...


@dataclass(frozen=True)
class CaptureSavedSearchPage:
    items: tuple[SavedCaptureSearch, ...]
    total: int
    page: int
    size: int = 20


@dataclass(frozen=True)
class ExternalNoteReference:
    authority_key: str
    note_id: str


@dataclass(frozen=True)
class ExternalReferenceAvailability:
    state: Literal["available", "unavailable"]
    reason: str | None = None


@dataclass(frozen=True)
class ResolvedCaptureDetail:
    capture: CaptureDetail
    media: ExternalReferenceAvailability | None
    note_links: tuple[tuple[CaptureNoteLink, ExternalReferenceAvailability], ...]


@dataclass(frozen=True)
class CaptureContentResult:
    identity: CaptureIdentity
    kind: Literal["summary", "audio"]
    text: str | None = None
    artifact_reference: str | None = None
```

`ExternalMediaReference` carries its owner authority and item id just like the Note reference.
`CaptureCapabilities` rejects unknown/missing action keys at construction so the UI never treats an
omitted destructive capability as supported. `ExternalNoteReference` is opaque and never resolved
inside the Collections database. Validation returns new normalized objects and never logs rejected
content.

- [ ] **Step 4: Run model tests**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_collections_capture_models.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the pure contract**

```bash
git add tldw_chatbook/Library/collections_capture_models.py Tests/Library/test_collections_capture_models.py
git commit -m 'feat(collections): define capture reader contracts'
```

### Task 6: Implement coherent Local capture persistence

**Files:**
- Create: `tldw_chatbook/Library/collections_capture_repository.py`
- Create: `Tests/Library/test_collections_capture_repository.py`

- [ ] **Step 1: Write Local repository failures first**

Cover same-URL saves under two Local profiles sharing one database path (they remain distinct),
canonical-URL upsert within one authority, deterministic tag merge, omitted-value preservation, explicit archived
resave, independent text/clean-HTML fields, deterministic content replacement/hash, revision CAS,
status/favorite/tags/note updates, archive/restore, hard-delete tombstone creation, FTS, all allowed
filters/sorts, and stable id tie-breakers. Seed 45 records and prove pages 1–3 and exact totals.

Use a controlled second connection between count and rows; Local must still return one snapshot.

- [ ] **Step 2: Run and witness failure**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_collections_capture_repository.py -q
```

Expected: collection/import failure.

- [ ] **Step 3: Implement the repository on the existing DB wrapper**

Expose synchronous methods intended for `asyncio.to_thread`:

```python
class CollectionsCaptureRepository:
    def list_page(self, request: CapturePageRequest) -> CapturePage: ...
    def get_detail(self, identity: CaptureIdentity) -> CaptureDetail | None: ...
    def save_capture(self, request: CaptureSaveRequest) -> CaptureSaveOutcome: ...
    def update_capture(self, identity, *, expected_revision: int, changes) -> CaptureDetail: ...
    def list_saved_searches(self, *, page: int, size: int = 20) -> CaptureSavedSearchPage: ...
    def create_saved_search(self, name: str, request: CapturePageRequest) -> SavedCaptureSearch: ...
    def update_saved_search(self, search_id: str, *, name: str, request: CapturePageRequest, expected_revision: int) -> SavedCaptureSearch: ...
    def delete_saved_search(self, search_id: str, *, expected_revision: int) -> CaptureActionResult: ...
    def list_highlights(self, identity: CaptureIdentity) -> tuple[CaptureHighlight, ...]: ...
    def save_highlight(self, identity: CaptureIdentity, draft: CaptureHighlightDraft) -> CaptureHighlight: ...
    def delete_highlight(self, identity: CaptureIdentity, highlight_id: str, *, expected_revision: int) -> CaptureActionResult: ...
    def list_note_links(self, identity: CaptureIdentity) -> tuple[CaptureNoteLink, ...]: ...
    def link_note(self, identity: CaptureIdentity, note_reference: ExternalNoteReference) -> CaptureNoteLink: ...
    def unlink_note(self, identity: CaptureIdentity, link_id: str) -> CaptureActionResult: ...
    def hard_delete(self, identity: CaptureIdentity, *, expected_revision: int) -> CaptureActionResult: ...
```

Use `db.read_transaction()` for the paired count/page and detail aggregates, and `db.transaction()`
for mutations. Every SQL identifier and sort fragment comes from a literal allowlist; all values
are parameters. Every statement includes `authority_key`; canonical URL and tag uniqueness are
authority-scoped, including when two profiles resolve to the same database file. Generate Local ids without exposing paths. Store only authority-qualified external
Media/Note references and never declare cross-database foreign keys.

- [ ] **Step 4: Run repository and migration tests together**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_collections_capture_repository.py Tests/DB/test_library_collections_capture_migration.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Local persistence**

```bash
git add tldw_chatbook/Library/collections_capture_repository.py Tests/Library/test_collections_capture_repository.py
git commit -m 'feat(collections): persist local capture reading state'
```

### Task 7: Add commit-first Local extraction and inert reader content

**Files:**
- Modify: `tldw_chatbook/Library/collections_capture_repository.py`
- Create: `Tests/Library/test_collections_capture_extraction.py`
- Reuse unchanged: `tldw_chatbook/Local_Ingestion/web_article_ingestion.py`
- Reuse unchanged: `tldw_chatbook/Subscriptions/html_text.py`

- [ ] **Step 1: Write lifecycle and trust-boundary tests**

Assert save returns a committed `queued` capture before extraction runs; queued becomes processing
then ready/failed; startup converts stale processing to interrupted; Retry preserves omitted status
and favorite; handled fetch failure preserves the capture with a bounded reason. Reject non-HTTP(S),
unsafe/SSRF targets, redirect/byte overflow, control characters, scripts, event handlers, remote
assets, and active content. Assert stored text is inert and clean HTML is optional rather than
fabricated.

- [ ] **Step 2: Run and witness failure**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_collections_capture_extraction.py -q
```

Expected: FAIL because extraction transitions do not exist.

- [ ] **Step 3: Compose the existing secure extractor off-loop**

Add repository claim/complete/fail/interruption methods guarded by identity and revision. In the
Local service task, call existing `extract_article_for_ingest(url, options)` through
`asyncio.to_thread`; do not add a fetch stack. Normalize its readable text with
`readable_body_text`/`strip_control_characters`. If an upstream path supplies HTML, reduce it to
inert display text unless the existing renderer can prove a safe HTML subset; unsafe HTML fails
closed to metadata/text-only rather than entering Textual markup.

- [ ] **Step 4: Run extraction and URL-security regressions**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_collections_capture_extraction.py Tests/Local_Ingestion -k 'web or url or article' -q
```

Expected: PASS. If the repository has no `Tests/Local_Ingestion` directory after rebase, run only
the discovered focused tests that own `web_article_ingestion.py` and record their exact paths.

- [ ] **Step 5: Commit extraction**

```bash
git add tldw_chatbook/Library/collections_capture_repository.py Tests/Library/test_collections_capture_extraction.py
git commit -m 'feat(collections): extract local captures after commit'
```

### Task 8: Add crash-safe offline copies without a new file framework

**Files:**
- Create: `tldw_chatbook/Library/collections_offline_store.py`
- Modify: `tldw_chatbook/Library/collections_capture_repository.py`
- Create: `Tests/Library/test_collections_offline_store.py`
- Reuse unchanged: `tldw_chatbook/Utils/private_paths.py`

- [ ] **Step 1: Write containment, quota, and crash-matrix failures**

Cover 50 MiB/copy and 1 GiB/authority admission, concurrent reservations, ready+staging quota
accounting, absolute/`..`/symlink/root-escape rejection, cross-authority access, owner-only modes,
hash/size validation, crash before publish, crash after publish before ready, missing ready file,
purge tombstones, abandoned temporary files, and bounded cursor-resumable batches.

- [ ] **Step 2: Run and witness failure**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_collections_offline_store.py -q
```

Expected: collection/import failure.

- [ ] **Step 3: Implement the two-phase lifecycle with existing primitives**

Derive `collections_archives/<authority-fingerprint>/` under the resolved private user-data root and
call `secure_private_directory`, `atomic_private_write_bytes`, `open_private_binary`, and
`unlink_private_file`. Store only normalized relative names. Use this ordered protocol:

```text
1. transaction: validate quota and insert staging reservation
2. off-loop: write+fsync temporary sibling and atomically publish final file
3. transaction: verify reservation and mark ready with hash/size/type
4. startup batch: reconcile staging, ready-missing, purge, and abandoned temp state
```

Hard delete first makes capture/file rows inaccessible through a purge tombstone, then removes the
file, then finishes owned-row deletion. Each batch commits a durable cursor and has a fixed item
limit; no unbounded startup scan. If the seam is unavailable, return capability reason
`offline_store_unavailable` and leave the capture unchanged.

- [ ] **Step 4: Run file and private-path tests**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_collections_offline_store.py Tests/Utils/test_private_paths.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the managed-file seam**

```bash
git add tldw_chatbook/Library/collections_offline_store.py tldw_chatbook/Library/collections_capture_repository.py Tests/Library/test_collections_offline_store.py
git commit -m 'feat(collections): manage private offline capture copies'
```

### Task 9: Preserve legacy v1 data through a recovery-only seam

**Files:**
- Create: `tldw_chatbook/Library/collections_legacy_recovery.py`
- Modify: `tldw_chatbook/Library/library_collections_service.py`
- Create: `Tests/Library/test_collections_legacy_recovery.py`
- Modify: `Tests/Library/test_library_collections_service.py`

- [ ] **Step 1: Write complete recovery and read-only failures**

Seed more than 40 active/deleted collections and memberships. Assert bounded inspection reaches all
pages and distinguishes deleted rows. Export to a validated user-selected path and assert one
coherent snapshot, stable order, every record beyond page 1, preserved stored text, no invented
capture fields, exact envelope/version, and atomic publication. Inject a writer during export and
prove the file is entirely pre- or post-write. Reject unsafe paths and prove logs exclude values.

Assert generic create/rename/membership/delete/restore entry points produce structured
`legacy_read_only` and never call capture APIs.

- [ ] **Step 2: Run and witness failure**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_collections_legacy_recovery.py Tests/Library/test_library_collections_service.py -q
```

Expected: FAIL because recovery/export and read-only cutover are absent.

- [ ] **Step 3: Implement bounded inspection and streaming export**

Expose a headless `export_json(destination, *, overwrite_identity)` method. Revalidate the supplied
path with `validate_path_simple(..., require_exists=False)` and privacy-safe failure copy; the reader
task owns `EnhancedFileSave` and overwrite confirmation. Capture whether the target is missing or
its confirmed `(st_dev, st_ino)`. Use `db.read_transaction()` for the entire export, write JSON incrementally to a
mode-`0o600` sibling created by `tempfile.mkstemp`, flush and `fsync`, recheck the target is still
missing or has the same identity, and publish with one `os.link` no-clobber or `os.replace` overwrite.
After successful `os.link`, unlink the temporary sibling before reporting success; tests assert no
temporary export remains after either publication path.
Remove the sibling on every failure and never log the selected path or record values. The exact
envelope is:

```json
{
  "format": "tldw-chatbook-legacy-collections",
  "version": 1,
  "exported_at": "<UTC timestamp>",
  "collections": [],
  "memberships": []
}
```

Collections order by `collection_id`; memberships by `collection_id, membership_id`. Keep the
recovery service usable when capture schema is too new only after verifying the expected v1 tables.
Convert current legacy mutations to one typed error path:

```python
class LegacyCollectionsReadOnlyError(LibraryCollectionsServiceError):
    reason = "legacy_read_only"
```

Each compatibility mutation raises this typed error before beginning a transaction; current callers
serialize/display its stable `reason` and safe recovery copy. Retain inspection methods only for the
recovery widget and compatibility tests.

- [ ] **Step 4: Run legacy, migration, and no-data-loss tests**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_collections_legacy_recovery.py Tests/Library/test_library_collections_service.py Tests/DB/test_library_collections_capture_migration.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit legacy recovery**

```bash
git add tldw_chatbook/Library/collections_legacy_recovery.py tldw_chatbook/Library/library_collections_service.py Tests/Library/test_collections_legacy_recovery.py Tests/Library/test_library_collections_service.py
git commit -m 'feat(collections): preserve legacy data for recovery only'
```

### Task 10: Normalize Local and Server behind a capture-specific scope service

**Files:**
- Create: `tldw_chatbook/Library/collections_capture_service.py`
- Create: `tldw_chatbook/Library/server_collections_capture_service.py`
- Create: `Tests/Library/test_collections_capture_scope_service.py`
- Create: `Tests/Library/test_server_collections_capture_service.py`
- Modify: `Tests/tldw_api/test_media_reading_client.py`

- [ ] **Step 1: Write authority and capability tests first**

Parameterize Local and Server fakes over list/detail/save/update/favorite/status/tags/freeform note,
archive/Undo, saved searches, highlights, and linked Notes. Prove Local profile/database identity,
Server profile/principal identity, dataset replacement, no merging, no workspace partitioning,
source-neutral layout preferences, and late-result fences. Prove capability unknown/supported/
unsupported reasons, exact-true snapshot attestation, malformed/missing/false refusal, cache keys by
profile/principal/API version/capability snapshot, credential invalidation, safe-probe downgrade, and
feature-route 404 isolation.

Inject Media and Note owner resolvers and prove detail-open outcomes for available, missing,
deleted/moved, unauthorized, and transient-failure references. A missing reference never removes or
mutates the capture. For Server note links, reject a different authority unless the Reading API
explicitly returns and the Notes owner recognizes that authority; the current API therefore accepts
only the active capture's server profile/principal and never a workspace authority.

Server save tests must distinguish confirmed save, response error, and transport-unknown outcome;
unknown never retries automatically. A confirmed save followed by placement/detail failure remains
confirmed and marks the page stale.

- [ ] **Step 2: Run and witness missing adapters**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_collections_capture_scope_service.py Tests/Library/test_server_collections_capture_service.py -q
```

Expected: collection/import failure.

- [ ] **Step 3: Implement one UI-facing protocol and two thin adapters**

Define only operations required by this reader:

```python
class CollectionsCaptureBackend(Protocol):
    async def list_page(self, request: CapturePageRequest) -> CapturePage: ...
    async def get_detail(self, identity: CaptureIdentity) -> CaptureDetail: ...
    async def save_capture(self, request: CaptureSaveRequest) -> CaptureSaveOutcome: ...
    async def update_capture(self, identity, expected_revision, changes) -> CaptureDetail: ...
    async def retry_extraction(self, identity: CaptureIdentity) -> CaptureActionResult: ...
    async def list_saved_searches(self, page: int, size: int = 20) -> CaptureSavedSearchPage: ...
    async def save_saved_search(self, search: SavedCaptureSearch) -> SavedCaptureSearch: ...
    async def delete_saved_search(self, search_id: str) -> CaptureActionResult: ...
    async def list_highlights(self, identity: CaptureIdentity) -> tuple[CaptureHighlight, ...]: ...
    async def save_highlight(self, identity: CaptureIdentity, draft: CaptureHighlightDraft) -> CaptureHighlight: ...
    async def delete_highlight(self, identity: CaptureIdentity, highlight_id: str) -> CaptureActionResult: ...
    async def list_note_links(self, identity: CaptureIdentity) -> tuple[CaptureNoteLink, ...]: ...
    async def link_note(self, identity: CaptureIdentity, note: ExternalNoteReference) -> CaptureNoteLink: ...
    async def unlink_note(self, identity: CaptureIdentity, link_id: str) -> CaptureActionResult: ...
    async def save_offline_copy(self, identity: CaptureIdentity) -> CaptureOfflineCopy: ...
    async def delete_offline_copy(self, identity: CaptureIdentity) -> CaptureActionResult: ...
    async def summarize(self, identity: CaptureIdentity) -> CaptureContentResult: ...
    async def listen(self, identity: CaptureIdentity) -> CaptureContentResult: ...
    async def hard_delete(self, identity: CaptureIdentity, expected_revision: int) -> CaptureActionResult: ...
    async def capabilities(self) -> CaptureCapabilities: ...
    async def probe_capability(self, action: str) -> CaptureCapability: ...
```

The Local adapter uses `asyncio.to_thread` around repository/file/extraction operations. The Server
adapter calls existing `TLDWAPIClient` Reading methods and maps Pydantic responses without passing
through `MediaReadingScopeService`. It reads docs-info through the current client, accepts only
`capabilities.get("hasReadingSnapshotPagesV1") is True`, and reports
`server_page_snapshot_unavailable` otherwise. Do not infer support from a well-shaped page.

The UI uses a source-neutral sort allowlist. Map it exactly at the Server boundary and implement the
same semantics in Local SQL:

```python
SERVER_SORT = {
    "saved_desc": "created_desc",
    "saved_asc": "created_asc",
    "updated_desc": "updated_desc",
    "updated_asc": "updated_asc",
    "title_asc": "title_asc",
    "title_desc": "title_desc",
    "relevance": "relevance",
}
```

Permit `relevance` only with nonblank search text. Every adapter method first checks the action's
tri-state capability; unsupported implementations return the source-owned reason and unknown
destructive/data-creating actions remain disabled. Archive and Undo use `update_capture(status=...)`
and therefore share revision/fence behavior rather than adding a second mutation path.

The scope service holds one active `CaptureAuthority`, clears page/detail/saved-search snapshots on
authority changes, retains archive receipts keyed by originating authority, and never includes
workspace in a Server key. Raw path/principal material is hashed into the opaque key and fingerprint
and is not logged.

`CollectionsCaptureScopeService.get_detail()` wraps the backend's raw `CaptureDetail` in
`ResolvedCaptureDetail`. It accepts two injected async callables—`resolve_media_reference` and
`resolve_note_reference`—that delegate to the owning Media and Notes services. Each returns a
bounded `ExternalReferenceAvailability`: not-found/deleted/moved/permission failures become
`unavailable` with a safe reason, while unexpected transport failures remain a retryable detail
substate. The Server adapter validates note-link authority before invoking the Notes resolver.

- [ ] **Step 4: Run contract and client tests**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_collections_capture_scope_service.py Tests/Library/test_server_collections_capture_service.py Tests/tldw_api/test_media_reading_client.py -q
```

Expected: PASS with more than 40 rows reaching pages 2 and 3 for both supported backends.

- [ ] **Step 5: Commit the authority seam**

```bash
git add tldw_chatbook/Library/collections_capture_service.py tldw_chatbook/Library/server_collections_capture_service.py Tests/Library/test_collections_capture_scope_service.py Tests/Library/test_server_collections_capture_service.py Tests/tldw_api/test_media_reading_client.py
git commit -m 'feat(collections): unify local and server capture authority'
```

### Task 11: Build the fenced Collections session/controller

**Files:**
- Create: `tldw_chatbook/UI/Library_Modules/library_collections_capture_controller.py`
- Create: `Tests/UI/test_library_collections_capture_controller.py`

- [ ] **Step 1: Write state-machine examples before controller code**

Cover requested/applied scope, first authoritative selection, selected-versus-loaded identity,
injected settle delay, Enter bypass, A-retained-while-B-loads copy, identity-sensitive action
disablement, one guarded last-page retry after shrink, repeated-shrink stale recovery, list/detail/
mutation/extraction generations, source/unmount invalidation, mutation-before-read invalidation,
conflict draft preservation, stale last-good rows, exact-total suppression, archive receipt/Undo
placement, and authority-hidden receipts.

- [ ] **Step 2: Run and witness failure**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_collections_capture_controller.py -q
```

Expected: collection/import failure.

- [ ] **Step 3: Implement immutable state plus one orchestration owner**

Use frozen state snapshots and injected `detail_settle_seconds`/clock for deterministic tests. Every
completion carries this fence:

```python
@dataclass(frozen=True)
class CaptureRequestFence:
    destination: Literal["collections"]
    authority_key: str
    scope_key: str
    item_id: str | None
    revision: str | int | None
    generation: int
```

Apply a result only when the complete fence equals current state. The controller calls only
`CollectionsCaptureScopeService`, returns renderable state/messages to the screen, and contains no
Textual widget queries. Reconcile only fields proven by a mutation response; any failed follow-up
keeps the last good page stale and disables totals/paging/identity-sensitive actions.

- [ ] **Step 4: Run controller tests**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_collections_capture_controller.py -q
```

Expected: PASS without fixed sleeps.

- [ ] **Step 5: Commit orchestration**

```bash
git add tldw_chatbook/UI/Library_Modules/library_collections_capture_controller.py Tests/UI/test_library_collections_capture_controller.py
git commit -m 'feat(collections): fence capture reader sessions'
```

### Task 12: Mount the adaptive Library/Items/Work reader

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_collections_capture_reader.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/__init__.py`
- Create: `Tests/UI/test_library_collections_capture_reader.py`
- Create: `Tests/UI/test_library_collections_reader_geometry.py`
- Reuse unchanged: `tldw_chatbook/Widgets/enhanced_file_picker.py`

- [ ] **Step 1: Write mounted behavior and geometry failures**

Use `Tests.UI.consolidated_css.ConsolidatedCSSApp` and the real Library hierarchy. Cover contextual
All/Saved/Reading/Read/Archived/Favorites/saved-search rows, bounded More continuation, compact Items
rows, Quick Capture, filters/sorts, pages, empty/error/stale states, selected/loaded copy, Read/
Highlights/Notes/Info modes, freeform versus linked Notes, provenance, missing external references,
capability reasons, toolbar More overflow, archive/Undo, hard-delete confirmation, legacy recovery,
both grips, Escape graduation, pointer/keyboard paths, and focus evacuation/restoration.

Pin the pure resolver profile and exact outputs:

```python
profile = AdaptiveReaderLayoutProfile(work_min_width=48, work_comfort_width=56)
expected = {
    160: (30, 40, 80),
    120: (0, 56, 54),
    100: (0, 42, 48),
    80: (0, 0, 70),
}
```

Mounted 160x50/120x35/100x30/80x24 tests must read the settled
`shell.content_size.width`, resolve once from that measured width, and assert exact Library, Items,
Work, and both grip regions—never terminal width and never two acceptable layouts.

- [ ] **Step 2: Run and witness missing widgets**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_collections_capture_reader.py Tests/UI/test_library_collections_reader_geometry.py -q
```

Expected: collection/import failure or old generic-container assertions.

- [ ] **Step 3: Compose the existing shell and destination-owned panes**

Use one `LibraryAdaptiveReaderShell` with Collections profile 48/56. Keep global Library navigation
in the Library slot and add contextual scope rows only while Collections is active. Build Items and
Work in the new widget file; do not subclass Media widgets or put capture state in the shared shell.
Render content as inert Textual text/Markdown with markup disabled or escaped. Never show URL query
strings in rows. Keep Work mounted in every geometry and expose labelled restore controls for both
optional panes.

The screen owns workers and focus but delegates all capture transitions to the controller. On
source change clear visible page/detail/total before requesting the new authority. On unmount,
invalidate every generation and cancel/ignore delayed detail work.

**More > Legacy Collections data…** mounts the recovery inspector whenever compatible v1 rows
exist. The widget owns `EnhancedFileSave`, overwrite confirmation, and the captured target identity;
it passes only the confirmed destination and precondition to the headless legacy recovery service.
Cancellation writes nothing and returns focus to the recovery action.

- [ ] **Step 4: Run mounted and shared-shell regressions**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_collections_capture_reader.py Tests/UI/test_library_collections_reader_geometry.py Tests/UI/test_library_adaptive_reader_shell.py Tests/Library/test_library_adaptive_reader_state.py -q
```

Expected: PASS with no horizontal overflow.

- [ ] **Step 5: Commit the reader topology**

```bash
git add tldw_chatbook/Widgets/Library/library_collections_capture_reader.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/__init__.py Tests/UI/test_library_collections_capture_reader.py Tests/UI/test_library_collections_reader_geometry.py
git commit -m 'feat(collections): mount adaptive capture reader'
```

### Task 13: Wire preferences, services, extraction recovery, and app lifecycle

**Files:**
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/Library/__init__.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/Library/test_library_collections_config.py`
- Create: `Tests/Library/test_collections_capture_app_wiring.py`
- Modify: `Tests/UI/test_profile_owned_settings_paths.py`

- [ ] **Step 1: Write configuration and lifecycle tests first**

Assert `[library.collections_reader]` owns only `items_open`/`items_width`; `[library.reader]` owns
Library open/width/custom opt-in; defaults are fixed 40 with custom widths disabled; responsive
collapse never persists; source-neutral preferences survive source switches. Assert Local authority
uses the resolved profile and configured existing `library_collections_db_path`; Server uses current
profile/principal; workspace changes do not rewire Server. Assert startup interruption recovery and
bounded offline scavenging run off-loop, and teardown invalidates/cancels work. Assert Media and
Media provenance resolution delegates to
`media_reading_scope_service.get_backing_media_item()` with the stored backing Media ID—not the
Reading capture ID—and Note reference resolution delegates to
`notes_scope_service.get_note_detail()` with the capture's Local/Server authority. Assert Server
Media provenance reaches the backing Media `get_media_item` client path and never
`get_reading_item`; Local reaches Local Media detail. Both resolvers produce explicit unavailable
states on missing/unauthorized records, and the Note resolver never sends a Server workspace scope.

- [ ] **Step 2: Run and witness failure**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_collections_config.py Tests/Library/test_collections_capture_app_wiring.py Tests/UI/test_profile_owned_settings_paths.py -q
```

Expected: FAIL because the destination preference and capture wiring are absent.

- [ ] **Step 3: Add the minimum configuration and wiring**

Extend the existing destination reader normalization tuple with `collections_reader`; do not add a
legacy fallback. Add the default section:

```toml
[library.collections_reader]
items_open = true
items_width = 40
```

Wire `CollectionsCaptureRepository`, legacy recovery, Local adapter, Server adapter from the
existing server-context provider, and `CollectionsCaptureScopeService`. Expose explicit app
attributes such as `collections_capture_scope_service` and `collections_legacy_recovery_service`;
do not reuse `library_collections_service` for captures. Schedule interruption/file reconciliation
through existing background-worker ownership after the profile path is resolved.

Build two narrow async resolver callables at wiring time instead of a new reference framework. The
Media resolver maps Local/Server authority to `MediaReadingBackend` and calls
`self.media_reading_scope_service.get_backing_media_item(mode=..., media_id=...)` with the stored
backing Media ID. That scope method routes Server mode to the backing Media `get_media_item`
operation, not the Reading List `get_reading_item` operation, and routes Local mode to Local Media
detail. The Note resolver maps to `ScopeType.LOCAL_NOTE` or `ScopeType.SERVER_NOTE` and calls
`self.notes_scope_service.get_note_detail(...)`; Local supplies the current local user id and Server
supplies no workspace. Both verify the returned source id before marking the reference available and
map owner not-found/deleted/permission exceptions to bounded reasons.

- [ ] **Step 4: Run configuration, app, and Library smoke tests**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_collections_config.py Tests/Library/test_collections_capture_app_wiring.py Tests/UI/test_profile_owned_settings_paths.py Tests/UI/test_library_entry_compose_once.py -q
```

Expected: PASS; Library still composes once.

- [ ] **Step 5: Commit lifecycle wiring**

```bash
git add tldw_chatbook/config.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/app.py tldw_chatbook/Library/__init__.py tldw_chatbook/UI/Screens/library_screen.py Tests/Library/test_library_collections_config.py Tests/Library/test_collections_capture_app_wiring.py Tests/UI/test_profile_owned_settings_paths.py
git commit -m 'feat(collections): wire capture authority and preferences'
```

### Task 14: Complete cutover, CSS, and cross-reader regression coverage

**Files:**
- Delete: `tldw_chatbook/Library/library_collections_state.py`
- Delete: `tldw_chatbook/UI/Library_Modules/library_collections_browse_controller.py`
- Delete: `tldw_chatbook/Widgets/Library/library_collections_panel.py`
- Modify: `tldw_chatbook/Library/library_tool_contract.py`
- Modify: `tldw_chatbook/Library/local_library_tool_service.py`
- Modify: `tldw_chatbook/MCP/server.py`
- Modify: `tldw_chatbook/UI/Console_Modules/library_activity.py`
- Modify: `tldw_chatbook/runtime_policy/registry.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/MCP/test_library_tools.py`
- Modify: `Tests/MCP/test_local_control_service.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_core.py`
- Modify: `Tests/Library/test_local_library_tool_service.py`
- Delete or replace: `Tests/Library/test_library_collections_state.py`
- Delete or replace: `Tests/UI/test_library_collections_browse_controller.py`
- Delete or replace: `Tests/Widgets/test_library_collections_panel.py`
- Modify: `Tests/UI/test_library_adaptive_reader_closeout.py`
- Modify: `Tests/Live/test_library_adaptive_reader_closeout.py`
- Modify or replace: `Tests/UI/test_product_maturity_phase39_library_collections.py`
- Modify: `Docs/superpowers/reviews/2026-08-31-library-collections-cutover-inventory.md`

- [ ] **Step 1: Turn the inventory into failing cutover assertions**

Assert no current tool, MCP description, Home/rail copy, RAG help, or Library action describes
generic containers as Collections. The Python compatibility methods `create_collection`,
`rename_collection`, `add_item_to_collection`, `delete_collection`, and `restore_collection` must
remain callable on `LocalLibraryCollectionsService` and return the structured reason
`legacy_read_only`; they are never absent and never save captures. Recovery must remain reachable whenever v1 rows exist, including with
schema version greater than 2. Convert obsolete phase-39 tests to capture reader tests or move their
still-valid legacy assertions into the recovery suite; do not preserve tests for the retired UI.
Assert `CAPABILITY_REGISTRY` has no `library.collections.*` entries and no visible “Library
Collections & agent tools” label, while the unrelated `library.templates`, `library.media`, and
`library.notes` policy entries remain unchanged.

- [ ] **Step 2: Run the cutover tests and witness old references**

```bash
../../.venv/bin/python -m pytest Tests/MCP/test_library_tools.py Tests/MCP/test_local_control_service.py Tests/RuntimePolicy/test_runtime_policy_core.py Tests/Library/test_local_library_tool_service.py Tests/UI/test_product_maturity_phase39_library_collections.py -q
```

Expected: FAIL on old generic-container behavior/copy.

- [ ] **Step 3: Retire the old product surfaces and style the new reader**

Follow every inventory row. Keep only the distinctly labelled legacy recovery adapter. Delete the
old generic state/controller/panel after all route imports and handlers point to the capture reader;
remove `collection` from the generic Library item/tool contract and stop constructing that backend
in MCP/Console activity. This retires current list/get/search tool exposure only; it does not remove
the compatibility mutation methods on `LocalLibraryCollectionsService`, which remain the explicit
`legacy_read_only` boundary. In `runtime_policy/registry.py`, remove only the
`_resource("library.collections", actions=(LIST, DETAIL))` row, rename the visible group to
“Library agent tools (local),” and preserve the stable internal capability id plus the unrelated
template/media/notes resources so saved policy configuration does not need a migration. Update the
policy tests to assert absence rather than remapping old ids to `collections.reading_list.*`.
Update `_agentic_terminal.tcss` for compact capture rows, explicit stale/error/loading states, quiet
Work typography, mode toolbar, More overflow, recovery inspector, and both focused grips. Use the
existing theme variables and shared shell selectors; do not introduce a second visual system.

Regenerate CSS with the repository's builder discovered after rebase:

```bash
../../.venv/bin/python -m tldw_chatbook.css.build_css
```

Expected: the modular bundle changes only through generation and contains the new selectors.

- [ ] **Step 4: Run production-shaped cross-reader suites**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_adaptive_reader_closeout.py Tests/UI/test_library_collections_reader_geometry.py Tests/UI/test_library_media_reader_shell.py Tests/UI/test_library_media_reader_flow.py Tests/UI/test_library_adaptive_reader_shell.py -q
```

Expected: PASS for Collections, Media, Conversations, Notes, Prompts, and Skills at the production
hierarchy/stylesheet boundary. Add the exact destination-specific files discovered in the existing
closeout parametrization rather than creating a parallel harness.

- [ ] **Step 5: Run the focused feature regression set**

```bash
../../.venv/bin/python -m pytest Tests/DB/test_library_collections_capture_migration.py Tests/Library/test_collections_capture_models.py Tests/Library/test_collections_capture_repository.py Tests/Library/test_collections_capture_extraction.py Tests/Library/test_collections_offline_store.py Tests/Library/test_collections_legacy_recovery.py Tests/Library/test_collections_capture_scope_service.py Tests/Library/test_server_collections_capture_service.py Tests/UI/test_library_collections_capture_controller.py Tests/UI/test_library_collections_capture_reader.py Tests/UI/test_library_collections_reader_geometry.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the cutover**

```bash
git add tldw_chatbook/Library/library_collections_state.py tldw_chatbook/Library/library_tool_contract.py tldw_chatbook/Library/local_library_tool_service.py tldw_chatbook/MCP/server.py tldw_chatbook/UI/Console_Modules/library_activity.py tldw_chatbook/UI/Library_Modules/library_collections_browse_controller.py tldw_chatbook/Widgets/Library/library_collections_panel.py tldw_chatbook/runtime_policy/registry.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/MCP/test_library_tools.py Tests/MCP/test_local_control_service.py Tests/RuntimePolicy/test_runtime_policy_core.py Tests/Library/test_local_library_tool_service.py Tests/Library/test_library_collections_state.py Tests/UI/test_library_collections_browse_controller.py Tests/Widgets/test_library_collections_panel.py Tests/UI/test_library_adaptive_reader_closeout.py Tests/Live/test_library_adaptive_reader_closeout.py Tests/UI/test_product_maturity_phase39_library_collections.py Docs/superpowers/reviews/2026-08-31-library-collections-cutover-inventory.md
git commit -m 'feat(collections): retire generic containers from current surfaces'
```

### Task 15: Perform isolated live walkthroughs and close TASK-18919

**Files:**
- Create: `Docs/superpowers/reviews/2026-08-31-library-collections-live-verification.md`
- Modify: `Docs/superpowers/reviews/2026-08-31-library-collections-cutover-inventory.md`
- Modify: `backlog/tasks/task-18919 - Build-the-Local-and-Server-Collections-capture-reader.md`

- [ ] **Step 1: Verify the branch before live data creation**

```bash
git diff --check origin/dev...HEAD
../../.venv/bin/python -m pytest Tests/Live/test_library_adaptive_reader_closeout.py -q
```

Expected: PASS. Do not run the entire repository suite unless the user explicitly opts into a full
sweep; the production-shaped and focused cross-reader suites above are mandatory either way.

- [ ] **Step 2: Walk through isolated Local authority at all four sizes**

Use a temporary config/data root and a seeded Local database with at least 45 captures. Record the
resolved database path fingerprint before and after. At 160x50, 120x35, 100x30, and 80x24 record
terminal size, measured shell width, requested state, mode, Library/Items/Work/grip geometry, focus,
and overflow. Cover untouched startup, route activation, all collapse combinations, resize
restoration, keyboard traversal, pages 1–3, Quick Capture commit-before-extract, extraction failure/
Retry, Read/Highlights/Notes/Info, archive/Undo, hard-delete tombstone cleanup, legacy inspection,
and a complete export beyond page 1. Use condition-based waits.

- [ ] **Step 3: Walk through enabled Server authority at all four sizes**

Use an isolated Server profile/principal against a deployment containing PR A and at least 45
captures. First record docs-info exact `hasReadingSnapshotPagesV1: true`; if absent, record the
expected blocked state and stop without bypassing it. Identify returned captures by authoritative
content, not lack of exceptions. Cover pages 1–3, source replacement, workspace non-effect,
confirmed save, simulated/controlled unknown outcome with no retry, explicit retry warning,
capability reasons, modes, archive/Undo, and source switch back to Local. Record no credentials,
URLs with query data, bodies, private paths, or stable principal ids.

- [ ] **Step 4: Self-review against the spec and security boundaries**

```bash
git diff --stat origin/dev...HEAD
git diff --check origin/dev...HEAD
rg -n 'MediaReadingScopeService|save_to_read_it_later' tldw_chatbook/Library tldw_chatbook/UI/Library_Modules tldw_chatbook/Widgets/Library
rg -n 'library_collections|collection_items' tldw_chatbook | rg -v 'legacy|recovery|DB/Library_Collections_DB.py'
```

Expected: capture code does not use the Media normalizer; remaining v1 names are justified in the
signed inventory. Review URL/path/body logging, SQL allowlists, HTML/markup escaping, capability
fail-closed behavior, generation fences, file containment, and destructive confirmations.

- [ ] **Step 5: Complete task evidence only after all acceptance criteria pass**

Check every acceptance criterion, add concise Implementation Notes with both PR links/commits,
tests, live evidence, ADR-107, trade-offs, and modified files, and add a lessons entry only if this
work produced a concrete reusable incident. Then:

```bash
backlog task edit 18919 -s Done
git add 'backlog/tasks/task-18919 - Build-the-Local-and-Server-Collections-capture-reader.md' Docs/superpowers/reviews/2026-08-31-library-collections-live-verification.md Docs/superpowers/reviews/2026-08-31-library-collections-cutover-inventory.md
git commit -m 'docs(collections): close capture reader verification'
```

Expected: TASK-18919 is Done only after all ten acceptance criteria, both repository prerequisites,
focused automated evidence, production-shaped cross-reader suites, and Local/Server live evidence
are complete.

## Acceptance-criteria trace

| TASK-18919 criterion | Implementation tasks |
| --- | --- |
| #1 capture product, not folders/Media | 3, 9, 14 |
| #2 explicit Local/Server authority | 5, 10, 13 |
| #3 Library/Items/Work topology | 12, 13 |
| #4 exact scoped paging and server attestation | 1, 2, 5, 6, 10 |
| #5 trustworthy Read/Highlights/Notes/Info | 6, 7, 10, 12 |
| #6 Local commit-first and Server unknown outcome | 7, 10, 11, 12 |
| #7 schema, revisions, offline files, legacy recovery | 4, 6, 8, 9 |
| #8 tri-state capabilities and destructive safety | 8, 10, 11, 12 |
| #9 loading/recovery/focus/exact geometry | 11, 12, 14, 15 |
| #10 shared contracts, concurrent writer, production/live | 1, 2, 10, 14, 15 |
