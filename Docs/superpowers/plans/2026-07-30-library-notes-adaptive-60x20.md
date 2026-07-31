# Library Notes Adaptive 60×20 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing Library Database Notes workflow lossless and fully keyboard-usable at 60×20 while preserving its capable wide-screen utilities and current storage, sync, export, and Console-handoff boundaries.

**Architecture:** Add the ADR-027 host-independent `DatabaseNoteSessionCoordinator` as the sole owner of the active Database Note draft, validation, serialized saves, conflict gating, destructive admission, and flush outcomes. `LibraryScreen` remains the Textual host and service adapter; `LibraryNotesCanvas` remains presentation-only, with stable Edit/Preview/Context surfaces and an idempotent state-application seam. Compact behavior is selected from the measured outer `#library-shell-grid` width, while the existing wide workflow retains direct inline utilities.

**Tech Stack:** Python 3.11+, Textual `>=8.0.0,<9`, Rich cell-width helpers, asyncio, pytest/pytest-asyncio, Textual Pilot, TCSS, Backlog.md

## Global Constraints

- Do not begin Task 1 until dependency TASK-400 is Done and both supported manifests constrain Textual to `>=8.0.0,<9`.
- Preserve the existing Library route, Database Note storage/schema, sync ownership, export formats, immediate-create behavior, and Console handoff.
- Preserve direct wide access to keywords, metadata, Use in Console, Copy, Markdown/text export, and Delete.
- Do not import Textual, File Notes, database handles, navigation, or application globals from `library_notes_session.py`.
- Never truncate, strip, sanitize, or otherwise rewrite a raw draft and then report it saved. Invalid input is a typed veto and remains visible and dirty.
- No save cancellation as serialization. An already-running threaded service call cannot be cancelled safely.
- Do not add a dedicated Notes route/workbench, backlinks placeholders, history/undo, crash-draft persistence, schema changes, or a generic controller shared with File Notes.
- Reuse `Tests/UI/test_library_shell.py::LibraryHarness`; do not create another Library harness or extract shared test infrastructure in this task.
- Runtime tests use only temporary profiles, databases/directories, synthetic notes, and stub services.
- Preserve unrelated dirty-worktree changes and stage only files named by the current task.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/027-portable-database-note-session-coordinator.md`

Reason: ADR-027 already records the new long-lived, host-independent Database
Note draft/save/conflict/destructive/flush interface and its separation from
Textual presentation and ADR-021 File Notes authority. No additional ADR is
needed unless implementation changes that boundary.

## File and Interface Map

### Create

- `tldw_chatbook/Library/library_notes_session.py`
  - `DatabaseNoteSessionPort` protocol.
  - Normalized load/save reply types.
  - `DatabaseNoteSessionCoordinator`.
  - Serialized/coalesced save driver, conflict gates, typed flush outcomes,
    untouched-create eligibility, and destructive admission.
- `Tests/Library/test_library_notes_session.py`
  - Textual-free coordinator tests with a controllable async fake port.
- `Docs/superpowers/qa/library-notes-adaptive-60x20/capture_library_notes.py`
  - Synthetic Pilot capture/soak driver.
- `Docs/superpowers/qa/library-notes-adaptive-60x20/README.md`
  - Final keyboard UAT and ADR-011 evidence.

### Modify

- `tldw_chatbook/Library/library_notes_state.py`
  - Immutable normalized detail/draft/session/focus/display types.
  - Lossless payload validation.
  - Plain one-row cell-width ellipsis.
  - Truthful Navigator/status helpers.
- `tldw_chatbook/UI/Screens/library_screen.py`
  - Own the coordinator and Textual service-port adapter.
  - Autosave debounce, workflow/region transitions, responsive breakpoint,
    focus capture/restore, footer context, and central recompose
    capture/rehydration.
  - Remove the legacy competing draft/save/conflict snapshots after the
    coordinator path is green.
- `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
  - Stable Navigator/Edit/Preview/Context/Create/Sync presentation.
  - Persistent labels, direct choice controls, compatible wide utilities,
    focusable scroll owners, and idempotent `apply_session_state`.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Authoritative compact/wide Notes geometry and state styles.
- `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Generated output only; regenerate with `build_css.py`.
- `Tests/Library/test_library_notes_state.py`
  - Pure validation, display, focus, and ellipsis contracts.
- `Tests/UI/test_library_shell.py`
  - Real-bundle Pilot, concurrency, accessibility, geometry, lifecycle, and
    responsive contracts.
- `Tests/UI/test_css_build_integrity.py`
  - Source/bundle and fallback/source parity contracts.
- `backlog/tasks/task-1333 - Adapt-Library-Notes-for-lossless-60x20-workflow.md`
  - Plan link, completed acceptance criteria, implementation evidence, and
    final status only after all gates pass.

### Explicitly Not Modified

- Database schema or migrations.
- `tldw_chatbook/Notes/sync_engine.py` and sync policy.
- ADR-021 File Notes controller/repository.
- A standalone Notes route.

---

### Task 0: Enforce the TASK-400 runtime prerequisite

**Files:**
- Read: `backlog/tasks/task-400 - Fix-MCP-navigation-crash-by-requiring-Textual-8.md`
- Read: `pyproject.toml:41`
- Read: `requirements.txt:21`
- Read: `.github/workflows/test.yml`
- Test: `Tests/CI/test_textual_runtime_contract.py`

**Interfaces:**
- Consumes: TASK-400's supported Textual runtime contract.
- Produces: a hard go/no-go decision for TASK-1333 implementation.

- [ ] **Step 1: Verify TASK-400 is complete**

Run:

```bash
backlog task 400 --plain
```

Expected: `Status: Done`, with every acceptance criterion checked and
implementation notes present. If not, stop TASK-1333 execution and complete
TASK-400 through its own approved plan first.

- [ ] **Step 2: Verify both manifests and the exact-minimum CI lane**

Run:

```bash
rg -n 'textual>=8\\.0\\.0,<9|textual==8\\.0\\.0|textual-minimum' \
  pyproject.toml requirements.txt .github/workflows/test.yml
```

Expected: the supported manifests contain `textual>=8.0.0,<9` and CI contains
an exact `textual==8.0.0` minimum-version lane.

- [ ] **Step 3: Run the runtime contract**

Run:

```bash
.venv/bin/python -m pytest -q Tests/CI/test_textual_runtime_contract.py --tb=short
```

Expected: PASS. Do not make TASK-400 changes in a TASK-1333 commit.

---

### Task 1: Define pure lossless Notes state contracts

**Files:**
- Modify: `tldw_chatbook/Library/library_notes_state.py`
- Modify: `Tests/Library/test_library_notes_state.py`

**Interfaces:**
- Produces: `NormalizedDatabaseNote`, `DatabaseNoteDraft`,
  `DatabaseNoteSavePayload`, `NoteValidationVeto`,
  `LibraryNoteSessionSnapshot`, `LibraryNotesFocusIdentity`,
  `validate_database_note_draft()`, and `ellipsize_note_title_cells()`.
- Consumes: existing shared input validation without accepting transformed
  output.

- [ ] **Step 1: Write the failing pure-state tests**

Append tests covering:

```python
def test_save_payload_preserves_valid_raw_content_exactly():
    draft = DatabaseNoteDraft(
        note_id="n-1",
        title="[draft] <plan>",
        body="line 1\\n<script example>\\x3c/script>",
        keywords_text="alpha, βeta",
        revision=7,
    )
    result = validate_database_note_draft(draft)
    assert result == DatabaseNoteSavePayload(
        title="[draft] <plan>",
        body="line 1\\n<script example>\\x3c/script>",
        keywords=("alpha", "βeta"),
        revision=7,
    )


@pytest.mark.parametrize(
    ("draft", "field"),
    (
        (DatabaseNoteDraft("n", "x" * 301, "", "", 1), "title"),
        (DatabaseNoteDraft("n", "ok", "x" * (2_000_000 + 1), "", 1), "body"),
        (DatabaseNoteDraft("n", "ok", "", "x" * 101, 1), "keywords"),
        (DatabaseNoteDraft("n", "bad\\x00title", "", "", 1), "title"),
    ),
)
def test_invalid_or_transforming_payload_is_a_typed_veto(draft, field):
    result = validate_database_note_draft(draft)
    assert isinstance(result, NoteValidationVeto)
    assert result.field == field


def test_keyword_delimiter_whitespace_is_syntax_not_content():
    draft = DatabaseNoteDraft("n", "ok", "", " alpha , , βeta ", 1)
    result = validate_database_note_draft(draft)
    assert isinstance(result, DatabaseNoteSavePayload)
    assert result.keywords == ("alpha", "βeta")


def test_keyword_limit_is_per_semantic_token_not_aggregate_field():
    keywords = ", ".join(f"topic-{index:02d}" for index in range(20))
    assert len(keywords) > DATABASE_NOTE_KEYWORD_MAX_CHARS
    result = validate_database_note_draft(
        DatabaseNoteDraft("n", "ok", "", keywords, 1)
    )
    assert isinstance(result, DatabaseNoteSavePayload)
    assert result.keywords == tuple(
        f"topic-{index:02d}" for index in range(20)
    )


def test_casefold_duplicate_keywords_are_vetoed_not_silently_deduplicated():
    result = validate_database_note_draft(
        DatabaseNoteDraft("n", "ok", "", "Alpha, alpha", 1)
    )
    assert isinstance(result, NoteValidationVeto)
    assert result.field == "keywords"


def test_cell_ellipsis_honors_wide_unicode_and_keeps_raw_title():
    raw = "研究計画 [draft] roadmap"
    visible = ellipsize_note_title_cells(raw, 10)
    assert cell_len(visible) <= 10
    assert visible.endswith("…")
    assert raw == "研究計画 [draft] roadmap"
```

Also add equality/immutability tests for the session snapshot and focus tuple,
plus separate empty-total and empty-filter copy tests.

- [ ] **Step 2: Run the pure-state tests and verify red**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Library/test_library_notes_state.py --tb=short
```

Expected: FAIL on the new imports/types/helpers.

- [ ] **Step 3: Add the immutable types and lossless builder**

Implement frozen dataclasses with these minimum fields:

```python
@dataclass(frozen=True)
class NormalizedDatabaseNote:
    note_id: str
    title: str
    body: str
    keywords: tuple[str, ...]
    version: int
    created_at: str
    modified_at: str


@dataclass(frozen=True)
class DatabaseNoteDraft:
    note_id: str
    title: str
    body: str
    keywords_text: str
    revision: int


@dataclass(frozen=True)
class DatabaseNoteSavePayload:
    title: str
    body: str
    keywords: tuple[str, ...]
    revision: int


@dataclass(frozen=True)
class NoteValidationVeto:
    field: Literal["title", "body", "keywords"]
    message: str
    revision: int
```

Use constants `DATABASE_NOTE_TITLE_MAX_CHARS = 300`,
`DATABASE_NOTE_BODY_MAX_CHARS = 2_000_000`, and
`DATABASE_NOTE_KEYWORD_MAX_CHARS = 100`.

`validate_database_note_draft()` must:

1. Check raw lengths before calling any sanitizer.
2. Reject title-leading/trailing whitespace if the database would strip it.
3. Require `sanitize_string(raw, max_length=limit) == raw`.
4. Use `validate_text_input(..., allow_html=False)` for title and semantic
   keyword tokens.
5. Apply the 100-character limit to each delimiter-trimmed semantic keyword,
   not to the aggregate comma-separated input.
6. Compare keyword identity by `casefold()` because the incumbent service
   deduplicates case-insensitively. If two entered tokens collide, veto the
   draft rather than silently dropping, merging, or re-casing either token;
   otherwise preserve each token's first-entered spelling and order exactly.
7. Permit markup/code in the body while still rejecting a sanitizer
   round-trip mismatch.
8. Return the exact title/body and semantic keyword tokens, never sanitized
   replacement text.

Implement cell ellipsis using `rich.cells.get_character_cell_size` so the
returned string plus `…` never exceeds the requested cell budget.

- [ ] **Step 4: Run the pure-state tests and verify green**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Library/test_library_notes_state.py --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit the pure contracts**

```bash
git add \
  tldw_chatbook/Library/library_notes_state.py \
  Tests/Library/test_library_notes_state.py
git commit -m "feat(notes): define lossless session state"
```

---

### Task 2: Build the portable serialized-save coordinator

**Files:**
- Create: `tldw_chatbook/Library/library_notes_session.py`
- Create: `Tests/Library/test_library_notes_session.py`
- Consume: `tldw_chatbook/Library/library_notes_state.py`

**Interfaces:**
- Produces: `DatabaseNoteSessionPort`, `DatabaseNotePortLoadReply`,
  `DatabaseNotePortSaveReply`, `NoteLoadOutcome`, `NoteSaveOutcome`,
  `NoteFlushOutcome`, and `DatabaseNoteSessionCoordinator`.
- Guarantees: no Textual/File Notes imports and one active save driver.

- [ ] **Step 1: Create a controllable fake port and failing load/mutation tests**

Start `Tests/Library/test_library_notes_session.py` with an async fake that
records calls and can gate each save on an `asyncio.Event`:

```python
class FakeDatabaseNotePort:
    def __init__(self, detail: NormalizedDatabaseNote):
        self.load_reply = DatabaseNotePortLoadReply.loaded(detail)
        self.save_calls: list[tuple[str, int, DatabaseNoteSavePayload]] = []
        self.save_gates: list[asyncio.Event] = []
        self.save_replies: list[DatabaseNotePortSaveReply] = []

    async def load_note(self, note_id: str):
        return self.load_reply

    async def save_note(self, note_id, expected_version, payload):
        self.save_calls.append((note_id, expected_version, payload))
        if self.save_gates:
            await self.save_gates.pop(0).wait()
        return self.save_replies.pop(0)
```

Add failing tests that prove:

- `await open_session(note_id)` loads through the injected port and one
  normalized detail seeds baseline, exact draft, revision 0, and version;
- a stale in-flight `open_session()` cannot replace a newer session;
- typed missing and failed load replies leave no falsely loaded session;
- one genuine mutation increments revision and marks dirty;
- programmatic snapshot reads do not mutate revision;
- construction and normal save work when `textual` imports are blocked;
- an invalid payload makes zero port calls and returns validation veto;
- an explicit no-op Save returns acknowledged success.

- [ ] **Step 2: Run the coordinator tests and verify red**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Library/test_library_notes_session.py --tb=short
```

Expected: FAIL because the coordinator module does not exist.

- [ ] **Step 3: Implement the protocol, typed replies, and session lifecycle**

Define:

```python
class DatabaseNoteSessionPort(Protocol):
    async def load_note(
        self, note_id: str
    ) -> DatabaseNotePortLoadReply: ...

    async def save_note(
        self,
        note_id: str,
        expected_version: int,
        payload: DatabaseNoteSavePayload,
    ) -> DatabaseNotePortSaveReply: ...


class PortSaveKind(StrEnum):
    SAVED = "saved"
    CONFLICT = "conflict"
    FAILED = "failed"


class PortLoadKind(StrEnum):
    LOADED = "loaded"
    MISSING = "missing"
    FAILED = "failed"


class NoteLoadOutcomeKind(StrEnum):
    LOADED = "loaded"
    MISSING = "missing"
    FAILED = "failed"
    STALE = "stale"


@dataclass(frozen=True)
class DatabaseNotePortLoadReply:
    kind: PortLoadKind
    detail: NormalizedDatabaseNote | None = None
    message: str = ""


@dataclass(frozen=True)
class NoteLoadOutcome:
    kind: NoteLoadOutcomeKind
    note_id: str
    message: str = ""


@dataclass(frozen=True)
class DatabaseNotePortSaveReply:
    kind: PortSaveKind
    version: int | None = None
    modified_at: str = ""
    keywords: tuple[str, ...] | None = None
    message: str = ""
```

Require the load reply to carry a detail only for `LOADED`; this keeps an
ordinary service failure distinct from a genuinely missing note. The
coordinator constructor accepts only the injected port and a clock callable.

Expose this unambiguous public load boundary:

```python
async def open_session(
    self,
    note_id: str,
    *,
    untouched_create_token: str | None = None,
) -> NoteLoadOutcome: ...
```

`open_session()` increments and captures a request token before awaiting
`self._port.load_note(note_id)`. Only the current request may apply a `LOADED`
reply. `MISSING`, `FAILED`, and stale replies return typed outcomes without
seeding a session. `invalidate_session_request()` increments the token for
Back, route changes, and note switches. A private
`_start_loaded_session(detail, untouched_create_token=...)` increments the
session generation, stores the normalized baseline, creates the canonical raw
draft, and clears all prior operation tokens. The screen never calls that
private helper or loads through the port itself.

`mutate()` rejects destructive-running mutations, otherwise changes only
fields supplied by the caller, increments the revision exactly once per
genuine value change, marks dirty, and sets `pending_save_requested=True` when
a save is active.

- [ ] **Step 4: Implement the unbounded serialized/coalesced save driver**

Use one `_save_task` and one `_pending_save_requested` flag:

```python
async def request_save(self, *, explicit: bool) -> NoteSaveOutcome:
    if self._destructive is not None:
        return NoteSaveOutcome.blocked("A destructive action is in progress.")
    if not self._snapshot.dirty:
        if explicit:
            self._untouched_create_token = None
        return NoteSaveOutcome.acknowledged("Saved — no changes.")
    self._pending_save_requested = True
    if self._save_task is None or self._save_task.done():
        self._save_task = asyncio.create_task(self._drive_saves())
    return await asyncio.shield(self._save_task)
```

`_drive_saves()` loops until current revision equals saved revision and the
pending flag is clear. Each attempt captures note id, session generation,
expected version, revision, and validated payload; it clears the flag before
awaiting the port. On success it always accepts the returned version for the
same session, patches the baseline from the payload actually saved, advances
`saved_revision`, and only clears dirty/reports `Saved HH:MM` when the current
revision is the saved revision. Newer edits trigger another iteration with no
two-attempt cap. Validation/conflict/failure stop chaining and retain dirty.
Untouched-create eligibility is cleared by a genuine edit or only after an
explicit no-op Save is acknowledged; a blocked, failed, conflicted, or
validation-vetoed request never revokes it.

- [ ] **Step 5: Add and pass three-revision serialization tests**

Add tests that gate the first two saves, type revisions 2/3/4 while those saves
are running, then assert:

```python
assert [call[2].revision for call in port.save_calls] == [1, 4]
assert coordinator.snapshot.saved_revision == 4
assert coordinator.snapshot.draft_revision == 4
assert coordinator.snapshot.dirty is False
assert max_concurrent_port_saves == 1
```

Also test ordinary failure stops retry chaining and a later mutation or
explicit Save retries the latest revision.

Run:

```bash
.venv/bin/python -m pytest -q Tests/Library/test_library_notes_session.py --tb=short
```

Expected: PASS.

- [ ] **Step 6: Verify host independence and commit**

Run:

```bash
rg -n 'textual|FileNotes|file_notes|ChaChaNotes|LibraryScreen' \
  tldw_chatbook/Library/library_notes_session.py
```

Expected: no matches except explanatory docstrings that do not import or
reference runtime types.

```bash
git add \
  tldw_chatbook/Library/library_notes_session.py \
  Tests/Library/test_library_notes_session.py
git commit -m "feat(notes): serialize database note sessions"
```

---

### Task 3: Add conflict, flush, untouched-create, and destructive gates

**Files:**
- Modify: `tldw_chatbook/Library/library_notes_session.py`
- Modify: `Tests/Library/test_library_notes_session.py`

**Interfaces:**
- Produces: `ConflictAction`, `ConflictOutcome`, `DestructiveKind`,
  `DestructiveAdmission`, and complete typed `flush()` behavior.
- Guarantees: one conflict operation token and one destructive admission token
  per active session.

- [ ] **Step 1: Write failing conflict-operation tests**

Add tests for:

```python
async def test_reload_applies_only_when_captured_revision_is_still_current():
    # Seed conflict at revision 2, gate load_note(), request Reload,
    # mutate body to revision 3, release the load.
    outcome = await reload_task
    assert outcome.kind is ConflictOutcomeKind.DRAFT_CHANGED
    assert coordinator.snapshot.body == "revision 3"
    assert coordinator.snapshot.in_conflict is True


async def test_overwrite_rebases_then_uses_serialized_save_driver():
    # Seed conflict, make load return version 8, mutate during the fetch,
    # release it, and assert the save uses v8 and the latest revision.
    assert port.save_calls[-1][1] == 8
    assert port.save_calls[-1][2].body == "latest local body"


async def test_duplicate_or_opposite_conflict_action_is_ignored():
    first = asyncio.create_task(coordinator.resolve_conflict(ConflictAction.RELOAD))
    duplicate = await coordinator.resolve_conflict(ConflictAction.OVERWRITE)
    assert duplicate.kind is ConflictOutcomeKind.ALREADY_RUNNING
    release_load.set()
    await first
```

Cover renewed conflicts, missing-note Overwrite, unchanged-revision
missing-note Reload, ordinary fetch failure, and stale operation/session
tokens.

- [ ] **Step 2: Write failing flush and destructive-admission tests**

Add tests that prove:

- `flush()` waits through every coalesced revision and permits navigation only
  when clean;
- validation, failure, conflict, and missing-note outcomes veto navigation;
- `open_session(..., untouched_create_token="create-7")` exposes discard after
  the matching typed load succeeds;
- first genuine edit and explicit no-op Save clear discard eligibility;
- blocked `Ctrl+S` during pending/running Discard leaves the admitted token and
  discard eligibility unchanged;
- validation, failure, and conflict outcomes never clear untouched eligibility
  unless a genuine edit already did so;
- stale create tokens cannot admit discard;
- destructive admission first flushes, then atomically blocks mutation, save,
  autosave, duplicate destructive requests, and stale note/session/version
  tuples;
- Cancel is accepted only before the service mutation is marked running;
- Escape/Cancel after `mark_destructive_running()` is rejected;
- failure unlocks the unchanged draft and success closes the session.

- [ ] **Step 3: Run the coordinator tests and verify red**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Library/test_library_notes_session.py --tb=short
```

Expected: FAIL on missing conflict/destructive APIs.

- [ ] **Step 4: Implement conflict generation and operation tokens**

When a port save returns `CONFLICT`, retain the latest canonical draft, clear
automatic chaining, set `in_conflict`, and increment
`conflict_generation`. `resolve_conflict()` must capture note id, session
generation, conflict generation, draft revision, and a monotonically
increasing operation token before its first await. Every completion verifies
the full tuple.

Reload may replace the draft only when the captured draft revision still
matches. Overwrite loads a fresh version without changing the draft, rebases
the expected version, and invokes the same save driver. Neither path owns UI,
focus, or navigation.

- [ ] **Step 5: Implement typed flush and destructive admission**

Define an admission value that contains:

```python
@dataclass(frozen=True)
class DestructiveAdmission:
    kind: DestructiveKind
    note_id: str
    session_generation: int
    expected_version: int
    operation_token: int
    create_token: str | None = None
```

`request_destructive_admission()` awaits `flush()`, validates untouched-create
eligibility when needed, then sets the gate without another await.
`mark_destructive_running()` revalidates the complete tuple immediately before
the host service call. `cancel_destructive()` works only while pending.
`finish_destructive(success=False)` clears the gate without changing draft;
success ends the active session.

- [ ] **Step 6: Run all coordinator tests and commit**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Library/test_library_notes_session.py --tb=short
```

Expected: PASS.

```bash
git add \
  tldw_chatbook/Library/library_notes_session.py \
  Tests/Library/test_library_notes_session.py
git commit -m "feat(notes): gate conflicts and destructive actions"
```

---

### Task 4: Adapt existing services behind the coordinator port

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:83-110`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:989-1103`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:4033-4185`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:5321-5810`
- Modify: `Tests/UI/test_library_shell.py:5271-6570`

**Interfaces:**
- Consumes: ADR-027 coordinator APIs and existing
  `notes_scope_service`/`notes_service`.
- Produces: a private screen-owned `_LibraryDatabaseNoteSessionPort` adapter.

- [ ] **Step 1: Write failing normalized-adapter Pilot tests**

Add tests that open a note whose detail and keywords arrive from separate
existing services, then assert the coordinator receives one complete
`NormalizedDatabaseNote`. Add a gated detail/keyword race where the user opens
another note before the second await and prove the first normalized result is
discarded.

Assert the adapter returns distinct typed load replies for loaded, genuinely
missing, and ordinary service-failure states; a transient fetch error must not
be treated as deletion.

While either detail or keywords is pending, assert a focusable Back action
remains visible; activating it invalidates the request and returns Navigator.
A typed load failure keeps Back plus actionable retry status instead of a
permanent loading placeholder.

Add a save-reply normalization matrix:

```python
@pytest.mark.parametrize(
    ("service_result", "expected_kind", "expected_version"),
    (
        (True, PortSaveKind.SAVED, 3),
        ({"version": 4, "keywords": ["a"]}, PortSaveKind.SAVED, 4),
        (False, PortSaveKind.CONFLICT, None),
    ),
)
```

Also assert `ConflictError` maps to `CONFLICT` and unexpected exceptions map to
typed `FAILED` without escaping to Textual.

- [ ] **Step 2: Run the focused adapter/UI tests and verify red**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py \
  -k 'note and (normalized or coordinator or save_reply)' \
  --tb=short
```

Expected: FAIL because the screen has no coordinator adapter.

- [ ] **Step 3: Add the private async service adapter**

Implement `_LibraryDatabaseNoteSessionPort` in `library_screen.py`, injected
with:

- `run_service_call`, reusing `_run_library_service_call`;
- `notes_scope_service`;
- `notes_service`;
- user id;
- UTC clock.

`load_note()` fetches detail and keywords internally and returns one
`DatabaseNotePortLoadReply`. It maps a successful detail/keyword pair to one
normalized value, a successful empty detail to `MISSING`, and any exception to
`FAILED`; it never conflates a transient failure with deletion. The loaded
value preserves raw title/body, converts keyword records to semantic strings,
normalizes version to `int`, and retains created/modified metadata.

`save_note()` calls the existing local `save_note` seam with exact payload
values. It maps:

- `True` to saved `expected_version + 1`;
- a mapping to its returned version/keywords;
- `False` or `ConflictError` to conflict;
- any other exception to failed.

It does not create/delete notes and does not expose the backing services to the
coordinator.

- [ ] **Step 4: Make the coordinator the canonical active editor state**

Construct one coordinator in `LibraryScreen.__init__`. Replace
`_library_note_detail`, `_library_note_version`, `_library_note_dirty`,
`_library_note_conflict_snapshot`, and `_library_note_preview_snapshot` as
authorities with `coordinator.snapshot`. During the transition, keep
compatibility properties only where old tests/callers still need a read:

```python
@property
def _library_note_dirty(self) -> bool:
    return self._library_note_session.snapshot.dirty
```

Do not maintain two mutable copies.

Detail loading calls `coordinator.open_session(note_id)`. The coordinator owns
the injected port call plus request freshness; the screen owns only
pending/failed presentation and applies the returned typed outcome if its
route still targets Notes. Save/autosave calls `request_save()`. Flush uses
typed coordinator `flush()`. Conflict buttons call `resolve_conflict()`.

Represent pending/failed detail load as a small screen-owned presentation
state with Back and Retry. Back increments the load token before leaving so a
late normalized reply cannot reopen the editor.

Handle typed outcomes centrally:

- saved: patch cached list baseline with the payload actually saved;
- validation veto: retain draft, show exact message, route/focus the field;
- failure: show retry copy, retain focus/draft;
- conflict: show Editor conflict state and focus callout;
- missing: apply the spec's safe Navigator/retained-draft posture.

- [ ] **Step 5: Remove the legacy competing save/conflict code**

Delete or reduce to adapter calls:

- `_sanitize_note_content()` save truncation;
- `_library_note_keywords_from_input()` sanitizing persistence path;
- `_read_library_note_editor_fields()` as persistence authority;
- `_save_library_note()`'s direct service call;
- `_resolve_library_note_conflict()`'s direct load/save calls;
- preview/conflict draft snapshots.

Keep unrelated media/prompt sanitizers and Notes export formatting intact.

- [ ] **Step 6: Run the coordinator and incumbent Notes save tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/Library/test_library_notes_session.py \
  Tests/UI/test_library_shell.py \
  -k 'note and (detail or save or autosave or flush or conflict)' \
  --tb=short
```

Expected: PASS, including incumbent explicit save, autosave, Back flush,
optimistic conflict, stale detail, and list-patch tests.

- [ ] **Step 7: Commit the host integration**

```bash
git add \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_shell.py
git commit -m "refactor(notes): host portable session coordinator"
```

---

### Task 5: Build stable Edit, Preview, and Context presentation

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py:28-410`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:3402-3572`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:5816-6159`
- Modify: `Tests/UI/test_library_shell.py:6929-7251`

**Interfaces:**
- Produces: stable mounted Editor/Preview/Context surfaces and idempotent
  `LibraryNotesCanvas.apply_session_state(...)`.
- Consumes: immutable coordinator snapshots; emits only user intent.

- [ ] **Step 1: Write failing stable-composition tests**

Add Pilot tests that capture object identities for title, body, Preview body,
Context, conflict callout, confirmation, and status; then toggle
Preview/Context/status/conflict/confirmation and assert each object identity
is unchanged.

Add a guarded-sync test:

```python
revision = screen._library_note_session.snapshot.draft_revision
timer = screen._library_notes_autosave_timer
canvas.apply_session_state(screen._library_note_presentation_state())
canvas.apply_session_state(screen._library_note_presentation_state())
assert screen._library_note_session.snapshot.draft_revision == revision
assert screen._library_notes_autosave_timer is timer
```

Add tests for Preview keyboard focus/scroll, Editor caret/selection/body-scroll
restoration, Context keyword mutation, compact Context utility reachability,
and wide direct utility reachability without entering Context.

- [ ] **Step 2: Run the focused presentation tests and verify red**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py \
  -k 'note and (preview or context or stable or widget_identity or wide_utility)' \
  --tb=short
```

Expected: FAIL because current toggles recompose different editor children.

- [ ] **Step 3: Compose all editor-session surfaces once**

For `mode="editor"`, mount:

- `#library-note-editor-region`;
- focusable `#library-note-preview-region` with
  `#library-note-preview-body`;
- scrollable `#library-note-context-region`;
- always-mounted conflict callout/actions;
- always-mounted delete confirmation/actions;
- one Editor status and one Context status.

Use persistent `Static(markup=False)` labels for title, Body, and keywords.
Use one-row markup-disabled, cell-ellipsized Preview/Context titles. Hidden
surfaces use `display=False`, are disabled where applicable, and are removed
from focus order/action queries.

Keep wide keyword, metadata, Console, Copy, Markdown/text export, and Delete
inline. Context may duplicate them at wide width; compact hides only the
inline copies.

- [ ] **Step 4: Implement idempotent presentation synchronization**

`apply_session_state()` receives the immutable snapshot plus
region/presentation/compact/validation/conflict/delete/transfer display state.
It:

- updates widget values only when different;
- updates Preview Markdown from canonical draft;
- updates status/meta/Context title;
- toggles surfaces/classes without recomposition;
- disables mutation/actions during destructive running;
- runs Input/TextArea assignments under a screen-owned
  `_library_note_presentation_syncing` guard.

The three change handlers return immediately under that guard and otherwise
send the exact current value to `coordinator.mutate()`.

- [ ] **Step 5: Convert Preview, Context, conflict, and confirmation handlers**

Preview/Context actions change presentation fields and call
`apply_session_state()`; they do not save or recompose. Conflict and
confirmation visibility use the same seam. Navigator ↔ Editor remains an
allowed workflow recompose.

Exports, Copy, and Console handoff read `coordinator.snapshot.draft` so they
work identically from Editor, Preview, and Context.

- [ ] **Step 6: Run presentation and incumbent utility tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py \
  -k 'note and (preview or context or export or copy or console or conflict or delete)' \
  --tb=short
```

Expected: PASS with stable identities and no draft/autosave changes from
presentation synchronization.

- [ ] **Step 7: Commit stable presentation**

```bash
git add \
  tldw_chatbook/Widgets/Library/library_notes_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_shell.py
git commit -m "feat(notes): add stable editor context surfaces"
```

---

### Task 6: Make Navigator, Create, Sync, and transfers explicit

**Files:**
- Modify: `tldw_chatbook/Library/library_notes_state.py`
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py:102-510`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:6684-6899`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:10124-10568`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:11217-11857`
- Modify: `Tests/Library/test_library_notes_state.py`
- Modify: `Tests/UI/test_library_shell.py:5049-5270`
- Modify: `Tests/UI/test_library_shell.py:7301-8365`

**Interfaces:**
- Produces: explicit Navigator action groups, direct sort/sync choices,
  truthful empty states, create/discard state, and visible transfer feedback.

- [ ] **Step 1: Write failing Navigator and direct-choice tests**

Add tests proving:

- zero total notes renders `No notes yet` plus New;
- zero filter matches renders `No notes match "<query>"` plus Clear and never
  claims there are no notes;
- Filter has a persistent visible label and its query/status are one-row
  markup-disabled/ellipsized;
- Browse row exposes New, Sort, Select;
- Transfer row exposes Sync, Import, Export;
- Sort opens one direct chooser with Newest/Oldest/Title; selecting one value
  applies it without cycling;
- Sync direction and conflict controls expose all direct choices;
- multi-select replaces the two normal action rows with exactly one selection
  action row and one status row.

- [ ] **Step 2: Write failing create/discard and transfer-status tests**

Add gated fake services and prove:

- Create has Back and prevents duplicate Blank/template activation;
- failed create stays in Create with visible actionable status;
- an invalid/transforming template payload is visibly vetoed before the
  create service and is never silently truncated or sanitized;
- successful create opens Editor, clears filter, focuses title, and exposes
  `Discard new note`;
- discard is removed after first genuine edit or explicit no-op Save;
- import, whole-source/selected export, per-note export, Copy, and Console
  handoff display running/success/failure in the active status channel;
- actions with external side effects are disabled while running and cannot be
  started twice.

- [ ] **Step 3: Run the focused workflow tests and verify red**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/Library/test_library_notes_state.py \
  Tests/UI/test_library_shell.py \
  -k 'note and (empty or sort or sync or create or discard or transfer or handoff or multiselect)' \
  --tb=short
```

Expected: FAIL on cycling controls, ambiguous empty state, missing discard,
and missing visible operation status.

- [ ] **Step 4: Implement Navigator display state and direct controls**

Extend the pure list/display state with:

- total count separate from rendered/filter count;
- sort-choice visibility;
- selection mode;
- active operation status;
- explicit empty-state kind.

Compose named one-row containers:

- `#library-notes-browse-actions`;
- `#library-notes-transfer-actions`;
- `#library-notes-sort-choices`;
- `#library-notes-selection-actions`;
- `#library-notes-selection-status`;
- `#library-notes-status`.

Replace `next_notes_sort_mode()` activation with direct
`#library-notes-sort-newest`, `-oldest`, and `-title` actions. Keep the pure
sort helper for compatibility where still used.

Replace Sync direction/conflict cycling with explicit choice groups whose
individual values are visible and keyboard-focusable. Preserve existing
config persistence and conflict-policy coercion.

- [ ] **Step 5: Add create tokens and untouched-note discard**

Guard the host create worker with a monotonically increasing create token and
running flag. Build its payload through the same lossless validation helper
used by coordinator saves; a typed veto stays in Create and makes no service
call. On success, call `coordinator.open_session(created_note_id,
untouched_create_token=create_token)` and require the matching typed loaded
outcome. Render Discard only from coordinator eligibility.

Discard requests destructive admission with that token, marks it running
immediately before the existing delete service call, and uses the standard
delete completion path. A stale create token, duplicate activation, or edit
cannot delete.

- [ ] **Step 6: Add one active-region operation status channel**

Track typed operation kind/token/status in the screen. Every external transfer
captures a token, disables duplicate activation, and updates the existing
one-row status surface:

- `<Action>…`;
- `<Action> complete.`;
- `<Action> failed — <safe next action>.`

Late completions verify the active token/region before updating UI. Existing
notifications may remain as secondary feedback, but are not the only feedback.

- [ ] **Step 7: Run focused and incumbent workflow tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/Library/test_library_notes_state.py \
  Tests/UI/test_library_shell.py \
  -k 'note and (list or filter or sort or sync or create or import or export or copy or console or select or discard)' \
  --tb=short
```

Expected: PASS.

- [ ] **Step 8: Commit explicit workflows**

```bash
git add \
  tldw_chatbook/Library/library_notes_state.py \
  tldw_chatbook/Widgets/Library/library_notes_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/Library/test_library_notes_state.py \
  Tests/UI/test_library_shell.py
git commit -m "feat(notes): expose compact knowledge workflow"
```

---

### Task 7: Add compact stage navigation, focus identity, and local shortcuts

**Files:**
- Modify: `tldw_chatbook/Library/library_notes_state.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:779-1215`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:1220-1695`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:3402-3572`
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/UI/test_screen_footer_hints.py`

**Interfaces:**
- Produces: fixed measured-width breakpoint, compact Library/Notes stage,
  complete focus tuple, central recompose seam, local Back/Escape hierarchy,
  and region-specific footer context.

- [ ] **Step 1: Write failing 119/120 breakpoint and stage tests**

Use `LibraryHarness` sizes that settle
`#library-shell-grid.region.width` at 119 and 120. Also exercise representative
compact terminal sizes `(60, 20)`, `(80, 24)`, and `(100, 30)`. Assert:

```python
assert screen._library_notes_compact is True   # measured width 119
assert screen._library_notes_compact is False  # measured width 120
```

Cross each size repeatedly and assert no oscillation. At compact:

- Library opens with the rail stage;
- selecting Notes shows Navigator as the only work region;
- selecting a note switches to Editor;
- Editor Back returns Navigator;
- Navigator Back returns and focuses its Library rail row;
- an unsafe dirty/saving/error/conflict session outranks rail focus and resumes
  its exact Notes region.

Resize 80→100→60 without crossing the breakpoint, and 120→170→120 on the wide
side. Spy on the presentation transition/capture seam and assert zero calls:
the compact flag, Notes stage/region, coordinator snapshot, semantic focus,
caret/selection, and scroll offsets remain unchanged. At 80×24 and 100×30,
assert surplus rows accrue only to the region's named scroll/content owner.

- [ ] **Step 2: Write failing complete focus-tuple round trips**

At 170→60→170 and 60→170→60, capture/restore:

- Library rail row;
- Navigator Filter and `note-row:<note_id>` plus list scroll;
- title/body and body selection/caret/scroll;
- Preview body plus scroll;
- Context action plus Context scroll;
- Create `create-template:<key>` plus scroll;
- Sync folder/direction/conflict/auto/run plus scroll and live status/activity.

Also force an unrelated whole-screen recompose during a dirty edit and assert
draft, caret, selection, scroll, presentation, and semantic focus restore.

- [ ] **Step 3: Write failing local shortcut/footer tests**

Cover:

- `Ctrl+N` only in Notes and only after a successful flush;
- `/` focuses Filter only from Navigator and never steals text-entry keys;
- `Ctrl+S` saves from Editor/Preview/Context;
- `Ctrl+S` visibly refuses during conflict resolution or destructive state;
- Escape follows confirmation → Context → Editor/Preview → Navigator → rail;
- Escape cannot bypass failure/conflict;
- footer copy changes exactly per spec and compact hides word/token/DB
  indicators while wide restores their existing contents.

- [ ] **Step 4: Run focused responsive tests and verify red**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_screen_footer_hints.py \
  -k 'note and (compact or breakpoint or focus or resize or shortcut or escape or footer or recompose)' \
  --tb=short
```

Expected: FAIL because current Library has no Notes compact stage/focus tuple.

- [ ] **Step 5: Implement invariant breakpoint measurement**

After mount/layout and on resize, measure only:

```python
width = self.query_one("#library-shell-grid").region.width
compact = width < 120
```

Apply compact classes/display only when `compact` changes. Do not derive the
breakpoint from content width or any region whose allocation changes with the
compact class.

Return immediately when the newly measured value equals
`self._library_notes_compact`; same-side resizes must do no presentation-state
capture, rehydration, class toggling, focus restoration, or coordinator work.

At compact, hide the rail or canvas by stage without removing either from the
screen. Notes opens Navigator; selecting a note changes the Notes region to
Editor; Context is explicit. At wide, retain the rail plus canvas and direct
inline utilities.

- [ ] **Step 6: Implement pure focus identity and central recompose capture**

Store only portable values in `LibraryNotesFocusIdentity`: stage, region,
note id, semantic role, body selection endpoints, and scroll offset. Map live
widget ids/attributes to semantic roles before resize/recompose and resolve
roles back afterward.

Override the one screen-level `refresh(..., recompose=True)` seam with the same
Textual signature:

```python
def refresh(self, *regions, repaint=True, layout=False, recompose=False):
    restore = self._capture_library_notes_recompose_state() if recompose else None
    result = super().refresh(
        *regions, repaint=repaint, layout=layout, recompose=recompose
    )
    if restore is not None:
        self.call_after_refresh(self._rehydrate_library_notes_after_recompose, restore)
    return result
```

The capture contains coordinator identity plus presentation/focus state, never
a second draft. Rehydration applies the coordinator snapshot under the
programmatic-sync guard, clamps selection/scroll, focuses the semantic role,
and calls `scroll_visible(animate=False)`.

- [ ] **Step 7: Implement local action eligibility and footer context**

Add local bindings/actions for `ctrl+n`, `/`, `ctrl+s`, and `escape`.
`check_action()` or the action body must return unavailable outside the exact
Notes region and while an input needs the literal key.

Register region-specific footer shortcuts through the existing persisting
footer API. When compact Notes is active, set
`#footer-word-count`, `#footer-token-count`, and
`#internal-db-size-indicator` to `display=False`; restore them on wide/exit
without replacing their text.

- [ ] **Step 8: Run focus, route, and footer tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_screen_footer_hints.py \
  Tests/UI/test_screen_navigation.py \
  -k 'library or notes' \
  --tb=short
```

Expected: PASS.

- [ ] **Step 9: Commit responsive behavior**

```bash
git add \
  tldw_chatbook/Library/library_notes_state.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_screen_footer_hints.py
git commit -m "feat(notes): preserve focus across compact stages"
```

---

### Task 8: Enforce exact 60×20 geometry in source, fallback, and bundle CSS

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:791-870`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:768-1528`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_css_build_integrity.py`
- Modify: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Produces: exact 15-row Notes content allocations and one scroll owner per
  region in both harness and bundled-app styles.

- [ ] **Step 1: Write failing CSS parity tests**

Add helpers that extract selector bodies from `LibraryScreen.DEFAULT_CSS`, the
source TCSS, and the generated bundle. Assert geometry-critical properties
match for:

- `#library-shell-grid.library-notes-compact`;
- `#library-canvas.library-notes-compact`;
- Navigator action/status/list rows;
- Editor normal/validation/conflict/delete body heights;
- Preview body;
- Context/Create/Sync scroll viewports;
- one-row headers/status/actions.

Assert source CSS and the bundled `_agentic_terminal` module are byte-equal
apart from the bundle module markers.

- [ ] **Step 2: Write failing 60×20 state-matrix geometry tests**

Parameterize normal, filtered-empty, sort-choice, selection, loading,
untouched-new, validation, conflict, delete-confirmation, Preview, Context,
Create, and Sync. For each state at `(60, 20)`, assert:

- the terminal allocation is Main navigation 3 + Library header 1 + Notes
  content 15 + footer 1 = 20;
- Notes content box height is 15;
- the exact allocation from the specification;
- every required visible control has positive size and lies inside the
  terminal;
- the named scroll/content owner meets its numeric floor;
- no page-level horizontal overflow;
- the focused control is visible.

Include long Unicode/Rich-like titles and assert one-row plain headers with no
stored-text mutation.

Repeat the normal Navigator/Editor/Context allocations at `(80, 24)` and
`(100, 30)` and assert that added height belongs only to the specified
list/body/viewport scroll owner; fixed headers, actions, and status rows remain
one row.

- [ ] **Step 3: Run CSS/geometry tests and verify red**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_css_build_integrity.py \
  Tests/UI/test_library_shell.py \
  -k 'note and (css or geometry or 60x20 or allocation or long_title)' \
  --tb=short
```

Expected: FAIL because current controls use three-row inputs, margins, borders,
and stacked utility rows that exceed the compact budget.

- [ ] **Step 4: Add matching fallback and source compact rules**

Under the compact class:

- remove decorative shell/canvas vertical border, padding, and margins;
- allocate the Notes canvas exactly 15 rows;
- make headers, labeled one-line inputs, actions, and status rows exactly one
  row;
- assign the table's body/list/viewport heights;
- keep exactly one scroll owner in each region;
- hide wide-only inline utilities without hiding Context equivalents;
- use `text-wrap: nowrap`, `text-overflow: ellipsis`, and one-row max height
  for dynamic headers/status.

Wide rules retain current border treatment, direct utilities, and flexible
body growth.

- [ ] **Step 5: Regenerate the CSS bundle**

Run:

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected: successful build. Inspect the generated diff; only the header
timestamp and source-derived `_agentic_terminal.tcss` changes are allowed.

- [ ] **Step 6: Run CSS parsing, parity, and geometry tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_css_build_integrity.py \
  Tests/QA/test_agentic_terminal_css_tokens.py \
  Tests/UI/test_library_shell.py \
  -k 'css or note' \
  --tb=short
```

Expected: PASS.

- [ ] **Step 7: Commit exact geometry**

```bash
git add \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/UI/test_css_build_integrity.py \
  Tests/UI/test_library_shell.py
git commit -m "style(notes): fit complete workflow at 60x20"
```

---

### Task 9: Prove destructive and concurrent interaction safety in Pilot

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:11455-11857`
- Modify: `Tests/UI/test_library_shell.py:5464-6880`

**Interfaces:**
- Consumes: coordinator save/conflict/destructive gates.
- Produces: end-to-end proof that Textual events and threaded service calls
  cannot bypass those gates.

- [ ] **Step 1: Add bounded gated fakes**

Extend the existing `_DelayedSaveLibraryNotesScopeService` pattern with
bounded `threading.Event` gates for detail, save, create, and delete. Always
use `_GATED_RELEASE_TIMEOUT_SECONDS`; every test releases gates in `finally`
so a failing assertion cannot wedge interpreter shutdown.

Track maximum concurrent save calls and each service payload/version/token.

- [ ] **Step 2: Write failing save-chain and conflict-interleaving tests**

Drive with actual widgets/Pilot:

- three successive edits while saves are in flight;
- explicit Save during an in-flight save;
- Back waiting for the complete chain;
- edit during Overwrite fetch and save;
- edit during Reload fetch;
- Overwrite then Reload, Reload then Overwrite, and duplicate activation;
- conflict originating from Editor, Preview, and Context.

Assert the newest canonical draft always wins, only one service mutation is
active, stale operations cannot apply, and Back stays put until a successful
typed flush.

- [ ] **Step 3: Write failing destructive event-race tests**

Enter discard/delete admission, then send real:

- title/body/keyword input;
- `Ctrl+S`;
- Escape;
- duplicate Delete/Discard/confirm activation.

Test both pending and service-running phases. Assert no mutation/save/cancel or
second delete reaches the service after running begins. On compact Context
Delete, assert confirmation moves to Editor; Cancel/failure restores Context
focus/scroll; success returns Navigator. Test stale version/note/session/create
tokens immediately before the service call.

- [ ] **Step 4: Run the new race tests and verify red if any host gate is missing**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py \
  -k 'note and (coalesc or inflight or overwrite or reload or destructive or discard or duplicate or race)' \
  --tb=short
```

Expected before final host wiring: at least one new assertion fails. If all
pass because prior tasks already completed the host wiring, inspect the tests
to ensure they use real widget/key events and gated service calls rather than
calling coordinator methods directly.

- [ ] **Step 5: Close only the observed host-adapter gaps**

Route all Textual mutation/save/destructive handlers through coordinator
admission checks. Disable fields/buttons in `apply_session_state()` while
destructive pending/running. Immediately before the service delete call,
invoke `mark_destructive_running(admission)` and return without side effects
if it rejects the tuple.

Do not add duplicate UI-side authority for session state; missing behavior is
fixed in the screen adapter or coordinator according to ADR-027 ownership.

- [ ] **Step 6: Run coordinator plus race suites**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/Library/test_library_notes_session.py \
  Tests/UI/test_library_shell.py \
  -k 'note and (save or conflict or flush or destructive or discard or delete or race)' \
  --tb=short
```

Expected: PASS, with no unbounded gate waits.

- [ ] **Step 7: Commit interaction safety**

```bash
git add \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_shell.py
git commit -m "test(notes): prove concurrent edit safety"
```

---

### Task 10: Prove lifecycle stability, accessibility, and capability parity

**Files:**
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/UI/test_non_obscuring_focus_contract.py`

**Interfaces:**
- Produces: objective keyboard, focus, mount, worker, timer, and capability
  parity gates for compact and wide Notes.

- [ ] **Step 1: Add the complete keyboard capability matrix**

Parameterize every row of the specification's Existing capability parity
table at `(60, 20)` and `(170, 48)`. Drive only keys/Enter and assert each
capability's target handler/service/status is reached. Do not assert generic
Space activation.

Add deterministic Tab/Shift+Tab order tests for normal Editor, conflict,
delete confirmation, Context, and post-recompose rehydration. Assert visible,
non-obscuring focus and persistent labels for filter/title/body/keywords.

- [ ] **Step 2: Add repeated resize/toggle/recompose lifecycle tests**

After a quiet baseline:

1. Cross 119/120 at least 50 times.
2. Resize 80→100→60 and 120→170→120 at least 50 times per sequence without
   crossing the breakpoint.
3. Toggle Preview/Context at least 50 times.
4. Invoke representative non-Notes whole-screen recomposes while dirty.
5. Route away/back at least 50 times after successful flush.

Assert:

- coordinator object identity survives same-screen recomposes;
- stable editor widget identities survive presentation/breakpoint toggles;
- same-side resize sequences invoke no Notes presentation transition, capture,
  rehydration, class, focus, or coordinator-state work;
- 80×24 and 100×30 retain one named scroll/content owner for all surplus rows;
- no unbounded mount/remove growth;
- `library_note_save`, conflict, create, delete, and autosave workers/timers
  return to baseline;
- unmount cleanup leaves no Notes timer/worker behind;
- remount restores only selected note/list-editor/filter/sort, never transient
  Context/Preview/confirmation.

- [ ] **Step 3: Add focused ADR-011 heartbeat/backlog evidence**

Reuse the existing UI responsiveness monitor conventions. During a bounded
30-second compact/wide + Preview/Context + route-switch soak, record a 100 ms
heartbeat and assert:

```python
assert snapshot.max_heartbeat_lag_ms <= 250
assert snapshot.active_workers == baseline.active_workers
assert snapshot.active_timers == baseline.active_timers
```

If CI timing makes the 30-second evidence unsuitable as a normal unit test,
keep the deterministic 50-cycle leak test in pytest and run the timed soak
from the QA script in Task 11; document both outputs.

- [ ] **Step 4: Run accessibility and lifecycle tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_non_obscuring_focus_contract.py \
  -k 'notes or library_note' \
  --tb=short
```

Expected: PASS.

- [ ] **Step 5: Run the interface detector on changed UI sources**

Run:

```bash
node .agents/skills/impeccable/scripts/detect.mjs \
  --scope layout \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_notes_canvas.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss
```

Expected: zero blocking findings. Review advisory output manually against the
terminal-specific design.

- [ ] **Step 6: Commit accessibility/lifecycle gates**

```bash
git add \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_non_obscuring_focus_contract.py
git commit -m "test(notes): gate compact keyboard lifecycle"
```

---

### Task 11: Run full verification, capture UAT evidence, and close TASK-1333

**Files:**
- Create: `Docs/superpowers/qa/library-notes-adaptive-60x20/capture_library_notes.py`
- Create: `Docs/superpowers/qa/library-notes-adaptive-60x20/README.md`
- Create: `Docs/superpowers/qa/library-notes-adaptive-60x20/*.svg`
- Modify: `backlog/tasks/task-1333 - Adapt-Library-Notes-for-lossless-60x20-workflow.md`

**Interfaces:**
- Produces: reproducible synthetic UAT evidence and complete Backlog Definition
  of Done.

- [ ] **Step 1: Run focused state/coordinator/UI suites**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/Library/test_library_notes_state.py \
  Tests/Library/test_library_notes_session.py \
  Tests/Notes/test_notes_scope_service_library_canvas.py \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_screen_footer_hints.py \
  Tests/UI/test_non_obscuring_focus_contract.py \
  Tests/UI/test_css_build_integrity.py \
  Tests/QA/test_agentic_terminal_css_tokens.py \
  --tb=short
```

Expected: PASS.

- [ ] **Step 2: Run Notes/Library regressions**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/Library \
  Tests/Notes \
  Tests/UI/test_library_multiselect_notes.py \
  Tests/UI/test_library_selection_updates.py \
  Tests/UI/test_screen_navigation.py \
  Tests/UI/test_library_content_hub.py \
  Tests/UI/test_post_release_workspaces_library_depth.py \
  --tb=short
```

Expected: PASS.

- [ ] **Step 3: Run static and diff hygiene**

Run:

```bash
.venv/bin/python -m mypy \
  tldw_chatbook/Library/library_notes_state.py \
  tldw_chatbook/Library/library_notes_session.py
git diff --check
```

Expected: PASS/no output. If the repository's formatter/linter is available in
the execution environment, also run its configured changed-file command; do
not introduce a new formatter configuration.

- [ ] **Step 4: Run the full project suite**

Run:

```bash
.venv/bin/python -m pytest -q Tests --tb=short
```

Expected: PASS. Any unrelated pre-existing failure must be reproduced on the
pre-task commit and documented; do not weaken or skip it silently.

- [ ] **Step 5: Create the synthetic capture/soak driver**

Base `capture_library_notes.py` on `LibraryHarness` and the existing synthetic
service fakes. It must write:

- `notes-60x20-navigator.svg`;
- `notes-60x20-editor.svg`;
- `notes-60x20-context.svg`;
- `notes-60x20-conflict.svg`;
- `notes-170x48-wide-editor.svg`.

Use `pilot.app.export_screenshot(title=..., simplify=True)`. The same driver
runs the 30-second 100 ms heartbeat soak, 50 breakpoint crossings, and 50
route switches, then prints a JSON summary containing maximum heartbeat lag,
final active worker/timer counts, and mount/remove deltas.

- [ ] **Step 6: Run final keyboard UAT and inspect evidence**

Run:

```bash
.venv/bin/python \
  Docs/superpowers/qa/library-notes-adaptive-60x20/capture_library_notes.py
wc -c Docs/superpowers/qa/library-notes-adaptive-60x20/*.svg
rg -n 'Traceback|Unhandled exception|Unable to mount|Internal Error|<.* object at 0x' \
  Docs/superpowers/qa/library-notes-adaptive-60x20/*.svg
```

Expected: all SVGs are non-empty; the error scan has no matches; heartbeat gap
is at most 250 ms; worker/timer/mount counts return to bounded baseline.
Visually inspect each SVG for clipped actions, wrapped one-row headers,
invisible focus, ambiguous status, and page-level overflow.

- [ ] **Step 7: Write the QA report**

Record in `README.md`:

- exact commit and commands;
- terminal sizes/states;
- capability matrix result;
- focus/keyboard result;
- 60×20 allocations;
- heartbeat/worker/timer/mount numbers;
- SVG inventory;
- any accepted limitation already named by the spec.

- [ ] **Step 8: Self-review against all 12 acceptance criteria**

Run:

```bash
git diff --stat
git diff --check
git status --short
```

Inspect every changed line for:

- a second mutable draft authority;
- silent persistence transformation;
- missing stale-token checks;
- Textual/File Notes imports in the coordinator;
- compact-only loss of a wide utility;
- direct service calls from the canvas;
- unrelated worktree changes.

- [ ] **Step 9: Update and close TASK-1333 only after all gates pass**

Use Backlog CLI to:

1. Check all 12 acceptance criteria and all Definition of Done items.
2. Add concise Implementation Notes naming the coordinator boundary,
   presentation/geometry changes, modified files, verification commands,
   UAT evidence, trade-offs, and ADR-027.
3. Set TASK-1333 to Done.

Do not mark Done if TASK-400 is not Done, any required test is failing, UAT
evidence is missing, or any acceptance criterion remains unchecked.

- [ ] **Step 10: Commit completion evidence**

```bash
git add \
  Docs/superpowers/qa/library-notes-adaptive-60x20 \
  'backlog/tasks/task-1333 - Adapt-Library-Notes-for-lossless-60x20-workflow.md'
git commit -m "docs(notes): record adaptive workflow verification"
```

## Execution Handoff

Recommended execution is **Subagent-Driven** with a fresh implementation
worker per numbered task and review between tasks. **Inline Execution** is also
supported using `superpowers:executing-plans` in bounded batches. In either
mode, Task 0 is a hard stop until TASK-400 is Done.
