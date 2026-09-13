# Library Destinations Adaptive Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate Library Conversations, Notes, Prompts, and Skills into the shipped Media-style adaptive reader while preserving destination authority and delivering four independently releasable PRs.

**Architecture:** Extract Media's pane geometry and structural widget into one Library-local adaptive shell with a pure requested-versus-effective layout resolver. Keep concrete list, work-pane, draft, trust, import, conflict, and recovery behavior in destination-owned state and widgets; `LibraryScreen` remains the orchestration owner. Land Conversations with the extraction, then Notes, Prompts, and Skills on branches created from the previously merged PR.

**Tech Stack:** Python 3.11+, Textual 8.x, dataclasses, asyncio workers, SQLite-backed existing services, pytest/Textual Pilot, Hypothesis, TCSS.

**ADR required:** yes

**ADR path:** `backlog/decisions/086-library-adaptive-reader-shell.md`

**Reason:** The programme establishes a durable Library-wide structural boundary, preference owner, and cross-destination interface.

---

## Source documents and PR topology

- Specification: `Docs/superpowers/specs/2026-08-24-library-destinations-adaptive-reader-design.md`
- Architecture: `backlog/decisions/086-library-adaptive-reader-shell.md`
- Media precedent: `Docs/superpowers/specs/2026-08-23-library-media-netnewswire-reader-design.md`
- Compose-once constraint: `Docs/superpowers/specs/2026-08-13-library-compose-once-design.md`
- Testing lessons: `backlog/docs/lessons-testing-evidence.md`
- Live verification lessons: `backlog/docs/lessons-live-verification.md`
- Backlog lessons: `backlog/docs/lessons-backlog-hygiene.md`

| PR | Backlog task | Branch | Depends on |
| --- | --- | --- | --- |
| 1 | TASK-22031 | `codex/task-22031-library-adaptive-reader-conversations` | This design/plan commit or merged `dev` containing it |
| 2 | TASK-22032 | `codex/task-22032-library-adaptive-reader-notes` | Merged PR 1 |
| 3 | TASK-22033 | `codex/task-22033-library-adaptive-reader-prompts` | Merged PR 2 |
| 4 | TASK-22034 | `codex/task-22034-library-adaptive-reader-skills` | Merged PR 3 |

Do not implement later destinations on an unmerged stack. Before every PR, fetch `origin/dev`,
search open branches/PRs for the task id, create or reuse one dedicated worktree, then verify the
task dependency is present in the new base.

## File responsibility map

### Shared foundation and PR 1

- Create `tldw_chatbook/Library/library_adaptive_reader_state.py` — pure layout profile,
  preferences, normalization, effective geometry, comfort expansion, explicit-open priority, and
  hysteresis.
- Create `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py` — generic three-slot
  structural widget and collapse/expand grips; no destination behavior.
- Modify `tldw_chatbook/Library/library_media_reader_state.py` — retain Media detail-session logic
  and re-export compatibility layout names from the shared state module.
- Modify `tldw_chatbook/Widgets/Library/library_media_reader_shell.py` — become a thin
  Media-labelled compatibility wrapper over the shared shell.
- Create `tldw_chatbook/Library/library_conversation_reader_state.py` — conversation-specific
  selected/loaded identity, bounded transcript pages, mode, Find, stale/error, and generation fence.
- Create `tldw_chatbook/Widgets/Library/library_conversation_reader.py` — permanent Read/Info work
  pane and progressively mounted transcript.
- Modify `tldw_chatbook/Widgets/Library/library_conversations_canvas.py` — retain list/filter/pager/
  bulk/export responsibilities and remove the embedded preview as the authoritative work surface.
- Modify `tldw_chatbook/UI/Screens/library_screen.py` — compose the shared shell, run existing
  service workers, apply generation-fenced results, persist explicit choices, and keep global focus
  ownership.
- Modify `tldw_chatbook/config.py`, `tldw_chatbook/UI/Screens/settings_appearance_defaults.py`, and
  `tldw_chatbook/UI/Screens/settings_screen.py` — exact `[library.reader]` and destination-list
  preference ownership with Media fallback.
- Modify `tldw_chatbook/Widgets/Library/__init__.py` — export the shared shell and new work pane.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` and regenerate
  `tldw_chatbook/css/tldw_cli_modular.tcss` — shared shell and conversation work-pane geometry.
- Create `Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md` —
  append one signed-off before/after table per PR.

### PR 2: Notes

- Create `tldw_chatbook/Library/library_notes_reader_state.py` — Notes work mode and
  selected/loaded identity around the existing note session snapshot; no second draft owner.
- Create `tldw_chatbook/Widgets/Library/library_note_work_pane.py` — Edit/Preview/Info plus existing
  create/import/sync/conflict/recovery work surfaces.
- Modify `tldw_chatbook/Widgets/Library/library_notes_canvas.py` — retain the concrete Notes list,
  tree, paging, filter, sort, and existing bulk actions as the destination-list pane.
- Modify `tldw_chatbook/Library/library_notes_state.py`,
  `tldw_chatbook/Library/library_notes_session.py`, and
  `tldw_chatbook/UI/Screens/library_screen.py` — project existing coordinator state into the shell
  without changing save/conflict ownership.
- Modify the shared TCSS source and regenerate the bundle.

### PR 3: Prompts

- Create `tldw_chatbook/Library/library_prompts_reader_state.py` — Basic/Advanced/Info mode,
  selected/loaded identity, hidden-field validation routing, and one lossless draft reference.
- Create `tldw_chatbook/Widgets/Library/library_prompt_work_pane.py` — editor, history, collection,
  provenance, import, lifecycle, and recovery work surfaces.
- Modify `tldw_chatbook/Widgets/Library/library_prompts_canvas.py` — retain browse/filter/pager/
  selection responsibilities as the list pane and delegate work content.
- Modify `tldw_chatbook/Library/library_prompts_state.py` and
  `tldw_chatbook/UI/Screens/library_screen.py` — preserve existing artifact preparation,
  capabilities, conditional writes, memberships, and history controllers.
- Modify the shared TCSS source and regenerate the bundle.

### PR 4: Skills

- Create `tldw_chatbook/Library/library_skills_reader_state.py` — Overview/Edit/Trust/Files mode,
  selected/loaded identity, reviewed fingerprint, and stale-trust projection.
- Create `tldw_chatbook/Widgets/Library/library_skill_work_pane.py` — destination-owned overview,
  editor, trust review, supporting files, import, and recovery surfaces.
- Modify `tldw_chatbook/Widgets/Library/library_skills_canvas.py` — retain browse/filter/sort and
  list actions while delegating work content.
- Modify `tldw_chatbook/Library/library_skills_state.py` and
  `tldw_chatbook/UI/Screens/library_screen.py` — preserve LocalSkillsService and trust-service
  authority.
- Modify the shared TCSS source and regenerate the bundle.

## PR 1 — TASK-22031: shared shell and Conversations

### Task 1: Claim the task and pin the capability inventory

**Files:**
- Modify: `backlog/tasks/task-22031 - Share-Library-adaptive-reader-shell-and-migrate-Conversations.md`
- Create: `Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md`

- [ ] **Step 1: Verify the dependency and absence of duplicate work**

Run:

```bash
git fetch origin dev
git branch -a | rg '22031|adaptive-reader'
gh pr list --repo rmusser01/tldw_chatbook --state open --search '22031 adaptive reader'
backlog task 22031 --plain
```

Expected: TASK-22031 is `To Do`; no competing implementation PR exists; ADR-086 and the spec are
present in the base.

- [ ] **Step 2: Put TASK-22031 in progress and add its Backlog implementation plan**

Run:

```bash
backlog task edit 22031 -s 'In Progress' -a @codex
backlog task edit 22031 --plan $'1. Inventory Media and Conversations capabilities\n2. Extract and prove the shared shell\n3. Migrate shared preferences and adaptive geometry\n4. Add the fenced Conversations reader\n5. Run automated and live verification\n\nADR required: yes\nADR path: backlog/decisions/086-library-adaptive-reader-shell.md\nReason: implements the accepted Library structural boundary.'
```

Expected: the task contains an Implementation Plan but no Implementation Notes yet.

- [ ] **Step 3: Record the before/after inventory before production edits**

Create a table with rows for every currently visible Media and Conversations capability, its
current selector/handler, new region, and preservation test. Include Media Read/Analysis/Highlights/
Info, Find, Read Later, Console handoff, metadata, delete/Undo, bulk export/delete, rich-preview
fallback, external detail, list search/paging, and Conversations filter/paging/export/bulk/Open in
Console.

- [ ] **Step 4: Commit the planning checkpoint**

```bash
git add 'backlog/tasks/task-22031 - Share-Library-adaptive-reader-shell-and-migrate-Conversations.md' Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md
git commit -m 'docs(library): inventory adaptive reader capabilities'
```

### Task 2: Extract the current Media layout policy without changing behavior

**Files:**
- Create: `tldw_chatbook/Library/library_adaptive_reader_state.py`
- Modify: `tldw_chatbook/Library/library_media_reader_state.py`
- Create: `Tests/Library/test_library_adaptive_reader_state.py`
- Modify: `Tests/Library/test_library_media_reader_state.py`

- [ ] **Step 1: Write failing examples and properties**

Start with the public shape:

```python
@dataclass(frozen=True)
class AdaptiveReaderLayoutProfile:
    list_min_width: int = 32
    list_target_width: int = 40
    list_comfort_width: int = 56
    list_max_width: int = 72
    work_min_width: int = 44
    work_comfort_width: int = 44


@dataclass(frozen=True)
class AdaptiveReaderLayoutPreferences:
    library_open: bool = True
    items_open: bool = True
    custom_widths_enabled: bool = False
    library_width: int = 28
    items_width: int = 40


def resolve_adaptive_reader_layout(
    width: int,
    preferences: AdaptiveReaderLayoutPreferences,
    profile: AdaptiveReaderLayoutProfile,
    *,
    previous: AdaptiveReaderEffectiveLayout | None = None,
    priority: PaneName | None = None,
) -> AdaptiveReaderEffectiveLayout: ...
```

Characterization tests must compare the shared resolver with the current Media resolver over wide,
medium, compact, and minimum widths. At this checkpoint they prove only existing collapse priority,
custom-width behavior, hysteresis, non-negative widths, and width-budget invariants. Do not enable
comfort expansion or destination-specific editor profiles yet.

- [ ] **Step 2: Run the focused tests and witness failure**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_adaptive_reader_state.py -q
```

Expected: collection/import failure because the shared state module does not exist.

- [ ] **Step 3: Implement the minimal pure resolver and Media aliases**

Move only Media's existing structural constants, preference normalization, effective-layout
dataclass, and resolver into the new module without changing returned geometry. Keep
`LibraryMediaReaderSessionState`, detail request, selection, settlement, modes, and external-detail
behavior in `library_media_reader_state.py`. Re-export old Media names so current imports remain
valid during the extraction checkpoint. The profile may expose the approved future comfort fields,
but the resolver must not consume them until Task 4.

- [ ] **Step 4: Run focused and existing Media state tests**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py -q
```

Expected: PASS.

- [ ] **Step 5: Mutation-check the behavior-preserving collapse guard**

Temporarily reverse the existing Library-before-Items collapse order; rerun the new file and confirm
at least one characterization test fails. Restore it and rerun green.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Library/library_adaptive_reader_state.py tldw_chatbook/Library/library_media_reader_state.py Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py
git commit -m 'refactor(library): extract adaptive reader layout policy'
```

### Task 3: Extract the structural shell without changing Media behavior

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py`
- Modify: `tldw_chatbook/Widgets/Library/library_media_reader_shell.py`
- Modify: `tldw_chatbook/Widgets/Library/__init__.py`
- Create: `Tests/UI/test_library_adaptive_reader_shell.py`
- Modify: `Tests/UI/test_library_media_reader_shell.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify generated: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Write failing shell composition and geometry tests**

Pin a constructor that receives concrete widgets rather than schemas:

```python
shell = LibraryAdaptiveReaderShell(
    library=Static("Library"),
    items=Static("Items"),
    work=Static("Work"),
    layout=effective,
    id_prefix="probe",
    library_label="Library",
    items_label="Items",
)
```

Assert exactly three content widgets plus two five-column grips, stable child identity after
`sync_layout`, Enter/Space and pointer collapse messages, no focus inside hidden content, and every
child region inside screen bounds.

- [ ] **Step 2: Run the shell tests and witness failure**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_adaptive_reader_shell.py -q
```

Expected: import failure for the new shell.

- [ ] **Step 3: Implement the generic shell and thin Media wrapper**

The generic shell may emit only:

```python
class PaneToggleRequested(Message):
    def __init__(self, pane: Literal["library", "items"]) -> None: ...


class AdaptiveReaderShellResized(Message):
    pass
```

Keep Media's existing ids/classes through wrapper parameters so this checkpoint changes no Media
handler or selector behavior. Grips collapse/expand only; do not add dragging.

- [ ] **Step 4: Add shared TCSS and regenerate the bundle**

```bash
../../.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected: the generated bundle contains the shared shell selectors once and retains Media-specific
reader selectors.

- [ ] **Step 5: Run shell and Media regression tests**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_media_reader_shell.py Tests/UI/test_library_media_side_by_side.py Tests/UI/test_library_media_reader_flow.py -q
```

Expected: PASS before Conversations is wired.

- [ ] **Step 6: Commit the behavior-preserving extraction checkpoint**

```bash
git add tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py tldw_chatbook/Widgets/Library/library_media_reader_shell.py tldw_chatbook/Widgets/Library/__init__.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_media_reader_shell.py
git commit -m 'refactor(library): share adaptive reader shell structure'
```

### Task 4: Add adaptive geometry and migrate preference ownership

**Files:**
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/UI/Screens/settings_appearance_defaults.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:5178-5390`
- Modify: `Tests/test_config_library_defaults.py`
- Modify: `Tests/UI/test_settings_appearance_defaults.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Tests/UI/test_library_media_reader_shell.py`

- [ ] **Step 1: Write failing config normalization tests**

Pin these effective sections:

```python
settings["library"]["reader"] == {
    "library_open": True,
    "custom_widths_enabled": False,
    "library_width": 28,
}
settings["library"]["conversations_reader"] == {
    "items_open": True,
    "items_width": 40,
}
```

Also prove: absent `library.reader` falls back to the three legacy Media keys; each partially
populated shared key falls back independently; load performs no disk write; first explicit toggle
writes `library.reader.library_open`; Media Items still writes `library.media_reader.items_open`;
all five destination list sections round-trip `items_open/items_width`; unknown keys survive
Settings deep merge.

Add geometry tests proving collapsed Library grows Items from 40 toward 56, custom 64 stays 64 when
it fits, responsive growth never mutates preferences, the editor profile protects 48 columns,
explicit open protects the requested pane, and hysteresis prevents flap.

- [ ] **Step 2: Run the focused tests and witness failures**

```bash
../../.venv/bin/python -m pytest Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_library_media_reader_shell.py -q
```

Expected: new shared-section assertions fail.

- [ ] **Step 3: Implement comfort expansion, normalization, and Settings ownership**

Enable the approved comfort-growth branch only after the shell extraction is green. Use one shared
`custom_widths_enabled`; keep saved list widths even when disabled. Do not eagerly rewrite old
config. Rename user-facing Settings copy from Media-only widths to shared Library reader geometry
and destination list widths. Settings save emits one `library.reader` section plus Media,
Conversations, Notes, Prompts, and Skills list sections without deleting legacy/future keys.

- [ ] **Step 4: Update mounted-screen refresh without data reload**

Replace `request_library_media_layout_refresh` with a Library-reader refresh entry point that
updates current normalized preferences and calls only the shell layout sync. Preserve a Media-named
compatibility method until all existing callers/tests migrate in this PR.

- [ ] **Step 5: Run focused tests**

```bash
../../.venv/bin/python -m pytest Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_library_media_reader_shell.py -q
```

Expected: PASS.

- [ ] **Step 6: Mutation-check the new geometry separately from extraction**

Temporarily remove comfort growth, confirm the collapsed-Library expansion test fails, restore it,
and rerun the Task 4 focused tests green.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/config.py tldw_chatbook/UI/Screens/settings_appearance_defaults.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/library_screen.py Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_library_media_reader_shell.py
git commit -m 'feat(library): share adaptive reader preferences'
```

### Task 5: Add the conversation reader state and bounded transcript loader

**Files:**
- Create: `tldw_chatbook/Library/library_conversation_reader_state.py`
- Create: `Tests/Library/test_library_conversation_reader_state.py`
- Modify: `Tests/Chat/test_chat_conversation_service.py` only if characterization coverage is absent

- [ ] **Step 1: Characterize the existing bounded service seam**

Prove `ChatConversationService.get_library_conversation_messages` returns exact `message_total`,
stable order, per-message revision, page offsets, and long-message continuation. Do not add a DB
method.

- [ ] **Step 2: Write failing destination-owned state tests**

Use a state shape such as:

```python
@dataclass(frozen=True)
class ConversationReaderRequest:
    destination: Literal["conversations"]
    conversation_id: str
    version: int
    generation: int


@dataclass(frozen=True)
class ConversationReaderState:
    selected_id: str | None = None
    loaded_id: str | None = None
    loaded_version: int | None = None
    generation: int = 0
    mode: Literal["read", "info"] = "read"
    messages: tuple[ConversationMessageView, ...] = ()
    message_total: int = 0
    complete: bool = False
    find_query: str = ""
    find_matches: tuple[ConversationFindMatch, ...] = ()
    error: str | None = None
```

Tests cover initial page, appended page, long-message continuation replacement, duplicate-page
defence, complete Find, selected/loaded mismatch banner, stale generation/version, deletion, retry,
mode preservation, and bulk read-only preview.

- [ ] **Step 3: Run and witness failure**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_conversation_reader_state.py -q
```

Expected: import failure.

- [ ] **Step 4: Implement minimal pure state transitions**

Keep IO out of the module. Append only results matching destination, id, version, and generation.
Find operates over the complete normalized message tuple; the controller must finish bounded page/
continuation loading before presenting a complete match count.

- [ ] **Step 5: Run and mutation-check stale settlement**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_conversation_reader_state.py Tests/Chat/test_chat_conversation_service.py -q
```

Expected: PASS; removing any fence dimension makes a focused stale-result test fail.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Library/library_conversation_reader_state.py Tests/Library/test_library_conversation_reader_state.py Tests/Chat/test_chat_conversation_service.py
git commit -m 'feat(library): model bounded conversation reader state'
```

### Task 6: Mount Conversations list and permanent work pane together

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_conversation_reader.py`
- Modify: `tldw_chatbook/Widgets/Library/library_conversations_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:8030-8103,10065-10158,11718-11994,18517-18599,33832-33959,35246-35279,36302-36349`
- Modify: `tldw_chatbook/Widgets/Library/__init__.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify generated: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_library_conversation_reader.py`
- Modify: `Tests/Widgets/Library/test_library_conversations_canvas.py`
- Modify: `Tests/UI/test_library_multiselect_conversations.py`
- Modify: `Tests/UI/test_library_entry_compose_once.py`

- [ ] **Step 1: Write the failing three-role journey**

With two conversations and delayed detail responses, assert the shell mounts Library, list, and
work once; selecting B immediately marks B selected while A remains labelled loaded; item actions
disable until B settles; late A cannot overwrite B; `/` focuses the list filter only outside text
inputs; global F6 reaches visible roles without a screen-local binding.

- [ ] **Step 2: Write failing transcript/mode tests**

Assert first message page appears before completion, subsequent pages mount without replacing the
shell/list, complete Find reaches a later page, Info shows title/version/message count/keywords, and
Open in Console uses the existing handoff. Bulk mode leaves the last single item visibly read-only.

- [ ] **Step 3: Run and witness failures**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_conversation_reader.py Tests/Widgets/Library/test_library_conversations_canvas.py Tests/UI/test_library_multiselect_conversations.py -q
```

Expected: the permanent reader assertions fail.

- [ ] **Step 4: Implement list/work composition and workers**

Use the existing `ChatConversationService.get_library_conversation_messages` through an off-loop
worker. Load in bounded pages and bounded long-message continuations. Every callback captures the
request object and calls the pure settlement function. Keep the prior work widget/state until the
new identity settles; do not recompose `LibraryScreen`.

- [ ] **Step 5: Implement truthful errors, deletion, focus, and restore controls**

Initial, stale refresh, filtered-empty, detail error, unavailable/deleted, and retry states must
retain the appropriate pane. Async completion moves focus only when the initiating focus-intent
generation remains current.

- [ ] **Step 6: Regenerate CSS and run focused tests**

```bash
../../.venv/bin/python tldw_chatbook/css/build_css.py
../../.venv/bin/python -m pytest Tests/Library/test_library_conversations_state.py Tests/Library/test_library_conversation_reader_state.py Tests/Widgets/Library/test_library_conversations_canvas.py Tests/UI/test_library_conversation_reader.py Tests/UI/test_library_multiselect_conversations.py Tests/UI/test_library_entry_compose_once.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_conversation_reader.py tldw_chatbook/Widgets/Library/library_conversations_canvas.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/__init__.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_library_conversation_reader.py Tests/Widgets/Library/test_library_conversations_canvas.py Tests/UI/test_library_multiselect_conversations.py Tests/UI/test_library_entry_compose_once.py
git commit -m 'feat(library): add permanent Conversations reader'
```

### Task 7: Verify and close PR 1

**Files:**
- Modify: `Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md`
- Modify: `backlog/tasks/task-22031 - Share-Library-adaptive-reader-shell-and-migrate-Conversations.md`

- [ ] **Step 1: Run the reachable automated suites**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py Tests/Library/test_library_conversations_state.py Tests/Library/test_library_conversation_reader_state.py Tests/Widgets/Library/test_library_conversations_canvas.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_media_reader_shell.py Tests/UI/test_library_media_reader_flow.py Tests/UI/test_library_media_side_by_side.py Tests/UI/test_library_media_image_preview.py Tests/UI/test_library_multiselect_media.py Tests/UI/test_library_conversation_reader.py Tests/UI/test_library_multiselect_conversations.py Tests/UI/test_library_entry_compose_once.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py Tests/test_config_library_defaults.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/Library/library_adaptive_reader_state.py tldw_chatbook/Library/library_conversation_reader_state.py tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py tldw_chatbook/Widgets/Library/library_conversation_reader.py
git diff --check
```

Expected: all selected tests pass, Ruff passes, diff check is clean. If unrelated failures appear,
run the identical command on a clean `origin/dev` worktree and compare failure identities.

- [ ] **Step 2: Perform real TUI verification**

Exercise Media and Conversations at 160x50, 120x35, 100x30, and 80x24. Capture text/SVG or
screenshots proving pane geometry, list expansion, full-title improvement, both restore controls,
long transcript traversal, Find, bulk preview, errors/retry, and no focus loss. Confirm no footer
advertises an unimplemented binding.

- [ ] **Step 3: Run architecture and generated-artifact checks**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_canvas_scoped_sync.py Tests/UI/test_library_recompose_ratchet.py Tests/RuntimePolicy -q
../../.venv/bin/python -m pytest -q
../../.venv/bin/python -m ruff check .
../../.venv/bin/python -m ruff format --check .
../../.venv/bin/python -m compileall -q tldw_chatbook
git diff --check origin/dev...HEAD
```

Expected: the repository-wide suite, lint, format check, compile check, RuntimePolicy, and diff
check all pass. If the clean base already fails, record the exact baseline command/output and prove
the branch adds no new failure; do not silently reduce the gate to focused tests.

- [ ] **Step 4: Draft closeout evidence while keeping the task In Progress**

Append the after inventory and draft concise Implementation Notes with files, tradeoffs, focused and
repository-wide commands, live evidence, ADR-086, and any generalized lesson actually learned. Do
not check final ACs or mark the task Done yet.

- [ ] **Step 5: Request review, address findings, rebase, and repeat the gates**

Use `superpowers:requesting-code-review`, address findings through
`superpowers:receiving-code-review`, rebase on latest `dev`, then rerun Steps 1–3 exactly. The task
remains In Progress until this final verification is green.

- [ ] **Step 6: Complete Backlog hygiene and commit closeout**

Only now check every TASK-22031 AC, finalize Implementation Notes, and mark Done:

```bash
backlog task edit 22031 -s Done --notes 'Implemented the shared Library adaptive shell and permanent Conversations reader; see PR verification and capability inventory. ADR: backlog/decisions/086-library-adaptive-reader-shell.md.'
git add 'backlog/tasks/task-22031 - Share-Library-adaptive-reader-shell-and-migrate-Conversations.md' Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md
git commit -m 'docs(library): close adaptive Conversations migration'
git diff --check origin/dev...HEAD
```

Use `superpowers:finishing-a-development-branch` for the PR/merge decision.

## PR 2 — TASK-22032: Notes

### Task 8: Claim Notes and inventory its existing contracts

**Files:**
- Modify: `backlog/tasks/task-22032 - Migrate-Library-Notes-to-the-adaptive-reader-shell.md`
- Modify: `Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md`

- [ ] **Step 1: Verify the merged dependency and duplicate-work guard**

```bash
git fetch origin dev
gh pr list --repo rmusser01/tldw_chatbook --state merged --search '22031 adaptive reader' --json number,mergeCommit,mergedAt,url
gh pr list --repo rmusser01/tldw_chatbook --state open --search '22032 Notes adaptive reader'
backlog task 22032 --plain
```

Expected: the merged query reports PR 1 with a merge commit already in `origin/dev`, TASK-22032 is
`To Do`, and there is no competing PR. If the merge query is ambiguous, stop and resolve the exact
dependency before branching.

- [ ] **Step 2: Claim the task and add the Backlog plan**

```bash
backlog task edit 22032 -s 'In Progress' -a @codex
backlog task edit 22032 --plan $'1. Inventory Notes capabilities and draft authority\n2. Add presentation-only reader state\n3. Split the persistent list and work pane\n4. Verify workflows, geometry, and focus\n\nADR required: yes\nADR path: backlog/decisions/086-library-adaptive-reader-shell.md\nReason: consumes the accepted Library structural boundary without changing Notes authority.'
```

- [ ] **Step 3: Append the before inventory**

Record list/tree/filter/sort/paging/bulk, Edit/Preview, templates, create, import, sync, lasting sync,
conflict, transfer, export, delete/recovery, File Notes exclusions, and focus/compact behavior. For
each row name the current selector/handler, its reader region, and a preservation test.

- [ ] **Step 4: Commit the inventory checkpoint**

```bash
git add 'backlog/tasks/task-22032 - Migrate-Library-Notes-to-the-adaptive-reader-shell.md' Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md
git commit -m 'docs(library): inventory Notes reader migration'
```

### Task 9: Add Notes reader state without creating a second draft owner

**Files:**
- Create: `tldw_chatbook/Library/library_notes_reader_state.py`
- Modify: `tldw_chatbook/Library/library_notes_state.py`
- Test: `Tests/Library/test_library_notes_reader_state.py`
- Test existing: `Tests/Library/test_library_notes_session.py`, `Tests/Library/test_library_notes_state.py`

- [ ] **Step 1: Write failing reader-state tests**

Pin a presentation-only state shape:

```python
@dataclass(frozen=True)
class NotesReaderState:
    selected_id: str | None = None
    loaded_id: str | None = None
    loaded_version: int | None = None
    generation: int = 0
    mode: Literal["edit", "preview", "info"] = "edit"
    session: LibraryNoteSessionSnapshot | None = None
    error: str | None = None
```

Cover clean mount, mode changes, selected/loaded identity, generation/version fencing,
conflict/deletion, and object identity of the existing snapshot as the sole draft source.

- [ ] **Step 2: Run the new tests and witness failure**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_notes_reader_state.py -q
```

Expected: import failure because the reader-state module does not exist.

- [ ] **Step 3: Implement minimal presentation transitions**

Store no title/body/keywords fields in the new state. `preview` reads the referenced snapshot's
current draft; `info` labels persisted versus unsaved values; settlements require matching note id,
version, and generation.

- [ ] **Step 4: Run and mutation-check draft ownership**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_notes_reader_state.py Tests/Library/test_library_notes_session.py Tests/Library/test_library_notes_state.py -q
```

Expected: PASS. Temporarily make `set_mode` increment `draft_revision`, confirm a focused test
fails, restore the implementation, and rerun green.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Library/library_notes_reader_state.py tldw_chatbook/Library/library_notes_state.py Tests/Library/test_library_notes_reader_state.py Tests/Library/test_library_notes_session.py Tests/Library/test_library_notes_state.py
git commit -m 'feat(library): model Notes reader presentation state'
```

### Task 10: Split the Notes list and permanent work pane

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_note_work_pane.py`
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/__init__.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify generated: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_library_notes_reader.py`
- Modify: `Tests/Widgets/Library/test_library_notes_canvas.py`
- Modify: `Tests/UI/test_library_multiselect_notes.py`
- Modify Notes import/sync/folder journey tests

- [ ] **Step 1: Write failing persistent-pane journeys**

In `Tests/UI/test_library_notes_reader.py`, prove list/work widget identity survives selection,
Edit/Preview/Info, save, validation, conflict, and retry. Add an 80-column geometry case proving the
48-column editor minimum and both restore controls remain on-screen.

- [ ] **Step 2: Write failing workflow-preservation journeys**

Prove create/templates/import/sync/recovery replace only work content; list selection goes through
the existing dirty-draft gate; Preview renders unsaved content; bulk mode leaves the last single
note read-only rather than hiding context.

- [ ] **Step 3: Run the red suite**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_notes_reader.py Tests/Widgets/Library/test_library_notes_canvas.py Tests/UI/test_library_multiselect_notes.py Tests/UI/test_library_note_import_flow.py Tests/UI/test_library_notes_files_sync_journey.py Tests/UI/test_library_notes_folder_navigator.py -q
```

Expected: new permanent-pane and geometry assertions fail.

- [ ] **Step 4: Extract and mount the concrete work pane**

Move editor/work composition into `LibraryNoteWorkPane`; leave list/tree/paging/bulk in
`LibraryNotesCanvas`. Mount both through the shared shell using the editor profile and
`[library.notes_reader]`; retain the existing coordinator, session snapshot, sync controllers,
confirmations, file pickers, scoped locks, and dirty-navigation authority.

- [ ] **Step 5: Regenerate CSS and run focused suites**

```bash
../../.venv/bin/python tldw_chatbook/css/build_css.py
../../.venv/bin/python -m pytest Tests/Library/test_library_notes_reader_state.py Tests/Library/test_library_notes_session.py Tests/Library/test_library_notes_state.py Tests/Library/test_library_notes_tree_state.py Tests/Library/test_library_note_import_state.py Tests/Library/test_library_notes_lasting_sync_state.py Tests/Widgets/Library/test_library_notes_canvas.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_multiselect_notes.py Tests/UI/test_library_note_import_flow.py Tests/UI/test_library_notes_files_sync_journey.py Tests/UI/test_library_notes_folder_navigator.py Tests/UI/test_library_notes_lasting_sync_flow.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Library/library_notes_reader_state.py tldw_chatbook/Widgets/Library/library_note_work_pane.py tldw_chatbook/Widgets/Library/library_notes_canvas.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/__init__.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_library_notes_reader.py Tests/Widgets/Library/test_library_notes_canvas.py Tests/UI/test_library_multiselect_notes.py
git commit -m 'feat(library): migrate Notes to adaptive reader'
```

### Task 11: Verify and close PR 2

- [ ] **Step 1: Run automated gates**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_notes_reader_state.py Tests/Library/test_library_notes_session.py Tests/Library/test_library_notes_state.py Tests/Library/test_library_notes_tree_state.py Tests/Library/test_library_note_import_state.py Tests/Library/test_library_notes_lasting_sync_state.py Tests/Widgets/Library/test_library_notes_canvas.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_multiselect_notes.py Tests/UI/test_library_note_import_flow.py Tests/UI/test_library_notes_files_sync_journey.py Tests/UI/test_library_notes_folder_navigator.py Tests/UI/test_library_notes_lasting_sync_flow.py Tests/UI/Library_Modules/test_library_note_import_controller.py Tests/UI/Library_Modules/test_library_notes_sync_controller.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_entry_compose_once.py Tests/UI/test_library_canvas_scoped_sync.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/Library/library_notes_reader_state.py tldw_chatbook/Widgets/Library/library_note_work_pane.py tldw_chatbook/Widgets/Library/library_notes_canvas.py
../../.venv/bin/python -m pytest -q
../../.venv/bin/python -m ruff check .
../../.venv/bin/python -m ruff format --check .
../../.venv/bin/python -m compileall -q tldw_chatbook
../../.venv/bin/python -m pytest Tests/RuntimePolicy -q
git diff --check origin/dev...HEAD
```

Expected: every command exits 0; compare any unrelated failure with a clean `origin/dev` worktree.

- [ ] **Step 2: Perform the live matrix**

Verify 160x50, 120x35, 100x30, and 80x24: clean Edit mount, unsaved Preview, conflicts, templates,
sync, both restore controls, focus continuity, and no clipped editor actions. Capture evidence.

- [ ] **Step 3: Draft task and inventory closeout without marking Done**

Append the after inventory and draft Implementation Notes with focused/full-suite commands, live
evidence and ADR-086. Keep TASK-22032 In Progress and its ACs unchecked until review completes.

- [ ] **Step 4: Review, rebase, and repeat the gates**

Use the review and receiving-review skills, address findings, rebase on latest `dev`, and rerun
Steps 1–2 exactly.

- [ ] **Step 5: Mark Done, commit closeout, and merge**

Check every AC and finalize notes only after final review/rebase verification:

```bash
backlog task edit 22032 -s Done --notes 'Migrated Database Notes to the shared adaptive reader while retaining the existing draft, sync, conflict, and recovery authorities. ADR: backlog/decisions/086-library-adaptive-reader-shell.md.'
git add 'backlog/tasks/task-22032 - Migrate-Library-Notes-to-the-adaptive-reader-shell.md' Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md
git commit -m 'docs(library): close adaptive Notes migration'
git diff --check origin/dev...HEAD
```

Use the verification and finishing-branch skills; merge before creating PR 3.

## PR 3 — TASK-22033: Prompts

### Task 12: Claim Prompts and inventory its existing contracts

**Files:**
- Modify: `backlog/tasks/task-22033 - Migrate-Library-Prompts-to-the-adaptive-reader-shell.md`
- Modify: `Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md`

- [ ] **Step 1: Verify PR 2 is merged and no duplicate work exists**

Run `git fetch origin dev`, verify the PR 2 merge SHA is an ancestor of `origin/dev`, query open PRs
for `22033`, and run `backlog task 22033 --plain`. Expected: merged dependency, To Do task, no
competing implementation.

- [ ] **Step 2: Claim and plan TASK-22033**

```bash
backlog task edit 22033 -s 'In Progress' -a @codex
backlog task edit 22033 --plan $'1. Inventory Prompt capabilities and draft authority\n2. Add one lossless reader projection\n3. Split persistent list and work pane\n4. Verify hidden fields, workflows, geometry, and focus\n\nADR required: yes\nADR path: backlog/decisions/086-library-adaptive-reader-shell.md\nReason: consumes the accepted Library structural boundary without changing Prompt authority.'
```

- [ ] **Step 3: Inventory and commit**

Append local browse/search/paging/sort/collections, bulk selection, Basic/Advanced editor, block
artifacts/recipes, capabilities, validation, import, history preview/restore, memberships,
lifecycle actions, conditional updates, conflicts, and delete recovery, with selector/handler,
target region, and preservation test. Commit the task plus inventory with
`docs(library): inventory Prompt reader migration`.

### Task 13: Add one lossless Prompt reader draft projection

**Files:**
- Create: `tldw_chatbook/Library/library_prompts_reader_state.py`
- Modify: `tldw_chatbook/Library/library_prompts_state.py`
- Create: `Tests/Library/test_library_prompts_reader_state.py`
- Modify: `Tests/Library/test_library_prompts_state.py`

- [ ] **Step 1: Write failing state and lossless-save tests**

Pin Basic as default, Basic/Advanced/Info projections, selected/loaded identity, destination/id/
version/generation settlement, and one referenced `PromptEditorState`/`PromptBlockEditorState`.
Include a non-default Advanced-only value, edit and save from Basic, and require byte-for-byte
preservation in the `prepare_prompt_artifact_save` payload. Validation must return owning mode and
target control without mutating the draft.

- [ ] **Step 2: Run the red suite**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_prompts_reader_state.py Tests/Library/test_library_prompts_seam.py -q
```

Expected: import failure for the new reader module.

- [ ] **Step 3: Implement the minimal projection**

Reference the existing editor/block states and artifact-save seam; do not copy their fields or add
artifact definitions. Accept settlements only when every fence dimension matches.

- [ ] **Step 4: Run and mutation-check hidden-field preservation**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_prompts_reader_state.py Tests/Library/test_library_prompts_state.py Tests/Library/test_library_prompts_seam.py Tests/Prompt_Management/test_prompt_preservation.py Tests/Prompt_Management/test_prompt_artifact_codec.py -q
```

Expected: PASS. Temporarily drop the Advanced-only merge, confirm the regression fails, restore,
and rerun green.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Library/library_prompts_reader_state.py tldw_chatbook/Library/library_prompts_state.py Tests/Library/test_library_prompts_reader_state.py Tests/Library/test_library_prompts_state.py
git commit -m 'feat(library): model lossless Prompt reader state'
```

### Task 14: Split Prompt list and work pane while retaining controllers

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_prompt_work_pane.py`
- Modify: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/__init__.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify generated: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_library_prompts_reader.py`
- Modify existing prompt canvas/browse/collections/history tests

- [ ] **Step 1: Write failing persistent-pane journeys**

In the new reader test, retain list widget identity through Basic/Advanced/Info, import, collection
membership, history preview/restore, validation, conflict, deletion, retry, and selected/loaded
mismatch. Add an 80-column geometry/focus case and prove loaded-item actions are disabled while stale.

- [ ] **Step 2: Run the red suite**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_prompts_reader.py Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_prompt_browse_controller.py Tests/UI/test_library_prompt_collections.py Tests/UI/test_library_prompt_history_controller.py -q
```

Expected: permanent work-pane assertions fail.

- [ ] **Step 3: Split and mount the concrete work pane**

Keep browse/paging/bulk in `LibraryPromptsCanvas`; move persistent editor/lifecycle/history/
provenance content into `LibraryPromptWorkPane`; wire the shared editor profile and
`[library.prompts_reader]`. Preserve capability negotiation, local/server ownership, conditional
writes, memberships, and history controllers. Add no generic field/action schema.

- [ ] **Step 4: Regenerate CSS and run focused suites**

```bash
../../.venv/bin/python tldw_chatbook/css/build_css.py
../../.venv/bin/python -m pytest Tests/Library/test_library_prompts_reader_state.py Tests/Library/test_library_prompts_state.py Tests/Library/test_library_prompts_seam.py Tests/UI/test_library_prompts_reader.py Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_prompt_browse_controller.py Tests/UI/test_library_prompt_collections.py Tests/UI/test_library_prompt_history_controller.py Tests/Prompt_Management/test_prompt_preservation.py Tests/Prompt_Management/test_prompt_artifact_codec.py Tests/Prompt_Management/test_prompt_collection_membership.py Tests/Prompt_Management/test_prompt_history_normalizers.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Library/library_prompts_reader_state.py tldw_chatbook/Widgets/Library/library_prompt_work_pane.py tldw_chatbook/Widgets/Library/library_prompts_canvas.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/__init__.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_library_prompts_reader.py Tests/UI/test_library_prompts_canvas.py
git commit -m 'feat(library): migrate Prompts to adaptive reader'
```

### Task 15: Verify and close PR 3

- [ ] **Step 1: Run automated gates**

Run the Task 14 focused command plus shared shell/config, compose-once/scoped-sync, then:

```bash
../../.venv/bin/python -m ruff check tldw_chatbook/Library/library_prompts_reader_state.py tldw_chatbook/Widgets/Library/library_prompt_work_pane.py tldw_chatbook/Widgets/Library/library_prompts_canvas.py
../../.venv/bin/python -m pytest -q
../../.venv/bin/python -m ruff check .
../../.venv/bin/python -m ruff format --check .
../../.venv/bin/python -m compileall -q tldw_chatbook
../../.venv/bin/python -m pytest Tests/RuntimePolicy -q
git diff --check origin/dev...HEAD
```

Expected: all commands exit 0.

- [ ] **Step 2: Live-verify and draft closeout**

At 160x50, 120x35, 100x30, and 80x24 verify hidden-field preservation, validation focus,
history/provenance truthfulness, bulk preview, import, errors/retry, and restore controls. Append the
after inventory and draft notes/ADR while TASK-22033 remains In Progress.

- [ ] **Step 3: Review, rebase, and repeat the gates**

Use the review skills, address findings, rebase on latest `dev`, and rerun Steps 1–2 exactly.

- [ ] **Step 4: Mark Done, commit closeout, and merge**

Only after final verification, check every AC, finalize Implementation Notes, mark TASK-22033 Done,
commit `docs(library): close adaptive Prompt migration`, run `git diff --check origin/dev...HEAD`,
and merge before PR 4.

## PR 4 — TASK-22034: Skills

### Task 16: Claim Skills and inventory trust/file boundaries

**Files:**
- Modify: `backlog/tasks/task-22034 - Migrate-Library-Skills-to-the-adaptive-reader-shell.md`
- Modify: `Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md`

- [ ] **Step 1: Verify PR 3 is merged and no duplicate work exists**

Fetch `origin/dev`, verify the PR 3 merge SHA is an ancestor, query open PRs for `22034`, and run
`backlog task 22034 --plain`. Expected: merged dependency, To Do task, no competing implementation.

- [ ] **Step 2: Claim and plan TASK-22034**

```bash
backlog task edit 22034 -s 'In Progress' -a @codex
backlog task edit 22034 --plan $'1. Inventory Skills and trust/file authority\n2. Add revision-aware reader presentation state\n3. Split persistent list and work pane\n4. Verify trust, files, geometry, and focus\n\nADR required: yes\nADR path: backlog/decisions/086-library-adaptive-reader-shell.md\nReason: consumes the accepted Library structural boundary without changing Skills trust authority.'
```

- [ ] **Step 3: Inventory and commit**

Append browse/filter/sort, Basic/Advanced edit fields, import, trust setup/unlock/review/approve/
reset, changed files, script grants, supporting/bundle files, tool picker, conflict, deletion, and
recovery. Record existing read/write support per file action; do not infer edit capability. Commit
the task and inventory with `docs(library): inventory Skills reader migration`.

### Task 17: Model Overview/Edit/Trust/Files without changing trust authority

**Files:**
- Create: `tldw_chatbook/Library/library_skills_reader_state.py`
- Modify: `tldw_chatbook/Library/library_skills_state.py`
- Create: `Tests/Library/test_library_skills_reader_state.py`
- Modify: `Tests/Library/test_library_skills_state.py`

- [ ] **Step 1: Write failing revision-aware presentation tests**

Cover Overview default, explicit Edit/Trust/Files, selected/loaded identity, destination/id/version/
generation fencing, reviewed fingerprint/revision, stale review after trust-relevant change, and
unchanged trust after display-only mode changes.

- [ ] **Step 2: Run the red suite**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_skills_reader_state.py -q
```

Expected: import failure for the new reader-state module.

- [ ] **Step 3: Implement a projection of existing service truth**

Compose only from `LocalSkillsService` detail and existing trust fields. The state may identify the
reviewed/current revision and changed files, but must not compute an independent trust verdict.

- [ ] **Step 4: Run and mutation-check trust staleness**

```bash
../../.venv/bin/python -m pytest Tests/Library/test_library_skills_reader_state.py Tests/Library/test_library_skills_state.py Tests/Library/test_skill_trust_review_preview.py Tests/Skills/test_skill_file_trust_material.py Tests/Skills/test_skill_trust_service.py -q
```

Expected: PASS. Temporarily ignore the service's reviewed fingerprint, confirm the stale-review
test fails, restore, and rerun green.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Library/library_skills_reader_state.py tldw_chatbook/Library/library_skills_state.py Tests/Library/test_library_skills_reader_state.py Tests/Library/test_library_skills_state.py
git commit -m 'feat(library): model revision-aware Skills reader state'
```

### Task 18: Split Skills list and work pane

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_skill_work_pane.py`
- Modify: `tldw_chatbook/Widgets/Library/library_skills_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/__init__.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify generated: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_library_skills_reader.py`
- Modify existing Skills canvas/flow/trust/import tests

- [ ] **Step 1: Write failing persistent-pane and truthfulness journeys**

Retain list identity through Overview/Edit/Trust/Files, import, trust review, grants, conflict,
deletion, retry, and rapid A→B selection. Prove actions disable while selected/loaded mismatch;
binary, unavailable, and read-only files are labelled accurately; 80-column geometry retains both
restore controls and the protected editor width.

- [ ] **Step 2: Run the red suite**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_skills_reader.py Tests/UI/test_library_skills_canvas.py Tests/Skills/test_skills_library_flow.py Tests/Skills/test_skills_import.py -q
```

Expected: permanent work-pane and truthfulness assertions fail.

- [ ] **Step 3: Split and mount the concrete work pane**

Keep browse/filter/sort in `LibrarySkillsCanvas`; place editor/trust/file/import/recovery content in
`LibrarySkillWorkPane`; wire the shared editor profile and `[library.skills_reader]`. Render files
read-only unless the inventory proves an existing edit seam. Generation-fence detail/trust workers
and delegate every trust decision to the current service.

- [ ] **Step 4: Regenerate CSS and run focused suites**

```bash
../../.venv/bin/python tldw_chatbook/css/build_css.py
../../.venv/bin/python -m pytest Tests/Library/test_library_skills_reader_state.py Tests/Library/test_library_skills_state.py Tests/Library/test_skill_script_grant_panel.py Tests/Library/test_skill_trust_review_preview.py Tests/UI/test_library_skills_reader.py Tests/UI/test_library_skills_canvas.py Tests/Skills/test_skills_library_flow.py Tests/Skills/test_skills_import.py Tests/Skills/test_local_skills_service.py Tests/Skills/test_local_skills_bundle_io.py Tests/Skills/test_read_skill_file.py Tests/Skills/test_skill_file_trust_material.py Tests/Skills/test_skill_trust_service.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Library/library_skills_reader_state.py tldw_chatbook/Widgets/Library/library_skill_work_pane.py tldw_chatbook/Widgets/Library/library_skills_canvas.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/__init__.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_library_skills_reader.py Tests/UI/test_library_skills_canvas.py
git commit -m 'feat(library): migrate Skills to adaptive reader'
```

### Task 19: Verify and close the programme

- [ ] **Step 1: Run Skills and cross-destination automated gates**

Run the Task 18 command plus Media, Conversations, Notes, Prompts reader journeys, shared shell/
config, compose-once/scoped-sync, then:

```bash
../../.venv/bin/python -m ruff check tldw_chatbook/Library/library_skills_reader_state.py tldw_chatbook/Widgets/Library/library_skill_work_pane.py tldw_chatbook/Widgets/Library/library_skills_canvas.py
../../.venv/bin/python -m pytest -q
../../.venv/bin/python -m ruff check .
../../.venv/bin/python -m ruff format --check .
../../.venv/bin/python -m compileall -q tldw_chatbook
../../.venv/bin/python -m pytest Tests/RuntimePolicy -q
git diff --check origin/dev...HEAD
```

Expected: all commands exit 0 and no prior destination regresses.

- [ ] **Step 2: Perform the live matrix**

At 160x50, 120x35, 100x30, and 80x24 verify Overview default, explicit Edit, trust revision/
staleness, read-only Files, imports, errors/retry, focus continuity, and restore controls. Capture
evidence and compare every destination against its inventory.

- [ ] **Step 3: Draft TASK-22034 and inventory closeout**

Audit TASK-22031–22034 status on current `origin/dev`, append the final after table, check every
prior task is Done, and draft TASK-22034 notes/ADR while it remains In Progress. Add a generalized
lesson only if an actual incident warrants it.

- [ ] **Step 4: Review, rebase, and repeat the gates**

Use the review skills, address findings, rebase on latest `dev`, and rerun Steps 1–2 exactly.

- [ ] **Step 5: Mark Done, commit closeout, and merge**

Only after final verification, check every TASK-22034 AC, finalize notes, mark it Done, commit
`docs(library): close adaptive Skills migration`, run `git diff --check origin/dev...HEAD`, and
merge. After merge, evaluate Library versus Watchlists only as a report; do not extract a global
framework without a new approved ADR and task.

## Final programme acceptance gate

The programme is complete only when:

- TASK-22031, TASK-22032, TASK-22033, and TASK-22034 are Done on `origin/dev` with checked ACs,
  implementation plans, notes, ADR links, automated evidence, and live evidence.
- All five Library reader consumers use the shared structural shell while their behavior remains
  concrete and destination-owned.
- Library collapse expands the active list toward its comfort cap without changing saved width.
- Shared `[library.reader]` and per-destination list sections are normalized, backward compatible,
  and free of responsive writes.
- Media retains every inventoried behavior.
- Conversations provides complete transcript/Find; Notes provides clean Edit/current-draft Preview;
  Prompts preserves Advanced-only data from Basic; Skills reports revision-specific trust and keeps
  unsupported file editing unavailable.
- The final capability inventory has no unexplained missing or newly invented behavior.
