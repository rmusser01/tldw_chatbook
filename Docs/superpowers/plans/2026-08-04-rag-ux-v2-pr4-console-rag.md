# RAG UX v2 — PR-4: Console RAG visibility + honesty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Console's RAG staging visible, truthful, and consistent (critique RAG-40…47) — a staged-evidence strip the user can see and un-stage, consume-on-send semantics that match the modal's own promise, one behavior for the queryless entry points, guarded prefill, editable source scope, and the small-fry fixes.

**Architecture:** All staging state stays on the single existing field (`ChatScreen._pending_console_launch_context`) with its single audited fan-out (`_sync_console_pending_launch_surfaces`) — the new strip joins that fan-out; no second source of truth. The scout map (ledger-referenced) carries exact seams; line refs there are verified against base `80ba9e580`.

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` ONLY (main venv), cwd = worktree root.

## Global Constraints

- Work ONLY in `.worktrees/rag-v2-pr4` (branch `feat/rag-v2-console-rag`, base `80ba9e580`). Absolute paths or `git -C`. NEVER `git stash`. Never `git add -A`.
- **NEVER a bare `python3 -c` importing app modules** (multiple live-config incidents). Probes = pytest under `Tests/`, deleted after.
- Targeted test gates only; SINGLE foreground Bash calls (timeout ≤600000); never idle-wait; never end a turn "standing by".
- No real network in tests. TDD per behavior change.
- Escaping is the TERMINAL step of any display pipeline; staged-evidence titles/snippets in the new strip are library content — escape at render (`_escape_all_brackets` family; check what the tray uses and match it).
- CSS: source (`css/components/_agentic_terminal.tcss`) → `python3 tldw_chatbook/css/build_css.py` → `python3 tldw_chatbook/css/check_bundle_sync.py`; never hand-edit the bundle; on merge conflict regenerate.
- Do NOT add a third citation formatter — Console formats via `format_local_evidence_context`, PR-3 via `format_evidence_for_cited_answer`; both consume one `EvidenceBundle`. The strip renders bundle fields directly.
- Tests pinning literal strings ("Run Library RAG" at `test_console_native_chat_flow.py:3811`, `test_console_internals_decomposition.py:2535`; chip label at `test_console_status_chips.py:51`) get exact-copy updates when copy changes — never weakened.
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Staged-evidence strip + consume-on-send + truthful chip count (RAG-40 + the sticky-staging defect)

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_staged_evidence_strip.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (compose seam between `ConsoleStatusChips` ~:15012-15022 and `ConsoleComposerBar` ~:15024; `_sync_console_pending_launch_surfaces` ~:14170-14178; `_consume_pending_console_launch` ~:3529-3552; `_capture_console_staged_rag` ~:5160-5167; `_build_console_control_state` staged count ~:8164), `tldw_chatbook/Chat/console_display_state.py` (chip count label ~:440), CSS source + bundle
- Test: `Tests/UI/test_console_staged_context.py` (extend), new `Tests/UI/test_console_staged_evidence_strip.py`, `Tests/UI/test_console_local_citation_capture.py` (consume semantics), `Tests/UI/test_console_status_chips.py` (count label)

**The defects (scout-verified):**
1. Staging is invisible on the main surface — the only reader is the Inspector-rail tray, and the success path never auto-opens the rail (only failure does, ~:14338 vs ~:14305-14326).
2. `_consume_pending_console_launch` NEVER clears the field — staged evidence rides every send forever, contradicting the modal's promise "staged for your next send. Running again replaces it." (`console_rag_settings_modal.py:111-114`).
3. The chip hardcodes `staged_source_count=1` while the bundle carries N references (`test_console_local_citation_capture.py:233` pins multi-result staging).

**Produces:**
- `ConsoleStagedEvidenceStrip` mounted between chips and composer: hidden when nothing staged; when staged, one compact row per bundle reference (max 3 shown + "+N more"), each with escaped title + source label, plus ONE un-stage button (`#console-unstage-evidence`) that clears the whole launch context and syncs all surfaces. Strip joins `_sync_console_pending_launch_surfaces`.
- **Consume-on-send:** after `_capture_console_staged_rag` successfully hands the bundle to the send path, the launch context is CLEARED and surfaces synced — the modal's copy becomes true. The clear happens only on successful capture (a blocked/failed send keeps the staging). Immediately after consumption the strip shows a one-send transient "Evidence sent with this message · N sources" line (cleared on the next sync), so the round trip is visible even on unpersisted sessions where the transcript's `Sources (N)` row cannot appear (scout §9).
- Chip reads `Sources: N staged` with the real reference count.

- [ ] **Step 1: Read** the scout's §1 seams in the current tree (line refs may drift): the three writer sites, the fan-out method, the capture provider, the tray (`console_staged_context.py`) for row-rendering + escaping idioms to reuse, and the modal copy. Read `Tests/UI/test_console_staged_context.py` for harness idiom.
- [ ] **Step 2: Failing tests** (one per behavior): (a) strip hidden with no staging; (b) staging via Console's own run → strip lists the bundle's references (escaped — feed a `[bold]` title) with real count; (c) un-stage clears context + strip + chip + tray in one action; (d) send consumes: after a send that captured evidence, a SECOND send captures nothing (pin the clear), and the transient "sent" line rendered then cleared; (e) blocked send does NOT consume; (f) chip label shows N (update the hardcoded-1 pins to real counts — exact-copy).
- [ ] **Step 3: RED → Step 4: implement (widget + wiring + CSS + bundle regen) → Step 5: GREEN**; run the named files plus `Tests/UI/test_console_workbench_contract.py` and `Tests/UI/test_console_internals_decomposition.py -k "rag or staged"`.
- [ ] **Step 6: Commit** `feat(console): staged-evidence strip, consume-on-send, truthful chip count`.

### Task 2: One behavior for queryless entry points (RAG-41/42)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_run_console_library_rag_from_visible_action` toast site ~:14090-14094)
- Test: `Tests/UI/test_console_internals_decomposition.py` or the modal test file (read first; extend where the run-action tests live)

**Scout facts:** four entry points exist — control-bar action + hidden CommandStrip (run-or-toast), Inspector readiness button (disabled when blank), chip (opens modal). The toast fires only when stored query AND composer draft are both empty. The modal's Run callback re-enters the same run method and keeps its own non-blank gate (`console_rag_settings_modal.py:153,166-171,178-179`) — no loop.

- [ ] **Step 1: Failing test:** queryless control-bar action → the RAG settings modal opens (assert pushed screen type), no toast. A second test: modal Cancel afterwards leaves no query set and no retrieval run (the loop-guard proof).
- [ ] **Step 2: RED → implement** (replace the toast with `self._open_console_rag_settings()`) **→ GREEN.** Delete the now-dead `CONSOLE_LIBRARY_RAG_QUERY_EMPTY_MESSAGE` constant IF nothing else reads it (grep; exact-copy test updates).
- [ ] **Step 3: Commit** `feat(console): queryless Run Library RAG opens the modal instead of toasting`.

### Task 3: Prefill guards (RAG-43)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (both prefill sites: modal ~:8919-8922, run-fallback ~:14085-14091), possibly a small helper
- Test: `Tests/UI/test_console_rag_settings_modal.py` (the composer-draft fallback tests at ~:164, ~:211 — extend)

**Scout facts:** two unguarded sites; the run-fallback additionally STORES the draft as the query. Reuse `Chat/console_paste_attach.py` shape detectors (`_is_absolute_token`, `extract_dropped_path`, `looks_attachable`) and the urlparse precedent — do NOT write a new regex family.

- [ ] **Step 1: Failing tests:** a path draft (`/Users/x/notes.md`), a `file://` URI, an `https://` URL, and a >200-char draft each produce NO prefill (modal opens empty) and are never stored as the query by the run-fallback (which should then behave as queryless → Task 2's modal-open). A normal question draft still prefills.
- [ ] **Step 2: RED → implement one shared guard helper used by BOTH sites → GREEN** (both named test files + the Task-2 tests).
- [ ] **Step 3: Commit** `fix(console): don't prefill path/URL/oversized drafts as RAG queries`.

### Task 4: Editable source scope in the modal (RAG-44)

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_rag_settings_modal.py` (scope Static ~:141-147 → toggle row; `ConsoleRagSettingsResult` ~:23-33 gains `source_types`), `tldw_chatbook/UI/Screens/chat_screen.py` (Console-local scope state replacing the bare constant read at ~:14098; label builder ~:13445; callback ~:8959; readiness-card label)
- Test: `Tests/UI/test_console_rag_settings_modal.py`, `Tests/Library/test_library_rag_scope.py` (the three Console pins must stay green)

**Design rulings (binding):**
- This edits **`source_types`** (which KINDS) — explicitly NOT `EffectiveScope` (which item ids); the modal copy must not conflate them (scout §5's two-concepts warning).
- Vocabulary = Library's four (`notes`, `media`, `conversations`, `prompts`) with display labels from `LIBRARY_RAG_SOURCE_TYPES` — no fifth vocabulary. DEFAULT stays exactly today's three (prompts OFF) — no behavior change until the user touches a toggle.
- Console-local persistence: a screen attribute captured/restored with the screen state (mirror how `_console_library_rag_query` persists); NOT shared with Library's screen-local set (scout: not shareable today without promotion — out of scope).
- The readiness-card scope label and the modal derive from the same state via one builder (the PR-2 scope-summary lesson: two seams, one builder).

- [ ] **Step 1: Failing tests:** (a) modal shows four toggles, current selection checked, display-cased labels; (b) toggling media off + Run → the retrieval request's `source_types` excludes media (assert on the fake service's recorded request); (c) Cancel discards toggle changes; (d) the readiness-card label reflects the stored scope ("Scope: Notes, Conversations (Media, Prompts off)" style — reuse Library's summary grammar); (e) default remains the current three with no stored state.
- [ ] **Step 2: RED → implement → GREEN**; run the modal file + `test_library_rag_scope.py` + `test_console_scope_row.py` (the row↔chip agreement must hold).
- [ ] **Step 3: Commit** `feat(console): editable RAG source scope in the settings modal`.

### Task 5: Small-fry (RAG-45/46/47)

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py` (~:1003-1015 status pair), CSS chip rules (`_agentic_terminal.tcss` ~:2116-2133), `tldw_chatbook/Widgets/Console/console_session_surface.py` (`CONSOLE_NEW_TAB_BUTTON_WIDTH` ~:36)
- Test: nearest owning test files (read `test_console_workspace_context_rail.py`, `test_console_composer_menu.py:660` chip-label pin, `test_console_rail_sections.py`)

- [ ] **RAG-45:** the workspace status pair labeled "Scope" shows CONVERSATION identity (the comment at ~:963-966 admits the collision) and renders bare on fresh sessions. Rename the label to what it shows (e.g. "Conversation") AND give the empty state a placeholder ("—"). Update pins.
- [ ] **RAG-46:** chip hit-target: raise `.console-control-chip` min-width to 12 and padding to `0 2` (source + bundle regen); do NOT restructure the strip (the ±1-row shift is composer-height-driven — record as accepted with the scout's two options noted for a future pass; changing composer layout is out of scope).
- [ ] **RAG-47:** "Temporar" — widen `CONSOLE_NEW_TAB_BUTTON_WIDTH` to fit "Temporary" + padding (13); verify the sibling `#console-new-chat-tab` still fits; no test pins the literal, add one.
- [ ] **Commit** `fix(console): scope-pair honesty, chip target, Temporary button width`.

### Task 6: Targeted verification + live check

- [ ] **Step 1: Targeted gate**, ONE foreground call: `Tests/UI/test_console_staged_context.py Tests/UI/test_console_staged_evidence_strip.py Tests/UI/test_console_rag_settings_modal.py Tests/UI/test_console_status_chips.py Tests/UI/test_console_shell_chip_actions.py Tests/UI/test_console_local_citation_capture.py Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_scope_row.py Tests/Library/test_library_rag_scope.py Tests/UI/test_library_rag_handoffs.py` plus `Tests/UI/test_console_internals_decomposition.py -k "rag or staged or scope"`. Zero new failures.
- [ ] **Step 2:** collect-only sweep (`Tests/UI/ Tests/Library/`) 0 errors; ruff on changed files; `check_bundle_sync.py`.
- [ ] **Step 3: Live check** — the proven recipe (scratch profile `verify_ragpr4`, COPY ChaChaNotes+media+`chromadb/` BEFORE first launch, `[first_run]` flags; socket `ragpr4-805d`; live-config before/after proof; cleanup mandatory; tmux column arithmetic via python char-index, never byte offsets; wheel-scroll `\x1b[<64;COL;ROW M`). Optionally configure the real provider via the repo-root Anthropic key ONLY inside the scratch config (delete after) to exercise a full stage→send round trip. Verify: (a) staging renders the strip with real titles + count; (b) un-stage clears strip+chip+tray; (c) a send consumes — strip shows the transient then clears, chip returns to `RAG: off`; (d) queryless control-bar action opens the modal; (e) path-shaped composer draft does not prefill; (f) modal scope toggles change the label and the retrieval; (g) "Temporary" renders unclipped; (h) workspace pair no longer bare. Evidence files; honest NOT-OBSERVABLE where timing defeats capture.
- [ ] **Step 4:** report; commit fixes; NO PR creation (controller ships).
