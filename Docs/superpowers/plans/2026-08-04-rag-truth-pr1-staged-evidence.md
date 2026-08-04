# RAG Truth PR-T1 — Staged-Evidence Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the five truth-split defects from the 2026-08-04 re-score critique (21/40): the Console Inspector's three staged-evidence lies, the silently swallowed first send, staged evidence lost on navigation, scope-off results remaining stageable, and the false "No Library sources yet" banner.

**Architecture:** One store (`ChatScreen._pending_console_launch_context`) causes three defects — fix by (a) centralizing every count/label on the existing `console_staged_source_count` truth function and giving the Inspector the strip's one-send memory, (b) making Library-staged launches carry a real evidence bundle, and (c) serializing the launch + sent-notice into native console state so navigation stops destroying it. The swallowed send is the `""` session-id sentinel on a fresh profile's first send — resolve the session at dispatch time, make every refusal path speak, and guard the no-op `Button.press()`. The Library panel gets a one-snapshot scope filter in `from_values` and a change-gated recovery sync.

**Tech Stack:** Python ≥3.11, Textual 8.2.7, pytest.

**Branch/worktree:** `feat/rag-truth-staged-evidence` in `.worktrees/rag-truth-pr1`, base `0c6f80487`. All paths relative to the worktree root. Full seam evidence: the scout map in the SDD workspace (`scout-map.md`) — every line ref below was verified at `0c6f80487`; re-verify before editing, never edit blind.

## Global Constraints

- **Consume-on-send predicate byte-equality:** the release predicate in `chat_screen.py:5327-5334` must stay byte-equal to `ConsoleChatController._capture_rag_context`'s prepend predicate. Any change to when the launch clears re-verifies against `console_chat_controller.py:5711+`. Release must never cost the send its evidence (`chat_screen.py:5379-5395` clear-first pattern; pinned by `Tests/UI/test_console_staged_evidence_strip.py:579`).
- **Always-mounted + display-toggle, never recompose the shell** (strip `console_staged_evidence_strip.py:50,105-109`; composer bar idiom). The sent-notice is cleared by events, not timers.
- **Escaping is terminal:** all new "Sources N"/"sent" copy renders on `markup=False` Statics; `_safe_display_text` (`console_display_state.py:73`) is the escaper.
- **Targeted tests only** (owner ruling): each task's gate = touched test files + `pytest Tests/ --collect-only -q | tail -3`. Never full Tests/UI. venv-only pytest (`source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate`); pytest under Tests/ is the ONLY python entry point (bare `python3 -c` importing app modules writes the LIVE config). `git stash` FORBIDDEN.
- **Protected oracles:** pre-existing tests change ONLY where this plan names a deliberate contract change; name every such change in the task report.
- **Quiet register** for all user-facing copy; no "Owner:/Recovery:" internal register.
- **Backlog IDs:** new tasks file at **2330+** (cross-worktree max 2320, scanned 2026-08-04; re-scan at ship).
- **CSS bundle:** never hand-edit; regenerate via `python3 -m tldw_chatbook.css.build_css` if any `.tcss` source changes; the class-coverage contract (`Tests/UI/test_css_class_coverage_contract.py`) must stay green — style or registry any new class.

## File Structure

| File | Responsibility in this PR |
|---|---|
| `tldw_chatbook/Chat/console_display_state.py` | T1: `source_count` on `ConsoleStagedContextState`; Inspector sent-memory copy |
| `tldw_chatbook/Widgets/Console/console_staged_context.py` | T1: tray renders source count, not row count |
| `tldw_chatbook/UI/Screens/chat_screen.py` | T1: inspector state threading; T2: (via library payload) send-gating check; T3: serialize/restore launch+notice; T4: dispatch-time session resolution, press guard |
| `tldw_chatbook/UI/Screens/library_screen.py` | T2: attach evidence bundle at `:17123`; T6: change-gated recovery sync |
| `tldw_chatbook/Chat/console_chat_controller.py` | T4: silent refusal paths append system rows |
| `tldw_chatbook/Library/library_rag_state.py` | T5: one-snapshot scope filter in `from_values` |
| `backlog/tasks/` | T7: new true-defect tasks at 2330+; close 2075 |
| Tests per task | named in each task |

---

### Task 1: One truth for the staged count; give the Inspector the strip's memory (D1a + D1b)

**Files:**
- Modify: `tldw_chatbook/Chat/console_display_state.py` (`ConsoleStagedContextState.from_live_work` :495-528; `console_staged_source_count` :550-570; inspector builder :864-898)
- Modify: `tldw_chatbook/Widgets/Console/console_staged_context.py:63-67`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_console_rag_source_status` :12893-12909; `_build_console_inspector_state` :12938+ threading `sent_source_count`)
- Test: `Tests/UI/test_console_staged_context.py`, `Tests/UI/test_console_staged_evidence_strip.py`, `Tests/UI/test_console_internals_decomposition.py` (additive only)

**Interfaces:**
- Produces: `ConsoleStagedContextState.source_count: int` (from `console_staged_source_count(launch)`); tray heading count renders it. Inspector `rag_status` vocabulary gains one value: when no launch is pending AND the one-send notice is armed, the row reads `sent with the last message · N sources` instead of `not staged` (N = the notice count). The `not staged` literal remains for the genuinely-empty state.
- Consumes: `_console_evidence_sent_notice` (`chat_screen.py:3184`), armed at release (`:5384`), cleared by events (`:5431-5436`, `:14453`, `:14550`).

- [ ] **Step 1: Write failing tests.**
  - `test_console_staged_context.py`: build a `ConsoleStagedContextState` via the REAL `from_live_work` with a 5-reference evidence bundle (use the real `build_library_rag_evidence_bundle` fixtures or a launch payload carrying a 5-ref `evidence_bundle` — NOT a hand-built 1-row state; the existing 1-row test at `:19-36` is exactly the gap that let "Sources 18" ship) and assert the tray renders `5`, not `len(rows)`.
  - Inspector: with a pending launch → unchanged behavior; with `pending_launch=None` and `sent_source_count=5` → row text contains `sent with the last message · 5 sources`; with neither → `not staged`.
  - Agreement test (new): after simulating release (launch None, notice 5), the strip's sent line count and the inspector's sent count come from the same number.
- [ ] **Step 2: Run to verify failures** (`pytest Tests/UI/test_console_staged_context.py Tests/UI/test_console_internals_decomposition.py -q -k "source_count or sent_with"` — adapt -k to your test names).
- [ ] **Step 3: Implement.** `from_live_work` computes `source_count=console_staged_source_count(launch)` and carries it on the dataclass; `console_staged_context.py:64` renders `str(state.source_count)`. `_console_rag_source_status(pending_launch, sent_source_count)` (widen the signature; the caller at `:12983` passes `self._console_evidence_sent_notice`): pending → existing derivation; none+notice → the new sentence; none+none → `not staged`. Inspector builder threads it through `console_display_state.py:864-898` (keep `markup=False` surfaces; run recipe line updates consistently).
- [ ] **Step 4: Run the targeted gate** (the three test files + `pytest Tests/ --collect-only -q | tail -3`).
- [ ] **Step 5: Commit** — `fix(console): tray counts sources not rows; inspector remembers the sent evidence`

---

### Task 2: Library-staged launches carry a real evidence bundle (D1c)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:17123-17126` (`_stage_library_rag_result_in_console` → `build_library_rag_console_live_work_payload` call)
- Test: `Tests/UI/test_library_rag_handoffs.py`, `Tests/UI/test_console_live_work_handoffs.py` (additive)

**Interfaces:**
- Consumes: the Console-side Run precedent at `chat_screen.py:14694-14696` (`launch_payload["evidence_bundle"] = build_library_rag_evidence_bundle(...).to_payload()`) — mirror its construction for the Library side's selected result.
- Produces: every launch payload carries `evidence_bundle`; `console_staged_source_count` stops falling through to the literal `1` (`console_display_state.py:565-570`); `console_prompted_source_count` stops returning 0 for Library-staged sends (`:592-594`).

- [ ] **Step 1: Write failing tests.** Stage from Library (the real handoff path the existing `test_console_staged_evidence_strip.py:620` uses) and assert the claimed launch's payload contains a non-empty `evidence_bundle` whose reference count matches the staged selection; assert the chip reads the real count.
- [ ] **Step 2: Verify failures.**
- [ ] **Step 3: Implement** — build the bundle from the selected result at the Library call site (one selected result today → a 1-ref bundle; construct via the same builder Console Run uses so the payload shape is byte-compatible). CHECK the named blast radius: `_console_send_blocked_reason` (`chat_screen.py:17694-17702`) gates sends on `evidence_state.available_count == 0` — adding bundles to Library launches changes that path's inputs; write one test pinning that a Library-staged launch with 1 available ref is sendable and a 0-ref bundle blocks with the existing copy.
- [ ] **Step 4: Targeted gate + collect sweep.**
- [ ] **Step 5: Commit** — `fix(library): Use in Console stages a real evidence bundle`

---

### Task 3: Staged evidence survives navigation (D3)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_serialize_native_console_state` :15912-15932; `_restore_native_console_state` :16026-16108; `_consume_pending_console_launch` early-return :3657-3658)
- Test: `Tests/UI/test_console_live_work_handoffs.py` (new navigate-away-and-back case), `Tests/UI/test_screen_navigation.py` if it hosts the save/restore harness (read it first)

**Interfaces:**
- Consumes: `ConsoleLiveWorkLaunch.to_pending_payload()` (`Chat/console_live_work.py:97-105`) and `ConsoleLiveWorkLaunch.from_pending`; `NATIVE_CONSOLE_STATE_VERSION`.
- Produces: `pending_console_launch` + `console_evidence_sent_notice` keys in native console state, restored on screen re-creation. A restored launch must NOT re-claim the `PendingHandoffStore` channel (the `:3657` early return must treat a restored launch as already-claimed).

**Context:** screens are never reused (`app.py:6267-6289` constructs fresh instances per navigation); the launch was screen-instance state, so ANY navigation away destroyed it. The critique blamed Library's Run; the scout proved Run is pure — teardown is the destroyer. The live check's "plain navigation preserves it" control was a mis-observation or an unconsumed-handoff case.

- [ ] **Step 1: Write the failing test**: stage a launch (real handoff path), serialize native console state, build a fresh screen, restore, assert the pending launch (with its evidence bundle — Task 2 landed) and the strip render; assert the handoff store is NOT re-claimed; assert an armed sent-notice round-trips.
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement** serialize/restore with legacy-key tolerance (absent keys → no launch, no notice — old payloads must restore cleanly; check whether `NATIVE_CONSOLE_STATE_VERSION` conventions require a bump by reading its consumers). The launch payload's `evidence_bundle` is already `to_payload()`-shaped and round-trips.
- [ ] **Step 4: Targeted gate + collect sweep.**
- [ ] **Step 5: Commit** — `fix(console): staged evidence and sent-notice survive navigation`

---

### Task 4: The swallowed send — resolve the session at dispatch; no silent refusals; guard the no-op press (D2)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (dispatch `:17847-17870`; post-await guards `:17504-17522`; Enter handler `:21656-21675`)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`_active_run_rejection` :8041-8078; `_session_closed_result`; `submit_draft` ensure_session `:1674-1677`)
- Test: `Tests/UI/test_console_send_draft_snapshot.py` (additive fresh-profile cases), `Tests/UI/test_console_composer_undo.py` if touched

**Interfaces:**
- Produces: `target_session_id` is never `""` — the screen resolves/creates the session at dispatch time (mirroring the mount-time creator's `settings=self._default_console_session_settings()` at `:17297-17303`, closing the no-settings race the scout found in `submit_draft`'s bare `ensure_session`); every stash map, worker group, and guard keys on the resolved id. Silent refusal paths (`_active_run_rejection`, `_session_closed_result`) append a SYSTEM row exactly like `_block` does (`console_chat_controller.py:5698-5702`) — no refusal may leave the transcript unchanged. The Enter handler checks the pressed button's `display`/`disabled` and restores the stash when the press was a no-op (Textual 8.2.7 `Button.press` returns immediately when `disabled or not display` — verified against the installed package).

- [ ] **Step 1: Write failing tests** (fresh-profile arrangement — no active session before the send; the existing suite all assumes one, which is exactly the gap):
  - First send with no active session: draft is either delivered (row appears, durably persisted) or restored-with-visible-refusal; assert the stash/guard maps are keyed on the RESOLVED session id.
  - `_active_run_rejection` and `_session_closed_result` each append a visible SYSTEM row.
  - No-op press: disable/hide the send button, Enter → stash restored to composer, no silent loss; second Enter path (the `:21656-21661` duplicate guard) cannot permanently swallow a draft.
- [ ] **Step 2: Verify failures.**
- [ ] **Step 3: Implement** per the interface block. Prefer dispatch-time resolution over post-hoc re-keying (one place, before any await). Keep `send_refusal_copy`/`run_state_for` behavior for real sessions byte-identical; the double-send gate now reads the resolved id.
- [ ] **Step 4: Targeted gate + collect sweep.**
- [ ] **Step 5: Commit** — `fix(console): first-send session sentinel, silent refusals, and no-op press swallow`

---

### Task 5: Scope-off results leave the stage (D4)

**Files:**
- Modify: `tldw_chatbook/Library/library_rag_state.py` (`LibraryRagPanelState.from_values` :1565-1578; `can_use_console` :1681-1683; heading/coverage recompute :1571)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:16584-16604` (docstring only — the "scope only affects the NEXT run" claim becomes false)
- Test: `Tests/UI/test_product_maturity_gate16_library_search_rag.py` or `Tests/UI/test_library_shell.py` (additive)

**Interfaces:**
- Produces: `from_values` filters `result_rows` to `scope.selected_source_types` BEFORE resolving `selected_result`/`can_use_console` — a scope toggled off hides its rows in the same recompose the toggle already triggers, clears a now-invalid selection, and recomputes `results_heading_text`/`coverage_note` from the filtered set. **Design ruling (this plan):** hide, don't grey — the scope line already claims the source is off; showing its rows is the lie. The answer region is untouched (its staleness guards `_library_rag_answer_query/_mode` already exist; an answer grounded in newly-hidden rows keeps rendering — it answers the query it names).
- Consumes: `LibraryRagResultRow.source_type`; the toggle's existing `refresh(recompose=True)` (`library_screen.py:16603`).

- [ ] **Step 1: Write failing tests**: search across sources, toggle one off → its rows absent from `result_rows`, selection on a hidden row cleared, `can_use_console` False for it, `u`/stage path (`library_screen.py:17073-17091`) refuses; toggle back on → rows return (state, not re-query); heading and coverage note reflect the filtered set.
- [ ] **Step 2: Verify failures.**
- [ ] **Step 3: Implement** the filter inside `from_values` (pure, one snapshot — no change in library_screen beyond the docstring correction).
- [ ] **Step 4: Targeted gate + collect sweep.**
- [ ] **Step 5: Commit** — `fix(library): scope-off results are hidden and unstageable in the same snapshot`

---

### Task 6: The false "No Library sources yet" banner (D5, closes task-2075)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`_sync_library_rag_scope_toggle_and_run_gate_widgets` :17398-17429; read the lock warning at :17376-17383 first)
- Test: `Tests/UI/test_product_maturity_gate16_library_search_rag.py` (cold-boot case per 2075 AC3), `Tests/UI/test_library_shell.py`

**Interfaces:**
- Produces: a change-gated recovery sync — cache the last `library_rag_scope_shows_recovery(...)` result on the screen; only when it flips, mirror the full-refresh block (`:17481-17493`: `set_class` `has-recovery` + remove `#library-rag-scope-recovery`/`#library-rag-open-import-export` + re-mount `library_rag_scope_recovery_children`). Steady-state snapshots take the no-op path, preserving RAG-27's no-eject guarantee (2075 AC2). Covers BOTH producers: the cold-boot in-place snapshot (`:2420-2462`) and the timeout all-zeros snapshot (`_apply_source_snapshot_timeout` :2464-2477).
- Consumes: `library_rag_scope_shows_recovery` (`library_search_rag_panel.py:167-176`), gate copy `library_rag_state.py:127-128`.

- [ ] **Step 1: Write failing tests**: drive `_apply_local_source_snapshot`'s in-place branch with real counts after a zero-count state → recovery block gone and `has-recovery` cleared without a full refresh; the timeout producer's all-zeros snapshot → recovery appears; steady-state repeat snapshot → no mount/remove churn (assert widget identity stable). Cold-boot arrangement per 2075 AC3 (`default_tab = "search"` boot).
- [ ] **Step 2: Verify failures.**
- [ ] **Step 3: Implement** with the sync's no-yield constraint respected (non-awaited `Widget.remove()`/`mount()` scheduling, or hold `_library_rag_panel_refresh_lock` — read the four other refresh callers before choosing; say which you chose and why in the report).
- [ ] **Step 4: Targeted gate + collect sweep.**
- [ ] **Step 5: Commit** — `fix(library): recovery banner syncs on in-place snapshots (closes the cold-boot lie)`

---

### Task 7: Backlog truth — file the real defects, close 2075, note the 2211/2212 mismatch

**Files:**
- Create: `backlog/tasks/task-2330 - Console inspector staged-evidence truth (critique D1).md` — RESOLVED-BY-THIS-PR record of D1a/b/c (so the critique finding has a durable task trail), status Done with implementation notes pointing at Tasks 1-2.
- Create: `backlog/tasks/task-2331 - Console first-send swallowed draft (critique D2).md` — same, Done, pointing at Task 4.
- Modify: `backlog/tasks/task-2075*` — check ACs, add Implementation Notes, status Done (Task 6 closed it).
- Modify: `backlog/tasks/task-2211*` and `task-2212*` — append a dated note: the 2026-08-04 critique initially mapped its P0s to these IDs; scout analysis showed they describe DIFFERENT defects (strip-vs-tray vocabulary; post-cancel evidence loss) which REMAIN open and valid.

- [ ] **Step 1:** Re-run the cross-worktree ID scan (`git worktree list` + ls backlog/tasks) — if 2330/2331 are taken, leapfrog and note it.
- [ ] **Step 2:** Write the four files per the repo's task frontmatter format (copy an existing task's shape).
- [ ] **Step 3: Commit** — `chore(backlog): file critique D1/D2 records, close 2075, annotate 2211/2212 mismatch`

---

### Task 8: Whole-branch review, live verification, ship

- [ ] **Step 1: Whole-branch review** (strongest model), full diff from `0c6f80487`. Named watch-items: the release-predicate byte-equality constraint (Tasks 1-3 all orbit the launch store); Task 2's send-gating blast radius; Task 3's restore-must-not-reclaim; Task 4's guard re-keying vs the worker group name; Task 5's interaction with the answer region; Task 6 under the refresh lock. Point it at the ledger's deferred minors.
- [ ] **Step 2: ONE fix wave + scoped re-review** for findings.
- [ ] **Step 3: Live verification** (scratch profile recipe, tmux, session-suffixed socket; NO API key needed except ONE paid send to verify D2's fix on a genuinely fresh profile — budget 2 calls):
  1. Fresh profile, FIRST send with staged evidence → message renders, DB row exists, Inspector shows `sent with the last message · N sources`, tray count = source count (not 18).
  2. Stage in Console → navigate to Library → back → strip still shows the staged evidence; Run in Library, return → Console staging intact.
  3. Library "Use in Console" → chip shows the real count; send blocked/allowed per bundle availability.
  4. Toggle a scope off with results showing → rows vanish, selection clears, `u` refuses; toggle on → rows return.
  5. Cold-boot with `default_tab = "search"` → no false banner above real counts; simulate/observe the timeout path if cheaply reachable.
- [ ] **Step 4: Docs** — update Console/Library user-guide pages where behavior changed (staged evidence persistence, scope filtering); stamp with the live-check commit.
- [ ] **Step 5: Ship** — merge latest origin/dev (regenerate CSS bundle on conflict, never hand-merge), targeted gates, fresh backlog ID re-scan, push, PR (`RAG Truth PR-T1: staged-evidence truth — inspector honesty, surviving navigation, first-send integrity, scope honesty, banner fix`), merge on verified, confirm `.merged` (exit-1-still-merges trap).

---

### Task 9: Non-RAG handoffs deliver their staged content (Task-2 review bonus find; execute BEFORE Task 8)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_stage_handoff_as_console_live_work` :16557-16632 — the `"rag" in source` gate at :16584)
- Test: `Tests/UI/test_console_live_work_handoffs.py` (additive)

**Interfaces:**
- Context (verified by the Task-2 review): media (`library_screen.py:3361`, source="library"), conversations (`:3318`, source="library"), and notes (`library_screen.py:7258`, source="notes") "Use in Console" handoffs never enter the bundle-building branch, and `capture_console_staged_evidence_for_chat` (`chat_rag_events.py:1601-1640`) returns `LocalRagContextResult(None, None)` whenever `payload.get("evidence_bundle")` isn't a mapping — so those three handoffs SILENTLY send with zero staged content reaching the model while the strip/tray show content staged. Live content-loss bug.
- Produces: every handoff staged through `_stage_handoff_as_console_live_work` carries an `evidence_bundle` built from the handoff's content (reuse the same bundle-construction the `"rag"` branch uses, adapted to the handoff's source/authority fields — read that branch first; do NOT invent a new bundle shape). The consume-on-send predicate and release path stay untouched.

- [ ] **Step 1: Write failing tests**: stage a media handoff (real path) → launch payload carries a non-empty `evidence_bundle`; `capture_console_staged_evidence_for_chat` returns real context (not (None, None)); repeat for a notes handoff. Pin that the existing `"rag"` branch behavior is byte-unchanged.
- [ ] **Step 2: Verify failures.**
- [ ] **Step 3: Implement** — build the bundle for all sources at the one seam (drop or widen the `"rag" in source` conditional; whichever you choose, say why in the report). Check the same send-gating blast radius Task 2 pinned (`_console_send_blocked_reason` evidence gate).
- [ ] **Step 4: Targeted gate** (`Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_console_staged_evidence_strip.py -q` + collect sweep).
- [ ] **Step 5: Commit** — `fix(console): non-RAG handoffs deliver their staged content to the model`

## Self-Review Notes

- Coverage: D1a/b → T1, D1c → T2, D3 → T3, D2 → T4, D4 → T5, D5 → T6; backlog truth → T7; ship → T8. The critique's P1s outside this PR's scope (MCP tool honesty, paid-moment visibility, provider split-brain) belong to PR-T2/T3 per the program plan — not lost, deliberately out of scope here.
- Type consistency: `source_count` named identically in T1's dataclass and tray; `sent_source_count` parameter name consistent between `_console_rag_source_status` and the inspector builder; T3 serializes whatever launch shape T1/T2 settle (T3 runs after both).
- Known unknowns delegated with read-first instructions: `NATIVE_CONSOLE_STATE_VERSION` bump conventions (T3), the recovery-sync locking choice (T6), the save/restore test harness location (T3).
