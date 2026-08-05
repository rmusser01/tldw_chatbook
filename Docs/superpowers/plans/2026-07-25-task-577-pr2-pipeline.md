# task-577 PR2 — Dead-Pipeline Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 2 of task-577 — retire the dead legacy chat pipeline: `chat_events.py` (whole file), `chat_events_sidebar.py`, `chat_streaming_events.py`, `chat_worker_handler.py`, the dead ~75% of `worker_events.py`, and the app.py legacy dispatch fabric + zero-reader reactives — while preserving the two LIVE seams the gate scout proved: `app.chat_wrapper`→`chat_wrapper_function` (MediaWindow_v2 media analysis) and `chat_token_events.py`.

**Architecture:** 562-method deletion campaign, 4 tasks in callers-before-callees order: relocations first, app.py fabric second, file deletions third, guards/docs last. The PR1 scout verdicts + the PR2 gate-resolution scout (2026-07-25, post-c7bcc6fdd) are binding ground truth; every deletion still re-gates at implementation time.

**Spec:** `Docs/superpowers/specs/2026-07-25-task-577-enhanced-window-retirement-design.md` Phase 2 (P1-P4) — user-approved; Ambiguous Gates A/B now RESOLVED by scout: A=LIVE (keep reduced worker_events), B=DEAD (delete Action handlers).

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign`, branch `claude/task-577-pr2-pipeline` off dev `c7bcc6fdd`. Subagent shells start in the MAIN checkout; a hook strips a LEADING `cd` — prepend `true; cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign; ` to EVERY Bash command.
- Test prefix: `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`. Tests/UI never mixed with other dirs. FOREGROUND ONLY. Stage explicit paths; never `.superpowers/`.
- After every commit: venv pyflakes on touched production files + `python -c "import tldw_chatbook.app"` + the task's suites + `pytest --collect-only -q` on affected test roots.
- **LIVE KEEP-SET (never touch):** `app.chat_wrapper` (app.py:9825) + `worker_events.chat_wrapper_function` (:1082) + the `app.py:247` worker_events import; `chat_token_events.py`; `Widgets/Chat_Widgets/chat_message.py`/`chat_message_enhanced.py` (widget modules — TTS handlers + chat_token_events query them, documented empty-result fallbacks); the TTS handlers at app.py ~:5894-5992; `MediaWindow_v2.py` + `media_viewer_panel.py`; `ServerWorkerHandler`/`MiscWorkerHandler`/`AIGenerationHandler` + the registry; `conv_char_events.py` (CCP stratum — only receives the relocation); `Tests/fixtures/event_handler_mocks.py` (live consumer: test_llm_management_events — prune only entries whose last consumers die); `Chat/chat_models.py:58` `current_ai_message_widget` dataclass field (INDEPENDENT of the app attr).
- Gate discipline: per-symbol grep-gates pasted verbatim; unexpected live hit ⇒ DEFER; callers before callees; reviewers re-run gates.
- Scout line refs are post-PR1 (c7bcc6fdd) and may drift; locate by symbol.

---

### Task 1: symbol relocations + the handoff-helper gate

**Files:** Modify `tldw_chatbook/Event_Handlers/conv_char_events.py`, `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py` (removal of relocated symbols happens in T3 — this task only ADDS the new homes and repoints importers), possibly a new small module for the handoff helpers, `Tests/Chat/test_answer_citations.py`, `Tests/UI/test_chat_first_handoffs.py` (import repoints).

**Interfaces (T3 depends on these):** after T1, NOTHING outside `chat_events.py` imports FROM `chat_events.py` except app.py (T2's job) and the dying test files.

- [ ] **Step 1 — the handoff-helper gate (decides this task's shape):** enumerate ALL production callers of `apply_current_handoff_context` and `attach_current_handoff_citation_validation` (grep both names across tldw_chatbook/). Known: `attach_...` is called from worker_events (dead) + chat_streaming_events (dead). `apply_current_handoff_context`'s callers are UNKNOWN — this is the gate. Outcomes: (a) a LIVE caller exists (e.g. chat_screen handoff staging) → RELOCATE both helpers (+ their private dependencies) verbatim into a new `tldw_chatbook/Chat/chat_handoff_context.py` (or an existing live Chat/ module if one is clearly the right home), repoint the live caller + the two test files' imports; (b) zero live callers → the helpers die with chat_events in T3 and their pinning tests (`test_answer_citations.py`, the two `test_chat_first_handoffs.py` references at ~:939/:963) are casualties — delete/update the tests NOW so T3 is clean. Paste the gate output; state the outcome explicitly.
- [ ] **Step 2 — relocate `load_branched_conversation_history_ui`:** first GATE the CCP consumer itself: `conv_char_events.py:1302`'s enclosing handler — is it reachable (its trigger)? Regardless of the answer, the relocation is the low-risk move (CCP stratum is out of scope — keep it importable): copy the function + any chat_events-private helpers it uses verbatim into `conv_char_events.py` (private, renamed `_load_branched_conversation_history_ui` if convention fits), change `conv_char_events.py:44` to stop importing from chat_events, and leave the chat_events original in place (T3 deletes it with the file). pyflakes conv_char_events.
- [ ] **Step 3 — verify:** import smoke; `grep -rn "from.*chat_events import\|import chat_events" tldw_chatbook/ | grep -v "app.py\|chat_events"` → only hits that die in T2/T3 (paste); run `Tests/Chat/test_answer_citations.py` (if kept) + `Tests/UI/test_chat_first_handoffs.py` + `Tests/Event_Handlers/` (conv_char tests if any exist — locate by grep).
- [ ] **Step 4 — commit:** `refactor(chat): task-577 PR2 T1 — rehome branched-history loader + resolve handoff-helper liveness`

### Task 2: app.py fabric + reactives

**Files:** Modify `tldw_chatbook/app.py` only (+ test casualties by grep).

**Deletion set (each gated; scout refs post-PR1):**
- The two chat_events imports (`:253` aliased, `:254` plain) — LAST, after every user below is gone.
- Action arms + methods: `:5832` `@on(ChatMessage.Action) handle_chat_message_action`, `:5847` `@on(ChatMessageEnhanced.Action) handle_chat_message_enhanced_action` (Gate B resolved DEAD — widgets unmountable; the ChatMessage/ChatMessageEnhanced IMPORTS at :311-312 stay ONLY if the live TTS handlers still need them — they do; keep imports).
- Dead dispatch arms: `_execute_tab_switch` chat arms (`:8359`, `:8362`) — NOTE: `watch_current_tab` returns unconditionally at `:8187` (screen-nav hard-True) — delete `_execute_tab_switch` WHOLE + the legacy body of `watch_current_tab` beyond its early-return shape IF gates confirm zero other callers (spec P3 explicitly includes this); Collapsible arms `:9078-9091` (`#chat-active-character-info-collapsible`), `:9097-9119` (`#chat-conversations`), `:9061` (`#chat-notes-collapsible` — id composed nowhere live, gate it); input arms `:9208` (`#chat-prompt-search-input`), `:9216` (`#chat-template-search-input`); the `:9285` log-string cleanup (arm dispatches to ccp_handlers on a nowhere-composed id — gate: delete the whole arm).
- Streaming arms: `:9428` `@on(StreamingChunk)`, `:9432` `@on(StreamDone)` + the `:96` `StreamingChunk, StreamDone` import (T3 deletes the classes — arms/import must go FIRST, here).
- Fabric: `_build_handler_map` (`:5201`) + the `:3362` assignment + `button_handler_map` attribute — CONFIRMED write-only; the INGEST/CCP/other map folds disappear with it (the source dicts in their own modules remain, unreferenced-but-out-of-scope). `on_button_pressed` (`:9153`) STAYS as the no-op screen-nav guard.
- `set_current_ai_message_widget`/`get_current_ai_message_widget` (`:6457-6469`) + the `current_ai_message_widget` attr (`:2800`) — gate: remaining users are chat_streaming_events (dies T3) + chat_worker_handler (dies T3) — deleting here is safe ONLY if nothing else reads the attr; since T3 deletes the readers, delete the accessors+attr in T3 instead if ordering demands; otherwise here with the gate proving only-T3-victims reference them (either placement acceptable — callers-before-callees governs).
- Reactives with ZERO live readers (scout-verified): `current_chat_conversation_id` (`:2899`), `current_chat_is_ephemeral` (`:2895`), `current_chat_active_character_data` (`:2902`), `active_chat_tab_id` (`:2909`), `chat_sessions` (`:2910`) + the `:2907-2908` stale comment block. Same ordering note as above: their only remaining readers are the T3-dying files — if deleting them here breaks import of those files (module-level references? no — runtime only), delete here; the T3 files' runtime reads never execute.

- [ ] **Step 1:** gates for every item above (paste); classify each arm's method: whole-method delete vs arm-only.
- [ ] **Step 2:** edit in dependency order; read every edited method end-to-end; pyflakes; import smoke.
- [ ] **Step 3:** test casualties by grep (app.py-pinning tests referencing deleted arms/reactives — enumerate, update/delete).
- [ ] **Step 4:** run `Tests/Event_Handlers/ Tests/test_smoke.py` + `Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_screen_state.py` + collect-only sweeps. Green.
- [ ] **Step 5 — commit:** `refactor(chat): task-577 PR2 T2 — retire app.py legacy dispatch fabric, dead arms, zero-reader chat reactives`

### Task 3: the file deletions + worker-family gutting

**Files:** `git rm`: `Event_Handlers/Chat_Events/chat_events.py`, `chat_events_sidebar.py`, `chat_streaming_events.py`, `Event_Handlers/worker_handlers/chat_worker_handler.py`. Modify: `Event_Handlers/worker_events.py` (gut to the live core), `app.py` (`:230` chat_worker_handler import + `:5194` registration; the worker_events `:96` import already gone in T2), `Tests/` casualties.

- [ ] **Step 1 — gates:** re-run the full external-reference sweep for all four files (post T1/T2 it must be: zero production references outside the deleted set; paste). `chat_events.py`'s deferred-CWE-import dangling and `chat_events_sidebar` both close here.
- [ ] **Step 2 — worker_events gutting:** delete `StreamingChunk`/`StreamingChunkWithLogits`/`StreamDone` (:44-60) + `handle_api_call_worker_state_changed` (:88-1081) + their now-unused imports (pyflakes drives); KEEP `chat_wrapper_function` (:1082+) byte-intact + whatever imports it needs. Post-edit: read the whole remaining file; `app.chat_wrapper` → `chat_wrapper_function` still works (import smoke + the media-analysis test if one exists — locate by grep `MediaAnalysisRequestEvent` in Tests/).
- [ ] **Step 3 — test casualties (by grep, brief list = expected majority):** DELETE `Tests/Event_Handlers/Chat_Events/test_chat_events.py`, `test_chat_events_sidebar.py`, `Tests/Chat/test_chat_sidebar_media_search.py`, `Tests/Event_Handlers/test_worker_answer_citations.py`; UPDATE `Tests/UI/test_ux_audit_smoke.py` (drop the chat_events import :14-16, keep the LIVE MediaWindow half), `Tests/Character_Chat/test_dead_attach_removed.py` (:27-29 imports chat_events — invert/remove that assertion), `Tests/UI/test_legacy_entrypoints_retired.py` (:223-230 imports chat_events for the CHAT_BUTTON_HANDLERS negative pins — REWRITE those pins as module-absence pins; :74-80 deferral note deleted), `Tests/Chat/test_answer_citations.py` + `test_chat_first_handoffs.py` handoff-helper imports per T1's outcome. PRUNE `event_handler_mocks.py` entries whose last consumers died — but the fixture module itself SURVIVES (test_llm_management_events).
- [ ] **Step 4 — verify:** pyflakes all touched; import smoke; collect-only on every affected Tests/ root; suites: `Tests/Event_Handlers/ Tests/test_smoke.py Tests/LLM_Management/test_llm_management_events.py`, `Tests/UI/test_chat_first_handoffs.py Tests/UI/test_ux_audit_smoke.py Tests/UI/test_media_handoffs.py Tests/UI/test_media_window_v2_parity.py`, `Tests/Chat/`. Green.
- [ ] **Step 5 — commit:** `refactor(chat): task-577 PR2 T3 — retire chat_events + streaming/worker chat family; worker_events reduced to the live media-analysis core`

### Task 4: guards + doc hygiene + task closure

**Files:** `Tests/UI/test_legacy_entrypoints_retired.py`, `tldw_chatbook/Chat/Chat-Uploads-Documentation.md` (+ archival docs sweep), `backlog/tasks/task-577 - ....md`, `backlog/decisions/026-...md` (addendum), CSS if any dead selectors surfaced (via source+`build_css.sh` only).

- [ ] **Step 1 — retirement guards:** add the four retired modules to RETIRED_MODULES/RETIRED_FILES; add `test_task_577_pr2_pipeline_retired` pinning: modules unimportable; `worker_events` no longer defines `StreamingChunk`/`StreamDone`/`handle_api_call_worker_state_changed` but STILL defines `chat_wrapper_function`; `app` (module-level check via importlib or source grep, not instantiation) no longer defines `_build_handler_map`. Match verified reality incl. any DEFERRED items.
- [ ] **Step 2 — doc hygiene (PR1-review carries):** fix or retire `tldw_chatbook/Chat/Chat-Uploads-Documentation.md` (presents CWE as the live attachment UI — attachments are live Console features: either rewrite its UI-layer section to name the Console surface or delete the doc if wholly stale — read it first, decide, report); sweep `Chat/TABBED_CHATS_LESSONS_LEARNED.md` + `Docs/Development/` archival mentions — add a one-line "(retired in task-577)" header note where the doc is archival, no rewrites.
- [ ] **Step 3 — task-577 closure:** AC #1 [x] (all units deleted-or-recorded with gate evidence — enumerate the deferred/kept list: chat_wrapper core kept-live, AIGenerationHandler structural, TTS query fallbacks, CCP stratum untouched), AC #2 [x] (guard pins), AC #3 [x] (suites green, boot clean, Console + handoffs unaffected); `## PR2 progress` section; `status: Done`. ADR-026: one-line addendum recording the pipeline-retirement completion + the chat_wrapper media-analysis survivor.
- [ ] **Step 4 — full verification:** the T3 suite set + `Tests/UI/test_legacy_entrypoints_retired.py` alone + `Tests/UI/test_css_bundle_sync_guard.py Tests/UI/test_css_build_integrity.py` (if CSS touched) + collect-only Tests/ (expect only the 3 known pre-existing errors).
- [ ] **Step 5 — commit:** `refactor(chat): task-577 PR2 T4 — retirement guards, doc hygiene, task-577 closure`

---

### Controller-level

- Final whole-branch review (sonnet): cumulative gates; Console + media-analysis + TTS + token-counter live-path integrity; spec-vs-reality incl. both PR1 danglings CLOSED; docs coherence.
- PR2 to `dev` (body: gate resolutions A/B, the chat_wrapper survivor, closure of both declared danglings), Qodo adjudication, STOP for user merge-go. task-577 Done only after merge.
