# RAG Truth PR-T3 — MCP Tool Honesty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The MCP surface stops lying to the agents that consume it. A `search_rag` result that found nothing useful says so instead of reporting `OK · 10 results`; a run that produced a result always shows it or says why it didn't; a refusal reads as a refusal; the audit trail is populated when you look at it and records what was actually called; and no execution path bypasses the permission gate and the log.

**Architecture:** Five findings, five seams — but three of them (F1 honesty copy, F2 silent drops, F4 illegible refusals) live in **one method pair** (`_summarize_tool_result` + `show_tool_result`), so they ship together or the good copy gets silently deleted by a mode switch. The honesty vocabulary is not invented: `library_rag_score_suffix` and `library_rag_all_matches_weak` are pure functions taking plain floats and are called directly; the coverage-note and quiet-no-match *shapes* are mirrored (their Library types and diagnostics don't exist on this path). One correctness fix precedes all of it: keyword retrieval fabricates `score: 1.0`, so bands must not be layered until that reads `None` like the Library's does.

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest.

**Branch/worktree:** `feat/rag-truth-mcp-honesty` in `.worktrees/rag-truth-pr3`, base `e4f7aa24e`. Full seam evidence: `scout-map.md` in the SDD workspace — every line ref verified at that commit; **re-verify before editing, never edit blind.** PR-T1 had three refuted premises and PR-T2 had two; expect the same rigor.

## Global Constraints

- **Never claim a number or a quality you cannot source.** A fabricated score is worse than no score — the Library deliberately shows *no* band for keyword rows (`library_rag_state.py:604-611`) because FTS relevance was judged misleading. Mirror that judgment; do not invent a band to fill a slot.
- **Reuse the vocabulary, don't reinvent it.** `library_rag_score_suffix` (`:600-629`) and `library_rag_all_matches_weak` (`:1505-1525`) are pure and directly callable. `library_rag_coverage_note` (`:1537-1612`) and `library_rag_empty_state_quiet_copy` (`:930-980`) take Library types — **mirror their shape and register, do not fake their inputs.**
- **Band vocabulary is gated on result shape.** `_summarize_tool_result` is generic across every MCP tool; `list_characters` must never grow a "match: strong". Only rows carrying a numeric `score` get banding.
- **Escaping is terminal; `markup=False`** on every Static rendering tool output (the Hub convention — tool output is untrusted; the `builtin:` branch executes in-process code).
- **Quiet register**, verb-first. Exemplars: `_ORIGIN_SENTENCES`/`_EMPTY_STATE_COPY` (`mcp_inspector.py:130-163`), `LIBRARY_RAG_*_COPY`.
- **Targeted tests only:** each task's gate = the files it touched + `pytest Tests/ --collect-only -q | tail -3`. **The machine is heavily contended — ONE test file per foreground Bash command with a generous timeout; never a full suite; never end a turn waiting on a background command.**
- venv-only pytest; pytest under `Tests/` is the ONLY python entry point (a bare `python3 -c` importing app modules writes the LIVE config). `git stash` FORBIDDEN. **Push after every task** (durability begins at origin — a cleaner destroyed a reviewed wave on 2026-08-06).
- **Protected oracles:** the stale-drop guard tests (`Tests/UI/test_mcp_inspector.py:1991-2018` and its same-name-different-server sibling) pin *silence* deliberately — a fallback is additive; **removing the guard is not authorized.** Exact-string result pins (`:1602, 1631, 1652, 1678, 1703, 1725`) change only where a task names the change.
- **Backlog IDs:** next free is **2531** (135-worktree scan); re-scan at ship.

## File Structure

| File | Responsibility in this PR |
|---|---|
| `tldw_chatbook/MCP/tools.py` | T1: stop fabricating a keyword score |
| `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` | T2: bands/all-weak/no-match in `_summarize_tool_result`; T3: drop fallbacks + refusal legibility; T6: Route B |
| `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` | T3: render-failure toast, in-flight Run re-enable; T4: thread argument names; T5: audit resync |
| `tldw_chatbook/MCP/execution_log.py` + `unified_control_plane_service.py` | T4: argument provenance |
| `tldw_chatbook/UI/MCP_Modules/mcp_schema_form.py` | T6b: the unreadable boolean checkbox (task-2272 item 1) |
| `tldw_chatbook/Chat/console_provider_gateway.py` + `Chat_Deps.py` | T7: carry the real status |
| `backlog/tasks/`, `Docs/User_Guide/` | T8 |

---

### Task 1: Stop fabricating a keyword match score

**Files:** Modify `tldw_chatbook/RAG_Search/simplified/search_service.py:167` **or** `tldw_chatbook/MCP/tools.py:318` (decide — see below). Test: `Tests/MCP/test_rag_search_tool.py` (additive).

**Interfaces:**
- Produces: a keyword-mode row carries `score: None` (not `1.0`), so no downstream surface can band it as a strong match. Semantic rows keep their real float.
- **Decide and justify:** fixing at `search_service.py:167` changes the value for *every* consumer of that service (honest, wider blast radius); fixing at `MCP/tools.py:318` scopes it to the MCP payload (narrower, leaves the fabrication in place for others). **Read both call sites and every consumer of `SimplifiedRAGSearchService` before choosing; say which and why in your report.** The Library's own precedent is to null it at the service boundary (`library_rag_state.py:604-611` documents *why*: FTS relevance is misleading, so no band beats a wrong band).

- [ ] **Step 1: Write the failing test** — a keyword-mode `perform_rag_search` result carries `score is None` for every row; a semantic-mode result keeps real floats. Use the real fixtures in `Tests/MCP/test_rag_search_tool.py` (`:67` semantic, `:87` keyword), which currently do not assert score values.
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement** at the chosen site, with a comment naming the reason (mirroring the Library's).
- [ ] **Step 4: Gate.** `pytest Tests/MCP/test_rag_search_tool.py -q`, plus whatever suite covers the other consumers if you fixed at the service, then the collect sweep.
- [ ] **Step 5: Commit** — `fix(rag): keyword rows carry no match score instead of a fabricated 1.0`

---

### Task 2: The MCP result says what it found (F1)

**Files:** Modify `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:356-386` (`_summarize_tool_result`) and its note wiring in `show_tool_result` (`:2083-2098`). Test: `Tests/UI/test_mcp_inspector.py` (additive; the exact-string pins at `:1602,1631,1652,1678,1703,1725` change only if a named case is affected — name each).

**Interfaces:**
- Consumes: `library_rag_score_suffix` and `library_rag_all_matches_weak` from `Library/library_rag_state.py` — **called directly, not copied.** (If importing Library into the MCP module creates a layering problem, say so and propose the smaller move — PR-T2 shipped a real circular-import regression from exactly this kind of cross-package import, so **check the direction before you write it**.)
- Produces, for a list result whose rows carry a numeric `score`:
  - the summary line keeps `OK · source · duration · N results`;
  - the quiet interpretation line gains the all-weak notice when `library_rag_all_matches_weak(rows)` — mirroring `LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX`'s register ("No strong semantic matches — results below are weak.");
  - the raw Collapsible's rows are unchanged (it is the evidence, not the summary).
- **Rows without a numeric score get no band and no notice** — `list_characters` must be byte-identical to today. Pin that.
- Empty list keeps its existing quiet line; the error shape keeps its existing handling.

- [ ] **Step 1: Write failing tests** — (a) ten rows all scoring < 0.2 → the all-weak notice appears; (b) rows with a mix incl. ≥ 0.5 → no all-weak notice; (c) a result whose rows have no `score` key → summary and note byte-identical to today; (d) an empty list → today's quiet line unchanged.
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement.** Keep `markup=False` on the note Static. Do not touch the raw body.
- [ ] **Step 4: Gate.** `pytest Tests/UI/test_mcp_inspector.py -q`, then the collect sweep.
- [ ] **Step 5: Commit** — `feat(mcp): tool results say when every match is weak`

---

### Task 3: A run that ran always says something (F2 render half + F4)

**Files:** Modify `mcp_inspector.py` (`_handle_test_run` `:1966-1982`; `show_tool_result` drops `:2050-2062`) and `mcp_workbench.py` (`_show_tool_test_result` `:3427-3438`; the in-flight branch `:3277-3288`). Test: `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_workbench.py` (additive).

**Interfaces:**
- Produces: **no path from "Run pressed" to "nothing happened".** Specifically:
  - the stale-drop guard (`:2050-2058`) keeps dropping the *render* (its tests pin that silence — do not weaken) but now also **surfaces a toast** naming the tool whose result arrived late, so the user learns the run completed;
  - the `NoMatches` drop (`:2059-2062`) does the same;
  - `_show_tool_test_result`'s `except Exception` (`:3434-3438`) toasts instead of only logging;
  - `_handle_test_run`'s two silent returns (`:1968-1975`) toast;
  - the in-flight-duplicate branch (`:3277-3288`) **re-enables the Run button** it left disabled.
- **F4 (refusal legibility):** a refusal must never read as `Failed`. `PermissionError` from `local_control_service.execute_tool` (`:571-589`) and the `ValueError("Server-source tools are display-only.")` at `unified_control_plane_service.py:2196` currently surface via the generic exception path as `Failed · Nms`. Classify them as blocked (existing `Blocked · not run` vocabulary + the "Change in Permissions" jump where applicable). Also fix `_handle_test_run:1976-1979`, the one result write with no status prefix at all.
- **Also (task-2270's rider, in this seam):** `_decision_note` says "This tool is set to Off." for a synthesized `origin == "gate_error"` gate (`mcp_workbench.py:3178-3188`) — but the tool is *not* set to Off; the resolver failed. Use the honest sentence (`_UNKNOWN_ORIGIN_SENTENCE`, "Permission state could not be resolved."). **This is a deliberate contract change to `test_decision_note_unknown_origin_degrades_to_bare_sentence` — name it in your report.**

- [ ] **Step 1: Write failing tests** for each silent path (stale drop → toast fired; NoMatches → toast; render exception → toast; in-flight duplicate → Run re-enabled), for the two misclassified refusals rendering as blocked-not-failed, and for the `gate_error` sentence.
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement.** The protected stale-drop tests must still pass unmodified (they assert the result Static stays empty — a toast is a different surface).
- [ ] **Step 4: Gate.** Each touched test file in its own foreground command, then the collect sweep.
- [ ] **Step 5: Commit** — `fix(mcp): a completed run is never silent, and a refusal never reads as a failure`

---

### Task 4: The audit trail records what was actually called (F2 log half)

**Files:** Modify `tldw_chatbook/MCP/execution_log.py` callers — `unified_control_plane_service.py:2274-2318` (`test_hub_tool`) and `Agents/mcp_tool_provider.py:713-719` — to supply `registered_argument_names`; thread it from `HubTool.input_schema` (`MCP/hub_tool_catalog.py:44`). Test: `Tests/MCP/test_execution_log.py` (additive), `Tests/MCP/test_control_plane_tool_execute.py`.

**Interfaces:**
- Context: `build_record`'s `registered_argument_names` parameter (`execution_log.py:89`) exists, is threaded through `execute_hub_tool`, and **no caller in the tree ever supplies it** — so every row records `argument_names: []` and `unknown_argument_count == len(arguments)`. The schema is available at both call sites (it is what the Test Tool form is built from).
- Produces: a `search_rag` run logs `argument_names` containing the supplied-and-registered names (e.g. `query`, `limit`, `use_semantic`) with `unknown_argument_count == 0`. **Names only — never values** (that is the existing privacy contract; `Tests/MCP/test_execution_log.py:75-93` pins the shape).
- The agent bridge gets the same fix for free — say in your report whether its schema source is the same one.

- [ ] **Step 1: Write failing tests** at the production default (today's tests only cover the case where the parameter IS supplied — the None-default production path is uncovered).
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Gate + collect sweep.**
- [ ] **Step 5: Commit** — `fix(mcp): execution log records the argument names it was built to record`

---

### Task 5: Audit mode is populated when you look at it (F3; closes task-2272 item 2)

**Files:** Modify `mcp_workbench.py` — call `_sync_audit_mode` (`:1263-1303`) from `_run_tool_test`'s `finally` (`:3424-3425`). Test: `Tests/UI/test_mcp_workbench.py` (additive; `Tests/UI/test_mcp_audit_mode.py` has no refresh-on-run coverage — that gap is the bug).

**Interfaces:** a completed tool run repopulates the audit table without pressing `r`. `_sync_audit_mode` is exception-guarded already. If refreshing Findings in the same call is unwanted per-run, split the read+`update_entries` pair (`:1293-1298`) — say which you did.

- [ ] **Step 1: Write the failing test** — run a tool, assert the audit entries include the new row with no manual refresh.
- [ ] **Step 2: Verify failure.** **Step 3: Implement.** **Step 4: Gate + sweep.**
- [ ] **Step 5: Commit** — `fix(mcp): a completed run lands in the audit trail immediately`

---

### Task 6: The ungated execution hatch, and the unreadable boolean

**Files:** Modify `mcp_inspector.py:2401-2419` (`_run_advanced_action`); `mcp_schema_form.py` (the boolean field). Test: `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_schema_form.py`.

**Interfaces — Route B (the scout's most serious find):** the Advanced panel's `tool.execute` action (`unified_control_plane_service.py:261-283`) reaches `local_control_service.execute_tool` **without the permission gate and without any execution-log record** (`_record_tool_execution` lives only inside `execute_hub_tool`). An Off tool runs; nothing is logged.

**Ruling (do not relitigate):** route it through `execute_hub_tool` so it is gated and logged like every other run. If that proves structurally impossible in this seam, the fallback is to **remove the descriptor** — an undocumented, ungated, unlogged execution path is not an acceptable thing to ship knowingly. Report which you did and why; if you route it, pin that an Off tool is refused there.

**Also (task-2272 item 1):** `use_semantic` renders as an empty box — no glyph, no label, unreadable and un-clickable (only Tab+Space works). A boolean the user cannot read is a lie about what was run. Fix the Checkbox composition/styling; live-verifiable in Task 9's check. Any new class must be styled in the CSS **source** with the bundle regenerated (`python3 -m tldw_chatbook.css.build_css`, never hand-edited) — `Tests/UI/test_css_class_coverage_contract.py` will fail the branch otherwise.

- [ ] **Step 1: Write failing tests** — an Off tool invoked via the Advanced action is refused (and logged if routed); the boolean field renders a readable, clickable labeled checkbox.
- [ ] **Step 2: Verify failure.** **Step 3: Implement.** **Step 4: Gate** (incl. the CSS contract) **+ sweep.**
- [ ] **Step 5: Commit** — `fix(mcp): the advanced execute hatch is gated and logged; boolean fields are readable`

---

### Task 7: One status per error (F5)

**Files:** Modify `tldw_chatbook/Chat/console_provider_gateway.py:1639-1645` (carry the status on the re-raise) and its `_QueueItem.error` producer (`:1626-1628`). Test: `Tests/Chat/test_console_provider_failure_copy.py` (additive).

**Interfaces:**
- Context: the real status (400) is frozen into prose at `:174`, then the re-raise at `:1639-1645` omits `status_code=`, so `ChatProviderError`'s default (`Chat_Deps.py:57`, 502) is what `describe_stream_failure` reads — producing `provider returned HTTP 502 (… Status: 400.)`.
- Produces: one status, the real one. **`Tests/Chat/test_console_chat_controller.py:1606` pins `"HTTP 502"` for a genuine 502 response — that must keep passing** (it is about a real gateway error, not the wrapper default). Verify you haven't disturbed it.

- [ ] **Step 1: Write the failing test** — a provider 400 surfaces as one consistent status in the user-facing copy.
- [ ] **Step 2: Verify failure.** **Step 3: Implement** (carry the code on the queue item; pass it at the raise). **Step 4: Gate** (both named files) **+ sweep.**
- [ ] **Step 5: Commit** — `fix(chat): provider errors report the real status, not the wrapper's default`

---

### Task 8: Docs and backlog

- [ ] **Step 1:** Update `Docs/User_Guide/` MCP pages for: what a tool result now tells you (bands/all-weak), that a completed run always reports, the audit trail populating live, and the Advanced hatch's new gating. Stamp per each file's convention.
- [ ] **Step 2:** Fresh cross-worktree ID scan (2531 at plan time). Close **task-2272** (both items, if Task 6 shipped item 1) and **task-2270's rider** (note the main body remains open). File anything deliberately left — in particular a coverage-note equivalent for the MCP retrieval path (it needs a `semantic_scope_coverage` diagnostic that `SimplifiedRAGSearchService` does not produce; explicitly out of scope here).
- [ ] **Step 3:** Two commits, docs and backlog separately.

---

### Task 9: Whole-branch review, live verification, ship

- [ ] **Step 1: Whole-branch review** (strongest available model). Composition watch-items: T1's score change × T2's banding (does any surface now show *no* band where it used to show something?); T2's copy × T3's drop fallbacks (does the better copy survive a mode switch?); T3's refusal reclassification × the pinned blocked tests; T4's argument names × the privacy contract (names only, never values); T6's routing × the permission gate's existing tests. Point it at the ledger's deferred minors.
- [ ] **Step 2: ONE fix wave + one scoped re-review.** Residuals adjudicated (parked with rulings, or escalated if load-bearing).
- [ ] **Step 3: Live verification** (scratch profile; the proven recipe — copy the three DBs + chromadb before first launch, `[first_run] setup_started/completed = true`, session-suffixed tmux socket, python char-index for click columns). Scenarios: (1) a nonsense `search_rag` query reports honestly (no bare `OK · N results` implying success); (2) an all-weak result shows the weak notice; (3) a run whose panel is closed/mode-switched mid-flight still tells the user it completed; (4) an Off tool reads as blocked, not failed; (5) the audit list shows a run without pressing `r`, with real argument names; (6) the `use_semantic` checkbox is readable and clickable; (7) the Advanced execute hatch refuses an Off tool.
- [ ] **Step 4: Docs stamp** with the live-check commit.
- [ ] **Step 5: Ship** — merge latest origin/dev (regenerate the CSS bundle on conflict, never hand-merge), targeted gates, fresh ID re-scan, push, PR, merge on verified, confirm `.merged`.

---

## Self-Review Notes

- Coverage: F1 → T1+T2; F2 → T3 (render) + T4 (log) + T6 (Route B); F3 → T5; F4 → T3; F5 → T7; task-2272 → T5+T6; task-2270 rider → T3.
- Deliberately out of scope, to be filed: a coverage note on the MCP path (needs a new upstream diagnostic), task-2270's main body (inspector badge staleness in other views), task-2375 (Console handoff kinds — different surface, needs a design decision).
- Known unknowns delegated with read-first instructions: where to null the keyword score (T1), whether importing Library helpers into the MCP module is layering-safe (T2 — PR-T2 shipped a real cycle from this exact move), whether Route B can route through `execute_hub_tool` (T6).
