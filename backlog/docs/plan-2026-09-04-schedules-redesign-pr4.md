# Schedules Redesign — PR-4 (FINAL): Responsive floor, keyboard map, tab retirement

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The redesign completes: one surface (the unified Queue), the spec-§12 keyboard map, an 80×24-safe responsive floor, and the Automations/Conflicts/Results tabs retired — with every unique capability they held relocated (audit trail, results detail + actions, conflicts view, run-now/edit for definitions) and the accumulated cleanup riders resolved.

**Architecture:** One new `WorkbenchHostScreen` pattern (a minimal pushed `Screen` hosting a given workbench widget class as a FRESH instance, Esc-to-close — the spec's no-reparenting rule; the repo has no precedent, this invents it) serves all four push needs: the narrow-width detail push, the results view, the conflicts view, and the audit view. The unified list drops its family filter (all server families render read-only rows with the existing unsupported-family fallback — retiring the Automations tab must not make agent_task invisible); definition rows gain run-now/edit routing; the Results tab's rendering+actions relocate into the pushed results view reachable from the rail and the panes' now-live results rows. The §12 keyboard map lands with its collisions resolved (legacy fixed-direction transfer keys and their duplicated flow DELETE — the Runs-on dropdown is the transfer surface); Up/Down detail-row traversal is new `DetailValueRow`/pane work. The retirement then deletes the three TabPanes, tab-gated code, legacy transfer buttons, and the dead branches.

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite, pytest.

**Spec:** `backlog/docs/spec-2026-09-02-schedules-screen-redesign.md` §11-phase-4/§12/§13 (tracked). Planning rulings (binding):
1. **Family filter drops** (the ledgered revisit): all families list in the unified Queue; non-`recurring_question` rows are read-only (the `_UNSUPPORTED_FAMILY_NOTE` fallback `DefinitionDetail` already has) — visibility is the honest v1, authoring/editing stays recurring_question-only. Spec §13 scopes out only the Create entry.
2. **Keyboard per spec §12 exactly**: `1-4`/`f` chips, `/` search, `n` create, `p` pause/resume, `m` move owner (the dropdown opener), `r` mark read, `Esc` back/close, `Up/Down` detail-row traversal. Legacy `m/M/y/k` fixed-direction transfer keys DELETE with their duplicated flow (`_begin_automation_transfer`/`_cancel_automation_transfer`, `_run_owner_*`'s legacy twins) and `TaskDetail`'s legacy transfer buttons — the Runs-on row is the one transfer surface. Run-now is NOT a global key: it is a detail-pane button (DefinitionDetail gains one beside pause/resume; TaskDetail already has one).
3. **Push pattern**: `WorkbenchHostScreen(widget_factory, title)` — fresh instance per push, Esc pops, no state reparenting; data flows through the same service seams the tab versions used. The conflicts badge repoints to it (the tab-flip breaks with the tab bar).
4. **Results relocation semantics preserved**: the 200-row sync-window listing + honest cap line, read/dismiss/mark-solved (+`r` per the map), detail rendering, Mark-all — all in the pushed results view; `DefinitionDetail`'s dangling "See Results tab" string and dead unread row become a live "view results" activation (affordance on) pushing the view filtered to that definition.
5. **Stale rider coordinates**: the PR-2-era isinstance-branch line numbers are stale — re-locate by PATTERN (single-primitive guards made dead by `include_projections=False` and the unified loader), verify each is genuinely unreachable before deleting.
6. **Responsive floor**: <84 cols the detail pane no longer blank-hides — Enter on a row pushes the hosted detail; chips collapse to the cycling control at ~80; every operation reachable at 80×24 via push or modal (pinned at 80×24 and ~110). The `_definitions_stale` TabActivated consumer re-homes (no tabs) onto the push/pop lifecycle.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-redesign-pr4`, branch `feat/schedules-single-surface` off current `origin/dev`. Never the main checkout; NEVER `git stash` (baselines: `git show <ref>:<path>` / throwaway detached worktree); no pkill beyond own PIDs; `git --no-pager`; FOREGROUND pytest only; tmp_path DBs.
- NO schema migration; NO service-seam changes except deletions of now-unconsumed legacy paths (verify zero consumers before deleting).
- Survey with exact seams: `redesign-pr4-survey.md` in the SDD workspace.
- Diagnostics pin (SCRIPT, `--write`+JSON) on logger changes; census COUNT parity vs dev CI; bare-type CSS ratchet (class-target, never ancestor-scoped bare types — bit us in PR-3); CSS via build flow source+bundle; `CSS_PATH = BUNDLED_STYLESHEET` for geometry tests; painted assertions; escape/Text discipline.
- Retirement = preservation flips: tests pinning the RETIRED surfaces update deliberately (each change cited with the retirement rationale); tests pinning KEPT behavior stay untouched.
- UI change ⇒ `Docs/User_Guide/` schedules page — this one is a REWRITE (single surface).
- Commit trailer on every commit:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WocisXw6SEEG6nb1aKFHtv
```

---

### Task 1: `WorkbenchHostScreen` + conflicts repoint

**Files:** Create `tldw_chatbook/UI/Screens/scheduling/workbench_host_screen.py`. Modify `schedules_workbench.py` (the conflicts badge pushes the hosted `ConflictsTab` instead of flipping tabs — the tab itself stays until Task 5). Tests: new `Tests/UI/test_workbench_host_screen.py` + the badge test updates.

**Interfaces (produced; Tasks 2/3/6 consume):** `WorkbenchHostScreen(widget_factory: Callable[[], Widget], *, title: str)` — pushes with the factory-built fresh instance, styles via the existing features CSS (class-targeted — the ratchet), `Esc` pops, footer hint registered; a `dismissed` callback slot for pop-time refresh (the staleness re-home consumer).

- [ ] TDD: push/pop round-trip (fresh instance each push — two pushes yield distinct widgets); Esc pops; badge push renders the conflicts content painted; the pane behind survives the round-trip untouched. FAIL → implement → PASS → commit `feat(ui): WorkbenchHostScreen push pattern + conflicts repoint`.

### Task 2: Results relocation

**Files:** Modify `results_tab.py` (the widget must compose cleanly inside the host screen — factor its tab-coupling out if any; add an optional `definition_id` filter mode preserving the cap-line honesty), `schedules_workbench.py` (rail: a "Results (N)" affordance beside Mark-all-read pushing the hosted view), `definition_detail.py` (+`task_detail.py` if reminders reference results — check): the unread row gains `affordance=True` and the dangling "See Results tab" string becomes the live "view results" activation pushing the view filtered to the definition. Tests: results-tab file + pane files.

- [ ] TDD: pushed view lists/read/dismiss/mark-solved/mark-all work hosted (the existing results tests' behaviors re-pinned in the pushed context); definition-filtered push shows only that definition's results with the honest cap line; the unread row activation; badge/unread refresh on pop (the dismissed callback). FAIL → implement → PASS → commit `feat(scheduling): results view relocated to the pushed surface`.

### Task 3: Definition-row actions + all-families listing + audit relocation

**Files:** Modify `unified_rows.py` (drop the family filter per ruling 1 — bucket/glyph handling for unknown families defers to honest defaults; search_blob etc. still safe), `schedules_workbench.py` (Queue definition rows: run-now routed to the existing local/server run-now paths by owner, `e` edit-in-full opens the modal for recurring_question rows — both routed by row kind; non-recurring_question rows: read-only + honest refusals), `definition_detail.py` (a Run-now button beside pause/resume — the tab's `r` relocates here as a button per ruling 2; the History "view runs" pointer becomes a live activation pushing a hosted audit view — a small `definition_audit_view.py` widget reusing `list_automation_definition_audit` + the tab's rendering). Tests: unified-list + pane + workbench files.

- [ ] TDD: agent_task rows visible/read-only with the honest note (bucket sanity for unknown families pinned); run-now both owners from the Queue; edit-in-full routing; audit view pushed with painted events; family-gated refusals. FAIL → implement → PASS → commit `feat(scheduling): definition actions on the queue + all-families listing + audit view`.

### Task 4: Keyboard map + detail-row traversal

**Files:** Modify `schedules_workbench.py` (BINDINGS + `SCHEDULES_SHORTCUTS` to the §12 map: `1-4`/`f` chips, `/` focus-search, `n` create chooser, `p` pause/resume for the selected row (reminders toggle enable, definitions lifecycle — routed by kind, honest refusal where locked), `m` opens the Runs-on dropdown on the selected row's detail, `r` mark-read (selected definition row's unread → read; refusal copy when nothing unread), `Esc` existing semantics); `detail_value_row.py`/pane files (Up/Down traversal across focusable rows within the pane — new key handling; Textual's default chain is Tab-based, arrows are new work; keep the editor-open input-ownership rule from PR-3). LEGACY DELETIONS: the `m/M/y/k` transfer keys + `_begin_automation_transfer`/`_cancel_automation_transfer` + `TaskDetail`'s legacy transfer buttons + the `_run_owner_*` duplication's legacy side (verify zero remaining consumers each — ruling 2). Tests across the affected files; every deleted-surface test updated WITH the retirement citation.

- [ ] TDD: each §12 key end-to-end (painted/effect assertions); traversal round-trip incl. skip-non-focusable and editor-open ownership; deleted keys genuinely gone (no-op or rebound); footer tuple matches. FAIL → implement → PASS → commit `feat(scheduling): spec keyboard map + row traversal, legacy transfer surface retired`.

### Task 5: The retirement + dead-code sweep

**Files:** Modify `schedules_workbench.py` (delete the Automations/Conflicts/Results TabPanes + ALL tab-gated code — the TabbedContent likely reduces to the bare Queue content, remove it if so; `_definitions_stale`'s TabActivated consumer re-homes onto push/pop dismissed callbacks + the existing refresh seams; the second `DefinitionDetail` instance and `_refresh_*_tab` paths for retired tabs go), `task_detail.py` (dead managed-by branch :1514-1519 region — verify unreachable then delete), the stale-rider sweep per ruling 5 (re-locate by pattern; delete only what's provably dead), the per-tick reminder tick-skip guard (the asymmetry: mirror the definition branch's guard), `results_tab.py`/`conflicts_tab.py` renames if their "tab" identity is now wrong (judgment: keep module names, update docstrings — smallest diff). Tests: the big deliberate update — every retired-surface test removed/rewritten WITH citations; kept-behavior tests untouched.

- [ ] TDD: the workbench composes single-surface (painted); all relocated capabilities still reachable (smoke each: conflicts push, results push, audit push, run-now, edit); staleness re-home pinned (an Automations-era mutation path → the Queue refreshes via the new consumer); no orphaned imports/dead services (grep sweep recorded). FAIL → implement → PASS → commit `feat(scheduling): single-surface workbench — tabs retired`.

### Task 6: Responsive floor + docs rewrite + gates

**Files:** Modify `schedules_workbench.py` + CSS (per ruling 6: <84 the detail region hides and Enter pushes the hosted detail — the SAME widget classes fresh-instanced with the row's data; chips collapse to the cycling control at ~80; the rail/status strip degrade readably), `Docs/User_Guide/` schedules page (REWRITE for the single surface: the list, chips, keys, editing, transfers, results/conflicts/audit pushes, responsive behavior). Gates.

- [ ] TDD: 80×24 full-operation smoke (create/edit/pause/transfer/results each reachable via push or modal — painted at that size with `CSS_PATH = BUNDLED_STYLESHEET`); ~110 and full-width layouts pinned; chip collapse; pushed-detail data correctness. Then FULL gates: `Tests/Scheduling/ -q`; the assembled UI set (accounting for the deliberate retirement updates); census COUNT parity vs dev CI; the bare-type + boot-css ratchets checked with attribution; pin script; ruff; bundle byte-identical. Commit `feat(scheduling): responsive floor + single-surface docs`.

---

## After the tasks
Final whole-branch review (opus; relocation-completeness + retirement-cleanliness + 80×24 lenses) → one fix wave → PR `feat(scheduling): single-surface schedules workbench (redesign PR-4, final)` → paged bot read → adjudicate → the full-cycle merge pipeline (in-loop merge; peer window if >2 laps). Post-merge: PROGRAM CLOSE-OUT — memory, lessons file (the accumulated cross-program lessons), backlog follow-up tasks (backfill normalizer, post-transfer history identity, keyring isolation, auth_token precedence, MRO double-unmount, C1-mirror... verify which remain), TASK-18940 status, and the redesign spec marked Delivered.
