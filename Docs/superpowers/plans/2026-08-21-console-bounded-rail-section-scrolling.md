# Console Bounded Rail Section Scrolling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Make every direct Console Context and Inspector section readable at expanded browser-like sizes by showing at most 20 content lines, scrolling overflow locally, and keeping every section reachable at constrained terminal heights.

**Architecture:** Introduce one presentation-only bounded-section widget and pure rail-layout policies, then let `ConsoleLeftRail` and `ConsoleInspectorRail` own their respective post-refresh reconciliation. Context uses an atomic allocator with normal header-fit and short-height outer-scroll modes; Inspector adds explicit semantic ownership and rail-local navigation. Existing domain widgets remain responsible for their content/state, while the rails own allocation, focus recovery, and outer-fold truth.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/Textual pilot compositor tests, modular TCSS built by `python -m tldw_chatbook.css.build_css`.

**ADR required:** yes

**ADR path:** `backlog/decisions/077-console-bounded-rail-section-scrolling.md`

**Reason:** ADR-077 already records the long-lived nested-scroll, ownership, focus, and constrained-height interaction model; this plan implements that approved decision without introducing a new architectural choice.

---

## Scope and file map

**Create**

- `tldw_chatbook/UI/Console_Modules/rail_section_layout.py` — pure Context allocation and outer-hint predicates.
- `tldw_chatbook/Widgets/Console/console_bounded_section.py` — shared 20-line viewport and local fold hint.
- `tldw_chatbook/Widgets/Console/console_inspector_ownership.py` — Inspector grouping, strict/resilient policy, and safe diagnostics.
- `Tests/UI/test_console_rail_section_layout.py` — pure allocator and fold-predicate tests.
- `Tests/UI/test_console_bounded_section.py` — widget boundary, hint, scrolling, focus, and resize tests.
- `Tests/UI/test_console_rail_reconciliation.py` — atomic Context and Inspector reconciliation tests.
- `Tests/UI/test_console_inspector_navigation.py` — rail-local `n/p`, footer, and F1 behavior tests.

**Modify**

- `tldw_chatbook/UI/Console_Modules/left_rail.py` — wrap Context bodies, track active section, allocate atomically, and recover focus.
- `tldw_chatbook/UI/Console_Modules/right_rail.py` — specialized Inspector sections, compact siblings, outer hint, and navigation anchors.
- `tldw_chatbook/Widgets/Console/console_run_inspector.py` — expose semantically owned row/action groups and reject leftovers.
- `tldw_chatbook/Widgets/Console/console_staged_context.py` — separate header chrome from the bounded Sources body.
- `tldw_chatbook/Widgets/Console/console_changed_files_section.py` — separate header from body while retaining the 12-entry data cap.
- `tldw_chatbook/Widgets/Console/console_settings_summary.py` — remove the legacy nine-line geometry override.
- `tldw_chatbook/Widgets/Console/__init__.py` — export the shared widgets/policy types.
- `tldw_chatbook/UI/Screens/chat_screen.py` — coalesced invalidations, live-card replacement, Inspector-local keys, footer/help truth.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` — shared bounded-section, outer-hint, active/focus, and short-height styles; retire legacy caps.
- `tldw_chatbook/css/tldw_cli_modular.tcss` plus generated CSS manifests — rebuild deterministic CSS outputs.
- `Tests/UI/test_console_left_rail.py`, `test_console_right_rail.py`, `test_console_run_inspector.py`, `test_console_session_settings.py` — component contracts.
- `Tests/UI/test_console_shell_regions.py`, `test_console_resize_reflow.py`, `test_css_build_integrity.py` — production-CSS size matrix and resize/build regressions.
- `Docs/User_Guide/console.md` and `Docs/User_Guide/console/context-and-rag.md` — user-facing scrolling and navigation behavior.
- `backlog/tasks/task-19428 - Bound-Console-Context-and-Inspector-sections-with-20-line-scroll-limits.md` — plan, completed acceptance criteria, and implementation notes.

## Task 1: Lock the pure rail policies

**Files:**

- Create: `tldw_chatbook/UI/Console_Modules/rail_section_layout.py`
- Create: `Tests/UI/test_console_rail_section_layout.py`

**Step 1: Write failing policy tests**

Define immutable demand/result records and test these exact outcomes:

- local slot predicate is true only for `D > A > 0`;
- `D=20, A=20` has no local hint, while `D=21, A=20` retains one;
- counterfactual outer slot predicate is true only when `D_outer > R` (where `R` is the viewport without the hint);
- `10 -> 11 -> 10` and terminal grow/shrink remove the outer slot without sticky overflow;
- normal header-fit allocation funds the active section first, breaks ties in DOM order, water-fills unused rows up to 20, and returns `A=0` plus `no_room=True` only for unfunded open non-empty bodies;
- reaching `A=D<=20` releases that section's reserved hint cost to another section, and redistribution repeats until the complete allocation and hint-cost set is stable;
- short-height allocation gives every open non-empty body an honest base (`1` for `D=1`, otherwise `1 + hint`) and gives the active body `min(D, 20, max(1, H-3))` without changing open preferences;
- close/empty fallback chooses nearest preceding valid section, then first following, then `None`.

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/UI/test_console_rail_section_layout.py -q
```

Expected: FAIL because `rail_section_layout` does not exist.

**Step 2: Implement the smallest pure policy**

Add `ContextSectionDemand`, `ContextSectionAllocation`, `ContextAllocationResult`, `allocate_context_sections(...)`, `local_hint_required(...)`, `outer_hint_required(...)`, and `fallback_active_section(...)`. Keep Textual imports out of this module and preserve input DOM order.

**Step 3: Run the focused test**

Run the Task 1 command again. Expected: PASS.

**Step 4: Commit**

```bash
git add tldw_chatbook/UI/Console_Modules/rail_section_layout.py Tests/UI/test_console_rail_section_layout.py
git commit -m "feat(console): define bounded rail layout policy"
```

## Task 2: Build the shared bounded-section widget

**Files:**

- Create: `tldw_chatbook/Widgets/Console/console_bounded_section.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Create: `Tests/UI/test_console_bounded_section.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`

**Step 1: Write failing widget tests**

Use a minimal Textual pilot to assert:

- 0, 1, 20, and 21 rendered content-line boundaries;
- 21 lines render a 20-line content viewport plus a separate one-line `▼ more — scroll` hint;
- the hint widget stays mounted while overflow exists, becomes blank at local scroll end, and reappears after scrolling upward;
- no-overflow bodies add no focus stop;
- overflowing bodies expose one viewport focus stop and reveal a focused descendant fully;
- Up/Down, Page Up/Page Down, Home/End scroll the focused local viewport using Textual's native scroll actions;
- content shrink and viewport resize clamp `scroll_y`, remove stale hints, and call the owner recovery callback when a focused descendant disappears;
- short bodies do not consume wheel input, and wheel events at a local scroll boundary bubble to the outer scroller.

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/UI/test_console_bounded_section.py -q
```

Expected: FAIL because the widget does not exist.

**Step 2: Implement `ConsoleBoundedSection`**

Compose one `VerticalScroll` content viewport plus an always-mounted, non-focusable `Static` hint. Accept owner-supplied content and allocation, expose uncapped measured demand, and coalesce `request_reconcile()` through `call_after_refresh`. Apply the 20-line ceiling to content only; never count header chrome or the hint row. Use equality guards so a reconciliation that changes nothing does not schedule another layout pass.

Leave Textual's native Up/Down, Page Up/Page Down, Home/End actions intact and do not repost wheel events: a local viewport stops pointer scrolling only when it actually moves, allowing boundary handoff naturally. Preserve `scroll_y` across in-place content sync and same-instance hide/show, clamping only after demand or allocation shrinks.

**Step 3: Add shared TCSS**

Give the content viewport `min-height: 0`, hidden horizontal overflow, and allocation-driven height. Make hint and active/focus decoration dimensionally stable. Do not add a hard `max-height: 20` that would override a smaller owner allocation. Relocate the existing `.console-rail-section-body` bottom padding outside the measured content viewport so physical line demand is not inflated by decorative spacing.

**Step 4: Run focused tests**

Run the Task 2 command. Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_bounded_section.py tldw_chatbook/Widgets/Console/__init__.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_console_bounded_section.py
git commit -m "feat(console): add bounded rail section widget"
```

## Task 3: Integrate atomic Context allocation

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/left_rail.py`
- Create: `Tests/UI/test_console_rail_reconciliation.py`
- Modify: `Tests/UI/test_console_left_rail.py`

**Step 1: Write failing Context reconciliation tests**

Cover:

- all seven direct named Context bodies (Sessions, Workspaces, Conversations, Model, Agent, Details, Character) use `ConsoleBoundedSection` in stable DOM order while their headers remain simultaneous in normal mode;
- one post-refresh snapshot reads outer height, fixed header/chrome heights, open states, uncapped demands, and active ID before applying the complete allocation set;
- multiple same-tick body mutations produce one reconciliation and never expose mixed old/new allocations;
- active-first water filling, DOM-order ties, `· no room`, and enabled `[>]` appear exactly when allocation is zero;
- `[>]` changes transient active priority without writing persisted open preferences;
- when headers alone cannot fit, outer Context scrolling activates, no non-empty open body gets zero rows, and the active header plus first body row are revealed;
- local and outer hints follow their exact copy/predicate contracts.

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/UI/test_console_rail_reconciliation.py Tests/UI/test_console_left_rail.py -q
```

Expected: FAIL on missing bounded bodies/allocation behavior.

**Step 2: Refactor Context composition**

Keep each existing header/control row outside its bounded content body. Add stable section descriptors in the exact order Sessions, Workspaces, Conversations, Model, Agent, Details, Character, and wrap only those seven bodies without changing their domain IDs or preference keys. Keep the pinned `#console-agent-fleet-summary` chrome outside all bounded bodies; its visibility only invalidates allocation. The nested Agent fleet subsection remains ordinary Agent content, never another scroll owner.

**Step 3: Add the owner coordinator**

Implement `ConsoleLeftRail.request_allocation_reconcile()` as a coalesced post-refresh operation. Snapshot every input once, call the pure allocator, then apply all allocations, `no room` states, outer mode, and hint state atomically. Reconcile body widgets before recomputing the outer hint.

**Step 4: Run focused tests and commit**

Run the Task 3 command. Expected: PASS.

```bash
git add tldw_chatbook/UI/Console_Modules/left_rail.py Tests/UI/test_console_rail_reconciliation.py Tests/UI/test_console_left_rail.py
git commit -m "feat(console): allocate Context sections atomically"
```

## Task 4: Implement Context activation, focus order, and invalidations

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/left_rail.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_rail_reconciliation.py`
- Modify: `Tests/UI/test_console_resize_reflow.py`

**Step 1: Add failing interaction tests**

Test the complete transition table:

- initial mount has no active section;
- opening a closed section sets active before saving;
- keyboard focus or pointer press in a header, viewport, or descendant activates it, while wheel-only input does not;
- closing, emptying, or removing the active section uses the specified fallback;
- mode changes retain a still-valid active section;
- rail collapse/reopen on the same mounted screen retains active state, while unmount/remount resets it;
- local offsets survive in-place sync and same-mounted-screen collapse/reopen, clamp after content/allocation shrink, and never write rail preferences;
- Tab/Shift+Tab follows enabled header controls, overflowing viewport, enabled body descendants, then next header, in reverse for Shift+Tab;
- removal recovers next enabled body control, previous, header, then Context toggle;
- workspace, conversations, model, agent, details, character, fleet visibility, toggles, and outer resize each coalesce an allocation invalidation.

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/UI/test_console_rail_reconciliation.py Tests/UI/test_console_resize_reflow.py -q
```

Expected: FAIL on activation/focus/invalidation cases.

**Step 2: Implement activation and focus recovery**

Store transient active ID on the mounted `ConsoleLeftRail`, never in preferences. Add input/focus handlers at the rail boundary so children do not need domain-specific activation code. Underline the active header in addition to the existing focus background, and underline the rail title when the outer scroller owns focus; keep all geometry unchanged.

**Step 3: Wire named invalidations**

Replace direct geometry writes with calls to `request_allocation_reconcile()` after each existing sync/mutation seam, including pinned fleet-summary display changes. Do not manually invoke reconciliation from tests; await the production post-refresh path.

**Step 4: Run focused tests and commit**

Run the Task 4 command. Expected: PASS.

```bash
git add tldw_chatbook/UI/Console_Modules/left_rail.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_rail_reconciliation.py Tests/UI/test_console_resize_reflow.py
git commit -m "feat(console): reconcile Context focus and mutations"
```

## Task 5: Make Inspector semantic ownership exhaustive

**Files:**

- Create: `tldw_chatbook/Widgets/Console/console_inspector_ownership.py`
- Modify: `tldw_chatbook/Widgets/Console/console_run_inspector.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/UI/Console_Modules/right_rail.py`
- Modify: `Tests/UI/test_console_run_inspector.py`
- Modify: `Tests/UI/test_console_right_rail.py`

**Step 1: Write failing ownership tests**

Create a table-driven inventory that covers every currently emitted row label and action ID exactly once, plus the specialized cards and compact Project Instructions/Scope/run-status rows. Assert the approved owners: Project Instructions compact, Sources, Scope, Changed Files, run-status compact, Run, Source Readiness, Tools, Approvals, Artifacts, Selected Conversation, Session Defaults, Selected Message, Changes, Chat Dictionaries, World Books, Session Settings, and final Live Work. Specifically assert Send blocked/Recovery action under Run, RAG/source under Source Readiness, and Review Changes under Changes before dictionaries. Make the inventory fail when a new emitted `_ROW_IDS`, `_ROW_GROUPS`, `_ACTION_GROUPS`, dictionary, or World Book item lacks an explicit owner.

Also test:

- STRICT raises `UnownedInspectorContentError` for any unknown row/action;
- `TLDW_CONSOLE_STRICT_INSPECTOR_OWNERSHIP=1` selects STRICT in opt-in development/tests;
- RESILIENT is the explicit production default, renders known groups, omits unknowns, never creates Other, emits one diagnostic per structural fingerprint using only `row:<label>` / `action:<widget_id>`, and shows `Status: Inspector data incomplete`;
- the next valid state clears the status in place when known structure is unchanged; a changed known structure follows the existing recompose path.

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/UI/test_console_run_inspector.py Tests/UI/test_console_right_rail.py -q
```

Expected: FAIL because ownership policy is absent and leftovers still render.

**Step 2: Implement the injected policy**

Add `InspectorOwnershipPolicy` (`STRICT`, `RESILIENT`), `UnownedInspectorContentError`, the exhaustive stable-ID classifier, and safe fingerprint logging. Inject the policy into `ConsoleRunInspector`; do not read environment state inside the classifier itself. At the Inspector composition boundary, translate `TLDW_CONSOLE_STRICT_INSPECTOR_OWNERSHIP=1` to STRICT; otherwise have ordinary production `ConsoleInspectorRail` explicitly inject RESILIENT. Component/unit harnesses explicitly inject STRICT. Replace leftover rendering with policy handling.

**Step 3: Run focused tests and commit**

Run the Task 5 command. Expected: PASS.

```bash
git add tldw_chatbook/Widgets/Console/console_inspector_ownership.py tldw_chatbook/Widgets/Console/console_run_inspector.py tldw_chatbook/Widgets/Console/__init__.py tldw_chatbook/UI/Console_Modules/right_rail.py Tests/UI/test_console_run_inspector.py Tests/UI/test_console_right_rail.py
git commit -m "feat(console): enforce Inspector content ownership"
```

## Task 6: Bound every Inspector section and retire legacy caps

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/right_rail.py`
- Modify: `tldw_chatbook/Widgets/Console/console_staged_context.py`
- Modify: `tldw_chatbook/Widgets/Console/console_changed_files_section.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_summary.py`
- Modify: `tldw_chatbook/Widgets/Console/console_run_inspector.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `Tests/UI/test_console_right_rail.py`
- Modify: `Tests/UI/test_console_run_inspector.py`
- Modify: `Tests/UI/test_console_run_inspector_worldbooks.py`
- Modify: `Tests/UI/test_console_session_settings.py`
- Modify: `Tests/UI/test_console_rail_reconciliation.py`

**Step 1: Write failing section-boundary tests**

Assert every true Inspector section has external header chrome and one bounded body, including final Live Work. Keep Scope and `console-inspector-run-status-summary` as compact sibling rows with no section header or local viewport. Verify 20/21 behavior for Sources, Run, Session Settings, and a swapped Live Work card. Verify a one-row settings body consumes one content line. Pin `ConsoleChangedFilesSection.MAX_VISIBLE_ROWS == 12` as a data cap, not a viewport cap.

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/UI/test_console_right_rail.py Tests/UI/test_console_run_inspector.py Tests/UI/test_console_run_inspector_worldbooks.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_rail_reconciliation.py -q
```

Expected: FAIL on legacy 6/10 Sources and 9-line settings geometry and missing bounded groups.

**Step 2: Refactor specialized widgets**

Remove Sources' inline 6/10 height writes and sync-time reapplication. Remove `CONSOLE_SETTINGS_SUMMARY_MAX_HEIGHT` and the CSS nine-row minimum/maximum. Keep changed-file truncation logic unchanged. Move each specialized internal header outside its bounded body without changing user-visible labels or action IDs. Move `.console-inspector-group-heading` top margin outside the measured bounded content box so it cannot consume an invisible content row.

**Step 3: Compose semantic Inspector sections**

Have `ConsoleRunInspector` build its exhaustive owned row/action groups in approved order, each with stable heading and `ConsoleBoundedSection`, preserving row IDs, structural fingerprints, and in-place updates; Changes remains at its existing tail position before dictionaries. Keep `ConsoleInspectorRail` responsible only for specialized top-level sections, compact Scope/run-status siblings, outer reconciliation, and navigation across all descendant boundaries. Replacing the Live Work card must replace only that body's content and request local then outer reconciliation.

**Step 4: Run focused tests and commit**

Run the Task 6 command. Expected: PASS.

```bash
git add tldw_chatbook/UI/Console_Modules/right_rail.py tldw_chatbook/Widgets/Console/console_staged_context.py tldw_chatbook/Widgets/Console/console_changed_files_section.py tldw_chatbook/Widgets/Console/console_settings_summary.py tldw_chatbook/Widgets/Console/console_run_inspector.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_console_right_rail.py Tests/UI/test_console_run_inspector.py Tests/UI/test_console_run_inspector_worldbooks.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_rail_reconciliation.py
git commit -m "feat(console): bound Inspector section bodies"
```

## Task 7: Add Inspector outer-fold truth and local section navigation

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/right_rail.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Create: `Tests/UI/test_console_inspector_navigation.py`
- Modify: `Tests/UI/test_console_right_rail.py`

**Step 1: Write failing outer-hint tests**

Assert a pinned, non-focusable `▼ more sections — scroll` slot exists exactly when `D_outer > R`, displays text before the end, is blank at the end, and disappears on 11→10 shrink or viewport growth. Local mutations must reconcile their section first and then invalidate the outer hint.

**Step 2: Write failing navigation/help tests**

Verify rail-local `n/p` only when focus is inside `#console-right-rail` and not an editable input. Cover boundary, non-boundary, collapse-button, outer-body, and no-anchor cases; no wrap; header reveal; focus overflowing viewport, else first enabled control, else outer body; preserve local scroll offsets/open state. Assert footer adds `n/p Sections` only while Inspector owns focus, refreshes on enter/leave, and F1 computes groups from live focus at invocation.

Also prove Inspector local offsets survive in-place sync and same-mounted-screen collapse/reopen, clamp after content/allocation shrink, and cause no persisted preference writes.

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/UI/test_console_inspector_navigation.py Tests/UI/test_console_right_rail.py -q
```

Expected: FAIL because navigation and counterfactual outer-hint reconciliation are absent.

**Step 3: Implement rail-local actions**

Handle `n/p` at the Inspector rail boundary instead of adding screen-global bindings. Use the approved anchor rules, stop at first/last boundary, and leave editable widgets untouched. Add focus enter/leave notification to the screen's existing footer registration and evaluate the same live predicate in `action_show_workbench_help()`.

**Step 4: Implement outer reconciliation/focus recovery**

Use the pure counterfactual predicate and a mounted hint slot. Recover a disappearing Inspector target to next, previous, header, outer body, then collapse control. Add the dimension-stable active underline to local header or outer rail title.

**Step 5: Run focused tests and commit**

Run the Task 7 command. Expected: PASS.

```bash
git add tldw_chatbook/UI/Console_Modules/right_rail.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_inspector_navigation.py Tests/UI/test_console_right_rail.py
git commit -m "feat(console): navigate bounded Inspector sections"
```

## Task 8: Prove live reconciliation and production-CSS geometry

**Files:**

- Modify: `Tests/UI/test_console_shell_regions.py`
- Modify: `Tests/UI/test_console_resize_reflow.py`
- Modify: `Tests/UI/test_console_rail_reconciliation.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify generated CSS outputs under `tldw_chatbook/css/`
- Modify: `Tests/UI/test_css_build_integrity.py` only if a new selector contract needs explicit source/bundle coverage.

**Step 1: Add failing production-CSS compositor tests**

Using real shell composition and production CSS, cover:

- 235x52 and 160x45 with both rails expanded/all sections open;
- 120x30 default and explicit all-open in normal header-fit mode;
- 80x24 default hidden and explicit all-open with Context outer fallback;
- exact 20/21 local boundaries, Context constrained reprioritization, Inspector 10→11→10 outer transition, content shrink, live row/card mutation, terminal resize, recompose, scroll clamping, focus recovery, and hit-test containment;
- local hint and outer hint remain separate rows and never overlap content or rail handles.
- production CSS counts internal content padding/margins in `D` while excluding external header spacing and both hint rows.

Run only the new/changed IDs first, for example:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_shell_regions.py::test_bounded_rails_expanded_size_matrix \
  Tests/UI/test_console_resize_reflow.py::test_bounded_rails_reconcile_across_resize \
  Tests/UI/test_console_rail_reconciliation.py -q
```

Expected: FAIL until final TCSS/layout wiring is correct.

**Step 2: Finish layout CSS and rebuild generated CSS**

Adjust only rail/section selectors required by the failing evidence. Then run:

```bash
../../.venv/bin/python -B -m tldw_chatbook.css.build_css
```

Do not hand-edit `tldw_cli_modular.tcss`.

**Step 3: Run the exact compositor tests**

Run the Task 8 focused IDs again. Expected: PASS.

**Step 4: Verify CSS source/bundle integrity**

```bash
../../.venv/bin/python -B -m pytest Tests/UI/test_css_build_integrity.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_chatbook/css Tests/UI/test_console_shell_regions.py Tests/UI/test_console_resize_reflow.py Tests/UI/test_console_rail_reconciliation.py Tests/UI/test_css_build_integrity.py
git commit -m "test(console): prove bounded rail geometry"
```

## Task 9: Update the Console guide

**Files:**

- Modify: `Docs/User_Guide/console.md`
- Modify: `Docs/User_Guide/console/context-and-rag.md`

**Step 1: Document the user contract**

Explain, in user language:

- direct bodies show up to 20 content lines;
- `▼ more — scroll` means more within the current section;
- `▼ more sections — scroll` means more sections below the rail;
- `[>]` temporarily prioritizes a constrained Context section without changing saved open/closed choices;
- short terminals use outer Context scrolling so all headers/bodies remain reachable;
- Tab/Shift+Tab order and Inspector-local `n/p` navigation.

Do not expose STRICT/RESILIENT implementation terminology in the user guide.

**Step 2: Commit**

```bash
git add Docs/User_Guide/console.md Docs/User_Guide/console/context-and-rag.md
git commit -m "docs(console): explain bounded rail navigation"
```

## Task 10: Focused final verification and completion

**Files:**

- Modify: `backlog/tasks/task-19428 - Bound-Console-Context-and-Inspector-sections-with-20-line-scroll-limits.md`
- No production changes unless a focused failure proves one is needed.

**Step 1: Run the complete changed-functionality suite only**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_rail_section_layout.py \
  Tests/UI/test_console_bounded_section.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_inspector_navigation.py \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_run_inspector.py \
  Tests/UI/test_console_run_inspector_worldbooks.py \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_staged_context.py \
  Tests/UI/test_console_changed_files_section.py \
  Tests/UI/test_console_changed_files_wiring.py \
  Tests/UI/test_console_inspector_section.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_rail_sections.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_console_resize_reflow.py \
  Tests/UI/test_css_build_integrity.py -q
```

Expected: PASS. Do not run the repository-wide suite.

**Step 2: Run scoped static checks**

```bash
../../.venv/bin/python -B -m ruff check \
  tldw_chatbook/UI/Console_Modules/rail_section_layout.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  tldw_chatbook/UI/Console_Modules/right_rail.py \
  tldw_chatbook/Widgets/Console/console_bounded_section.py \
  tldw_chatbook/Widgets/Console/console_inspector_ownership.py \
  tldw_chatbook/Widgets/Console/console_run_inspector.py \
  tldw_chatbook/Widgets/Console/console_staged_context.py \
  tldw_chatbook/Widgets/Console/console_changed_files_section.py \
  tldw_chatbook/Widgets/Console/console_settings_summary.py \
  Tests/UI/test_console_rail_section_layout.py \
  Tests/UI/test_console_bounded_section.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_inspector_navigation.py
../../.venv/bin/python -B -m ruff format --check \
  tldw_chatbook/UI/Console_Modules/rail_section_layout.py \
  tldw_chatbook/Widgets/Console/console_bounded_section.py \
  tldw_chatbook/Widgets/Console/console_inspector_ownership.py \
  Tests/UI/test_console_rail_section_layout.py \
  Tests/UI/test_console_bounded_section.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_console_inspector_navigation.py
git diff --check
```

If an unchanged legacy file has a pre-existing Ruff baseline failure, prove it is unchanged from the merge base and report it; do not broaden this task into unrelated cleanup.

**Step 3: Self-review against ADR/spec**

Confirm exact hint copy, 20 content lines plus separate hint, counterfactual outer predicate, every invalidation seam, all ownership groups, non-global `n/p`, no user data in diagnostics, no preference writes from transient prioritization, and no manual test-only reconciliation.

**Step 4: Complete documentation and backlog hygiene**

Check every acceptance criterion, record ADR-077, list the core changes/trade-offs, and add the exact focused verification evidence to TASK-19428 Implementation Notes. Add a testing lesson only if implementation produces a new, evidence-backed reusable trap.

When and only when all acceptance criteria and focused gates are green:

```bash
backlog task edit 19428 -s Done
```

Commit the completed task record and any evidence-backed lesson added after Task 9.
