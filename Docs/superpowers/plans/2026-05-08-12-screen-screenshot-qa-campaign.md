# 12-Screen Screenshot QA Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and correct all 12 top-level destination screens through actual rendered screenshot approval, one screen per PR.

**Architecture:** This is a serial QA-and-correction campaign. A shared Backlog parent task tracks the campaign, each screen gets one child task and one branch/PR, and every screen must pass the same screenshot approval gate before PR creation. Existing mounted geometry tests remain regression coverage, but actual screenshots are the approval artifact.

**Tech Stack:** Python 3.12, Textual, textual-serve/textual-web, Playwright/browser automation, pytest, Backlog.md, GitHub PRs.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-05-08-12-screen-screenshot-qa-campaign-design.md`
- Visual parity reference: `Docs/superpowers/specs/2026-05-08-destination-visual-parity-correction-design.md`
- Product tracker: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Existing visual QA evidence: `Docs/superpowers/qa/product-maturity/phase-3/2026-05-08-destination-visual-parity-correction.md`

## Shared Files And Responsibilities

- `backlog/tasks/`: Create one parent campaign task and one child task per screen.
- `Docs/superpowers/qa/product-maturity/screen-qa/<screen>/notes.md`: Per-screen evidence record, rejection log, tests, and approval status.
- `Docs/superpowers/qa/product-maturity/screen-qa/<screen>/baseline-<timestamp>.png`: Baseline actual screenshot from running app.
- `Docs/superpowers/qa/product-maturity/screen-qa/<screen>/final-<timestamp>.png`: Final actual screenshot shown for approval.
- `tldw_chatbook/UI/Screens/<screen>_screen.py`: Screen-specific layout/interaction fixes when needed.
- `tldw_chatbook/Widgets/`: Shared widgets only when the target screen already uses them or a shared-shell bug blocks the screen.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`: Source TCSS for destination visual corrections when needed.
- `tldw_chatbook/css/tldw_cli_modular.tcss`: Regenerated CSS only through `tldw_chatbook/css/build_css.py`; never edit directly.
- `Tests/UI/test_destination_visual_parity_correction.py`: Existing compact/default geometry regression suite.
- `Tests/UI/test_destination_shells.py`: Destination mounting and behavior coverage.
- `Tests/UI/test_screen_navigation.py`: Top-level navigation safety.
- Screen-specific tests listed in each task below.

## Shared Commands

Use repository venv Python:

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
```

Refresh before each screen branch:

```bash
git fetch origin dev
```

Create each screen worktree from latest `origin/dev`:

```bash
git worktree add -b codex/screen-qa-<screen> .worktrees/codex-screen-qa-<screen> origin/dev
```

Run CSS build when TCSS changes:

```bash
$PY tldw_chatbook/css/build_css.py
```

Run baseline destination verification when layout changes:

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py Tests/UI/test_destination_shells.py Tests/UI/test_screen_navigation.py --tb=short
```

Run hygiene before every commit:

```bash
git diff --check
```

## Screenshot Capture Protocol

Preferred path:

```bash
textual-serve "$PY -m tldw_chatbook.app" --host 127.0.0.1 --port 8765
```

Then use browser automation to open `http://127.0.0.1:8765`, navigate to the target screen, interact with one primary workflow path, and capture a PNG.

Fallback path: run the app in a real terminal and use a system screenshot. Record the fallback reason in `notes.md`.

Do not use SVG exports, generated mockups, or code-rendered layouts for approval.

## Evidence Notes Template

Each screen task should create:

```markdown
# <Screen> Screenshot QA Notes

Date:
Branch:
Commit:
Screen:
Viewport:
Launch method:

## Baseline Screenshot

- Path:
- Defects:

## Interaction Smoke

- Goal:
- Steps:
- Result:

## Fixes

- Summary:

## Final Screenshot

- Path:
- User approval: pending

## Verification

- Commands:
- Results:

## Residual Risks

- None recorded.
```

---

### Task 1: Backlog And QA Scaffolding

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/screen-qa/README.md`
- Create: `Docs/superpowers/qa/product-maturity/screen-qa/<screen>/notes.md` for all 12 screens
- Modify: `backlog/tasks/` through Backlog.md tooling
- Test: `git diff --check`

- [ ] **Step 1: Create parent Backlog task**

Use the Backlog MCP tool or CLI to create a parent task titled:

`12-screen actual screenshot QA campaign`

Acceptance criteria:

- Baseline screenshot evidence exists for each of 12 screens.
- Final screenshot evidence exists for each of 12 screens.
- User approval is recorded for each screen.
- One PR is opened and merged per screen.
- Campaign README and per-screen notes are updated.

- [ ] **Step 2: Create 12 child Backlog tasks**

Create one child task under the campaign parent for each screen:

- `Screen QA: Console`
- `Screen QA: Home`
- `Screen QA: Library`
- `Screen QA: Artifacts`
- `Screen QA: Personas`
- `Screen QA: Watchlists`
- `Screen QA: Schedules`
- `Screen QA: Workflows`
- `Screen QA: MCP`
- `Screen QA: ACP`
- `Screen QA: Skills`
- `Screen QA: Settings`

Each child task must include acceptance criteria:

- Baseline actual screenshot captured.
- Interaction smoke path exercised.
- Final actual screenshot captured.
- User approval recorded before PR.
- Focused tests pass.
- PR merged before next screen starts unless user explicitly overrides.

- [ ] **Step 3: Create QA README**

Create `Docs/superpowers/qa/product-maturity/screen-qa/README.md`:

```markdown
# Screen Screenshot QA Campaign

This folder tracks actual rendered screenshot approval for the 12 top-level destination screens. Geometry dumps and SVG exports are regression evidence only. A screen is approved only after the user approves an actual screenshot from the running app.

| Order | Screen | Backlog Task | Branch | Baseline | Final | Approved | PR | Merged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Console | pending | `codex/screen-qa-console` | pending | pending | pending | pending | pending |
| 2 | Home | pending | `codex/screen-qa-home` | pending | pending | pending | pending | pending |
| 3 | Library | pending | `codex/screen-qa-library` | pending | pending | pending | pending | pending |
| 4 | Artifacts | pending | `codex/screen-qa-artifacts` | pending | pending | pending | pending | pending |
| 5 | Personas | pending | `codex/screen-qa-personas` | pending | pending | pending | pending | pending |
| 6 | Watchlists | pending | `codex/screen-qa-watchlists` | pending | pending | pending | pending | pending |
| 7 | Schedules | pending | `codex/screen-qa-schedules` | pending | pending | pending | pending | pending |
| 8 | Workflows | pending | `codex/screen-qa-workflows` | pending | pending | pending | pending | pending |
| 9 | MCP | pending | `codex/screen-qa-mcp` | pending | pending | pending | pending | pending |
| 10 | ACP | pending | `codex/screen-qa-acp` | pending | pending | pending | pending | pending |
| 11 | Skills | pending | `codex/screen-qa-skills` | pending | pending | pending | pending | pending |
| 12 | Settings | pending | `codex/screen-qa-settings` | pending | pending | pending | pending | pending |
```

- [ ] **Step 4: Create per-screen notes**

Create the 12 `notes.md` files from the template in:

```text
Docs/superpowers/qa/product-maturity/screen-qa/console/notes.md
Docs/superpowers/qa/product-maturity/screen-qa/home/notes.md
Docs/superpowers/qa/product-maturity/screen-qa/library/notes.md
Docs/superpowers/qa/product-maturity/screen-qa/artifacts/notes.md
Docs/superpowers/qa/product-maturity/screen-qa/personas/notes.md
Docs/superpowers/qa/product-maturity/screen-qa/watchlists/notes.md
Docs/superpowers/qa/product-maturity/screen-qa/schedules/notes.md
Docs/superpowers/qa/product-maturity/screen-qa/workflows/notes.md
Docs/superpowers/qa/product-maturity/screen-qa/mcp/notes.md
Docs/superpowers/qa/product-maturity/screen-qa/acp/notes.md
Docs/superpowers/qa/product-maturity/screen-qa/skills/notes.md
Docs/superpowers/qa/product-maturity/screen-qa/settings/notes.md
```

- [ ] **Step 5: Verify and commit scaffolding**

Run:

```bash
git diff --check
```

Expected: exits `0`.

Commit:

```bash
git add backlog Docs/superpowers/qa/product-maturity/screen-qa
git commit -m "Track 12-screen screenshot QA campaign"
```

---

### Task 2: Console Screen PR

**Files:**
- Modify as needed: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify as needed: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify as needed: `tldw_chatbook/Widgets/Console/console_control_bar.py`
- Modify as needed: `tldw_chatbook/Widgets/Console/console_run_inspector.py`
- Modify as needed: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify if TCSS changes: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/console/notes.md`
- Test: `Tests/UI/test_console_internals_decomposition.py`
- Test: `Tests/UI/test_console_live_work_handoffs.py`
- Test: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`

- [ ] **Step 1: Create branch from latest dev**

```bash
git fetch origin dev
git worktree add -b codex/screen-qa-console .worktrees/codex-screen-qa-console origin/dev
```

- [ ] **Step 2: Capture baseline screenshot**

Run the app through `textual-serve`, open Console, type a short draft, paste a long draft, and save:

```text
Docs/superpowers/qa/product-maturity/screen-qa/console/baseline-<timestamp>.png
```

Update `console/notes.md` with defects.

- [ ] **Step 3: Fix only Console defects**

Keep scope to Console readability, composer visibility/resize, transcript dominance, inspector visibility, provider/RAG blocked states, footer/status bar, and action row clipping.

- [ ] **Step 4: Run focused Console tests**

```bash
$PY -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions --tb=short
```

Expected: all selected tests pass.

- [ ] **Step 5: Capture final screenshot and request approval**

Save:

```text
Docs/superpowers/qa/product-maturity/screen-qa/console/final-<timestamp>.png
```

Do not open the PR until the user approves this screenshot.

- [ ] **Step 6: Commit and PR**

```bash
git diff --check
git add tldw_chatbook Tests Docs/superpowers/qa/product-maturity/screen-qa/console
git commit -m "Screen QA: approve Console"
gh pr create --base dev --head codex/screen-qa-console --title "Screen QA: Console" --body-file Docs/superpowers/qa/product-maturity/screen-qa/console/notes.md
```

---

### Task 3: Home Screen PR

**Files:**
- Modify as needed: `tldw_chatbook/UI/Screens/home_screen.py`
- Modify as needed: `tldw_chatbook/Home/dashboard_state.py`
- Modify as needed: `tldw_chatbook/Home/active_work_adapter.py`
- Modify as needed: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/home/notes.md`
- Test: `Tests/UI/test_home_screen.py`
- Test: `Tests/Home/test_dashboard_state.py`
- Test: `Tests/Home/test_active_work_adapter.py`

- [ ] **Step 1: Create branch from latest dev**

```bash
git fetch origin dev
git worktree add -b codex/screen-qa-home .worktrees/codex-screen-qa-home origin/dev
```

- [ ] **Step 2: Capture baseline screenshot**

Open Home, inspect attention queue, active work, selected item, next-best action, recent work, and footer. Save `baseline-<timestamp>.png`.

- [ ] **Step 3: Interaction smoke**

Exercise one Home action path: select/open an active work row if present, or verify a recoverable empty/ready state if no active work exists.

- [ ] **Step 4: Fix only Home defects**

Keep scope to dashboard readability, status/notification clarity, selected item inspector, next-best/recent visibility, active-work controls, and screenshot-visible footer.

- [ ] **Step 5: Run focused Home tests**

```bash
$PY -m pytest -q Tests/UI/test_home_screen.py Tests/Home/test_dashboard_state.py Tests/Home/test_active_work_adapter.py Tests/UI/test_unified_shell_phase2_home_adapter.py --tb=short
```

- [ ] **Step 6: Capture final screenshot, get approval, commit, PR**

Use the same final screenshot approval and PR gate as Console.

---

### Task 4: Library Screen PR

**Files:**
- Modify as needed: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify as needed: `tldw_chatbook/Library/`
- Modify as needed: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/library/notes.md`
- Test: `Tests/UI/test_product_maturity_phase3_library_contract_layout.py`
- Test: `Tests/UI/test_product_maturity_gate16_library_search_rag.py`
- Test: `Tests/UI/test_product_maturity_phase39_library_collections.py`
- Test: `Tests/UI/test_product_maturity_phase3_knowledge_entry.py`

- [ ] **Step 1: Create branch from latest dev**

```bash
git fetch origin dev
git worktree add -b codex/screen-qa-library .worktrees/codex-screen-qa-library origin/dev
```

- [ ] **Step 2: Capture baseline screenshot**

Capture Library in default mode and at least one mode switch screenshot if defects depend on mode, especially Search/RAG and Collections.

- [ ] **Step 3: Interaction smoke**

Switch modes: Sources -> Search/RAG -> Collections. If possible, type a Search/RAG query or select/create a local Collection.

- [ ] **Step 4: Fix only Library defects**

Keep scope to source browser/detail/inspector, mode strip density, Search/RAG panel visibility, Collections panel visibility, Import/Export placement, and Console staging affordance.

- [ ] **Step 5: Run focused Library tests**

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_product_maturity_phase39_library_collections.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py --tb=short
```

- [ ] **Step 6: Capture final screenshot, get approval, commit, PR**

Use the same final screenshot approval and PR gate as Console.

---

### Task 5: Artifacts Screen PR

**Files:**
- Modify as needed: `tldw_chatbook/UI/Screens/artifacts_screen.py`
- Modify as needed: `tldw_chatbook/Artifacts/`
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/artifacts/notes.md`
- Test: `Tests/UI/test_console_live_work_handoffs.py`
- Test: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Create branch from latest dev**

```bash
git fetch origin dev
git worktree add -b codex/screen-qa-artifacts .worktrees/codex-screen-qa-artifacts origin/dev
```

- [ ] **Step 2: Capture baseline screenshot**

Open Artifacts, verify list/detail/provenance/actions are visible or honestly empty.

- [ ] **Step 3: Interaction smoke**

Exercise Chatbook artifact open/resume path if data exists; otherwise verify empty state and disabled Console launch reason.

- [ ] **Step 4: Fix only Artifacts defects**

Keep scope to generated output list, Chatbook visibility, provenance, detail preview, Console reopen action, and export/recovery copy.

- [ ] **Step 5: Run focused Artifacts tests**

```bash
$PY -m pytest -q Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_keeps_console_launch_disabled_without_chatbooks Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_launches_latest_local_chatbook_in_console Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_reopens_console_saved_chatbook_with_provenance Tests/UI/test_destination_shells.py --tb=short
```

- [ ] **Step 6: Capture final screenshot, get approval, commit, PR**

Use the same final screenshot approval and PR gate as Console.

---

### Task 6: Personas Screen PR

**Files:**
- Modify as needed: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify as needed: `tldw_chatbook/Character_Chat/`
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/personas/notes.md`
- Test: `Tests/UI/test_destination_shells.py`
- Test: `Tests/UI/test_navigation_label_language.py`

- [ ] **Step 1: Create branch from latest dev**

```bash
git fetch origin dev
git worktree add -b codex/screen-qa-personas .worktrees/codex-screen-qa-personas origin/dev
```

- [ ] **Step 2: Capture baseline screenshot**

Open Personas and wait deterministically for loading, empty, summary, or service-error state.

- [ ] **Step 3: Interaction smoke**

Select persona/character summary if available, or verify service/empty recovery path.

- [ ] **Step 4: Fix only Personas defects**

Keep scope to local behavior snapshot, character/persona/profile labels, attach-to-Console affordance, loading/error state clarity, and footer visibility.

- [ ] **Step 5: Run focused Personas tests**

```bash
$PY -m pytest -q Tests/UI/test_destination_shells.py Tests/UI/test_navigation_label_language.py --tb=short
```

- [ ] **Step 6: Capture final screenshot, get approval, commit, PR**

Use the same final screenshot approval and PR gate as Console.

---

### Task 7: Watchlists Screen PR

**Files:**
- Modify as needed: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify as needed: `tldw_chatbook/Subscriptions/`
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/watchlists/notes.md`
- Test: `Tests/UI/test_console_live_work_handoffs.py`
- Test: `Tests/UI/test_command_palette_providers.py`

- [ ] **Step 1: Create branch from latest dev**

```bash
git fetch origin dev
git worktree add -b codex/screen-qa-watchlists .worktrees/codex-screen-qa-watchlists origin/dev
```

- [ ] **Step 2: Capture baseline screenshot**

Open Watchlists and verify no visible Collections confusion in nav/header/body.

- [ ] **Step 3: Interaction smoke**

Inspect active run/follow state if present; otherwise verify empty/recovery state and disabled Console follow reason.

- [ ] **Step 4: Fix only Watchlists defects**

Keep scope to monitored-source list, run status, alerts, recovery, Console follow, and removal of stale `W+C`/Collections language.

- [ ] **Step 5: Run focused Watchlists tests**

```bash
$PY -m pytest -q Tests/UI/test_console_live_work_handoffs.py::test_watchlists_destination_keeps_console_follow_disabled_without_active_run Tests/UI/test_console_live_work_handoffs.py::test_watchlists_destination_routes_latest_active_run_to_console Tests/UI/test_command_palette_providers.py --tb=short
```

- [ ] **Step 6: Capture final screenshot, get approval, commit, PR**

Use the same final screenshot approval and PR gate as Console.

---

### Task 8: Schedules Screen PR

**Files:**
- Modify as needed: `tldw_chatbook/UI/Screens/schedules_screen.py`
- Modify as needed: `tldw_chatbook/Schedules/`
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/schedules/notes.md`
- Test: `Tests/UI/test_console_live_work_handoffs.py`
- Test: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Create branch from latest dev**

```bash
git fetch origin dev
git worktree add -b codex/screen-qa-schedules .worktrees/codex-screen-qa-schedules origin/dev
```

- [ ] **Step 2: Capture baseline screenshot**

Open Schedules and verify timing/triggers/status/recovery panes are visible.

- [ ] **Step 3: Interaction smoke**

Inspect active run or digest fallback if present; otherwise verify recoverable unavailable state.

- [ ] **Step 4: Fix only Schedules defects**

Keep scope to schedule list, job timing, paused/running/failed state, digest fallback, Console launch, and recovery copy.

- [ ] **Step 5: Run focused Schedules tests**

```bash
$PY -m pytest -q Tests/UI/test_console_live_work_handoffs.py::test_schedules_destination_keeps_console_follow_disabled_without_active_run Tests/UI/test_console_live_work_handoffs.py::test_schedules_destination_routes_latest_active_run_to_console Tests/UI/test_console_live_work_handoffs.py::test_schedules_destination_routes_latest_digest_output_to_console Tests/UI/test_destination_shells.py --tb=short
```

- [ ] **Step 6: Capture final screenshot, get approval, commit, PR**

Use the same final screenshot approval and PR gate as Console.

---

### Task 9: Workflows Screen PR

**Files:**
- Modify as needed: `tldw_chatbook/UI/Screens/workflows_screen.py`
- Modify as needed: `tldw_chatbook/Workflows/`
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/workflows/notes.md`
- Test: `Tests/UI/test_console_live_work_handoffs.py`
- Test: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Create branch from latest dev**

```bash
git fetch origin dev
git worktree add -b codex/screen-qa-workflows .worktrees/codex-screen-qa-workflows origin/dev
```

- [ ] **Step 2: Capture baseline screenshot**

Open Workflows and verify procedure list, detail, inspector/actions, and unavailable states.

- [ ] **Step 3: Interaction smoke**

Select a workflow/run context if present; otherwise verify disabled Console launch reason.

- [ ] **Step 4: Fix only Workflows defects**

Keep scope to procedure library, steps/inputs/outputs, dry-run/approval labels, Console launch, and recovery copy.

- [ ] **Step 5: Run focused Workflows tests**

```bash
$PY -m pytest -q Tests/UI/test_console_live_work_handoffs.py::test_workflows_destination_keeps_console_launch_disabled_without_active_run Tests/UI/test_console_live_work_handoffs.py::test_workflows_destination_routes_latest_active_run_to_console Tests/UI/test_destination_shells.py --tb=short
```

- [ ] **Step 6: Capture final screenshot, get approval, commit, PR**

Use the same final screenshot approval and PR gate as Console.

---

### Task 10: MCP Screen PR

**Files:**
- Modify as needed: `tldw_chatbook/UI/Screens/mcp_screen.py`
- Modify as needed: `tldw_chatbook/MCP/`
- Modify as needed: `tldw_chatbook/UI/Widgets/` or `tldw_chatbook/Widgets/` MCP panel files if already used by MCP
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/mcp/notes.md`
- Test: `Tests/UI/test_unified_mcp_panel.py`
- Test: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Create branch from latest dev**

```bash
git fetch origin dev
git worktree add -b codex/screen-qa-mcp .worktrees/codex-screen-qa-mcp origin/dev
```

- [ ] **Step 2: Capture baseline screenshot**

Open MCP and verify server-first hierarchy, tools, auth, permissions, readiness, and audit/status.

- [ ] **Step 3: Interaction smoke**

Open/select an MCP server row or verify recoverable unavailable state if no servers exist.

- [ ] **Step 4: Fix only MCP defects**

Keep scope to server tree density, detail pane, tool/readiness inspector, disabled run/action reasons, and avoiding Settings absorption.

- [ ] **Step 5: Run focused MCP tests**

```bash
$PY -m pytest -q Tests/UI/test_unified_mcp_panel.py Tests/UI/test_destination_shells.py --tb=short
```

- [ ] **Step 6: Capture final screenshot, get approval, commit, PR**

Use the same final screenshot approval and PR gate as Console.

---

### Task 11: ACP Screen PR

**Files:**
- Modify as needed: `tldw_chatbook/UI/Screens/acp_screen.py`
- Modify as needed: `tldw_chatbook/ACP/`
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/acp/notes.md`
- Test: `Tests/UI/test_destination_shells.py`
- Test: `Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py`

- [ ] **Step 1: Create branch from latest dev**

```bash
git fetch origin dev
git worktree add -b codex/screen-qa-acp .worktrees/codex-screen-qa-acp origin/dev
```

- [ ] **Step 2: Capture baseline screenshot**

Open ACP and verify runtime/session/setup state is readable.

- [ ] **Step 3: Interaction smoke**

Verify ACP setup-needed state and recovery copy; do not move ACP runtime setup into Settings.

- [ ] **Step 4: Fix only ACP defects**

Keep scope to agent/session/runtimes/diffs/terminal layout, setup-needed clarity, Console launch unavailable state, and recovery actions.

- [ ] **Step 5: Run focused ACP tests**

```bash
$PY -m pytest -q Tests/UI/test_destination_shells.py Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py --tb=short
```

- [ ] **Step 6: Capture final screenshot, get approval, commit, PR**

Use the same final screenshot approval and PR gate as Console.

---

### Task 12: Skills Screen PR

**Files:**
- Modify as needed: `tldw_chatbook/UI/Screens/skills_screen.py`
- Modify as needed: `tldw_chatbook/Skills_Interop/`
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/skills/notes.md`
- Test: `Tests/UI/test_destination_shells.py`
- Test: `Tests/UI/test_unified_shell_phase234_maturity_gate.py`

- [ ] **Step 1: Create branch from latest dev**

```bash
git fetch origin dev
git worktree add -b codex/screen-qa-skills .worktrees/codex-screen-qa-skills origin/dev
```

- [ ] **Step 2: Capture baseline screenshot**

Open Skills and verify local skills list, readiness, validation/import states, and attach/stage affordances.

- [ ] **Step 3: Interaction smoke**

Select a listed local skill if available; otherwise verify empty/local-only state and recovery copy.

- [ ] **Step 4: Fix only Skills defects**

Keep scope to Agent Skills discovery, validation/readiness, attach/stage affordance, local/server distinction, and empty/error states.

- [ ] **Step 5: Run focused Skills tests**

```bash
$PY -m pytest -q Tests/UI/test_destination_shells.py Tests/UI/test_unified_shell_phase234_maturity_gate.py --tb=short
```

- [ ] **Step 6: Capture final screenshot, get approval, commit, PR**

Use the same final screenshot approval and PR gate as Console.

---

### Task 13: Settings Screen PR

**Files:**
- Modify as needed: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify as needed: `tldw_chatbook/config.py`
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/settings/notes.md`
- Test: `Tests/UI/test_destination_shells.py`
- Test: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Create branch from latest dev**

```bash
git fetch origin dev
git worktree add -b codex/screen-qa-settings .worktrees/codex-screen-qa-settings origin/dev
```

- [ ] **Step 2: Capture baseline screenshot**

Open Settings and verify category/detail/impact panes, global defaults, and boundary copy.

- [ ] **Step 3: Interaction smoke**

Change category focus or inspect Appearance/global preferences without moving MCP/ACP runtime controls into Settings.

- [ ] **Step 4: Fix only Settings defects**

Keep scope to category navigation, readable detail pane, impact/boundary explanation, appearance/global defaults, and footer/status visibility.

- [ ] **Step 5: Run focused Settings tests**

```bash
$PY -m pytest -q Tests/UI/test_destination_shells.py Tests/UI/test_screen_navigation.py --tb=short
```

- [ ] **Step 6: Capture final screenshot, get approval, commit, PR**

Use the same final screenshot approval and PR gate as Console.

---

### Task 14: Campaign Closeout

**Files:**
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/README.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: parent and child Backlog task files through Backlog.md tooling
- Test: `Tests/UI/test_destination_visual_parity_correction.py`
- Test: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Verify all screen PRs are merged**

Run:

```bash
gh pr list --state open --base dev --search "Screen QA:"
```

Expected: no remaining open Screen QA PRs unless intentionally paused.

- [ ] **Step 2: Update campaign README**

Replace all `pending` cells with final task IDs, screenshot paths, PR numbers, approval status, and merge status.

- [ ] **Step 3: Update product tracker**

Add a short note that actual screenshot approval supersedes the earlier geometry-only visual parity claim.

- [ ] **Step 4: Run final focused suite**

```bash
$PY -m pytest -q Tests/UI/test_destination_visual_parity_correction.py Tests/UI/test_destination_shells.py Tests/UI/test_screen_navigation.py --tb=short
git diff --check
```

Expected: tests pass and diff check exits `0`.

- [ ] **Step 5: Mark Backlog tasks Done**

Only mark the campaign parent and all children Done after all PRs are merged and approval evidence is recorded.

- [ ] **Step 6: Commit closeout**

```bash
git add backlog Docs/superpowers/qa/product-maturity/screen-qa Docs/superpowers/trackers/product-maturity-roadmap.md
git commit -m "Close 12-screen screenshot QA campaign"
```
