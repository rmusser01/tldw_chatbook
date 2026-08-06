# User Guide G4 (Settings) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write the Settings section of the User Guide — `settings.md` plus one
child page for RAG defaults — per the approved spec
`Docs/superpowers/specs/2026-07-25-user-guide-by-screen-design.md`, from the
completed IA survey of dev @ `fb2df0c8a` (2026-08-01).

**Architecture:** Pure-docs change on branch `claude/user-guide-g4-settings`
(worktree `/private/tmp/tldw-guide-g1`), one PR at the user merge gate.
Authoring inputs (session scratchpad): `g4_inventory_shell.md` (shell,
navigation, save models, category list, palette) and `g4_inventory_panes.md`
(per-pane controls, config keys, validation copy, honesty flags). Drafting by
subagents; live verification, captures, stamps, and commits stay with the
controller.

## Survey decisions

1. **One page plus ONE child.** The spec wants a single `settings.md`; the
   survey confirms that holds for 21 of 22 categories, because the shell is
   uniform and 11 categories are read-only "(view)" stubs that collapse into a
   single table. **RAG breaks out** to `settings/rag.md`: it is a second
   workbench (profile objects, `Set active` / `Clone…` / `Rename…` / `Delete` /
   `Backfill`, its own `a`/`c`/`b` keys, three modals including a destructive
   re-index confirm, a built-in-is-read-only trap, and a first-run panel) whose
   procedures cannot be summarized at field-group level without losing
   warnings. This is the spec's own "split later only if a pane outgrows it".
2. **Form-heavy panes are summarized at field-group level** — explicitly
   licensed by `_template.md` for Settings. Behavioral controls (buttons that
   probe, discover, reset, encrypt, backfill) are enumerated individually.
3. **The headline concept is that five save models coexist**, each named
   on-screen by the State banner badge (`Draft — save with s`,
   `Draft — save/revert below`, `Auto-saved`, `Applies immediately`,
   `Managed in editor`, `Per-item Save/Reset`, `Validate, then Save`,
   `Read-only here`). The page leads with this.

## Keyboard truth this phase must state

- `s` save / `r` revert / `t` test, plus RAG-only `a` / `c` / `b`.
- **A focused text field swallows all of them** — the app's own escape hatch is
  **Esc first**, and the footer re-labels the hints as `Esc, s` while a field
  has focus. Probe-verified: `s` in the category filter typed a literal "s".
- `/` focuses the category filter from anywhere and *re-arms* (select-all)
  rather than typing a slash when the filter already has focus.
- **Settings has no `Ctrl+<digit>`** — it is the last of the 13 destinations in
  `SHELL_DESTINATION_ORDER` and the ten hotkey digits run out before it. Reach
  it by clicking `Settings` or via `Ctrl+P`.
- `F6` does nothing on this screen (no pane-cycle target).

## Global constraints

Template order and authoring rules as G1-G3; verbatim labels; honest
limitations with backlog refs; captures at 200×50 with the cdnjs font strip;
`PYTHONPATH=/private/tmp/tldw-guide-g1` + codebase guard on by-path scripts;
**post-merge duplicate-ID check** (this programme already shipped six colliding
ids; see the renumbering commits on this branch); commit trailer.

## Honesty flags the pages must carry

- **Appearance `Animations` and `Smooth scrolling` write config that nothing
  reads.** Same for `Palette limit (themes)` (legacy window only) and
  `Web font size (px)` (browser/Textual-Web only, not the TUI).
- **`Theme` is startup-only**; `Preview` is the only in-session feedback, and
  the Theme editor never writes `[general].default_theme`.
- **Splash settings are startup-only**, and `Animation speed (x)` is saved to
  the wrong section (**task-2706**, filed this phase).
- **Privacy & Security changes nothing** — it is a read-out; credential
  mutation is declared `not available yet - password-gated flow required`.
- **Storage needs a restart**, and `Active files` can legitimately differ from
  `Database paths (configured)` when a user profile is set.
- **`Settings & Preferences: Open Config File` only prints the path.**
- RAG's Search group is persisted now but consumed when Library runs a query.

## Task list

### Task 1: `settings.md` (subagent)
Sections per `_template.md`, in the survey's recommended order: what the screen
is for → getting there (no hotkey) → layout tour (header, `Mode:` strip,
`Settings Sections` rail with its `>` / `(view)` / `*` legend, `Preference
Detail`, `Scope Inspector`, `▼ more — scroll the inspector`) → moving around →
**how saving works** (the badge table, `Save (s)` / `Revert (r)` incl. their
`— no changes` labels, the `State:` banner grammar, the keeps-this-draft
promise, the local-only note, Storage's restart caveat) → the grouped category
map → per-group sections (Core, Interface, Data & Privacy, Troubleshooting,
Expert, Domain Defaults + Image Gen, pointing at the RAG child) → common tasks
→ keyboard & commands (Esc-first rule up front) → related → quirks.
Capture: `images/settings/overview.svg`.

### Task 2: `settings/rag.md` (subagent)
Profiles card and its lifecycle verbs, the built-in read-only trap, the
preview-vs-edit decoupling, the five field collapsibles with the `⚠`
rebuild legend, index status and `Backfill`, `a`/`c`/`b`, and all three modals
(clone/rename name modal, unsaved-changes switch confirm, delete confirm) plus
the `Re-index required` warning. Capture: `../images/settings/rag.svg`.

### Task 3: captures + live verification + stamps (controller)
Scenes: settings overview (Overview category), a draft category showing the
dirty State banner, and the RAG profiles card. Execute the Common tasks live
or by pilot; file backlog tasks for any new confirmed defect (sweep first);
stamp both pages `fb2df0c8a — 2026-08-01`.

### Task 4: index wiring + link sweep
`index.md`: the Settings row → link; the legacy `Customize` row → link; check
whether the globals table needs any further qualification. Full sweep
`BROKEN: []`; template conformance per page.

### Task 5: whole-branch review + PR (user gate)
Reviewer brief as G3, **including the explicit "report and STOP — do not edit,
commit, push" instruction**. Run the merge-time drift check
(`git log fb2df0c8a..origin/dev -- tldw_chatbook/UI/Screens/settings*.py
tldw_chatbook/Widgets/settings_*`) **before** the review as well as at merge —
G3 shipped a review round against stale code because dev moved mid-phase.
Push, open PR against dev. **Do NOT merge — user gate.**
