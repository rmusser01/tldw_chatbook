---
id: TASK-17651
title: >-
  Project skills (.SKILLS/) folder discovery and prompt-driven import
status: In Progress
assignee: []
created_date: '2026-08-17 00:00'
labels:
  - skills
  - workspaces
  - ux
priority: high
dependencies:
  - TASK-17650
---

## Description (the why)

A user resuming work on an existing project has no way to bring that project's
skills into the app short of importing them one directory at a time through
Library ▸ Skills. Introduce a project-local `.SKILLS/` convention: on app
startup in (or under) a directory containing one, and after creating a
workspace bound to such a directory, offer a prompt-driven (never silent)
import. Imports stay quarantined behind the ADR-009 trust boundary.

Spec: `Docs/superpowers/specs/2026-08-17-workspace-create-modal-and-project-skills-design.md` §5.
Plan: `Docs/superpowers/plans/2026-08-17-project-skills-import.md`.

## Acceptance Criteria (the what)

- [x] Launching the app from a project (or subdirectory, up to the first .git ancestor) containing `.SKILLS/` offers an import prompt listing discovered skills; the first-run wizard suppresses it for that launch
- [x] Declining re-prompts only when the skill set's fingerprint changes; "Never for this folder" is permanent; `[skills] project_skills_prompt_enabled = false` disables the feature
- [x] Creating a workspace with a bound folder containing `.SKILLS/` chains the same offer after creation
- [x] Imports run through the existing importer with trust_approved=False (quarantined), never overwrite existing names silently, and the modal states the one-time trust review expectation with a route to Library ▸ Skills
- [x] Discovery refuses symlinked `.SKILLS/` dirs and entries, caps entries (50) and frontmatter reads (64 KiB), pre-flags invalid names, and renders repo-sourced text escaped
- [x] ADR for the convention added; config key documented; User Guide updated

## Implementation Plan (the how)

Execute `Docs/superpowers/plans/2026-08-17-project-skills-import.md` (6 tasks:
discovery → ledger → import modal → startup trigger → create-modal chaining →
ADR/docs + live verification). Starts only after TASK-17650 is on dev.

## Implementation Notes

Shipped as 9 commits (`e511355df`..`964cb04df`, TASK-17651's 6 plan tasks
plus 3 self-caught fix rounds):

- **Task 1 (discovery)** — `tldw_chatbook/Skills_Interop/project_skills_discovery.py`:
  pure `discover_project_skills(root)` — per-skill directories (`SKILL.md`)
  and loose `*.md` files, symlink refusal on both the `.SKILLS/`/`.skills/`
  dir and its entries, 50-entry / 64 KiB caps, invalid-name pre-flagging via
  the importer's own name grammar, skipped-entry reasons, and a
  sha256 fingerprint over the recognized skill files. Two same-day fix
  rounds: `670250506` (description parsing didn't match the importer's own
  frontmatter grammar) and `c950b271d` (the fingerprint originally stat'd
  the containing directory, which doesn't change mtime on POSIX when a file
  inside it is edited in place — switched to stat'ing the skill file itself;
  also fixed the ancestor walk to resolve symlinked ancestors before the
  `$HOME`/`.git`/fs-root stop checks).
- **Task 2 (ledger)** — `tldw_chatbook/Skills_Interop/project_skills_prompt.py`:
  `ProjectSkillsPromptLedger` at `<user_data_dir>/skills/project_prompts.json`,
  resolved-path keys, `should_offer_project_skills_prompt` gating truth table
  (no entry / declined+same fingerprint / declined+changed / never /
  kill-switch). Writes use a writer-unique temp filename (pid + thread id) —
  the two pre-existing sites with a fixed-temp-name equivalent
  (`local_skills_service.py`, `skill_trust_store.py`) were deliberately left
  alone and are now tracked as TASK-17963. Fix round `7005a09b9`: resolve
  ledger keys consistently and make advisory writes crash-proof (swallow
  `OSError`, never abort the caller).
- **Task 3 (import modal)** — `tldw_chatbook/Widgets/project_skills_import_modal.py`:
  `ProjectSkillsImportModal` (offer phase → results phase), verbatim
  spec §5.5 trust line, escaped repo-sourced text, `trust_approved=False` on
  every import, and the shared `maybe_offer_project_skills_import(app,
  discoveries)` helper that sequentially chains multiple discoveries and
  posts `NavigateToScreen("skills")` on Review.
- **Task 4 (startup trigger)** — wired into `app.py`'s `_post_mount_setup`
  next to the first-run wizard; wizard wins (skills offer defers to next
  launch when the wizard fires this launch), worker-thread discovery so
  startup stays unblocked. Fix round `971650bb5` (controller-completed after
  the implementer died at session limit): the startup discovery worker
  needed `exit_on_error=False` plus a full-body `try/except` so a discovery
  crash can never take the app down, and a re-entrancy guard
  (`_project_skills_offer_active`) so a second startup call while a modal
  chain is already open can never stack a duplicate modal or leak the flag
  on worker failure.
- **Task 5 (create-modal chaining)** — `WorkspaceCreateModal._add_folder`
  runs `discover_project_skills` synchronously per added folder (bounded
  scan, consistent with the existing synchronous path-validation call in the
  same handler) and annotates the row `"— contains N project skill(s)"`;
  `WorkspaceCreateResult.project_skills` carries the discoveries for
  successfully bound folders; all three creation surfaces (Console,
  Settings, Library) call `maybe_offer_project_skills_import` after their
  existing post-create sync.
- **Task 6 (this task) — ADR, config docs, User Guide, live verification**:
  - **ADR-069** (`backlog/decisions/069-project-skills-folder-convention.md`):
    the `.SKILLS/`/`.skills/` convention, both triggers, the fingerprint
    ledger, the kill-switch, and — the substantive decision — import-copy
    (not live-load) as the only design that keeps imported content inside
    ADR-009's trust boundary; live-load would put trust-sensitive prompt
    content structurally outside it.
  - **Config docs**: added a `[skills]` heading (none existed before) with
    `# project_skills_prompt_enabled = true  # offer .SKILLS/ import at
    startup; spec 2026-08-17` to `config.py`'s `CONFIG_TOML_CONTENT` docs
    template, next to `[console]`'s `workspace_root` doc line (the brief's
    cited `:2588` had drifted to `:2722` in this worktree's `config.py`;
    verified by grepping the live symbol before editing).
  - **User Guide**: `Docs/User_Guide/library/skills.md` gained a full
    "Project skills (`.SKILLS/`)" subsection (convention, both triggers,
    trust-review framing, kill-switch) plus two Quirks bullets and a
    Related-docs correction (the page previously said it "owns no
    `config.toml` keys" — no longer true); `Docs/User_Guide/console/
    sessions-tabs-workspaces.md`, `settings.md`, and `library.md` (the three
    workspace pages PR A's own docs commit touched) each gained the
    folder-row annotation + chained-offer description and a fresh "Verified
    against" stamp.
  - **Live verification** (isolated `$HOME`/`TLDW_CONFIG_PATH` scratch
    profile; real config confirmed byte-identical + same mtime before and
    after): all 5 brief scenarios plus the kill-switch clause of AC#2,
    driven end-to-end in a real launched TUI — see the full evidence log in
    `.superpowers/sdd/2026-08-17-project-skills-import/task-6-report.md`.
    One process-launch wrinkle worth recording: this venv's editable install
    (`__editable__.tldw_chatbook-*.pth`) maps `tldw_chatbook` to a **stale,
    different worktree** (`task-2512-mcp-unified`), so `python -m
    tldw_chatbook.app` only resolves this worktree's code when its root is
    either the process cwd or on `PYTHONPATH` — launching with cwd set to a
    fixture project directory (as the scenarios require) needed an explicit
    `PYTHONPATH=<this worktree>` alongside the isolated `HOME`, or every
    launch would import the wrong branch's code with a
    `ModuleNotFoundError` en route. Also found, live, a genuine (separately
    filed, not blocking) grammar bug: the folder-row annotation reads "1
    project skill(s)" for the singular case — TASK-17964.

Deviations from the plan: none structural. The three same-day fix rounds
above (Tasks 1, 2, 4) were caught by each task's own RED/GREEN discipline or
immediate self-review, not by this close-out task.

**Post-review gate connection (final review 2026-08-17, Finding 1):** the
final whole-branch review found that AC#2's gating ("declining re-prompts
only when the fingerprint changes; 'Never' is permanent; the kill-switch
disables the feature") was consulted ONLY by the startup trigger
(`startup_discovery_for`) -- the create-modal trigger's own path
(`WorkspaceCreateModal._add_folder` -> `maybe_offer_project_skills_import`)
scanned and offered unconditionally, bypassing the kill-switch, "Never",
and fingerprint gating entirely, which violated spec §5.3's "declining in
one place silences the other" and this AC. Fixed at the declared choke
point: `maybe_offer_project_skills_import` (every call site routes through
it) now checks the kill-switch first, filters every discovery through
`should_offer_project_skills_prompt` + the ledger before offering anything,
and suppresses when `app.skills_scope_service` is absent; `_add_folder`
skips the scan entirely (not just the offer) when the kill-switch is off.
New RED/GREEN pilot coverage: `test_never_for_folder_silences_create_
trigger`, `test_kill_switch_suppresses_create_offer`,
`test_kill_switch_suppresses_folder_discovery_scan`,
`test_missing_skills_scope_service_suppresses_offer` (all in
`Tests/Skills/test_project_skills_import_modal.py` /
`Tests/Workspaces/test_workspace_create_modal.py`). AC#2 stays ticked with
this fix in place; see `.superpowers/sdd/2026-08-17-project-skills-import/
finalfix-report.md` for the full fix-wave record (also covers 6 other
review findings: in-flight-import dismissal races, a recovery-dialog
startup stacking bug, a docs inaccuracy about unparseable frontmatter, a
missing precedence log line, and a coroutine-leak hardening fix).

Modified/added files (cumulative, all 6 tasks): `tldw_chatbook/Skills_Interop/
project_skills_discovery.py` (new), `project_skills_prompt.py` (new),
`tldw_chatbook/Widgets/project_skills_import_modal.py` (new),
`tldw_chatbook/Widgets/workspace_create_modal.py`, `tldw_chatbook/app.py`,
`tldw_chatbook/UI/Console_Modules/workspace.py`,
`tldw_chatbook/UI/Screens/{settings_screen,library_screen}.py`,
`tldw_chatbook/config.py`, `backlog/decisions/069-project-skills-folder-
convention.md` (new) + its README index row, `Docs/User_Guide/library/
skills.md`, `Docs/User_Guide/console/sessions-tabs-workspaces.md`,
`Docs/User_Guide/{settings,library}.md`, plus the `Tests/Skills/` and
`Tests/Workspaces/` suites for each of Tasks 1-5.

Follow-ups filed: TASK-17963 (fixed-temp-name write race, pre-existing sites
in `local_skills_service.py`/`skill_trust_store.py`), TASK-17964 (offer-modal
footer test coverage, Checkbox-escape assertion, the "1 project skill(s)"
pluralization fix, a stale-discovery-after-remove/rescan fix, and an
off-thread-discovery risk decision for `_add_folder`).
