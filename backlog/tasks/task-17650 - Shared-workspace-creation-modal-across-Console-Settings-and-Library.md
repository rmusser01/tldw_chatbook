---
id: TASK-17650
title: >-
  Shared workspace creation modal across Console, Settings, and Library
status: In Progress
assignee: []
created_date: '2026-08-17 00:00'
labels:
  - workspaces
  - ux
priority: high
dependencies: []
---

## Description (the why)

Workspace creation today collects nothing (Console/Library) or a name only
(Settings), and folder binding — the thing that makes a workspace do anything
for agents — is a separate post-creation Settings-only action. Users are left
with a new "Workspace N" entry and no idea what it changes (task-713/task-714
complaints). Replace instant creation on all three surfaces with one shared
modal that collects a name + optional folder bindings and explains, truthfully,
what a workspace does.

Spec: `Docs/superpowers/specs/2026-08-17-workspace-create-modal-and-project-skills-design.md` §4.
Plan: `Docs/superpowers/plans/2026-08-17-workspace-create-modal.md`.

## Acceptance Criteria (the what)

- [x] Creating a workspace from Console rail, Settings ▸ Workspaces, or Library opens the shared modal; escape cancels with nothing created
- [x] Folder paths are validated inline as they are added (missing/home/root/sensitive/nested-overlap rejected before Create), via a validator shared with `add_folder_binding`
- [x] Name collisions render inline and keep the modal open
- [x] Console make_active path reproduces the full session-activation sequence including the TASK-713 toast; unchecked path only resyncs context
- [x] Folders bind read-only per ADR-028
- [x] User Guide pages updated and the 2026-07-26 settings-workspaces spec carries a supersession note

## Implementation Plan (the how)

Execute `Docs/superpowers/plans/2026-08-17-workspace-create-modal.md` (7 tasks:
validator extraction → browse-from-modal spike → full modal → Console/Settings/
Library wiring → docs + live verification).

## Implementation Notes

**Approach.** Seven tasks executed per the plan: extracted a pure
`validate_folder_binding_path` from `add_folder_binding` (Task 1); spiked
pushing `SelectDirectory` from a `ModalScreen` (Task 2); built the full
`WorkspaceCreateModal` — name prefill via `next_local_workspace_identity`,
optional multi-folder list with inline per-add validation, a
"Switch to this workspace" checkbox default-on, Create/Cancel with
escape-cancels-with-nothing-created (Task 3); wired Console's rail `New`,
Settings ▸ Workspaces' new "Create workspace…" button (replacing the old
inline name-input row), and Library's "Create local workspace" button, each
keeping its own post-create sync callback (Task 4-6); this task closed out
docs, the supersession note, and live verification (Task 7).

**Live verification (Task 7, isolated scratch profile — real config/data dirs
confirmed untouched before/after).** Console: created "PROJECT-ALPHA" with a
bound folder and "Switch" checked → toast "Created PROJECT-ALPHA and switched
Console to it.", rail switched, a `PROJECT-ALPHA Chat` tab opened
(TASK-713 toast + full activation sequence confirmed). Escape on a fresh
modal → dismissed with nothing created (rail unchanged). Console again with
"Switch" unchecked → toast "Created Workspace 2." only, rail stayed on
PROJECT-ALPHA (unchecked path correctly skips activation). Settings ▸
Workspaces: list refreshed live to show both new workspaces with correct
folder counts ("PROJECT-ALPHA (active) - 1 folders", "Workspace 2 - 0
folders"). Library: "Create local workspace" opened the identical shared
modal; default Create → toast "Created local workspace Workspace 3 and made
it active; Console now targets it.", and Console's rail reflected
"Workspace 3" as active on next visit — confirming all three surfaces drive
the same modal and stay in sync.

**Read-only folders (AC5).** `WorkspaceCreateModal._create()` calls
`add_folder_binding(workspace_id, folder)` with no `allow_write` argument,
and `add_folder_binding`'s signature defaults `allow_write: bool = False`
(`registry_service.py:788`) — confirmed by source read plus the full green
`Tests/Workspaces/` suite (253 passed) rather than by toggling "Allow write"
live in Settings.

**Name-collision + inline-validation ACs.** Live-confirmed the happy path
(valid folder add, duplicate-name path not separately driven live); AC
coverage for name collisions and inline rejection classes is carried by
`Tests/Workspaces/test_workspace_create_modal.py`
(`test_duplicate_name_error_keeps_modal_open`,
`test_invalid_folder_shows_inline_error`, plus the Task 1 validator unit
tests), all green in the Task 7 gate run.

**Deviations / findings, not part of this task's AC scope:**
- **Validator-split ruling (Task 1-2, pre-existing before this close-out):**
  `add_folder_binding` now calls the shared `validate_folder_binding_path`
  so the modal and the direct-add path can never drift on rejection rules.
- **Settings activation-default change (spec decision #2):** unlike the
  retired inline-name-input flow (which never activated), Settings-created
  workspaces now activate by default via the modal's checkbox — documented
  in `Docs/User_Guide/settings.md`'s refreshed Workspaces section and
  walkthrough step 5.
- **Live-verification-only defect found and filed separately, not fixed
  here (out of this task's AC scope):** the Name Input, folder-path Input,
  and "Switch to this workspace" Checkbox each render as an empty bordered
  box (zero content rows) while they hold keyboard focus — value/label
  invisible until focus moves away; the underlying value is unaffected
  (confirmed via blur-to-reveal and via the unchecked-checkbox path's
  correct functional result above). A similarly-shaped collapsed-content
  row was also seen on the pre-existing, unrelated "Show archived" Checkbox
  in Settings ▸ Workspaces, suggesting a broader pre-existing rendering
  interaction rather than something newly introduced only by this modal.
  Filed as `task-17961` for root-cause and fix; not blocking here because
  every functional outcome (values persist, Create/Cancel/toast semantics)
  was independently confirmed correct despite the cosmetic bug. This also
  explains a gap in the shipped Pilot coverage: `test_workspace_create_modal
  .py::test_make_active_checkbox_carried_on_result` asserts via
  `checkbox.value = False` rather than a real keypress/click, so it could
  not have caught a focused-widget rendering defect (the exact blind-spot
  shape documented in `backlog/docs/lessons-live-verification.md`'s
  TASK-13154.1 entry).

**Files touched this task (Task 7 only; Tasks 1-6 shipped the feature
itself):**
- `Docs/superpowers/specs/2026-07-26-settings-workspaces-category-design.md`
  (supersession note, §1)
- `Docs/User_Guide/settings.md` (Workspaces section + walkthrough step 5 +
  Verified-against stamp)
- `Docs/User_Guide/console/sessions-tabs-workspaces.md` (Workspaces table +
  three Common-tasks walkthroughs + Verified-against stamp)
- `Docs/User_Guide/library.md` ("Create local workspace" row + Verified-
  against stamp)
- `backlog/tasks/task-17961 - Focused-compact-Input-Checkbox-content-invisible-in-WorkspaceCreateModal.md`
  (new — the rendering defect found during live verification)
- This file (ACs ticked, Implementation Notes added; `status:` left
  untouched per instruction)
