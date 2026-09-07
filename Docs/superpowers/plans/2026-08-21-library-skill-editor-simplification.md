# Library Skill Editor Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Do not
> delegate in this session unless the user explicitly requests subagents.

**Goal:** Give eligible Skills a concise Basic editor while preserving exact
SKILL.md data, independent invocation choices, trust safety, and the complete
Advanced tool/runtime workbench.

**Architecture:** The existing immutable `SkillEditorState` remains the only
draft. Basic and Advanced are mounted presentations over that state, switched
with targeted `display` updates. The screen owns one profile-local preference;
trust expansion is derived independently from live safety state. The Advanced
tool chooser uses Textual's native `SelectionList`, while a pure reconciliation
helper preserves untouched ordered, duplicate, and unknown allowlist names.

**Tech Stack:** Python 3.11+, Textual 8.x, immutable dataclasses, existing
LocalSkillsService/SKILL.md grammar, existing agent tool providers, pytest.

---

## ADR check

ADR required: no

ADR path: `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`.
This task implements ADR-076's Skill-specific Basic/Advanced preference. It
does not change trust, runtime approval, SKILL.md storage, service ownership,
or agent tool authorization.

## File ownership

- `tldw_chatbook/Library/library_skills_state.py`: mode coercion, invocation
  copy, trust-detail expansion predicate, and exact allowlist reconciliation.
- `tldw_chatbook/Widgets/Library/library_skills_canvas.py`: dual-mounted
  Basic/Advanced presentation, native searchable tool picker, compact/detailed
  trust disclosure, and lifecycle-valid action strip.
- `tldw_chatbook/UI/Screens/library_screen.py`: profile preference, catalog
  presentation input, canonical draft capture, explicit tool-selection edits,
  lifecycle interlock, and current handler routing.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`: bounded picker/editor
  containment only where production geometry proves it necessary.
- `Tests/Library/test_library_skills_state.py`: pure mode, invocation, trust,
  and allowlist contracts.
- `Tests/UI/test_library_skills_canvas.py`: mounted editor, trust, lifecycle,
  picker, focus/undo/scroll, persistence, safety, and geometry.
- `Docs/User_Guide/library/skills.md`: ASCII-only Basic/Advanced and lifecycle
  documentation.

## Task 1: Pin pure Skill disclosure and allowlist authority

- [ ] Add RED pure tests for invalid preference -> Basic; independent user and
  agent invocation including neither/reference-only copy; healthy versus
  actionable trust expansion; and exact allowlist preservation.
- [ ] Add `SkillEditorMode`/coercion and the smallest pure invocation/trust
  helpers. Do not add a generic editor-mode framework.
- [ ] Add an exact allowlist reconciliation helper: before an explicit picker
  edit return the captured sequence unchanged; after an edit, remove
  deselected known names, retain untouched names and duplicates in original
  order, retain unknown names, and append newly selected known names in stable
  chooser order.
- [ ] Prove the required inverse by sorting/deduplicating the captured
  allowlist; restore immediately after the exact test fails.

## Task 2: Compose Basic and Advanced over one mounted draft

- [ ] Add mounted RED tests for the approved Basic fields: Name, Description,
  Instructions, independent You/Agent invocation, relevant Argument hint,
  trust summary, and no painted Advanced-only picker/context/files/model data.
- [ ] Mount common identity/instructions plus Basic and Advanced regions once.
  Switch only `display`; never recompose for a mode change.
- [ ] Basic shows Argument hint only while user invocation is enabled.
  Neither/reference-only renders explicit non-invocable copy. Agent configured
  on but trust/runtime unavailable distinguishes configuration from effective
  availability.
- [ ] Advanced shows context in plain language, supporting files, technical
  warnings, and read-only imported model metadata only when a value exists.
- [ ] Capture current fields before switching. Preserve TextArea undo, both
  regions' scroll, and semantic focus; a newer visible focus wins over stale
  restoration intent.

## Task 3: Add the bounded native tool restriction picker

- [ ] Add RED tests with 60 catalog names plus ordered duplicates and unknown
  imported names. Assert one bounded `SelectionList`, constant descendant
  count, search, keyboard selection, and unknown disabled/selected rows. The
  chooser renders one row per unique name while the separate captured content
  sequence retains duplicates.
- [ ] Build the screen's read-only skill-eligible catalog from the existing
  builtin provider and, when enabled, the existing local provider's
  `list_catalog()` output. Do not include skills, MCP tools, or runtime approval
  state; the allowlist narrows availability and never grants permission.
- [ ] Keep the picker selection in the mounted canvas. Filtering, opening,
  catalog refresh, mode switching, Cancel, and save-without-picker-change must
  not emit a draft change. Guard programmatic option rebuild messages; only a
  user-driven `SelectionList.SelectedChanged` reconciles and posts one
  canonical allowlist edit.
- [ ] Preserve unknown imported names visibly and losslessly. Catalog loss or
  gain changes only option availability/warnings until a user selects.

## Task 4: Make trust and lifecycle composition truthful

- [ ] Add RED mounted tests for exact lifecycle sets:
  - new: Save skill, Cancel;
  - saved clean: Back to list, More actions;
  - saved dirty: Save changes, Discard changes;
  - conflict: Reload;
  - delete armed: Delete, Cancel;
  - mutation: progress plus readable disabled reason.
- [ ] Keep trust/script actions only inside the trust region. Healthy trust is
  one line with View details; pending, changed, quarantined, script-access,
  manifest-error, and other actionable states expand automatically in either
  Basic or Advanced.
- [ ] Mount lifecycle actions once and patch visibility/labels/disabled state
  without replacing fields. `More actions` is one inline disclosure containing
  only valid saved-clean lifecycle actions such as Delete; Escape closes it and
  restores the opener. Remove the separate top Back control so saved-clean has
  exactly one Back to list action.
- [ ] Route through existing save/reload/delete/trust handlers. Preserve dirty
  exit guards, optimistic conflict, delete confirmation, script grants, and
  trust review/approval ownership.
- [ ] After first save, status must name trust review as the next step rather
  than imply the Skill is agent-ready.

## Task 5: Preference, geometry, review, and closeout

- [ ] Add RED screen tests for default Basic, remembered Advanced across
  editor re-entry/restart, invalid fallback, and safety-driven trust expansion
  that leaves the remembered display preference unchanged.
- [ ] Read/write only `library.skill_editor_mode` through the existing
  app-config mirror and off-loop config persistence. A failed write warns but
  does not change the live mode or draft.
- [ ] At exact 100x30 and 170x48 with `TldwCli.CSS_PATH`, prove Basic's core
  fields/actions and Advanced's bounded picker/trust controls are compositor-
  visible, contained, and keyboard reachable.
- [ ] Run only the touched state/Skills canvas owners. Run Ruff on the final
  changed Python inventory, rebuild/check CSS only if CSS changes, run the
  Impeccable detector/audit, and run `git diff --check`.
- [ ] Update the ASCII-only guide, check TASK-19025 ACs, record exact
  evidence/inverses/ADR outcome, self-review, and mark Done through Backlog
  CLI. Explicitly state that repository-wide pytest was not run.

## Required inverse checks

Apply one mutation at a time and immediately restore it:

1. Sort/deduplicate the captured allowlist; exact preservation test fails.
2. Let catalog refresh rewrite unknown/duplicate names; no-edit test fails.
3. Recompose on mode switch; widget identity/undo/focus test fails.
4. Hide an actionable trust state in Basic; safety disclosure test fails.
5. Expose Delete on a new draft or Save on saved-clean; lifecycle test fails.

## Focused verification boundary

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Library/test_library_skills_state.py \
  Tests/UI/test_library_skills_canvas.py \
  -k 'skill and (basic or advanced or disclosure or lifecycle or action or dirty or conflict or mutation or trust or invocation or allowlist or tool or focus or undo or scroll or geometry)'
```

No repository-wide pytest claim is permitted for this task.
