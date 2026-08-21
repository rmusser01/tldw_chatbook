# Library Prompt Editor Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Do not
> delegate in this session unless the user explicitly requests subagents.

**Goal:** Give eligible Prompts a concise Basic editor and lifecycle-valid
actions without changing their canonical block representation, safety gates,
or power-user Advanced tools.

**Architecture:** The existing immutable `PromptBlockEditorState` remains the
only Prompt draft. Basic and Advanced are two presentations over that same
mounted state. Both regions stay mounted; targeted `display` and button-state
updates switch modes and lifecycle actions without remounting text editors.
The screen owns one profile-local preference and derives a temporary forced
Advanced presentation for incompatible or unsafe records.

**Tech Stack:** Python 3.11+, Textual 8.x, immutable dataclasses, existing
Prompt artifact/block compiler, TOML config owner, pytest.

---

## ADR check

ADR required: no

ADR path: N/A. This task directly implements the Prompt half of accepted
ADR-076. It adds no storage schema, service contract, editor base class, or
cross-source abstraction.

## File ownership

- `tldw_chatbook/Library/library_prompts_state.py`: pure mode coercion and
  Basic eligibility/reason contract.
- `tldw_chatbook/Widgets/Prompts/prompt_block_editor.py`: optional embedded
  host-owned lifecycle footer, preserving standalone/Console behavior.
- `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`: dual-mounted
  Basic/Advanced presentation, inline More actions, and targeted display sync.
- `tldw_chatbook/UI/Screens/library_screen.py`: preference ownership,
  mode/action handlers, draft capture, stable-block Basic edits, persistence,
  and existing action routing.
- `Tests/Library/test_library_prompts_state.py`: pure eligibility/coercion.
- `Tests/UI/test_prompt_block_editor.py`: embedded-footer compatibility.
- `Tests/UI/test_library_prompts_canvas.py`: mounted editor, lifecycle,
  round-trip, focus/undo/scroll, safety, and geometry.
- `Docs/User_Guide/library/prompts.md`: Basic/Advanced and lifecycle actions.

## Task 1: Pin mode and eligibility authority

- [ ] Add RED pure tests for invalid preference -> Basic; simple legacy and
  one-block-per-lane structured Prompt eligibility; Recipe, multi-block,
  compatibility, stale-conversion, conflict, and non-updatable saved records
  forcing Advanced with exact readable reasons.
- [ ] Add the smallest pure coercion and reason helpers in
  `library_prompts_state.py`. Do not add a generic disclosure type.
- [ ] Run only the exact new state tests and prove the eligibility inverse by
  allowing a multi-block Prompt into Basic, then restore immediately.

## Task 2: Compose Basic and Advanced over one mounted draft

- [ ] Add mounted RED tests for the approved Basic field set: Name,
  Description, Instructions, Message template, mode switch, meta/status, and
  no Advanced-only metadata/block/history/membership content painted.
- [ ] Add a host-owned-lifecycle option to `PromptBlockEditor`; default keeps
  all current standalone/Console actions, while the Library embedding retains
  only explicit structured Save-as-Prompt/Recipe actions inside Advanced.
- [ ] Mount common identity fields plus Basic and Advanced regions once.
  Toggle `display` only; never recompose for mode changes.
- [ ] Use existing Basic TextArea IDs so current save/export/Console seams keep
  reading the live draft. Incrementally reconcile Basic content edits into the
  exact existing block ID (or add one deterministic block to an empty lane)
  and sync the hidden Advanced editor through its native incremental method.
- [ ] Prove legacy and structured round trips preserve IDs, order, syntax,
  wrappers, mapping hints, and version-history behavior.

## Task 3: Own and persist Prompt display preference

- [ ] Add RED screen tests for default Basic, remembered Advanced across
  editor re-entry/restart, invalid value fallback, and forced Advanced that
  leaves the remembered Basic preference unchanged.
- [ ] Read/write only `library.prompt_editor_mode` through existing app-config
  mirror plus off-loop CLI config persistence. Mode switching captures live
  metadata before display changes but does not dirty the draft itself.
- [ ] Preserve current semantic focus when it remains visible; otherwise move
  to the corresponding mode control. Keep each mounted region's native scroll
  and TextArea undo stack intact.
- [ ] Prove a failed preference write warns without changing the live mode or
  Prompt data.

## Task 4: Make actions lifecycle-valid

- [ ] Add RED mounted tests for exact action sets:
  - new: Save prompt, Cancel;
  - saved clean: Use in Console, More actions;
  - saved dirty: Save changes, Discard changes;
  - conflict: Save as new, Reload;
  - mutation: progress plus disabled reason.
- [ ] Mount lifecycle actions once and patch visibility/labels/disabled state
  in place as dirty/save/mutation state changes. Do not recompose editor fields.
- [ ] Move Export, Copy Markdown, Duplicate, Collections, History, and Delete
  under one inline More actions region. Escape closes it and restores the More
  actions opener before existing editor Back handling.
- [ ] Route every action to its current handler; keep guarded discard,
  optimistic conflict, mutation interlock, and dirty navigation unchanged.
- [ ] Prove the bypass inverse by exposing Delete on a new draft, then restore.

## Task 5: Production geometry, review, and closeout

- [ ] At exact 100x30 and 170x48 with `TldwCli.CSS_PATH`, prove Basic's four
  fields and fixed lifecycle actions are painted and keyboard reachable;
  Advanced retains bounded scrolling and all structured content.
- [ ] Gate a mode switch, move focus after dispatch, and prove stale intent
  cannot steal the newer live focus.
- [ ] Run the touched state/block-editor/Prompt canvas owners only. Run Ruff on
  the final changed Python inventory, CSS build/parity only if CSS changes, the
  Impeccable post-edit detector, and `git diff --check`.
- [ ] Update the ASCII-only user guide, check TASK-19024 ACs, record exact
  evidence/inverses/ADR outcome, perform self/spec/quality review, and mark Done
  via Backlog CLI. Explicitly state that repository-wide pytest was not run.

## Required inverse checks

Apply one mutation at a time and immediately restore it:

1. Admit a multi-block Prompt into Basic; eligibility test fails.
2. Recompose on mode switch; native widget identity/undo/focus test fails.
3. Persist forced Advanced over remembered Basic; restart preference test fails.
4. Expose a lifecycle-invalid action (Delete on new or Save on clean); action
   set test fails.
5. Flatten Basic content instead of updating the stable block; round-trip test
   fails.

## Focused verification boundary

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Library/test_library_prompts_state.py \
  Tests/UI/test_prompt_block_editor.py \
  Tests/UI/test_library_prompts_canvas.py \
  -k 'prompt and (basic or advanced or disclosure or lifecycle or action or dirty or conflict or mutation or roundtrip or focus or undo or scroll or geometry)'
```

No full-suite claim is permitted for this task.
