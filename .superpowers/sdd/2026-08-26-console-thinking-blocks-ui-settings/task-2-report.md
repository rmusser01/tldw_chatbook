# Task 2 report — lazy Console thinking disclosures

## Result

Reused `ConsoleActivityDisclosure` for trusted thinking activity refs. The first
live evidence opens lazily, updates the same disclosure/Assistant-turn/answer
widgets, and auto-collapses at the first answer, explicitly owned tool, or
terminal boundary. Any mouse, Enter, Space, or existing `o` toggle cancels that
pending automatic action. Historical and late proprietary evidence starts
collapsed; collapsed bodies stay unmounted and resolve from the owning
`ThinkingEnvelope` only when expansion, Copy, or Inspector display projection
needs them.

Thinking refs participate in existing selection/navigation and expansion state.
Copy uses the full trusted body while the Assistant answer and its speech surface
remain unchanged. Proprietary detail is always the exact
`PROPRIETARY_THINKING_NOTICE`. Session changes prune thinking refs, pending/manual
state, and recycled expansion IDs. Activity reconciliation is by trusted ID, so
inserting thinking does not detach an already-expanded Tool detail or its text
selection.

## TDD evidence

Initial RED:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/UI/test_console_thinking_disclosures.py \
  Tests/UI/test_console_assistant_turn.py -q

13 failed, 40 passed
```

The failures were the missing disclosure projection/detail resolver plus the
expected lazy/lifecycle, manual-win, proprietary, and identity contracts. A later
exact Copy RED proved that trusted refs could not pass the legacy real-message-only
`select_message` gate:

```text
1 failed: selected_message_id stayed None and no thinking body reached clipboard
```

Focused GREEN after the production change and Copy/selection fix:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/UI/test_console_thinking_disclosures.py \
  Tests/UI/test_console_assistant_turn.py -q

54 passed, 1 existing dependency warning
```

The required broad target was run once:

```text
196 items: 192 passed, 4 failed
```

Two Task-2 regressions were then fixed and rerun directly (`2 passed`): message
ingest had grouped causal units twice, and the old activity-replacement test
asserted detachment even though the approved contract now preserves the same Tool
detail and selection. The other two failures are unchanged stale citation tests in
`test_console_transcript_selection_contract.py`: BASE already plans every Assistant
as `assistant-turn:<id>`, while those tests still look for a top-level
`message:<assistant-id>` row/build counter. Task 2 does not change that no-thinking
path and did not broaden production to undo Assistant-turn grouping.

## Verification

```text
CSS build: complete
CSS source/bundle sync: pass
Ruff check on all changed Python: pass
Ruff format check on new disclosure tests and the small disclosure widget: pass
git diff --check: pass
```

The test runner also emitted the repository's existing dependency-version warning
and temporary-directory cleanup warnings; neither changed test outcomes.

## Review fix round 1

Four Priority-1 review findings were reproduced before implementation:

- the real `ChatScreen` Inspector could not resolve a projected thinking ID, so
  no thinking Excerpt row existed;
- four lifecycle cases stayed in the wrong expansion state: an ordinal-less Tool
  did not close pending live thinking, and live proprietary evidence began
  collapsed at each answer/Tool/terminal boundary case;
- watermark pruning removed the Assistant owner of a selected thinking row; and
- two-sided tail trimming hid the selected thinking row's Assistant owner.

The bounded fix keeps the store's ordinary ownership guard intact, resolves a
current thinking ID through its Assistant owner, and supplies the full lazy body
to the Inspector (including bodies longer than 90 characters and the exact
proprietary notice). A stale-session projection still returns no Inspector rows.
Live-block identity is now the lifecycle fact even when proprietary presentation
status remains `unavailable`; genuinely new Tool IDs close the pending block
without inventing an activity round ordinal, while an already-observed Tool does
not close a later block. Closed-live state is session-scoped and bounded to blocks
still present. Finally, thinking activity IDs share their Assistant causal tuple in
`_ownership_by_message_id`, protecting the whole unit during watermark pruning and
two-sided windowing.

Fresh review-fix evidence:

```text
Task 2 disclosure/Assistant suite: 58 passed, 1 warning in 29.68s
Inspector/pruning/windowing subset: 17 passed, 2 warnings in 37.43s
Rendered real-Inspector full-body follow-up: 1 passed, 2 warnings in 3.08s
Ruff check on all six changed Python files: pass
git diff --check: pass
```

Ruff's whole-file format check reports the six touched legacy files would be
reformatted; no broad formatting churn was applied. No CSS changed in this review
round, so the already-passing Task 2 CSS source/bundle check did not require a
rebuild.

## Decisions

- Keep `detail_available` separate from mounted detail children so a collapsed
  disclosure retains its chevron without retaining private text in the DOM.
- Keep `_expanded_tool_output_ids` as the compatibility state and admit only trusted
  hashed thinking activity IDs alongside Tool IDs; no parallel controller or binding.
- Resolve displayable text from the current owning envelope and synthesize proprietary
  copy from the shared constant. No presentation body enters the conversation tree.
- Reconcile activity children by ID. Existing Tool expansion, focus, and nested text
  selection survive a thinking row inserted before them.
- Reuse one causal grouping result inside `set_messages`; thinking projection does not
  add a second full grouping walk.
- ADR required: no. This is direct implementation of the approved UI plan and Task 1
  owner/projection contracts; it adds no storage, schema, dependency, or runtime boundary.

## Files

- `tldw_chatbook/Widgets/Console/console_assistant_turn.py`
- `tldw_chatbook/Widgets/Console/console_transcript.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
- `tldw_chatbook/css/tldw_cli_modular.tcss`
- `Tests/UI/test_console_thinking_disclosures.py`
- `Tests/UI/test_console_native_transcript.py`
- `Tests/UI/test_console_native_chat_flow.py`
- `Tests/UI/test_console_transcript_pruning.py`
- `Tests/UI/test_console_transcript_two_sided_window.py`
