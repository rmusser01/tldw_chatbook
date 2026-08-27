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
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
- `tldw_chatbook/css/tldw_cli_modular.tcss`
- `Tests/UI/test_console_thinking_disclosures.py`
- `Tests/UI/test_console_native_transcript.py`
