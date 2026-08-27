# Task 1 report — trusted thinking activity projection

## Result

Added the exact proprietary notice constant, expanded the incumbent activity status
vocabulary, introduced a frozen owner-reference value, and added a pure projection
that merges supported thinking blocks into existing Assistant activity order without
changing transcript ownership or storing presentation state.

## TDD evidence

RED:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_thinking_presentation.py \
  Tests/Chat/test_console_turn_grouping.py -q

2 collection errors:
- PROPRIETARY_THINKING_NOTICE was missing
- ConsoleThinkingActivityRef was missing
```

GREEN after the production change and collision-hardening test:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_thinking_presentation.py \
  Tests/Chat/test_console_turn_grouping.py -q

34 passed
```

Nearest pure presentation regression:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_thinking_presentation.py \
  Tests/Chat/test_console_turn_grouping.py \
  Tests/Chat/test_console_activity_presentation.py -q

171 passed
```

Static checks:

```text
../../.venv/bin/python -m ruff format <owned files>
../../.venv/bin/python -m ruff check <owned files>
git diff --check

Ruff: all checks passed
git diff --check: clean
```

The test runner also emitted the repository's existing dependency-version warning and
temporary-directory cleanup warnings after successful completion; neither changed the
test result.

## Decisions

- UUID5 activity IDs hash a JSON tuple of session, Assistant owner, and stable block
  ID. Raw/imported block text cannot enter a Textual identity, and delimiter-bearing
  hostile components cannot alias one another.
- `live_block_id` is an explicit process-local input. Durable terminal status alone
  never implies a live disclosure.
- Proprietary evidence always projects `Thinking` plus `unavailable`; the body remains
  absent from the reference and later UI code must use the application constant.
- Existing safe Thinking/Planning TOOL markers are model-round anchors when present.
  The ordered projection otherwise uses existing TOOL sequence order as the legacy/
  direct fallback, while leaving causal ownership and message identity unchanged.

## Known concerns

- Task 1 is pure presentation state, so no Textual widget/CSS or visual inspection was
  applicable. Disclosure lifecycle, lazy bodies, and painted status layout remain
  owned by later UI tasks.
- Planning-marker renaming and same-round suppression remain explicitly deferred to
  Task 4 of the approved plan.
