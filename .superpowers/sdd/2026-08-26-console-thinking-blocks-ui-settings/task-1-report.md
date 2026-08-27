# Task 1 report — trusted thinking activity projection

## Result

Added the exact proprietary notice constant, expanded the incumbent activity status
vocabulary, introduced a frozen owner-reference value, and added a pure projection
that merges supported thinking blocks into existing Assistant activity order without
changing transcript ownership or storing presentation state.

Review fix round 1 hardened that projection in two places: activity identity now
includes an explicit stable generation fact, and visual ordering now consumes explicit
model-round ownership instead of inferring rounds from TOOL-marker position or kind.

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

Review fix round 1 RED (against `236b8a448d`):

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_thinking_presentation.py \
  Tests/Chat/test_console_turn_grouping.py \
  -k "fresh_generations or variant_switch or explicit_round_ownership" -q

3 failed, 34 deselected:
- two identity regressions failed because `generation_id` was not accepted
- the ordering regression failed because `activity_round_ordinal` was not accepted
```

Review fix round 1 GREEN, including fresh capture generations, variant switching,
durable restore, explicit multi-row/skipped-round/trailing order, and the no-anchor
control:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_thinking_presentation.py \
  Tests/Chat/test_console_turn_grouping.py -q

38 passed
```

Nearest pure presentation regression:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_thinking_presentation.py \
  Tests/Chat/test_console_turn_grouping.py \
  Tests/Chat/test_console_activity_presentation.py -q

175 passed
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

- UUID5 activity IDs hash a JSON tuple of session, Assistant owner, explicit stable
  generation identity, and stable block ID. Fresh captures for the same Assistant
  therefore cannot leak selection/expansion state across attempts, while rerendering
  or restoring the same generation remains stable. Raw/imported identity text cannot
  enter a Textual identity, and delimiter-bearing hostile components cannot alias.
- `generation_id` is mandatory so a caller cannot silently fall back to an unsafe
  Assistant/block-only namespace. The caller must reuse an existing generation fact:
  `ConsoleVariant.id` for an installed current variant, a frozen existing generation
  attempt token identity during live replacement, or a durable persisted-message
  generation identity after restoration. It must never be minted during rerender.
- `live_block_id` is an explicit process-local input. Durable terminal status alone
  never implies a live disclosure.
- Proprietary evidence always projects `Thinking` plus `unavailable`; the body remains
  absent from the reference and later UI code must use the application constant.
- `ConsoleChatMessage.activity_round_ordinal` is the only model-round activity anchor.
  The merge preserves activity order and inserts each block immediately before the
  first activity explicitly owned by its round. Multiple rows in a round share the
  same ordinal; skipped-thinking rounds require no synthetic block; a final unanchored
  block sits before a trailing unowned row. With no anchors, all blocks remain grouped
  before existing activity rows instead of inventing an interleave from row positions.

## Known concerns

- Task 1 is pure presentation state, so no Textual widget/CSS or visual inspection was
  applicable. Disclosure lifecycle, lazy bodies, and painted status layout remain
  owned by later UI tasks.
- Task 2 must supply the stable `generation_id` described above when it calls the pure
  projection. In particular, it must freeze the existing live generation-attempt fact
  for the attempt lifetime rather than generate identity from a render pass.
- The later agent-activity producer must stamp every model-round-owned TOOL/Planning
  row with that round's exact `activity_round_ordinal`; trailing post-run summaries
  stay `None`. Until producers adopt this contract, the fail-closed no-anchor layout
  deliberately groups thinking blocks without guessed per-tool interleaving.
- Planning-marker renaming and same-round suppression remain explicitly deferred to
  Task 4 of the approved plan.
