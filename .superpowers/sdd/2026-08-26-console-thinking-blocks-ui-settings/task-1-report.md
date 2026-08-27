# Task 1 report — trusted thinking activity projection

## Result

Added the exact proprietary notice constant, expanded the incumbent activity status
vocabulary, introduced a frozen owner-reference value, and added a pure projection
that merges supported thinking blocks into existing Assistant activity order without
changing transcript ownership or storing presentation state.

Review fix round 1 hardened that projection in two places: activity identity now
includes an explicit stable generation fact, and visual ordering now consumes explicit
model-round ownership instead of inferring rounds from TOOL-marker position or kind.

Review fix round 2 moved generation identity to the durable capture evidence itself.
Each `ThinkingCapture` now allocates one capture-scoped namespace and embeds it in every
block ID it creates. Presentation hashes that stored block ID alone, so no render caller
can re-key an activity when a local Assistant later receives its persisted message ID.

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

Review fix round 2 RED (against `8ed3eac927`):

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_thinking_capture.py::test_block_ids_are_capture_unique_bounded_and_provider_text_free \
  Tests/Chat/test_console_chat_store.py::test_thinking_activity_identity_survives_variant_and_durable_lifecycle -q

2 failed:
- fresh same-owner captures produced the same block ID
- the projection still required caller-manufactured session/generation identity
```

Review fix round 2 GREEN drives a real `CharactersRAGDB`/
`ChatPersistenceService`/`ConsoleChatStore` lifecycle. The literal activity ID remains
equal while live and unpersisted, after variant finalization, after the message gains a
persisted ID, after switching away and back, and after a fresh store hydrates the
selected generation; a second capture for the same Assistant differs:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_thinking_capture.py::test_block_ids_are_capture_unique_bounded_and_provider_text_free \
  Tests/Chat/test_console_chat_store.py::test_thinking_activity_identity_survives_variant_and_durable_lifecycle -q

2 passed
```

Nearest foundation and presentation regression:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/Chat/test_thinking_blocks.py \
  Tests/Chat/test_console_thinking_capture.py \
  Tests/Chat/test_console_thinking_presentation.py \
  Tests/Chat/test_console_turn_grouping.py \
  Tests/Chat/test_console_activity_presentation.py -q

208 passed
```

Nearest store/variant lifecycle regression:

```text
PYTHONPATH=. ../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_chat_store.py::test_message_completed_subscription_emits_each_successful_regeneration \
  Tests/Chat/test_console_chat_store.py::test_completion_generation_remains_monotonic_across_same_id_restore \
  Tests/Chat/test_console_chat_store.py::test_message_completed_subscription_add_variant_emits_but_selection_does_not \
  Tests/Chat/test_console_chat_store.py::test_store_adds_regenerated_variant_and_selects_it \
  Tests/Chat/test_console_chat_store.py::test_collapsed_buffer_variant_stream_finalizes_full_content \
  Tests/Chat/test_console_chat_store.py::test_thinking_activity_identity_survives_variant_and_durable_lifecycle -q

6 passed
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

- One standard-library `uuid4().hex` namespace is allocated once in each
  `ThinkingCapture` and embedded in its block IDs alongside the existing trusted owner
  digest, round, and sequence. IDs remain ASCII, under the existing 128-character
  schema bound, provider-text-free, and are persisted by the incumbent V1 envelope;
  existing durable block IDs are read unchanged and require no migration.
- UUID5 activity IDs hash the stored block ID alone. This is the only identity fact
  proven unchanged across live capture, variant installation and switching, first
  persistence, and durable hydration. It also keeps raw/hostile imported block IDs out
  of Textual identity strings. Presentation callers no longer mint or choose a
  generation namespace.
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
- The later agent-activity producer must stamp every model-round-owned TOOL/Planning
  row with that round's exact `activity_round_ordinal`; trailing post-run summaries
  stay `None`. Until producers adopt this contract, the fail-closed no-anchor layout
  deliberately groups thinking blocks without guessed per-tool interleaving.
- Planning-marker renaming and same-round suppression remain explicitly deferred to
  Task 4 of the approved plan.
