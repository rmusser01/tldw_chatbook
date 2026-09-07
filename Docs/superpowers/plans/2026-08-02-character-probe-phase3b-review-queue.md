# Character Probe Evals — Phase 3b (Review Queue) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a completed character-probe run readable — an ordered, filterable queue of conversations, a full transcript view, tags applied by keystroke, and an explicit "reviewed" verdict that survives closing the app.

**Architecture:** The queue replaces the neutral placeholder Phase 2 mounts for a character-probe run group (`evals_screen.py:1993`). Ordering, filtering, and hint computation are pure functions in the engine, so they are testable without Textual and cannot accidentally run during a bench run. The UI is a queue list plus a conversation view, both keyboard-first.

**Tech Stack:** Python ≥3.11, Textual, pytest. Engine from phases 1 and 3a; UI alongside Phase 2's `character_bench_editor.py`.

## Global Constraints

Copied from `Docs/superpowers/specs/2026-08-01-character-probe-eval-design.md` and phases 1-3a's conventions. Every task's requirements implicitly include this section.

- **The two bench types never share a detail surface.** The queue renders in the slot the word-bench `ResultsGrid` occupies, and neither leaks into the other's pane.
- **No logprobs / top-K / normalizer / canary vocabulary** anywhere in character-probe UI. A degenerate-canary warning must never appear on this bench type.
- **Review is a queue, not a grid.** "A 3D grid of conversations has no good 2D rendering and the reader's task is sequential anyway."
- **Keyboard-first.** "Reviewing dozens of conversations by mouse in a terminal app is untenable; moving between turns and conversations and applying tags must be a few keystrokes."
- **"Reviewed" is explicit, and "nothing notable" is a real verdict.** A conversation is done when the reviewer says so, not when it happens to carry a tag — otherwise "a clean, in-character, well-handled exchange is indistinguishable from one nobody has opened, and the progress count lies."
- **Ordering hints, never verdicts.** Hints reorder the queue; they are "never tags and never scores — if they become the judgment, the tool has quietly invented the metric it claims not to have."
- **Hints are computed at review time, on demand, never during the run.** "Nothing about a run's cost or duration may depend on hinting."
- **No composite score, anywhere.** "No view anywhere sums tags into a number."
- **User-authored text is a markup hazard.** Any Static carrying a card name, probe text, or model output takes `markup=False`; any Button label or tooltip interpolating it uses `escape_markup`. Whitespace shown with `␣` via `snippet_editor.render_snippet_cell`; newlines guarded to one line with `guard_single_line`'s `⏎` where a value must stay on one row.
- **Fail loudly, never silently default** — a corrupt row or missing record raises a named error identifying it; a write affecting no rows raises rather than reporting success.
- **`character_ids` are ints**; every eval id is a str. Do not normalise them. Note that `composed_system_prompts` in the run snapshot is keyed by **`str(card_id)`** (storage.py:570-573) — a reader keying by card id must `str()` it.
- **Tests must drive real widgets.** "The review UI's tests must drive real clicks and keypresses and assert the annotation persisted. Programmatically setting a widget's value passes while the feature is unusable." Every behavioural UI test presses or types through `pilot`, and asserts what is in the database afterwards.
- **Painted geometry is the arbiter.** This pane has pushed a control out of reach three times. Any task adding rows asserts the controls below it stay hit-testable — `screen.get_widget_at(*control.region.center)` resolves to the control — at 160x45 AND 235x52.
- Google-style docstrings (Args/Returns/Raises) on public callables; parameterized SQL only; CSS in `css/features/_evals.tcss` regenerated via `build_css.py`, never hand-edited.
- Run tests foreground: `/private/tmp/tldw-venv/bin/python -m pytest <paths> -p no:randomly` from the clone root. Never `-q`. **Pass `timeout: 600000` on the Bash call** — the harness auto-backgrounds anything past 120s and a backgrounded pytest has stalled this workflow twice.

## What already exists (verified — do not rebuild it)

- `Conversation(card_id: int, probe_index: int, sample_index: int, target_id: str, turns: tuple[ConversationTurn, ...], error: str)` and `ConversationTurn(user: str, reply: str, error: str)` (`models.py:170,197`). `turns` holds only the turns that ran; a partial conversation keeps its completed turns and records why it stopped in `Conversation.error`. **`ConversationTurn.error` is reserved and never populated today** — read `Conversation.error` for failure, never `turn.error == ""` as "this turn succeeded".
- `ConversationTurn.reply` may legitimately be `""` — "the model said nothing" is a real observation this eval exists to surface, not a malformed row.
- `CardSnapshot(id, name, description, system_prompt, personality, scenario, first_message, post_history_instructions, message_example)` (`models.py:158-166`). `first_message` is the card's opening message the conversation view must render.
- `load_conversations(db, run_group_id)` (`storage.py:880`), `load_probe_run_snapshot(db, run_group_id)` (:644). The snapshot holds `cards`, `probes`, `targets`, `sampler`, `extra_tags`, and `composed_system_prompts`.
- `annotate_turn` (:921), `load_turn_annotations` (:964), `mark_conversation_reviewed` (:990), `load_review_state` (:1028).
- **Phase 3a** ships `tags.py` with `Tag`, `TAG_KINDS`, `BUILTIN_TAGS`, `canonical_slug`, `coerce_tag`, `resolve_vocabulary`, `tag_by_slug`, and `storage.run_group_vocabulary(db, run_group_id)`. `annotate_turn` already rejects a slug outside the run's vocabulary.
- `EvalsScreen._character_run_group(group)` (`evals_screen.py:490`) is the existing predicate that keeps `ResultsGrid`/`EvalsCellInspector` away from a character-probe run group; the placeholder it guards is `#evals-detail-character-run-placeholder` (:1993).
- `snippet_editor.render_snippet_cell` (`snippet_editor.py:145`) and `snippet_editor.guard_single_line` (extracted in Phase 2) are the whitespace and single-line conventions.

## File Structure

- `tldw_chatbook/Evals/character_probe/hints.py` (new) — the four ordering hints as pure functions over already-loaded conversations plus the run snapshot. No DB, no UI, so "never during the run" is structural rather than a convention.
- `tldw_chatbook/Evals/character_probe/review_queue.py` (new) — pure ordering, filtering, and progress over `(conversations, review_state, hints)`. Its own file so the queue's rules are testable without mounting anything.
- `tldw_chatbook/UI/Evals/conversation_view.py` (new) — the transcript widget.
- `tldw_chatbook/UI/Evals/review_pane.py` (new) — the queue list, filters, progress, and the mark-reviewed affordance; replaces Phase 2's placeholder.
- `tldw_chatbook/UI/Evals/tag_picker.py` (new) — applying tags to a turn and creating a bench tag. Its own file because both the turn affordance and the create flow live in it and it is the one place `TAG_KINDS` reaches the UI.
- `tldw_chatbook/UI/Screens/evals_screen.py` — mount the review pane where the placeholder was; own the keyboard bindings that belong to the screen.
- Tests mirror each under `Tests/Evals/character_probe/` and `Tests/UI/`.

---

### Task 1: Ordering hints

**Files:**
- Create: `tldw_chatbook/Evals/character_probe/hints.py`
- Test: `Tests/Evals/character_probe/test_hints.py`

**Interfaces:**
- Consumes: `Conversation`, `ConversationTurn`, `CardSnapshot` (`models.py`); the run snapshot's `composed_system_prompts` mapping.
- Produces: `HINT_KINDS: tuple[str, ...]`; `Hint(kind: str, turn_index: int, detail: str)` frozen dataclass; `compute_hints(conversations: Sequence[Conversation], snapshot: Mapping[str, Any]) -> dict[tuple[int, int, int, str], tuple[Hint, ...]]` keyed by `(card_id, probe_index, sample_index, target_id)`.

Four hints, all from the spec: **empty or very short replies**, **replies containing text from the card's own system prompt (a leak)**, **refusal-shaped openings**, and **replies near-identical across targets**. They are hints, never tags and never scores — nothing here returns a number a caller could rank by, and `Hint` deliberately carries no severity or weight.

The near-identical check compares replies **across cells**, which is exactly why this runs on demand at review time rather than at write time: computing it during a run would add a cross-cell pass to every run for a signal only the reviewer uses.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.Evals.character_probe.hints import HINT_KINDS, Hint, compute_hints
from tldw_chatbook.Evals.character_probe.models import Conversation, ConversationTurn


def _conv(card_id=1, probe=0, sample=0, target="t-1", replies=("a reply",)):
    return Conversation(
        card_id=card_id,
        probe_index=probe,
        sample_index=sample,
        target_id=target,
        turns=tuple(ConversationTurn(user="q", reply=r) for r in replies),
    )


SNAPSHOT = {
    "composed_system_prompts": {"1": {"t-1": "You are Vex, a rooftop thief."}},
}


def test_the_hint_kinds_are_the_four_the_spec_names():
    assert set(HINT_KINDS) == {
        "empty-or-short",
        "prompt-leak",
        "refusal-shaped",
        "near-identical",
    }


def test_an_empty_reply_is_hinted():
    hints = compute_hints([_conv(replies=("",))], SNAPSHOT)
    kinds = [h.kind for h in hints[(1, 0, 0, "t-1")]]
    assert "empty-or-short" in kinds


def test_a_very_short_reply_is_hinted():
    hints = compute_hints([_conv(replies=("ok",))], SNAPSHOT)
    assert any(h.kind == "empty-or-short" for h in hints[(1, 0, 0, "t-1")])


def test_an_ordinary_reply_is_not_hinted_as_short():
    hints = compute_hints(
        [_conv(replies=("Vex leans back against the chimney and grins at you.",))],
        SNAPSHOT,
    )
    assert not any(
        h.kind == "empty-or-short" for h in hints.get((1, 0, 0, "t-1"), ())
    )


def test_a_reply_quoting_the_cards_system_prompt_is_hinted_as_a_leak():
    hints = compute_hints(
        [_conv(replies=("You are Vex, a rooftop thief. How can I help?",))],
        SNAPSHOT,
    )
    assert any(h.kind == "prompt-leak" for h in hints[(1, 0, 0, "t-1")])


def test_a_reply_merely_naming_the_character_is_not_a_leak():
    """Saying its own name is in character, not a prompt leak."""
    hints = compute_hints(
        [_conv(replies=("Vex is not someone you want to cross, friend.",))],
        SNAPSHOT,
    )
    assert not any(
        h.kind == "prompt-leak" for h in hints.get((1, 0, 0, "t-1"), ())
    )


@pytest.mark.parametrize(
    "reply",
    [
        "I'm sorry, but I can't help with that.",
        "I cannot assist with this request.",
        "As an AI language model, I am unable to do that.",
    ],
)
def test_a_refusal_shaped_opening_is_hinted(reply):
    hints = compute_hints([_conv(replies=(reply,))], SNAPSHOT)
    assert any(h.kind == "refusal-shaped" for h in hints[(1, 0, 0, "t-1")])


def test_a_refusal_later_in_a_long_reply_is_not_hinted_as_an_opening():
    reply = (
        "Vex laughs at the suggestion and paces the rooftop for a while, "
        "turning it over. Eventually he says he cannot help with that."
    )
    hints = compute_hints([_conv(replies=(reply,))], SNAPSHOT)
    assert not any(
        h.kind == "refusal-shaped" for h in hints.get((1, 0, 0, "t-1"), ())
    )


def test_near_identical_replies_across_targets_are_hinted_on_both():
    shared = "The night is cold and the tiles are slick underfoot tonight."
    hints = compute_hints(
        [
            _conv(target="t-1", replies=(shared,)),
            _conv(target="t-2", replies=(shared,)),
        ],
        {"composed_system_prompts": {"1": {"t-1": "p", "t-2": "p"}}},
    )
    assert any(h.kind == "near-identical" for h in hints[(1, 0, 0, "t-1")])
    assert any(h.kind == "near-identical" for h in hints[(1, 0, 0, "t-2")])


def test_the_same_reply_from_one_target_alone_is_not_near_identical():
    hints = compute_hints(
        [_conv(target="t-1", replies=("The night is cold and slick underfoot.",))],
        SNAPSHOT,
    )
    assert not any(
        h.kind == "near-identical" for h in hints.get((1, 0, 0, "t-1"), ())
    )


def test_different_cards_with_the_same_reply_are_not_compared():
    """Near-identical means across TARGETS for one cell, not across cards."""
    shared = "The night is cold and the tiles are slick underfoot tonight."
    hints = compute_hints(
        [
            _conv(card_id=1, target="t-1", replies=(shared,)),
            _conv(card_id=2, target="t-1", replies=(shared,)),
        ],
        {"composed_system_prompts": {"1": {"t-1": "p"}, "2": {"t-1": "p"}}},
    )
    assert not any(
        h.kind == "near-identical" for h in hints.get((1, 0, 0, "t-1"), ())
    )


def test_a_hint_carries_the_turn_it_describes():
    hints = compute_hints([_conv(replies=("a full and ordinary reply here", ""))], SNAPSHOT)
    short = [h for h in hints[(1, 0, 0, "t-1")] if h.kind == "empty-or-short"]
    assert short and short[0].turn_index == 1


def test_a_hint_has_no_score_or_severity():
    """Hints are never verdicts; nothing here may be ranked or summed."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(Hint)}
    assert fields == {"kind", "turn_index", "detail"}


def test_a_conversation_with_no_hints_is_absent_rather_than_empty():
    hints = compute_hints(
        [_conv(replies=("Vex leans back against the chimney and grins at you.",))],
        SNAPSHOT,
    )
    assert (1, 0, 0, "t-1") not in hints


def test_a_failed_conversation_with_no_turns_produces_no_hints_and_does_not_raise():
    conv = Conversation(
        card_id=1, probe_index=0, sample_index=0, target_id="t-1",
        turns=(), error="provider exploded",
    )
    assert compute_hints([conv], SNAPSHOT) == {}


def test_a_snapshot_missing_composed_prompts_does_not_raise():
    assert compute_hints([_conv(replies=("ok",))], {}) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_hints.py -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named '...hints'`

- [ ] **Step 3: Write minimal implementation**

`hints.py`. Structure it as one function per hint plus `compute_hints` combining them, so each is independently testable and none can grow a score:

```python
"""Cheap heuristics that reorder the review queue.

These are HINTS and never verdicts. Nothing here returns a number, a
severity, or a rank: the design spec is explicit that "if they become the
judgment, the tool has quietly invented the metric it claims not to have."
``Hint`` therefore carries a kind, the turn it describes, and a human
sentence -- nothing a caller could sort by magnitude.

Computed on demand at review time, never during a run. The near-identical
check compares replies ACROSS cells, so running it at write time would add a
cross-cell pass to every run for a signal only the reviewer uses; the spec
requires that "nothing about a run's cost or duration may depend on hinting."
This module imports no database and no UI, which makes that structural.
"""
```

Constants to define with a stated reason each, not bare magic numbers:

- `_SHORT_REPLY_CHARS = 24` — below this a reply is flagged `empty-or-short`. An empty reply always flags.
- `_LEAK_SHINGLE_WORDS = 6` — a run of this many consecutive words shared with the card's composed system prompt counts as a leak. Word-shingling rather than substring matching is what makes `test_a_reply_merely_naming_the_character_is_not_a_leak` pass: a single shared token is not a leak.
- `_REFUSAL_OPENING_CHARS = 80` — refusal shapes only count within this much of the reply's start, which is what `test_a_refusal_later_in_a_long_reply_is_not_hinted_as_an_opening` pins.
- `_REFUSAL_PATTERNS` — a tuple of lowercase regexes covering the three shapes the test parametrizes (`i'm sorry`, `i cannot/can't assist|help`, `as an ai`). Keep them as regexes over the normalised opening, not substring `in` checks, so `I am unable` and `I'm unable` both match.
- `_NEAR_IDENTICAL_RATIO = 0.9` — `difflib.SequenceMatcher(None, a, b).ratio()` at or above this counts as near-identical. Compare per `(card_id, probe_index, sample_index, turn_index)` across differing `target_id`s only, which is what the card-isolation test pins.

`compute_hints` returns a dict keyed by `(card_id, probe_index, sample_index, target_id)` containing only conversations that produced at least one hint — an absent key means "nothing stood out", which the queue renders as no hint rather than as an empty badge.

Guard every snapshot read: `snapshot.get("composed_system_prompts") or {}`, and key it with `str(card_id)` (the snapshot stores string keys — storage.py:570-573).

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_hints.py -p no:randomly`
Expected: PASS (18 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/hints.py Tests/Evals/character_probe/test_hints.py
git commit -m "feat(evals): ordering hints for the character-probe review queue (task-1691 phase 3b)"
```

---

### Task 2: The queue — ordering, filtering, progress

**Files:**
- Create: `tldw_chatbook/Evals/character_probe/review_queue.py`
- Test: `Tests/Evals/character_probe/test_review_queue.py`

**Interfaces:**
- Consumes: `Conversation` (`models.py`); `load_review_state`'s return shape — `dict[(card_id, probe_index, sample_index, target_id), {"reviewed_at": str, "note": str}]`; `compute_hints`' return shape (Task 1).
- Produces: `QueueEntry(key: tuple[int, int, int, str], conversation: Conversation, reviewed: bool, hints: tuple[Hint, ...])`; `QueueFilter(card_id: Optional[int], probe_index: Optional[int], target_id: Optional[str], unreviewed_only: bool)`; `build_queue(conversations, review_state, hints, queue_filter=QueueFilter()) -> tuple[QueueEntry, ...]`; `queue_progress(entries) -> tuple[int, int]` returning `(reviewed_count, total)`.

The spec: the run group "presents an ordered queue, filterable by card, probe, target, or 'not yet reviewed', so a reviewer can take a deliberate slice — *'just the villain card across all three models'* — in one sitting."

**Ordering:** hinted conversations first (they are "likely-interesting material"), then a stable deterministic order — `(card_id, probe_index, sample_index, target_id)` — so the queue does not reshuffle under the reviewer between sessions. Within the hinted group, the same stable key applies. **Reviewing a conversation must not move it**; the queue is stable across a review so the reviewer's position is meaningful.

**Progress counts against the FILTERED queue**, and `queue_progress` returns two integers rather than a percentage — a percentage is one step from a score.

- [ ] **Step 1: Write the failing test**

```python
from tldw_chatbook.Evals.character_probe.hints import Hint
from tldw_chatbook.Evals.character_probe.models import Conversation, ConversationTurn
from tldw_chatbook.Evals.character_probe.review_queue import (
    QueueFilter,
    build_queue,
    queue_progress,
)


def _conv(card_id=1, probe=0, sample=0, target="t-1"):
    return Conversation(
        card_id=card_id, probe_index=probe, sample_index=sample,
        target_id=target, turns=(ConversationTurn(user="q", reply="r"),),
    )


def test_an_empty_run_produces_an_empty_queue():
    assert build_queue([], {}, {}) == ()


def test_every_conversation_becomes_one_entry():
    convs = [_conv(target="t-1"), _conv(target="t-2")]
    assert len(build_queue(convs, {}, {})) == 2


def test_entries_are_in_a_stable_deterministic_order():
    convs = [
        _conv(card_id=2, probe=0), _conv(card_id=1, probe=1), _conv(card_id=1, probe=0),
    ]
    keys = [e.key for e in build_queue(convs, {}, {})]
    assert keys == [(1, 0, 0, "t-1"), (1, 1, 0, "t-1"), (2, 0, 0, "t-1")]


def test_hinted_conversations_sort_ahead_of_unhinted_ones():
    convs = [_conv(card_id=1), _conv(card_id=2)]
    hints = {(2, 0, 0, "t-1"): (Hint("empty-or-short", 0, "empty reply"),)}
    keys = [e.key for e in build_queue(convs, {}, hints)]
    assert keys[0] == (2, 0, 0, "t-1")


def test_hinted_conversations_keep_a_stable_order_among_themselves():
    convs = [_conv(card_id=2), _conv(card_id=1)]
    hints = {
        (1, 0, 0, "t-1"): (Hint("refusal-shaped", 0, "x"),),
        (2, 0, 0, "t-1"): (Hint("empty-or-short", 0, "y"),),
    }
    keys = [e.key for e in build_queue(convs, {}, hints)]
    assert keys == [(1, 0, 0, "t-1"), (2, 0, 0, "t-1")]


def test_reviewing_a_conversation_does_not_move_it():
    convs = [_conv(card_id=1), _conv(card_id=2)]
    before = [e.key for e in build_queue(convs, {}, {})]
    state = {(1, 0, 0, "t-1"): {"reviewed_at": "now", "note": ""}}
    after = [e.key for e in build_queue(convs, state, {})]
    assert before == after


def test_an_entry_knows_it_was_reviewed():
    convs = [_conv()]
    state = {(1, 0, 0, "t-1"): {"reviewed_at": "now", "note": "fine"}}
    assert build_queue(convs, state, {})[0].reviewed is True


def test_an_entry_with_no_review_state_is_not_reviewed():
    assert build_queue([_conv()], {}, {})[0].reviewed is False


def test_filtering_by_card():
    convs = [_conv(card_id=1), _conv(card_id=2)]
    entries = build_queue(convs, {}, {}, QueueFilter(card_id=2))
    assert [e.key[0] for e in entries] == [2]


def test_filtering_by_probe():
    convs = [_conv(probe=0), _conv(probe=1)]
    entries = build_queue(convs, {}, {}, QueueFilter(probe_index=1))
    assert [e.key[1] for e in entries] == [1]


def test_filtering_by_target():
    convs = [_conv(target="t-1"), _conv(target="t-2")]
    entries = build_queue(convs, {}, {}, QueueFilter(target_id="t-2"))
    assert [e.key[3] for e in entries] == ["t-2"]


def test_filtering_to_unreviewed_only():
    convs = [_conv(card_id=1), _conv(card_id=2)]
    state = {(1, 0, 0, "t-1"): {"reviewed_at": "now", "note": ""}}
    entries = build_queue(convs, state, {}, QueueFilter(unreviewed_only=True))
    assert [e.key[0] for e in entries] == [2]


def test_filters_combine():
    convs = [
        _conv(card_id=1, target="t-1"), _conv(card_id=1, target="t-2"),
        _conv(card_id=2, target="t-1"),
    ]
    entries = build_queue(convs, {}, {}, QueueFilter(card_id=1, target_id="t-2"))
    assert [e.key for e in entries] == [(1, 0, 0, "t-2")]


def test_a_filter_matching_nothing_yields_an_empty_queue():
    entries = build_queue([_conv()], {}, {}, QueueFilter(card_id=99))
    assert entries == ()


def test_progress_counts_reviewed_against_total():
    convs = [_conv(card_id=1), _conv(card_id=2)]
    state = {(1, 0, 0, "t-1"): {"reviewed_at": "now", "note": ""}}
    assert queue_progress(build_queue(convs, state, {})) == (1, 2)


def test_progress_counts_against_the_filtered_queue_not_the_whole_run():
    convs = [_conv(card_id=1), _conv(card_id=2)]
    state = {(1, 0, 0, "t-1"): {"reviewed_at": "now", "note": ""}}
    entries = build_queue(convs, state, {}, QueueFilter(card_id=2))
    assert queue_progress(entries) == (0, 1)


def test_progress_of_an_empty_queue_is_zero_of_zero():
    assert queue_progress(()) == (0, 0)


def test_a_failed_conversation_is_still_queued():
    """A partial or failed conversation is still evidence and stays reviewable."""
    conv = Conversation(
        card_id=1, probe_index=0, sample_index=0, target_id="t-1",
        turns=(), error="provider exploded",
    )
    assert len(build_queue([conv], {}, {})) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_review_queue.py -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named '...review_queue'`

- [ ] **Step 3: Write minimal implementation**

Two frozen dataclasses and two functions. `QueueFilter`'s fields all default to `None`/`False` so `QueueFilter()` means "everything". Sort with a two-part key — `(0 if hinted else 1, card_id, probe_index, sample_index, target_id)` — which gives hinted-first plus a stable tie-break in one pass, and note in a comment that the reviewed flag is deliberately absent from the sort key so reviewing does not reshuffle the queue under the reviewer.

`queue_progress` returns a `(reviewed, total)` tuple, not a ratio: a percentage is one refactor away from the composite score this design forbids. Say so in its docstring.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/review_queue.py Tests/Evals/character_probe/test_review_queue.py
git commit -m "feat(evals): the character-probe review queue's ordering and filters (task-1691 phase 3b)"
```

---

### Task 3: The conversation view

**Files:**
- Create: `tldw_chatbook/UI/Evals/conversation_view.py`
- Modify: `tldw_chatbook/css/features/_evals.tcss` (+ regenerate the bundle)
- Test: `Tests/UI/test_evals_conversation_view.py`

**Interfaces:**
- Consumes: `Conversation`, `ConversationTurn`, `CardSnapshot`; `Hint` (Task 1); `snippet_editor.render_snippet_cell`.
- Produces: `ConversationView(conversation, card, hints=(), annotations=None, id=...)` — a `Vertical` with `#evals-cv-opening` (the card's first message), one `#evals-cv-turn-{index}` block per turn containing `.evals-cv-user` and `.evals-cv-reply`, and `#evals-cv-error` when the conversation failed; `ConversationView.turn_count() -> int`; `ConversationView.focus_turn(index) -> None`.

The spec: it "renders the full exchange as turns, with the card's opening message and each scripted user turn in place, and a tag affordance on every model turn." The tag affordance itself is Task 5 — this task renders the transcript and leaves a per-turn anchor for it.

**Every string here is model output or user-authored card text**, so every `Static` takes `markup=False`. A reply is multi-line prose and is NOT collapsed to one line — `guard_single_line` is for row labels, not for the transcript. Leading and trailing whitespace still renders through `render_snippet_cell` so "the model replied with three spaces" is visible rather than looking empty.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Evals.character_probe.hints import Hint
from tldw_chatbook.Evals.character_probe.models import (
    CardSnapshot, Conversation, ConversationTurn,
)
from tldw_chatbook.UI.Evals.conversation_view import ConversationView

CARD = CardSnapshot(id=1, name="Vex", first_message="You find me on the roof.")
CONV = Conversation(
    card_id=1, probe_index=0, sample_index=0, target_id="t-1",
    turns=(
        ConversationTurn(user="What do you think about lying?", reply="Everyone lies."),
        ConversationTurn(user="And to protect someone?", reply="Then it's mercy."),
    ),
)


class _Host(App):
    def __init__(self, conversation=CONV, card=CARD, hints=(), annotations=None):
        super().__init__()
        self._args = (conversation, card, hints, annotations)

    def compose(self) -> ComposeResult:
        conversation, card, hints, annotations = self._args
        yield ConversationView(
            conversation, card, hints=hints, annotations=annotations, id="cv"
        )


@pytest.mark.asyncio
async def test_the_cards_opening_message_is_rendered():
    async with _Host().run_test() as pilot:
        opening = pilot.app.query_one("#evals-cv-opening")
        assert "You find me on the roof." in opening.render_str_or_text()


@pytest.mark.asyncio
async def test_every_turn_renders_its_user_prompt_and_reply():
    async with _Host().run_test() as pilot:
        view = pilot.app.query_one(ConversationView)
        assert view.turn_count() == 2
        assert len(view.query(".evals-cv-user")) == 2
        assert len(view.query(".evals-cv-reply")) == 2


@pytest.mark.asyncio
async def test_turns_render_in_order():
    async with _Host().run_test() as pilot:
        view = pilot.app.query_one(ConversationView)
        users = [w.render_str_or_text() for w in view.query(".evals-cv-user")]
        assert "lying" in users[0]
        assert "protect someone" in users[1]


@pytest.mark.asyncio
async def test_an_empty_reply_renders_visibly_rather_than_as_nothing():
    """An empty reply is a real observation, not a blank row."""
    conv = Conversation(
        card_id=1, probe_index=0, sample_index=0, target_id="t-1",
        turns=(ConversationTurn(user="q", reply=""),),
    )
    async with _Host(conversation=conv).run_test() as pilot:
        reply = pilot.app.query_one(".evals-cv-reply")
        assert reply.render_str_or_text().strip() != ""


@pytest.mark.asyncio
async def test_a_whitespace_only_reply_shows_the_whitespace_marker():
    conv = Conversation(
        card_id=1, probe_index=0, sample_index=0, target_id="t-1",
        turns=(ConversationTurn(user="q", reply="   "),),
    )
    async with _Host(conversation=conv).run_test() as pilot:
        assert "␣" in pilot.app.query_one(".evals-cv-reply").render_str_or_text()


@pytest.mark.asyncio
async def test_a_reply_containing_markup_renders_literally():
    conv = Conversation(
        card_id=1, probe_index=0, sample_index=0, target_id="t-1",
        turns=(ConversationTurn(user="q", reply="[bold]not bold[/]"),),
    )
    async with _Host(conversation=conv).run_test() as pilot:
        assert "[bold]" in pilot.app.query_one(".evals-cv-reply").render_str_or_text()


@pytest.mark.asyncio
async def test_a_card_name_containing_markup_renders_literally():
    card = CardSnapshot(id=1, name="Vex[/]v2", first_message="hi")
    async with _Host(card=card).run_test() as pilot:
        text = pilot.app.query_one(ConversationView).render_str_or_text()
        assert "[/]" in text


@pytest.mark.asyncio
async def test_a_failed_conversation_shows_its_error_and_keeps_its_turns():
    conv = Conversation(
        card_id=1, probe_index=0, sample_index=0, target_id="t-1",
        turns=(ConversationTurn(user="q", reply="a reply"),),
        error="provider exploded",
    )
    async with _Host(conversation=conv).run_test() as pilot:
        assert "provider exploded" in (
            pilot.app.query_one("#evals-cv-error").render_str_or_text()
        )
        assert len(pilot.app.query(".evals-cv-reply")) == 1


@pytest.mark.asyncio
async def test_a_conversation_that_never_ran_a_turn_still_renders_its_error():
    conv = Conversation(
        card_id=1, probe_index=0, sample_index=0, target_id="t-1",
        turns=(), error="cancelled before the first call",
    )
    async with _Host(conversation=conv).run_test() as pilot:
        assert pilot.app.query_one("#evals-cv-error")
        assert pilot.app.query_one(ConversationView).turn_count() == 0


@pytest.mark.asyncio
async def test_a_successful_conversation_shows_no_error_block():
    async with _Host().run_test() as pilot:
        assert not pilot.app.query("#evals-cv-error")


@pytest.mark.asyncio
async def test_a_hint_renders_on_the_turn_it_describes():
    hints = (Hint("empty-or-short", 1, "the reply was two characters"),)
    async with _Host(hints=hints).run_test() as pilot:
        turn = pilot.app.query_one("#evals-cv-turn-1")
        assert "empty-or-short" in turn.render_str_or_text() or "short" in (
            turn.render_str_or_text()
        )


@pytest.mark.asyncio
async def test_a_hint_never_renders_as_a_tag_or_a_score():
    """Hints are never verdicts. No number, no applied-tag styling.

    `.evals-cv-applied-tag` is the class an APPLIED tag chip carries (Task 5).
    A hint must never wear it -- the two must stay visually distinguishable,
    because a hint that reads as a tag has become the verdict the spec
    forbids. Deliberately NOT `.evals-cv-tag`, which is Task 5's per-turn tag
    BUTTON and will exist on every turn.
    """
    hints = (Hint("refusal-shaped", 0, "opens with a refusal"),)
    async with _Host(hints=hints).run_test() as pilot:
        turn = pilot.app.query_one("#evals-cv-turn-0")
        assert turn.query(".evals-cv-hint")
        assert not turn.query(".evals-cv-applied-tag")


@pytest.mark.asyncio
async def test_no_logprob_or_canary_vocabulary_appears():
    async with _Host().run_test() as pilot:
        text = pilot.app.query_one(ConversationView).render_str_or_text().lower()
        for forbidden in ("logprob", "top-k", "top_k", "canary", "normalizer"):
            assert forbidden not in text


@pytest.mark.asyncio
async def test_an_existing_annotation_renders_on_its_turn():
    annotations = {1: {"tags": ["broke-character"], "note": "slipped here"}}
    async with _Host(annotations=annotations).run_test() as pilot:
        turn = pilot.app.query_one("#evals-cv-turn-1")
        assert "broke-character" in turn.render_str_or_text()
        assert "slipped here" in turn.render_str_or_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (235, 52)])
async def test_the_last_turn_is_reachable_at_realistic_sizes(size):
    turns = tuple(
        ConversationTurn(user=f"question {i}", reply="a reply " * 20) for i in range(12)
    )
    conv = Conversation(
        card_id=1, probe_index=0, sample_index=0, target_id="t-1", turns=turns,
    )
    async with _Host(conversation=conv).run_test(size=size) as pilot:
        view = pilot.app.query_one(ConversationView)
        view.scroll_end(animate=False)
        await pilot.pause()
        last = pilot.app.query_one("#evals-cv-turn-11")
        hit = pilot.app.screen.get_widget_at(*last.region.center)[0]
        assert hit is last or last in hit.ancestors
```

`render_str_or_text()` is a placeholder for however this repo's existing tests read a widget's painted text — **grep `Tests/UI/test_evals_character_bench_editor.py` and `test_evals_snippet_editor.py` for the real helper and use it verbatim**. Do not add a new one.

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_conversation_view.py -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named '...conversation_view'`

- [ ] **Step 3: Write minimal implementation**

A `Vertical` whose `compose()` yields: the card's opening message (`#evals-cv-opening`, `markup=False`), then per turn a `Vertical(id=f"evals-cv-turn-{index}")` containing the scripted user prompt (`.evals-cv-user`) and the reply (`.evals-cv-reply`), each a `Static(markup=False)` rendering through `render_snippet_cell` so whitespace stays visible; then any hints for that turn as a plain line (`.evals-cv-hint`) and any existing annotation, whose applied tags carry `.evals-cv-applied-tag` (`.evals-cv-annotation` on the block); then `#evals-cv-error` **only when `conversation.error`** is non-empty.

Keep `.evals-cv-hint` and `.evals-cv-applied-tag` visually distinct and never share a class between them. Task 5 adds a per-turn tag *button* with id `#evals-cv-tag-{turn_index}` inside this same block — that is a third thing again, and none of the three may be styled as another.

`focus_turn(index)` scrolls the turn block into view and focuses it — Task 7's keyboard navigation calls it, and Task 5 mounts the tag affordance inside the same block.

Style hints and annotations differently and give hints no tag-like class: `test_a_hint_never_renders_as_a_tag_or_a_score` pins that a hint is not mistakable for an applied tag.

CSS: `#evals-conversation-view` gets `height: 1fr; overflow-y: auto;`. Note the difference from Phase 2's editor, which took `overflow-y: auto` with no height because `#evals-detail-pane` bounds it — here the view is one of two siblings in a split pane and must take its share explicitly. Verify the real parent before choosing, and regenerate the bundle with `/private/tmp/tldw-venv/bin/python tldw_chatbook/css/build_css.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_conversation_view.py -p no:randomly`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Evals/conversation_view.py tldw_chatbook/css Tests/UI/test_evals_conversation_view.py
git commit -m "feat(evals): the character-probe conversation view (task-1691 phase 3b)"
```

---

### Task 4: The review pane replaces the placeholder

**Files:**
- Create: `tldw_chatbook/UI/Evals/review_pane.py`
- Modify: `tldw_chatbook/UI/Screens/evals_screen.py`
- Modify: `tldw_chatbook/css/features/_evals.tcss` (+ regenerate the bundle)
- Test: `Tests/UI/test_evals_review_pane.py`

**Interfaces:**
- Consumes: `build_queue`, `queue_progress`, `QueueFilter` (Task 2); `compute_hints` (Task 1); `ConversationView` (Task 3); `load_conversations`, `load_probe_run_snapshot`, `load_review_state`, `load_turn_annotations`, `mark_conversation_reviewed` (storage).
- Produces: `ReviewPane(view_model, run_group_id, id="evals-review-pane")` with `#evals-review-list` (one `#evals-review-row-{index}` per queued conversation), `#evals-review-progress`, the filter controls `#evals-review-filter-card` / `-probe` / `-target` / `-unreviewed`, and `#evals-review-mark-reviewed`; message `ReviewPane.ConversationSelected(key)`.

This replaces the neutral placeholder Phase 2 mounts at `evals_screen.py:1993` (`#evals-detail-character-run-placeholder`). Keep `_character_run_group` (`evals_screen.py:490`) as the predicate — it already keeps `ResultsGrid`/`EvalsCellInspector` away from this bench type; this task changes only what mounts in the true branch.

**"Reviewed" is explicit.** `#evals-review-mark-reviewed` writes review state via `mark_conversation_reviewed` even when the conversation carries no annotations at all — "nothing notable" is a real verdict and the progress count depends on it.

**Hints are computed here, on demand**, when the pane loads a run group — never in the run worker.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_selecting_a_character_run_group_renders_the_review_pane(
    evals_app, reviewable_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        assert pilot.app.screen.query("#evals-review-pane")
        assert not pilot.app.screen.query("#evals-detail-character-run-placeholder")


@pytest.mark.asyncio
async def test_a_word_bench_run_group_still_renders_its_results_grid(
    evals_app, word_bench_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=word_bench_run_group)
        await pilot.pause()
        assert not pilot.app.screen.query("#evals-review-pane")


@pytest.mark.asyncio
async def test_every_conversation_gets_a_queue_row(evals_app, reviewable_run_group):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        assert len(pilot.app.screen.query(".evals-review-row")) == 4


@pytest.mark.asyncio
async def test_clicking_a_queue_row_shows_that_conversation(
    evals_app, reviewable_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        await pilot.click("#evals-review-row-1")
        await pilot.pause()
        assert pilot.app.screen.query_one("#evals-conversation-view")


@pytest.mark.asyncio
async def test_progress_starts_at_zero_of_the_queue_length(
    evals_app, reviewable_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        text = pilot.app.screen.query_one("#evals-review-progress").render_str_or_text()
        assert "0" in text and "4" in text


@pytest.mark.asyncio
async def test_marking_reviewed_persists_and_advances_progress(
    evals_app, reviewable_run_group, evals_db
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        await pilot.click("#evals-review-row-0")
        await pilot.click("#evals-review-mark-reviewed")
        await pilot.pause()

        from tldw_chatbook.Evals.character_probe.storage import load_review_state
        assert len(load_review_state(evals_db, reviewable_run_group)) == 1
        text = pilot.app.screen.query_one("#evals-review-progress").render_str_or_text()
        assert "1" in text


@pytest.mark.asyncio
async def test_a_conversation_with_no_annotations_can_be_marked_reviewed(
    evals_app, reviewable_run_group, evals_db
):
    """Reading a conversation and finding nothing notable is a real verdict."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        await pilot.click("#evals-review-row-0")
        await pilot.click("#evals-review-mark-reviewed")
        await pilot.pause()

        from tldw_chatbook.Evals.character_probe.storage import load_turn_annotations
        assert load_turn_annotations(evals_db, reviewable_run_group) == {}
        from tldw_chatbook.Evals.character_probe.storage import load_review_state
        assert load_review_state(evals_db, reviewable_run_group)


@pytest.mark.asyncio
async def test_review_state_survives_reopening_the_run_group(
    evals_app, reviewable_run_group
):
    """The queue is resumable across sessions, which annotation work requires."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        await pilot.click("#evals-review-row-0")
        await pilot.click("#evals-review-mark-reviewed")
        await pilot.pause()

        # Navigate away and back, so the pane is rebuilt from the database
        # rather than from whatever it still holds in memory.
        pilot.app.screen.select(kind="bench", id=bench_id_of_that_run)
        await pilot.pause()
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        text = pilot.app.screen.query_one("#evals-review-progress").render_str_or_text()
        assert "1" in text


@pytest.mark.asyncio
async def test_filtering_to_unreviewed_only_hides_a_reviewed_conversation(
    evals_app, reviewable_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        await pilot.click("#evals-review-row-0")
        await pilot.click("#evals-review-mark-reviewed")
        await pilot.click("#evals-review-filter-unreviewed")
        await pilot.pause()
        assert len(pilot.app.screen.query(".evals-review-row")) == 3


@pytest.mark.asyncio
async def test_a_card_name_with_markup_renders_literally_in_a_queue_row(
    evals_app, run_group_with_markup_card_name
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=run_group_with_markup_card_name)
        await pilot.pause()
        row = pilot.app.screen.query_one("#evals-review-row-0")
        assert "[/]" in row.render_str_or_text()


@pytest.mark.asyncio
async def test_no_logprob_or_canary_vocabulary_appears_in_the_pane(
    evals_app, reviewable_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        text = pilot.app.screen.query_one("#evals-review-pane").render_str_or_text().lower()
        for forbidden in ("logprob", "top-k", "top_k", "canary", "normalizer"):
            assert forbidden not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (235, 52)])
async def test_mark_reviewed_stays_hit_testable_at_realistic_sizes(
    evals_app, reviewable_run_group, size
):
    async with evals_app.run_test(size=size) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        button = pilot.app.screen.query_one("#evals-review-mark-reviewed")
        hit = pilot.app.screen.get_widget_at(*button.region.center)[0]
        assert hit is button or button in hit.ancestors
```

The reopen test needs `bench_id_of_that_run` alongside the run-group fixture — add it if the fixture module does not already expose one. Build `reviewable_run_group` (2 cards × 2 probes × 1 target × 1 sample = 4 conversations), `word_bench_run_group`, and `run_group_with_markup_card_name` on `Tests/UI/test_evals_character_run_e2e.py`'s existing fixtures — **import them rather than writing new run-group construction**, and remember that fixture's own lesson: build target rows the way the real "+ New target" form writes them, never as hand-built dicts.

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_review_pane.py -p no:randomly`
Expected: FAIL — the placeholder still mounts; `#evals-review-pane` does not exist.

- [ ] **Step 3: Write minimal implementation**

`ReviewPane` is a `Horizontal` split: the queue list on the left (`#evals-review-list`, one `Button` per entry with the card name, probe index, target name, a reviewed marker, and a hint marker), the conversation on the right. Above the list, the four filter controls; below it, `#evals-review-progress` rendering `f"{reviewed} of {total} reviewed"` — two integers, never a percentage.

On mount, load once: `load_conversations`, `load_probe_run_snapshot`, `load_review_state`, `load_turn_annotations`, then `compute_hints`. Rebuild only the list when a filter changes; rebuild only the right-hand side when the selection changes — recomposing the whole pane on every keystroke is what made Phase 2's card picker drop focus.

In `evals_screen.py`, replace the placeholder mount (`:1993`) with `ReviewPane(...)`. Leave `_character_run_group` and the `ResultsGrid` guards exactly as they are — they already encode "never the word-bench surface for this bench type", and Task 4 must not weaken that.

CSS: give the list a bounded width and the conversation side `1fr`. Regenerate the bundle.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_review_pane.py Tests/UI/test_evals_screen.py Tests/UI/test_evals_character_run_e2e.py -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Evals/review_pane.py tldw_chatbook/UI/Screens/evals_screen.py tldw_chatbook/css Tests/UI/test_evals_review_pane.py
git commit -m "feat(evals): the character-probe review queue pane (task-1691 phase 3b)"
```

---

### Task 5: Applying a tag to a turn

**Files:**
- Create: `tldw_chatbook/UI/Evals/tag_picker.py`
- Modify: `tldw_chatbook/UI/Evals/conversation_view.py`
- Test: `Tests/UI/test_evals_tag_picker.py`

**Interfaces:**
- Consumes: `run_group_vocabulary` (Phase 3a), `annotate_turn`, `Tag`, `TAG_KINDS`.
- Produces: `TagPicker(vocabulary, selected_slugs, note, id=...)` with `#evals-tag-search`, one `#evals-tag-row-{index}` per matching tag grouped under a heading per kind, `#evals-tag-note`, `#evals-tag-apply`; message `TagPicker.Applied(slugs: tuple[str, ...], note: str)`. `ConversationView` gains `#evals-cv-tag-{turn_index}` — the per-turn affordance that opens it.

The spec: "the UI offers existing tags before creating new ones, to limit the `broke-character` / `OOC` / `out-of-character` fragmentation that per-bench extension invites." So the picker leads with the run's vocabulary and search filters it; creating a new one is Task 6 and sits **after** the existing list, never before it.

Tags are grouped by kind with the kind named, because a reviewer choosing between `notable` and `broke-character` is choosing a kind as much as a label.

- [ ] **Step 1: Write the failing test**

Cover, driving everything through `pilot`: every vocabulary tag renders a row; rows are grouped under their kind's heading; clicking a row selects it and clicking again deselects; search filters case-insensitively and does not drop a selection filtered out of view; a tag label containing markup renders literally; applying posts `Applied` with the selected slugs and the typed note; applying with **no** tags but a typed note still posts (a note alone is a real observation); and the geometry assertion that `#evals-tag-apply` stays hit-testable at 160x45 and 235x52 with a full ten-tag vocabulary.

Then the integration test that matters most, driving the real widget chain and asserting the **database**:

```python
@pytest.mark.asyncio
async def test_applying_a_tag_from_the_conversation_view_persists_it(
    evals_app, reviewable_run_group, evals_db
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        await pilot.click("#evals-review-row-0")
        await pilot.pause()
        await pilot.click("#evals-cv-tag-0")
        await pilot.pause()
        await pilot.click("#evals-tag-row-0")
        await pilot.click("#evals-tag-apply")
        await pilot.pause()

        from tldw_chatbook.Evals.character_probe.storage import load_turn_annotations
        stored = load_turn_annotations(evals_db, reviewable_run_group)
        assert stored, "no annotation was written"
        (key, value), = stored.items()
        assert key[4] == 0  # turn_index
        assert value["tags"]
```

and its converse, which is the one that catches an unusable affordance:

```python
@pytest.mark.asyncio
async def test_the_applied_tag_renders_on_the_turn_without_reopening_the_run(
    evals_app, reviewable_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        await pilot.click("#evals-review-row-0")
        await pilot.click("#evals-cv-tag-0")
        await pilot.pause()
        await pilot.click("#evals-tag-row-0")
        await pilot.click("#evals-tag-apply")
        await pilot.pause()
        turn = pilot.app.screen.query_one("#evals-cv-turn-0")
        assert "broke-character" in turn.render_str_or_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_tag_picker.py -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named '...tag_picker'`

- [ ] **Step 3: Write minimal implementation**

Model `TagPicker` on Phase 2's `CardPicker` (`card_picker.py`) — it solved the same problems and its traps are already paid for: rows live in their own `#evals-tag-picker-rows` container so filtering never rebuilds the search `Input` and drops focus, and selection is tracked by slug so a tag filtered out of view stays selected. **Row ids are positional over the filtered list**, exactly as `CardPicker`'s are; do not use a row id to mean a specific tag after a search.

Every tag label is user-authored (a bench may relabel a built-in), so labels go through `escape_markup` in `Button` labels and `markup=False` in any `Static`.

`ConversationView` gains a tag button per turn, mounted inside that turn's block, opening the picker seeded with the turn's existing tags and note. On `Applied`, the pane calls `annotate_turn` and refreshes just that turn's annotation line — not the whole view.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_tag_picker.py Tests/UI/test_evals_conversation_view.py Tests/UI/test_evals_review_pane.py -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Evals/tag_picker.py tldw_chatbook/UI/Evals/conversation_view.py Tests/UI/test_evals_tag_picker.py
git commit -m "feat(evals): apply review tags to a conversation turn (task-1691 phase 3b)"
```

---

### Task 6: Creating a bench tag, kind required

**Files:**
- Modify: `tldw_chatbook/UI/Evals/tag_picker.py`
- Modify: `tldw_chatbook/UI/Evals/review_pane.py`
- Test: `Tests/UI/test_evals_tag_picker.py`

**Interfaces:**
- Consumes: `canonical_slug`, `TAG_KINDS`, `coerce_tag` (Phase 3a); `save_character_bench`, `load_character_bench`.
- Produces: `#evals-tag-new-name`, `#evals-tag-new-kind` (one control per kind, no pre-selection), `#evals-tag-new-create`, `#evals-tag-new-error`; message `TagPicker.TagCreated(tag: Tag)`.

The spec is unusually firm here: "**Creating a tag requires choosing its kind** — the extension flow asks, with no default. A kind guessed on the user's behalf would quietly mis-group observations in the summary, and `notable` is not a safe fallback: it would make genuine failures invisible in exactly the view meant to surface them."

So: **no kind is pre-selected**, and Create is refused with a visible reason until one is chosen. Do not make `notable` the default "to be helpful".

A created tag is added to the **bench's** `extra_tags` — it outlives this run. But the run being reviewed keeps the vocabulary its snapshot captured (Phase 3a, task 4), so a newly created tag is usable in **this** review only if the pane also adds it to its in-memory vocabulary. Decide and state which you implement:
- **Recommended:** add it to the bench AND to the pane's live vocabulary for this session, so the reviewer can use the tag they just created. Note in the code that `run_group_vocabulary` will not return it on a later reopen of this older run, and why that is the correct provenance trade-off.
- The alternative — refusing to use a new tag until the bench is re-run — is defensible but frustrating; if you pick it, say so and make the UI state it plainly rather than silently failing the next apply.

- [ ] **Step 1: Write the failing test**

Cover, all through `pilot`: creating a tag with a name and a kind adds it to the bench's `extra_tags` **in the database**; Create with a name but **no kind** is refused with a visible error naming the missing kind, and writes nothing; no kind control is selected when the picker opens; a name that canonicalises to an existing slug relabels rather than duplicating; a name with no usable characters is refused with a visible error; the created tag is immediately selectable in this picker and, when applied, persists through `annotate_turn`; and the geometry assertion that `#evals-tag-new-create` stays hit-testable at both sizes with a full vocabulary above it.

The load-bearing one:

```python
@pytest.mark.asyncio
async def test_creating_a_tag_without_choosing_a_kind_is_refused_and_writes_nothing(
    evals_app, reviewable_run_group, evals_db, bench_id_of_that_run
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        await pilot.click("#evals-review-row-0")
        await pilot.click("#evals-cv-tag-0")
        await pilot.pause()
        await pilot.click("#evals-tag-new-name")
        await pilot.press(*"meta commentary")
        await pilot.click("#evals-tag-new-create")
        await pilot.pause()

        assert pilot.app.screen.query_one("#evals-tag-new-error").visible
        from tldw_chatbook.Evals.character_probe.storage import load_character_bench
        assert load_character_bench(evals_db, bench_id_of_that_run).extra_tags == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_tag_picker.py -k new -p no:randomly`
Expected: FAIL — `NoMatches: #evals-tag-new-name`

- [ ] **Step 3: Write minimal implementation**

The create form sits **below** the existing-tag list, so the spec's "offers existing tags before creating new ones" is true of the layout and not only of the code. One control per kind (`RadioSet` or three toggle Buttons — follow whatever this slice already uses for a small exclusive choice; grep `bench_editor.py` before inventing one), with **nothing selected initially**.

On Create: canonicalise the name; if it yields nothing, or no kind is selected, render the reason in `#evals-tag-new-error` and return without writing. Otherwise build the `Tag`, append it to the bench's `extra_tags` via `save_character_bench`, add it to the picker's live vocabulary, and post `TagCreated`.

`#evals-tag-new-error` is a `Static(markup=False)` carrying a user-typed name — same convention as Phase 2's `#evals-cb-form-error`, including that a failed create does **not** recompose, so the typed name survives.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_tag_picker.py Tests/Evals/character_probe -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Evals/tag_picker.py tldw_chatbook/UI/Evals/review_pane.py Tests/UI/test_evals_tag_picker.py
git commit -m "feat(evals): create a bench tag, kind required (task-1691 phase 3b)"
```

---

### Task 7: Keyboard-first review

**Files:**
- Modify: `tldw_chatbook/UI/Evals/review_pane.py`
- Modify: `tldw_chatbook/UI/Evals/conversation_view.py`
- Test: `Tests/UI/test_evals_review_keyboard.py`

**Interfaces:**
- Consumes: `ReviewPane`, `ConversationView`, `TagPicker` (Tasks 3-6).
- Produces: `ReviewPane.BINDINGS` — `j`/`k` next/previous conversation, `n`/`p` next/previous turn, `t` open the tag picker on the focused turn, `r` mark the current conversation reviewed, `u` toggle the unreviewed-only filter.

The spec: "Reviewing dozens of conversations by mouse in a terminal app is untenable; moving between turns and conversations and applying tags must be a few keystrokes."

**Every test here presses real keys through `pilot` and asserts the persisted result or the moved focus** — never calls the action method directly. An action method that works while its binding is unreachable is exactly the defect family this project has shipped four times.

Check for conflicts before choosing: `EvalsScreen` and the app shell already own bindings, and a single-letter binding that collides with an existing one, or that fires while an `Input` has focus, is worse than no binding. **Confirm the pane's bindings do not fire while the reviewer is typing in `#evals-tag-note`, `#evals-tag-search`, or `#evals-tag-new-name`** — that is a test, not an assumption.

- [ ] **Step 1: Write the failing test**

Cover: `j` moves the selection to the next conversation and the conversation view follows; `k` moves back; `j` at the end of the queue does not wrap or crash; `n`/`p` move the focused turn within a conversation and the focus is really on the turn block; `t` opens the tag picker for the focused turn (assert the picker is mounted and seeded with **that** turn's existing tags); `r` writes review state to the database and advances the progress line; `u` toggles the unreviewed-only filter and the row count changes; and — the one that catches a binding that silently steals keystrokes:

```python
@pytest.mark.asyncio
async def test_typing_a_note_does_not_trigger_the_review_bindings(
    evals_app, reviewable_run_group, evals_db
):
    """`r`, `j`, `t`, `u` are letters. A reviewer must be able to type them."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewable_run_group)
        await pilot.pause()
        await pilot.click("#evals-review-row-0")
        await pilot.click("#evals-cv-tag-0")
        await pilot.pause()
        await pilot.click("#evals-tag-note")
        await pilot.press(*"jerky turn, refused up front")
        await pilot.pause()

        note = pilot.app.screen.query_one("#evals-tag-note")
        assert note.value == "jerky turn, refused up front"
        from tldw_chatbook.Evals.character_probe.storage import load_review_state
        assert load_review_state(evals_db, reviewable_run_group) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_review_keyboard.py -p no:randomly`
Expected: FAIL — no bindings exist, so `j` moves nothing.

- [ ] **Step 3: Write minimal implementation**

Add `BINDINGS` to `ReviewPane` with `show=False` for the ones that would crowd the footer, and a short footer hint for `r` and `t` — the two a first-time reviewer will not guess. Actions operate on the pane's current selection and focused turn, both of which the pane already tracks from Tasks 4 and 5.

Textual routes a key to the focused widget first, so a binding on the pane will not fire while an `Input` inside it has focus — but **verify that against the real widget tree rather than trusting it**, because the picker may be mounted as a modal whose focus behaviour differs. If a collision does occur, gate the actions on `isinstance(self.app.focused, Input)` rather than removing the binding.

Document each binding's choice in a comment where it is defined: `j`/`k` and `n`/`p` are chosen for one-hand navigation without a modifier, `r` and `t` for their initials.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_review_keyboard.py Tests/UI/test_evals_review_pane.py Tests/UI/test_evals_tag_picker.py -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Evals/review_pane.py tldw_chatbook/UI/Evals/conversation_view.py Tests/UI/test_evals_review_keyboard.py
git commit -m "feat(evals): keyboard-first character-probe review (task-1691 phase 3b)"
```

---

## Phase 3b exit criteria

- Selecting a character-probe run group renders the review queue, not the placeholder and never the word-bench results grid.
- The queue is ordered with hinted conversations first, is filterable by card, probe, target, and not-yet-reviewed, and does not reshuffle when a conversation is reviewed.
- The conversation view shows the card's opening message and every turn, with empty and whitespace-only replies visible rather than blank.
- A tag can be applied to a specific turn by keystroke and is in the database afterwards.
- A tag can be created only with an explicitly chosen kind; a create without one is refused visibly and writes nothing.
- "Reviewed" can be recorded with zero annotations, the progress count reflects it, and it survives reopening the run group.
- Hints render as hints — never as tags, never as a number, never in the run path.
- Every new control is hit-testable at 160x45 and 235x52.
- `Tests/Evals` and the word-bench UI suites remain green and behaviourally untouched.

## Not in Phase 3b (deliberate)

The summary — per-tag counts across cards, probes, and targets — is Phase 4. Cross-run comparison is out of scope entirely (the spec: annotations are per run group, and non-determinism makes comparison unsound without seeds). A rich probe editor stays a follow-up. And no view here sums tags into a number.
