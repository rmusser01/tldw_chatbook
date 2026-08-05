# Character Probe Evals — Phase 4 (Summary) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer the question the eval was built for — which model broke character most, which card was hardest, which probe broke things regardless of model — as per-tag counts, and never as a score.

**Architecture:** Aggregation is a pure function over the annotations Phase 3b records; the UI is one more surface inside the review pane. Small by design: the hard constraint here is what the summary must *refuse* to do.

**Tech Stack:** Python ≥3.11, Textual, pytest. Engine from phases 1 and 3a; UI alongside Phase 3b's review pane.

## Global Constraints

Copied from `Docs/superpowers/specs/2026-08-01-character-probe-eval-design.md`. Every task's requirements implicitly include this section.

- **The summary reports per-tag counts and never a composite score.** The spec, verbatim: "Ranking models by 'fewest bad tags' would invent the objective metric this eval exists precisely because we lack — and would be wrong anyway, since `notable` and `positive` tags are not penalties. **No view anywhere sums tags into a number.**"
- **Every tag carries a kind** (`failure`, `notable`, `positive`), and the summary groups by kind so it cannot imply "fewer tags is better".
- **No logprobs / top-K / normalizer / canary vocabulary** anywhere in character-probe UI.
- **The three questions the summary answers**, from the spec: which model broke character most, which card was hardest, which probe broke things regardless of model. Those are the three groupings — by target, by card, by probe.
- User-authored text is a markup hazard: `markup=False` on any Static carrying a card name, probe text, or tag label; `escape_markup` in Button labels and tooltips.
- Fail loudly, never silently default.
- `character_ids` are ints; every eval id is a str. Do not normalise them.
- Tests must drive real widgets — press or type through `pilot`, assert what is in the database or on the screen, not what a helper returned.
- Painted geometry is the arbiter — new controls stay hit-testable at 160x45 AND 235x52.
- Google-style docstrings; CSS in `css/features/_evals.tcss` regenerated via `build_css.py`, never hand-edited.
- Run tests foreground: `/private/tmp/tldw-venv/bin/python -m pytest <paths> -p no:randomly`. Never `-q`. **Pass `timeout: 600000` on the Bash call.**

## What already exists (assumed from phases 3a and 3b — verify before use)

- `tags.py`: `Tag`, `TAG_KINDS`, `BUILTIN_TAGS`, `tag_by_slug`, `resolve_vocabulary`.
- `storage.run_group_vocabulary(db, run_group_id)`, `load_turn_annotations(db, run_group_id)` returning `dict[(card_id, probe_index, sample_index, target_id, turn_index), {"tags": [...], "note": str}]`, `load_review_state`, `load_probe_run_snapshot`.
- `review_queue.queue_progress(entries) -> (reviewed, total)`.
- `ReviewPane` (`UI/Evals/review_pane.py`) with its queue list and conversation side.

## File Structure

- `tldw_chatbook/Evals/character_probe/summary.py` (new) — pure aggregation over loaded annotations plus the run snapshot. No DB, no UI, no ranking.
- `tldw_chatbook/UI/Evals/summary_view.py` (new) — the three grouped tables and the coverage line.
- `tldw_chatbook/UI/Evals/review_pane.py` — a way to reach the summary from the queue.
- Tests mirror each under `Tests/Evals/character_probe/` and `Tests/UI/`.

---

### Task 1: Aggregating tag counts

**Files:**
- Create: `tldw_chatbook/Evals/character_probe/summary.py`
- Test: `Tests/Evals/character_probe/test_summary.py`

**Interfaces:**
- Consumes: `load_turn_annotations`' return shape; the run snapshot's `cards`, `probes`, `targets`; `Tag` and `tag_by_slug` (Phase 3a).
- Produces: `TagCount(slug: str, label: str, kind: str, count: int)`; `GroupedCounts(group_key: str | int, group_label: str, counts: tuple[TagCount, ...])`; `summarise(annotations, snapshot, vocabulary) -> dict[str, tuple[GroupedCounts, ...]]` keyed by exactly `"by_target"`, `"by_card"`, `"by_probe"`.

`count` is the number of **annotations carrying that tag** within the group — a count of observations, which is a fact. It is not a score, and nothing in this module combines counts across kinds. `summarise` returns counts grouped by kind-ordered tag; it must not return a total, a ratio, a rank, or a sort by "worst".

**The three groupings are the three questions**, no more: by target ("which model broke character most"), by card ("which card was hardest"), by probe ("which probe broke things regardless of model").

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.Evals.character_probe.summary import summarise
from tldw_chatbook.Evals.character_probe.tags import BUILTIN_TAGS

SNAPSHOT = {
    "cards": [{"id": 1, "name": "Vex"}, {"id": 2, "name": "Marlow"}],
    "probes": [{"turns": ["q1"]}, {"turns": ["q2"]}],
    "targets": [{"id": "t-1", "name": "llama-8b"}, {"id": "t-2", "name": "llama-70b"}],
}


def _annotations(*entries):
    """entries: (card_id, probe, sample, target, turn, [slugs])"""
    return {
        (c, p, s, t, turn): {"tags": list(slugs), "note": ""}
        for c, p, s, t, turn, slugs in entries
    }


def test_an_empty_run_summarises_to_empty_groups():
    result = summarise({}, SNAPSHOT, BUILTIN_TAGS)
    assert set(result) == {"by_target", "by_card", "by_probe"}
    assert all(
        all(not g.counts for g in groups) for groups in result.values()
    )


def test_a_tag_is_counted_under_its_target():
    annotations = _annotations((1, 0, 0, "t-1", 0, ["broke-character"]))
    groups = summarise(annotations, SNAPSHOT, BUILTIN_TAGS)["by_target"]
    by_key = {g.group_key: g for g in groups}
    counts = {c.slug: c.count for c in by_key["t-1"].counts}
    assert counts["broke-character"] == 1
    assert not by_key["t-2"].counts


def test_a_tag_is_counted_under_its_card():
    annotations = _annotations((2, 0, 0, "t-1", 0, ["refused"]))
    groups = summarise(annotations, SNAPSHOT, BUILTIN_TAGS)["by_card"]
    by_key = {g.group_key: g for g in groups}
    assert {c.slug: c.count for c in by_key[2].counts}["refused"] == 1


def test_a_tag_is_counted_under_its_probe():
    annotations = _annotations((1, 1, 0, "t-1", 0, ["refused"]))
    groups = summarise(annotations, SNAPSHOT, BUILTIN_TAGS)["by_probe"]
    by_key = {g.group_key: g for g in groups}
    assert {c.slug: c.count for c in by_key[1].counts}["refused"] == 1


def test_one_annotation_with_two_tags_counts_once_for_each():
    annotations = _annotations((1, 0, 0, "t-1", 0, ["refused", "broke-character"]))
    groups = {g.group_key: g for g in summarise(annotations, SNAPSHOT, BUILTIN_TAGS)["by_target"]}
    counts = {c.slug: c.count for c in groups["t-1"].counts}
    assert counts["refused"] == 1
    assert counts["broke-character"] == 1


def test_the_same_tag_on_two_turns_counts_twice():
    annotations = _annotations(
        (1, 0, 0, "t-1", 0, ["broke-character"]),
        (1, 0, 0, "t-1", 1, ["broke-character"]),
    )
    groups = {g.group_key: g for g in summarise(annotations, SNAPSHOT, BUILTIN_TAGS)["by_target"]}
    assert {c.slug: c.count for c in groups["t-1"].counts}["broke-character"] == 2


def test_a_group_carries_its_human_label():
    annotations = _annotations((1, 0, 0, "t-1", 0, ["refused"]))
    groups = {g.group_key: g for g in summarise(annotations, SNAPSHOT, BUILTIN_TAGS)["by_card"]}
    assert groups[1].group_label == "Vex"


def test_every_group_from_the_snapshot_appears_even_with_no_annotations():
    """A model nobody tagged is a result, not an absence."""
    annotations = _annotations((1, 0, 0, "t-1", 0, ["refused"]))
    groups = summarise(annotations, SNAPSHOT, BUILTIN_TAGS)["by_target"]
    assert {g.group_key for g in groups} == {"t-1", "t-2"}


def test_counts_are_ordered_by_tag_kind_not_by_magnitude():
    """Ordering by count would be a ranking; kind order is the spec's order."""
    annotations = _annotations(
        (1, 0, 0, "t-1", 0, ["in-character"]),
        (1, 0, 0, "t-1", 1, ["in-character"]),
        (1, 0, 0, "t-1", 2, ["broke-character"]),
    )
    groups = {g.group_key: g for g in summarise(annotations, SNAPSHOT, BUILTIN_TAGS)["by_target"]}
    kinds = [c.kind for c in groups["t-1"].counts]
    assert kinds == sorted(kinds, key=lambda k: ("failure", "notable", "positive").index(k))
    assert kinds[0] == "failure"


def test_a_tag_count_carries_its_kind_and_label():
    annotations = _annotations((1, 0, 0, "t-1", 0, ["broke-character"]))
    groups = {g.group_key: g for g in summarise(annotations, SNAPSHOT, BUILTIN_TAGS)["by_target"]}
    count = groups["t-1"].counts[0]
    assert count.kind == "failure"
    assert count.label == "Broke character"


def test_a_tag_outside_the_vocabulary_raises_naming_it():
    """Phase 3a rejects these at write time; a corrupt row must not be silent."""
    annotations = _annotations((1, 0, 0, "t-1", 0, ["no-such-tag"]))
    with pytest.raises(KeyError) as exc:
        summarise(annotations, SNAPSHOT, BUILTIN_TAGS)
    assert "no-such-tag" in str(exc.value)


def test_an_annotation_with_only_a_note_contributes_no_counts():
    annotations = _annotations((1, 0, 0, "t-1", 0, []))
    groups = {g.group_key: g for g in summarise(annotations, SNAPSHOT, BUILTIN_TAGS)["by_target"]}
    assert not groups["t-1"].counts


def test_the_summary_exposes_no_total_ratio_or_rank():
    """The one constraint this whole phase exists to hold."""
    import dataclasses
    from tldw_chatbook.Evals.character_probe.summary import GroupedCounts, TagCount

    assert {f.name for f in dataclasses.fields(TagCount)} == {
        "slug", "label", "kind", "count",
    }
    assert {f.name for f in dataclasses.fields(GroupedCounts)} == {
        "group_key", "group_label", "counts",
    }
    import tldw_chatbook.Evals.character_probe.summary as mod
    exported = {n for n in dir(mod) if not n.startswith("_")}
    for forbidden in ("rank", "score", "total", "ratio", "best", "worst"):
        assert not any(forbidden in name.lower() for name in exported)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_summary.py -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named '...summary'`

- [ ] **Step 3: Write minimal implementation**

Two frozen dataclasses and one function, with the module docstring carrying the constraint so nobody adds a total later:

```python
"""Per-tag counts across a character-probe run.

This module deliberately produces NO composite score, total, ratio, or rank.
The design spec: "Ranking models by 'fewest bad tags' would invent the
objective metric this eval exists precisely because we lack -- and would be
wrong anyway, since `notable` and `positive` tags are not penalties. No view
anywhere sums tags into a number."

A count of how many times a tag was applied within a group is a fact about
what a reader observed. The moment counts of different kinds are combined,
it stops being a fact and becomes the metric this eval refuses to invent.
``TagCount`` and ``GroupedCounts`` therefore carry no field a caller could
sort by across kinds, and this module exports no such helper.
"""
```

`summarise` walks the annotations once, resolving each slug through `tag_by_slug` (which raises naming the slug for anything outside the vocabulary — that is the fail-loudly path the test pins), and accumulates into three `dict[group_key, Counter]`. Then it materialises every group present in the snapshot — including groups with no annotations, so "nobody tagged this model" reads as a result rather than a missing row — ordering each group's counts by `TAG_KINDS.index(kind)` then by slug, never by count.

Read groups from the snapshot: targets from `snapshot["targets"]` (`id`, `name`), cards from `snapshot["cards"]` (`id`, `name`), probes from `snapshot["probes"]` by index with a label like `f"Probe {index + 1}"`. Guard each read with `or []`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/summary.py Tests/Evals/character_probe/test_summary.py
git commit -m "feat(evals): per-tag count aggregation for a character-probe run (task-1691 phase 4)"
```

---

### Task 2: The summary view

**Files:**
- Create: `tldw_chatbook/UI/Evals/summary_view.py`
- Modify: `tldw_chatbook/UI/Evals/review_pane.py`
- Modify: `tldw_chatbook/css/features/_evals.tcss` (+ regenerate the bundle)
- Test: `Tests/UI/test_evals_summary_view.py`

**Interfaces:**
- Consumes: `summarise`, `GroupedCounts`, `TagCount` (Task 1); `queue_progress` (Phase 3b); `TAG_KINDS`.
- Produces: `SummaryView(grouped, coverage, id="evals-summary-view")` with `#evals-summary-by-target`, `#evals-summary-by-card`, `#evals-summary-by-probe`, and `#evals-summary-coverage`; `#evals-review-summary` in `ReviewPane` to reach it, and `#evals-review-back-to-queue` to return.

**The coverage line is what keeps the summary honest.** It states how much of the run has actually been reviewed — `queue_progress`' two integers — because counts over a half-read run mean something different from counts over a finished one, and a reader who cannot see that will over-read the table. It is not a score and is never rendered as a percentage.

Each grouping renders as a small table: one row per group, one column per tag that has a non-zero count anywhere in that grouping, with the tag's **kind shown in the header**. A group with no counts renders its row with zeros rather than being dropped.

- [ ] **Step 1: Write the failing test**

Cover, driving through `pilot`: the three groupings each render with a row per group; a group with no annotations still renders a row; the tag columns show each tag's kind; a card name containing markup renders literally; a bench-relabelled tag shows its label rather than its slug; the coverage line shows reviewed-of-total as two integers; the summary is reachable from the queue by clicking `#evals-review-summary` and returns by `#evals-review-back-to-queue`; no logprob/top-K/canary/normalizer vocabulary appears; and the geometry assertion at both sizes.

Plus the two that hold the phase's one hard line:

```python
@pytest.mark.asyncio
async def test_the_summary_shows_no_percentage_and_no_composite_number(
    evals_app, reviewed_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=reviewed_run_group)
        await pilot.pause()
        await pilot.click("#evals-review-summary")
        await pilot.pause()
        text = pilot.app.screen.query_one("#evals-summary-view").render_str_or_text()
        assert "%" not in text
        for forbidden in ("score", "rank", "overall", "total"):
            assert forbidden not in text.lower()


@pytest.mark.asyncio
async def test_groups_are_not_ordered_by_how_many_failure_tags_they_carry(
    evals_app, run_group_with_lopsided_tags
):
    """Sorting by failure count IS a ranking, however it is rendered."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="run_group", id=run_group_with_lopsided_tags)
        await pilot.pause()
        await pilot.click("#evals-review-summary")
        await pilot.pause()
        rows = pilot.app.screen.query(".evals-summary-target-row")
        labels = [r.render_str_or_text() for r in rows]
        # The heavily-tagged target is t-2; a ranking would float it first.
        assert "llama-8b" in labels[0]
```

`render_str_or_text()` above stands in for however this repo's tests actually read a widget's painted text — **grep `Tests/UI/test_evals_character_bench_editor.py` for the real helper and use it verbatim**, as Phase 3b's tasks do. Do not add a new one.

Build `reviewed_run_group` and `run_group_with_lopsided_tags` on Phase 3b's review fixtures — import them rather than constructing run groups again, and keep that lineage's rule: target rows built the way the real "+ New target" form writes them, never hand-built dicts.

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_summary_view.py -p no:randomly`
Expected: FAIL — `NoMatches: #evals-review-summary`

- [ ] **Step 3: Write minimal implementation**

`SummaryView` is a `Vertical` yielding the coverage line first — it frames everything below it — then the three grouped tables under plain headings naming the question each answers ("By model", "By character", "By probe"). Every cell carrying a name or label is a `Static(markup=False)`; tag labels in headers go through `escape_markup` if they land in a `Button`.

In `ReviewPane`, add `#evals-review-summary` beside the progress line and swap the right-hand side between the conversation view and the summary, with `#evals-review-back-to-queue` to return. Recompute `summarise` when entering the summary, not on every annotation — the counts are a read of the database, and a reviewer tagging turns does not need the table rebuilt under them mid-keystroke.

Order the rows by the snapshot's own order (targets as configured, cards as configured, probes by index). That is deliberate and is what `test_groups_are_not_ordered_by_how_many_failure_tags_they_carry` pins: any count-derived ordering is a ranking wearing a different hat.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_summary_view.py Tests/UI/test_evals_review_pane.py Tests/UI/test_evals_review_keyboard.py -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Evals/summary_view.py tldw_chatbook/UI/Evals/review_pane.py tldw_chatbook/css Tests/UI/test_evals_summary_view.py
git commit -m "feat(evals): the character-probe run summary (task-1691 phase 4)"
```

---

### Task 3: The live pass

**Files:**
- Modify: `Docs/superpowers/specs/2026-08-01-character-probe-eval-design.md` (record the result)
- Test: manual, recorded

**Interfaces:**
- Consumes: everything phases 1-4 built.
- Produces: a recorded live-verification note.

The spec closes with a requirement no automated test satisfies: "**A live pass against a real llama.cpp instance with real character cards is required before this is called done.**" This project has shipped four features whose tests were green and which no user could operate — a dead Run button, an untoggleable checkbox, steering that never reached the model, and a target resolution that silently reused a steered row. Every one was found by driving the real app, not by a suite.

- [ ] **Step 1: Prepare the run**

Start the llama.cpp instance the earlier phases used (`127.0.0.1:9099`), and make sure the profile has at least two real character cards with distinct voices and a probe file with at least one multi-turn probe and one single-turn probe.

- [ ] **Step 2: Drive the whole journey by hand**

In a real terminal at 160x45, then again at 235x52: import the probe set, create a character bench, pick both cards, save, check the estimate matches cards × probes × targets × samples × turns, run it, and watch the run reach a terminal state.

- [ ] **Step 3: Review by keyboard only**

With hands off the mouse: move through the queue with `j`/`k`, move through turns with `n`/`p`, apply a tag with `t`, mark a conversation reviewed with `r`, and filter to unreviewed with `u`. Type a note containing the letters `j`, `r`, `t`, and `u` and confirm none of them triggers a binding.

- [ ] **Step 4: Confirm what the tests cannot**

Check by eye: replies render readably at both sizes; an empty reply is visibly empty rather than a blank gap; hints read as hints and not as verdicts; the summary's coverage line matches what you actually reviewed; and closing and reopening the app preserves every annotation and the reviewed count.

- [ ] **Step 5: Record the result and commit**

Append a short "Live verification" section to the design spec: the date, the llama.cpp model, the terminal sizes, what worked, and every defect found. **A defect found here is a task, not a footnote** — file it before closing the phase.

```bash
git add Docs/superpowers/specs/2026-08-01-character-probe-eval-design.md
git commit -m "docs(evals): record the character-probe live verification pass (task-1691 phase 4)"
```

---

## Phase 4 exit criteria

- The summary answers the three questions the spec names, grouped by target, card, and probe.
- Counts are grouped by tag kind, and no view anywhere sums them into a number, a ratio, a rank, or a percentage.
- Group ordering comes from the run's own configuration, never from counts.
- The coverage line states how much of the run has actually been reviewed, as two integers.
- A tag outside the run's vocabulary raises naming it rather than being silently dropped.
- The live pass is done, recorded in the spec, and every defect it found is filed.

## Not in Phase 4 (deliberate)

Cross-run comparison stays out of scope — annotations are per run group, and non-determinism makes comparison unsound without seeds. Export of any kind is a follow-up. And there is still no composite score.
