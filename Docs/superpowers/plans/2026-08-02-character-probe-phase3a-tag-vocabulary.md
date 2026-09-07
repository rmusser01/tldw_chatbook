# Character Probe Evals — Phase 3a (Tag Vocabulary) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the character-probe eval a real tag vocabulary — built-in defaults, kinds that stop the summary implying "fewer tags is better", per-bench extension that cannot omit a kind — and make deleting a bench take its annotations with it.

**Architecture:** Engine-only, no UI. A new `character_probe/tags.py` owns the vocabulary; `CharacterProbeConfig.extra_tags` stops being an untyped `tuple[dict, ...]` placeholder and becomes validated `Tag` objects; `annotate_turn` validates against **the vocabulary the run captured**, not today's. Phase 3b consumes all of it.

**Tech Stack:** Python ≥3.11, pytest. No Textual in this phase.

## Global Constraints

Copied from `Docs/superpowers/specs/2026-08-01-character-probe-eval-design.md` and phases 1-2's established conventions. Every task's requirements implicitly include this section.

- **Every tag carries a kind** — `failure`, `notable`, or `positive` — "so the summary cannot imply 'fewer tags is better'".
- **Creating a tag requires choosing its kind; there is no default.** The spec is explicit that `notable` is *not* a safe fallback: "it would make genuine failures invisible in exactly the view meant to surface them." A missing kind is an error, never a guess.
- **Tags are stored as canonical slugs.** The UI offering existing tags before new ones is Phase 3b's job; this phase makes the canonical form unambiguous so that offer is possible.
- **Annotations attach to a specific run's answers.** "A re-run produces new answers to annotate rather than silently inheriting old judgments. Deleting a run group cascades both, since they describe those answers and mean nothing without them."
- **No logprobs / top-K / normalizer / canary vocabulary** anywhere in character-probe code. That vocabulary judges distributions; this eval reads generated text.
- **No composite score, anywhere.** "No view anywhere sums tags into a number." This phase must not add a helper that would make one easy.
- **Fail loudly, never silently default** — a corrupt row or missing record raises a named error identifying it; a write affecting no rows raises rather than reporting success.
- **`character_ids` are ints** (`character_cards.id`); every eval id is a str. Do not normalise them.
- **Back-compatibility is required.** `extra_tags` already round-trips as raw dicts through `save_character_bench` (storage.py:279) and `load_character_bench` (storage.py:480), and is already embedded in every existing run snapshot (storage.py:556). A bench or snapshot written before this phase must still load.
- Google-style docstrings (Args/Returns/Raises) on public callables; parameterized SQL only.
- Run tests foreground: `/private/tmp/tldw-venv/bin/python -m pytest <paths> -p no:randomly` from the clone root. Never `-q`. **Pass `timeout: 600000` on the Bash call** — the harness auto-backgrounds anything past 120s and a backgrounded pytest has stalled this workflow.

## What already exists (verified against the merged code — do not rebuild it)

- `eval_probe_turn_annotations` and `eval_probe_review_state` tables (`DB/Evals_DB.py:316,333`), with `tags` as a JSON list of slugs.
- `EvalsDB.upsert_probe_turn_annotation` / `list_probe_turn_annotations` / `upsert_probe_review_state` / `list_probe_review_state` (`Evals_DB.py:1816-1935`).
- `character_probe/storage.py`: `annotate_turn` (:921), `load_turn_annotations` (:964), `mark_conversation_reviewed` (:990), `load_review_state` (:1028).
- `CharacterProbeConfig.extra_tags: tuple[dict, ...] = ()` (`models.py:119`) — an untyped placeholder with **no validation and no built-in defaults anywhere in the repo**.
- `_probe_run_snapshot` already writes `"extra_tags": list(config.extra_tags)` (`storage.py:556`), and `load_probe_run_snapshot` (:644) reads the snapshot back. **This is the seam Task 4 uses** — it means a run's vocabulary is already captured, matching the spec's snapshot-provenance rule.

## File Structure

- `tldw_chatbook/Evals/character_probe/tags.py` (new) — the `Tag` model, `TAG_KINDS`, `BUILTIN_TAGS`, `canonical_slug`, `resolve_vocabulary`, `tag_by_slug`. Its own file because Phase 3b's UI, Phase 4's summary, and `storage.py` all import it, and it must pull no DB and no UI.
- `tldw_chatbook/Evals/character_probe/models.py` — `extra_tags` becomes `tuple[Tag, ...]`, validated at construction.
- `tldw_chatbook/Evals/character_probe/storage.py` — serialise/deserialise `Tag`; add `run_group_vocabulary`; validate in `annotate_turn`; cascade on delete.
- `tldw_chatbook/DB/Evals_DB.py` — the cascade delete.
- Tests mirror each under `Tests/Evals/character_probe/`.

---

### Task 1: The tag model and the built-in vocabulary

**Files:**
- Create: `tldw_chatbook/Evals/character_probe/tags.py`
- Test: `Tests/Evals/character_probe/test_tags.py`

**Interfaces:**
- Consumes: nothing (stdlib only — this module must stay importable without a DB).
- Produces: `TAG_KINDS: tuple[str, str, str]`; `Tag(slug: str, label: str, kind: str)` frozen dataclass validating both fields at construction; `BUILTIN_TAGS: tuple[Tag, ...]` — exactly the ten tags below.

The ten built-ins are an owner decision recorded in the spec's Tags section. Copy the slugs, labels, and kinds **verbatim** — a reviewer's muscle memory and Phase 4's grouping both depend on them being stable.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.Evals.character_probe.tags import (
    BUILTIN_TAGS,
    TAG_KINDS,
    Tag,
)


def test_the_three_kinds_are_exactly_the_specs_three():
    assert TAG_KINDS == ("failure", "notable", "positive")


def test_the_builtin_vocabulary_is_the_ten_the_spec_names():
    assert {t.slug for t in BUILTIN_TAGS} == {
        "broke-character",
        "refused",
        "leaked-prompt",
        "generic-assistant-voice",
        "contradicted-card",
        "ignored-the-question",
        "notable",
        "surprising",
        "in-character",
        "handled-well",
    }


def test_each_builtin_carries_the_kind_the_spec_assigns_it():
    by_slug = {t.slug: t.kind for t in BUILTIN_TAGS}
    assert by_slug["broke-character"] == "failure"
    assert by_slug["refused"] == "failure"
    assert by_slug["leaked-prompt"] == "failure"
    assert by_slug["generic-assistant-voice"] == "failure"
    assert by_slug["contradicted-card"] == "failure"
    assert by_slug["ignored-the-question"] == "failure"
    assert by_slug["notable"] == "notable"
    assert by_slug["surprising"] == "notable"
    assert by_slug["in-character"] == "positive"
    assert by_slug["handled-well"] == "positive"


def test_builtin_slugs_are_unique():
    slugs = [t.slug for t in BUILTIN_TAGS]
    assert len(slugs) == len(set(slugs))


def test_a_tag_without_a_valid_kind_is_rejected_naming_the_kind():
    with pytest.raises(ValueError) as exc:
        Tag(slug="whatever", label="Whatever", kind="bad")
    assert "bad" in str(exc.value)
    assert "failure" in str(exc.value)


def test_a_tag_with_an_empty_kind_is_rejected_rather_than_defaulted():
    """The spec: a guessed kind mis-groups observations; `notable` is not safe."""
    with pytest.raises(ValueError):
        Tag(slug="whatever", label="Whatever", kind="")


def test_a_tag_with_a_non_canonical_slug_is_rejected_naming_the_slug():
    with pytest.raises(ValueError) as exc:
        Tag(slug="Broke Character", label="Broke character", kind="failure")
    assert "Broke Character" in str(exc.value)


def test_a_tag_with_an_empty_label_is_rejected():
    with pytest.raises(ValueError):
        Tag(slug="broke-character", label="", kind="failure")


def test_a_tag_is_frozen():
    tag = BUILTIN_TAGS[0]
    with pytest.raises(Exception):
        tag.slug = "mutated"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_tags.py -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Evals.character_probe.tags'`

- [ ] **Step 3: Write minimal implementation**

`tldw_chatbook/Evals/character_probe/tags.py`:

```python
"""The character-probe review vocabulary.

Every tag carries a kind so no view can imply "fewer tags is better": a
``positive`` tag and a ``failure`` tag are both observations, and the summary
groups by kind rather than counting them together. Creating a tag therefore
REQUIRES a kind -- the design spec rules out guessing one, and rules out
``notable`` as a fallback specifically because it would hide genuine failures
in the view meant to surface them.

Stdlib only, deliberately: this module is imported by the engine, the review
UI, and the summary, and must never drag a database or Textual behind it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: The only kinds a tag may carry. Order is display order, worst first.
TAG_KINDS: tuple[str, str, str] = ("failure", "notable", "positive")

#: A canonical slug: lowercase, digits, and single interior hyphens.
_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Tag:
    """One review tag.

    Args:
        slug: The canonical stored form (see ``canonical_slug``).
        label: What a reviewer reads. Never empty.
        kind: One of ``TAG_KINDS``.

    Raises:
        ValueError: If the slug is not canonical, the label is blank, or the
            kind is not one of ``TAG_KINDS`` -- naming the offending value.
    """

    slug: str
    label: str
    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.slug, str) or not _SLUG_RE.match(self.slug):
            raise ValueError(
                f"Tag slug must be lowercase-hyphenated, got {self.slug!r}."
            )
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError(f"Tag {self.slug!r} needs a non-empty label.")
        if self.kind not in TAG_KINDS:
            raise ValueError(
                f"Tag {self.slug!r} has kind {self.kind!r}; must be one of "
                f"{', '.join(TAG_KINDS)}. A kind is never guessed -- the wrong "
                f"one mis-groups the observation in the summary."
            )


#: The vocabulary every bench starts with (spec, Tags section). A bench
#: extends this through ``CharacterProbeConfig.extra_tags``; it never
#: replaces it.
BUILTIN_TAGS: tuple[Tag, ...] = (
    Tag("broke-character", "Broke character", "failure"),
    Tag("refused", "Refused", "failure"),
    Tag("leaked-prompt", "Leaked the card's prompt", "failure"),
    Tag("generic-assistant-voice", "Generic assistant voice", "failure"),
    Tag("contradicted-card", "Contradicted the card", "failure"),
    Tag("ignored-the-question", "Ignored the question", "failure"),
    Tag("notable", "Notable", "notable"),
    Tag("surprising", "Surprising", "notable"),
    Tag("in-character", "In character", "positive"),
    Tag("handled-well", "Handled well", "positive"),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_tags.py -p no:randomly`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/tags.py Tests/Evals/character_probe/test_tags.py
git commit -m "feat(evals): the character-probe tag vocabulary and its kinds (task-1691 phase 3a)"
```

---

### Task 2: Canonical slugs and per-bench extension

**Files:**
- Modify: `tldw_chatbook/Evals/character_probe/tags.py`
- Test: `Tests/Evals/character_probe/test_tags.py`

**Interfaces:**
- Consumes: `Tag`, `BUILTIN_TAGS`, `TAG_KINDS` (Task 1).
- Produces: `canonical_slug(text: str) -> str`; `resolve_vocabulary(extra_tags: Sequence[Mapping[str, Any]] | Sequence[Tag] = ()) -> tuple[Tag, ...]`; `tag_by_slug(vocabulary: Sequence[Tag], slug: str) -> Tag` raising `KeyError` naming the slug and listing the vocabulary.

`canonical_slug` is what limits the `broke-character` / `OOC` / `out-of-character` fragmentation the spec warns about: the UI will offer existing tags first, but only a single canonical form makes "existing" a decidable question.

An extra tag whose slug matches a built-in **overrides** it — that is how a bench relabels `notable` to something domain-specific without forking the kind system. An override may not change the kind of a built-in: that would make two benches' `failure` counts mean different things in the same summary.

- [ ] **Step 1: Write the failing test**

```python
from tldw_chatbook.Evals.character_probe.tags import (
    BUILTIN_TAGS,
    Tag,
    canonical_slug,
    resolve_vocabulary,
    tag_by_slug,
)


def test_canonical_slug_lowercases_and_hyphenates():
    assert canonical_slug("Broke Character") == "broke-character"
    assert canonical_slug("  Out Of Character  ") == "out-of-character"
    assert canonical_slug("OOC") == "ooc"


def test_canonical_slug_collapses_runs_and_strips_punctuation():
    assert canonical_slug("broke   character!!") == "broke-character"
    assert canonical_slug("re-broke  --  character") == "re-broke-character"


def test_canonical_slug_rejects_text_with_no_usable_characters():
    import pytest
    with pytest.raises(ValueError):
        canonical_slug("   !!!   ")


def test_resolve_vocabulary_with_no_extras_is_exactly_the_builtins():
    assert resolve_vocabulary(()) == BUILTIN_TAGS


def test_resolve_vocabulary_appends_an_extra_tag():
    vocab = resolve_vocabulary(
        [{"slug": "meta-commentary", "label": "Meta commentary", "kind": "failure"}]
    )
    assert len(vocab) == len(BUILTIN_TAGS) + 1
    assert vocab[-1] == Tag("meta-commentary", "Meta commentary", "failure")


def test_an_extra_tag_may_relabel_a_builtin_in_place():
    vocab = resolve_vocabulary(
        [{"slug": "notable", "label": "Worth a second look", "kind": "notable"}]
    )
    assert len(vocab) == len(BUILTIN_TAGS)
    assert tag_by_slug(vocab, "notable").label == "Worth a second look"


def test_an_extra_tag_may_not_change_a_builtins_kind():
    import pytest
    with pytest.raises(ValueError) as exc:
        resolve_vocabulary(
            [{"slug": "refused", "label": "Refused", "kind": "positive"}]
        )
    assert "refused" in str(exc.value)


def test_an_extra_tag_without_a_kind_is_rejected_naming_the_slug():
    import pytest
    with pytest.raises(ValueError) as exc:
        resolve_vocabulary([{"slug": "meta-commentary", "label": "Meta"}])
    assert "meta-commentary" in str(exc.value)


def test_an_extra_tags_slug_is_canonicalised_rather_than_rejected():
    """A bench author types a label; the stored slug is canonical."""
    vocab = resolve_vocabulary(
        [{"slug": "Meta Commentary", "label": "Meta commentary", "kind": "notable"}]
    )
    assert tag_by_slug(vocab, "meta-commentary").label == "Meta commentary"


def test_an_extra_tag_missing_a_label_falls_back_to_its_slug():
    vocab = resolve_vocabulary([{"slug": "meta-commentary", "kind": "notable"}])
    assert tag_by_slug(vocab, "meta-commentary").label == "meta-commentary"


def test_resolve_vocabulary_accepts_tag_objects_as_well_as_mappings():
    vocab = resolve_vocabulary([Tag("meta-commentary", "Meta", "notable")])
    assert tag_by_slug(vocab, "meta-commentary").kind == "notable"


def test_two_extras_with_the_same_slug_keep_the_last():
    vocab = resolve_vocabulary([
        {"slug": "meta", "label": "First", "kind": "notable"},
        {"slug": "meta", "label": "Second", "kind": "notable"},
    ])
    assert tag_by_slug(vocab, "meta").label == "Second"


def test_tag_by_slug_raises_naming_the_slug_and_the_vocabulary():
    import pytest
    with pytest.raises(KeyError) as exc:
        tag_by_slug(BUILTIN_TAGS, "no-such-tag")
    assert "no-such-tag" in str(exc.value)
    assert "broke-character" in str(exc.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_tags.py -p no:randomly`
Expected: FAIL — `ImportError: cannot import name 'canonical_slug'`

- [ ] **Step 3: Write minimal implementation**

Append to `tags.py`:

```python
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def canonical_slug(text: str) -> str:
    """The single stored form of a tag name.

    One canonical form is what makes "does this tag already exist?" a
    decidable question, which is what limits the ``broke-character`` /
    ``OOC`` / ``out-of-character`` fragmentation per-bench extension invites.

    Args:
        text: A human-typed tag name.

    Returns:
        str: Lowercase, with every run of non-alphanumerics collapsed to a
        single hyphen and leading/trailing hyphens removed.

    Raises:
        ValueError: If nothing usable survives -- an empty slug would collide
            with every other empty slug and silently merge unrelated tags.
    """
    slug = _NON_SLUG_RE.sub("-", str(text).strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"{text!r} has no characters usable in a tag slug.")
    return slug


def _coerce_tag(raw: Any) -> Tag:
    """One extra-tag entry as a validated ``Tag``.

    Args:
        raw: A ``Tag``, or a mapping with ``slug`` and ``kind`` and an
            optional ``label``.

    Returns:
        Tag: The validated tag, its slug canonicalised.

    Raises:
        ValueError: If the entry is not a mapping, has no slug, or omits the
            kind -- naming the slug, since a guessed kind mis-groups the
            observation in the summary.
    """
    if isinstance(raw, Tag):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError(f"An extra tag must be a mapping or Tag, got {raw!r}.")
    raw_slug = raw.get("slug")
    if not raw_slug:
        raise ValueError(f"An extra tag needs a slug: {dict(raw)!r}.")
    slug = canonical_slug(str(raw_slug))
    kind = raw.get("kind")
    if not kind:
        raise ValueError(
            f"Extra tag {slug!r} has no kind. Every tag states one of "
            f"{', '.join(TAG_KINDS)} -- it is never guessed."
        )
    label = str(raw.get("label") or slug)
    return Tag(slug=slug, label=label, kind=str(kind))


def resolve_vocabulary(extra_tags: Sequence[Any] = ()) -> tuple[Tag, ...]:
    """The full tag vocabulary for one bench: built-ins plus its extras.

    An extra whose slug matches a built-in relabels it in place rather than
    appending a duplicate. It may NOT change a built-in's kind: two benches
    whose ``failure`` sets mean different things cannot be read in one
    summary.

    Args:
        extra_tags: The bench's ``extra_tags``, as ``Tag`` objects or as the
            raw mappings older rows and run snapshots store.

    Returns:
        tuple[Tag, ...]: Built-ins in their declared order, with overrides
        applied in place, then each new extra in the order supplied.

    Raises:
        ValueError: If an extra is malformed, omits its kind, or tries to
            change a built-in's kind -- naming the slug in every case.
    """
    builtin_kinds = {tag.slug: tag.kind for tag in BUILTIN_TAGS}
    resolved: dict[str, Tag] = {tag.slug: tag for tag in BUILTIN_TAGS}
    for raw in extra_tags or ():
        tag = _coerce_tag(raw)
        builtin_kind = builtin_kinds.get(tag.slug)
        if builtin_kind is not None and tag.kind != builtin_kind:
            raise ValueError(
                f"Extra tag {tag.slug!r} would change the built-in kind "
                f"{builtin_kind!r} to {tag.kind!r}. Built-in kinds are fixed "
                f"so one summary can read every bench."
            )
        resolved[tag.slug] = tag
    return tuple(resolved.values())


def tag_by_slug(vocabulary: Sequence[Tag], slug: str) -> Tag:
    """One tag from a vocabulary.

    Args:
        vocabulary: The bench's resolved vocabulary.
        slug: The canonical slug to find.

    Returns:
        Tag: The matching tag.

    Raises:
        KeyError: If no tag matches -- naming the slug and the vocabulary, so
            a stored annotation referencing a retired tag says which one.
    """
    for tag in vocabulary:
        if tag.slug == slug:
            return tag
    known = ", ".join(t.slug for t in vocabulary)
    raise KeyError(f"No tag {slug!r} in this bench's vocabulary ({known}).")
```

Add `from typing import Any, Mapping, Sequence` to the module's imports.

Note that `resolved` is a dict keyed by slug and built from `BUILTIN_TAGS` first, so Python's insertion order gives built-ins-then-extras for free and an override lands in the built-in's original position — which is what `test_an_extra_tag_may_relabel_a_builtin_in_place` pins.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_tags.py -p no:randomly`
Expected: PASS (22 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/tags.py Tests/Evals/character_probe/test_tags.py
git commit -m "feat(evals): canonical tag slugs and per-bench vocabulary extension (task-1691 phase 3a)"
```

---

### Task 3: `extra_tags` becomes validated, and keeps loading old rows

**Files:**
- Modify: `tldw_chatbook/Evals/character_probe/models.py`
- Modify: `tldw_chatbook/Evals/character_probe/storage.py`
- Test: `Tests/Evals/character_probe/test_bench_storage.py`

**Interfaces:**
- Consumes: `resolve_vocabulary`, `Tag` (Task 2).
- Produces: `CharacterProbeConfig.extra_tags: tuple[Tag, ...]`, validated in `__post_init__` regardless of the `strict` flag; `storage._tags_to_json(tags)` / `storage._tags_from_json(raw)` used by both `save_character_bench` and `_probe_run_snapshot`.

`extra_tags` currently accepts anything and stores it verbatim (`models.py:119`, `storage.py:279/480/556`). Validation belongs at construction so a malformed tag cannot reach the database at all.

**`strict` does NOT gate this.** `strict=False` exists so a *draft* bench with no characters and no targets can be created and reloaded (Phase 2, task 5). A malformed tag is not a draft state — it is corrupt data, and the spec's "fail loudly" rule applies to it in every mode.

**Back-compat matters here.** Existing `eval_tasks` rows and every existing run snapshot hold `extra_tags` as raw dicts. `_tags_from_json` must accept those. A row whose tags are corrupt raises naming the bench, per fail-loudly — it does not silently drop them.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.Evals.character_probe.models import CharacterProbeConfig
from tldw_chatbook.Evals.character_probe.storage import (
    load_character_bench,
    save_character_bench,
)
from tldw_chatbook.Evals.character_probe.tags import Tag


def _config(**kwargs):
    base = dict(
        name="villain probes",
        probe_set_id="ps-1",
        character_ids=(1,),
        target_ids=("t-1",),
    )
    base.update(kwargs)
    return CharacterProbeConfig(**base)


def test_extra_tags_round_trip_as_tag_objects(db):
    bench_id = save_character_bench(
        db,
        _config(extra_tags=(Tag("meta-commentary", "Meta commentary", "failure"),)),
    )
    loaded = load_character_bench(db, bench_id)
    assert loaded.extra_tags == (Tag("meta-commentary", "Meta commentary", "failure"),)


def test_extra_tags_supplied_as_mappings_are_validated_and_coerced(db):
    bench_id = save_character_bench(
        db,
        _config(extra_tags=({"slug": "Meta Commentary", "kind": "notable"},)),
    )
    loaded = load_character_bench(db, bench_id)
    assert loaded.extra_tags[0].slug == "meta-commentary"
    assert loaded.extra_tags[0].kind == "notable"


def test_an_extra_tag_without_a_kind_is_rejected_at_construction():
    with pytest.raises(ValueError) as exc:
        _config(extra_tags=({"slug": "meta-commentary"},))
    assert "meta-commentary" in str(exc.value)


def test_a_malformed_extra_tag_is_rejected_even_when_not_strict():
    """strict=False is for DRAFT benches, not for corrupt tags."""
    with pytest.raises(ValueError):
        CharacterProbeConfig(
            name="draft",
            probe_set_id="ps-1",
            character_ids=(),
            target_ids=(),
            extra_tags=({"slug": "meta", "kind": "not-a-kind"},),
            strict=False,
        )


def test_a_bench_row_written_before_this_phase_still_loads(db):
    """extra_tags shipped as raw dicts; existing rows must not break."""
    bench_id = save_character_bench(db, _config())
    row = db.get_task(bench_id)
    config_data = dict(row["config_data"])
    config_data["extra_tags"] = [
        {"slug": "meta-commentary", "label": "Meta commentary", "kind": "failure"}
    ]
    db.update_task(bench_id, {"config_data": config_data})

    loaded = load_character_bench(db, bench_id)
    assert loaded.extra_tags == (Tag("meta-commentary", "Meta commentary", "failure"),)


def test_a_bench_row_with_corrupt_tags_raises_naming_the_bench(db):
    bench_id = save_character_bench(db, _config())
    row = db.get_task(bench_id)
    config_data = dict(row["config_data"])
    config_data["extra_tags"] = ["not-a-mapping"]
    db.update_task(bench_id, {"config_data": config_data})

    with pytest.raises(ValueError) as exc:
        load_character_bench(db, bench_id)
    assert bench_id in str(exc.value)


def test_a_bench_with_no_extra_tags_still_loads_as_empty(db):
    bench_id = save_character_bench(db, _config())
    assert load_character_bench(db, bench_id).extra_tags == ()
```

Use the file's existing fixtures and helpers rather than inventing new ones. Verified for you against the real file:

- The database fixture is **`db`** (`test_bench_storage.py:14`), not `evals_db`.
- There is already a **`config` fixture** (`:19`) returning a ready `CharacterProbeConfig`. Prefer it over the `_config()` helper sketched above, passing only the field under test.
- **`_corrupt_config_field(db, task_id, key, raw_json_value)`** (`:172`) exists for exactly the "write a bad value into a stored row" job. Use it for the back-compat and corrupt-row tests instead of hand-rolling row mutation — `get_task`/`update_task` do exist (`Evals_DB.py:873,786`), but the helper is what this file's own tests use.

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_bench_storage.py -k tag -p no:randomly`
Expected: FAIL — `extra_tags` round-trips as plain dicts, so the `Tag` equality assertions fail.

- [ ] **Step 3: Write minimal implementation**

First, in `tags.py`, rename `_coerce_tag` to `coerce_tag` (Task 2 defined it module-private) and update its internal caller — importing an underscore name across modules is worse than exporting it.

`CharacterProbeConfig` is `@dataclass(frozen=True)` (verified at `models.py:39`), so `__post_init__` cannot assign to a field directly; it must go through `object.__setattr__`, which is the same mechanism the existing `strict` handling already lives alongside.

In `models.py`, widen the field's declared type so a caller may pass mappings:

```python
    extra_tags: tuple[Any, ...] = ()
```

and in `__post_init__`, after the existing `character_ids` type checks:

```python
        # Validate and canonicalise the bench's tag extensions. NOT gated on
        # `strict`: that flag exists so a DRAFT bench with no characters and
        # no targets can be created and reloaded (phase 2), and a malformed
        # tag is corrupt data rather than a draft state.
        from .tags import coerce_tag, resolve_vocabulary

        coerced = tuple(coerce_tag(raw) for raw in self.extra_tags or ())
        object.__setattr__(self, "extra_tags", coerced)
        resolve_vocabulary(coerced)  # rejects a kind change to a built-in
```

The import is function-local because `tags.py` is stdlib-only and `models.py` is imported by it in neither direction today — keeping the import inside `__post_init__` avoids creating a module-level cycle if `tags.py` ever needs a model. If you confirm no cycle exists, a module-level import is cleaner; say which you chose and why in your report.

In `storage.py`, replace the three raw round-trip sites with a pair of helpers:

```python
def _tags_to_json(tags: Sequence[Any]) -> list[dict[str, str]]:
    """The stored form of a bench's extra tags.

    Args:
        tags: Validated ``Tag`` objects.

    Returns:
        list[dict[str, str]]: One JSON-safe mapping per tag.
    """
    return [{"slug": t.slug, "label": t.label, "kind": t.kind} for t in tags]


def _tags_from_json(raw: Any, owner_id: str) -> tuple[Tag, ...]:
    """Extra tags read back from a stored row or run snapshot.

    Accepts the raw mappings written before the vocabulary existed, so rows
    predating this phase still load.

    Args:
        raw: The stored ``extra_tags`` value, or None.
        owner_id: The bench or run-group id, named in any error.

    Returns:
        tuple[Tag, ...]: Validated tags, empty when none were stored.

    Raises:
        ValueError: If a stored entry is malformed -- naming ``owner_id``, so
            a corrupt row identifies itself rather than failing anonymously.
    """
    try:
        return tuple(coerce_tag(entry) for entry in raw or ())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{owner_id} has a corrupt extra_tags entry: {exc}") from exc
```

Use `_tags_to_json` at storage.py:279 and :556, and `_tags_from_json` at :480 — pass the bench id at :480 and the run-group id wherever the snapshot is read.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe -p no:randomly`
Expected: PASS — the new tag tests plus every existing character_probe test.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/models.py tldw_chatbook/Evals/character_probe/storage.py tldw_chatbook/Evals/character_probe/tags.py Tests/Evals/character_probe/test_bench_storage.py
git commit -m "feat(evals): validate a bench's extra tags at construction (task-1691 phase 3a)"
```

---

### Task 4: The vocabulary a run captured

**Files:**
- Modify: `tldw_chatbook/Evals/character_probe/storage.py`
- Test: `Tests/Evals/character_probe/test_conversation_storage.py`

**Interfaces:**
- Consumes: `resolve_vocabulary` (Task 2), `_tags_from_json` (Task 3), `load_probe_run_snapshot` (storage.py:644, already shipped).
- Produces: `run_group_vocabulary(db: EvalsDB, run_group_id: str) -> tuple[Tag, ...]`.

A reviewer annotates a run's answers, so the vocabulary that applies is **the one the run captured**, not the bench's current one. `_probe_run_snapshot` already writes `extra_tags` into the snapshot (storage.py:556) — this task reads it back. The spec's snapshot-provenance rule ("a card edited or deleted after the run does not change what the run shows") applies to tags for the same reason: a tag removed from the bench today must not orphan an annotation recorded last week.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.Evals.character_probe.storage import run_group_vocabulary
from tldw_chatbook.Evals.character_probe.tags import BUILTIN_TAGS, Tag


def test_a_run_with_no_extra_tags_has_exactly_the_builtins(db, probe_run_group):
    assert run_group_vocabulary(db, probe_run_group) == BUILTIN_TAGS


def test_a_runs_vocabulary_includes_the_benchs_extras_as_of_the_run(
    db, probe_run_group_with_extra_tags
):
    vocab = run_group_vocabulary(db, probe_run_group_with_extra_tags)
    assert Tag("meta-commentary", "Meta commentary", "failure") in vocab


def test_editing_the_bench_after_the_run_does_not_change_the_runs_vocabulary(
    db, probe_run_group_with_extra_tags, bench_id_of_that_run
):
    """Snapshot provenance: the run is annotated with what it captured."""
    from tldw_chatbook.Evals.character_probe.storage import (
        load_character_bench,
        save_character_bench,
    )
    config = load_character_bench(db, bench_id_of_that_run)
    save_character_bench(
        db,
        type(config)(
            name=config.name,
            probe_set_id=config.probe_set_id,
            character_ids=config.character_ids,
            target_ids=config.target_ids,
            extra_tags=(),
        ),
        bench_id_of_that_run,
    )
    vocab = run_group_vocabulary(db, probe_run_group_with_extra_tags)
    assert any(t.slug == "meta-commentary" for t in vocab)


def test_an_unknown_run_group_raises_naming_it(db):
    with pytest.raises(Exception) as exc:
        run_group_vocabulary(db, "no-such-group")
    assert "no-such-group" in str(exc.value)
```

Build `probe_run_group`, `probe_run_group_with_extra_tags`, and `bench_id_of_that_run` on what `Tests/Evals/character_probe/test_conversation_storage.py` already has — verified for you: the `db` fixture (`:26`), the `bench` fixture (`:174`), and the helpers `_seed_run(db)` (`:44`), `_conversation(...)` (`:31`), `_bench_config(**overrides)` (`:146`), `_cards()` (`:161`), and `_target_row(db, name, config)` (`:179`). Extend those rather than writing a second set of run-group construction.

`save_character_bench(db, config, task_id=None)` is the real signature (`storage.py:225`) — a new bench creates a row, passing a `task_id` updates in place — so the positional form in the test above is correct.

Note `_target_row`'s default name is `"steered"`: read what it actually writes into `config` before reusing it, and follow this slice's standing rule that a target row in a test is built the way the real "+ New target" form writes one, never as a hand-built dict.

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_conversation_storage.py -k vocabulary -p no:randomly`
Expected: FAIL — `ImportError: cannot import name 'run_group_vocabulary'`

- [ ] **Step 3: Write minimal implementation**

```python
def run_group_vocabulary(db: EvalsDB, run_group_id: str) -> tuple[Tag, ...]:
    """The tag vocabulary one run group was created under.

    A reviewer annotates a run's answers, so the vocabulary that applies is
    the one the run captured -- not the bench's current one. This is the same
    provenance rule the card snapshot follows: editing the bench afterwards
    must not change what the run shows, and must not orphan an annotation
    already recorded against a tag the bench has since dropped.

    Args:
        db: The evals database handle.
        run_group_id: The run group to read.

    Returns:
        tuple[Tag, ...]: Built-in tags plus whatever extras the snapshot
        recorded, resolved through ``resolve_vocabulary``.

    Raises:
        ValueError: If the snapshot is missing or its stored tags are corrupt
            -- naming the run group.
    """
    snapshot = load_probe_run_snapshot(db, run_group_id)
    return resolve_vocabulary(
        _tags_from_json(snapshot.get("extra_tags"), run_group_id)
    )
```

Check what `load_probe_run_snapshot` already raises for an unknown group and let that propagate rather than adding a second error path.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/storage.py Tests/Evals/character_probe/test_conversation_storage.py
git commit -m "feat(evals): a run group's tag vocabulary comes from its snapshot (task-1691 phase 3a)"
```

---

### Task 5: An annotation cannot reference a tag the run never had

**Files:**
- Modify: `tldw_chatbook/Evals/character_probe/storage.py`
- Test: `Tests/Evals/character_probe/test_conversation_storage.py`

**Interfaces:**
- Consumes: `run_group_vocabulary` (Task 4), `canonical_slug` (Task 2).
- Produces: `annotate_turn` (storage.py:921) validating and canonicalising its `tags` argument; unchanged signature.

Today `annotate_turn` writes whatever slugs it is handed. A typo'd slug becomes an annotation nobody can filter for and a tag Phase 4's summary cannot group, because it has no kind. Validating at the write matches the fail-loudly rule and keeps the summary's grouping total.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.Evals.character_probe.storage import (
    annotate_turn,
    load_turn_annotations,
)


def test_a_known_tag_is_stored(db, probe_run_group):
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0,
        tags=["broke-character"], note="third turn",
    )
    stored = load_turn_annotations(db, probe_run_group)
    assert stored[(1, 0, 0, "t-1", 0)]["tags"] == ["broke-character"]


def test_an_unknown_tag_is_rejected_naming_it(db, probe_run_group):
    with pytest.raises(ValueError) as exc:
        annotate_turn(
            db, probe_run_group, 1, 0, 0, "t-1", 0,
            tags=["brok-charcter"], note="",
        )
    assert "brok-charcter" in str(exc.value)


def test_nothing_is_written_when_one_tag_of_several_is_unknown(
    db, probe_run_group
):
    with pytest.raises(ValueError):
        annotate_turn(
            db, probe_run_group, 1, 0, 0, "t-1", 0,
            tags=["broke-character", "no-such-tag"], note="",
        )
    assert load_turn_annotations(db, probe_run_group) == {}


def test_a_non_canonical_tag_is_canonicalised_rather_than_rejected(
    db, probe_run_group
):
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0,
        tags=["Broke Character"], note="",
    )
    stored = load_turn_annotations(db, probe_run_group)
    assert stored[(1, 0, 0, "t-1", 0)]["tags"] == ["broke-character"]


def test_duplicate_tags_are_stored_once(db, probe_run_group):
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0,
        tags=["broke-character", "broke-character"], note="",
    )
    stored = load_turn_annotations(db, probe_run_group)
    assert stored[(1, 0, 0, "t-1", 0)]["tags"] == ["broke-character"]


def test_an_annotation_with_no_tags_but_a_note_is_allowed(
    db, probe_run_group
):
    """A note without a tag is a real observation, not an empty write."""
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0, tags=[], note="odd phrasing",
    )
    stored = load_turn_annotations(db, probe_run_group)
    assert stored[(1, 0, 0, "t-1", 0)]["note"] == "odd phrasing"


def test_a_benchs_extra_tag_is_accepted(db, probe_run_group_with_extra_tags):
    annotate_turn(
        db, probe_run_group_with_extra_tags, 1, 0, 0, "t-1", 0,
        tags=["meta-commentary"], note="",
    )
    stored = load_turn_annotations(db, probe_run_group_with_extra_tags)
    assert stored[(1, 0, 0, "t-1", 0)]["tags"] == ["meta-commentary"]
```

Reuse the fixtures Task 4 extended. `test_conversation_storage.py` already holds this slice's annotation tests (`test_a_turn_annotation_persists_with_its_tags_and_note`, `test_a_conversation_can_be_reviewed_with_no_annotations`, `test_review_state_is_scoped_to_its_run_group`) — add these beside them rather than starting a new file.

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_conversation_storage.py -p no:randomly`
Expected: FAIL — an unknown slug is written rather than rejected, so the `pytest.raises` tests fail.

- [ ] **Step 3: Write minimal implementation**

In `annotate_turn`, before the `db.upsert_probe_turn_annotation(...)` call:

```python
    vocabulary = run_group_vocabulary(db, run_group_id)
    known = {tag.slug for tag in vocabulary}
    canonical: list[str] = []
    for raw in tags or ():
        slug = canonical_slug(str(raw))
        if slug not in known:
            raise ValueError(
                f"{slug!r} is not a tag in this run's vocabulary "
                f"({', '.join(sorted(known))}). Annotations are grouped by tag "
                f"kind, so an unknown tag would have no kind to group under."
            )
        if slug not in canonical:
            canonical.append(slug)
```

then pass `canonical` where `tags` was passed. Update the docstring's Raises section to name the new `ValueError`.

Validation happens before the write, so a rejected annotation leaves nothing behind — which is what `test_nothing_is_written_when_one_tag_of_several_is_unknown` pins.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/storage.py Tests/Evals/character_probe/test_conversation_storage.py
git commit -m "feat(evals): reject an annotation tag outside the run's vocabulary (task-1691 phase 3a)"
```

---

### Task 6: Deleting a bench takes its annotations with it

**Files:**
- Modify: `tldw_chatbook/DB/Evals_DB.py`
- Test: `Tests/Evals/character_probe/test_conversation_storage.py`

**Interfaces:**
- Consumes: `EvalsDB.delete_task` (Evals_DB.py:850), the two annotation tables (Evals_DB.py:316,333).
- Produces: `EvalsDB.delete_probe_annotations_for_run_groups(run_group_ids: Sequence[str]) -> int` returning rows removed; `delete_task` calling it for the task's run groups.

The spec: "Deleting a run group cascades both, since they describe those answers and mean nothing without them." Neither table has a foreign key or an `ON DELETE CASCADE` today, and `delete_task` (the only delete path — there is no `delete_run_group`) does not touch them. Phase 2 made a character bench deletable from the UI, so orphaned annotation rows are reachable now, not hypothetically.

- [ ] **Step 1: Write the failing test**

```python
def test_deleting_a_bench_removes_its_turn_annotations(
    db, probe_run_group, bench_id_of_that_run
):
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0,
        tags=["broke-character"], note="",
    )
    assert load_turn_annotations(db, probe_run_group)

    db.delete_task(bench_id_of_that_run)

    assert db.list_probe_turn_annotations(probe_run_group) == []


def test_deleting_a_bench_removes_its_review_state(
    db, probe_run_group, bench_id_of_that_run
):
    from tldw_chatbook.Evals.character_probe.storage import mark_conversation_reviewed

    mark_conversation_reviewed(db, probe_run_group, 1, 0, 0, "t-1", note="fine")
    assert db.list_probe_review_state(probe_run_group)

    db.delete_task(bench_id_of_that_run)

    assert db.list_probe_review_state(probe_run_group) == []


def test_deleting_a_bench_leaves_another_benchs_annotations_alone(
    db, probe_run_group, bench_id_of_that_run, second_probe_run_group
):
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0, tags=["refused"], note="",
    )
    annotate_turn(
        db, second_probe_run_group, 1, 0, 0, "t-1", 0,
        tags=["refused"], note="",
    )

    db.delete_task(bench_id_of_that_run)

    assert db.list_probe_turn_annotations(second_probe_run_group)


def test_deleting_a_word_bench_touches_no_probe_annotation_rows(
    db, probe_run_group, seeded_word_bench_id
):
    annotate_turn(
        db, probe_run_group, 1, 0, 0, "t-1", 0, tags=["refused"], note="",
    )
    db.delete_task(seeded_word_bench_id)
    assert db.list_probe_turn_annotations(probe_run_group)
```

`second_probe_run_group` needs a run group under a **different** bench — extend the fixture module accordingly. `seeded_word_bench_id` should come from the word-bench helpers the Evals tests already use; grep for them rather than writing a new one.

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_conversation_storage.py -k deleting -p no:randomly`
Expected: FAIL — the annotation rows survive `delete_task`.

- [ ] **Step 3: Write minimal implementation**

In `Evals_DB.py`:

```python
    def delete_probe_annotations_for_run_groups(
        self, run_group_ids: Sequence[str]
    ) -> int:
        """Remove every character-probe annotation for these run groups.

        Annotations and review state describe one run's answers and mean
        nothing without them, so they are removed with the run rather than
        left orphaned.

        Args:
            run_group_ids: The run groups whose annotations to remove.

        Returns:
            int: Rows removed across both tables. Zero is a normal result --
            a run nobody reviewed has no annotations.
        """
        ids = [str(rg) for rg in run_group_ids if rg]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        removed = 0
        conn = self._get_connection()
        with conn:
            for table in (
                "eval_probe_turn_annotations",
                "eval_probe_review_state",
            ):
                cursor = conn.execute(
                    f"DELETE FROM {table} WHERE run_group_id IN ({placeholders})",
                    ids,
                )
                removed += cursor.rowcount
        return removed
```

The table names are interpolated from a two-element literal tuple in this function's own body, never from a caller — the `run_group_id` values stay parameterized.

**`EvalsDB` has no `transaction()` context manager** — verified. The `conn = self._get_connection()` / `with conn:` form above is what `delete_task` (Evals_DB.py:850-858) and its neighbours actually use; follow it. (The repo's CLAUDE.md shows a `db.transaction()` idiom, and an automated reviewer has previously flagged this class for not using it — that convention belongs to a different DB class in this codebase. Match the file, not the doc.)

In `delete_task`, after the task's own soft-delete succeeds, collect the task's run group ids and call the new method. Find how run groups are associated with a task (`eval_runs.run_group_id` scoped by `task_id`) and read them before the delete, since a soft-deleted task may no longer resolve them.

Note the asymmetry to preserve: `delete_task` is a **soft** delete for the task itself; the annotation rows are **hard**-deleted, because they carry no `deleted_at` column and nothing reads them for an undeleted view. Say so in a comment so a later reader does not "fix" it.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe Tests/DB -p no:randomly`
Expected: PASS — the new cascade tests plus every existing DB test.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/Evals_DB.py Tests/Evals/character_probe/test_conversation_storage.py
git commit -m "feat(evals): deleting a bench cascades its probe annotations (task-1691 phase 3a)"
```

---

## Phase 3a exit criteria

- Every tag in the system carries one of exactly three kinds, and nothing can create one without stating its kind.
- The ten built-in tags exist with the slugs and kinds the spec names.
- A bench's `extra_tags` are validated at construction, canonicalised, and cannot change a built-in's kind.
- A run group's vocabulary comes from its own snapshot, so editing the bench afterwards does not change or orphan what the run captured.
- An annotation cannot reference a tag outside its run's vocabulary, and a rejected annotation writes nothing.
- Deleting a bench removes its annotations and review state, and no other bench's.
- `Tests/Evals/character_probe` and `Tests/Evals` remain green; nothing in `Tests/UI` needed to change, because this phase ships no UI.

## Not in Phase 3a (deliberate)

The conversation view, the queue, keyboard navigation, the tag-application UI, the create-a-tag flow, and ordering hints are Phase 3b — this phase gives that UI a vocabulary to render and a validated place to write. The summary is Phase 4. No view that sums tags into a number is ever in scope.
