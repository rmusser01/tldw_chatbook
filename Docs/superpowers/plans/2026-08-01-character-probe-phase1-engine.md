# Character Probe Evals — Phase 1 (Engine + Storage) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the engine and storage for character-probe evals — probe parsing, bench config, card snapshots, prompt assembly, the conversation runner, and annotation storage — with no UI.

**Architecture:** A second bench type in the existing Evals slice, discriminated by `config_data.bench_type = "character_probe"` exactly as word_bench discriminates today. It reuses `eval_models` targets (and task-1611 steering), `eval_runs`/`run_group_id`, and the dataset inline-samples convention; it reuses **none** of word_bench's measurement stack. Everything here lives under `tldw_chatbook/Evals/character_probe/`, a sibling package to `word_bench/`.

**Tech Stack:** Python ≥3.11, `dataclasses`, `asyncio` (with `asyncio.to_thread` for the blocking chat gateway), SQLite via `EvalsDB`/`CharactersRAGDB`, pytest.

## Global Constraints

Copied verbatim from `Docs/superpowers/specs/2026-08-01-character-probe-eval-design.md`. Every task's requirements implicitly include this section.

- **No logprobs, no top-K, no normalizer, no truncated mass, and no canary/degenerate vocabulary.** Only generated text matters. A character-probe target's readiness means only: can we reach this model and get text back.
- **`chat_api_call` is synchronous and must never be called from the event loop.** Every call dispatches through `asyncio.to_thread`. The bench's own `concurrency` setting bounds the thread fan-out.
- **Cancel stops scheduling; it cannot abort a turn already in flight.** In-flight turns run to completion and are recorded.
- **Steering composes ahead of the card's system prompt.** Both preserved, neither discarded; the composed result is recorded in the snapshot.
- **A card with no `first_message` starts with the user's first scripted turn.** No synthetic greeting is invented.
- **Per-sample seed is `seed + sample_index`** so a seeded run is reproducible *and* its samples differ.
- **Turns are delimited explicitly:** a line of `---` separates turns within a probe; a line of `===` separates probes. Interior whitespace is preserved exactly; leading/trailing blank lines around a turn are stripped.
- **The ordered turn list lives in the `metadata` JSON**, never in `actual_output`.
- **`sample_id` composes `(card_id, probe_index, sample_index)`.**
- **Tags carry a kind** (`failure`/`notable`/`positive`); creating a tag requires choosing one, with no default.
- Tests: real in-memory `EvalsDB`, fake chat callable. Every behavioural test must fail against pre-change code. Google-style docstrings with Args/Returns/Raises on public callables. Parameterized SQL only.
- Run tests foreground: `/private/tmp/tldw-venv/bin/python -m pytest <paths> -p no:randomly` from the clone root. Never `-q`.

## File Structure

- `tldw_chatbook/Evals/character_probe/__init__.py` — package marker.
- `tldw_chatbook/Evals/character_probe/models.py` — `Probe`, `ProbeSet`, `CharacterProbeConfig`, `CardSnapshot`, `ConversationTurn`, `Conversation`, `TagKind`. Pure data, no I/O.
- `tldw_chatbook/Evals/character_probe/probe_format.py` — the `---`/`===` text format parser and serializer. Pure string handling.
- `tldw_chatbook/Evals/character_probe/storage.py` — probe-set persistence, bench save/load, conversation persistence, annotation and review-state tables.
- `tldw_chatbook/Evals/character_probe/cards.py` — read-only card snapshotting across the `ChaChaNotes_DB` boundary.
- `tldw_chatbook/Evals/character_probe/prompt.py` — system-prompt composition and message-list assembly.
- `tldw_chatbook/Evals/character_probe/runner.py` — the conversation runner, `to_thread` bridge, concurrency, cancel.
- Tests mirror each module under `Tests/Evals/character_probe/`.

---

### Task 1: Probe models and the text format

**Files:**
- Create: `tldw_chatbook/Evals/character_probe/__init__.py`
- Create: `tldw_chatbook/Evals/character_probe/models.py`
- Create: `tldw_chatbook/Evals/character_probe/probe_format.py`
- Test: `Tests/Evals/character_probe/test_probe_format.py`

**Interfaces:**
- Produces: `Probe(turns: tuple[str, ...])`; `ProbeSet(probes: tuple[Probe, ...])`; `parse_probe_text(text: str) -> ProbeSet`; `format_probe_text(probe_set: ProbeSet) -> str`; constants `TURN_DELIMITER = "---"`, `PROBE_DELIMITER = "==="`.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.Evals.character_probe.models import Probe, ProbeSet
from tldw_chatbook.Evals.character_probe.probe_format import (
    format_probe_text,
    parse_probe_text,
)


def test_single_probe_single_turn():
    assert parse_probe_text("What do you think about lying?") == ProbeSet(
        probes=(Probe(turns=("What do you think about lying?",)),)
    )


def test_turns_split_on_the_turn_delimiter():
    text = "What do you think about lying?\n---\nAnd if it protected someone?"
    assert parse_probe_text(text) == ProbeSet(
        probes=(
            Probe(
                turns=(
                    "What do you think about lying?",
                    "And if it protected someone?",
                )
            ),
        )
    )


def test_probes_split_on_the_probe_delimiter():
    text = "First probe\n===\nSecond probe"
    parsed = parse_probe_text(text)
    assert len(parsed.probes) == 2
    assert parsed.probes[0].turns == ("First probe",)
    assert parsed.probes[1].turns == ("Second probe",)


def test_a_turn_may_span_multiple_paragraphs():
    """The whole point of the delimiter format: complex prompts are the subject."""
    text = "Describe your earliest memory.\n\nTake your time, and include what you could smell."
    parsed = parse_probe_text(text)
    assert parsed.probes[0].turns == (
        "Describe your earliest memory.\n\nTake your time, and include what you could smell.",
    )


def test_interior_whitespace_is_preserved_exactly():
    text = "Line one\n    indented line\nLine three"
    assert parsed_turn(text) == "Line one\n    indented line\nLine three"


def parsed_turn(text: str) -> str:
    return parse_probe_text(text).probes[0].turns[0]


def test_blank_lines_around_a_turn_are_stripped():
    text = "\n\nWhat is your name?\n\n\n---\n\nAnd your age?\n"
    parsed = parse_probe_text(text)
    assert parsed.probes[0].turns == ("What is your name?", "And your age?")


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="no probes"):
        parse_probe_text("   \n\n  ")


def test_a_probe_with_no_turns_is_rejected():
    with pytest.raises(ValueError, match="probe 2"):
        parse_probe_text("Real probe\n===\n   \n===\nAnother")


def test_round_trip_through_format_and_parse():
    original = ProbeSet(
        probes=(
            Probe(turns=("One\n\nwith a paragraph", "Two")),
            Probe(turns=("Three",)),
        )
    )
    assert parse_probe_text(format_probe_text(original)) == original
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_probe_format.py -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Evals.character_probe'`

- [ ] **Step 3: Write minimal implementation**

`tldw_chatbook/Evals/character_probe/__init__.py`:

```python
"""Character probe evals: scripted question sets run against character cards,
collected as conversations for human review.

Sibling package to ``word_bench``. Shares its targets, run groups, and rail,
but none of its measurement stack -- see
``Docs/superpowers/specs/2026-08-01-character-probe-eval-design.md``.
"""
```

`tldw_chatbook/Evals/character_probe/models.py`:

```python
"""Pure data for character probe evals. No I/O, no Textual, no provider calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    """One scripted exchange: an ordered list of user turns.

    A "one-off" question is simply a probe with a single turn -- there is no
    separate type for it. Turn text is verbatim, including interior newlines,
    because prompt formatting changes model behaviour.
    """

    turns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("A probe needs at least one turn.")


@dataclass(frozen=True)
class ProbeSet:
    """An ordered collection of probes, the unit a bench runs."""

    probes: tuple[Probe, ...]
```

`tldw_chatbook/Evals/character_probe/probe_format.py`:

```python
"""The plain-text probe format: `---` between turns, `===` between probes.

Turns are delimited explicitly rather than by line breaks so a single turn can
be a multi-paragraph prompt -- complex prompts are exactly what this eval
exists to study, and a newline-delimited format could not express one.
"""

from __future__ import annotations

from .models import Probe, ProbeSet

#: A line containing only this separates turns within a probe.
TURN_DELIMITER = "---"
#: A line containing only this separates probes within a set.
PROBE_DELIMITER = "==="


def _split_on_delimiter(text: str, delimiter: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.strip() == delimiter:
            chunks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    chunks.append("\n".join(current))
    return chunks


def _clean_turn(raw: str) -> str:
    """Strip only leading/trailing blank lines; interior whitespace is data."""
    return raw.strip("\n").strip() if not raw.strip() else raw.strip("\n")


def parse_probe_text(text: str) -> ProbeSet:
    """Parse the plain-text probe format into a ``ProbeSet``.

    Args:
        text: The file's contents.

    Returns:
        ProbeSet: The parsed probes, in file order.

    Raises:
        ValueError: If the text contains no probes, or if any probe has no
            turns (naming the 1-based probe number so the author can find it).
    """
    probe_chunks = _split_on_delimiter(text, PROBE_DELIMITER)
    probes: list[Probe] = []
    for index, chunk in enumerate(probe_chunks, start=1):
        if not chunk.strip():
            if len(probe_chunks) == 1 or index in (1, len(probe_chunks)):
                # A wholly empty document, or trailing/leading delimiter noise.
                continue
            raise ValueError(f"probe {index} has no turns")
        turns = [
            _clean_turn(raw)
            for raw in _split_on_delimiter(chunk, TURN_DELIMITER)
            if raw.strip()
        ]
        if not turns:
            raise ValueError(f"probe {index} has no turns")
        probes.append(Probe(turns=tuple(turns)))
    if not probes:
        raise ValueError("The probe file contains no probes.")
    return ProbeSet(probes=tuple(probes))


def format_probe_text(probe_set: ProbeSet) -> str:
    """Render a ``ProbeSet`` back to the plain-text format.

    Args:
        probe_set: The set to render.

    Returns:
        str: Text that ``parse_probe_text`` round-trips to an equal ProbeSet.
    """
    return f"\n{PROBE_DELIMITER}\n".join(
        f"\n{TURN_DELIMITER}\n".join(probe.turns) for probe in probe_set.probes
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_probe_format.py -p no:randomly`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/ Tests/Evals/character_probe/
git commit -m "feat(evals): probe models and the delimited probe text format (task-1691 phase 1)"
```

---

### Task 2: Probe-set storage on the dataset convention

**Files:**
- Create: `tldw_chatbook/Evals/character_probe/storage.py`
- Test: `Tests/Evals/character_probe/test_probe_storage.py`

**Interfaces:**
- Consumes: `ProbeSet`, `Probe`, `parse_probe_text` (Task 1).
- Produces: `PROBE_DATASET_TYPE = "character_probe"`; `save_probe_set(db: EvalsDB, name: str, probe_set: ProbeSet, dataset_id: str | None = None) -> str`; `load_probe_set(db: EvalsDB, dataset_id: str) -> ProbeSet`; `is_probe_set(dataset_row: Mapping[str, Any]) -> bool`.

Probe sets reuse the existing inline-samples convention that snippets use (`metadata[RESERVED_LOCAL_DATASET_SAMPLES_KEY]`), with `metadata["dataset_type"] = "character_probe"` discriminating them — the same shape `bench_type` gives `eval_tasks`. `EvalsDB.create_dataset` restricts `format` to `huggingface|json|csv|custom`; probe sets use `"custom"`.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import Probe, ProbeSet
from tldw_chatbook.Evals.character_probe.storage import (
    PROBE_DATASET_TYPE,
    is_probe_set,
    load_probe_set,
    save_probe_set,
)


@pytest.fixture
def db():
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def probe_set():
    return ProbeSet(
        probes=(
            Probe(turns=("What do you think about lying?", "And to protect someone?")),
            Probe(turns=("Describe your earliest memory.\n\nInclude the smell.",)),
        )
    )


def test_save_then_load_round_trips(db, probe_set):
    dataset_id = save_probe_set(db, "starter", probe_set)
    assert load_probe_set(db, dataset_id) == probe_set


def test_saved_set_is_marked_as_a_probe_set(db, probe_set):
    dataset_id = save_probe_set(db, "starter", probe_set)
    row = db.get_dataset(dataset_id)
    assert (row.get("metadata") or {}).get("dataset_type") == PROBE_DATASET_TYPE
    assert is_probe_set(row) is True


def test_a_snippet_dataset_is_not_a_probe_set(db):
    dataset_id = db.create_dataset(
        name="snippets", format="custom", source_path="inline:snippets"
    )
    assert is_probe_set(db.get_dataset(dataset_id)) is False


def test_saving_with_an_existing_id_replaces_its_probes(db, probe_set):
    dataset_id = save_probe_set(db, "starter", probe_set)
    replacement = ProbeSet(probes=(Probe(turns=("Only one now",)),))
    assert save_probe_set(db, "starter", replacement, dataset_id=dataset_id) == dataset_id
    assert load_probe_set(db, dataset_id) == replacement


def test_loading_a_non_probe_dataset_raises(db):
    dataset_id = db.create_dataset(
        name="snippets", format="custom", source_path="inline:snippets"
    )
    with pytest.raises(ValueError, match="not a probe set"):
        load_probe_set(db, dataset_id)


def test_loading_a_missing_dataset_raises(db):
    with pytest.raises(ValueError, match="could not be found"):
        load_probe_set(db, "nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_probe_storage.py -p no:randomly`
Expected: FAIL — `ImportError: cannot import name 'save_probe_set'`

- [ ] **Step 3: Write minimal implementation**

Append to `tldw_chatbook/Evals/character_probe/storage.py`:

```python
"""Persistence for character probe evals.

Probe sets reuse the dataset inline-samples convention that snippets already
use, discriminated by ``metadata["dataset_type"]`` -- the same shape
``config_data.bench_type`` gives ``eval_tasks``. Nothing here writes SQL
directly; every call goes through ``EvalsDB``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ...DB.Evals_DB import EvalsDB
from ...UI.Evals.snippet_editor import RESERVED_LOCAL_DATASET_SAMPLES_KEY
from .models import Probe, ProbeSet

#: Marks a dataset row as holding probes rather than snippets.
PROBE_DATASET_TYPE = "character_probe"


def is_probe_set(dataset_row: Mapping[str, Any]) -> bool:
    """Whether a dataset row holds probes rather than snippets.

    Args:
        dataset_row: A row as returned by ``EvalsDB.get_dataset``/``list_datasets``.

    Returns:
        bool: True when the row is marked as a probe set.
    """
    metadata = dataset_row.get("metadata") or {}
    return metadata.get("dataset_type") == PROBE_DATASET_TYPE


def _probe_set_to_samples(probe_set: ProbeSet) -> list[dict[str, Any]]:
    return [
        {"index": index, "turns": list(probe.turns)}
        for index, probe in enumerate(probe_set.probes)
    ]


def _samples_to_probe_set(samples: Any) -> ProbeSet:
    if not isinstance(samples, list):
        return ProbeSet(probes=())
    probes = [
        Probe(turns=tuple(str(turn) for turn in sample.get("turns") or ()))
        for sample in samples
        if isinstance(sample, Mapping) and sample.get("turns")
    ]
    return ProbeSet(probes=tuple(probes))


def save_probe_set(
    db: EvalsDB,
    name: str,
    probe_set: ProbeSet,
    dataset_id: Optional[str] = None,
) -> str:
    """Persist a probe set, creating or replacing a dataset row.

    Args:
        db: The evals database handle.
        name: Display name for the dataset row.
        probe_set: The probes to store.
        dataset_id: An existing probe-set dataset to overwrite; when omitted a
            new dataset row is created.

    Returns:
        str: The dataset id holding the probes.
    """
    metadata = {
        "dataset_type": PROBE_DATASET_TYPE,
        RESERVED_LOCAL_DATASET_SAMPLES_KEY: _probe_set_to_samples(probe_set),
    }
    if dataset_id is None:
        return db.create_dataset(
            name=name,
            format="custom",
            source_path=f"inline:{name}",
            metadata=metadata,
        )
    db.update_dataset(dataset_id, metadata=metadata)
    return dataset_id


def load_probe_set(db: EvalsDB, dataset_id: str) -> ProbeSet:
    """Read a probe set back.

    Args:
        db: The evals database handle.
        dataset_id: The dataset row to read.

    Returns:
        ProbeSet: The stored probes, in order.

    Raises:
        ValueError: If the dataset does not exist, or is not a probe set --
            loading a snippet dataset as probes would otherwise silently yield
            an empty set and look like an authoring mistake.
    """
    row = db.get_dataset(dataset_id)
    if row is None:
        raise ValueError(f"Probe set {dataset_id!r} could not be found.")
    if not is_probe_set(row):
        raise ValueError(f"Dataset {dataset_id!r} is not a probe set.")
    metadata = row.get("metadata") or {}
    return _samples_to_probe_set(metadata.get(RESERVED_LOCAL_DATASET_SAMPLES_KEY))
```

**Note for the implementer:** verify `EvalsDB.update_dataset` exists and accepts `metadata=`; if it does not, add the smallest possible method following `update_task`'s shape (parameterized, soft-delete aware) rather than writing SQL here.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_probe_storage.py -p no:randomly`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/storage.py Tests/Evals/character_probe/test_probe_storage.py
git commit -m "feat(evals): probe sets persist on the dataset inline-samples convention (task-1691 phase 1)"
```

---

### Task 3: Bench config save and load

**Files:**
- Modify: `tldw_chatbook/Evals/character_probe/models.py`
- Modify: `tldw_chatbook/Evals/character_probe/storage.py`
- Test: `Tests/Evals/character_probe/test_bench_storage.py`

**Interfaces:**
- Produces: `BENCH_TYPE = "character_probe"`; `CharacterProbeConfig(name, probe_set_id, character_ids, target_ids, description="", concurrency=1, samples_per_cell=1, seed=None, temperature=0.8, max_tokens=512, extra_tags=())`; `save_character_bench(db, config, task_id=None) -> str`; `load_character_bench(db, task_id) -> CharacterProbeConfig`; `is_character_bench(task_row) -> bool`.

`character_ids` are **ints** (`character_cards.id`), unlike every eval id which is `str` — do not normalise them to strings.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import CharacterProbeConfig
from tldw_chatbook.Evals.character_probe.storage import (
    BENCH_TYPE,
    is_character_bench,
    load_character_bench,
    save_character_bench,
)


@pytest.fixture
def db():
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def config():
    return CharacterProbeConfig(
        name="villain probes",
        probe_set_id="ps-1",
        character_ids=(3, 7),
        target_ids=("t-1",),
        samples_per_cell=2,
        seed=1234,
    )


def test_save_then_load_round_trips(db, config):
    task_id = save_character_bench(db, config)
    assert load_character_bench(db, task_id) == config


def test_saved_bench_is_marked_with_its_type(db, config):
    task_id = save_character_bench(db, config)
    row = db.get_task(task_id)
    assert (row.get("config_data") or {}).get("bench_type") == BENCH_TYPE
    assert is_character_bench(row) is True


def test_character_ids_survive_as_integers(db, config):
    """character_cards.id is an INTEGER; every eval id is TEXT. Do not merge them."""
    task_id = save_character_bench(db, config)
    assert load_character_bench(db, task_id).character_ids == (3, 7)


def test_defaults_are_conservative():
    config = CharacterProbeConfig(
        name="n", probe_set_id="p", character_ids=(1,), target_ids=("t",)
    )
    assert config.samples_per_cell == 1
    assert config.seed is None
    assert config.concurrency == 1
    assert config.extra_tags == ()


def test_editing_an_existing_bench_updates_in_place(db, config):
    task_id = save_character_bench(db, config)
    edited = CharacterProbeConfig(**{**config.__dict__, "name": "renamed"})
    assert save_character_bench(db, edited, task_id=task_id) == task_id
    assert load_character_bench(db, task_id).name == "renamed"


def test_samples_per_cell_below_one_is_rejected():
    with pytest.raises(ValueError, match="samples_per_cell"):
        CharacterProbeConfig(
            name="n",
            probe_set_id="p",
            character_ids=(1,),
            target_ids=("t",),
            samples_per_cell=0,
        )


def test_a_bench_needs_at_least_one_character():
    with pytest.raises(ValueError, match="at least one character"):
        CharacterProbeConfig(
            name="n", probe_set_id="p", character_ids=(), target_ids=("t",)
        )


def test_loading_a_word_bench_as_a_character_bench_raises(db):
    task_id = db.create_task(
        name="word bench",
        description="",
        task_type="logprob",
        config_data={"bench_type": "word_bench"},
    )
    with pytest.raises(ValueError, match="not a character probe bench"):
        load_character_bench(db, task_id)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_bench_storage.py -p no:randomly`
Expected: FAIL — `ImportError: cannot import name 'CharacterProbeConfig'`

- [ ] **Step 3: Write minimal implementation**

Append to `models.py`:

```python
@dataclass(frozen=True)
class CharacterProbeConfig:
    """A character probe bench definition.

    ``character_ids`` are ``character_cards.id`` INTEGERs, unlike every eval
    id in this slice, which is TEXT. They are deliberately not normalised to
    strings: the cross-database lookup in ``cards.py`` binds them as integers.
    """

    name: str
    probe_set_id: str
    character_ids: tuple[int, ...]
    target_ids: tuple[str, ...]
    description: str = ""
    concurrency: int = 1
    samples_per_cell: int = 1
    seed: Optional[int] = None
    temperature: float = 0.8
    max_tokens: int = 512
    extra_tags: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        if self.samples_per_cell < 1:
            raise ValueError("samples_per_cell must be 1 or more.")
        if self.concurrency < 1:
            raise ValueError("concurrency must be 1 or more.")
        if not self.character_ids:
            raise ValueError("A character probe bench needs at least one character.")
        if not self.target_ids:
            raise ValueError("A character probe bench needs at least one target.")
```

Add `from typing import Optional` to `models.py`'s imports.

Append to `storage.py`:

```python
#: Discriminates a character probe bench from a word bench in ``eval_tasks``.
BENCH_TYPE = "character_probe"


def is_character_bench(task_row: Mapping[str, Any]) -> bool:
    """Whether an ``eval_tasks`` row is a character probe bench.

    Args:
        task_row: A row as returned by ``EvalsDB.get_task``/``list_tasks``.

    Returns:
        bool: True when the row carries this bench type.
    """
    return (task_row.get("config_data") or {}).get("bench_type") == BENCH_TYPE


def save_character_bench(
    db: EvalsDB, config: CharacterProbeConfig, task_id: Optional[str] = None
) -> str:
    """Persist a character probe bench.

    Mirrors ``word_bench.storage.save_bench``: a new bench creates an
    ``eval_tasks`` row, an existing one updates in place. The probe set is
    fixed at creation for the same reason a word bench's dataset is -- see
    that function's own docstring.

    Args:
        db: The evals database handle.
        config: The bench to persist.
        task_id: An existing bench to update; omit to create.

    Returns:
        str: The bench's ``eval_tasks`` id.

    Raises:
        ConflictError: If the name collides with another task's (including a
            soft-deleted one -- the UNIQUE index has no ``deleted_at``
            exemption).
    """
    config_data = {
        "bench_type": BENCH_TYPE,
        "probe_set_id": config.probe_set_id,
        "character_ids": list(config.character_ids),
        "target_ids": list(config.target_ids),
        "concurrency": config.concurrency,
        "samples_per_cell": config.samples_per_cell,
        "seed": config.seed,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "extra_tags": list(config.extra_tags),
    }
    if task_id is not None:
        db.update_task(
            task_id,
            name=config.name,
            description=config.description,
            config_data=config_data,
        )
        return task_id
    return db.create_task(
        name=config.name,
        description=config.description,
        task_type="generation",
        config_data=config_data,
    )


def load_character_bench(db: EvalsDB, task_id: str) -> CharacterProbeConfig:
    """Read a character probe bench back.

    Args:
        db: The evals database handle.
        task_id: The bench to read.

    Returns:
        CharacterProbeConfig: The stored bench.

    Raises:
        ValueError: If the task does not exist or is not a character probe
            bench -- loading a word bench here would otherwise produce a
            config with empty characters and look like data loss.
    """
    row = db.get_task(task_id)
    if row is None:
        raise ValueError(f"Bench {task_id!r} could not be found.")
    if not is_character_bench(row):
        raise ValueError(f"Bench {task_id!r} is not a character probe bench.")
    data = row.get("config_data") or {}
    return CharacterProbeConfig(
        name=row.get("name") or "",
        description=row.get("description") or "",
        probe_set_id=str(data.get("probe_set_id") or ""),
        character_ids=tuple(int(cid) for cid in data.get("character_ids") or ()),
        target_ids=tuple(str(tid) for tid in data.get("target_ids") or ()),
        concurrency=int(data.get("concurrency") or 1),
        samples_per_cell=int(data.get("samples_per_cell") or 1),
        seed=data.get("seed"),
        temperature=float(data.get("temperature", 0.8)),
        max_tokens=int(data.get("max_tokens") or 512),
        extra_tags=tuple(data.get("extra_tags") or ()),
    )
```

Add `CharacterProbeConfig` to `storage.py`'s import from `.models`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_bench_storage.py -p no:randomly`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/ Tests/Evals/character_probe/test_bench_storage.py
git commit -m "feat(evals): character probe bench config save and load (task-1691 phase 1)"
```

---

### Task 4: Card snapshots across the database boundary

**Files:**
- Create: `tldw_chatbook/Evals/character_probe/cards.py`
- Test: `Tests/Evals/character_probe/test_cards.py`

**Interfaces:**
- Produces: `CardSnapshot(id, name, system_prompt, personality, scenario, first_message, post_history_instructions, message_example)` (added to `models.py`); `snapshot_cards(chacha_db, character_ids: Sequence[int]) -> tuple[CardSnapshot, ...]`.

Character cards live in `ChaChaNotes_DB` (`CharactersRAGDB`), a different database from `EvalsDB` with no foreign keys between them. Snapshotting copies the card's text into the run so later edits or deletions never rewrite history.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.Evals.character_probe.cards import snapshot_cards
from tldw_chatbook.Evals.character_probe.models import CardSnapshot


class _FakeCharacterDB:
    """Stands in for CharactersRAGDB; only get_character_card_by_id is used."""

    def __init__(self, cards):
        self._cards = cards

    def get_character_card_by_id(self, character_id):
        return self._cards.get(character_id)


def _card(**overrides):
    base = {
        "id": 1,
        "name": "Vex",
        "system_prompt": "You are Vex.",
        "personality": "sardonic",
        "scenario": "a rooftop at night",
        "first_message": "You again.",
        "post_history_instructions": "Stay in character.",
        "message_example": "<START>",
    }
    base.update(overrides)
    return base


def test_snapshot_copies_every_field_used_in_prompting():
    db = _FakeCharacterDB({1: _card()})
    (snapshot,) = snapshot_cards(db, [1])
    assert snapshot == CardSnapshot(
        id=1,
        name="Vex",
        system_prompt="You are Vex.",
        personality="sardonic",
        scenario="a rooftop at night",
        first_message="You again.",
        post_history_instructions="Stay in character.",
        message_example="<START>",
    )


def test_missing_fields_become_empty_strings_not_none():
    db = _FakeCharacterDB({1: {"id": 1, "name": "Sparse"}})
    (snapshot,) = snapshot_cards(db, [1])
    assert snapshot.system_prompt == ""
    assert snapshot.first_message == ""


def test_order_follows_the_requested_ids():
    db = _FakeCharacterDB({1: _card(id=1, name="A"), 2: _card(id=2, name="B")})
    assert [c.name for c in snapshot_cards(db, [2, 1])] == ["B", "A"]


def test_a_missing_card_raises_naming_the_id():
    db = _FakeCharacterDB({1: _card()})
    with pytest.raises(ValueError, match="99"):
        snapshot_cards(db, [1, 99])


def test_no_ids_raises():
    with pytest.raises(ValueError, match="at least one character"):
        snapshot_cards(_FakeCharacterDB({}), [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_cards.py -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named '...character_probe.cards'`

- [ ] **Step 3: Write minimal implementation**

Append to `models.py`:

```python
@dataclass(frozen=True)
class CardSnapshot:
    """A character card's text, copied at run time.

    Cards live in ``ChaChaNotes_DB`` while runs live in ``Evals_DB``, with no
    foreign keys between them. Copying the text into the run means editing or
    deleting a card later never rewrites what a past run shows -- the same
    provenance rule word_bench applies to snippets.
    """

    id: int
    name: str
    system_prompt: str = ""
    personality: str = ""
    scenario: str = ""
    first_message: str = ""
    post_history_instructions: str = ""
    message_example: str = ""
```

`tldw_chatbook/Evals/character_probe/cards.py`:

```python
"""Read-only card access across the ChaChaNotes/Evals database boundary."""

from __future__ import annotations

from typing import Any, Sequence

from .models import CardSnapshot

#: Card fields copied into a run. Every one participates in prompt assembly;
#: anything not listed here (images, timestamps, versions) is deliberately
#: excluded because it cannot change what the model sees.
_SNAPSHOT_FIELDS = (
    "system_prompt",
    "personality",
    "scenario",
    "first_message",
    "post_history_instructions",
    "message_example",
)


def snapshot_cards(chacha_db: Any, character_ids: Sequence[int]) -> tuple[CardSnapshot, ...]:
    """Copy each requested card's prompting text, in the requested order.

    Args:
        chacha_db: A ``CharactersRAGDB``-shaped handle; only
            ``get_character_card_by_id`` is used, so a fake needs just that.
        character_ids: ``character_cards.id`` values, as INTEGERs.

    Returns:
        tuple[CardSnapshot, ...]: One snapshot per id, in the order given.

    Raises:
        ValueError: If no ids are supplied, or a card cannot be found -- the
            message names the missing id so the caller can drop it from the
            bench rather than guessing which card vanished.
    """
    if not character_ids:
        raise ValueError("A character probe run needs at least one character.")
    snapshots: list[CardSnapshot] = []
    for character_id in character_ids:
        row = chacha_db.get_character_card_by_id(character_id)
        if not row:
            raise ValueError(
                f"Character card {character_id} could not be found; "
                "remove it from the bench or restore the card."
            )
        snapshots.append(
            CardSnapshot(
                id=int(row.get("id", character_id)),
                name=str(row.get("name") or ""),
                **{field: str(row.get(field) or "") for field in _SNAPSHOT_FIELDS},
            )
        )
    return tuple(snapshots)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_cards.py -p no:randomly`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/cards.py tldw_chatbook/Evals/character_probe/models.py Tests/Evals/character_probe/test_cards.py
git commit -m "feat(evals): snapshot character cards across the DB boundary (task-1691 phase 1)"
```

---

### Task 5: Prompt assembly

**Files:**
- Create: `tldw_chatbook/Evals/character_probe/prompt.py`
- Test: `Tests/Evals/character_probe/test_prompt.py`

**Interfaces:**
- Consumes: `CardSnapshot` (Task 4).
- Produces: `compose_system_prompt(card: CardSnapshot, steering: str | None) -> str`; `build_messages(card: CardSnapshot, steering: str | None, scripted_turns: Sequence[str], replies_so_far: Sequence[str]) -> list[dict[str, str]]`.

- [ ] **Step 1: Write the failing test**

```python
from tldw_chatbook.Evals.character_probe.models import CardSnapshot
from tldw_chatbook.Evals.character_probe.prompt import build_messages, compose_system_prompt


def _card(**overrides):
    base = dict(id=1, name="Vex", system_prompt="You are Vex.", first_message="You again.")
    base.update(overrides)
    return CardSnapshot(**base)


def test_steering_is_placed_ahead_of_the_card_prompt():
    composed = compose_system_prompt(_card(), "Answer in English.")
    assert composed.startswith("Answer in English.")
    assert composed.endswith("You are Vex.")


def test_no_steering_yields_the_card_prompt_unchanged():
    assert compose_system_prompt(_card(), None) == "You are Vex."


def test_no_card_prompt_yields_the_steering_alone():
    assert compose_system_prompt(_card(system_prompt=""), "Be brief.") == "Be brief."


def test_first_message_seeds_an_assistant_turn():
    messages = build_messages(_card(), None, ["Hello?"], [])
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "assistant", "content": "You again."}
    assert messages[2] == {"role": "user", "content": "Hello?"}


def test_a_card_without_a_first_message_starts_at_the_user_turn():
    """No synthetic greeting is invented -- that would evaluate text the
    character never had."""
    messages = build_messages(_card(first_message=""), None, ["Hello?"], [])
    assert [m["role"] for m in messages] == ["system", "user"]


def test_prior_replies_accumulate_in_order():
    messages = build_messages(
        _card(), None, ["One", "Two", "Three"], ["Reply one", "Reply two"]
    )
    assert [m["role"] for m in messages] == [
        "system", "assistant", "user", "assistant", "user", "assistant", "user",
    ]
    assert messages[-1] == {"role": "user", "content": "Three"}
    assert messages[-2] == {"role": "assistant", "content": "Reply two"}


def test_personality_and_scenario_reach_the_system_prompt():
    card = _card(system_prompt="You are Vex.", personality="sardonic", scenario="a rooftop")
    composed = compose_system_prompt(card, None)
    assert "sardonic" in composed
    assert "rooftop" in composed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_prompt.py -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named '...character_probe.prompt'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Assembling the messages a character probe sends.

Steering composes AHEAD of the card's own system prompt: steering is a
model-level instruction ("answer in English") and the card is the content it
operates on. Both are preserved -- silently dropping either would evaluate
something other than what the bench describes.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .models import CardSnapshot


def compose_system_prompt(card: CardSnapshot, steering: Optional[str]) -> str:
    """Build the system prompt for one card under one target's steering.

    Args:
        card: The snapshotted card.
        steering: The target's own system prompt, or None when unsteered.

    Returns:
        str: Steering first, then the card's persona text. Empty parts are
        omitted rather than contributing blank lines.
    """
    parts = [
        steering or "",
        card.system_prompt,
        f"Personality: {card.personality}" if card.personality else "",
        f"Scenario: {card.scenario}" if card.scenario else "",
        card.post_history_instructions,
    ]
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def build_messages(
    card: CardSnapshot,
    steering: Optional[str],
    scripted_turns: Sequence[str],
    replies_so_far: Sequence[str],
) -> list[dict[str, str]]:
    """Build the message list for the next turn of a conversation.

    The card's ``first_message`` seeds an opening assistant turn as it does in
    real roleplay. A card without one starts at the user's first scripted turn
    -- no greeting is invented, because inventing one would evaluate text the
    character never had.

    Args:
        card: The snapshotted card.
        steering: The target's own system prompt, or None.
        scripted_turns: All of the probe's user turns.
        replies_so_far: The model's replies to the preceding turns; its length
            determines which scripted turn comes next.

    Returns:
        list[dict[str, str]]: ``role``/``content`` messages, ending with the
        user turn awaiting a reply.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": compose_system_prompt(card, steering)}
    ]
    if card.first_message:
        messages.append({"role": "assistant", "content": card.first_message})
    for index, reply in enumerate(replies_so_far):
        messages.append({"role": "user", "content": scripted_turns[index]})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": scripted_turns[len(replies_so_far)]})
    return messages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_prompt.py -p no:randomly`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/prompt.py Tests/Evals/character_probe/test_prompt.py
git commit -m "feat(evals): character probe prompt assembly with steering precedence (task-1691 phase 1)"
```

---

### Task 6: The conversation runner

**Files:**
- Create: `tldw_chatbook/Evals/character_probe/runner.py`
- Modify: `tldw_chatbook/Evals/character_probe/models.py`
- Test: `Tests/Evals/character_probe/test_runner.py`

**Interfaces:**
- Consumes: `CardSnapshot`, `Probe`, `CharacterProbeConfig`, `build_messages`.
- Produces: `ConversationTurn(user, reply, error="")`; `Conversation(card_id, probe_index, sample_index, target_id, turns, error="")`; `CharacterProbeRunner(chat_fn, cancel_token=None)` with `async def run(cards, probe_set, targets, config, progress=None) -> list[Conversation]`.

`chat_fn` is the injected provider callable with the shape `chat_fn(messages, model, temperature, max_tokens, seed) -> str`. In production it wraps `Chat_Functions.chat_api_call`; in tests it is a fake. **It is synchronous and MUST be dispatched through `asyncio.to_thread`.**

- [ ] **Step 1: Write the failing test**

```python
import asyncio

import pytest

from tldw_chatbook.Evals.character_probe.models import (
    CardSnapshot,
    CharacterProbeConfig,
    Probe,
    ProbeSet,
)
from tldw_chatbook.Evals.character_probe.runner import CharacterProbeRunner


class _FakeChat:
    def __init__(self, reply="ok", fail_on=None):
        self.calls = []
        self._reply = reply
        self._fail_on = fail_on

    def __call__(self, messages, model, temperature, max_tokens, seed):
        self.calls.append(
            {"messages": messages, "model": model, "seed": seed, "temperature": temperature}
        )
        if self._fail_on is not None and len(self.calls) == self._fail_on:
            raise RuntimeError("provider exploded")
        return f"{self._reply}-{len(self.calls)}"


def _card(card_id=1):
    return CardSnapshot(id=card_id, name=f"card{card_id}", system_prompt="sys")


def _config(**overrides):
    base = dict(
        name="b", probe_set_id="ps", character_ids=(1,), target_ids=("t-1",)
    )
    base.update(overrides)
    return CharacterProbeConfig(**base)


def _targets():
    return [{"id": "t-1", "model_id": "m", "system_prompt": None}]


def test_turns_run_in_order_and_each_sees_the_previous_reply():
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One", "Two")),))
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, _targets(), _config())
    )
    (conversation,) = conversations
    assert [t.user for t in conversation.turns] == ["One", "Two"]
    second_call_messages = chat.calls[1]["messages"]
    assert second_call_messages[-2] == {"role": "assistant", "content": "ok-1"}


def test_a_failed_turn_ends_only_its_own_conversation():
    chat = _FakeChat(fail_on=1)
    probe_set = ProbeSet(probes=(Probe(turns=("One",)), Probe(turns=("Two",))))
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, _targets(), _config())
    )
    failed = [c for c in conversations if c.error]
    survived = [c for c in conversations if not c.error]
    assert len(failed) == 1 and "provider exploded" in failed[0].error
    assert len(survived) == 1


def test_partial_turns_are_kept_when_a_later_turn_fails():
    chat = _FakeChat(fail_on=2)
    probe_set = ProbeSet(probes=(Probe(turns=("One", "Two")),))
    (conversation,) = asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, _targets(), _config())
    )
    assert conversation.turns[0].reply == "ok-1"
    assert conversation.error


def test_per_sample_seed_is_offset_so_samples_differ():
    """A single fixed seed would return N identical answers -- see the spec."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    asyncio.run(
        CharacterProbeRunner(chat).run(
            [_card()], probe_set, _targets(), _config(samples_per_cell=3, seed=100)
        )
    )
    assert sorted(call["seed"] for call in chat.calls) == [100, 101, 102]


def test_no_seed_passes_none():
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, _targets(), _config())
    )
    assert chat.calls[0]["seed"] is None


def test_the_grid_covers_cards_probes_targets_and_samples():
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)), Probe(turns=("Two",))))
    targets = [
        {"id": "t-1", "model_id": "m1", "system_prompt": None},
        {"id": "t-2", "model_id": "m2", "system_prompt": None},
    ]
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run(
            [_card(1), _card(2)],
            probe_set,
            targets,
            _config(character_ids=(1, 2), target_ids=("t-1", "t-2"), samples_per_cell=2),
        )
    )
    assert len(conversations) == 2 * 2 * 2 * 2


def test_target_steering_reaches_the_system_prompt():
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    targets = [{"id": "t-1", "model_id": "m", "system_prompt": "Be terse."}]
    asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, targets, _config())
    )
    system = chat.calls[0]["messages"][0]["content"]
    assert system.startswith("Be terse.")


def test_cancelling_stops_scheduling_but_keeps_completed_turns():
    """Cancel cannot abort an in-flight turn (to_thread survives cancellation),
    so it means: start nothing further, keep what finished."""
    from tldw_chatbook.Evals.character_probe.runner import CancelToken

    token = CancelToken()

    def chat(messages, model, temperature, max_tokens, seed):
        token.cancel()  # cancelled while the first turn is in flight
        return "first reply"

    probe_set = ProbeSet(probes=(Probe(turns=("One", "Two", "Three")),))
    (conversation,) = asyncio.run(
        CharacterProbeRunner(chat, cancel_token=token).run(
            [_card()], probe_set, _targets(), _config()
        )
    )
    assert len(conversation.turns) == 1
    assert conversation.turns[0].reply == "first reply"
    assert "Cancelled" in conversation.error


def test_the_blocking_chat_callable_never_runs_on_the_event_loop():
    """chat_api_call is a plain def; calling it inline would freeze the TUI."""
    seen = {}

    def chat(messages, model, temperature, max_tokens, seed):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return "ok"

    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    asyncio.run(CharacterProbeRunner(chat).run([_card()], probe_set, _targets(), _config()))
    assert seen["on_loop"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_runner.py -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named '...character_probe.runner'`

- [ ] **Step 3: Write minimal implementation**

Append to `models.py`:

```python
@dataclass(frozen=True)
class ConversationTurn:
    """One scripted user turn and the model's reply to it."""

    user: str
    reply: str
    error: str = ""


@dataclass(frozen=True)
class Conversation:
    """One cell: a card, a probe, a target, and one sample of the exchange."""

    card_id: int
    probe_index: int
    sample_index: int
    target_id: str
    turns: tuple[ConversationTurn, ...]
    error: str = ""
```

`tldw_chatbook/Evals/character_probe/runner.py`:

```python
"""Runs character probe conversations.

Every provider call goes through ``asyncio.to_thread``: the app's chat gateway
(``Chat_Functions.chat_api_call``) is a plain synchronous ``def``, and calling
it from the event loop would block the whole TUI. Conversations run
concurrently under the bench's ``concurrency`` setting; turns WITHIN a
conversation are strictly sequential, because turn N needs turn N-1's reply.

Cancelling stops SCHEDULING further turns and conversations. It cannot abort a
turn already in flight -- ``to_thread`` survives task cancellation -- so an
in-flight provider call always runs to completion and is recorded.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping, Optional, Sequence

from .models import (
    CardSnapshot,
    CharacterProbeConfig,
    Conversation,
    ConversationTurn,
    ProbeSet,
)
from .prompt import build_messages

#: The injected provider callable. Synchronous by contract.
ChatCallable = Callable[..., str]


class CancelToken:
    """Cancels a whole run; see the module docstring for what that means."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


class CharacterProbeRunner:
    """Runs a bench's full grid of conversations."""

    def __init__(
        self, chat_fn: ChatCallable, cancel_token: Optional[CancelToken] = None
    ) -> None:
        self._chat = chat_fn
        self._cancel = cancel_token or CancelToken()

    async def _run_conversation(
        self,
        card: CardSnapshot,
        probe_index: int,
        turns: Sequence[str],
        target: Mapping[str, Any],
        sample_index: int,
        config: CharacterProbeConfig,
    ) -> Conversation:
        steering = target.get("system_prompt")
        seed = None if config.seed is None else config.seed + sample_index
        collected: list[ConversationTurn] = []
        replies: list[str] = []
        error = ""
        for turn_index, user_turn in enumerate(turns):
            if self._cancel.is_cancelled:
                error = "Cancelled before this turn ran."
                break
            messages = build_messages(card, steering, turns, replies)
            try:
                reply = await asyncio.to_thread(
                    self._chat,
                    messages=messages,
                    model=target.get("model_id"),
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    seed=seed,
                )
            except Exception as exc:  # noqa: BLE001 -- any provider failure ends this conversation only
                error = f"Turn {turn_index + 1} failed: {exc}"
                break
            reply_text = str(reply or "")
            replies.append(reply_text)
            collected.append(ConversationTurn(user=user_turn, reply=reply_text))
        return Conversation(
            card_id=card.id,
            probe_index=probe_index,
            sample_index=sample_index,
            target_id=str(target.get("id")),
            turns=tuple(collected),
            error=error,
        )

    async def run(
        self,
        cards: Sequence[CardSnapshot],
        probe_set: ProbeSet,
        targets: Sequence[Mapping[str, Any]],
        config: CharacterProbeConfig,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[Conversation]:
        """Run every (card x probe x target x sample) conversation.

        Args:
            cards: Snapshotted cards, already resolved.
            probe_set: The scripts to run.
            targets: ``eval_models`` rows; ``system_prompt`` supplies steering.
            config: The bench, supplying concurrency, samples, seed, sampler.
            progress: Optional ``(done, total)`` callback fired as each
                conversation completes.

        Returns:
            list[Conversation]: Every conversation, including failed and
            partial ones -- a failed cell is still evidence and stays
            reviewable.
        """
        jobs = [
            (card, probe_index, probe.turns, target, sample_index)
            for card in cards
            for probe_index, probe in enumerate(probe_set.probes)
            for target in targets
            for sample_index in range(config.samples_per_cell)
        ]
        semaphore = asyncio.Semaphore(config.concurrency)
        done = 0
        total = len(jobs)

        async def _guarded(job) -> Conversation:
            nonlocal done
            card, probe_index, turns, target, sample_index = job
            async with semaphore:
                conversation = await self._run_conversation(
                    card, probe_index, turns, target, sample_index, config
                )
            done += 1
            if progress is not None:
                progress(done, total)
            return conversation

        return list(await asyncio.gather(*(_guarded(job) for job in jobs)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_runner.py -p no:randomly`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/character_probe/runner.py tldw_chatbook/Evals/character_probe/models.py Tests/Evals/character_probe/test_runner.py
git commit -m "feat(evals): character probe conversation runner on a thread bridge (task-1691 phase 1)"
```

---

### Task 7: Conversation, annotation, and review-state persistence

**Files:**
- Modify: `tldw_chatbook/Evals/character_probe/storage.py`
- Modify: `tldw_chatbook/DB/Evals_DB.py` (two new tables + their CRUD)
- Test: `Tests/Evals/character_probe/test_conversation_storage.py`

**Interfaces:**
- Consumes: `Conversation`, `ConversationTurn`, `CharacterProbeConfig`.
- Produces: `conversation_sample_id(card_id: int, probe_index: int, sample_index: int) -> str`; `save_conversations(db, run_group_id, run_ids: Mapping[str, str], conversations) -> None`; `load_conversations(db, run_group_id) -> list[Conversation]`; `annotate_turn(db, run_group_id, card_id, probe_index, sample_index, target_id, turn_index, tags: Sequence[str], note: str) -> None`; `load_turn_annotations(db, run_group_id) -> dict[tuple, dict]`; `mark_conversation_reviewed(db, run_group_id, card_id, probe_index, sample_index, target_id, note: str = "") -> None`; `load_review_state(db, run_group_id) -> dict[tuple, dict]`.

The ordered turn list lives in `eval_results.metadata` JSON, never in `actual_output` — that column is shaped for a single answer and cannot represent a conversation.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import Conversation, ConversationTurn
from tldw_chatbook.Evals.character_probe.storage import (
    annotate_turn,
    conversation_sample_id,
    load_conversations,
    load_review_state,
    load_turn_annotations,
    mark_conversation_reviewed,
    save_conversations,
)


@pytest.fixture
def db():
    return EvalsDB(db_path=":memory:", client_id="test")


def _conversation(card_id=1, probe_index=0, sample_index=0, target_id="t-1"):
    return Conversation(
        card_id=card_id,
        probe_index=probe_index,
        sample_index=sample_index,
        target_id=target_id,
        turns=(
            ConversationTurn(user="One", reply="Reply one"),
            ConversationTurn(user="Two", reply="Reply two"),
        ),
    )


def _seed_run(db):
    task_id = db.create_task(
        name="probe bench", description="", task_type="generation",
        config_data={"bench_type": "character_probe"},
    )
    model_id = db.create_model(name="m", provider="llama_cpp", model_id="m")
    run_id = db.create_run(name="r", task_id=task_id, model_id=model_id)
    return run_id, model_id


def test_sample_id_composes_card_probe_and_sample():
    assert conversation_sample_id(3, 1, 2) == "3:1:2"


def test_conversations_round_trip(db):
    run_id, target_id = _seed_run(db)
    original = _conversation(target_id=target_id)
    save_conversations(db, "rg-1", {target_id: run_id}, [original])
    (loaded,) = load_conversations(db, "rg-1")
    assert loaded.turns == original.turns
    assert loaded.card_id == original.card_id


def test_turns_are_stored_in_metadata_not_actual_output(db):
    """actual_output is shaped for a single answer; a conversation is not one."""
    run_id, target_id = _seed_run(db)
    save_conversations(db, "rg-1", {target_id: run_id}, [_conversation(target_id=target_id)])
    row = db.get_run_results(run_id)[0]
    assert "Reply one" in str(row.get("metadata"))


def test_a_turn_annotation_persists_with_its_tags_and_note(db):
    run_id, target_id = _seed_run(db)
    save_conversations(db, "rg-1", {target_id: run_id}, [_conversation(target_id=target_id)])
    annotate_turn(db, "rg-1", 1, 0, 0, target_id, 1, ["broke-character"], "drifted here")
    stored = load_turn_annotations(db, "rg-1")[(1, 0, 0, target_id, 1)]
    assert stored["tags"] == ["broke-character"]
    assert stored["note"] == "drifted here"


def test_re_annotating_the_same_turn_replaces_it(db):
    run_id, target_id = _seed_run(db)
    annotate_turn(db, "rg-1", 1, 0, 0, target_id, 0, ["refused"], "")
    annotate_turn(db, "rg-1", 1, 0, 0, target_id, 0, ["in-character"], "fine actually")
    stored = load_turn_annotations(db, "rg-1")[(1, 0, 0, target_id, 0)]
    assert stored["tags"] == ["in-character"]


def test_a_conversation_can_be_reviewed_with_no_annotations(db):
    """'Nothing notable' is a real verdict and needs its own home."""
    mark_conversation_reviewed(db, "rg-1", 1, 0, 0, "t-1")
    state = load_review_state(db, "rg-1")[(1, 0, 0, "t-1")]
    assert state["reviewed_at"]
    assert load_turn_annotations(db, "rg-1") == {}


def test_review_state_is_scoped_to_its_run_group(db):
    mark_conversation_reviewed(db, "rg-1", 1, 0, 0, "t-1")
    assert load_review_state(db, "rg-2") == {}


def test_character_probe_never_imports_the_word_bench_measurement_stack():
    """This eval reads generated text only. Importing the capture client,
    normalizer, or canary code would let distribution vocabulary leak into a
    surface that has no distributions -- pinned the way
    Tests/UI/test_evals_bench_editor.py pins the same rule for the editor."""
    import pathlib

    package = pathlib.Path("tldw_chatbook/Evals/character_probe")
    forbidden = ("capture_client", "normalize_logprobs", "CANARY", "top_k", "logprobs")
    for module in package.glob("*.py"):
        source = module.read_text()
        for token in forbidden:
            assert token not in source, f"{module.name} mentions {token}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_conversation_storage.py -p no:randomly`
Expected: FAIL — `ImportError: cannot import name 'conversation_sample_id'`

- [ ] **Step 3: Write minimal implementation**

In `tldw_chatbook/DB/Evals_DB.py`, add two tables to the schema alongside the existing ones (follow the surrounding `CREATE TABLE` style, including `client_id` and timestamp columns):

```python
conn.execute("""
    CREATE TABLE IF NOT EXISTS eval_probe_turn_annotations (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        run_group_id TEXT NOT NULL,
        card_id INTEGER NOT NULL,
        probe_index INTEGER NOT NULL,
        sample_index INTEGER NOT NULL,
        target_id TEXT NOT NULL,
        turn_index INTEGER NOT NULL,
        tags TEXT NOT NULL,          -- JSON list of tag slugs
        note TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        client_id TEXT NOT NULL,
        UNIQUE(run_group_id, card_id, probe_index, sample_index, target_id, turn_index)
    )
""")
conn.execute("""
    CREATE TABLE IF NOT EXISTS eval_probe_review_state (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        run_group_id TEXT NOT NULL,
        card_id INTEGER NOT NULL,
        probe_index INTEGER NOT NULL,
        sample_index INTEGER NOT NULL,
        target_id TEXT NOT NULL,
        note TEXT NOT NULL DEFAULT '',
        reviewed_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        client_id TEXT NOT NULL,
        UNIQUE(run_group_id, card_id, probe_index, sample_index, target_id)
    )
""")
conn.execute(
    "CREATE INDEX IF NOT EXISTS idx_probe_annotations_group "
    "ON eval_probe_turn_annotations (run_group_id)"
)
conn.execute(
    "CREATE INDEX IF NOT EXISTS idx_probe_review_group "
    "ON eval_probe_review_state (run_group_id)"
)
```

Then add matching methods on `EvalsDB` (`upsert_probe_turn_annotation`, `list_probe_turn_annotations`, `upsert_probe_review_state`, `list_probe_review_state`), each parameterized and each following the surrounding methods' transaction style. Append to `character_probe/storage.py`:

```python
def conversation_sample_id(card_id: int, probe_index: int, sample_index: int) -> str:
    """Compose the ``eval_results.sample_id`` for one conversation.

    ``run_id`` already scopes the target (one run row per target, as word
    benches do), so the sample id only needs the remaining three axes.

    Args:
        card_id: The character card's integer id.
        probe_index: Zero-based index of the probe within its set.
        sample_index: Zero-based sample number for this cell.

    Returns:
        str: The composed id, stable across runs.
    """
    return f"{card_id}:{probe_index}:{sample_index}"


def save_conversations(
    db: EvalsDB,
    run_group_id: str,
    run_ids: Mapping[str, str],
    conversations: Sequence[Conversation],
) -> None:
    """Persist every conversation into ``eval_results``.

    The ordered turn list goes into the ``metadata`` JSON, never into
    ``actual_output`` -- that column holds one answer and cannot represent a
    conversation.

    Args:
        db: The evals database handle.
        run_group_id: The group these conversations belong to.
        run_ids: target id -> ``eval_runs`` id for this group.
        conversations: What the runner produced, including failed ones.
    """
    for conversation in conversations:
        db.store_run_result(
            run_id=run_ids[conversation.target_id],
            sample_id=conversation_sample_id(
                conversation.card_id, conversation.probe_index, conversation.sample_index
            ),
            input_data={
                "card_id": conversation.card_id,
                "probe_index": conversation.probe_index,
                "user_turns": [turn.user for turn in conversation.turns],
            },
            actual_output="",
            metadata={
                "run_group_id": run_group_id,
                "turns": [
                    {"user": turn.user, "reply": turn.reply, "error": turn.error}
                    for turn in conversation.turns
                ],
                "error": conversation.error,
            },
        )
```

`load_conversations`, `annotate_turn`, `load_turn_annotations`, `mark_conversation_reviewed`, and `load_review_state` follow the same shape: read through the new `EvalsDB` methods, rebuild the frozen dataclasses, and key the returned dicts by the tuple `(card_id, probe_index, sample_index, target_id[, turn_index])`.

**Note for the implementer:** `db.store_run_result`/`db.get_run_results` may be named differently — check `Evals_DB.py` and use the existing methods rather than adding parallel ones. If a needed reader genuinely does not exist, add the smallest one following the surrounding style.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe/test_conversation_storage.py -p no:randomly`
Expected: PASS (8 tests)

- [ ] **Step 5: Run the whole phase and commit**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/Evals/character_probe Tests/Evals/word_bench -p no:randomly`
Expected: PASS — the word_bench suite must be untouched by this phase.

```bash
git add tldw_chatbook/Evals/character_probe/storage.py tldw_chatbook/DB/Evals_DB.py Tests/Evals/character_probe/
git commit -m "feat(evals): persist probe conversations, turn annotations, and review state (task-1691 phase 1)"
```

---

## Phase 1 exit criteria

- `Tests/Evals/character_probe/` passes in full, and `Tests/Evals/word_bench/` is unchanged.
- A bench can be saved, loaded, and run end to end against a fake chat callable, producing conversations that persist and reload.
- No module under `character_probe/` references word_bench's measurement concepts — no distribution
  vocabulary appears anywhere in this package's source or surface.

  **Amended after the whole-branch review (was: "imports word_bench's capture client, normalizer,
  or canary code").** `character_probe/targets.py` deliberately imports
  `word_bench.storage.model_steering`, the app's single existing reader for a target's steering out
  of an `eval_models` row's `config` JSON. Importing that module pulls word_bench's own imports
  transitively — `storage` → `capture_client` → `normalizer` → `httpx` — so the criterion as
  originally worded is **not** met by the import graph.
  The reuse is correct and stays: duplicating that reader is exactly what produced Critical C1, in
  which the runner read a key no `eval_models` row has ever carried and every real run silently
  dropped its steering. The criterion's *intent* — that none of the measurement stack's ideas leak
  into an eval that reads only generated text — is fully met, so the criterion is amended to say
  what is actually true rather than left silently unmet.

  **The in-repo hygiene test is weaker than the rule it names.**
  `Tests/Evals/character_probe/test_conversation_storage.py::test_character_probe_never_imports_the_word_bench_measurement_stack`
  greps each module's SOURCE TEXT for forbidden tokens. It cannot see an import graph at all, so it
  passes on exactly the situation described above. TASK-1754 tracks both halves of the remedy:
  moving `model_steering`/`_steering_field` (pure functions of a row dict, with no word_bench
  dependencies) into a shared home neither package's stack rides on, and strengthening the test to
  assert on the real import graph — e.g. `sys.modules` after a fresh package import — instead of on
  tokens.

### Forward-looking caveats for Phase 2

Two consequences of Phase 1's fail-loudly choices that a UI author will meet:

- **`_stored_int_field` is strict against JSON floats.** `load_character_bench` now rejects a
  stored `512.0` as a non-integer `max_tokens` (previously `int()` silently truncated it). A form
  control or JSON payload that emits whole numbers as floats will therefore make a bench fail to
  load. Coerce at the UI boundary, not by loosening the loader.
- **`eval_models` has `UNIQUE(name, provider, model_id)`.** The prefix-steered rejection in
  `targets.resolve_target` tells the user to use a chat-mode target instead, and steering is
  immutable per row (there is no `update_model`), so that means creating a NEW row. A Phase 2
  "duplicate this target" affordance must offer a **different name** — reusing the name raises
  `ConflictError`.

## Not in Phase 1 (deliberate)

Phase 2 (import + card selection UI), Phase 3 (review queue), and Phase 4 (summary) follow as separate plans. The starter probe set, the tag vocabulary with kinds, ordering hints, and the Estimate all belong to those phases — Phase 1 stops at the engine.
