"""Persistence for character probe evals.

Probe sets reuse the dataset inline-samples convention that snippets already
use, discriminated by ``metadata["dataset_type"]`` -- the same shape
``config_data.bench_type`` gives ``eval_tasks``. Nothing here writes SQL
directly; every call goes through ``EvalsDB``.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Optional

from ...DB.Evals_DB import EvalsDB
from ...Evaluations_Interop.evaluation_normalizers import (
    RESERVED_LOCAL_DATASET_SAMPLES_KEY,
)
from .models import (
    CardSnapshot,
    CharacterProbeConfig,
    Conversation,
    ConversationTurn,
    Probe,
    ProbeSet,
)
from .prompt import compose_system_prompt
from .targets import ResolvedTarget, resolve_targets

#: Marks a dataset row as holding probes rather than snippets.
PROBE_DATASET_TYPE = "character_probe"


def is_probe_set(dataset_row: Mapping[str, Any]) -> bool:
    """Whether a dataset row holds probes rather than snippets.

    Args:
        dataset_row: A row as returned by ``EvalsDB.get_dataset``/``list_datasets``.

    Returns:
        bool: True when the row is marked as a probe set. A row whose
        ``metadata`` is missing or is not itself a mapping (corrupt data)
        is treated as "not a probe set" rather than raising, so a caller
        that only wants a yes/no answer never has to catch an unrelated
        ``AttributeError`` for that case.
    """
    metadata = dataset_row.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    return metadata.get("dataset_type") == PROBE_DATASET_TYPE


def _probe_set_to_samples(probe_set: ProbeSet) -> list[dict[str, Any]]:
    return [{"turns": list(probe.turns)} for probe in probe_set.probes]


def _samples_to_probe_set(samples: Any, dataset_id: str) -> ProbeSet:
    """Rebuild a ``ProbeSet`` from a dataset row's stored samples list.

    Every corrupt shape raises here, naming the dataset (and the offending
    sample's index). Returning an empty or partial set instead would produce
    exactly the outcome ``load_probe_set``'s docstring promises it prevents:
    a bench that runs, produces nothing, and looks like an authoring mistake
    rather than the damaged row it is.

    Args:
        samples: The value stored under the dataset's samples key.
        dataset_id: The owning dataset, only for the error messages below.

    Returns:
        ProbeSet: The stored probes, in order. An explicitly-stored empty
        list yields an empty ``ProbeSet`` and does NOT raise -- that is a
        deliberately empty probe set, not a missing one.

    Raises:
        ValueError: If the samples key is absent or does not hold a list;
            if any entry is not a mapping; if an entry's ``turns`` is
            missing, empty, or a bare string (a string would otherwise
            iterate character by character into a probe of one-character
            turns); if any turn is not a string; or if the turns violate
            ``Probe``'s own rules (no turns, or an empty/whitespace-only
            turn) -- re-raised with the dataset named.
    """
    if not isinstance(samples, list):
        raise ValueError(
            f"Probe set {dataset_id!r} has no stored samples list "
            f"(found {type(samples).__name__}); the dataset is marked as a "
            "probe set but its probes are missing or corrupt."
        )
    probes: list[Probe] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(
                f"Probe set {dataset_id!r} sample {index} is not a mapping: "
                f"{sample!r}"
            )
        turns = sample.get("turns")
        if isinstance(turns, str):
            raise ValueError(
                f"Probe set {dataset_id!r} sample {index} stores its turns as "
                f"a string ({turns!r}); a probe's turns are a list, and "
                "iterating a string here would silently produce a probe of "
                "one-character turns."
            )
        if not turns or not isinstance(turns, Sequence):
            raise ValueError(
                f"Probe set {dataset_id!r} sample {index} has no turns: "
                f"{turns!r}"
            )
        for turn in turns:
            if not isinstance(turn, str):
                raise ValueError(
                    f"Probe set {dataset_id!r} sample {index} has a "
                    f"non-string turn: {turn!r}"
                )
        try:
            probes.append(Probe(turns=tuple(turns)))
        except ValueError as exc:
            raise ValueError(
                f"Probe set {dataset_id!r} sample {index} is invalid: {exc}"
            ) from None
    return ProbeSet(probes=tuple(probes))


def save_probe_set(
    db: EvalsDB,
    name: str,
    probe_set: ProbeSet,
    dataset_id: str | None = None,
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

    Raises:
        ValueError: If ``dataset_id`` is given but does not identify a live
            dataset row -- ``EvalsDB.update_dataset`` reports this by
            returning ``False`` rather than raising, and silently ignoring
            that would look like a successful save that in fact wrote
            nothing (see ``LocalEvaluationsService.update_dataset`` for the
            same convention).
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
    updated = db.update_dataset(dataset_id, name=name, metadata=metadata)
    if not updated:
        raise ValueError(f"Probe set {dataset_id!r} could not be updated.")
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
        ValueError: Propagated from ``_samples_to_probe_set`` if the row IS
            marked as a probe set but its samples are missing or corrupt --
            same reason, one layer in.
    """
    row = db.get_dataset(dataset_id)
    if row is None:
        raise ValueError(f"Probe set {dataset_id!r} could not be found.")
    if not is_probe_set(row):
        raise ValueError(f"Dataset {dataset_id!r} is not a probe set.")
    metadata = row.get("metadata") or {}
    return _samples_to_probe_set(
        metadata.get(RESERVED_LOCAL_DATASET_SAMPLES_KEY), dataset_id
    )


#: Discriminates a character probe bench from a word bench in ``eval_tasks``.
BENCH_TYPE = "character_probe"


def is_character_bench(task_row: Mapping[str, Any]) -> bool:
    """Whether an ``eval_tasks`` row is a character probe bench.

    Args:
        task_row: A row as returned by ``EvalsDB.get_task``/``list_tasks``.

    Returns:
        bool: True when the row carries this bench type. A row whose
        ``config_data`` is missing or is not itself a mapping (corrupt
        data) is treated as "not a character bench" rather than raising --
        byte-for-byte the guard ``is_probe_set`` above already carries, and
        for the same reason: a caller that only wants a yes/no answer must
        never have to catch an unrelated ``AttributeError``. ``(x or {})``
        alone does NOT provide this: it rescues ``None`` and every other
        falsy value, but a TRUTHY non-mapping -- a string, a list -- sails
        straight through to ``.get`` and raises.
    """
    config_data = task_row.get("config_data")
    if not isinstance(config_data, Mapping):
        return False
    return config_data.get("bench_type") == BENCH_TYPE


def save_character_bench(
    db: EvalsDB, config: CharacterProbeConfig, task_id: Optional[str] = None
) -> str:
    """Persist a character probe bench.

    Mirrors ``word_bench.storage.save_bench``: a new bench creates an
    ``eval_tasks`` row, an existing one updates in place. ``task_type`` is
    ``"generation"`` because ``EvalsDB.create_task``'s ``CHECK`` constraint
    permits only a small fixed set of DB-level literals, none of which name
    this eval type; ``config_data.bench_type`` is the real discriminator,
    the same convention word_bench uses for its own ``task_type`` literal.
    ``config_format`` is ``"custom"`` for the same reason word_bench passes
    it explicitly -- ``create_task`` has no default for that parameter, and
    omitting it (as an earlier draft of this function did) raises a
    ``TypeError`` before a single row is written.

    ``probe_set_id`` lives only in ``config_data``, never passed through as
    ``create_task``'s ``dataset_id``: that column carries a real ``FOREIGN
    KEY`` to ``eval_datasets(id)``, and a probe set is only sometimes an
    ``eval_datasets`` row (see ``save_probe_set`` above) -- forcing every
    bench to reference one there would reject a bench referencing a probe
    set by a not-yet-persisted or external id.

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
        ValueError: If ``task_id`` is given (the edit path) and
            ``update_task`` matched no row -- the bench was deleted (by this
            process or another) between whenever the caller loaded it and
            this call. ``update_task`` itself only returns ``False`` here,
            never raises; silently returning ``task_id`` anyway would tell
            the caller "saved" for a write that persisted nothing, the same
            failure mode ``save_probe_set`` above already guards against for
            ``update_dataset``.
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
        updated = db.update_task(
            task_id,
            name=config.name,
            description=config.description,
            config_data=config_data,
        )
        if not updated:
            raise ValueError(
                f"Bench {task_id!r} could not be updated; it may have been "
                "deleted."
            )
        return task_id
    return db.create_task(
        name=config.name,
        description=config.description,
        task_type="generation",
        config_format="custom",
        config_data=config_data,
    )


def _stored_int_field(
    data: Mapping[str, Any], key: str, default: int, task_id: str
) -> int:
    """One integer field out of a stored ``config_data``, missing-vs-falsy aware.

    A present-but-falsy stored value (``0``, most notably a zero
    ``max_tokens``) is a real value the caller explicitly chose, not an
    absent one -- ``data.get(key) or default`` cannot tell those apart,
    since ``0`` and a missing key are both falsy, and would silently
    replace a deliberately-stored ``0`` with ``default`` on every load.
    Only a genuinely missing key, or an explicit stored ``None``, reads as
    ``default`` here.

    A present value that is negative or not integer-shaped is instead
    treated as corrupt data and rejected outright, matching the
    "fail loudly on a corrupt row" contract this module already applies to
    ``bench_type`` (``is_character_bench``/``load_character_bench`` below)
    and, for the probe-set half of this file, ``is_probe_set``/
    ``load_probe_set`` -- silently coercing a negative or garbage stored
    value into ``default`` would look like a normal, freshly-created bench
    rather than the sign of a hand-edited or otherwise damaged row that it
    actually is. ``0`` itself is not rejected here: whether it is a
    sensible value for a *particular* field (e.g. ``max_tokens``) is that
    field's own concern -- ``CharacterProbeConfig.__post_init__`` already
    enforces a ``>= 1`` floor for ``concurrency``/``samples_per_cell``, so
    a stored ``0`` for either of those still fails loudly, just one layer
    up, the moment this function's caller constructs the config below.

    Args:
        data: The bench's ``config_data``, already parsed into a dict.
        key: The field to read.
        default: Value to use when the key is absent or explicitly ``None``.
        task_id: The owning bench's id, only for the error message below.

    Returns:
        int: The stored value, or ``default``.

    Raises:
        ValueError: If the key is present with a value that is not
            integer-shaped, or is negative -- naming both the bench id and
            the field. "Integer-shaped" means a genuine ``int``, checked by
            type rather than by attempting ``int(value)``: the coercion
            this function used to perform accepted the numeric STRING
            ``"512"`` (``int("512")`` succeeds) and silently truncated the
            float ``2.7`` to ``2``, neither of which this docstring ever
            claimed. ``bool`` is rejected too -- it is an ``int`` subclass,
            but ``True`` is never a stored setting.
    """
    value = data.get(key)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"Bench {task_id!r} has a non-integer {key!r}: {value!r}"
        )
    if value < 0:
        raise ValueError(f"Bench {task_id!r} has a negative {key!r}: {value!r}")
    return value


def _stored_seed(data: Mapping[str, Any], task_id: str) -> Optional[int]:
    """The bench's optional ``seed``, validated rather than passed through.

    ``seed`` is the one genuinely optional numeric setting -- ``None`` means
    "no seed", which is the default and is NOT an error. Everything else
    must be a real ``int``: an unvalidated stored value reaches the runner
    as ``config.seed + sample_index`` on the first cell of a grid, so a
    stored string detonates with a bare ``TypeError`` mid-run, after real
    provider calls have already been paid for and with nothing in the
    message naming the bench or the field.

    A NEGATIVE seed is accepted, unlike every other numeric field here:
    llama.cpp uses ``-1`` for "pick a random seed", so it is a real value a
    user may deliberately store.

    Args:
        data: The bench's ``config_data``, already parsed into a dict.
        task_id: The owning bench's id, only for the error message below.

    Returns:
        Optional[int]: The stored seed, or ``None`` when unseeded.

    Raises:
        ValueError: If the stored seed is present and is not an ``int``
            (``bool`` included), naming both the bench and the field.
    """
    value = data.get("seed")
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Bench {task_id!r} has a non-integer 'seed': {value!r}")
    return value


def _stored_temperature(data: Mapping[str, Any], task_id: str, default: float) -> float:
    """The bench's ``temperature``, with the same named-error contract.

    ``float(data.get("temperature", 0.8))`` -- what this replaced -- raises
    a bare ``TypeError`` on a stored ``null`` and a bare ``ValueError`` on a
    stored string, two lines below ``_stored_int_field``, whose entire
    purpose is to name the bench and the field when that happens.

    Args:
        data: The bench's ``config_data``, already parsed into a dict.
        task_id: The owning bench's id, only for the error message below.
        default: Value to use when the key is absent or explicitly ``None``.

    Returns:
        float: The stored temperature, or ``default``.

    Raises:
        ValueError: If the stored value is not a real number (``bool``
            excluded, being an ``int`` subclass that is never a sampler
            setting) or is negative -- naming the bench and the field.
    """
    value = data.get("temperature")
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Bench {task_id!r} has a non-numeric 'temperature': {value!r}"
        )
    if value < 0:
        raise ValueError(
            f"Bench {task_id!r} has a negative 'temperature': {value!r}"
        )
    return float(value)


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
            config with empty characters and look like data loss. Also
            propagated from ``_stored_int_field`` (see its own docstring)
            for a corrupt ``concurrency``/``samples_per_cell``/
            ``max_tokens``, from ``_stored_seed``/``_stored_temperature``
            for a corrupt ``seed``/``temperature``, and from
            ``CharacterProbeConfig.__post_init__`` for a stored
            ``concurrency``/``samples_per_cell`` below its ``>= 1`` floor.
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
        concurrency=_stored_int_field(data, "concurrency", 1, task_id),
        samples_per_cell=_stored_int_field(data, "samples_per_cell", 1, task_id),
        seed=_stored_seed(data, task_id),
        temperature=_stored_temperature(data, task_id, 0.8),
        max_tokens=_stored_int_field(data, "max_tokens", 512, task_id),
        extra_tags=tuple(data.get("extra_tags") or ()),
    )


#: get_run_results is a paginated API (default limit=1000), same reason
#: word_bench.storage._RESULTS_PAGE_SIZE pages it: a run with more cells
#: than one page would otherwise silently load only part of the group.
_RESULTS_PAGE_SIZE = 1000


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


def _probe_run_snapshot(
    config: CharacterProbeConfig,
    cards: Sequence[CardSnapshot],
    probe_set: ProbeSet,
    targets: Sequence[ResolvedTarget],
) -> dict[str, Any]:
    """The fully-resolved configuration one run group ran with.

    This is what makes a run self-describing, which the design spec requires
    twice over: "at run time the card's actual text is copied into the run
    snapshot", and "the sampler settings are stored in the snapshot so every
    run is self-describing". Without it, ``CardSnapshot``'s whole provenance
    purpose is defeated the moment the run ends -- the cards were copied
    across the database boundary and then thrown away, so a card edited
    afterwards silently rewrites what a past run appears to have asked.

    Card TEXT is stored, not just ids, for the same reason word_bench stores
    snippet text rather than only hashes: a run must still render after its
    cards are edited or deleted, and there are no foreign keys across the
    ``ChaChaNotes_DB``/``Evals_DB`` boundary to prevent either.

    ``composed_system_prompts`` records the ACTUAL composed text per card
    per target -- what the model was really told, steering included, after
    macro resolution. The spec asks for exactly this ("the run snapshot
    records the composed result so what actually ran is never in doubt"),
    and it is not derivable from the parts later: field order, labelling,
    and macro resolution all live in ``prompt.py``, which is free to change.

    Args:
        config: The bench being run.
        cards: The snapshotted cards, in run order.
        probe_set: The scripts being run.
        targets: The resolved targets, in run order.

    Returns:
        dict[str, Any]: A JSON-serialisable snapshot.
    """
    return {
        "bench_name": config.name,
        "bench_description": config.description,
        "probe_set_id": config.probe_set_id,
        "sampler": {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
            "samples_per_cell": config.samples_per_cell,
            "concurrency": config.concurrency,
        },
        "extra_tags": list(config.extra_tags),
        "targets": [
            {
                "id": target.id,
                "name": target.name,
                "provider": target.provider,
                "model_id": target.model_id,
                "system_prompt": target.steering,
            }
            for target in targets
        ],
        "cards": [asdict(card) for card in cards],
        "probes": [{"turns": list(probe.turns)} for probe in probe_set.probes],
        "composed_system_prompts": {
            # Card ids are INTEGERs and JSON object keys are strings, so
            # these come back out of storage as strings; a reader keying by
            # card id must str() it, exactly as this writer does.
            str(card.id): {
                target.id: compose_system_prompt(card, target.steering)
                for target in targets
            }
            for card in cards
        },
    }


def create_probe_run_group(
    db: EvalsDB,
    task_id: str,
    config: CharacterProbeConfig,
    cards: Sequence[CardSnapshot],
    probe_set: ProbeSet,
    targets: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, str]]:
    """Open a run group for one probe run: one ``eval_runs`` row per target.

    Mirrors ``word_bench.storage.create_run_group`` seam for seam -- a run
    per target sharing a ``run_group_id``, with the run's own snapshot in
    ``config_overrides["snapshot"]`` -- so the two bench types stay legible
    against each other and a later reader can find a whole group with one
    ``list_runs(run_group_id=...)`` call.

    Call this BEFORE running, then pass the returned ``run_ids`` to
    ``save_conversations``. The snapshot is written at launch, from the
    resolved cards and targets the run is about to use, so it records what
    actually ran rather than what the (mutable) bench row says afterwards.

    Args:
        db: The evals database handle.
        task_id: The bench's ``eval_tasks`` id.
        config: The bench being run.
        cards: The snapshotted cards, in run order.
        probe_set: The scripts being run.
        targets: ``eval_models`` rows for the run's targets.

    Returns:
        tuple[str, dict[str, str]]: The new ``run_group_id``, and
        target id -> ``eval_runs`` id for every target.

    Raises:
        ValueError: Propagated from ``resolve_targets`` for an empty target
            list, duplicate target ids, or a malformed/prefix-steered row.
        InputError: From ``EvalsDB.create_run`` if ``task_id`` or a target
            id does not name a live row -- a target id IS an ``eval_models``
            id, which ``create_run`` validates.
    """
    resolved = resolve_targets(targets)
    group_id = uuid.uuid4().hex
    snapshot = _probe_run_snapshot(config, cards, probe_set, resolved)
    # One conversation per card x probe x sample, per target -- the target
    # axis is the run itself, so it is NOT part of a single run's count.
    per_target_cells = len(cards) * len(probe_set.probes) * config.samples_per_cell
    run_ids: dict[str, str] = {}
    for target in resolved:
        run_id = db.create_run(
            name=f"{config.name or 'Character probe'} · "
            f"{target.name or target.model_id}",
            task_id=task_id,
            model_id=target.id,
            config_overrides={"snapshot": snapshot, "target_id": target.id},
        )
        db.update_run(
            run_id, {"run_group_id": group_id, "total_samples": per_target_cells}
        )
        run_ids[target.id] = run_id
    return group_id, run_ids


def load_probe_run_snapshot(db: EvalsDB, run_group_id: str) -> dict[str, Any]:
    """Read back the snapshot ``create_probe_run_group`` wrote.

    Every run in a group carries the same snapshot (written once, at
    launch), so the first run found answers for the group -- the same
    approach ``word_bench.storage._load_run_group_snapshot`` takes, and for
    the same reason: the snapshot, never the live ``eval_tasks`` row, is
    what a past run must render from.

    Args:
        db: The evals database handle.
        run_group_id: The group to read.

    Returns:
        dict[str, Any]: The stored snapshot. ``{}`` for a group whose runs
        were created some other way and carry no snapshot, rather than
        raising -- absent provenance is not a corrupt group.

    Raises:
        ValueError: If no runs share this ``run_group_id`` -- naming the
            group, rather than returning ``{}`` and letting a caller mistake
            "this group does not exist" for "this group has no snapshot".
    """
    runs = db.list_runs(run_group_id=run_group_id, limit=10_000)
    if not runs:
        raise ValueError(f"No runs found for run group {run_group_id!r}.")
    overrides = runs[0].get("config_overrides") or {}
    snapshot = overrides.get("snapshot")
    return snapshot if isinstance(snapshot, Mapping) else {}


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

    Every run named in ``run_ids`` is stamped with ``run_group_id`` (via
    ``EvalsDB.update_run``), mirroring ``word_bench.storage.
    create_run_group``'s own convention of tagging every run in a group so
    later reads can find them all with one ``list_runs(run_group_id=...)``
    call. This runs unconditionally, even for a target whose conversations
    all failed before producing a turn -- ``load_conversations`` below has
    no other way to discover which runs belong to this group.

    EVERY check this function makes runs before the FIRST write. An unknown
    target used to be detected inside the write loop, so a group could be
    left half-committed -- earlier conversations stored, the rest not, and
    the run group loading back as if it were complete, since nothing
    distinguishes a missing conversation from one that was never meant to
    exist.

    That is a claim about VALIDATION ordering only, not about atomicity: the
    write loop itself is NOT transactional. ``EvalsDB.store_result`` commits
    each row in its own ``with conn`` block, so an error raised by the
    database mid-loop still leaves the preceding rows committed. The one
    reachable way to provoke that is two conversations sharing a
    ``sample_id`` -- the same ``(card_id, probe_index, sample_index)`` under
    one target -- which trips ``eval_results``'s UNIQUE
    ``(run_id, sample_id)`` on the second write. A runner-produced grid
    cannot contain such a pair (every cell is one point of a product of
    distinct axes), so this is a hand-assembled-input hazard rather than a
    live one; making the loop genuinely atomic needs a batch write on
    ``EvalsDB``, which is not this function's to add.

    Args:
        db: The evals database handle.
        run_group_id: The group these conversations belong to.
        run_ids: target id -> ``eval_runs`` id for this group.
        conversations: What the runner produced, including failed ones.

    Raises:
        ValueError: If a conversation's ``target_id`` has no entry in
            ``run_ids`` -- storing it would otherwise raise an opaque
            ``KeyError`` naming neither the conversation nor the run group.
        ValueError: If a run id in ``run_ids`` does not name a live run.
            ``EvalsDB.update_run`` returns nothing and silently no-ops on an
            unmatched id, so a stale run id would otherwise look like a
            successful stamp that in fact wrote nothing -- the same failure
            mode ``save_probe_set``/``save_character_bench`` above already
            guard against for ``update_dataset``/``update_task``.
    """
    unknown = sorted(
        {c.target_id for c in conversations if c.target_id not in run_ids}
    )
    if unknown:
        raise ValueError(
            f"No run id supplied for target(s) {unknown!r} in run group "
            f"{run_group_id!r}; run_ids covers {sorted(run_ids)!r}."
        )
    for run_id in set(run_ids.values()):
        if db.get_run(run_id) is None:
            raise ValueError(
                f"Run {run_id!r} (in run_ids for run group {run_group_id!r}) "
                "does not exist; it may have been deleted."
            )

    for run_id in set(run_ids.values()):
        db.update_run(run_id, {"run_group_id": run_group_id})

    for conversation in conversations:
        db.store_result(
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


def _turn_field(payload: Mapping[str, Any], field: str, row_id: Any) -> str:
    """One string field out of a stored turn payload, missing-vs-corrupt aware.

    Args:
        payload: One entry of a stored ``metadata["turns"]`` list.
        field: ``"user"``, ``"reply"``, or ``"error"``.
        row_id: The owning ``eval_results`` row's id, only for the error
            message below.

    Returns:
        str: The field's value, or ``""`` if the key is genuinely absent.

    Raises:
        ValueError: If the key is present with a non-``str`` value -- naming
            both the row and the offending field, rather than handing a
            malformed value on to ``ConversationTurn``.
    """
    value = payload.get(field, "")
    if not isinstance(value, str):
        raise ValueError(
            f"eval_results row {row_id!r} has a non-string {field!r} in a "
            f"stored turn: {value!r}"
        )
    return value


def _conversation_from_row(row: Mapping[str, Any], target_id: str) -> Conversation:
    """Rebuild one ``Conversation`` from its ``eval_results`` row.

    Args:
        row: A row as returned by ``EvalsDB.get_run_results``, already
            JSON-decoded.
        target_id: The owning run's target, supplied by the caller
            (``load_conversations``) rather than read from the row itself --
            a target is implicit in which run a row belongs to, per
            ``save_conversations``'s own docstring.

    Returns:
        Conversation: The reconstructed conversation.

    Raises:
        ValueError: If ``sample_id`` is not shaped like
            ``conversation_sample_id`` produces, or if ``metadata`` is
            missing its turn list or carries a non-string field -- naming
            the offending row rather than raising an opaque ``KeyError``/
            ``TypeError`` out of the reconstruction below.
    """
    row_id = row.get("id")
    sample_id = row.get("sample_id") or ""
    parts = sample_id.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"eval_results row {row_id!r} has a malformed sample_id "
            f"{sample_id!r}; expected 'card:probe:sample'."
        )
    try:
        card_id, probe_index, sample_index = (int(part) for part in parts)
    except ValueError:
        raise ValueError(
            f"eval_results row {row_id!r} has a non-integer sample_id "
            f"{sample_id!r}."
        ) from None

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"eval_results row {row_id!r} has a non-mapping metadata; "
            "expected a conversation payload."
        )
    turns_payload = metadata.get("turns")
    if not isinstance(turns_payload, list):
        raise ValueError(
            f"eval_results row {row_id!r} is missing its turns list in "
            "metadata."
        )
    turns = []
    for turn in turns_payload:
        if not isinstance(turn, Mapping):
            raise ValueError(
                f"eval_results row {row_id!r} has a non-mapping turn entry: "
                f"{turn!r}"
            )
        turns.append(
            ConversationTurn(
                user=_turn_field(turn, "user", row_id),
                reply=_turn_field(turn, "reply", row_id),
                error=_turn_field(turn, "error", row_id),
            )
        )

    error = metadata.get("error", "")
    if not isinstance(error, str):
        raise ValueError(
            f"eval_results row {row_id!r} has a non-string metadata error: "
            f"{error!r}"
        )

    return Conversation(
        card_id=card_id,
        probe_index=probe_index,
        sample_index=sample_index,
        target_id=target_id,
        turns=tuple(turns),
        error=error,
    )


def load_conversations(db: EvalsDB, run_group_id: str) -> list[Conversation]:
    """Rebuild every conversation stored under one run group.

    Mirrors ``word_bench.storage.load_grid``'s own approach: find every run
    sharing ``run_group_id`` (via ``EvalsDB.list_runs``), then drain each
    run's ``eval_results`` a page at a time so a group with more cells than
    one page is never silently truncated.

    Args:
        db: The evals database handle.
        run_group_id: The run group to read, as passed to
            ``save_conversations``.

    Returns:
        list[Conversation]: Every conversation stored under this group,
        across every target. A run group with no runs at all (nothing has
        been saved for it yet) returns an empty list rather than raising --
        the review queue may probe a group speculatively.

    Raises:
        ValueError: Propagated from ``_conversation_from_row`` for a row
            whose ``sample_id`` or ``metadata`` is corrupt.
    """
    conversations: list[Conversation] = []
    for run in db.list_runs(run_group_id=run_group_id, limit=10_000):
        target_id = run["model_id"]
        offset = 0
        while True:
            page = db.get_run_results(
                run["id"], limit=_RESULTS_PAGE_SIZE, offset=offset
            )
            if not page:
                break
            for row in page:
                conversations.append(_conversation_from_row(row, target_id))
            if len(page) < _RESULTS_PAGE_SIZE:
                break
            offset += _RESULTS_PAGE_SIZE
    return conversations


def annotate_turn(
    db: EvalsDB,
    run_group_id: str,
    card_id: int,
    probe_index: int,
    sample_index: int,
    target_id: str,
    turn_index: int,
    tags: Sequence[str],
    note: str,
) -> None:
    """Record or replace one conversation turn's reviewer annotation.

    This is the "it broke character on the third turn" home -- keyed all the
    way down to ``turn_index``, distinct from ``mark_conversation_reviewed``
    below, which has no turn axis at all. Re-annotating the same turn
    replaces its tags and note rather than accumulating a second row (see
    ``EvalsDB.upsert_probe_turn_annotation``).

    Args:
        db: The evals database handle.
        run_group_id: The run group the conversation belongs to.
        card_id: The character card's id.
        probe_index: The probe's zero-based index within its probe set.
        sample_index: The zero-based sample number for this cell.
        target_id: The target's id, as used in ``save_conversations``'s
            ``run_ids``.
        turn_index: The zero-based turn within the conversation.
        tags: Tag slugs describing this turn.
        note: Free-text reviewer note for this turn.
    """
    db.upsert_probe_turn_annotation(
        run_group_id=run_group_id,
        card_id=card_id,
        probe_index=probe_index,
        sample_index=sample_index,
        target_id=target_id,
        turn_index=turn_index,
        tags=list(tags),
        note=note,
    )


def load_turn_annotations(
    db: EvalsDB, run_group_id: str
) -> dict[tuple[int, int, int, str, int], dict[str, Any]]:
    """Every turn annotation recorded for one run group.

    Args:
        db: The evals database handle.
        run_group_id: The run group to read.

    Returns:
        A dict keyed by ``(card_id, probe_index, sample_index, target_id,
        turn_index)``, each value ``{"tags": [...], "note": str}``. Empty
        for a run group with no annotations, never raising.
    """
    return {
        (
            row["card_id"],
            row["probe_index"],
            row["sample_index"],
            row["target_id"],
            row["turn_index"],
        ): {"tags": row["tags"], "note": row["note"]}
        for row in db.list_probe_turn_annotations(run_group_id)
    }


def mark_conversation_reviewed(
    db: EvalsDB,
    run_group_id: str,
    card_id: int,
    probe_index: int,
    sample_index: int,
    target_id: str,
    note: str = "",
) -> None:
    """Mark one whole conversation reviewed.

    This is the ONLY home for "I read this and nothing was notable" -- a
    conversation is markable reviewed with zero turn annotations, which is a
    common, meaningful outcome the review queue's progress count depends on,
    not an edge case. Re-marking a reviewed conversation replaces its note
    and refreshes ``reviewed_at`` (see
    ``EvalsDB.upsert_probe_review_state``).

    Args:
        db: The evals database handle.
        run_group_id: The run group the conversation belongs to.
        card_id: The character card's id.
        probe_index: The probe's zero-based index within its probe set.
        sample_index: The zero-based sample number for this cell.
        target_id: The target's id, as used in ``save_conversations``'s
            ``run_ids``.
        note: Free-text reviewer note for the whole conversation.
    """
    db.upsert_probe_review_state(
        run_group_id=run_group_id,
        card_id=card_id,
        probe_index=probe_index,
        sample_index=sample_index,
        target_id=target_id,
        note=note,
    )


def load_review_state(
    db: EvalsDB, run_group_id: str
) -> dict[tuple[int, int, int, str], dict[str, Any]]:
    """Every conversation's review state for one run group.

    Args:
        db: The evals database handle.
        run_group_id: The run group to read.

    Returns:
        A dict keyed by ``(card_id, probe_index, sample_index, target_id)``,
        each value ``{"reviewed_at": str, "note": str}``. Empty for a run
        group with nothing marked reviewed, never raising.
    """
    return {
        (
            row["card_id"],
            row["probe_index"],
            row["sample_index"],
            row["target_id"],
        ): {"reviewed_at": row["reviewed_at"], "note": row["note"]}
        for row in db.list_probe_review_state(run_group_id)
    }
