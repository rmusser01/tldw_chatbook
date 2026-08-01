"""Persistence for character probe evals.

Probe sets reuse the dataset inline-samples convention that snippets already
use, discriminated by ``metadata["dataset_type"]`` -- the same shape
``config_data.bench_type`` gives ``eval_tasks``. Nothing here writes SQL
directly; every call goes through ``EvalsDB``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from ...DB.Evals_DB import EvalsDB
from ...Evaluations_Interop.evaluation_normalizers import (
    RESERVED_LOCAL_DATASET_SAMPLES_KEY,
)
from .models import CharacterProbeConfig, Probe, ProbeSet

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
    """
    row = db.get_dataset(dataset_id)
    if row is None:
        raise ValueError(f"Probe set {dataset_id!r} could not be found.")
    if not is_probe_set(row):
        raise ValueError(f"Dataset {dataset_id!r} is not a probe set.")
    metadata = row.get("metadata") or {}
    return _samples_to_probe_set(metadata.get(RESERVED_LOCAL_DATASET_SAMPLES_KEY))


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
