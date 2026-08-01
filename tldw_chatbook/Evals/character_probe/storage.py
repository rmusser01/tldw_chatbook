"""Persistence for character probe evals.

Probe sets reuse the dataset inline-samples convention that snippets already
use, discriminated by ``metadata["dataset_type"]`` -- the same shape
``config_data.bench_type`` gives ``eval_tasks``. Nothing here writes SQL
directly; every call goes through ``EvalsDB``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...DB.Evals_DB import EvalsDB
from ...Evaluations_Interop.evaluation_normalizers import (
    RESERVED_LOCAL_DATASET_SAMPLES_KEY,
)
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
