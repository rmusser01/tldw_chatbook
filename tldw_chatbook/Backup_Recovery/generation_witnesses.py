"""Passive paired-generation reader shared by admitted local owners."""

from pathlib import Path

from . import bootstrap
from .activation import ActivationStore, _private, _source_scope_admitted_from_records


def _witnesses(path, lease):
    """Check paired generations against this actual admitted storage group."""
    root, names = lease.execution_context(path)
    selected = bootstrap.effective_config_path()
    with bootstrap._control_observation(root) as (records, registry):
        if path is not None and not _source_scope_admitted_from_records(
            names or (), path, records, registry
        ):
            raise ValueError("projection_source_scope_unavailable")
        return _paired_witnesses_from_records(
            path, root, names, selected, records, registry
        )


def _paired_witnesses(path, root, names, selected):
    """Read paired local control metadata; this grants no storage admission."""
    with bootstrap._control_observation(root) as (records, registry):
        return _paired_witnesses_from_records(
            path, root, names, selected, records, registry
        )


def _paired_witnesses_from_records(path, root, names, selected, records, registry):
    pending, profiles, associations = records
    if not bootstrap._startup_permission_from_records(
        selected, root, pending, profiles, registry
    )[0]:
        raise ValueError("projection_generation_unavailable")
    held = set(names or ())
    profile = next((p for p in profiles if p["selector"] == str(selected)), None)
    if profile:
        held.update(profile["namespaces"])
    results = []
    seen = {}
    for selector in sorted({p["selector"] for p in profiles + associations}):
        profile = next((p for p in profiles if p["selector"] == selector), None)
        association = next((p for p in associations if p["selector"] == selector), None)
        witness = profile.get("activation") if profile else None
        other = association["activation"] if association else None
        candidates = (witness, other)
        relevant = (
            selector == str(selected)
            or any(
                item and held.intersection(item["namespaces"]) for item in candidates
            )
            or path is not None
            and any(
                bootstrap._overlap(path, Path(p))
                for p in (profile or {}).get("roots", [])
            )
        )
        if not relevant or witness is None and other is None:
            continue
        if witness is None or witness != other or names is None:
            raise ValueError("projection_generation_unavailable")
        if registry is None or not set(witness["namespaces"]) <= registry.keys():
            raise ValueError("projection_generation_unavailable")
        roots = sorted(
            {p for name in witness["namespaces"] for p in registry[name]["roots"]}
        )
        if roots != profile["roots"]:
            raise ValueError("projection_generation_unavailable")
        for name in witness["namespaces"]:
            if name in seen and not bootstrap._same_activation_generation(
                seen[name], witness
            ):
                raise ValueError("projection_generation_unavailable")
            seen[name] = witness
        store = ActivationStore(Path(witness["store_root"]))
        with (
            _private(store.root),
            _private(store._generation(witness["generation"])) as parent,
        ):
            if (
                store._required(parent, witness["generation"]).owners
                != witness["owners"]
            ):
                raise ValueError("projection_generation_unavailable")
        if witness not in results:
            results.append(witness)
    return results
