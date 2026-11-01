"""Passive paired-generation reader shared by admitted local owners."""

import sys
from pathlib import Path

from . import activation as _activation
from . import bootstrap
from .activation import ActivationStore, _private, _source_scope_admitted


def _witnesses(path, lease):
    """Check paired local generations against this actual admitted storage group."""
    from . import storage_admission as storage

    root, names = lease.execution_context(path)
    selected = bootstrap.effective_config_path()
    hold = storage._ordinary_hold(lease=lease)
    key = ("witness", str(selected), str(path))
    before = storage._derived_before(hold, key)
    stock = _witness_metadata_current()
    reused, value = (
        storage._derived_reuse(hold, key, before) if stock else (False, None)
    )
    stock = stock and _witness_metadata_current()
    if not stock:
        reused = False
        storage._note_derived(hold, key, None, None, before)
    if reused:
        return value
    if path is not None and not _source_scope_admitted(root, names or (), path):
        raise ValueError("projection_source_scope_unavailable")
    result = _paired_witnesses(path, root, names, selected)
    evidence = storage._metadata_evidence(hold, (() if path is None else (path,)))
    storage._note_derived(
        hold, key,
        evidence if stock and _witness_metadata_current() else None,
        result, before,
    )
    return result


def _paired_witnesses(path, root, names, selected):
    """Read paired local control metadata; this grants no storage admission."""
    from .mobile_preview_budget import current, path_cost

    scope = current()
    if scope is not None:
        scope.read_budget.reserve(4096 + path_cost(root) + path_cost(selected))
    if not bootstrap.startup_permission(selected, root)[0]:
        from .storage_admission import _mobile_discovery_pending

        if not _mobile_discovery_pending(root, names, selected):
            raise ValueError("projection_generation_unavailable")
    _, profiles, associations = bootstrap._control_records(root)
    registry = bootstrap._registry(root)
    if scope is not None:
        scope.read_budget.value(profiles, text_copies=2)
        scope.read_budget.value(associations, text_copies=2)
        scope.read_budget.value(registry, text_copies=2)
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


def _witness_metadata_current():
    """Decline optional metadata reuse when its original readers changed."""
    if any(owner.get(name) is not value for owner, name, value in _WITNESS_METADATA_BINDINGS):
        return False
    for cls, lookup, dictionary in _WITNESS_METADATA_CLASSES:
        if (vars(cls).get("__getattribute__") is not lookup
                or vars(cls).get("__dict__") is not dictionary):
            return False
    for module, namespace, filename, spec, origin in _WITNESS_METADATA_MODULES:
        if (sys.modules.get(namespace["__name__"]) is not module
                or vars(module) is not namespace or namespace.get("__file__") != filename
                or namespace.get("__spec__") is not spec or getattr(spec, "origin", None) != origin):
            return False
    for owner, name, descriptor, original, code, namespace, defaults, keywords, items, closure, values in _WITNESS_METADATA_READERS:
        if (owner.get(name) is not descriptor or original.__code__ is not code
                or original.__globals__ is not namespace or original.__defaults__ is not defaults
                or original.__kwdefaults__ is not keywords or original.__closure__ is not closure
                or keywords is not None and (len(keywords) != len(items)
                    or any(key not in keywords or keywords[key] is not value for key, value in items))
                or any(cell.cell_contents is not value for cell, value in values)):
            return False
    return True


# Exact readers skipped by this owner's optional derivation; no native authority.
_WITNESS_METADATA_BINDINGS = (
    (globals(), "bootstrap", bootstrap),
    (globals(), "_activation", _activation),
    (globals(), "ActivationStore", ActivationStore),
    (vars(_activation), "ActivationStore", ActivationStore),
    (vars(_activation), "_Requirement", _activation._Requirement),
    (vars(_activation), "bootstrap", bootstrap),
)
_WITNESS_METADATA_CLASSES = tuple(
    (cls, vars(cls).get("__getattribute__"), vars(cls).get("__dict__"))
    for cls in (ActivationStore,)
)
_WITNESS_METADATA_READERS = tuple(
    (row_owner, row_name, descriptor, function, function.__code__, function.__globals__,
     function.__defaults__, function.__kwdefaults__, tuple((function.__kwdefaults__ or {}).items()),
     function.__closure__, tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()))
    for owner, names in (
        (globals(), ("_witnesses", "_paired_witnesses", "_source_scope_admitted", "_private", "_witness_metadata_current")),
        (vars(_activation), ("_source_scope_admitted", "_private", "_key", "_identifier", "_read", "lexical_path", "pinned_directory")),
        (vars(ActivationStore), ("__init__", "_generation", "_required")),
        (vars(bootstrap), (
            "_records", "_control_records", "_registry", "_registry_read_lock",
            "pinned_directory", "_read", "_strings", "_paths", "_key",
            "_activation_witness", "_validate_ancestry_record",
            "_open_verified_parent", "_native_close",
            "startup_permission", "lexical_path", "_overlap", "identity_view",
            "inode_token", "inode_token_for", "_same_activation_generation",
        )),
    )
    for name in names
    for original in (owner[name],)
    for callback in (original.__func__ if type(original) is staticmethod else original,)
    for row_owner, row_name, descriptor, function in (
        (owner, name, original, callback),
        *(((vars(callback), "__wrapped__", callback.__wrapped__, callback.__wrapped__),)
          if hasattr(callback, "__wrapped__") else ()),
    )
)
_WITNESS_METADATA_MODULES = tuple(
    (module, vars(module), module.__file__, module.__spec__, getattr(module.__spec__, "origin", None))
    for name in dict.fromkeys(row[5]["__name__"] for row in _WITNESS_METADATA_READERS)
    for module in (sys.modules[name],)
)
