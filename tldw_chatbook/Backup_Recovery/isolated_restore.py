"""Explicit separate-profile publication and fresh-process launch (ADR-126)."""

from __future__ import annotations

import os
import subprocess  # nosec B404
import sys
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .archive_models import SealedArchive
    from .journal import Journal, _IsolatedProfile
    from .restore_plan import RestorePlan
from uuid import uuid4

_selected_installation: str | None = None


def installation_client_id() -> str:
    """Return the verified startup identity, preserving ordinary CLI behavior."""
    if _selected_installation is not None:
        return _selected_installation
    from .activation import replacement_installation_id

    return replacement_installation_id() or "tldw_cli_local_instance_v1"


def restore_isolated(
    archive: SealedArchive, plan: RestorePlan, control_root: Path, cancel: Event
) -> str:
    """Publish explicit new roots; retain fences through catalog durability.

    Multi-profile plans register every selected config and return the first source
    profile's new ID in lexical order. Other IDs remain in the operation journal.
    The caller selects existing destination parents disjoint from control storage.
    """
    from . import archive_reader as reader
    from . import bootstrap
    from .control_records import admission_authority, register_pending
    from .journal import Journal, observe_artifact
    from .native_files import create_private_directory, pinned_directory
    from .restore_plan import RestorePlan, _ancestor, recheck_targets
    from .space import require_capacity
    from .staging import stage_restore

    if (
        type(plan) is not RestorePlan
        or plan.mode != "isolated"
        or plan.target is not None
    ):
        raise ValueError("isolated_plan_required")
    doc = reader.verify_sealed(archive, cancel)
    if plan.archive_digest != archive.digest:
        raise ValueError("archive_plan_mismatch")
    recheck_targets(plan)
    selected = dict(plan.restore)
    profiles = []
    for row in doc.files:
        if row.owner_id == "config" and row.logical_id in selected:
            source = row.logical_id.split(":")[1]
            data = dict(plan.selectors).get(f"profile:{source}:paths.data_dir")
            if data is None:
                raise ValueError("config_data_destination_required")
            profiles.append(
                {
                    "source_profile": source,
                    "profile_id": uuid4().hex,
                    "installation_id": uuid4().hex,
                    "config": str(selected[row.logical_id]),
                    "data": str(data),
                }
            )
    profiles.sort(key=lambda row: row["source_profile"])
    if not profiles:
        raise ValueError("isolated_config_required")
    control_root = Path(control_root)
    if not control_root.is_absolute() or ".." in control_root.parts:
        raise ValueError("invalid_control_root")
    root = bootstrap.default_bootstrap_root()
    ancestors = sorted(
        {_ancestor(path) for _, path in (*plan.destinations, *plan.selectors)}
    )
    if any(
        bootstrap._overlap(parent, control_root) or bootstrap._overlap(parent, root)
        for parent in ancestors
    ):
        raise ValueError("isolated_destination_parent_overlaps_control")
    for parent in ancestors:
        with pinned_directory(parent) as fd:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError("private_destination_parent_required")
    if not control_root.exists():
        create_private_directory(control_root)
    operation_id = uuid4().hex
    journal = Journal(control_root, operation_id)
    work = control_root / ("isolated-" + operation_id)
    create_private_directory(work)
    retained = None
    if doc.credential_policy != "exclude":
        if archive.encrypted_source is None:
            raise ValueError("encrypted_acquisition_required")
        require_capacity({control_root: archive.encrypted_source.identity[2]})
        proof = reader.retain_encrypted(archive, work / "credentials.age", cancel)
        if reader._hash(proof.path, cancel) != proof.digest:
            raise ValueError("encrypted_retention_changed")
        ciphertext = observe_artifact(proof.path)
        if reader._identity(proof.path.stat(follow_symlinks=False)) != proof.identity:
            raise ValueError("encrypted_retention_changed")
        retained = {
            "ciphertext": ciphertext,
            "plaintext_digest": proof.plaintext_digest,
            "manifest_digest": proof.manifest_digest,
        }
    candidate = stage_restore(
        archive,
        plan,
        work / "staging",
        cancel,
        journal=journal,
        isolated_profiles=tuple(profiles),
        retained_credentials=retained,
    )
    reader._check(cancel)
    authority = admission_authority(root)
    names = ("isolated." + operation_id,)
    authority.register(names[0], tuple(ancestors))
    selectors = tuple(Path(row["config"]) for row in profiles)
    register_pending(root, operation_id, names, control_root, selectors)
    _finish_isolated(candidate, plan, journal, names, operation_id, cancel)
    return profiles[0]["profile_id"]


def _finish_isolated(
    candidate: Path,
    plan: RestorePlan,
    journal: Journal,
    names: tuple[str, ...],
    generation: str,
    cancel: Event,
) -> None:
    """Finish/retry only this explicit journal under freshly held native authority."""
    from . import bootstrap
    from .archive_reader import _check
    from .control_records import (
        UNBOUND_NAMESPACE,
        _existing_admission_authority,
        _recover_activation_pairs,
    )
    from .journal import _Prepared
    from .publication import _plan_digest, finalize_candidate, publish_candidate

    root = bootstrap.default_bootstrap_root()
    _check(cancel)
    authority = _existing_admission_authority(root)
    with authority.maintenance(
        tuple(names) + (UNBOUND_NAMESPACE,), 30, cancel=cancel
    ) as session:
        with journal._locked(exclusive=False) as parent:
            records = journal._records(parent)
        if not any(row.event == "prepared" for row in records):
            from .bootstrap import _read
            from .native_files import pinned_directory

            with pinned_directory(candidate) as parent:
                descriptor = _read(parent, "candidate.json")
            journal.prepare_publication(
                candidate,
                plan,
                bootstrap_root=root,
                namespaces=tuple(names),
                selectors=tuple(
                    Path(row["config"]) for row in descriptor["isolated_profiles"]
                ),
                generation=generation,
            )
        with journal._locked(exclusive=False) as parent:
            records = journal._records(parent)
        prepared = _Prepared.model_validate(
            next(row.evidence for row in records if row.event == "prepared")
        )
        if (
            prepared.mode != "isolated"
            or prepared.publication is None
            or prepared.publication.bootstrap_root != str(root)
            or prepared.generation != generation
            or tuple(prepared.publication.namespaces) != tuple(names)
            or prepared.publication.plan_digest != _plan_digest(plan)
        ):
            raise ValueError("isolated_recovery_context_invalid")
        # Only this existing operation may complete a split native activation pair.
        # Ordinary readers remain fenced until its exact before/after states match.
        if records[-1].event != "committed":
            _recover_activation_pairs(journal, prepared, session)
        _check(cancel)
        if records[-1].event != "committed":
            publish_candidate(candidate, plan, journal, None)
        finalize_candidate(candidate, plan, journal, session=session)


def _launch_descriptor(profile_id: str, control_root: Path) -> _IsolatedProfile:
    """Return the verified profile entry without changing its launch contract."""
    return _launch_state(profile_id, control_root)[0]


def _launch_state(profile_id: str, control_root: Path):
    """Read actual committed local association; a catalog is only a locator."""
    from . import bootstrap
    from .activation import ActivationStore, _private
    from .journal import Journal, _evidence_digest, _Prepared, observe_artifact
    from .profile_catalog import ProfileCatalog, _name

    catalog = ProfileCatalog(control_root)
    config, data = catalog.resolve(profile_id)
    root = bootstrap.default_bootstrap_root()
    allowed, reason = bootstrap.startup_permission(config, root)
    if not allowed:
        raise ValueError(reason)
    _, profiles, associations = bootstrap._control_records(root)
    registry = bootstrap._registry(root)
    profile = bootstrap._binding(config, profiles, registry)
    association = next(
        (row for row in associations if row["selector"] == str(config)), None
    )
    witness = profile.get("activation") if profile else None
    if witness is None or association is None or association["activation"] != witness:
        raise ValueError("isolated_activation_required")
    if witness["store_root"] != str(control_root / "activation"):
        raise ValueError("isolated_control_mismatch")
    operation_id = witness["operation_id"]
    expected_root = control_root / ("operation-" + bootstrap._key(operation_id))
    if not expected_root.is_dir():
        raise ValueError("isolated_commit_required")
    journal = Journal(control_root, operation_id)
    with journal._locked(exclusive=False) as parent:
        records = journal._records(parent)
    if not records or records[-1].event != "committed":
        raise ValueError("isolated_commit_required")
    prepared = _Prepared.model_validate(
        next(row.evidence for row in records if row.event == "prepared")
    )
    entry = next(
        (row for row in prepared.isolated_profiles if row.profile_id == profile_id),
        None,
    )
    if (
        entry is None
        or (entry.config, entry.data) != (str(config), str(data))
        or prepared.publication.bootstrap_root != str(root)
        or witness["generation"] != prepared.generation
        or witness["namespaces"] != prepared.publication.namespaces
    ):
        raise ValueError("isolated_mapping_changed")
    catalog_event = next(row for row in records if row.event == "catalog_registered")
    actual = observe_artifact(catalog.root / _name(profile_id))
    if actual not in catalog_event.evidence["records"] or records[-1].evidence[
        "catalog_digest"
    ] != _evidence_digest(catalog_event.evidence):
        raise ValueError("isolated_catalog_changed")
    activation = next(
        row.evidence for row in records if row.event == "activation_recorded"
    )
    store = ActivationStore(control_root / "activation")
    with _private(store._generation(prepared.generation)) as parent:
        required = store._required(parent, prepared.generation)
    if required.owners != witness["owners"] or required.owners != activation["owners"]:
        raise ValueError("isolated_activation_changed")
    # Paired admission verifies roots; additionally require the catalog data locator
    # in that exact footprint, without following a writable alias elsewhere.
    if not any(
        data == Path(path) or Path(path) in data.parents for path in profile["roots"]
    ):
        raise ValueError("isolated_data_uncovered")
    return entry, witness


def profile_requirements(profile_id: str, control_root: Path) -> dict:
    """Present checked owner requirements without granting or probing capabilities."""
    from .activation import ActivationStore

    entry, witness = _launch_state(profile_id, control_root)
    required = tuple(witness["owners"])
    store = ActivationStore(Path(witness["store_root"]))
    pending = tuple(
        owner for owner in required if not store.allowed(witness["generation"], owner)
    )
    if _launch_state(profile_id, control_root) != (entry, witness):
        raise ValueError("isolated_activation_changed")
    return {
        "generation": witness["generation"],
        "required_owners": required,
        "pending_owners": pending,
        "requirements_checked": True,
        "needs_setup": bool(pending),
    }


def _launch_environment() -> dict[str, str]:
    """Preserve only platform/terminal selectors; provider and app overrides drop."""
    allowed = (
        "HOME",
        "USER",
        "LOGNAME",
        "PATH",
        "TERM",
        "COLORTERM",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "SYSTEMROOT",
        "WINDIR",
        "TMPDIR",
    )
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env.update(
        HF_HUB_DISABLE_PROGRESS_BARS="1",
        TQDM_DISABLE="1",
        TRANSFORMERS_VERBOSITY="error",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        LOGURU_DIAGNOSE="0",
    )
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    return env


def select_profile(
    profile_id: str, control_root: Path, *, launch_attempt: str | None = None
) -> None:
    """Select once at process startup, before config/services or admission imports."""
    global _selected_installation
    if _selected_installation is not None or "tldw_chatbook.config" in sys.modules:
        raise ValueError("profile_requires_fresh_process")
    entry = _launch_descriptor(profile_id, control_root)
    if launch_attempt is not None:
        from .profile_open import _select

        _select(profile_id, control_root, launch_attempt)
    env = _launch_environment()
    env["TLDW_CONFIG_PATH"] = entry.config
    os.environ.clear()
    os.environ.update(env)
    os.chdir(Path(entry.config).parent)
    import keyring
    from keyring.backends.null import Keyring

    keyring.set_keyring(Keyring())
    _selected_installation = entry.installation_id


def launch_profile(
    profile_id: str, control_root: Path, *, launch_attempt: str | None = None
) -> int:
    """Launch a fresh supported CLI; child independently rechecks all authority."""
    entry = _launch_descriptor(profile_id, control_root)
    extra = []
    if launch_attempt is not None:
        from .profile_open import _expected

        _expected(profile_id, control_root, launch_attempt)
        extra = ["--recovery-launch-attempt", launch_attempt]
    # The executable/module are local constants; verified selectors are separate
    # argv entries and the child repeats admission with a filtered environment.
    return subprocess.call(  # nosec B603
        [
            sys.executable,
            "-P",
            "-m",
            "tldw_chatbook",
            "--recovery-profile",
            profile_id,
            "--recovery-control-root",
            str(control_root),
            *extra,
        ],
        cwd=Path(entry.config).parent,
        env=_launch_environment(),
    )
