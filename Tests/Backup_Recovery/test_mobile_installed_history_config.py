"""Installed config physics; full mobile source authentication is public-test owned."""

import copy
import json
from contextlib import contextmanager
from dataclasses import replace
from threading import Event

import pytest

from Tests.Backup_Recovery.test_held_sqlite_rollback import (
    replacement_case,
    run_capture,
)
from Tests.Backup_Recovery.test_mobile_empty_history import _case, _mapping_proof
from Tests.private_profile import private_profile_test
from tldw_chatbook.Backup_Recovery import publication
from tldw_chatbook.Backup_Recovery.mobile_empty_history import _empty_history_members
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget


@pytest.mark.parametrize("bad", ("hash", "profile", "duplicate", "mutable", "extra"))
def test_installed_config_data_is_exact_profile_bound_semantics(tmp_path, bad):
    document, target, item, _ = _case(tmp_path)
    config = next(row for row in target.items if row.owner == "config")
    proof = _mapping_proof(config)
    raw = config.path.read_bytes()
    data = ((config.logical_id, raw),)
    if bad == "hash":
        data = ((config.logical_id, raw + b"\n"),)
    elif bad == "profile":
        data = (("profile:foreign:config", raw),)
    elif bad == "duplicate":
        data = data + data
    elif bad == "mutable":
        data = ((config.logical_id, bytearray(raw)),)
    elif bad == "extra":
        data = data + (("profile:other:config", raw),)
    with pytest.raises(ValueError, match="^mobile_canonical_config_data_changed$"):
        _empty_history_members(
            document, target, _TraceReadBudget(), proof, _config_data=data
        )
    assert _empty_history_members(
        document,
        target,
        _TraceReadBudget(),
        proof,
        _config_data=((config.logical_id, raw),),
    ) == {item.logical_id: item}


def test_default_mapping_still_rejects_content_identical_replacement(tmp_path):
    document, target, item, _ = _case(tmp_path)
    config = next(row for row in target.items if row.owner == "config")
    proof = _mapping_proof(config)
    raw = config.path.read_bytes()
    replacement = config.path.with_name("replacement.toml")
    replacement.write_bytes(raw)
    replacement.chmod(0o600)
    replacement.replace(config.path)
    with pytest.raises(ValueError, match="^mobile_canonical_selector_review_required$"):
        _empty_history_members(document, target, _TraceReadBudget(), proof)
    # Pure semantic input ONLY; this call itself is not native publication proof.
    assert _empty_history_members(
        document,
        target,
        _TraceReadBudget(),
        proof,
        _config_data=((config.logical_id, raw),),
    ) == {item.logical_id: item}


@contextmanager
def _published_case(
    tmp_path, monkeypatch, helper_resource_root, *, validated=False, applied_mode=None
):
    """Real ordinary publication/session; controlled mobile classification facts.

    This isolates native installed-copy checks, not source archive authentication.
    The separate public mobile scenario provides that complete proof.
    """
    from tldw_chatbook.Backup_Recovery import crypto
    from tldw_chatbook.Backup_Recovery.journal import (
        _CandidateReceipt,
        _MobileCaptureVerified,
        _Prepared,
    )
    from tldw_chatbook.Backup_Recovery.mobile_canonical_copy import _CanonicalCopy
    from tldw_chatbook.Backup_Recovery.models import FileMetadata, StorageItem
    from tldw_chatbook.Backup_Recovery.restore_plan import LocalSnapshotSource

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    metadata = (
        None
        if applied_mode is None
        else {"version": 1, "mode": applied_mode, "mtime_ns": 123456789000}
    )
    with replacement_case(tmp_path, monkeypatch, config_metadata=metadata) as case:
        candidate, original_plan, journal, session, _, selector = case
        config = next(
            row for row in original_plan.target.items if row.owner == "config"
        )
        proof = _mapping_proof(config)
        descriptor = json.loads((candidate / "candidate.json").read_bytes())
        row = next(
            row
            for row in descriptor["artifacts"]
            if row["logical_id"] == config.logical_id
        )
        # Original canonical source facts are controlled unit inputs. Current
        # native publication, journal, real session and copied config are genuine.
        proof = replace(
            proof,
            prepared=proof.prepared.model_copy(
                update={
                    "selector_review": [
                        proof.prepared.selector_review[0].model_copy(
                            update={"fingerprint": row["sha256"]}
                        )
                    ],
                }
            ),
        )
        rollback = run_capture(case, tmp_path)
        publication.publish_candidate(
            candidate, original_plan, journal, rollback, session=session
        )
        if validated:
            publication._validate_installed(
                journal, candidate, original_plan, session=session
            )
        root = selector.parent / "preserved-history"
        root.mkdir(mode=0o700)
        history = StorageItem(
            "agents.history",
            "profile:profile:agents.history:controlled",
            root,
            "included_directory",
            (config.logical_id,),
            metadata=FileMetadata(
                1,
                "profile:profile:agents.history:controlled",
                "",
                None,
                "directory",
                0o700,
                0,
                "private",
            ),
        )
        plan = replace(
            original_plan,
            target=replace(
                original_plan.target, items=(*original_plan.target.items, history)
            ),
            preserve=(*original_plan.preserve, (history.logical_id, root)),
            safety_scope=(*original_plan.safety_scope, history.logical_id),
            local_snapshot=LocalSnapshotSource(
                journal.root.parent, "controlled-original-mobile", "a" * 64
            ),
        )
        with journal._locked(exclusive=True) as parent:
            records = journal._records(parent)
            receipt = _CandidateReceipt.model_validate(records[0].evidence)
            prepared = _Prepared.model_validate(
                next(r.evidence for r in records if r.event == "prepared")
            )
            source = _CanonicalCopy(
                _MobileCaptureVerified.model_construct(
                    manifest_digest=receipt.manifest_digest
                ),
                "a" * 64,
                False,
                proof.prepared,
            )
            work = candidate / "isolated-installed-check"
            work.mkdir(mode=0o700)
            private = work / "config.toml"
            private.write_bytes(selector.read_bytes())
            private.chmod(0o600)
            # The exact current EX remains held through every helper call.
            values = {
                "plan": plan,
                "document": object(),
                "manifest_digest": receipt.manifest_digest,
                "work": work,
                "session": session,
                "candidates": {config.logical_id: private},
                "source": source,
                "prepared": prepared,
                "receipt": receipt,
                "descriptor": descriptor,
                "records": records,
            }
            yield values, config, private, journal


def _install_sink(monkeypatch, values, config, *, mutate=None):
    from tldw_chatbook.Backup_Recovery import later_rollback

    reached = []

    def sink(plan, document, manifest_digest, work, session, cancel, **kwargs):
        budget, custody = kwargs["_read_budget"], kwargs["_custody"]
        assert custody in session._scopes and not custody.close_failed
        assert 0 < budget._remaining < 16 * 1024**2
        assert kwargs["_config_data"] == (
            (config.logical_id, values["candidates"][config.logical_id].read_bytes()),
        )
        assert set(kwargs) == {"_read_budget", "_custody", "_config_data"}
        reached.append((budget, custody))
        if mutate is not None:
            mutate()
        return {}, {}

    monkeypatch.setattr(later_rollback, "_preserved_builtin_validation", sink)
    return reached


@pytest.mark.parametrize("validated", (False, True))
def test_real_publication_config_transition_keeps_original_proof_and_retires(
    tmp_path, monkeypatch, helper_resource_root, validated
):
    with _published_case(
        tmp_path, monkeypatch, helper_resource_root, validated=validated
    ) as (values, config, _, _journal):
        old = values["source"].prepared.model_dump()
        old_leaf = values["source"].prepared.selector_review[0].witnesses[-1]
        assert config.path.stat().st_ino != old_leaf.inode
        reached = _install_sink(monkeypatch, values, config)
        assert publication._installed_history_preservation(**values) == ({}, {})
        assert len(reached) == 1
        budget, custody = reached[0]
        assert custody not in values["session"]._scopes
        assert (
            not custody.close_failed
            and not custody.fds
            and not custody.directory_iterators
        )
        assert values["source"].prepared.model_dump() == old
        assert 0 < budget._remaining < 16 * 1024**2


@pytest.mark.parametrize(
    "fault",
    (
        "live_inode",
        "live_bytes",
        "live_mode",
        "live_symlink",
        "live_link",
        "private_inode_late",
        "private_bytes",
        "private_mode",
        "private_symlink",
        "private_link",
        "live_late",
        "ancestor_mode",
        "wrong_session",
        "wrong_manifest",
        "wrong_prepared",
        "missing_progress",
        "duplicate_installed",
        "reversed_phase",
        "budget",
    ),
)
def test_installed_producer_refuses_actual_physical_and_context_changes(
    tmp_path, monkeypatch, helper_resource_root, fault
):
    from tldw_chatbook.Backup_Recovery import (
        mobile_builtin_preservation as preservation,
    )
    from tldw_chatbook.Backup_Recovery.journal import _Event
    from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit
    from tldw_chatbook.Utils.platform_files import os

    with _published_case(tmp_path, monkeypatch, helper_resource_root) as (
        values,
        config,
        private,
        _journal,
    ):
        original_mode = config.path.parent.stat().st_mode & 0o777

        def replace_bytes(path):
            replacement = path.with_name(path.name + ".new")
            replacement.write_bytes(path.read_bytes())
            replacement.chmod(0o600)
            replacement.replace(path)

        late = None
        if fault == "live_inode":
            replace_bytes(config.path)
        elif fault == "live_bytes":
            config.path.write_bytes(config.path.read_bytes() + b"\n")
        elif fault == "live_mode":
            config.path.chmod(0o640)
        elif fault == "live_symlink":
            other = config.path.with_name("kept-config")
            config.path.rename(other)
            config.path.symlink_to(other)
        elif fault == "live_link":
            os.link(config.path, config.path.with_name("linked-config"))
        elif fault == "private_bytes":
            private.write_bytes(private.read_bytes() + b"\n")
        elif fault == "private_mode":
            private.chmod(0o640)
        elif fault == "private_symlink":
            other = private.with_name("kept-copy")
            private.rename(other)
            private.symlink_to(other)
        elif fault == "private_link":
            os.link(private, private.with_name("linked-copy"))
        elif fault == "private_inode_late":
            late = lambda: replace_bytes(private)
        elif fault == "live_late":
            late = lambda: replace_bytes(config.path)
        elif fault == "ancestor_mode":
            config.path.parent.chmod(0o750)
        elif fault == "wrong_session":
            values["session"] = copy.copy(values["session"])
        elif fault == "wrong_manifest":
            values["manifest_digest"] = "0" * 64
        elif fault == "wrong_prepared":
            values["prepared"] = values["prepared"].model_copy(
                update={"generation": "foreign"}
            )
        elif fault == "missing_progress":
            values["records"] = [
                r for r in values["records"] if r.event != "artifact_published"
            ]
        elif fault == "duplicate_installed":
            values["prepared"] = values["prepared"].model_copy(
                update={"installed_paths": values["prepared"].installed_paths * 2}
            )
        elif fault == "reversed_phase":
            values["records"] = [
                *values["records"],
                _Event.model_construct(event="rollback_started"),
            ]
        elif fault == "budget":
            budget = _TraceReadBudget()
            budget._remaining = 4095
            monkeypatch.setattr(
                preservation, "_preservation_budget", lambda *args: budget
            )
        reached = _install_sink(monkeypatch, values, config, mutate=late)
        try:
            with pytest.raises(
                (ValueError, OSError, RuntimeError, _TraceReadLimit),
                match=(
                    "mobile_|maintenance_session_inactive|Too many levels|capture_|Operation not permitted"
                ),
            ):
                publication._installed_history_preservation(**values)
            assert len(reached) == (1 if late else 0)
        finally:
            if fault == "ancestor_mode":
                config.path.parent.chmod(original_mode)


@pytest.mark.parametrize("validated", (False, True))
def test_native_private_publication_mode_precedes_recorded_applied_mode(
    tmp_path, monkeypatch, helper_resource_root, validated
):
    with _published_case(
        tmp_path,
        monkeypatch,
        helper_resource_root,
        validated=validated,
        applied_mode=0o700,
    ) as (values, config, _, _journal):
        assert config.path.stat().st_mode & 0o777 == (0o700 if validated else 0o600)
        reached = _install_sink(monkeypatch, values, config)
        assert publication._installed_history_preservation(**values) == ({}, {})
        assert len(reached) == 1


def test_two_profile_config_data_remains_unique_and_hash_bound(tmp_path):
    from tldw_chatbook.Backup_Recovery.archive_models import ArchiveManifest
    from tldw_chatbook.Backup_Recovery.models import DiscoveryContext
    from tldw_chatbook.Backup_Recovery.recovery_files import _tree_member_id

    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir(mode=0o700)
    two.mkdir(mode=0o700)
    document, target, first, _ = _case(one)
    second_document, second_target, second, second_record = _case(two)
    second_config = next(r for r in second_target.items if r.owner == "config")
    config_key = "profile:second:config"
    context = DiscoveryContext(second_config.path, "second")
    key = _tree_member_id(context, "agents.history", second.path, second.path)
    second_config = replace(second_config, logical_id=config_key)
    second_runs = replace(
        next(r for r in second_target.items if r.owner == "db.agent_runs"),
        logical_id="profile:second:db.agent_runs",
        dependencies=(config_key,),
    )
    second = replace(
        second,
        logical_id=key,
        dependencies=(config_key,),
        metadata=replace(second.metadata, root_id=key),
    )
    second_record = second_record.model_copy(update={"logical_id": key, "root_id": key})
    producer = second_document.producer_inventory[0].model_copy(
        update={"logical_id": key, "dependencies": (config_key,)}
    )
    combined = ArchiveManifest.model_construct(
        profile_ids=(*document.profile_ids, "second"),
        directories=(*document.directories, second_record),
        files=(),
        producer_inventory=(*document.producer_inventory, producer),
    )
    target = replace(target, items=(*target.items, second_config, second_runs, second))
    first_config = next(r for r in target.items if r.owner == "config")
    proof = _mapping_proof(first_config)
    proof = replace(
        proof,
        prepared=proof.prepared.model_copy(
            update={
                "target_selector": str(second_config.path),
                "selector_review": [
                    *proof.prepared.selector_review,
                    *_mapping_proof(second_config).prepared.selector_review,
                ],
            }
        ),
    )
    data = (
        (first_config.logical_id, first_config.path.read_bytes()),
        (config_key, second_config.path.read_bytes()),
    )
    assert _empty_history_members(
        combined, target, _TraceReadBudget(), proof, _config_data=data
    ) == {first.logical_id: first, second.logical_id: second}
    with pytest.raises(ValueError, match="^mobile_canonical_config_data_changed$"):
        _empty_history_members(
            combined,
            target,
            _TraceReadBudget(),
            proof,
            _config_data=((data[0][0], data[1][1]), (data[1][0], data[0][1])),
        )


@pytest.mark.parametrize("mask", (1, 2, 3, 4, 5, 6))
def test_partial_installed_data_refuses_before_source_journal(
    tmp_path, monkeypatch, mask
):
    from tldw_chatbook.Backup_Recovery import later_rollback
    from tldw_chatbook.Backup_Recovery.models import Inventory
    from tldw_chatbook.Backup_Recovery.restore_plan import (
        LocalSnapshotSource,
        RestorePlan,
    )

    plan = RestorePlan(
        "archive",
        "replace",
        (),
        (),
        (),
        "target",
        target=Inventory((), True, "target", ()),
        local_snapshot=LocalSnapshotSource(tmp_path, "source", "0" * 64),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("partial installed data reached source journal")

    monkeypatch.setattr(later_rollback, "verify_snapshot_source", forbidden)
    with pytest.raises(ValueError, match="^mobile_canonical_config_data_changed$"):
        later_rollback._preserved_builtin_validation(
            plan,
            object(),
            "manifest",
            tmp_path,
            None,
            Event(),
            _config_data=() if mask & 1 else None,
            _read_budget=object() if mask & 2 else None,
            _custody=object() if mask & 4 else None,
        )


@private_profile_test
def test_installed_metadata_custody_uncertain_native_close_is_not_retried(
    request, tmp_path, monkeypatch, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.admission import AdmissionError
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    native_os = storage.os
    closes, reached = [], {}
    holds = len(storage._failed_capture_holds)

    class NativeClose:
        def __getattr__(self, name):
            return getattr(native_os, name)

        def close(self, fd):
            info = native_os.fstat(fd)
            selected = (info.st_dev, info.st_ino) == reached.get("private_identity")
            native_os.close(fd)
            if selected:
                closes.append(fd)
                raise OSError("native close completed with uncertain outcome")

    with (
        pytest.raises(
            (RecoveryRequired, AdmissionError), match="^capture_resources_not_retired$"
        ),
        _published_case(tmp_path, monkeypatch, helper_resource_root) as (
            values,
            config,
            private,
            _journal,
        ),
    ):
        reached["session"] = values["session"]
        info = private.stat()
        reached["private_identity"] = (info.st_dev, info.st_ino)
        parent_info = private.parent.stat()
        reached["parent_identity"] = (parent_info.st_dev, parent_info.st_ino)
        sink = _install_sink(monkeypatch, values, config)
        monkeypatch.setattr(storage, "os", NativeClose())
        publication._installed_history_preservation(**values)
        pytest.fail("uncertain copy close returned to consumer")
    assert len(closes) == 1 and not sink
    session = reached["session"]
    failed = [
        r
        for r in session._scopes
        if isinstance(r, storage._CaptureFileDescriptors) and r.close_failed
    ]
    assert len(failed) == 1 and closes[0] in failed[0].fds
    parents = [fd for fd in failed[0].fds if fd != closes[0]]
    assert len(parents) == 1
    info = native_os.fstat(parents[0])
    assert (info.st_dev, info.st_ino) == reached["parent_identity"]
    import stat

    assert stat.S_ISDIR(info.st_mode)
    with pytest.raises(OSError):
        native_os.fstat(closes[0])
    assert not session._active and len(storage._failed_capture_holds) > holds


def test_installed_caller_keeps_native_snapshot_proof_after_copy_loop(
    tmp_path, monkeypatch, helper_resource_root
):
    from Tests.Backup_Recovery.test_later_rollback import _completed
    from tldw_chatbook.Backup_Recovery import later_rollback, recovery_copies
    from tldw_chatbook.Backup_Recovery.journal import _Rollback

    with _completed(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        operation,
        original_config,
    ):
        control = tmp_path / "control"
        plan = later_rollback.preview_rollback(
            operation,
            control_root=control,
            old_password=b"original",
            target=case[1].target,
            cancel=Event(),
        )
        observed, reached = [], []
        native_verify = later_rollback.verify_snapshot_source
        native_preservation = publication._installed_history_preservation

        def verify(*args, **kwargs):
            result = native_verify(*args, **kwargs)
            observed.append(result[1])
            return result

        def preserve(*args, **kwargs):
            reached.append(args[6])
            return native_preservation(*args, **kwargs)

        monkeypatch.setattr(later_rollback, "verify_snapshot_source", verify)
        monkeypatch.setattr(publication, "_installed_history_preservation", preserve)
        result = recovery_copies.rollback(
            operation,
            control_root=control,
            old_password=b"original",
            new_password=b"new safety",
            cancel=Event(),
            approved_plan=plan,
        )
        assert result != operation and reached
        assert all(type(source) is _Rollback for source in reached)
        assert all(any(source is proof for proof in observed) for source in reached)
        assert case[-1].read_bytes() == original_config


def test_installed_selected_history_coexists_with_unselected_nonempty_tree(
    tmp_path, monkeypatch, helper_resource_root
):
    with _published_case(tmp_path, monkeypatch, helper_resource_root) as (
        values,
        config,
        _,
        _journal,
    ):
        plan = values["plan"]
        selected = next(
            item for item in plan.target.items if item.owner == "agents.history"
        )
        root = selected.path.with_name("agent_history")
        child = root / "imports"
        child.mkdir(mode=0o700, parents=True)
        payload = child / "retained-payload"
        payload.write_bytes(b"ordinary native retained history")
        payload.chmod(0o600)
        sibling_id = selected.logical_id + "-ordinary"
        sibling = replace(
            selected,
            logical_id=sibling_id,
            path=root,
            metadata=replace(selected.metadata, root_id=sibling_id),
        )
        descendant = replace(
            sibling,
            logical_id=sibling_id + "-child",
            path=child,
            dependencies=(sibling_id, config.logical_id),
            metadata=replace(
                sibling.metadata, parent_id=sibling_id, relative_path="imports"
            ),
        )
        values["plan"] = replace(
            plan,
            target=replace(
                plan.target, items=(*plan.target.items, sibling, descendant)
            ),
            preserve=(
                *plan.preserve,
                (sibling.logical_id, root),
                (descendant.logical_id, child),
            ),
        )
        assert sibling_id not in values["plan"].safety_scope
        before = payload.read_bytes()
        reached = _install_sink(monkeypatch, values, config)
        assert publication._installed_history_preservation(**values) == ({}, {})
        assert len(reached) == 1
        assert payload.read_bytes() == before


@pytest.mark.parametrize(
    "fault", ("root", "parent", "relative", "dependency", "preserve")
)
def test_installed_selected_history_still_requires_exact_native_shape(
    tmp_path, monkeypatch, helper_resource_root, fault
):
    with _published_case(tmp_path, monkeypatch, helper_resource_root) as (
        values,
        config,
        _,
        _journal,
    ):
        plan = values["plan"]
        selected = next(
            item for item in plan.target.items if item.owner == "agents.history"
        )
        changed = selected
        if fault == "root":
            changed = replace(
                selected, metadata=replace(selected.metadata, root_id="foreign")
            )
        elif fault == "parent":
            changed = replace(
                selected, metadata=replace(selected.metadata, parent_id="foreign")
            )
        elif fault == "relative":
            changed = replace(
                selected, metadata=replace(selected.metadata, relative_path="child")
            )
        elif fault == "dependency":
            changed = replace(selected, dependencies=(config.logical_id, "foreign"))
        preserve = (
            tuple(pair for pair in plan.preserve if pair[0] != selected.logical_id)
            if fault == "preserve"
            else plan.preserve
        )
        values["plan"] = replace(
            plan,
            preserve=preserve,
            target=replace(
                plan.target,
                items=tuple(
                    changed if item is selected else item for item in plan.target.items
                ),
            ),
        )
        reached = _install_sink(monkeypatch, values, config)
        with pytest.raises(
            ValueError, match="^mobile_installed_config_context_changed$"
        ):
            publication._installed_history_preservation(**values)
        assert not reached


@pytest.mark.parametrize("routing", ("unselected", "no_config_transition"))
def test_unselected_history_retains_original_no_data_call_shape(
    tmp_path, monkeypatch, helper_resource_root, routing
):
    from tldw_chatbook.Backup_Recovery import later_rollback

    with _published_case(tmp_path, monkeypatch, helper_resource_root) as (
        values,
        _,
        _,
        _journal,
    ):
        plan = values["plan"]
        if routing == "unselected":
            values["plan"] = replace(plan, safety_scope=())
        else:
            values["plan"] = replace(
                plan,
                target=replace(
                    plan.target,
                    items=tuple(
                        replace(item, dependencies=("foreign",))
                        if item.owner == "agents.history"
                        else item
                        for item in plan.target.items
                    ),
                ),
            )
        reached = []

        def fallback(*args, **kwargs):
            reached.append((args, kwargs))
            return {}, {}

        monkeypatch.setattr(later_rollback, "_preserved_builtin_validation", fallback)
        assert publication._installed_history_preservation(**values) == ({}, {})
        assert len(reached) == 1 and reached[0][1] == {}
        assert reached[0][0][0] is values["plan"]
