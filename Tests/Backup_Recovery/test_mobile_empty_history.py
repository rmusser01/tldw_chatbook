"""R176 fixed mapping/physical negatives; public qualification stays separate."""

from dataclasses import replace
from threading import Event

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.archive_models import (
    ArchiveManifest,
    Directory,
    Metadata,
    Payload,
    ProducerItem,
)
from tldw_chatbook.Backup_Recovery.mobile_empty_history import (
    _empty_history_members,
    _empty_history_observation,
)
from tldw_chatbook.Backup_Recovery.models import (
    DiscoveryContext,
    FileMetadata,
    Inventory,
    StorageItem,
)
from tldw_chatbook.Backup_Recovery.recovery_files import _tree_member_id
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget


def _case(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir(mode=0o700)
    shared = profile / "data" / "shared"
    shared.mkdir(parents=True, mode=0o700)
    root = shared / "tool_sandbox"
    root.mkdir(mode=0o700)
    context = DiscoveryContext(profile / "config.toml", "fixture-profile")
    context.config_path.write_text(
        f'[Paths]\ndata_dir = "{profile / "data"}"\n[general]\nusers_name = "shared"\n'
    )
    context.config_path.chmod(0o600)
    key = _tree_member_id(context, "agents.history", root, root)
    config_key = "profile:fixture-profile:config"
    runs_key = "profile:fixture-profile:db.agent_runs"
    info = root.stat()
    item = StorageItem(
        "agents.history",
        key,
        root,
        "included_directory",
        (config_key,),
        metadata=FileMetadata(
            1, key, "", None, "directory", 0o700, info.st_mtime_ns, "private"
        ),
    )
    config = StorageItem("config", config_key, context.config_path, "included", ())
    runs = StorageItem(
        "db.agent_runs",
        runs_key,
        shared / "agent-runs" / "runs.db",
        "included",
        (config_key,),
    )
    target = Inventory((config, runs, item), True, "fixture-scope", ())
    record = Directory(
        logical_id=key,
        root_id=key,
        parent_id=None,
        relative_path="",
        metadata=Metadata(version=1, mode=0o700, mtime_ns=info.st_mtime_ns),
        synthetic=False,
    )
    producer = ProducerItem(
        logical_id=key,
        owner_id=item.owner,
        status=item.status,
        dependencies=item.dependencies,
    )
    # Deliberately only a pure mapping input. This bypass is NEVER a claim of an
    # authenticated archive or native rollback; the public end-to-end test owns that.
    document = ArchiveManifest.model_construct(
        profile_ids=(context.profile_id,),
        directories=(record,),
        files=(),
        producer_inventory=(producer,),
    )
    return document, target, item, record


def _mapping_proof(config):
    """Unit-only source facts, never an authenticated journal/rollback grant."""
    from tldw_chatbook.Backup_Recovery.journal import (
        _MobilePrepared,
        _MobileSelectorReview,
        _MobileWitness,
    )
    from tldw_chatbook.Backup_Recovery.mobile_canonical_copy import _CanonicalCopy
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import _selector_observation

    fingerprint, witnesses = _selector_observation(config.path)
    row = _MobileSelectorReview(
        selector=str(config.path),
        fingerprint=fingerprint,
        witnesses=[
            _MobileWitness(path=p, device=d, inode=i, mode=m)
            for p, d, i, m in witnesses
        ],
        destination_identity=None,
        targets=[],
    )
    prepared = _MobilePrepared.model_construct(
        source_selector=str(config.path),
        target_selector=str(config.path),
        selector_review=[row],
    )
    return _CanonicalCopy(None, "unit-only", False, prepared)


def _mapping(document, target, budget):
    config = next(item for item in target.items if item.owner == "config")
    return _empty_history_members(document, target, budget, _mapping_proof(config))


def test_empty_native_default_container_mapping_is_not_payload_authority(tmp_path):
    document, target, item, record = _case(tmp_path)
    assert _mapping(document, target, _TraceReadBudget()) == {item.logical_id: item}
    child = document.directories[0].model_copy(
        update={
            "logical_id": item.logical_id + ":child",
            "root_id": item.logical_id,
            "parent_id": item.logical_id,
            "relative_path": "history",
        }
    )
    nonempty = document.model_copy(
        update={"directories": (*document.directories, child)}
    )
    assert _mapping(nonempty, target, _TraceReadBudget()) == {}
    payload = Payload(
        logical_id=item.logical_id + ":payload",
        root_id=item.logical_id,
        parent_id=item.logical_id,
        relative_path="run.jsonl",
        owner_id="agents.history",
        payload="payload/run.jsonl",
        size=0,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        metadata=Metadata(version=1, mode=0o600, mtime_ns=record.metadata.mtime_ns),
    )
    with_file = document.model_copy(update={"files": (payload,)})
    assert _mapping(with_file, target, _TraceReadBudget()) == {}


@pytest.mark.parametrize(
    "change",
    (
        "owner",
        "dependency",
        "shared",
        "metadata",
        "root_id",
        "duplicate_current",
        "duplicate_record",
        "wrong_profile",
        "current_child",
        "runs_dependency",
    ),
)
def test_empty_history_mapping_refuses_changed_native_shape(tmp_path, change):
    document, target, item, record = _case(tmp_path)
    replacement = item
    if change == "owner":
        replacement = replace(item, owner="eval.definitions")
    elif change == "dependency":
        replacement = replace(item, dependencies=())
    elif change == "shared":
        replacement = replace(item, shared_group="foreign")
    elif change == "metadata":
        replacement = replace(
            item, metadata=replace(item.metadata, mtime_ns=item.metadata.mtime_ns + 1)
        )
    elif change == "root_id":
        replacement = replace(item, metadata=replace(item.metadata, root_id="foreign"))
    elif change == "duplicate_current":
        target = replace(target, items=(*target.items, item))
    elif change == "duplicate_record":
        document = document.model_copy(update={"directories": (record, record)})
    elif change == "wrong_profile":
        document = document.model_copy(update={"profile_ids": ("other",)})
    elif change == "current_child":
        target = replace(
            target,
            items=(
                *target.items,
                replace(
                    item,
                    logical_id=item.logical_id + ":child",
                    path=item.path / "history",
                    metadata=replace(
                        item.metadata,
                        parent_id=item.logical_id,
                        relative_path="history",
                    ),
                ),
            ),
        )
    else:
        target = replace(
            target,
            items=tuple(
                replace(row, dependencies=()) if row.owner == "db.agent_runs" else row
                for row in target.items
            ),
        )
    if replacement is not item:
        target = replace(
            target,
            items=tuple(replacement if row is item else row for row in target.items),
        )
    try:
        matched = _mapping(document, target, _TraceReadBudget())
    except ValueError:
        return
    assert matched == {}  # No alternative path or malformed shape is preserved.


@pytest.mark.parametrize(
    "change",
    (
        "none",
        "child",
        "symlink",
        "mode",
        "metadata",
        "late_child",
        "late_second_child",
        "root_swap",
        "cancel",
    ),
)
def test_native_empty_directory_observation_checks_real_names_and_identity(
    tmp_path, monkeypatch, change
):
    from tldw_chatbook.Backup_Recovery import bootstrap

    _, _, item, record = _case(tmp_path)
    root = item.path
    before = root.stat()
    if change == "child":
        (root / "history").write_text("must not be hidden")
    elif change == "symlink":
        root.rename(root.with_name("retained-root"))
        root.symlink_to(root.with_name("retained-root"), target_is_directory=True)
    elif change == "mode":
        root.chmod(0o755)
    elif change == "metadata":
        record = record.model_copy(
            update={
                "metadata": record.metadata.model_copy(
                    update={"mtime_ns": record.metadata.mtime_ns + 1}
                )
            }
        )
    census = bootstrap._bounded_names
    calls = 0

    def changing(parent, maximum, budget, custody):
        nonlocal calls
        result = census(parent, maximum, budget, custody)
        calls += 1
        if (calls == 1 and change == "late_child") or (
            calls == 2 and change == "late_second_child"
        ):
            (root / "late-history").write_text("must refuse")
        elif calls == 1 and change == "root_swap":
            root.rename(root.with_name("retained-root"))
            root.mkdir(mode=0o700)
        return result

    monkeypatch.setattr(bootstrap, "_bounded_names", changing)
    cancel = Event()
    if change == "cancel":
        cancel.set()
    budget = _TraceReadBudget()
    with storage._preview_reads(_read_budget=budget):
        scope = storage._local.preview_scope
        if change == "none":
            observed = _empty_history_observation(
                item, record, budget, scope.custody, cancel
            )
            assert observed[:2] == (before.st_dev, before.st_ino)
            assert not tuple(root.iterdir())
            assert root.stat().st_mode == before.st_mode
        else:
            with pytest.raises((ValueError, OSError, RuntimeError)):
                _empty_history_observation(item, record, budget, scope.custody, cancel)
        assert scope.custody.directory_iterators == []
        assert scope.custody.fds == []
    assert budget._remaining < 16 * 1024**2


@private_profile_test
def test_empty_history_ambiguous_native_iterator_close_keeps_exclusion(
    request, tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    _, _, item, record = _case(tmp_path)
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    authority = admission_authority(tmp_path / "authority")
    authority.register("history", (item.path.parent,))
    native = bootstrap.os
    close_calls = []

    class Iterator:
        def __init__(self, iterator):
            self.iterator = iterator

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.iterator)

        def close(self):
            close_calls.append(self.iterator)
            self.iterator.close()
            raise OSError("injected ambiguous native iterator close")

    class Calls:
        def __getattr__(self, name):
            return getattr(native, name)

        def scandir(self, fd):
            return Iterator(native.scandir(fd))

    limits = ArchiveLimits()
    from tldw_chatbook.Backup_Recovery.admission import AdmissionError

    with (
        pytest.raises(
            (bootstrap.RecoveryRequired, AdmissionError),
            match="^capture_resources_not_retired$",
        ),
        authority.maintenance(("history",), 2) as session,
        session._capture_bound_sources((), work, limits, limits.expanded_bytes),
    ):
        scope = storage._local.capture_scope
        custody = storage._CaptureFileDescriptors(scope)
        with monkeypatch.context() as fault:
            fault.setattr(bootstrap, "os", Calls())
            _empty_history_observation(
                item, record, _TraceReadBudget(), custody, Event()
            )
    assert custody.close_failed and custody in scope.resources
    assert custody.directory_iterators
    assert len(close_calls) == 1
    import subprocess
    import sys
    from pathlib import Path

    probe = """from pathlib import Path
from Tests.real_profile_guard import install
install()
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
import sys
try:
    with admission_authority(Path(sys.argv[1])).maintenance(('history',), 0.1):
        raise AssertionError('ambiguous close released exclusion')
except AdmissionTimeout as error:
    assert str(error) == 'admission_timeout'
    sys.exit(23)
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(authority.control_root.parent)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 23, result.stdout + result.stderr
    assert len(close_calls) == 1


def test_unrelated_local_snapshot_keeps_native_no_work_short_circuit(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import later_rollback

    def unexpected(*args, **kwargs):
        pytest.fail("unrelated no-work snapshot authenticated or acquired sources")

    monkeypatch.setattr(later_rollback, "verify_snapshot_source", unexpected)
    plan = SimpleNamespace(
        local_snapshot=object(),
        target=Inventory((), True, "empty", ()),
        safety_scope=(),
        preserve=(),
    )
    assert later_rollback._preserved_builtin_validation(
        plan, None, "unused", tmp_path, None, Event()
    ) == ({}, {})


def test_actual_public_mobile_rollback_rejects_partial_source_options(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_mobile_native_lineage_semantics import (
        test_native85_archive_witnesses_canonical_history_and_public_rollback,
    )
    from tldw_chatbook.Backup_Recovery import mobile_empty_history, replacement

    original_root = mobile_empty_history._native_default_history_root
    native_proof_controls = []

    def root_with_native_negatives(config, proof, budget, *, _custody=None):
        if _custody is not None:
            session = storage._local.maintenance_session
            assert _custody in session._scopes and not _custody.close_failed
        if not native_proof_controls:
            row = next(
                row
                for row in proof.prepared.selector_review
                if row.selector == str(config.path)
            )
            changed_row = row.model_copy(update={"fingerprint": "0" * 64})
            copied = replace(
                proof,
                prepared=proof.prepared.model_copy(
                    update={
                        "selector_review": [
                            changed_row if entry is row else entry
                            for entry in proof.prepared.selector_review
                        ]
                    }
                ),
            )
            with pytest.raises(
                ValueError, match="^mobile_canonical_selector_review_required$"
            ):
                original_root(config, copied, budget, _custody=_custody)
            foreign = replace(
                config, path=config.path.with_name("wrong-selected-profile.toml")
            )
            with pytest.raises(
                ValueError, match="^mobile_canonical_history_binding_changed$"
            ):
                original_root(foreign, proof, budget, _custody=_custody)
            from tldw_chatbook.Backup_Recovery.mobile_canonical_copy import (
                _mobile_preserved_members,
            )

            with pytest.raises(ValueError, match="^recovery_copy_pending$"):
                _mobile_preserved_members(
                    replace(proof, pending=True), None, None, _read_budget=budget
                )
            native_proof_controls.extend(
                ("copied_changed_fingerprint", "wrong_selected_profile", "pending")
            )
        if proof.pending and "held_pending_source" not in native_proof_controls:
            from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
            from tldw_chatbook.Backup_Recovery.mobile_canonical_copy import (
                _mobile_preserved_members,
            )

            with pytest.raises(ValueError, match="^recovery_copy_pending$"):
                _mobile_preserved_members(proof, None, None, _read_budget=budget)
            # Even a copied false bit cannot suppress the actual public pending
            # registry recheck. This intentionally minimal target is negative-only.
            with pytest.raises(RecoveryRequired, match="^recovery_pending$"):
                _mobile_preserved_members(
                    replace(proof, pending=False),
                    None,
                    Inventory((config,), True, "pending-boundary-negative", ()),
                    _read_budget=budget,
                )
            native_proof_controls.append("held_pending_source")
        return original_root(config, proof, budget, _custody=_custody)

    monkeypatch.setattr(
        mobile_empty_history, "_native_default_history_root", root_with_native_negatives
    )
    original = replacement._checked_originals
    reached = []

    def check(*args, **kwargs):
        if kwargs.get("_source_read_budget") is not None:
            from itertools import product

            names = ("_source_read_budget", "_source_journal", "_source_archive")
            for mask in product((False, True), repeat=3):
                if not any(mask) or all(mask):
                    continue
                incomplete = dict(
                    kwargs,
                    **{
                        name: kwargs[name] if keep else None
                        for name, keep in zip(names, mask, strict=True)
                    },
                )
                with pytest.raises(ValueError, match="^mobile_safety_source_required$"):
                    original(*args, **incomplete)
                reached.append(mask)
        return original(*args, **kwargs)

    monkeypatch.setattr(replacement, "_checked_originals", check)
    import copy
    from itertools import product

    from tldw_chatbook.Backup_Recovery import publication
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.journal import Journal

    native_prepare = publication._prepare
    publication_controls = []

    def prepare_with_native_controls(*args, **kwargs):
        if kwargs.get("_source_read_budget") is not None:
            names = (
                "_source_archive",
                "_source_read_budget",
                "_source_journal",
                "_source_session",
                "_source_cancel",
            )

            def never_locked(*unused, **options):
                pytest.fail("invalid publication source acquired journal")

            with monkeypatch.context() as isolated:
                isolated.setattr(Journal, "_locked", never_locked)
                for mask in product((False, True), repeat=5):
                    if not any(mask) or all(mask):
                        continue
                    bad = dict(
                        kwargs,
                        **{
                            name: kwargs[name] if keep else None
                            for name, keep in zip(names, mask, strict=True)
                        },
                    )
                    with pytest.raises(
                        ValueError, match="^mobile_safety_source_required$"
                    ):
                        native_prepare(*args, **bad)
                    publication_controls.append(mask)
                for name in names:
                    with pytest.raises(
                        ValueError, match="^mobile_safety_source_required$"
                    ):
                        native_prepare(*args, **dict(kwargs, **{name: object()}))
                with pytest.raises(
                    RecoveryRequired, match="maintenance_session_inactive"
                ):
                    native_prepare(
                        *args,
                        **dict(
                            kwargs, _source_session=copy.copy(kwargs["_source_session"])
                        ),
                    )
                with pytest.raises(
                    ValueError, match="^mobile_safety_source_journal_changed$"
                ):
                    native_prepare(*args, **dict(kwargs, _source_journal=args[0]))
                with pytest.raises(ValueError, match="^mobile_safety_source_required$"):
                    native_prepare(
                        *args,
                        **dict(
                            kwargs,
                            _source_archive=replace(
                                kwargs["_source_archive"], digest="0" * 64
                            ),
                        ),
                    )
        return native_prepare(*args, **kwargs)

    monkeypatch.setattr(publication, "_prepare", prepare_with_native_controls)
    test_native85_archive_witnesses_canonical_history_and_public_rollback(
        tmp_path, monkeypatch
    )
    assert len(set(publication_controls)) == 30
    assert native_proof_controls == [
        "copied_changed_fingerprint",
        "wrong_selected_profile",
        "pending",
        "held_pending_source",
    ]
    assert set(reached) == {
        (False, False, True),
        (False, True, False),
        (True, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, False),
    }


def test_all_none_source_options_keep_generic_journal_entry():
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.replacement import _checked_originals

    class EnteredGenericJournal(Exception):
        pass

    def locked(*, exclusive):
        assert exclusive is True
        raise EnteredGenericJournal

    # Classification control only: no source acquisition or owner authority.
    from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan

    journal = SimpleNamespace(_locked=locked)
    plan = RestorePlan(
        "classification",
        "replace",
        (),
        (),
        (),
        "empty",
        target=Inventory((), True, "empty", ()),
    )
    with pytest.raises(EnteredGenericJournal):
        _checked_originals(plan, journal, None, cancel=Event())


@pytest.mark.parametrize("section", ("Paths", "paths"))
def test_ordinary_native_history_root_ignores_runs_override(tmp_path, section):
    document, target, item, _ = _case(tmp_path)
    config = next(row for row in target.items if row.owner == "config")
    config.path.write_text(
        f'[{section}]\ndata_dir = "{config.path.parent / "data"}"\n[general]\nusers_name = "shared"\n[database]\nagent_runs_db_path = "{tmp_path / "elsewhere" / "runs.db"}"\n'
    )
    target = replace(
        target,
        items=tuple(
            replace(row, path=tmp_path / "elsewhere" / "runs.db")
            if row.owner == "db.agent_runs"
            else row
            for row in target.items
        ),
    )
    assert _mapping(document, target, _TraceReadBudget()) == {item.logical_id: item}


def test_nondefault_history_root_keeps_ordinary_refusal_mapping(tmp_path):
    document, target, _, _ = _case(tmp_path)
    config = next(row for row in target.items if row.owner == "config")
    with config.path.open("a") as out:
        out.write(f'\n[tools]\nfile_sandbox_root = "{tmp_path / "external"}"\n')
    assert _mapping(document, target, _TraceReadBudget()) == {}


@pytest.mark.parametrize("change", ("fingerprint", "witness", "selected", "file"))
def test_native_history_config_rejects_stale_or_foreign_source_facts(tmp_path, change):
    from tldw_chatbook.Backup_Recovery.mobile_empty_history import (
        _native_default_history_root,
    )

    _, target, _, _ = _case(tmp_path)
    config = next(row for row in target.items if row.owner == "config")
    proof = _mapping_proof(config)
    row = proof.prepared.selector_review[0]
    if change == "fingerprint":
        row = row.model_copy(update={"fingerprint": "0" * 64})
    elif change == "witness":
        row = row.model_copy(
            update={
                "witnesses": [
                    *row.witnesses[:-1],
                    row.witnesses[-1].model_copy(
                        update={"inode": row.witnesses[-1].inode + 1}
                    ),
                ]
            }
        )
    elif change == "selected":
        proof = replace(
            proof,
            prepared=proof.prepared.model_copy(
                update={
                    "source_selector": str(tmp_path / "other"),
                    "target_selector": str(tmp_path / "other"),
                }
            ),
        )
    else:
        with config.path.open("a") as out:
            out.write("\n# changed after source observation\n")
    proof = replace(
        proof, prepared=proof.prepared.model_copy(update={"selector_review": [row]})
    )
    with pytest.raises(
        ValueError,
        match="^mobile_canonical_(selector_review_required|history_binding_changed)$",
    ):
        _native_default_history_root(config, proof, _TraceReadBudget())


@pytest.mark.parametrize(
    "mode", ("lowercase_precedence", "sanitized_user", "default_user", "default_data")
)
def test_native_config_default_case_and_user_selection(tmp_path, monkeypatch, mode):
    from tldw_chatbook.Backup_Recovery.mobile_empty_history import (
        _native_default_history_root,
    )

    _, target, _, _ = _case(tmp_path)
    config = next(row for row in target.items if row.owner == "config")
    base = tmp_path / "selected-data"
    if mode == "lowercase_precedence":
        text = f'[Paths]\ndata_dir = "{tmp_path / "ignored-data"}"\n[paths]\ndata_dir = "{base}"\n[general]\nusers_name = "shared"\n'
        name = "shared"
    elif mode == "sanitized_user":
        text = f'[Paths]\ndata_dir = "{base}"\n[general]\nusers_name = "two users!"\n'
        name = "two_users_"
    elif mode == "default_user":
        text = f'[Paths]\ndata_dir = "{base}"\n'
        name = "default_user"
    else:
        home = tmp_path / "private-home"
        home.mkdir(mode=0o700)
        monkeypatch.setenv("HOME", str(home))
        text = ""
        base = home / ".local" / "share" / "tldw_cli"
        name = "default_user"
    config.path.write_text(text)
    assert (
        _native_default_history_root(config, _mapping_proof(config), _TraceReadBudget())
        == base / name / "tool_sandbox"
    )


@pytest.mark.parametrize("replacement_kind", ["file", "symlink"])
def test_expected_safety_directory_never_falls_back_to_a_file_reader(
    tmp_path, monkeypatch, replacement_kind
):
    from tldw_chatbook.Backup_Recovery import publication
    from tldw_chatbook.Backup_Recovery.journal import _Object

    directory = tmp_path / "observed-directory"
    directory.mkdir(mode=0o700)
    expected = _Object.model_validate(publication._observe_safety_source(directory))
    assert publication._safety_source_matches(expected)
    directory.rename(tmp_path / "retained-original-directory")
    if replacement_kind == "file":
        directory.write_bytes(b"never read substituted payload")
    else:
        payload = tmp_path / "unselected-payload"
        payload.write_bytes(b"never follow substituted symlink")
        directory.symlink_to(payload)
    reads = []

    def forbidden_reader(*args, **kwargs):
        reads.append((args, kwargs))
        raise ValueError("test_file_reader_reached")

    monkeypatch.setattr(publication, "observe_artifact", forbidden_reader)
    assert not publication._safety_source_matches(expected)
    assert reads == []


def test_safety_file_observation_keeps_native_framing(tmp_path):
    from tldw_chatbook.Backup_Recovery import publication
    from tldw_chatbook.Backup_Recovery.journal import _Object, observe_artifact

    path = tmp_path / "ordinary-file"
    path.write_bytes(b"ordinary retained payload")
    path.chmod(0o600)
    expected = _Object.model_validate(observe_artifact(path, metadata=True))
    assert publication._safety_source_matches(expected)
    path.write_bytes(b"changed")
    assert not publication._safety_source_matches(expected)


@pytest.mark.parametrize("mask", range(1, 31))
def test_publication_partial_native_tuple_refuses_before_journal(mask, tmp_path):
    from tldw_chatbook.Backup_Recovery.publication import _prepare

    class NeverLocked:
        def _locked(self, **kwargs):
            pytest.fail("partial tuple acquired the new journal")

    names = (
        "_source_archive",
        "_source_read_budget",
        "_source_journal",
        "_source_session",
        "_source_cancel",
    )
    kwargs = {
        name: object() if mask & (1 << bit) else None for bit, name in enumerate(names)
    }
    with pytest.raises(ValueError, match="^mobile_safety_source_required$"):
        _prepare(NeverLocked(), None, None, tmp_path, (), (), "test", **kwargs)


def test_publication_ordinary_call_shape_and_all_none_stay_native(
    monkeypatch, tmp_path
):
    from tldw_chatbook.Backup_Recovery import publication
    from tldw_chatbook.Backup_Recovery.journal import Journal

    calls = []
    native = publication._prepare
    monkeypatch.setattr(
        publication, "_prepare", lambda *args, **kwargs: calls.append(kwargs)
    )
    Journal.prepare_publication(
        object(),
        None,
        None,
        bootstrap_root=tmp_path,
        namespaces=(),
        selectors=(),
        generation="test",
    )
    assert len(calls) == 1
    assert calls[0] == {"replacement_profiles": (), "incoming_credentials": None}

    class EnteredGenericJournal(Exception):
        pass

    class OrdinaryJournal:
        def _locked(self, **kwargs):
            raise EnteredGenericJournal

    with pytest.raises(EnteredGenericJournal):
        native(OrdinaryJournal(), None, None, tmp_path, (), (), "test")


@private_profile_test
@pytest.mark.parametrize("fault", ["none", "file", "symlink", "close"])
def test_provisional_history_metadata_uses_held_directory_only_custody(
    request, tmp_path, monkeypatch, fault
):
    """Native held metadata mechanics only; this is not a canonical-source proof."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import bootstrap, publication
    from tldw_chatbook.Backup_Recovery.admission import AdmissionError
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_empty_history import (
        _publication_history_observation,
    )

    _, target, item, _ = _case(tmp_path)
    config = next(row for row in target.items if row.owner == "config")
    root = tmp_path / "bootstrap"
    authority = admission_authority(root)
    authority.register("native-history-profile", (config.path.parent,))
    bind_profile(root, config.path, ("native-history-profile",), authority.control_root)
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    context = SimpleNamespace(
        bootstrap_root=str(root),
        namespaces=("native-history-profile",),
        selectors=(str(config.path),),
    )
    plan = SimpleNamespace(
        target=target,
        preserve=((item.logical_id, item.path),),
        safety_scope=(item.logical_id,),
    )
    native_observe = publication._observe_safety_directory
    native_pin = storage._CaptureFileDescriptors.pinned_directory
    native_os = storage.os
    selected_fds, closed, buckets, file_reads = set(), [], [], []

    def forbid_file(*args, **kwargs):
        file_reads.append(args)
        pytest.fail("provisional history read a file body")

    @contextmanager
    def pin(custody, selected):
        with native_pin(custody, selected) as fd:
            if selected == item.path:
                selected_fds.add(fd)
            yield fd

    class Calls:
        def __getattr__(self, name):
            return getattr(native_os, name)

        def close(self, fd):
            native_os.close(fd)
            if fault == "close" and fd in selected_fds:
                closed.append(fd)
                raise OSError("native close completed but outcome was uncertain")

    def observe(path, *, _custody=None):
        assert _custody is not None and _custody in session._scopes
        buckets.append(_custody)
        if fault in {"file", "symlink"}:
            retained = item.path.with_name("retained-original-history")
            item.path.rename(retained)
            if fault == "file":
                item.path.write_bytes(b"must never enter file observer")
            else:
                item.path.symlink_to(retained, target_is_directory=True)
        return native_observe(path, _custody=_custody)

    monkeypatch.setattr(publication, "observe_artifact", forbid_file)
    monkeypatch.setattr(publication, "_observe_safety_directory", observe)
    monkeypatch.setattr(storage._CaptureFileDescriptors, "pinned_directory", pin)
    monkeypatch.setattr(storage, "os", Calls())
    holds = len(storage._failed_capture_holds)

    def exercise():
        nonlocal session
        with authority.maintenance(
            ("native-history-profile", "bootstrap.unbound"), 2
        ) as session:
            before = tuple(session._scopes)
            result = _publication_history_observation(
                item, plan, context, session, _TraceReadBudget(), Event()
            )
            assert result["kind"] == "directory" and result["size"] == 0
            assert tuple(session._scopes) == before

    session = None
    if fault == "none":
        exercise()
    elif fault == "close":
        with pytest.raises(
            (bootstrap.RecoveryRequired, AdmissionError),
            match="^capture_resources_not_retired$",
        ):
            exercise()
        assert len(closed) == 1
        assert len(storage._failed_capture_holds) == holds + 1
        assert storage._failed_capture_holds[-1][2] is session
        assert buckets[0].close_failed and buckets[0] in session._scopes
    else:
        with pytest.raises(OSError):
            exercise()
    assert file_reads == []
    assert len(buckets) == 1
    if fault != "close":
        assert not buckets[0].close_failed and buckets[0] not in session._scopes
        assert len(storage._failed_capture_holds) == holds


def test_pending_copy_is_refused_at_public_boundary_before_default_reads(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import mobile_canonical_copy

    _, target, _, _ = _case(tmp_path)
    config = next(row for row in target.items if row.owner == "config")
    proof = replace(_mapping_proof(config), pending=True)

    def forbidden(*args, **kwargs):
        pytest.fail("pending public copy entered target/default observation")

    monkeypatch.setattr(mobile_canonical_copy, "_current_canonical_target", forbidden)
    with pytest.raises(ValueError, match="^recovery_copy_pending$"):
        mobile_canonical_copy._mobile_preserved_members(
            proof, None, target, _read_budget=_TraceReadBudget()
        )
