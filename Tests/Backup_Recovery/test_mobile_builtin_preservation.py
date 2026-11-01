"""Native file accounting and unchanged installed builtin preservation."""

import hashlib
from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits, _TraceReadLimit
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget


@pytest.mark.parametrize(
    "mode", ["default", "budget", "short", "zero", "read_limit", "slice_limit"]
)
def test_native_copy_prices_reads_and_partial_write_views(tmp_path, monkeypatch, mode):
    root = tmp_path / "data"
    root.mkdir(mode=0o700)
    source = root / "source.yaml"
    payload = b"native fixed source\n" * 8
    source.write_bytes(payload)
    source.chmod(0o600)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    destination = stage / "copy.yaml"
    authority = admission_authority(tmp_path / "bootstrap")
    authority.register("source", (root,))
    limits = ArchiveLimits()
    budget = _TraceReadBudget()
    counts = {"read": 0, "write": 0}
    native = storage.os

    class FileCalls:
        def __getattr__(self, name):
            return getattr(native, name)

        def read(self, fd, size):
            counts["read"] += 1
            return native.read(fd, size)

        def write(self, fd, data):
            counts["write"] += 1
            if mode == "zero":
                return 0
            count = native.write(fd, data[:3] if mode == "short" else data)
            if mode == "slice_limit":
                budget.reserve(budget._remaining)
            return count

    with authority.maintenance(("source",), 2) as session:
        info = source.stat()
        with session._capture_bound_sources(
            ((source, info.st_dev, info.st_ino),), stage, limits, limits.expanded_bytes
        ):
            options = {} if mode == "default" else {"_read_budget": budget}
            if mode == "read_limit":
                budget.reserve(budget._remaining)
            with monkeypatch.context() as fault:
                fault.setattr(storage, "os", FileCalls())
                if mode in {"read_limit", "slice_limit"}:
                    with pytest.raises(_TraceReadLimit):
                        storage.copy_capture_file(
                            "eval.definitions",
                            source,
                            destination,
                            Event(),
                            max_bytes=len(payload),
                            **options,
                        )
                elif mode == "zero":
                    with pytest.raises(OSError, match="capture_write_unavailable"):
                        storage.copy_capture_file(
                            "eval.definitions",
                            source,
                            destination,
                            Event(),
                            max_bytes=len(payload),
                            **options,
                        )
                else:
                    storage.copy_capture_file(
                        "eval.definitions",
                        source,
                        destination,
                        Event(),
                        max_bytes=len(payload),
                        **options,
                    )
            assert not session._scopes[-1].resources
        assert all(
            not scope.active and not scope.resources for scope in session._scopes
        )
    with authority.normal(("source",)):
        assert source.read_bytes() == payload
    if mode in {"default", "budget", "short"}:
        assert destination.read_bytes() == payload
    if mode == "short":
        assert counts["write"] > 1
    if mode == "read_limit":
        assert counts == {"read": 0, "write": 0}
    if mode != "default":
        assert budget._remaining < 16 * 1024**2


def test_capture_digest_uses_original_reader_with_explicit_budget(tmp_path):
    root = tmp_path / "data"
    root.mkdir(mode=0o700)
    source = root / "source.yaml"
    payload = b"actual digest source"
    source.write_bytes(payload)
    source.chmod(0o600)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = admission_authority(tmp_path / "bootstrap")
    authority.register("source", (root,))
    limits = ArchiveLimits()
    budget = _TraceReadBudget()
    with authority.maintenance(("source",), 2) as session:
        info = source.stat()
        with session._capture_bound_sources(
            ((source, info.st_dev, info.st_ino),), stage, limits, limits.expanded_bytes
        ):
            assert not hasattr(session._scopes[-1], "read_budget")
            assert storage._digest_recovery_file(
                "eval.definitions", source, max_bytes=len(payload), _read_budget=budget
            ) == (len(payload), hashlib.sha256(payload).hexdigest())
        assert all(
            not scope.active and not scope.resources for scope in session._scopes
        )
    assert budget._remaining < 16 * 1024**2
    with authority.normal(("source",)):
        assert source.read_bytes() == payload


def test_empty_capture_file_keeps_positive_native_probe_bound(tmp_path):
    root = tmp_path / "data"
    root.mkdir(mode=0o700)
    source = root / "empty.yaml"
    source.write_bytes(b"")
    source.chmod(0o600)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = admission_authority(tmp_path / "bootstrap")
    authority.register("source", (root,))
    limits = ArchiveLimits()
    budget = _TraceReadBudget()
    with authority.maintenance(("source",), 2) as session:
        info = source.stat()
        with session._capture_bound_sources(
            ((source, info.st_dev, info.st_ino),), stage, limits, limits.expanded_bytes
        ):
            storage.copy_capture_file(
                "eval.definitions",
                source,
                stage / "empty",
                Event(),
                max_bytes=1,
                _read_budget=budget,
            )
            assert storage._digest_recovery_file(
                "eval.definitions", stage / "empty", max_bytes=1, _read_budget=budget
            ) == (0, hashlib.sha256(b"").hexdigest())
            assert not session._scopes[-1].resources
        assert all(
            not scope.active and not scope.resources for scope in session._scopes
        )
    assert source.read_bytes() == b""
    assert budget._remaining < 16 * 1024**2


def test_preview_digest_requires_exact_live_native_allowance(tmp_path):
    source = tmp_path / "data.yaml"
    source.write_bytes(b"current scope")
    source.chmod(0o600)
    budget = _TraceReadBudget()
    foreign = _TraceReadBudget()
    with storage._preview_reads(_read_budget=budget):
        with pytest.raises(ValueError, match="mobile_preview_budget_changed"):
            storage._digest_recovery_file(
                "eval.definitions", source, max_bytes=100, _read_budget=foreign
            )
        with pytest.raises(TypeError, match="native_read_budget_required"):
            storage._digest_recovery_file(
                "eval.definitions", source, max_bytes=100, _read_budget=object()
            )
        assert storage._digest_recovery_file(
            "eval.definitions", source, max_bytes=100, _read_budget=budget
        ) == (13, hashlib.sha256(b"current scope").hexdigest())
    assert foreign._remaining == 16 * 1024**2


def test_actual_canonical_capture_has_exact_installed_builtin_members(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed
    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )
    from tldw_chatbook.Character_Chat.visual_identity import ensure_builtin_samira
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    def seed(stores, *, legacy, **options):
        result = _small_native_seed(stores, legacy=legacy, **options)
        if not legacy:
            owner = CharactersRAGDB(stores.chat_path, "builtin-capture-fixture")
            try:
                ensure_builtin_samira(owner)
            finally:
                owner.close_connection()
        return result

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        _, journal, config, _, original_preview, inventory, authority = values
    # The old small fixture's fixed names omit newly included package sources.
    # Use the installed capture-name enrollment before fresh held discovery.
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        _preview_selected_mobile,
    )

    names = _capture_names(authority, inventory)
    with authority.maintenance(names, 2) as session:
        with session._discovery_reads(private_sqlite=True):
            inventory = discover((config,))
            preview = _preview_selected_mobile(config, config)
        assert inventory.complete, inventory.issues
        assert preview.source_fingerprint == original_preview.source_fingerprint
        before = {
            path: path.read_bytes() for path in preview.capture_scope if path.is_file()
        }
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            document = archive_reader._manifest(
                captured.canonical.manifest_bytes, ArchiveLimits(), True
            )
            files = [
                row
                for row in document.files
                if row.owner_id == "persona.visual_identity_builtin"
            ]
            assert len(files) == 31
            # Pure exact member matching is not mobile completion or restoration
            # authority. Its input is the actual capture and native inventory.
            from tldw_chatbook.Backup_Recovery.mobile_builtin_preservation import (
                _member_observation,
                _ordinary_builtin_members,
                _verify_current_builtin_files,
            )

            budget = _TraceReadBudget()
            members = _ordinary_builtin_members(document, inventory, budget)
            assert {row.logical_id for row in files} <= set(members)
            assert all(
                item.owner == "persona.visual_identity_builtin"
                for item in members.values()
            )
            from dataclasses import replace

            from tldw_chatbook.DB.recovery_core import core_adapters
            from tldw_chatbook.Persona_Visual.recovery import recovery_adapters

            first = files[0]
            producer = next(
                row
                for row in document.producer_inventory
                if row.logical_id == first.logical_id
            )
            tampered = document.model_copy(
                update={
                    "producer_inventory": tuple(
                        row.model_copy(update={"dependencies": ()})
                        if row is producer
                        else row
                        for row in document.producer_inventory
                    )
                }
            )
            with pytest.raises(
                ValueError, match="mobile_canonical_builtin_members_changed"
            ):
                _ordinary_builtin_members(tampered, inventory, _TraceReadBudget())
            foreign = replace(
                inventory,
                items=tuple(
                    replace(item, path=tmp_path / "foreign")
                    if item.logical_id == first.logical_id
                    else item
                    for item in inventory.items
                ),
            )
            with pytest.raises(
                ValueError, match="mobile_canonical_builtin_members_changed"
            ):
                _ordinary_builtin_members(document, foreign, _TraceReadBudget())
            altered = document.model_copy(
                update={
                    "files": tuple(
                        row.model_copy(update={"relative_path": "wrong.png"})
                        if row is first
                        else row
                        for row in document.files
                    )
                }
            )
            with pytest.raises(
                ValueError, match="mobile_canonical_builtin_members_changed"
            ):
                _ordinary_builtin_members(altered, inventory, _TraceReadBudget())

            preview_budget = budget
            # Stage validation is a separate real caller; its mapping, copies,
            # digests and both observations share one original native allowance.
            budget = _TraceReadBudget()
            members = _ordinary_builtin_members(document, inventory, budget)
            stage_costs = {"mapped_remaining": budget._remaining}
            stage = tmp_path / "builtin-validation"
            stage.mkdir(mode=0o700)
            core = next(
                item
                for item in inventory.items
                if item.owner == "db.chachanotes.primary"
                and item.path == preview.destination.chat_path
            )
            selected = [
                item for item in members.values() if item.status == "included"
            ] + [core]
            held = tuple(
                (item.path, item.path.stat().st_dev, item.path.stat().st_ino)
                for item in selected
            )
            records = {
                row.logical_id: row for row in (*document.directories, *document.files)
            }
            with session._capture_bound_sources(
                held, stage, ArchiveLimits(), ArchiveLimits().expanded_bytes
            ):
                scope = storage._local.capture_scope
                custody = storage._CaptureFileDescriptors(scope)
                budget.reserve(65536 + 1024)
                raw = bytearray(65536)
                view = memoryview(raw)
                buffer = (raw, view, [view])
                candidates = {}
                cancel = Event()
                try:
                    _verify_current_builtin_files(document, members, preview_budget)
                    bad_digest = document.model_copy(
                        update={
                            "files": tuple(
                                row.model_copy(update={"sha256": "0" * 64})
                                if row is first
                                else row
                                for row in document.files
                            )
                        }
                    )
                    with pytest.raises(
                        ValueError, match="mobile_canonical_builtin_content_changed"
                    ):
                        _verify_current_builtin_files(
                            bad_digest, members, _TraceReadBudget()
                        )
                    observed = {
                        key: _member_observation(
                            item, records[key], budget, custody, cancel, buffer
                        )
                        for key, item in members.items()
                    }
                    with pytest.raises(
                        ValueError, match="mobile_canonical_builtin_content_changed"
                    ):
                        _member_observation(
                            members[first.logical_id],
                            first.model_copy(update={"sha256": "0" * 64}),
                            _TraceReadBudget(),
                            custody,
                            cancel,
                            buffer,
                        )
                    for key in (
                        first.logical_id,
                        next(
                            key
                            for key, item in members.items()
                            if item.status == "included_directory"
                        ),
                    ):
                        record = records[key]
                        changed = record.model_copy(
                            update={
                                "metadata": record.metadata.model_copy(
                                    update={"mtime_ns": record.metadata.mtime_ns + 1}
                                )
                            }
                        )
                        with pytest.raises(
                            ValueError,
                            match="mobile_canonical_builtin_metadata_changed",
                        ):
                            _member_observation(
                                members[key],
                                changed,
                                _TraceReadBudget(),
                                custody,
                                cancel,
                                buffer,
                            )
                    stage_costs["before_observed_remaining"] = budget._remaining
                    for key, item in members.items():
                        path = stage / hashlib.sha256(key.encode()).hexdigest()
                        if item.status == "included_directory":
                            path.mkdir(mode=0o700)
                        else:
                            record = records[key]
                            storage.copy_capture_file(
                                item.owner,
                                item.path,
                                path,
                                cancel,
                                max_bytes=max(1, record.size),
                                _read_budget=budget,
                            )
                            assert storage._digest_recovery_file(
                                item.owner,
                                path,
                                max_bytes=max(1, record.size),
                                _read_budget=budget,
                            ) == (record.size, record.sha256)
                        candidates[key] = path
                    stage_costs["copied_remaining"] = budget._remaining
                    core_copy = stage / "core.sqlite"
                    core_owner = next(
                        row for row in core_adapters() if row.owner_id == core.owner
                    )
                    core_owner.capture(core, core_copy, cancel)
                    candidates[core.logical_id] = core_copy
                    asset_owner = next(
                        row
                        for row in recovery_adapters()
                        if row.owner_id == "persona.visual_identity_builtin"
                    )
                    root = next(
                        item
                        for item in members.values()
                        if item.metadata.parent_id is None
                    )
                    assert (
                        asset_owner.validate_dependencies(
                            root, candidates[root.logical_id], candidates
                        )
                        == ()
                    )
                    candidates[first.logical_id].write_bytes(
                        b"tampered private validation copy"
                    )
                    assert asset_owner.validate_dependencies(
                        root, candidates[root.logical_id], candidates
                    ) == ("asset_digest_mismatch",)
                    assert {
                        key: _member_observation(
                            item, records[key], budget, custody, cancel, buffer
                        )
                        for key, item in members.items()
                    } == observed
                    stage_costs["after_observed_remaining"] = budget._remaining
                    (tmp_path / "builtin-stage-costs.json").write_text(
                        __import__("json").dumps(stage_costs, indent=2) + "\n"
                    )
                finally:
                    custody.retire()
                assert not scope.resources
            assert {path: path.read_bytes() for path in before} == before
            with journal._locked(exclusive=False) as parent:
                assert not any(
                    row.event in {"mobile_owner_committed", "mobile_committed"}
                    for row in journal._records(parent)
                )


def test_existing_bound_public_builtin_start(tmp_path, monkeypatch):
    """Actual existing profile must carry its current included builtin scope."""
    import json
    import traceback

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery import bootstrap, recovery_service
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Character_Chat.visual_identity import ensure_builtin_samira
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    def seed(stores, *, legacy, **options):
        result = _small_native_seed(stores, legacy=legacy, **options)
        if not legacy:
            owner = CharactersRAGDB(stores.chat_path, "builtin-existing-binding")
            try:
                ensure_builtin_samira(owner)
            finally:
                owner.close_connection()
        return result

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        _, journal, config, stores, preview, inventory, authority = values
    from tldw_chatbook.Chat import console_trace_read_budget

    cutover_budgets = []
    native_cutover_budget = console_trace_read_budget._mobile_cutover_budget

    def observed_cutover_budget():
        actual = native_cutover_budget()
        cutover_budgets.append(actual)
        return actual

    monkeypatch.setattr(
        console_trace_read_budget, "_mobile_cutover_budget", observed_cutover_budget
    )

    def observe_cutover():
        try:
            assert require_mobile_cutover(preview.source_dir, stores) is None
        finally:
            (tmp_path / "builtin-cutover-counters.json").write_text(
                json.dumps(
                    {
                        "observation_point": "ordinary_gate_finally_after_service_close",
                        "budgets": [
                            {
                                "identity": id(actual),
                                "parent_identity": (
                                    id(actual._parent)
                                    if actual._parent is not None
                                    else None
                                ),
                                "remaining": actual._remaining,
                            }
                            for actual in cutover_budgets
                        ],
                    },
                    indent=2,
                )
            )

    errors = []
    issue_code = recovery_service.issue_code

    def issue(error, **kwargs):
        errors.append(
            {
                "type": type(error).__name__,
                "code": str(error),
                "stack": traceback.format_tb(error.__traceback__),
            }
        )
        return issue_code(error, **kwargs)

    monkeypatch.setattr(recovery_service, "issue_code", issue)
    digest_errors = []
    native_digest = storage._digest_recovery_file

    def observed_digest(*args, **kwargs):
        try:
            return native_digest(*args, **kwargs)
        except Exception as error:
            if not digest_errors:
                scope = (
                    getattr(storage._local, "discovery_scope", None)
                    or getattr(storage._local, "capture_scope", None)
                    or getattr(storage._local, "preview_scope", None)
                )
                digest_errors.append(
                    {
                        "type": type(error).__name__,
                        "code": str(error),
                        "owner": args[0],
                        "path": str(args[1]),
                        "scope": type(scope).__name__,
                        "stack": traceback.format_tb(error.__traceback__),
                    }
                )
            raise

    monkeypatch.setattr(storage, "_digest_recovery_file", observed_digest)
    # Diagnostic only: the installed predicate decides first and is returned unchanged.
    from tldw_chatbook.Backup_Recovery import sqlite_validation
    from tldw_chatbook.Backup_Recovery.mobile_captured_sqlite import _CapturedSQLite

    expired_observations = []
    native_expired = sqlite_validation._Restrictions.expired

    def observed_expired(restriction):
        result = native_expired(restriction)
        scope = getattr(restriction, "capture_scope", None)
        if result and not expired_observations and type(scope) is _CapturedSQLite:
            observed_at = sqlite_validation.monotonic()
            members = []
            failing_member = None
            for key, member in scope._connections.items():
                actual = member[1]
                if actual is restriction:
                    failing_member = key
                members.append(
                    {
                        "member": key,
                        "deadline": actual.deadline,
                        "steps": actual.steps,
                        "cancelled": actual.cancel is not None
                        and actual.cancel.is_set(),
                    }
                )
            expired_observations.append(
                {
                    "native_result": result,
                    "observed_at_after_native_call": observed_at,
                    "deadline": restriction.deadline,
                    "steps": restriction.steps,
                    "cancelled": restriction.cancel is not None
                    and restriction.cancel.is_set(),
                    "seconds_limit": sqlite_validation._SECONDS,
                    "step_limit": sqlite_validation._STEP_BUDGET,
                    "progress_interval": sqlite_validation._PROGRESS_INTERVAL,
                    "operation_id": scope.journal.operation_id,
                    "member": failing_member,
                    "retained_members": members,
                }
            )
        return result

    monkeypatch.setattr(sqlite_validation._Restrictions, "expired", observed_expired)

    source_before = mobile_consolidation_fixture.fingerprints(preview.source_dir)
    target_before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in stores.paths
        if path.is_file()
    }
    root = authority.control_root.parent
    _, profiles = bootstrap._records(root)
    registry = bootstrap._registry(root)
    package = [
        str(item.path)
        for item in inventory.items
        if item.owner == "persona.visual_identity_builtin"
        and item.status == "included_directory"
    ]
    package_items = [
        item
        for item in inventory.items
        if item.owner == "persona.visual_identity_builtin"
        and item.status in {"included", "included_directory"}
    ]

    def package_facts():
        return {
            str(item.path): {
                "inode": item.path.stat().st_ino,
                "device": item.path.stat().st_dev,
                "mode": item.path.stat().st_mode,
                "mtime_ns": item.path.stat().st_mtime_ns,
                "sha256": hashlib.sha256(item.path.read_bytes()).hexdigest()
                if item.path.is_file()
                else None,
            }
            for item in package_items
        }

    package_before = package_facts()
    budgets = []
    from tldw_chatbook.Chat.console_trace_read_budget import _mobile_budget_pair

    native_init = _TraceReadBudget.__init__

    def counted(actual):
        native_init(actual)
        budgets.append(actual)

    monkeypatch.setattr(_TraceReadBudget, "__init__", counted)

    def pairs():
        return [
            {
                "coupled": _mobile_budget_pair(row._parent, row),
                "aggregate_remaining": row._parent._remaining,
                "native_remaining": row._remaining,
            }
            for row in budgets
            if row._parent is not None
        ]

    state = None
    service = RecoveryService(journal.root.parent)
    try:
        operation = service.start_mobile_consolidation(
            preview,
            destination_confirmed=True,
            rollback_password=b"test-private-mobile-rollback",
        )
        state = dict(service.wait(operation))
        assert state["state"] == "succeeded", (state, errors)
    finally:
        service.close()
        source_after = mobile_consolidation_fixture.fingerprints(preview.source_dir)
        target_after = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in stores.paths
            if path.is_file()
        }
        profile_violations, network_violations = take_violations(), blocked_attempts()
        (tmp_path / "builtin-existing-binding.json").write_text(
            json.dumps(
                {
                    "state": state,
                    "errors": errors,
                    "digest_errors": digest_errors,
                    "expired_observations": expired_observations,
                    "selector": str(config),
                    "profiles": profiles,
                    "registry": registry,
                    "included_package_directories": package,
                    "source_before": source_before,
                    "source_after": source_after,
                    "target_before": target_before,
                    "target_after": target_after,
                    "profile_violations": profile_violations,
                    "network_violations": network_violations,
                    "service_closed": True,
                    "profiles_after_public": bootstrap._records(root)[1],
                    "registry_after_public": bootstrap._registry(root),
                    "package_before": package_before,
                    "package_after": package_facts(),
                    "pairs_after_public_close": pairs(),
                },
                indent=2,
                default=str,
            )
        )
        assert source_before == source_after
        assert not profile_violations and not network_violations

    assert bootstrap._records(root)[1] == profiles
    assert package_facts() == package_before
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        require_mobile_cutover,
    )

    observe_cutover()
    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
    from tldw_chatbook.Backup_Recovery.recovery_copies import list_recovery_copies

    copies = [
        row
        for row in list_recovery_copies(journal.root.parent)
        if row.operation_id == operation
    ]
    assert (
        len(copies) == 1
        and copies[0].status == "verified"
        and not copies[0].pending_operation
    )
    edited = CharactersRAGDB(stores.chat_path, "builtin-later-edit")
    try:
        note = edited.add_note("New safety copy note", "Retain exact post-import edit")
    finally:
        edited.close_connection()
    selected = tuple(dict.fromkeys((preview.source_selector, preview.target_selector)))
    current = preview_capture(selected, options={})
    assert current.complete
    later = RecoveryService(journal.root.parent)
    report_path = tmp_path / "builtin-existing-binding.json"
    report = json.loads(report_path.read_text())
    try:
        plan = later.preview_rollback(
            operation, old_password=b"test-private-mobile-rollback", target=current
        )
        report["later_preview"] = True
        token = later.start_rollback(
            operation,
            plan,
            old_password=b"test-private-mobile-rollback",
            new_password=b"new-test-only-safety-password",
        )
        replaced = dict(later.wait(token))
        report["replacement"] = replaced
        assert replaced["state"] == "succeeded", (replaced, errors)
        replacement_id = replaced["result"]["journal_operation_id"]
        assert replacement_id != operation and later.pending_operations() == ()
    finally:
        later.close()
        report.update(
            errors=errors,
            source_after_replacement=mobile_consolidation_fixture.fingerprints(
                preview.source_dir
            ),
            target_after_replacement={
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in stores.paths
            },
            package_after_replacement=package_facts(),
            pairs_after_all_services_close=pairs(),
            profile_violations=take_violations(),
            network_violations=blocked_attempts(),
        )
        report_path.write_text(json.dumps(report, indent=2, default=str))
        assert report["source_after_replacement"] == source_before
        assert report["package_after_replacement"] == package_before
        assert not report["profile_violations"] and not report["network_violations"]
    restored = CharactersRAGDB(stores.chat_path, "builtin-restored-reader")
    try:
        assert restored.get_all_conversation_ids() == ["shared-chat"]
        assert restored.get_note_by_id(note) is None
    finally:
        restored.close_connection()
    safety = next(
        row
        for row in list_recovery_copies(journal.root.parent)
        if row.operation_id == replacement_id
    )
    assert safety.status == "verified" and not safety.pending_operation
    saved = archive_reader.acquire(
        safety.path,
        tmp_path / "new-safety",
        ArchiveLimits(),
        b"new-test-only-safety-password",
        Event(),
    )
    document = archive_reader.verify_sealed(saved)
    core = next(
        row for row in document.files if row.owner_id == "db.chachanotes.primary"
    )
    import sqlite3
    import zipfile
    from contextlib import closing

    snapshot = tmp_path / "safety-core.db"
    with zipfile.ZipFile(saved.path) as packed:
        snapshot.write_bytes(packed.read(core.payload))
    snapshot.chmod(0o600)
    with closing(
        sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True)
    ) as connection:
        assert connection.execute(
            "SELECT content FROM notes WHERE id=?", (note,)
        ).fetchone() == ("Retain exact post-import edit",)
    # Direct native restore authenticates the new canonical generation;
    # imported batch rows are deliberately absent from this preimage.
    observe_cutover()
    report.update(
        safety_copy_verified=True,
        restored_gate_accepted=True,
        source_final=mobile_consolidation_fixture.fingerprints(preview.source_dir),
        package_final=package_facts(),
        profile_violations=take_violations(),
        network_violations=blocked_attempts(),
    )
    report_path.write_text(json.dumps(report, indent=2, default=str))
    assert (
        report["source_final"] == source_before
        and report["package_final"] == package_before
    )
    assert not report["profile_violations"] and not report["network_violations"]
    assert len(pairs()) == 1 and pairs()[0]["coupled"]
