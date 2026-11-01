"""Actual native source registration keeps accounting separate from authority."""

import pytest

from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit
from tldw_chatbook.Backup_Recovery.storage_admission import _CaptureFileDescriptors
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget


def test_explicit_registration_budget_preserves_native_mapping(tmp_path):
    authority = admission_authority(tmp_path / "bootstrap")
    root = tmp_path / "source"
    root.mkdir(mode=0o700)
    budget = _TraceReadBudget()
    custody = _CaptureFileDescriptors(None)
    try:
        authority.register(
            "actual.source", (root,), _read_budget=budget, _custody=custody
        )
        assert budget._remaining < 16 * 1024**2
        with authority.normal(("actual.source",)):
            assert root.is_dir()
        before = (authority.control_root / "registry.json").read_bytes()
        authority.register("actual.source", (root,))
        assert (authority.control_root / "registry.json").read_bytes() == before
    finally:
        custody.retire()
    assert not custody.fds


def test_registration_budget_refuses_before_publication(tmp_path):
    authority = admission_authority(tmp_path / "bootstrap")
    root = tmp_path / "source"
    root.mkdir(mode=0o700)
    before = {
        p.name: p.read_bytes() for p in authority.control_root.iterdir() if p.is_file()
    }
    budget = _TraceReadBudget()
    budget.reserve(budget._remaining)
    custody = _CaptureFileDescriptors(None)
    try:
        with pytest.raises(_TraceReadLimit):
            authority.register(
                "actual.source", (root,), _read_budget=budget, _custody=custody
            )
    finally:
        custody.retire()
    assert {
        p.name: p.read_bytes() for p in authority.control_root.iterdir() if p.is_file()
    } == before
    assert not custody.fds


@pytest.mark.parametrize("reason", ["cancel", "foreign_budget", "remap"])
def test_budgeted_registration_retains_native_refusals(tmp_path, reason):
    from threading import Event

    from tldw_chatbook.Backup_Recovery.admission import (
        AdmissionCancelled,
        AdmissionError,
    )

    authority = admission_authority(tmp_path / "bootstrap")
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    authority.register("actual.source", (source,))
    before = {
        p.name: (p.stat().st_ino, p.read_bytes())
        for p in authority.control_root.iterdir()
        if p.is_file()
    }
    custody = _CaptureFileDescriptors(None)
    budget = _TraceReadBudget()
    cancel = Event()
    if reason == "cancel":
        cancel.set()
    error = (
        AdmissionCancelled
        if reason == "cancel"
        else ValueError
        if reason == "foreign_budget"
        else AdmissionError
    )
    try:
        with pytest.raises(error):
            authority.register(
                "actual.source",
                (other,),
                _read_budget=object() if reason == "foreign_budget" else budget,
                _custody=custody,
                cancel=cancel,
            )
    finally:
        custody.retire()
    assert {
        p.name: (p.stat().st_ino, p.read_bytes())
        for p in authority.control_root.iterdir()
        if p.is_file()
    } == before
    assert not custody.fds


@pytest.mark.parametrize("include_absent_sqlite", [False, True])
def test_actual_bound_config_capture_names_forward_one_budget(
    tmp_path, monkeypatch, include_absent_sqlite
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_admission import prepared
    from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.storage_admission import (
        _config_capture_bindings,
        _config_capture_sources,
    )
    from tldw_chatbook.Chat.console_trace_read_budget import _mobile_operation_budgets

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    with prepared(tmp_path, monkeypatch) as values:
        _, _, selector, _, _, inventory, authority = values
    profiles = bootstrap._records(authority.control_root.parent)[1]
    expected_names = _capture_names(
        authority, inventory, include_absent_sqlite=include_absent_sqlite
    )
    budget, native = _mobile_operation_budgets()
    custody = _CaptureFileDescriptors(None)
    try:
        bindings = _config_capture_bindings(
            authority.control_root.parent,
            (selector,),
            _read_budget=budget,
            _custody=custody,
        )
        assert len(bindings) == 1 and bindings[0][0] == profiles[0]
        sources = _config_capture_sources(
            authority.control_root.parent,
            (selector,),
            bindings,
            _read_budget=budget,
            _custody=custody,
        )
        assert sources and all(row[2] == selector for row in sources)
        names = _capture_names(
            authority,
            inventory,
            include_absent_sqlite=include_absent_sqlite,
            _read_budget=budget,
            _custody=custody,
        )
        assert names == expected_names
        assert set(profiles[0]["namespaces"]) <= set(names)
        assert bootstrap._records(authority.control_root.parent)[1] == profiles
    finally:
        custody.retire()
    assert not custody.fds and native._remaining == 16 * 1024**2
    assert budget._remaining < 4 * 1024**3 - 512


def test_complete_samira_reuses_registered_names_with_ordinary_budget(
    tmp_path, monkeypatch
):
    """Component only: already registered sources, no stage/bootstrap/archive."""
    import hashlib
    import json
    import stat
    from threading import Event

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Character_Chat.visual_identity import ensure_builtin_samira
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    def seed(stores, *, legacy, **options):
        result = _small_native_seed(stores, legacy=legacy, **options)
        if not legacy:
            owner = CharactersRAGDB(stores.chat_path, "capture-names-samira")
            try:
                ensure_builtin_samira(owner)
            finally:
                owner.close_connection()
        return result

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        _, _, config, stores, preview, inventory, authority = values
    assert inventory.complete, inventory.issues
    package = tuple(
        item.path
        for item in inventory.items
        if item.owner == "persona.visual_identity_builtin" and item.path is not None
    )
    assert package and any(
        item.owner == "persona.visual_identity_builtin" and item.status == "included"
        for item in inventory.items
    )
    # Real public import already registered these namespaces before later stage.
    # Establish exactly that native registry state, without rebinding a profile.
    profiles = bootstrap._records(authority.control_root.parent)[1]
    expected = _capture_names(authority, inventory, include_absent_sqlite=False)
    registry = (authority.control_root / "registry.json").read_bytes()
    assert bootstrap._records(authority.control_root.parent)[1] == profiles

    paths = set(package) | set(stores.paths) | set(preview.capture_scope) | {config}

    def facts():
        result = {}
        for path in paths:
            if not path.exists():
                result[str(path)] = None
                continue
            info = path.lstat()
            result[str(path)] = [
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest()
                if stat.S_ISREG(info.st_mode)
                else None,
            ]
        return result

    before = facts()
    budget = _TraceReadBudget()
    report = {
        "scope": "Complete Samira inventory, existing native namespaces; ordinary16 component only",
        "package_paths": len(package),
        "inventory_items": len(inventory.items),
        "error": None,
    }
    scope = None
    try:
        with storage._preview_reads(_read_budget=budget):
            scope = storage._local.preview_scope
            actual = _capture_names(
                authority,
                inventory,
                include_absent_sqlite=False,
                _read_budget=budget,
                _custody=scope.custody,
                cancel=Event(),
            )
            assert actual == expected
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "code": str(error)}
        raise
    finally:
        report.update(
            remaining=budget._remaining,
            facts_unchanged=facts() == before,
            profiles_unchanged=bootstrap._records(authority.control_root.parent)[1]
            == profiles,
            registry_unchanged=(authority.control_root / "registry.json").read_bytes()
            == registry,
            custody_fds=[] if scope is None else list(scope.custody.fds),
            network=blocked_attempts(),
            profile_guard=take_violations(),
        )
        (tmp_path / "capture-names-samira.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        assert (
            report["facts_unchanged"]
            and report["profiles_unchanged"]
            and report["registry_unchanged"]
        )
        assert (
            not report["custody_fds"]
            and not report["network"]
            and not report["profile_guard"]
        )


@pytest.mark.parametrize(
    "case", ["narrower_growth_alias_unicode", "omitted", "zero", "row_exhaustion"]
)
def test_capture_name_pipeline_keeps_native_resolutions(tmp_path, monkeypatch, case):
    """Native namespace composition control, not full capture/Complete evidence."""
    import inspect
    from collections import Counter
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery import admission, bootstrap
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem

    authority = admission_authority(tmp_path / "bootstrap")
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o700)
    child = parent / "child"
    child.write_bytes(b"child")
    child.chmod(0o600)
    authority.register("existing.child", (child,))
    if case == "omitted":
        config = parent / "config.toml"
        config.write_text('name="profile"\n')
        config.chmod(0o600)
        authority.register("existing.parent", (parent,))
        bind_profile(
            authority.control_root.parent,
            config,
            ("existing.parent", "existing.child"),
            authority.control_root,
        )
        child.unlink()
        selected = (parent,)
    else:
        fresh = tmp_path / "Ω😀"
        fresh.mkdir(mode=0o700)
        leaf = fresh / "leaf"
        leaf.write_bytes(b"leaf")
        leaf.chmod(0o600)
        alias = tmp_path / "alias"
        alias.symlink_to(parent, target_is_directory=True)
        selected = (parent, alias, fresh, leaf)
    inventory = Inventory(
        tuple(
            StorageItem("fixture.source", str(i), path, "included", ())
            for i, path in enumerate(selected)
        ),
        False,
        "component",
        (),
    )
    before = (authority.control_root / "registry.json").read_bytes()
    profiles = bootstrap._records(authority.control_root.parent)[1]
    budget = _TraceReadBudget()
    custody = _CaptureFileDescriptors(None)
    observed, expected, roots = Counter(), {}, set()
    native_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        caller = inspect.currentframe().f_back
        result = native_resolve(path, *args, **kwargs)
        if caller.f_code is _capture_names.__code__:
            assert kwargs == {"strict": True}
            if "raw" not in caller.f_locals:
                roots.add(result)
            else:
                state = caller.f_locals
                root = state["root"]
                expected.setdefault(
                    root,
                    Counter(
                        raw
                        for row in state["registry"].values()
                        for raw in row["roots"]
                        if Path(raw) not in state["omitted"]
                    ),
                )
                observed[root, str(path)] += 1
        return result

    monkeypatch.setattr(Path, "resolve", resolve)
    if case == "zero":
        budget.reserve(budget._remaining)
    if case == "row_exhaustion":
        native_price = admission._price_paths

        def price(values, actual, **kwargs):
            caller = inspect.currentframe().f_back
            if (
                caller.f_code is _capture_names.__code__
                and isinstance(values, list)
                and actual is budget
            ):
                actual.reserve(actual._remaining)
            return native_price(values, actual, **kwargs)

        monkeypatch.setattr(admission, "_price_paths", price)
    try:
        if case in {"zero", "row_exhaustion"}:
            with pytest.raises(_TraceReadLimit):
                _capture_names(
                    authority,
                    inventory,
                    include_absent_sqlite=False,
                    _read_budget=budget,
                    _custody=custody,
                )
            assert not observed
            if case == "zero":
                assert not roots
            assert (authority.control_root / "registry.json").read_bytes() == before
        else:
            actual = _capture_names(
                authority,
                inventory,
                include_absent_sqlite=False,
                _read_budget=budget,
                _custody=custody,
            )
            assert set(expected) == roots
            assert observed == Counter(
                {
                    (root, raw): count
                    for root, rows in expected.items()
                    for raw, count in rows.items()
                }
            )
            assert (
                "existing.child" in actual
                if case != "omitted"
                else "existing.parent" in actual
            )
            if case == "narrower_growth_alias_unicode":
                assert any(name.startswith("backup.source.") for name in actual)
                assert len({tuple(rows) for rows in expected.values()}) > 1
            else:
                assert all(str(child) not in rows for rows in expected.values())
            assert actual == _capture_names(
                authority, inventory, include_absent_sqlite=False
            )
            assert bootstrap._records(authority.control_root.parent)[1] == profiles
    finally:
        custody.retire()
    assert not custody.fds
