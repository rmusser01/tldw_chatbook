"""Related installed source members retain one finite original admission."""

import errno
import inspect
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_raw_native_retirement as retirement_cases
from tldw_chatbook.Agents import hook_permissions
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
)
from tldw_chatbook.MCP.local_store import LocalMCPStore, LocalMCPStoreState
from tldw_chatbook.MCP.permission_store import MCPPermissionStore

configured_source = retirement_cases.configured_source
local_root = retirement_cases.local_root


@pytest.fixture
def installed_source_case(configured_source, monkeypatch, request):
    """Use actual canonical producers installed against one private profile."""
    kind = request.param
    data = configured_source.get_user_data_dir()
    if kind == "permission":
        source = MCPPermissionStore(data / "mcp_permissions.json")
        source.set_kill_switch(True)
        source.set_kill_switch(False)
        path, route, read = source.path, "mcp_store", source.get_kill_switch
    elif kind == "catalog":
        source = LocalMCPStore(data / "local_mcp_store.json")
        source.save(LocalMCPStoreState())
        path, route, read = source.path, "mcp_store", source.load
    else:
        assert kind == "hook"
        monkeypatch.setattr(hook_permissions, "config", configured_source)
        source = hook_permissions.HookPermissions()
        path, route, read = (
            data / "hook_permissions.json",
            "hook_permissions",
            source.snapshot,
        )
        initial = source.snapshot()
        assert initial.ready and initial.rows == () and initial.store_path == path
    assert path.is_file(), "fixture must use the actual persisted source"
    try:
        yield SimpleNamespace(
            kind=kind, source=source, path=path, route=route, read=read
        )
    finally:
        if kind == "hook":
            source.close()


@contextmanager
def _original_scope_acquisitions(source):
    """Count untouched acquisition bodies only under this source's exact scope."""
    scope_code = inspect.unwrap(raw._scope).__code__
    acquire_function = storage.acquire_storage
    acquire_code = acquire_function.__code__
    check_function, participant_function = raw._check, raw._participant_state
    check_code, participant_code = (
        check_function.__code__,
        participant_function.__code__,
    )
    reuse_function = storage._reuse_evidence
    reuse_code = reuse_function.__code__
    evidence_code = storage._observe_evidence.__code__
    close_code = raw._close_descriptor.__code__
    private_module = sys.modules["tldw_chatbook.Utils.private_paths"]
    parent_function = private_module._open_verified_parent
    parent_code = parent_function.__code__
    runtime_function = raw._runtime_operation
    runtime_code = runtime_function.__code__
    requests, returned, entries, closes = [], [], [], []
    active_acquires, active_closes, active_reuses = {}, {}, {}
    active_checks, active_native = {}, {}
    check_rows, native_rows, acquire_rows = {}, {}, {}
    native_overflow = dict(calls=0, exits=0, inclusive_seconds=0.0)
    source_prefix = Path(__file__).resolve().parents[2].as_posix() + "/"
    # contextlib.__enter__ resumes this generator directly from the measured read.
    entering_frame = sys._getframe(2)
    measured_read_root = (
        id(entering_frame)
        if entering_frame.f_code is _assert_one_original_read_acquisition.__code__
        else None
    )
    del entering_frame
    previous = sys.getprofile()
    native_module = sys.modules.get("tldw_chatbook.Utils.windows_files")
    native_type = getattr(native_module, "_Native", None)
    native_open = (
        inspect.getattr_static(native_type, "open_handle")
        if native_type is not None
        else None
    )
    native_code = native_open.__code__ if native_open is not None else None
    native_info = (
        inspect.getattr_static(native_type, "info") if native_type is not None else None
    )
    native_info_code = native_info.__code__ if native_info is not None else None
    requested_names, object_opens, open_identities = {}, {}, {}
    observed = SimpleNamespace(
        requests=requests,
        returned=returned,
        entries=entries,
        closes=closes,
        active_acquires=active_acquires,
        active_closes=active_closes,
        native_opens=0,
        native_open_requests=[],
        native_object_opens=[],
        native_object_overflow=0,
        native_request_overflow=0,
        native_opens_outside_acquisitions=0,
        acquisition_details=[],
        detail_overflow=0,
        all_acquisition_count=0,
        all_acquisition_callers=[],
        check_callers=[],
        parent_walk_depth=0,
        parent_walk_calls=0,
        parent_walk_exits=0,
        native_open_buckets=[],
        native_open_overflow=native_overflow,
        attribution_overflow=0,
        native_ancestor_limit_hits=0,
        measured_read_root_observed=measured_read_root is not None,
        native_open_observer_available=native_open is not None,
    )

    def caller(frame):
        if frame is None:
            return ("<unknown>", "<unknown>", 0)
        filename = frame.f_code.co_filename.replace("\\", "/")
        filename = (
            filename[len(source_prefix) :]
            if filename.casefold().startswith(source_prefix.casefold())
            else "<outside-repository>"
        )
        return filename, frame.f_code.co_qualname, frame.f_lineno

    def native_bucket(frame):
        check_caller = check_route = acquisition_caller = fallback = None
        source_proof = participant_seen = truncated = False
        acquisition_kind = "none"
        parent_walk_runtime_check = False
        parent = frame.f_back
        for _ in range(32):
            if parent is None or id(parent) == measured_read_root:
                break
            if fallback is None:
                candidate = caller(parent)
                if candidate[0] not in (
                    "<outside-repository>",
                    "tldw_chatbook/Utils/windows_files.py",
                ):
                    fallback = candidate
            if parent.f_code is participant_code:
                participant_seen = True
            if parent.f_code is check_code and check_caller is None:
                check_caller = caller(parent.f_back)
                state = raw._states.get(parent.f_locals["operation"])
                check_route = state.route if state is not None else "unissued"
                source_proof = participant_seen
                checked = active_checks.get(id(parent))
                if checked is not None:
                    parent_walk_runtime_check = checked[0]["parent_walk_runtime_check"]
            if parent.f_code is acquire_code and acquisition_caller is None:
                acquisition_caller = caller(parent.f_back)
                acquisition_kind = (
                    "direct_member" if id(parent) in active_acquires else "other"
                )
            if check_caller is not None and acquisition_caller is not None:
                break
            parent = parent.f_back
        else:
            if parent is not None and id(parent) != measured_read_root:
                truncated = True
                observed.native_ancestor_limit_hits += 1
                if acquisition_caller is None:
                    acquisition_kind = "unknown_beyond_limit"
        if check_caller is not None or acquisition_caller is not None:
            fallback = None
        return (
            check_caller,
            check_route,
            source_proof,
            parent_walk_runtime_check,
            acquisition_caller,
            acquisition_kind,
            fallback,
            truncated,
        )

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        # This original synchronous body has one profile return, including unwind.
        if frame.f_code is parent_code:
            if event == "call":
                observed.parent_walk_depth += 1
                observed.parent_walk_calls += 1
            elif event == "return":
                assert observed.parent_walk_depth > 0
                observed.parent_walk_depth -= 1
                observed.parent_walk_exits += 1
        # Read identity already returned by the original native open; no extra IO.
        if (
            frame.f_code is native_info_code
            and event == "return"
            and result is not None
            and frame.f_back is not None
            and frame.f_back.f_code is native_code
        ):
            open_identities[id(frame.f_back)] = (
                result.volume,
                (result.index_high << 32) | result.index_low,
            )
        # Whole measured-thread read: include surrounding config work for hooks.
        if frame.f_code is native_code:
            if event == "call":
                observed.native_opens += 1
                values = frame.f_locals
                request_key = (
                    values["name"],
                    values["parent"] is not None,
                    values["directory"],
                    values["metadata"],
                )
                request_row = requested_names.get(request_key)
                if request_row is None and len(requested_names) < 256:
                    request_row = dict(
                        zip(
                            ("name", "relative_to_parent", "directory", "metadata"),
                            request_key,
                            strict=True,
                        ),
                        calls=0,
                    )
                    requested_names[request_key] = request_row
                    observed.native_open_requests.append(request_row)
                if request_row is None:
                    observed.native_request_overflow += 1
                else:
                    request_row["calls"] += 1
                if not active_acquires:
                    observed.native_opens_outside_acquisitions += 1
                for detail in active_acquires.values():
                    if detail is not None:
                        detail["native_opens"] += 1
                key = native_bucket(frame)
                row = native_rows.get(key)
                if row is None and len(native_rows) < 64:
                    row = dict(
                        zip(
                            (
                                "check_caller",
                                "check_route",
                                "source_proof",
                                "parent_walk_runtime_check",
                                "acquisition_caller",
                                "acquisition_kind",
                                "fallback_caller",
                                "ancestry_truncated",
                            ),
                            key,
                            strict=True,
                        ),
                        calls=0,
                        exits=0,
                        inclusive_seconds=0.0,
                    )
                    native_rows[key] = row
                    observed.native_open_buckets.append(row)
                if row is None:
                    row = native_overflow
                row["calls"] += 1
                if len(active_native) < 32:
                    active_native[id(frame)] = row, time.perf_counter()
                else:
                    observed.attribution_overflow += 1
            elif event == "return" and id(frame) in active_native:
                row, started = active_native.pop(id(frame))
                row["inclusive_seconds"] += time.perf_counter() - started
                row["exits"] += 1
                identity = open_identities.pop(id(frame), None)
                if result is not None and identity is not None:
                    object_row = object_opens.get(identity)
                    if object_row is None and len(object_opens) < 256:
                        object_row = dict(
                            identity=identity, calls=0, requested_names=[]
                        )
                        object_opens[identity] = object_row
                        observed.native_object_opens.append(object_row)
                    if object_row is None:
                        observed.native_object_overflow += 1
                    else:
                        object_row["calls"] += 1
                        name = frame.f_locals["name"]
                        if (
                            name not in object_row["requested_names"]
                            and len(object_row["requested_names"]) < 4
                        ):
                            object_row["requested_names"].append(name)
        if frame.f_code is check_code:
            if event == "call":
                state = raw._states.get(frame.f_locals["operation"])
                route = state.route if state is not None else "unissued"
                parent_walk_runtime_check = (
                    observed.parent_walk_depth > 0
                    and frame.f_back is not None
                    and frame.f_back.f_code is runtime_code
                )
                key = route, caller(frame.f_back), parent_walk_runtime_check
                row = check_rows.get(key)
                if row is None and len(check_rows) < 64:
                    row = dict(
                        route=route,
                        caller=key[1],
                        parent_walk_runtime_check=parent_walk_runtime_check,
                        calls=0,
                        exits=0,
                        inclusive_seconds=0.0,
                    )
                    check_rows[key] = row
                    observed.check_callers.append(row)
                if row is not None and len(active_checks) < 32:
                    row["calls"] += 1
                    active_checks[id(frame)] = row, time.perf_counter()
                else:
                    observed.attribution_overflow += 1
            elif event == "return" and id(frame) in active_checks:
                row, started = active_checks.pop(id(frame))
                row["inclusive_seconds"] += time.perf_counter() - started
                row["exits"] += 1
        if (
            frame.f_code is evidence_code
            and event == "return"
            and frame.f_back is not None
            and id(frame.f_back) in active_reuses
        ):
            row = active_reuses[id(frame.f_back)]
            row["evidence_observation_returned"] = isinstance(result, tuple)
            if isinstance(result, tuple):
                differences = []
                for entry, actual in zip(
                    frame.f_locals["entries"], result, strict=True
                ):
                    for kind, wanted, observed_stamps in zip(
                        ("posture", "content"),
                        (entry.posture, entry.content),
                        actual,
                        strict=True,
                    ):
                        for (path, expected), observed_stamp in zip(
                            wanted, observed_stamps, strict=True
                        ):
                            if expected != observed_stamp:
                                differences.append(
                                    dict(
                                        kind=kind,
                                        leaf=path.name,
                                        expected=repr(expected),
                                        observed=repr(observed_stamp),
                                    )
                                )
                row["evidence_differences"] = differences[:16]
                row["evidence_difference_overflow"] = max(0, len(differences) - 16)
        if frame.f_code is reuse_code:
            if event == "call":
                parent = frame.f_back
                for _ in range(16):
                    if parent is None or id(parent) in active_acquires:
                        break
                    parent = parent.f_back
                detail = active_acquires.get(id(parent))
                if detail is not None:
                    if len(detail["reuse"]) >= 4:
                        observed.detail_overflow += 1
                        return
                    values = frame.f_locals
                    hold = storage._holds.get((os.getpid(), str(values["root"])))
                    evidence = (
                        hold.evidence.get(str(values["selector"]))
                        if hold is not None
                        else None
                    )
                    row = dict(
                        selector_evidence_present=evidence is not None,
                        selector_confirmed=evidence is not None and evidence.confirmed,
                        names=(
                            "absent"
                            if evidence is None
                            else "unbound"
                            if evidence.names == (storage.UNBOUND_NAMESPACE,)
                            else "bound"
                        ),
                        outcome="unfinished",
                    )
                    detail["reuse"].append(row)
                    active_reuses[id(frame)] = row
            elif event == "return" and id(frame) in active_reuses:
                row = active_reuses.pop(id(frame))
                row["outcome"] = (
                    "lease"
                    if type(result) is storage.StorageLease
                    else "none_or_unwind"
                    if result is None
                    else "unexpected"
                )
                row["return_line"] = frame.f_lineno
        elif frame.f_code is acquire_code:
            if event == "call":
                observed.all_acquisition_count += 1
                key = caller(frame.f_back)
                row = acquire_rows.get(key)
                if row is None and len(acquire_rows) < 64:
                    row = dict(caller=key, calls=0)
                    acquire_rows[key] = row
                    observed.all_acquisition_callers.append(row)
                if row is None:
                    observed.attribution_overflow += 1
                else:
                    row["calls"] += 1
                parent = frame.f_back
                if (
                    parent is not None
                    and parent.f_code is scope_code
                    and parent.f_locals.get("source") is source
                ):
                    state = parent.f_locals["state"]
                    assert not state.active
                    request = (
                        frame.f_locals["path"],
                        tuple(frame.f_locals["related_paths"]),
                    )
                    requests.append(request)
                    detail = None
                    if len(observed.acquisition_details) < 16:
                        detail = dict(
                            member_count=1 + len(request[1]), native_opens=0, reuse=[]
                        )
                        observed.acquisition_details.append(detail)
                    else:
                        observed.detail_overflow += 1
                    active_acquires[id(frame)] = detail
            elif event == "return" and id(frame) in active_acquires:
                active_acquires.pop(id(frame))
                assert isinstance(result, storage.StorageLease)
                returned.append(result)
        elif (
            frame.f_code is scope_code
            and event == "return"
            and frame.f_locals.get("source") is source
            and type(result) is raw._RawOperation
            and frame.f_locals.get("previous") is not result
        ):
            state = raw._states[result]
            assert state.active and state.pinned and state.participant is not None
            assert state.source is source and state.leases and state.pins
            assert all(lease in storage._live_leases for lease in state.leases)
            for fd in state.pins.values():
                raw.os.fstat(fd)
            entries.append((result, state, tuple(state.leases)))
        elif frame.f_code is close_code:
            state = frame.f_locals["state"]
            if state.source is not source:
                return
            if event == "call":
                fd = frame.f_locals["fd"]
                raw.os.fstat(fd)
                active_closes[id(frame)] = state, fd
            elif event == "return" and id(frame) in active_closes:
                state, fd = active_closes.pop(id(frame))
                try:
                    raw.os.fstat(fd)
                except OSError as error:
                    closes.append(
                        (error.errno == errno.EBADF, fd not in state.descriptors)
                    )
                else:
                    closes.append((False, fd not in state.descriptors))

    sys.setprofile(observe)
    try:
        yield observed
    finally:
        sys.setprofile(previous)
        assert storage.acquire_storage is acquire_function
        assert acquire_function.__code__ is acquire_code
        assert raw._check is check_function and check_function.__code__ is check_code
        assert raw._participant_state is participant_function
        assert participant_function.__code__ is participant_code
        assert sys.modules.get("tldw_chatbook.Utils.private_paths") is private_module
        assert private_module._open_verified_parent is parent_function
        assert parent_function.__code__ is parent_code
        assert parent_function.__globals__ is vars(private_module)
        assert raw._runtime_operation is runtime_function
        assert runtime_function.__code__ is runtime_code
        assert runtime_function.__globals__ is vars(raw)
        assert observed.parent_walk_depth == 0
        assert observed.parent_walk_calls == observed.parent_walk_exits
        assert not active_checks and not active_native
        assert all(row["calls"] == row["exits"] for row in check_rows.values())
        assert all(row["calls"] == row["exits"] for row in native_rows.values())
        assert native_overflow["calls"] == native_overflow["exits"]
        assert observed.all_acquisition_count == sum(
            row["calls"] for row in acquire_rows.values()
        )
        assert (
            observed.native_opens
            == sum(row["calls"] for row in native_rows.values())
            + native_overflow["calls"]
        )
        assert storage._reuse_evidence is reuse_function
        assert reuse_function.__code__ is reuse_code
        assert not active_reuses
        if native_open is not None:
            assert sys.modules.get("tldw_chatbook.Utils.windows_files") is native_module
            assert native_module._Native is native_type
            assert inspect.getattr_static(native_type, "open_handle") is native_open
            assert native_open.__code__ is native_code
            assert native_open.__globals__ is vars(native_module)
            assert inspect.getattr_static(native_type, "info") is native_info
            assert native_info.__code__ is native_info_code
            assert not open_identities


def _assert_one_original_read_acquisition(case, request):
    before = case.path.read_bytes()
    with _original_scope_acquisitions(case.source) as observed:
        output = case.read()
    if case.kind == "permission":
        assert output is False
        assert json.loads(case.path.read_bytes())["kill_switch"] is False
    elif case.kind == "catalog":
        assert isinstance(output, LocalMCPStoreState)
        assert output.to_dict() == LocalMCPStoreState().to_dict()
    else:
        assert output.ready and output.rows == () and output.store_path == case.path
    assert case.path.read_bytes() == before
    assert len(observed.entries) == 1
    operation, state, leases = observed.entries[0]
    assert state.route == case.route and state.selected == case.path
    if case.kind == "permission":
        expected = (
            case.path,
            case.path.with_suffix(".json.tmp"),
            case.path.with_suffix(".json.bak"),
        )
    elif case.kind == "catalog":
        expected = (case.path, case.path.with_suffix(".json.tmp"))
    else:
        assert state.temporary is not None
        assert state.temporary.parent == case.path.parent
        assert state.temporary.name.startswith(".hook_permissions.json.")
        assert state.temporary.name.endswith(".tmp")
        expected = (
            case.path,
            case.path.with_name(case.path.name + ".lock"),
            state.temporary,
            case.path.parent,
        )
    assert state.paths + state.directories == expected
    admitted = tuple(
        member
        for primary, related in observed.requests
        for member in (primary, *related)
    )
    assert admitted == expected, "all original members retain their request order"
    observation_lease = state.mcp_observation_lease
    if observation_lease is not None and all(
        observation_lease is not lease for lease in observed.returned
    ):
        assert isinstance(observation_lease, storage.StorageLease)
        assert sum(lease is observation_lease for lease in leases) == 1
        assert tuple(
            lease for lease in leases if lease is not observation_lease
        ) == tuple(observed.returned)
    else:
        assert tuple(observed.returned) == leases
    assert observed.closes and all(
        closed and removed for closed, removed in observed.closes
    )
    assert not observed.active_acquires and not observed.active_closes
    assert not state.active and not state.uncertain
    assert not state.pins and not state.files and not state.descriptors
    assert operation not in raw._states and operation not in storage._raw_operations
    assert all(lease not in storage._live_leases for lease in leases)
    request.node.user_properties.extend(
        [
            ("original_acquisitions", len(observed.requests)),
            ("original_members", len(admitted)),
            ("native_descriptor_closes", len(observed.closes)),
            ("native_whole_read_opens", observed.native_opens),
            ("native_open_requests", json.dumps(observed.native_open_requests)),
            ("native_object_opens", json.dumps(observed.native_object_opens)),
            ("native_request_overflow", observed.native_request_overflow),
            ("native_object_overflow", observed.native_object_overflow),
            (
                "native_opens_outside_direct_acquisitions",
                observed.native_opens_outside_acquisitions,
            ),
            ("original_acquisition_details", json.dumps(observed.acquisition_details)),
            ("acquisition_detail_overflow", observed.detail_overflow),
            ("all_original_acquisitions", observed.all_acquisition_count),
            (
                "all_original_acquisition_callers",
                json.dumps(observed.all_acquisition_callers),
            ),
            ("original_check_callers", json.dumps(observed.check_callers)),
            ("original_parent_walk_calls", observed.parent_walk_calls),
            ("original_parent_walk_exits", observed.parent_walk_exits),
            ("native_open_ancestry_buckets", json.dumps(observed.native_open_buckets)),
            ("native_open_bucket_overflow", json.dumps(observed.native_open_overflow)),
            ("native_attribution_overflow", observed.attribution_overflow),
            ("native_ancestor_limit_hits", observed.native_ancestor_limit_hits),
            ("elapsed_counts_include_normal_and_exceptional_exits", True),
            ("check_and_native_elapsed_overlap_do_not_add", True),
            ("elapsed_is_observer_instrumented", True),
            ("native_ancestor_limit", 32),
            ("measured_read_root_observed", observed.measured_read_root_observed),
            (
                "native_whole_read_opens_available",
                observed.native_open_observer_available,
            ),
        ]
    )
    assert observed.measured_read_root_observed
    assert observed.attribution_overflow == 0
    assert observed.native_open_overflow["calls"] == 0
    assert observed.detail_overflow == 0
    assert len(observed.requests) == 1, observed.requests
    return observed


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_nested_installed_permission_read_checks_each_scope_once(
    installed_source_case, request
):
    observed = _assert_one_original_read_acquisition(installed_source_case, request)
    counts = {}
    for row in observed.check_callers:
        # Source lines move; the original caller and route define this boundary.
        key = row["route"], *row["caller"][:2]
        counts[key] = counts.get(key, 0) + row["calls"]
    prefix = "tldw_chatbook/Backup_Recovery/"
    scope = ("mcp_store", prefix + "raw_participants.py", "_scope")
    source = ("mcp_store", prefix + "mcp_source_participants.py", "_operation")
    file = ("mcp_store", prefix + "raw_participants.py", "_file")
    assert counts[source] == 1, counts
    assert counts[file] == 1, counts
    request.node.user_properties.extend(
        [
            ("original_scope_checks", counts[scope]),
            ("original_full_raw_checks", sum(counts.values())),
        ]
    )
    # One outer entry plus two nested entries, each freshly checked once.
    assert counts[scope] == 3, counts
    assert counts == {scope: 3, source: 1, file: 1}


@pytest.mark.parametrize(
    "installed_source_case", ["permission", "catalog", "hook"], indirect=True
)
def test_installed_related_members_share_one_original_acquisition(
    installed_source_case, request
):
    _assert_one_original_read_acquisition(installed_source_case, request)


@pytest.mark.parametrize(
    "installed_source_case", ["permission", "catalog", "hook"], indirect=True
)
def test_bound_warm_related_members_share_one_original_acquisition(
    installed_source_case, configured_source, local_root, request
):
    case = installed_source_case
    selector = configured_source._get_effective_config_path()
    key = (os.getpid(), str(local_root))
    previous = storage._startups.pop(key, None)
    if previous is not None:
        previous.close()
    assert key not in storage._holds, "fixture scope change requires retired owners"
    authority = admission_authority(local_root)
    authority.register("related-profile", (case.path.parent, selector))
    bind_profile(local_root, selector, ("related-profile",), local_root / "admission")
    # local_root's existing finalizer owns this replacement startup lease.
    storage._startups[key] = startup = storage.acquire_storage()
    assert startup.execution_scope()[1] == ("related-profile",)

    deadline = time.monotonic() + 10
    reads = 0
    while True:
        output = case.read()
        if case.kind == "hook":
            assert output.ready and output.store_path == case.path
        reads += 1
        with storage._lock:
            hold = storage._holds[key]
            evidence = hold.evidence.get(str(selector))
            primary = hold.path_evidence.get((str(selector), str(case.path)))
            selector_confirmed = evidence is not None and evidence.confirmed
            primary_confirmed = primary is not None and primary.confirmed
        # Two real reads also give the stable primary a chance to confirm; a
        # fresh random hook temporary is deliberately never pre-admitted here.
        if reads >= 2 and selector_confirmed:
            break
        remaining = deadline - time.monotonic()
        assert remaining > 0, "original selector evidence did not confirm"
        time.sleep(min(0.05, remaining))

    request.node.user_properties.extend(
        [
            ("bound_selector_evidence_confirmed", selector_confirmed),
            ("bound_primary_evidence_confirmed", primary_confirmed),
            ("bound_warm_reads", reads),
            ("bound_evidence_settle_ns", storage._EVIDENCE_SETTLE_NS),
        ]
    )
    _assert_one_original_read_acquisition(case, request)
