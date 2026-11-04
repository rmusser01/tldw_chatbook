"""Raw reference proof is issued only by an actual authenticated native capture."""

import json
import sqlite3
from dataclasses import FrozenInstanceError
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_captured_sqlite import _fingerprints
from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture


@pytest.mark.parametrize("distinct_selectors", (False, True))
def test_captured_shared_raw_owners_prove_exact_references_without_writes(
    tmp_path, monkeypatch, distinct_selectors
):
    from tldw_chatbook.Backup_Recovery.inventory import discover, read_profile_config
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        _preview_selected_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )
    from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir

    with _capture_fixture(
        tmp_path, monkeypatch, distinct_selectors=distinct_selectors
    ) as values:
        session, journal, config, _, previous, _, _ = values
        data = user_data_dir(read_profile_config(config))
        persona_path = data / "tldw_chatbook_personas.json"
        permission_path = data / "mcp_permissions.json"
        persona_path.write_text(
            json.dumps(
                {
                    "profiles": [
                        {
                            "id": "persona-1",
                            "name": "Private fixture",
                            "is_active": True,
                            "policy_rules": [],
                        },
                        {
                            "id": "malformed-persona",
                            "version": "private-invalid-version",
                        },
                        {"id": "inactive-persona", "is_active": False},
                    ]
                }
            )
        )
        from Tests.MCP.test_permission_resolution import _valid_lifecycle

        permission_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kill_switch": False,
                    "profiles": {
                        "default": {"global_default": "ask", "servers": {}},
                        "profile-1": {"servers": {}},
                        "removed-profile": {
                            "profile_kind": "tool_pack_tombstone",
                            "tool_pack_lifecycle": _valid_lifecycle(origin="tombstone"),
                            "servers": {},
                        },
                    },
                }
            )
        )
        # Real Workspace references are captured with the raw owners they depend on.
        from tldw_chatbook.Backup_Recovery.profile_paths import legacy_mobile_dir

        with sqlite3.connect(
            legacy_mobile_dir(read_profile_config(config)) / "workspaces.db"
        ) as connection:
            connection.execute(
                "UPDATE workspace_records SET assistant_defaults=? WHERE workspace_id='conflicting-workspace'",
                (
                    json.dumps(
                        {
                            "assistant_kind": "persona",
                            "assistant_id": "persona-1",
                            "persona_memory_mode": "read_only",
                            "tool_policy_profile_id": "profile-1",
                        }
                    ),
                ),
            )
            connection.execute(
                "INSERT INTO workspace_agent_backfill VALUES ('assistant_defaults','2026-10-25')"
            )
        connection.close()
        persona_path.chmod(0o600)
        permission_path.chmod(0o600)
        # Seed only test-owned raw stores before this operation's actual review.
        with session._discovery_reads(private_sqlite=True):
            inventory = discover(
                tuple(dict.fromkeys((config, previous.target_selector)))
            )
            preview = _preview_selected_mobile(config, previous.target_selector)
        assert inventory.complete
        archives = tmp_path / "archives"
        archives.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=archives,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            before = _fingerprints(tmp_path / "profile")
            with captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads:
                from tldw_chatbook.Backup_Recovery.storage_admission import (
                    _mobile_admission,
                )

                assert reads.read_budget is _mobile_admission(session)[2]
                assert reads.read_budget._parent is None
                assert 0 < reads.read_budget._remaining <= 16 * 1024**2
                workspace = reads.workspace_plan()
                assert (
                    not workspace.blockers
                )  # Referenced defaults are backfill-exempt.
                assert {
                    (proof.kind, proof.reference_id)
                    for proof in workspace.authority_references
                } == {
                    ("persona", "persona-1"),
                    ("permission_profile", "profile-1"),
                }
                record = next(
                    dict(row.values)
                    for row in workspace.rows
                    if row.table == "workspace_records"
                )
                from tldw_chatbook.Workspaces.registry_service import (
                    _workspace_from_row,
                )

                assert (
                    _workspace_from_row(record).assistant_defaults.assistant_id
                    == "persona-1"
                )
                persona = reads.authority_reference("persona", "persona-1")
                profile = reads.authority_reference("permission_profile", "profile-1")
                assert persona.reference_id == "persona-1"
                assert profile.reference_id == "profile-1"
                assert persona.owner_id == "personas"
                assert profile.owner_id == "mcp.permissions"
                assert len(profile.member_digest) == 64
                assert len(profile.provenance_digest) == 64
                with pytest.raises(ValueError, match="lifecycle_blocked"):
                    reads.authority_reference("permission_profile", "removed-profile")
                original_permissions = permission_path.read_bytes()
                try:
                    changed = json.loads(original_permissions)
                    changed["profiles"]["default"]["global_default"] = "allow"
                    permission_path.write_text(json.dumps(changed))
                    with pytest.raises(ValueError, match="source_coverage_unproven"):
                        reads.authority_reference("permission_profile", "profile-1")
                    changed = json.loads(original_permissions)
                    changed["kill_switch"] = True
                    permission_path.write_text(json.dumps(changed))
                    with pytest.raises(ValueError, match="source_coverage_unproven"):
                        reads.authority_reference("permission_profile", "profile-1")
                finally:
                    permission_path.write_bytes(original_permissions)

                assert "Private fixture" not in repr(persona)
                with pytest.raises(
                    ValueError, match="mobile_authority_policy_invalid"
                ) as invalid:
                    reads.authority_reference("persona", "malformed-persona")
                assert "private-invalid-version" not in str(invalid.value)
                with pytest.raises(ValueError, match="persona_unavailable"):
                    reads.authority_reference("persona", "inactive-persona")
                with pytest.raises(FrozenInstanceError):
                    persona.reference_id = "different"
                with pytest.raises(ValueError, match="reference_missing"):
                    reads.authority_reference("persona", "missing-persona")
                with pytest.raises(ValueError, match="reference_missing"):
                    reads.authority_reference("permission_profile", "missing-profile")
            assert _fingerprints(tmp_path / "profile") == before
            with pytest.raises(ValueError, match="inactive"):
                reads.authority_reference("persona", "persona-1")


@pytest.mark.parametrize(
    "payload,error_name",
    (
        (b"[general]\nflag=true\n[plugin]\nsetting=7\n", None),
        (b"[general\n", "TOMLDecodeError"),
        (b"general=7\n", "ValidationError"),
        (b"__reserved__={}\n", "ValueError"),
        (b"\xff", "UnicodeDecodeError"),
    ),
)
def test_owned_config_parser_preserves_native_selector_decoding(
    tmp_path, payload, error_name
):
    import tomllib

    from pydantic import ValidationError

    from tldw_chatbook.Backup_Recovery.inventory import (
        _parse_profile_config,
        read_profile_config,
    )
    from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY

    if payload == b"__reserved__={}\n":
        payload = (DISCOVERY_CONTEXT_KEY + "={}\n").encode()
    selected = tmp_path / "selected.toml"
    selected.write_bytes(payload)
    errors = {
        "TOMLDecodeError": tomllib.TOMLDecodeError,
        "ValidationError": ValidationError,
        "ValueError": ValueError,
        "UnicodeDecodeError": UnicodeDecodeError,
    }
    for read, value in (
        (read_profile_config, selected),
        (_parse_profile_config, payload),
    ):
        if error_name is not None:
            with pytest.raises(errors[error_name]):
                read(value)
        else:
            result = read(value)
            assert result["general"] == {"flag": True}
            assert result["plugin"] == {"setting": 7}
            assert result["paths"] == {}


@pytest.mark.parametrize("phase", ["raw", "payload", "json", "sealed"])
def test_authority_budget_refuses_before_unpriced_native_allocation(
    tmp_path, monkeypatch, phase
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery import (
        mobile_captured_authority as authority,
    )
    from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit

    original_reference = authority._authority_reference
    original_bytes, original_payload = authority._checked_bytes, authority._payload
    original_validate, original_regular = (
        authority._validate_reference,
        archive_reader._regular,
    )
    original_hash = archive_reader._hash
    state = {"armed": False, "allocated": False, "hashes": 0}

    def exhaust(scope):
        assert not state["armed"]
        state["armed"] = True
        state["budget"] = scope.read_budget
        state["budget"].reserve(state["budget"]._remaining)

    def allocation(*args, **kwargs):
        state["allocated"] = True
        raise AssertionError(f"unpriced {phase} allocation reached")

    class Stream:
        def __init__(self, actual, scope):
            self.actual, self.scope = actual, scope

        def fileno(self):
            if not state["armed"]:
                exhaust(self.scope)
            return self.actual.fileno()

        def read(self, *args):
            return allocation()

    @contextmanager
    def regular(path, **kwargs):
        with original_regular(path, **kwargs) as stream:
            if not state["armed"] and phase == "raw" and path == state.get("raw_path"):
                # All original native parent/stream custody has already entered.
                yield Stream(stream, state["scope"])
            else:
                if (
                    not state["armed"]
                    and phase == "sealed"
                    and state["hashes"] == 2
                    and path == state.get("sealed_path")
                ):
                    exhaust(state["scope"])
                yield stream

    def reference(scope, *args):
        state["scope"] = scope
        return original_reference(scope, *args)

    def checked(scope, path):
        if phase == "raw" and path.name == "tldw_chatbook_personas.json":
            state["raw_path"] = path
        return original_bytes(scope, path)

    def hashed(*args, **kwargs):
        result = original_hash(*args, **kwargs)
        if args[0] == state.get("sealed_path"):
            state["hashes"] += 1
        return result

    def payload(scope, sealed, member):
        if phase == "payload":
            original_zip = authority.zipfile.ZipFile

            @contextmanager
            def opened(*args, **kwargs):
                with original_zip(*args, **kwargs) as archive:

                    class Archive:
                        def infolist(self):
                            return archive.infolist()

                        def getinfo(self, name):
                            info = archive.getinfo(name)
                            assert name == member.payload
                            # Native central-directory/stream custody is already real.
                            exhaust(scope)
                            return info

                    yield Archive()

            with monkeypatch.context() as observe:
                observe.setattr(authority, "zipfile", SimpleNamespace(ZipFile=opened))
                observe.setattr(authority, "bytearray", allocation, raising=False)
                return original_payload(scope, sealed, member)
        if phase == "sealed":
            state["sealed_path"] = sealed.path
            with monkeypatch.context() as observe:
                observe.setattr(
                    authority, "zipfile", SimpleNamespace(ZipFile=allocation)
                )
                return original_payload(scope, sealed, member)
        return original_payload(scope, sealed, member)

    def validate(*args, **kwargs):
        if phase != "json":
            return original_validate(*args, **kwargs)
        exhaust(state["scope"])
        with monkeypatch.context() as observe:
            observe.setattr(
                authority,
                "json",
                SimpleNamespace(loads=allocation, dumps=json.dumps),
            )
            return original_validate(*args, **kwargs)

    monkeypatch.setattr(authority, "_authority_reference", reference)
    monkeypatch.setattr(authority, "_checked_bytes", checked)
    monkeypatch.setattr(authority, "_payload", payload)
    monkeypatch.setattr(authority, "_validate_reference", validate)
    monkeypatch.setattr(archive_reader, "_regular", regular)
    monkeypatch.setattr(archive_reader, "_hash", hashed)
    with pytest.raises(_TraceReadLimit):
        test_captured_shared_raw_owners_prove_exact_references_without_writes(
            tmp_path, monkeypatch, False
        )
    assert state["armed"], "the actual intended allocation boundary was not reached"
    assert not state["allocated"]
    assert state["budget"] is state["scope"].read_budget
    assert not state["scope"].active
    assert not state["scope"].custody.fds
    assert not state["scope"].custody.close_failed


def test_public_authority_refs_keep_original_pair_and_exact_discovery_binding(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager
    from copy import copy
    from dataclasses import replace

    from Tests.Backup_Recovery import test_mobile_public_cutover as public
    from Tests.Backup_Recovery.test_mobile_runs_selector import (
        _assert_missing_history_child_refuses,
    )
    from tldw_chatbook.Agents import recovery as run_logs
    from tldw_chatbook.Backup_Recovery import (
        archive_reader,
        mobile_preview_budget,
        mobile_runs_history,
        mobile_runs_selector,
    )
    from tldw_chatbook.Backup_Recovery import (
        mobile_captured_authority as authority,
    )
    from tldw_chatbook.Backup_Recovery import mobile_history_files as history_files
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.inventory import read_profile_config
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        discovery_context,
    )
    from tldw_chatbook.Backup_Recovery.profile_paths import (
        legacy_mobile_dir,
        user_data_dir,
    )
    from tldw_chatbook.Backup_Recovery.storage_admission import _mobile_admission
    from tldw_chatbook.Chat.console_trace_read_budget import _mobile_budget_pair
    from tldw_chatbook.Skills_Interop import recovery as script_recovery

    original_fixture = public._capture_fixture
    original_selected, original_reference = (
        authority._selected_source,
        authority._authority_reference,
    )
    original_payload = authority._payload
    seen, checked_binding, payloads = set(), [], []
    native_members = history_files._members
    native_logs = run_logs._RunLogs.discover_for_stores
    native_config = mobile_runs_selector._source_config
    native_names = run_logs._run_log_names
    history_children, history_checked = [], []
    history_state = {"scope": None, "names": None}

    def members(captured, document, *, _read_budget=None, _scope=None):
        if _scope is None:
            return native_members(captured, document, _read_budget=_read_budget)
        native = _scope.read_budget
        parent = _mobile_admission(_scope.session)[4]
        assert _read_budget is native and _mobile_budget_pair(parent, native)
        if not history_checked:
            _assert_missing_history_child_refuses(_scope, _read_budget, monkeypatch)

            def forbidden(*args, **kwargs):
                pytest.fail("wrong history counter reached native selector read")

            with monkeypatch.context() as observe:
                observe.setattr(mobile_runs_selector, "_source_config", forbidden)
                for wrong in (None, parent):
                    with pytest.raises(ValueError, match="selector_scope_required"):
                        native_members(
                            captured, document, _read_budget=wrong, _scope=_scope
                        )
        history_state.update(scope=_scope, names=None)
        selected = []

        def config(scope, budget):
            assert scope is _scope and budget is native
            assert storage._local.discovery_scope is None
            return native_config(scope, budget)

        def names(config, *, _read_budget=None):
            assert _read_budget is native
            assert storage._local.discovery_scope is None
            selected.append(True)
            return native_names(config, _read_budget=_read_budget)

        try:
            with monkeypatch.context() as observe:
                observe.setattr(mobile_runs_selector, "_source_config", config)
                observe.setattr(run_logs, "_run_log_names", names)
                result = native_members(
                    captured, document, _read_budget=_read_budget, _scope=_scope
                )
            assert selected == [True]
            assert result[1] is history_state["names"]
            return result
        finally:
            history_state["scope"] = None

    def logs(owner, config, stores=None, **options):
        scope = options.get("_runs_scope")
        if scope is None:
            return native_logs(owner, config, stores, **options)
        assert scope is history_state["scope"]
        child = mobile_preview_budget.current()
        native, parent = scope.read_budget, _mobile_admission(scope.session)[4]
        assert child is storage._local.discovery_scope and child.read_budget is parent
        assert options["_read_budget"] is parent and _mobile_budget_pair(parent, native)
        names = options["_run_names"]
        assert type(names) is frozenset
        if history_state["names"] is None:
            history_state["names"] = names
        assert history_state["names"] is names
        if not history_checked:
            before = native._remaining, parent._remaining
            binding = scope.discovery_bindings[child]

            def forbidden(*args, **kwargs):
                pytest.fail("invalid RunLogs binding reached generic inventory")

            def reject(candidate=config, **changes):
                with pytest.raises(ValueError):
                    native_logs(owner, candidate, stores, **(options | changes))

            with monkeypatch.context() as observe:
                observe.setattr(
                    script_recovery, "default_script_output_root", forbidden
                )
                reject(_runs_scope=copy(scope))
                reject(_runs_scope=child.sqlite_copies)
                reject(_runs_scope=None)
                reject(_run_names=None)
                reject(_run_names=set(names))
                missing = native._remaining, parent._remaining
                reject(_read_budget=None)
                assert missing == (native._remaining, parent._remaining)
                reject(_read_budget=native)
                foreign = dict(config)
                context = discovery_context(config)
                foreign[DISCOVERY_CONTEXT_KEY] = replace(
                    context, config_path=context.config_path.with_name("foreign.toml")
                )
                reject(foreign)
                for holder, field, value in (
                    (child, "read_budget", native),
                    (child.sqlite_copies, "read_budget", native),
                    (child, "custody", copy(child.custody)),
                    (child.custody, "close_failed", True),
                ):
                    with monkeypatch.context() as fault:
                        fault.setattr(holder, field, value)
                        reject()
                with monkeypatch.context() as fault:
                    fault.setattr(storage._local, "discovery_scope", None)
                    reject()
                previous = history_children[0]
                assert previous is not child and not previous.active
                with monkeypatch.context() as fault:
                    fault.setattr(storage._local, "discovery_scope", previous)
                    with pytest.raises(storage.bootstrap.RecoveryRequired):
                        native_logs(owner, config, stores, **options)
            assert scope.discovery_bindings[child] is binding
            assert mobile_preview_budget.current() is child
            assert _mobile_budget_pair(parent, native)
            assert 0 < native._remaining <= before[0]
            assert 0 < parent._remaining <= before[1]
            history_checked.append(scope)
        return native_logs(owner, config, stores, **options)

    def payload(scope, sealed, member):
        if not payloads:
            before = scope.read_budget._remaining

            def reject(*args, _expected_debit=0):
                def physical_hash(*args, **kwargs):
                    pytest.fail(
                        "invalid authority association reached physical hash IO"
                    )

                prior = scope.read_budget._remaining
                try:
                    with monkeypatch.context() as observe:
                        observe.setattr(archive_reader, "_hash", physical_hash)
                        return original_payload(*args)
                finally:
                    assert scope.read_budget._remaining == prior - _expected_debit

            with pytest.raises(ValueError, match="mobile_sqlite_inactive"):
                reject(copy(scope), sealed, member)
            for candidate, selected in ((copy(sealed), member), (sealed, copy(member))):
                with pytest.raises(
                    ValueError, match="mobile_authority_capture_changed"
                ):
                    reject(scope, candidate, selected)
            partition, document, _ = scope._canonical_archive
            for candidate in (
                replace(sealed, encrypted_source=None),
                replace(sealed, manifest_bytes=b"{}"),
                replace(
                    sealed,
                    encrypted_source=replace(sealed.encrypted_source, digest="0" * 64),
                ),
                replace(
                    sealed,
                    encrypted_source=replace(
                        sealed.encrypted_source, plaintext_digest="0" * 64
                    ),
                ),
                replace(
                    sealed,
                    encrypted_source=replace(
                        sealed.encrypted_source, manifest_digest="0" * 64
                    ),
                ),
            ):
                with monkeypatch.context() as observe:
                    observe.setattr(
                        scope, "_canonical_archive", (partition, document, candidate)
                    )
                    with pytest.raises(
                        ValueError, match="mobile_authority_capture_changed"
                    ):
                        reject(
                            scope,
                            candidate,
                            member,
                            _expected_debit=(
                                1024
                                if candidate.encrypted_source is not None
                                and candidate.encrypted_source.manifest_digest
                                == "0" * 64
                                else 0
                            ),
                        )
            with monkeypatch.context() as observe:
                observe.setattr(
                    scope, "_canonical_archive", (copy(partition), document, sealed)
                )
                with pytest.raises(
                    ValueError, match="mobile_native_partition_required"
                ):
                    reject(scope, sealed, member)
            with monkeypatch.context() as observe:
                observe.setattr(
                    scope,
                    "_canonical_archive",
                    (partition, document.model_copy(deep=True), sealed),
                )
                with pytest.raises(
                    ValueError, match="mobile_authority_capture_changed"
                ):
                    reject(scope, sealed, member)
            with monkeypatch.context() as observe:
                observe.setattr(
                    scope.captured, "canonical", copy(scope.captured.canonical)
                )
                with pytest.raises(
                    ValueError, match="mobile_native_capture_cohort_changed"
                ):
                    reject(scope, sealed, member)
            assert scope.read_budget._remaining == before - 1024
        original_hash = archive_reader._hash
        hashes = []

        def hashed(path, cancel, **kwargs):
            assert path == sealed.path and cancel is scope.cancel
            assert kwargs["_custody"] is scope.custody
            assert kwargs["_read_budget"] is scope.read_budget
            result = original_hash(path, cancel, **kwargs)
            hashes.append(result)
            return result

        def unused_manifest(*args, **kwargs):
            pytest.fail("authority proof reconstructed unused manifest data")

        with monkeypatch.context() as observe:
            observe.setattr(archive_reader, "_hash", hashed)
            observe.setattr(archive_reader, "verify_sealed", unused_manifest)
            result = original_payload(scope, sealed, member)
        assert hashes == [sealed.digest] * 4
        payloads.append(result)
        return result

    @contextmanager
    def fixture(*args, **kwargs):
        with original_fixture(*args, **kwargs) as values:
            _, _, selector, _, _, _, _ = values
            config = read_profile_config(selector)
            root = user_data_dir(config)
            for name, payload in (
                (
                    "tldw_chatbook_personas.json",
                    {
                        "profiles": [
                            {"id": "persona-1", "is_active": True, "policy_rules": []}
                        ]
                    },
                ),
                (
                    "mcp_permissions.json",
                    {
                        "schema_version": 1,
                        "kill_switch": False,
                        "profiles": {
                            "default": {"global_default": "ask", "servers": {}},
                            "profile-1": {"servers": {}},
                        },
                    },
                ),
            ):
                path = root / name
                path.write_text(json.dumps(payload))
                path.chmod(0o600)
            connection = sqlite3.connect(legacy_mobile_dir(config) / "workspaces.db")
            try:
                cursor = connection.execute(
                    "UPDATE workspace_records SET assistant_defaults=? WHERE workspace_id='same-workspace'",
                    (
                        json.dumps(
                            {
                                "assistant_kind": "persona",
                                "assistant_id": "persona-1",
                                "persona_memory_mode": "read_only",
                                "tool_policy_profile_id": "profile-1",
                            }
                        ),
                    ),
                )
                assert cursor.rowcount == 1
                connection.execute(
                    "INSERT OR REPLACE INTO workspace_agent_backfill VALUES ('assistant_defaults','2026-10-25')"
                )
                connection.commit()
            finally:
                connection.close()
            yield values

    def selected(scope, *args):
        admission = _mobile_admission(scope.session)
        child = mobile_preview_budget.current()
        assert scope.read_budget is admission[2]
        assert child is not None and child.read_budget is admission[4]
        if not history_children:
            history_children.append(child)
        assert _mobile_budget_pair(child.read_budget, scope.read_budget)
        if not checked_binding:
            binding = scope.discovery_bindings[child]
            before = scope.read_budget._remaining, scope.read_budget._parent._remaining

            def unopened(*args, **kwargs):
                pytest.fail("missing source child reached selector IO")

            with monkeypatch.context() as observe:
                observe.setattr(storage._local, "discovery_scope", None)
                observe.setattr(authority, "_checked_bytes", unopened)
                with pytest.raises(ValueError, match="discovery_child_changed"):
                    original_selected(scope, *args)
            assert before == (
                scope.read_budget._remaining,
                scope.read_budget._parent._remaining,
            )
            for replacement in (
                (binding[0], binding[1], [], binding[3]),
                (binding[0], binding[1], binding[2], []),
            ):
                with monkeypatch.context() as observe:
                    observe.setitem(scope.discovery_bindings, child, replacement)
                    with pytest.raises(ValueError, match="discovery_child_changed"):
                        mobile_preview_budget.current()
            with monkeypatch.context() as observe:
                observe.setattr(child, "custody", copy(child.custody))
                with pytest.raises(ValueError, match="readers_not_retired"):
                    mobile_preview_budget.current()
            with monkeypatch.context() as observe:
                observe.setattr(child.custody, "close_failed", True)
                with pytest.raises(ValueError, match="readers_not_retired"):
                    mobile_preview_budget.current()
            with monkeypatch.context() as observe:
                observe.setattr(child, "read_budget", scope.read_budget)
                with pytest.raises(ValueError, match="discovery_child_changed"):
                    mobile_preview_budget.current()
            with monkeypatch.context() as observe:
                observe.setattr(scope, "read_budget", admission[4])
                with pytest.raises(ValueError, match="mobile_sqlite_admission_changed"):
                    mobile_preview_budget.current()
            assert before == (
                scope.read_budget._remaining,
                scope.read_budget._parent._remaining,
            )
            assert mobile_preview_budget.current() is child
            parent = admission[4]
            try:
                admission[4] = None
                with monkeypatch.context() as observe:
                    observe.setattr(scope.read_budget, "_parent", None)
                    observe.setattr(child, "read_budget", None)
                    observe.setattr(child.sqlite_copies, "read_budget", None)
                    with pytest.raises(
                        ValueError, match="mobile_sqlite_admission_changed"
                    ):
                        mobile_preview_budget.current()
            finally:
                admission[4] = parent
            assert before == (
                scope.read_budget._remaining,
                scope.read_budget._parent._remaining,
            )
            assert mobile_preview_budget.current() is child
            checked_binding.append(True)
        return original_selected(scope, *args)

    def reference(scope, kind, identity):
        admission = _mobile_admission(scope.session)
        native, parent = scope.read_budget, scope.read_budget._parent
        assert admission[2] is native and admission[4] is parent
        assert _mobile_budget_pair(parent, native)
        before = native._remaining, parent._remaining
        original_reserve = type(native).reserve
        witnessed = []

        def reserve(actual, amount):
            if actual is not native or not amount or witnessed:
                return original_reserve(actual, amount)
            assert _mobile_budget_pair(parent, actual)
            reservation = actual._remaining, parent._remaining
            result = original_reserve(actual, amount)
            assert reservation[0] - actual._remaining == amount
            assert reservation[1] - parent._remaining == amount
            assert _mobile_budget_pair(parent, actual)
            witnessed.append(amount)
            return result

        with monkeypatch.context() as observe:
            observe.setattr(type(native), "reserve", reserve)
            result = original_reference(scope, kind, identity)
        assert witnessed
        assert 0 < native._remaining < before[0] <= 16 * 1024**2
        assert before[1] - parent._remaining >= before[0] - native._remaining
        assert _mobile_budget_pair(parent, native)
        seen.add((kind, identity))
        return result

    monkeypatch.setattr(mobile_runs_history, "_members", members)
    monkeypatch.setattr(run_logs._RunLogs, "discover_for_stores", logs)
    monkeypatch.setattr(public, "_capture_fixture", fixture)
    monkeypatch.setattr(authority, "_selected_source", selected)
    monkeypatch.setattr(authority, "_authority_reference", reference)
    monkeypatch.setattr(authority, "_payload", payload)
    # Executes the genuine public producer and retains every existing assertion.
    public.test_public_service_completes_same_native_target(tmp_path, monkeypatch)
    assert checked_binding
    assert history_checked
    assert not history_checked[0].active
    assert not history_checked[0].custody.fds
    assert not history_checked[0].custody.close_failed
    assert seen == {("persona", "persona-1"), ("permission_profile", "profile-1")}


def test_private_receiver_loss_cannot_fall_back_to_coordinator(tmp_path, monkeypatch):
    from copy import copy

    from tldw_chatbook.Backup_Recovery import (
        mobile_captured_authority as authority,
    )
    from tldw_chatbook.Backup_Recovery import mobile_preview_budget, storage_admission

    selected, reference = authority._selected_source, authority._authority_reference
    reached, checked_issuer = [], []

    def check_issuer(scope, *args):
        if not checked_issuer:
            assert storage_admission._local.discovery_scope is None
            before = scope.read_budget._remaining
            with (
                pytest.raises(ValueError, match="mobile_preview_budget_changed"),
                scope.session._discovery_reads(
                    private_sqlite=True, _read_budget=scope.read_budget
                ),
            ):
                pytest.fail(
                    "unsupported explicit captured-discovery budget was admitted"
                )
            assert scope.read_budget._remaining == before
            checked_issuer.append(True)
        return reference(scope, *args)

    def check(scope, *args):
        if not reached:
            assert scope.read_budget._parent is None
            child = storage_admission._local.discovery_scope
            assert child is not None and child.read_budget is None
            assert child.sqlite_copies.read_budget is None
            assert mobile_preview_budget.current() is None
            binding = scope.discovery_bindings[child]
            assert binding == (
                scope,
                child.sqlite_copies,
                child.resources,
                child.sqlite_copies.resources,
            )
            for holder in (child, child.sqlite_copies):
                assert type(holder.custody) is storage_admission._CaptureFileDescriptors
                assert holder.custody.scope is holder
                assert any(resource is holder.custody for resource in holder.resources)
            before = scope.read_budget._remaining
            for replacement in (
                (binding[0], binding[1], [], binding[3]),
                (binding[0], binding[1], binding[2], []),
            ):
                with monkeypatch.context() as observe:
                    observe.setitem(scope.discovery_bindings, child, replacement)
                    with pytest.raises(ValueError, match="discovery_child_changed"):
                        mobile_preview_budget.current()
            with monkeypatch.context() as observe:
                observe.setattr(child, "read_budget", scope.read_budget)
                observe.setattr(child.sqlite_copies, "read_budget", scope.read_budget)
                with pytest.raises(ValueError, match="discovery_child_changed"):
                    mobile_preview_budget.current()
            with monkeypatch.context() as observe:
                observe.setattr(child, "custody", copy(child.custody))
                with pytest.raises(ValueError, match="readers_not_retired"):
                    mobile_preview_budget.current()
            with monkeypatch.context() as observe:
                observe.setattr(child.sqlite_copies.custody, "close_failed", True)
                with pytest.raises(ValueError, match="readers_not_retired"):
                    mobile_preview_budget.current()
            with monkeypatch.context() as observe:
                observe.setattr(scope, "read_budget", None)
                with pytest.raises(ValueError, match="mobile_sqlite_admission_changed"):
                    mobile_preview_budget.current()
            native, ledger = scope.admission[2], scope.admission[3]
            try:
                scope.admission[2] = scope.admission[3] = None
                with monkeypatch.context() as observe:
                    observe.setattr(
                        storage_admission._local, "mobile_sqlite_scope", None
                    )
                    observe.setattr(scope, "read_budget", None)
                    observe.setattr(scope, "discovery_bindings", None)
                    with pytest.raises(
                        ValueError, match="mobile_native_reader_changed"
                    ):
                        mobile_preview_budget.current()
            finally:
                scope.admission[2], scope.admission[3] = native, ledger
            assert scope.read_budget._remaining == before
            assert mobile_preview_budget.current() is None
            reached.append(True)
        return selected(scope, *args)

    monkeypatch.setattr(authority, "_selected_source", check)
    monkeypatch.setattr(authority, "_authority_reference", check_issuer)
    test_captured_shared_raw_owners_prove_exact_references_without_writes(
        tmp_path, monkeypatch, False
    )
    assert reached and checked_issuer


def test_authority_hash_budget_refuses_before_native_buffer_allocation(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery import mobile_captured_authority as authority
    from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit

    original_payload, original_regular = authority._payload, archive_reader._regular
    state = {"armed": False, "allocated": False}

    def allocation(*args, **kwargs):
        state["allocated"] = True
        raise AssertionError("unpriced native hash workspace allocated")

    @contextmanager
    def regular(path, **kwargs):
        with original_regular(path, **kwargs) as stream:
            if path == state.get("path") and not state["armed"]:
                state["armed"] = True
                scope = state["scope"]
                scope.read_budget.reserve(scope.read_budget._remaining)

                class Stream:
                    fileno = stream.fileno
                    readinto = stream.readinto
                    read = allocation

                yield Stream()
            else:
                yield stream

    def payload(scope, sealed, member):
        state.update(scope=scope, path=sealed.path)
        with monkeypatch.context() as observe:
            observe.setattr(archive_reader, "_regular", regular)
            observe.setattr(archive_reader, "bytearray", allocation, raising=False)
            return original_payload(scope, sealed, member)

    monkeypatch.setattr(authority, "_payload", payload)
    with pytest.raises(_TraceReadLimit):
        test_captured_shared_raw_owners_prove_exact_references_without_writes(
            tmp_path, monkeypatch, False
        )
    assert state["armed"], "the actual native hash stream was not reached"
    assert not state["allocated"]
    assert not state["scope"].active
    assert not state["scope"].custody.fds
    assert not state["scope"].custody.close_failed


@pytest.mark.parametrize(
    "content", [b"", b"native bytes" * 7000], ids=("empty", "short-reads")
)
def test_budgeted_seal_hash_handles_actual_short_reads_and_ordinary_default(
    tmp_path, monkeypatch, content
):
    import hashlib
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.models import _TraceReadBudget

    path = tmp_path / "sealed.zip"
    path.write_bytes(content)
    original_regular = archive_reader._regular
    counts = {"read": 0, "readinto": 0}

    @contextmanager
    def regular(*args, **kwargs):
        with original_regular(*args, **kwargs) as stream:

            class Stream:
                fileno = stream.fileno

                def read(self, width):
                    counts["read"] += 1
                    return stream.read(width)

                def readinto(self, buffer):
                    counts["readinto"] += 1
                    return stream.readinto(memoryview(buffer)[:113])

            yield Stream()

    monkeypatch.setattr(archive_reader, "_regular", regular)
    expected = hashlib.sha256(content).hexdigest()
    budget = _TraceReadBudget()
    assert archive_reader._hash(path, Event(), _read_budget=budget) == expected
    assert counts["readinto"] >= 1 and counts["read"] == 0
    assert 0 < budget._remaining < 16 * 1024**2
    counts["readinto"] = 0
    assert archive_reader._hash(path, Event()) == expected
    assert counts["read"] >= 1 and counts["readinto"] == 0


@pytest.mark.parametrize("failure", ["changed", "cancel"])
def test_budgeted_seal_hash_preserves_actual_change_and_cancellation_refusal(
    tmp_path, monkeypatch, failure
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery.models import _TraceReadBudget

    path = tmp_path / "sealed.zip"
    path.write_bytes(b"native sealed bytes")
    original_regular = archive_reader._regular
    cancel = Event()
    reached = []

    @contextmanager
    def regular(*args, **kwargs):
        with original_regular(*args, **kwargs) as stream:

            class Stream:
                fileno = stream.fileno

                def readinto(self, buffer):
                    count = stream.readinto(buffer)
                    if count and not reached:
                        reached.append(True)
                        if failure == "changed":
                            with path.open("ab") as output:
                                output.write(b"changed")
                        else:
                            cancel.set()
                    return count

            yield Stream()

    monkeypatch.setattr(archive_reader, "_regular", regular)
    expected, message = (
        (ValueError, "sealed_changed")
        if failure == "changed"
        else (InterruptedError, "cancelled")
    )
    with pytest.raises(expected, match=message):
        archive_reader._hash(path, cancel, _read_budget=_TraceReadBudget())
    assert reached


@pytest.mark.parametrize("context", ["manifest", "seal"])
def test_authority_budget_refuses_before_native_hash_context(
    tmp_path, monkeypatch, context
):
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import archive_reader
    from tldw_chatbook.Backup_Recovery import mobile_captured_authority as authority
    from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit

    original_payload, original_hash = authority._payload, archive_reader._hash
    state = {"armed": False, "allocated": False}

    def allocation(*args, **kwargs):
        state["allocated"] = True
        raise AssertionError(f"unpriced native {context} hash context allocated")

    def exhaust(scope):
        assert not state["armed"]
        state["armed"] = True
        state["counter"] = scope.read_budget
        scope.read_budget.reserve(scope.read_budget._remaining)

    def hashed(*args, **kwargs):
        exhaust(state["scope"])
        with monkeypatch.context() as observe:
            observe.setattr(
                archive_reader, "hashlib", SimpleNamespace(sha256=allocation)
            )
            return original_hash(*args, **kwargs)

    def payload(scope, sealed, member):
        state["scope"] = scope
        with monkeypatch.context() as observe:
            if context == "manifest":
                # The actual native reference has already passed all source/member reads.
                exhaust(scope)
                observe.setattr(
                    authority, "hashlib", SimpleNamespace(sha256=allocation)
                )
            else:
                observe.setattr(archive_reader, "_hash", hashed)
            return original_payload(scope, sealed, member)

    monkeypatch.setattr(authority, "_payload", payload)
    with pytest.raises(_TraceReadLimit):
        test_captured_shared_raw_owners_prove_exact_references_without_writes(
            tmp_path, monkeypatch, False
        )
    assert state["armed"], "the actual native hash context boundary was not reached"
    assert not state["allocated"]
    assert state["counter"] is state["scope"].read_budget
    assert not state["scope"].active
    assert not state["scope"].custody.fds
    assert not state["scope"].custody.close_failed


def test_cold_ordinary_discovery_keeps_captured_reader_lazy(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(
        tmp_path,
        "ordinary",
        "private-discovery",
        script=r"""
import sys
from pathlib import Path

blocked = "tldw_chatbook.Backup_Recovery.mobile_captured_sqlite"
assert blocked not in sys.modules

class NoCapturedReader:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == blocked:
            raise AssertionError("ordinary_discovery_loaded_captured_reader")

sys.meta_path.insert(0, NoCapturedReader())
from tldw_chatbook.Backup_Recovery.admission import Admission
from tldw_chatbook.Backup_Recovery.mobile_preview_budget import current
from tldw_chatbook.Backup_Recovery.storage_admission import _local

source = Path.home() / "source"
source.mkdir(mode=0o700)
authority = Admission(Path.home() / "control")
authority.register("core", (source,))
for _ in range(2):
    with authority.maintenance(("core",), 1) as session:
        assert getattr(_local, "mobile_sqlite_scope", None) is None
        with session._discovery_reads(private_sqlite=True):
            scope = _local.discovery_scope
            assert scope.session is session and scope.sqlite_copies is not None
            assert scope.read_budget is None and scope.sqlite_copies.read_budget is None
            assert blocked not in sys.modules
            assert current() is None
            assert blocked not in sys.modules
        assert _local.discovery_scope is None
    assert _local.maintenance_session is None and blocked not in sys.modules
print("retired and reopened")
""",
    )


@pytest.mark.parametrize(
    "fault",
    [None, "discover", "project", "config"],
    ids=["reuse", "discover-error", "project-error", "config-mismatch"],
)
def test_authority_member_retains_native_copy_data_and_fresh_checks(
    tmp_path, monkeypatch, fault
):
    from contextlib import nullcontext

    from tldw_chatbook.Backup_Recovery import inventory as inventory_module
    from tldw_chatbook.Backup_Recovery import mobile_captured_authority as authority
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.mobile_captured_sqlite import _CapturedSQLite
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _MobileCapturePartition,
    )

    class DiscoveryFailure(RuntimeError):
        pass

    member = authority._member
    workspace_plan = _CapturedSQLite.workspace_plan
    discover = inventory_module.discover
    project = _MobileCapturePartition.project
    sqlite_target = storage._PreviewScope.sqlite_target
    verify_sources = storage._PreviewScope.verify_sources
    capture_inventory = storage.MaintenanceSession._discover_capture_inventory
    states, active = [], []
    workspace_active = False

    def workspace(scope):
        nonlocal workspace_active
        workspace_active = True
        try:
            return workspace_plan(scope)
        finally:
            workspace_active = False

    def targeted(copies):
        return bool(active) and copies.native_scope is storage._local.discovery_scope

    def target(copies, source):
        result = sqlite_target(copies, source)
        if targeted(copies):
            state = active[-1]
            if state["child"] is None:
                state["child"], state["copies"] = copies.native_scope, copies
            assert state["child"] is copies.native_scope and state["copies"] is copies
            state["targets"][state["phase"]] += 1
        return result

    def verify(copies):
        result = verify_sources(copies)
        if targeted(copies):
            active[-1]["verified"].append(active[-1]["phase"])
        return result

    def discovered(*args, **kwargs):
        if not active:
            return discover(*args, **kwargs)
        state = active[-1]
        state["discovered"] += 1
        result = discover(*args, **kwargs)
        state["inventory"] = result
        if fault == "discover":
            raise DiscoveryFailure("after_actual_discovery")
        return result

    def captured(session, *args, **kwargs):
        if not active or fault != "config":
            return capture_inventory(session, *args, **kwargs)
        child = storage._local.discovery_scope
        assert child is not None and child.config_sources
        active[-1]["child"], active[-1]["copies"] = child, child.sqlite_copies
        with monkeypatch.context() as observe:
            observe.setattr(child, "config_sources", ())
            return capture_inventory(session, *args, **kwargs)

    def projected(partition, session, native_inventory):
        if not active:
            return project(partition, session, native_inventory)
        state = active[-1]
        child, copies = state["child"], state["copies"]
        assert (
            child is storage._local.discovery_scope and child.active and copies.active
        )
        assert state["verified"] == ["discover"]
        assert native_inventory is state["inventory"]
        before = copies.copied_bytes
        images = {source: value[1] for source, value in copies.snapshots.items()}
        assert images and before > 0
        with (
            pytest.raises(
                storage.bootstrap.RecoveryRequired, match="nested_discovery_scope"
            ),
            session._discovery_reads(private_sqlite=True),
        ):
            pytest.fail("general nested discovery was admitted")
        state["phase"] = "project"
        result = project(partition, session, native_inventory)
        assert copies.copied_bytes == before
        assert images == {
            source: value[1] for source, value in copies.snapshots.items()
        }
        assert state["targets"]["project"] > 0
        state["projected"] += 1
        state["phase"] = "after-project"
        if fault == "project":
            raise DiscoveryFailure("after_actual_projection")
        return result

    def checked(scope, *args):
        # Observe the entire real workspace phase, including any extra proof.
        if (fault is None and not workspace_active) or (
            fault == "config" and workspace_active
        ):
            return member(scope, *args)
        assert storage._local.discovery_scope is None
        state = {
            "phase": "discover",
            "child": None,
            "copies": None,
            "targets": {"discover": 0, "project": 0},
            "verified": [],
            "discovered": 0,
            "projected": 0,
            "inventory": None,
        }
        active.append(state)
        try:
            return member(scope, *args)
        finally:
            active.pop()
            child, copies = state["child"], state["copies"]
            assert child is not None and copies is not None
            assert storage._local.discovery_scope is None
            assert not child.active and not copies.active
            assert child.resources == copies.resources == []
            assert not child.custody.fds and not copies.custody.fds
            assert not child.custody.close_failed and not copies.custody.close_failed
            assert copies.directory is None
            assert all(previous["child"] is not child for previous in states)
            if fault == "config":
                assert state["discovered"] == state["projected"] == 0
            elif fault == "discover":
                assert state["discovered"] == 1 and state["projected"] == 0
                assert state["verified"] == ["discover", "discover"]
            else:
                assert state["discovered"] == state["projected"] == 1
                assert state["verified"] == ["discover", "after-project"]
            states.append(state)

    monkeypatch.setattr(authority, "_member", checked)
    monkeypatch.setattr(_CapturedSQLite, "workspace_plan", workspace)
    monkeypatch.setattr(inventory_module, "discover", discovered)
    monkeypatch.setattr(
        storage.MaintenanceSession, "_discover_capture_inventory", captured
    )
    monkeypatch.setattr(_MobileCapturePartition, "project", projected)
    monkeypatch.setattr(storage._PreviewScope, "sqlite_target", target)
    monkeypatch.setattr(storage._PreviewScope, "verify_sources", verify)
    with (
        pytest.raises(ValueError, match="capture_config_scope_changed")
        if fault == "config"
        else pytest.raises(DiscoveryFailure)
        if fault
        else nullcontext()
    ):
        test_captured_shared_raw_owners_prove_exact_references_without_writes(
            tmp_path, monkeypatch, False
        )
    assert states
    if fault is None:
        assert len(states) == 2


@pytest.mark.parametrize("consumer", ["selected", "member"])
def test_authority_consumers_require_actual_issued_child_before_work(
    tmp_path, monkeypatch, consumer
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import mobile_captured_authority as authority
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _MobileCapturePartition,
    )

    class ChildRefused(RuntimeError):
        pass

    selected, member = authority._selected_source, authority._member
    discovery = storage.MaintenanceSession._discovery_reads
    inventory = storage.MaintenanceSession._discover_capture_inventory
    project = _MobileCapturePartition.project
    state = {"inside": False, "reached": False, "work": False}

    def snapshot(scope):
        child = storage._local.discovery_scope
        assert child is not None and child.active
        assert scope.discovery_bindings[child][0] is scope
        state.update(
            scope=scope,
            child=child,
            budget=scope.read_budget,
            parent=scope.read_budget._parent,
            remaining=scope.read_budget._remaining,
            parent_remaining=scope.read_budget._parent._remaining
            if scope.read_budget._parent is not None
            else None,
            before=_fingerprints(tmp_path / "profile"),
        )
        state["reached"] = True

    def forbidden(*args, **kwargs):
        state["work"] = True
        raise AssertionError("missing authority child reached native work")

    def captured_inventory(session, *args, **kwargs):
        if state["inside"] and state["reached"]:
            return forbidden()
        return inventory(session, *args, **kwargs)

    def projected(partition, *args, **kwargs):
        if state["inside"] and state["reached"]:
            return forbidden()
        return project(partition, *args, **kwargs)

    @contextmanager
    def issued(session, *args, **kwargs):
        with discovery(session, *args, **kwargs):
            if state["inside"]:
                snapshot(state["scope"])
                with monkeypatch.context() as fault:
                    fault.setattr(storage._local, "discovery_scope", None)
                    yield
            else:
                yield

    def source(scope, *args):
        if consumer != "selected":
            return selected(scope, *args)
        snapshot(scope)
        with monkeypatch.context() as fault:
            fault.setattr(storage._local, "discovery_scope", None)
            fault.setattr(authority, "_checked_bytes", forbidden)
            with pytest.raises(ValueError, match="discovery_child_changed"):
                selected(scope, *args)
        raise ChildRefused("selected source refused missing child")

    def proof(scope, *args):
        state.update(scope=scope, inside=True)
        try:
            with pytest.raises(ValueError, match="discovery_child_changed"):
                member(scope, *args)
        finally:
            state["inside"] = False
        raise ChildRefused("member refused missing child")

    monkeypatch.setattr(authority, "_selected_source", source)
    if consumer == "member":
        monkeypatch.setattr(authority, "_member", proof)
        monkeypatch.setattr(storage.MaintenanceSession, "_discovery_reads", issued)
        monkeypatch.setattr(
            storage.MaintenanceSession,
            "_discover_capture_inventory",
            captured_inventory,
        )
        monkeypatch.setattr(_MobileCapturePartition, "project", projected)
    with pytest.raises(ChildRefused):
        test_captured_shared_raw_owners_prove_exact_references_without_writes(
            tmp_path, monkeypatch, False
        )
    assert state["reached"]
    assert not state["work"]
    scope, child = state["scope"], state["child"]
    assert scope.read_budget is state["budget"]
    assert scope.read_budget._parent is state["parent"]
    assert scope.read_budget._remaining == state["remaining"]
    assert (
        scope.read_budget._parent._remaining if state["parent"] is not None else None
    ) == state["parent_remaining"]
    assert _fingerprints(tmp_path / "profile") == state["before"]
    assert not scope.active and not child.active and not child.sqlite_copies.active
    assert not scope.custody.fds and not scope.custody.close_failed
    assert child.resources == child.sqlite_copies.resources == []
    assert not child.custody.fds and not child.sqlite_copies.custody.fds
    assert (
        not child.custody.close_failed and not child.sqlite_copies.custody.close_failed
    )
    assert child.sqlite_copies.directory is None
