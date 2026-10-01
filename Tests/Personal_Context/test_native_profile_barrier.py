"""Current-manifest denial on the real encrypted native repository."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest
from tldw_profile_core import canonical_bytes

from Tests.Personal_Context.native_barrier_helpers import v2_fixture


def test_valid_v2_manifest_never_qualifies():
    from tldw_chatbook.Personal_Context.native_codec import decode_native_profile
    from tldw_chatbook.Personal_Context.native_compatibility import (
        profile_compatibility,
    )

    decoded = decode_native_profile(
        "manifest", json.dumps(v2_fixture("01-manifest")["data"])
    )
    view = profile_compatibility(decoded, consumer_id="repository.read_compatibility")
    assert view.state == "v2_blocked"
    assert view.schema_version == 2
    assert view.evidence_retirement_epoch == decoded.value.evidence_retirement_epoch
    assert decoded.value.profile_id not in repr(view)
    with pytest.raises(FrozenInstanceError):
        view.state = "legacy_v1"


def test_v1_current_registry_success_and_unknown_route_denial(
    tmp_path, memory_protector
):
    from tldw_chatbook.Personal_Context.native_codec import decode_native_profile
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
        profile_compatibility,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    repo = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    decoded = decode_native_profile("manifest", canonical_bytes(manifest))
    view = profile_compatibility(decoded, consumer_id="repository.get_record")
    assert view.state == "legacy_v1"
    assert view.evidence_retirement_epoch is None
    assert view.manifest_version_id == manifest.current_version_id
    for incoming in ("unknown", True, [], "repository.get_record "):
        with pytest.raises(
            ProfileCompatibilityError,
            match="^personal_context_compatibility_unavailable$",
        ):
            profile_compatibility(decoded, consumer_id=incoming)
    with pytest.raises(ProfileCompatibilityError):
        profile_compatibility(
            replace(decoded, schema_version=2), consumer_id="repository.get_record"
        )


def test_compiled_registry_is_fixed_sorted_unique_and_validated():
    from tldw_chatbook.Personal_Context import native_compatibility as module

    consumers = module.NATIVE_PROFILE_CONSUMERS
    ids = tuple(row.consumer_id for row in consumers)
    assert ids == tuple(sorted(set(ids)))
    assert {row.owner_id for row in consumers} == {"repository", "service", "bootstrap"}
    assert len(module.NATIVE_PROFILE_CONSUMER_DIGEST) == 64
    assert not any(hasattr(module, name) for name in ("register", "add", "qualify"))
    module.validate_native_consumers()
    with pytest.raises(module.ProfileCompatibilityError):
        module._validate_consumers(consumers + consumers[:1])
    with pytest.raises(module.ProfileCompatibilityError):
        module._validate_consumers(tuple(reversed(consumers)))
    with pytest.raises(module.ProfileCompatibilityError):
        module._validate_consumers((replace(consumers[0], owner_id="unknown"),))
    with pytest.raises(module.ProfileCompatibilityError):
        module._validate_consumers(
            (replace(consumers[0], operations=("read", "read")),)
        )


@pytest.mark.parametrize("change", ["extra", "unknown", "reordered", "qualified"])
def test_unacknowledged_manifest_requirements_never_make_a_view(change):
    from tldw_chatbook.Personal_Context.native_codec import (
        NativeProfileDecodeError,
        decode_native_profile,
    )

    body = v2_fixture("01-manifest")["data"]
    if change == "extra":
        body["required_context_semantics"].append("unknown-semantic")
    elif change == "unknown":
        body["required_context_semantics"][0] = "unknown-semantic"
    elif change == "reordered":
        body["required_context_semantics"].reverse()
    else:
        body["qualified"] = True
    with pytest.raises(NativeProfileDecodeError):
        decode_native_profile("manifest", json.dumps(body))


@pytest.mark.parametrize("reopen", [False, True])
def test_v2_manifest_blocks_readable_v1_before_content_authentication(
    tmp_path, memory_protector, record_factory, monkeypatch, reopen
):
    from Tests.Personal_Context.native_barrier_helpers import (
        blocked_repository,
        sql_state,
    )
    from tldw_chatbook.Personal_Context.native_codec import decode_native_profile
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    repo, record = blocked_repository(tmp_path, memory_protector, record_factory)
    if reopen:
        repo.close()
        repo = PersonalContextRepository(repo.db_path, key_protector=memory_protector)
    before = sql_state(repo)
    raw_method = (
        "_decrypt_row_authenticated"
        if hasattr(repo, "_decrypt_row_authenticated")
        else "_decrypt_row"
    )
    original = getattr(repo, raw_method)
    authenticated = []

    def authenticate(row):
        authenticated.append(row["object_type"])
        return original(row)

    monkeypatch.setattr(repo, raw_method, authenticate)
    # Validate synthetic cryptography before relying on product-denial assertions.
    with repo._connect() as connection:
        row = connection.execute(
            "SELECT * FROM encrypted_objects WHERE object_type='manifest' AND version_id='blocked-manifest-v2'"
        ).fetchone()
        assert decode_native_profile("manifest", original(row)).schema_version == 2
    with pytest.raises(
        ProfileCompatibilityError, match="^personal_context_compatibility_unavailable$"
    ):
        repo.get_record(record.record_id)
    assert authenticated and set(authenticated) == {"manifest"}
    assert sql_state(repo) == before


_READ_CALLS = [
    ("get_manifest", (), {}),
    ("get_record", ("record-1",), {}),
    ("get_record", ("missing",), {}),
    ("get_record_derivation", ("missing",), {}),
    ("list_records", (), {}),
    ("get_scope", ("missing",), {}),
    ("list_scopes", (), {}),
    ("get_proposal", ("missing",), {}),
    ("list_proposals", (), {}),
    ("read_export_snapshot", (), {}),
    ("get_runtime_policy", ("missing",), {}),
    ("get_runtime_policy_version", ("missing",), {}),
    ("get_scope_binding", ("missing",), {}),
    ("get_scope_binding_version", ("missing",), {}),
    ("get_validated_scope_binding", ("missing",), {}),
    ("list_scope_bindings", (), {}),
    ("list_validated_scope_bindings", (), {}),
    ("is_scope_explicitly_unlinked", ("missing",), {}),
    ("get_undo", ("missing",), {"now": "2026-09-27T00:00:00Z"}),
    ("list_undo_ids", (), {"now": "2026-09-27T00:00:00Z"}),
    ("get_outbox_body", ("missing",), {}),
    ("list_pending_outbox", (), {}),
    ("list_dispatchable_outbox", (), {}),
    ("get_outbox_receipt", ("missing",), {}),
    ("get_outbox_quarantine_reason", ("missing",), {}),
    ("list_quarantine", (), {}),
    ("first_link_head_rows", (), {}),
    ("first_link_sync_heads", (), {}),
    ("first_link_reviewed_lineage", (), {}),
]


@pytest.mark.parametrize("method,args,kwargs", _READ_CALLS)
def test_current_barrier_covers_content_and_zero_row_metadata_reads(
    tmp_path, memory_protector, record_factory, method, args, kwargs
):
    from Tests.Personal_Context.native_barrier_helpers import (
        install_v2_manifest,
        sql_state,
    )
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    repo = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    repo.commit_record_version(
        record_factory(manifest.profile_id), expected_version_id=None
    )
    getattr(repo, method)(
        *args, **kwargs
    )  # Real V1 success, including legitimate empty results.
    install_v2_manifest(repo)
    before = sql_state(repo)
    with pytest.raises(ProfileCompatibilityError):
        getattr(repo, method)(*args, **kwargs)
    assert sql_state(repo) == before


def test_post_open_storage_marker_change_blocks_before_body_authentication(
    tmp_path, memory_protector, record_factory, monkeypatch
):
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )
    from tldw_chatbook.Personal_Context.repository import (
        SCHEMA_VERSION,
        PersonalContextRepository,
    )

    repo = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    record = record_factory(manifest.profile_id)
    repo.commit_record_version(record, expected_version_id=None)
    assert repo.get_record(record.record_id) == record
    with repo._transaction() as connection:
        connection.execute(
            "UPDATE personal_context_schema SET version=?", (SCHEMA_VERSION + 1,)
        )
    calls = []
    raw_method = (
        "_decrypt_row_authenticated"
        if hasattr(repo, "_decrypt_row_authenticated")
        else "_decrypt_row"
    )
    monkeypatch.setattr(repo, raw_method, lambda row: calls.append(row["object_type"]))
    with pytest.raises(ProfileCompatibilityError):
        repo.get_record(record.record_id)
    assert calls == []


@pytest.mark.parametrize("incoming_kind", ["manifest", "record", "scope", "outbox"])
def test_incoming_v2_canonical_objects_deny_atomically_under_healthy_v1(
    tmp_path, memory_protector, record_factory, incoming_kind
):
    from tldw_profile_core import ProfileManifest, ProfileScope, ScopeKind
    from tldw_profile_core.v2_contract import validate_v2_object

    from Tests.Personal_Context.native_barrier_helpers import sql_state
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    repo = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    initial = v2_fixture("01-manifest")["data"]
    manifest = ProfileManifest(
        profile_id="p1",
        revision=0,
        purge_generation=0,
        current_version_id="native-v1-manifest",
        created_at=initial["created_at"],
        updated_at=initial["updated_at"],
    )
    scope = ProfileScope(
        profile_id="p1",
        scope_id="s1",
        kind=ScopeKind.GLOBAL,
        version_id="native-v1-scope",
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
    )
    repo.create_profile_with_global_scope(manifest, scope)
    repo.commit_record_version(
        record_factory(manifest.profile_id), expected_version_id=None
    )
    fixed = v2_fixture(
        "01-manifest" if incoming_kind in ("manifest", "outbox") else "02-active-record"
    )["data"]
    if incoming_kind == "manifest":
        fixed["revision"] = 1
    value = validate_v2_object(fixed)
    before = sql_state(repo)
    with pytest.raises(ProfileCompatibilityError):
        if incoming_kind == "manifest":
            repo.commit_manifest_version(
                value, expected_version_id=manifest.current_version_id
            )
        elif incoming_kind == "record":
            repo.commit_record_version(value, expected_version_id=None)
        elif incoming_kind == "scope":
            from tldw_profile_core import ProfileScope, ScopeKind

            scope = ProfileScope(
                profile_id=manifest.profile_id,
                scope_id="s1",
                kind=ScopeKind.GLOBAL,
                version_id="s1-v2",
                created_at=manifest.created_at,
                updated_at=manifest.updated_at,
            )
            repo.commit_scope(scope.model_copy(update={"schema_version": 2}))
        else:
            repo.commit_outbox_body(
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=value.current_version_id,
                body={"version": 1, "manifest": fixed},
            )
    assert sql_state(repo) == before
    assert repo.get_manifest() == manifest


def test_incoming_v2_manifest_cannot_create_profile(tmp_path, memory_protector):
    from tldw_profile_core import ProfileScope, ScopeKind
    from tldw_profile_core.v2_contract import validate_v2_object

    from Tests.Personal_Context.native_barrier_helpers import sql_state
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    value = validate_v2_object(v2_fixture("01-manifest")["data"])
    scope = ProfileScope(
        scope_id="s1",
        profile_id=value.profile_id,
        kind=ScopeKind.GLOBAL,
        version_id="scope-v1",
        created_at=value.created_at,
        updated_at=value.updated_at,
    )
    repo = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    before = sql_state(repo)
    with pytest.raises(ProfileCompatibilityError):
        repo.create_profile_with_global_scope(value, scope)
    assert sql_state(repo) == before
    assert repo.get_manifest() is None


def test_stale_removed_handle_cannot_reinitialize_unknown_storage(
    tmp_path, memory_protector
):
    from tldw_profile_core import ProfileScope, ScopeKind

    from Tests.Personal_Context.native_barrier_helpers import sql_state
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )
    from tldw_chatbook.Personal_Context.repository import (
        SCHEMA_VERSION,
        PersonalContextRepository,
    )

    repo = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    scope = ProfileScope(
        profile_id=manifest.profile_id,
        scope_id="global",
        kind=ScopeKind.GLOBAL,
        version_id="global-v1",
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
    )
    repo.destroy_profile_content()
    assert memory_protector.is_empty
    repo.reinitialize_destroyed_profile(manifest, scope)
    assert repo.get_manifest() == manifest
    repo.destroy_profile_content()
    with repo._transaction() as connection:
        connection.execute(
            "UPDATE personal_context_schema SET version=?", (SCHEMA_VERSION + 1,)
        )
    before = sql_state(repo)
    with pytest.raises(ProfileCompatibilityError):
        repo.reinitialize_destroyed_profile(manifest, scope)
    assert sql_state(repo) == before
    assert memory_protector.is_empty


def test_compiled_inventory_covers_every_protected_native_call_site():
    import ast
    from pathlib import Path

    from tldw_chatbook.Personal_Context.native_compatibility import (
        NATIVE_PROFILE_CONSUMERS,
    )

    ids = {row.consumer_id for row in NATIVE_PROFILE_CONSUMERS}
    source = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/Personal_Context/repository.py"
    ).read_text()
    owner = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "PersonalContextRepository"
    )
    protected = {
        "_decrypt_row",
        "_mutation",
        "_head_row",
        "_iter_head_rows",
        "_get_local_body",
        "_get_local_version",
        "_commit_local_body",
        "_read_connection",
        "_require_legacy_on_connection",
        "_compatibility_on_connection",
        "_manifest_on_connection",
        "_require_supported_storage",
        "_list_canonical_heads",
    }
    raw_callers = []
    routes = set()
    for method in owner.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for call in ast.walk(method):
            if not isinstance(call, ast.Call) or not isinstance(
                call.func, ast.Attribute
            ):
                continue
            if (
                not isinstance(call.func.value, ast.Name)
                or call.func.value.id != "self"
            ):
                continue
            if call.func.attr == "_decrypt_row_authenticated":
                raw_callers.append(method.name)
            if call.func.attr not in protected:
                continue
            consumer = next(
                (kw.value for kw in call.keywords if kw.arg == "consumer_id"), None
            )
            assert consumer is not None, (method.name, call.func.attr)
            if isinstance(consumer, ast.Constant):
                assert consumer.value in ids
                route_owner = {
                    "_read_export_snapshot": "read_export_snapshot",
                }.get(method.name, method.name)
                assert consumer.value == "repository." + route_owner
                routes.add(consumer.value)
            else:
                assert isinstance(consumer, ast.Name) and consumer.id == "consumer_id"
                assert any(a.arg == "consumer_id" for a in method.args.kwonlyargs)
    assert sorted(raw_callers) == ["_decrypt_row", "_manifest_on_connection"]
    assert len(routes) >= 45


@pytest.mark.parametrize("reader", ["get_record", "list_records"])
@pytest.mark.parametrize(
    "damage",
    [
        "polarity",
        "expiry",
        "extra",
        "schema",
        "kind",
        "privacy",
        "nested_schema",
        "mixed",
    ],
)
def test_authenticated_v1_damage_is_quarantined_only_when_unambiguously_known(
    tmp_path, memory_protector, record_factory, reader, damage
):
    from Tests.Personal_Context.native_barrier_helpers import (
        replace_sealed_body,
        sql_state,
    )
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    repository = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    manifest = repository.create_provisional_profile()
    record = record_factory(manifest.profile_id, value="quarantine-canary")
    repository.commit_record_version(record, expected_version_id=None)
    control = record_factory(
        manifest.profile_id, record_id="healthy-control", value="healthy"
    )
    repository.commit_record_version(control, expected_version_id=None)
    body = json.loads(canonical_bytes(record))
    if damage in {"polarity", "mixed"}:
        body["payload"]["polarity"] = "INVALID"
    if damage in {"extra", "mixed"}:
        body["future_restriction"] = "private-future-canary"
    if damage == "expiry":
        body["expires_at"] = "2026-08-01T00:00:00.000Z"
    if damage == "schema":
        body["schema_version"] = 3
    if damage == "kind":
        body["kind"] = "future-kind"
    if damage == "privacy":
        body["controls"]["agent_visibility"] = "future-private"
    if damage == "nested_schema":
        body["payload"]["schema_version"] = 2
    replace_sealed_body(
        repository,
        object_type="record",
        object_id=record.record_id,
        version_id=record.version_id,
        raw=json.dumps(body).encode(),
    )
    before = sql_state(repository)

    def read():
        return (
            repository.get_record(record.record_id)
            if reader == "get_record"
            else repository.list_records()
        )

    if damage in {"polarity", "expiry"}:
        result = read()
        assert result is None if reader == "get_record" else result == [control]
        assert repository.get_record(control.record_id) == control
        quarantine = repository.list_quarantine()
        assert len(quarantine) == 1 and quarantine[0].reason_code == "integrity_failure"
    else:
        with pytest.raises(
            ProfileCompatibilityError,
            match="^personal_context_compatibility_unavailable$",
        ):
            read()
        assert sql_state(repository) == before
        assert repository.list_quarantine() == []


@pytest.mark.parametrize("reader", ["getter", "list"])
@pytest.mark.parametrize("location", ["record", "proposal", "proposed_record"])
@pytest.mark.parametrize(
    ("damage", "unsupported"),
    [
        ("hash_pattern", None),
        ("reference_limit", None),
        ("hash_limit", None),
        ("hash_pattern", "extra"),
        ("hash_pattern", "schema"),
        ("hash_pattern", "privacy"),
    ],
)
def test_known_v1_provenance_damage_preserves_healthy_reads_without_omitting_future_data(
    tmp_path,
    memory_protector,
    record_factory,
    proposal_factory,
    reader,
    location,
    damage,
    unsupported,
):
    from Tests.Personal_Context.native_barrier_helpers import (
        replace_sealed_body,
        sql_state,
    )
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    repository = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    manifest = repository.create_provisional_profile()
    kind = "record" if location == "record" else "proposal"
    if kind == "record":
        damaged = record_factory(manifest.profile_id)
        control = record_factory(
            manifest.profile_id,
            record_id="healthy-control",
            version_id="healthy-version",
            value="healthy",
        )
        repository.commit_record_version(damaged, expected_version_id=None)
        repository.commit_record_version(control, expected_version_id=None)
        object_id = damaged.record_id
    else:
        damaged = proposal_factory(manifest.profile_id)
        control = proposal_factory(manifest.profile_id, proposal_id="healthy-control")
        repository.commit_proposal(damaged)
        repository.commit_proposal(control)
        object_id = damaged.proposal_id
    body = json.loads(canonical_bytes(damaged))
    record_body = body if kind == "record" else body["proposed_record"]
    provenance = (
        record_body["provenance"]
        if location == "proposed_record"
        else body["provenance"]
    )
    if damage == "hash_pattern":
        provenance["source_hashes"] = ["g" * 64]
    elif damage == "reference_limit":
        provenance["source_references"] = [f"source-{index}" for index in range(33)]
    else:
        provenance["source_hashes"] = ["a" * 64] * 33
    if unsupported == "extra":
        provenance["future_restriction"] = "future-private"
    elif unsupported == "schema":
        record_body["payload"]["schema_version"] = 2
    elif unsupported == "privacy":
        record_body["controls"]["agent_visibility"] = "future-private"
    with repository._connect() as connection:
        version_id = connection.execute(
            "SELECT version_id FROM object_heads WHERE object_type=? AND object_id=?",
            (kind, object_id),
        ).fetchone()[0]
    replace_sealed_body(
        repository,
        object_type=kind,
        object_id=object_id,
        version_id=version_id,
        raw=json.dumps(body).encode(),
    )
    before = sql_state(repository)
    get = repository.get_record if kind == "record" else repository.get_proposal
    listing = repository.list_records if kind == "record" else repository.list_proposals

    if unsupported is not None:
        with pytest.raises(
            ProfileCompatibilityError,
            match="^personal_context_compatibility_unavailable$",
        ):
            get(object_id) if reader == "getter" else listing()
        assert sql_state(repository) == before
        assert repository.list_quarantine() == []
        return

    result = get(object_id) if reader == "getter" else listing()
    assert result is None if reader == "getter" else result == [control]
    assert get("healthy-control") == control
    quarantine = repository.list_quarantine()
    assert len(quarantine) == 1
    assert quarantine[0].object_type == kind
    assert quarantine[0].object_id == object_id
    assert quarantine[0].version_id == version_id
    assert quarantine[0].reason_code == "integrity_failure"


@pytest.mark.parametrize("kind", ["record", "scope", "proposal"])
@pytest.mark.parametrize("state", ["healthy", "missing", "quarantined"])
def test_canonical_getters_use_one_private_admission_per_snapshot(
    tmp_path,
    memory_protector,
    record_factory,
    proposal_factory,
    monkeypatch,
    kind,
    state,
):
    from tldw_profile_core import ProfileScope, ScopeKind

    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    repo = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    try:
        manifest = repo.create_provisional_profile()
        record = record_factory(manifest.profile_id)
        scope = ProfileScope(
            profile_id=manifest.profile_id,
            scope_id=record.scope_id,
            kind=ScopeKind.GLOBAL,
            version_id="scope-version",
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        proposal = proposal_factory(manifest.profile_id)
        repo.commit_record_version(record, expected_version_id=None)
        repo.commit_scope(scope)
        repo.commit_proposal(proposal)
        value, object_id, version_id = {
            "record": (record, record.record_id, record.version_id),
            "scope": (scope, scope.scope_id, scope.version_id),
            "proposal": (proposal, proposal.proposal_id, "proposal-version"),
        }[kind]
        # Proposal version IDs live in the envelope, rather than the V1 model.
        if state == "quarantined":
            with repo._read_connection(
                consumer_id="repository.get_proposal"
            ) as connection:
                version_id = connection.execute(
                    "SELECT version_id FROM object_heads WHERE object_type=? AND object_id=?",
                    (kind, object_id),
                ).fetchone()[0]
            repo.quarantine_object(kind, object_id, version_id, "integrity_failure")
        if state == "missing":
            object_id = "missing"
        admissions = []
        connect = repo._connect

        def counted_connect():
            connection = connect()
            admissions.append(connection)
            return connection

        monkeypatch.setattr(repo, "_connect", counted_connect)
        getter = getattr(repo, f"get_{kind}")
        for _ in range(2):
            admissions.clear()
            actual = getter(object_id)
            assert actual == (value if state == "healthy" else None)
            assert len(admissions) == 1, (
                "A getter must use one fresh guarded private snapshot"
            )
    finally:
        repo.close()


def test_reused_operation_rechecks_manifest_after_external_commit(
    tmp_path, memory_protector, record_factory
):
    from Tests.Personal_Context.native_barrier_helpers import install_v2_manifest
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    repo = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    record = record_factory(manifest.profile_id)
    repo.commit_record_version(record, expected_version_id=None)
    with repo.operation():
        assert repo.get_record(record.record_id) == record
        install_v2_manifest(repo)
        with pytest.raises(ProfileCompatibilityError):
            repo.get_record(record.record_id)
    with pytest.raises(ProfileCompatibilityError):
        repo.get_record(record.record_id)
