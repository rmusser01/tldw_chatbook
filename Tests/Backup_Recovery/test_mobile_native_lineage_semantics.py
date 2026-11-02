"""Actual native rows retain chronology, revocation history and current data."""

import hashlib
import json
import re
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_native_operation import captured_operation
from Tests.Backup_Recovery.test_mobile_owning_transactions import (
    _ordinary_imported_owner,
    _small_native_seed,
)

_PASSWORD = b"test-private-mobile-rollback"
_MAX_REVISION = 2**63 - 1


def _portrait(db, *, regions_only=False):
    from Tests.Character_Chat.test_portrait_performance import png
    from tldw_chatbook.Character_Chat.portrait_compositing import (
        PixelRect,
        PortraitCompositionRecipe,
        prepared_base_pixel_sha256,
    )
    from tldw_chatbook.Character_Chat.portrait_performance import (
        FRAME_ROLES,
        capture_performance_source,
        normalize_performance_frames,
    )

    raw = png(size=(16, 16))
    identity = db.add_character_card({"name": "Retained native portrait", "image": raw})
    source, _ = capture_performance_source(
        db, character_id=identity, requested_state="idle"
    )
    recipe = PortraitCompositionRecipe(
        1,
        16,
        16,
        prepared_base_pixel_sha256(raw),
        PixelRect(0, 0, 4, 4),
        PixelRect(10, 0, 4, 4),
        PixelRect(4, 9, 7, 6),
        True,
    )
    revision = db.save_character_portrait_regions(source, recipe, expected_revision=0)
    if not regions_only:
        frames = normalize_performance_frames({role: raw for role in FRAME_ROLES})
        db.apply_character_portrait_performance(
            source,
            frames,
            expected_revision=0,
            composition_recipe=recipe,
            expected_regions_revision=revision,
        )
    return identity


def _review_history(db, conversation, *, aborted=False):
    """Seed retained SQL history, not a reusable current review authority grant."""
    from Tests.DB.test_automatic_wake_attempts import claim, survivor
    from Tests.DB.test_automatic_work_budget import chain

    chain_id = chain(
        db, conversation=conversation, submission="retained-review-history"
    )
    child = survivor(db, chain_id, conversation=conversation)
    attempt = claim(
        db, chain_id, (child,), attempt="review-attempt", owner="review-owner"
    )
    if aborted:
        assert db.automatic_work.abort_wake(attempt.id, owner_id="review-owner")
    with db.transaction() as connection:
        connection.execute(
            "UPDATE automatic_wake_attempts SET review_boundary_id=? WHERE id=?",
            ("retained-review-boundary", attempt.id),
        )
        if not aborted:
            connection.execute(
                "INSERT INTO automatic_review_boundary_claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "retained-review-boundary",
                    attempt.id,
                    "review-owner",
                    chain_id,
                    "review-id",
                    "review-slot",
                    "slot-attempt",
                    1,
                    1,
                    "retained-grant",
                    "a" * 64,
                    "b" * 64,
                    "c" * 64,
                    json.dumps([child]),
                    json.dumps(["retained-settlement"]),
                ),
            )
    return attempt.id


def _rows(owner, table):
    # Literal test-only table callers below; no production SQL surface is exposed.
    with owner.transaction() as connection:
        return tuple(
            tuple(row)
            for row in connection.execute('SELECT * FROM "' + table + '" ORDER BY 1')
        )


def _native_state(chat, workspace, runs, registry, result, legacy):
    from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

    conversation = result["conversation"]
    old = chat.get_conversation_by_id(conversation)
    archived = chat.set_conversations_archived(
        (conversation,),
        archived=True,
        expected_versions={conversation: old["version"]},
    )
    assert archived["changed"] and not archived["failures"]
    registry.archive_workspace("same-workspace", expected_archive_revision=0)
    root = runs.db_path.parent.parent / "tool_sandbox"
    root.mkdir(mode=0o700, exist_ok=True)

    def binding(identity):
        return WorkspaceRuntimeBinding(
            binding_id=identity,
            workspace_id="same-workspace",
            binding_kind="local-filesystem",
            label="Retained binding",
            locator=str(root),
            status="ready",
            metadata={"allow_write": True},
        )

    for identity in ("binding-live", "binding-retired"):
        registry.save_runtime_binding(binding(identity))
        if legacy and identity == "binding-live":
            registry.save_runtime_binding(binding(identity))
        else:
            registry.remove_runtime_binding(identity)
    if legacy:
        with workspace.transaction() as connection:
            connection.execute(
                "UPDATE workspace_binding_authority SET revision=? WHERE binding_id=?",
                (_MAX_REVISION, "binding-retired"),
            )
    else:
        result["portrait"] = _portrait(chat)
        result["review_attempt"] = _review_history(runs, conversation)
        result["canonical_rows"] = {
            "chat": {
                table: _rows(chat, table)
                for table in (
                    "character_portrait_performances",
                    "character_portrait_performance_frames",
                    "character_portrait_regions",
                )
            },
            "workspaces": {
                table: _rows(workspace, table)
                for table in (
                    "workspace_binding_authority",
                    "workspace_records",
                )
            },
            "runs": {
                table: _rows(runs, table)
                for table in (
                    "automatic_review_boundary_claims",
                    "automatic_wake_attempts",
                    "automatic_work_runtime_owner",
                )
            },
        }
    with chat.transaction() as connection:
        result["archive"] = tuple(
            connection.execute(
                "SELECT archived,archived_at,version,last_modified FROM conversations WHERE id=?",
                (conversation,),
            ).fetchone()
        )
        result["sync_frontier"] = tuple(
            connection.execute(
                "SELECT version,timestamp FROM sync_log WHERE entity='conversations' AND entity_id=? ORDER BY version DESC LIMIT 1",
                (conversation,),
            ).fetchone()
        )
    assert result["sync_frontier"][0] < result["archive"][2]
    result["workspace_archive"] = registry.get_workspace("same-workspace")
    result["witnesses"] = _rows(workspace, "workspace_binding_authority")


def _install_native_seed(monkeypatch):
    from Tests.Backup_Recovery import mobile_consolidation_fixture

    seeded = {}

    def seed(stores, **kwargs):
        result = _small_native_seed(
            stores,
            **kwargs,
            native_seed=_native_state,
            legacy_chat_version=85,
            legacy_workspace_version=10,
        )
        seeded["legacy" if kwargs["legacy"] else "canonical"] = result
        return result

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    return seeded


def _assert_canonical_rows(stores, expected):
    for owner_id, family in (
        ("db.chachanotes.primary", "chat"),
        ("db.workspaces", "workspaces"),
        ("db.agent_runs", "runs"),
    ):
        with _ordinary_imported_owner(stores, owner_id) as owner:
            for table, rows in expected[family].items():
                actual = _rows(owner, table)
                for row in rows:
                    assert row in actual, (family, table, row[0])


def _native85_archive_witnesses_canonical_history_and_public_rollback(
    tmp_path, monkeypatch, *, after_import=None
):
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        require_mobile_cutover,
    )
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _mobile_budget_pair,
        _TraceReadBudget,
    )
    from tldw_chatbook.DB.mobile_consolidation_receipts import _read_owner_receipt
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    seeded = _install_native_seed(monkeypatch)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        _, journal, _, stores, preview, initial_inventory, _ = values
    empty_roots = {
        item.path: (
            item.path.stat().st_dev,
            item.path.stat().st_ino,
            item.path.stat().st_mode,
        )
        for item in initial_inventory.items
        if item.owner == "agents.history"
        and item.status == "included_directory"
        and item.metadata is not None
        and item.metadata.parent_id is None
        and not tuple(item.path.iterdir())
    }
    assert empty_roots
    control_root = journal.root.parent
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(preview.target_selector))
    source_before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in preview.source_dir.iterdir()
        if p.is_file()
    }
    issued = []
    native_init = _TraceReadBudget.__init__

    def count(budget):
        native_init(budget)
        issued.append(budget)

    monkeypatch.setattr(_TraceReadBudget, "__init__", count)
    report = {}
    service = RecoveryService(control_root)
    try:
        preview = service.preview_mobile_consolidation(
            preview.source_selector, preview.target_selector
        )
        operation_id = service.start_mobile_consolidation(
            preview, destination_confirmed=True, rollback_password=_PASSWORD
        )
        result = service.wait(operation_id)
        report["import_state"] = result["state"]
        report["pairs"] = [
            {
                "coupled": _mobile_budget_pair(b._parent, b),
                "native_remaining": b._remaining,
                "aggregate_remaining": b._parent._remaining,
            }
            for b in issued
            if b._parent is not None
        ]
        assert result["state"] == "succeeded", report
        assert not service.pending_operations()
        assert report["pairs"] and all(
            p["coupled"]
            and 0 < p["native_remaining"] < 16 * 1024**2
            and 0 < p["aggregate_remaining"] < 4 * 1024**3
            for p in report["pairs"]
        )
    finally:
        service.close()
        (tmp_path / "native85-public-import.json").write_text(
            json.dumps(report, indent=2)
        )
    assert require_mobile_cutover(preview.source_dir, stores) is None
    source = seeded["legacy"]
    canonical = seeded["canonical"]
    _assert_canonical_rows(stores, canonical["canonical_rows"])
    mappings = {}
    for owner_id, family in (
        ("db.chachanotes.primary", "chat"),
        ("db.workspaces", "workspaces"),
    ):
        with (
            _ordinary_imported_owner(stores, owner_id) as owner,
            owner.transaction() as connection,
        ):
            _, children = _read_owner_receipt(
                connection.connection if family == "chat" else connection,
                operation_id,
                preview.source_fingerprint,
                family,
                _TraceReadBudget(),
            )
            mappings.update(
                {
                    (r["data"]["domain"], r["data"]["original"]): r["data"]["mapped"]
                    for r in children
                    if r["kind"] == "identity"
                    and type(r["data"]["original"]) in (str, int)
                }
            )
    with (
        _ordinary_imported_owner(stores, "db.chachanotes.primary") as chat,
        chat.transaction() as connection,
    ):
        mapped = mappings["conversation", source["conversation"]]
        assert (
            tuple(
                connection.execute(
                    "SELECT archived,archived_at,version,last_modified FROM conversations WHERE id=?",
                    (mapped,),
                ).fetchone()
            )
            == source["archive"]
        )
        assert (
            tuple(
                connection.execute(
                    "SELECT version,timestamp FROM sync_log WHERE entity='conversations' AND entity_id=? ORDER BY version DESC LIMIT 1",
                    (mapped,),
                ).fetchone()
            )
            == source["sync_frontier"]
        )
        assert (
            tuple(
                connection.execute(
                    "SELECT archived,archived_at,version,last_modified FROM conversations WHERE id=?",
                    (canonical["conversation"],),
                ).fetchone()
            )
            == canonical["archive"]
        )
    with _ordinary_imported_owner(stores, "db.workspaces") as workspace:
        registry = LocalWorkspaceRegistryService(workspace)
        mapped_workspace = mappings["workspace", "same-workspace"]
        imported_workspace = registry.get_workspace(mapped_workspace)
        assert imported_workspace.archived_at == source["workspace_archive"].archived_at
        assert (
            imported_workspace.archive_revision
            == source["workspace_archive"].archive_revision
        )
        for identity, revision in source["witnesses"]:
            mapped = mappings["binding", identity]
            assert mapped != identity  # canonical tombstones reserve both names
            assert registry.binding_authority_revision(identity) == 2
            assert registry.binding_authority_revision(mapped) == revision
            assert not workspace.imported_binding_allowed(mapped)
            with registry.binding_authority_fence(mapped) as authority:
                assert authority is None
            with registry.project_index_detached_fence(
                mapped_workspace, mapped, 0
            ) as allowed:
                assert allowed is False
        assert (
            registry.get_runtime_binding(mappings["binding", "binding-retired"]) is None
        )
    if after_import is not None:
        after_import(stores, canonical["conversation"])
    # Public rollback restores the populated canonical partition, including its
    # revocation histories, instead of erasing or adopting source tombstones.
    selected = tuple(dict.fromkeys((preview.source_selector, preview.target_selector)))
    current = preview_capture(selected, options={})
    assert current.complete, current
    service = RecoveryService(control_root)
    try:
        plan = service.preview_rollback(
            operation_id, old_password=_PASSWORD, target=current
        )
        for root in empty_roots:
            matches = [
                item
                for item in current.items
                if item.owner == "agents.history"
                and item.path == root
                and item.status == "included_directory"
            ]
            assert matches
            for item in matches:
                assert item.logical_id in plan.safety_scope
                assert (item.logical_id, root) in plan.preserve
                assert item.logical_id not in dict(plan.restore)
        operation = service.start_rollback(
            operation_id,
            plan,
            old_password=_PASSWORD,
            new_password=b"native-history-safety-password",
        )
        result = service.wait(operation)
        assert result["state"] == "succeeded", dict(result)
        assert service.pending_operations() == ()
    finally:
        service.close()
    _assert_canonical_rows(stores, canonical["canonical_rows"])
    with _ordinary_imported_owner(stores, "db.workspaces") as workspace:
        assert (
            _rows(workspace, "workspace_binding_authority")
            == canonical["canonical_rows"]["workspaces"]["workspace_binding_authority"]
        )
        assert (
            _rows(workspace, "workspace_records")
            == canonical["canonical_rows"]["workspaces"]["workspace_records"]
        )
    assert source_before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in preview.source_dir.iterdir()
        if p.is_file()
    }

    for root, identity in empty_roots.items():
        info = root.stat()
        assert (info.st_dev, info.st_ino, info.st_mode) == identity
        assert not tuple(root.iterdir())

    protected_roots = tuple(
        dict.fromkeys(
            (
                preview.source_dir,
                preview.source_selector.parent,
                preview.target_selector.parent,
                *(
                    item.path
                    for item in initial_inventory.items
                    if item.path is not None
                    and item.metadata is not None
                    and item.metadata.root_id == item.logical_id
                    and item.metadata.parent_id is None
                    and item.metadata.kind == "directory"
                ),
            )
        )
    )
    return (
        control_root,
        stores,
        result["result"]["journal_operation_id"],
        protected_roots,
    )


def test_native85_archive_witnesses_canonical_history_and_public_rollback(
    tmp_path, monkeypatch
):
    _native85_archive_witnesses_canonical_history_and_public_rollback(
        tmp_path, monkeypatch
    )


def test_public_rollback_retains_post_import_conversation_edit_in_safety_copy(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import archive_reader, later_rollback
    from tldw_chatbook.Backup_Recovery.inert_extraction import (
        extract_inert,
        preview_inert_extraction,
    )
    from tldw_chatbook.Backup_Recovery.recovery_copies import _locked_copy
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _check, _installed_owner
    from tldw_chatbook.DB.private_sqlite import open_recovery_validation

    edited = {}

    def stored_row(connection, identity):
        # Ordinary Core parses its two DATETIME declarations; recovery validation
        # intentionally does not. Compare their exact stored text, not adapters.
        cursor = connection.execute(
            "SELECT *, CAST(created_at AS TEXT), CAST(last_modified AS TEXT) "
            "FROM conversations WHERE id=?",
            (identity,),
        )
        values = list(cursor.fetchone())
        columns = tuple(column[0] for column in cursor.description[:-2])
        for name, raw in zip(("created_at", "last_modified"), values[-2:], strict=True):
            values[columns.index(name)] = raw
        return tuple(values[:-2])

    def edit_existing(stores, identity):
        with _ordinary_imported_owner(stores, "db.chachanotes.primary") as owner:
            with owner.transaction() as connection:
                original = stored_row(connection, identity)
            version = owner.get_conversation_by_id(identity)["version"]
            assert (
                owner.update_conversation(
                    identity,
                    {"title": "Actual post-import edit retained only in SafetyCopy"},
                    expected_version=version,
                )
                is True
            )
            assert owner.get_conversation_by_id(identity)["version"] == version + 1
            with owner.transaction() as connection:
                changed = stored_row(connection, identity)
            assert changed != original
            edited.update(identity=identity, original=original, changed=changed)

    control_root, stores, operation, profile_roots = (
        _native85_archive_witnesses_canonical_history_and_public_rollback(
            tmp_path, monkeypatch, after_import=edit_existing
        )
    )
    with (
        _ordinary_imported_owner(stores, "db.chachanotes.primary") as owner,
        owner.transaction() as connection,
    ):
        assert stored_row(connection, edited["identity"]) == edited["original"]

    work = tmp_path / "retained-safety-readback"
    work.mkdir(mode=0o700)
    cancel = Event()
    with _locked_copy(control_root, operation, exclusive=False) as (entry, _, proof, _):
        assert entry.status == "verified" and not entry.pending_operation
        archive = later_rollback._acquire(
            entry, proof, work, b"native-history-safety-password", cancel
        )
        document = archive_reader.verify_sealed(archive, cancel)
        payloads = [
            row for row in document.files if row.owner_id == "db.chachanotes.primary"
        ]
        assert len(payloads) == 1
        payload = payloads[0]
        # Native coverage names a physical representative; co-resident Core
        # aliases carry the same authenticated shared group and payload bytes.
        producers = {row.logical_id: row for row in document.producer_inventory}
        group = producers[payload.logical_id].shared_group
        covered = [
            row
            for row in document.files
            if row.logical_id in proof.coverage.values()
            and (
                row.logical_id == payload.logical_id
                or group is not None
                and producers[row.logical_id].shared_group == group
            )
        ]
        assert covered
        assert all(
            (row.size, row.sha256) == (payload.size, payload.sha256) for row in covered
        )
        plan = preview_inert_extraction(
            archive,
            group_ids=tuple(
                group.group_id
                for group in document.dependency_groups
                if payload.logical_id in group.members
            ),
            destination=tmp_path / "retained-safety-extracted",
            limits=archive_reader.ArchiveLimits(),
            cancel=cancel,
            protected_roots=(control_root, work, *stores.paths, *profile_roots),
        )
        extracted = extract_inert(archive, plan, cancel=cancel)
        mapping = json.loads(extracted.report_path.read_text())
        rows = [
            row for row in mapping["files"] if row["logical_id"] == payload.logical_id
        ]
        assert len(rows) == 1
        candidate = plan.destination / rows[0]["output"]
        assert candidate.stat().st_size == payload.size
        assert hashlib.sha256(candidate.read_bytes()).hexdigest() == payload.sha256
        with open_recovery_validation(
            "db.chachanotes.primary",
            candidate,
            writable=False,
            with_restrictions=True,
            cancel=cancel,
        ) as (connection, restrictions):
            owner = _installed_owner("db.chachanotes.primary")
            issues, _ = _check(connection, owner, owner.schema_policy(), restrictions)
            assert issues == ()
            assert stored_row(connection, edited["identity"]) == edited["changed"]
        assert entry.status == "verified"


@pytest.mark.parametrize(
    "history", ("portrait", "regions", "review_claim", "aborted_review_reference")
)
def test_populated_native_source_history_refuses_before_owner_effects(
    tmp_path, monkeypatch, history
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_admission import (
        prepare,
        prepared,
    )
    from tldw_chatbook.Backup_Recovery import mobile_consolidation

    snapshots = {}
    statements = []
    original_open = mobile_consolidation._open_preview_read

    def observed_open(path):
        connection = original_open(path)
        connection.set_trace_callback(statements.append)
        return connection

    def native(chat, workspace, runs, registry, result, legacy):
        if legacy:
            if history in {"portrait", "regions"}:
                _portrait(chat, regions_only=history == "regions")
            else:
                _review_history(
                    runs,
                    result["conversation"],
                    aborted=history == "aborted_review_reference",
                )

    def seed(stores, **kwargs):
        result = _small_native_seed(
            stores,
            **kwargs,
            native_seed=native,
            legacy_chat_version=85,
            legacy_workspace_version=10,
        )
        snapshots.update(
            {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in stores.paths}
        )
        return result

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    monkeypatch.setattr(mobile_consolidation, "_open_preview_read", observed_open)
    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _stores,
        preview,
        inventory,
        _,
    ):
        reason = (
            "mobile_source_portrait_history_unsupported"
            if history in {"portrait", "regions"}
            else "mobile_source_review_boundary_history_unsupported"
        )
        assert any(issue.startswith(reason) for issue in preview.issues), preview.issues
        with pytest.raises(ValueError, match="mobile"):
            prepare(session, journal, config, preview, inventory)
        assert snapshots == {
            p: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshots
        }
        with journal._locked(exclusive=False) as directory:
            assert not journal._records(directory)
    protected = (
        "character_portrait_"
        if history in {"portrait", "regions"}
        else "(?:automatic_review_boundary_claims|automatic_wake_attempts)"
    )
    assert not any(
        re.match(r'SELECT \* FROM ["`]?' + protected, sql, re.IGNORECASE)
        for sql in statements
    )


@pytest.mark.parametrize("purpose", ("legacy", "canonical"))
def test_missing_live_binding_witness_refuses_before_native_owner_effects(
    tmp_path, monkeypatch, purpose
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_captured_sqlite import _fingerprints

    def native(chat, workspace, runs, registry, result, legacy):
        _native_state(chat, workspace, runs, registry, result, legacy)
        if legacy == (purpose == "legacy"):
            from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

            registry.save_runtime_binding(
                WorkspaceRuntimeBinding(
                    binding_id="missing-witness",
                    workspace_id="same-workspace",
                    binding_kind="local-filesystem",
                    label="Retained binding",
                    locator=str(runs.db_path.parent.parent / "tool_sandbox"),
                    status="ready",
                    metadata={"allow_write": True},
                )
            )
            with workspace.transaction() as connection:
                connection.execute(
                    "DELETE FROM workspace_binding_authority WHERE binding_id='missing-witness'"
                )

    monkeypatch.setattr(
        mobile_consolidation_fixture,
        "_seed",
        lambda stores, **kwargs: _small_native_seed(
            stores,
            **kwargs,
            native_seed=native,
            legacy_chat_version=85,
            legacy_workspace_version=10,
        ),
    )
    with captured_operation(tmp_path, monkeypatch) as captured:
        before = _fingerprints(tmp_path / "profile")
        with captured.journal._locked(exclusive=False) as fd:
            events = tuple(captured.journal._records(fd))
        with pytest.raises(ValueError, match="^mobile_binding_witness_missing$"):
            captured.consolidate(password=_PASSWORD, cancel=Event())
        assert _fingerprints(tmp_path / "profile") == before
        with captured.journal._locked(exclusive=False) as fd:
            assert tuple(captured.journal._records(fd)) == events


def test_binding_union_cannot_renew_exhausted_native_allowance(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import mobile_identity_plan
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadLimit

    _install_native_seed(monkeypatch)
    original_rows = mobile_identity_plan._rows
    original_reserve = mobile_identity_plan._reserve
    observed = []

    def consume_after_native_read(reads, purpose, family, table, **kwargs):
        rows = original_rows(reads, purpose, family, table, **kwargs)
        if purpose == "legacy" and table == "workspace_binding_authority":
            budget = reads.read_budget
            assert rows and kwargs["_read_budget"] is budget
            observed.append(budget)
            budget.reserve(budget._remaining)
        return rows

    def no_binding_allocation(domain, *args, **kwargs):
        assert domain.name != "binding", "union allocation escaped exhausted admission"
        return original_reserve(domain, *args, **kwargs)

    with captured_operation(tmp_path, monkeypatch) as captured:
        with captured.sqlite_reads(password=_PASSWORD, cancel=Event()) as reads:
            monkeypatch.setattr(
                mobile_identity_plan, "_rows", consume_after_native_read
            )
            monkeypatch.setattr(mobile_identity_plan, "_reserve", no_binding_allocation)
            original_budget = reads.read_budget
            for _ in range(2):
                with pytest.raises(_TraceReadLimit, match="mobile_trace_input_limit"):
                    reads.identity_plan()
                assert reads.read_budget is original_budget
                assert original_budget._remaining == 0
            assert observed and all(b is original_budget for b in observed)
        assert not reads.active and not reads.resources


def test_workspace_receipt_first_resume_retains_exact_witnesses(tmp_path, monkeypatch):
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from Tests.Backup_Recovery.test_mobile_owning_transactions import (
        _native_terminal_interruption,
    )
    from tldw_chatbook.Backup_Recovery import mobile_native_rows
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
        _resume_captured_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_workspace_plan import _TABLES

    _install_native_seed(monkeypatch)
    destination = tmp_path / "archives"
    destination.mkdir(mode=0o700)
    insert = mobile_native_rows._insert_row
    reached = []

    def fail_runs(connection, row, values, detach=False):
        if row.table == "agent_runs":
            reached.append("runs")
            raise OSError("native Runs transaction interrupted")
        return insert(connection, row, values, detach=detach)

    witness_tables = (
        "workspace_records",
        "workspace_runtime_bindings",
        "workspace_binding_authority",
        "mobile_import_authority_holds",
        "mobile_consolidation_batches",
    )
    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        _,
        _,
        preview,
        inventory,
        authority,
    ):
        source_before = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in preview.source_dir.iterdir()
            if p.is_file()
        }
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=destination,
            password=_PASSWORD,
            cancel=Event(),
        ) as captured:
            with monkeypatch.context() as fault:
                fault.setattr(mobile_native_rows, "_insert_row", fail_runs)
                with pytest.raises(
                    OSError, match="native Runs transaction interrupted"
                ):
                    captured.consolidate(password=_PASSWORD, cancel=Event())
            assert reached == ["runs"]
            with captured.owner("db.workspaces") as workspace:
                before = {table: _rows(workspace, table) for table in witness_tables}
                assert len(before["workspace_binding_authority"]) == 4
                assert sorted(
                    row[1] for row in before["workspace_binding_authority"]
                ) == [2, 2, 2, _MAX_REVISION]
                assert len(before["mobile_import_authority_holds"]) == 2
                assert before["mobile_consolidation_batches"]

    resumed_rows = []

    def refuse_committed_replay(connection, row, values, detach=False):
        assert row.table not in {*_TABLES, "conversations"}, (
            "committed owner replayed before native receipt verification"
        )
        resumed_rows.append(row.table)
        return insert(connection, row, values, detach=detach)

    with monkeypatch.context() as fault:
        fault.setattr(mobile_native_rows, "_insert_row", refuse_committed_replay)
        with (
            authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as fresh,
            _resume_captured_mobile(
                fresh,
                journal,
                destination=destination,
                password=_PASSWORD,
                cancel=Event(),
            ) as captured,
        ):
            # Component proof deliberately stops before terminal publication;
            # the separate public test qualifies full cutover and rollback.
            with _native_terminal_interruption(monkeypatch):
                captured.consolidate(password=_PASSWORD, cancel=Event())
            assert "agent_runs" in resumed_rows
            with captured.owner("db.workspaces") as workspace:
                assert {
                    table: _rows(workspace, table) for table in witness_tables
                } == before
            for owner_id, family in (
                ("db.chachanotes.primary", "chat"),
                ("db.workspaces", "workspaces"),
                ("db.agent_runs", "runs"),
            ):
                with (
                    captured.owner(owner_id) as owner,
                    owner.transaction() as connection,
                ):
                    assert (
                        connection.execute(
                            "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id=? AND batch_key=?",
                            (journal.operation_id, family),
                        ).fetchone()[0]
                        == 1
                    )
    assert source_before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in preview.source_dir.iterdir()
        if p.is_file()
    }


def test_initial_captured_sqlite_custody_refusal_retires_empty_scope(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import mobile_captured_sqlite, storage_admission
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadLimit

    _install_native_seed(monkeypatch)
    original = storage_admission._CaptureFileDescriptors.__init__
    observed = []

    def exhaust_before_native_allocation(custody, scope):
        if type(scope) is mobile_captured_sqlite._CapturedSQLite:
            observed.append(scope)
            scope.read_budget.reserve(scope.read_budget._remaining)
        original(custody, scope)

    with captured_operation(tmp_path, monkeypatch) as captured:
        with monkeypatch.context() as fault:
            fault.setattr(
                storage_admission._CaptureFileDescriptors,
                "__init__",
                exhaust_before_native_allocation,
            )
            with (
                pytest.raises(_TraceReadLimit, match="mobile_trace_input_limit"),
                captured.sqlite_reads(password=_PASSWORD, cancel=Event()),
            ):
                pytest.fail("exhausted initial custody admitted an acquisition")
        assert len(observed) == 1
        scope = observed[0]
        assert scope.custody is None
        assert not scope.active and not scope.resources and not scope.roots
        assert scope.read_budget._remaining == 0
        scope.retire()
