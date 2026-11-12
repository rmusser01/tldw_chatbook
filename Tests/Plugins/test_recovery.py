"""Recovery requires protected proof and exact retained definitions."""

import shutil
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import pytest

from Tests.Plugins import platform_support
from Tests.Plugins.test_coordinator import reviewed


def interrupt_at(stack, phase):
    def interrupt(observed):
        if observed == phase:
            raise OSError("injected interruption")

    stack.call(lambda: setattr(stack.coordinator, "progress", interrupt))


def lose_registry(stack):
    from tldw_chatbook.Plugins.coordinator import PluginCoordinator
    from tldw_chatbook.Plugins.registry import PluginRegistry

    def replace():
        path = stack.registry.path
        stack.registry.close()
        for member in (
            path,
            path.with_name(path.name + "-wal"),
            path.with_name(path.name + "-shm"),
        ):
            member.unlink(missing_ok=True)
        stack.registry = PluginRegistry(path, owner=stack.owner)
        stack.coordinator = PluginCoordinator(
            stack.registry, stack.authority, stack.owner
        )

    stack.call(replace)


@pytest.mark.parametrize(
    "phase,want,committed",
    [
        ("prepared", "aborted", False),
        ("registry_committed", "recovery_required", False),
        ("certified", "complete", True),
        ("marker_advanced", "complete", True),
        ("published", "complete", True),
    ],
)
def test_evidence_matrix_at_actual_boundaries(
    plugin_stack, native_package, phase, want, committed
):
    stack = plugin_stack
    review = reviewed(stack, native_package())
    interrupt_at(stack, phase)
    with pytest.raises(OSError):
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    stack.call(lambda: setattr(stack.coordinator, "progress", None))
    receipt = next(
        item
        for item in stack.call(stack.coordinator.recover)
        if item.operation_id == review.operation_id
    )
    assert (receipt.phase, receipt.committed) == (want, committed)
    if committed:
        assert (
            len(stack.call(stack.coordinator.published_snapshot)["installations"]) == 1
        )
    elif want == "recovery_required":
        with pytest.raises(PermissionError):
            stack.call(stack.coordinator.published_snapshot)
    else:
        assert stack.call(stack.coordinator.published_snapshot)["installations"] == []


@pytest.mark.parametrize("phase", ["certified", "marker_advanced"])
def test_database_loss_restores_all_installations_and_blocks_unknown_processes(
    plugin_stack, native_package, phase
):
    stack = plugin_stack
    first = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(first, first.operation_id))
    second = reviewed(stack, native_package())
    interrupt_at(stack, phase)
    with pytest.raises(OSError):
        stack.call(lambda: stack.coordinator.commit(second, second.operation_id))
    lose_registry(stack)
    receipts = stack.call(stack.coordinator.recover)
    assert any(
        item.operation_id == second.operation_id and item.committed for item in receipts
    )
    state = stack.call(stack.authority.verify_current)
    assert {item["installation_id"] for item in state["installations"]} == {
        first.installation_id,
        second.installation_id,
    }
    assert (
        len(stack.call(lambda: stack.registry.list_installations(limit=50, offset=0)))
        == 2
    )
    assert state["mappings"] == []
    for review in (first, second):
        with pytest.raises(PermissionError, match="recovery"):
            stack.call(
                lambda review=review: stack.owner.reserve_launch(
                    "launch",
                    review.installation_id,
                    None,
                    review.inspection.effective_digest,
                )
            )
    rows = stack.call(lambda: stack.owner.list_processes(limit=50, offset=0))
    assert len(rows) == 2
    assert all(
        row["state"] == "unresolved" and "pid" not in row["provenance"] for row in rows
    )
    stack.call(stack.coordinator.recover)
    assert len(stack.call(lambda: stack.owner.list_processes(limit=50, offset=0))) == 2


def test_recovery_reinspects_retained_bytes_without_original_link_source(
    plugin_stack, native_package
):
    stack = plugin_stack
    package = native_package()
    original = package / "skills" / "review" / "SKILL.md"
    data = original.read_bytes()
    original.unlink()
    (package / "skill-source.md").write_bytes(data)
    original.symlink_to("../../skill-source.md")
    review = reviewed(stack, package)
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    shutil.rmtree(package)
    lose_registry(stack)
    receipt = stack.call(stack.coordinator.recover)[0]
    assert receipt.committed
    state = stack.call(stack.authority.verify_current)
    assert state["revisions"][0]["link_targets"] == {
        "skills/review/SKILL.md": "skill-source.md"
    }
    assert state == stack.call(
        lambda: stack.registry.authority_projection(
            operation_result=state["operation_result"]
        )
    )


def test_changed_retained_material_quarantines_without_default_projection(
    plugin_stack, native_package
):
    stack = plugin_stack
    review = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    state = stack.call(stack.authority.verify_current)
    from pathlib import Path

    root = Path(state["revisions"][0]["materialized_identity"])
    file = root / "skills" / "review" / "SKILL.md"
    file.chmod(0o600)
    file.write_text("tampered")
    lose_registry(stack)
    assert stack.call(stack.coordinator.recover)[0].phase == "recovery_required"
    assert (
        stack.call(lambda: stack.registry.list_installations(limit=50, offset=0)) == ()
    )
    with pytest.raises(PermissionError):
        stack.call(stack.coordinator.published_snapshot)


def test_invalid_journal_never_promotes_registry(plugin_stack, native_package):
    stack = plugin_stack
    review = reviewed(stack, native_package())
    interrupt_at(stack, "registry_committed")
    with pytest.raises(OSError):
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    path = next((stack.authority.store_dir / "intents").iterdir())
    path.write_text("{}")
    assert stack.call(stack.coordinator.recover)[0].phase == "recovery_required"
    assert stack.call(stack.authority.load_marker).generation == 0


def child_environment(root):
    import json
    import os

    root.mkdir(parents=True, exist_ok=True)
    (root / "data" / "crash-test").mkdir(mode=0o700, parents=True, exist_ok=True)
    (root / "home").mkdir(mode=0o700, exist_ok=True)
    config = root / "config.toml"
    if not config.exists():
        config.write_text(
            f'[paths]\ndata_dir = {json.dumps(str(root / "data"))}\n[general]\nusers_name = "crash-test"\n[model_catalog]\nenabled = false\n'
        )
    env = os.environ.copy()
    env.update(
        TLDW_CONFIG_PATH=str(config),
        TLDW_TEST_MODE="1",
        HOME=str(root / "home"),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        XDG_DATA_HOME=str(root / "xdg-data"),
        XDG_CONFIG_HOME=str(root / "xdg-config"),
        XDG_CACHE_HOME=str(root / "xdg-cache"),
    )
    return env


def run_child(root, *arguments, barrier=None):
    import json
    import selectors
    import subprocess
    import sys
    from pathlib import Path

    worktree = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-I",
        "-c",
        "import sys,runpy;sys.path.insert(0," + repr(str(worktree)) + ");"
        "runpy.run_module('Tests.Plugins.recovery_worker',run_name='__main__')",
        *arguments,
        str(root),
    ]
    if barrier:
        command += ["--barrier", barrier]
    env = child_environment(root)
    log = root / "worker-stderr.log"
    with log.open("w") as errors:
        child = subprocess.Popen(
            command,
            cwd=worktree,
            env=env,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
        )
        try:
            if barrier:
                with selectors.DefaultSelector() as selector:
                    selector.register(child.stdout, selectors.EVENT_READ)
                    assert selector.select(20), log.read_text()
                line = child.stdout.readline()
                assert line, log.read_text()
                message = json.loads(line)
                assert message["barrier"] == barrier and message["pid"] == child.pid
                child.kill()  # This exact controlled owner only, never a discovered PID.
                child.wait(timeout=10)
                assert child.returncode < 0
                return message
            output, _ = child.communicate(timeout=25)
            assert child.returncode == 0, log.read_text()
            result = json.loads(output)
            assert Path(result["module"]).is_relative_to(worktree)
            assert (
                Path(result["profile"]).resolve()
                == (root / "data" / "crash-test").resolve()
            )
            return result
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            child.stdout.close()


@platform_support.managed_runtime
@pytest.mark.parametrize(
    "barrier,want,committed",
    [
        ("materialized", None, False),
        ("prepared", "aborted", False),
        ("registry_committed", "recovery_required", False),
        ("certified", "complete", True),
        ("marker_advanced", "complete", True),
        ("published", "complete", True),
    ],
)
def test_killed_owner_recovers_in_fresh_process(
    tmp_path, native_package, barrier, want, committed
):
    root = tmp_path / "child"
    package = native_package()
    barrier_result = run_child(
        root, "install", "--package", str(package), barrier=barrier
    )
    result = run_child(root, "recover")
    if want is None:
        assert result["receipts"] == []
    else:
        receipt = next(
            item
            for item in result["receipts"]
            if item["operation_id"] == barrier_result["operation_id"]
        )
        assert receipt["phase"] == want and receipt["committed"] is committed
    assert result["marker_generation"] == int(committed)
    if committed:
        again = run_child(root, "recover")
        assert again["receipts"] == result["receipts"]
        assert len(again["installations"]) == 1


@platform_support.managed_runtime
@pytest.mark.parametrize("barrier", ["certified", "marker_advanced"])
@pytest.mark.parametrize("loss", ["missing", "rollback"])
def test_fresh_process_recovers_registry_loss_with_runtime_gate(
    tmp_path, native_package, barrier, loss
):
    root = tmp_path / "child"
    package = native_package()
    first = run_child(
        root, "install", "--package", str(package), "--operation", "first"
    )
    registry = root / "data" / "crash-test" / "plugins" / "registry.sqlite3"
    backup = root / "old.sqlite3"
    shutil.copyfile(registry, backup)
    run_child(
        root,
        "install",
        "--package",
        str(package),
        "--operation",
        "second",
        barrier=barrier,
    )
    for path in (
        registry,
        registry.with_name(registry.name + "-wal"),
        registry.with_name(registry.name + "-shm"),
    ):
        path.unlink(missing_ok=True)
    if loss == "rollback":
        shutil.copyfile(backup, registry)
        registry.chmod(0o600)
    recovered = run_child(root, "recover")
    assert recovered["marker_generation"] == 2
    assert len(recovered["installations"]) == 2
    assert first["installations"][0] in recovered["installations"]
    assert len(recovered["blocked"]) == 2
    assert all(item["committed"] for item in recovered["receipts"])
    again = run_child(root, "recover")
    assert set(again["blocked"]) == set(recovered["blocked"])
    unknown = [
        row
        for row in again["processes"]
        if row["owner_session"] == "unknown:registry-reconstruction"
    ]
    assert len(unknown) == 2
    assert all(
        row["state"] == "unresolved" and "pid" not in row["provenance"]
        for row in unknown
    )


def test_new_registry_without_operation_hint_still_requires_commit_proof(
    plugin_stack, native_package
):
    stack = plugin_stack
    first = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(first, first.operation_id))
    second = reviewed(stack, native_package())
    interrupt_at(stack, "registry_committed")
    with pytest.raises(OSError):
        stack.call(lambda: stack.coordinator.commit(second, second.operation_id))

    def remove_hint():
        with stack.registry.transaction() as cursor:
            cursor.execute(
                "DELETE FROM operations WHERE operation_id=?", (second.operation_id,)
            )

    stack.call(remove_hint)
    receipts = stack.call(stack.coordinator.recover)
    assert any(item.phase == "recovery_required" for item in receipts)
    assert (
        len(stack.call(lambda: stack.registry.list_installations(limit=50, offset=0)))
        == 2
    )
    assert stack.call(stack.authority.load_marker).generation == 1


def test_secure_marker_snapshot_alone_recovers_and_repeats(
    plugin_stack, native_package
):
    stack = plugin_stack
    review = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    for folder in ("intents", "certificates"):
        for path in (stack.authority.store_dir / folder).iterdir():
            path.unlink()
    lose_registry(stack)
    assert stack.call(stack.coordinator.recover)[0].committed
    assert stack.call(stack.coordinator.recover)[0].committed
    assert (
        len(stack.call(lambda: stack.registry.list_installations(limit=50, offset=0)))
        == 1
    )


def test_ambiguous_certified_successors_never_choose_one(plugin_stack, native_package):
    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    stack = plugin_stack
    review = reviewed(stack, native_package())
    interrupt_at(stack, "certified")
    with pytest.raises(OSError):
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))

    def add_competing():
        evidence = stack.authority.verify_transition(review.operation_id)
        snapshot = evidence.snapshot
        result = dict(snapshot["operation_result"])
        result.pop("operation_id")
        snapshot["operation_result"]["operation_id"] = (
            stack.authority.issue_operation_id(1, "f" * 64, result, "e" * 64)
        )
        new = PluginMarker(
            generation=1,
            operation_id=snapshot["operation_result"]["operation_id"],
            recovery_snapshot_digest=snapshot_digest(snapshot),
        )
        stack.authority.prepare(snapshot, evidence.old, new)
        stack.authority.certify_commit(evidence.old, new)

    stack.call(add_competing)
    assert stack.call(stack.coordinator.recover)[0].phase == "recovery_required"
    assert stack.call(stack.authority.load_marker).generation == 0


def test_current_marker_snapshot_and_previous_evidence_are_retained(
    plugin_stack, native_package
):
    stack = plugin_stack
    for index in range(3):
        review = reviewed(stack, native_package())
        stack.call(
            lambda review=review, index=index: stack.coordinator.commit(
                review, review.operation_id
            )
        )
    stack.call(stack.coordinator.recover)
    assert len(list((stack.authority.store_dir / "snapshots").iterdir())) == 4
    assert len(list((stack.authority.store_dir / "intents").iterdir())) == 3
    assert len(list((stack.authority.store_dir / "certificates").iterdir())) == 3
    assert len(stack.call(stack.authority.verify_current)["installations"]) == 3


def test_reconstruction_preserves_existing_real_process_evidence(
    plugin_stack, native_package
):
    stack = plugin_stack
    review = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    token = stack.call(
        lambda: stack.owner.reserve_launch(
            "live", review.installation_id, None, review.inspection.effective_digest
        )
    )
    stack.call(
        lambda: stack.owner.publish_process(
            token, {"host_identity": "controlled-evidence", "pid": 12345}
        )
    )

    def roll_back_projection():
        with stack.registry.transaction() as cursor:
            cursor.execute("DELETE FROM selections")

    stack.call(roll_back_projection)
    assert stack.call(stack.coordinator.recover)[0].committed
    evidence = stack.call(lambda: stack.owner.list_processes(limit=50, offset=0))
    original = next(item for item in evidence if item["token"] == token)
    assert original["state"] == "published"
    assert original["provenance"] == {
        "host_identity": "controlled-evidence",
        "pid": 12345,
    }
    assert len(evidence) == 2


def test_reconstruction_preserves_blocked_component_definitions(
    plugin_stack, native_package
):
    stack = plugin_stack
    package = native_package(
        extension={"version": 1, "requires": {"skill:review": ["hook:missing"]}}
    )
    review = reviewed(stack, package)
    assert review.inspection.inventory["skill:review"].activation_blockers
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    before = stack.call(stack.authority.verify_current)
    lose_registry(stack)
    assert stack.call(stack.coordinator.recover)[0].committed
    assert (
        stack.call(
            lambda: stack.registry.authority_projection(
                operation_result=before["operation_result"]
            )
        )
        == before
    )


def test_retained_definition_mismatch_never_partially_reconstructs(
    plugin_stack, native_package
):
    from copy import deepcopy

    from tldw_chatbook.Plugins.recovery import retained_inspections

    stack = plugin_stack
    review = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    original = stack.call(stack.authority.verify_current)
    altered = deepcopy(original)
    altered["components"][0]["definition_digest"] = "0" * 64
    with pytest.raises(ValueError, match="retained_interpretation_mismatch"):
        retained_inspections(altered)
    assert (
        stack.call(
            lambda: stack.registry.authority_projection(
                operation_result=original["operation_result"]
            )
        )
        == original
    )


def test_aborted_prepare_does_not_poison_later_reviewed_commit(
    plugin_stack, native_package
):
    stack = plugin_stack
    review = reviewed(stack, native_package())
    interrupt_at(stack, "prepared")
    with pytest.raises(OSError):
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    stack.call(lambda: setattr(stack.coordinator, "progress", None))
    assert (
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id)).phase
        == "aborted"
    )
    replacement = reviewed(stack, Path(review.inspection.source_identity))
    assert stack.call(
        lambda: stack.coordinator.commit(replacement, replacement.operation_id)
    ).committed
    receipts = stack.call(stack.coordinator.recover)
    assert [(item.operation_id, item.phase) for item in receipts] == [
        (review.operation_id, "aborted"),
        (replacement.operation_id, "complete"),
    ]
    assert len(stack.call(stack.coordinator.published_snapshot)["installations"]) == 1


@pytest.mark.parametrize("old_generation", [1, 3], ids=["older", "equal"])
@pytest.mark.parametrize("changed_field", ["operation_id", "recovery_snapshot_digest"])
def test_prepared_intent_with_unrelated_old_marker_quarantines_before_abort(
    plugin_stack, native_package, old_generation, changed_field
):
    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    stack = plugin_stack
    markers = {}
    for generation in range(1, 4):
        review = reviewed(stack, native_package())
        stack.call(
            lambda review=review, generation=generation: stack.coordinator.commit(
                review, review.operation_id
            )
        )
        markers[generation] = stack.call(stack.authority.load_marker)
    # Successful same-stack control proves the accepted chain is publishable.
    assert all(receipt.committed for receipt in stack.call(stack.coordinator.recover))
    accepted = stack.call(stack.coordinator.published_snapshot)
    operation_id = "unrelated-prepared"

    def prepare_unrelated():
        changed = (
            "different-old-operation" if changed_field == "operation_id" else "0" * 64
        )
        old = markers[old_generation].model_copy(update={changed_field: changed})
        snapshot = stack.authority.verify_current()
        snapshot["operation_result"]["operation_id"] = operation_id
        new = PluginMarker(
            generation=old.generation + 1,
            operation_id=operation_id,
            recovery_snapshot_digest=snapshot_digest(snapshot),
        )
        # Use the real protected writer/MAC primitive to construct authenticated
        # out-of-lineage evidence. This is not a claim an untrusted caller can
        # forge a MAC; recovery must reject its exact old identity nevertheless.
        stack.authority._save_snapshot(snapshot, new)
        stack.authority._save_evidence(
            "prepared", stack.authority._transition(old, new)
        )
        evidence = stack.authority.verify_transition(operation_id)
        assert evidence.old == old and evidence.committed is False
        assert stack.registry.read_operation(operation_id) is None

    stack.call(prepare_unrelated)
    artifacts = {
        path: path.read_bytes()
        for folder in ("intents", "snapshots", "certificates")
        for path in (stack.authority.store_dir / folder).iterdir()
    }
    receipts = stack.call(stack.coordinator.recover)
    assert len(receipts) == 1
    assert receipts[0].operation_id == operation_id
    assert receipts[0].phase == "recovery_required"
    assert receipts[0].committed is False
    with pytest.raises(PermissionError):
        stack.call(stack.coordinator.published_snapshot)
    assert stack.call(stack.authority.load_marker) == markers[3]
    assert (
        stack.call(
            lambda: stack.registry.authority_projection(
                operation_result=accepted["operation_result"]
            )
        )
        == accepted
    )
    assert all(path.read_bytes() == payload for path, payload in artifacts.items())


@pytest.mark.parametrize(
    "prepare_generation", [0, 1, 3], ids=["genesis", "ancestor", "current"]
)
def test_prepared_intent_on_exact_accepted_endpoint_aborts_without_fencing(
    plugin_stack, native_package, prepare_generation
):
    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    stack = plugin_stack
    markers = {0: stack.call(stack.authority.load_marker)}
    for generation in range(1, 4):
        review = reviewed(stack, native_package())
        stack.call(
            lambda review=review, generation=generation: stack.coordinator.commit(
                review, review.operation_id
            )
        )
        markers[generation] = stack.call(stack.authority.load_marker)
    accepted = stack.call(stack.coordinator.published_snapshot)

    def prepare_legitimate():
        old = markers[prepare_generation]
        snapshot = stack.authority.verify_current()
        snapshot["operation_result"]["operation_id"] = "legitimate-prepared"
        new = PluginMarker(
            generation=old.generation + 1,
            operation_id="legitimate-prepared",
            recovery_snapshot_digest=snapshot_digest(snapshot),
        )
        stack.authority._save_snapshot(snapshot, new)
        stack.authority._save_evidence(
            "prepared", stack.authority._transition(old, new)
        )

    stack.call(prepare_legitimate)
    receipts = stack.call(stack.coordinator.recover)
    aborted = next(
        item for item in receipts if item.operation_id == "legitimate-prepared"
    )
    assert aborted.phase == "aborted" and aborted.committed is False
    assert sum(item.committed for item in receipts) == 3
    assert stack.call(stack.coordinator.published_snapshot) == accepted


def test_retired_issued_hints_reconstruct_current_but_newer_compound_loss_refuses(
    plugin_stack, native_package
):
    from Tests.Plugins.test_coordinator import reviewed

    stack = plugin_stack
    first = reviewed(stack, native_package())
    assert stack.call(
        lambda: stack.coordinator.commit(first, first.operation_id)
    ).committed
    second = reviewed(stack, native_package())
    assert stack.call(
        lambda: stack.coordinator.commit(second, second.operation_id)
    ).committed
    expected = stack.call(stack.coordinator.published_snapshot)
    # Simulate an old hint restored after an eligible oldest-prefix retirement.
    for purpose in ("committed", "prepared"):
        stack.authority._operation_path(purpose, first.operation_id).unlink()
    receipts = stack.call(stack.coordinator.recover)
    assert all(item.phase == "complete" for item in receipts)
    assert stack.call(stack.coordinator.published_snapshot) == expected
    third = reviewed(stack, native_package())

    def fail(phase):
        if phase == "registry_committed":
            raise OSError("compound loss")

    stack.call(lambda: setattr(stack.coordinator, "progress", fail))
    with pytest.raises(OSError):
        stack.call(lambda: stack.coordinator.commit(third, third.operation_id))
    stack.call(lambda: setattr(stack.coordinator, "progress", None))
    stack.authority._operation_path("prepared", third.operation_id).unlink()
    assert stack.call(stack.coordinator.recover)[0].phase == "recovery_required"
    assert stack.call(stack.authority.verify_current) == expected


def _legacy_fixture_namespace(stack):
    """Construct the documented pre-F7 metadata; never a production writer option."""
    import json

    metadata = stack.authority.store_dir / "metadata.json"
    value = json.loads(metadata.read_text())
    assert stack.call(stack.authority.load_marker).generation == 0
    metadata.write_text(json.dumps({"schema_version": 1, "salt": value["salt"]}))
    (stack.authority.store_dir / "legacy-cutover.json").unlink()


def _legacy_fixture_commit(stack, package, identity):
    """Execute the frozen F4 v1 protocol to author legacy recovery evidence."""
    from Tests.Plugins.test_coordinator import reviewed
    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    review = reviewed(stack, package)

    def write():
        coordinator = stack.coordinator
        retained = coordinator._materialize(review)
        old = stack.authority.load_marker()
        result = {
            "operation_id": identity,
            "installation_id": review.installation_id,
            "kind": "install",
            "revision_digest": retained.effective_digest,
            "result": "committed",
        }
        with stack.registry.transaction() as cursor:
            coordinator._apply_review(cursor, review, retained)
            snapshot = stack.registry.authority_projection(operation_result=result)
            new = PluginMarker(
                generation=old.generation + 1,
                operation_id=identity,
                recovery_snapshot_digest=snapshot_digest(snapshot),
            )
            stack.authority._save_snapshot(snapshot, new)
            stack.authority._save_evidence(
                "prepared", stack.authority._transition(old, new)
            )
            stack.registry.write_operation(cursor, result, phase="committed")
        stack.authority.certify_commit(old, new)
        stack.authority.advance_marker(old, new)
        coordinator._published = snapshot
        return stack.authority.verify_transition(identity)

    return stack.call(write)


def test_interrupted_cutover_requalifies_valid_legacy_writer_then_old_backup(
    plugin_stack, native_package, monkeypatch, tmp_path
):
    import sqlite3

    from tldw_chatbook.Plugins import authority_store

    stack = plugin_stack
    _legacy_fixture_namespace(stack)
    first = _legacy_fixture_commit(stack, native_package(), "legacy-first")
    backup = tmp_path / "old-registry.sqlite"

    def save_backup():
        with sqlite3.connect(backup) as destination:
            stack.registry._connection.backup(destination)

    stack.call(save_backup)
    old_metadata = stack.authority.metadata()
    write = authority_store._write

    def interrupt_metadata(path, payload, **kwargs):
        if path.name == "metadata.json" and payload.get("schema_version") == 2:
            raise OSError("interrupted before metadata switch")
        return write(path, payload, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(authority_store, "_write", interrupt_metadata)
        with pytest.raises(OSError):
            stack.call(lambda: stack.authority.ensure_issued_metadata((first,)))
    assert stack.authority.metadata() == old_metadata
    assert stack.call(stack.authority.verify_legacy_cutover).marker == first.new
    second = _legacy_fixture_commit(stack, native_package(), "legacy-second")
    expected = stack.call(stack.authority.verify_current)
    assert all(
        receipt.phase == "complete" for receipt in stack.call(stack.coordinator.recover)
    )
    final = stack.call(stack.authority.verify_legacy_cutover)
    assert final.marker == second.new
    assert {entry.operation_id for entry in final.entries} == {
        "legacy-first",
        "legacy-second",
    }
    assert stack.authority.metadata()["salt"] == old_metadata["salt"]
    assert stack.authority.metadata()["schema_version"] == 2
    stack.call(lambda: stack.registry.forget_operation_hints((first.new.operation_id,)))
    stack.call(lambda: stack.authority.retire_transition(first))

    def restore_backup():
        with sqlite3.connect(backup) as source:
            source.backup(stack.registry._connection)

    stack.call(restore_backup)
    assert all(
        receipt.phase == "complete" for receipt in stack.call(stack.coordinator.recover)
    )
    assert stack.call(stack.coordinator.published_snapshot) == expected
    assert stack.call(stack.authority.verify_legacy_cutover) == final


# TASK-40132: backups capture the plugin root (owner decision 2026-11-10).


def _plugin_root(tmp_path, *, process_state=None, config=None):
    """A selected profile whose plugin root holds real registry, package and data bytes."""
    from contextlib import closing

    from tldw_chatbook.Plugins.registry import _schema_statements
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    config = config or tmp_path / "selected.toml"
    config.write_text(
        f'[general]\nusers_name = "Ada"\n[paths]\ndata_dir = "{tmp_path / "data"}"\n'
    )
    root = tmp_path / "data" / "Ada" / "plugins"
    files = {
        "packages/inst/plugin.json": b'{"name": "inst"}',
        "packages-revisions/inst/digest/plugin.json": b'{"name": "inst", "v": 2}',
        "data/root-1/notes.txt": b"made by the plugin",
        "staging/op/snapshot/plugin.json": b'{"name": "git"}',
        # What a fetch leaves beside its snapshot (D1: never archived).
        "staging/op/host-git-config": b"[credential]\n\thelper = !token\n",
        "staging/op/host-ssh-input": b"Host *\n  IdentityFile ~/.ssh/id\n",
        "staging/op/host-ssh-config": b"Host x\n",
        "staging/op/repository.git/HEAD": b"ref: refs/heads/main\n",
        "staging/op/content/plugin.json": b'{"name": "git"}',
        # I1: anything else in a fetch folder is excluded too, known or not.
        "staging/op/host-git-config.lock": b"",
        "staging/op/later-member/state": b"unknown",
        "acquisition-custody/op.json": b'{"phase": "terminal"}',
        "cache/index.json": b"{}",
        "runtime.lock": b"",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o600)
    registry = root / "registry.sqlite3"
    with closing(sqlite3.connect(registry)) as connection:
        for migration in (
            "001_initial.sql",
            "002_authority.sql",
            "003_installation_alias.sql",
            "004_root_users.sql",
            "005_marketplace_sources.sql",
        ):
            for statement in _schema_statements(migration):
                connection.execute(statement)
        connection.execute("PRAGMA user_version=5")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("INSERT INTO installations(installation_id) VALUES ('inst')")
        connection.execute(
            "INSERT INTO data_roots(root_id, installation_id, path, generation)"
            " VALUES ('root-1', 'inst', ?, 0)",
            # As the runtime owner names it: parent links resolved (M8).
            (str(PluginRuntimeOwner(root).root / "data" / "root-1"),),
        )
        if process_state is not None:
            connection.execute(
                "INSERT INTO processes(token, operation_id, installation_id,"
                " revision_digest, owner_session, state, kind)"
                " VALUES ('t', 'op', 'inst', 'digest', 'session', ?, 'active_run')",
                (process_state,),
            )
        connection.commit()
    registry.chmod(0o600)
    for directory in (tmp_path / "data", *(p for p in root.parent.rglob("*") if p.is_dir())):
        directory.chmod(0o700)
    return config, root


def _plugin_rows(config, root):
    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    install_adapters()
    inventory = discover((config,))
    rows = {
        item.path: item
        for item in inventory.items
        if item.path is not None and (item.path == root or root in item.path.parents)
    }
    return inventory, rows


# The plugin root exists only where the runtime owner qualifies (POSIX here).
@platform_support.package_capture
def test_backup_preview_captures_plugin_root_except_runtime_lock(tmp_path):
    """AC1/AC2: registry, packages, revisions, every data root, staging snapshots and
    acquisition custody are captured; runtime.lock and every other member of a
    fetch folder are excluded (D1 as an allow-list, I1), each as one row whose
    contents are never listed, and the registry's live WAL/SHM stay transient
    SQLite sidecars."""
    config, root = _plugin_root(tmp_path)
    registry = root / "registry.sqlite3"
    reader = sqlite3.connect(registry)  # keeps -wal/-shm present, as a live store does
    try:
        reader.execute("PRAGMA wal_autocheckpoint=0")
        reader.execute("INSERT INTO installations(installation_id) VALUES ('later')")
        reader.commit()
        inventory, rows = _plugin_rows(config, root)
    finally:
        reader.close()
    statuses = {
        path.relative_to(root).as_posix(): (item.owner, item.status)
        for path, item in rows.items()
    }
    assert statuses.get("runtime.lock") == ("plugins", "intentionally_excluded")
    del statuses["runtime.lock"]
    # D1 (2026-11-10), I1: inside staging/<name>/ only snapshot/ is captured;
    # every other member is excluded at the directory level, unlisted inside.
    for relative in (
        "staging/op/host-git-config",
        "staging/op/host-ssh-input",
        "staging/op/host-ssh-config",
        "staging/op/repository.git",
        "staging/op/content",
        "staging/op/host-git-config.lock",
        "staging/op/later-member",
    ):
        assert statuses.pop(relative) == ("plugins", "intentionally_excluded"), relative
    assert not [
        relative
        for relative in statuses
        if relative.startswith("staging/op/") and relative.split("/")[2] != "snapshot"
    ]
    assert statuses.pop("registry.sqlite3-wal") == (
        "sqlite.transient",
        "intentionally_excluded",
    )
    assert statuses.pop("registry.sqlite3-shm") == (
        "sqlite.transient",
        "intentionally_excluded",
    )
    for relative in (
        "registry.sqlite3",
        "packages/inst/plugin.json",
        "packages-revisions/inst/digest/plugin.json",
        "data/root-1/notes.txt",
        "staging/op/snapshot/plugin.json",
        "acquisition-custody/op.json",
        "cache/index.json",
    ):
        assert statuses.pop(relative) == ("plugins", "included"), relative
    assert set(statuses.values()) == {("plugins", "included_directory")}
    assert "." in statuses
    assert not any(item.owner == "unknown" for item in inventory.items if item.path == root)


# The plugin root exists only where the runtime owner qualifies (POSIX here).
@platform_support.package_capture
def test_plugin_registry_capture_folds_wal_into_its_exact_catalog(tmp_path, monkeypatch):
    """AC3: the frozen catalog matches the installed migrations, and capture copies
    committed WAL frames through the SQLite backup route like the other owners."""
    from contextlib import closing
    from threading import Event

    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.models import FileMetadata, StorageItem
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Plugins.registry import PluginRegistry

    owners = {owner.owner_id: owner for owner in install_adapters()}
    assert "plugins" in owners
    from tldw_chatbook.Plugins.backup import REGISTRY_SCHEMAS

    with PluginRegistry._reference_schema() as reference:
        installed = tuple(
            row[3]
            for row in reference.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name"
            )
            if row[3] is not None
        )
    assert REGISTRY_SCHEMAS == ((5, installed),)
    _, root = _plugin_root(tmp_path)
    registry = root / "registry.sqlite3"
    writer = sqlite3.connect(registry)
    try:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("INSERT INTO installations(installation_id) VALUES ('wal-only')")
        writer.commit()
        assert Path(str(registry) + "-wal").stat().st_size > 0
        authority = application_authority(tmp_path, root, monkeypatch)
        adapter = owners["plugins"]
        item = StorageItem(
            "plugins",
            "profile:test:plugins:registry",
            registry,
            "included",
            (),
            metadata=FileMetadata(
                1, "profile:test:plugins", "registry.sqlite3",
                "profile:test:plugins", "file", 0o600, 0, "private",
            ),
        )
        stage = tmp_path / "stage"
        stage.mkdir(mode=0o700)
        destination = stage / "registry.sqlite3"
        with (
            authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
            session.capture_scope((registry,), stage),
        ):
            adapter.capture(item, destination, Event())
            assert adapter.validate(destination) == ()
        assert not Path(str(destination) + "-wal").exists()
        with closing(sqlite3.connect(destination)) as captured:
            assert captured.execute("PRAGMA user_version").fetchone()[0] == 5
            assert {
                row[0]
                for row in captured.execute("SELECT installation_id FROM installations")
            } == {"inst", "wal-only"}
    finally:
        writer.close()


@platform_support.package_capture
@pytest.mark.parametrize("writer", ["process", "lock"])
def test_live_plugin_writers_refuse_preview_and_capture(tmp_path, monkeypatch, writer):
    """AC4: an unsettled plugin process or a live runtime owner makes the preview
    refuse every plugin item with a reason, and capture refuses the copy."""
    from threading import Event

    import portalocker

    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.recovery_service import issue_message

    config, root = _plugin_root(
        tmp_path, process_state="published" if writer == "process" else None
    )
    notes = root / "data" / "root-1" / "notes.txt"
    if writer == "lock":
        # A capture item from an idle copy of the same layout.
        (tmp_path / "free").mkdir()
        free = _plugin_rows(*_plugin_root(tmp_path / "free"))[1][
            tmp_path / "free" / "data" / "Ada" / "plugins" / "data" / "root-1" / "notes.txt"
        ]
    handle = open(root / "runtime.lock", "r+b")
    try:
        if writer == "lock":
            portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
        inventory, rows = _plugin_rows(config, root)
        assert "plugin_writers_active" in inventory.issues
        assert not inventory.complete
        # C1: one unavailable root and no children keep the tree valid, so a
        # partial or Tools-free backup still works.
        assert {path: item.status for path, item in rows.items()} == {root: "unavailable"}
        assert "overlapping_owner_roots" not in inventory.issues
        assert any(
            item.owner == "plugins" and item.logical_id.endswith(":writers_active")
            for item in inventory.items
        )
        assert "quit Chatbook" in issue_message("plugin_writers_active")
        if writer == "lock":
            from tldw_chatbook.Plugins.backup import recovery_adapters

            (adapter,) = recovery_adapters()
            item = replace(free, path=notes)  # the same item, at this root
            authority = application_authority(tmp_path, root, monkeypatch)
            stage = tmp_path / "stage"
            stage.mkdir(mode=0o700)
            destination = stage / "notes.txt"
            with (
                authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
                session.capture_scope((item.path,), stage),
            ):
                with pytest.raises(ValueError, match="plugin_writers_active"):
                    adapter.capture(item, destination, Event())
                assert not destination.exists()
                portalocker.unlock(handle)
                adapter.capture(item, destination, Event())
            assert destination.read_bytes() == b"made by the plugin"
    finally:
        handle.close()


# The plugin root exists only where the runtime owner qualifies (POSIX here).
@platform_support.package_capture
def test_restored_registry_must_name_its_own_plugin_root(tmp_path):
    """AC8 defense: a registry whose data roots name another plugin root refuses."""
    from tldw_chatbook.Backup_Recovery.models import FileMetadata, StorageItem
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.Plugins.backup import recovery_adapters

    _, root = _plugin_root(tmp_path)
    (adapter,) = recovery_adapters()
    metadata = FileMetadata(
        1, "profile:p:plugins", "registry.sqlite3", "profile:p:plugins",
        "file", 0o600, 0, "private",
    )
    candidate = root / "registry.sqlite3"
    # M2: with no data roots, a revision's installed or fetched package path
    # still binds the registry to its original plugin root.
    from contextlib import closing

    bare = tmp_path / "bare-registry.sqlite3"
    shutil.copyfile(candidate, bare)
    bare.chmod(0o600)
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    owned = PluginRuntimeOwner(root).root  # the root as the registry names it
    # Closed, so the WAL is checkpointed into the file the restore route reads.
    with closing(sqlite3.connect(bare)) as connection, connection:
        connection.execute("DELETE FROM data_roots")
        connection.execute(
            "INSERT INTO revisions VALUES ('inst', 'digest', ?)",
            (
                '{"materialized_identity": "'
                + str(owned / "packages" / "inst")
                + '", "acquisition_reference": {"materialized_identity": "'
                + str(owned / "staging" / "op" / "snapshot")
                + '"}}',
            ),
        )
    for destination, refused in (
        (root, False),
        (tmp_path / "elsewhere" / "plugins", True),
    ):
        item = StorageItem(
            "plugins", "profile:p:plugins:registry", destination / "registry.sqlite3",
            "included", (), metadata=metadata,
        )
        with _preview_reads():
            if refused:
                with pytest.raises(ValueError, match="owner_relocation_unverified:plugins"):
                    adapter.relocate_restore(item, bare, {})
            else:
                adapter.relocate_restore(item, bare, {})
    for destination, refused in ((root, False), (tmp_path / "elsewhere" / "plugins", True)):
        item = StorageItem(
            "plugins", "profile:p:plugins:registry", destination / "registry.sqlite3",
            "included", (), metadata=metadata,
        )
        with _preview_reads():
            if refused:
                with pytest.raises(ValueError, match="owner_relocation_unverified:plugins"):
                    adapter.relocate_restore(item, candidate, {})
            else:
                adapter.relocate_restore(item, candidate, {})

@platform_support.package_capture
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_odd_staging_member_is_excluded_not_blocking(tmp_path, kind):
    """I-3 (r13g-rev3): whatever a fetch folder member is, it is never copied, so
    a link, a hard-linked file or a FIFO there is excluded and blocks nothing."""
    import os

    config, root = _plugin_root(tmp_path)
    member = root / "staging/op/odd-member"
    if kind == "symlink":
        member.symlink_to(root / "staging/op/snapshot")
    elif kind == "hardlink":
        os.link(root / "staging/op/host-git-config", member)
    else:
        os.mkfifo(member, 0o600)
    _, rows = _plugin_rows(config, root)
    assert rows[member].status == "intentionally_excluded", rows[member].status
    blocking = {"unsupported", "unavailable", "missing_required"}
    assert not [(path, item.status) for path, item in rows.items() if item.status in blocking]


@platform_support.package_capture
@pytest.mark.parametrize("planted", ["fetched", "snapshot-only"])
def test_replace_restore_refuses_a_fetch_folder_added_since_the_backup(tmp_path, planted):
    """I-1 (owner decision 2026-11-12, refuse early): a replace restore retires a
    fetch folder the backup lacks, which it cannot do while the folder holds
    members no backup captures. The preview refuses with a plugin reason that
    says what to do; a folder holding only its snapshot restores."""
    from tldw_chatbook.Backup_Recovery.recovery_service import issue_code, issue_message
    from tldw_chatbook.Plugins import backup

    config, root = _plugin_root(tmp_path)
    refuse = getattr(backup, "refuse_added_fetch_folders", None)
    assert refuse is not None, "no plugin-specific restore preview check"
    backed_up = _plugin_rows(config, root)[0].items
    archived = {item.logical_id: item for item in backed_up}
    refuse(backed_up, archived)  # the backup's own fetch folders never refuse
    added = root / "staging" / "zz-new"
    members = {"snapshot/plugin.json": b"{}"}
    if planted == "fetched":
        members.update({"repository.git/HEAD": b"ref", "host-git-config": b"x"})
    for relative, payload in members.items():
        (added / relative).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        (added / relative).write_bytes(payload)
        (added / relative).chmod(0o600)
    target = _plugin_rows(config, root)[0]
    if planted == "snapshot-only":
        refuse(target.items, archived)
        return
    with pytest.raises(ValueError) as refused:
        refuse(target.items, archived)
    code = issue_code(refused.value)
    assert code == "plugin_fetch_added_since_backup"
    message = issue_message(code)
    assert "downloaded from a repository since this backup" in message
    assert "a source refresh, an update check, an install or an update" in message
    assert "without the Tools and skills group" in message
    assert "except the snapshot folder" in message and "Uninstall" not in message


@platform_support.package_capture
def test_restored_registry_matches_its_root_through_a_symlinked_data_dir(tmp_path):
    """M8: the registry names its root as the runtime owner resolves it, so a
    restore onto the same root reached through a linked data folder is in place,
    and another root still refuses."""
    from tldw_chatbook.Backup_Recovery.models import FileMetadata, StorageItem
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.Plugins.backup import recovery_adapters

    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    _, root = _plugin_root(real)
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    (adapter,) = recovery_adapters()
    metadata = FileMetadata(
        1, "profile:p:plugins", "registry.sqlite3", "profile:p:plugins",
        "file", 0o600, 0, "private",
    )
    # The lexical destination, as the restore plan names it through the link.
    for destination, refused in (
        (link / root.relative_to(real), False),
        (tmp_path / "elsewhere" / "plugins", True),
    ):
        item = StorageItem(
            "plugins", "profile:p:plugins:registry", destination / "registry.sqlite3",
            "included", (), metadata=metadata,
        )
        with _preview_reads():
            if refused:
                with pytest.raises(ValueError, match="owner_relocation_unverified:plugins"):
                    adapter.relocate_restore(item, root / "registry.sqlite3", {})
            else:
                adapter.relocate_restore(item, root / "registry.sqlite3", {})


@platform_support.package_capture
@pytest.mark.parametrize("member", ["host-git-config", "later-member/state"])
def test_capture_refuses_an_included_item_at_an_excluded_staging_path(
    tmp_path, monkeypatch, member
):
    """M2: capture re-checks the staging allow-list, so an item claiming to be
    included at an excluded fetch member is refused and never copied."""
    from threading import Event

    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Plugins.backup import recovery_adapters

    config, root = _plugin_root(tmp_path)
    snapshot = _plugin_rows(config, root)[1][root / "staging/op/snapshot/plugin.json"]
    assert snapshot.status == "included"
    relative = "staging/op/" + member
    item = replace(
        snapshot,
        path=root / relative,
        metadata=replace(snapshot.metadata, relative_path=relative),
    )
    (adapter,) = recovery_adapters()
    authority = application_authority(tmp_path, root, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    destination = stage / "member"
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((item.path,), stage),
    ):
        with pytest.raises(ValueError, match="plugin_staging_member_excluded"):
            adapter.capture(item, destination, Event())
    assert not destination.exists()


@platform_support.package_capture
def test_excluded_staging_members_are_never_walked(tmp_path, monkeypatch):
    """I1/M9: a local-source repository.git can hold tens of thousands of files.
    An excluded member is one row whose contents are never listed, so they count
    toward no entry limit and produce no preview rows."""
    from tldw_chatbook.Backup_Recovery import file_inventory

    config, root = _plugin_root(tmp_path)
    objects = root / "staging/op/repository.git/objects"
    objects.mkdir(mode=0o700)
    for index in range(80):
        (objects / f"{index:02x}").write_bytes(b"object")
    # The plugin root itself holds about 35 entries; the planted tree alone is over.
    monkeypatch.setattr(file_inventory, "MAX_ENTRIES", 60)
    _, rows = _plugin_rows(config, root)
    assert rows[root].status == "included_directory"
    assert rows[root / "staging/op/repository.git"].status == "intentionally_excluded"
    assert not [path for path in rows if path == objects or objects in path.parents]


@platform_support.package_capture
def test_capture_holds_the_runtime_lock_even_when_none_existed(tmp_path, monkeypatch):
    """M3: after a restore runtime.lock is absent; capture creates and holds it, so
    no runtime owner can start while plugin bytes are copied."""
    from threading import Event

    import portalocker

    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration
    from tldw_chatbook.Plugins.backup import recovery_adapters

    config, root = _plugin_root(tmp_path)
    (root / "runtime.lock").unlink()
    _, rows = _plugin_rows(config, root)
    item = rows[root / "data" / "root-1" / "notes.txt"]
    observed = []
    original = _RawDeclaration.capture

    def probing(self, *args, **kwargs):
        try:
            with open(root / "runtime.lock", "rb") as handle:
                portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
                observed.append("free")
                portalocker.unlock(handle)
        except FileNotFoundError:
            observed.append("absent")
        except portalocker.exceptions.LockException:
            observed.append("held")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_RawDeclaration, "capture", probing)
    (adapter,) = recovery_adapters()
    authority = application_authority(tmp_path, root, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((item.path,), stage),
    ):
        adapter.capture(item, stage / "notes.txt", Event())
    assert observed == ["held"]


@platform_support.managed_runtime
@pytest.mark.asyncio
async def test_plugin_service_retries_after_a_backup_released_the_lock(tmp_path):
    """M3: a first plugin use while a capture held runtime.lock must not stay
    plugin_owner_busy until restart."""
    import portalocker

    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.service import PluginService

    root = tmp_path / "profile" / "plugins"
    root.mkdir(parents=True, mode=0o700)
    handle = open(root / "runtime.lock", "a+b")
    portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
    service = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: None,
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    try:
        assert await service.authority_status() == "plugin_owner_busy"
        portalocker.unlock(handle)
        handle.close()
        assert await service.authority_status() == "needs_setup"
    finally:
        handle.close()
        await service.aclose()


@pytest.mark.parametrize(
    ("phase", "posture", "approved"),
    [
        ("recovery_required", "ready", False),
        ("complete", "locked", False),
        ("complete", "recovery_required", False),
        ("complete", "ready", True),
        ("complete", "ready_reduced", True),
    ],
)
def test_only_completed_ready_recovery_approves_a_restore(
    tmp_path, monkeypatch, phase, posture, approved
):
    """M5: a recovery_required receipt or a non-ready posture approves nothing."""
    from types import SimpleNamespace

    from tldw_chatbook.Plugins import backup
    from tldw_chatbook.Plugins.review import OperationReceipt
    from tldw_chatbook.Plugins.service import PluginService

    calls = []
    monkeypatch.setattr(backup, "approve_restore", calls.append)
    monkeypatch.setattr(backup, "restore_review_required", lambda root: not calls)
    service = PluginService(tmp_path, workspace_lookup=lambda _: None)
    service._restore_review = True
    service._coordinator = SimpleNamespace(
        authority=SimpleNamespace(posture=lambda: posture),
        owner=SimpleNamespace(root=tmp_path / "plugins"),
    )
    service._finish_restore_review((OperationReceipt("op", phase, phase == "complete"),))
    assert calls == ([tmp_path / "plugins"] if approved else [])
    assert service._restore_review is not approved


def test_unlock_mismatch_copy_is_truthful_and_has_no_dead_end_step():
    """D2: a marker naming missing authority files is a restore mismatch, not a
    passphrase problem; other failures keep the retry advice."""
    from tldw_chatbook.Widgets.plugin_review_modal import authority_failure_message

    mismatch = authority_failure_message(FileNotFoundError("snapshot"), "unlock")
    assert "cannot recover these plugins yet" in mismatch
    assert "stay on disk" in mismatch and "Reset" not in mismatch
    assert "passphrase incorrect" in authority_failure_message(ValueError(), "unlock")
    assert "passphrase incorrect" in authority_failure_message(
        FileNotFoundError(), "setup"
    )


def test_live_writer_reason_gives_the_way_out_of_a_stale_process_record():
    """M4/M7: a live service clears by quitting; Console plugin use starts it too.
    A process row left by an earlier session clears only through the plugin
    runtime's recovery, so the copy gives those steps, ending in quit and reopen
    (opening Plugins takes the lock again) before backing up."""
    from tldw_chatbook.Backup_Recovery.recovery_service import issue_message

    message = issue_message("plugin_writers_active")
    assert "quit Chatbook" in message and "partial backup" in message
    assert "Console" in message
    steps = message[message.index("earlier session") :]
    assert steps.index("open Plugins") < steps.index("Authority…") < steps.index(
        "quit and reopen Chatbook"
    ) < steps.index("back up")


_HELD_LOCK_SETUP = r"""
import os
from pathlib import Path
# The profile exists before any import loads config (else a default one appears).
source=Path(os.environ['TLDW_CONFIG_PATH']);home=Path.home()
source.write_text(f'[general]\nusers_name = "Ada"\n[paths]\ndata_dir = "{home / "data"}"\n')
(home/'data').mkdir(mode=0o700)
from Tests.network_guard import install
install()
from Tests.Plugins.test_recovery import _plugin_root
from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
_plugin_root(home,config=source)
_populate_required_dependencies(preview_capture((source,),options={'allow_partial':True}))
print('HELD_LOCK_SETUP_COMPLETE',flush=True)
"""

# A separate process: the setup's store handles would hold the leases capture pauses.
_HELD_LOCK_BACKUPS = r"""
import json,os,zipfile
from pathlib import Path
from Tests.network_guard import install
install()
os.environ['PYTHON_KEYRING_BACKEND']='keyring.backends.null.Keyring'
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
import portalocker
from tldw_chatbook.Backup_Recovery import credentials
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
store=KeyringServerCredentialStore(keyring_backend=Keyring());credentials._credential_store=lambda:store
source=Path(os.environ['TLDW_CONFIG_PATH']);home=Path.home();root=home/'data'/'Ada'/'plugins'
# The app's plugin service holds this lock from first plugin use until quit.
handle=open(root/'runtime.lock','r+b');portalocker.lock(handle,portalocker.LOCK_EX|portalocker.LOCK_NB)
service=RecoveryService(home/'recovery-control')
try:
 for name,groups in (('prompts',('prompts',)),('whole',None)):
  options={'allow_partial':True}
  if groups:options['data_groups']=groups
  preview=service.preview_backup((source,),options=options)
  if groups:assert preview.complete and 'plugin_writers_active' not in preview.issues,preview.issues
  else:assert 'plugin_writers_active' in preview.issues and not preview.complete,preview.issues
  destination=home/(name+'.tldw-backup.zip')
  result=service.wait(service.start_backup((source,),preview.scope_digest,destination,options=options,password=None))
  assert result['state']=='succeeded',dict(result)
  with zipfile.ZipFile(destination) as archive:manifest=json.loads(archive.read('manifest.json'))
  assert not [row for row in manifest['files'] if row['owner_id']=='plugins'],name
finally:
 service.close();handle.close()
print('HELD_LOCK_BACKUPS_COMPLETE',flush=True)
"""


@platform_support.package_capture
@pytest.mark.timeout(600)
def test_held_plugin_lock_leaves_other_and_partial_backups_available(tmp_path):
    """C1: while the plugin service holds runtime.lock, a backup without Tools
    succeeds and a whole-profile partial backup succeeds without plugins."""
    from Tests.Backup_Recovery.test_config_sibling_capture import _child

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    selector = tmp_path / "profile-config" / "config.toml"
    selector.parent.mkdir(mode=0o700)
    setup = _child(home, _HELD_LOCK_SETUP, "setup", timeout=180, selector=selector)
    assert "HELD_LOCK_SETUP_COMPLETE" in setup
    output = _child(home, _HELD_LOCK_BACKUPS, "held-lock", timeout=300, selector=selector)
    assert "HELD_LOCK_BACKUPS_COMPLETE" in output

_RESTORE_PRIVATE = r"""
import asyncio,hashlib,json,os,shutil,sqlite3,sys,threading
from pathlib import Path
from tldw_chatbook.Backup_Recovery.storage_admission import admit_startup
admit_startup()
from Tests.network_guard import install
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.Plugins import authority_store
from tldw_chatbook.Plugins.service import PluginService
home=Path.home();root=get_user_data_dir()/'plugins'
trust=get_user_data_dir()/'skills'/'trust'/'plugins'
class FileKeyring:
 # The OS keyring's role: it outlives the profile and no backup holds it.
 def __init__(self,path):self.path=path
 def _values(self):return json.loads(self.path.read_text()) if self.path.exists() else {}
 def get_password(self,service,account):return self._values().get(service+'|'+account)
 def set_password(self,service,account,value):
  values=self._values();values[service+'|'+account]=value;self.path.write_text(json.dumps(values))
 def delete_password(self,service,account):
  values=self._values();del values[service+'|'+account];self.path.write_text(json.dumps(values))
authority_store.is_secure_keyring_backend=lambda candidate:isinstance(candidate,FileKeyring)
def marker(name='keyring.json'):
 return lambda store:authority_store.KeyringPluginMarkerStore(store,keyring_backend=FileKeyring(home/name))
def service():
 return PluginService(get_user_data_dir(),workspace_lookup=lambda _:None,marker_store_factory=marker())
def excluded(path):
 parts=path.relative_to(root).parts
 return parts[0]=='runtime.lock' or parts[0].startswith('registry.sqlite3') or (len(parts)>2 and parts[0]=='staging' and parts[2]!='snapshot')
def state():
 files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and not excluded(p)}
 with sqlite3.connect(f"file:{root/'registry.sqlite3'}?mode=ro",uri=True) as connection:registry=list(connection.iterdump())
 return {'files':files,'registry':registry}
"""

_PLUGIN_SEED = (
    r"""
import json,os
from pathlib import Path
selector=Path(os.environ['TLDW_CONFIG_PATH']);selector.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
data=Path.home()/'data';data.mkdir(mode=0o700,exist_ok=True)
selector.write_text('[general]\nusers_name="default_user"\n[first_run]\nsetup_completed=true\n[paths]\ndata_dir='+json.dumps(str(data))+'\n')
selector.chmod(0o600)
"""
    + _RESTORE_PRIVATE
    + r"""
async def main():
 from Tests.Plugins.acquisition_process_support import repository
 from tldw_chatbook.Plugins.git_source import select_local_source
 from tldw_chatbook.Plugins.source_service import SourceService
 package=home/'package';shutil.copytree(Path('Tests/Plugins/fixtures/native'),package)
 manifest=json.loads((package/'plugin.json').read_text())
 manifest['extensions']['io.github.rmusser01.chatbook']={'version':1}
 (package/'plugin.json').write_text(json.dumps(manifest))
 source=repository(package,home/'git-source')
 plugins=service()
 try:
  await plugins.bootstrap('test passphrase')
  # The Plugins screen's repository path: acquire, then review the acquired bytes.
  destination=await plugins.acquisition_destination('ui-git-install')
  snapshot=await plugins.acquire_snapshot(select_local_source(source),'ui-git-install',destination)
  review=await SourceService(plugins).review_acquired(snapshot,selection=('skill:review',),workspace_id=None)
  await plugins.commit(review,review.operation_id)
  trust_review=await plugins.review_trust(review.installation_id)
  await plugins.commit(trust_review,trust_review.operation_id)
  enable=await plugins.review_activation(review.installation_id,workspace_id=None,intent='enabled')
  await plugins.commit(enable,enable.operation_id)
  assert plugins.capture_maximum('global')['available_skills']
 finally:
  await plugins.aclose()
 return review.installation_id,destination.parent
# The core stores every profile's preview depends on, as the selective-restore tests build them.
from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
_populate_required_dependencies(preview_capture((selector,),options={'allow_partial':True}))
installation,staged=asyncio.run(main())
assert (staged/'snapshot').is_dir() and (staged/'repository.git').is_dir() and (staged/'content').is_dir()
# A remote fetch also leaves copies of the host git/SSH configuration (D1).
for name in ('host-git-config','host-ssh-input','host-ssh-config'):
 (staged/name).write_bytes(b'[credential]\n\thelper = !secret-token\n');(staged/name).chmod(0o600)
data=root/'data'/'kept-root'
for path in (root/'data',data):path.mkdir(mode=0o700,exist_ok=True)
(data/'notes.txt').write_bytes(b'made by the plugin');(data/'notes.txt').chmod(0o600)
(home/'plugin-seed.json').write_text(json.dumps({'installation':installation,'staged':str(staged),**state()}))
print('PLUGIN_SEED_COMPLETE',flush=True)
"""
)

# Only the Tools group (MCP, skills and plugins): whole-profile replacement is
# far slower and adds nothing to the plugin owner's evidence.
_PLUGIN_BACKUP_RESTORE = r"""
import json,os,shutil,sys,threading,time,zipfile
from pathlib import Path
started=time.monotonic()
def phase(name):print('PHASE',name,round(time.monotonic()-started),flush=True)
from Tests.network_guard import install
install()
os.environ['PYTHON_KEYRING_BACKEND']='keyring.backends.null.Keyring'
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import credentials
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
store=KeyringServerCredentialStore(keyring_backend=Keyring());credentials._credential_store=lambda:store
source=Path(os.environ['TLDW_CONFIG_PATH']);home=Path.home();first=sys.argv[1]=='first'
options={'data_groups':('tools',)}
destination=home/'tools.tldw-backup.zip'
def done(service,operation):
 result=service.wait(operation);assert result['state']=='succeeded',dict(result);return result
service=RecoveryService(home/'recovery-control')
try:
 if first:
  preview=service.preview_backup((source,),options=options)
  assert preview.complete,preview.issues
  plugins=[item for item in preview.items if item.owner=='plugins' and item.path is not None]
  root=next(item.path for item in plugins if item.metadata and item.metadata.relative_path=='')
  status={item.path.relative_to(root).as_posix():item.status for item in plugins}
  staged=Path(json.loads((home/'plugin-seed.json').read_text())['staged']).relative_to(root).as_posix()
  assert status['registry.sqlite3']=='included' and status['runtime.lock']=='intentionally_excluded'
  assert status[staged+'/snapshot/plugin.json']=='included'
  assert {status[staged+'/'+name] for name in ('host-git-config','host-ssh-input','host-ssh-config','repository.git','content')}=={'intentionally_excluded'}
  assert not [key for key in status if key.startswith((staged+'/repository.git/',staged+'/content/'))],status
  assert any(key.startswith('acquisition-custody/') and value=='included' for key,value in status.items()),status
  done(service,service.start_backup((source,),preview.scope_digest,destination,options=options,password=None))
  phase('backup')
  # D1: the fetch's host configuration and git working copies never reach the archive.
  with zipfile.ZipFile(destination) as archive:
   manifest=json.loads(archive.read('manifest.json'))
   names={row['relative_path'] for row in manifest['files'] if row['owner_id']=='plugins'}
   assert not [name for name in names if '/host-' in name or '/repository.git' in name or '/content/' in name],names
   assert any(name.endswith('snapshot/plugin.json') for name in names),names
   assert not [row for row in manifest['files'] if b'secret-token' in archive.read(row['payload'])]
 inspection=service.start_inspection(destination,password=None);done(service,inspection)
 profile=service.summary(inspection)['profile_ids'][0]
 phase('inspection')
 if first:
  # AC8: a separate profile cannot take the plugin root's absolute paths.
  isolated=home/'isolated-parent';isolated.mkdir(mode=0o700)
  try:
   service.preview_restore(inspection,mode='isolated',profile_bases={profile:isolated/'restored'},target_configs={},external_destinations={},profile_names={profile:'Restored elsewhere'},target=None,data_groups=('tools',))
  except ValueError as error:
   assert str(error)=='owner_relocation_unverified:plugins',str(error)
  else:raise AssertionError('separate-profile plugin restore was planned')
  phase('isolated refused')
 def restore(**review):
  target=service.preview_backup((source,),options={})
  plan=service.preview_restore(inspection,mode='replace',profile_bases={},target_configs={profile:source},external_destinations={},profile_names={},target=target,data_groups=('tools',),**review)
  return service.wait(service.start_restore(inspection,plan,rollback_password=b'rollback-test-password'))
 if first:
  # M2: same profile, but its data folder moved since the backup. With no
  # registered data roots, the installed/fetched package paths still refuse.
  original=source.read_bytes()
  moved=home/'moved-data';moved.mkdir(mode=0o700)
  source.write_text(original.decode().replace(json.dumps(str(home/'data')),json.dumps(str(moved))))
  # A plain copy of the other stores: opening them here would hold leases.
  shutil.copytree(home/'data'/'default_user',moved/'default_user',ignore=shutil.ignore_patterns('plugins'))
  try:result=restore()
  except ValueError as error:assert str(error).startswith('owner_relocation_unverified'),str(error)
  else:assert result['state']!='succeeded' and 'owner_relocation_unverified' in result['issues'],dict(result)
  assert not (moved/'default_user'/'plugins'/'registry.sqlite3').exists()
  source.write_bytes(original)
  phase('moved data refused')
  # I-1 (owner decision 2026-11-12, refuse early): a plugin fetched since the
  # backup leaves a fetch folder the archive lacks. While it holds members no
  # backup captures, the review refuses with the plugin reason, before Start.
  plugin_root=next(item.path for item in plugins if item.metadata and item.metadata.relative_path=='')
  added=plugin_root/'staging'/'zz-new'
  for name,payload in (('snapshot/plugin.json',b'{}'),('repository.git/HEAD',b'ref'),('host-git-config',b'x')):
   (added/name).parent.mkdir(parents=True,exist_ok=True,mode=0o700);(added/name).write_bytes(payload);(added/name).chmod(0o600)
  try:restore()
  except ValueError as error:
   assert service.issue_code(error)=='plugin_fetch_added_since_backup',str(error)
  else:raise AssertionError('restore planned over a fetch folder added since the backup')
  # The copy's step: keep only snapshot/ there. The restore then retires it.
  shutil.rmtree(added/'repository.git');(added/'host-git-config').unlink()
  phase('newer fetch refused')
  next(item.path for item in preview.items if item.owner=='plugins' and item.path is not None and item.path.name=='notes.txt').write_bytes(b'changed after the backup')
 # The first attempt stops at the rollback snapshot's credential review (about
 # 10 s) for the restored skill trust; the user acknowledges and restores again.
 result=restore()
 phase('restore '+result['state'])
 if result['state']=='recovery_required' and result.get('review_issues'):
  issues=tuple(result['review_issues'])
  phase('credential review of %d skill trust items'%len(issues))
  assert all(code.startswith('credential_') for code in issues),issues
  done(service,service.start_recovery(result['result']['journal_operation_id'],action='abort'))
  phase('aborted')
  result=restore(acknowledged_credential_issues=issues)
 assert result['state']=='succeeded',dict(result)
 if first:assert not added.exists(),'the newer fetch folder was not retired'
 phase('restored')
finally:
 service.close()
print('PLUGIN_REPLACEMENT_COMPLETE',flush=True)
"""

_PLUGIN_REOPEN = _RESTORE_PRIVATE + r"""
from tldw_chatbook.Plugins.backup import restore_review_required
from tldw_chatbook.Plugins.admission import PluginUnavailable
from tldw_chatbook.Widgets.plugin_review_modal import authority_failure_message
seed=json.loads((home/'plugin-seed.json').read_text());unchanged=sys.argv[1]=='unchanged'
# AC5: registry rows, package bytes and plugin data are back, byte for byte.
assert state()==({key:seed[key] for key in ('files','registry')}),'restored plugin root differs'
assert (root/'data'/'kept-root'/'notes.txt').read_bytes()==b'made by the plugin'
# AC5: without its keyring marker the restored authority asks for recovery.
assert authority_store.PluginAuthorityStore(trust,marker('other-computer.json')(trust)).posture()=='recovery_required'
# AC7: the restored generation keeps plugins from running until recovery.
assert restore_review_required(root) is True
async def main():
 plugins=service()
 try:
  async def capture():
   return await plugins._call(lambda:plugins._admission.capture(seed['installation'],None,'restored-run'))
  async def refused(reason):
   try:await capture()
   except PluginUnavailable as error:assert reason in {None,str(error)},str(error)
   else:raise AssertionError('restored plugin ran before authority recovery')
  # AC7: the restore gate refuses first, whatever the authority's state.
  await refused('plugin_restore_review_required')
  posture=await plugins.authority_status()
  assert posture=='locked',posture
  if not unchanged:
   # D2: plugins changed after the backup. Unlock cannot verify the marker's
   # authority files, and the copy says so without a dead-end step.
   try:await plugins.unlock('test passphrase')
   except FileNotFoundError as error:message=authority_failure_message(error,'unlock')
   else:raise AssertionError('mismatched restored authority unlocked')
   assert 'cannot recover these plugins yet' in message and 'stay on disk' in message,message
   await refused('plugin_restore_review_required')
   return
  # M5: the authority dialog's own call. Unlocking recovers and approves.
  await plugins.unlock('test passphrase')
  assert await plugins.authority_status()=='ready'
  assert restore_review_required(root) is False
  # The git-installed plugin re-verifies its retained fetch after the restore.
  assert (await capture()).installation_id==seed['installation']
  assert plugins.capture_maximum('global')['available_skills']
  # Change plugins after the backup, for the next restore of the same archive.
  disable=await plugins.review_activation(seed['installation'],workspace_id=None,intent='disabled')
  await plugins.commit(disable,disable.operation_id)
 finally:
  await plugins.aclose()
if unchanged:
 # M4 (D1): recovery needs only snapshot/ and the custody record. Whatever else
 # the fetch folder holds here is not from the archive; remove it all first.
 staged=Path(seed['staged'])
 for path in staged.iterdir():
  if path.name!='snapshot':
   print('REMOVED',path.name,flush=True)
   shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
 assert [path.name for path in staged.iterdir()]==['snapshot']
asyncio.run(main())
assert (root/'data'/'kept-root'/'notes.txt').read_bytes()==b'made by the plugin'
print('PLUGIN_RESTORE_REVIEW_COMPLETE',flush=True)
"""


@platform_support.managed_runtime
# M12: the macOS lane's pytest step already takes ~19 of its 30 minutes, so a hang
# here must fail (with JUnit output, signal method) within 8 minutes. Measured on
# a shared 18-core host: 4.0 min at load 15-20, 5.4-6.9 min at load 25-40. Two
# Tools-group restores take 1.7-3.4 min each (mostly the restore journal's
# per-record revalidation); everything else is under 30 s.
@pytest.mark.timeout(480)
def test_plugin_backup_restores_in_place_and_waits_for_authority_recovery(tmp_path):
    """AC5/AC7/AC8/D1/D2 through the app's repository install path, actual
    capture, archive and same-profile replacement, twice: with plugins unchanged
    since the backup (Unlock recovers) and after a later plugin change (truthful
    refusal, plugins stay off, data kept)."""
    from Tests.Backup_Recovery.test_config_sibling_capture import _child

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    # A selector outside the default config folder, which holds recovery control,
    # as the selective-restore tests use: first binding proves its own container.
    selector = tmp_path / "profile-config" / "config.toml"

    durations = {}

    def run(script, *arguments, timeout):
        started = time.monotonic()
        try:
            output = _child(home, script, *arguments, timeout=timeout, selector=selector)
        finally:
            durations["-".join(arguments)] = round(time.monotonic() - started)
            print("child seconds:", durations, flush=True)
        print(*(line for line in output.splitlines() if line.startswith("PHASE")), sep="\n")
        return output

    # A child's limit names the child that hung; the test's limit bounds the sum.
    assert "PLUGIN_SEED_COMPLETE" in run(_PLUGIN_SEED, "plugins", timeout=60)
    assert "PLUGIN_REPLACEMENT_COMPLETE" in run(_PLUGIN_BACKUP_RESTORE, "first", timeout=360)
    unchanged = run(_PLUGIN_REOPEN, "unchanged", timeout=60)
    assert "PLUGIN_RESTORE_REVIEW_COMPLETE" in unchanged
    # M4: the excluded fetch members survived the in-place restore (preserved,
    # not archived) and were deleted before the plugin re-verified and ran.
    for name in ("repository.git", "content", "host-git-config", "host-ssh-input"):
        assert f"REMOVED {name}\n" in unchanged, name
    assert "PLUGIN_REPLACEMENT_COMPLETE" in run(_PLUGIN_BACKUP_RESTORE, "again", timeout=300)
    assert "PLUGIN_RESTORE_REVIEW_COMPLETE" in run(_PLUGIN_REOPEN, "changed", timeout=60)
