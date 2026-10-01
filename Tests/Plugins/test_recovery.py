"""Recovery requires protected proof and exact retained definitions."""

import shutil

import pytest

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
        stack.call(lambda: stack.coordinator.commit(review, "interrupted"))
    stack.call(lambda: setattr(stack.coordinator, "progress", None))
    receipt = next(
        item
        for item in stack.call(stack.coordinator.recover)
        if item.operation_id == "interrupted"
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
    stack.call(lambda: stack.coordinator.commit(first, "first"))
    second = reviewed(stack, native_package())
    interrupt_at(stack, phase)
    with pytest.raises(OSError):
        stack.call(lambda: stack.coordinator.commit(second, "second"))
    lose_registry(stack)
    receipts = stack.call(stack.coordinator.recover)
    assert any(item.operation_id == "second" and item.committed for item in receipts)
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
    stack.call(lambda: stack.coordinator.commit(review, "linked"))
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
    stack.call(lambda: stack.coordinator.commit(review, "changed"))
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
        stack.call(lambda: stack.coordinator.commit(review, "invalid"))
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
    run_child(root, "install", "--package", str(package), barrier=barrier)
    result = run_child(root, "recover")
    if want is None:
        assert result["receipts"] == []
    else:
        receipt = next(
            item
            for item in result["receipts"]
            if item["operation_id"] == "child-install"
        )
        assert receipt["phase"] == want and receipt["committed"] is committed
    assert result["marker_generation"] == int(committed)
    if committed:
        again = run_child(root, "recover")
        assert again["receipts"] == result["receipts"]
        assert len(again["installations"]) == 1


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
    stack.call(lambda: stack.coordinator.commit(first, "first"))
    second = reviewed(stack, native_package())
    interrupt_at(stack, "registry_committed")
    with pytest.raises(OSError):
        stack.call(lambda: stack.coordinator.commit(second, "no-proof"))

    def remove_hint():
        with stack.registry.transaction() as cursor:
            cursor.execute("DELETE FROM operations WHERE operation_id='no-proof'")

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
    stack.call(lambda: stack.coordinator.commit(review, "marker-only"))
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
        stack.call(lambda: stack.coordinator.commit(review, "one"))

    def add_competing():
        evidence = stack.authority.verify_transition("one")
        snapshot = evidence.snapshot
        snapshot["operation_result"]["operation_id"] = "two"
        new = PluginMarker(
            generation=1,
            operation_id="two",
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
                review, f"install-{index}"
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
    stack.call(lambda: stack.coordinator.commit(review, "first"))
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
    stack.call(lambda: stack.coordinator.commit(review, "blocked"))
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
    stack.call(lambda: stack.coordinator.commit(review, "first"))
    original = stack.call(stack.authority.verify_current)
    altered = deepcopy(original)
    altered["components"][0]["definition_digest"] = "0" * 64
    with pytest.raises(ValueError, match="constraints"):
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
        stack.call(lambda: stack.coordinator.commit(review, "aborted"))
    stack.call(lambda: setattr(stack.coordinator, "progress", None))
    assert stack.call(lambda: stack.coordinator.commit(review, "new-attempt")).committed
    receipts = stack.call(stack.coordinator.recover)
    assert [(item.operation_id, item.phase) for item in receipts] == [
        ("aborted", "aborted"),
        ("new-attempt", "complete"),
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
                review, f"accepted-{generation}"
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
                review, f"accepted-{generation}"
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
