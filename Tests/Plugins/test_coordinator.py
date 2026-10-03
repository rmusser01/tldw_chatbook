"""Reviewed commit evidence must survive real storage boundaries."""


def test_registry_flag_cannot_replace_commit_certificate():
    from tldw_chatbook.Plugins.recovery import recovery_action

    assert (
        recovery_action(
            registry_new=True,
            marker_new=False,
            certificate_valid=False,
            snapshot_valid=True,
        )
        == "review_required"
    )
    assert (
        recovery_action(
            registry_new=True,
            marker_new=False,
            certificate_valid=True,
            snapshot_valid=True,
        )
        == "complete_marker"
    )


import pytest


def reviewed(stack, package, *, workspace=None):
    from tldw_chatbook.Plugins.inspection import inspect_package

    return stack.call(
        lambda: stack.coordinator.review(
            inspect_package(package),
            selection=("skill:review",),
            workspace_id=workspace,
        )
    )


def test_install_is_disabled_exact_retry_does_not_duplicate(
    plugin_stack, native_package
):
    stack = plugin_stack
    review = reviewed(stack, native_package(), workspace="workspace-a")
    receipt = stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    assert receipt.committed and receipt.phase == "complete"
    assert (
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
        == receipt
    )
    state = stack.call(stack.authority.verify_current)
    assert len(state["installations"]) == 1
    assert state["installations"][0]["activation_default"] is False
    assert state["activation"] == []
    assert state["revision_trust"][0]["reviewed"] is False
    assert [row["component_id"] for row in state["selections"] if row["selected"]] == [
        "skill:review"
    ]
    assert stack.call(lambda: stack.coordinator.published_snapshot()) == state


def test_stale_package_review_cannot_mutate(plugin_stack, native_package):
    stack = plugin_stack
    package = native_package()
    review = reviewed(stack, package)
    (package / "skills" / "review" / "SKILL.md").write_text("changed")
    with pytest.raises(ValueError, match="stale"):
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    assert stack.call(stack.authority.verify_current)["installations"] == []


def test_review_binds_authority_and_cannot_be_forged(plugin_stack, native_package):
    from dataclasses import replace

    stack = plugin_stack
    review = reviewed(stack, native_package())
    with pytest.raises(ValueError, match="review"):
        stack.call(
            lambda: stack.coordinator.commit(
                replace(review, workspace_id="other"), review.operation_id
            )
        )
    other = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(other, other.operation_id))
    with pytest.raises(ValueError, match="stale"):
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))


def test_publication_stays_fenced_until_marker_and_registry_agree(
    plugin_stack, native_package
):
    stack = plugin_stack
    review = reviewed(stack, native_package())
    observed = []

    def progress(phase):
        observed.append(phase)
        if phase != "published":
            with pytest.raises(PermissionError):
                stack.coordinator.published_snapshot()
        if phase == "registry_committed":
            assert (
                stack.registry.read_operation(review.operation_id)["phase"]
                == "committed"
            )
            assert not stack.authority.verify_transition(review.operation_id).committed
            assert stack.authority.load_marker().generation == 0
        if phase == "certified":
            assert stack.authority.verify_transition(review.operation_id).committed
            assert stack.authority.load_marker().generation == 0

    stack.call(lambda: setattr(stack.coordinator, "progress", progress))
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    assert observed == [
        "materialized",
        "prepared",
        "registry_committed",
        "certified",
        "marker_advanced",
        "published",
    ]


def test_failed_registry_commit_never_certifies(
    plugin_stack, native_package, monkeypatch
):
    import sqlite3
    from contextlib import contextmanager

    stack = plugin_stack
    review = reviewed(stack, native_package())
    original = stack.registry.transaction

    @contextmanager
    def fail_commit():
        with original() as cursor:
            yield cursor
            raise sqlite3.OperationalError("injected disk full before COMMIT")

    monkeypatch.setattr(stack.registry, "transaction", fail_commit)
    with pytest.raises(sqlite3.OperationalError):
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    assert not stack.call(
        lambda: stack.authority.verify_transition(review.operation_id)
    ).committed
    assert stack.call(stack.authority.load_marker).generation == 0
    monkeypatch.setattr(stack.registry, "transaction", original)
    assert stack.call(stack.coordinator.recover)[0].phase == "aborted"
    assert (
        stack.call(lambda: stack.coordinator.published_snapshot())["installations"]
        == []
    )


def test_same_id_for_different_review_is_refused(plugin_stack, native_package):
    stack = plugin_stack
    first = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(first, first.operation_id))
    second = reviewed(stack, native_package())
    with pytest.raises(ValueError, match="operation"):
        stack.call(lambda: stack.coordinator.commit(second, first.operation_id))


def test_wrong_thread_and_main_thread_mutations_are_refused(plugin_stack):
    import asyncio

    with pytest.raises(RuntimeError, match="worker"):
        asyncio.run(plugin_stack.coordinator.recover())


def test_same_operation_retry_rejects_changed_review_fields(
    plugin_stack, native_package
):
    from dataclasses import replace

    stack = plugin_stack
    review = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    with pytest.raises(ValueError, match="review"):
        stack.call(
            lambda: stack.coordinator.commit(
                replace(review, workspace_id="retargeted"), review.operation_id
            )
        )


def test_full_disk_preserves_old_marker_and_disabled_projection(
    plugin_stack, native_package, monkeypatch
):
    from collections import namedtuple

    from tldw_chatbook.Plugins import coordinator

    stack = plugin_stack
    review = reviewed(stack, native_package())
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        coordinator.shutil, "disk_usage", lambda path: usage(1000, 1000, 0)
    )
    with pytest.raises(OSError, match="reserve"):
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    assert stack.call(stack.authority.load_marker).generation == 0
    stack.call(stack.coordinator.recover)
    assert stack.call(stack.coordinator.published_snapshot)["installations"] == []


def test_review_expiry_is_enforced_before_staging(
    plugin_stack, native_package, monkeypatch
):
    from tldw_chatbook.Plugins import coordinator

    stack = plugin_stack
    review = reviewed(stack, native_package())
    monkeypatch.setattr(coordinator.time, "monotonic", lambda: review.expires_at + 1)
    with pytest.raises(ValueError, match="expired"):
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    assert not (stack.owner.root / "packages").exists()


def test_retry_after_materialized_response_failure_reuses_exact_owned_bytes(
    plugin_stack, native_package
):
    stack = plugin_stack
    review = reviewed(stack, native_package())

    def interrupt(phase):
        if phase == "materialized":
            raise OSError("lost materialization response")

    stack.call(lambda: setattr(stack.coordinator, "progress", interrupt))
    with pytest.raises(OSError):
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    stack.call(lambda: setattr(stack.coordinator, "progress", None))
    assert stack.call(
        lambda: stack.coordinator.commit(review, review.operation_id)
    ).committed
    assert len(list((stack.owner.root / "packages").iterdir())) == 1


def test_projection_cannot_activate_itself(plugin_stack, native_package):
    stack = plugin_stack
    review = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))

    def modify_projection():
        with stack.registry.transaction() as cursor:
            cursor.execute("UPDATE installations SET activation_default=1")

    stack.call(modify_projection)
    with pytest.raises(PermissionError):
        stack.call(stack.coordinator.published_snapshot)
    assert (
        stack.call(stack.authority.verify_current)["installations"][0][
            "activation_default"
        ]
        is False
    )


def test_owner_loss_prevents_bootstrap_reset_and_commit(plugin_stack, native_package):
    stack = plugin_stack
    review = reviewed(stack, native_package())
    stack.call(stack.owner.close)
    for action in (
        lambda: stack.coordinator.bootstrap("pw"),
        lambda: stack.coordinator.reset(operation_id="reset"),
        lambda: stack.coordinator.commit(review, review.operation_id),
    ):
        with pytest.raises(PermissionError):
            stack.call(action)


def test_guarded_transaction_cannot_be_committed_by_progress_callback(
    plugin_stack, native_package
):
    import sqlite3

    stack = plugin_stack
    review = reviewed(stack, native_package())

    def progress(phase):
        if phase == "prepared":
            # The coordinator must preserve the actual F2 transaction authorizer.
            with pytest.raises(sqlite3.DatabaseError):
                stack.registry._connection.commit()

    stack.call(lambda: setattr(stack.coordinator, "progress", progress))
    assert stack.call(
        lambda: stack.coordinator.commit(review, review.operation_id)
    ).committed


def test_same_operation_retry_uses_secure_marker_when_journal_is_missing(
    plugin_stack, native_package
):
    stack = plugin_stack
    review = reviewed(stack, native_package())
    receipt = stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    for folder in ("intents", "certificates"):
        for path in (stack.authority.store_dir / folder).iterdir():
            path.unlink()
    assert (
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
        == receipt
    )
    assert (
        len(stack.call(lambda: stack.registry.list_installations(limit=50, offset=0)))
        == 1
    )


def test_review_issues_exact_mutation_identity(plugin_stack, native_package):
    from dataclasses import replace

    from tldw_chatbook.Plugins.inspection import inspect_package

    stack = plugin_stack
    review = stack.call(
        lambda: stack.coordinator.review(
            inspect_package(native_package()),
            selection=("skill:review",),
            workspace_id=None,
        )
    )
    assert review.operation_id.startswith("pi1.")
    with pytest.raises(ValueError):
        stack.call(lambda: stack.coordinator.commit(review, "caller-supplied"))
    receipt = stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    assert receipt.committed
    assert (
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
        == receipt
    )
    with pytest.raises(ValueError):
        stack.call(
            lambda: stack.coordinator.commit(
                replace(review, selection=()), review.operation_id
            )
        )
