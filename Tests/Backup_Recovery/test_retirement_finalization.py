"""Finalization must retain fresh absence proof after native retirement."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
from keyring.backends.null import Keyring

from Tests.Backup_Recovery.test_retained_config import (  # noqa: F401
    retained_case,
    retained_plan,
)
from tldw_chatbook.Backup_Recovery import credentials, crypto, publication, replacement
from tldw_chatbook.Backup_Recovery.bootstrap import _key, startup_permission
from tldw_chatbook.Backup_Recovery.control_records import (
    UNBOUND_NAMESPACE,
    register_pending,
)
from tldw_chatbook.Backup_Recovery.journal import Journal, _Prepared
from tldw_chatbook.Backup_Recovery.restore_plan import _fingerprint, _paths
from tldw_chatbook.Backup_Recovery.staging import stage_restore
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore


@contextmanager
def retired(case, tmp_path, monkeypatch, helper_resource_root):
    """Exercise the real executor independently of later-rollback eligibility."""
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    store = KeyringServerCredentialStore(keyring_backend=Keyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    archive, target, selector, state, authority = case
    original = retained_plan(case)
    # Bind an executor-level retirement plan to real locally observed state.
    # The separate later-rollback tests own authenticated absence eligibility.
    plan = replace(
        original,
        restore=(),
        retire=(
            (
                next(item.logical_id for item in target.items if item.path == state),
                state,
            ),
        ),
        metadata=(),
    )
    plan = replace(plan, target_fingerprint=_fingerprint(_paths(plan), target))
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    journal = Journal(control, "retirement-finalization")
    bootstrap = tmp_path / "bootstrap"
    register_pending(bootstrap, journal.operation_id, ("local",), control, (selector,))
    with authority.maintenance(("local", UNBOUND_NAMESPACE), 3) as session:
        candidate = stage_restore(
            archive, plan, tmp_path / "stage", Event(), journal=journal, session=session
        )
        journal.prepare_publication(
            candidate,
            plan,
            bootstrap_root=bootstrap,
            namespaces=("local",),
            selectors=(selector,),
            generation="retirement",
        )
        rollback = replacement.capture_verify_rollback(
            candidate,
            plan,
            journal,
            tmp_path / "rollback.tldw-backup.zip.age",
            session=session,
            password=b"retirement-test",
            work_root=tmp_path / "rollback-work",
            cancel=Event(),
            acknowledged_credential_issues=(),
        )
        publication.publish_candidate(
            candidate, plan, journal, rollback, session=session
        )
        with journal._locked(exclusive=False) as parent:
            prepared = _Prepared.model_validate(
                next(
                    row.evidence
                    for row in journal._records(parent)
                    if row.event == "prepared"
                )
            )
        assert prepared.installed_paths == []
        assert len(prepared.artifacts) == 1
        assert prepared.artifacts[0].action == "retire"
        assert not state.exists()
        yield candidate, plan, journal, session, bootstrap, selector, prepared


@pytest.mark.parametrize(
    "boundary,damage,error",
    [
        ("validation", "target", "installed_objects_changed"),
        ("validation", "retained", "installed_objects_changed"),
        ("validation", "parent", "publication_parent_changed"),
        ("activation_recorded", "target", "installed_objects_changed"),
        ("committed", "target", "installed_objects_changed"),
        ("pending", "target", "installed_objects_changed"),
    ],
)
def test_retirement_drift_during_finalization_keeps_pending_fence(
    retained_case,  # noqa: F811 - imported pytest fixture
    tmp_path,
    monkeypatch,
    helper_resource_root,
    boundary,
    damage,
    error,
):
    with retired(retained_case, tmp_path, monkeypatch, helper_resource_root) as case:
        candidate, plan, journal, session, bootstrap, selector, prepared = case
        injected = []

        def mutate():
            if injected:
                return
            artifact = prepared.artifacts[0]
            if damage == "parent":
                path = Path(artifact.retained).parent
                moved = path.with_name(path.name + "-moved")
                path.rename(moved)
                path.mkdir(mode=0o700)
                for child in moved.iterdir():
                    child.rename(path / child.name)
            else:
                path = Path(
                    artifact.target if damage == "target" else artifact.retained
                )
                path.write_bytes(b'[state]\nselected="unreviewed return"\n')
                path.chmod(0o600)
            injected.append(path)

        if boundary == "validation":
            validate = publication._validate_installed

            def after_validation(*args, **kwargs):
                proof = validate(*args, **kwargs)
                assert proof["artifacts"] == []
                mutate()
                return proof

            monkeypatch.setattr(publication, "_validate_installed", after_validation)
        elif boundary == "pending":
            pending = publication._pending

            def after_pending(*args, **kwargs):
                result = pending(*args, **kwargs)
                if kwargs.get("durable") and kwargs.get("committed") is not None:
                    mutate()
                return result

            monkeypatch.setattr(publication, "_pending", after_pending)
        else:
            append = journal._append

            def after_event(parent, event, evidence):
                result = append(parent, event, evidence)
                if event == boundary:
                    mutate()
                return result

            monkeypatch.setattr(journal, "_append", after_event)

        with pytest.raises(ValueError, match=error):
            publication.finalize_candidate(candidate, plan, journal, session=session)
        assert injected
        assert (
            bootstrap / ("pending-" + _key(journal.operation_id) + ".json")
        ).exists()
        assert not startup_permission(selector, bootstrap)[0]
        with journal._locked(exclusive=False) as parent:
            committed = "committed" in {row.event for row in journal._records(parent)}
            assert committed is (boundary in {"committed", "pending"})


def test_unchanged_retirement_finalization_is_idempotent(
    retained_case,  # noqa: F811 - imported pytest fixture
    tmp_path,
    monkeypatch,
    helper_resource_root,
):
    with retired(retained_case, tmp_path, monkeypatch, helper_resource_root) as case:
        candidate, plan, journal, session, bootstrap, selector, _ = case
        for _ in range(2):
            assert (
                publication.finalize_candidate(
                    candidate, plan, journal, session=session
                )
                == "retirement"
            )
        assert startup_permission(selector, bootstrap)[0]


def test_committed_retirement_retry_rechecks_absence_before_clearing_fence(
    retained_case,  # noqa: F811 - imported pytest fixture
    tmp_path,
    monkeypatch,
    helper_resource_root,
):
    with retired(retained_case, tmp_path, monkeypatch, helper_resource_root) as case:
        candidate, plan, journal, session, bootstrap, selector, prepared = case
        pending = publication._pending

        def interrupt_before_clear(*args, **kwargs):
            result = pending(*args, **kwargs)
            if kwargs.get("durable") and kwargs.get("committed") is not None:
                raise OSError("interrupted before pending clear")
            return result

        monkeypatch.setattr(publication, "_pending", interrupt_before_clear)
        with pytest.raises(OSError, match="interrupted before pending clear"):
            publication.finalize_candidate(candidate, plan, journal, session=session)
        monkeypatch.setattr(publication, "_pending", pending)
        with journal._locked(exclusive=False) as parent:
            assert journal._records(parent)[-1].event == "committed"
        validate = publication._validate_installed

        def recreate_after_validation(*args, **kwargs):
            proof = validate(*args, **kwargs)
            assert proof["artifacts"] == []
            Path(prepared.artifacts[0].target).write_bytes(b"returned during retry")
            return proof

        monkeypatch.setattr(
            publication, "_validate_installed", recreate_after_validation
        )
        with pytest.raises(ValueError, match="installed_objects_changed"):
            publication.finalize_candidate(candidate, plan, journal, session=session)
        assert (
            bootstrap / ("pending-" + _key(journal.operation_id) + ".json")
        ).exists()
        assert not startup_permission(selector, bootstrap)[0]
