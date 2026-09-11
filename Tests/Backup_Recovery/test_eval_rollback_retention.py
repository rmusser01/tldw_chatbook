"""Actual rolled-back generations retain independently captured original YAML."""

import shutil
from dataclasses import replace
from threading import Event

import pytest

from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from tldw_chatbook.Backup_Recovery import bootstrap, crypto, publication, replacement
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.models import (
    DISCOVERY_CONTEXT_KEY,
    DiscoveryContext,
    Inventory,
    StorageItem,
)
from tldw_chatbook.Evals.recovery import _DefinitionsAdapter


@pytest.fixture
def rolled_back_eval(tmp_path, monkeypatch, helper_resource_root, request):
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, _, _, _, selector = case
        covered = getattr(request, "param", True)
        selected = selector.parent / "retained-eval.yaml"
        selected.write_bytes(b"tasks:\n  original: true\n")
        selected.chmod(0o600)
        item = StorageItem(
            "eval.definitions",
            "profile:profile:eval.definitions",
            selected,
            "included",
            ("profile:profile:config",),
        )
        target = replace(plan.target, items=(*plan.target.items, item))
        source = replacement._acquired_source(candidate, plan, Event())
        plan = plan_restore(
            source,
            mode="replace",
            destinations={**dict(plan.destinations), **dict(plan.selectors)},
            target=target,
            profile_names=dict(plan.profile_names),
            safety_scope=(item.logical_id,) if covered else (),
            acknowledged_credential_issues=("credential_format_unreadable",),
        )
        candidate = stage_restore(source, plan, tmp_path / "eval-candidate", Event())
        control = tmp_path / "control"

        def stopped(*args, **kwargs):
            raise KeyboardInterrupt("actual pre-finalization interruption")

        with monkeypatch.context() as fault:
            fault.setattr(publication, "finalize_candidate", stopped)
            with pytest.raises(KeyboardInterrupt):
                replacement.replace(
                    plan,
                    candidate,
                    control_root=control,
                    rollback_password=b"old",
                    cancel=Event(),
                )
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        operation = pending[0]["operation_id"]
        assert (
            replacement.recover_replacement(
                operation,
                control_root=control,
                action="rollback",
                rollback_password=b"old",
                cancel=Event(),
            )
            == "rolled_back"
        )
        journal = Journal(control, operation)
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        assert rows[-1].event == "rolled_back"
        shutil.rmtree(candidate)
        assert not candidate.exists()
        context = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "originals")}
        yield journal, selected, selector, context


@pytest.mark.parametrize("rolled_back_eval", [False], indirect=True)
def test_known_original_without_rollback_coverage_refuses(rolled_back_eval):
    _, selected, _, context = rolled_back_eval
    assert selected.read_bytes() == b"tasks:\n  original: true\n"
    with pytest.raises(ValueError, match="eval_retained_originals_unverified"):
        _DefinitionsAdapter().discover(context)


def test_original_eval_retained_and_captured_without_incoming_owner_or_candidate(
    rolled_back_eval, tmp_path
):
    from tldw_chatbook.Backup_Recovery.activation import activation_permission
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Evals import _default_config_path

    journal, selected, selector, context = rolled_back_eval
    adapter = _DefinitionsAdapter()
    items = adapter.discover(context)
    assert {item.path for item in items} == {_default_config_path(), selected}
    assert all(item.status == "included" for item in items)
    assert selected.read_bytes() == b"tasks:\n  original: true\n"
    # Incoming manifest has no eval owner. Only the independently captured
    # original target/safety proof may supply this retained source.
    assert (
        b'"eval.definitions"'
        not in (journal.root / "verified-manifest.json").read_bytes()
    )
    assert not activation_permission("eval.definitions", config_selector=selector)
    authority = admission_authority(tmp_path / "bootstrap")
    names = _capture_names(authority, Inventory(items, True, "owned-component", ()))
    stage = tmp_path / "recaptured"
    stage.mkdir(mode=0o700)
    item = next(row for row in items if row.path == selected)
    with (
        authority.maintenance(names, 3) as session,
        session.capture_scope((selected,), stage),
    ):
        adapter.capture(item, stage / "retained.yaml", Event())
    assert (stage / "retained.yaml").read_bytes() == b"tasks:\n  original: true\n"
    assert not activation_permission("eval.definitions", config_selector=selector)


@pytest.mark.parametrize(
    "change", ["plan", "terminal", "association", "missing", "alias", "unrelated"]
)
def test_rolled_back_eval_requires_current_provenance_and_exact_path(
    rolled_back_eval, tmp_path, change
):
    journal, selected, _, context = rolled_back_eval
    if change == "unrelated":
        other = selected.with_name("eval_config.yaml")
        other.write_bytes(selected.read_bytes())
        assert other not in {
            row.path for row in _DefinitionsAdapter().discover(context)
        }
        return
    if change in {"missing", "alias"}:
        saved = selected.with_name("saved-original")
        selected.rename(saved)
        if change == "alias":
            selected.symlink_to(saved)
            with pytest.raises(ValueError):
                _DefinitionsAdapter().discover(context)
        else:
            rows = _DefinitionsAdapter().discover(context)
            assert (
                next(row for row in rows if row.path == selected).status
                == "missing_required"
            )
        return
    if change == "plan":
        path = journal.root / "restore-plan.json"
    elif change == "terminal":
        path = max(journal.root.glob("[0-9]*.json"))
    else:
        path = next((tmp_path / "bootstrap").glob("activation-*.json"))
    if change == "terminal":
        path.rename(tmp_path / "removed-terminal")
    else:
        path.write_bytes(b"{}")
    with pytest.raises((ValueError, RuntimeError)):
        _DefinitionsAdapter().discover(context)
