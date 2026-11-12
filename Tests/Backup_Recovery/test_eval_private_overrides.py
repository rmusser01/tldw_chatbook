"""Shipped Eval defaults and actual private override file lifetimes."""

import copy

import pytest
import yaml

from Tests.Backup_Recovery.test_participant_lifetimes import local_root as _local_root
from tldw_chatbook.Backup_Recovery import storage_admission as storage

local_root = _local_root


@pytest.fixture
def eval_profile(tmp_path, monkeypatch, local_root):
    from tldw_chatbook import Evals

    selector = tmp_path / "config.toml"
    selector.write_text("", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    defaults = {
        "task_types": ["question_answer", "generation"],
        "budget": {"default_limit": 10, "warning_threshold": 0.8},
        "error_handling": {"max_retries": 3},
        "optional_fields": {"task": {"max_samples": 42}},
    }
    resource = tmp_path / "shipped.yaml"

    def publish(values):
        resource.write_text(yaml.safe_dump(values), encoding="utf-8")

    publish(defaults)
    monkeypatch.setattr(Evals, "_default_config_path", lambda: resource)
    return selector, selector.parent / "eval_overrides.yaml", defaults, publish


def test_default_loader_reads_complete_shipped_resource_from_checkout(
    tmp_path, monkeypatch, local_root
):
    from tldw_chatbook.Evals import _default_config_path
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    shipped = yaml.safe_load(_default_config_path().read_text(encoding="utf-8"))
    source = EvalConfigLoader()
    assert source.get_task_types() == shipped["task_types"]
    assert source.persistence_error is None
    assert source.config_path == tmp_path / "eval_overrides.yaml"


def test_missing_override_is_clean_and_read_does_not_create_it(eval_profile):
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    _, path, defaults, _ = eval_profile
    source = EvalConfigLoader()
    assert source.get("budget") == defaults["budget"]
    assert source.persistence_safe_point() == "ready"
    assert not path.exists()


def test_overrides_inherit_new_defaults_and_replace_lists(eval_profile):
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    _, path, defaults, publish = eval_profile
    path.write_text(
        "budget: {default_limit: 20}\ntask_types: [custom]\n", encoding="utf-8"
    )
    source = EvalConfigLoader()
    assert source.get("budget.warning_threshold") == 0.8
    assert source.get_task_types() == ["custom"]
    updated = copy.deepcopy(defaults)
    updated["budget"]["default_limit"] = 15
    updated["error_handling"]["max_retries"] = 4
    publish(updated)
    source.reload()
    assert source.get("budget.default_limit") == 20
    assert source.get("error_handling.max_retries") == 4


def test_mutable_draft_save_is_sparse_and_preserves_null(eval_profile):
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    _, path, _, _ = eval_profile
    source = EvalConfigLoader()
    source.get("budget")["default_limit"] = 20
    source.update({"optional_fields": {"task": {"max_samples": None}}})
    source.save()
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "budget": {"default_limit": 20},
        "optional_fields": {"task": {"max_samples": None}},
    }
    assert source.persistence_safe_point() == "ready"


def test_explicit_default_value_is_dirty_then_remains_pinned(eval_profile):
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    _, path, defaults, publish = eval_profile
    source = EvalConfigLoader()
    source.update({"budget": {"default_limit": 10}})
    assert source.persistence_safe_point() == "needs_user_save_or_discard"
    source.save()
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "budget": {"default_limit": 10}
    }
    updated = copy.deepcopy(defaults)
    updated["budget"]["default_limit"] = 15
    publish(updated)
    source.reload()
    assert source.get("budget.default_limit") == 10


def test_failed_save_keeps_draft_and_persisted_overrides(eval_profile):
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    _, path, _, _ = eval_profile
    source = EvalConfigLoader()
    source.update({"budget": {"default_limit": 20}})
    pause = storage._begin_local_pause()
    try:
        source.save()
        assert source.persistence_error == "eval_save_failed"
        assert source.get("budget.default_limit") == 20
        assert not path.exists()
    finally:
        pause.resume()
    source.save()
    assert source.persistence_safe_point() == "ready"


def test_export_contains_effective_values_and_does_not_save_local_draft(eval_profile):
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    selector, path, _, _ = eval_profile
    source = EvalConfigLoader()
    source.update({"budget": {"default_limit": 20}})
    exported = selector.parent / "export.yaml"
    source.save(str(exported))
    data = yaml.safe_load(exported.read_text(encoding="utf-8"))
    assert data["budget"] == {"default_limit": 20, "warning_threshold": 0.8}
    assert data["error_handling"] == {"max_retries": 3}
    assert not path.exists()
    assert source.persistence_safe_point() == "needs_user_save_or_discard"


def test_recovery_discovers_private_override_and_normal_absence(eval_profile):
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Evals.recovery import _DefinitionsAdapter

    selector, path, _, _ = eval_profile
    context = DiscoveryContext(selector, "eval-profile")
    adapter = _DefinitionsAdapter()
    rows = adapter.discover({DISCOVERY_CONTEXT_KEY: context})
    assert len(rows) == 1
    assert rows[0].path == path and rows[0].status == "unused"
    path.write_text("budget: {default_limit: 20}\n", encoding="utf-8")
    rows = adapter.discover({DISCOVERY_CONTEXT_KEY: context})
    assert rows[0].path == path and rows[0].status == "included"


def test_explicit_custom_file_remains_full_configuration(eval_profile):
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    selector, _, _, _ = eval_profile
    path = selector.parent / "custom.yaml"
    path.write_text("task_types: [custom]\n", encoding="utf-8")
    source = EvalConfigLoader(str(path))
    assert source.get_task_types() == ["custom"]
    assert source.get("budget") is None
    source.update({"task_types": ["other"]})
    source.save()
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"task_types": ["other"]}


def test_bad_private_override_reload_preserves_the_actual_draft(eval_profile):
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    _, path, _, _ = eval_profile
    source = EvalConfigLoader()
    source.get("budget")["default_limit"] = 20
    path.write_text("budget: [\n", encoding="utf-8")
    source.reload()
    assert source.persistence_error == "eval_load_failed"
    assert source.get("budget.default_limit") == 20
    assert source.persistence_safe_point() == "needs_user_save_or_discard"
    source.save()
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "budget": {"default_limit": 20}
    }


def test_failed_initial_private_read_does_not_pin_all_shipped_defaults(eval_profile):
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    _, path, defaults, publish = eval_profile
    path.write_text("budget: [\n", encoding="utf-8")
    source = EvalConfigLoader()
    assert source.persistence_error == "eval_load_failed"
    assert source.get("budget.default_limit") == 10
    source.update({"budget": {"default_limit": 20}})
    source.save()
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "budget": {"default_limit": 20}
    }
    updated = copy.deepcopy(defaults)
    updated["error_handling"]["max_retries"] = 4
    publish(updated)
    source.reload()
    assert source.get("error_handling.max_retries") == 4


def test_explicit_pin_blocks_maintenance_until_saved(eval_profile):
    import time
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    source = EvalConfigLoader()
    source.update({"budget": {"default_limit": 10}})
    participant = raw._raw_participant(source)
    participant.close_admission()
    try:
        assert not participant.drain(time.monotonic() + 0.1)
    finally:
        participant.resume()
    source.save()
    participant.close_admission()
    try:
        assert participant.drain(time.monotonic() + 0.1)
    finally:
        participant.resume()


def test_global_loader_follows_profile_and_keeps_held_drafts(eval_profile, monkeypatch):
    from tldw_chatbook.Evals import config_loader

    selector, _, _, _ = eval_profile
    monkeypatch.setattr(config_loader, "_config_loader", None)
    first = config_loader.get_eval_config()
    first.get("budget")["default_limit"] = 20
    alternate = selector.parent / "other" / "config.toml"
    alternate.parent.mkdir()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(alternate))
    second = config_loader.get_eval_config()
    assert second is not first
    assert second.config_path == alternate.parent / "eval_overrides.yaml"
    assert second.get("budget.default_limit") == 10
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    assert config_loader.get_eval_config() is first
    assert first.get("budget.default_limit") == 20


@pytest.mark.parametrize("canonical", [True, False])
def test_full_rollback_omits_only_the_canonical_unused_declaration(
    eval_profile, canonical
):
    from dataclasses import replace
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.journal import (
        _Prepared,
        _Rollback,
        observe_artifact,
    )
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
        Inventory,
        StorageItem,
    )
    from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan
    from tldw_chatbook.Evals.recovery import (
        _DefinitionsAdapter,
        _original_definition_paths,
    )

    selector, _, _, _ = eval_profile
    context = DiscoveryContext(selector, "rollback")
    declaration = _DefinitionsAdapter().discover({DISCOVERY_CONTEXT_KEY: context})[0]
    assert declaration.status == "unused"
    if not canonical:
        declaration = replace(declaration, path=selector.parent / "legacy.yaml")
    config_item = StorageItem(
        "config", "profile:rollback:config", selector, "included", ()
    )
    target = Inventory((config_item, declaration), True, "owned-component", ())
    plan = RestorePlan("0" * 64, "replace", (), (), (), "fixture", target=target)
    # Strict receipt parsing remains real; this unit exercises the post-authentication
    # selection step separately from the full native publication fixture.
    proof = _Rollback(
        ciphertext=observe_artifact(selector),
        sealed_digest="0" * 64,
        manifest_digest="0" * 64,
        coverage={},
    )
    rows = [SimpleNamespace(event="rollback_verified", evidence=proof.model_dump())]
    prepared = _Prepared(generation="fixture", mode="replace")
    binding = {"roots": [str(selector.parent)]}
    if canonical:
        assert _original_definition_paths(selector, binding, rows, prepared, plan) == ()
    else:
        with pytest.raises(ValueError, match="eval_retained_owner_unverified"):
            _original_definition_paths(selector, binding, rows, prepared, plan)


@pytest.mark.parametrize("kind", ["directory", "dangling-link", "fifo"])
def test_recovery_blocks_nonregular_private_override(eval_profile, kind):
    import os

    from tldw_chatbook.Backup_Recovery.inventory import BLOCKING
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Evals.recovery import _DefinitionsAdapter

    selector, path, _, _ = eval_profile
    if kind == "directory":
        path.mkdir()
        (path / "private.yaml").write_text("budget: 7", encoding="utf-8")
    elif kind == "dangling-link":
        try:
            path.symlink_to(path.with_name("missing.yaml"))
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                pytest.skip("Windows account cannot create file symlinks")
            raise
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("Platform has no FIFO creation primitive")
        os.mkfifo(path)
    context = DiscoveryContext(selector, "eval-profile")
    (item,) = _DefinitionsAdapter().discover({DISCOVERY_CONTEXT_KEY: context})
    assert item.path == path
    assert item.status in BLOCKING


def test_recovery_blocks_unreadable_private_override(eval_profile, monkeypatch):
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery.inventory import BLOCKING
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Evals.recovery import _DefinitionsAdapter

    selector, path, _, _ = eval_profile
    path.write_text("budget: {default_limit: 7}", encoding="utf-8")
    original = Path.stat
    attempted = []

    def unreadable(selected, *args, **kwargs):
        if selected == path:
            attempted.append(True)
            raise PermissionError("override metadata is unreadable")
        return original(selected, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", unreadable)
    context = DiscoveryContext(selector, "eval-profile")
    (item,) = _DefinitionsAdapter().discover({DISCOVERY_CONTEXT_KEY: context})
    assert attempted
    assert item.path == path
    assert item.status in BLOCKING
