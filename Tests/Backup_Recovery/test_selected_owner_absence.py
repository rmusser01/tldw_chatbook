"""Selected replacement includes an owner's authenticated empty source state."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_restore_data_groups import _document
from Tests.Backup_Recovery.test_selective_restore_service import _SELECTED_SERVICE

_HISTORY = r'''
    history_writer = """
import asyncio
from tldw_chatbook.Chat.prompt_history import PromptHistory, default_prompt_history_path
async def main():
    history = PromptHistory(default_prompt_history_path())
    await history.append('New prompt history after backup')
    assert history.size == 1
asyncio.run(main())
"""
    created = subprocess.run([sys.executable, '-c', history_writer],
                             capture_output=True, text=True, timeout=30)
    assert created.returncode == 0, created.stderr[-4000:]
'''


_HISTORY_AUDIT = r"""
    import hashlib
    from tldw_chatbook.Backup_Recovery import publication
    from Tests.Backup_Recovery.thread_diagnostics import observe_recovery_failures
    stop_history_failures = observe_recovery_failures(Path.home() / 'history-recovery-failures.log')
    def history_mapping():
        root = bootstrap.default_bootstrap_root()
        registry = bootstrap._registry(root)
        profiles = bootstrap._records(root)[1]
        own = next(row for row in profiles if row['selector'] == str(source))
        child_names = {name for name, entry in registry.items()
                       if str(history_path) in entry['roots']}
        assert child_names and child_names <= set(own['namespaces'])
        assert str(history_path) in own['roots']
        assert any(Path(path) in history_path.parents and Path(path).is_dir()
                   for name in own['namespaces'] for path in registry[name]['roots'])
        assert any(history_inode_token in registry[name]['historical'] for name in child_names)
        return {
            'registry': hashlib.sha256((root / 'admission' / 'registry.json').read_bytes()).hexdigest(),
            'profiles': {row['selector']: [row['namespaces'], row['roots']] for row in profiles},
        }
    original_publish = publication.publish_candidate
    publication_mappings = []
    def observe_history_publication(*args, **kwargs):
        evidence = history_mapping()
        publication_mappings.append(evidence)
        evidence['history_path'] = str(history_path)
        evidence['history_inode_token'] = history_inode_token
        evidence['preserved'] = {
            str(path): [hashlib.sha256(content).hexdigest(), device, inode]
            for path, (content, device, inode) in preserved.items()
        }
        proof = Path.home() / 'history-publication-evidence.json'
        proof.write_text(json.dumps(evidence))
        proof.chmod(0o600)
        return original_publish(*args, **kwargs)
    publication.publish_candidate = observe_history_publication
    def assert_history_mapping():
        current = history_mapping()
        assert all(current == {key: value for key, value in before.items()
                               if key in ('registry', 'profiles')}
                   for before in publication_mappings)
"""


_HISTORY_UNDO = r"""
    assert history_path.read_bytes() == history_before
    assert_history_mapping()
    current = service.preview_backup((source,), options={})
    undo = service.preview_rollback(rollback_operation,
        old_password=b'second-safety-password', target=current)
    assert undo.effective_groups == ('prompts',)
    assert {path for _, path in undo.restore} == {prompt_path}
    assert {path for _, path in undo.retire} == {history_path}
    succeeded(service, service.start_rollback(rollback_operation, undo,
        old_password=b'second-safety-password', new_password=b'third-safety-password'))
    assert not history_path.exists()
    read_prompts({'prepare': saved_text, 'newer': None})
    assert_preserved()
    assert_history_mapping()
"""


def _history_service_script():
    """Build the genuine history fixture for completion and fresh Finish cases."""
    script = _SELECTED_SERVICE.replace("ENCRYPTED", "False").replace(
        "SCENARIO", repr("prompts")
    )
    script = script.replace(
        "    assert preview.complete, preview.issues",
        "    assert preview.complete, preview.issues\n"
        "    assert next(row for row in preview.items if row.owner == 'chat.prompt_history').status == 'unused'",
    )
    script = script.replace(
        "    target = service.preview_backup((source,), options={})",
        _HISTORY + "\n    target = service.preview_backup((source,), options={})\n"
        "    history_row = next(row for row in target.items if row.owner == 'chat.prompt_history')\n"
        "    assert history_row.status == 'included'\n"
        "    history_path = history_row.path\n"
        "    history_before = history_path.read_bytes()\n"
        "    from tldw_chatbook.Utils.platform_files import os as native_os\n"
        "    history_info = native_os.stat(history_path)\n"
        "    history_inode_token = f'inode:{history_info.st_dev}:{history_info.st_ino}'",
    )
    script = script.replace(
        "    assert not plan.retire",
        "    assert plan.retire == ((history_row.logical_id, history_path),)\n"
        "    assert (history_row.logical_id, history_path) not in plan.preserve",
    )
    script = script.replace(
        "    operation = result['result']['journal_operation_id']",
        "    assert not history_path.exists()\n"
        "    assert_history_mapping()\n"
        "    operation = result['result']['journal_operation_id']",
    )
    script = script.replace(
        "        assert {path for _, path in rollback.restore} == {prompt_path}",
        "        assert {path for _, path in rollback.restore} == {prompt_path, history_path}",
    )
    script = script.replace(
        "    assert bootstrap.startup_permission(source, bootstrap.default_bootstrap_root())[0]",
        _HISTORY_UNDO + "\n"
        "    assert bootstrap.startup_permission(source, bootstrap.default_bootstrap_root())[0]",
    )
    marker = "    result = succeeded(service, service.start_restore(inspection, plan,"
    assert script.count(marker) == 1
    script = script.replace(marker, _HISTORY_AUDIT + "\n" + marker)
    # Every real producer readback runs in a new process. Explicitly check the
    # startup fence as well as its existing native database admission.
    marker = "selector = Path(os.environ['TLDW_CONFIG_PATH'])"
    assert script.count(marker) == 1
    script = script.replace(
        marker,
        marker + "\nfrom tldw_chatbook.Backup_Recovery import bootstrap\n"
        "assert bootstrap.startup_permission(selector, bootstrap.default_bootstrap_root())[0]",
    )
    return script


def test_selected_prompts_replace_empty_history_and_rollback_restores_it(tmp_path):
    script = _history_service_script()
    _run(tmp_path, "selected-owner-absence", "success", script=script, timeout=180)


@pytest.mark.parametrize(
    "change",
    [
        None,
        "excluded",
        "missing",
        "owner",
        "source_profile",
        "target_profile",
        "dependency",
        "unselected",
        "relocated_slot",
        "shared_publication",
    ],
)
def test_owner_absence_needs_exact_selected_source_profile_declaration(
    tmp_path, change
):
    """Exercise pure mapping; the native service test owns filesystem authority."""
    from dataclasses import replace

    from tldw_chatbook.Backup_Recovery.models import (
        FileMetadata,
        Inventory,
        StorageItem,
    )
    from tldw_chatbook.Backup_Recovery.restore_groups import (
        resolve_archive_groups,
        selected_absent_target_ids,
    )
    from tldw_chatbook.Backup_Recovery.restore_plan import RetainedConfig

    doc = _document(("config", "db.prompts.primary"))
    source_id = "profile:profile:chat.prompt_history"
    current_id = "profile:current:chat.prompt_history"
    config_id = "profile:current:config"
    declaration = type(doc.producer_inventory[0])(
        logical_id=source_id,
        owner_id="chat.prompt_history",
        status="unused",
        dependencies=("profile:profile:config",),
    )
    if change == "excluded":
        declaration = declaration.model_copy(
            update={"status": "intentionally_excluded"}
        )
    elif change == "owner":
        declaration = declaration.model_copy(update={"owner_id": "ui.state"})
    elif change == "source_profile":
        declaration = declaration.model_copy(
            update={"logical_id": "profile:other:chat.prompt_history"}
        )
    elif change == "dependency":
        declaration = declaration.model_copy(update={"dependencies": ()})
    elif change == "relocated_slot":
        declaration = declaration.model_copy(
            update={"logical_id": source_id + ":" + "a" * 64}
        )
    if change != "missing":
        doc = doc.model_copy(
            update={"producer_inventory": (*doc.producer_inventory, declaration)}
        )
    scope = resolve_archive_groups(doc, ("prompts",))
    if change == "unselected":
        scope = replace(scope, effective_groups=("library",), member_ids=())
    config_path = tmp_path / "config.toml"
    history_path = tmp_path / "prompt_history.jsonl"
    if change == "target_profile":
        current_id = "profile:unreviewed:chat.prompt_history"
    history = StorageItem(
        "chat.prompt_history",
        current_id,
        history_path,
        "included",
        (config_id,),
        metadata=FileMetadata(
            1, current_id, history_path.name, None, "file", 0o600, 1, "private"
        ),
    )
    target = Inventory(
        (StorageItem("config", config_id, config_path, "included", ()), history),
        True,
        "pure",
        (),
    )
    retained = RetainedConfig(
        "profile:profile:config", config_id, config_path, "0" * 64
    )
    restore = (
        (("profile:another:chat.prompt_history", history_path),)
        if change == "shared_publication"
        else ()
    )
    if change in {"relocated_slot", "shared_publication"}:
        with pytest.raises(ValueError, match="selected_absence_"):
            selected_absent_target_ids(doc, scope, target, restore, (retained,))
        return
    actual = selected_absent_target_ids(doc, scope, target, restore, (retained,))
    assert actual == (frozenset({current_id}) if change is None else frozenset())


@pytest.mark.parametrize("target_status", ["unused", "intentionally_excluded"])
def test_empty_history_slot_is_satisfied_independently_of_active_history(
    tmp_path, target_status
):
    """An absent .bak and a saved timestamped file are distinct same-owner slots."""
    from tldw_chatbook.Backup_Recovery.models import (
        FileMetadata,
        Inventory,
        StorageItem,
    )
    from tldw_chatbook.Backup_Recovery.restore_groups import (
        resolve_archive_groups,
        selected_absent_target_ids,
    )

    doc = _document(("config", "config.history"))
    config_id = "profile:profile:config"
    absent_id = "profile:profile:config.history:" + "a" * 64
    declaration = type(doc.producer_inventory[0])(
        logical_id=absent_id,
        owner_id="config.history",
        status="unused",
        dependencies=(config_id,),
    )
    doc = doc.model_copy(
        update={"producer_inventory": (*doc.producer_inventory, declaration)}
    )
    scope = resolve_archive_groups(doc, ("settings",))
    config = tmp_path / "config.toml"
    saved = tmp_path / "config_backup_20260915_120000.toml"
    saved_id = "profile:profile:config.history"
    target = Inventory(
        (
            StorageItem("config", config_id, config, "included", ()),
            StorageItem(
                "config.history",
                absent_id,
                tmp_path / "config.toml.bak",
                target_status,
                (config_id,),
            ),
            StorageItem(
                "config.history",
                saved_id,
                saved,
                "included",
                (config_id,),
                metadata=FileMetadata(
                    1, saved_id, saved.name, None, "file", 0o600, 1, "private"
                ),
            ),
        ),
        True,
        "pure",
        (),
    )
    if target_status == "intentionally_excluded":
        with pytest.raises(ValueError, match="selected_absence_mapping_required"):
            selected_absent_target_ids(
                doc, scope, target, ((config_id, config), (saved_id, saved)), ()
            )
        return
    assert (
        selected_absent_target_ids(
            doc, scope, target, ((config_id, config), (saved_id, saved)), ()
        )
        == frozenset()
    )
