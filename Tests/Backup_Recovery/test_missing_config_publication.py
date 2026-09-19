"""Installed config files reuse their config scope without enrolling new roots."""

from dataclasses import replace
from threading import Event

import pytest

from Tests.Backup_Recovery.test_default_service_container import (
    default_container as default_container,  # noqa: PLC0414
)
from tldw_chatbook.Backup_Recovery import bootstrap, replacement, service_storage
from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed
from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.restore_plan import (
    _fingerprint,
    _paths,
    plan_restore,
)
from tldw_chatbook.runtime_policy.recovery import recovery_adapters as runtime_adapters


@pytest.fixture(params=["ui.state", "ui.emoji_recents", "runtime.source_state"])
def missing_file(default_container, monkeypatch, request):
    service, archive, parent, target, _ = default_container
    selector = parent / "config.toml"
    selector.write_bytes(b'[general]\nusers_name="fixture"\n')
    selector.chmod(0o600)
    config = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "fixture")}
    adapters = {
        owner.owner_id: owner for owner in (*recovery_adapters(), *runtime_adapters())
    }
    config_item = adapters["config"].discover(config)[0]
    absent = adapters[request.param].discover(config)[0]
    assert absent.status == "unused" and not absent.path.exists()
    target = replace(target, items=(*target.items, config_item, absent))
    plan = plan_restore(
        archive, mode="replace", destinations={"root": parent}, target=target
    )
    doc = verify_sealed(archive)
    config_record = doc.files[0].model_copy(
        update={
            "logical_id": "import-config",
            "owner_id": "config",
            "relative_path": "config.toml",
        }
    )
    file_record = doc.files[0].model_copy(
        update={
            "logical_id": "import-file",
            "owner_id": request.param,
            "relative_path": absent.path.name,
        }
    )
    producer = doc.producer_inventory[0]
    doc = doc.model_copy(
        update={
            "files": (config_record, file_record),
            "producer_inventory": (
                producer.model_copy(
                    update={
                        "logical_id": "import-config",
                        "owner_id": "config",
                        "status": "included",
                        "dependencies": (),
                    }
                ),
                producer.model_copy(
                    update={
                        "logical_id": "import-file",
                        "owner_id": request.param,
                        "status": "included",
                        "dependencies": ("import-config",),
                    }
                ),
            ),
        }
    )
    plan = replace(
        plan, restore=(("import-config", selector), ("import-file", absent.path))
    )
    plan = replace(plan, target_fingerprint=_fingerprint(_paths(plan), target))
    monkeypatch.setattr(
        replacement, "_first_binding_inventory", lambda _plan, _selector: target
    )
    root = bootstrap.default_bootstrap_root()
    protected = (
        root,
        service.control_root,
        service_storage.work_root(service.control_root) / "candidate",
    )
    replacement._ensure_first_bindings(
        plan, (selector,), root, Event(), protected=protected, document=doc
    )
    authority = admission_authority(root)
    return plan, doc, authority, protected, absent


def test_missing_installed_config_file_uses_existing_scope(missing_file):
    plan, doc, authority, protected, absent = missing_file
    before = bootstrap._registry(authority.control_root.parent)
    profiles = bootstrap._records(authority.control_root.parent)
    scopes = replacement._register_publication_parents(
        plan, authority, protected, document=doc
    )
    after = bootstrap._registry(authority.control_root.parent)
    assert after == before
    assert bootstrap._records(authority.control_root.parent) == profiles
    assert scopes == {absent.path: plan.restore[0][1]}
    assert not absent.path.exists()
    assert not any(str(absent.path.parent) in row["roots"] for row in after.values())


@pytest.mark.parametrize(
    "case",
    [
        "unknown_name",
        "wrong_owner",
        "wrong_dependency",
        "local_wrong_dependency",
        "unselected_file",
        "foreign_root",
        "foreign_profile",
        "missing_controls",
        "existing_file",
        "symlink",
    ],
)
def test_missing_config_file_keeps_collision_and_owner_refusals(
    missing_file, case, tmp_path
):
    plan, doc, authority, protected, absent = missing_file
    if case == "unknown_name":
        plan = replace(
            plan,
            restore=(
                plan.restore[0],
                ("import-file", absent.path.with_name("unknown.txt")),
            ),
        )
    elif case == "wrong_owner":
        doc = doc.model_copy(
            update={
                "files": (
                    doc.files[0],
                    doc.files[1].model_copy(update={"owner_id": "external.files"}),
                )
            }
        )
    elif case == "wrong_dependency":
        doc = doc.model_copy(
            update={
                "producer_inventory": (
                    doc.producer_inventory[0],
                    doc.producer_inventory[1].model_copy(update={"dependencies": ()}),
                )
            }
        )
    elif case == "local_wrong_dependency":
        plan = replace(
            plan,
            target=replace(
                plan.target,
                items=tuple(
                    replace(item, dependencies=()) if item is absent else item
                    for item in plan.target.items
                ),
            ),
        )
    elif case == "unselected_file":
        plan = replace(plan, restore=(plan.restore[0],))
        # An unrelated unmapped file must not be enrolled opportunistically.
        before = bootstrap._registry(authority.control_root.parent)
        replacement._register_publication_parents(
            plan, authority, protected, document=doc
        )
        assert bootstrap._registry(authority.control_root.parent) == before
        return
    elif case == "foreign_root":
        absent.path.write_bytes(b"foreign")
        absent.path.chmod(0o600)
        authority.register("foreign", (absent.path,))
    elif case == "foreign_profile":
        from tldw_chatbook.Backup_Recovery.control_records import bind_profile

        selector = absent.path.parent / "other.toml"
        selector.write_bytes(b"")
        selector.chmod(0o600)
        authority.register("foreign", (selector,))
        bind_profile(
            authority.control_root.parent,
            selector,
            ("foreign",),
            authority.control_root,
        )
    elif case == "missing_controls":
        plan = replace(plan, preserve=())
    elif case == "existing_file":
        absent.path.write_bytes(b"changed")
        absent.path.chmod(0o600)
    elif case == "symlink":
        other = tmp_path / "other"
        other.write_bytes(b"other")
        other.chmod(0o600)
        absent.path.symlink_to(other)
    if case != "symlink":
        plan = replace(plan, target_fingerprint=_fingerprint(_paths(plan), plan.target))
    before = bootstrap._registry(authority.control_root.parent)
    with pytest.raises(ValueError):
        replacement._register_publication_parents(
            plan, authority, protected, document=doc
        )
    assert bootstrap._registry(authority.control_root.parent) == before
