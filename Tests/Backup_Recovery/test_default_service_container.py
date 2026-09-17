"""Only untouched synthetic containers may contain excluded default service work."""

from dataclasses import replace
from threading import Event

import pytest

from Tests.Backup_Recovery.test_restore_plan import producer, sealed
from tldw_chatbook.Backup_Recovery import bootstrap, inventory, service_storage
from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.staging import stage_restore


@pytest.fixture
def default_container(tmp_path, monkeypatch):
    config = tmp_path / "default-config" / "config.toml"
    monkeypatch.setattr(service_storage, "default_config_path", lambda: config)
    control = service_storage.default_control_root()
    service_storage.ensure_storage(control)
    service = RecoveryService(control)
    item = inventory._service_control_exclusion()[0]
    root = config.parent / "recovery-bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    admission_authority(root)
    fixed = inventory._fixed_control_exclusion()[0]
    live = config.parent / "note.txt"
    live.write_bytes(b"current saved note")
    live.chmod(0o600)
    target = Inventory(
        (StorageItem("ui.state", "file", live, "included", ()), item, fixed),
        True,
        "local",
        (),
    )

    def synthetic(doc):
        producer(doc)
        doc["directories"][0]["synthetic"] = True

    archive = sealed(tmp_path, mutate=synthetic)
    yield service, archive, config.parent, target, item
    service.close()


def test_default_service_archive_and_candidate_preserve_existing_container(
    default_container,
):
    service, external_archive, destination, target, item = default_container
    operation = service.start_inspection(external_archive.path, password=None)
    assert service.wait(operation)["state"] == "succeeded"
    archive = service.inspection(operation)
    assert service_storage.work_root(service.control_root) in archive.path.parents
    marker = service.control_root / "service.json"
    before = marker.read_bytes()
    plan = service.preview_restore(
        operation, mode="replace", destinations={"root": destination}, target=target
    )
    assert dict(plan.preserve)[item.logical_id] == item.path
    assert destination not in dict(plan.restore).values()
    candidate = stage_restore(
        archive,
        plan,
        service_storage.work_root(service.control_root) / "candidate",
        Event(),
    )
    assert candidate.is_dir()
    assert marker.read_bytes() == before
    assert (destination / "note.txt").read_bytes() == b"current saved note"


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "nonsynthetic",
        "isolated",
        "missing_preserve",
        "missing_target",
        "wrong_status",
        "wrong_owner",
        "wrong_id",
        "nondefault",
        "control",
        "service_root",
        "new_container",
        "restore_child",
        "retire_child",
        "publish_parent",
        "container_parent",
        "bad_marker",
    ],
)
def test_service_container_proof_retains_all_mutation_boundaries(
    default_container, tmp_path, case
):
    service, archive, destination, target, item = default_container
    plan = plan_restore(
        archive, mode="replace", destinations={"root": destination}, target=target
    )
    work = service_storage.work_root(service.control_root) / "candidate"
    if case == "isolated":
        plan = replace(plan, mode="isolated")
    elif case == "missing_preserve":
        plan = replace(plan, preserve=())
    elif case == "missing_target":
        plan = replace(plan, target=None)
    elif case in {"wrong_status", "wrong_owner", "wrong_id"}:
        altered = replace(
            item,
            **{
                "wrong_status": {"status": "included_directory"},
                "wrong_owner": {"owner": "ui.state"},
                "wrong_id": {"logical_id": "other"},
            }[case],
        )
        plan = replace(
            plan,
            target=replace(target, items=(target.items[0], altered)),
            preserve=((altered.logical_id, altered.path),),
        )
    elif case == "nondefault":
        work = tmp_path / "another-work" / "candidate"
    elif case == "control":
        work = service.control_root / "candidate"
    elif case == "service_root":
        destination = item.path
    elif case == "new_container":
        destination = tmp_path / "missing-container"
    elif case in {"restore_child", "retire_child"}:
        field = "restore" if case == "restore_child" else "retire"
        plan = replace(
            plan, **{field: ((case, service.control_root / "service.json"),)}
        )
    elif case == "publish_parent":
        plan = replace(plan, restore=(("root", destination),))
    elif case == "container_parent":
        plan = replace(plan, containers=(("local:container", destination),))
    elif case == "bad_marker":
        (service.control_root / "service.json").write_text("{}")
    assert service_storage.is_preserved_service_container(
        work, destination, plan, synthetic=case != "nonsynthetic"
    ) is (case == "valid")


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "unknown_protected",
        "nondefault_protected",
        "unknown_sibling",
        "service_replaced",
        "unpreserved_bootstrap",
        "bootstrap_mutation",
        "foreign_bootstrap_registry",
    ],
)
def test_first_default_config_binding_preserves_exact_service_only(
    default_container, tmp_path, case
):
    from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed
    from tldw_chatbook.Backup_Recovery.replacement import _first_config_container

    service, archive, destination, target, _item = default_container
    selector = destination / "config.toml"
    selector.write_bytes(b"[general]\n")
    selector.chmod(0o600)
    target = replace(
        target,
        items=(
            *target.items,
            StorageItem("config", "config", selector, "included", ()),
        ),
    )
    plan = plan_restore(
        archive, mode="replace", destinations={"root": destination}, target=target
    )
    protected = (
        bootstrap.default_bootstrap_root(),
        service.control_root,
        service_storage.work_root(service.control_root) / "candidate",
    )
    if case == "unknown_protected":
        protected += (destination / "unknown-control",)
    elif case == "nondefault_protected":
        protected += (destination / "another-work" / "candidate",)
    elif case == "unknown_sibling":
        (destination / "unowned.txt").write_bytes(b"unknown")

    registry = bootstrap._registry(bootstrap.default_bootstrap_root())
    if case == "unpreserved_bootstrap":
        plan = replace(
            plan,
            preserve=tuple(
                row
                for row in plan.preserve
                if row[0] != "recovery.control:fixed-bootstrap"
            ),
        )
    elif case == "bootstrap_mutation":
        plan = replace(
            plan,
            retire=(("bad", bootstrap.default_bootstrap_root() / "registry.json"),),
        )
    elif case == "foreign_bootstrap_registry":
        registry["foreign"] = {
            "roots": [str(bootstrap.default_bootstrap_root() / "foreign")]
        }

    def prove():
        return _first_config_container(
            selector,
            target,
            (),
            registry,
            (),
            (selector,),
            protected,
            plan=plan,
            document=verify_sealed(archive),
        )

    if case in {
        "unknown_protected",
        "nondefault_protected",
        "unknown_sibling",
        "unpreserved_bootstrap",
        "bootstrap_mutation",
        "foreign_bootstrap_registry",
    }:
        with pytest.raises(ValueError, match="replacement_config_container_unverified"):
            prove()
        return
    initial = prove()
    if case == "service_replaced":
        old = service_storage.work_root(service.control_root)
        old.rename(tmp_path / "old-work")
        old.mkdir(mode=0o700)
        assert prove() != initial
    else:
        (
            service_storage.work_root(service.control_root) / "internal-progress"
        ).write_bytes(b"work changes")
        assert prove() == initial


@pytest.mark.parametrize("kind", ["arbitrary", "nondefault_service"])
def test_other_nested_archive_and_staging_paths_still_refuse(default_container, kind):
    from tldw_chatbook.Backup_Recovery.archive_reader import acquire
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    _service, archive, destination, target, _item = default_container
    if kind == "nondefault_service":
        control = destination / "other-service" / "control"
        nested_work = service_storage.ensure_storage(control)
    else:
        nested_work = destination / "arbitrary-work"
        nested_work.mkdir(mode=0o700)
    nested = acquire(
        archive.path, nested_work / "source", ArchiveLimits(), None, Event()
    )
    with pytest.raises(ValueError, match="^archive_destination_alias$"):
        plan_restore(
            nested, mode="replace", destinations={"root": destination}, target=target
        )
    plan = plan_restore(
        archive, mode="replace", destinations={"root": destination}, target=target
    )
    with pytest.raises(ValueError, match="^staging_destination_alias$"):
        stage_restore(archive, plan, nested_work / "candidate", Event())


def test_first_default_binding_never_enrolls_control_or_config_parent(
    default_container, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import replacement
    from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed

    service, archive, destination, target, _item = default_container
    selector = destination / "config.toml"
    selector.write_bytes(b"[general]\n")
    selector.chmod(0o600)
    target = replace(
        target,
        items=(
            *target.items,
            StorageItem("config", "config", selector, "included", ()),
        ),
    )
    plan = plan_restore(
        archive, mode="replace", destinations={"root": destination}, target=target
    )
    monkeypatch.setattr(
        replacement, "_first_binding_inventory", lambda _plan, _selector: target
    )
    root = bootstrap.default_bootstrap_root()
    replacement._ensure_first_bindings(
        plan,
        (selector,),
        root,
        Event(),
        protected=(
            service.control_root,
            service_storage.work_root(service.control_root) / "candidate",
        ),
        document=verify_sealed(archive),
    )
    pending, profiles = bootstrap._records(root)
    assert not pending and len(profiles) == 1
    roots = {path for row in profiles for path in row["roots"]}
    assert roots == {str(selector), str(destination / "note.txt")}
    assert not any(
        str(destination) in row["roots"] for row in bootstrap._registry(root).values()
    )


@pytest.mark.parametrize("covered", [True, False])
def test_default_container_new_payload_requires_existing_child_authority(
    default_container, covered
):
    from tldw_chatbook.Backup_Recovery import replacement
    from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.restore_plan import _fingerprint, _paths

    service, archive, destination, target, _item = default_container
    child = destination / "owned-child"
    child.mkdir(mode=0o700)
    plan = plan_restore(
        archive, mode="replace", destinations={"root": destination}, target=target
    )
    plan = replace(
        plan,
        restore=(("new", child / "new.txt" if covered else destination / "new.txt"),),
    )
    plan = replace(plan, target_fingerprint=_fingerprint(_paths(plan), plan.target))
    root = bootstrap.default_bootstrap_root()
    authority = admission_authority(root)
    authority.register("owned-child", (child,))
    doc = verify_sealed(archive)
    # Exercise the config-parent requirement with the verified synthetic root.
    doc = doc.model_copy(
        update={
            "files": tuple(
                row.model_copy(update={"owner_id": "config"}) for row in doc.files
            )
        }
    )
    protected = (
        root,
        service.control_root,
        service_storage.work_root(service.control_root) / "candidate",
    )
    if covered:
        replacement._register_publication_parents(
            plan, authority, protected, document=doc
        )
    else:
        with pytest.raises(
            ValueError, match="^replacement_destination_parent_required$"
        ):
            replacement._register_publication_parents(
                plan, authority, protected, document=doc
            )
    assert not any(
        str(destination) in row["roots"] for row in bootstrap._registry(root).values()
    )
