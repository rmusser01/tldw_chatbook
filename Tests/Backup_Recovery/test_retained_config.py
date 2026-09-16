"""Selected replacement uses checked local config only as a private dependency."""

import hashlib
import json
import subprocess  # nosec B404 - fixed native ACL command for the private fixture
import sys
import zipfile
from dataclasses import replace
from pathlib import PurePosixPath
from threading import Event

import pytest

from Tests.Backup_Recovery.test_archive_reader import manifest
from tldw_chatbook.Backup_Recovery import bootstrap, publication, restore_plan
from tldw_chatbook.Backup_Recovery.archive_reader import acquire
from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
from tldw_chatbook.Backup_Recovery.control_records import (
    UNBOUND_NAMESPACE,
    admission_authority,
    bind_profile,
)
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.models import (
    DISCOVERY_CONTEXT_KEY,
    DiscoveryContext,
    Inventory,
)
from tldw_chatbook.Backup_Recovery.staging import stage_restore
from tldw_chatbook.Utils.platform_files import os as native_os


@pytest.fixture
def retained_case(tmp_path, monkeypatch, request):
    live = tmp_path / "live"
    live.mkdir(mode=0o700)
    selector = live / "config.toml"
    selector.write_bytes(b'[general]\nusers_name="Local unchanged"\n')
    selector.chmod(0o600)
    state = live / "ui_state.toml"
    state.write_bytes(b'[state]\nselected="original"\n')
    state.chmod(0o600)
    local_profile = hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    config = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, local_profile)}
    owners = {owner.owner_id: owner for owner in recovery_adapters()}
    items = tuple(owners[owner].discover(config)[0] for owner in ("config", "ui.state"))
    target = Inventory(items, True, "observed", ())
    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    authority = admission_authority(root)
    authority.register(
        "local",
        (selector,) if getattr(request, "param", None) == "selector-only" else (live,),
    )
    if getattr(request, "param", None) != "never-bound":
        bind_profile(root, selector, ("local",), root / "admission")
    payloads = {
        "config": b'[general]\nusers_name="Imported ignored"\n',
        "state": b'[state]\nselected="incoming"\n',
    }
    doc = manifest()
    doc["owners"] = [
        {"owner_id": owner, "schema_version": 1, "capabilities": []}
        for owner in ("config", "ui.state")
    ]
    doc["directories"] = [
        {**doc["directories"][0], "logical_id": key, "root_id": key, "synthetic": True}
        for key in ("config-root", "state-root")
    ]
    doc["files"] = []
    doc["producer_inventory"] = []
    for key, owner, relative in (
        ("config", "config", "config.toml"),
        ("state", "ui.state", "ui_state.toml"),
    ):
        logical = "profile:profile:" + owner
        root_id = key + "-root"
        doc["files"].append(
            {
                "logical_id": logical,
                "root_id": root_id,
                "parent_id": root_id,
                "relative_path": relative,
                "owner_id": owner,
                "payload": "payload/" + key,
                "size": len(payloads[key]),
                "sha256": hashlib.sha256(payloads[key]).hexdigest(),
            }
        )
        doc["producer_inventory"].append(
            {
                "logical_id": logical,
                "owner_id": owner,
                "status": "included",
                "dependencies": [] if key == "config" else ["profile:profile:config"],
                "shared_group": None,
            }
        )
        doc["producer_inventory"].append(
            {
                "logical_id": root_id,
                "owner_id": owner,
                "status": "included_directory",
                "dependencies": [],
                "shared_group": None,
            }
        )
    doc["dependency_groups"][0]["members"] = [row["logical_id"] for row in doc["files"]]
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as packed:
        packed.writestr("manifest.json", json.dumps(doc))
        for key, content in payloads.items():
            packed.writestr("payload/" + key, content)
    archive = acquire(path, tmp_path / "acquired", ArchiveLimits(), None, Event())
    return archive, target, selector, state, authority


def retained_plan(case, *, selectors=None, **changes):
    archive, target, selector, _, _ = case
    relation = restore_plan.RetainedConfig(
        "profile:profile:config",
        target.items[0].logical_id,
        selector,
        restore_plan.retained_config_observation(selector),
    )
    return restore_plan.plan_restore(
        archive,
        mode="replace",
        destinations={"state-root": selector.parent, **(selectors or {})},
        target=target,
        retained_configs=(replace(relation, **changes),),
    )


def test_selected_plan_preserves_config_and_captures_it_only_as_support(retained_case):
    plan = retained_plan(retained_case)
    assert dict(plan.restore) == {"profile:profile:ui.state": retained_case[3]}
    assert (
        plan.retained_configs[0].target_config_id,
        retained_case[2],
    ) in plan.preserve
    assert plan.safety_scope == (plan.retained_configs[0].target_config_id,)
    assert restore_plan.required_rollback_dependencies(plan) == ()


@pytest.mark.parametrize(
    "field,value",
    [
        ("archive_config_id", "profile:profile:ui.state"),
        ("target_config_id", "profile:other:config"),
        ("observation", "0" * 64),
    ],
)
def test_unverified_retained_relation_refuses(retained_case, field, value):
    with pytest.raises(ValueError, match="retained_config"):
        retained_plan(retained_case, **{field: value})


@pytest.mark.parametrize("mutation", ["content", "identity"])
def test_retained_config_drift_invalidates_plan(retained_case, mutation):
    plan = retained_plan(retained_case)
    selector = retained_case[2]
    if mutation == "content":
        selector.write_bytes(b'[general]\nusers_name="changed"\n')
    else:
        other = selector.with_name("replacement.toml")
        other.write_bytes(selector.read_bytes())
        other.chmod(0o600)
        other.replace(selector)
    with pytest.raises(ValueError, match="changed"):
        restore_plan.recheck_targets(plan)


def test_retained_config_staging_requires_native_session(retained_case, tmp_path):
    plan = retained_plan(retained_case)
    with pytest.raises(ValueError, match="retained_config_native_session_required"):
        stage_restore(retained_case[0], plan, tmp_path / "work", Event())


def test_native_stage_keeps_support_out_of_publication(retained_case, tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.plan_records import load_plan

    archive, _, selector, state, authority = retained_case
    plan = retained_plan(retained_case)
    before = (selector.read_bytes(), selector.stat().st_ino)
    (tmp_path / "control").mkdir(mode=0o700)
    journal = Journal(tmp_path / "control", "retained-config-stage")
    with authority.maintenance(("local", UNBOUND_NAMESPACE), 3) as session:
        candidate = stage_restore(
            archive, plan, tmp_path / "work", Event(), session=session, journal=journal
        )
    descriptor = json.loads((candidate / "candidate.json").read_bytes())
    assert [row["destination"] for row in descriptor["artifacts"]] == [str(state)]
    assert (selector.read_bytes(), selector.stat().st_ino) == before
    assert load_plan(journal) == plan
    assert publication._plan_digest(
        replace(plan, retained_configs=())
    ) != publication._plan_digest(plan)


@pytest.mark.parametrize("retained_case", ["never-bound"], indirect=True)
@pytest.mark.parametrize("changed", [False, True])
def test_retained_voice_selectors_validate_local_paths_without_rewriting_config(
    retained_case, tmp_path, changed
):
    import toml

    from Tests.Backup_Recovery.test_file_inventory import VOICE_LOCATION_KEYS

    archive, _, selector, state, authority = retained_case
    data = {"general": {"users_name": "Local unchanged"}}
    mapping = {}
    for location in VOICE_LOCATION_KEYS:
        path = selector.parent.joinpath("voices", *location)
        table = data if len(location) == 1 else data.setdefault(location[0], {})
        table[location[-1]] = str(path)
        mapping["profile:profile:" + ".".join(location)] = path
    selector.write_text(toml.dumps(data))
    bind_profile(tmp_path / "bootstrap", selector, ("local",), authority.control_root)
    before = selector.read_bytes(), selector.stat().st_ino
    if changed:
        mapping["profile:profile:HIGGS_VOICE_SAMPLES_DIR"] = selector.parent / "wrong"
    plan = retained_plan(retained_case, selectors=mapping)

    with authority.maintenance(("local", UNBOUND_NAMESPACE), 3) as session:
        if changed:
            with pytest.raises(ValueError, match="retained_config_selector_unverified"):
                stage_restore(
                    archive, plan, tmp_path / "work", Event(), session=session
                )
        else:
            candidate = stage_restore(
                archive, plan, tmp_path / "work", Event(), session=session
            )
            document = json.loads((candidate / "candidate.json").read_bytes())
            assert [row["destination"] for row in document["artifacts"]] == [str(state)]
    assert (selector.read_bytes(), selector.stat().st_ino) == before


def test_retained_config_rejects_selector_with_extra_database_component(
    retained_case,
):
    import tomllib

    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Backup_Recovery.profile_paths import database_path
    from tldw_chatbook.Backup_Recovery.staging import (
        _document,
        _retained_config_targets,
    )

    archive, _, selector, _, _ = retained_case
    data = tomllib.loads(selector.read_text())
    plan = replace(
        retained_plan(retained_case),
        selectors=(
            (
                "profile:profile:database.prompts_db_path.extra",
                database_path(data, "prompts_db_path"),
            ),
        ),
    )
    owners = {owner.owner_id: owner for owner in install_adapters()}
    with pytest.raises(ValueError, match="invalid_config_selector"):
        _retained_config_targets(
            data, "profile", selector, _document(archive), plan, owners
        )


@pytest.mark.parametrize("kind", ["invalid", "public", "stale"])
def test_local_config_observation_requires_valid_private_bound_selector(
    retained_case, kind
):
    _, _, selector, _, _ = retained_case
    if kind == "invalid":
        selector.write_bytes(b"not = [valid")
    elif kind == "public":
        if sys.platform == "win32":
            subprocess.run(  # nosec B603 B607 - fixed ACL on the private fixture
                ["icacls", str(selector), "/grant", "*S-1-1-0:(R)"],
                check=True,
                capture_output=True,
            )
        else:
            selector.chmod(0o644)
        assert native_os.stat(selector).st_mode & 0o044
    else:
        selector.write_bytes(b'[general]\nusers_name="Changed after binding"\n')
    with pytest.raises(ValueError, match="retained_config"):
        restore_plan.retained_config_observation(selector)


@pytest.mark.parametrize("retained_case", ["never-bound"], indirect=True)
def test_first_use_config_review_keeps_bootstrap_and_selector_unchanged(retained_case):
    _, _, selector, _, authority = retained_case
    root = authority.control_root.parent

    def before():
        return {
            path.relative_to(root): (path.read_bytes(), path.stat().st_ino)
            for path in root.rglob("*")
            if path.is_file()
        }

    records = before()
    config = selector.read_bytes(), selector.stat().st_ino
    plan = retained_plan(retained_case)
    assert restore_plan.retained_config_names(plan) == (UNBOUND_NAMESPACE,)
    assert before() == records
    assert (selector.read_bytes(), selector.stat().st_ino) == config
    assert bootstrap._records(root)[1] == []


@pytest.mark.parametrize("retained_case", ["never-bound"], indirect=True)
@pytest.mark.parametrize("held_source", [False, True])
def test_first_use_config_staging_requires_actual_source_scope(
    retained_case, tmp_path, held_source
):
    archive, _, selector, state, authority = retained_case
    plan = retained_plan(retained_case)
    original = selector.read_bytes(), selector.stat().st_ino
    names = (UNBOUND_NAMESPACE, "local") if held_source else (UNBOUND_NAMESPACE,)
    with authority.maintenance(names, 3) as session:
        if held_source:
            candidate = stage_restore(
                archive, plan, tmp_path / "work", Event(), session=session
            )
            document = json.loads((candidate / "candidate.json").read_bytes())
            assert [row["destination"] for row in document["artifacts"]] == [str(state)]
        else:
            with pytest.raises(
                ValueError, match="retained_config_native_scope_required"
            ):
                stage_restore(
                    archive, plan, tmp_path / "work", Event(), session=session
                )
    assert bootstrap._records(authority.control_root.parent)[1] == []
    assert (selector.read_bytes(), selector.stat().st_ino) == original


@pytest.mark.parametrize("retained_case", ["never-bound"], indirect=True)
def test_first_use_config_pending_fence_refuses_observation(retained_case, tmp_path):
    from tldw_chatbook.Backup_Recovery.control_records import register_pending

    _, _, selector, _, authority = retained_case
    register_pending(
        authority.control_root.parent,
        "pending-first-use",
        ("local",),
        tmp_path / "pending-control",
        (selector,),
    )
    with pytest.raises(ValueError, match="retained_config_binding_required"):
        restore_plan.retained_config_observation(selector)


@pytest.mark.parametrize("retained_case", ["never-bound"], indirect=True)
def test_first_use_config_orphan_activation_refuses_observation(
    retained_case, tmp_path
):
    _, _, selector, _, authority = retained_case
    # A real on-disk orphaned half-pair must never become first-use eligibility.
    association = {
        "version": 1,
        "selector": str(selector),
        "activation": {
            "operation_id": "earlier-operation",
            "generation": "earlier-generation",
            "owners": ["ui.state"],
            "namespaces": ["local"],
            "store_root": str(tmp_path / "prior-activation"),
        },
    }
    path = authority.control_root.parent / (
        "activation-" + hashlib.sha256(str(selector).encode()).hexdigest() + ".json"
    )
    path.write_text(json.dumps(association))
    path.chmod(0o600)
    with pytest.raises(ValueError, match="retained_config_activation_changed"):
        restore_plan.retained_config_observation(selector)


def test_local_config_destination_semantics_refuse_foreign_sibling(
    retained_case, tmp_path
):
    archive, target, selector, _, authority = retained_case
    relation = retained_plan(retained_case).retained_configs[0]
    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    plan = restore_plan.plan_restore(
        archive,
        mode="replace",
        destinations={"state-root": other},
        target=target,
        retained_configs=(relation,),
    )
    before = (selector.read_bytes(), selector.stat().st_ino)
    with (
        authority.maintenance(("local", UNBOUND_NAMESPACE), 3) as session,
        pytest.raises(ValueError, match="owner_relocation_unverified:ui.state"),
    ):
        stage_restore(archive, plan, tmp_path / "work", Event(), session=session)
    assert (selector.read_bytes(), selector.stat().st_ino) == before
    assert not list(other.iterdir())


def test_native_stage_requires_all_bound_namespaces(retained_case, tmp_path):
    archive, _, _, _, authority = retained_case
    plan = retained_plan(retained_case)
    with (
        authority.maintenance((UNBOUND_NAMESPACE,), 3) as session,
        pytest.raises(ValueError, match="retained_config_native_scope_required"),
    ):
        stage_restore(archive, plan, tmp_path / "work", Event(), session=session)


def test_pre_selection_plan_serialization_remains_compatible(tmp_path):
    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.plan_records import load_plan

    archive = sealed(tmp_path)
    plan = restore_plan.plan_restore(
        archive,
        mode="isolated",
        destinations={"root": tmp_path / "new"},
        target=None,
    )
    (tmp_path / "control").mkdir(mode=0o700)
    journal = Journal(tmp_path / "control", "legacy-plan")
    stage_restore(archive, plan, tmp_path / "work", Event(), journal=journal)
    encoded = json.loads((journal.root / "restore-plan.json").read_bytes())
    assert (
        not {
            "retained_configs",
            "requested_groups",
            "effective_groups",
            "required_groups",
        }
        & encoded.keys()
    )
    assert load_plan(journal) == plan


def test_pre_selection_plan_digest_matches_existing_receipt_format():
    plan = restore_plan.RestorePlan(
        "a" * 64,
        "replace",
        (("old-root", PurePosixPath("/private/tmp/legacy-data")),),
        (),
        (),
        "b" * 64,
    )
    # Frozen output of the pre-selection implementation at 31a9cc490e.
    assert publication._plan_digest(plan) == (
        "1bf9e53c21974566eb6c0f2978a31e37943abc714162324f0035a4a73ec5bf24"
    )


@pytest.mark.parametrize(
    "retained_case,config_changed",
    [
        ("directory", False),
        ("selector-only", False),
        ("directory", True),
        ("directory", "review-content"),
        ("directory", "review-identity"),
    ],
    indirect=["retained_case"],
)
def test_native_replacement_and_later_rollback_keep_config_identity(
    retained_case,
    tmp_path,
    monkeypatch,
    helper_resource_root,
    config_changed,
):
    from keyring.backends.null import Keyring

    from tldw_chatbook.Backup_Recovery import (
        credentials,
        crypto,
        later_rollback,
        replacement,
    )
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    credential_store = KeyringServerCredentialStore(keyring_backend=Keyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: credential_store)
    archive, target, selector, state, authority = retained_case
    plan = retained_plan(retained_case)
    before = (selector.read_bytes(), selector.stat().st_ino)
    original_state = state.read_bytes()
    with authority.maintenance(("local", UNBOUND_NAMESPACE), 3) as session:
        candidate = stage_restore(
            archive, plan, tmp_path / "work", Event(), session=session
        )
    operation = replacement.replace(
        plan,
        candidate,
        control_root=tmp_path / "control",
        rollback_password=b"test-retained-config",
        cancel=Event(),
    )
    assert state.read_bytes() == b'[state]\nselected="incoming"\n'
    assert (selector.read_bytes(), selector.stat().st_ino) == before
    journal = Journal(tmp_path / "control", operation)
    with journal._locked(exclusive=False) as parent:
        rows = journal._records(parent)
    prepared = next(row.evidence for row in rows if row.event == "prepared")
    assert [row["target"] for row in prepared["artifacts"]] == [str(state)]
    assert prepared["safety_sources"][0]["source"]["path"] == str(selector)
    owners = {owner.owner_id: owner for owner in recovery_adapters()}
    context = {
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(
            selector, target.items[0].logical_id.split(":")[1]
        )
    }
    current = replace(
        target,
        items=tuple(
            owners[owner].discover(context)[0] for owner in ("config", "ui.state")
        ),
    )
    if config_changed is True:
        selector.write_bytes(b'[general]\nusers_name="Later local edit"\n')
        changed = (selector.read_bytes(), selector.stat().st_ino)
        with pytest.raises(ValueError, match="retained_config_binding_required"):
            later_rollback.preview_rollback(
                operation,
                control_root=tmp_path / "control",
                old_password=b"test-retained-config",
                target=current,
                cancel=Event(),
            )
        assert (selector.read_bytes(), selector.stat().st_ino) == changed
        assert state.read_bytes() == b'[state]\nselected="incoming"\n'
        return
    rollback = later_rollback.preview_rollback(
        operation,
        control_root=tmp_path / "control",
        old_password=b"test-retained-config",
        target=current,
        cancel=Event(),
    )
    assert {path for _, path in rollback.restore} == {state}
    assert rollback.retained_configs[0].config_path == selector
    if config_changed in {"review-content", "review-identity"}:
        if config_changed == "review-content":
            selector.write_bytes(selector.read_bytes() + b"# after review\n")
        else:
            replacement_config = selector.with_name("replacement.toml")
            replacement_config.write_bytes(selector.read_bytes())
            replacement_config.chmod(0o600)
            replacement_config.replace(selector)
        changed = (selector.read_bytes(), selector.stat().st_ino)
        unchanged_state = (state.read_bytes(), state.stat().st_ino)
        with pytest.raises(ValueError, match="retained_config_changed"):
            later_rollback.execute_rollback(
                operation,
                control_root=tmp_path / "control",
                old_password=b"test-retained-config",
                new_password=b"test-second-safety",
                cancel=Event(),
                approved_plan=rollback,
            )
        assert (selector.read_bytes(), selector.stat().st_ino) == changed
        assert (state.read_bytes(), state.stat().st_ino) == unchanged_state
        return
    later_rollback.execute_rollback(
        operation,
        control_root=tmp_path / "control",
        old_password=b"test-retained-config",
        new_password=b"test-second-safety",
        cancel=Event(),
        approved_plan=rollback,
    )
    assert state.read_bytes() == original_state
    assert (selector.read_bytes(), selector.stat().st_ino) == before


_RECOVER_RETAINED = r"""
import os, sys
from pathlib import Path
from threading import Event
from Tests.network_guard import install, blocked_attempts
install()
root, helper, action, operation = sys.argv[1:]
root = Path(root)
os.environ['TLDW_CONFIG_PATH'] = str(root / 'live' / 'config.toml')
from tldw_chatbook.Backup_Recovery import bootstrap, crypto, credentials, publication, replacement
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.plan_records import load_plan
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
from keyring.backends.null import Keyring
bootstrap.default_bootstrap_root = lambda: root / 'bootstrap'
crypto._package_resource_root = lambda: Path(helper)
store = KeyringServerCredentialStore(keyring_backend=Keyring())
credentials._credential_store = lambda: store
if action == 'start':
    journal = Journal(root / 'control', 'retained-config-source')
    plan = load_plan(journal)
    with journal._locked(exclusive=False) as parent:
        candidate = Path(journal._records(parent)[0].evidence['stage']['path'])
    publish = publication.publish_new
    def stop_after_native_publish(source, destination, **kwargs):
        publish(source, destination, **kwargs)
        if destination == root / 'live' / 'ui_state.toml':
            os._exit(91)
    publication.publish_new = stop_after_native_publish
    replacement.replace(plan, candidate, control_root=root / 'control', rollback_password=b'retained-recovery', cancel=Event())
    raise AssertionError('native crash boundary not reached')
else:
    result = replacement.recover_replacement(operation, control_root=root / 'control', action=action, rollback_password=b'retained-recovery', cancel=Event())
    assert result == ('committed' if action == 'finish' else 'rolled_back'), result
assert not blocked_attempts(), blocked_attempts()
"""


@pytest.mark.parametrize("action", ["finish", "rollback"])
def test_fresh_process_finish_and_abort_preserve_config(
    retained_case,
    tmp_path,
    helper_resource_root,
    action,
):
    import subprocess
    import sys

    from tldw_chatbook.Backup_Recovery.journal import Journal

    archive, _, selector, state, authority = retained_case
    before = (selector.read_bytes(), selector.stat().st_ino)
    original_state = state.read_bytes()
    plan = retained_plan(retained_case)
    (tmp_path / "control").mkdir(mode=0o700)
    journal = Journal(tmp_path / "control", "retained-config-source")
    with authority.maintenance(("local", UNBOUND_NAMESPACE), 3) as session:
        stage_restore(
            archive, plan, tmp_path / "work", Event(), session=session, journal=journal
        )

    def child(command, operation):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                _RECOVER_RETAINED,
                str(tmp_path),
                str(helper_resource_root),
                command,
                operation,
            ],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )

    crashed = child("start", "")
    assert crashed.returncode == 91, crashed.stderr[-4000:]
    assert (selector.read_bytes(), selector.stat().st_ino) == before
    pending, _ = bootstrap._records(tmp_path / "bootstrap")
    assert len(pending) == 1
    resumed = child(action, pending[0]["operation_id"])
    assert resumed.returncode == 0, resumed.stderr[-5000:]
    assert (selector.read_bytes(), selector.stat().st_ino) == before
    assert state.read_bytes() == (
        b'[state]\nselected="incoming"\n' if action == "finish" else original_state
    )
    assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
