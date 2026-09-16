"""Settings staging binds preservation evidence for finalization's native recheck."""

import json
from threading import Event

import pytest
import toml

from Tests.Backup_Recovery.test_preserved_group_paths import (
    preserved_case as preserved_case,  # noqa: PLC0414 - pytest fixture reuse
)
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery import publication
from tldw_chatbook.Backup_Recovery.control_records import UNBOUND_NAMESPACE
from tldw_chatbook.Backup_Recovery.journal import _Object, observe_artifact
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.staging import stage_restore


@pytest.fixture
def staged_settings(preserved_case, tmp_path):
    template, data, selector, store, authority = preserved_case

    def config_document(doc):
        doc["owners"][0]["owner_id"] = "config"
        doc["files"][0].update(
            owner_id="config", logical_id="profile:profile:config",
            relative_path="config.toml",
        )
        doc["directories"][0]["synthetic"] = True
        doc["producer_inventory"] = [
            {"logical_id": "root", "owner_id": "config", "status": "included_directory", "dependencies": []},
            {"logical_id": "profile:profile:config", "owner_id": "config", "status": "included", "dependencies": []},
        ]
        doc["dependency_groups"][0]["members"] = ["profile:profile:config"]

    archive = sealed(tmp_path, data=toml.dumps(data).encode(), mutate=config_document)
    plan = plan_restore(
        archive, mode="replace", target=template.target, data_groups=("settings",),
        destinations={"root": selector.parent}, profile_names={"profile": "Local"},
    )
    with authority.maintenance(("local", UNBOUND_NAMESPACE), 3) as session:
        candidate = stage_restore(
            archive, plan, tmp_path / "candidate", Event(), session=session,
        )
        record = _Object.model_validate(observe_artifact(candidate / "candidate.json"))
        yield plan, candidate, session, record, store


def check(case):
    plan, candidate, session, record, _ = case
    publication._check_preserved_settings(plan, session, candidate, record)


def test_staged_preservation_survives_repeated_native_rechecks(staged_settings):
    check(staged_settings)
    check(staged_settings)


@pytest.mark.parametrize("change", ["content", "inode", "wal", "shm", "journal", "parent"])
def test_staged_preservation_refuses_later_unselected_changes(staged_settings, change):
    check(staged_settings)
    store = staged_settings[-1]
    if change == "content":
        with store.open("ab") as output:
            output.write(b"changed")
    elif change == "inode":
        other = store.with_name("replacement.db")
        other.write_bytes(store.read_bytes())
        other.chmod(0o600)
        other.replace(store)
    elif change == "parent":
        prior = store.parent.with_name("previous-parent")
        store.parent.rename(prior)
        store.parent.mkdir(mode=0o700)
        (prior / store.name).rename(store)
    else:
        store.with_name(store.name + "-" + change).write_bytes(b"late companion")
    with pytest.raises(ValueError, match="preserved_group_state_changed"):
        check(staged_settings)


def test_tampered_candidate_cannot_replace_staged_preservation(staged_settings):
    plan, candidate, _, _, _ = staged_settings
    path = candidate / "candidate.json"
    doc = json.loads(path.read_bytes())
    doc["preserved_settings_fingerprint"] = "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="candidate_receipt_changed"):
        check(staged_settings)
    assert plan.effective_groups == ("settings",)


@pytest.mark.parametrize("proof", [None, "", "x" * 64])
def test_candidate_without_valid_preservation_refuses_before_publication(staged_settings, proof):
    plan, candidate, _, _, _ = staged_settings
    path = candidate / "candidate.json"
    doc = json.loads(path.read_bytes())
    if proof is None:
        del doc["preserved_settings_fingerprint"]
    else:
        doc["preserved_settings_fingerprint"] = proof
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="preserved_settings_proof_required"):
        publication._descriptor(candidate, plan)
