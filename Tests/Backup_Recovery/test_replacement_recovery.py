"""Real durable replacement recovery uses checked local originals and native moves."""

from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from tldw_chatbook.Backup_Recovery import bootstrap, crypto, publication, replacement
from tldw_chatbook.Backup_Recovery.journal import Journal


def test_actual_candidate_receipt_retains_complete_typed_plan(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.plan_records import load_plan

    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        assert load_plan(case[2]) == case[1]


def test_interrupted_original_retirement_can_really_reverse(
    tmp_path, monkeypatch, helper_resource_root
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, _, _, source, selector = case
        plan = replace(
            plan, acknowledged_credential_issues=("credential_format_unreadable",)
        )
        originals = {
            path: path.read_bytes()
            for path in (
                source,
                selector,
                Path(str(source) + "-wal"),
                Path(str(source) + "-shm"),
            )
        }
        retire = publication._retire

        def interrupted(item):
            retire(item)
            raise InterruptedError("after native original retirement")

        monkeypatch.setattr(publication, "_retire", interrupted)
        with pytest.raises(InterruptedError):
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        monkeypatch.setattr(publication, "_retire", retire)
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        operation = pending[0]["operation_id"]
        outcome = replacement.recover_replacement(
            operation,
            control_root=tmp_path / "control",
            action="rollback",
            rollback_password=b"rollback",
            cancel=Event(),
        )
        assert outcome == "rolled_back"
        assert all(path.read_bytes() == data for path, data in originals.items())
        assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            assert journal._records(parent)[-1].event == "rolled_back"


def test_installed_validation_failure_reverses_under_the_same_session(
    tmp_path, monkeypatch, helper_resource_root
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, _, _, source, selector = case
        plan = replace(
            plan, acknowledged_credential_issues=("credential_format_unreadable",)
        )
        original = {
            path: path.read_bytes()
            for path in (
                source,
                selector,
                Path(str(source) + "-wal"),
                Path(str(source) + "-shm"),
            )
        }
        seen = []
        reverse = replacement._rollback_replacement

        def checked(journal, prepared, session, check_credentials, cancel):
            session._check()
            assert not bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
            seen.append(id(session))
            return reverse(journal, prepared, session, check_credentials, cancel)

        monkeypatch.setattr(replacement, "_rollback_replacement", checked)
        validate = publication._validate_installed

        def rejected(*args):
            validate(*args)
            raise ValueError("installed owner rejection")

        monkeypatch.setattr(publication, "_validate_installed", rejected)
        with pytest.raises(ValueError, match="replacement_rolled_back"):
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        assert seen and all(
            path.read_bytes() == data for path, data in original.items()
        )
        assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]


_CHILD = r"""
import os,sys,json
from pathlib import Path
from threading import Event
from dataclasses import replace
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery import bootstrap,crypto,publication,replacement
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.plan_records import load_plan
root,helper,mode,action,boundary,operation=map(str,sys.argv[1:])
root=Path(root)
bootstrap.default_bootstrap_root=lambda:root/'bootstrap'
os.environ['TLDW_CONFIG_PATH']=str(root/'live/config.toml')
crypto._package_resource_root=lambda:Path(helper)
if boundary=='retire_native':
 original=publication._retire
 def retire(*args,**kwargs):
  original(*args,**kwargs);os._exit(91)
 publication._retire=retire
if boundary=='publish_native':
 original=publication.publish_new
 def publish(*args,**kwargs):
  original(*args,**kwargs);os._exit(91)
 publication.publish_new=publish
if boundary in ('move_observed','activation_recorded','rolled_back','committed'):
 original=Journal._append
 def append(self,parent,event,evidence):
  original(self,parent,event,evidence)
  if event==boundary:os._exit(91)
 Journal._append=append
if boundary=='metadata':
 original=publication._installed_metadata
 def metadata(*args,**kwargs):
  original(*args,**kwargs);os._exit(91)
 publication._installed_metadata=metadata
if boundary=='activation_pair':
 from tldw_chatbook.Backup_Recovery import control_records
 original=control_records._publish_activation_record
 def pair(*args,**kwargs):
  original(*args,**kwargs);os._exit(91)
 control_records._publish_activation_record=pair
if boundary=='activation_requirement':
 from tldw_chatbook.Backup_Recovery.activation import ActivationStore
 original=ActivationStore.require
 def require(*args,**kwargs):
  original(*args,**kwargs);os._exit(91)
 ActivationStore.require=require
if boundary=='reverse_native':
 original=publication._reverse_native_move
 def reverse(*args,**kwargs):
  original(*args,**kwargs);os._exit(91)
 publication._reverse_native_move=reverse
if mode=='start':
 journal=Journal(root/'control','held-sqlite')
 plan=load_plan(journal)
 with journal._locked(exclusive=False) as parent:
  candidate=Path(journal._records(parent)[0].evidence['stage']['path'])
 plan=replace(plan,acknowledged_credential_issues=('credential_format_unreadable',))
 replacement.replace(plan,candidate,control_root=root/'control',rollback_password=b'rollback',cancel=Event())
else:
 result=replacement.recover_replacement(operation,control_root=root/'control',action=action,rollback_password=b'rollback',cancel=Event())
 assert result==('rolled_back' if action=='rollback' else 'committed'),result
 from tldw_chatbook.Backup_Recovery.isolated_restore import installation_client_id
 assert len(installation_client_id())==32
 assert not blocked_attempts(),blocked_attempts()
 print(result)
"""


def _child(tmp_path, helper, mode, action, boundary, operation="", *, expected=0):
    import subprocess
    import sys

    log = tmp_path / ("child-" + mode + "-" + action + "-" + boundary + ".log")
    with log.open("w+") as output:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _CHILD,
                str(tmp_path),
                str(helper),
                mode,
                action,
                boundary,
                operation,
            ],
            stdout=output,
            stderr=output,
            timeout=35,
            check=False,
        )
        output.seek(0)
        assert result.returncode == expected, output.read()[-7000:]


@pytest.mark.parametrize(
    "boundary",
    [
        "retire_native",
        "publish_native",
        "move_observed",
        "metadata",
        "activation_recorded",
        "activation_pair",
        "activation_requirement",
    ],
)
@pytest.mark.parametrize("action", ["finish", "rollback"])
def test_fresh_process_reconciles_real_killed_publication(
    tmp_path, monkeypatch, helper_resource_root, boundary, action
):
    with replacement_case(
        tmp_path, monkeypatch, prepared=False, tree=boundary == "metadata"
    ) as case:
        source, selector = case[4:]
        original = {
            path: path.read_bytes()
            for path in (
                source,
                selector,
                Path(str(source) + "-wal"),
                Path(str(source) + "-shm"),
            )
        }
        _child(tmp_path, helper_resource_root, "start", action, boundary, expected=91)
        pending = _pending_rows(tmp_path)
        assert (
            len(pending) == 1
            and not bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
        )
        operation = pending[0]["operation_id"]
        if action == "finish" and boundary == "retire_native":
            import shutil

            shutil.rmtree(tmp_path / "input")
            (tmp_path / "replacement.zip").unlink()
        _child(tmp_path, helper_resource_root, "recover", action, "none", operation)
        assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
        if action == "rollback":
            assert all(path.read_bytes() == data for path, data in original.items())
        else:
            import sqlite3
            from contextlib import closing

            with closing(sqlite3.connect(source)) as db:
                assert db.execute("SELECT query FROM research_runs").fetchall() == [
                    ("old",)
                ]
            assert b"broken" not in selector.read_bytes()
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            assert journal._records(parent)[-1].event == (
                "rolled_back" if action == "rollback" else "committed"
            )


@pytest.mark.parametrize(
    "boundary",
    [
        "reverse_native",
        "metadata",
        "rolled_back",
        "activation_pair",
        "activation_requirement",
    ],
)
def test_fresh_process_resumes_interrupted_reverse(
    tmp_path, monkeypatch, helper_resource_root, boundary
):
    with replacement_case(
        tmp_path, monkeypatch, prepared=False, tree=boundary == "metadata"
    ) as case:
        source, selector = case[4:]
        original = {
            path: path.read_bytes()
            for path in (
                source,
                selector,
                Path(str(source) + "-wal"),
                Path(str(source) + "-shm"),
            )
        }
        _child(
            tmp_path,
            helper_resource_root,
            "start",
            "rollback",
            "activation_recorded",
            expected=91,
        )
        pending = _pending_rows(tmp_path)
        operation = pending[0]["operation_id"]
        _child(
            tmp_path,
            helper_resource_root,
            "recover",
            "rollback",
            boundary,
            operation,
            expected=91,
        )
        assert not bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
        _child(tmp_path, helper_resource_root, "recover", "rollback", "none", operation)
        assert all(path.read_bytes() == data for path, data in original.items())
        assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]


@pytest.mark.parametrize("damage", ["password", "retained", "plan", "sibling"])
def test_recovery_refuses_changed_local_evidence_before_any_reverse(
    tmp_path, monkeypatch, helper_resource_root, damage
):
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        selector = case[-1]
        _child(
            tmp_path,
            helper_resource_root,
            "start",
            "rollback",
            "retire_native",
            expected=91,
        )
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        operation = pending[0]["operation_id"]
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        intent = rows[-1].evidence
        retained = Path(intent["destination"])
        if damage == "retained":
            retained.write_bytes(b"foreign original")
        elif damage == "plan":
            (journal.root / "restore-plan.json").write_bytes(b"{}")
        elif damage == "sibling":
            (retained.parent / "unexpected").write_bytes(b"unrecorded topology")
        before = {
            path: path.read_bytes() for path in (retained, selector) if path.is_file()
        }
        monkeypatch.setattr(
            crypto, "_package_resource_root", lambda: helper_resource_root
        )
        with pytest.raises((ValueError, RuntimeError)):
            replacement.recover_replacement(
                operation,
                control_root=tmp_path / "control",
                action="rollback",
                rollback_password=b"wrong" if damage == "password" else b"rollback",
                cancel=Event(),
            )
        assert all(path.read_bytes() == value for path, value in before.items())
        assert not bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
        with journal._locked(exclusive=False) as parent:
            assert not any(
                row.event == "rollback_started" for row in journal._records(parent)
            )


@pytest.mark.parametrize("drift", [False, True])
def test_actual_original_credentials_reuse_or_remap_and_new_scopes_are_retained(
    tmp_path, monkeypatch, helper_resource_root, drift
):
    from Tests.Backup_Recovery.test_replacement import _credential_candidate

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, plan, store, backend, targets = _credential_candidate(
            case, tmp_path, monkeypatch
        )
        original = {path: path.read_bytes() for path in (case[-1], targets)}
        finalize = publication.finalize_candidate

        def stopped(*args, **kwargs):
            raise KeyboardInterrupt("process interruption before finalization")

        monkeypatch.setattr(publication, "finalize_candidate", stopped)
        with pytest.raises(KeyboardInterrupt):
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        monkeypatch.setattr(publication, "finalize_candidate", finalize)
        pending, _ = bootstrap._records(tmp_path / "bootstrap")
        operation = pending[0]["operation_id"]
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        prepared = next(row.evidence for row in rows if row.event == "prepared")
        scoped = dict(backend.values)
        if drift:
            store.set_secret("peer", "api_key", "changed-old-secret")
            assert (
                replacement.recover_replacement(
                    operation,
                    control_root=tmp_path / "control",
                    action="rollback",
                    rollback_password=b"rollback",
                    cancel=Event(),
                )
                == "rolled_back"
            )
            import json

            purpose = json.loads(targets.read_bytes())["targets"][0][
                "auth_reference"
            ].removeprefix("keyring:")
            assert store.get_secret("peer", purpose) == "current-shared-secret"
            assert store.get_secret("peer", "api_key") == "changed-old-secret"
            assert case[-1].read_bytes() == original[case[-1]]
            assert bootstrap.startup_permission(case[-1], tmp_path / "bootstrap")[0]
        else:
            assert (
                replacement.recover_replacement(
                    operation,
                    control_root=tmp_path / "control",
                    action="rollback",
                    rollback_password=b"rollback",
                    cancel=Event(),
                )
                == "rolled_back"
            )
            assert all(path.read_bytes() == value for path, value in original.items())
            assert backend.values == scoped
            with journal._locked(exclusive=False) as parent:
                started = next(
                    row
                    for row in journal._records(parent)
                    if row.event == "rollback_started"
                )
            assert started.evidence["retained_credential_scopes"] == sorted(
                prepared["credential_scopes"]
            )


_NATIVE_REVERSE = r"""
import asyncio, os, sqlite3, subprocess, sys, time
from dataclasses import replace
from pathlib import Path
from threading import Event
import pytest
from Tests import network_guard
network_guard.install()
from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from tldw_chatbook.Backup_Recovery import bootstrap, crypto, publication, replacement
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.rag_inventory import _Projections
from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant
from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
original={}
base=Path.home()/'case';base.mkdir(mode=0o700)
def extras(live):
 root=live/'chromadb'
 store=ChromaVectorStore(root,collection_name='retained')
 store.add([str(i) for i in range(1100)],[[1.,0.]]*1100,['original']*1100,[{'doc_id':str(i)} for i in range(1100)])
 async def drain():
  participant._maintenance_close_admission()
  assert await participant._maintenance_drain(time.monotonic()+10)
 asyncio.run(drain())
 subprocess.run([sys.executable,'-c',"import os,sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA wal_autocheckpoint=0'); c.execute(\"UPDATE collections SET name='wal-retained'\"); c.commit(); os._exit(0)",str(root/'chroma.sqlite3')],check=True,timeout=15)
 original.update({str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()})
 assert original['chroma.sqlite3-wal']
 config={DISCOVERY_CONTEXT_KEY:DiscoveryContext(live/'config.toml','profile')}
 return tuple(replace(item,status='intentionally_excluded',dependencies=(*item.dependencies,'profile:profile:research.local')) for item in _Projections('rag.projections')._tree(config,root))
with pytest.MonkeyPatch.context() as patch:
 patch.setattr(crypto,'_package_resource_root',lambda:Path(HELPER_ROOT))
 with replacement_case(base,patch,extras=extras,prepared=False) as case:
  candidate,plan,_,_,source,selector=case
  root=source.parent/'chromadb'
  plan=replace(plan,acknowledged_credential_issues=('credential_format_unreadable',))
  retire=publication._retire
  def interrupted(item):
   retire(item)
   if item.target==str(root):raise InterruptedError('after native projection retirement')
  patch.setattr(publication,'_retire',interrupted)
  try:replacement.replace(plan,candidate,control_root=base/'control',rollback_password=b'rollback',cancel=Event())
  except InterruptedError:pass
  else:raise AssertionError('real projection retirement did not interrupt')
  patch.setattr(publication,'_retire',retire)
  assert not root.exists()
  pending,_=bootstrap._records(base/'bootstrap')
  assert replacement.recover_replacement(pending[0]['operation_id'],control_root=base/'control',action='rollback',rollback_password=b'rollback',cancel=Event())=='rolled_back'
  assert {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}==original
  assert bootstrap.startup_permission(selector,base/'bootstrap')[0]
  # Native reopen is the fixture's private read probe; ordinary restored retrieval
  # remains gated on the new generation's projection reconciliation review.
  import chromadb
  from chromadb.config import Settings
  client=chromadb.PersistentClient(path=str(root),settings=Settings(anonymized_telemetry=False,migrations='validate'))
  try:
   collection=client.get_collection('wal-retained',embedding_function=None)
   assert collection.count()==1100
   row=collection.get(ids=['17'],include=['embeddings','documents'])
   assert row['documents']==['original'] and row['embeddings'].tolist()==[[1.,0.]]
  finally:client.close()
 assert not network_guard.blocked_attempts(),network_guard.blocked_attempts()
print('retired and reopened')
"""


def test_actual_native_projection_group_is_reversed_and_readable(
    tmp_path, helper_resource_root
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(
        tmp_path,
        "native",
        "reverse",
        script="HELPER_ROOT="
        + repr(str(helper_resource_root))
        + "\n"
        + _NATIVE_REVERSE,
    )


def _pending_rows(tmp_path):
    import json

    return [
        json.loads(path.read_text())
        for path in (tmp_path / "bootstrap").glob("pending-*.json")
    ]


def test_committed_remaining_fence_only_allows_finalize_retry(
    tmp_path, monkeypatch, helper_resource_root
):
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        _child(
            tmp_path, helper_resource_root, "start", "finish", "committed", expected=91
        )
        operation = _pending_rows(tmp_path)[0]["operation_id"]
        monkeypatch.setattr(
            crypto, "_package_resource_root", lambda: helper_resource_root
        )
        with pytest.raises(ValueError, match="later_rollback_required"):
            replacement.recover_replacement(
                operation,
                control_root=tmp_path / "control",
                action="rollback",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        assert not bootstrap.startup_permission(case[-1], tmp_path / "bootstrap")[0]
        _child(tmp_path, helper_resource_root, "recover", "finish", "none", operation)
        assert bootstrap.startup_permission(case[-1], tmp_path / "bootstrap")[0]


def test_typed_move_intent_rejects_foreign_mapping_and_public_append(
    tmp_path, monkeypatch, helper_resource_root
):
    from copy import deepcopy

    from tldw_chatbook.Backup_Recovery.journal import _validate

    with replacement_case(tmp_path, monkeypatch, prepared=False):
        _child(
            tmp_path,
            helper_resource_root,
            "start",
            "rollback",
            "retire_native",
            expected=91,
        )
        journal = Journal(
            tmp_path / "control", _pending_rows(tmp_path)[0]["operation_id"]
        )
        with journal._locked(exclusive=False) as parent:
            records = journal._records(parent)
        for field in ("destination", "source", "parents"):
            proof = deepcopy(records[-1].evidence)
            if field == "destination":
                proof[field] = str(tmp_path / "foreign")
            elif field == "source":
                proof[field]["inode"] += 1
            else:
                proof[field].append(proof[field][0])
            with pytest.raises(ValueError, match="journal_evidence_invalid"):
                _validate("move_intended", proof, records[:-1])
        with pytest.raises(ValueError, match="recovery_execution_required"):
            journal.record("move_intended", records[-1].evidence)


def test_finish_validation_failure_reverses_with_the_unlocked_password(
    tmp_path, monkeypatch, helper_resource_root
):
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        source, selector = case[4:]
        original = {
            path: path.read_bytes()
            for path in (
                source,
                selector,
                Path(str(source) + "-wal"),
                Path(str(source) + "-shm"),
            )
        }
        _child(
            tmp_path,
            helper_resource_root,
            "start",
            "finish",
            "publish_native",
            expected=91,
        )
        operation = _pending_rows(tmp_path)[0]["operation_id"]
        monkeypatch.setattr(
            crypto, "_package_resource_root", lambda: helper_resource_root
        )
        validate = publication._validate_installed

        def rejected(*args):
            validate(*args)
            raise ValueError("installed validation refusal")

        monkeypatch.setattr(publication, "_validate_installed", rejected)
        with pytest.raises(ValueError, match="replacement_rolled_back"):
            replacement.recover_replacement(
                operation,
                control_root=tmp_path / "control",
                action="finish",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        assert all(path.read_bytes() == value for path, value in original.items())
        assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]


def test_foreign_activation_intent_cannot_authorize_pair_repair(
    tmp_path, monkeypatch, helper_resource_root
):
    import json

    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        _child(
            tmp_path,
            helper_resource_root,
            "start",
            "finish",
            "activation_pair",
            expected=91,
        )
        operation = _pending_rows(tmp_path)[0]["operation_id"]
        intent = next((tmp_path / "bootstrap").glob("activation-update-*.json"))
        document = json.loads(intent.read_text())
        document["after"][0]["activation"]["generation"] = "foreign"
        intent.write_text(json.dumps(document))
        paths = [*(tmp_path / "bootstrap").glob("*.json"), case[-1]]
        before = {path: path.read_bytes() for path in paths}
        monkeypatch.setattr(
            crypto, "_package_resource_root", lambda: helper_resource_root
        )
        with pytest.raises(ValueError, match="activation_recovery_context_invalid"):
            replacement.recover_replacement(
                operation,
                control_root=tmp_path / "control",
                action="finish",
                rollback_password=b"rollback",
                cancel=Event(),
            )
        assert all(path.read_bytes() == value for path, value in before.items())
        assert not bootstrap.startup_permission(case[-1], tmp_path / "bootstrap")[0]


def test_replacement_does_not_read_or_capture_unselected_sibling_payloads(
    tmp_path, monkeypatch, helper_resource_root
):
    import os
    import stat

    def extras(live):
        os.mkfifo(live / "unrelated-stream", 0o600)
        (live / "unrelated-link").symlink_to(live / "absent-external-target")
        return ()

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False, extras=extras) as case:
        candidate, plan = case[:2]
        plan = replace(
            plan, acknowledged_credential_issues=("credential_format_unreadable",)
        )
        operation = replacement.replace(
            plan,
            candidate,
            control_root=tmp_path / "control",
            rollback_password=b"rollback",
            cancel=Event(),
        )
        assert stat.S_ISFIFO((tmp_path / "live/unrelated-stream").lstat().st_mode)
        assert (tmp_path / "live/unrelated-link").is_symlink()
        journal = Journal(tmp_path / "control", operation)
        with journal._locked(exclusive=False) as parent:
            rollback = next(
                row
                for row in journal._records(parent)
                if row.event == "rollback_verified"
            )
        assert not any("unrelated" in key for key in rollback.evidence["coverage"])


def test_rollback_identity_uses_exact_profile_binding_with_shared_held_namespace(
    tmp_path, monkeypatch, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        admission_authority(tmp_path / "bootstrap").register(
            "shared", (case[-1].parent,)
        )
        _child(
            tmp_path,
            helper_resource_root,
            "start",
            "rollback",
            "retire_native",
            expected=91,
        )
        pending = _pending_rows(tmp_path)[0]
        assert pending["namespaces"] == ["profile", "shared"]
        _child(
            tmp_path,
            helper_resource_root,
            "recover",
            "rollback",
            "none",
            pending["operation_id"],
        )
        assert bootstrap.startup_permission(case[-1], tmp_path / "bootstrap")[0]


def test_committed_forward_identity_uses_exact_binding_with_shared_held_namespace(
    tmp_path, monkeypatch, helper_resource_root
):
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        admission_authority(tmp_path / "bootstrap").register(
            "shared", (case[-1].parent,)
        )
        _child(
            tmp_path, helper_resource_root, "start", "finish", "committed", expected=91
        )
        pending = _pending_rows(tmp_path)[0]
        assert pending["namespaces"] == ["profile", "shared"]
        _child(
            tmp_path,
            helper_resource_root,
            "recover",
            "finish",
            "none",
            pending["operation_id"],
        )
        assert bootstrap.startup_permission(case[-1], tmp_path / "bootstrap")[0]
