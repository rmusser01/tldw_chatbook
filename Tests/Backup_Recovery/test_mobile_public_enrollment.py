"""Actual unbound populated start must pre-admit its native source re-reads."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_mobile_public_operator import _START_CANONICAL

_POPULATED = r"""
import json,sys
from pathlib import Path
from Tests.real_profile_guard import install as install_profile_guard,take_violations
install_profile_guard()
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Utils.launch_options import adopt_config_flag
adopt_config_flag(sys.argv[1:])
from tldw_chatbook.Backup_Recovery import bootstrap,mobile_consolidation,recovery_service
from tldw_chatbook.Backup_Recovery.inventory import read_profile_config
from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path,legacy_mobile_dir
from tldw_chatbook.Backup_Recovery.mobile_consolidation import CanonicalConsoleStores
from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed
from Tests.Backup_Recovery.mobile_consolidation_fixture import fingerprints
selector=effective_config_path()
source=legacy_mobile_dir(read_profile_config(selector))
_small_native_seed(CanonicalConsoleStores(*(source/name for name in ('chat.db','workspaces.db','runs.db'))),legacy=True)
before=fingerprints(source)
pending,profiles=bootstrap._records(bootstrap.default_bootstrap_root())
assert not pending
assert bootstrap._binding(selector,profiles,bootstrap._registry(bootstrap.default_bootstrap_root())) is None
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService,default_control_root
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
from tldw_chatbook.Backup_Recovery import credentials
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
class EmptyIsolatedKeyring:
 def get_password(self,service,username):return None
fixture_credentials=KeyringServerCredentialStore(keyring_backend=EmptyIsolatedKeyring())
credentials._credential_store=lambda:fixture_credentials
service=RecoveryService(default_control_root())
preview=service.preview_mobile_consolidation(selector,selector)
assert preview.has_data and not preview.issues,preview.issues
budgets=[]
costs={}
limits=[]
native_init=_TraceReadBudget.__init__
def issued(self):
 native_init(self)
 budgets.append(self)
_TraceReadBudget.__init__=issued
native_reserve=_TraceReadBudget.reserve
def charged(self,amount):
 import traceback
 frame=sys._getframe(1)
 key=(Path(frame.f_code.co_filename).name,frame.f_lineno)
 costs[key]=costs.get(key,0)+amount
 try:return native_reserve(self,amount)
 except Exception:
  if not limits:limits.append({'remaining':self._remaining,'requested':amount,'stack':traceback.format_stack(limit=12)})
  raise
_TraceReadBudget.reserve=charged
observations=[]
inspection_errors=[]
native_inspect=mobile_consolidation._inspect_store
def checked(path,adapter,**kwargs):
 from tldw_chatbook.Backup_Recovery.storage_admission import _local
 scope=getattr(_local,'discovery_scope',None) or getattr(_local,'preview_scope',None)
 actual=kwargs.get('_read_budget')
 captured=getattr(_local,'mobile_sqlite_scope',None)
 budgeted=(len(budgets)==2 and (actual is budgets[0] if captured is None else captured.read_budget is budgets[1]))
 observations.append({'path':str(path),'scope':type(scope).__name__,'domain':'native' if captured is not None else 'coordinator','budgeted':budgeted})
 assert observations[-1]['budgeted'],'populated start source inspection lacks original allowance'
 try:return native_inspect(path,adapter,**kwargs)
 except Exception as error:
  import traceback
  inspection_errors.append({'type':type(error).__name__,'message':str(error),'stack':traceback.format_exc(limit=8)})
  raise
mobile_consolidation._inspect_store=checked
phases=[]
native_pairs=[]
from tldw_chatbook.Backup_Recovery import mobile_enrollment,mobile_native_operation,mobile_consolidation_apply,journal as native_journal
from tldw_chatbook.Backup_Recovery.storage_admission import _mobile_admission
def snapshot(name):
 if len(budgets)==2:phases.append({'phase':name,'aggregate_remaining':budgets[0]._remaining,'native_remaining':budgets[1]._remaining})
native_discover=mobile_enrollment.discover
def discovery(*args,**kwargs):
 snapshot('enrollment_inventory_enter')
 try:return native_discover(*args,**kwargs)
 finally:snapshot('enrollment_inventory_exit')
mobile_enrollment.discover=discovery
native_capture=mobile_consolidation_apply._verified_capture
def capture(*args,**kwargs):
 snapshot('capture_enter')
 try:return native_capture(*args,**kwargs)
 finally:snapshot('capture_exit')
mobile_consolidation_apply._verified_capture=capture
native_consolidate=mobile_native_operation._consolidate
def consolidate(captured,**kwargs):
 admission=_mobile_admission(captured.session)
 native_pairs.append(admission[2] is budgets[1] and admission[4] is budgets[0])
 snapshot('native_proof_enter')
 try:return native_consolidate(captured,**kwargs)
 finally:
  native_pairs.append(admission[2] is budgets[1] and admission[4] is budgets[0])
  snapshot('native_proof_exit')
mobile_native_operation._consolidate=consolidate
native_append=native_journal.Journal._append
def append(self,parent,event,*args,**kwargs):
 if event in ('mobile_owner_opened','mobile_owner_retired','mobile_committed'):snapshot(event+'_enter')
 result=native_append(self,parent,event,*args,**kwargs)
 if event in ('mobile_owner_opened','mobile_owner_retired','mobile_committed'):snapshot(event+'_exit')
 return result
native_journal.Journal._append=append
errors=[]
native_issue=recovery_service.issue_code
def issue(error,**kwargs):
 errors.append((type(error).__name__,str(error)))
 return native_issue(error,**kwargs)
recovery_service.issue_code=issue
try:
 operation=service.start_mobile_consolidation(preview,destination_confirmed=True,rollback_password=b'test-private-mobile-rollback')
 state=service.wait(operation)
 report={'state':state['state'],'phase':state['phase'],'issues':state['issues'],'review_issues':state.get('review_issues',()),'worker_review_issues':service.current().get('review_issues',()),'errors':errors,'inspection_errors':inspection_errors,'reads':observations,'budgets':len(budgets),'phases':phases,'native_pairs':native_pairs,'costs':sorted(((str(k),v) for k,v in costs.items()),key=lambda row:-row[1]),'limits':limits,'source_unchanged':before==fingerprints(source),'profile_guard':take_violations(),'network_guard':blocked_attempts()}
 Path(sys.argv[-1]).write_text(json.dumps(report,indent=2))
 assert report['source_unchanged'] and not report['profile_guard'] and not report['network_guard'],report
 assert observations and all(row['budgeted'] for row in observations),report
 assert len(budgets)==2 and native_pairs and all(native_pairs),report
 assert state['state']=='succeeded',report
finally:service.close()
"""


_PREFETCH = (
    _POPULATED.split("service=RecoveryService", 1)[0]
    + r"""
from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
budget=_TraceReadBudget()
costs={}
native_reserve=_TraceReadBudget.reserve
def charged(self,amount):
 frame=sys._getframe(1)
 key=(Path(frame.f_code.co_filename).name,frame.f_lineno)
 costs[key]=costs.get(key,0)+amount
 return native_reserve(self,amount)
_TraceReadBudget.reserve=charged
error=None
observed=None
try:
 with _preview_reads(_read_budget=budget):
  observed=mobile_consolidation._inspect_store(source/'chat.db',mobile_consolidation._adapters()['db.chachanotes.primary'],_read_budget=budget)
except Exception:
 import traceback
 error=traceback.format_exc(limit=10)
report={'error':error,'version':observed[0] if observed else None,'counts':len(observed[2]) if observed else None,'remaining':budget._remaining,'costs':sorted(((str(k),v) for k,v in costs.items()),key=lambda row:-row[1]),'source_unchanged':before==fingerprints(source),'profile_guard':take_violations(),'network_guard':blocked_attempts()}
Path(sys.argv[-1]).write_text(json.dumps(report,indent=2))
assert error is None and observed[0]==81 and observed[2] and report['source_unchanged'] and not report['profile_guard'] and not report['network_guard'],report
"""
)


@pytest.mark.parametrize("native_reader_only", [False, True])
def test_unbound_populated_start_prices_initial_and_held_source_reviews(
    tmp_path, native_reader_only
):
    home = tmp_path / "home"
    selector = home / "selected" / "config.toml"
    selector.parent.mkdir(parents=True, mode=0o700)
    (home / "data").mkdir(mode=0o700)
    (home / "canonical").mkdir(mode=0o700)
    selector.write_text(
        '[general]\nusers_name="operator"\n[first_run]\nsetup_completed=true\n'
        "[splash_screen]\nenabled=false\n"
        '[tldw_api]\nbase_url=""\napi_url=""\nurl=""\n'
        f'[paths]\ndata_dir="{home / "data"}"\n'
        f'[database]\nchachanotes_db_path="{home / "canonical" / "chat.db"}"\n'
        f'workspaces_db_path="{home / "canonical" / "workspaces.db"}"\n'
    )
    selector.chmod(0o600)
    repository = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ,
        HOME=str(home),
        USERPROFILE=str(home),
        TLDW_TEST_CONFIG_ROOT=str(home / ".test-config"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(repository),
    )
    environment.pop("TLDW_CONFIG_PATH", None)
    for name, script in (
        ("startup", _START_CANONICAL),
        ("populated", _PREFETCH if native_reader_only else _POPULATED),
    ):
        log = tmp_path / f"native-{name}.log"
        result = tmp_path / f"native-{name}.json"
        with log.open("w") as output:
            child = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-c",
                    script,
                    "--config",
                    str(selector),
                    str(result),
                ],
                cwd=repository,
                env=environment,
                stdout=output,
                stderr=output,
                timeout=300,
                check=False,
            )
        assert child.returncode == 0, log.read_text()[-12000:]
    assert json.loads((tmp_path / "native-populated.json").read_text())[
        "source_unchanged"
    ]


def test_native_budgeted_catalog_digest_keeps_exact_json_framing():
    import hashlib

    from tldw_chatbook.Backup_Recovery.mobile_consolidation import _digest
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    values = ("", 'ascii " \\ \t\n\r\b\f', "\x00\x1f\x7f", "é漢😀", "\ud800")
    expected = hashlib.sha256(
        json.dumps(values, sort_keys=True, default=str).encode()
    ).hexdigest()
    with _preview_reads(_read_budget=_TraceReadBudget()):
        assert _digest(values) == expected
        assert _digest(()) == hashlib.sha256(b"[]").hexdigest()
        nested = {
            "types": [None, True, False, -123456789, 1.25, b"\x00\xff"],
            "text": values,
        }
        assert (
            _digest(nested)
            == hashlib.sha256(
                json.dumps(nested, sort_keys=True, default=str).encode()
            ).hexdigest()
        )


@pytest.mark.parametrize("held", [False, True])
def test_native_prefetch_refuses_payload_and_retires_both_scope_kinds(
    tmp_path, monkeypatch, held
):
    from contextlib import ExitStack, closing, nullcontext

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.mobile_preview_budget import current, table_cost
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _restrict_connection
    from tldw_chatbook.Backup_Recovery.storage_admission import (
        _mobile_admission,
        _preview_reads,
    )
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    data = tmp_path / "private-owner"
    data.mkdir(mode=0o700)
    path = data / "workspaces.db"
    owner = WorkspaceDB(path, "native-budget-test")
    owner.close()
    before = path.read_bytes()
    authority = admission_authority(root)
    authority.register("test.mobile.owner", (data,))
    budget = _TraceReadBudget()
    statements = []
    lease = None
    # The held issuer also charges its mandatory final source observation.
    # Exhaustion there remains a refusal; its finally must still retire custody.
    with (
        pytest.raises(_TraceReadLimit) if held else nullcontext(),
        ExitStack() as contexts,
    ):
        if held:
            session = contexts.enter_context(
                authority.maintenance(("bootstrap.unbound", "test.mobile.owner"), 2)
            )
            _mobile_admission(session)[2] = budget
            contexts.enter_context(
                session._discovery_reads(private_sqlite=True, _read_budget=budget)
            )
        else:
            contexts.enter_context(_preview_reads(_read_budget=budget))
        scope = current()
        with closing(
            connect_private_sqlite("recovery.mobile.preview", path, read_only=True)
        ) as connection:
            _restrict_connection(connection)
            lease = next(
                item
                for item in scope.resources
                if getattr(item, "connection", None) is connection
            )
            connection.set_trace_callback(statements.append)
            # Consume the real allowance, leaving only the fixed scalar size
            # projection. The following payload reservation must fail before
            # the caller's payload SELECT can execute.
            projection = 2048 + 128 + 32 * (len("schema_version") + len("version"))
            budget.reserve(budget._remaining - projection)
            with pytest.raises(_TraceReadLimit):
                table_cost(connection, "schema_version", ("version",))
                connection.execute("SELECT version FROM schema_version").fetchall()
        assert lease.resource_closed and lease.connection is None
    assert not scope.active
    assert not scope.custody.fds and not scope.custody.directory_iterators
    copies = scope.sqlite_copies if held else scope
    assert not copies.active and copies.directory is None
    assert any("SELECT count(*)" in sql for sql in statements)
    assert not any(sql == "SELECT version FROM schema_version" for sql in statements)
    assert path.read_bytes() == before


def test_materialization_refusal_uses_fixed_body_free_public_code():
    from tldw_chatbook.Backup_Recovery.recovery_service import issue_code
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadLimit

    assert (
        issue_code(_TraceReadLimit("private arbitrary exception detail"))
        == "mobile_materialization_limit"
    )


def test_budgeted_literal_catalog_still_refuses_extra_native_index(
    tmp_path, monkeypatch
):
    import sqlite3
    from contextlib import closing

    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        _adapters,
        _inspect_store,
    )
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: tmp_path / "bootstrap"
    )
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    path = private / "workspaces.db"
    owner = WorkspaceDB(path, "catalog-budget-test")
    owner.close()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE INDEX uninstalled_version_index ON schema_version(version)"
        )
        connection.commit()
    before = path.read_bytes()
    budget = _TraceReadBudget()
    with (
        _preview_reads(_read_budget=budget),
        pytest.raises(ValueError, match="^unsupported_schema$"),
    ):
        _inspect_store(path, _adapters()["db.workspaces"], _read_budget=budget)
    assert path.read_bytes() == before


def test_native_toml_reservation_precedes_codec_and_keeps_default(monkeypatch):
    import tomllib

    from tldw_chatbook.Backup_Recovery.mobile_preview_budget import toml_loads
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )

    raw = b'[native]\nvalue="unchanged"\n'
    assert toml_loads(raw) == {"native": {"value": "unchanged"}}
    calls = []
    native = tomllib.loads

    def observed(value):
        calls.append(value)
        return native(value)

    monkeypatch.setattr(tomllib, "loads", observed)
    budget = _TraceReadBudget()
    with _preview_reads(_read_budget=budget):
        budget.reserve(budget._remaining)
        with pytest.raises(_TraceReadLimit):
            toml_loads(raw)
    assert not calls


def test_budgeted_subscription_native_catalog_keeps_default_acceptance(
    tmp_path, monkeypatch
):
    """The separate operational adapter uses the genuine initial read scope."""
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.mobile_preview_budget import current
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.recovery_operations import recovery_adapters
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: tmp_path / "bootstrap"
    )
    path = tmp_path / "subscriptions.db"
    owner = SubscriptionsDB(path)
    owner.add_subscription(
        "Budgeted inert history",
        "rss",
        "https://example.invalid/feed",
        auto_pause_threshold=5,
    )
    owner.close()
    before = path.read_bytes()
    adapter = next(
        row for row in recovery_adapters() if row.owner_id == "db.subscriptions"
    )
    assert adapter.validate(path) == ()
    budget = _TraceReadBudget()
    with _preview_reads(_read_budget=budget):
        scope = current()
        assert scope.read_budget is budget
        assert adapter.validate(path) == ()
        assert budget._remaining < 16 * 1024**2
    assert not scope.active and scope.directory is None
    assert not scope.custody.fds and not scope.custody.directory_iterators
    assert path.read_bytes() == before


@pytest.mark.parametrize("entry", ["finish", "rollback", "start"])
def test_cold_mobile_finish_installs_all_native_owners_before_admission(
    tmp_path, entry
):
    """Stop before maintenance; a cold import must register shipped declarations."""
    script = r"""
from threading import Event
import sys
from Tests.real_profile_guard import install as profile_guard, take_violations
from Tests.network_guard import install as network_guard, blocked_attempts
profile_guard()
network_guard()
from tldw_chatbook.Backup_Recovery import mobile_consolidation_recovery as recovery
from tldw_chatbook.Backup_Recovery import owner_registry
assert owner_registry.registered() == ()
class AdmissionBoundaryReached(Exception):
    pass
def stop_before_admission(*args, **kwargs):
    actual = owner_registry.registered()
    assert actual, "cold mobile finish has no installed owner declarations"
    expected = owner_registry.install_adapters()
    assert actual == expected
    raise AdmissionBoundaryReached
recovery._recovery_pending_session = stop_before_admission
try:
    if sys.argv[1] == "finish":
        recovery._finish_import(None, "no-operation-created", b"unit-test-only", Event(), _budgets=None)
    elif sys.argv[1] == "rollback":
        from tldw_chatbook.Backup_Recovery.mobile_rollback import _rollback_import
        _rollback_import(None, "no-operation-created", b"unit-test-only", Event(), _budgets=None)
    else:
        from tldw_chatbook.Backup_Recovery import mobile_enrollment
        from types import SimpleNamespace
        mobile_enrollment._start_session = stop_before_admission
        recovery._start_mobile(SimpleNamespace(control_root=None), "no-operation-created", Event(), None, b"unit-test-only")
except AdmissionBoundaryReached:
    pass
else:
    raise AssertionError("must stop before any native admission or discovery")
assert not take_violations() and not blocked_attempts()
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script, entry],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    (tmp_path / "cold-owner-registration.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
