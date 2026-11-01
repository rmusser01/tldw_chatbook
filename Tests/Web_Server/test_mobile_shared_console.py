"""The HTTP adapter cannot construct a second Console execution graph."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Web_Server.mobile_service import MobileChatService


@pytest.mark.asyncio
async def test_canonical_adapter_reads_without_starting_host():
    starts = []

    class Hosts:
        async def get_or_start(self, identity):
            starts.append(identity)
            raise AssertionError("A Library read must not start a host")

        async def aclose(self):
            raise AssertionError("The server owns the registry")

    reader = SimpleNamespace(
        sessions_page=lambda **kw: {"items": [], "page": 1, "has_more": False}
    )
    service = MobileChatService(
        hosts=Hosts(),
        storage_identity="canonical",
        broker=None,
        readers=reader,
        require_cutover=lambda: None,
    )
    assert service.sessions_page() == {"items": [], "page": 1, "has_more": False}
    await service.shutdown()
    assert starts == []


@pytest.fixture
def mobile_surface_gate(monkeypatch):
    from Tests.Web_Server.test_mobile_api import mobile_surface_gate as fixture

    # Preserve the named API fixture's actual configuration gate.
    return fixture.__wrapped__(monkeypatch)


@pytest.fixture
async def shared_mobile_case(tmp_path, monkeypatch):
    """Real native App/runtime + authenticated control child; recording lifecycle."""
    import os
    import tempfile

    from Tests.console_provider_doubles import AgentModeGateway
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook import app_served
    from tldw_chatbook.Backup_Recovery.inventory import read_profile_config
    from tldw_chatbook.Backup_Recovery.profile_paths import (
        canonical_console_paths,
        database_path,
        user_data_dir,
    )
    from tldw_chatbook.Canvas.control_protocol import (
        CanvasControlBroker,
        CanvasControlClient,
    )
    from tldw_chatbook.Chat.chat_conversation_scope_service import (
        ChatConversationScopeService,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_runtime import dispose_console_runtime
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.Web_Server.mobile_service import CanonicalMobileReads
    from tldw_chatbook.Web_Server.served_console_hosts import (
        ServedConsoleHost,
        ServedConsoleHosts,
    )

    test_root = (
        __import__("pathlib")
        .Path(os.environ["TLDW_TEST_CONFIG_ROOT"])
        .resolve(strict=True)
    )
    assert test_root.is_relative_to(
        __import__("pathlib").Path(tempfile.gettempdir()).resolve()
    )
    assert test_root.name.startswith("tldw_test_config_")
    selector = (
        __import__("pathlib").Path(os.environ["TLDW_CONFIG_PATH"]).resolve(strict=True)
    )
    assert selector.is_relative_to(test_root)
    assert (
        __import__("pathlib")
        .Path(os.environ["HOME"])
        .resolve(strict=True)
        .is_relative_to(test_root)
    )
    values = read_profile_config(selector)
    selected_data = user_data_dir(values).resolve()
    assert selected_data.is_relative_to(test_root)
    assert all(
        path.resolve().is_relative_to(test_root)
        for path in (
            *canonical_console_paths(values),
            database_path(values, "media_db_path"),
        )
    )
    selected_data.mkdir(mode=0o700, parents=True, exist_ok=True)
    app = _build_test_app(configured_default="library", user_data_dir=selected_data)
    core = CharactersRAGDB(tmp_path / "canonical-chat.db", "native-mobile-test")
    media = MediaDatabase(tmp_path / "canonical-media.db", "native-mobile-test")
    media.close_connection()
    app.chachanotes_db = core
    app.local_chat_conversation_service = ChatConversationService(core)
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=app.local_chat_conversation_service, server_service=None
    )
    app._ui_ready = True
    app.console_runtime.set_provider_gateway(AgentModeGateway())
    conversation_id = core.add_conversation({"title": "Canonical saved chat"})
    core.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "Canonical saved body",
        }
    )
    core.close_connection()
    broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
    await broker.start()
    case = SimpleNamespace(
        app=app,
        conversation_id=conversation_id,
        host_start_count=0,
        mobile_db_opens=[],
        host=None,
        storage_identity="canonical-admitted",
        trap_armed=True,
    )

    class NativeAppService:
        app_service_id = "native-child"

        def __init__(self, **callbacks):
            self.callbacks = callbacks
            self.child = None

        async def start(self, width, height):
            assert (width, height) == (80, 24)
            case.host_start_count += 1
            launch = broker.issue_child(self.app_service_id)
            self.child = CanvasControlClient(
                launch.environment,
                runtime_snapshot_id="a" * 64,
                handler=lambda request: app_served._handle_served_control_request(
                    app, request
                ),
            )
            app.served_canvas_control = self.child
            await self.child.start()

        async def stop(self):
            await dispose_console_runtime(app)
            await self.child.aclose()

    def factory(identity):
        assert identity == case.storage_identity
        case.host = ServedConsoleHost(NativeAppService)
        return case.host

    hosts = ServedConsoleHosts(factory)
    readers = CanonicalMobileReads(
        core.db_path, media.db_path, core.db_path.parent / "agent_runs.db"
    )
    service = MobileChatService(
        hosts=hosts,
        storage_identity="canonical-admitted",
        broker=broker,
        readers=readers,
        require_cutover=lambda: None,
    )
    case.service, case.hosts, case.broker = service, hosts, broker
    # Post-retirement constructors remain a trap against mobile/ DB selection.
    for owner in (CharactersRAGDB, MediaDatabase):
        native = owner.__init__

        def recording(actual, db_path, *args, _native=native, **kwargs):
            if (
                case.trap_armed
                and "mobile" in __import__("pathlib").Path(db_path).parts
            ):
                case.mobile_db_opens.append(str(db_path))
                raise AssertionError("Retired mobile DB opened")
            return _native(actual, db_path, *args, **kwargs)

        monkeypatch.setattr(owner, "__init__", recording)
    try:
        yield case
    finally:
        await service.shutdown()
        await hosts.aclose()
        await dispose_console_runtime(app)
        with core.quiesce_connections(timeout_seconds=2):
            pass
        assert core.registered_connection_count() == 0
        await broker.aclose()


async def _open_from_client(case, client_id, conversation_id, tmp_path):
    from Tests.Web_Server.test_mobile_api import _csrf, _logged_in_client
    from tldw_chatbook.Canvas.web_auth import CSRF_HEADER_NAME

    client_root = tmp_path / client_id
    client_root.mkdir()
    async with _logged_in_client(client_root, service=case.service) as client:
        headers = {CSRF_HEADER_NAME: await _csrf(client)}
        response = await client.post(
            "/api/v1/mobile/console/host", json={"protocol_version": 1}, headers=headers
        )
        assert response.status == 200, await response.text()
        host = (await response.json())["host"]
        payload = {"conversation_id": conversation_id}
        response = await client.post(
            "/api/v1/mobile/commands/tickets",
            json={
                "protocol_version": 1,
                "action": "open",
                "target": host,
                "payload": payload,
            },
            headers=headers,
        )
        assert response.status == 200, await response.text()
        ticket = await response.json()
        response = await client.post(
            "/api/v1/mobile/commands",
            json={
                "protocol_version": 1,
                "ticket": ticket["ticket"],
                "payload": payload,
            },
            headers=headers,
        )
        assert response.status == 200, await response.text()
        receipt = await response.json()
        assert receipt["state"] == "applied", receipt
        queried = await client.get("/api/v1/mobile/commands/" + ticket["command_id"])
        assert queried.status == 200, await queried.text()
        assert (await queried.json())["native_id"] == receipt["native_id"]
        snapshot = await client.get(
            "/api/v1/mobile/console/snapshot?protocol_version=1"
        )
        assert snapshot.status == 200, await snapshot.text()
        return next(
            row
            for row in (await snapshot.json())["chats"]
            if row["session_id"] == receipt["native_id"]
        )


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_two_clients_open_one_saved_console_owner(
    shared_mobile_case, mobile_surface_gate, tmp_path
):
    import asyncio

    case = shared_mobile_case
    first, second = await asyncio.gather(
        _open_from_client(case, "phone-a", case.conversation_id, tmp_path),
        _open_from_client(case, "phone-b", case.conversation_id, tmp_path),
    )
    assert first["session_id"] == second["session_id"]
    assert first["incarnation_id"] == second["incarnation_id"]
    assert case.host_start_count == 1
    assert case.host.current_attachment is None
    assert case.mobile_db_opens == []
    assert case.app.console_runtime.chat_store.active_session_id is None


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_native_create_lost_ack_queries_original_without_replay(
    shared_mobile_case, monkeypatch
):
    case = shared_mobile_case
    host = await case.service.bootstrap_host()
    assert case.host_start_count == 1
    assert case.app.console_runtime.chat_store is None
    payload = {"workspace_id": "global"}
    ticket = await case.service.issue_command_ticket("create", host, payload)
    request = case.broker.request

    async def lost_ack(child_id, message_type, *args, **kwargs):
        response = await request(child_id, message_type, *args, **kwargs)
        if message_type == "console.command.request":
            raise TimeoutError("Lost acknowledgement after native apply")
        return response

    monkeypatch.setattr(case.broker, "request", lost_ack)
    with pytest.raises(TimeoutError):
        await case.service.submit_command(ticket["ticket"], payload)
    receipt = await case.service.query_command(ticket["command_id"])
    assert receipt.state == "applied"
    monkeypatch.setattr(case.broker, "request", request)
    duplicate = await case.service.submit_command(ticket["ticket"], payload)
    assert duplicate == receipt
    store = case.app.console_runtime.chat_store
    assert len(store.sessions()) == 1
    session = store.sessions()[0]
    assert receipt.native_id == session.id
    assert store.active_session_id is None
    assert case.host_start_count == 1 and case.mobile_db_opens == []
    await case.hosts.aclose()
    closed_receipt = await case.service.query_command(ticket["command_id"])
    assert closed_receipt.native_id == receipt.native_id
    assert case.host_start_count == 1


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_reads_and_reconnect_do_not_start_host(shared_mobile_case):
    from tldw_chatbook.Web_Server.mobile_service import MobileOwnerUnavailable

    case = shared_mobile_case
    page = case.service.sessions_page()
    assert [row["id"] for row in page["items"]] == [case.conversation_id]
    saved = await case.service.session_messages(case.conversation_id)
    assert saved["messages"][0]["text"] == "Canonical saved body"
    with pytest.raises(MobileOwnerUnavailable):
        await case.service.snapshot()
    events = case.service.events()
    with pytest.raises(MobileOwnerUnavailable):
        await anext(events)
    await events.aclose()
    assert case.host_start_count == 0
    assert case.app.console_runtime.chat_store is None


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_cutover_and_native_admission_refuse_before_chat_effects(
    shared_mobile_case,
):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        MobileConsolidationRefused,
    )
    from tldw_chatbook.Web_Server.mobile_service import MobileOwnerUnavailable

    case = shared_mobile_case

    def refuse():
        raise MobileConsolidationRefused("review_required")

    case.service.require_cutover = refuse
    with pytest.raises(MobileConsolidationRefused):
        await case.service.bootstrap_host()
    assert case.host_start_count == 0
    case.service.require_cutover = lambda: None
    case.app.console_runtime.ensure_activity_receipt_service = lambda: None
    with pytest.raises(MobileOwnerUnavailable, match="execution_owner_unavailable"):
        await case.service.bootstrap_host()
    assert case.host_start_count == 1
    assert case.app.console_runtime.chat_store is None
    assert case.app.console_runtime._remote_receipts is None


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_stale_host_target_and_payload_never_create_second_session(
    shared_mobile_case,
):
    from tldw_chatbook.Canvas.control_protocol import ControlProtocolError

    case = shared_mobile_case
    host = await case.service.bootstrap_host()
    with pytest.raises(ControlProtocolError):
        await case.service.issue_command_ticket(
            "create",
            host.model_copy(update={"server_epoch": "old"}),
            {"workspace_id": "global"},
        )
    assert case.app.console_runtime.chat_store is None
    ticket = await case.service.issue_command_ticket(
        "create", host, {"workspace_id": "global"}
    )
    with pytest.raises(ControlProtocolError):
        await case.service.submit_command(ticket["ticket"], {"workspace_id": "changed"})
    assert case.app.console_runtime.chat_store is None
    receipt = await case.service.submit_command(
        ticket["ticket"], {"workspace_id": "global"}
    )
    assert receipt.state == "applied"
    snapshot = await case.service.snapshot()
    from tldw_chatbook.Chat.console_remote_models import ConsoleTarget

    row = next(
        row for row in snapshot["chats"] if row["session_id"] == receipt.native_id
    )
    target = ConsoleTarget(
        **host.model_dump(), session_id=row["session_id"], incarnation_id="stale"
    )
    ticket = await case.service.issue_command_ticket(
        "send",
        target,
        {"text": "not sent", "expected_revisions": row["expected_revisions"]},
    )
    refused = await case.service.submit_command(
        ticket["ticket"],
        {"text": "not sent", "expected_revisions": row["expected_revisions"]},
    )
    assert refused.state == "refused"
    assert len(case.app.console_runtime.chat_store.sessions()) == 1


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_overflow_and_disconnect_preserve_native_owner_and_other_phone(
    shared_mobile_case,
):
    import asyncio

    case = shared_mobile_case
    await case.service.bootstrap_host()
    streams = [case.service.events(client_id=name) for name in ("phone-a", "phone-b")]
    pending = [asyncio.create_task(anext(events)) for events in streams]
    for _ in range(100):
        if len(case.app.console_runtime._remote_projection._clients) == 2:
            break
        await asyncio.sleep(0.01)
    assert len(case.app.console_runtime._remote_projection._clients) == 2
    buffer = case.broker.console_events("native-child")
    for sequence in range(1000):
        buffer.publish("console.changed", {"sequence": sequence})
    buffer.note_changed(buffer.read(0).sequence)
    first = await asyncio.gather(*pending)
    assert all(event.type == "reset" for event in first)
    assert len(buffer.read(0).events) <= 64
    await streams[0].aclose()
    assert case.app.console_runtime._remote_projection._clients == {"phone-b"}
    response = await case.broker.request(
        "native-child", "console.events.request", {"request": {}}, timeout=1
    )
    assert "batch" in response.payload
    assert not case.app.console_runtime._disposed
    await streams[1].aclose()
    assert not case.app.console_runtime._remote_projection._clients
    assert case.host_start_count == 1


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_native_snapshot_transport_race_retains_reset(
    shared_mobile_case, monkeypatch
):
    case = shared_mobile_case
    await case.service.bootstrap_host()
    buffer = case.broker.console_events("native-child")
    for value in range(70):
        buffer.publish("console.changed", {"sequence": value})
    assert buffer.read(0).reset_required
    request = case.broker.request

    async def during_capture(child_id, message_type, *args, **kwargs):
        response = await request(child_id, message_type, *args, **kwargs)
        if message_type == "console.snapshot.request":
            buffer.publish("console.changed", {"sequence": 9999})
        return response

    monkeypatch.setattr(case.broker, "request", during_capture)
    snapshot = await case.service.snapshot()
    assert snapshot["transport_sequence"] == 70
    assert snapshot["sequence"] != 70
    assert snapshot["reset_required"]
    assert buffer.read(70).events[-1]["payload"]["sequence"] == 9999
    assert buffer.read(70).reset_required
    monkeypatch.setattr(case.broker, "request", request)
    settled = await case.service.snapshot()
    assert settled["transport_sequence"] == 71
    assert not buffer.read(71).reset_required


def test_verified_empty_cutover_routes_same_public_canonical_target(tmp_path):
    """One isolated fresh public producer, then actual native adapter ownership."""
    import hashlib
    import json
    import os
    import stat
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from Tests.Backup_Recovery.test_mobile_public_operator import (
        _NO_DATA,
        _START_CANONICAL,
    )

    # Source-claim inventory inspects each ancestor. A shared sticky /tmp
    # parent deliberately refuses missing-leaf pinning; use the OS user-private
    # temporary cohort, as the native public operator fixture does normally.
    parent = Path(tempfile.gettempdir()).resolve(strict=True)
    parent_info = parent.stat()
    assert parent != Path("/private/tmp")
    assert parent_info.st_uid == os.geteuid()
    assert not (parent_info.st_mode & (0o077 | stat.S_ISVTX))
    original_artifacts = tmp_path
    tmp_path = Path(
        tempfile.mkdtemp(prefix="task9-mobile-public-", dir=parent)
    ).resolve(strict=True)
    base_info = tmp_path.stat()
    assert tmp_path.parent == parent and base_info.st_uid == os.geteuid()
    assert not (base_info.st_mode & (0o077 | stat.S_ISVTX))
    (original_artifacts / "standalone-root.json").write_text(
        json.dumps(
            {
                "path": str(tmp_path),
                "device": base_info.st_dev,
                "inode": base_info.st_ino,
                "uid": base_info.st_uid,
                "mode": stat.S_IMODE(base_info.st_mode),
                "parent": str(parent),
                "parent_uid": parent_info.st_uid,
                "parent_mode": stat.S_IMODE(parent_info.st_mode),
            },
            indent=2,
        )
    )
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    selector = home / "selected" / "config.toml"
    assert selector.is_relative_to(tmp_path)
    assert home.is_relative_to(tmp_path)
    selector.parent.mkdir(parents=True, mode=0o700)
    (home / "data").mkdir(mode=0o700)
    (home / "canonical").mkdir(mode=0o700)
    selector.write_text(
        '[general]\nusers_name="operator"\ndefault_tab="library"\n'
        "[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n"
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
    environment.pop("TLDW_TEST_MOBILE_CLI", None)
    environment.pop("XDG_CONFIG_HOME", None)
    environment.pop("XDG_DATA_HOME", None)
    assert "XDG_CONFIG_HOME" not in environment and "XDG_DATA_HOME" not in environment
    scripts = [
        ("startup", _START_CANONICAL),
        ("cutover", _NO_DATA),
        ("adapter", _PUBLIC_NATIVE_ADAPTER),
    ]
    sources = {
        str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in repository.joinpath("tldw_chatbook").rglob("*.py")
    }
    (tmp_path / "source-environment.json").write_text(
        json.dumps(
            {
                "source": sources,
                "python": sys.executable,
                "private_home": str(home),
                "selected_config": str(selector),
                "environment": {
                    key: environment[key]
                    for key in (
                        "HOME",
                        "USERPROFILE",
                        "TLDW_TEST_CONFIG_ROOT",
                        "TLDW_TEST_MODE",
                        "TLDW_DISABLE_CONFIG_WATCH",
                        "PYTHONPATH",
                    )
                },
            },
            indent=2,
        )
    )
    for name, script in scripts:
        (tmp_path / f"{name}.py").write_text(script)
        log = tmp_path / f"{name}.log"
        with log.open("w") as output:
            result = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-c",
                    script,
                    "--config",
                    str(selector),
                    str(tmp_path / f"{name}.json"),
                ],
                cwd=repository,
                env=environment,
                stdout=output,
                stderr=output,
                timeout=120,
                check=False,
            )
        assert result.returncode == 0, log.read_text()[-12000:]
    assert (
        json.loads((tmp_path / "cutover.json").read_text())["finished_state"]
        == "succeeded"
    )
    result = json.loads((tmp_path / "adapter.json").read_text())
    assert result["starts"] == 1 and result["same_native_session"]
    assert result["source_absent"] and result["guards_clean"]


_PUBLIC_NATIVE_ADAPTER = r"""
import asyncio,json,sys
from pathlib import Path
from Tests.real_profile_guard import install as install_profile_guard,take_violations
install_profile_guard()
from Tests.network_guard import install,blocked_attempts,NetworkMode,set_mode
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Utils.launch_options import adopt_config_flag
adopt_config_flag(sys.argv[1:])
from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path,legacy_mobile_dir,canonical_console_paths,database_path
from tldw_chatbook.Backup_Recovery.inventory import read_profile_config
from tldw_chatbook.Backup_Recovery.mobile_consolidation import require_mobile_cutover,CanonicalConsoleStores
selector=effective_config_path()
assert selector.is_relative_to(Path.home())
assert all(__import__('os').environ.get(key) is None for key in ('XDG_CONFIG_HOME','XDG_DATA_HOME'))
values=read_profile_config(selector)
source=legacy_mobile_dir(values)
stores=CanonicalConsoleStores(*canonical_console_paths(values))
assert require_mobile_cutover(source,stores) is None
from tldw_chatbook.app import TldwCli
from tldw_chatbook import app_served
from tldw_chatbook.Canvas.control_protocol import CanvasControlBroker,CanvasControlClient
from tldw_chatbook.Chat.console_remote_models import ConsoleTarget
from tldw_chatbook.Web_Server.mobile_service import MobileChatService,CanonicalMobileReads
from tldw_chatbook.Web_Server.served_console_hosts import ServedConsoleHost,ServedConsoleHosts
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
async def main():
 app=TldwCli()
 broker=CanvasControlBroker(runtime_snapshot_id='b'*64)
 await broker.start()
 assert broker._server.sockets[0].getsockname()[0]=='127.0.0.1'
 # Same numeric-loopback-only policy as the native @loopback_network tests.
 # The listener is owned by this process; DNS and external egress stay blocked.
 set_mode(NetworkMode.LOOPBACK_ONLY)
 starts=[]
 class NativeChild:
  app_service_id='task9-public-native'
  def __init__(self,**callbacks):self.child=None
  async def start(self,width,height):
   assert (width,height)==(80,24)
   starts.append((width,height))
   launch=broker.issue_child(self.app_service_id)
   self.child=CanvasControlClient(launch.environment,runtime_snapshot_id='b'*64,handler=lambda request:app_served._handle_served_control_request(app,request))
   app.served_canvas_control=self.child
   await self.child.start()
   assert self.child._writer.get_extra_info('peername')[0]=='127.0.0.1'
  async def stop(self):
   await app._shutdown_app_owned_lifecycles()
   await self.child.aclose()
 def factory(identity):
  assert identity==stores.identity
  return ServedConsoleHost(NativeChild)
 hosts=ServedConsoleHosts(factory)
 reads=CanonicalMobileReads(stores.chat_path,database_path(values,'media_db_path'),stores.run_path)
 service=MobileChatService(hosts=hosts,storage_identity=stores.identity,broker=broker,readers=reads,require_cutover=lambda:require_mobile_cutover(source,stores))
 try:
  assert app.chachanotes_db.db_path==stores.chat_path
  native_core=app.chachanotes_db
  baseline=service.sessions_page()
  assert isinstance(baseline['items'],list)
  assert starts==[]
  host=await service.bootstrap_host()
  native_runtime=app.console_runtime
  payload={'workspace_id':'global'}
  ticket=await service.issue_command_ticket('create',host,payload)
  receipt=await service.submit_command(ticket['ticket'],payload)
  assert receipt.state=='applied',receipt
  store=app.console_runtime.chat_store
  session=next(s for s in store.sessions() if s.id==receipt.native_id)
  saved=app.chachanotes_db.add_conversation({'title':'Public canonical native'})
  app.chachanotes_db.add_message({'conversation_id':saved,'sender':'user','content':'Same-target history'})
  payload={'conversation_id':saved}
  ticket=await service.issue_command_ticket('open',host,payload)
  opened=await service.submit_command(ticket['ticket'],payload)
  assert opened.state=='applied',opened
  opened_session=next(s for s in store.sessions() if s.id==opened.native_id)
  second=await service.issue_command_ticket('open',host,payload)
  repeated=await service.submit_command(second['ticket'],payload)
  assert repeated.native_id==opened_session.id
  row=await service.snapshot(target=ConsoleTarget(**host.model_dump(),session_id=opened_session.id,incarnation_id=opened_session.incarnation_id))
  assert row['console_id']==host.console_id
  assert (await service.session_messages(saved))['messages'][0]['text']=='Same-target history'
  assert (await service.query_command(ticket['command_id'])).native_id==opened_session.id
  assert store.active_session_id is None and len(store.sessions())==2
  assert require_mobile_cutover(source,stores) is None
  assert not source.exists()
  assert not blocked_attempts() and not take_violations()
  Path(sys.argv[-1]).write_text(json.dumps({'starts':len(starts),'baseline_saved_count':len(baseline['items']),'same_native_session':True,'source_absent':not source.exists(),'guards_clean':True,'stores':[str(p) for p in stores.paths],'app_source':str(Path(sys.modules['tldw_chatbook.app'].__file__).resolve())}))
 finally:
  await service.shutdown()
  await hosts.aclose()
  await broker.aclose()
  set_mode(NetworkMode.BLOCKED)
  await app._shutdown_app_owned_lifecycles()
  await app.tts_service.close()
  await app.tts_service.wait_closed()
 assert not blocked_attempts() and not take_violations()
 assert not native_runtime._preparation_reads
 native_core.close_connection()
 with native_core.quiesce_connections(timeout_seconds=2):pass
 assert native_core.registered_connection_count()==0
 result=json.loads(Path(sys.argv[-1]).read_text())
 result['native_closed']=True
 result['core_connections_after_close']=native_core.registered_connection_count()
 Path(sys.argv[-1]).write_text(json.dumps(result))
asyncio.run(main())
"""


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_phone_created_native_approval_remains_answerable_without_view(
    shared_mobile_case, monkeypatch
):
    import asyncio

    from Tests.Chat.test_console_runtime_lifetime import _pending_call
    from tldw_chatbook.Agents.run_context import use_run_id

    case = shared_mobile_case
    host = await case.service.bootstrap_host()
    payload = {"workspace_id": "global"}
    ticket = await case.service.issue_command_ticket("create", host, payload)
    receipt = await case.service.submit_command(ticket["ticket"], payload)
    controller = case.app.console_runtime.chat_controller
    monkeypatch.setattr(
        case.app, "call_from_thread", lambda fn, *args, **kwargs: fn(*args, **kwargs)
    )
    controller.mcp_approval_timeout_seconds = lambda: 2

    def request():
        with use_run_id("native-approval-run"):
            return controller.request_mcp_approvals(
                [_pending_call()], session_id=receipt.native_id
            )

    native = asyncio.create_task(asyncio.to_thread(request))
    discovery = await case.service.snapshot()
    row = next(
        row for row in discovery["chats"] if row["session_id"] == receipt.native_id
    )
    from tldw_chatbook.Chat.console_remote_models import ConsoleTarget

    target = ConsoleTarget(
        **host.model_dump(),
        session_id=receipt.native_id,
        incarnation_id=row["incarnation_id"],
    )
    snapshot = None
    try:
        for _ in range(100):
            snapshot = await case.service.snapshot(target=target)
            if snapshot["decisions"] or native.done():
                break
            await asyncio.sleep(0.01)
        assert snapshot["decisions"], (
            "Explicit phone-created native session must retain its viewless approval"
        )
        decision = snapshot["decisions"][0]
        exact = target.model_copy(
            update={"round_id": decision["round_id"], "run_id": decision["run_id"]}
        )
        projection = case.app.console_runtime._remote_projection
        projection.register_client("phone-a")
        projection.register_client("phone-b")
        await case.service.set_answerable("phone-a", exact, True)
        await case.service.set_answerable("phone-b", exact, True)
        projection.disconnect_client("phone-a")
        assert projection._answerable_clients[exact] == {"phone-b"}
        assert (await case.service.snapshot(target=target))["decisions"][0][
            "answerable"
        ]
        projection.disconnect_client("phone-b")
        assert exact not in projection._answerable_clients
        assert not (await case.service.snapshot(target=target))["decisions"][0][
            "answerable"
        ]
    finally:
        for round_id in tuple(controller._pending_approval_rounds):
            controller.resolve_pending_approval(
                {"write_file": "deny"}, round_id=round_id
            )
        await native


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision_kind", "retain"),
    [("approval", True), ("question", True), ("question", False)],
)
async def test_native_agent_tool_approval_and_first_persisted_identity(
    shared_mobile_case, tmp_path, monkeypatch, decision_kind, retain
):
    import asyncio
    import json
    from uuid import uuid4

    from Tests.console_provider_doubles import AgentModeGateway
    from tldw_chatbook.Chat.console_provider_gateway import ProviderToolCalls
    from tldw_chatbook.Chat.console_remote_models import ConsoleTarget
    from tldw_chatbook.Chat.console_session_settings import (
        default_console_session_settings,
    )

    class QuestionGateway(AgentModeGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            async for item in super().stream_chat(resolution, messages, **kwargs):
                if isinstance(item, ProviderToolCalls):
                    call = item.tool_calls[0]
                    call = dict(
                        call,
                        function=dict(
                            call["function"],
                            arguments=json.dumps(
                                {
                                    "questions": [
                                        {
                                            "question": "Continue?",
                                            "header": "Check",
                                            "options": [
                                                {"label": "Yes"},
                                                {"label": "No"},
                                            ],
                                        }
                                    ]
                                }
                            ),
                        ),
                    )
                    item = ProviderToolCalls((call,), metadata=item.metadata)
                yield item

    case = shared_mobile_case
    if decision_kind == "question":
        case.app.console_runtime.set_provider_gateway(
            QuestionGateway(tool_name="ask_user")
        )
    async with case.app.run_test(size=(100, 32)):
        registry = case.app.workspace_registry_service
        workspace_id = "phone-project-" + uuid4().hex[:12]
        registry.create_workspace(workspace_id=workspace_id, name=workspace_id)
        folder = tmp_path / "project"
        folder.mkdir()
        (folder / "README.md").write_text("Actual native tool evidence")
        binding = registry.add_folder_binding(workspace_id, folder, allow_write=True)
        host = await case.service.bootstrap_host()
        if not retain:
            from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost

            # Remove only the opener capability; the native denial remains intact.
            monkeypatch.setattr(
                InterruptRoundHost, "retain_decision_target", lambda *_: None
            )
        payload = {"workspace_id": workspace_id}
        ticket = await case.service.issue_command_ticket("create", host, payload)
        receipt = await case.service.submit_command(ticket["ticket"], payload)
        assert receipt.state == "applied"
        runtime = case.app.console_runtime
        controller, store = runtime.chat_controller, runtime.chat_store
        controller.mcp_approval_timeout_seconds = lambda: 15
        store.replace_session_settings(
            receipt.native_id,
            default_console_session_settings(
                case.app.app_config, "openai", "gpt-4o-mini"
            ),
        )
        row = next(
            row
            for row in (await case.service.snapshot())["chats"]
            if row["session_id"] == receipt.native_id
        )
        target = ConsoleTarget(
            **host.model_dump(),
            session_id=receipt.native_id,
            incarnation_id=row["incarnation_id"],
        )
        payload = {
            "text": "List the project folder",
            "expected_revisions": row["expected_revisions"],
        }
        ticket = await case.service.issue_command_ticket("send", target, payload)
        assert case.app._local_skill_trust_service is None
        assert case.app.local_skills_service._trust_service is None
        assert case.app.local_skills_service._plugin_service is None
        send = await case.service.submit_command(ticket["ticket"], payload)
        assert send.state == "pending"
        decision = None
        terminal_reviews = []
        for _ in range(240):
            snapshot = await case.service.snapshot(target=target)
            if snapshot["decisions"]:
                current = snapshot["decisions"][0]
                if current["kind"] == "project_instruction_binding":
                    assert (
                        current["supported"] is False
                        and current["action"] == "open_terminal"
                    )
                    assert runtime.resolve_project_instruction_binding(
                        current["round_id"], "select", binding.binding_id
                    )
                    terminal_reviews.append(current["kind"])
                    await asyncio.sleep(0.05)
                    continue
                if current["kind"] == "project_instruction_dispatch":
                    assert (
                        current["supported"] is False
                        and current["action"] == "open_terminal"
                    )
                    assert runtime.resolve_project_instruction_dispatch(
                        current["round_id"], "proceed"
                    )
                    terminal_reviews.append(current["kind"])
                    await asyncio.sleep(0.05)
                    continue
                # Native registration precedes payload publication; wait for
                # the ordinary question's validated payload to become visible.
                if current["kind"] == "question" and not current["supported"]:
                    await asyncio.sleep(0.05)
                    continue
                decision = current
                break
            status = controller.run_state_for(target.session_id).status.value
            if status in {"blocked", "failed"} or not retain and status == "completed":
                break
            await asyncio.sleep(0.05)
        session = next(s for s in store.sessions() if s.id == target.session_id)
        assert session.persisted_conversation_id
        assert (
            runtime._remote_saved_sessions[session.persisted_conversation_id]
            == session.id
        )
        assert (
            controller._interrupt_host.has_retained_decision_target(session.id)
            is retain
        )
        if retain:
            assert decision is not None, (
                snapshot,
                controller.run_state_for(target.session_id),
            )
            assert decision["supported"] and decision["kind"] == decision_kind, decision
            decision_target = target.model_copy(
                update={"run_id": decision["run_id"], "round_id": decision["round_id"]}
            )
            payload = (
                {
                    "kind": "approval",
                    "answer": {
                        "decisions": {
                            call["call_id"]: "approve_once"
                            for call in decision["calls"]
                        }
                    },
                }
                if decision_kind == "approval"
                else {
                    "kind": "question",
                    "answer": {
                        "answers": [
                            {
                                "question": "Continue?",
                                "selected": ["Yes"],
                                "other_text": None,
                            }
                        ]
                    },
                }
            )
            answer = await case.service.issue_command_ticket(
                "decision", decision_target, payload
            )
            response = await case.service.submit_command(answer["ticket"], payload)
            assert response.state in {"accepted", "applied"}
        else:
            assert decision is None
        gateway = runtime.provider_gateway
        for _ in range(200):
            if any(
                "MOBILE_AGENT_ROUND_TRIP_COMPLETE" in str(m.content)
                for m in store.messages_for_session(session.id)
            ):
                break
            await asyncio.sleep(0.05)
        assert gateway.tool_result_messages
        tool_wire = json.dumps(gateway.tool_result_messages)
        if decision_kind == "approval":
            assert "README.md" in tool_wire
        elif retain:
            assert "Yes" in tool_wire
        else:
            assert "Tool not permitted: ask_user" in tool_wire
        assert gateway.stream_calls[0]["tool_advertised"] is retain
        assert any(
            "MOBILE_AGENT_ROUND_TRIP_COMPLETE" in str(m.content)
            for m in store.messages_for_session(session.id)
        )
        assert len(gateway.stream_calls) >= 2
        assert store.active_session_id is None
        (tmp_path / "native-agent-result.json").write_text(
            json.dumps(
                {
                    "cold_before_send": True,
                    "retained": retain,
                    "first_persist_registered": True,
                    "tool_executed": retain,
                    "terminal_reviews": terminal_reviews,
                    "model_rounds": len(gateway.stream_calls),
                },
                indent=2,
            )
        )


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_lost_subscription_ack_releases_only_its_native_claim(
    shared_mobile_case, monkeypatch
):
    case = shared_mobile_case
    await case.service.bootstrap_host()
    original = case.broker.request

    async def request(child, message_type, payload, **kwargs):
        result = await original(child, message_type, payload, **kwargs)
        if payload.get("request", {}).get("operation") == "register":
            raise TimeoutError("lost subscription acknowledgement")
        return result

    monkeypatch.setattr(case.broker, "request", request)
    stream = case.service.events(client_id="uncertain-phone")
    with pytest.raises(TimeoutError, match="lost subscription"):
        await anext(stream)
    assert not case.app.console_runtime._remote_projection._clients
    assert not case.app.console_runtime._disposed
