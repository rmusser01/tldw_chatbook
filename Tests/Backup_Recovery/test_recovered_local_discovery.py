"""Restored local provider discovery stays inert before connection review."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_recovery_service import _SERVICE_ISOLATED

_RESTORED_CHILD = r"""
import asyncio
from pathlib import Path
import sys
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
select_profile(sys.argv[1], Path(sys.argv[2]))
from tldw_chatbook import config
from tldw_chatbook.Chat import local_server_discovery as discovery
from tldw_chatbook.LLM_Calls.recovery_review import ProviderReconnectRequired
import httpx
created = []
original = httpx.AsyncClient
def observe(*args, **kwargs):
    created.append(True)
    return original(*args, **kwargs)
httpx.AsyncClient = observe
async def main():
    refused = False
    try:
        if sys.argv[3] == 'discover':
            await discovery.discover_local_servers(config.load_settings())
        else:
            await discovery.probe_models_endpoint('http://127.0.0.1:8080',
                provider_key='llama_cpp')
    except ProviderReconnectRequired:
        refused = True
    assert refused and not created and not blocked_attempts(), (
        refused, created, blocked_attempts())
asyncio.run(main())
"""


@pytest.mark.parametrize("route", ["discover", "probe"])
def test_actual_restored_discovery_refuses_before_client_or_wire(tmp_path, route):
    launch = """
 import subprocess
 child = subprocess.run([sys.executable, '-c', CHILD, profile, str(control), sys.argv[2]],
     capture_output=True, text=True, timeout=20)
 assert child.returncode == 0, child.stderr[-5000:] + child.stdout[-1000:]
"""
    script = "CHILD=" + repr(_RESTORED_CHILD) + "\n" + _SERVICE_ISOLATED.replace(
        " assert service.profiles()[0]['status']=='restoration_validated'",
        " assert service.profiles()[0]['status']=='restoration_validated'" + launch,
    )
    _run(tmp_path, "isolated", route, script=script)


_ORDINARY = r"""
import asyncio
import sys
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Chat import local_server_discovery as discovery
import httpx
route, outcome = sys.argv[1:]
clients = []
requests = []
entered = asyncio.Event()
release = asyncio.Event()
async def transport(request):
    requests.append(request.url.path)
    entered.set()
    if outcome == 'cancel':
        await release.wait()
    return httpx.Response(200, json={'data': [{'id': 'ordinary-model'}]})
original = httpx.AsyncClient
def client_factory(*args, **kwargs):
    client = original(*args, **kwargs, transport=httpx.MockTransport(transport),
        trust_env=False)
    clients.append(client)
    return client
httpx.AsyncClient = client_factory
async def main():
    supplied = client_factory() if outcome == 'borrowed' else None
    if route == 'discover':
        call = discovery.discover_local_servers({}, http_client=supplied)
    else:
        call = discovery.probe_models_endpoint('http://127.0.0.1:8080',
            provider_key='llama_cpp', http_client=supplied)
    task = asyncio.create_task(call)
    if outcome == 'cancel':
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError('cancelled discovery returned normally')
    else:
        result = await task
        if route == 'discover':
            assert result and all(row.model_ids == ('ordinary-model',) for row in result)
        else:
            assert result.ok and result.model_ids == ('ordinary-model',), result
    assert requests and clients
    if supplied is not None:
        assert not supplied.is_closed
        await supplied.aclose()
    assert all(client.is_closed for client in clients)
asyncio.run(main())
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["discover", "probe"])
@pytest.mark.parametrize("outcome", ["complete", "cancel", "borrowed"])
def test_ordinary_discovery_preserves_transport_ownership(tmp_path, route, outcome):
    _run(tmp_path, route, outcome, script=_ORDINARY)


def test_cold_console_opens_without_network_before_review(tmp_path):
    from Tests.Backup_Recovery.test_profile_open import _CHILD, _LAUNCH

    script = (
        "CHILD="
        + repr(_CHILD)
        + "\n"
        + _SERVICE_ISOLATED.replace(
            'users_name="original"\\n',
            'users_name="original"\\ndefault_tab="chat"\\n[first_run]\\nsetup_completed=true\\n[splash_screen]\\nenabled=false\\n',
        ).replace(
            " assert service.profiles()[0]['status']=='restoration_validated'",
            " assert service.profiles()[0]['status']=='restoration_validated'"
            + _LAUNCH,
        )
    )
    _run(tmp_path, "isolated", "mounted", script=script, timeout=70)
