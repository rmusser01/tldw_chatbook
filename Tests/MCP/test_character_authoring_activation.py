"""Character writes retain actual MCP recovery authority in native workers."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_mcp_recovery_review import _APPROVED_SETUP

_TOOLS = r"""
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.MCP.tools import MCPTools

tools=MCPTools.__new__(MCPTools)
delegate._tools=tools
if state=='update_character':
    # Fresh CharactersRAGDB seeds the built-in character at id 1.
    arguments={'character_id':2,'expected_version':1,'fields':{'description':'Updated'}}
else:
    arguments={'name':'Ada','fields':{'description':'Created'}}
records=[]
original=tools._write_character
def persist(*args):
    # A real, worker-owned memory database avoids an open file's storage lease
    # masking a missing MCP source lease in the maintenance assertion.
    db=CharactersRAGDB(':memory:', 'mcp-authoring-admission')
    tools.chachanotes_db=db
    try:
        if state=='update_character':
            assert original({'name':'Ada','description':'Original'})['id']==arguments['character_id']
        receipt=original(*args)
        assert 'id' in receipt,receipt
        records.append(db.get_character_card_by_id(receipt['id']))
        return receipt
    finally:db.close()
tools._write_character=persist
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None:startup.close()
"""


_RETENTION = (
    _APPROVED_SETUP
    + _TOOLS
    + r"""
import threading
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
from tldw_chatbook.MCP.activation import _guard

entered=threading.Event();release=threading.Event();finished=threading.Event()
receipts=[];sources=[]
def write(*args):
    sources.extend(_guard.captured_sources(tools))
    entered.set()
    try:
        assert release.wait(5)
        receipts.append(persist(*args))
        return receipts[-1]
    finally:finished.set()
tools._write_character=write

async def run():
    waiter=asyncio.create_task(plane.execute_hub_tool(
        'builtin:tldw_chatbook',state,arguments,
        timeout_seconds=.3 if route=='timeout' else 4))
    try:
        assert await asyncio.to_thread(entered.wait,3)
        if route=='cancel':waiter.cancel()
        try:await waiter
        except asyncio.CancelledError:assert route=='cancel'
        except RuntimeError as error:assert route=='timeout' and 'Timed out' in str(error)
        else:raise AssertionError('waiter unexpectedly completed')
        assert not finished.is_set()
        try:
            with authority.maintenance(tuple(witness['namespaces']),.03):
                raise AssertionError('character write lost MCP source admission')
        except AdmissionTimeout:pass
        # These are the actual restored sources admitted by the outer plane,
        # including context/targets that MCPTools cannot discover on its own.
        for name,owner in owners_by_name.items():
            assert (owner,user/name) in sources,(owner,sources)
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait,5)

asyncio.run(run())
assert len(receipts)==1 and 'id' in receipts[0],receipts
record=records[0]
assert record['description']==('Updated' if state=='update_character' else 'Created')
assert record['version']==(2 if state=='update_character' else 1)
with authority.maintenance(tuple(witness['namespaces']),1):pass
assert all((user/name).read_bytes()==payload for name,payload in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("tool_name", ["create_character", "update_character"])
@pytest.mark.parametrize("outcome", ["cancel", "timeout"])
def test_character_worker_retains_reviewed_sources_until_persistence(
    tmp_path, tool_name, outcome
):
    _run(tmp_path, outcome, tool_name, script=_RETENTION)


_STANDALONE = (
    _APPROVED_SETUP
    + _TOOLS
    + r"""
from mcp_unified.gateway import GatewayRequestContext
from tldw_chatbook.MCP.activation import MCPActivationRequired
from tldw_chatbook.MCP.builtin_tool_policy import standalone_character_write_refusal
from tldw_chatbook.MCP.gateway_runtime import ChatbookGatewayRuntime
from tldw_chatbook.MCP.recovery_activation import _record_name
from tldw_chatbook.MCP.server import TldwMCPServer,_describe_local_tools

plane.permission_store.set_tool_state('builtin:tldw_chatbook',state,'allow')
assert standalone_character_write_refusal(state) is None
instance=TldwMCPServer.__new__(TldwMCPServer)
instance.tools=tools
instance.mcp=ChatbookGatewayRuntime(name='test',version='1',tool_descriptors=_describe_local_tools())
instance._register_tools();instance.mcp.finalize()
context=GatewayRequestContext(request_id='character-recovery')
if route=='inactive':
    binding=activation._generation(witness['generation'])/_record_name('mcp.local',user/'local_mcp_store.json')
    binding.unlink()
    # The current permission is still allowed; an independently invalidated
    # definition source must stop execution before the first database effect.
    assert standalone_character_write_refusal(state) is None
    try:asyncio.run(instance.mcp.call_tool(state,arguments,context))
    except MCPActivationRequired:pass
    else:raise AssertionError('standalone write bypassed MCP recovery review')
    assert not records
else:
    receipt=asyncio.run(instance.mcp.call_tool(state,arguments,context))
    assert receipt['version']==(2 if state=='update_character' else 1),receipt
    record=records[0]
    assert record['description']==('Updated' if state=='update_character' else 'Created')
assert all((user/name).read_bytes()==payload for name,payload in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("tool_name", ["create_character", "update_character"])
@pytest.mark.parametrize("state", ["active", "inactive"])
def test_standalone_character_write_requires_current_native_review(
    tmp_path, tool_name, state
):
    pytest.importorskip("mcp_unified.gateway")
    _run(tmp_path, state, tool_name, script=_STANDALONE)
