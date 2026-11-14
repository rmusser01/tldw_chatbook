import asyncio
from dataclasses import asdict,replace
import json,os
from pathlib import Path
import sys,tomllib
r=Path(__file__).parent
case=r/'direct-ready'
for name,sub in {'HOME':'home','XDG_CONFIG_HOME':'xdg-config','XDG_DATA_HOME':'xdg-data','XDG_CACHE_HOME':'xdg-cache','TLDW_CONFIG_PATH':'config.toml'}.items():os.environ[name]=str(case/sub)
os.environ['PYTHON_KEYRING_BACKEND']='keyring.backends.null.Keyring'
sys.path.insert(0,'/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook')
import httpx
from tldw_chatbook.Agents.native_tools import provider_supports_native_tools
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
config=tomllib.loads((case/'config.toml').read_text())
requests=[]
async def request(req):
    assert req.method=='GET' and req.url.host=='127.0.0.1' and req.url.port==8010
    assert req.url.path in {'/health','/v1/models','/props'}
async def response(res):requests.append({'method':res.request.method,'path':res.request.url.path,'status':res.status_code})
async def main():
    before=(r/'counter.json').read_bytes()
    async with httpx.AsyncClient(trust_env=False,event_hooks={'request':[request],'response':[response]}) as client:
        gateway=ConsoleProviderGateway(http_client=client,config_provider=lambda:config)
        resolution=await gateway.resolve_for_send(ConsoleProviderSelection(provider='llama_cpp',base_url='http://127.0.0.1:8010',explicit_model=config['chat_defaults']['model'],max_tokens=4096))
        policy=resolution.reasoning_replay
        assert resolution.ready and policy is not None
        actual=provider_supports_native_tools(resolution.execution_key,reasoning_replay=policy)
        control=provider_supports_native_tools(resolution.execution_key,reasoning_replay=replace(policy,native_tools=True))
        assert not actual and control
        result={'ready':resolution.ready,'provider':resolution.execution_key,'model':resolution.model,'reasoning_policy':asdict(policy),'native_tools_supported':actual,'pure_native_fact_only_control':control,'native_overrides':config['console'].get('reasoning_native_tool_overrides',{}),'get_observations':requests,'model_requests':0,'counter_unchanged':(r/'counter.json').read_bytes()==before,'control_is_detached_not_app_override':True}
        assert result['counter_unchanged']
        (r/'capability-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
asyncio.run(main())
