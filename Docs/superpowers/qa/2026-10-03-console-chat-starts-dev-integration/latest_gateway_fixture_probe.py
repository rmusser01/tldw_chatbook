import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import Tests.conftest
import asyncio
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.console_provider_support import resolve_console_provider_identity
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
async def main():
 identity=resolve_console_provider_identity('custom-hosted')
 print('IDENTITY',identity)
 config={'api_settings': {identity.readiness_key: {'api_base_url':'https://custom-hosted-workspace.example.test','model':'m','api_key':'fixture-key'}}}
 gateway=ConsoleProviderGateway(config_provider=lambda:config,environ={})
 result=await gateway.resolve_for_send(ConsoleProviderSelection(provider='custom-hosted',explicit_model='m'))
 print('RESOLUTION',result)
 print('VISIBLE_COPY',result.visible_copy)
asyncio.run(main())
