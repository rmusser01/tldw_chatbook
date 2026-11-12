from pathlib import Path
p=Path('Tests/Chat/test_console_provider_gateway.py'); s=p.read_text()
s=s.replace('    resolve_console_provider_identity,\n)', '    CUSTOM_OPENAI_EXECUTION_KEYS,\n    resolve_console_provider_identity,\n)',1)
s=s.replace('if identity.readiness_key in PROVIDERS_REQUIRING_BASE_URL_KEYS:', 'if (\n            identity.readiness_key in PROVIDERS_REQUIRING_BASE_URL_KEYS\n            or identity.execution_key in CUSTOM_OPENAI_EXECUTION_KEYS\n        ):',1)
# Only the four endpoint-entry cases use the installed default engine.
for name in ['test_custom_endpoint_openai_compatible_resolves_entry_url','test_custom_endpoint_declared_env_key_flows_to_resolution','test_custom_endpoint_stored_key_flows_to_resolution','test_custom_endpoint_resolution_carries_raw_selected_provider']:
 start=s.index('async def '+name); end=s.find('\n\n@pytest',start); end=end if end!=-1 else len(s)
 part=s[start:end]; assert part.count('assert resolved.execution_key == "custom-openai-api"')==1
 part=part.replace('assert resolved.execution_key == "custom-openai-api"','assert resolved.execution_key == "custom-hosted"'); s=s[:start]+part+s[end:]
# Await optional metadata at the assertion boundary, preserving production's nonblocking send.
anchor='class _SettlementBoundary:'
helper='''async def _settle_gateway_metadata(gateway):
    tasks = tuple(gateway._context_window_refreshes) + tuple(
        gateway._reasoning_metadata_refreshes.values()
    )
    if tasks:
        await asyncio.gather(*tasks)


'''
s=s.replace(anchor,helper+anchor,1)
start=s.index('async def test_resolve_for_send_normalizes_scheme_less_llamacpp_base_url_before_http'); end=s.index('\n\n@pytest',start)
part=s[start:end].replace('    assert resolved.ready is True','    await _settle_gateway_metadata(gateway)\n\n    assert resolved.ready is True',1); s=s[:start]+part+s[end:]
for name in ['test_console_persisted_explicit_keyless_llamacpp_sends_no_authorization','test_console_llamacpp_explicit_stored_source_reaches_probe_and_chat']:
 start=s.index('async def '+name);end=s.index('\n\n@pytest',start);part=s[start:end]
 part=part.replace('        chunks = [','        await _settle_gateway_metadata(gateway)\n        chunks = [',1)
 part=part.replace('        ("GET", "/props"),','        ("GET", "/props"),\n        ("GET", "/props"),',1)
 if name.endswith('stored_source_reaches_probe_and_chat'):
  part=part.replace('        "Bearer stored-llama-request-canary",\n    ]','        "Bearer stored-llama-request-canary",\n        "Bearer stored-llama-request-canary",\n    ]',1)
 s=s[:start]+part+s[end:]
# Capture the real migrated hosted transport, with a valid terminal wire response.
start=s.index('class _CapturedMistralSession:');end=s.index('\n\nclass _FakeMistralPostResponse',start);part=s[start:end]
part=part.replace('        return _FakeMistralPostResponse()', '''        import requests

        response = requests.Response()
        response.status_code = 200
        response.headers["Content-Type"] = "application/json"
        response._content = b'{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}'
        return response

    def close(self):
        return None''');s=s[:start]+part+s[end:]
start=s.index('async def test_console_send_keeps_each_mistral_credential_on_its_own_endpoint');end=s.index('\n\n@pytest',start);part=s[start:end]
part=part.replace('from tldw_chatbook.LLM_Calls import LLM_API_Calls','from tldw_chatbook.LLM_Calls import hosted_chat, mistral',1).replace('        LLM_API_Calls,\n        "create_default_session",','        hosted_chat,\n        "create_default_session",',1).replace('        LLM_API_Calls,\n        "get_runtime_config_snapshot",','        mistral,\n        "get_runtime_config_snapshot",',1)
s=s[:start]+part+s[end:];p.write_text(s)
p=Path('Tests/UI/test_console_cost_chip_screen.py');s=p.read_text();s=s.replace('class _AnthropicWaitingGateway:', 'class _AnthropicWaitingGateway(_AnthropicCostGateway):',1).replace('class _AnthropicReadinessWaitingGateway:', 'class _AnthropicReadinessWaitingGateway(_AnthropicCostGateway):',1);p.write_text(s)
