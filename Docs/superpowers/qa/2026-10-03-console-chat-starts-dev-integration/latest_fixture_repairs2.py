from pathlib import Path
p=Path('Tests/Chat/test_console_provider_gateway.py');s=p.read_text()
s=s.replace('if identity.readiness_key in PROVIDERS_REQUIRING_API_KEY_KEYS:', 'if (\n            identity.readiness_key in PROVIDERS_REQUIRING_API_KEY_KEYS\n            or identity.execution_key in CUSTOM_OPENAI_EXECUTION_KEYS\n        ):',1)
s=s.replace("response._content = b'{\"choices\":[{\"message\":", "response._content = b'{\"choices\":[{\"index\":0,\"message\":",1)
# Explicit rollback uses the same entry/URL authority, with only execution changed.
old='@pytest.mark.asyncio\nasync def test_custom_endpoint_openai_compatible_resolves_entry_url() -> None:'
new='''@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("console_settings", "execution_key"),
    [({}, "custom-hosted"), ({"custom_endpoints_use_engine": False}, "custom-openai-api")],
)
async def test_custom_endpoint_openai_compatible_resolves_entry_url(
    console_settings, execution_key
) -> None:'''
assert old in s;s=s.replace(old,new,1)
start=s.index('async def test_custom_endpoint_openai_compatible_resolves_entry_url');end=s.index('\n\n@pytest',start);part=s[start:end].replace('config_provider=lambda: {\n','config_provider=lambda: {\n            "console": console_settings,\n',1).replace('assert resolved.execution_key == "custom-hosted"','assert resolved.execution_key == execution_key',1);s=s[:start]+part+s[end:];p.write_text(s)
