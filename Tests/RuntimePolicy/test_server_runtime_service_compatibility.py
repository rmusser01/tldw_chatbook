import inspect

import tldw_chatbook.Server_Runtime_Interop.server_runtime_service as runtime_module
from tldw_chatbook.Server_Runtime_Interop.server_runtime_service import ServerRuntimeService


def test_server_runtime_service_module_does_not_reference_legacy_config_client_builders():
    source = inspect.getsource(runtime_module)

    assert "build_runtime_api_client_from_config" not in source
    assert "build_runtime_api_client(app_config" not in source


def test_server_runtime_service_from_config_delegates_through_provider_seam():
    service = ServerRuntimeService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerRuntimeService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client


def test_server_runtime_service_from_app_config_delegates_through_provider_seam():
    service = ServerRuntimeService.from_app_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerRuntimeService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client
