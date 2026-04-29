from types import SimpleNamespace


def test_app_wires_server_evaluation_scope_when_provider_backed_service_has_no_client(monkeypatch):
    from tldw_chatbook import app as app_module

    class LocalUnavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("local evaluations unavailable")

    server_service = SimpleNamespace(client=None, client_provider=object())

    monkeypatch.setattr(app_module, "EvaluationOrchestrator", LocalUnavailable)
    monkeypatch.setattr(
        app_module.ServerEvaluationsService,
        "from_config",
        classmethod(lambda cls, app_config, *, policy_enforcer=None: server_service),
    )

    app_like = SimpleNamespace(
        app_config={"tldw_api": {"base_url": "https://server.test"}},
        service_policy_enforcer=object(),
    )

    app_module.TldwCli._wire_evaluation_services(app_like)

    assert app_like.local_evaluation_service is None
    assert app_like.server_evaluation_service is server_service
    assert app_like.evaluation_scope_service is not None
    assert app_like.evaluation_scope_service.server_service is server_service
