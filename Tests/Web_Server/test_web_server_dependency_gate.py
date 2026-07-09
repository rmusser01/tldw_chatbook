from tldw_chatbook.Web_Server import serve


def test_check_web_server_available_refreshes_lazy_dependency_state(monkeypatch):
    """The web server gate must not trust the lazy default False cache.

    Args:
        monkeypatch: Pytest fixture for replacing dependency-gate state.
    """
    calls = []

    def fake_check_web_server_deps() -> bool:
        calls.append("checked")
        serve.DEPENDENCIES_AVAILABLE["web"] = True
        return True

    monkeypatch.setitem(serve.DEPENDENCIES_AVAILABLE, "web", False)
    monkeypatch.setattr(
        serve,
        "check_web_server_deps",
        fake_check_web_server_deps,
        raising=False,
    )

    assert serve.check_web_server_available() is True
    assert calls == ["checked"]


def test_check_web_server_available_handles_probe_exceptions(monkeypatch):
    """The availability gate must fail closed when dependency probing crashes.

    Args:
        monkeypatch: Pytest fixture for replacing dependency-gate state.
    """

    def broken_check_web_server_deps() -> bool:
        raise RuntimeError("broken optional dependency import")

    monkeypatch.setitem(serve.DEPENDENCIES_AVAILABLE, "web", False)
    monkeypatch.setitem(serve.DEPENDENCIES_AVAILABLE, "textual_serve", True)
    monkeypatch.setattr(
        serve,
        "check_web_server_deps",
        broken_check_web_server_deps,
        raising=False,
    )

    assert serve.check_web_server_available() is False
    assert serve.DEPENDENCIES_AVAILABLE["web"] is False
    assert serve.DEPENDENCIES_AVAILABLE["textual_serve"] is False


def test_main_cli_runner_serve_uses_web_dependency_gate(monkeypatch, tmp_path):
    """The installed CLI serve flag must use the refreshed web dependency gate.

    Args:
        monkeypatch: Pytest fixture for replacing app startup dependencies.
        tmp_path: Pytest fixture for creating isolated package/config files.
    """
    import atexit
    import signal
    import sys

    from tldw_chatbook import app as app_module

    package_root = tmp_path / "package"
    css_dir = package_root / "css"
    css_dir.mkdir(parents=True)
    (css_dir / "tldw_cli_modular.tcss").write_text("", encoding="utf-8")
    (css_dir / "build_css.py").write_text("", encoding="utf-8")

    run_calls = []

    def fake_run_web_server(**kwargs):
        run_calls.append(kwargs)

    monkeypatch.setattr(app_module, "__file__", str(package_root / "app.py"))
    monkeypatch.setattr(app_module, "DEFAULT_CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(app_module, "initialize_early_logging", lambda: object())
    monkeypatch.setattr(app_module, "supports_emoji", lambda: False)
    monkeypatch.setattr(app_module, "get_char", lambda _emoji, fallback: fallback)
    monkeypatch.setattr(atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(serve, "check_web_server_available", lambda: True)
    monkeypatch.setattr(serve, "run_web_server", fake_run_web_server)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tldw-cli",
            "--serve",
            "--host",
            "127.0.0.1",
            "--port",
            "8941",
            "--web-title",
            "Test Title",
            "--debug",
        ],
    )

    app_module.main_cli_runner()

    assert run_calls == [
        {
            "host": "127.0.0.1",
            "port": 8941,
            "title": "Test Title",
            "debug": True,
        }
    ]
