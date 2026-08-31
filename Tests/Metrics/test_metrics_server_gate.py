"""The Prometheus listener must not bind unless the user asked for it.

TASK-25914. Before this, ``init_metrics_server`` bound a socket whenever
``prometheus_client`` was importable, so installing the ``dev`` or ``debugging``
extra silently opened an unauthenticated HTTP listener. Dependency presence is
not consent.

These tests force ``PROMETHEUS_AVAILABLE`` on. Without that the gate under test
is masked -- the function would return ``False`` because the optional dependency
is absent, not because the gate works, and every assertion here would pass for
the wrong reason.
"""

from __future__ import annotations

import logging

import pytest

import tldw_chatbook.Metrics.metrics as prometheus_metrics


class _RecordingBinder:
    """Stands in for ``prometheus_client.start_http_server``."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, port, addr=None, **kwargs):
        self.calls.append({"port": port, "addr": addr, **kwargs})


@pytest.fixture
def binder(monkeypatch: pytest.MonkeyPatch) -> _RecordingBinder:
    recorder = _RecordingBinder()
    monkeypatch.setattr(prometheus_metrics, "PROMETHEUS_AVAILABLE", True)
    monkeypatch.setattr(
        prometheus_metrics, "start_http_server", recorder, raising=False
    )
    return recorder


def _with_config(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    resolved = {"enabled": False, "port": 8000, "bind_address": "127.0.0.1"}
    resolved.update(overrides)
    monkeypatch.setattr(
        prometheus_metrics, "_metrics_server_config", lambda: dict(resolved)
    )


def test_listener_does_not_bind_when_disabled(monkeypatch, binder):
    _with_config(monkeypatch, enabled=False)

    started = prometheus_metrics.init_metrics_server()

    assert started is False
    assert binder.calls == [], "no socket may be bound while metrics are disabled"


def test_listener_binds_when_explicitly_enabled(monkeypatch, binder):
    _with_config(monkeypatch, enabled=True, port=9123)

    started = prometheus_metrics.init_metrics_server()

    assert started is True
    assert len(binder.calls) == 1
    assert binder.calls[0]["port"] == 9123


def test_listener_binds_to_configured_address(monkeypatch, binder):
    _with_config(monkeypatch, enabled=True, bind_address="127.0.0.1")

    prometheus_metrics.init_metrics_server()

    assert binder.calls[0]["addr"] == "127.0.0.1"


def test_metrics_disabled_by_default(monkeypatch):
    """The shipped default must be off, not merely settable to off."""
    monkeypatch.setattr(
        prometheus_metrics,
        "_get_cli_setting",
        lambda section, key, default: default,
    )

    assert prometheus_metrics._metrics_server_config()["enabled"] is False


def test_bind_address_defaults_to_loopback(monkeypatch):
    """prometheus_client defaults to 0.0.0.0; we must not inherit that."""
    monkeypatch.setattr(
        prometheus_metrics,
        "_get_cli_setting",
        lambda section, key, default: default,
    )

    assert prometheus_metrics._metrics_server_config()["bind_address"] == "127.0.0.1"


def test_started_listener_reports_address_and_port(monkeypatch, binder, caplog):
    _with_config(monkeypatch, enabled=True, port=9123, bind_address="127.0.0.1")

    with caplog.at_level(logging.INFO):
        prometheus_metrics.init_metrics_server()

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "9123" in logged
    assert "127.0.0.1" in logged


def test_collection_still_runs_while_listener_disabled(monkeypatch, binder):
    """Disabling the listener must not disable metric collection itself.

    Scope note: with ``prometheus_client`` absent from the test environment the
    metric classes are no-op stand-ins, so this asserts the call path stays
    reachable and raises nothing -- not that a value lands in a registry. That
    stronger claim was verified out-of-band against the real dependency and is
    recorded in TASK-25914's implementation notes (counter 3.0, histogram count
    1.0, listener off).
    """
    _with_config(monkeypatch, enabled=False)
    assert prometheus_metrics.init_metrics_server() is False

    prometheus_metrics.log_counter("task_25914_probe", value=1)
    prometheus_metrics.log_histogram("task_25914_probe_seconds", value=0.25)


def test_metrics_port_env_var_overrides_configured_port(monkeypatch):
    """METRICS_PORT kept working as a port override (it predates this task).

    It deliberately does NOT enable the listener: AC#1 requires an explicit
    config setting for that. The unimplemented 2026-08-12 launch-diagnostics
    plan proposed env-alone-opts-in; this task's criteria say otherwise, so the
    env var moves the port and nothing else.
    """
    monkeypatch.setenv("METRICS_PORT", "9555")
    monkeypatch.setattr(
        prometheus_metrics,
        "_get_cli_setting",
        lambda section, key, default: default,
    )

    resolved = prometheus_metrics._metrics_server_config()

    assert resolved["port"] == 9555
    assert resolved["enabled"] is False, "env var must not enable the listener"


# --- coercion: the gate must fail CLOSED on odd values, not open -------------
# Review finding 1/3/5 on TASK-25914. bool("false") is True in Python, so a user
# who quotes the boolean out of YAML/env habit meant "off" and would have got an
# unauthenticated listener -- the same fail-open shape this task set out to
# remove, relocated from dependency-presence to value coercion.


@pytest.mark.parametrize("raw", ["false", "0", "no", "off", ""])
def test_stringly_false_does_not_enable_the_listener(monkeypatch, raw):
    monkeypatch.setattr(
        prometheus_metrics,
        "_get_cli_setting",
        lambda section, key, default: raw if key == "enabled" else default,
    )

    assert prometheus_metrics._metrics_server_config()["enabled"] is False


@pytest.mark.parametrize("raw", ["true", "1", "yes"])
def test_stringly_true_still_enables_the_listener(monkeypatch, raw):
    monkeypatch.setattr(
        prometheus_metrics,
        "_get_cli_setting",
        lambda section, key, default: raw if key == "enabled" else default,
    )

    assert prometheus_metrics._metrics_server_config()["enabled"] is True


@pytest.mark.parametrize("raw", [0, "", None, [], 123])
def test_unusable_bind_address_falls_back_to_loopback(monkeypatch, raw):
    """str(0) is "0", which getaddrinfo resolves to 0.0.0.0 -- all interfaces."""
    monkeypatch.setattr(
        prometheus_metrics,
        "_get_cli_setting",
        lambda section, key, default: raw if key == "bind_address" else default,
    )

    assert prometheus_metrics._metrics_server_config()["bind_address"] == "127.0.0.1"


@pytest.mark.parametrize("raw", [0, 70000, "garbage", None])
def test_unusable_port_falls_back_to_default(monkeypatch, raw):
    monkeypatch.setattr(
        prometheus_metrics,
        "_get_cli_setting",
        lambda section, key, default: raw if key == "port" else default,
    )

    assert prometheus_metrics._metrics_server_config()["port"] == 9090


def test_garbage_env_port_falls_back_to_the_configured_port(monkeypatch):
    """A junk env var must not discard a deliberately configured port."""
    monkeypatch.setenv("METRICS_PORT", "not-a-port")
    monkeypatch.setattr(
        prometheus_metrics,
        "_get_cli_setting",
        lambda section, key, default: 9000 if key == "port" else default,
    )

    assert prometheus_metrics._metrics_server_config()["port"] == 9000


def test_non_loopback_bind_is_logged_as_a_warning(monkeypatch, binder, caplog):
    """Exposing the endpoint beyond localhost must be hard to miss."""
    _with_config(monkeypatch, enabled=True, port=9123, bind_address="0.0.0.0")

    with caplog.at_level(logging.DEBUG):
        prometheus_metrics.init_metrics_server()

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "a non-loopback bind should warn"
    assert "0.0.0.0" in " ".join(warnings)


def test_shipped_config_template_ships_metrics_off():
    """Pins the default that actually ships, not the module constant."""
    import tomllib

    from tldw_chatbook import config as config_module

    section = tomllib.loads(config_module.CONFIG_TOML_CONTENT)["metrics"]
    assert section["enabled"] is False
    assert section["bind_address"] == "127.0.0.1"


@pytest.mark.parametrize("raw", ["on", "enable", "sure", "maybe", object()])
def test_unrecognised_enabled_value_fails_closed(monkeypatch, raw):
    """An value the coercion helper does not understand must mean OFF.

    Note the asymmetry in config.coerce_bool_setting: "off" is in its falsy set
    but "on" is not in its truthy set. For a gate whose default is off that is
    the safe direction -- both land on "do not bind" -- so this pins the
    behaviour rather than treating it as a bug to route around.
    """
    monkeypatch.setattr(
        prometheus_metrics,
        "_get_cli_setting",
        lambda section, key, default: raw if key == "enabled" else default,
    )

    assert prometheus_metrics._metrics_server_config()["enabled"] is False


def test_real_config_chain_resolves_a_user_enabled_listener(tmp_path):
    """Pins the actual get_cli_setting("metrics", ...) chain, not a stub.

    Every other test here monkeypatches the resolver, so all of them would stay
    green if the config lookup shape silently stopped resolving -- a failure
    this repo has had before (the dotted-section trap, TASK-1771). This one
    writes a real config file and reads it through the real loader.

    Out-of-process because config is cached per-process and TLDW_CONFIG_PATH is
    consulted at import time.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[metrics]\nenabled = true\nport = 9191\nbind_address = "127.0.0.1"\n',
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, TLDW_CONFIG_PATH=str(config_file))
    env.pop("METRICS_PORT", None)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json;"
            "import tldw_chatbook.Metrics.metrics as m;"
            "print('RESULT=' + json.dumps(m._metrics_server_config()))",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = next(
        ln for ln in result.stdout.splitlines() if ln.startswith("RESULT=")
    )
    resolved = json.loads(line[len("RESULT=") :])

    assert resolved == {
        "enabled": True,
        "port": 9191,
        "bind_address": "127.0.0.1",
    }
