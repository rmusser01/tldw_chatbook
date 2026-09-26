"""Opt-in paid inference-cloud (Together / Fireworks / Cerebras) live probes.

Three live probes, one per provider, each behind a DOUBLE gate: the
per-provider opt-in flag (``TLDW_LIVE_TOGETHER`` / ``TLDW_LIVE_FIREWORKS``
/ ``TLDW_LIVE_CEREBRAS`` set to ``1``) PLUS the provider API key env var
(``TOGETHER_API_KEY`` / ``FIREWORKS_API_KEY`` / ``CEREBRAS_API_KEY``). An
optional ``TLDW_LIVE_<PROVIDER>_MODEL`` override picks the probe model
(default: the first listed model). Without cloud keys the probes skip
cleanly; the gate-matrix and profile-isolation structural tests below run
always.

Each probe (Phase 1's ``test_live_databricks_api.py`` subprocess pattern):
model listing, one non-streaming engine round-trip, one streamed engine
round-trip, and a wire-envelope capture.

ALLOWANCE RECONCILIATION: the three registry records ship EMPTY
response/choice/message allowances behind a PROVISIONAL PENDING FIRST
LIVE CAPTURE comment (Phase 2 Task 2 captured no cloud fixtures -- no
provider keys in that environment). The child's printed evidence line --
listing route, model id, and the response envelope's sorted KEY NAMES --
is exactly what reconciles each record's provisional allowances against
reality: run the probe once, compare the emitted ``response_key_names``
(and later choice/message key captures) with the strict core shape, amend
the record's allowance sets (never silently, never guessed). The
memory-to-verify notes in the registry (Together: top-level ``prompt`` +
choice ``logprobs``; Cerebras: top-level ``time_info``) are confirmed or
refuted by this evidence line, not by this test's assertions.

Privacy follows ``test_live_databricks_api.py``: the child suppresses all
provider content; the only strings it ever prints are structural
metadata -- route names, model ids, and sorted response KEY NAMES --
never prompt text, response text, or credentials.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

import pytest

# (provider key, opt-in flag env, api-key env, model-override env)
_LIVE_PROVIDER_ENV_VARS: tuple[tuple[str, str, str, str], ...] = (
    (
        "together",
        "TLDW_LIVE_TOGETHER",
        "TOGETHER_API_KEY",
        "TLDW_LIVE_TOGETHER_MODEL",
    ),
    (
        "fireworks",
        "TLDW_LIVE_FIREWORKS",
        "FIREWORKS_API_KEY",
        "TLDW_LIVE_FIREWORKS_MODEL",
    ),
    (
        "cerebras",
        "TLDW_LIVE_CEREBRAS",
        "CEREBRAS_API_KEY",
        "TLDW_LIVE_CEREBRAS_MODEL",
    ),
)
_LIVE_PROVIDER_KEYS = frozenset(key for key, _, _, _ in _LIVE_PROVIDER_ENV_VARS)


def _provider_env_names(provider_key: str) -> tuple[str, str, str]:
    """Return (opt-in flag env, api-key env, model-override env) names."""
    for key, flag_env, api_key_env, model_env in _LIVE_PROVIDER_ENV_VARS:
        if key == provider_key:
            return flag_env, api_key_env, model_env
    raise AssertionError(f"unknown live provider: {provider_key}")


def _live_enabled(provider_key: str, environ: Mapping[str, str]) -> bool:
    """Double gate: explicit opt-in flag AND a non-empty API key."""
    flag_env, api_key_env, _ = _provider_env_names(provider_key)
    return environ.get(flag_env, "").strip() == "1" and bool(
        environ.get(api_key_env, "").strip()
    )


def _live_subprocess_environment(
    provider_key: str, profile: Path, environ: Mapping[str, str]
) -> dict[str, str]:
    env = dict(environ)
    env.update(
        {
            "HOME": str(profile / "home"),
            "XDG_CONFIG_HOME": str(profile / "xdg-config"),
            "XDG_DATA_HOME": str(profile / "xdg-data"),
            "TLDW_CONFIG_PATH": str(profile / "config" / "config.toml"),
            "INFERENCE_CLOUD_LIVE_DATA_DIR": str(profile / "data"),
            "INFERENCE_CLOUD_LIVE_PROVIDER": provider_key,
        }
    )
    return env


@pytest.mark.parametrize(
    ("provider_key", "flag", "api_key", "expected"),
    [
        ("together", "", "", False),
        ("together", "1", "", False),
        ("together", "", "sk-test", False),
        ("together", "yes", "sk-test", False),  # the flag must be exactly 1
        ("together", "1", "sk-test", True),
        ("fireworks", "", "", False),
        ("fireworks", "1", "", False),
        ("fireworks", "", "fw-test", False),
        ("fireworks", "1", "fw-test", True),
        ("cerebras", "", "", False),
        ("cerebras", "1", "", False),
        ("cerebras", "", "csk-test", False),
        ("cerebras", "1", "csk-test", True),
    ],
)
def test_live_gate_requires_opt_in_flag_and_api_key(
    provider_key: str, flag: str, api_key: str, expected: bool
) -> None:
    flag_env, api_key_env, _ = _provider_env_names(provider_key)
    environ = {flag_env: flag, api_key_env: api_key}
    assert _live_enabled(provider_key, environ) is expected


def test_live_subprocess_isolates_profile_before_chatbook_imports() -> None:
    profile = Path("/tmp/inference-cloud-live-structural-test")
    for provider_key in sorted(_LIVE_PROVIDER_KEYS):
        env = _live_subprocess_environment(provider_key, profile, {})
        assert env["HOME"] == str(profile / "home")
        assert env["XDG_CONFIG_HOME"] == str(profile / "xdg-config")
        assert env["XDG_DATA_HOME"] == str(profile / "xdg-data")
        assert env["TLDW_CONFIG_PATH"] == str(profile / "config" / "config.toml")
        assert env["INFERENCE_CLOUD_LIVE_DATA_DIR"] == str(profile / "data")
        assert env["INFERENCE_CLOUD_LIVE_PROVIDER"] == provider_key
    # The child resolves every base URL from its own table, and an always-run
    # test below pins that table against the registry records' default URLs.
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    for provider_key, _, _, _ in _LIVE_PROVIDER_ENV_VARS:
        record = RECORDS_BY_KEY[provider_key]
        assert f'"{provider_key}": "{record.default_base_url}"' in _LIVE_CHILD
    # The scratch config is written (and permissioned) BEFORE any chatbook
    # import, so the engine resolves credentials from the isolated profile.
    assert _LIVE_CHILD.index("config_path.write_text") < _LIVE_CHILD.index(
        "from tldw_chatbook"
    )
    assert _LIVE_CHILD.index("config_path.chmod") < _LIVE_CHILD.index(
        "from tldw_chatbook"
    )
    # The child names every provider's key env explicitly (double gate's
    # second half reaches the subprocess environment unchanged) and never
    # interpolates the key value itself into anything it prints.
    for _, _, api_key_env, _ in _LIVE_PROVIDER_ENV_VARS:
        assert api_key_env in _LIVE_CHILD
    compile(_LIVE_CHILD, "<inference-cloud-live-child>", "exec")


_LIVE_CHILD = dedent(
    r"""
    from __future__ import annotations

    import json
    import os
    from pathlib import Path

    import requests

    provider_key = os.environ["INFERENCE_CLOUD_LIVE_PROVIDER"].strip()
    api_key_env = {
        "together": "TOGETHER_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
    }[provider_key]
    model_env = {
        "together": "TLDW_LIVE_TOGETHER_MODEL",
        "fireworks": "TLDW_LIVE_FIREWORKS_MODEL",
        "cerebras": "TLDW_LIVE_CEREBRAS_MODEL",
    }[provider_key]
    api_key = os.environ[api_key_env].strip()
    requested_model = os.environ.get(model_env, "").strip()
    data_dir = Path(os.environ["INFERENCE_CLOUD_LIVE_DATA_DIR"])
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # --- Probe 0: base URL table (pinned to the registry records by an
    # always-run structural test; kept inline so no chatbook import happens
    # before the scratch profile config is in place) ---
    base_url = {
        "together": "https://api.together.xyz/v1",
        "fireworks": "https://api.fireworks.ai/inference/v1",
        "cerebras": "https://api.cerebras.ai/v1",
    }[provider_key].rstrip("/")

    # --- Probe 1: model listing ---
    listing_route = "/models"
    listing_response = requests.get(
        f"{base_url}/models", headers=headers, timeout=60
    )
    if listing_response.status_code != 200:
        raise SystemExit(
            f"listing {base_url}/models returned {listing_response.status_code}"
        )
    payload = listing_response.json()
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise SystemExit("listing payload has no data list")
    model_ids = [
        item.get("id")
        for item in entries
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not model_ids:
        raise SystemExit("listing worked but listed zero models")
    model = requested_model or model_ids[0]
    if requested_model and requested_model not in model_ids:
        raise SystemExit(f"requested model {requested_model!r} not in listing")

    # Write the scratch config AFTER the probes that must not depend on it.
    config_path.write_text(
        "\n".join(
            (
                "[general]",
                'users_name = "inference-cloud-live-test"',
                "",
                "[paths]",
                f"data_dir = {json.dumps(str(data_dir))}",
                "",
                f"[api_settings.{provider_key}]",
                f"api_key_env_var = {json.dumps(api_key_env)}",
                f"api_base_url = {json.dumps(base_url)}",
                f"model = {json.dumps(model)}",
                "streaming = true",
                "",
            )
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o600)

    from loguru import logger

    logger.remove()

    from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
        build_hosted_chat_handler,
    )
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    record = RECORDS_BY_KEY[provider_key]
    if not record.engine_driven:
        raise SystemExit(f"{provider_key} is not an engine-driven record")
    if record.default_base_url is None or base_url != record.default_base_url.rstrip(
        "/"
    ):
        raise SystemExit(
            f"{provider_key} probe base URL drifted from the registry record"
        )

    handler = build_hosted_chat_handler(record)
    messages = [
        {"role": "system", "content": "Reply with exactly the word: ready"},
        {"role": "user", "content": "ready?"},
    ]

    # --- Probe 2: non-streaming round-trip (strict engine end-to-end) ---
    response = handler(
        input_data=[dict(message) for message in messages],
        streaming=False,
    )
    text = response["choices"][0]["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise SystemExit("chat round-trip returned empty text")
    if not isinstance(response.get("usage"), dict):
        raise SystemExit("chat round-trip returned no usage")

    # --- Probe 3: streamed round-trip ---
    stream = handler(
        input_data=[dict(message) for message in messages],
        streaming=True,
    )
    consumed = list(stream)
    terminal = stream.terminal_turn
    if not consumed:
        raise SystemExit("streamed round-trip produced no visible chunks")
    if terminal.usage is None:
        raise SystemExit("streamed round-trip terminal turn has no usage")
    stream.close()

    # --- Probe 4: wire-envelope capture (structural metadata only) ---
    envelope_response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={"model": model, "messages": messages[:1], "max_tokens": 8},
        timeout=90,
    )
    envelope_response.raise_for_status()
    key_names = sorted(envelope_response.json().keys())

    # Only structural metadata ever reaches stdout: no content, no keys.
    print(
        json.dumps(
            {
                "provider": provider_key,
                "listing_route": listing_route,
                "model_count": len(model_ids),
                "model": model,
                "response_key_names": key_names,
            }
        )
    )
    """
)


@pytest.mark.allow_network
@pytest.mark.integration
@pytest.mark.parametrize("provider_key", sorted(_LIVE_PROVIDER_KEYS))
def test_live_inference_cloud_provider(provider_key: str, tmp_path: Path) -> None:
    """One paid inference-cloud verification run behind the double gate."""
    if not _live_enabled(provider_key, os.environ):
        flag_env, api_key_env, _ = _provider_env_names(provider_key)
        pytest.skip(f"Set {flag_env}=1 and {api_key_env} to opt in.")
    env = _live_subprocess_environment(
        provider_key, tmp_path / provider_key, os.environ
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _LIVE_CHILD],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=360,
            text=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Live {provider_key} verification timed out.")
    if completed.returncode != 0:
        pytest.fail(
            f"Live {provider_key} verification failed: {completed.stderr.strip()}"
        )
    # The child prints one JSON line of structural metadata (probe
    # evidence): provider, listing route, model count/id, and the response
    # envelope's key names -- the source for reconciling the record's
    # PROVISIONAL response_allowances (amend, never silent).
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, "child printed no probe evidence"
    evidence = json.loads(lines[-1])
    assert evidence["provider"] == provider_key
    assert evidence["listing_route"] == "/models"
    assert evidence["model_count"] >= 1
    assert isinstance(evidence["model"], str) and evidence["model"].strip()
    assert "choices" in evidence["response_key_names"]
