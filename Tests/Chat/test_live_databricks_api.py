"""Opt-in paid Databricks (AI Gateway) live verification checks.

Four live probes behind one gate: model listing (settles spec open item O-1 —
which listing route the gateway exposes), one non-streaming chat round-trip,
one streamed round-trip, and a wire-envelope capture that records the
response's top-level key names so unexpected extras can be encoded as the
provider record's ``response_allowances`` (never guessed, never silent).

Privacy follows ``test_live_moonshot_zai_api.py``: the child suppresses all
provider content; the only strings it ever prints are structural metadata —
route names, model ids, and sorted response KEY NAMES — never prompt text,
response text, or credentials.
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


def _live_enabled(environ: Mapping[str, str]) -> bool:
    return bool(environ.get("DATABRICKS_TOKEN", "").strip()) and bool(
        environ.get("DATABRICKS_HOST", "").strip()
    )


def _live_subprocess_environment(profile: Path, environ: Mapping[str, str]) -> dict[str, str]:
    env = dict(environ)
    env.update(
        {
            "HOME": str(profile / "home"),
            "XDG_CONFIG_HOME": str(profile / "xdg-config"),
            "XDG_DATA_HOME": str(profile / "xdg-data"),
            "TLDW_CONFIG_PATH": str(profile / "config" / "config.toml"),
            "DATABRICKS_LIVE_DATA_DIR": str(profile / "data"),
        }
    )
    return env


@pytest.mark.parametrize(
    ("token", "host", "expected"),
    [
        ("", "", False),
        ("dapi-test", "", False),
        ("", "https://adb-1.azuredatabricks.net", False),
        ("dapi-test", "https://adb-1.azuredatabricks.net", True),
    ],
)
def test_live_gate_requires_token_and_workspace_host(
    token: str, host: str, expected: bool
) -> None:
    environ = {"DATABRICKS_TOKEN": token, "DATABRICKS_HOST": host}
    assert _live_enabled(environ) is expected


def test_live_subprocess_isolates_profile_before_chatbook_imports() -> None:
    profile = Path("/tmp/databricks-live-structural-test")
    env = _live_subprocess_environment(profile, {})
    assert env["HOME"] == str(profile / "home")
    assert env["XDG_CONFIG_HOME"] == str(profile / "xdg-config")
    assert env["XDG_DATA_HOME"] == str(profile / "xdg-data")
    assert env["TLDW_CONFIG_PATH"] == str(profile / "config" / "config.toml")
    assert env["DATABRICKS_LIVE_DATA_DIR"] == str(profile / "data")
    assert _LIVE_CHILD.index("config_path.write_text") < _LIVE_CHILD.index(
        "from tldw_chatbook"
    )
    assert _LIVE_CHILD.index("config_path.chmod") < _LIVE_CHILD.index(
        "from tldw_chatbook"
    )
    compile(_LIVE_CHILD, "<databricks-live-child>", "exec")


_LIVE_CHILD = dedent(
    r"""
    from __future__ import annotations

    import json
    import os
    from pathlib import Path

    import requests

    token = os.environ["DATABRICKS_TOKEN"].strip()
    host = os.environ["DATABRICKS_HOST"].strip().rstrip("/")
    requested_model = os.environ.get("DATABRICKS_TEST_MODEL", "").strip()
    data_dir = Path(os.environ["DATABRICKS_LIVE_DATA_DIR"])
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # --- Probe 1: model listing (spec open item O-1) ---
    listing_route = None
    model_ids: list[str] = []
    listing_response = requests.get(
        f"{host}/openai/v1/models", headers=headers, timeout=60
    )
    if listing_response.status_code == 200:
        payload = listing_response.json()
        entries = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(entries, list):
            listing_route = "/openai/v1/models"
            model_ids = [
                item.get("id")
                for item in entries
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            ]
    if listing_route is None:
        serving_response = requests.get(
            f"{host}/api/2.0/serving-endpoints", headers=headers, timeout=60
        )
        if serving_response.status_code == 200:
            payload = serving_response.json()
            entries = payload.get("endpoints") if isinstance(payload, dict) else None
            if isinstance(entries, list):
                listing_route = "/api/2.0/serving-endpoints"
                model_ids = [
                    item.get("name")
                    for item in entries
                    if isinstance(item, dict) and isinstance(item.get("name"), str)
                ]
    if listing_route is None:
        raise SystemExit(
            "O1: neither /openai/v1/models nor /api/2.0/serving-endpoints "
            f"returned 200 ({listing_response.status_code}/"
            f"{serving_response.status_code})"
        )
    if not model_ids:
        raise SystemExit(
            f"O1: route {listing_route} worked but listed zero models; "
            "configure an AI Gateway external model on the workspace first"
        )
    model = requested_model or model_ids[0]
    if requested_model and requested_model not in model_ids:
        raise SystemExit(f"requested model {requested_model!r} not in listing")

    # Write the scratch config AFTER the probes that must not depend on it.
    # The BARE host exercises the engine's /openai/v1 append rule live.
    config_path.write_text(
        "\n".join(
            (
                "[general]",
                'users_name = "databricks-live-test"',
                "",
                "[paths]",
                f"data_dir = {json.dumps(str(data_dir))}",
                "",
                "[api_settings.databricks]",
                'api_key_env_var = "DATABRICKS_TOKEN"',
                f"api_base_url = {json.dumps(host)}",
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
    from tldw_chatbook.provider_registry import DATABRICKS

    handler = build_hosted_chat_handler(DATABRICKS)
    messages = [
        {"role": "system", "content": "Reply with exactly the word: ready"},
        {"role": "user", "content": "ready?"},
    ]

    # --- Probe 2: non-streaming round-trip ---
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
        f"{host}/openai/v1/chat/completions",
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
def test_live_databricks_gateway(tmp_path: Path) -> None:
    """One paid Databricks verification run behind the token+host gate."""
    if not _live_enabled(os.environ):
        pytest.skip("Set DATABRICKS_TOKEN and DATABRICKS_HOST to opt in.")
    env = _live_subprocess_environment(tmp_path / "databricks", os.environ)
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
        pytest.fail("Live Databricks verification timed out.")
    if completed.returncode != 0:
        pytest.fail(f"Live Databricks verification failed: {completed.stderr.strip()}")
    # The child prints one JSON line of structural metadata (probe evidence):
    # listing route (O-1 settlement), model count/id, and the response
    # envelope's key names — the source for any response_allowances entries.
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, "child printed no probe evidence"
    evidence = json.loads(lines[-1])
    assert evidence["listing_route"].startswith("/")
    assert evidence["model_count"] >= 1
    assert "choices" in evidence["response_key_names"]
