"""Opt-in local Models verification; fixture counters are not live evidence."""

import os
from pathlib import Path

import pytest


def live_inputs() -> dict[str, Path]:
    if os.environ.get("TLDW_LLAMA_SNAPSHOT_LIVE") != "1":
        pytest.skip(
            "Set TLDW_LLAMA_SNAPSHOT_LIVE=1 with local server/model/media assets"
        )
    names = ("SERVER", "MODEL", "MMPROJ", "IMAGE_A", "IMAGE_B")
    result = {}
    for name in names:
        raw = os.environ.get(f"TLDW_LLAMA_SNAPSHOT_{name}")
        if not raw or not Path(raw).is_file():
            pytest.fail(f"Missing local snapshot live input: {name}")
        result[name] = Path(raw)
    return result


def cache_counter(payload: dict) -> int:
    timings = payload.get("timings")
    value = timings.get("cache_n") if isinstance(timings, dict) else None
    if type(value) is not int or value < 0:
        pytest.fail("Missing or invalid timings.cache_n; no live cache evidence")
    return value


def assert_media_reuse(evidence: dict[str, tuple[int, int]]) -> None:
    cold, native, same, different = (
        evidence[name] for name in ("cold_a", "native_ab", "restored_aa", "restored_ab")
    )
    if cold[1] != same[1] or native[1] != different[1]:
        pytest.fail("Prompt totals changed across the controlled restart")
    if same[0] <= max(cold[0], native[0]):
        pytest.fail("Restored same-image reuse does not exceed the native text prefix")
    if different[0] > native[0]:
        pytest.fail("Restored different-image reuse exceeds the native media boundary")


async def configure_live_profile() -> None:
    """Persist only to pytest's already-isolated profile, before app construction."""
    import asyncio

    from tldw_chatbook.config import save_settings_to_cli_config
    from tldw_chatbook.LLM_Management.snapshot_settings import (
        SnapshotPreferences,
        save_snapshot_preferences,
    )

    assert await asyncio.to_thread(
        save_settings_to_cli_config, {"splash_screen": {"enabled": False}}
    )
    assert await asyncio.to_thread(
        save_snapshot_preferences, SnapshotPreferences(enabled=True, keep_count=10)
    )


@pytest.mark.asyncio
@pytest.mark.loopback_network
@pytest.mark.timeout(
    4800
)  # Six potentially ten-minute mutations plus local CPU startup.
async def test_models_live_persistence_and_media_reuse(
    tmp_path, monkeypatch, record_property
):
    assets = live_inputs()  # Before application imports, sockets, or process creation.

    import asyncio
    import base64
    import hashlib
    import json
    import mimetypes
    import shlex
    import socket

    import httpx
    from textual.widgets import Button, Collapsible, DataTable, Input, Select

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
        current_server_claim,
        server_process,
        stop_server_process,
    )
    from tldw_chatbook.LLM_Management.snapshot_settings import load_snapshot_preferences
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
    from tldw_chatbook.Widgets.llamacpp_snapshot_manager import LlamaCppSnapshotManager

    # Root conftest owns config/HOME isolation; the mounted app's data root is
    # explicitly scratch-owned too. No service/client/store/lifecycle is replaced.
    data_root = tmp_path.resolve() / "live-data"
    data_root.mkdir(mode=0o700)
    monkeypatch.setattr("tldw_chatbook.app.get_user_data_dir", lambda: data_root)
    await configure_live_profile()
    # The real Models view probes Ollama on mount; that unrelated external
    # discovery is not part of this isolated llama.cpp verification.
    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        lambda self: asyncio.sleep(0, result=False),
    )

    def image_data(name):
        payload = assets[name].read_bytes()
        media_type = mimetypes.guess_type(assets[name].name)[0]
        if media_type not in {"image/png", "image/jpeg", "image/webp"}:
            pytest.fail("Live images must use PNG, JPEG, or WebP extensions")
        return hashlib.sha256(
            payload
        ).digest(), f"data:{media_type};base64,{base64.b64encode(payload).decode()}"

    image_a, image_b = await asyncio.gather(
        asyncio.to_thread(image_data, "IMAGE_A"),
        asyncio.to_thread(image_data, "IMAGE_B"),
    )
    if image_a[0] == image_b[0]:
        pytest.fail("Live image inputs must have distinct SHA-256 byte identities")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    # Production preflight rejects an intervening listener; never adopt it.
    base_url = f"http://127.0.0.1:{port}"
    app = _build_test_app()
    service = app.llamacpp_snapshot_service
    processes = []
    evidence = {}

    async with app.run_test(size=(140, 45)) as pilot:
        try:
            await app.push_screen(LLMScreen(app))
            await pilot.pause()
            window = app.screen.query_one(LLMManagementWindow)
            manager = window.query_one(LlamaCppSnapshotManager)

            async def capture(label):
                await pilot.pause()
                app.save_screenshot(f"{label}.svg", path=str(tmp_path))
                print(f"Live Models stage: {label}", flush=True)

            async def until(predicate, label, seconds=120):
                try:
                    async with asyncio.timeout(seconds):
                        while not predicate():
                            await pilot.pause(0.05)
                except TimeoutError:
                    pytest.fail(f"Live Models boundary did not settle: {label}")

            await until(
                lambda: service.view().storage_location is not None, "private storage"
            )
            window.query_one("#llamacpp-gguf-source-mode", Select).value = "external"
            for selector, value in {
                "exec-path": str(assets["SERVER"]),
                "model-path": str(assets["MODEL"]),
                "host": "127.0.0.1",
                "port": str(port),
                "additional-args": shlex.join(
                    (
                        "--ctx-size",
                        "8192",
                        "--parallel",
                        "1",
                        "--flash-attn",
                        "off",
                        "--fit",
                        "off",
                        "--device",
                        "none",
                        "--n-gpu-layers",
                        "0",
                        "--mmproj",
                        str(assets["MMPROJ"]),
                        "--mmproj-device",
                        "none",
                        "--no-mmproj-offload",
                        "--swa-full",
                        "--cache-ram",
                        "0",
                        "--no-warmup",
                    )
                ),
            }.items():
                window.query_one(f"#llamacpp-{selector}", Input).value = value
            await pilot.pause()

            async def press(selector):
                control = window.query_one(selector, Button)
                # Textual ignores keyboard presses during its active feedback.
                await until(
                    lambda: not control.disabled and not control.has_class("-active"),
                    "action enabled and accepting keyboard input",
                )
                control.focus()
                await pilot.pause()
                await pilot.press("enter")

            async def start():
                await press("#llamacpp-start-server-button")
                await until(
                    lambda: server_process(app, "llamacpp") is not None, "owned child"
                )
                processes.append(server_process(app, "llamacpp"))
                async with asyncio.timeout(120):
                    while service.view().status != "idle":
                        await press("#snapshot-refresh")
                        await pilot.pause(0.25)
                        if processes[-1].poll() is not None:
                            pytest.fail("Selected live server exited before readiness")
                assert len(service.view().slots) == 1
                await capture(f"started-{len(processes)}")

            async def stop():
                await press("#llamacpp-stop-server-button")
                await until(
                    lambda: current_server_claim(app, "llamacpp") is None,
                    "confirmed Stop",
                )
                assert processes[-1].poll() is not None
                print(f"Live Models stopped/reaped child {len(processes)}", flush=True)

            async def mutate(action):
                seen = False
                complete = asyncio.Event()

                def changed():
                    nonlocal seen
                    view = service.view()
                    seen |= view.operation_id is not None
                    if seen and (
                        view.operation_id is None or view.status == "outcome_unknown"
                    ):
                        complete.set()

                unsubscribe = service.subscribe(changed)
                try:
                    if action == "restore":
                        manager.query_one("#snapshot-records", DataTable).move_cursor(
                            row=0
                        )
                        await pilot.pause()
                    await press(f"#snapshot-{action}")
                    if action == "restore":
                        await until(
                            lambda: bool(app.screen.query("#confirm-button")),
                            "Restore confirmation",
                        )
                        app.screen.query_one("#confirm-button", Button).press()
                    await until(lambda: seen, "mutation admission", seconds=10)
                    await until(complete.is_set, "acknowledged mutation", seconds=620)
                    view = service.view()
                    await capture(f"{action}-{len(processes)}")
                    if view.status != "idle" or view.message is not None:
                        pytest.fail(
                            "Live snapshot mutation was not a clean acknowledged completion: "
                            f"status={view.status}, reason={view.message}"
                        )
                finally:
                    unsubscribe()

            async def chat(client, image=None):
                content = [
                    {"type": "text", "text": "Describe this input briefly and calmly."}
                ]
                if image is not None:
                    content.append({"type": "image_url", "image_url": {"url": image}})
                # Ordinary OpenAI-compatible send, deliberately no id_slot.
                try:
                    response = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "local",
                            "messages": [{"role": "user", "content": content}],
                            "max_tokens": 1,
                            "temperature": 0,
                            "seed": 1,
                            "stream": False,
                            "cache_prompt": True,
                        },
                    )
                    if response.status_code != 200:
                        pytest.fail(
                            "Live ordinary chat request failed; response content withheld"
                        )
                    payload = response.json()
                except (httpx.HTTPError, ValueError):
                    pytest.fail(
                        "Live ordinary chat transport failed; request content withheld",
                        pytrace=False,
                    )
                cached = cache_counter(payload)
                processed = payload["timings"].get("prompt_n")
                if type(processed) is not int or processed < 0:
                    pytest.fail(
                        "Missing or invalid timings.prompt_n; no live cache evidence"
                    )
                await press("#snapshot-refresh")
                await until(
                    lambda: service.view().status == "idle", "post-chat observations"
                )
                print(f"Live chat cache_n={cached}, prompt_n={processed}", flush=True)
                return cached, cached + processed

            async with httpx.AsyncClient(
                base_url=base_url, trust_env=False, follow_redirects=False, timeout=600
            ) as client:
                await start()
                evidence["cold_text"] = await chat(client)
                await mutate("save")
                await stop()
                await start()
                await mutate("restore")
                evidence["restored_text"] = await chat(client)
                if (
                    evidence["cold_text"][1] != evidence["restored_text"][1]
                    or evidence["restored_text"][0] <= evidence["cold_text"][0]
                ):
                    pytest.fail(
                        "Restored text did not demonstrate matching-prefix reuse"
                    )
                await stop()

                await start()
                evidence["cold_a"] = await chat(client, image_a[1])
                evidence["native_ab"] = await chat(client, image_b[1])
                await stop()
                await start()
                populated_a = await chat(client, image_a[1])
                if populated_a != evidence["cold_a"]:
                    pytest.fail("Fresh image control changed before snapshot creation")
                await mutate("save")
                newest = service.view().catalog.records[0]
                if "427291b" not in newest.compatibility.build_info:
                    pytest.fail(
                        "Live oracle requires the reviewed llama.cpp build 427291b"
                    )
                await stop()
                await start()
                await mutate("restore")
                evidence["restored_aa"] = await chat(client, image_a[1])
                await stop()
                await start()
                await mutate("restore")
                evidence["restored_ab"] = await chat(client, image_b[1])
                assert_media_reuse(evidence)
                record_property(
                    "snapshot_live_counters", json.dumps(evidence, sort_keys=True)
                )
                print(
                    "Snapshot live cache_n/prompt_total: "
                    + json.dumps(evidence, sort_keys=True)
                )

                # Real files and UI preference persistence, not a mocked catalog.
                original_ids = {
                    record.snapshot_id for record in service.view().catalog.records
                }
                assert load_snapshot_preferences().keep_count == 10
                for _ in range(11):
                    await mutate("save")
                retained = service.view().catalog.records
                assert len(retained) == 10
                assert original_ids.isdisjoint(
                    record.snapshot_id for record in retained
                )
                await capture("retention-default-ten")
                manager.query_one(
                    "#snapshot-details-panel", Collapsible
                ).collapsed = False
                manager.query_one("#snapshot-keep", Input).value = "2"
                await press("#snapshot-apply")
                await until(
                    lambda: load_snapshot_preferences().keep_count == 2,
                    "persisted retention preference",
                )
                assert len(service.view().catalog.records) == 10
                await mutate("save")
                assert len(service.view().catalog.records) == 2
                await capture("retention-lowered-after-save")

                selected = service.view().catalog.records[0]
                manager.query_one("#snapshot-records", DataTable).move_cursor(row=0)
                await pilot.pause()
                await press("#snapshot-delete")
                await until(
                    lambda: bool(app.screen.query("#confirm-button")),
                    "Delete confirmation",
                )
                await capture("delete-confirmation")
                await pilot.press("escape")
                assert selected in service.view().catalog.records
                await press("#snapshot-delete")
                await until(
                    lambda: bool(app.screen.query("#confirm-button")),
                    "Delete confirmation again",
                )
                app.screen.query_one("#confirm-button", Button).press()
                await until(
                    lambda: len(service.view().catalog.records) == 1,
                    "confirmed Delete",
                )
                assert selected not in service.view().catalog.records
                assert not (service.store.catalog / selected.filename).exists()
                assert processes[-1].poll() is None
                await capture("deleted-one-server-still-running")
                record_property("snapshot_live_retention_delete", "passed")
        finally:
            await stop_server_process(app, "llamacpp", "Live verification server")
            async with asyncio.timeout(30):
                await asyncio.gather(
                    *(
                        worker.wait()
                        for worker in app.workers
                        if worker.group == "llamacpp_server"
                    )
                )
            await service.shutdown()
            assert all(process.poll() is not None for process in processes), (
                "An owned child was not reaped"
            )


@pytest.mark.parametrize("flag", [None, "", "0", "true", "yes"])
def test_live_gate_requires_exact_opt_in(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("TLDW_LLAMA_SNAPSHOT_LIVE", raising=False)
    else:
        monkeypatch.setenv("TLDW_LLAMA_SNAPSHOT_LIVE", flag)
    with pytest.raises(pytest.skip.Exception, match="Set TLDW_LLAMA_SNAPSHOT_LIVE=1"):
        live_inputs()


@pytest.mark.parametrize("missing", ["SERVER", "MODEL", "MMPROJ", "IMAGE_A", "IMAGE_B"])
def test_opted_in_missing_asset_fails_without_exposing_path(
    monkeypatch, tmp_path, missing
):
    monkeypatch.setenv("TLDW_LLAMA_SNAPSHOT_LIVE", "1")
    asset = tmp_path / "private-asset"
    asset.touch()
    for name in ("SERVER", "MODEL", "MMPROJ", "IMAGE_A", "IMAGE_B"):
        monkeypatch.setenv(f"TLDW_LLAMA_SNAPSHOT_{name}", str(asset))
    monkeypatch.setenv(
        f"TLDW_LLAMA_SNAPSHOT_{missing}", str(tmp_path / "private-missing")
    )
    with pytest.raises(pytest.fail.Exception) as failure:
        live_inputs()
    assert str(failure.value) == f"Missing local snapshot live input: {missing}"


def test_opted_in_local_assets_are_returned(monkeypatch, tmp_path):
    monkeypatch.setenv("TLDW_LLAMA_SNAPSHOT_LIVE", "1")
    expected = {}
    for name in ("SERVER", "MODEL", "MMPROJ", "IMAGE_A", "IMAGE_B"):
        asset = tmp_path / name
        asset.touch()
        expected[name] = asset
        monkeypatch.setenv(f"TLDW_LLAMA_SNAPSHOT_{name}", str(asset))
    assert live_inputs() == expected


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"timings": {}},
        {"timings": {"cache_n": None}},
        {"timings": {"cache_n": True}},
        {"timings": {"cache_n": -1}},
        {"timings": {"cache_n": "3"}},
    ],
)
def test_missing_or_invalid_live_counter_cannot_become_zero(payload):
    with pytest.raises(
        pytest.fail.Exception, match="Missing or invalid timings.cache_n"
    ):
        cache_counter(payload)


@pytest.mark.parametrize("value", [0, 3, 8192])
def test_live_counter_accepts_observed_nonnegative_integer(value):
    assert cache_counter({"timings": {"cache_n": value}}) == value


@pytest.mark.asyncio
async def test_live_profile_setup_uses_real_isolated_bulk_save_api():
    from tldw_chatbook.config import get_cli_setting
    from tldw_chatbook.LLM_Management.snapshot_settings import load_snapshot_preferences

    await configure_live_profile()
    assert get_cli_setting("splash_screen", "enabled") is False
    preferences = load_snapshot_preferences()
    assert preferences.enabled is True
    assert preferences.keep_count == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["0", "1"])
async def test_actual_live_entry_gates_before_process_or_network(
    monkeypatch, tmp_path, flag
):
    import socket
    import subprocess

    import httpx

    monkeypatch.setenv("TLDW_LLAMA_SNAPSHOT_LIVE", flag)
    monkeypatch.delenv("TLDW_LLAMA_SNAPSHOT_SERVER", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Live gate crossed a process/network boundary")

    with monkeypatch.context() as guarded:
        guarded.setattr(socket, "socket", forbidden)
        guarded.setattr(subprocess, "Popen", forbidden)
        guarded.setattr(httpx, "AsyncClient", forbidden)
        expected = pytest.skip.Exception if flag == "0" else pytest.fail.Exception
        with pytest.raises(expected):
            await test_models_live_persistence_and_media_reuse(
                tmp_path, guarded, lambda *args: None
            )


def test_native_media_boundary_accepts_positive_differential_evidence():
    assert_media_reuse(
        {
            "cold_a": (0, 64),
            "native_ab": (5, 80),
            "restored_aa": (63, 64),
            "restored_ab": (5, 80),
        }
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("restored_aa", (5, 64)),
        ("restored_ab", (6, 80)),
        ("restored_aa", (63, 65)),
        ("restored_ab", (5, 81)),
    ],
)
def test_native_media_boundary_rejects_text_only_reuse_or_changed_totals(key, value):
    evidence = {
        "cold_a": (0, 64),
        "native_ab": (5, 80),
        "restored_aa": (63, 64),
        "restored_ab": (5, 80),
    }
    evidence[key] = value
    with pytest.raises(pytest.fail.Exception):
        assert_media_reuse(evidence)
