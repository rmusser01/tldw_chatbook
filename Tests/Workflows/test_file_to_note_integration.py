"""Joined controls/HTTP/Notes qualification; opt-in real-app localhost UAT.

No application imports at collection: both bootstrap and per-test TOML must
exist before the application's configuration-owning modules are imported.
"""

import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import httpx
import keyring
import pytest
import toml

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from Tests.private_profile import is_private_profile_child, private_profile_test

FIXTURE = ROOT / "Tests/fixtures/workflows/file_to_note.json"
MODEL = (
    "../../../Working/Language_Models/gemma-4-26B-A4B/"
    "gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf"
)
SOURCE = "The library opens at nine. Returns are free. Workshops run on Friday."
EDIT = "Human-edited summary"


def write_profile(root):
    """Prepare all authority before imports; never borrow a host profile."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = root / "data"
    config = root / "config/config.toml"
    data.mkdir(parents=True, exist_ok=True, mode=0o700)
    config.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    db_keys = set(
        re.findall(
            r'"([a-z_]+_db_path)"', (ROOT / "tldw_chatbook/config.py").read_text()
        )
    )
    values = {
        "paths": {"data_dir": str(data)},
        "database": {
            **{key: str(data / (key + ".db")) for key in db_keys},
            # Shutdown persists merged defaults. Predeclare these so strict
            # authority equality also holds on a genuinely fresh profile.
            "USER_DB_BASE_DIR": str(data),
            "check_integrity_on_startup": False,
            "integrity_check_timeout": 30,
        },
        "general": {"default_tab": "workflows"},
        "first_run": {"setup_completed": True},
        "splash_screen": {"enabled": False},
        "model_catalog": {"auto_refresh_enabled": False},
        "tldw_api": {"base_url": "http://127.0.0.1:1", "auth_token": ""},
        "providers": {"llama_cpp": [MODEL]},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://localhost:9099",
                "credential_source": "none",
                "timeout": 120,
            }
        },
        "embedding_config": {"model_cache_dir": str(root / "cache")},
        "logging": {"file_log_level": "INFO"},
        "web_server": {"enabled": False},
        "metrics": {"enabled": False},
        "hooks": {"enabled": False},
    }
    config.write_text(toml.dumps(values), encoding="utf-8")
    config.chmod(0o600)
    assert tomllib.loads(config.read_text()) == values
    return config


@pytest.fixture
def tmp_path(tmp_path, request):
    # The ancestor autouse fixture imports config AFTER requesting tmp_path.
    # Precreate its exact selected TOML, without creating its HOME directory.
    root = (
        Path(os.environ["TLDW_TEST_CONFIG_ROOT"])
        if is_private_profile_child(request)
        else tmp_path / "test_data"
    )
    if is_private_profile_child(request) and os.environ.get("TASK6_LIVE_PROFILE"):
        # The standalone launcher already prepared this exact profile. A fresh
        # restart must inspect its saved config, never replace it with defaults.
        assert root.resolve() == Path(os.environ["TASK6_LIVE_PROFILE"]).resolve()
        tomllib.loads((root / "config/config.toml").read_text())
    else:
        write_profile(root)
    return tmp_path


@pytest.fixture
def isolated_profile(isolate_test_environment, monkeypatch, request):
    if getattr(request.function, "_private_profile_test", False) and not (
        is_private_profile_child(request)
    ):
        return None  # Only the fresh child constructs or exercises the real app.
    from tldw_chatbook import config

    root = isolate_test_environment.resolve()
    assert Path(os.environ["TLDW_CONFIG_PATH"]).resolve().is_relative_to(root)
    tomllib.loads(Path(os.environ["TLDW_CONFIG_PATH"]).read_text())
    assert type(keyring.get_keyring()).__module__ == "keyring.backends.null"
    if os.environ.get("TASK6_LIVE_PROFILE"):
        root = Path(os.environ["TASK6_LIVE_PROFILE"]).resolve()
        persistent_config = root / "config/config.toml"
        tomllib.loads(persistent_config.read_text())
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(persistent_config))
    config.refresh_runtime_config_from_cli_config()
    paths = {"config": os.environ["TLDW_CONFIG_PATH"]}
    for name in dir(config):
        if name.startswith("get_") and name.endswith("_db_path"):
            paths[name] = str(getattr(config, name)())
    paths["data"] = str(config.get_user_data_dir())
    paths["cache"] = str(config.get_model_cache_dir())
    for name, value in paths.items():
        assert Path(value).resolve().is_relative_to(root), (name, value)
    assert Path(config.__file__).resolve().is_relative_to(ROOT)
    paths.update(
        {
            "bootstrap_root": os.environ["TLDW_TEST_CONFIG_ROOT"],
            "per_test_root": str(isolate_test_environment),
            **{
                name: value
                for name, value in os.environ.items()
                if name
                in {
                    "HOME",
                    "USERPROFILE",
                    "XDG_DATA_HOME",
                    "XDG_CONFIG_HOME",
                    "XDG_CACHE_HOME",
                    "HF_HOME",
                    "HF_HUB_CACHE",
                    "TIKTOKEN_CACHE_DIR",
                    "TMPDIR",
                    "PYTHON_KEYRING_BACKEND",
                    "PYTHONPATH",
                }
            },
        }
    )
    return root, paths


async def wait_for(pilot, predicate, seconds=30):
    async with asyncio.timeout(seconds):
        while not predicate():
            await pilot.pause(0.05)


async def press(app, pilot, selector):
    from textual.widgets import Button

    await wait_for(pilot, lambda: bool(app.screen.query(selector)))
    button = app.screen.query_one(selector, Button)
    await wait_for(
        pilot, lambda: not button.has_class("-active") and not button.disabled
    )
    assert not button.disabled, selector
    button.focus()
    button.scroll_visible(animate=False)
    await pilot.pause()
    assert app.focused is button
    region = button.region
    assert region.width and region.height
    painted, _ = app.screen.get_widget_at(
        region.x + region.width // 2, region.y + (region.height - 1) // 2
    )
    assert painted is button or button in painted.ancestors, selector
    await pilot.press("enter")
    await pilot.pause()


def permit(app, state="ask"):
    store = app.unified_mcp_service.permission_store
    payload = store.load()
    payload["profiles"]["default"]["servers"]["agent:builtin"] = {
        "tools": {
            name: {"state": state}
            for name in ("workflow_read_file", "workflow_local_model", "create_note")
        }
    }
    store.save(payload)


async def boot(app, pilot, root, *, seed=True):
    await wait_for(pilot, lambda: bool(app.screen.query("#workflow-run")), 60)
    assert app.chachanotes_db is not None
    assert app.notes_scope_service.local_notes_service is app.notes_service
    assert app.get_authoritative_runtime_source() == "local"
    assert (
        Path(app.unified_mcp_service.permission_store.path)
        .resolve()
        .is_relative_to(root)
    )
    await app.ensure_workflow_authoring()
    if not seed:
        assert app.workflow_documents.list_workflows(), (
            "restart must not recreate a definition"
        )
    if not app.workflow_documents.list_workflows():
        revision = await asyncio.to_thread(
            app.workflow_documents.create, FIXTURE.read_text()
        )
        await app.workflow_drafts.select(revision.workflow_id, revision.revision_id)
        # Reload through the production navigation route after document import.
        app.action_shell_destination("home")
        await wait_for(pilot, lambda: not app.screen.query("#workflow-run"))
        app.action_shell_destination("workflows")
        await wait_for(pilot, lambda: bool(app.screen.query("#workflow-run")))
    permit(app)
    tomllib.loads(Path(os.environ["TLDW_CONFIG_PATH"]).read_text())


async def setup(app, pilot, source, model, capture_setup=None):
    from textual.widgets import Input

    await press(app, pilot, "#workflow-run")
    app.screen.query_one("#workflow-source", Input).value = str(source)
    app.screen.query_one("#workflow-model", Input).value = model
    await pilot.pause()
    if capture_setup:
        capture_setup("setup-inputs")
    await press(app, pilot, "#workflow-setup-review")
    if capture_setup:
        capture_setup("setup-destinations")
    await press(app, pilot, "#workflow-start")
    await wait_for(pilot, lambda: app._workflow_session.view().state == "approval")


async def approve(app, pilot, step):
    session = app._workflow_session
    await wait_for(pilot, lambda: session.view().state == "approval")
    assert session.view().step_id == step, session.view()
    await press(app, pilot, "#workflow-effect-approve")


@pytest.mark.loopback_network
@pytest.mark.parametrize(
    "hold_nested_mount", [False, True], ids=["ordinary", "held-nested-mount"]
)
@private_profile_test
async def test_saved_file_controls_real_http_edited_local_note(
    request, isolated_profile, monkeypatch, hold_nested_mount
):
    """Catches lost reviewed text or recaptured mutable provider/Notes bindings."""
    from textual.widgets import TextArea

    from Tests.LLM_Calls.test_llamacpp_bounded import _answer, _listener, _response
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.config import save_setting_to_cli_config

    root, _ = isolated_profile
    source = root / "source.txt"
    source.write_text(SOURCE)
    source.chmod(0o600)
    async with _listener(_response(_answer("Generated response canary"))) as (
        origin,
        requests,
    ):
        save_setting_to_cli_config(
            "api_settings",
            "llama_cpp",
            {
                "api_url": origin,
                "credential_source": "none",
                "timeout": 5,
            },
        )
        app = TldwCli()
        assert (
            Path(__import__("tldw_chatbook.app", fromlist=["x"]).__file__)
            .resolve()
            .is_relative_to(ROOT)
        )
        async with app.run_test(size=(110, 36)) as pilot:
            await boot(app, pilot, root)
            await setup(app, pilot, source, "owned-peer-model")
            # tldw_server is absent; changing the current provider after launch
            # must not redirect the already captured request or Note owner.
            save_setting_to_cli_config(
                "api_settings",
                "llama_cpp",
                {
                    "api_url": "http://127.0.0.1:1",
                    "credential_source": "none",
                    "timeout": 5,
                },
            )
            await approve(app, pilot, "ingest")
            await approve(app, pilot, "summarize")
            session = app._workflow_session
            await wait_for(pilot, lambda: session.view().state == "review")
            review = app.screen.query_one("#workflow-review-text", TextArea)
            review.load_text(EDIT)
            await pilot.pause()
            await press(app, pilot, "#workflow-review-accept")
            await approve(app, pilot, "save")
            await wait_for(pilot, lambda: session.view().state == "completed")
            saved_note = await app.notes_scope_service.get_note_detail(
                scope="local_note",
                user_id=app.notes_user_id,
                note_id=session.view().note_id,
            )
            assert saved_note["content"] == "Human-edited summary"
            assert session.view().note_id == saved_note["id"]
            request_count = len(requests)
            with app.chachanotes_db.transaction() as cursor:
                inserted_note_count = cursor.execute(
                    "SELECT count(*) FROM notes"
                ).fetchone()[0]
            assert request_count == 1
            assert inserted_note_count == 1
            assert requests[0][0] == "POST /v1/chat/completions HTTP/1.1"
            assert requests[0][2]["model"] == "owned-peer-model"
            assert requests[0][2]["messages"] == [
                {"role": "user", "content": "Summarize in three bullets: " + SOURCE}
            ]
            assert requests[0][2]["stream"] is False
            assert requests[0][2]["max_tokens"] == 512
            await press(app, pilot, "#workflow-open-note")
            await wait_for(pilot, lambda: bool(app.screen.query("#library-note-body")))
            await wait_for(
                pilot,
                lambda: (
                    app.screen.query_one("#library-note-body", TextArea).text == EDIT
                ),
            )
            pane = app.screen.query_one("#library-note-work-pane")
            if hold_nested_mount:
                state = pane.presentation_state
                from textual.widget import Widget
                from textual.widgets import Button

                entered, release = asyncio.Event(), asyncio.Event()
                original_mount = Widget.mount_composed_widgets

                async def held_mount(widget, children):
                    if (
                        widget.id == "library-note-mode-controls"
                        and pane in widget.ancestors
                    ):
                        entered.set()
                        await release.wait()
                    await original_mount(widget, children)

                monkeypatch.setattr(Widget, "mount_composed_widgets", held_mount)
                follow_up = []

                def observe_then_focus():
                    context = pane.query_one("#library-note-context", Button)
                    follow_up.append(context.has_class("is-active"))
                    context.focus()

                pane.queue_after_recompose(observe_then_focus)
                recomposing = asyncio.create_task(pane.recompose())
                try:
                    await asyncio.wait_for(entered.wait(), 10)
                    assert pane.query(f"#{pane.authority_id}")
                    assert pane.query("#library-note-title")
                    assert not pane.query("#library-note-edit")
                    pane.apply_session_state(replace(state, presentation="preview"))
                    latest = replace(state, region="context")
                    pane.apply_session_state(latest)
                    assert pane.presentation_state is latest
                finally:
                    release.set()
                    await asyncio.wait_for(recomposing, 10)
                # No second apply: completed mount must paint the newest intent
                # BEFORE the existing focus callback runs.
                assert follow_up == [True]
                assert pane.query_one("#library-note-context-region").display
                assert not pane.query_one("#library-note-preview-region").display
                assert pane.query_one("#library-note-body", TextArea).text == EDIT
                assert app.focused is pane.query_one("#library-note-context")
                # A no-op rebuild must not revoke readiness of an intact tree.
                with monkeypatch.context() as pruning:
                    pruning.setattr(pane, "_pruning", True)
                    await pane.recompose()
                pane.apply_session_state(state)
                assert pane.query_one("#library-note-edit", Button).has_class(
                    "is-active"
                )


def capture(app, folder, name, captures):
    """Export real-app compositor evidence, never claim native terminal fonts."""
    path = folder / (name + ".svg")
    path.write_text(app.export_screenshot(simplify=True), encoding="utf-8")
    captures.append(
        {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    )


def note_rows(app):
    with app.chachanotes_db.transaction() as cursor:
        return [
            dict(row)
            for row in cursor.execute(
                "SELECT id, title, content FROM notes ORDER BY id"
            )
        ]


@pytest.mark.loopback_network
@pytest.mark.skipif(
    not os.environ.get("TASK6_LIVE_PROFILE"), reason="explicit isolated live UAT only"
)
@private_profile_test
async def test_live_full_app(request, isolated_profile, monkeypatch):
    """Real app, real localhost:9099, actual Notes; separately invoked restart."""
    from textual.widgets import TextArea

    from tldw_chatbook import app as app_module
    from tldw_chatbook import config

    root, paths = isolated_profile
    config_before_text = (root / "config/config.toml").read_text()
    phase = os.environ["TASK6_LIVE_PHASE"]
    width, height = map(int, os.environ["TASK6_LIVE_SIZE"].split("x"))
    folder = Path(os.environ["TASK6_ARTIFACTS"])
    prefix = f"live-{width}x{height}"
    manifest_path = folder / f"{prefix}-{phase}.json"
    requests, captures, responses = [], [], []
    log_offsets = {path: path.stat().st_size for path in root.glob("data/**/*.log")}

    def current_logs():
        return "\n".join(
            path.read_bytes()[log_offsets.get(path, 0) :].decode(errors="replace")
            for path in root.glob("data/**/*.log")
        )

    def verify_before_quit():
        assert not re.search(r"unhandled_exception|app_stopping", current_logs())
        report["prequit_log_scan_clean"] = True

    original_send = httpx.AsyncClient.send

    async def observed_send(client, request, **kwargs):
        # Pass through the real HTTP transport. Count attempts before dispatch.
        assert request.url.host == "127.0.0.1" and request.url.port == 9099, str(
            request.url
        )
        requests.append({"method": request.method, "url": str(request.url)})
        assert request.method == "POST"
        payload = json.loads(request.content)
        assert payload["model"] == MODEL
        assert payload["stream"] is False and payload["max_tokens"] == 512
        assert "authorization" not in request.headers
        return await original_send(client, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", observed_send)
    assert Path(app_module.__file__).resolve().is_relative_to(ROOT)
    app = app_module.TldwCli()
    report = {
        "phase": phase,
        "pid": os.getpid(),
        "size": [width, height],
        "paths": paths,
        "app_file": app_module.__file__,
        "provider": "llama_cpp",
        "model": MODEL,
        "endpoint": "http://localhost:9099",
        "requests": requests,
        "captures": captures,
        "versions": {
            name: importlib.metadata.version(name) for name in ("textual", "httpx")
        },
        "python": sys.version,
        "evidence_kind": "real TldwCli.run_test driver/compositor; not native PTY",
    }
    try:
        async with app.run_test(size=(width, height)) as pilot:
            await boot(app, pilot, root, seed=phase != "restart")
            paths["permission_store"] = str(
                app.unified_mcp_service.permission_store.path
            )
            report["notes_before"] = note_rows(app)
            if phase == "restart":
                previous = json.loads((folder / f"{prefix}-walk.json").read_text())
                assert previous["pid"] != os.getpid()
                for key in ("config", "data", "cache", "permission_store"):
                    assert paths[key] == previous["paths"][key]
                for key in paths:
                    if key.endswith("_db_path"):
                        assert paths[key] == previous["paths"][key]
                assert note_rows(app) == previous["notes_after"]
                assert (
                    app.workflow_documents.get_head(previous["workflow_id"]).revision_id
                    == previous["revision_id"]
                )
                assert app.ensure_workflow_session().view() is None
                await pilot.pause(1)
                assert "Unsaved review canary" not in "\n".join(
                    strip.text for strip in app.screen._compositor.render_strips()
                )
                assert requests == []
                capture(app, folder, prefix + "-restart", captures)
                verify_before_quit()
                await pilot.press("ctrl+q")
            else:
                source = root / "source.txt"
                source.write_text(SOURCE, encoding="utf-8")
                source.chmod(0o600)
                await setup(
                    app,
                    pilot,
                    source,
                    MODEL,
                    lambda state: capture(app, folder, prefix + "-" + state, captures),
                )
                session = app._workflow_session
                report.update(
                    workflow_id=session.view().workflow_id,
                    revision_id=session.view().revision_id,
                )
                capture(app, folder, prefix + "-file-ask", captures)
                await approve(app, pilot, "ingest")
                capture(app, folder, prefix + "-model-ask", captures)
                await approve(app, pilot, "summarize")
                await wait_for(
                    pilot, lambda: session.view().state in {"review", "failed"}, 150
                )
                assert session.view().state == "review", session.view()
                responses.append(session.view().review_text)
                review = app.screen.query_one("#workflow-review-text", TextArea)
                review.load_text(EDIT)
                review.focus()
                review.scroll_visible(animate=False)
                await pilot.pause()
                capture(app, folder, prefix + "-edited-review", captures)
                # Navigation preserves the exact review; quit Stay on another screen.
                app.action_shell_destination("home")
                await wait_for(pilot, lambda: not app.screen.query("#workflow-run"))
                await pilot.press("ctrl+q")
                await wait_for(pilot, lambda: bool(app.screen.query("#cancel-button")))
                capture(app, folder, prefix + "-offscreen-stay", captures)
                await press(app, pilot, "#cancel-button")
                assert session.view().review_text == EDIT
                app.action_shell_destination("workflows")
                await wait_for(
                    pilot, lambda: bool(app.screen.query("#workflow-review-text"))
                )
                await press(app, pilot, "#workflow-review-accept")
                await wait_for(pilot, lambda: session.view().state == "approval")
                capture(app, folder, prefix + "-note-ask", captures)
                await approve(app, pilot, "save")
                await wait_for(pilot, lambda: session.view().state == "completed")
                saved = await app.notes_scope_service.get_note_detail(
                    scope="local_note",
                    user_id=app.notes_user_id,
                    note_id=session.view().note_id,
                )
                assert saved["content"] == EDIT
                report["approved_note_id"] = saved["id"]
                assert len(note_rows(app)) == len(report["notes_before"]) + 1
                capture(app, folder, prefix + "-completed", captures)
                await press(app, pilot, "#workflow-open-note")
                await wait_for(
                    pilot, lambda: bool(app.screen.query("#library-note-body"))
                )
                await wait_for(
                    pilot,
                    lambda: (
                        app.screen.query_one("#library-note-body", TextArea).text
                        == EDIT
                    ),
                )
                assert app.screen._notes_state.selected_note_id == saved["id"]
                capture(app, folder, prefix + "-open-note", captures)
                app.action_shell_destination("workflows")
                await wait_for(pilot, lambda: bool(app.screen.query("#workflow-run")))
                # Cancel before file authority: neither HTTP nor Note.
                await setup(app, pilot, source, MODEL)
                await press(app, pilot, "#workflow-cancel")
                await wait_for(pilot, lambda: session.view().state == "cancelled")
                capture(app, folder, prefix + "-cancelled", captures)
                assert len(requests) == 1
                # Reject an actual generated review, then quit a third pending one.
                for ending in ("reject", "quit"):
                    await setup(app, pilot, source, MODEL)
                    await approve(app, pilot, "ingest")
                    await approve(app, pilot, "summarize")
                    await wait_for(
                        pilot, lambda: session.view().state in {"review", "failed"}, 150
                    )
                    assert session.view().state == "review", session.view()
                    responses.append(session.view().review_text)
                    if ending == "reject":
                        await press(app, pilot, "#workflow-review-reject")
                        await wait_for(
                            pilot, lambda: session.view().state == "rejected"
                        )
                        capture(app, folder, prefix + "-rejected", captures)
                    else:
                        app.screen.query_one(
                            "#workflow-review-text", TextArea
                        ).load_text("Unsaved review canary")
                        await pilot.pause()
                        await pilot.press("ctrl+q")
                        await wait_for(
                            pilot, lambda: bool(app.screen.query("#confirm-button"))
                        )
                        capture(app, folder, prefix + "-quit-loss", captures)
                        report["notes_after"] = note_rows(app)
                        verify_before_quit()
                        await press(app, pilot, "#confirm-button")
                assert len(requests) == 3
                assert len(report["notes_after"]) == len(report["notes_before"]) + 1
            await wait_for(pilot, lambda: app._shutting_down)
            report["controls_result"] = "passed"
    finally:
        config_after_text = (root / "config/config.toml").read_text()
        before_config = tomllib.loads(config_before_text)
        after_config = tomllib.loads(config_after_text)
        snapshots = root / "evidence"
        snapshots.mkdir(exist_ok=True, mode=0o700)
        for label, content in (
            ("before", config_before_text),
            ("after", config_after_text),
        ):
            (snapshots / f"{prefix}-{phase}-{label}.toml").write_text(content)
        report["config_evidence"] = {
            "before_sha256": hashlib.sha256(config_before_text.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(config_after_text.encode()).hexdigest(),
            "both_parsed": True,
            "changed_sections": [
                key
                for key in before_config.keys() | after_config.keys()
                if before_config.get(key) != after_config.get(key)
            ],
            "relevant_after": {
                key: after_config.get(key)
                for key in ("paths", "database", "model_catalog", "tldw_api")
            },
            "llama_cpp_after": after_config["api_settings"]["llama_cpp"],
            "authority_unchanged": all(
                before_config[key] == after_config[key]
                for key in ("paths", "database", "tldw_api")
            ),
            "model_authority_unchanged": all(
                values["api_settings"]["llama_cpp"]["api_url"]
                == "http://localhost:9099"
                and values["api_settings"]["llama_cpp"]["credential_source"] == "none"
                and values["model_catalog"]["auto_refresh_enabled"] is False
                for values in (before_config, after_config)
            ),
        }
        # Re-evaluate the persisted authority, not merely the initial snapshot.
        config.refresh_runtime_config_from_cli_config()
        effective_after = {
            name: str(getattr(config, name)())
            for name in paths
            if name.startswith("get_") and name.endswith("_db_path")
        }
        effective_after.update(
            config=str(config.get_cli_config_path()),
            data=str(config.get_user_data_dir()),
            cache=str(config.get_model_cache_dir()),
            permission_store=str(app.unified_mcp_service.permission_store.path),
        )
        report["config_evidence"]["effective_paths_after"] = effective_after
        report["config_evidence"]["effective_paths_unchanged"] = all(
            value == paths[name] and Path(value).resolve().is_relative_to(root)
            for name, value in effective_after.items()
        )
        logs = list(root.glob("data/**/*.log"))
        log_text = current_logs()
        report["log_scan"] = {
            "files": [str(path) for path in logs],
            "unhandled_exception_count": log_text.count("unhandled_exception"),
            "app_stopping_count": log_text.count("event=app_stopping"),
            "matching_lines": [
                line
                for line in log_text.splitlines()
                if re.search(r"unhandled_exception|app_stopping", line)
            ],
            "payload_found": any(
                value and value in log_text
                for value in [
                    SOURCE,
                    "Summarize in three bullets: " + SOURCE,
                    EDIT,
                    "Unsaved review canary",
                    *responses,
                ]
            ),
        }
        for index, path in enumerate(logs):
            shutil.copyfile(path, folder / f"{prefix}-{phase}-app-{index}.log")
        manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert not report["log_scan"]["unhandled_exception_count"]
    assert report["log_scan"]["app_stopping_count"] == 1
    assert report["prequit_log_scan_clean"]
    assert not report["log_scan"]["payload_found"]
    assert report["config_evidence"]["authority_unchanged"]
    assert report["config_evidence"]["model_authority_unchanged"]
    assert report["config_evidence"]["effective_paths_unchanged"]
    report["result"] = "passed"
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # Standalone preparation is stdlib/third-party only, never app/core imports.
    if len(sys.argv) == 2:
        print(write_profile(Path(sys.argv[1])))
    else:
        # Execute pytest in a fresh interpreter after scrubbing ambient secrets
        # and cache selectors. Never inspect any host profile or credential value.
        profile, phase, size, artifacts = map(str, sys.argv[1:])
        profile = str(Path(profile).resolve())
        artifacts = str(Path(artifacts).resolve())
        Path(profile, "probes").mkdir(mode=0o700, exist_ok=True)
        environment = os.environ.copy()
        removed = []
        for name in tuple(environment):
            if re.search(
                r"KEY|TOKEN|PASSWORD|SECRET|CREDENTIAL|PROXY|CACHE|BASE_URL|API_URL|^HF_|^HUGGING|^TRANSFORMERS_|^TLDW_|^TASK6_",
                name,
            ):
                removed.append(name)
                del environment[name]
        environment.update(
            {
                "TLDW_TEST_CONFIG_ROOT": profile,
                "TLDW_TEST_PRIVATE_PROFILE_NODE": (
                    "Tests/Workflows/test_file_to_note_integration.py::test_live_full_app"
                ),
                "TASK6_LIVE_PROFILE": profile,
                "TASK6_LIVE_PHASE": phase,
                "TASK6_LIVE_SIZE": size,
                "TASK6_ARTIFACTS": artifacts,
                "PYTHONPATH": str(ROOT),
                "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
                "XDG_CACHE_HOME": profile + "/cache",
                "HF_HOME": profile + "/cache/huggingface",
                "TIKTOKEN_CACHE_DIR": profile + "/cache/tiktoken",
                "HF_HUB_OFFLINE": "1",
            }
        )
        Path(artifacts, f"live-{size}-{phase}-environment.json").write_text(
            json.dumps({"removed_names_only": sorted(removed)}, indent=2)
        )
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "--tb=short",
            "--show-capture=no",
            "--basetemp=" + profile + f"/probes/{size}-{phase}",
            str(Path(__file__).resolve()) + "::test_live_full_app",
        ]
        with Path(artifacts, f"live-{size}-{phase}.txt").open("w") as output:
            result = subprocess.run(
                command,
                env=environment,
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        raise SystemExit(result.returncode)
