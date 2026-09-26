"""Settings/Console SSH binding surfaces (Phase 5a, Task 20).

Spec: ``Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings-design.md``
"User-facing surfaces". Three surfaces are pinned here:

- The Settings → Workspaces "SSH folders" editor mirrors the folder editor:
  an add form (target, absolute remote path, ro/rw, interpreter override)
  submitting to ``add_ssh_binding``; per-binding rows
  ``locator [ro|rw] <live-status>`` rendered from the status cache; toggle
  via ``set_ssh_binding_access``; remove via ``remove_runtime_binding``.
- The add-time probe is ADVISORY: the save always lands, the probe runs in
  a ``@work(thread=True)`` worker (never inline, never blocking the UI),
  and its result updates the row when it lands.
- The Console project-instruction picker and the Alt+W switcher label SSH
  bindings with cache-status chips (listing/labeling only).
"""

from __future__ import annotations

import shlex
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_configuration_hub import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _open_settings_category,
    _visible_text,
)
import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from tldw_chatbook.Tools.remote_binding_status import (
    BindingState,
    RemoteBindingStatusCache,
    cached_status_display,
    set_remote_binding_status_cache,
)
from tldw_chatbook.Tools.remote_workspace_transport import TransportFailureKind
from tldw_chatbook.UI.Console_Modules.workspace import (
    _console_workspace_ssh_status_chips,
)
from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError


@pytest.fixture()
def fresh_status_cache():
    """Isolate the process-wide remote-binding status cache per test."""
    cache = RemoteBindingStatusCache()
    set_remote_binding_status_cache(cache)
    yield cache
    set_remote_binding_status_cache(None)


def _fake_ssh(tmp_path: Path) -> Path:
    """Write the executable fake ``ssh`` (Task 6 ``ssh -G`` pattern)."""
    lines = [
        "#!/bin/sh",
        "port=''",
        "user=''",
        "host=''",
        "prev=''",
        'for arg in "$@"; do',
        '  if [ "$prev" = "-p" ]; then port="$arg"; fi',
        '  if [ "$prev" = "-l" ]; then user="$arg"; fi',
        '  host="$arg"',
        '  prev="$arg"',
        "done",
        'case "$host" in',
        # dev/devbox deliberately share one resolved destination.
        "  dev|devbox)",
        '  [ -n "$port" ] || port=2222; [ -n "$user" ] || user=me;',
        "  printf 'hostname devbox.example.com\\n'; "
        'printf \'port %s\\nuser %s\\n\' "$port" "$user"; exit 0 ;;',
        "esac",
        'echo "unknown host" >&2',
        "exit 255",
    ]
    script = tmp_path / "fake-ssh"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)
    return script


async def _open_workspaces_ssh_pane(tmp_path, workspace_id: str):
    """Build the app, open Settings → Workspaces, select ``workspace_id``."""
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id=workspace_id, name="SSH Surfaces")
    fake_ssh = _fake_ssh(tmp_path)
    host = DestinationHarness(app, "settings")
    return app, registry, fake_ssh, host


async def _wait_for_widget(
    pilot, screen, selector: str, timeout: float = 5.0, *, contains: str | None = None,
    expect_type: type | None = None,
):
    """Poll for ``selector`` instead of trusting a fixed pause.

    Every SSH-pane mutation (add/toggle/remove, and the advisory probe
    landing) re-renders the rows through the async category-swap worker, so
    a post-action ``query_one`` can run while the widget is being replaced
    and hit ``NoMatches`` on slow runs. Wait for the widget -- and, when
    asked, its content/type -- to (re)appear; on timeout fall through to
    ``query_one`` so ``NoMatches`` keeps its original context.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matches = screen.query(selector)
        if matches:
            widget = matches.first()
            if (expect_type is None or isinstance(widget, expect_type)) and (
                contains is None
                or contains in str(getattr(widget, "renderable", ""))
            ):
                return widget
        await pilot.pause(0.1)
    return (
        screen.query_one(selector, expect_type)
        if expect_type is not None
        else screen.query_one(selector)
    )


async def _add_ssh_via_pane(
    screen, pilot, registry, fake_ssh: Path, target: str, path: str
):
    """Fill the add form and press Add (registry spied to use the fake ssh).

    The form only exists once the workspace-selection pane swap finishes,
    so wait for each control instead of trusting the caller's fixed pause.
    """
    target_input = await _wait_for_widget(
        pilot, screen, "#settings-workspace-ssh-target", expect_type=Input
    )
    target_input.value = target
    path_input = await _wait_for_widget(
        pilot, screen, "#settings-workspace-ssh-path", expect_type=Input
    )
    path_input.value = path
    original = registry.add_ssh_binding

    def spied(workspace_id, raw_locator, **kwargs):
        return original(workspace_id, raw_locator, ssh_bin=str(fake_ssh), **kwargs)

    registry.add_ssh_binding = spied
    try:
        add_button = await _wait_for_widget(
            pilot, screen, "#settings-workspace-ssh-add", expect_type=Button
        )
        add_button.press()
        await pilot.pause()  # let the add handler run before returning
    finally:
        registry.add_ssh_binding = original


# --- status display words (shared helper) -----------------------------------


def test_cached_status_display_words(fresh_status_cache) -> None:
    """The four cache states map to the spec's live-status vocabulary."""
    cache = fresh_status_cache
    # Cold/unseen stays optimistic READY.
    assert cached_status_display("binding-cold") == "ready"

    cache.record_transport_failure(
        "b-unreachable", TransportFailureKind.UNREACHABLE, "unreachable or auth failed"
    )
    assert (
        cached_status_display("b-unreachable")
        == "unreachable (unreachable or auth failed)"
    )

    cache.record_transport_failure(
        "b-nopython", TransportFailureKind.INTERPRETER_MISSING, "host lacks python3"
    )
    assert cached_status_display("b-nopython") == "unreachable (host lacks python3)"

    cache.record_transport_failure(
        "b-oldpy",
        TransportFailureKind.PYTHON_TOO_OLD,
        "python ≥ 3.10 required (found 3.8.5)",
    )
    assert (
        cached_status_display("b-oldpy")
        == "unreachable (python ≥ 3.10 required (found 3.8.5))"
    )

    cache.record_missing("b-missing", "root absent")
    assert cached_status_display("b-missing") == "missing on host"

    cache.record_pin_failure("b-stale")
    assert cached_status_display("b-stale") == "identity stale"

    cache.record_success("b-recovered")
    assert cached_status_display("b-recovered") == "ready"


# --- Settings → Workspaces pane ---------------------------------------------


def test_advisory_probe_preserves_ping_recorded_reason(
    monkeypatch, fresh_status_cache
) -> None:
    """The real probe never overwrites a taxonomy row the ping recorded."""
    from tldw_chatbook.Tools.remote_workspace_executor import (
        RemoteWorkspaceExecutionError,
        RemoteWorkspaceToolExecutor,
    )
    from tldw_chatbook.Tools import remote_workspace_transport as transport_module

    cache = fresh_status_cache
    # The manager reads [console_ssh] config; keep this unit test off it.
    monkeypatch.setattr(transport_module, "get_master_manager", lambda: object())

    class _PingRecordedThenRaised:
        def ping(self):
            cache.record_transport_failure(
                "b-real", TransportFailureKind.INTERPRETER_MISSING, "host lacks python3"
            )
            raise RemoteWorkspaceExecutionError(
                "interpreter_missing", "host lacks python3", admitted=False
            )

    monkeypatch.setattr(
        RemoteWorkspaceToolExecutor,
        "for_ssh",
        classmethod(lambda cls, *a, **k: _PingRecordedThenRaised()),
    )
    settings_screen_module._settings_ssh_advisory_probe(
        "b-real", "ssh://devbox/srv/app", "python3"
    )
    assert cache.status("b-real").reason == "host lacks python3"


def test_advisory_probe_records_unreached_contract_failures(
    monkeypatch, fresh_status_cache
) -> None:
    """A ping failure the taxonomy never saw degrades to unreachable."""
    from tldw_chatbook.Tools.remote_workspace_executor import (
        RemoteWorkspaceExecutionError,
        RemoteWorkspaceToolExecutor,
    )
    from tldw_chatbook.Tools import remote_workspace_transport as transport_module

    cache = fresh_status_cache
    monkeypatch.setattr(transport_module, "get_master_manager", lambda: object())

    class _PingRaisedUnrecorded:
        def ping(self):
            raise RemoteWorkspaceExecutionError("protocol_failure", admitted=False)

    monkeypatch.setattr(
        RemoteWorkspaceToolExecutor,
        "for_ssh",
        classmethod(lambda cls, *a, **k: _PingRaisedUnrecorded()),
    )
    settings_screen_module._settings_ssh_advisory_probe(
        "b-contract", "ssh://devbox/srv/app", "python3"
    )
    status = cache.status("b-contract")
    assert str(status.state) == "BLOCKED"
    assert "probe failed" in str(status.reason)

    def _unbuildable(cls, *a, **k):
        raise ValueError("bad locator")

    monkeypatch.setattr(
        RemoteWorkspaceToolExecutor, "for_ssh", classmethod(_unbuildable)
    )
    settings_screen_module._settings_ssh_advisory_probe(
        "b-unbuilt", "ssh://devbox/srv/app", "python3"
    )
    unbuilt = cache.status("b-unbuilt")
    assert str(unbuilt.state) == "BLOCKED"
    assert "probe could not run" in str(unbuilt.reason)


@pytest.mark.timeout(240)
@private_profile_test
@pytest.mark.asyncio
async def test_ssh_form_renders_and_submits_add_ssh_binding(
    request, tmp_path, monkeypatch, fresh_status_cache
) -> None:
    """The add form submits a composed locator with access + interpreter."""
    app, registry, fake_ssh, host = await _open_workspaces_ssh_pane(
        tmp_path, "ws-ssh-add"
    )

    probe_metadata: list = []

    def noop_probe(binding_id, locator, python, metadata=None):
        probe_metadata.append(metadata)
        return None

    monkeypatch.setattr(
        settings_screen_module, "_settings_ssh_advisory_probe", noop_probe
    )
    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        (await _wait_for_widget(
            pilot, screen, "#settings-workspace-row-ws-ssh-add", expect_type=Button
        )).press()

        # The form renders: target, absolute remote path, ro/rw choice,
        # interpreter override, submit.
        assert await _wait_for_widget(
            pilot, screen, "#settings-workspace-ssh-target", expect_type=Input
        )
        assert await _wait_for_widget(
            pilot, screen, "#settings-workspace-ssh-path", expect_type=Input
        )
        assert await _wait_for_widget(
            pilot, screen, "#settings-workspace-ssh-python", expect_type=Input
        )
        access_button = await _wait_for_widget(
            pilot, screen, "#settings-workspace-ssh-access", expect_type=Button
        )
        assert str(access_button.label) == "Access: read-only"

        calls: list[tuple[str, dict]] = []
        original = registry.add_ssh_binding

        def spied(workspace_id, raw_locator, **kwargs):
            calls.append((raw_locator, kwargs))
            return original(workspace_id, raw_locator, ssh_bin=str(fake_ssh), **kwargs)

        registry.add_ssh_binding = spied
        try:
            screen.query_one("#settings-workspace-ssh-target", Input).value = (
                "me@devbox:2222"
            )
            screen.query_one("#settings-workspace-ssh-path", Input).value = "/srv/app"
            screen.query_one("#settings-workspace-ssh-python", Input).value = (
                "python3.11"
            )
            # Flip the access choice to read-write before submitting.
            screen.query_one("#settings-workspace-ssh-access", Button).press()
            await pilot.pause(0.1)
            screen.query_one("#settings-workspace-ssh-add", Button).press()
            await pilot.pause(0.4)
        finally:
            registry.add_ssh_binding = original

        assert calls == [
            (
                "ssh://me@devbox:2222/srv/app",
                {"allow_write": True, "python_interpreter": "python3.11"},
            )
        ]
        (binding,) = registry.list_ssh_bindings("ws-ssh-add")
        assert binding.metadata["access"] == "rw"
        assert binding.metadata["python"] == "python3.11"
        # The advisory probe ran (noop stub) and its receipt replaced the
        # transient "SSH folder added…" line next to the form.
        for _ in range(50):
            await pilot.pause(0.05)
            if "Probe: ready." in _visible_text(screen):
                break
        assert "Probe: ready." in _visible_text(screen)
        # The probe gets the recorded ssh -G destination, so it verifies
        # the host before capturing identity.
        assert probe_metadata == [binding.metadata]
        # The row renders locator + access + live status (noop probe left
        # the cold-cache optimistic ready).
        row = await _wait_for_widget(
            pilot, screen, f"#settings-workspace-ssh-{binding.binding_id}"
        )
        assert "me@devbox:2222/srv/app" in str(row.renderable)
        assert "[rw] ready" in str(row.renderable)


@pytest.mark.timeout(240)
@private_profile_test
@pytest.mark.asyncio
async def test_advisory_probe_failure_still_saves_with_reason(
    request, tmp_path, monkeypatch, fresh_status_cache
) -> None:
    """A failed add-time probe never blocks the save; the reason shows."""
    app, registry, fake_ssh, host = await _open_workspaces_ssh_pane(
        tmp_path, "ws-ssh-probe"
    )
    cache = fresh_status_cache

    def failing_probe(binding_id, locator, python, metadata=None):
        cache.record_transport_failure(
            binding_id, TransportFailureKind.INTERPRETER_MISSING, "host lacks python3"
        )

    monkeypatch.setattr(
        settings_screen_module, "_settings_ssh_advisory_probe", failing_probe
    )
    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        (await _wait_for_widget(
            pilot, screen, "#settings-workspace-row-ws-ssh-probe", expect_type=Button
        )).press()

        await _add_ssh_via_pane(screen, pilot, registry, fake_ssh, "devbox", "/srv/app")

        # Save ALWAYS allowed: the binding persisted despite the probe
        # failure, and the row's live status went degraded from the cache.
        (binding,) = registry.list_ssh_bindings("ws-ssh-probe")
        row = await _wait_for_widget(
            pilot,
            screen,
            f"#settings-workspace-ssh-{binding.binding_id}",
            contains="[ro] unreachable (host lacks python3)",
        )
        assert "[ro] unreachable (host lacks python3)" in str(row.renderable)
        # The advisory outcome is surfaced next to the add form.
        assert "host lacks python3" in _visible_text(screen)


@pytest.mark.timeout(240)
@private_profile_test
@pytest.mark.asyncio
async def test_probe_dispatched_to_worker_never_blocks_ui(
    request, tmp_path, monkeypatch, fresh_status_cache
) -> None:
    """The probe body runs off the UI thread; interactions continue meanwhile."""
    app, registry, fake_ssh, host = await _open_workspaces_ssh_pane(
        tmp_path, "ws-ssh-block"
    )
    cache = fresh_status_cache
    started = threading.Event()
    release = threading.Event()

    def blocking_probe(binding_id, locator, python, metadata=None):
        started.set()
        release.wait(timeout=10)
        cache.record_transport_failure(
            binding_id, TransportFailureKind.UNREACHABLE, "unreachable or auth failed"
        )

    monkeypatch.setattr(
        settings_screen_module, "_settings_ssh_advisory_probe", blocking_probe
    )
    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        (await _wait_for_widget(
            pilot, screen, "#settings-workspace-row-ws-ssh-block", expect_type=Button
        )).press()

        await _add_ssh_via_pane(screen, pilot, registry, fake_ssh, "devbox", "/srv/app")

        # The handler returned while the probe body was still parked: the
        # save landed (its receipt visible), and the pane stayed
        # interactive (toggle still works with the probe in flight) --
        # the UI thread never ran the probe.
        for _ in range(40):
            if started.is_set():
                break
            await pilot.pause(0.05)
        assert started.is_set(), "probe worker should have been scheduled"
        assert not release.is_set()
        assert len(registry.list_ssh_bindings("ws-ssh-block")) == 1
        assert "SSH folder added (read-only). Probing host…" in _visible_text(screen)
        (binding,) = registry.list_ssh_bindings("ws-ssh-block")
        toggle = f"#settings-workspace-ssh-toggle-{binding.binding_id}"
        (await _wait_for_widget(pilot, screen, toggle, expect_type=Button)).press()
        await pilot.pause(0.3)
        assert registry.list_ssh_bindings("ws-ssh-block")[0].metadata["access"] == "rw"

        release.set()
        row = await _wait_for_widget(
            pilot,
            screen,
            f"#settings-workspace-ssh-{binding.binding_id}",
            contains="unreachable",
        )
        assert (
            "[rw] unreachable (unreachable or auth failed)"
            in str(row.renderable)
        )


@pytest.mark.timeout(240)
@private_profile_test
@pytest.mark.asyncio
async def test_ssh_toggle_and_remove_call_through(
    request, tmp_path, monkeypatch, fresh_status_cache
) -> None:
    """ro/rw toggle and remove call the registry; failures surface inline."""
    app, registry, fake_ssh, host = await _open_workspaces_ssh_pane(
        tmp_path, "ws-ssh-edit"
    )

    def noop_probe(binding_id, locator, python, metadata=None):
        return None

    monkeypatch.setattr(
        settings_screen_module, "_settings_ssh_advisory_probe", noop_probe
    )
    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        (await _wait_for_widget(
            pilot, screen, "#settings-workspace-row-ws-ssh-edit", expect_type=Button
        )).press()

        await _add_ssh_via_pane(screen, pilot, registry, fake_ssh, "devbox", "/srv/app")
        (binding,) = registry.list_ssh_bindings("ws-ssh-edit")
        toggle = f"#settings-workspace-ssh-toggle-{binding.binding_id}"
        remove = f"#settings-workspace-ssh-remove-{binding.binding_id}"

        # Refused toggle explains inline and changes nothing.
        with monkeypatch.context() as patch:

            def refuse(*args, **kwargs):
                raise WorkspaceRegistryServiceError("Access change refused. Retry.")

            patch.setattr(registry, "set_ssh_binding_access", refuse)
            (await _wait_for_widget(pilot, screen, toggle, expect_type=Button)).press()
            await pilot.pause(0.3)
            assert "Access change refused" in _visible_text(screen)
        assert registry.list_ssh_bindings("ws-ssh-edit")[0].metadata["access"] == "ro"

        (await _wait_for_widget(pilot, screen, toggle, expect_type=Button)).press()
        await pilot.pause(0.3)
        assert "SSH folder access: read-write." in _visible_text(screen)
        assert registry.list_ssh_bindings("ws-ssh-edit")[0].metadata["access"] == "rw"

        # Refused remove explains inline and keeps the binding.
        with monkeypatch.context() as patch:

            def refuse(*args, **kwargs):
                raise WorkspaceRegistryServiceError("Removal refused. Retry.")

            patch.setattr(registry, "remove_runtime_binding", refuse)
            (await _wait_for_widget(pilot, screen, remove, expect_type=Button)).press()
            await pilot.pause(0.3)
            assert "Removal refused" in _visible_text(screen)
        assert len(registry.list_ssh_bindings("ws-ssh-edit")) == 1

        (await _wait_for_widget(pilot, screen, remove, expect_type=Button)).press()
        await pilot.pause(0.3)
        assert registry.list_ssh_bindings("ws-ssh-edit") == ()
        assert "SSH folder removed." in _visible_text(screen)


@pytest.mark.timeout(240)
@private_profile_test
@pytest.mark.asyncio
async def test_ssh_add_validation_keeps_drafts(
    request, fresh_status_cache, tmp_path
) -> None:
    """A relative path or missing target explains inline and adds nothing."""
    app, registry, fake_ssh, host = await _open_workspaces_ssh_pane(
        tmp_path, "ws-ssh-valid"
    )
    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        (await _wait_for_widget(
            pilot, screen, "#settings-workspace-row-ws-ssh-valid", expect_type=Button
        )).press()

        target_input = await _wait_for_widget(
            pilot, screen, "#settings-workspace-ssh-target", expect_type=Input
        )
        target_input.value = "devbox"
        path_input = await _wait_for_widget(
            pilot, screen, "#settings-workspace-ssh-path", expect_type=Input
        )
        path_input.value = "srv/app"
        (await _wait_for_widget(
            pilot, screen, "#settings-workspace-ssh-add", expect_type=Button
        )).press()
        await pilot.pause(0.3)
        assert "absolute" in _visible_text(screen)
        assert registry.list_ssh_bindings("ws-ssh-valid") == ()
        # Drafts survive the rejection (the refused add recomposes nothing).
        assert (
            screen.query_one("#settings-workspace-ssh-path", Input).value == "srv/app"
        )


# --- Console picker + switcher chips ----------------------------------------


@pytest.mark.asyncio
async def test_picker_lists_ssh_bindings_with_chips(fresh_status_cache) -> None:
    """SSH options carry a cache-status chip; folder options do not."""
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController

    fresh_status_cache.record_transport_failure(
        "ssh-b1", TransportFailureKind.UNREACHABLE, "unreachable or auth failed"
    )
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._current_chat_store_accessor = lambda: SimpleNamespace(
        sessions=lambda: [SimpleNamespace(id="session-a")]
    )
    captured: list[object] = []

    async def capture(modal):
        captured.append(modal)
        return SimpleNamespace(action="cancel", binding_id=None)

    controller._screen = SimpleNamespace(
        app=SimpleNamespace(push_screen_wait=AsyncMock())
    )
    controller._screen.app.push_screen_wait.side_effect = capture

    selections = (
        SimpleNamespace(
            binding=SimpleNamespace(
                binding_id="folder-b1",
                display_name="",
                label="project",
                binding_kind="local-filesystem",
            )
        ),
        SimpleNamespace(
            binding=SimpleNamespace(
                binding_id="ssh-b1",
                display_name="",
                label="/srv/app",
                binding_kind="ssh-filesystem",
            )
        ),
    )
    await controller._select_project_instruction_binding(
        "session-a", selections, ""
    )

    (modal,) = captured
    folder_option, ssh_option = modal._options
    assert folder_option.binding_id == "folder-b1"
    assert "ssh:" not in folder_option.label
    assert ssh_option.binding_id == "ssh-b1"
    assert ssh_option.label.endswith(
        "ssh: unreachable (unreachable or auth failed)"
    )


def test_switcher_ssh_status_chips(fresh_status_cache) -> None:
    """Worst per-binding state becomes one chip per workspace row."""
    cache = fresh_status_cache
    cache.record_transport_failure(
        "ssh-a1", TransportFailureKind.UNREACHABLE, "unreachable or auth failed"
    )
    cache.record_pin_failure("ssh-a2")
    ssh_binding_a1 = SimpleNamespace(binding_id="ssh-a1")
    ssh_binding_a2 = SimpleNamespace(binding_id="ssh-a2")
    ssh_binding_b1 = SimpleNamespace(binding_id="ssh-b1")
    folder_binding = SimpleNamespace(binding_id="folder-b1")

    class _Registry:
        def list_ssh_bindings(self, workspace_id):
            return {
                "ws-a": (ssh_binding_a1, ssh_binding_a2),
                "ws-b": (ssh_binding_b1,),
                "ws-healthy": (folder_binding,),
            }.get(workspace_id, ())

    workspaces = (
        SimpleNamespace(workspace_id="ws-a"),
        SimpleNamespace(workspace_id="ws-b"),
        SimpleNamespace(workspace_id="ws-healthy"),
    )
    chips = _console_workspace_ssh_status_chips(_Registry(), workspaces)
    # BLOCKED outranks STALE_IDENTITY; cold/ready and folder-only rows stay
    # chip-free.
    assert chips == {"ws-a": "ssh: unreachable (unreachable or auth failed)"}


@pytest.mark.asyncio
async def test_switcher_modal_renders_ssh_chip(fresh_status_cache) -> None:
    """The Alt+W modal appends the chip to the workspace display name."""
    from textual.app import App

    from tldw_chatbook.Widgets.Console.console_workspace_switcher_modal import (
        ConsoleWorkspaceSwitcherModal,
    )

    modal = ConsoleWorkspaceSwitcherModal(
        workspaces=(SimpleNamespace(workspace_id="ws-a", name="Alpha", archived=False),),
        active_workspace_id=None,
        show_archived=False,
        ssh_status_chips={"ws-a": "ssh: unreachable (host lacks python3)"},
    )

    class _Host(App):
        def on_mount(self) -> None:
            self.push_screen(modal)

    async with _Host().run_test() as pilot:
        await pilot.pause()
        labels = [
            str(widget.label)
            for widget in modal.query(Button)
            if "console-workspace-switcher-option" in (widget.classes or set())
        ]
        assert any(
            "Alpha" in label and "ssh: unreachable (host lacks python3)" in label
            for label in labels
        )
