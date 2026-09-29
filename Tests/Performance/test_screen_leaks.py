"""Departed screens must be collectable (TASK-33264, PERF-05).

The 2026-09-27 audit measured every Settings visit retaining its whole
SettingsScreen (10/10, ~+10.7 MB per visit) through an app-level
``theme_changed_signal`` subscription that was never dropped: Textual's
``Signal`` keys subscribers in a ``WeakKeyDictionary`` whose values (the
subscribing callbacks) strongly reference the key, so the entry only goes
away when that signal next publishes. Departed Personas screens (6/10,
~+14 MB each) were pinned by Textual's thread-worker ``active_worker``
ContextVar, which stays set in each pool thread's context after the work
returns and holds the Worker -- and through it the screen that started it.

This boots the real app on a private profile, visits each non-reusable
destination ten times with Home in between, and asserts that no departed
instance is still reachable after a full collection.
"""

from __future__ import annotations

import ast
import asyncio
import gc
import os
import weakref
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test

#: The AC's visit count. Retention above zero at any N is the bug; ten makes
#: an intermittent pin (a pool-thread ContextVar, which depends on which
#: thread ran the last worker) show up reliably.
VISITS = 10

#: Non-reusable routes (a fresh instance per visit), hotkey -> screen class.
LEAK_ROUTES: tuple[tuple[str, str], ...] = (
    ("f4", "SettingsScreen"),
    ("ctrl+4", "PersonasScreen"),
)


def _scratch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate the private profile selected before this interpreter's imports."""
    for name in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    config_file = Path(os.environ["TLDW_CONFIG_PATH"])
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "[first_run]\nsetup_completed = true\n\n[splash_screen]\nenabled = false\n"
    )
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "screen_leaks")
    from tldw_chatbook.config import load_settings

    load_settings(force_reload=True)


async def _visit(pilot, key: str, expected: str) -> None:
    deadline = asyncio.get_running_loop().time() + 30.0
    await pilot.press(key)
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause()
        if type(pilot.app.screen).__name__ == expected:
            break
    assert type(pilot.app.screen).__name__ == expected, (
        f"{key} never reached {expected}; stuck on {type(pilot.app.screen).__name__}"
    )
    # Let the visit's mount workers (thread workers included) run and finish.
    for _ in range(8):
        await asyncio.sleep(0.05)
        await pilot.pause()


def _holder_chain(target: object, app: object, ignore: set[int]) -> str:
    """Name the shortest referrer path from ``target`` to a live root.

    A root is the running app, a ``contextvars.Context`` or a module dict --
    whatever first explains why ``target`` survived the collection.
    """
    import contextvars
    import types

    parents: dict[int, tuple[object, int | None]] = {id(target): (target, None)}
    frontier = [target]
    ignore = ignore | {id(parents), id(frontier)}
    while frontier and len(parents) < 200_000:
        next_frontier = []
        for node in frontier:
            for ref in gc.get_referrers(node):
                if id(ref) in parents or id(ref) in ignore or isinstance(ref, types.FrameType):
                    continue
                parents[id(ref)] = (ref, id(node))
                is_root = (
                    ref is app
                    or isinstance(ref, contextvars.Context)
                    or (isinstance(ref, dict) and "__name__" in ref and "__loader__" in ref)
                )
                if is_root:
                    chain, key = [], id(ref)
                    while key is not None:
                        obj, key = parents[key]
                        chain.append(type(obj).__qualname__)
                    return " -> ".join(chain)
                next_frontier.append(ref)
        frontier = next_frontier
    return "no root found"


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_departed_screens_are_collectable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """No Settings, Personas or Settings-Theme-picker instance outlives its visit."""
    _scratch_env(monkeypatch)
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    departed: dict[str, list[weakref.ref]] = {name: [] for _, name in LEAK_ROUTES}
    departed["ThemePicker"] = []
    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(20):
            await asyncio.sleep(0.05)
            await pilot.pause()
        await _visit(pilot, "ctrl+1", "HomeScreen")
        for key, name in LEAK_ROUTES:
            for _ in range(VISITS):
                await _visit(pilot, key, name)
                if name == "SettingsScreen":
                    # The Theme category mounts ThemePicker, the other
                    # theme_changed_signal subscriber on this screen.
                    await pilot.press("escape", "/", *"Theme", "enter")
                    for _ in range(40):
                        await pilot.pause(0.05)
                        if app.screen.query("ThemePicker"):
                            break
                    # No local may hold a query result: the coroutine would
                    # keep the last screen alive.
                    assert app.screen.query("ThemePicker"), "Theme category never mounted"
                    departed["ThemePicker"].append(
                        weakref.ref(app.screen.query_one("ThemePicker"))
                    )
                departed[name].append(weakref.ref(app.screen))
                await _visit(pilot, "ctrl+1", "HomeScreen")

        for _ in range(3):
            gc.collect()
        alive = {
            name: [ref() for ref in refs if ref() is not None]
            for name, refs in departed.items()
        }
        ignore = {id(alive), *(id(screens) for screens in alive.values())}
        report = {
            name: f"{len(screens)}/{VISITS} retained"
            + (
                f"; holder chain: {_holder_chain(screens[0], app, ignore)}"
                if screens
                else ""
            )
            for name, screens in alive.items()
        }
        del alive
        assert all(r.startswith("0/") for r in report.values()), "\n".join(
            f"{name}: {line}" for name, line in report.items()
        )


# ---------------------------------------------------------------------------
# AC#3 guard: a node that subscribes to an App-level Signal must unsubscribe in
# its unmount handler. The mounted tour above only sees the screens it visits;
# this static pass sees every class, including screens that do not exist yet.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The Signals Textual's ``App`` owns (textual/app.py). They live as long as
#: the app, so a subscriber that never unsubscribes lives that long too.
APP_SIGNALS = frozenset(
    {
        "theme_changed_signal",
        "screen_change_signal",
        "app_suspend_signal",
        "app_resume_signal",
        "mode_change_signal",
    }
)
_UNMOUNT_HANDLERS = frozenset({"on_unmount", "_on_unmount"})


def _signal_call(node: ast.AST, method: str) -> str | None:
    """``<expr>.<app signal>.<method>(...)`` -> the signal name, else None.

    ``self.<signal>`` is the App touching its own signal and is ignored.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr in APP_SIGNALS
    ):
        return None
    owner = node.func.value.value
    if isinstance(owner, ast.Name) and owner.id == "self":
        return None
    return node.func.value.attr


def _unpaired_app_signal_subscriptions(source: str, filename: str) -> list[str]:
    """Classes that subscribe to an App signal but never drop it on unmount."""
    findings = []
    for cls in ast.walk(ast.parse(source, filename)):
        if not isinstance(cls, ast.ClassDef):
            continue
        subscribed = {
            (sig, node.lineno)
            for node in ast.walk(cls)
            if (sig := _signal_call(node, "subscribe"))
        }
        dropped = {
            sig
            for method in cls.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            and method.name in _UNMOUNT_HANDLERS
            for node in ast.walk(method)
            if (sig := _signal_call(node, "unsubscribe"))
        }
        findings += [
            f"{filename}:{line} {cls.name} subscribes {sig} without unsubscribing in on_unmount"
            for sig, line in sorted(subscribed)
            if sig not in dropped
        ]
    return findings


def test_app_signal_subscribers_unsubscribe_on_unmount() -> None:
    """Every App-signal subscription in the package is dropped on unmount.

    Textual's ``Signal`` holds subscribers in a ``WeakKeyDictionary`` whose
    values (the callbacks) reference the key, so an entry survives until that
    signal next publishes -- for ``theme_changed_signal`` that is the next
    theme change, and every Settings visit leaked its screen until then.
    """
    findings = []
    for path in sorted((REPO_ROOT / "tldw_chatbook").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "_signal.subscribe(" in source:
            findings += _unpaired_app_signal_subscriptions(
                source, str(path.relative_to(REPO_ROOT))
            )
    assert not findings, "\n".join(findings)


def test_signal_guard_flags_a_subscription_without_unsubscribe() -> None:
    """Negative control: the scanner is not vacuous."""
    leaky = (
        "class W:\n"
        "    def on_mount(self):\n"
        "        self.app.theme_changed_signal.subscribe(self, self.f)\n"
    )
    paired = leaky + (
        "    def on_unmount(self):\n"
        "        self.app.theme_changed_signal.unsubscribe(self)\n"
    )
    assert _unpaired_app_signal_subscriptions(leaky, "w.py")
    assert not _unpaired_app_signal_subscriptions(paired, "w.py")


def test_fresh_context_executor_jobs_do_not_share_context() -> None:
    """A context variable one job sets is gone for the next job on the same thread."""
    import contextvars

    from tldw_chatbook.Utils.text_selection_crash_guard import FreshContextExecutor

    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="unset")
    executor = FreshContextExecutor(max_workers=1)
    try:
        assert executor.submit(marker.set, "job-1").result(timeout=5) is not None
        assert executor.submit(marker.get).result(timeout=5) == "unset"
    finally:
        executor.shutdown(wait=True)
