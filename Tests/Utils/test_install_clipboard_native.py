"""Exercise the bounded clipboard subprocess without touching the desktop."""

from __future__ import annotations

import asyncio
import os

import psutil
import pytest

from tldw_chatbook.Utils import install_clipboard


class _ClipboardApp:
    is_web = False
    is_headless = False

    def copy_to_clipboard(self, command):
        raise AssertionError("cancelled copying must not fall back")


class _ClipboardWidget:
    def __init__(self):
        self.app = _ClipboardApp()
        self.notices = []

    def notify(self, *args, **kwargs):
        self.notices.append(args)


@pytest.fixture
def clipboard_module(tmp_path, monkeypatch):
    # Substitute only the external desktop library in the real child process.
    # Imports, stdin transport, exit status, timeout and process cleanup stay real.
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("CLIPBOARD_TEST_PATH", str(tmp_path / "clipboard.txt"))
    module = tmp_path / "pyperclip.py"
    return module, tmp_path / "clipboard.txt"


@pytest.mark.parametrize("outcome", ["success", "silent", "raises"])
@pytest.mark.asyncio
async def test_native_copy_requires_matching_readback(clipboard_module, outcome):
    module, clipboard = clipboard_module
    module.write_text(
        "import os\nfrom pathlib import Path\n"
        "path = Path(os.environ['CLIPBOARD_TEST_PATH'])\n"
        "def copy(text):\n"
        + (
            "    raise RuntimeError('unavailable')\n"
            if outcome == "raises"
            else "    path.write_text(text, encoding='utf-8')\n"
        )
        + "def paste():\n"
        + (
            "    return 'stale value'\n"
            if outcome == "silent"
            else "    return path.read_text(encoding='utf-8')\n"
        )
    )
    command = 'pip install "tldw_chatbook[pdf]" # café'
    assert await install_clipboard._copy_native(command) is (outcome == "success")
    if outcome != "raises":
        assert clipboard.read_text(encoding="utf-8") == command


@pytest.mark.asyncio
async def test_process_owned_qt_clipboard_does_not_confirm_delivery(clipboard_module):
    module, _ = clipboard_module
    # Qt can read its own value, but it loses X11 ownership when this helper
    # exits. Its in-process value is not evidence another application can paste.
    module.write_text(
        "value = ''\n"
        "def copy_qt(text):\n"
        "    global value\n"
        "    value = text\n"
        "copy = copy_qt\n"
        "def paste(): return value\n"
    )
    assert (
        await install_clipboard._copy_native('pip install "tldw_chatbook[pdf]"')
        is False
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX clipboard subprocess tree")
@pytest.mark.asyncio
async def test_timed_out_clipboard_descendant_cannot_overwrite_a_later_copy(
    clipboard_module, monkeypatch
):
    module, clipboard = clipboard_module
    marker = module.parent / "child.pid"
    release = module.parent / "release"
    # A real descendant waits for release before writing. The timeout must
    # terminate that descendant as well as the immediate Python helper.
    child = (
        "import os,time; from pathlib import Path; "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); "
        f"gate=Path({str(release)!r}); target=Path({str(clipboard)!r})\n"
        "while not gate.exists(): time.sleep(0.01)\n"
        "target.write_text('stale')\n"
    )
    module.write_text(
        "import subprocess,sys\n"
        f"def copy(text): subprocess.run([sys.executable, '-c', {child!r}], check=True)\n"
        "def paste(): return 'stale'\n"
    )
    assert await install_clipboard._copy_native("first") is False
    assert marker.exists(), "the child must start before timeout proves cleanup"
    child_pid = int(marker.read_text())
    release.touch()
    clipboard.write_text("second")
    # A dead descendant may briefly remain a zombie; neither state can write.
    try:
        observed = psutil.Process(child_pid).status()
    except psutil.NoSuchProcess:
        pass
    else:
        assert observed in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD), observed
    assert clipboard.read_text() == "second"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX clipboard subprocess tree")
async def test_cancelled_copy_reaps_descendant_before_returning(
    clipboard_module, monkeypatch
):
    module, clipboard = clipboard_module
    marker = module.parent / "child.pid"
    release = module.parent / "release"
    child = (
        "import os,time; from pathlib import Path; "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); "
        f"gate=Path({str(release)!r}); target=Path({str(clipboard)!r})\n"
        "while not gate.exists(): time.sleep(0.01)\n"
        "target.write_text('cancelled command')\n"
    )
    module.write_text(
        "import subprocess,sys\n"
        f"def copy(text): subprocess.run([sys.executable, '-c', {child!r}], check=True)\n"
        "def paste(): return 'cancelled command'\n"
    )
    for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    widget = _ClipboardWidget()
    pending = asyncio.create_task(
        install_clipboard.copy_install_command(widget, "first")
    )
    try:
        async with asyncio.timeout(2):
            while not marker.exists():
                await asyncio.sleep(0.01)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert widget.notices == []
        try:
            status = psutil.Process(int(marker.read_text())).status()
        except psutil.NoSuchProcess:
            pass
        else:
            assert status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
        clipboard.write_text("later command")
        release.touch()
        assert clipboard.read_text() == "later command"
    finally:
        release.touch()
        if not pending.done():
            pending.cancel()


@pytest.mark.asyncio
async def test_later_copy_waits_for_an_active_native_copy(monkeypatch):
    entered = []
    release = asyncio.Event()
    started = asyncio.Event()

    async def native_copy(command):
        entered.append(command)
        started.set()
        if command == "first":
            await release.wait()
        return True

    for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(install_clipboard, "_copy_native", native_copy)
    widget = _ClipboardWidget()
    first = asyncio.create_task(install_clipboard.copy_install_command(widget, "first"))
    await started.wait()
    second = asyncio.create_task(
        install_clipboard.copy_install_command(widget, "second")
    )
    try:
        # Give the later event handler its turn while the first write is held.
        await asyncio.sleep(0)
        assert entered == ["first"]
    finally:
        release.set()
        await asyncio.gather(first, second)
    assert entered == ["first", "second"]
