"""Fresh-process import boundaries for the audio policy reducer."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


_GUARDS = """
import importlib.abc
import sys

forbidden_attempts = []
blocked = ('sounddevice', 'pyaudio', 'torch', 'mlx', 'transformers',
           'huggingface_hub', 'onnxruntime', 'parakeet_mlx')

class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            forbidden_attempts.append(fullname)
            raise AssertionError('forbidden audio/model import: ' + fullname)

sys.meta_path.insert(0, Guard())

def no_network(event, args):
    if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo',
                 'socket.bind', 'socket.sendto'):
        raise AssertionError('network is forbidden in import probes')

sys.addaudithook(no_network)
"""


def _probe(script, tmp_path):
    env = dict(os.environ)
    env.update(
        HOME=str(tmp_path),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(tmp_path / "config.toml"),
        TLDW_TEST_MODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", _GUARDS + textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module",
    [
        "voice_process_types",
        "voice_turn_coordinator",
        "voice_transcription",
        "voice_phrase_sequencer",
        "voice_process_core",
    ],
)
def test_audio_policy_import_has_no_parent_or_model_graph(module, tmp_path):
    _probe(
        f"""
        blocked += ('textual', 'tldw_chatbook.Chat', 'tldw_chatbook.TTS',
                    'tldw_chatbook.app', 'tldw_chatbook.DB',
                    'tldw_chatbook.LLM_Calls', 'tldw_chatbook.LLM_Provider_Catalog')
        import importlib
        importlib.import_module('tldw_chatbook.Audio.{module}')
        assert not forbidden_attempts, forbidden_attempts
        loaded = sorted(name for name in sys.modules if any(
            name == prefix or name.startswith(prefix + '.') for prefix in blocked
        ))
        assert not loaded, loaded
        """,
        tmp_path,
    )


@pytest.mark.parametrize("package", ["UI", "Widgets"])
def test_ui_package_installs_static_compatibility_idempotently(package, tmp_path):
    _probe(
        f"""
        import importlib
        import tldw_chatbook
        from textual.widgets import Static
        if hasattr(Static, 'renderable'):
            del Static.renderable
        package = importlib.import_module('tldw_chatbook.{package}')
        widget = Static('before')
        assert widget.renderable == widget.content
        widget.renderable = 'after'
        assert widget.content == 'after'
        shim = Static.renderable
        importlib.reload(package)
        assert Static.renderable is shim
        assert not forbidden_attempts, forbidden_attempts
        """,
        tmp_path,
    )
