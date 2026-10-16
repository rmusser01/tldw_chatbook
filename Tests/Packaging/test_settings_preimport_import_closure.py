"""Keep category-specific Settings services outside screen pre-import."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "deferred_prefix",
    [
        "tldw_chatbook.UI.Screens.settings_rag_profile_adapter",
        "tldw_chatbook.Tool_Packs",
        "tldw_chatbook.Widgets.Settings_Widgets.tool_pack_import_review",
    ],
)
def test_preimport_defers_settings_category_services(
    tmp_path: Path, deferred_prefix: str
) -> None:
    """The full registry walk must not execute unopened Settings services."""
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "HOME": str(tmp_path / "home"),
        "USERPROFILE": str(tmp_path / "home"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "PYTHONPATH": str(repo_root),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from types import SimpleNamespace
import tldw_chatbook
from tldw_chatbook.app import TldwCli
import tldw_chatbook.UI.Screens.chat_screen

assert Path(tldw_chatbook.__file__).resolve().is_relative_to(Path.cwd())
routes = TldwCli._screen_preimport_route_order(SimpleNamespace())
assert routes
for route in routes:
    route.load_screen_class()
prefix = sys.argv[1]
loaded = sorted(name for name in sys.modules if name == prefix or name.startswith(prefix + "."))
assert not loaded, f"Pre-import executed unopened Settings services: {loaded}"
""",
            deferred_prefix,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-4000:]


def test_category_first_use_preserves_original_exports_and_assigned_patch_points(
    tmp_path,
):
    """The real registry stays lazy; first use retains class and patch identity."""
    repo_root = Path(__file__).resolve().parents[2]
    for name in ("home", "data", "config", "cache"):
        (tmp_path / name).mkdir()
    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "HOME": str(tmp_path / "home"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "PYTHONPATH": str(repo_root),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from importlib import import_module
from types import SimpleNamespace
from Tests import network_guard
network_guard.install()
network_guard.set_mode(network_guard.NetworkMode.BLOCKED)
from tldw_chatbook.app import TldwCli
import tldw_chatbook.UI.Screens.chat_screen
for route in TldwCli._screen_preimport_route_order(SimpleNamespace()):
    assert route.load_screen_class() is not None
prefixes = (
    "tldw_chatbook.UI.Plugins_Modules",
    "tldw_chatbook.Agents.agent_presets",
    "tldw_chatbook.Terminal",
    "tldw_chatbook.UI.Screens.settings_advanced_config",
    "tldw_chatbook.UI.Screens.settings_appearance_defaults",
    "tldw_chatbook.UI.Screens.settings_privacy_security",
    "tldw_chatbook.UI.Screens.settings_provider_view_model",
    "tldw_chatbook.UI.Screens.settings_web_search",
    "tldw_chatbook.Video_Generation.video_templates",
    "tldw_chatbook.Web_Scraping",
    "tldw_chatbook.Widgets.settings_advanced_config_panel",
    "tldw_chatbook.Widgets.settings_agents_panel",
    "tldw_chatbook.Widgets.settings_image_gen_panel",
    "tldw_chatbook.Widgets.settings_splash_screen_viewer",
    "tldw_chatbook.Widgets.settings_video_gen_panel",
    "tldw_chatbook.Widgets.settings_web_search_panel",
)
assert not [name for name in sys.modules if any(name == p or name.startswith(p + ".") for p in prefixes)]
from tldw_chatbook.UI.Screens import settings_screen as settings
for name, (module, member) in settings._CATEGORY_EXPORTS.items():
    assert getattr(settings, name) is getattr(import_module(module, settings.__package__), member)
# A real consumer must honor an explicitly assigned module patch, not bypass it
# through a local import. Constructor identity is unchanged after removing it.
original = settings.WebSearchSettings
sentinel = SimpleNamespace(changed=None)
settings.WebSearchSettings = lambda *args: sentinel
fake = SimpleNamespace(_web_search_settings=None, _settings_drafts={}, _web_search_changed=lambda: None)
assert settings.SettingsScreen._web_search_model(fake) is sentinel
assert sentinel.changed is fake._web_search_changed
del settings.WebSearchSettings
assert settings.WebSearchSettings is original
assert not network_guard.drain_blocked_attempts()
""",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-4000:]
