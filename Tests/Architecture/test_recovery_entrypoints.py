"""Real launch imports must refuse before config, effects, or defaults."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tldw_chatbook.Backup_Recovery.control_records import register_pending

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "route",
    [
        "from tldw_chatbook.cli import main_cli_runner; main_cli_runner()",
        'import runpy; runpy.run_module("tldw_chatbook", run_name="__main__")',
        'import runpy; runpy.run_module("tldw_chatbook.app", run_name="__main__")',
        'import runpy; runpy.run_module("tldw_chatbook.app", run_name="__mp_main__")',
        "from tldw_chatbook.Web_Server.serve import main; main()",
        'import runpy; runpy.run_module("tldw_chatbook.MCP", run_name="__main__")',
        "import tldw_chatbook.config",
        'import runpy; runpy.run_module("tldw_chatbook.MCP.server", run_name="__main__")',
        'import runpy; runpy.run_module("tldw_chatbook.RAG_Search.backfill", run_name="__main__")',
        'import runpy; runpy.run_module("tldw_chatbook.TTS.utils.download_models", run_name="__main__")',
        'import runpy; runpy.run_module("tldw_chatbook.TTS.backends.higgs_voice_manager", run_name="__main__")',
        'import runpy; runpy.run_module("tldw_chatbook.Config_Files.create_custom_template", run_name="__main__")',
        f"import runpy; runpy.run_path({str(ROOT / 'tldw_chatbook/TTS/backends/chatterbox_process.py')!r}, run_name='__main__')",
    ],
)
@pytest.mark.parametrize("config_present", [True, False])
def test_real_route_refuses_before_side_effect_imports(tmp_path, route, config_present):
    home = tmp_path / "home"
    config = tmp_path / "custom.toml"
    config.write_text("broken [")
    root = home / ".config" / "tldw_cli" / "recovery-bootstrap"
    root.parent.mkdir(parents=True, mode=0o700)
    register_pending(root, "op", ("p",), tmp_path / "missing-control", (config,))
    if not config_present:
        config.unlink()
    env = dict(
        os.environ, HOME=str(home), TLDW_CONFIG_PATH=str(config), PYTHONPATH=str(ROOT)
    )
    script = (
        """
import sys
class Forbidden:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"tldw_chatbook.DB.ChaChaNotes_DB", "tldw_chatbook.Utils.optional_deps", "tldw_chatbook.Metrics.metrics", "mcp_unified.gateway", "tldw_chatbook.RAG_Search.simplified", "tldw_chatbook.TTS.TTS_Generation", "torch", "transformers"}:
            raise AssertionError("side_effect_import_before_fence:" + fullname)
sys.meta_path.insert(0, Forbidden())
"""
        + route
    )
    result = subprocess.run(
        [sys.executable, "-c", script, "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "Recovery required: recovery_pending" in result.stderr, result.stderr
    assert "side_effect_import_before_fence" not in result.stderr
    assert config.exists() is config_present
    assert not (home / ".local").exists()
    assert not (home / ".config" / "tldw_cli" / "config.toml").exists()


def test_actual_multiprocessing_spawn_rechecks_pending_before_app_import(tmp_path):
    home = tmp_path / "home"
    config = tmp_path / "config.toml"
    config.write_text("broken [")
    root = home / ".config" / "tldw_cli" / "recovery-bootstrap"
    root.parent.mkdir(parents=True, mode=0o700)
    register_pending(root, "op", ("p",), tmp_path / "missing-control", (config,))
    script = tmp_path / "spawn_probe.py"
    script.write_text("""
import multiprocessing
import runpy

def worker():
    raise AssertionError("worker reached effects")

if __name__ == "__mp_main__":
    runpy.run_module("tldw_chatbook.app", run_name="__mp_main__")
if __name__ == "__main__":
    process = multiprocessing.get_context("spawn").Process(target=worker)
    process.start()
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join()
        raise AssertionError("spawn hung")
    print("child_exit", process.exitcode)
""")
    env = dict(
        os.environ, HOME=str(home), TLDW_CONFIG_PATH=str(config), PYTHONPATH=str(ROOT)
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "child_exit 1" in result.stdout
    assert "Recovery required: recovery_pending" in result.stderr
    assert "worker reached effects" not in result.stderr
    assert not (home / ".local").exists()


def test_bootstrap_dependency_leaf_and_script_route_inventory():
    import ast
    import tomllib

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["scripts"] == {
        "tldw-cli": "tldw_chatbook.cli:main_cli_runner",
        "tldw-serve": "tldw_chatbook.Web_Server.serve:main",
    }
    tree = ast.parse((ROOT / "tldw_chatbook/Backup_Recovery/bootstrap.py").read_text())
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert set(imports) <= {
        "__future__",
        "contextlib",
        "pathlib",
        "Utils.private_paths",
        "profile_paths",
        "typing",
    }


def test_console_enrolls_before_loading_app_and_releases_on_process_exit(tmp_path):
    import selectors
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )

    home = tmp_path / "home"
    root = home / ".config" / "tldw_cli" / "recovery-bootstrap"
    config = tmp_path / "config.toml"
    config.write_text("preferences")
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    authority = admission_authority(root)
    authority.register("profile", (config, data))
    script = """
import sys
class PauseBeforeApp:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tldw_chatbook.app":
            print("before_app", flush=True)
            sys.stdin.readline()
            raise SystemExit(0)
sys.meta_path.insert(0, PauseBeforeApp())
from tldw_chatbook.cli import main_cli_runner
main_cli_runner()
"""
    env = dict(
        os.environ, HOME=str(home), TLDW_CONFIG_PATH=str(config), PYTHONPATH=str(ROOT)
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        cwd=ROOT,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(child.stdout, selectors.EVENT_READ)
            assert ready.select(15)
        assert child.stdout.readline().strip() == "before_app"
        with pytest.raises(
            RecoveryRequired, match="close_unenrolled_clients_and_restart"
        ):
            bind_profile(root, config, ("profile",), root / "admission")
        assert not (home / ".local").exists()
        child.stdin.write("exit\n")
        child.stdin.flush()
        _, errors = child.communicate(timeout=15)
        assert child.returncode == 0, errors
        bind_profile(root, config, ("profile",), root / "admission")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


# Every syntactic package __main__ route is explicitly classified. New routes
# require review even if they happen to reach a lower-level guard transitively.
ROUTE_CLASSIFICATIONS = {
    "app.py": "enrolled_app",
    "__main__.py": "enrolled_app",
    "MCP/server.py": "enrolled_mcp",
    "MCP/__main__.py": "enrolled_mcp",
    "Web_Server/serve.py": "enrolled_web",
    "RAG_Search/backfill.py": "enrolled_headless",
    "Config_Files/create_custom_template.py": "enrolled_config",
    "Utils/paths.py": "enrolled_config",
    "TTS/utils/download_models.py": "enrolled_tts_direct",
    "TTS/backends/higgs_voice_manager.py": "enrolled_tts_direct",
    "TTS/backends/chatterbox_process.py": "enrolled_direct_worker",
    "Tools/_grep_worker.py": "external_read_only_worker",
    "css/check_bundle_sync.py": "development_tool",
    "css/build_css.py": "development_tool",
    "css/Themes/theme_tester.py": "development_example",
    "Prompt_Management/Prompts_Interop.py": "temporary_example",
    "Web_Scraping/Article_Scraper/main.py": "external_demo",
    "Chunking/templates/example_usage.py": "development_example",
    "Third_Party/aider/waiting.py": "vendored_example",
    "Third_Party/textual_fspicker/example_enhanced.py": "vendored_example",
    "Third_Party/textual_fspicker/__main__.py": "vendored_example",
    "Widgets/Tamagotchi/examples/simple_tamagotchi.py": "development_example",
}


def test_every_package_main_route_has_reviewed_classification():
    import ast

    current = set()
    package = ROOT / "tldw_chatbook"
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text())
        if any(
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "__name__"
            and any(
                isinstance(value, ast.Constant) and value.value == "__main__"
                for value in node.comparators
            )
            for node in ast.walk(tree)
        ):
            current.add(path.relative_to(package).as_posix())
    assert current == set(ROUTE_CLASSIFICATIONS)


@pytest.mark.parametrize("editable", [False, True])
@pytest.mark.parametrize("pending", [False, True])
def test_chatterbox_actual_script_path_without_pythonpath(tmp_path, editable, pending):
    import sysconfig
    import venv

    runtime = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(runtime)
    interpreter = runtime / "bin" / "python"
    packages = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    # Read shared dependencies without executing their editable-install .pth hooks.
    (packages / "dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
    if editable:
        (packages / "chatbook-editable.pth").write_text(str(ROOT) + "\n")
    marker = tmp_path / "runtime-import-reached"
    (packages / "sitecustomize.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "class StopBeforeModelEffects:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'chatterbox':\n"
        f"            Path({str(marker)!r}).write_text('runtime reached after guard')\n"
        "            raise SystemExit(0)\n"
        "sys.meta_path.insert(0, StopBeforeModelEffects())\n"
    )
    home = tmp_path / "home"
    config = tmp_path / "custom-config"
    config.write_text("private config")
    root = home / ".config" / "tldw_cli" / "recovery-bootstrap"
    root.parent.mkdir(parents=True, mode=0o700)
    if pending:
        register_pending(root, "op", ("p",), tmp_path / "control", (config,))
    env = dict(
        os.environ, HOME=str(home), TLDW_CONFIG_PATH=str(config), PYTHONUNBUFFERED="1"
    )
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    provenance = subprocess.run(
        [
            str(interpreter),
            "-c",
            "import importlib.util; print(importlib.util.find_spec('tldw_chatbook') is not None)",
        ],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert provenance.returncode == 0, provenance.stderr
    assert provenance.stdout.strip() == str(editable)
    result = subprocess.run(
        [
            str(interpreter),
            str(ROOT / "tldw_chatbook/TTS/backends/chatterbox_process.py"),
        ],
        env=env,
        cwd=tmp_path,
        input="",
        text=True,
        capture_output=True,
        timeout=20,
    )
    if pending:
        assert result.returncode != 0
        assert "Recovery required: recovery_pending" in result.stderr, result.stderr
        assert not marker.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert marker.read_text() == "runtime reached after guard"
    assert not (home / ".local").exists()
    assert not (home / ".config" / "tldw_cli" / "config.toml").exists()
