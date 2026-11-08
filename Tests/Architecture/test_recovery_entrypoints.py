"""Real launch imports must refuse before config, effects, or defaults."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tldw_chatbook.Backup_Recovery.control_records import register_pending

ROOT = Path(__file__).resolve().parents[2]


# TASK-38991: the two main routes do not refuse on stderr. They open the
# recovery-only host (cli.main_cli_runner -> launcher.startup_unlock ->
# minimal_recovery, b5251e9a6e / 87aa6d1d03), which never imports the app.
MAIN_ROUTES = [
    "from tldw_chatbook.cli import main_cli_runner; main_cli_runner()",
    'import runpy; runpy.run_module("tldw_chatbook", run_name="__main__")',
]

# The real minimal_recovery runs the Textual host; this stub records the call
# and returns the real host's exit status, so the child exits.
RECOVERY_HOST_STUB = """
class ForbidApp:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tldw_chatbook.app":
            print("app_import_before_fence", file=sys.stderr, flush=True)
            raise AssertionError("app_import_before_fence")
sys.meta_path.insert(0, ForbidApp())
from tldw_chatbook.Backup_Recovery import launcher
def recovery_only_host(reason, **_):
    print("recovery_only_host:" + reason, file=sys.stderr, flush=True)
    return 0
launcher.minimal_recovery = recovery_only_host
"""


def _run_pending_route(tmp_path, route, config_present, prelude=""):
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
        + prelude
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
    assert "side_effect_import_before_fence" not in result.stderr
    assert config.exists() is config_present
    assert not (home / ".local").exists()
    assert not (home / ".config" / "tldw_cli" / "config.toml").exists()
    return result


@pytest.mark.parametrize(
    "route",
    [
        'import runpy; runpy.run_module("tldw_chatbook.app", run_name="__main__")',
        'import runpy; runpy.run_module("tldw_chatbook.app", run_name="__mp_main__")',
        "from tldw_chatbook.Web_Server.serve import main; main()",
        'import runpy; runpy.run_module("tldw_chatbook.MCP", run_name="__main__")',
        "import tldw_chatbook.config",
        'import runpy; runpy.run_module("tldw_chatbook.MCP.server", run_name="__main__")',
        'import runpy; runpy.run_module("tldw_chatbook.RAG_Search.backfill", run_name="__main__")',
        'import runpy; runpy.run_module("tldw_chatbook.TTS.backends.higgs_voice_manager", run_name="__main__")',
        'import runpy; runpy.run_module("tldw_chatbook.Config_Files.create_custom_template", run_name="__main__")',
        f"import runpy; runpy.run_path({str(ROOT / 'tldw_chatbook/TTS/backends/chatterbox_process.py')!r}, run_name='__main__')",
    ],
)
@pytest.mark.parametrize("config_present", [True, False])
def test_real_route_refuses_before_side_effect_imports(tmp_path, route, config_present):
    result = _run_pending_route(tmp_path, route, config_present)
    assert result.returncode != 0
    assert "Recovery required: recovery_pending" in result.stderr, result.stderr


@pytest.mark.parametrize("route", MAIN_ROUTES)
@pytest.mark.parametrize("config_present", [True, False])
def test_main_route_opens_recovery_only_host_without_app_import(
    tmp_path, route, config_present
):
    result = _run_pending_route(
        tmp_path, route, config_present, prelude=RECOVERY_HOST_STUB
    )
    assert "app_import_before_fence" not in result.stderr, result.stderr
    assert "recovery_only_host:recovery_pending" in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr


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


def _bootstrap_import_inventory(tree):
    import ast
    from collections import Counter

    def collect(node, owner="<module>"):
        found = Counter()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            key = (
                owner,
                type(node).__name__,
                getattr(node, "level", 0),
                getattr(node, "module", None),
                tuple((alias.name, alias.asname) for alias in node.names),
            )
            found[key] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested = node.name if owner == "<module>" else owner + "." + node.name
            for child in node.body:
                found.update(collect(child, nested))
        else:
            for child in ast.iter_child_nodes(node):
                found.update(collect(child, owner))
        return found

    return collect(tree)


def _assert_bootstrap_import_inventory(tree):
    from collections import Counter

    # Pin eager and lazy owners separately; a permitted lazy import cannot move
    # ahead of the fence or silently acquire another name, alias or occurrence.
    expected = Counter(
        {
            ("<module>", "Import", 0, None, (("functools", None),)): 1,
            ("<module>", "Import", 0, None, (("hashlib", None),)): 1,
            ("<module>", "Import", 0, None, (("json", None),)): 1,
            ("<module>", "Import", 0, None, (("re", None),)): 1,
            ("<module>", "Import", 0, None, (("stat", None),)): 1,
            ("<module>", "Import", 0, None, (("threading", None),)): 1,
            ("<module>", "ImportFrom", 0, "__future__", (("annotations", None),)): 1,
            (
                "<module>",
                "ImportFrom",
                0,
                "collections.abc",
                (("Callable", None), ("Iterable", None), ("Mapping", None)),
            ): 1,
            ("<module>", "ImportFrom", 0, "contextlib", (("contextmanager", None),)): 1,
            ("<module>", "ImportFrom", 0, "pathlib", (("Path", None),)): 1,
            (
                "<module>",
                "ImportFrom",
                0,
                "tldw_chatbook.Utils.platform_files",
                (("fcntl", None), ("os", None)),
            ): 1,
            (
                "<module>",
                "ImportFrom",
                0,
                "typing",
                (("ParamSpec", None), ("TypeVar", None)),
            ): 1,
            ("<module>", "ImportFrom", 1, "limits", (("_TraceReadLimit", None),)): 1,
            (
                "<module>",
                "ImportFrom",
                1,
                "profile_paths",
                (
                    ("default_config_path", None),
                    ("effective_config_path", None),
                    ("lexical_path", None),
                ),
            ): 1,
            (
                "<module>",
                "ImportFrom",
                2,
                "Utils.private_paths",
                (("_native_close", None), ("_open_verified_parent", None)),
            ): 1,
            (
                "_binding",
                "ImportFrom",
                1,
                "mobile_preview_budget",
                (("read_options", None),),
            ): 1,
            (
                "_bounded_names",
                "ImportFrom",
                1,
                "journal",
                (("_artifact_component_limit", None),),
            ): 1,
            (
                "_control_records",
                "ImportFrom",
                1,
                "mobile_preview_budget",
                (("read_options", None),),
            ): 1,
            (
                "_read",
                "ImportFrom",
                1,
                "mobile_preview_budget",
                (("read_options", None),),
            ): 1,
            (
                "_registry",
                "ImportFrom",
                1,
                "mobile_preview_budget",
                (("read_options", None),),
            ): 1,
            (
                "effective_roots",
                "ImportFrom",
                1,
                "admission",
                (("_price_paths", None),),
            ): 1,
            (
                "effective_roots",
                "ImportFrom",
                1,
                "effective_roots",
                (("effective_roots", "prove_absence"),),
            ): 1,
        }
    )
    assert _bootstrap_import_inventory(tree) == expected


def test_bootstrap_dependency_leaf_and_script_route_inventory():
    import ast
    import tomllib

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["scripts"] == {
        "tldw-cli": "tldw_chatbook.cli:main_cli_runner",
        "tldw-serve": "tldw_chatbook.Web_Server.serve:main",
    }
    tree = ast.parse((ROOT / "tldw_chatbook/Backup_Recovery/bootstrap.py").read_text())
    _assert_bootstrap_import_inventory(tree)
    limits = ast.parse((ROOT / "tldw_chatbook/Backup_Recovery/limits.py").read_text())
    assert _bootstrap_import_inventory(limits) == {
        ("<module>", "ImportFrom", 0, "dataclasses", (("dataclass", None),)): 1
    }
    refusal = next(
        node
        for node in limits.body
        if isinstance(node, ast.ClassDef) and node.name == "_TraceReadLimit"
    )
    assert ast.dump(refusal.bases[0]) == ast.dump(
        ast.Name(id="RuntimeError", ctx=ast.Load())
    )
    assert (
        len(refusal.bases) == 1 and not refusal.keywords and not refusal.decorator_list
    )
    assert len(refusal.body) == 1
    assert isinstance(refusal.body[0], ast.Expr)
    assert isinstance(refusal.body[0].value, ast.Constant)
    assert isinstance(refusal.body[0].value.value, str)


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
    config.write_text("[preferences]\n")
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
    "TTS/backends/chatterbox_process.py": "enrolled_direct_worker",
    "Tools/_grep_worker.py": "external_read_only_worker",
    "css/check_bundle_sync.py": "development_tool",
    "css/build_css.py": "development_tool",
    "css/Themes/theme_tester.py": "development_example",
    "Prompt_Management/Prompts_Interop.py": "temporary_example",
    "Web_Scraping/Article_Scraper/main.py": "external_demo",
    "Third_Party/aider/waiting.py": "vendored_example",
    "Third_Party/textual_fspicker/example_enhanced.py": "vendored_example",
    "Third_Party/textual_fspicker/__main__.py": "vendored_example",
    "Widgets/Tamagotchi/examples/simple_tamagotchi.py": "development_example",
    # TASK-38991 review. recovery_host: the startup-independent recovery CLI
    # (`tldw-cli recovery`'s recovery_main). app_spawned_pipe_worker: a child
    # an admitted app or recovery process spawns; its inputs are argv, pipes
    # or paths the parent staged, and it neither loads config nor enrolls.
    # config_fenced_worker: an app-spawned child whose imports load
    # tldw_chatbook.config (lab_runner: Chunking/__init__ -> Chunk_Lib ->
    # config, which writes config.toml under a fresh HOME), so it reaches
    # config's import-time recovery fence (review r10a-rev1).
    # remote_host_worker: the generated bundle runs on the SSH host, not here.
    "Backup_Recovery/__main__.py": "recovery_host",
    "Audio/diarizer_worker.py": "app_spawned_pipe_worker",
    "Audio/voice_process_entry.py": "app_spawned_pipe_worker",
    "Backup_Recovery/age_worker.py": "app_spawned_pipe_worker",
    "Backup_Recovery/detached_worker.py": "app_spawned_pipe_worker",
    "Backup_Recovery/rag_projection_validation.py": "app_spawned_pipe_worker",
    "Chat/console_trace_regex_worker.py": "app_spawned_pipe_worker",
    "Chunking/lab_runner.py": "config_fenced_worker",
    "DB/private_sqlite_helper_entry.py": "app_spawned_pipe_worker",
    "Terminal/posix_launcher.py": "app_spawned_pipe_worker",
    "Tools/workspace_tool_worker.py": "app_spawned_pipe_worker",
    "Web_Server/artifact_share_server.py": "app_spawned_pipe_worker",
    "Tools/project_index_scan_worker.py": "external_read_only_worker",
    "Tools/build_remote_worker_bundle.py": "development_tool",
    "Tools/remote_worker_bundle.py": "remote_host_worker",
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
    import site
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
    # Plain paths share installed dependencies without executing their .pth hooks.
    shared = tmp_path / "shared" / "site-packages"
    shared.mkdir(parents=True)
    pth_marker = tmp_path / "shared-pth-executed"
    (shared / "must-not-run.pth").write_text(
        str(ROOT)
        + "\n"
        + f"import pathlib; pathlib.Path({str(pth_marker)!r}).write_text('executed')\n"
    )
    dependency_paths = [
        Path(value).resolve()
        for value in dict.fromkeys((*site.getsitepackages(), *sys.path))
        if value
        and Path(value).is_absolute()
        and Path(value).name in {"site-packages", "dist-packages"}
        and Path(value).is_dir()
    ]
    dependency_paths.append(shared)
    (packages / "dependencies.pth").write_text(
        "\n".join(str(path) for path in dependency_paths) + "\n"
    )
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
        os.environ,
        HOME=str(home),
        TLDW_CONFIG_PATH=str(config),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    provenance = subprocess.run(
        [
            str(interpreter),
            "-c",
            (
                "import importlib.util, sys; import pydantic; assert pydantic.BaseModel; "
                "assert sys.dont_write_bytecode; "
                "print(importlib.util.find_spec('tldw_chatbook') is not None)"
            ),
        ],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert provenance.returncode == 0, provenance.stderr
    assert not pth_marker.exists(), "shared dependency .pth was executed"
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
    assert not pth_marker.exists(), "direct launch executed a shared dependency .pth"

@pytest.mark.parametrize(
    "mutation",
    ["eager", "lazy", "duplicate", "name", "alias", "move", "if", "try"],
)
def test_bootstrap_import_inventory_rejects_unreviewed_phase_or_site(mutation):
    import ast
    import copy

    tree = ast.parse((ROOT / "tldw_chatbook/Backup_Recovery/bootstrap.py").read_text())
    _assert_bootstrap_import_inventory(tree)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "effective_roots"
    )
    site = next(node for node in owner.body if isinstance(node, ast.ImportFrom))
    if mutation == "eager":
        tree.body.append(ast.parse("import tldw_chatbook.config").body[0])
    elif mutation == "lazy":
        owner.body.append(ast.parse("import tldw_chatbook.config").body[0])
    elif mutation == "duplicate":
        owner.body.append(copy.deepcopy(site))
    elif mutation == "name":
        site.names[0].name = "unreviewed_name"
    elif mutation == "alias":
        site.names[0].asname = "unreviewed_alias"
    elif mutation == "move":
        owner.body.remove(site)
        tree.body.append(site)
    elif mutation == "if":
        tree.body.append(
            ast.If(test=ast.Constant(value=True), body=[copy.deepcopy(site)], orelse=[])
        )
    else:
        tree.body.append(
            ast.Try(
                body=[copy.deepcopy(site)],
                handlers=[],
                orelse=[],
                finalbody=[ast.Pass()],
            )
        )
    with pytest.raises(AssertionError):
        _assert_bootstrap_import_inventory(tree)


@pytest.mark.parametrize("requested", ["bootstrap", "pydantic", "tldw_chatbook.config", "tldw_chatbook.Backup_Recovery.admission", "tldw_chatbook.Backup_Recovery.journal"])
def test_bootstrap_cold_import_has_only_reviewed_leaf_dependencies(tmp_path, requested):
    home = tmp_path / "home"
    home.mkdir()
    config = tmp_path / "never-created.toml"
    code = r"""
import importlib, sys
assert sys.dont_write_bytecode
from Tests import network_guard
network_guard.install()
allowed = {
    'tldw_chatbook', 'tldw_chatbook.Utils',
    'tldw_chatbook.Utils.tiktoken_runtime',
    'tldw_chatbook.Utils.private_paths',
    'tldw_chatbook.Utils.platform_files',
    'tldw_chatbook.Utils.windows_files',
    'tldw_chatbook.Backup_Recovery',
    'tldw_chatbook.Backup_Recovery.bootstrap',
    'tldw_chatbook.Backup_Recovery.limits',
    'tldw_chatbook.Backup_Recovery.profile_paths',
}
class LeafOnly:
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith('tldw_chatbook') and fullname not in allowed) or fullname.split('.')[0] in {'pydantic','textual','loguru','tiktoken','torch'}:
            raise AssertionError('bootstrap_import_not_leaf:' + fullname)
sys.meta_path.insert(0, LeafOnly())
heavy = {'pydantic','textual','loguru','tiktoken','torch'}
assert not any(name.split('.')[0] in heavy for name in sys.modules)
requested = sys.argv[1]
if requested != 'bootstrap':
    importlib.import_module(requested)
from tldw_chatbook.Backup_Recovery import bootstrap, limits
assert bootstrap._TraceReadLimit is limits._TraceReadLimit
assert not any(name.split('.')[0] in heavy for name in sys.modules)
assert {name for name in sys.modules if name.startswith('tldw_chatbook')} <= allowed
assert not network_guard.blocked_attempts()
print('cold leaf imported')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, requested],
        cwd=ROOT,
        env=dict(
            os.environ,
            HOME=str(home),
            USERPROFILE=str(home),
            XDG_CONFIG_HOME=str(home / ".config"),
            XDG_DATA_HOME=str(home / ".local" / "share"),
            TLDW_CONFIG_PATH=str(config),
            PYTHONPATH=str(ROOT),
            PYTHONDONTWRITEBYTECODE="1",
        ),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if requested == "bootstrap":
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "cold leaf imported"
    else:
        assert result.returncode != 0
        assert "bootstrap_import_not_leaf:" + requested in result.stderr
    assert not config.exists()
    assert not (home / ".config").exists()
    assert not (home / ".local").exists()
