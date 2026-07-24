# TASK-545 Installed Distribution Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify truthful sdist and wheel artifacts, install and
exercise the wheel outside the checkout, and prevent every installed
application startup path from writing generated CSS into the installed target.

**Architecture:** Setuptools remains the build backend. A fresh-tree integration
fixture owns artifact construction, archive inspection, target installation,
private process state, and complete target hashing. `Packaging/check_manifest.py`
becomes the reusable release checker for the same content contract. One small
predicate in `app.py` distinguishes source trees from installed packages; the
existing CSS bootstrap sites keep their current source behavior but do nothing
to package files when installed.

**Tech Stack:** Python 3.11+, setuptools 77+, `build`, pip, `tarfile`,
`zipfile`, `email.parser`, `configparser`, pytest, Textual.

**ADR required:** yes

**ADR path:**
`backlog/decisions/025-immutable-installed-distribution-assets.md`

**Reason:** Packaging ownership, installed runtime immutability, build-tool
minimums, and startup write boundaries are architectural decisions. ADR-025
already records the approved decision; this plan implements it without a new
ADR.

**Design:**
`Docs/superpowers/specs/2026-07-24-installed-distribution-integrity-design.md`

**Backlog:** `TASK-545`

**Dependencies:** TASK-490 through TASK-497 are complete on this branch. Preserve
their privacy, deletion/migration, eval, and tool-worker invariants. Do not start
the larger application-state decomposition.

---

## File map

| Path | Responsibility |
| --- | --- |
| `MANIFEST.in` | Canonical sdist inclusion and exclusion rules consumed by setuptools. |
| `Packaging/MANIFEST.in` | Retired misplaced manifest; remove it when the root manifest is added. |
| `pyproject.toml` | Build-backend floor, SPDX license metadata, package discovery exclusions, and explicit wheel data. |
| `requirements-test.txt` | Makes the no-isolation artifact gate deterministic in test environments. |
| `Packaging/check_manifest.py` | Standalone required/forbidden artifact and metadata verifier. |
| `tldw_chatbook/app.py` | Source-tree predicate and the three guarded CSS bootstrap sites. |
| `Tests/Packaging/test_installed_distribution.py` | Fresh build, archive, checker, isolated install, loader, entry-point, privacy, and immutability regressions. |
| `Tests/Web_Server/test_web_server_dependency_gate.py` | Fast `main_cli_runner()` installed-tree no-build regression. |
| `Tests/UI/test_product_maturity_phase6_packaging_data_safety.py` | Cheap source-seam assertions for packaging declarations. |
| `Tests/CI/test_github_actions_test_workflow.py` | Ensures CI installs the build tools needed by the integration marker. |
| `Packaging/PACKAGING_CHECKLIST.md` | Maintainer commands and truthful artifact expectations. |
| `backlog/tasks/task-545 - Verify-installed-distributions-and-immutable-packaged-assets.md` | Plan link, acceptance state, and final implementation notes. |

Do not add a packaging framework, installer wrapper, application-state owner,
or recursive package-data catch-all. Keep the integration helpers in one test
module because no second consumer exists.

---

### Task 1: Pin explicit distribution declarations with a fresh artifact test

**Files:**

- Create: `Tests/Packaging/test_installed_distribution.py`
- Create: `MANIFEST.in`
- Delete: `Packaging/MANIFEST.in`
- Modify: `pyproject.toml:1-20`
- Modify: `pyproject.toml:399-420`
- Modify: `requirements-test.txt`
- Modify:
  `Tests/UI/test_product_maturity_phase6_packaging_data_safety.py:359-386`
- Modify: `Tests/CI/test_github_actions_test_workflow.py:11-17`

- [ ] **Step 1: Create the fresh-build fixture and archive readers**

Create `Tests/Packaging/test_installed_distribution.py` with module-level
`pytestmark = pytest.mark.integration`. Keep all build output under
`tmp_path_factory`; never clean the checkout.

```python
from __future__ import annotations

import configparser
from email.parser import Parser
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import NamedTuple
import zipfile

import pytest


pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_NAMES = {
    "academic_paper",
    "code_documentation",
    "conversation",
    "ebook_chapters",
    "json",
    "legal_document",
    "paragraphs",
    "rolling_summarize",
    "semantic",
    "sentences",
    "tokens",
    "words",
    "xml",
}


class BuiltDistributions(NamedTuple):
    source_root: Path
    dist_dir: Path
    sdist: Path
    wheel: Path


def _copy_build_inputs(destination: Path) -> None:
    ignored = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".DS_Store",
        "build",
        "dist",
        "*.egg-info",
    )
    for name in ("tldw_chatbook", "Packaging"):
        shutil.copytree(REPO_ROOT / name, destination / name, ignore=ignored)

    seen_test_trees: set[tuple[int, int]] = set()
    for name in ("Tests", "tests", "STests"):
        source = REPO_ROOT / name
        if not source.is_dir():
            continue
        stat = source.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen_test_trees:
            continue
        seen_test_trees.add(identity)
        shutil.copytree(source, destination / name, ignore=ignored)

    for name in (
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "CLAUDE.md",
        "CHANGELOG.md",
        "requirements.txt",
    ):
        source = REPO_ROOT / name
        if source.is_file():
            shutil.copy2(source, destination / name)


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> BuiltDistributions:
    source_root = tmp_path_factory.mktemp("distribution-source")
    _copy_build_inputs(source_root)
    dist_dir = source_root / "dist"
    command = [
        sys.executable,
        "-m",
        "build",
        "--sdist",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(dist_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=source_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert "`project.license` as a TOML table is deprecated" not in (
        completed.stdout + completed.stderr
    )
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(sdists) == 1
    assert len(wheels) == 1
    return BuiltDistributions(source_root, dist_dir, sdists[0], wheels[0])


def _sdist_members(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        files = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {name.split("/", 1)[0] for name in files}
    assert len(roots) == 1
    return {name.split("/", 1)[1] for name in files if "/" in name}


def _wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {name for name in archive.namelist() if not name.endswith("/")}
```

Use the existing interpreter rather than a newly resolved environment.
`--no-isolation` and later `pip --no-deps` are the no-network contract.

- [ ] **Step 2: Add the failing artifact content and metadata test**

Add exact required and forbidden assertions. Parse metadata and entry points
from the wheel instead of matching unstructured text.

```python
def test_built_artifacts_match_distribution_contract(
    built_distributions: BuiltDistributions,
) -> None:
    sdist_members = _sdist_members(built_distributions.sdist)
    wheel_members = _wheel_members(built_distributions.wheel)

    required_sdist = {
        "LICENSE",
        "README.md",
        "CLAUDE.md",
        "CHANGELOG.md",
        "MANIFEST.in",
        "pyproject.toml",
        "requirements.txt",
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/css/components/stats_screen.css",
        "tldw_chatbook/Config_Files/rag_pipelines.toml",
        "tldw_chatbook/Evals/config/eval_config.yaml",
        "tldw_chatbook/Third_Party/aider/LICENSE.txt",
        "tldw_chatbook/Third_Party/textual_fspicker/LICENSE",
    }
    required_wheel = {
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/Config_Files/rag_pipelines.toml",
        "tldw_chatbook/Evals/config/eval_config.yaml",
        "tldw_chatbook/Third_Party/aider/LICENSE.txt",
        "tldw_chatbook/Third_Party/textual_fspicker/LICENSE",
    }
    assert not required_sdist - sdist_members
    assert not required_wheel - wheel_members

    wheel_templates = {
        Path(name).stem
        for name in wheel_members
        if name.startswith("tldw_chatbook/Chunking/templates/")
        and name.endswith(".json")
    }
    assert wheel_templates == TEMPLATE_NAMES

    forbidden_wheel = {
        "tldw_chatbook/css/components/stats_screen.css",
        "tldw_chatbook/Config_Files/embedding_configs_examples.toml",
        "tldw_chatbook/Config_Files/pipeline_configs/custom_pipelines_example.toml",
        "tldw_chatbook/Chunking/templates/README.md",
        "tldw_chatbook/Chunking/templates/example_usage.py",
        "tldw_chatbook/Evals/DEVELOPER_GUIDE.md",
    }
    assert forbidden_wheel.isdisjoint(wheel_members)
    for members in (sdist_members, wheel_members):
        assert not any(
            name.startswith(("Tests/", "tests/", "STests/"))
            or "/__pycache__/" in name
            or name.endswith((".pyc", ".pyo", ".DS_Store"))
            for name in members
        )

    with zipfile.ZipFile(built_distributions.wheel) as archive:
        metadata_name = next(
            name for name in wheel_members if name.endswith(".dist-info/METADATA")
        )
        entry_points_name = next(
            name
            for name in wheel_members
            if name.endswith(".dist-info/entry_points.txt")
        )
        metadata = Parser().parsestr(
            archive.read(metadata_name).decode("utf-8")
        )
        entry_points = configparser.ConfigParser()
        entry_points.read_string(
            archive.read(entry_points_name).decode("utf-8")
        )

    with tarfile.open(built_distributions.sdist, "r:gz") as archive:
        pkg_info = next(
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith("/PKG-INFO")
        )
        pkg_info_stream = archive.extractfile(pkg_info)
        assert pkg_info_stream is not None
        sdist_metadata = Parser().parsestr(
            pkg_info_stream.read().decode("utf-8")
        )

    assert metadata["Metadata-Version"] == "2.4"
    assert metadata["License-Expression"] == "AGPL-3.0-or-later"
    assert "LICENSE" in (metadata.get_all("License-File") or [])
    assert sdist_metadata["Metadata-Version"] == "2.4"
    assert sdist_metadata["License-Expression"] == "AGPL-3.0-or-later"
    assert "LICENSE" in (sdist_metadata.get_all("License-File") or [])
    assert any(
        name.endswith(".dist-info/licenses/LICENSE") for name in wheel_members
    )
    assert dict(entry_points["console_scripts"]) == {
        "tldw-cli": "tldw_chatbook.cli:main_cli_runner",
        "tldw-serve": "tldw_chatbook.Web_Server.serve:main",
    }
```

- [ ] **Step 3: Run the artifact test to verify RED**

Run:

```bash
python -m pytest \
  Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract \
  -q -p no:cacheprovider
```

Expected: FAIL because the sdist lacks the root-manifest metadata, the wheel
lacks the RAG/chunking/eval/license data, and the build still emits legacy
metadata. If it fails for a copy or syntax error, correct the test and rerun
until the failure names a verified distribution defect.

- [ ] **Step 4: Add fast failing source-seam assertions**

Extend
`test_phase6_packaging_config_and_data_safety_source_seams_are_present()`:

```python
setuptools = pyproject["tool"]["setuptools"]
assert setuptools["include-package-data"] is False
assert "tldw_chatbook.Chunking.templates*" in setuptools["packages"]["find"]["exclude"]

for owner in (
    "tldw_chatbook.css",
    "tldw_chatbook.Config_Files",
    "tldw_chatbook.Chunking",
    "tldw_chatbook.Evals",
    "tldw_chatbook.Third_Party.aider",
    "tldw_chatbook.Third_Party.textual_fspicker",
):
    assert owner in setuptools["package-data"]

assert pyproject["build-system"]["requires"] == ["setuptools>=77.0"]
assert project["license"] == "AGPL-3.0-or-later"
assert project["license-files"] == ["LICENSE"]
assert (REPO_ROOT / "MANIFEST.in").is_file()
assert not (REPO_ROOT / "Packaging" / "MANIFEST.in").exists()
```

Add a CI dependency test:

```python
def test_ci_installs_distribution_build_dependencies() -> None:
    requirements = (PROJECT_ROOT / "requirements-test.txt").read_text()

    assert "build" in requirements.splitlines()
    assert "setuptools>=77" in requirements.splitlines()
```

Run both focused tests and confirm they fail before changing declarations.

- [ ] **Step 5: Move and correct the sdist manifest**

Move `Packaging/MANIFEST.in` to root `MANIFEST.in`. Retain the existing
metadata and artifact exclusions, then add:

```text
include MANIFEST.in
include pyproject.toml
recursive-include tldw_chatbook/Config_Files *.json *.md rag_pipelines.toml
recursive-include tldw_chatbook/Chunking/templates *.json
recursive-include tldw_chatbook/Evals/config *.yaml
recursive-include tldw_chatbook/css *.tcss *.css
include tldw_chatbook/Third_Party/aider/LICENSE.txt
include tldw_chatbook/Third_Party/textual_fspicker/LICENSE
recursive-exclude Tests *
recursive-exclude tests *
recursive-exclude STests *
```

Keep cache, bytecode, virtual-environment, egg-info, and OS exclusions. Do not
recursively include all `Third_Party` content.

- [ ] **Step 6: Make wheel content and metadata explicit**

Change the build and project metadata:

```toml
[build-system]
requires = ["setuptools>=77.0"]
build-backend = "setuptools.build_meta"

[project]
license = "AGPL-3.0-or-later"
license-files = ["LICENSE"]
```

Add:

```toml
[tool.setuptools]
include-package-data = false
```

Extend package discovery without disabling legitimate namespace packages:

```toml
exclude = [
    "Tests*",
    ".venv*",
    "tldw_chatbook.Chunking.templates*",
]
```

Extend the existing package-data table:

```toml
"tldw_chatbook.Config_Files" = [
    "*.json",
    "*.md",
    "rag_pipelines.toml",
]
"tldw_chatbook.Chunking" = ["templates/*.json"]
"tldw_chatbook.Evals" = ["config/*.yaml"]
"tldw_chatbook.Third_Party.aider" = ["LICENSE.txt"]
"tldw_chatbook.Third_Party.textual_fspicker" = ["LICENSE"]
```

Keep the current explicit TCSS rules. Do not add `*.css` to wheel package data;
`stats_screen.css` is an sdist-only build input.

- [ ] **Step 7: Make no-isolation builds reproducible in test environments**

Add exact lines to `requirements-test.txt`:

```text
build
setuptools>=77
```

Do not add a second packaging dependency.

- [ ] **Step 8: Run GREEN checks**

Run:

```bash
python -m pytest \
  Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py::test_phase6_packaging_config_and_data_safety_source_seams_are_present \
  Tests/CI/test_github_actions_test_workflow.py::test_ci_installs_distribution_build_dependencies \
  -q -p no:cacheprovider
```

Expected: 3 passed. Confirm the build output contains no legacy
`project.license` deprecation.

- [ ] **Step 9: Commit Task 1**

```bash
git add \
  MANIFEST.in \
  Packaging/MANIFEST.in \
  pyproject.toml \
  requirements-test.txt \
  Tests/Packaging/test_installed_distribution.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py \
  Tests/CI/test_github_actions_test_workflow.py
git commit -m "fix(packaging): declare distribution content explicitly"
```

---

### Task 2: Make the release checker enforce the same contract

**Files:**

- Modify: `Packaging/check_manifest.py`
- Modify: `Tests/Packaging/test_installed_distribution.py`

- [ ] **Step 1: Add a checker subprocess helper**

```python
def _run_manifest_checker(
    built: BuiltDistributions,
    dist_dir: Path,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(built.source_root / "Packaging" / "check_manifest.py"),
            str(dist_dir),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )
```

- [ ] **Step 2: Add RED tests for positive, duplicate, and forbidden cases**

```python
def test_release_checker_accepts_fresh_artifacts(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    result = _run_manifest_checker(
        built_distributions,
        built_distributions.dist_dir,
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_release_checker_rejects_multiple_wheels(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    shutil.copytree(built_distributions.dist_dir, dist_dir)
    shutil.copy2(
        built_distributions.wheel,
        dist_dir / f"duplicate-{built_distributions.wheel.name}",
    )

    result = _run_manifest_checker(built_distributions, dist_dir, tmp_path)

    assert result.returncode == 1
    assert "exactly one wheel" in (result.stdout + result.stderr).lower()


def test_release_checker_rejects_sdist_only_css_in_wheel(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    shutil.copytree(built_distributions.dist_dir, dist_dir)
    wheel = next(dist_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(
            "tldw_chatbook/css/components/stats_screen.css",
            "forbidden",
        )

    result = _run_manifest_checker(built_distributions, dist_dir, tmp_path)

    assert result.returncode == 1
    assert "stats_screen.css" in result.stdout + result.stderr


def test_release_checker_rejects_missing_runtime_data(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    shutil.copytree(built_distributions.dist_dir, dist_dir)
    wheel = next(dist_dir.glob("*.whl"))
    rewritten = wheel.with_suffix(".rewritten")
    missing = "tldw_chatbook/Evals/config/eval_config.yaml"
    with (
        zipfile.ZipFile(wheel) as source,
        zipfile.ZipFile(rewritten, "w") as destination,
    ):
        for member in source.infolist():
            if member.filename != missing:
                destination.writestr(member, source.read(member.filename))
    rewritten.replace(wheel)

    result = _run_manifest_checker(built_distributions, dist_dir, tmp_path)

    assert result.returncode == 1
    assert missing in result.stdout + result.stderr
```

Run these tests from a `cwd` without `dist/`.

Expected: the positive case fails because the current checker ignores the
explicit path; the negative cases are not considered meaningful until the
positive case reaches the real artifacts.

- [ ] **Step 3: Replace first-match/warning logic with exact validation**

Keep `check_manifest.py` standard-library-only. Change the callable boundary to
`def check_distribution(dist_dir: Path = Path("dist")) -> bool:`.

Add an optional CLI positional argument:

```python
parser = argparse.ArgumentParser()
parser.add_argument("dist_dir", nargs="?", type=Path, default=Path("dist"))
args = parser.parse_args()
raise SystemExit(0 if check_distribution(args.dist_dir) else 1)
```

Require exactly one `*.tar.gz` and exactly one `*.whl`; report all candidates
when the count is wrong. Normalize the sdist top-level directory, read wheel
members directly, and validate:

- every required file and every declared glob;
- exactly the thirteen JSON template names;
- forbidden root test/cache/OS paths;
- wheel-forbidden CSS, example TOML/Markdown, and
  `Chunking/templates/example_usage.py`;
- one wheel `METADATA` file with `License-Expression` and `License-File`;
- one sdist `PKG-INFO` file with the same Core Metadata 2.4 license contract;
- one project license under `.dist-info/licenses/`;
- one `entry_points.txt` with the exact two console targets; and
- the two vendored license paths.

Use `fnmatch.fnmatchcase()` for archive paths. Print each missing or forbidden
path and return `False`; do not merely print a non-Python file count.

- [ ] **Step 4: Run checker tests to GREEN**

Run:

```bash
python -m pytest \
  Tests/Packaging/test_installed_distribution.py \
  -q -p no:cacheprovider \
  -k "release_checker or built_artifacts"
```

Expected: 5 passed.

- [ ] **Step 5: Run the copied checker directly**

Use the fixture's failure diagnostics or a fresh temporary build; do not run
the checker against a stale repository `dist/`. Confirm the copied checker exits
0 against exactly one newly built sdist and wheel.

- [ ] **Step 6: Commit Task 2**

```bash
git add Packaging/check_manifest.py Tests/Packaging/test_installed_distribution.py
git commit -m "test(packaging): make artifact checker authoritative"
```

---

### Task 3: Drive installed execution red, then guard every CSS bootstrap site

**Files:**

- Modify: `Tests/Packaging/test_installed_distribution.py`
- Modify: `Tests/Web_Server/test_web_server_dependency_gate.py:53-106`
- Modify: `tldw_chatbook/app.py:10096-10492`

- [ ] **Step 1: Add target installation and immutable hash helpers**

Extend the packaging integration module:

```python
import hashlib
import json


def _install_wheel(
    built: BuiltDistributions,
    target: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--target",
        str(target),
        str(built.wheel),
    ]
    completed = subprocess.run(
        command,
        cwd=target.parent,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _target_hashes(target: Path) -> dict[str, str]:
    return {
        path.relative_to(target).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }


def _private_child_env(state_root: Path, target: Path) -> dict[str, str]:
    state_root = state_root.resolve(strict=True)
    config_root = state_root / "config"
    data_root = state_root / "data"
    temp_root = state_root / "tmp"
    for path in (config_root, data_root, temp_root):
        path.mkdir(parents=True, mode=0o700, exist_ok=True)

    env = os.environ.copy()
    for name in ("TLDW_TEST_CONFIG_ROOT", "TLDW_TEST_CONFIG_ROOT_OWNER"):
        env.pop(name, None)
    env.update(
        {
            "HOME": str(state_root),
            "USERPROFILE": str(state_root),
            "APPDATA": str(data_root),
            "LOCALAPPDATA": str(data_root),
            "XDG_CONFIG_HOME": str(config_root),
            "XDG_DATA_HOME": str(data_root),
            "TLDW_CONFIG_PATH": str(config_root / "config.toml"),
            "TMPDIR": str(temp_root),
            "TEMP": str(temp_root),
            "TMP": str(temp_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(target.resolve(strict=True)),
            "EXPECTED_TARGET": str(target.resolve(strict=True)),
            "EXPECTED_TEMPLATES": json.dumps(sorted(TEMPLATE_NAMES)),
        }
    )
    return env
```

Snapshot only after pip finishes so pip-created metadata/bytecode is the
baseline. Hash the complete target, including scripts and `.dist-info`.

- [ ] **Step 2: Add the installed resource and factory probe**

Run a child interpreter from a directory outside the repository. Define the
inline probe as a module constant:

```python
INSTALLED_PROBE = r"""
from pathlib import Path
import json
import os
import tomllib

import tldw_chatbook
from tldw_chatbook.Chunking.chunking_templates import ChunkingTemplateManager
from tldw_chatbook.Evals.config_loader import EvalConfigLoader
from tldw_chatbook.RAG_Search.pipeline_loader import PipelineLoader
from tldw_chatbook.app import TldwCli, get_app

package_root = Path(tldw_chatbook.__file__).resolve().parent
expected_target = Path(os.environ["EXPECTED_TARGET"]).resolve()
expected_templates = set(json.loads(os.environ["EXPECTED_TEMPLATES"]))
assert package_root.is_relative_to(expected_target)
assert (package_root / "css" / "tldw_cli_modular.tcss").is_file()

with (package_root / "Config_Files" / "rag_pipelines.toml").open("rb") as stream:
    assert "plain" in tomllib.load(stream)["pipelines"]

loader = PipelineLoader(config_dir=package_root / "Config_Files")
loader.load_pipeline_config()
assert "plain" in loader.pipelines
assert set(ChunkingTemplateManager().get_available_templates()) == expected_templates
assert "code_execution" in EvalConfigLoader().get_task_types()
assert (package_root / "Third_Party" / "aider" / "LICENSE.txt").is_file()
assert (
    package_root / "Third_Party" / "textual_fspicker" / "LICENSE"
).is_file()
assert isinstance(get_app(), TldwCli)
print(package_root)
"""
```

Pass this complete snippet to `python -c` and never import test helpers from the
checkout. The resolved package origin is printed for diagnostics.

- [ ] **Step 3: Discover and run the installed console scripts**

Search `<target>/bin` and `<target>/Scripts` with `shutil.which()` so POSIX and
Windows wrappers are supported. Run:

```text
<installed tldw-cli> --help
<installed tldw-serve> --help
```

with the private environment and unrelated `cwd`. Require return code 0 and
include command/stdout/stderr on failure.

Before the probe, snapshot `_target_hashes(target)`. After the resource probe
and both help commands:

- compare the full hash dictionaries for equality;
- concatenate captured stdout/stderr;
- recursively read private-root text logs; and
- reject `Building modular CSS`, `Failed to build modular CSS`, and
  `Error handling CSS file`.

The test orchestration is:

```python
def _run_child(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return completed


def test_installed_wheel_loaders_entry_points_and_assets_are_immutable(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    state_root.mkdir(mode=0o700)
    run_root.mkdir()
    _install_wheel(built_distributions, target)
    env = _private_child_env(state_root, target)
    before = _target_hashes(target)
    results = [
        _run_child([sys.executable, "-c", INSTALLED_PROBE], run_root, env)
    ]

    script_path = os.pathsep.join(
        str(path) for path in (target / "bin", target / "Scripts")
    )
    for name in ("tldw-cli", "tldw-serve"):
        script = shutil.which(name, path=script_path)
        assert script is not None, (
            f"missing installed script {name!r}; "
            f"target files: {sorted(_target_hashes(target))}"
        )
        results.append(_run_child([script, "--help"], run_root, env))

    after = _target_hashes(target)
    process_text = "\n".join(
        result.stdout + "\n" + result.stderr for result in results
    )
    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in state_root.rglob("*.log*")
        if path.is_file()
    )
    observed_text = process_text + "\n" + log_text
    for forbidden in (
        "Building modular CSS",
        "Failed to build modular CSS",
        "Error handling CSS file",
    ):
        assert forbidden not in observed_text
    assert after == before
```

- [ ] **Step 4: Run the installed regression to verify RED**

Run:

```bash
python -m pytest \
  Tests/Packaging/test_installed_distribution.py \
  -q -p no:cacheprovider \
  -k "installed"
```

Expected: FAIL on the current `tldw-cli --help` CSS rebuild/failure signal. The
package origin and packaged loader assertions must already pass; otherwise fix
Task 1/2 rather than weakening the installed assertion.

- [ ] **Step 5: Add fast RED tests for the source-tree decision**

In `Tests/Web_Server/test_web_server_dependency_gate.py`, add:

```python
from pathlib import Path

import pytest


def test_source_tree_requires_adjacent_pyproject(tmp_path):
    from tldw_chatbook import app as app_module

    package_root = tmp_path / "checkout" / "tldw_chatbook"
    package_root.mkdir(parents=True)
    assert app_module._is_source_tree(package_root) is False

    (package_root.parent / "pyproject.toml").write_text("", encoding="utf-8")
    assert app_module._is_source_tree(package_root) is True
```

Change `test_main_cli_runner_serve_uses_web_dependency_gate()` so its fake
installed package has `css/build_css.py` but no bundle and no adjacent
`pyproject.toml`. Patch `subprocess.run` to raise if called. The test must still
reach the mocked web-server call.

Add:

```python
def test_get_app_does_not_build_css_outside_source_tree(monkeypatch, tmp_path):
    import subprocess
    from tldw_chatbook import app as app_module

    package_root = tmp_path / "installed" / "tldw_chatbook"
    css_dir = package_root / "css"
    css_dir.mkdir(parents=True)
    (css_dir / "build_css.py").write_text("", encoding="utf-8")
    expected = object()

    monkeypatch.setattr(app_module, "__file__", str(package_root / "app.py"))
    monkeypatch.setattr(app_module, "TldwCli", lambda: expected)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("installed CSS build attempted"),
    )

    assert app_module.get_app() is expected
```

Add one narrow source-seam assertion that the predicate is used at the three
known CSS bootstrap sites. This protects the `python -m tldw_chatbook.app` path
that `Web_Server.serve.create_server()` launches without starting a server in
the unit test:

```python
def test_all_css_bootstrap_sites_use_source_tree_guard() -> None:
    from tldw_chatbook import app as app_module

    source = Path(app_module.__file__).read_text(encoding="utf-8")
    assert source.count("if _is_source_tree(package_root):") == 3
```

Run the three tests. Expected: import failure for `_is_source_tree`, or an
installed CSS build attempt from the existing code.

- [ ] **Step 6: Add one source-tree predicate and guard all three sites**

Before the direct execution block in `app.py`, add:

```python
def _is_source_tree(package_root: Path) -> bool:
    """Return whether package files are inside a build-capable source tree."""
    return (package_root.parent / "pyproject.toml").is_file()
```

At direct `app.py` execution, `get_app()`, and `main_cli_runner()`:

1. compute `package_root = Path(__file__).parent`;
2. call `_is_source_tree(package_root)` before `css_dir.mkdir()`, stat/glob
   scans, or `subprocess.run()`; and
3. retain each path's current source behavior and error handling inside the
   guard.

Do not move CLI parsing, logging, signal handling, metrics startup, or
application state. Do not package `stats_screen.css` in the wheel as a fallback.

- [ ] **Step 7: Run unit and installed tests to GREEN**

Run:

```bash
python -m pytest \
  Tests/Web_Server/test_web_server_dependency_gate.py \
  Tests/Packaging/test_installed_distribution.py \
  Tests/Local_Ingestion/test_ingest_spawn_bootstrap.py \
  -q -p no:cacheprovider
```

Expected: all selected tests pass; installed origin is under the target; no CSS
build signal is present; the before/after target hashes are identical.

- [ ] **Step 8: Commit Task 3**

```bash
git add \
  tldw_chatbook/app.py \
  Tests/Web_Server/test_web_server_dependency_gate.py \
  Tests/Packaging/test_installed_distribution.py
git commit -m "fix(packaging): keep installed application assets immutable"
```

---

### Task 4: Document the release gate and run final branch verification

**Files:**

- Modify: `Packaging/PACKAGING_CHECKLIST.md`
- Modify:
  `backlog/tasks/task-545 - Verify-installed-distributions-and-immutable-packaged-assets.md`
- Modify:
  `Docs/superpowers/specs/2026-07-24-installed-distribution-integrity-design.md`

- [ ] **Step 1: Correct the packaging checklist**

Document:

- root `MANIFEST.in` is canonical;
- wheel package data is explicit and sdist-only files stay out;
- SPDX license metadata and vendored notices are checked;
- builds must start from fresh output;
- `python Packaging/check_manifest.py <dist-dir>` verifies exactly one sdist
  and wheel; and
- the installed regression command is:

```bash
python -m pytest \
  Tests/Packaging/test_installed_distribution.py \
  -m integration -q -p no:cacheprovider
```

Correct the stale local smoke command from `tldw-chatbook` to `tldw-cli`. Do not
add upload or release automation.

- [ ] **Step 2: Run focused packaging and startup gates**

```bash
python -m pytest \
  Tests/Packaging/test_installed_distribution.py \
  Tests/Web_Server/test_web_server_dependency_gate.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py \
  Tests/UI/test_css_build_integrity.py \
  Tests/CI/test_github_actions_test_workflow.py \
  Tests/Local_Ingestion/test_ingest_spawn_bootstrap.py \
  -q -p no:cacheprovider
```

Record exact passed/skipped/warning counts.

- [ ] **Step 3: Re-run the completed eval/tool cross-task sentinels**

```bash
python -m pytest Tests/Evals -q -p no:cacheprovider
python -m pytest \
  Tests/Tools \
  Tests/Agents/test_mcp_tool_provider.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/Chat/test_console_agent_swap.py \
  -q -p no:cacheprovider
```

These are regression gates only. Do not broaden TASK-545 into eval/tool changes.

- [ ] **Step 4: Run static verification**

Run Ruff only on changed Python:

```bash
python -m ruff check \
  Packaging/check_manifest.py \
  tldw_chatbook/app.py \
  Tests/Packaging/test_installed_distribution.py \
  Tests/Web_Server/test_web_server_dependency_gate.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py \
  Tests/CI/test_github_actions_test_workflow.py
```

Compile changed Python:

```bash
python -m py_compile \
  Packaging/check_manifest.py \
  tldw_chatbook/app.py \
  Tests/Packaging/test_installed_distribution.py \
  Tests/Web_Server/test_web_server_dependency_gate.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py \
  Tests/CI/test_github_actions_test_workflow.py
```

Then run:

```bash
git diff --check
```

- [ ] **Step 5: Commit release documentation before the final artifact build**

```bash
git add Packaging/PACKAGING_CHECKLIST.md
git commit -m "docs(packaging): document installed artifact gate"
```

- [ ] **Step 6: Build and check from the committed source state**

Commit implementation/doc changes before this step. Create a new directory
with `mktemp -d` under `/private/tmp`, archive `HEAD` into it with `git archive`,
and run:

```bash
python -m build --sdist --wheel --no-isolation
python Packaging/check_manifest.py dist
```

from the extracted committed tree. Require both commands to exit 0. Inspect the
new wheel, not any repository `dist/`.

- [ ] **Step 7: Self-review every TASK-545 acceptance criterion**

Verify with fresh evidence:

1. one fresh sdist and wheel pass required/forbidden checks;
2. exact runtime data and three licenses are present;
3. installed imports and loaders resolve under the target;
4. both installed help commands succeed with private state;
5. entry points and the app factory leave all target hashes unchanged;
6. integration tests use neither checkout imports nor dependency resolution;
7. metadata is Core Metadata 2.4 with the SPDX license expression and license
   file.

Inspect `git diff` and `git status`. Preserve all unrelated `.superpowers/sdd/`
scratch files.

- [ ] **Step 8: Reconcile Backlog and design documentation**

Only after every gate is green:

- change every TASK-545 acceptance checkbox to `[x]`;
- add concise `## Implementation Notes` containing the approach, tradeoffs,
  changed files, ADR-025 link, and exact test evidence;
- set TASK-545 to Done through the Backlog CLI; and
- update the design status to Implemented with a link to this plan.

- [ ] **Step 9: Commit closeout documentation**

```bash
git add \
  Packaging/PACKAGING_CHECKLIST.md \
  Docs/superpowers/specs/2026-07-24-installed-distribution-integrity-design.md \
  'backlog/tasks/task-545 - Verify-installed-distributions-and-immutable-packaged-assets.md'
git commit -m "docs(packaging): record installed distribution verification"
```

Do not begin application-state decomposition in this commit or this task.
