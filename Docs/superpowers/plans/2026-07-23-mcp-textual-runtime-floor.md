# MCP Textual Runtime Floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent MCP navigation crashes by declaring, testing, and continuously verifying the Textual 8.x runtime contract.

**Architecture:** Packaging metadata remains the enforcement boundary: both supported installation manifests declare Textual 8.x, while semantic source tests prevent version-range drift. A focused CI lane installs the exact minimum Textual release and mounts the MCP workbench and tools mode so later code cannot silently outgrow the declared floor.

**Tech Stack:** Python 3.11+, PEP 508 requirements, `tomllib`, `packaging`, pytest, Textual 8.x, GitHub Actions

## Global Constraints

- Supported Textual range is exactly `>=8.0.0,<9`.
- `pyproject.toml` is authoritative; `requirements.txt` mirrors the range because documented and CI installation paths use it directly.
- Do not add a `Select.NULL` compatibility alias, MCP exception handler, or application startup guard.
- The minimum-version CI lane must install exactly Textual 8.0.0.
- Preserve unrelated dirty-worktree changes.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/022-textual-8-runtime-floor.md`

Reason: the fix changes dependency and runtime support policy, ends Textual
3–7 support, and fails closed on unreviewed future Textual major versions.

---

### Task 1: Add the red dependency-contract regression

**Files:**
- Create: `Tests/CI/test_textual_runtime_contract.py`
- Modify: `Tests/UI/test_product_maturity_phase6_packaging_data_safety.py:102`

**Interfaces:**
- Consumes: PEP 508 requirement strings from `pyproject.toml` and `requirements.txt`
- Produces: `_textual_requirement(entries: Iterable[str]) -> Requirement` and a semantic Textual 8.x contract enforced for both manifests

- [x] **Step 1: Create the semantic dependency test**

Create `Tests/CI/test_textual_runtime_contract.py` with:

```python
from collections.abc import Iterable
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _textual_requirement(entries: Iterable[str]) -> Requirement:
    for entry in entries:
        candidate = entry.split("#", 1)[0].strip()
        if not candidate:
            continue
        requirement = Requirement(candidate)
        if requirement.name.lower() == "textual":
            return requirement
    raise AssertionError("Textual requirement is missing")


def _assert_textual_8_only(requirement: Requirement) -> None:
    assert Version("7.999.999") not in requirement.specifier
    assert Version("8.0.0") in requirement.specifier
    assert Version("8.999.999") in requirement.specifier
    assert Version("9.0.0") not in requirement.specifier


def test_pyproject_supports_only_textual_8_x() -> None:
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    requirement = _textual_requirement(pyproject["project"]["dependencies"])

    _assert_textual_8_only(requirement)


def test_development_requirements_support_only_textual_8_x() -> None:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(
        encoding="utf-8"
    )

    requirement = _textual_requirement(requirements.splitlines())

    _assert_textual_8_only(requirement)
```

- [x] **Step 2: Update the existing packaging seam to the intended contract**

In `Tests/UI/test_product_maturity_phase6_packaging_data_safety.py`, replace:

```python
assert "textual>=3.3.0" in project["dependencies"]
```

with:

```python
assert "textual>=8.0.0,<9" in project["dependencies"]
```

- [x] **Step 3: Run the tests and verify the red state**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/CI/test_textual_runtime_contract.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py::test_phase6_packaging_config_and_data_safety_source_seams_are_present \
  --tb=short
```

Expected: FAIL because `pyproject.toml` still accepts Textual 7 and 9,
`requirements.txt` has an unconstrained `textual` entry, and the existing
packaging seam expects the new declaration.

### Task 2: Correct both installation manifests and document the upgrade

**Files:**
- Modify: `pyproject.toml:41`
- Modify: `requirements.txt:21`
- Modify: `CHANGELOG.md` under `Unreleased` → `Changed`
- Test: `Tests/CI/test_textual_runtime_contract.py`
- Test: `Tests/UI/test_product_maturity_phase6_packaging_data_safety.py`

**Interfaces:**
- Consumes: the red semantic dependency contract from Task 1
- Produces: matching Textual 8.x constraints for package installs and requirements-file installs

- [x] **Step 1: Change the authoritative package dependency**

In `pyproject.toml`, replace:

```toml
"textual>=3.3.0",
```

with:

```toml
"textual>=8.0.0,<9",
```

- [x] **Step 2: Mirror the range in the development requirements**

In `requirements.txt`, replace:

```text
textual
```

with:

```text
textual>=8.0.0,<9
```

- [x] **Step 3: Add the user-facing upgrade note**

Add this bullet under `CHANGELOG.md` → `Unreleased` → `Changed`:

```markdown
- Textual 8.x is now required (`>=8.0.0,<9`). This corrects the previously
  overstated Textual 3.3 compatibility range, which could crash when opening
  MCP because the screen uses Textual 8's `Select.NULL` API. Existing source
  checkouts should reinstall dependencies after pulling this update.
```

- [x] **Step 4: Run the dependency tests and verify green**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/CI/test_textual_runtime_contract.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py::test_phase6_packaging_config_and_data_safety_source_seams_are_present \
  --tb=short
```

Expected: `3 passed`.

- [x] **Step 5: Commit the runtime contract**

```bash
git add \
  pyproject.toml \
  requirements.txt \
  CHANGELOG.md \
  Tests/CI/test_textual_runtime_contract.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py
git commit -m "fix: require Textual 8 for MCP runtime"
```

### Task 3: Add a continuously enforced minimum-version CI lane

**Files:**
- Modify: `Tests/CI/test_github_actions_test_workflow.py`
- Modify: `.github/workflows/test.yml`

**Interfaces:**
- Consumes: the Textual 8.x package contract from Task 2
- Produces: GitHub Actions job `textual-minimum` that installs Textual 8.0.0 and runs the focused dependency and MCP suites

- [x] **Step 1: Write the failing workflow contract test**

Append to `Tests/CI/test_github_actions_test_workflow.py`:

```python
def test_ci_exercises_mcp_against_minimum_textual() -> None:
    workflow = _workflow_text()

    assert "  textual-minimum:" in workflow
    assert 'pip install "textual==8.0.0"' in workflow
    assert "Tests/CI/test_textual_runtime_contract.py" in workflow
    assert "Tests/UI/test_mcp_workbench.py" in workflow
    assert "Tests/UI/test_mcp_tools_mode.py" in workflow
```

- [x] **Step 2: Run the workflow contract and verify the red state**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/CI/test_github_actions_test_workflow.py::test_ci_exercises_mcp_against_minimum_textual \
  --tb=short
```

Expected: FAIL because `.github/workflows/test.yml` has no
`textual-minimum` job.

- [x] **Step 3: Add the minimum-version workflow job**

Insert this job before `all-tests` in `.github/workflows/test.yml`:

```yaml
  textual-minimum:
    name: MCP - Minimum Textual 8.0.0
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.11
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: 'pip'

    - name: Install core and test dependencies at the Textual floor
      run: |
        python -m pip install --upgrade pip setuptools wheel
        pip install -e .
        pip install pytest pytest-asyncio pytest-timeout
        pip install "textual==8.0.0"

    - name: Verify packaging and MCP at the Textual floor
      run: |
        pytest \
          Tests/CI/test_textual_runtime_contract.py \
          Tests/UI/test_mcp_workbench.py \
          Tests/UI/test_mcp_tools_mode.py \
          --timeout=180 \
          --tb=short
```

- [x] **Step 4: Include the lane in aggregate workflow status**

Change the `test-summary` job dependency list from:

```yaml
needs: [unit-tests, integration-tests, ui-tests]
```

to:

```yaml
needs: [unit-tests, integration-tests, ui-tests, textual-minimum]
```

- [x] **Step 5: Run the workflow contract and verify green**

Run:

```bash
.venv/bin/python -m pytest -q Tests/CI/test_github_actions_test_workflow.py --tb=short
```

Expected: all tests in the file PASS.

- [x] **Step 6: Commit the CI floor**

```bash
git add .github/workflows/test.yml Tests/CI/test_github_actions_test_workflow.py
git commit -m "ci: test MCP on minimum Textual"
```

### Task 4: Verify the supported runtime and close TASK-503

**Files:**
- Modify: `backlog/tasks/task-503 - Fix-MCP-navigation-crash-by-requiring-Textual-8.md` through the Backlog CLI
- Verify: `Docs/superpowers/specs/2026-07-23-mcp-textual-runtime-floor-design.md`
- Verify: `backlog/decisions/022-textual-8-runtime-floor.md`

**Interfaces:**
- Consumes: the corrected manifests and minimum-version CI lane
- Produces: verification evidence and complete Backlog implementation notes

- [x] **Step 1: Run focused tests on the normal environment**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/CI/test_textual_runtime_contract.py \
  Tests/CI/test_github_actions_test_workflow.py \
  Tests/UI/test_product_maturity_phase6_packaging_data_safety.py \
  Tests/UI/test_mcp_workbench.py \
  Tests/UI/test_mcp_tools_mode.py \
  --tb=short
```

Expected: all selected tests PASS. On the rebased `dev` suite, nine existing
`MCPWorkbench._clear_tool_view` unawaited-coroutine warnings may remain.

- [x] **Step 2: Replay the MCP suites against exactly Textual 8.0.0**

Run:

```bash
minimum_textual_dir="$(mktemp -d /private/tmp/tldw-textual-minimum.XXXXXX)"
.venv/bin/python -m pip install --target "$minimum_textual_dir" textual==8.0.0
PYTHONPATH="$minimum_textual_dir" .venv/bin/python -m pytest -q \
  Tests/CI/test_textual_runtime_contract.py \
  Tests/UI/test_mcp_workbench.py \
  Tests/UI/test_mcp_tools_mode.py \
  --tb=short
```

Expected after the `dev` rebase: all 209 tests PASS; the same nine pre-existing
coroutine warnings may remain.

- [x] **Step 3: Run static hygiene**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [x] **Step 4: Update TASK-503 acceptance criteria and notes**

Use `backlog task edit 503 --check-ac` for all five acceptance criteria, set
implementation notes summarizing the bounded Textual range, semantic
dual-manifest test, minimum-version CI lane, changelog, ADR-022, and exact-floor
test evidence, then set the task to `Done`.

- [x] **Step 5: Commit closeout records**

```bash
git add \
  Docs/superpowers/specs/2026-07-23-mcp-textual-runtime-floor-design.md \
  Docs/superpowers/plans/2026-07-23-mcp-textual-runtime-floor.md \
  backlog/decisions/022-textual-8-runtime-floor.md \
  backlog/decisions/README.md \
  "backlog/tasks/task-503 - Fix-MCP-navigation-crash-by-requiring-Textual-8.md"
git commit -m "docs: record Textual runtime decision"
```

## Self-Review

- Spec coverage: the plan covers the bounded dependency declarations,
  dual-manifest regression, stale packaging assertion, exact-floor CI lane,
  changelog notice, normal-runtime tests, exact-floor tests, ADR, and Backlog
  closeout.
- Placeholder scan: no deferred implementation steps or undefined code remain.
- Type consistency: `_textual_requirement` consumes both manifest entry
  iterables and always returns `packaging.requirements.Requirement`; both tests
  use the same semantic range assertions.
