# TASK-15674 Effective Config Startup Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close TASK-15674 with a real app lifecycle regression proving approved
quit persistence honors `TLDW_CONFIG_PATH`, while correcting the earlier
unconfirmed cross-profile attribution.

**Architecture:** Keep production unchanged. A fresh Python subprocess receives a
fully disposable environment before importing `tldw_chatbook`, mounts the real
`TldwCli`, and drives its approved quit path. The parent test owns all before/after
file comparisons. Documentation records the distinction between observed
fingerprint drift and proven writer identity.

**Tech Stack:** Python 3.12, pytest, Textual `run_test`, subprocess, TOML config,
Backlog.md CLI.

**ADR required:** no

**ADR path:** N/A

**Reason:** Regression-only characterization of the existing effective-config
boundary; no new storage, security, runtime, or cross-module decision.

---

### Task 1: Add the real lifecycle regression

**Files:**

- Create: `Tests/ProductionApp/test_config_profile_isolation.py`
- Verify unchanged: `tldw_chatbook/config.py`
- Verify unchanged: `tldw_chatbook/app.py`

- [x] **Step 1: Write the isolated lifecycle test**

Create `Tests/ProductionApp/test_config_profile_isolation.py` with this intended
shape (adjust only for observed Textual lifecycle details):

```python
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PREFIX = "TASK15674_RESULT="
EFFECTIVE_SENTINEL = "PRIVATE_EFFECTIVE_CONFIG_VALUE"
DECOY_SENTINEL = "PRIVATE_DECOY_CONFIG_VALUE"


def _write_sparse_config(path: Path, *, data_dir: Path, marker: str) -> bytes:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    serialized = textwrap.dedent(
        f"""
        [general]
        default_tab = "chat"
        profile_marker = {json.dumps(marker)}

        [first_run]
        setup_started = true
        setup_completed = true

        [splash_screen]
        enabled = false

        [paths]
        data_dir = {json.dumps(str(data_dir))}

        [model_catalog]
        auto_refresh_enabled = false
        """
    ).lstrip()
    path.write_text(serialized, encoding="utf-8")
    return serialized.encode("utf-8")


def _result_payload(stdout: str) -> dict[str, bool]:
    records = [
        line.removeprefix(RESULT_PREFIX)
        for line in stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    assert len(records) == 1, "isolated lifecycle emitted no unique result record"
    return json.loads(records[0])


def test_real_app_quit_persists_only_the_effective_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    xdg_config = tmp_path / "xdg-config"
    xdg_data = tmp_path / "xdg-data"
    xdg_cache = tmp_path / "xdg-cache"
    temp_dir = tmp_path / "tmp"
    effective_data = tmp_path / "effective-data"
    decoy_data = tmp_path / "decoy-data"
    for directory in (
        home,
        xdg_config,
        xdg_data,
        xdg_cache,
        temp_dir,
        effective_data,
        decoy_data,
    ):
        directory.mkdir(parents=True, mode=0o700)

    effective_config = tmp_path / "effective" / "config.toml"
    decoy_default = home / ".config" / "tldw_cli" / "config.toml"
    _write_sparse_config(
        effective_config,
        data_dir=effective_data,
        marker=EFFECTIVE_SENTINEL,
    )
    decoy_before = _write_sparse_config(
        decoy_default,
        data_dir=decoy_data,
        marker=DECOY_SENTINEL,
    )

    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "TERM", "COLORTERM")
        if key in os.environ
    }
    env.update({
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(xdg_config),
        "XDG_DATA_HOME": str(xdg_data),
        "XDG_CACHE_HOME": str(xdg_cache),
        "TMPDIR": str(temp_dir),
        "TLDW_CONFIG_PATH": str(effective_config),
        "TLDW_TEST_MODE": "1",
        "PYTHONPATH": str(REPO_ROOT),
    })

    snippet = f"""
import asyncio
import json
import os
from pathlib import Path

import tldw_chatbook.config as config_module
import tldw_chatbook.app as app_module

state = {{"mounted": False, "effective_path_selected": False,
         "persistence_called": False, "persistence_succeeded": False}}
real_persist = config_module.persist_cli_config_for_shutdown

def observed_persist():
    state["persistence_called"] = True
    state["effective_path_selected"] = (
        config_module.get_cli_config_path()
        == Path(os.environ["TLDW_CONFIG_PATH"])
    )
    result = real_persist()
    state["persistence_succeeded"] = result is True
    return result

app_module.persist_cli_config_for_shutdown = observed_persist

async def main():
    app = app_module.TldwCli()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        state["mounted"] = True
        await app._confirm_and_quit()

try:
    asyncio.run(main())
except BaseException as error:
    print({RESULT_PREFIX!r} + json.dumps(
        {{"phase": "lifecycle", "error_type": type(error).__name__}},
        sort_keys=True,
    ))
    raise SystemExit(2)
print({RESULT_PREFIX!r} + json.dumps(state, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"isolated lifecycle exited with status {result.returncode}"
    )
    assert _result_payload(result.stdout) == {
        "effective_path_selected": True,
        "mounted": True,
        "persistence_called": True,
        "persistence_succeeded": True,
    }
    assert decoy_default.read_bytes() == decoy_before
    assert effective_config.is_file()
    assert effective_config.read_bytes()
    captured = result.stdout + result.stderr
    assert EFFECTIVE_SENTINEL not in captured
    assert DECOY_SENTINEL not in captured
```

The child inherits only locale/terminal process variables, never credentials,
provider settings, proxy configuration, or application overrides. The test
deliberately does not require `effective_config` bytes to change: a future
idempotent persistence optimization is valid. The explicit selected-path boolean
and sparse decoy make the production path mutation in Step 3 observable.

- [x] **Step 2: Run the unmodified characterization**

Run:

```bash
python -B -m pytest \
  Tests/ProductionApp/test_config_profile_isolation.py::test_real_app_quit_persists_only_the_effective_config \
  -q
```

Expected: PASS on current `dev`. If it fails, stop and diagnose before changing
production; the approved design is based on current behavior already satisfying
the contract.

- [x] **Step 3: Mutation-prove the production boundary**

With `apply_patch`, temporarily replace the selection in
`tldw_chatbook.config._get_effective_config_path()`:

```python
candidate = DEFAULT_CONFIG_PATH
```

Run only the named test from Step 2. Expected: FAIL because the sparse decoy is
selected and rewritten. Restore the original production line exactly:

```python
candidate = Path(override).expanduser() if override else DEFAULT_CONFIG_PATH
```

Re-run the named test. Expected: PASS. Do not stage or commit the mutation.

- [x] **Step 4: Run the focused config controls**

Run:

```bash
python -B -m pytest \
  Tests/ProductionApp/test_config_profile_isolation.py \
  Tests/test_config_private_bootstrap.py::test_default_application_config_directory_is_created_as_0700 \
  Tests/test_config_private_bootstrap.py::test_existing_default_config_directory_is_hardened_before_read \
  Tests/test_config_persistence_owner.py::test_shutdown_persistence_uses_only_effective_path \
  -q
```

Expected: all selected tests pass. Do not expand to full test files or the full
suite.

- [x] **Step 5: Commit the regression**

```bash
git add Tests/ProductionApp/test_config_profile_isolation.py
git diff --cached --check
git commit -m "test: lock effective config lifecycle isolation"
```

---

### Task 2: Correct the historical evidence

**Files:**

- Modify: `Docs/superpowers/qa/2026-08-09-comfyui-h3-console-generation-uat.md`
- Modify: `backlog/tasks/task-3401.14 - UAT-end-to-end-ComfyUI-H3-generation-through-Console.md`
- Modify: `backlog/docs/lessons-live-verification.md`
- Modify: `backlog/tasks/task-15674 - Honor-TLDW_CONFIG_PATH-when-persisting-startup-defaults.md`

- [x] **Step 1: Correct the UAT and parent-task wording**

Preserve the facts that a fingerprint change was observed and the validated
snapshot was restored as a precaution. Replace the unsupported causal claim with
the controlled result: current `dev` persisted only the effective profile during a
fully isolated startup-to-approved-quit lifecycle, while the decoy default stayed
byte-identical.

- [x] **Step 2: Record the generalizable lesson**

Rewrite the TASK-3401.14/TASK-15674 incident in
`backlog/docs/lessons-live-verification.md` to explain that fingerprint drift proves
mutation, not writer identity. Retain the concrete incident: the original restore
was appropriate, but a controlled decoy/effective-profile reproduction did not
reproduce a cross-profile app write.

- [x] **Step 3: Prepare accurate TASK-15674 closeout notes**

Update the task description to remove the reproduced-defect claim. Check all five
acceptance criteria and add concise Implementation Notes containing:

- regression-only approach and no production change;
- approved-quit persistence seam and scratch isolation;
- mutation RED and focused GREEN evidence;
- exact modified files;
- ADR required: no / path: N/A / reason;
- documentation correction and lesson update.

Keep status In Progress until Task 3 verification passes.

---

### Task 3: Verify and close out

**Files:**

- Verify: `Tests/ProductionApp/test_config_profile_isolation.py`
- Modify: `Docs/superpowers/plans/2026-08-12-effective-config-startup-isolation-plan.md`
- Modify: `backlog/tasks/task-15674 - Honor-TLDW_CONFIG_PATH-when-persisting-startup-defaults.md`

- [x] **Step 1: Run touched-file test and static gates**

Run the exact focused pytest command from Task 1, then:

```bash
python -m ruff check \
  Tests/ProductionApp/test_config_profile_isolation.py
```

Compile the new Python test with `py_compile` to a `TemporaryDirectory`; do not
write `.pyc` files into the repository.

- [x] **Step 2: Run repository hygiene checks**

Run `git diff --check`, then verify:

- no config value, credential, private source path, media, build, or cache artifact
  is staged;
- `tldw_chatbook/config.py` and `tldw_chatbook/app.py` have no final diff;
- the exact branch diff satisfies the spec and all five acceptance criteria.

- [x] **Step 3: Mark TASK-15674 Done**

After all gates pass, use Backlog.md CLI to set TASK-15674 to Done with final
Implementation Notes and all five acceptance criteria checked.

- [x] **Step 4: Commit closeout documentation**

Stage only the four documentation/task files plus this plan, verify the cached
diff, and commit:

```bash
git commit -m "docs: close config profile isolation task"
```

- [x] **Step 5: Verify final branch state**

Confirm normal `git status --short` is clean, production code is unchanged from
`origin/dev`, and the final report states focused test counts and warnings without
claiming a broad suite was run.

Final verification (2026-08-12): the isolated branch was rebased without conflicts
onto current `origin/dev` and was zero commits behind. The exact focused selection
passed 4 tests in 6.54 seconds with one dependency-compatibility warning; Ruff and
temporary-output `py_compile` passed. Scope, privacy, artifact, diff, and
production-code checks passed. No broad suite was run.
