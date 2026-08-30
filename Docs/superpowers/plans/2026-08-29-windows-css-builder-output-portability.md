# Windows CSS Builder Output Portability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the complete CSS builder run through strict CP1252 standard output without changing generated CSS or substantive failure behavior.

**Architecture:** Keep output policy local to the builder by replacing every direct decorative/dynamic-path print with ASCII literals and numeric counts. Adapt the existing end-to-end scratch-tree test to enforce CP1252 at the real `main()` entry point, so no process-wide encoding mutation or new console abstraction is needed.

**Tech Stack:** Python 3.11+ standard library (`io`, `sys`), pytest, existing CSS manifest builder

**Backlog:** `TASK-24531`

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a portable presentation fix inside an existing build script; dependency, artifact, and staleness contracts remain unchanged.

---

## File Map

- Modify `tldw_chatbook/css/build_css.py`: emit ASCII-only phase progress and completion output without interpolating paths or module names.
- Modify `Tests/UI/test_css_staleness_manifest.py`: enforce strict CP1252 around the real builder entry path while retaining manifest/staleness checks.
- Modify `backlog/tasks/task-24531 - Make-CSS-builder-output-CP1252-safe.md`: record evidence and close the task after focused verification.

No generated stylesheet, dependency, output wrapper, environment variable, or Windows code-page setting is changed.

### Task 1: Make the existing builder integration test reproduce strict CP1252 failure

**Files:**
- Modify: `Tests/UI/test_css_staleness_manifest.py:1-25, 282-340`

- [ ] **Step 1: Add only the standard-library imports needed by the existing integration test**

Add `io` and `sys` in the standard-library import group:

```python
import io
import sys
```

- [ ] **Step 2: Convert the existing scratch tree into a non-CP1252 path and distinctive real source**

Inside `TestBuilderIntegration.test_main_end_to_end_manifest_and_staleness`, replace the package root and source setup with:

```python
package = tmp_path / "checkout-漢" / "tldw_chatbook"
css_dir = package / "css"
(css_dir / "core").mkdir(parents=True)
distinctive_rule = "Screen { color: #123456; }"
(css_dir / "core" / "_base.tcss").write_text(
    distinctive_rule + "\n",
    encoding="utf-8",
)
```

The source itself remains ASCII; the non-representable character exists only in the valid checkout path, proving progress output does not leak dynamic paths.

- [ ] **Step 3: Replace the print no-op with a strict stream around the real entry point**

Delete:

```python
monkeypatch.setattr("builtins.print", lambda *a, **k: None)
```

and run the existing `bc.main()` call through an encoding-enforcing wrapper:

```python
captured_bytes = io.BytesIO()
strict_stdout = io.TextIOWrapper(
    captured_bytes,
    encoding="cp1252",
    errors="strict",
    write_through=True,
)
monkeypatch.setattr(sys, "stdout", strict_stdout)

bc.main()
strict_stdout.flush()
output = captured_bytes.getvalue().decode("cp1252")
```

Do not call `reconfigure`, set `PYTHONIOENCODING`, or replace `builtins.print`; the test must exercise the production print calls.

- [ ] **Step 4: Assert all four phases, generated artifacts, and source content**

Immediately after the build, assert non-vacuous output:

```python
assert "Processing CSS module 1 of 1" in output
assert "CSS build complete" in output
assert "Widget defaults build complete" in output
assert "Screen CSS build complete" in output
assert "checkout-" not in output
```

Retain the existing five generated-file and manifest assertions, then add:

```python
bundle = (css_dir / "tldw_cli_modular.tcss").read_text(encoding="utf-8")
assert distinctive_rule in bundle
```

Keep the unchanged-content mtime and real-edit staleness assertions exactly as existing outcome controls.

- [ ] **Step 5: Run the integration test and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_css_staleness_manifest.py::TestBuilderIntegration::test_main_end_to_end_manifest_and_staleness
```

Expected before the fix: FAIL with `UnicodeEncodeError` on the first checkmark or, if that is locally bypassed, on a later emoji or non-CP1252 output path.

### Task 2: Replace every direct builder print with ASCII-safe bounded output

**Files:**
- Modify: `tldw_chatbook/css/build_css.py:290-320, 360-370, 455-465, 500-510`

- [ ] **Step 1: Make module progress numeric and path-free**

Change the module loop to:

```python
for index, module in enumerate(CSS_MODULES, start=1):
    print(f"Processing CSS module {index} of {len(CSS_MODULES)}")
    content = (css_dir / module).read_text(encoding="utf-8")
```

Continue using `module` internally for reads and generated separators; only direct output becomes bounded ASCII.

- [ ] **Step 2: Make each completion block ASCII-only and omit output paths**

Use these exact phase markers and numeric summaries:

```python
print("CSS build complete")
print(f"Total size: {len(''.join(combined_css)):,} characters")
```

```python
print("Widget defaults build complete")
print(
    f"Widget defaults: {len(blocks)} classes, "
    f"{len(own):,} + {len(scoped):,} characters"
)
```

```python
print("Screen CSS build complete")
print(
    f"Screen CSS: {len(blocks)} classes, "
    f"{len(own):,} + {len(scoped):,} characters"
)
```

The existing final usage instructions are already ASCII and remain unchanged.

- [ ] **Step 3: Prove every direct print literal is ASCII**

Run:

```bash
../../.venv/bin/python - <<'PY'
import ast
from pathlib import Path

path = Path("tldw_chatbook/css/build_css.py")
tree = ast.parse(path.read_text(encoding="utf-8"))
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
        for part in node.args:
            for literal in ast.walk(part):
                if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                    literal.value.encode("ascii")
PY
```

Expected: exit 0. This is a scoped implementation check; the integration test remains the behavioral proof.

- [ ] **Step 4: Run the strict CP1252 integration test and verify GREEN**

Run the same single-test command from Task 1 Step 5.

Expected: 1 passed; captured output includes every phase marker and the generated bundle contains the distinctive source rule.

- [ ] **Step 5: Commit the portability fix**

```bash
git add -- tldw_chatbook/css/build_css.py Tests/UI/test_css_staleness_manifest.py
git diff --cached --check
git commit -m "fix: make CSS builder output encoding-safe"
```

### Task 3: Verify fail-loud and artifact semantics, then close `TASK-24531`

**Files:**
- Test: `Tests/UI/test_css_staleness_manifest.py`
- Test: `Tests/UI/test_css_build_integrity.py`
- Test: `Tests/UI/test_css_bundle_sync_guard.py`
- Test: `Tests/UI/test_widget_css_consolidation.py`
- Modify: `backlog/tasks/task-24531 - Make-CSS-builder-output-CP1252-safe.md`

- [ ] **Step 1: Run the focused manifest/staleness suite**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_css_staleness_manifest.py
```

Expected: all tests pass, including missing inputs, malformed manifests, build races, unchanged-content mtime, and real-edit staleness.

- [ ] **Step 2: Run the focused generated-CSS integrity controls**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_css_build_integrity.py \
  Tests/UI/test_css_bundle_sync_guard.py \
  Tests/UI/test_widget_css_consolidation.py
```

These are the existing modules that directly pin modular bundle completeness/reproducibility plus generated widget/screen sheet production. Do not run unrelated visual/UI suites and do not run the full repository suite without explicit user opt-in.

Expected: all selected tests pass and no checked-in stylesheet content changes solely because output copy changed.

- [ ] **Step 3: Run scoped lint, compilation, and whitespace checks**

Run:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/css/build_css.py \
  Tests/UI/test_css_staleness_manifest.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/css/build_css.py \
  Tests/UI/test_css_staleness_manifest.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/css/build_css.py \
  Tests/UI/test_css_staleness_manifest.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 4: Self-review artifact neutrality**

Run:

```bash
git diff --name-only
git diff -- tldw_chatbook/css/build_css.py Tests/UI/test_css_staleness_manifest.py
```

Confirm no generated `.tcss` file changed, output contains no module/path interpolation, and all filesystem/race exceptions still propagate.

- [ ] **Step 5: Complete and verify the Backlog record**

Check every acceptance criterion, add concise Implementation Notes with commands/results and `ADR required: no`, then run:

```bash
backlog task edit 24531 -s Done
backlog task 24531 --plain
```

Verify the CLI reports the expected `task-24531` file path and all criteria are checked before committing.

- [ ] **Step 6: Commit task closeout**

```bash
git add -- "backlog/tasks/task-24531 - Make-CSS-builder-output-CP1252-safe.md"
git diff --cached --check
git commit -m "docs: close CSS builder portability task"
```
