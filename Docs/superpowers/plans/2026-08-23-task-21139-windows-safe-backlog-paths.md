# TASK-21139 Windows-Safe Backlog Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore Windows checkout by renaming TASK-21130 without changing its contents and make the existing stdlib Backlog guard reject future Windows-incompatible task filenames.

**Architecture:** Keep the policy in `scripts/check_backlog_task_ids.py`, the existing owner of all three Backlog task buckets and all current local/CI entry points. Add one pure basename classifier and one non-recursive bucket scanner, compose their result with the unchanged duplicate-ID result in `main()`, and repair only the incompatible TASK-21130 path.

**Tech Stack:** Python 3.11+ standard library, pytest, Ruff, Backlog.md task files, GitHub Actions, Git/GitHub CLI

---

## File map

- Modify `scripts/check_backlog_task_ids.py`: define the Windows basename policy, scan the existing task buckets, report every incompatible path, and combine that result with duplicate-ID validation.
- Modify `Tests/Architecture/test_derived_artifact_checkers.py`: focused unit and CLI-composition coverage for the classifier and scanner.
- Rename `backlog/tasks/task-21130 - TTS profile v3->v4 migration snapshots the entire reference-BLOB table into memory twice.md` to `backlog/tasks/task-21130 - TTS profile v3-to-v4 migration snapshots the entire reference-BLOB table into memory twice.md`: remove the checkout blocker while preserving task bytes.
- Modify `backlog/docs/lessons-backlog-hygiene.md`: record the concrete Windows-checkout incident and the new authoring-time guard.
- Modify `backlog/tasks/task-21139 - Restore Windows checkout for Backlog task paths.md`: track this plan, checked acceptance criteria, implementation notes, and final status.
- Create `Docs/superpowers/plans/2026-08-23-task-21139-windows-safe-backlog-paths.md`: this executable plan.
- No workflow file changes: `scripts/preflight.sh`, `.github/workflows/backlog-guard.yml`, and `.github/workflows/derived-artifacts.yml` already invoke the shared checker.

## ADR check

ADR required: no

ADR path: N/A

Reason: this is a focused CI-portability repair within an existing stdlib guard. It does not alter storage, schemas, runtime ownership, security boundaries, dependencies, or cross-module contracts.

## Execution constraints

- Follow `@superpowers:test-driven-development` for behavior changes and `@ponytail` full: add only the two small functions and reporting composition required by the approved design.
- Run only tests and static checks related to the modified checker/workflows. Do not run the broad local suite.
- If a scoped local or CI test fails for a reason unrelated to this task, capture the evidence and create a Backlog task; do not broaden TASK-21139 to repair it.
- Preserve `duplicate_ids()` and its return contract exactly.
- Use `apply_patch` for every repository edit, including the filename move.

### Task 1: Pin the Windows basename contract with failing focused tests

**Files:**
- Modify: `Tests/Architecture/test_derived_artifact_checkers.py:12-14,337-403`
- Test: `Tests/Architecture/test_derived_artifact_checkers.py`

- [ ] **Step 1: Add pytest for compact table-driven policy coverage**

Add the existing test dependency beside the current standard-library imports:

```python
import pytest
```

- [ ] **Step 2: Add a failing test for every forbidden-character class**

Add direct pure-classifier tests so `/` and NUL are testable even though POSIX cannot create basenames containing them:

```python
@pytest.mark.parametrize("character", '<>:"/\\|?*')
def test_windows_task_name_rejects_every_reserved_character(character):
    reason = backlog_ids.windows_incompatible_reason(f"task-1 - Bad{character}Name.md")
    assert reason == f"contains Windows-reserved character {character!r}"


@pytest.mark.parametrize("character", ["\x00", "\x01", "\x1f"])
def test_windows_task_name_rejects_ascii_controls(character):
    reason = backlog_ids.windows_incompatible_reason(f"task-1 - Bad{character}Name.md")
    assert reason == f"contains ASCII control U+{ord(character):04X}"
```

- [ ] **Step 3: Add failing tests for trailing characters and device stems**

Cover case-insensitivity, every device family, superscript aliases, the first-period rule, and valid boundaries:

```python
@pytest.mark.parametrize("name", ["task-1 - Bad.md.", "task-1 - Bad.md "])
def test_windows_task_name_rejects_trailing_dot_or_space(name):
    assert backlog_ids.windows_incompatible_reason(name) == "ends with a dot or space"


@pytest.mark.parametrize(
    "name, stem",
    [
        ("CON", "CON"), ("prn.md", "PRN"), ("Aux.notes.md", "AUX"),
        ("NUL.tar.gz", "NUL"), ("com1.md", "COM1"), ("COM9.log", "COM9"),
        ("com¹.log", "COM¹"), ("COM².log", "COM²"),
        ("Com³.tar.gz", "COM³"),
        ("lpt1.md", "LPT1"), ("LPT9.log", "LPT9"),
        ("lpt¹.log", "LPT¹"), ("LPT².log", "LPT²"),
        ("Lpt³.tar.gz", "LPT³"),
    ],
)
def test_windows_task_name_rejects_reserved_device_stems(name, stem):
    assert backlog_ids.windows_incompatible_reason(name) == (
        f"uses reserved Windows device name {stem!r}"
    )


@pytest.mark.parametrize(
    "name",
    [
        "task-1 - Ordinary title.md", "task-2 - Version v3-to-v4 (safe).md",
        "task-3 - CON appears after the task prefix.md", "COM0.md", "COM10.md",
        "COM⁴.md", "LPT0.md", "LPT10.md", "LPT⁴.md",
    ],
)
def test_windows_task_name_accepts_valid_names(name):
    assert backlog_ids.windows_incompatible_reason(name) is None
```

- [ ] **Step 4: Add a failing scanner test using POSIX-creatable invalid files**

Exercise direct bucket contents, absent optional buckets, stable full-path labels outside the repository, and directory exclusion:

```python
def test_windows_incompatible_paths_scan_files_in_each_existing_bucket(tmp_path):
    tasks = tmp_path / "tasks"
    completed = tmp_path / "completed"
    archive = tmp_path / "archive" / "tasks"
    tasks.mkdir()
    completed.mkdir()
    archive.mkdir(parents=True)
    invalid = {
        tasks / "task-1 - Bad>Name.md": "contains Windows-reserved character '>'",
        completed / "task-2 - Control\x1f.md": "contains ASCII control U+001F",
        archive / "NUL.tar.gz": "uses reserved Windows device name 'NUL'",
        archive / "task-3 - Trailing.md ": "ends with a dot or space",
    }
    for path in invalid:
        path.write_text("id: TASK-1\n", encoding="utf-8")
    (tasks / "task-4 - Safe.md").write_text("id: TASK-4\n", encoding="utf-8")
    (tasks / "task-5 - Directory?.md").mkdir()

    assert backlog_ids.windows_incompatible_paths(
        tasks, completed, archive, tmp_path / "absent"
    ) == {path.resolve().as_posix(): reason for path, reason in invalid.items()}
```

- [ ] **Step 5: Add a failing CLI-composition test**

Prove one invocation reports duplicate IDs and incompatible paths rather than short-circuiting after the first problem:

```python
def test_main_reports_duplicate_ids_and_windows_paths_together(tmp_path, capsys):
    first = tmp_path / "task-42 - First?.md"
    second = tmp_path / "task-42 - Second.md"
    first.write_text("id: TASK-42\n", encoding="utf-8")
    second.write_text("id: TASK-42\n", encoding="utf-8")

    assert backlog_ids.main(["--tasks-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    report = captured.out + captured.err
    assert "Duplicate backlog task IDs" in report
    assert "Windows-incompatible Backlog task paths" in report
    assert first.resolve().as_posix() in report
    assert "Keep punctuation in task content" in report
```

- [ ] **Step 6: Run only the new tests and verify RED for the missing API**

Run:

```bash
pytest -q \
  Tests/Architecture/test_derived_artifact_checkers.py::test_windows_task_name_rejects_every_reserved_character \
  Tests/Architecture/test_derived_artifact_checkers.py::test_windows_task_name_rejects_ascii_controls \
  Tests/Architecture/test_derived_artifact_checkers.py::test_windows_task_name_rejects_trailing_dot_or_space \
  Tests/Architecture/test_derived_artifact_checkers.py::test_windows_task_name_rejects_reserved_device_stems \
  Tests/Architecture/test_derived_artifact_checkers.py::test_windows_task_name_accepts_valid_names \
  Tests/Architecture/test_derived_artifact_checkers.py::test_windows_incompatible_paths_scan_files_in_each_existing_bucket \
  Tests/Architecture/test_derived_artifact_checkers.py::test_main_reports_duplicate_ids_and_windows_paths_together
```

Expected: FAIL because `windows_incompatible_reason` and `windows_incompatible_paths` do not exist. Confirm the failure is the intended missing-behavior failure, not fixture or collection breakage.

- [ ] **Step 7: Commit the RED tests**

```bash
git add Tests/Architecture/test_derived_artifact_checkers.py
git commit -m "test(backlog): pin Windows-safe task filenames"
```

### Task 2: Implement the minimal stdlib classifier and scanner

**Files:**
- Modify: `scripts/check_backlog_task_ids.py:1-55,91-191`
- Test: `Tests/Architecture/test_derived_artifact_checkers.py`

- [ ] **Step 1: Extend the module contract and define the policy constants**

Update the module docstring to cover duplicate IDs and Windows-representable paths. Add only these constants after the existing regexes:

```python
WINDOWS_RESERVED_CHARACTERS = frozenset('<>:"/\\|?*')
WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{suffix}" for suffix in (*map(str, range(1, 10)), "¹", "²", "³")}
    | {f"lpt{suffix}" for suffix in (*map(str, range(1, 10)), "¹", "²", "³")}
)
WINDOWS_PATH_RESOLUTION = (
    "Keep punctuation in task content, but rename the task file with a "
    "Windows-safe spelling and update live path references."
)
```

- [ ] **Step 2: Add the pure basename classifier**

Place it before `duplicate_ids()` so scanner and tests share one source of truth:

```python
def windows_incompatible_reason(name: str) -> str | None:
    """Return why a basename cannot be represented by Win32, else ``None``."""
    for character in name:
        if ord(character) <= 0x1F:
            return f"contains ASCII control U+{ord(character):04X}"
        if character in WINDOWS_RESERVED_CHARACTERS:
            return f"contains Windows-reserved character {character!r}"
    if name.endswith((".", " ")):
        return "ends with a dot or space"
    device_stem = name.split(".", 1)[0].casefold()
    if device_stem in WINDOWS_RESERVED_DEVICE_NAMES:
        return f"uses reserved Windows device name {device_stem.upper()!r}"
    return None
```

- [ ] **Step 3: Add the non-recursive bucket scanner**

Use `iterdir()` rather than `glob("*.md")` so invalid extensions/trailing characters cannot hide from the portability guard. Keep directories out of scope:

```python
def windows_incompatible_paths(*task_dirs: Path) -> dict[str, str]:
    """Return directly contained files whose basenames Win32 rejects."""
    invalid: dict[str, str] = {}
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        for path in sorted(task_dir.iterdir()):
            if not path.is_file():
                continue
            reason = windows_incompatible_reason(path.name)
            if reason:
                invalid[_label(path)] = reason
    return invalid
```

- [ ] **Step 4: Compose path reporting with duplicate-ID reporting**

Add a focused reporter and update `main()` without altering `duplicate_ids()`:

```python
def _report_windows_incompatible(invalid: dict[str, str]) -> None:
    print("::error::Windows-incompatible Backlog task paths:")
    for path, reason in sorted(invalid.items()):
        print(f"  {path}: {reason}")


# Inside main(), after duplicate_ids(...):
invalid_paths = windows_incompatible_paths(*task_dirs)
# Report filename_dupes, frontmatter_dupes, and invalid_paths independently.
# Print RESOLUTION only when duplicate IDs exist.
# Print WINDOWS_PATH_RESOLUTION only when invalid_paths exists.
# Return 1 when any of the three collections is non-empty.
```

Update `main()`'s docstring and success text to say IDs are unique and paths are Windows-compatible.

- [ ] **Step 5: Run the new tests and verify GREEN**

Run the exact seven-node command from Task 1, Step 6.

Expected: all selected tests PASS.

- [ ] **Step 6: Run the existing Backlog checker regression tests**

Run:

```bash
pytest -q \
  Tests/Architecture/test_derived_artifact_checkers.py::test_duplicate_task_ids_are_caught_in_both_namespaces \
  Tests/Architecture/test_derived_artifact_checkers.py::test_duplicate_task_ids_are_caught_across_buckets \
  Tests/Architecture/test_derived_artifact_checkers.py::test_default_scope_is_every_bucket_the_cli_resolves \
  Tests/Architecture/test_derived_artifact_checkers.py::test_an_absent_optional_bucket_is_not_an_error \
  Tests/Architecture/test_derived_artifact_checkers.py::test_unique_task_ids_pass
```

Expected: 5 passed; `duplicate_ids()` behavior remains unchanged.

- [ ] **Step 7: Demonstrate the real repository now fails for the intended path**

Run:

```bash
python3 scripts/check_backlog_task_ids.py
```

Expected: exit 1 and output naming TASK-21130's `v3->v4` path with `contains Windows-reserved character '>'`. This is the integration RED before the repair.

- [ ] **Step 8: Commit the minimal guard**

```bash
git add scripts/check_backlog_task_ids.py
git commit -m "fix(backlog): reject Windows-incompatible task paths"
```

### Task 3: Repair TASK-21130 without changing task content

**Files:**
- Rename: `backlog/tasks/task-21130 - TTS profile v3->v4 migration snapshots the entire reference-BLOB table into memory twice.md` -> `backlog/tasks/task-21130 - TTS profile v3-to-v4 migration snapshots the entire reference-BLOB table into memory twice.md`
- Verify: repository references to the old path

- [ ] **Step 1: Record the source task's byte hash and identity**

Run:

```bash
shasum -a 256 'backlog/tasks/task-21130 - TTS profile v3->v4 migration snapshots the entire reference-BLOB table into memory twice.md'
sed -n '1,45p' 'backlog/tasks/task-21130 - TTS profile v3->v4 migration snapshots the entire reference-BLOB table into memory twice.md'
```

Expected: record the SHA-256 and confirm frontmatter `id: TASK-21130`. Do not edit any bytes inside the task.

- [ ] **Step 2: Move the file with apply_patch**

Use an `apply_patch` move from the old path to the new `v3-to-v4` path with no content edits.

- [ ] **Step 3: Prove the task bytes and identity are unchanged**

Run:

```bash
shasum -a 256 'backlog/tasks/task-21130 - TTS profile v3-to-v4 migration snapshots the entire reference-BLOB table into memory twice.md'
git add -- \
  'backlog/tasks/task-21130 - TTS profile v3->v4 migration snapshots the entire reference-BLOB table into memory twice.md' \
  'backlog/tasks/task-21130 - TTS profile v3-to-v4 migration snapshots the entire reference-BLOB table into memory twice.md'
git diff --cached --summary -- \
  'backlog/tasks/task-21130 - TTS profile v3->v4 migration snapshots the entire reference-BLOB table into memory twice.md' \
  'backlog/tasks/task-21130 - TTS profile v3-to-v4 migration snapshots the entire reference-BLOB table into memory twice.md'
git diff --cached -- \
  'backlog/tasks/task-21130 - TTS profile v3->v4 migration snapshots the entire reference-BLOB table into memory twice.md' \
  'backlog/tasks/task-21130 - TTS profile v3-to-v4 migration snapshots the entire reference-BLOB table into memory twice.md'
```

Expected: SHA-256 exactly matches Step 1; the staged diff identifies a pure rename and shows no content hunk. Leave the verified rename staged for the Task 3 commit.

- [ ] **Step 4: Audit references to the old path**

Run:

```bash
git grep -n -F \
  'task-21130 - TTS profile v3->v4 migration snapshots the entire reference-BLOB table into memory twice.md' \
  -- . \
  ':!Docs/superpowers/specs/2026-08-23-task-21139-windows-safe-backlog-paths-design.md' \
  ':!Docs/superpowers/plans/2026-08-23-task-21139-windows-safe-backlog-paths.md'
```

Expected: no matches outside the approved design and this implementation plan, which intentionally retain the old literal as historical evidence and as the rename procedure. Update any resolvable link, command, or live path value with `apply_patch`; do not rewrite unrelated `v3->v4` prose or TASK-21130's content.

- [ ] **Step 5: Demonstrate integration GREEN on the real inventory**

Run:

```bash
python3 scripts/check_backlog_task_ids.py
pytest -q Tests/CI/test_backlog_task_id_uniqueness.py
```

Expected: checker exits 0 with no false positive across live/completed/archive; the focused local uniqueness module passes.

- [ ] **Step 6: Commit the pure path repair**

```bash
git add 'backlog/tasks/task-21130 - TTS profile v3->v4 migration snapshots the entire reference-BLOB table into memory twice.md' 'backlog/tasks/task-21130 - TTS profile v3-to-v4 migration snapshots the entire reference-BLOB table into memory twice.md'
git commit -m "fix(backlog): make TASK-21130 path Windows-safe"
```

### Task 4: Document the incident and run the complete scoped local gate

**Files:**
- Modify: `backlog/docs/lessons-backlog-hygiene.md`
- Test: `Tests/Architecture/test_derived_artifact_checkers.py`
- Test: `Tests/CI/test_backlog_task_id_uniqueness.py`
- Test: `Tests/CI/test_derived_artifacts_workflow.py`

- [ ] **Step 1: Add the evidence-based Backlog hygiene lesson**

Append a short entry following the existing file's format:

```markdown
### Backlog filenames must survive every supported checkout platform (TASK-21139)

**Incident:** On 2026-08-22, commit `46cb7bc1f` added TASK-21130 with `>` in its
tracked filename. Git for Windows fetched the repository but `actions/checkout`
failed with exit 128 before project tests in runs `32617893248` and `32617893237`.

**Lesson:** Keep task content expressive, but keep files directly inside the live,
completed, and archived Backlog buckets Win32-compatible. The shared stdlib
Backlog guard is the authoring-time source of truth; a Windows job cannot report
this cleanly because checkout fails before repository code can run.
```

- [ ] **Step 2: Run the focused architecture tests for the modified checker**

Run:

```bash
pytest -q \
  Tests/Architecture/test_derived_artifact_checkers.py::test_windows_task_name_rejects_every_reserved_character \
  Tests/Architecture/test_derived_artifact_checkers.py::test_windows_task_name_rejects_ascii_controls \
  Tests/Architecture/test_derived_artifact_checkers.py::test_windows_task_name_rejects_trailing_dot_or_space \
  Tests/Architecture/test_derived_artifact_checkers.py::test_windows_task_name_rejects_reserved_device_stems \
  Tests/Architecture/test_derived_artifact_checkers.py::test_windows_task_name_accepts_valid_names \
  Tests/Architecture/test_derived_artifact_checkers.py::test_windows_incompatible_paths_scan_files_in_each_existing_bucket \
  Tests/Architecture/test_derived_artifact_checkers.py::test_main_reports_duplicate_ids_and_windows_paths_together \
  Tests/Architecture/test_derived_artifact_checkers.py::test_duplicate_task_ids_are_caught_in_both_namespaces \
  Tests/Architecture/test_derived_artifact_checkers.py::test_duplicate_task_ids_are_caught_across_buckets \
  Tests/Architecture/test_derived_artifact_checkers.py::test_default_scope_is_every_bucket_the_cli_resolves \
  Tests/Architecture/test_derived_artifact_checkers.py::test_an_absent_optional_bucket_is_not_an_error \
  Tests/Architecture/test_derived_artifact_checkers.py::test_unique_task_ids_pass
```

Expected: all 12 selected Backlog checker nodes pass, including every new test and the five existing regressions.

- [ ] **Step 3: Run the focused CI contract tests**

Run:

```bash
pytest -q \
  Tests/CI/test_backlog_task_id_uniqueness.py \
  Tests/CI/test_derived_artifacts_workflow.py::test_every_checker_runs \
  Tests/CI/test_derived_artifacts_workflow.py::test_backlog_guard_delegates_to_the_shared_script
```

Expected: all selected tests pass; both workflows still delegate to the modified shared script.

- [ ] **Step 4: Run static checks only on modified Python files**

Run:

```bash
ruff check scripts/check_backlog_task_ids.py Tests/Architecture/test_derived_artifact_checkers.py
ruff format --check scripts/check_backlog_task_ids.py Tests/Architecture/test_derived_artifact_checkers.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit the lesson**

```bash
git add backlog/docs/lessons-backlog-hygiene.md
git commit -m "docs(backlog): record Windows checkout filename trap"
```

### Task 5: Independently review, rebase, and open the focused PR

**Files:**
- Review: all files changed from `origin/dev...HEAD`
- Modify only if a review finding is verified and within TASK-21139's acceptance criteria

- [ ] **Step 1: Request independent correctness and simplification reviews**

Use `@superpowers:requesting-code-review` for a correctness/spec review and the repository's `@ponytail-review` for deletion/simplification opportunities. Give each reviewer the approved spec, plan, and `origin/dev...HEAD` diff. Resolve only concrete, reproducible findings; do not add speculative abstractions.

- [ ] **Step 2: Apply verified review findings through focused RED/GREEN changes**

For each accepted behavior change, first add or tighten the smallest focused test, run it RED, implement the minimal fix, and rerun the same test GREEN. Commit review fixes separately. If a finding is inaccurate, document the exact code/test evidence rather than changing correct code.

- [ ] **Step 3: Rebase onto the latest dev**

Run:

```bash
git fetch origin dev
git rebase origin/dev
```

Expected: branch is based on current `origin/dev`. Resolve conflicts without dropping unrelated dev changes or changing TASK-21130's contents.

- [ ] **Step 4: Rerun the complete scoped local gate after rebase**

Repeat Task 3, Step 5 and Task 4, Steps 2-4.

Expected: checker, focused pytest nodes, Ruff, format, and `git diff --check` all exit 0. Create a Backlog task for any unrelated failure rather than repairing it here.

- [ ] **Step 5: Push and open the PR**

```bash
git push --force-with-lease -u origin codex/task-21139-windows-backlog-paths
gh pr create --base dev --head codex/task-21139-windows-backlog-paths \
  --title "fix(backlog): restore Windows-safe task checkout" \
  --body-file /tmp/task-21139-pr.md
```

The PR body must link TASK-21139 and its approved spec, explain the pure TASK-21130 rename, list the exact scoped commands/results, state `ADR required: no`, and call out the two Windows evidence jobs.

### Task 6: Address Qodo and prove Windows checkout

**Files:**
- Modify only files required by verified, in-scope review findings

- [ ] **Step 1: Wait for Qodo and required checks to post**

Inspect the PR review threads and checks with `gh pr view`, `gh api`, and `gh pr checks`. Do not infer approval from a queued or absent review.

- [ ] **Step 2: Evaluate Qodo feedback rigorously**

Use `@superpowers:receiving-code-review`: reproduce each finding against the approved contract. For an accepted behavior finding, add a focused failing test before the fix; for a documentation-only finding, edit only the affected text. Reply with evidence and resolve each addressed thread.

- [ ] **Step 3: Rerun only the impacted focused test plus the complete scoped gate**

Run the smallest test that proves each fix, then repeat Task 3, Step 5 and Task 4, Steps 2-4 once after all review changes.

Expected: all scoped checks pass. File a Backlog task for every unrelated failing test or check, including the exact run/job evidence.

- [ ] **Step 4: Verify both formerly blocked Windows job types pass checkout**

Inspect logs for the PR's current head and verify `actions/checkout` succeeds before project steps in:

- `GGUF source evidence - windows-latest` (the job type that failed in run `32617893248`)
- `Artifact leases - Python 3.11 - windows-latest` (the job type that failed in Tests run `32617893237`)

Expected: neither log contains `error: invalid path`; both advance beyond checkout. A later unrelated failure does not invalidate the acceptance criterion, but it must receive its own Backlog task before continuing.

### Task 7: Close TASK-21139 and merge

**Files:**
- Modify: `backlog/tasks/task-21139 - Restore Windows checkout for Backlog task paths.md`

- [ ] **Step 1: Complete the task record only after current-head evidence exists**

Use `apply_patch` to check all acceptance criteria and add concise Implementation Notes covering the classifier/scanner and unchanged `duplicate_ids()` contract; byte-identical TASK-21130 rename and reference audit; focused test/Ruff evidence; both Windows job URLs; review/Qodo disposition; the lesson file; and `ADR required: no`, `ADR path: N/A`, with its reason.

Then use the Backlog CLI to set the task to Done:

```bash
backlog task edit 21139 -s Done
```

- [ ] **Step 2: Verify the closeout diff and rerun the guard**

Run:

```bash
python3 scripts/check_backlog_task_ids.py
git diff --check
git status --short
```

Expected: checker and whitespace gate pass; status contains only the intentional TASK-21139 closeout edit.

- [ ] **Step 3: Commit and push closeout**

```bash
git add 'backlog/tasks/task-21139 - Restore Windows checkout for Backlog task paths.md'
git commit -m "docs(backlog): complete TASK-21139"
git push --force-with-lease
```

- [ ] **Step 4: Confirm current-head reviews/checks and merge**

Confirm all configured required checks and review threads are terminal and acceptable on the new head. Merge the PR using the repository's accepted method, then confirm the PR reports merged and `dev` contains the Windows-safe TASK-21130 path.
