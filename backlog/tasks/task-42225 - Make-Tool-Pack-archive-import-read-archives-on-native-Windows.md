---
id: TASK-42225
title: "Make Tool Pack archive import read archives on native Windows"
status: Done
assignee:
- '@claude'
created_date: 2026-11-21 13:30
updated_date: 2026-11-21 13:50
labels:
- tool-packs
- windows
- bug
priority: high
references:
- tldw_chatbook/Tool_Packs/importer.py
- Tests/Tool_Packs/test_import_safety.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tool Pack import cannot read any archive on native Windows. ADR-107 and `Docs/Design/User_Settings.md` say the import contract is supported there, so this is a bug, not a gap by design. A triage on 2026-11-21 of the 109 `Tests/Tool_Packs` tests that fail on Windows traced 25 of them to the archive reader, `_read_regular_archive`:

- **C2.** The reader refuses to run when `os.O_NOFOLLOW` is not an integer. Windows never defines it, so every *Inspect* (Settings → Tool Profiles → Import) fails with `archive_invalid` before reading a byte.
- **C3** (hidden behind C2). The reader's file identity includes `st_ctime_ns`, and it compares `fstat` of the open descriptor with `stat` of the path. On CPython 3.12 for Windows, `fstat` reports ChangeTime as `st_ctime`, but `stat` reports CreationTime. Once C2 is fixed, archives fail at random, and every archive that was rewritten in place fails.
- **Text mode** (found while fixing C2 and C3). The reader opens without `O_BINARY`. The Windows CRT therefore reads in text mode: it drops CR before LF and stops at the first Ctrl-Z byte. A canonical archive is a stored ZIP whose CRC and length fields are binary, so a few percent of real archives contain 0x1A and are cut short.

The receipt store's separate Windows failure (C1) is handled in PR #3684 and is out of scope here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 On native Windows, inspecting a valid Tool Pack archive succeeds, including an archive rewritten in place and an archive whose bytes contain CRLF or Ctrl-Z
- [x] #2 On POSIX, a host without `O_NOFOLLOW` still refuses the archive before opening it, and `st_ctime` is still part of the file identity
- [x] #3 A path whose identity differs from the descriptor that was read is refused with `archive_invalid` on every platform, and the test proves the identity comparison is reached
- [x] #4 The new regression tests are not platform-skipped
- [x] #5 Against `dev` on Windows, the touched test files show no new failures, and the 25 archive-reader tests pass across repeated runs (one also needs PR #3684's receipt store)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. C2: default `O_NOFOLLOW` to 0 on `nt` only, so POSIX stays fail-closed. On Windows, symlink substitution is still caught by the descriptor-versus-`lstat` identity check.
2. C3: compare `st_birthtime_ns` on `nt` (both `fstat` and `stat` report CreationTime there), and keep `st_ctime_ns` elsewhere.
3. Add `O_BINARY` to the open flags (0 on POSIX).
4. Add regression tests to `test_import_safety.py` with no `nt` skips. Cover a valid archive; an archive rewritten in place; an exact binary read with CRLF and Ctrl-Z; a real archive containing 0x1A; a forced identity mismatch per field through a patched `os.stat`; Windows ignoring `st_ctime` while POSIX compares it; and POSIX without `O_NOFOLLOW` refusing before open.
5. On Windows, run the three affected test files repeatedly, a dev baseline, negative controls for each partial fix, and the whole `Tests/Tool_Packs` directory.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Change.** `_read_regular_archive` in `tldw_chatbook/Tool_Packs/importer.py`:
- `O_NOFOLLOW` defaults to 0 on `nt` and stays required elsewhere.
- `O_BINARY` is added to the open flags.
- The identity uses `st_birthtime_ns` on `nt` and `st_ctime_ns` elsewhere.

Twelve new tests in `Tests/Tool_Packs/test_import_safety.py`, none platform-skipped.

**Deviation from the triage's validated patch.** The patch covered C2 and C3 only, and it passed the 25 tests because their fixture archives happen to contain no 0x1A byte. On Windows, with C2 and C3 applied but without `O_BINARY`, two of the new tests fail:
- an exact read of CRLF and Ctrl-Z bytes comes back truncated;
- a real canonical archive containing 0x1A is refused with `archive_invalid`.

So `O_BINARY` ships in the same change. The triage's C4 makes the same fix in `publication.py`.

**The rewrite test.** It sleeps 50 ms before rewriting, which is longer than one filesystem clock tick, so ChangeTime moves away from CreationTime.

**The forced-mismatch test.** It reaches the identity comparison on Windows. The older `test_rejects_archive_path_substitution_during_descriptor_read` does not: it passes there only because the CRT refuses to `os.replace` an open file. The new test asserts that the patched `stat` was called once with `follow_symlinks=False`. A passthrough `stat` (`[None]`) shows that the harness alone does not break the read.

**The POSIX-gate test.** It patches `os.name` inside `monkeypatch.context()`. A test-wide patch broke `pathlib` in conftest's teardown on Windows.

**Windows verification**, on branch `claude/windows-tool-pack-archive-reader` (base `dev` e5626ea043), 2026-11-21. Environment: Windows 10, CPython 3.12.10, `Tests.windows_private_fixture_runner` with a 0o700 private root under `~/.chatbook-uat-*`.
- **The three files** (`test_importer.py`, `test_import_safety.py`, `test_limits_and_performance.py`):
  - dev baseline: 26 failed, 34 passed;
  - with the fix, in 6 of 6 runs: 2 failed, 70 passed, with no test failing that passed on dev.
  - 24 of the 25 target tests pass in every run.
  - The 25th, `test_archive_and_receipt_capacity_accept_exact_and_reject_one_over`, now passes its archive half. It then fails constructing `ToolPackReceiptStore` (`receipt_store.py:497`), which is C1.
  - The other failure is `test_rejects_symlink_source_without_following_it` (WinError 1314, symlink privilege; C6, also fails on dev).
- **On PR #3684's head (1db0ab7632) plus this diff**, in 3 of 3 runs: 71 passed, 1 failed (the same C6 test). All 25 targets pass.
- **Whole `Tests/Tool_Packs`**: dev 137 failed / 282 passed; with the fix 113 failed / 318 passed. 24 tests went from fail to pass, the 12 new tests pass, and none went from pass to fail.
- **Negative controls** (new tests run on modified copies of dev's importer):
  - Unfixed: 11 of 12 fail. The POSIX gate test passes by design.
  - C2 only, in 3 of 3 runs: the rewrite test, both Ctrl-Z tests and the Windows `st_ctime` test fail. The plain valid-archive test also fails in 2 of 3 runs (C3 at random).
  - C2 and C3 without `O_BINARY`, in 2 of 2 runs: only the two Ctrl-Z tests fail.

**Not covered.**
- No Linux run: this host has no WSL or Docker, and no PR CI lane runs `Tests/Tool_Packs`. The POSIX branches were checked by reading the code only.
- Symlink rejection on Windows was not live-tested, because this host cannot create symlinks.
- Inspect → import → remove end to end with the real receipt store still waits on PR #3684.

`./scripts/preflight.sh`: see the PR description.
<!-- SECTION:NOTES:END -->
