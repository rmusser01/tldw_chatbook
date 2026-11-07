# TASK-37235: Library Media post-save publication observer

Status: implementer verification complete; ready for fresh scoped Superpowers review. Task remains In Progress; AC5, task closure and delivery are root-owned. No commit, push, rebase or merge was performed.

## Implementation notes

The Edit Save journey now waits for the accepted current Media page, five attached rows, and the opened row's edited title in addition to the existing saved Reader condition. Its exact highlighted-id assertion remains separate. All original service-target and Reader assertions are preserved; every other original test/helper is unchanged.

The repair and two new parameterized controls live only in `Tests/UI/test_library_media_actions_target_opened_item.py`, which is already enrolled as a whole-file PR selection. `_edit_save_has_published` does not inspect selected-cue state. The native-mount control holds the resident canvas's real `mount_all` after native removal, while Textual's native rendering batch is held. It verifies saved target 3, canonical identity, zero DOM rows, and a false readiness condition, releases inside the App context, then requires current replacement row objects and the original cue. The missing-cue arm then calls the real `apply_reader_state` with selected flags removed; readiness must remain true, and only the unchanged semantic cue assertion is required to raise `AssertionError`.

Production source, native recompose/batching, profile/path guards, test timeouts, caps and global census enrollment were not changed. No full suite was run.

ADR required: no. ADR path: N/A; existing [ADR-084](../../../../backlog/decisions/084-library-media-reader-ia.md) and [ADR-055](../../../../backlog/decisions/055-library-destructive-action-reversibility-rule.md) apply. This is a routine test-publication correction preserving [TASK-34000.18](../../../../backlog/tasks/task-34000.18%20-%20Library-Media-Read-later-trash-and-Edit-metadata-act-on-the-item-opened-by-link-not-row-1.md), especially AC1's requirement after the list refresh lands. Atomic owner: [TASK-37235](../../../../backlog/tasks/task-37235%20-%20Observe-Library-Media-cues-after-native-replacement-rows-publish.md).

## Native TDD evidence

Working directory: `C:/Users/GDesktop-1/.codex/worktrees/textual-settings-patterns/tldw_tui`.
Interpreter: `C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8`.
Runner: `.test-tmp/textual-37031-37034/run-targeted-external-profile.py`; native 0700 run/profile/TEMP roots and unchanged 180-second pytest case timeout. One App batch ran at a time.

Authoritative RED03 executed both new controls against the extracted but unchanged old Reader-only observer. Both failed at `the post-save observer settled before native replacement rows published`, after all saved-target, canonical-identity, native-batch and empty-row assertions passed. Actual JUnit: 2 cases, 2 failures, 0 errors/skips; SHA256 `a45a9c9c0816e06684b4caddf208d0499d8c87e6ebdb47d695fa9d984667d854`. Source snapshot SHA256 `8b5d6a5695497304ff0ed291448f6ae1aa8c201336040cccd2275dc6eb5704fb`. Native private root: `C:/Users/GDesktop-1/AppData/Local/codex-textual-test-531902b211d04111a7e1e6003d3fdf3c`.

Earlier qualification attempts are preserved and are not represented as clean RED batches. RED01 had one intended observer failure and one Pilot pump-drain timeout during the intentional hold. RED02 had one intended observer failure and one transient Reader `NoMatches`. The final control polls the real Reader condition without asking Pilot to drain the intentionally held pump, and retries the same query misses as the existing shared condition helper. No product behavior or timeout policy was altered to resolve those diagnostic issues.

Commands, prefix each with the interpreter above:

```text
.test-tmp/textual-37031-37034/run-targeted-external-profile.py media-37235-red-20261028-03 Tests/UI/test_library_media_actions_target_opened_item.py::test_edit_save_observer_waits_for_replacement_rows_before_checking_cue
.test-tmp/textual-37031-37034/run-targeted-external-profile.py media-37235-green-20261028-01 Tests/UI/test_library_media_actions_target_opened_item.py
```

All run directories preserve `command.json`, `run.log`, `result.xml`, and `verification.json` under `.test-tmp/textual-37031-37034/`. Native 0700 receipt directories require native/escalated reads; their permissions were not weakened. Frozen original investigation: `.test-tmp/textual-37031-37034/media-highlight-investigation.md` and adjacent diagnostic/source receipts.

## Final verification

GREEN01 ran the complete original file: **12 passed, 0 failures/errors/skips**, 2 inherited warnings, 585.55 seconds on the loaded local Windows host. This includes the exact originally failing journey, both new native controls, and all ten original cases. Actual JUnit SHA256: `1d2d7269a0cb19ca0f9a22e7f37a683a899fd459ae34f2f3b8c9c05069485477`. Native private root: `C:/Users/GDesktop-1/AppData/Local/codex-textual-test-29626541e53d4b7280122e4837d9031e`. Both authoritative RED03 and GREEN01 verification receipts list one actual report, with no child reports. Actual XML was parsed and inspected, including case-level failures/errors/skips; no profile/path-guard bypass or setup error is represented as RED.

Fresh command: interpreter above plus `.test-tmp/textual-37031-37034/media-37235-proof.py` (exit 0). The reproducible script and JSON receipt remain at those paths. Proof JSON SHA256: `51a2c898fa94956fa4e62007e89c9e631020bce78f124500a6a984dd4b894562`.

The proof establishes:

- Every original journey assertion has identical AST to HEAD `012ca47c39a16c10aa9b9b0a2298287945e22a89`; every other original function/helper also has identical AST. Only the original post-save readiness call changes.
- Comparing authoritative RED03 to executed GREEN01, only `_edit_save_has_published` changes executable AST. The native regression's logic is identical across RED/GREEN; its cleanup/query-poll corrections preceded RED03.
- `ruff check --no-cache` and scoped `git diff --check` pass. Full-file `ruff format --check --no-cache` exits 1 for the same ten inherited formatting regions; full baseline/final formatter edits match exactly, so there is no new formatting debt. Unrelated existing formatting was preserved.
- Production working-tree diff is empty. No production, dependency, CSS, timeout/cap or native/profile-policy edits were made by this task. Root-owned CI/capacity changes were left untouched.

Executed/final test source SHA256: `66241205a4ef70430e4fb0feda973fde267febcdab939cebc0d162b7add33567`. Source snapshots are preserved under `.test-tmp/textual-37031-37034/media-37235-evidence/`. Inherited output includes the unset pytest-asyncio loop-scope warning, Pydantic `exclude_if` deprecation, pydub/audioop deprecation and optional-backend startup notices; output is not claimed pristine. The local elapsed time is not a CI capacity qualification. Current-head hosted CI and final census/capacity qualification remain root-owned.

AC1–4 have implementer evidence above. AC5 remains pending root's scoped review and final evidence/delivery recording. Task acceptance checkboxes/status were not edited by this implementer.


Portable frozen sources: [authoritative RED03](media-37235-red03-source.py.txt), [executed GREEN](media-37235-green-source.py.txt), [parsed proof](media-37235-proof.json). Native XML/logs remain at the explicit private paths and hashes above; portable text uses normalized LF. Fresh scoped review and actual successor-head CI remain pending.
