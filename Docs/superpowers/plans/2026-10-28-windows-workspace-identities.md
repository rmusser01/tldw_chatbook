# Windows Workspace Handle Identity Compatibility Plan

> **For agentic workers:** Execute inline using the existing systematic-debugging and test-first workflow.

**Goal:** Preserve complete native directory identities so unchanged Windows project folders remain usable with Python 3.12+.
**Architecture:** Read the retained handle's FILE_ID_INFO using GetFileInformationByHandleEx; keep the existing no-follow handle, reparse detection, current-directory verification, and cleanup.
**Tech Stack:** stdlib ctypes; no new dependencies.
**Backlog:** TASK-37174

ADR required: no
ADR path: backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md; backlog/decisions/069-console-project-instruction-local-state-and-preflight.md
Reason: native metadata representation correction preserving the existing security contract, not a change to identity policy or authority.

- [x] Reproduce a mismatch between Python stat and the retained native handle.
- [x] Read the full 64-bit volume serial and 128-bit file identifier from that same retained handle; fail closed if unavailable.
- [x] Regenerate the embedded remote worker from source and check drift.
- [x] Run targeted root-pin tests and the actual Console folder-listing regression; retain replacement/reparse/cleanup protections.
- [x] Update architecture/task/lesson evidence and review the diff.

RED evidence: a native handle/stat comparison failed on this machine because the old reader returned only the low 32 bits of the 64-bit volume serial reported by Python. File identifiers matched on this NTFS directory. The original real worker regression returned root_pin_failed for the same representation mismatch.

References: https://docs.python.org/3.12/whatsnew/3.12.html#os ; https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_id_info


Progress:
- The native identity regression changed from a mismatched-volume assertion failure to passing.
- The selected root-pin/confinement checks passed, along with the additional note controls. The first real worker rerun still used the original checkout through the shared interpreter's isolated editable install.
- A dedicated worktree runtime now imports this checkout under `-I` while reusing the already-installed dependencies and their normal pywin32 bootstrap. The original Console folder-listing regression then passed (1 passed, 7 deselected).
- Independent review found no actionable defect in the ABI, full-width identity conversion, same-handle metadata reads, reparse checks, current-directory verification, or cleanup.
- The canonical remote worker was regenerated; the builder drift check and a standard-library-only import both pass. Regeneration also captures two existing source/bundle assignment differences; the artifact was not hand-edited.

Final verification: 67 integrated targeted tests passed (195.48s), followed by 10 final refusal/guard/hooked-recovery checks (17.60s). E9/F lint passed, seven source/test files passed formatting, existing service formatter differences match HEAD, whitespace checks passed, and the canonical generated worker matches source and imports with the standard library only. No full-repository suite was run.

Upstream reconciliation: this worktree started from an older dev base. Current dev already includes the equivalent repair and broader tests in 6fe22b80ba; retain those files without a duplicate implementation. The reporting user was on Fedora Linux, so the Windows finding is separate from the reported tool-discovery errors.

Post-rebase verification on current dev: 86 targeted tests passed (553.63s), including native/fenced reload recovery, typed preparation guards, skill gates, the upstream native root-pin controls, and isolated real folder-access/policy cases. Lint, changed-file formatting, generated-worker drift, task/ADR uniqueness, and patch whitespace checks pass. Fedora and live-model reproduction remain unconfirmed.
