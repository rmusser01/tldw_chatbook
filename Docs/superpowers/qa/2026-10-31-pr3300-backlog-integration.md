# PR3300 backlog integration — verification in progress

Source `af1addda74c8dfa4b8b524cddf13814e4acd7d67`; rebased head `fd3375eaf4faf60a24a3d2bda7f2f95f62701b92` on dev `db7e04cf484b4c0b803366b00f2bbae1b5550042`. Repair changes remain uncommitted. This is not a final passing or merge-ready claim.

The user waives hosted CI and size limits and substitutes independent Superpowers review for Qodo. Drafts and PR3267/3297 remain excluded; this session does no release or main promotion. Local targeted checks, current-head review and resolved threads remain required.

## Changes under verification

- Pending Terminal-return reading state follows the actually published conversation and completes only after position application. Deferred whole sync, deferred transcript preparation, replaced regions and queued callbacks cannot relabel a prior conversation's position.
- Capturing Terminal input receives Paste through the existing bounded controller/manager path. One current runtime view binds the selected session and its safe bracketed-paste mode; viewport repaint lag supplies no mode. Refusal shows only the runtime safe reason or fixed generic text.
- Priority voice Escape yields while the actual Terminal viewport captures input and keeps its existing route elsewhere.
- Released-mode copy no longer promises that ordinary focus resumes input. Ordinary padding uses the existing token. Textual's established Python `line_pad` workaround remains because its CSS integer parser rejects zero, including token-expanded zero.
- New regressions enroll in the existing admission-sensitive Console P0 lane. Historical full Terminal integration enrollment remains TASK-33622.22.

## Retained native receipts

All directories are siblings of this worktree. Each runner stores command, JUnit, log, result and before/after hashes; every completed receipt below had frozen source hashes.

| Receipt | Actual result | Qualification |
| --- | --- | --- |
| r-68cccf91 | 2 expected failures, no errors/skips | Held-sync empty restoration and stolen priority Escape reproduced before repair. |
| r-39b0921d | 2 passes, 2 expected failures | Return/Escape small checks pass; plain and bracketed Paste still drop before repair. |
| r-103ed829 | 5 passes, 4 failures | Fast checks pass; all mounted cases stopped before bodies at unsupported zero `line-pad` CSS. CSS proposal removed; this is not mounted behavior evidence. |
| r-a22cf718 | 4 expected failures | Independent review reproduced outer-sync/publication mismatch, early pending-state clearing, stale paste mode, and discarded safe refusal reason. |
| r-3186d3b4 | 1 expected failure | Capture relabeled a still-painted old owner's offset with the newly activating session. |
| r-49e1d6c0 | 2 expected failures | Repeated Return/Open lost or replaced the original pending reading position. |
| r-180a33f3 | 3 passes, 1 expected failure | Repeated Return/Open and owner guard pass; replacing the inner transcript still allowed a queued stale restore before the final repair. |
| r-55d8d003 | 14 passes, 5 failures, no errors/skips | All small controls, including replaced-transcript and bracketed-mode checks, pass. All five mounted cases stop at the new attach-readiness helper before test bodies: four never witness completed attach within20s, and one exposes a misspelled resume-field name. These failures do not qualify mounted behavior. |

Additional frozen receipts: `r-393f3dc6` passes all291 rail-state, P0 enrollment and bounded input contracts (105.027s pytest; no failures/errors/skips). `r-90c2caeb` repeats one mounted case after the field correction and still stops before its body: attach reconciliation remains running, with retry count0 and exhaustion false. The unchanged20s prerequisite remains under diagnosis; this is not a product return-state failure or passing mounted evidence.

The initial independent review and subsequent integration review are outside the tracked tree in the PR3298/PR3300 preparation parents. Final review and final-head test evidence are pending. The subsequent source review finds no remaining Critical or Important source blockers, including repeated Return/Open and replaced inner transcripts. Final exact-head approval remains pending targeted integration evidence. The new readiness helper now uses the actual `_console_attach_resume_in_progress` field and reports attach phase flags on timeout; the four incomplete-attach results remain under diagnosis.

## Platform boundary

The existing ADR099 Windows refusal is unchanged. Local `wsl --list --quiet` exited1 with installation help; Docker/Podman commands were absent. No fresh POSIX real-PTY execution has been claimed. Incoming author POSIX evidence is historical, not new local evidence. Portable mounted tests use the real manager with its inert recording backend and do not qualify actual POSIX shell lifecycle.

## Further targeted evidence before the final current-base rebase

`r-b1e5063f` and `r-5160c941` stopped at the unchanged20s attach prerequisite. Sanitized stacks showed a workspace reader waiting on storage initialization/locking while a hook-permissions reader traversed the native registry. These instantaneous stacks do not establish a deadlock or a single stalled syscall. Identity-preserving `sys.monitoring` observation `r-7798d07f` completed attach in1.687s; it is diagnostic, not feature qualification. The earlier `r-bfa644e9` failed at monitoring registration and did not exercise the product. No admission guard or timeout was weakened.

Uninstrumented `r-12f8ef76` passed all3 mounted return-state cases at the original20s prerequisite, including maintenance/projection deferral and queued owner switch. The remaining2 cases reached input assertions but wrongly expected backend writes. Source inspection established that RecordingTerminalBackend intentionally supplies no read/wait-for-exit methods, so the manager starts no runtime bridge; existing tests consume the real bounded input actor. The new tests were corrected accordingly, with exact bytes and empty-queue refusal/release/modal controls. `r-61fefe24` then passed the full mounted Paste path; voice-Escape stopped at the intermittent attach prerequisite before its body. All these completed receipts have frozen sources. Final voice/current-base controls remain pending.
