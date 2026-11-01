# PR3300 backlog integration — final targeted evidence

Source `af1addda74c8dfa4b8b524cddf13814e4acd7d67`; final integration is based on dev `63b764c858cc5aedeba7f0c7ba28b57500e90568` (PR3299 merge). All Terminal behavior cases now have passing current-base receipts. Two static failures are explicitly qualified below; this is not an all-green full-suite claim. Exact committed-head approval is recorded in the external independent review.

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

Historical status at that stage (superseded by Final current-base evidence below): the initial independent review and subsequent integration review are outside the tracked tree in the PR3298/PR3300 preparation parents. Final review and final-head test evidence are pending. The subsequent source review finds no remaining Critical or Important source blockers, including repeated Return/Open and replaced inner transcripts. Final exact-head approval remains pending targeted integration evidence. At that stage, the duplicate readiness helper used the actual `_console_attach_resume_in_progress` field and reported attach phase flags on timeout; those incomplete-attach results were still under diagnosis.

## Platform boundary

The existing ADR099 Windows refusal is unchanged. Local `wsl --list --quiet` exited1 with installation help; Docker/Podman commands were absent. No fresh POSIX real-PTY execution has been claimed. Incoming author POSIX evidence is historical, not new local evidence. Portable mounted tests use the real manager with its inert recording backend and do not qualify actual POSIX shell lifecycle.

## Further targeted evidence before the final current-base rebase

`r-b1e5063f` and `r-5160c941` stopped at the unchanged20s attach prerequisite. Sanitized stacks showed a workspace reader waiting on storage initialization/locking while a hook-permissions reader traversed the native registry. These instantaneous stacks do not establish a deadlock or a single stalled syscall. Identity-preserving `sys.monitoring` observation `r-7798d07f` completed attach in1.687s; it is diagnostic, not feature qualification. The earlier `r-bfa644e9` failed at monitoring registration and did not exercise the product. No admission guard or timeout was weakened.

Uninstrumented `r-12f8ef76` passed all3 mounted return-state cases at the original20s prerequisite, including maintenance/projection deferral and queued owner switch. The remaining2 cases reached input assertions but wrongly expected backend writes. Source inspection established that RecordingTerminalBackend intentionally supplies no read/wait-for-exit methods, so the manager starts no runtime bridge; existing tests consume the real bounded input actor. The new tests were corrected accordingly, with exact bytes and empty-queue refusal/release/modal controls. `r-61fefe24` then passed the full mounted Paste path; voice-Escape stopped at the intermittent attach prerequisite before its body. All these completed receipts have frozen sources. Voice/current-base controls were still pending at that stage; their completed results are in Final current-base evidence below.

## Final current-base evidence

- `r-82683942`:24 passes,2 startup failures,0 errors/skips;334.727s pytest. All small return/input controls, arm/create, ordinary return, narrow geometry, rail capture/release, combined PR3299 cue behavior and the explicit-input compatibility control passed. Paste and second-session cases stopped at obsolete startup preparation before their behavior assertions.
- `r-b325a897`:6 passes,0 failures/errors/skips;111.677s pytest,117.64s wall. All3 mounted deferred-return/owner-switch cases, full Paste admission/release/modal path, capturing voice Escape and second-session selection/focus pass. Both test modules now reuse current dev's real `_mounted(..., messages=False)` readiness: actual attach/resume, existing focus timers completed, actual composer focus. This adopts that existing fixture's30s attach allowance from the duplicate20s helper; action bounds and assertions are unchanged. The change does not establish why earlier native startup stalled. Independent review found no fixture bypass. `r-676b3ae2` had0tests due to a mistaken selector, then the command was corrected.
- `r-425dfddb`:1 intended causal failure,0errors/skips;22.141s pytest. After real attach, the test substituted only the original `finish_center_swap` and `_resync_returned_transcript` fromfd3375eaf4faf60a24a3d2bda7f2f95f62701b92. Actual Return with maintenance-held publication discarded pending reading state. Reference SHA256:1de33dab57b333bca71a8ac09bb73f3e6a0bebe8937761fe4c7304984574cad4. Final test bytes were restored exactly; production was never changed. Earlier controls rf50c2b28/r2f4a7b09/r36c35045 stopped at startup and are not causal proof.
- `r-ba2c6572`:12 passes,2 qualified static failures,0errors/skips;119.416s pytest,124.58s wall. The restored fixed mounted Return case passes. Exact ChatScreen24598lines/741methods, real token/hex/spacing governance, generated bundle reproducibility and parser bracketed-paste enable/disable/reset pass.

All completed receipts above freeze before/after sources. The inherited cue test preserves actual viewport Escape/x/y and unchanged draft checks, verifies the composer hidden/unfocusable throughout Terminal ownership, and uses actual Return before normal composer typing and full-row cue paint checks. It preserves the user's own collapse preference. No production change followed the current-base behavior receipts.

### Qualified static failures

`test_active_css_extension_is_in_hex_floor` fails in its own negative-control regex: it expects `components/active.css`, but native Windows Path emits a backslash. The entire test module is identical to dev63b764c858 after newline normalization (SHA2565156627e02e7ad2df21dfcf316c684606ef57d090a95633e7695dd6d93e9e681), and this case substitutes its own miniature CSS inputs. The real project token/hex/spacing assertions passed. No test was weakened or changed for this baseline portability mismatch.

Boot CSS measures615783bytes against608090. The same current-dev checkout's five boot sources measure615566: this PR adds217bytes only to the generated bundle. This size gate falls under the user's explicit waiver. Its ceiling and snapshot remain unchanged. The source counts and comparison are retained in `static-exception-evidence.json` beside this worktree.

The changed regression tests pass Ruff E9/F63/F7/F82. The changed production helpers/new test passed prior focused Ruff; ChatScreen's18 existing diagnostics match current dev by code/message/source text in `lint-parity-current-base.json`. `git diff --check` passes. The earlier291 passing rail/P0/input contracts remain qualified evidence; no full suite, new provider call or fresh POSIX shell run is claimed.
