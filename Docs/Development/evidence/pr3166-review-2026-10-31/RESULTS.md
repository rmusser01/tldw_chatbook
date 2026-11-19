# PR3166 independent-review repair — 2026-10-31

Local branch `codex/backlog-pr3166-repair`; review input `e021a376b2a8e8230556ca2cb235bd417f7ed951`, integrated base `b1ca1b11fa44a8e0ff0a09dfbffe940072861ba4`. Root owns the final current-dev rebase, independent review and all remote actions. No remote action, merge, release or full/performance sweep was performed here.

ADR required: no. ADR path: N/A; existing ADR-126/230 apply. The repair restores the existing checked-read freshness limit without changing native admission, source/owner validation, config locks, callback affinity or physical retirement.

## Review changes

1. `ConsoleReadinessConfigProjection._refresh` retains the authenticated proof's original `at` when rebinding only the converged settings owner. The projection uses that same timestamp; custom reads retain their completion time before optional tokenizer preparation. Tokenizer work cannot grant an additional freshness window. A regression holds only the tokenizer seam while the original native config reader issues the real proof, then checks actual presentation refusal/replay, zero main-thread native entries and successful fresh-read recovery. A non-expired arm still renders.
2. Incoming native-attachment TASK-35645 is TASK-38267. The already-landed config-rebuild TASK-35645 remains untouched. Root allocated 38267 after the 71-worktree census; the same-title and all-ref history checks found no competing task. The task retains its previous status/content plus dated provenance. Maintained incoming references were updated; historical raw evidence/revision IDs were not rewritten. Root must repeat the collision check before publication.
3. The exact previously failing six-case cadence selection now has an actual-code decision trace. The first case's three tail ticks had no changed values/owners and made no projection. The later backstop observed a different config generation and cache owners. These are separate observations. Added deterministic controls establish that a late presentation publication is a new input at the same clock, renders exactly once, and stays owed if its first projection refuses. There is no product cadence change.

## Focused receipts

Each test receipt records the starting HEAD, exact command, native private root, exit code, elapsed time and SHA-256 of its adjacent source `.patch`. Applying that patch to the recorded HEAD reconstructs the tested product/test sources. Diagnostic plugins are retained alongside the receipts. The two new regression test files pass Ruff and its formatter; production retains the same pre-existing E721 exact-type guard and formatting debt as the review input (`static-review.json`). No new lint issue was added. The freshness test received formatting-only line wrapping after its green run.

| Receipt | Result | Meaning |
| --- | --- | --- |
| `tokenizer-age-red` | 1 failed, 1 passed | Original product accepted an expired native proof after the tokenizer hold. |
| `tokenizer-age-green` | 48 passed | Complete focused checked-display and tokenizer files, including owner/source/custom/cancellation guards. |
| `cadence-combined-trace` | 6 passed | Exact prior combined cadence selection, original assertions and bounds, original-code monitoring. |
| `cadence-deterministic-inputs` | 51 passed | Existing input/decision controls plus late publication and refused-publication retry. |
| `cadence-decision-red`, `cadence-decision-followup` | 1 passed each | Earlier bounded first-case traces; the first label is diagnostic, not an expected-red regression. |
| `cadence-original-tail-window` | 1 passed | Test-only additional 0.2s scheduling window before tail counting; no causal attribution. |

The exact combined raw output is compressed in `cadence-combined-trace.log.gz` with its uncompressed hash in `.raw.json`. The readable `.log` relocates only trace lines; `.trace.json` retains decision data and omits repeated synthetic config/session representations. No failing receipt is replaced by a later pass.

## Explicit unresolved qualifications

- **Historical cadence 6-versus-5 remains unattributed.** The earlier `pr3166-integration-2026-10-31/cadence-source-retry.log` failed at line372 before the later backstop. No changed-input/publication trace exists for that failure. The exact-b1ca product-source control in the earlier handoff passed; it did **not** reproduce 6-versus-5 and does not establish a baseline defect. Current passing traces and deterministic controls do not retroactively explain that failure.
- **All three connected tab-publication contracts remain unqualified**: `after_close`, `coalesced`, `same_session_drain` in `test_console_tab_publication_read_barrier.py`. No original assertion or timeout was removed or raised. TASK-35679 AC2 remains open and the task remains In Progress. Root explicitly deferred final qualification until PR3166 is rebased after the preceding native-admission work lands.
- `tabs-current-red` reached cleanup with the released original reader still running after the original ten-second bound. Its exact original native ancestry reached `Admission._tokens` / Windows path checks; the captured future was running, not cancelled or abandoned. This is not a deadlock verdict.
- `tabs-loop-retirement` and `tabs-setup-state` are failed fixture experiments. Their stronger quiet helper required attachment reconciliation that the harness did not promise. Those edits were reverted entirely; the final tab test is identical to review input.
- `tabs-native-child-after-close` and `tabs-native-owner-trace` run the original decorated node directly in a separate native-private child via its existing exact child marker and the Windows default-owner fixture runner. This avoids a second nested pytest process/profile but is not a proven fixture fix: setup still failed its original ten-second predicate. The trace identified one original config reader, one pending producer/non-retired read, and `console-sync` / `console-readiness-config` workers. Its original native stack was `config_participants.operation → raw _scope → acquire_storage → Admission.open_existing/pinned_directory → WindowsOS.fstat`. It physically returned an authentic proof after **10.641s**; the preceding original read returned without a proof after **3.485s**. There was no initializer wait in this captured stack. The generic `run_preparation_read.<locals>.invoke` source-shape diagnostic did not resolve its defining nested code; no replacement/ownership verdict is inferred from that metadata.

The current child trace is bounded evidence of the exact outstanding owner. It does not qualify keeper-only tab DOM publication or same-session retirement. The historical performance conclusions in `windows-performance-2026-10-18/MEMORY_RESULTS.md` remain unchanged: no consistent overall speedup was established.

## Local hygiene

Only receipt-owned private roots may be removed after checking live process command lines; final cleanup is recorded in `cleanup.json`. Product/test source snapshots, original failing logs, current passing receipts and exact raw decision data remain reviewable. Hosted CI and size limits retain the user's 2026-10-31 waiver; no size pin or diagnostic baseline was changed by this repair.
