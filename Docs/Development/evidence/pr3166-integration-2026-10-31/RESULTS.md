# PR3166 current-dev integration — 2026-10-31

Original head: `34f88d6322ee2819d3e189e97eda5bc0e3f419f1`.
Integration base: `b1ca1b11fa44a8e0ff0a09dfbffe940072861ba4`.
Rebased head before these repairs: `fd922602dfa9711476da79e171ce0ead93eeab97`.
Branch: `codex/backlog-pr3166-repair`. No remote publication or merge performed.

ADR required: no. Existing ADR-126/230 own finite reads, exact sources and retirement; no authority or storage policy changes. TASK35679 remains In Progress with AC2 open. Historical measurements and their adverse results remain in `../windows-performance-2026-10-18/`; this integration makes no new speedup claim.

## Integration decisions

The rebase preserves current dev's native admission, configuration locks, caller qualification, run cadence and refresh-ahead threshold. Independent doc/lesson additions were retained. The current Home readiness tuple and current transcript session/library outcome seams are reflected in the old fixture fakes.

Deferred Library construction now calls the existing app `ensure_notes_sync_runtime_started` after attaching a successfully prepared owner, on the UI loop. Failed construction or detached/changed app ownership cannot start it. The regression was red before the repair and green afterward.

Conversation appearance reuse now requires the same service even in the current dev stale-display route; replacing a service cannot serve the prior service's map. Both ordinary and stale-display routes are covered. Resume reconciliation logs retain a literal diagnostic without an exception object, preventing user-bearing exception text from reaching persistent logs; the new regression observed the original leak before repair.

Only incoming Settings task35587 was renumbered to TASK38237. The already-landed session-close TASK35587 remains intact. The replacement task records the root coordinator's census and buffered allocation; current refs/worktrees still require a collision recheck before publication. Raw historical evidence/revision identifiers are unchanged.

## Verification method

`run_tests.py` creates native Windows private run/config/temp directories with the repository's `platform_files.os.mkdir(..., 0o700)`. It sets TLDW_TEST_CONFIG_ROOT, TEMP and TMP before child imports and invokes the existing `Tests.windows_private_fixture_runner`, using the shared Python 3.12 environment with `-X utf8`. Native user execution preserves real ownership/admission; no guard is patched. Each completed run has a command/exit/root receipt and raw log. Later receipts include a hash of the uncommitted product/test diff. Runs overlap; counts must not be added as distinct cases.

| Receipt | Result | Meaning |
| --- | --- | --- |
| library-start-red | 1 failed, 5 passed | Missing deferred-owner start observed. |
| library-home-green | 48 passed | Library owner lifecycle, lazy boot and hidden Home behavior. |
| appearance-red | 1 failed, 1 passed | Replaced service leaked through stale-display reuse before repair. |
| transcript-appearance-green | 7 passed, 9 failed | Appearance repair green; transcript fake still used obsolete session field in this intermediate run. |
| diagnostic-red | 1 failed | Original exception object reached the diagnostic sink. |
| reconciliation-native | 56 passed | Native config/workspace ownership, cancellation, retirement and body-free diagnostic regression. |
| console-current-units | 110 passed, 9 failed | Original fixture seams caused the nine transcript failures; retained as intermediate evidence. |
| current-fixture-contracts | 9 passed, 1 failed | All corrected transcript contracts passed; mounted cadence investigation below. |
| native-storage-css | 71 passed, 6 failed | Native ancestry/initialization, CSS and Home controls; Windows platform failures below. |
| base-cadence-native | 1 passed, 6 failed | Scoped exact-base product-source control reproduces all six Windows failures; original cadence case passed on base. |
| remaining-native-correct-path | 25 passed, 2 failed | Tokenizer, all contention and return-response cases passed; timer census drift below. |
| cadence-source-retry | 5 passed, 1 failed | One extra tail projection (6 vs5) still occurred before the guarded retry. |
| cadence-retry-first-failure | 1 passed | Isolated first cadence case passed; this does not resolve the combined-run failure. |
| census-base-control | 2 failed | Four project-index sites and its missing clock root reproduce on base package sources. |
| census-reviewed-current | 2 passed, 1 failed | All seven added classifications are present and typed; only the four inherited project-index sites remain. |
| exact-size-pins | 195 passed, 9 failed | All changed size rows passed; nine unchanged baseline modules exceed their existing rows. |

The first broad focused attempt used an outer 90-second timeout and terminated. `display-native-final` likewise cut off a tokenizer case whose existing subprocess budget is 150 seconds. The corrected run used the repository's unchanged 300-second outer default and the original 150-second inner budget: the tokenizer test passed. No product timeout or guard was raised. `remaining-native` is a retained no-tests collection error caused by a mistyped inventory test path.

## Limits and baseline distinctions

The six Windows native-files failures concern POSIX mode-bit expectations, four symlink operations unavailable to the current Windows token (WinError1314), and absent `os.mkfifo`. `product_baseline.py` loads exact b1ca1b11 product sources for changed modules while keeping current dependencies/tests. This is a scoped source control, not a full baseline checkout. All six failures reproduce there.

The timer census initially exposes seven Settings sites newly reachable from the prompt-summary scheduling path, plus four pre-existing project-index sites and its missing clock root. `census_baseline.py` substitutes exact base package sources for the static reader; the four project-index sites/root remain on base. The seven Settings sites were read individually: config-identity gating bounds the count/status/banner refresh; the provider-only branch cannot be entered by the INTERNAL_PROMPTS path. Only those seven classifications were added. The inherited project-index classifications/root remain unchanged.

All changed governed size rows were remeasured exactly under the user's 2026-10-31 hosted-CI/size waiver, with dated comments and no slack. The nine failing unchanged modules have identical product/test source at the supplied base. No unrelated budgets were raised.

Persistent diagnostic statements were reviewed individually before regenerating the inventory: runtime adds a type-name-only failure diagnostic; ChatScreen adds a literal warning; workspace now has literal diagnostics without exception bodies; app service wiring only reindents an existing diagnostic. `diagnostic-statements-final.log` retains the review input. Final rebuild matches exactly: 708 owners and 16 sink files. The source comparison parsed 78 changed Python files and found no new Ruff findings (147 existing before, 146 after); this is not a claim that inherited lint debt is absent. The final focused lint invocation also reported the existing F601 duplicate controller budget in test_module_size_ratchet.py; the exact-base comparison includes that unchanged finding.

## Remaining cadence investigation

The mounted cadence fixture originally mistook refused or still-changing ticks for quiescence. It now waits finite config/context reads and verifies matching accepted inputs before counting unchanged ticks. Original-code monitoring then located an additional real config-generation change during the tail's awaited readiness read: `_key` index0 changes, cached display age is1 second, screen remains open, and preparation refuses before the counted projection. The test must retain that refusal/replay and unconsumed backstop, not demand projection from an obsolete source. The six-case follow-up passed five cases but still observed an extra projection among three tail ticks (6 vs5) in the first case. That first case subsequently passed in isolation. The earlier exact-b1ca product-source control passed the first case and did not reproduce 6 vs5; that single passing control is not proof that the intermittent failure cannot occur on base, and it preceded the quiescence/retry fixture edits. The remaining tail-publication behavior is unresolved. Root requested no more cadence reruns before handoff. No production guard or timing bound was relaxed.

Fresh independent review and final current-base integration remain root-agent responsibilities. No full-suite sweep or new performance probe was run.

Owned test processes were absent before cleanup. `fixture-cleanup.json` records removal of only the validated private roots named by this folder's receipts.

Raw logs retain their original ANSI sequences and trailing spaces. The product/test/document patch passes `git diff --check` with raw `.log` artifacts excluded; the raw log contents intentionally remain untouched (normal Git line-ending conversion applies). Intermediate diff hashes identify the tested worktree state but are not standalone source snapshots.
