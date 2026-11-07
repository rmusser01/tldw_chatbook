# Session switcher native-report follow-up — 2026-09-07

Separate follow-up to merged PR #2469, based on dev
`29442a160bb69cf2183a3aa0d1e68c7f8990e3bf`. This does not reopen the
broader release qualification or the deferred TASK-31966 GC investigation.

## Report and corrections

The user reported duplicate Console tabs after reopening a conversation and
could not identify the selected switcher mode. The screenshot establishes the
duplicate tabs, not which mode caused them. Controlled tests reproduced
History-to-History and Character-to-History duplicates; the other two mode
combinations already reused the exact runtime.

History now opts into exact persisted-ID reuse after saved-record validation.
It preserves authority checks and missing-record refusal; IDs beginning with
`native:` are not mistaken for runtime IDs. Other cold-resume callers retain
their intentional fresh-session behavior. Warm activation uses the existing
guarded presentation path, including refreshing the current retrieval scope.
Existing duplicate tabs are not automatically closed or consolidated.

The existing border title now names `[Active]`, `[History]`, or
`[Character chats]`, including explicit History widening. The selected mode
uses the app's selected styling, independently of focus. The title uses primary
text color: visual inspection caught that inheriting the border's dark color
made the first implementation unreadable. No rows, shortcuts, or CSS budget
were added. Impeccable's label-before-color guidance informed this correction.

## Verification

- Initial duplicate/mode regression: 6 failed, 4 passed; review regressions for
  deleted records and prefixed IDs failed before their corrections.
- Real registry scope regression: cached one source remained stale after the
  inactive target's workspace changed to two; then passed after refresh repair.
- Covering run after activation repairs: **48 passed**, one expected CSS
  headroom warning, 81.47 seconds. Covers the new reuse tests, installed
  activation, trust, geometry, History widening/loading, and CSS budget.
- After the final title-color correction: the two mode tests pass at **52×20
  and 120×50**, checking actual compositor text and at least 4.5:1 title contrast
  with search or an inactive mode button focused. Geometry and CSS rerun:
  **12 passed**, one expected headroom warning. Boot CSS is **803,784/804,000
  bytes**, 216 bytes remaining; no cap raise.
- No added Ruff diagnostics against the base; changed focused files formatted,
  whitespace clean, and all six preflight derived-artifact guards pass.
- Covering resource observer: regular descriptors plateau at seven with no
  remaining SQLite file handles; only first-use FIFO/socket additions. Tests
  used the existing isolated compatible qualification dependency overlay, not
  a shared-environment dependency change.
- Independent static review: identity, deletion, and scope findings addressed;
  no remaining scoped findings. Only targeted tests were run, not a full sweep.

Reproduce the main functional and visual regressions with:

```sh
python -m pytest Tests/UI/test_console_switcher_reuse_and_modes.py -q
python -m pytest Tests/UI/test_console_character_activation_presentation.py Tests/UI/test_console_session_switcher_trust.py Tests/UI/test_console_character_switcher_geometry.py Tests/Performance/test_boot_css_byte_budget.py -q
```

## Remaining manual handoff

### PR #2487 Qodo remediation

Rebased without conflicts onto dev `0bb00beaf4e324385a10878bb1ceb380ec55e3e9`;
range-diff confirms the original patch was unchanged. Qodo's two findings were
addressed before merge: scope-display refresh is best effort, matching native
tab activation, and isolated unit tests supplement the installed UI coverage.
Cancellation still restores and propagates; genuine presentation failure still
restores the prior session. The failure was reproduced by one unit test and one
installed History test before the exception boundary was added.

The post-rebase covering run passed **60 tests in 82.87 seconds**, with only the
expected CSS headroom warning. Eleven isolated cases cover exact saved identity,
active-runtime preference, missing-record refusal, mode widening/selection, and
scope-error versus cancellation/presentation outcomes. The installed regression
also verifies reused runtime and composer focus during a scope-display outage.
Independent static review reported no remaining findings.

The diagnostic inventory was reviewed using its statement comparison before
regeneration: one new warning in `workspace.py`, containing a fixed message and
opaque runtime session ID, with existing exception logging semantics and no new
sink. No conversation content, credentials, paths, or URLs were added to its
format arguments. Native/external qualifications below remain unchanged.

A smaller verification run exposed an unawaited reconciliation coroutine.
Tracemalloc identified its allocation at the reconciliation worker handoff;
the worker could be cancelled before starting that eagerly created coroutine.
The handoff now uses an async `functools.partial`, so creation occurs only at
execution. A plain lambda was rejected by Textual's actual async-worker tests;
the final unit boundary also asserts an async-callable input. The final gate
covering twelve isolated cases, fourteen installed switcher cases, and all
forty-one activity-switcher cases passed **67 tests in 47.28 seconds, without
warnings**, with the same stable descriptor plateau. No GC policy changed.

The user's existing synthetic native app and its checkout were left unchanged;
the fixes were built in a separate worktree. Headless Textual evidence is not
new native-terminal acceptance. On a restarted fixed build, verify:

1. Ctrl+K and F3 visibly identify all three modes, even after focus moves.
2. Resume a saved conversation from History, switch away, then resume it again:
   the same tab is selected and the tab count does not grow.
3. Repeat across History and Character chats; exact transcript and composer
   focus must match. Existing duplicates from the older build may remain.

Task-31245 stays In Progress. Existing native, Windows, participant, and deferred
performance qualifications in the main QA report are not waived. Existing
ADR-120, ADR-031, and ADR-097 apply; no new architectural decision is required.

## 2026-09-25 qualification refresh

The merged code's focused tests were rerun on a clean isolated checkout of
`461668df00` (a descendant of PR #2487). Later config-participant admission
had made the mounted tests' per-case profile redirects invalid before any
switcher action. The test module now uses its collection-time private profile,
as other source-bound mounted suites do. Later Console visit ownership also
required the unit doubles to expose the optional visit callbacks and
`resume_if` argument. No product code changed.

The installed scope-refresh test uses Console's Alt+I Inspector shortcut; its
incidental pointer click had become out of bounds under the current layout.
Wait failures now identify the switcher stage. After these test-only repairs,
the focused unit file passed **12/12** and the installed switcher file passed
**14/14**, including all four History/Character reopening combinations with
both current and inactive tabs, exact saved identity, 52x20/120x50 mode paint,
and scope refresh. Ruff checks and whitespace checks passed. Earlier attempts
included an isolated private SQLite helper timeout and two 5-second wait
timeouts that passed on retry; keep those attempts in the evidence history.

The short native-terminal smoke remains unverified: the prior QA window was
removed, the shell-side controller daemon could not start in this host session,
and the connected desktop controller refuses terminal-app control. No native
keyboard or screenshot outcome is inferred from the mounted tests. TASK-31245
remains In Progress; Windows, participant, and performance evidence also remain
open.

## 2026-10-04 coalesced transcript publication repair

PR #2835 is merged. Fresh checks on `dev f800952214` initially passed 24 of
26 cases, with two Character activation waits failing after an inactive-tab
switch. Both passed individually, but bounded state probes reproduced genuine
postcommit `FAILED` results as well as one wait that expired before a later
successful completion. Passing retries did not establish a clean result.

The reproduced failure had the requested runtime active and composer focused,
but the transcript still belonged to the preceding runtime. A whole-Console
sync was already running, so the requested sync returned after coalescing its
work. The exact-ready check correctly rejected the stale transcript and rolled
back. Subsequent reconciliation could retire the visible failure.

Character activation now awaits the existing transcript renderer before its
immediate composer-focus and strict readiness proof. It retains the transcript
refresh lock, immutable target, modal ownership, precommit cancellation,
postcommit rollback, and ordinary exposed-destination requirements. No timeout
was raised, readiness check weakened, new resume path introduced, or visual
value changed. Existing ADR-120 governs; no new ADR is needed.

A deterministic installed regression holds the full sync before transcript
publication while opening an exact existing character conversation. After a
fixture identity-type correction, the valid RED test returned `FAILED` before
the repair; GREEN returned `OPENED` and the same single runtime with its exact
transcript owner and composer focus. The affected unit and reuse/mode cases
passed **27 tests**. The additional activation file initially had **14 setup
errors** from missing collection-time profile ownership, before any switcher
action; it now declares the same private-profile marker as the reuse suite,
and all **14 cases passed** without a diagnostic plugin. Independent review
found no production blocker, but identified an older rollback double missing
the awaited renderer seam. That fixture now reaches and asserts its intended
stale-transcript, missing-composer, and broken-focus paths; the activation and
reuse-decision unit files passed **40 tests**.

The changed tests pass Ruff and all four Python paths pass formatting.
Workspace's 69 inherited Ruff diagnostics match the exact base, with none added.
After rebasing onto `dev 9878fd251a`, the four-file gate passed **69 tests**
in 210.09 seconds with no warnings and no diagnostic plugin. All eleven
derived-artifact guards passed; the first sandboxed attempt could not obtain
the pinned Mermaid input, and the network-enabled rerun verified its hashes
and completed successfully. Changed-path Ruff, formatting, and whitespace
checks also passed after the rebase.
Native terminal, Windows, participant, and deferred performance qualifications
remain open. TASK-31245 remains In Progress.

### Qodo cancellation follow-up

Qodo's review of the original patch identified a real postcommit cancellation
gap: the renderer could suspend after a cold runtime was captured but before
its token reached the outer rollback owner. Caller shielding does not prevent
cancellation of that child task. The deterministic real-store RED run leaked
the cold runtime (one failure) while the warm-preservation case passed.

The opener now catches cancellation at that publication seam, synchronously
removes only its exact captured cold instance using the store's existing
ownership guard, and rethrows. The incumbent caller still drains its shielded
prior-session restoration; warm and unrelated runtimes are retained. No new
cleanup task or suspension point was added. The two regression cases and
covering activation/reuse-decision unit files passed **42 tests** in 5.83s.
Independent follow-up review found no issues. The new installed regression's
Google-style docstring now describes both fixture arguments, addressing Qodo's
documentation finding. ADR-120 and all qualification limits remain unchanged.

Final covering verification of this correction passed **29 mounted reuse/mode
and activation-presentation tests** in 180.60s with no warnings, in addition to
the 42 unit tests. All eleven artifact guards passed again. Changed tests pass
Ruff, all four Python paths pass formatting, and whitespace is clean; workspace
still has exactly the base's 69 inherited lint findings with no additions.
