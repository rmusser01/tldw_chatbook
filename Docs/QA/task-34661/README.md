# SQLite ownership investigation — historical RED and bounded repair

Latest replacement-checkout continuation: [separate remaining GC/non-GC latency evidence and pending designs](latency-attribution-2026-10-21.md),
after [finite History worker retirement and remaining latency](history-worker-continuation-2026-10-21.md),
after [frame-local tint reuse and newly pinned finite History ownership gap](render-tint-continuation-2026-10-21.md),
after [owner-local BoxModel and unchanged startup budgets](box-model-continuation-2026-10-21.md),
after [equal mode copy and exact test-owned retirement](modebar-continuation-2026-10-21.md),
after [creation-only draft defaults and strict caller retirement](draft-default-continuation-2026-10-21.md),
after [plain session tabs and targeted retirement](tab-label-continuation-2026-10-21.md),
after [due-timer admission](due-timer-continuation-2026-10-21.md),
after [owner-result reuse and Character labels](owner-result-label-continuation-2026-10-21.md),
[native readiness polling](credential-poll-continuation-2026-10-21.md) and
[native style-color ownership](style-colors-continuation-2026-10-21.md). Earlier:
the [visible-copy allocation and finite Workspace retirement](visible-copy-continuation-2026-10-21.md),
the [current-copy rendering and native worker settlement](copy-settlement-continuation-2026-10-21.md) and
the [2026-10-20 native mark ownership and latency receipt](continuation-2026-10-20.md).
Historical failures below remain retained; no bounded pass is whole-application
qualification.

Base: `61e28695bc4fa9a62e8f14771a529591576d96ce`. The fresh regression runs
actual `TldwCli`, real constructor stores and actual `run_test` shutdown in the
existing private-profile child. It does not fake database getters or IO services.

The first valid run failed on five still-queryable captured native handles:
Chat/config, Evaluations, Collections, Subscriptions and Workspace. A read-only
observer confirms all five remain open after actual app shutdown. The test's
own final cleanup closes them; its later closed-state observation is **not**
production retirement evidence.

A corrected Darwin `F_GETPATH` census observed **21 database-related descriptors**
after app shutdown, before test cleanup, with zero storage operations and pending
acquisitions. Eight default-executor threads remained at that observation boundary;
it precedes the pytest Runner's physical pool join. The total descriptor count is
not exclusive attribution to the five constructor handles or a repeated-run
plateau result. This child failed in 7.34s with one diagnostic-plugin
`PytestAssertRewriteWarning`, retained rather than suppressed.

Earlier invalid diagnostics remain local: the geometry fixture supplied a fake
app with no Chat owner, so its constructor AttributeError was setup-invalid and
not a native-retirement RED. The first census used `readlink(/dev/fd)` on Darwin;
its reported zero database descriptors is not usable evidence. Successful real
`SELECT 1` checks still independently prove the first valid run's five handles
were live.

Source tracing finds no constructor-store retirement at the current app shutdown
boundary. Four stores are composed for the app; Chat selects a reusable shared
config/Notes owner. Existing source close guards retain active borrowers and
failed-close references. `operation_owned_connection` retires only a handle newly
opened during a finite synchronous callback; it is not a global shutdown policy.
Existing evaluation `aclose()` already drains its own task set before DB close.

The proposed production correction requires a concrete app-versus-process
ownership decision and ADR approval. No production lifecycle changes, global
quiescence, GC, limit increases, cancellation substitutes or warning suppression
have been implemented. The intentionally RED test must not be described as a
green change. Parent qualification remains unwaived.

New raw aggregate logs stay local. Hand-audited SHA-256 receipts:

| Receipt | SHA-256 |
| --- | --- |
| `Tests/App/test_app_sqlite_retirement.py` | `910ee0cd0bb37695949b27cc87b7dc6ace195cd952ef32d87eb47abc4c99ce25` |
| `/private/tmp/app-sqlite-owner-probe-eaRCY3/real-app-native-inventory.log` | `813f97f8b4c9d95734bad1f2c0b64bf0c734ee2bcaa6ee3efeb6aec103aebf6b` |
| `/private/tmp/app-sqlite-owner-probe-eaRCY3/real-app-pytest/test_real_app_shutdown_retires0/private-profile-process/pytest.log` | `05d7d5c75c60e8377ca3bf423a5c6aeaaecc0f239446b8a028e3012defc5209f` |
| `/private/tmp/app-sqlite-owner-probe-eaRCY3/app_owner_probe.py` | `47fdd771d6acb82316e3940038e4ea9c4cb3cf59c5b40b1b15f3d6bf0676a94a` |
| `/private/tmp/app-sqlite-owner-probe-eaRCY3/app_owner_probe_v2.py` | `2f6f73966e3f5b56fe873914d120bb4a53f09f7b63599e95ac887e3b157f179f` |
| `/private/tmp/app-sqlite-owner-probe-eaRCY3/native-fd-inventory.log` | `f3882c88aca24d17b9c7bf00b7dc16dfe7c598429293e271ce07c7a3e22d3458` |
| `/private/tmp/app-sqlite-owner-probe-eaRCY3/native-fd-pytest/test_real_app_shutdown_retires0/private-profile-process/pytest.log` | `285c43fdc2a2e4bf4c216e362d56a2b8dc7208bf5adfe51c85f6569d8aec8516` |

The merged branch, dirty main, other worktrees and other sessions remain untouched.

## Approved bounded implementation — 2026-10-07

This section supersedes the earlier **production repair pending** status without
discarding the original REDs. The user approved the five-store written contract
and ADR225; the implementation is local, uncommitted and not integrated.

The fixed five-owner retirement boundary runs after successful producer and
Textual shutdown. It uses existing source close APIs and remembers Chat's exact
pre-existing native borrower without acquiring another connection. Raw/managed
transactions, tracked live cursors, uncertain foreign ownership, failed native
close and primary shutdown errors remain observable. Separate shared Chat
thread handles are not app-exclusive. Evaluation close now fences intake,
includes its existing tracked preparation tasks and refuses a transaction left
by a cancellation finalizer instead of silently rolling it back.

The new native contract selection passed **32 tests in 94.76s**, including actual
cold/warm/replacement app children and real SQLite borrower/failure controls.
An omission control still fails on a live captured connection after physical
executor join, before test cleanup. Earlier evaluation finalizer/preparation
REDs and the shared-Chat foreign-thread refusal RED remain local. The affected
Evaluation module passes 30 tests in 7.30s when its original default-path node
uses the existing bootstrap-profile fixture; the preceding ordinary combined
run's 32 passed / one source-selection setup failure is retained, not hidden.

The source helper lives in `app_sqlite_retirement.py`, called by the existing
lifecycle root. Two unchanged diagnostic nodes moved to `app_keep_alive.py`
and remain re-exported; their ASTs match the base exactly. Existing size caps
are unchanged: app.py 5712 lines; app_lifecycle.py 2134. The new helper, keep-alive
module, two new test files and Interrupt test pass Ruff/format. Inherited
Ruff code/message multisets are unchanged: app 150, lifecycle 70, Evaluation 58,
core-lifetime test 26. These are not whole-file lint-clean claims.

## Remaining descriptor finding — not resolved or waived

Repeated actual app sessions join their test-owned event loops/default executor
physically before native SQL and Darwin F_GETPATH measurement, and before owner
cleanup. The five constructor handles close, but **aggregate database descriptor
stability is not established**. A final-source observation records counts
**0, 3, 6** across three sessions; registered remaining owners are
`db.agent_runs`, with `agent_runs.db`, WAL and SHM descriptors. A separate
observation recorded seven DB-related descriptors. These owners are outside the
five stores explicitly approved by the current contract.

The source's AgentRuns close affects only its calling thread. Observed native
creation paths include Console bridge construction, automatic-work recovery,
FLEET wake seeding and later Console projection. Several finite paths already
use `run_owned_db_call` / `operation_owned_connection`; no blanket close or
missing-guard fix has been inferred. The creation stacks alone do not attribute
each surviving native handle to a unique callback or thread. The first new
observer invocation did not load in its private child and is not stack evidence;
its original assertion/descriptor failure remains valid. A later observer used
connection objects as ID-map keys incorrectly and its empty association is not
attribution. Both limitations are retained.

Other bounded runs passed, including one final association observer (one parent
passed in 18.88s). A timing-dependent passing run does not supersede the retained
FD-growth failures. No GC, FD ceiling/deadline increase, warning suppression,
foreign-thread closure or default-executor replacement was used. No AgentRuns
production repair has been implemented. Extending that owner's lifetime scope
requires the user's direction before altering the approved five-store contract.

## Affected regression disposition

The eight-module targeted aggregate is **183 passed / four failed in 373.63s**,
not green. Two failures are original real-app nodes refusing raw config source
selection before construction. A separate architecture selection is nine passed
/ one original import-setup failure / 48 deselected. The three original nodes
all pass using the existing bootstrap-profile fixture, **three passed in 5.91s**;
no getter, admission or assertion was changed.

The other two aggregate failures are original MCP upper-route maintenance cases
(`ordinary` / `approved`) with `AdmissionTimeout`. A fresh read-only subprocess
module census reproduces **two failed in 10.88s**, with no modified production
module loaded (`FOLLOWUP_MODULES_LOADED=[]` in both). MCP/Backup_Recovery source
and their original script/helper inputs compare byte-identical to the incoming
base. This excludes execution of the app-retirement repair in those failing
subprocesses; it does not identify their unique cause or waive those failures.

The cursor and mounted Interrupt modules pass in the affected aggregate and
their earlier exact RED/GREEN receipts remain linked from TASK34659/34660.
All eleven existing artifact guards passed. After this evidence reconciliation,
both Backlog guards pass across 5050 readable, uniquely identified,
Windows-compatible task files; whitespace checks pass. All 17 current audited
hashes compare equal. These documentation checks do not waive a failed test.
TASK34661 remains In Progress; descriptor stability and final integration review
are incomplete. TASK31966/TASK31245 and broader lifetime/FD qualification remain
unwaived. No fresh-context review, source commit, PR, push or merge is claimed.

## Five-store checkpoint hashes (historical)

All full new raw logs/XML stay local. This table is a hash/outcome receipt, not a
claim that raw aggregate dumps have been copied into this repository.

| Receipt | SHA-256 |
| --- | --- |
| `tldw_chatbook/app_sqlite_retirement.py` | `3901b8444c90b14b06a7c05d627c45dccde8b83e50424ada4d4838990bd12d98` |
| `tldw_chatbook/app.py` | `d8d6f204ba37ee9f604b0bf3395e5cdd70ed356cd66448c04931b12aa49db4e3` |
| `tldw_chatbook/app_lifecycle.py` | `9caee7868515718876b6f717656fe2e6843415469130941932d812d1ebb74338` |
| `tldw_chatbook/app_keep_alive.py` | `e5028aabd4cb95c057f8669cf072c41e4b26e7fee363e669842743076cb37a78` |
| `tldw_chatbook/Evals/eval_orchestrator.py` | `0b6e046b9d830991dc9a1a1aaff961fd8c5594614bf2933fbcf2867d5ff2fa12` |
| `Tests/App/test_app_sqlite_retirement.py` | `96475b653b35efdba500616a6b67f76b0351923b74e94d7a55018ffc38c9951b` |
| `Tests/App/test_app_sqlite_retirement_contract.py` | `3f97561af822a1b752f510420c8246cbec2e6217e77f0f0e13993f6cac7f51e1` |
| `/private/tmp/app-sqlite-final-native-controls-hEAUnS/tests.log` | `cdf2c12e7a60be18551c6d34355208f06ec3b25fdbee7a760dfa5f2b0b39a14c` |
| `/private/tmp/app-sqlite-shared-final-fd-witness-fxM3CV/tests.log` | `8ae3926418e0b2034af2728ecf1b4bbaedd575f07f1e4b3206cf7382f9c352cd` |
| `/private/tmp/app-sqlite-agent-origins-Lyri2I/tests.log` | `503bdd97c0f5b3ed2d504ddb4fb6f6e254804d1e2b703be63bee242a4b5c7faf` |
| `/private/tmp/app-sqlite-agent-owner-child-trace-PD5u6M/tests.log` | `5d2283e481f1a3b9d6b117e23a883b3f6b19158e409834c47201cfcdb852c677` |
| `/private/tmp/app-sqlite-affected-regressions-UORZC6/tests.log` | `bdf9be268b2a89770314d5d60d213d9dbdd1a2dc741ca2c5c71ae6d45bfc1aa0` |
| `/private/tmp/app-sqlite-bootstrap-controls-mzzXyX/tests.log` | `44903b31c29c3311903c81f5c6e31062015a1d7048f600c6ed21110da21f120f` |
| `/private/tmp/app-sqlite-mcp-import-witness-vjNj5X/tests.log` | `fe32f99e7bebd41fd4404cb5f5f742ae457b6778448855ed7406f80ad6bb7e64` |
| `/private/tmp/sqlite_retirement_observer.py` (final observation source) | `63871a311ef339d991a8e8ad41d5c31a2bf1efeebe7af3cb357f3ce5a1e1389a` |
| `/private/tmp/sqlite_owner_bootstrap_controls.py` | `ce5b6efadf43103c747a66623aa9e3944fe024cd21895ca20e555d8a70fa19c8` |
| `/private/tmp/sqlite_mcp_import_witness.py` | `f4dbe78f49092c76a732479879373ac97bf333f4def123e062f7d7bd1c3dba1f` |

## Approved AgentRuns extension and current bounded result

The user approved the remaining AgentRuns scope. This section supersedes the
earlier production/scope-pending wording, preserving every failed receipt.

Two separate supported ownership gaps are now reproduced and repaired locally:

- App-exit disposal cleared the runtime slot; a late real accessor constructed
  a second AgentRuns owner. The native RED fails once in 0.73s. Keeping the
  disposed runtime attached preserves its existing permanent intake latch;
  the same regression passes in 0.56s.
- Late count projection could reopen the old bridge's AgentRuns cache after
  disposal. The retained creation stack reaches the batched count through real
  Console attach reconciliation. Both single/batched native controls fail on
  a live cold handle, while two warm transaction controls pass. Reusing the
  existing operation-owned guard makes all four pass in 1.06s. This proves a
  supported gap, not unique attribution of every earlier FD-growth observation.

Unchecked source close also falsely reported disposal success for busy/failed
native owners (three REDs in 1.08s). Exact file-backed AgentRuns now uses the
existing settled-source helper at the original Console close boundary. An
incomplete result retains its owner/cache and reports a fixed category after
sibling cleanup; cancellation still wins. Custom/memory behavior is unchanged.
Final **nine native controls pass in 2.05s**, including a real foreign managed
borrower, raw transaction, managed operation, native close failure and explicit
retry after settlement. There is no foreign close, global sweep or second registry.

Final actual-app selection: **four passed in 47.72s** (cold, warm, replacement,
and three-session physical-join census). A call-through observer retains strong
native references, so GC cannot establish closure: at each of three actual
Runner/default-executor joins it records **zero retained native AgentRuns handles
and zero database-related descriptors**, before test cleanup. The final
uninstrumented three-session plateau also passes, **one parent in 15.78s**.
These are bounded local checks, not whole-app qualification or a raised FD budget.

The first new count probe's four config-source failures are setup-invalid, not
native leak REDs. Original affected modules remain **61 passed / five failed /
one warning in 13.13s**. Three failures directly refuse config source selection;
two turn controls time out/refuse work in that ordinary fixture mode. Seven
original controls pass in the existing bootstrap fixture (the eight-node run is
seven passed / one failed in 8.60s). Home still redirects its profile and refuses
construction. An authentic exact-base source export reproduces **the same Home
failure and the unchanged unawaited `to_thread` scheduler-failure warning**, one
failed / one passed / one warning in 2.25s. Neither is hidden or repaired by
changing admission, getters, test assertions or profile policy.

Affected worker, diagnostic and UI-census guards pass without artifact changes.
App/lifecycle size controls pass at unchanged ceilings; original extracted
bindings pass in existing bootstrap mode (one in 1.16s), after a retained ordinary
import/source refusal. The new test is Ruff/format clean; changed production
ranges format clean, inherited runtime30/bridge28/lifecycle70 code/message
multisets add nothing. No whole-file formatting/lint-clean claim for inherited
production files. Inline review checked all touched count/disposal callers and
borrower/error precedence; fresh-context review/integration remain pending.

Earlier MCP maintenance failures, the separate fake-worker coroutine warning,
and TASK31966/TASK31245 qualification remain unwaived. Changes remain local and
uncommitted; no PR/push/merge or other-session controls. Full raw dumps/XML and
the exact incoming source export stay LOCAL. Only audited hashes/outcomes appear
here; no aggregate dump publication or false full-copy claim.

### AgentRuns current source and retained receipt hashes

The prior table is a historical checkpoint; its old lifecycle hash is not the
current source. All other prior entries remain retained unchanged.

| Receipt | SHA-256 |
| --- | --- |
| `tldw_chatbook/Chat/console_runtime.py` | `ddc6e641c5eab5e302b8bfab601788ba9514fe423d21e266391656b791a9091d` |
| `tldw_chatbook/Chat/console_agent_bridge.py` | `c1ba259ef950494302d583795001030fbe4d7b8ff892e6fa8dec99a909cd7f3e` |
| `Tests/Chat/test_console_receipt_bootstrap.py` | `441dd4279a1a876dcb5e8c2b5bfffcbe968450f2e94276727b58af5ade7b1ef2` |
| `tldw_chatbook/app_lifecycle.py` | `450e8b13657c40e98a6731543c6cc007ee15c4281a9e7607ddd1a816e2cc2d47` |
| `/private/tmp/sqlite_agent_native_observer.py` | `9845da5ba5e68a32d3a0f670b4d3d3ab0435c5b9187fc22535347748b6725d5f` |
| `/private/tmp/agent_runs_bootstrap_controls.py` | `f0bdcfc8ab3a6cd6c92f01c2047413b1e25322617d32a6f8f6484128380d7fcf` |
| `/private/tmp/agent-runs-late-runtime-red-H2O7Z3/tests.log` | `0eb226cf3dc6e973ad48140dc23fd606e74266b4093f2c086f76f3f1d5a90373` |
| `/private/tmp/agent-runs-late-runtime-green-lYEDuh/tests.log` | `1b1a7f10ba1c6189c72d03f11895d8b6d8c97982a99f4ae5431e921c483bd81a` |
| `/private/tmp/agent-runs-late-counts-red-S59GmZ/tests.log` (setup-invalid) | `acd6467a2dbe5280328d0f5f742f93a0c6fdb687d852b5939380424cb043c235` |
| `/private/tmp/agent-runs-late-counts-native-red-AQSk7t/tests.log` | `23ff65a948a647c96cf40004df29737f23c464a262cfa2c7f4a838adad93f59a` |
| `/private/tmp/agent-runs-late-counts-green-5XTSGs/tests.log` | `42d4594984018690d79162e664cd26733e0c94ca6161d8f90e40c9606db2ab3a` |
| `/private/tmp/agent-runs-dispose-refusal-red-tu189S/tests.log` | `fdd7f6afe3acdebdf5f977e9f1bd2fc614dd5bf507eb7f2aa5e66c5e21da6468` |
| `/private/tmp/agent-runs-dispose-refusal-green-08cNGO/tests.log` | `fe4c57eddafb27e29f611c30325ad01c3140a57ab32893965205804b7bf2a28f` |
| `/private/tmp/agent-runs-final-native-contract-HaX8eN/tests.log` | `9e947321379a6fff64f4b0e020a7c5c3de7a3a0ac4177be9c21e9efd02323b2f` |
| `/private/tmp/agent-runs-final-physical-join-2wzH6f/tests.log` | `aa8792f50e03c7995bfdd112b8d3507c0fecad2df6e965e0db40e1602887daf4` |
| `/private/tmp/agent-runs-final-uninstrumented-bvROOZ/tests.log` | `7505260b71897f89bd06a874e4667aeb372977c05dfd526af48daad0b9c29773` |
| `/private/tmp/agent-runs-affected-ordinary-BxmOnz/tests.log` | `4319a4c26f32a9eb5fd43e134b7d287d7c0a246e75c5e1f1ab3946e6c875f4fa` |
| `/private/tmp/agent-runs-bootstrap-original-controls-EZlGxo/tests.log` | `4f27cb4626125c0fed7cf9974cc486513590456bb4de1f97878a4ed362d6e129` |
| `/private/tmp/agent-runs-incoming-controls-aQRc6C/tests.log` | `efef350b0bcf8be8da52a27bc5c5af5e8dd31b651c254a660fbda2bdfc8cce7a` |

### Fresh physical-run regression — 2026-10-07

The earlier no-work app/native checks do not complete AgentRuns retirement for
actual Console runs. The original mounted Interrupt body passes, then runtime
disposal correctly fails with `sqlite_retirement_incomplete:AgentRuns`: one
failed in 10.40s (`/private/tmp/agent-runs-interrupt-regression-PhTBO9/tests.log`),
reproduced with call-through native census in 10.15s
(`/private/tmp/agent-runs-interrupt-close-witness-5YanLd/tests.log`). This is not
the earlier executor shutdown hang; no original assertion/budget changed.

The census captures a live foreign `asyncio_0` handle from the actual runtime
bridge's `run_reply` roster read. Managed operations and the native transaction
are settled, but the physical worker has not retired its cache. Final runtime
close correctly refuses to close it from the UI thread. Fixture cleanup later
retires it; that cleanup is not production evidence. Both preview callers also
route through the same roster read without AgentRuns ownership guards.

Task 3 is incomplete pending causal worker-native RED/GREEN and the original
mounted regression. Reuse the existing finite-operation helper at the shared
reply/roster boundaries; preserve borrowed handles and strict foreign refusal.
No new ownership policy, registry, deadline, suppression or other-session control.

## Actual reply-worker retirement and bounded final verification

This section supersedes the preceding worker-incomplete result, not its failed
receipts. The first getter observer recorded **latest access**, not first
acquisition; its roster frame cannot uniquely attribute a surviving handle.

Actual cold reply completion/error/cancellation and preview controls reproduced
four leaks while four warm controls passed (**four failed / four passed, 2.66s**).
The existing reply-attempt wrapper now holds `operation_owned_connection` around
the entire physical reply; the shared eager roster read gives both preview
callers the same finite ownership. The same eight controls pass in 2.99s.
Warm handles/transactions remain borrowed; owning workers close their new handles.

One mounted-node pass (9.32s) and 31 native controls (11.81s) did not prove the
aggregate: the full original Interrupt module still failed **12 passed / one
failed in 119.18s**. A separate first-acquisition census reproduced the mounted
failure after its original preceding cases (**eight passed / one failed in
32.61s**). At native close, the actual foreign reply worker was still admitted
through `pending_notes_for_conversation`; logical caller cancellation had not
settled its shielded physical task. The strict close refusal was correct.

Runtime disposal now includes the controller's existing `_maintenance_background`
task census in its existing bounded drain. It does not cancel those physical
owner tasks on budget expiry. A real managed native callback held across logical
cancellation reproduced the omission (**one failed / one passed, 1.39s**).
The settled control now waits for physical completion; the zero-budget control
still retains native ownership and reports incomplete. No new task registry,
unbounded join, foreign close or deadline increase was introduced.

The first GREEN assertion incorrectly attempted SQL on a foreign owning thread
(nine passed / one thread-affinity assertion failure, 4.35s). The corrected
oracle uses SQLite's native `in_transaction` property on the same strongly held
connection, not foreign SQL or GC. Final native controls passed **33 in 7.41s**.
The complete original Interrupt module then passed both with the first-acquisition
census (**13 in 92.21s**) and uninstrumented (**13 in 101.00s**), without pytest
warnings. All ten observed retirement censuses had no admitted operations or
foreign native handles and successful current-thread close.

Inline review additionally caught a compatibility mistake: a controlled actual
in-memory owner close failure received the new file-backed incomplete result
(one failed in 0.53s). Restricting that exception result to exact **file-backed**
AgentRuns preserves original memory/custom behavior. The final current-source
native/compatibility/original-disposal selection passed **41 tests in 7.88s**,
37 deselected, without pytest warnings. This last memory-only branch correction
does not change the already verified file-backed Interrupt/physical-join paths.

The final actual three-app physical-join check passed in **16.35s** (15.91s call).
Its selected private child retained strong native references and reported zero
retained AgentRuns handles and zero database-related descriptors at **all three**
Runner/default-executor join boundaries, before test-owned cleanup. These bounded
results supersede the demonstrated scope gaps, not whole-application qualification.

Seven new/clean files pass whole-file Ruff and formatting. All six inherited
changed paths pass changed-range formatting; exact Ruff code/message multisets
remain app150/lifecycle70/evals58/core26/runtime30/bridge28. No inherited file is
claimed lint-clean. The final architecture selection is **49 passed / nine
failed in 12.66s**: all nine failed size sources and the exact ratchet test are
byte-identical to incoming `54954c0af5`. Failing paths are app_service_wiring,
console_chat_controller, console_chat_store, console_interrupt_rounds,
console_transcript, mcp_workbench, tldw_api/client, watchlists_collections_screen
and llm_screen. Changed app/lifecycle remain at **5712/5712 and 2134/2134**;
all six extracted-patch checks pass in existing bootstrap-profile mode. No
ceilings were raised and these incoming failures are not waived.

Inline self-review traced every touched shared helper and disposal caller,
same-thread close, borrower/failure retention, cancellation precedence, runtime
fencing and physical-task census. Reused existing owners/helpers rather than
introducing a lifetime framework. No schema, dependency, profile/getter/admission,
workflow, recovery, GC or warning policy change. ADR225 and its approved extension
apply. Prior artifact guards remain retained; affected documentation guards are
checked after reconciliation.

TASK34661's bounded native AC2/3 are now verified, but it remains In Progress
pending review/integration and the retained static-gate disposition. The earlier
MCP/config-source failures, separate fake-worker coroutine warning, nine incoming
size failures and TASK31966/TASK31245 qualification remain unresolved/unwaived.
Source is local/uncommitted; no fresh-context review, source commit, PR, push or
merge is claimed. No other sessions/worktrees were contacted or controlled.

### Final worker source and local receipt hashes

All full logs/XML remain LOCAL. These hand-audited outcomes/hashes do not publish
aggregate environment dumps. Earlier receipts and failed attempts remain intact.

| Receipt | SHA-256 |
| --- | --- |
| `tldw_chatbook/Chat/console_runtime.py` | `41228c108b935b2ab0d50d7180fbeb961c04f4af5f4fbfd3db9adba15d771f59` |
| `tldw_chatbook/Chat/console_agent_bridge.py` | `d1320f23d55e17824afa456923e32229ce87fe7a322ad6c9296511de580d41b6` |
| `Tests/Chat/test_console_agent_sqlite_ownership.py` | `dbf018d530639d032069964d879f82b2ad0feabf335707849eb87d327554cc56` |
| `/private/tmp/agent_runs_close_witness.py` (latest-access observer) | `8905fcfb1fb41d319321a703dd8b4a6bb9189e5f5908b6ddf4004c3219f5f7ea` |
| `/private/tmp/agent_runs_first_close_witness.py` | `adbf36e893d775839569f0082f26c76fe35f576f682ba0cfbd3073065a67d6f7` |
| `/private/tmp/agent-runs-interrupt-regression-PhTBO9/tests.log` | `e7bbe5ca4191a97a12cc1bf84335bb49e88612eada402dd98f1fbebc08208197` |
| `/private/tmp/agent-runs-interrupt-close-witness-5YanLd/tests.log` | `8001d759e1a92d5a476024079f2c35191f1fa29721d9bc9bf5d24aa6ba270fc5` |
| `/private/tmp/agent-runs-reply-native-red-eG31mI/tests.log` | `217be0cf40e43f44fa8d0e8ae287196b233d47a39ecbb0a7f20d59e0c103c156` |
| `/private/tmp/agent-runs-reply-native-green-aCNJQ4/tests.log` | `b0e4a516284425ba782bb142a36a8f8a9d3f53881cd9b3338b13012705c1cf49` |
| `/private/tmp/agent-runs-interrupt-worker-green-MGNdm8/tests.log` | `f9a96cf6dd0bd3ba13b7965d87d31b23250a5ca64bc95a52d3bbb7ec4cf78781` |
| `/private/tmp/agent-runs-worker-final-controls-b7rduA/tests.log` | `20779bf02d9abb2bd82ff234a0dc2fb69ec4c04a794d626c2a9e3caf7c7b269f` |
| `/private/tmp/agent-runs-interrupt-module-final-dtip7o/tests.log` | `309b6680fff63d6a88c8512296cbf1c61975516a1e9914446779804a410a6715` |
| `/private/tmp/agent-runs-interrupt-first-acquisition-43SUcW/tests.log` | `609b63e2c67625425d0a5ae40208d957eba17bebb4bceaf688e7db5b802d05e1` |
| `/private/tmp/agent-runs-physical-drain-red-C06czt/tests.log` | `a2b03d5471197e2212fb640385267716194065573cd9f82c863024b5f3c0cbe7` |
| `/private/tmp/agent-runs-physical-drain-green-eWGPpb/tests.log` | `a2d6a4ffda27072ff607b05194c9dcfd18fe2a8d58c667c2c4a9fd01610553bc` |
| `/private/tmp/agent-runs-physical-drain-final-U4YtOX/tests.log` | `06d1e2e0d40eea000dbd56d1cc87ba53eb4e5f2ea2b7906bacc7e6505733456d` |
| `/private/tmp/agent-runs-interrupt-final-green-sW2nwO/tests.log` | `380a2fa6cd3a768d32435e399b9b80cf3bfbc3e0de0f6a51492a1e7494b5a75b` |
| `/private/tmp/agent-runs-interrupt-uninstrumented-final-QTpgJ2/tests.log` | `6991f2fa2df852e1f8d6e24efdf9f607e9d8ae8c25f04422b98df0689fd61210` |
| `/private/tmp/agent-runs-memory-compatibility-red-OeXTCL/tests.log` | `7207a0ee380e326af677ba8b9da7285a5bf5c9058266357e13695d414cbd88b1` |
| `/private/tmp/agent-runs-worker-compat-final-86hCUb/tests.log` | `727c0ba6f411b3a778db9f9f6c811f9e65bdcb799d018b9910c517bc2d706ebb` |
| `/private/tmp/agent-runs-joined-final-MXBgZo/tests.log` | `4883538637fb58bada85eef53a9cef7d46d186104be63cc6250cfb506d7fd2fb` |
| `/private/tmp/agent-runs-joined-final-MXBgZo/pytest/test_repeated_real_apps_retire0/private-profile-process/pytest.log` | `ef6fdea76e243a3e669f1b576e88d829c82dfa6e10fec38ddcb931e429b643e8` |
| `/private/tmp/agent-runs-final-static-C3d8da/checks.log` | `20c97e98d632babfbd637f4d6912e71434b916d0a8ef06c8d6e5d5c0130ae3cf` |
| `/private/tmp/agent-runs-final-architecture-SLDzBm/tests.log` | `c1383d2b9f4785856844e40e26824174679e3dd8ddaee38a64a2c5fbad66517e` |
| `/private/tmp/agent-runs-final-guards-60HDXl/check_textual_worker_contract.log` | `9300edf0699bc6c50e88c96b641f1a8bd77b5e6dd9a06f120ab0c958f08b8fb6` |
| `/private/tmp/agent-runs-final-guards-60HDXl/check_persistent_diagnostic_inventory.log` | `27cdce6c1001c87c095892387b6c952f2808046a71d4bab19c54133b8f6a7118` |
| `/private/tmp/agent-runs-final-guards-60HDXl/check_ui_pr_gate_census.log` | `4cdbe59941c45c54bbbb5441a4e03d1491f81b3b93d28d8f78c4868e6d7ed8bb` |
| `/private/tmp/agent-runs-final-guards-60HDXl/check_backlog_task_ids.log` | `5bf79fb63c12c44377f00a748bcc04119aebb53201f8c49cb2e6ae07dda84a9b` |
| `/private/tmp/agent-runs-final-guards-60HDXl/check_backlog_task_files.log` | `ac476c2c355f7be2cbfb05a8486ae2b4fe6c13cf5f05411f59b554731cf2b76d` |

After task reconciliation, all five affected worker/diagnostic/UI-census/Backlog
guards pass; both Backlog guards verify 5050 readable, unique, Windows-compatible
tasks. Whitespace passes. All 29 final audited hashes verify; original cursor
and Interrupt source hashes remain unchanged. No pending test/guard session.

## Local commit and continued failure work — 2026-10-08

The user authorized a local commit and addressing the retained failures. The
pre-commit native/disposal selection passed 41 tests in 6.06s (37 deselected),
without pytest warnings; source hashes still match the verified repair. Raw
receipt: `/private/tmp/sqlite-ownership-commit-check-ZWJY72/tests.log`. This
supersedes prior no-commit authority only; no push, merge, full sweep or
other-session control is authorized or claimed. Remaining failures will be
tracked separately without erasing this checkpoint or relaxing existing guards.

## Buddy inbox worker ownership — 2026-10-08

Correcting the Home test's admitted profile exposed a real executor-thread
Workspace handle at app retirement. The existing Buddy membership read now
guards both actual Workspace and Chat sources with `operation_owned_connection`.
This closes only newly acquired handles on their owning thread, including failed
reads; native warm borrowers remain live. Valid cold success/error REDs and final
25-test coordinator/receipt GREEN retain their raw logs locally. The unchanged
Home journey and ordinary affected seven-module selection pass (115 in 60.76s).
Native app/borrower/failure/strict-warning selection passes 33 in 49.02s, including
the repeated physical-join child. Audited receipt: `Docs/QA/task-34662/README.md`.
Earlier setup-invalid probes, native failures, inherited static gates and wider
qualification gaps are not erased or waived. No other sessions were controlled.

## Continued native ownership repairs — 2026-10-08

Three real routes remained after the Buddy checkpoint. Fleet Wake startup recovery
now joins the controller's existing retained physical-task drain; zero-budget
disposal still refuses incomplete retirement without cancelling its native owner.
Workspace catalog reads reuse `operation_owned_connection`. The finite Library
worker closer map adds only the already-registered Subscriptions source. Cold
success/error native handles close before test cleanup; warm borrowers and active
transactions stay live. No foreign-thread SQL/close, GC, suppression or larger cap.

Held Wake RED: one failed/one passed; GREEN: two passed. Correct Workspace RED:
two failed/two passed. Actual Subscriptions public-cache RED: two failed/six passed.
Final native/architecture/size selection: 81 passed in 14.63s; actual Library and
Research navigation/disposal: 74 passed in 44.35s; original Subscriptions finite
worker controls: seven passed in 5.20s. Earlier 119 native controls passed before
two separately corrected architecture/profile failures. No false green aggregate.
Warm UI-ready and cold-import controls: nine passed/two expected census warnings;
1033/1033 warm modules (zero headroom), 681/686 cold. Late pytest shared-temp
housekeeping warnings remain in local receipts; no warning-free aggregate claim.

Raw receipts remain LOCAL under `/private/tmp/ownership-size-final-NIEdyZ`:

| Receipt | SHA256 |
| --- | --- |
| `wake-native-red.log` | `23dfdc910599d838ad9d77a946121da17b80d10afc70d4ab5a87c9db90dc437d` |
| `wake-native-green.log` | `e0021dca706a84af9e3b72f90eef84154fc0e0f9207f9b963df2041c040b5d9b` |
| `workspace-catalog-native-corrected-red.log` | `6b583d05d7cd05f8305a1ad5d340e484420f4d1a6c3cb655650bfe6052d64fb6` |
| `artifact-finite-worker-native-final-red.log` | `319c04bc5766a316e5c702c3b3c662b2fed45e9140672692a55710732bbf97db` |
| `artifact-research-green.log` | `89a1e2898bea583215a0ea6f0be7c853683bcc41a92999097f02c6544100e3b9` |
| `subscriptions-worker-public-controls.log` | `ea42d2c297bd6f699b5602348eda74759e51395f0698c46d086f265ce83aa412` |
| `import-native-size-final.log` | `231c5ab27d2c1f739736f4f31d4c566cbdc2840fd3d5d02ca862c7e57a3b840a` |
| `boot-existing-owners.log` | `55eef42ec4dfd96aac0bbf8c7adae78583771df36a21cb1efed87b8307287c45` |

ADR225 remains the accepted ownership contract. Task stays In Progress pending
review/integration. Separate closed-chat Send behavior and parent qualification
remain unresolved; no push, merge, full sweep or other-session control.

## Broader qualification reconciliation — 2026-10-08

Closed-chat Send is now locally committed and its complete385-test selection
passes. Additional causal Workspace/physical-Files ownership and exact fixture
repairs are verified in [the broader audited receipt](broader-qualification-2026-10-08.md).
That receipt supersedes historical pending-source wording only, retaining every
failure/warning and the separate50ms/native/Windows/human qualification gaps.
Review/integration remain pending; no task Done or whole qualification claim.
