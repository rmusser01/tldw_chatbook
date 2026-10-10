# Console fixes and verification â€” 2026-10-04

This repair includes the earlier DeepSeek/capture retry changes, the five documented provider audit defects, and the shared/native pause causes from the original investigation. The final combined pull request targets dev. This document records failed integration measurements as well as passing subsystem checks; the earlier investigation remains a historical baseline.

## Confirmed causes and changes

- DeepSeek's documented null/object choice logprobs were rejected by a shared strict parser. The scoped provider allowance preserves generic strict checks. A failed capture must settle its exact owned failed call before retry; a canonical fingerprint remains until persistence, and retry checks retain the original actor, source, policy, header, target and settings proof.
- Startup trace GC could delete a canonical admitted revision before reservation. Live canonical metadata references now protect revision ancestry; payload reachability remains separate, so another call sharing the policy cannot retain a purged call's private payload.
- Recovery height synchronization removed/reapplied unchanged classes and inline constraints, cascading stylesheet work. Exact class and full scalar-unit equality avoids that work while correcting actual external changes. Real mounted tests cover both row heights and painted retry targets.
- Finite character snapshots repeatedly prepared native connections; workspace availability repeated binding queries. Paired reads now share one owned connection, with the UI midpoint check and revision/generation fences retained.
- Refreshes repeated context inputs, agent count reads and terminal receipt acknowledgements. Fenced disposable presentation snapshots and exact terminal acknowledgement memoization reduce that fan-out. Dispatch authority uses live reads.
- Config refresh entered the same blocking rebuild/file locks as actions. Only presentation uses a nonblocking entry and coalesced retry. Entered native custody, errors and normal action waiting remain unchanged.
- Subscription indexing repeated guarded config/admission setup on the UI before starting a worker that already owns fresh setup. Only the duplicate UI preflight is removed; the existing worker still resolves its path and takes live admission.
- Scheduler default stop-path resolution ran on the UI loop before only the file read was offloaded. Both now execute in one existing owned worker, with fresh sentinel transitions, fail-safe errors and canceled-worker drain ownership preserved.
- Windows admission always rederived scope and rewalked common ancestors. Fresh union observations, exact native security bytes and NTFS change time allow qualified reuse without caching permission decisions. Elevated SQLite sidecars can be owned by the current token's default owner; a narrow current-token/exact-user-ACL custody rule retains foreign/shared/deny rejection.
- Groq, OpenRouter, Together, Fireworks and Cerebras had documented metadata/SSE shapes rejected by current profiles. Scoped allowances and provider-owned normalizers preserve nested usage/errors and strict terminal accounting. Latest dev's Together usage request, live replay and model-discovery fixes are retained.

## Integration measurements and limits

The full app, file-backed SQLite, native guards and normal timers/startup GC are real. Only the final provider adapter returns immediate data (two complete replies, one true SSE reply). Call-through timing overhead, host load and headless dispatch affect wall time; nested seam times overlap and must not be summed. Actual DeepSeek UAT is separate.

Registered limits were fixed before final measurement: each captured send <=15s, send UI stall <=1s, <=200 main-thread admissions, <=40,000 Windows native opens or <=16 POSIX helper starts. They have not been raised.

| Local Windows integration source | Send durations (s) | Maximum send UI stalls (s) | Result |
| --- | --- | --- | --- |
| First combined changes | 53.23 / 51.05 / 40.66 | 8.58 / 9.06 / 8.10 | RED; incorrect streaming test selection also exposed |
| Persistent context and native tree observations | 37.72 / 25.54 / 22.50 | 5.71 / 4.45 / 3.84 | RED; old heartbeat phase mapping could miss entry stalls |
| Union entry snapshot, presentation lock try-entry and scheduler offload | 49.87 / 55.33 / 47.12 | 8.04 / 8.85 / 10.11 | RED; interval-intersection heartbeat includes entry stalls |
| Observer replacing guarded config loaders | No complete conversation | Not qualified | INVALID; observer violated installed callable identity, no performance inference |
| Corrected observer and finite callback/readiness fixes | 21.84 / 14.73 / 19.03 | 3.54 / 3.44 / 3.70 | Valid RED; three complete calls/links, zero checkpoints |
| Sixth final-source run with output outside runner root | First send 23.24 | Detailed receipt unavailable | Output qualification failed; original send-budget failure remains recorded |
| Seventh, corrected output root, final source rebased on dev | 22.17 / 12.93 / 18.90 | 1.62 / 1.40 / 1.39 | Valid RED; three complete calls/links, zero checkpoints |

The first, second, third, fifth and seventh integration runs completed three actual capture calls, three user/reply pairs, three response links and no dispatch checkpoint. The later eighth checkpoint is a qualified before-provider functional failure, recorded separately below; it supplies no completed-conversation timing result. The second and third used the correct two-complete/one-SSE selection. The third exposed provider-settings refresh outside the shared presentation scope and remaining per-method admission inside finite database callbacks. Finite callback ownership and captured availability now pass targeted controls. Readiness publication/source provenance and eligible session-default convergence were reconciled with 18 targeted controls. The fourth receipt is excluded because its observer replaced exact guarded function attributes; two forced/cold-loader passivity controls preserve those attributes in the corrected observer. The fifth valid receipt has warm settings cache hits, main admissions 252/213/253 and all-thread native opens 169358/119920/138514. It remains RED. Further profiling measured one actual installed MCP get_kill_switch at 3003 native opens and 18 nested generation selections (about 0.613s unprofiled); one load used 2561 opens (about 0.520s). Fresh witness queries reread control records three times and the registry twice. Fresh observation bundling and off-loop catalog store reads now have targeted qualification; no fresh witness or permission decision is cached across calls.

The seventh source is commit 0ecd8327f79c1b14ea18c15f704850cc41b0ec9f, rebased on dev a78a9a900b4901e33c031f830dd2d80224d5147d. Reviewed Console sources remain byte-equivalent after normalizing line endings. First/second send main admissions fell to 154/103, but all-thread Windows opens remain 170602/104965. Worker configuration entry repeats 13 readiness reads and 14 context reads in the first send, while native directory validation repeats within each operation. These are current investigation targets, not a passing result. The initial elevated CI ran actual ownership negatives successfully but failed one creation-barrier fixture when its cleanup attempted chmod on the Administrator-owned pytest parent. The fixture now creates an explicit TokenUser-owned target; production foreign-owner refusal is unchanged, and the local exact test passes (1.07s). Elevated rerun is pending.

## Targeted evidence so far

- Native Windows grouped-observation controls: 81 passed, 7 genuine elevated-only skips locally; six related siblings used 71 opens instead of 211. Elevated runner qualification is pending.
- Fresh witness bundling: final review repair passes 61 relevant controls, with two POSIX-only skips on Windows. Independent review found and reproduced a Windows shared-parent ACL mutation missed by the initial completion snapshot; the final snapshot now checks the existing ancestor receipts with the established private-parent policy. The proposed POSIX held-fd completion race is corrected with bottom-up fresh named-parent proof and awaits its native rename and shared-mode controls on Linux/macOS. The actual paired reader changed from 119 native opens / three control-record reads / two registry reads to 89 opens / one read each. Native record replacement, pending evidence inserted during observation, successive empty results and registry ownership/publication controls remain fresh. The broader 147-pass/53-failure diagnostic run is classification evidence only; Windows symlink/fork/pipe limitations and fixture defects are not reported as a green suite.
- Bootstrap creation durability: 21 controls passed with one genuine ordinary-token foreign-owner privilege skip, plus 15 existing native integration controls passed. Exact persistent per-entry intent covers process death and failed barriers; protected unchanged ancestors are not flushed. The final storage-source original three mounted restore cases pass (106.10s, normal exit): successful publication, persistent pending failure, and live scheduler/native pause. Original timeouts and assertions remain unchanged.
- Metadata batching: 75 focused controls passed with normal exit, plus one real compactor VACUUM through the broadened finite callback. Three independent admissions became one callback interval; availability reduced two to one, preserving SQL and physical retirement. The original 12 native snapshot tests passed. Existing mounted metadata tests had two demonstrated baseline failures and one unreproduced Textual teardown failure; the grouped summary required private-process cleanup, so it is not a clean-exit receipt.
- Scheduler stop resolution/lifecycle/maintenance: 68 passed. Subscription startup/indexing selection: 28 passed; the final retained compatibility alias also passes the two actual starter/worker controls.
- Trace GC: 24 tests passed, including actual shared-policy purge privacy and admitted revision reservation.
- Recovery layout: 4 mounted tests passed; eight unchanged calls fell from hundreds of stylesheet applications to no applications.
- Presentation config lifetime/lock contention: 34 passed, including default action waiting, continuous native lifetime, partial-lock release and body error propagation.
- Historical agent presentation reads: cold main SQL/admission intervals 2 to 0, both SQL reads under one owned worker callback. Existing close checks retain two worker intervals. Final focused refresh/readiness/bridge bundle: 36 passed, plus two mounted persisted-rail controls. Live/action historical reads remain fresh.
- MCP catalog composition: documented worker-safe sync store reads move off the main loop; async catalog, profile callbacks and publication remain on the loop. Initial provider selection: 92 passed; final controller/profile/error/cancellation controls: 22 passed, 119 deselected. Actual permission scope ownership remains live until a canceled worker retires.
- Raw source coordinator: blocking selected-source/native reads run outside the coordinator, followed by exact operation/participant/lease rechecks. Qualified original RED controls cover the lock, post-proof revocation, actual accepted RMW continuation and final source/canonical mutation. The final formatted bundle passes 21 cases (43.45s), including nonvacuous seeded permission mutation, unrelated exact same-path refusal, thread/task and cancellation custody. An additional actual restored MCP integration passes (13.32s): selected and canonical paths differ, separate exact canonical custody is admitted before acceptance, and the real permission update finishes across pause while historical bytes remain unchanged. No fresh cross-call witness or permission cache is introduced.
- Captured JSON settings: Python mapping equality treated nested true/1 and 1/1.0 as unchanged retries. Canonical JSON artifact bytes retain scalar type and spelling, while frozen response-format arrays are thawed at the existing persistence boundary. Actual gateway RED controls and actual HTTP wire qualification are recorded in TASK-34368.
- Engine capture alignment: actual Together HTTP replay exposed dispatch pinning an engine preset api_base_url while independent trace reconstruction omitted that registry-derived branch (and engine continuations). The first call stored an alignment omission and unchanged retry correctly failed closed. This affects engine presets on all OS; independent reconstruction now includes both branches without weakening the changed-surface fence. Groq separately omitted response_format from its parameter map, dropping a requested schema on the actual HTTP body; the map now forwards it. Final qualification passes 620 targeted cases (74.42s, normal exit), including 330 engine cases and six actual Groq/Together/OpenAI HTTP cases. The separate 72 ownership/type controls pass with normal exit.
- Conversation settings initialization: cold/changed context was permanently copied as busy into the modal, disabling Apply/Save/Make default. Opening now waits for the existing finite checked read and rechecks controller/store/context owner and durable origin after awaits. Actual mounted completion controls pass, and chat closure during model/thinking waits returns without an orphaned modal. The final source targeted receipt is 23 passed / 432 deselected (21.62s), including actual mounted controls and both asynchronous chat-close edges.

- Merged provider/discovery qualification: 644 passed. Offline actual-adapter replay is distinguished from paid/live provider qualification. Structured OpenRouter reasoning blocks and Fireworks opt-in token-ID arrays remain outside the supported text contract and fail closed.

## Remaining qualification

Final source native matrix, final private DeepSeek setup/model/three-message UAT, merged-source review and precise final receipts remain pending. Targeted checks only; no full-suite sweep was authorized. Existing design-governance Windows path/style failures and pre-exceeded screen structural ceilings are disclosed in task notes. Three extra Windows restore controls reporting WinError5 reproduce on the exact whole-HEAD baseline at c78b9dd81c: publication._pending -> directory flush -> OpenFileById directory reopen is denied in all three. This is an existing native boundary defect; the initial exact creation/publication barrier remedy passes the three flows (54.98s), but a concrete retry after a failed creation barrier exposed missing durable creation evidence. That additional retry protocol passes 21 focused controls plus 15 existing native integration controls; the original three mounted restores now pass on the final frozen storage source (106.10s, normal exit). Do not infer broad restore qualification from the narrower native Console controls. Elevated custody CI now explicitly fails missing capability instead of silently skipping ownership negatives; genuine ordinary local capability skips remain.

## Governance

ADR-126 governs native admission/custody and entered config lifetime. ADR-097 governs canonical metadata and payload trace roots. ADR-179 governs provider normalization. Each task contains its own acceptance criteria, plan and evidence; task status stays in progress where final integrated qualification is pending.

Native matrix at 0ecd8327 (run37241740746) remains RED: Linux shared controls pass, actual sends7.22/5.40/4.41s and UI stalls0.55/0.26/0.26s meet time limits, but real private DB helpers59/35/34 exceed16. Mac798passed6skipped with3 dependency completeness failures, then probe skipped. Windows shared624passed40skipped102failed41errors: genuine Admin-default-owner pytest fixtures violated exact-user private construction. Separate required elevated custody runs each passed39 actual controls and failed the same fixture cleanup. All failures are retained; no budget, native check, or test suppression changed.

Creation-marker actual native RED: warm reused admission allowed a newly inserted malformed exact unfinished intent while full derivation refused (two failing controls). Added exact ancestor-intent absence stamps and reuse fallback. Final root follow-up through real TokenUser-default-owner launcher:32passed2 genuine ordinary-token privilege skips13.29s normalexit0. Launcher asserts actual native default owner, actual stdlib and child file owners, restores original owner after success/error, and refuses required elevated custody. Local actual owner controls2passed0.87s. The original required three-version elevated jobs keep their unmodified token. Mac dependency tracer retains unknown relative reads and adds call-site diagnostics. Final three-OS performance, elevated fixture rerun and real DeepSeek UAT still pending.

Follow-up pre-integration root gate: 19 actual targeted controls pass in 5.04 seconds with normal exit zero, including repository coordinator/full-proof publication races, complete exact creation-intent evidence and real Windows fixture owner/restoration/refusal. Differential AST/lint check over 24 changed/new Python files retains 88 existing diagnostics and introduces zero. YAML verification confirms all three native hosts, the original three elevated-custody Python versions and no allowed-failure setting. These subsystem receipts do not establish whole-app budget success. Mac relative-call diagnostics read frame metadata only, avoiding source-line formatting that would itself open files; unknown paths remain included pending actual native evidence.

### Checkpoint bced62b: functional regression and native qualification

The eighth local native whole-app test is a qualified **functional failure**, not
another three-message latency result. Exact committed production source
`bced62b9289a1085f05b598422ec59e281df205f` loaded with ordinary timers, maintenance,
native storage guards and the actual capture path enabled. It ran 1 failed /
3 passed in 66.20s. The first user turn durably committed at phase offset 6.977s,
then the controller failed at 15.458s with `agent_activation_required` before
any provider entry (`provider_calls=0`). There are no three-reply trace receipts
for this checkpoint. Windows send phase 15.530s and 96,498 opens cannot be used
as evidence that a completed conversation meets the registered budgets.
Evidence: `console-performance-fix-eighth.json`, `perf-eighth-local.xml` and log.

Native macOS in Actions run
[37244879946](https://github.com/rmusser01/tldw_chatbook/actions/runs/37244879946)
independently reproduces the same before-provider activation failure: zero
provider calls, successful durable commit, failed controller. Its 2.933s first
send is also not a completed-send performance result. The new finite controller
wrapper encloses the broad maintenance agent callback; the focused native
reproduction and scope correction are in progress. This regression therefore
has cross-platform scope.

The checkpoint's three genuine elevated Windows custody jobs are green with
`TLDW_REQUIRE_ELEVATED_CUSTODY=1`: Python 3.12.10, 3.13 and 3.14 each execute
**40 passed, zero skipped, zero errors**, in 14.412s / 17.794s / 18.675s. Each job
also verifies that the ordinary test-fixture launcher restores the original
administrative default owner before those capability controls. Original actual
elevated ownership and negative custody tests are retained. This qualifies the
previous elevated test gap at this revision; final-source native requalification
still follows.

The macOS shared controls report 907 tests: 877 passed, 21 failed, 9 skipped,
zero errors. Three dependency completeness failures now contain actual stacks
showing only relative lstat probes from the unchanged real-profile write audit
hook through `_protected` and CPython `realpath`. The observer is corrected by
exact installed code identity and raw input attribution, with native same-name
unknown-read and descriptor-write controls; no production admission dependency
or profile guard is weakened. Five native config-sync subprocess failures are
an obsolete character presentation fixture method. The remaining thirteen
pause-drain failures require independent lifetime investigation; they remain
reported until their actual source is proved. Linux and Windows full results
were still pending when this entry was recorded.


## Ninth actual native checkpoint and shared lifetime diagnosis

The uncontended Windows checkpoint a8fdda4832f52235f34683bebefe96426fb34737 completes all three provider calls, user/reply pairs, COMPLETE traces and response links, with no dispatch checkpoint. It remains performance RED: send times28.260/25.232/24.847s, maximum heartbeat stalls1.881/2.042/1.767s, and all-thread native opens157339/140124/126019. Main admissions127/114/124 satisfy the existing200 cap. The original15s/1s/40000-open thresholds are unchanged. The interval after durable commit and before trace reservation dominates; repeated readiness reads and finite database calls contribute overlapping measured times. The native receipt console-performance-fix-ninth.json and perf-ninth-local.xml retain actual loaded source hashes.

The second three-host checkpoint bced62b9289a1085f05b598422ec59e281df205f reports the same13 later pause-drain failures on Windows, Linux and macOS. An actual original-code Windows creation/retirement census proves one Settings-worker SQLite lease from the real shared-policy generation-settings mutation remains alive after the fixture closes its main-thread handle. This is an ordinary application worker-lifetime defect, not a Windows-only pause issue. The real persistence fixture and original drain assertions are retained. The finite Settings writer fix and body-time receiver changes are under targeted qualification. Five obsolete Character facade fixture failures are corrected at a8; macOS additionally has three observer-only relative realpath probes, with exact-origin observer qualification pending the next native Mac job.

The second Windows native matrix has907tests:849passed,18failed,40skipped,0errors; Linux907:877passed,18failed,12skipped; macOS907:877passed,21failed,9skipped. The independent Windows mounted restore step fails all three original controls: startup readmission fails on success, and persistent-pending failure reports a different underlying error. These are separate unresolved native results; the earlier local3pass106.10s does not supersede them. The whole Console probe at this checkpoint fails before its first provider call on all three hosts because an agent-wide Notes interval swallowed independent sources; a8 separates finite Notes callbacks from multi-source agent cleanup and passes12native source/pause/cancellation controls. These functional failures cannot be compared as completed-send timings.

A proposed fresh raw-parent tree snapshot failed its native count hypothesis:1903 to1973opens and924 to987security reads, unchanged7checks/14witnesses/13selections. The actual scope has one parent, so the optional helper and prospective1100 bound were withdrawn. This confirms the earlier isolated6-to12 observation. No production raw-parent check change remains, and no permission/path observation is cached across calls.

Settings lifetime final local qualification: the unchanged original 25-test ordered prefix followed by 22 actual native Settings controls passes 47/47, with no skips/errors and normal exit zero (148.196s). The actual original-code observer records transient drain refusal while the accepted worker is alive, then successful drain after physical retirement; final ordinary leases, pending acquisitions, repository/raw operations and retiring holds are all zero. All four owned source/test SHA-256 values remain identical throughout. The earlier native RED establishes both uncounted full writer bodies and a leaked worker-created SQLite handle; an actual copied-library A-to-B-to-A receiver mutation additionally proves the original writer could redirect committed settings. The fix captures the original standard writer/database/repository transaction receiver, retains producer custody through repeated cancellation, and retires only the worker-created handle. Borrowed, custom, subclass and in-memory contracts retain their existing calls. Evidence receipts settings-worker-native-lifetime-red.json and settings-worker-native-lifetime-green.json retain redacted source provenance and counters. Root independently reviewed the final source with no actionable finding. This locally qualifies the shared thirteen drain failures' actual source; final three-host receipts, mounted startup readmission and whole-app performance limits remain separate pending gates.


## Exact startup initializing publication qualification

Startup publication qualification at the frozen storage SHA-256 `58f452a282dad4f18fed0108ae99878e106d66d36947c3332c6f3372b4f7ee52` passes the three original mounted restore cases and all fifteen existing coordinator/native-proof race controls. The complete first bundle is 18 passed / 10 failed, zero errors/skips, normal parent exit one (552.294s), with unchanged storage/test bytes and final parent ordinary leases, pending acquisitions, repository/raw operations and retiring holds all zero. Actual original child receipts retain the exact pause-owned startup attempt, current native thread, TokenOwner=TokenUser and zero real-profile guard refusals: both success paths publish ordinary startup, and persistent pending still refuses with `recovery_scope_uncertain`. The original success controls retain their post-resume borrowed Notes connection and startup until normal child exit. The ten added revocation controls fail before their intended revocation: seven time out at the original monitor/finalization wait and three reach the unchanged outer child timeout. They provide no negative publication evidence yet. A matched original/new observer diagnostic will record actual monitor/restoring await chains, worker code stacks and native ownership at that wait, preserving the original 20/45 second limits and every guarded function. Receipts `startup-publication-first-run.json` and `startup-custody-first-run-failures.json` preserve the separate outcomes; the original native startup RED remains `mounted-startup-readmission-red.json`. Startup negative controls, final three-host results and integrated performance/UAT remain pending.

## Actual normal Send input-pump qualification

The original mounted Enter and Send-button routes fail three native controls (39.29s, normal exit one). A passive original-code observer holds the real hook permission snapshot after its native raw-config admission: Enter's app message pump and the button's Console message pump each await dispatch then to_thread and stop delivering callbacks. The repeated-cancellation control confirms no preparation worker exists, despite actual raw operations and leases being present. All held readers drain in cleanup; no guarded callable is replaced. Evidence: native-send-pump-red.xml/log. The correction hands actual pump callers to the existing Send worker before the first native await and retains reader ownership through repeated cancellation; direct and spoken worker callers preserve their original settled result. Final native green qualification follows.


An independent review additionally identifies owner drift during synchronous state publication. Two real native controls (30.44s, normal exit one) hold the original source read, then replace the runtime HookPermissions owner or its original bound reader in on_state. Both original-ready continuations still dispatch, proving the missing final receiver fence. The correction rechecks exact owner/reader immediately before consuming the captured Send, including after explicit review. Evidence: native-send-pump-owner-red.xml/log. Existing source guards remain installed; only the downstream dispatcher is recorded to prevent an unintended provider call in a RED run.

The first complete normal-Send bundle settles with 45 passed / 1 failed, zero errors/skips, normal exit one (692.27s). All five new native pump, repeated-cancellation and final hook-owner controls pass. The sole failure is the unchanged eager navigated Send-button allow-all control: its original five-second approved-dispatch wait expires with no dispatch; the continuation later records SENT at 7343ms during cleanup. This is an unresolved approval latency result, not a green bundle or proof of a lost Send. The original deadline stays unchanged while actual approval/read/publication timing is investigated. The receipt names native-send-pump-green.xml/log identify the attempted qualification and do not imply a passing result.

## First-use callable provenance checkpoint

The async MCP first-use controls first settle with seven failures and one positive control (74.73s, normal exit one). Six failures qualify changed calling contracts: class audit replacement before the helper's first import moved onto a worker; proxy, foreign, inherited and class-overridden screen builders were promoted to the standard keyword contract; and a custom catalog retained an obsolete inventory receiver. The seventh is a new expectation error: original native call ancestry proves three required permission loads, including the downgrade read-modify-write, rather than two. All three original fresh reads are retained.

Definition-time callable references, exact real bound methods/concrete screen receiver, and the previous custom catalog's late receiver resolution address those qualified contracts. The attempted focused completion settles with 56 passed / 21 failed / 196 deselected (352.82s, normal exit one) and unchanged source hashes. Its seven new contract controls, required three-load ancestry positive, 22 selected asynchronous controls, twelve original catalog/kill controls and ten queue compatibility controls pass. One additional queued control stops at its original four-second catalog-entry setup deadline; twenty legacy composer failures require isolated baseline comparisons for config aliases and bare-controller construction. They remain classification gaps, not qualified implementation failures or a green bundle. Independent source review also identifies omitted permission/catalog getters and permission/audit properties in the optimized source qualification; their reproduction and narrow custom fallback repairs remain pending.


### Hook indicator and complete snapshot producer qualification

The old eager Allow-all Send-button case failed in a 46-case native bundle
(45 pass, one failure at its unchanged five-second post-answer boundary).
Passive timing identified overlapping original visit_snapshot and fresh Send
reads, but accidentally recorded a generic contextmanager helper 52,583 times;
those wall spans are diagnostic only. They do not qualify performance budgets.
The original refresh implementation abandoned its to_thread reader on cancel.

Original native held-body controls then reproduce seven specific defects in
58.03 seconds: duplicate physical visits; premature cancellation completion
while actual issued configuration leases stay live; publication after owner,
genuine borrowed reader, accessor or publisher replacement; and indicator work
competing with an accepted fresh Send. Manual review remains a positive control.
Strengthened complete original visit-frame retirement, pending join and
non-error result checks requalify two lifetime failures plus that positive
control in 36.07 seconds. These receipts retain real guards and actual source
operation/lease observations, with no guarded callable replacement.

The fix shares only one in-flight presentation producer, captures strong
source/callback/generation references and drains physical completion through
repeated cancellation. It has no completed snapshot or permission cache.
Manual review's required final refresh remains. Ordinary refresh requests
during Send coalesce into one replay after Send releases its preparation.

Review exposed three correction issues; five custom-worker/callback controls
reproduce incompatible-owner error inheritance, hidden joined publisher failure
and deferred scheduler masking sent/original-error/cancel outcomes. Two more
actual native controls reproduce the same premature retirement in direct Send
and manual review. All seven fail on their intended assertions in 20.75 seconds.
All three remaining snapshot callsites now use the physical drain; matching
joiners retain the original publication error, incompatible source waiters
recapture after complete retirement, and best-effort presentation scheduling
preserves actual Send outcome. The current waiter's cancellation remains primary.
Original synchronous snapshot and complete visit frames are both observed,
since neither configuration lease exit alone nor coroutine cancellation proves
callback retirement. Exact source read-only review reports no remaining
Critical/Important finding at hooks SHA256
CB5B7AD902C07CAE23A8A6518430B7219B42BDC06839A44F8CBC4AB2F0A528B1.
Targeted GREEN, unchanged eager timing and whole-app budgets remain pending.

### Source-passive startup metadata measurement

A separate actual native diagnostic completes in 30.188 seconds with unchanged
source/guards: standard seed18.406 seconds;35 resource reads and35 issued
witnesses;31 durable assets;16,653 native opens and3,923 fresh security reads.
_identity has1,482 calls/5.813 seconds and_check_path105 calls/6.029 seconds;
these nested spans overlap and must not be summed. Final ordinary, pending,
operations, raw, retiring and visual states are zero; one startup source remains
until the child exits. This visual metadata code is unchanged from0ecd.
The preceding original pending-failure stage control timed out at45 seconds
before startup setup, with the actual native stack inside bundled Samira asset
security observations. It supplies no qualified publication-race outcome.

Fresh observations over the actual47 distinct paths use529 scalar opens versus
94 snapshot opens; one selected leaf plus12 fixed ancestors uses78 versus24.
Whole-set median wall time0.219 versus0.187 seconds and leaf0.047 versus0.047
seconds do not establish end-to-end seed speed. An actual7,292-byte native DACL
is accepted by scalar stat and refused by the existing admission snapshot with
ENOTSUP129. Any finite identity batching must preserve that scalar compatibility
and every existing identity/check/lease/retirement, without a permission cache
or changing the admission descriptor cap. Candidate qualification is pending.

Workflow static validation finds61 targeted node references, all real paths,
three native hosts and three original elevated Windows Python versions, with
no continue-on-error. The three-host job ceiling is75 minutes to accommodate
the expanded targeted bundle's measured >52-minute Windows component total;
individual deadlines,20-minute custody job and15s/1s/40000/16 budgets are unchanged.
No final six-job result, tenth whole-app result, fresh live UAT or combined PR
is claimed by these subsystem and diagnostic receipts.

### Qualified complete hook lifetime GREEN

The source-frozen native run hook-refresh-combined-green.xml/log passes all
24 selected cases in157.53 seconds, normal exit0. All eight before/after source
hashes match. This includes18 new physical refresh/snapshot/cancellation/error
controls, five actual mounted Enter/Send/native-owner controls, and the original
allow-all eager Send-button approval test with its unchanged five-second bound.
The latter passes (11.50-second complete call including setup and cleanup);
no approval deadline or assertion was changed. Concurrent visits join one actual
reader, canceled callers drain its complete original native body, stale source
publication is refused, manual review keeps its visit, and deferred indicators
do not mask the actual Send result, reader error or cancellation. Integrated
performance, startup publication custody and final native matrix remain pending.


## Windows workspace-root identity repair (AC22)

A genuine Windows 3.12 root-pin failure was uncovered by the retained original local-review tests: Python reported volume 10718190542197972492 while the same original retained HANDLE projected legacy DWORD volume 3198770700 (same inode 65583669577539523). The original unchanged-root pin refused that complete comparison. The native leaf now reads FileIdInfo from the same retained handle, matching CPython full64-bit volume/128-bit file IDs. It keeps fresh legacy attributes/reparse metadata, CPython-compatible unsupported-query legacy behavior and unchanged full comparison/refusal/cleanup. No expected ID is truncated. Existing ADR 101/32 authority boundaries remain.

Exclusive native evidence: original 3 failures/3 refusal positives became 16 passing cases including actual invalid-HANDLE retirement, changed high identity bits, real junction refusal, all original root-pin controls and the isolated stdlib import gate. Final original local-review module: 58 pass/1 actual Windows 1314 capability skip, with unchanged source hashes; its brief startup overlap is recorded and its duration is not performance evidence. No product loader, primary environment, source guard or watchdog ceiling was changed. The bounded unit harness declares its settings/catalog/prompt inputs at existing consumers and the documented ENV-first unchanged watchdog default.

Source-current isolated-helper verification uses an EvidenceRoot-only same3.12 environment: ordinary -I origins/hashes match managed app/worker/root-pin/filesystem/profile-core and dependency versions match primary. This is a Windows-specific metadata bug; remaining shared performance fixes and exact-head three-host qualification are tracked separately.

Primary sources: [CPython v3.12.10 Windows stat implementation](https://raw.githubusercontent.com/python/cpython/v3.12.10/Python/fileutils.c), [FILE_ID_INFO](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_id_info), [GetFileInformationByHandleEx](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getfileinformationbyhandleex).

Final startup and native metadata evidence:
The genuine standard seed read all 35 bundled resources through their actual issued source/repository/file leases and committed all 31 assets. Its native open count fell from the registered 17,053-open RED to 9,490, below the unchanged 12,000 limit. The frozen 16-control receipt is `samira-observation-final-green-fixed.xml` (16 passed, 46.77 seconds), with actual native ownership details in `samira-native-final-controls.json`. This establishes the seed open count; it does not establish whole-app pause or latency budgets.

The additional custom scalar route exposed a genuine uncertainty-retirement gap: the original rejected-junction opener returned the specific metadata-close error with its real protected HANDLE still alive, but the actual issued visual/raw state was removed. After the exact exception retention repair, eight focused native controls passed in 21.66 seconds (`samira-observation-custom-scalar-green.xml`). `samira-native-custom-scalar-controls.json` proves the original custom callback is preserved, uncertain raw/source custody is retained, bytes are refused, guards/network attempts remain zero, and each test-owned protected HANDLE was physically retired before child exit. All captured production/test hashes stayed stable in both frozen runs.

The post-seed startup run `startup-after-samira-summary.json` contains the three original mounted restore passes and all 15 coordinator passes (18 passed, 10 new failures, 529.78 seconds). The original live readmission used the managed storage module, actual TokenUser ownership and unchanged guards. The ten new failures precede the intended publication check: seven fail the original initial-finalization 20-second wait and three reach the original child 45-second bound. They remain fixture/setup qualification gaps, not authority failures or passing custody controls. A one-node passive stage/await/native census and a narrow observer activation draft are ready; the original bounds remain unchanged.

Final startup qualification: the unchanged original pending-failure node passed on the verified same3.12 source-current isolated runtime (20.33 seconds). The new fixture now installs its profile after construction/setup and selects only the exact current pause's actual startup Thread on the first callback; other new threads restore the prior observer immediately. The held native check-return notifies one captured-loop asyncio.Event, replacing preboundary millisecond polling. Every original guard, revocation, finalization body, thirty-second revoker, ten-second native hold, twenty-second initial wait and forty-five-second child bound remains intact. No production implementation changed during this fixture qualification.

The one cancellation control passed with the actual target/source/refusal/retirement receipt (24.41 seconds). The frozen full ten then passed in 209.72 seconds (`startup-owner-event-ten.xml`, `startup-owner-event-ten-receipt.json`). All ten receipts select the exact managed native startup thread, observe exactly one issued check, record the expected original refusal, restore the prior profile on five other new threads, record zero guard refusals, and prove tracked native retirement. Ten captured source/test/guard hashes remained stable and final ordinary/pending/operation/raw/retiring counts were zero. This resolves the earlier ten fixture setup gaps on this actual Windows runtime; final cross-host and integrated whole-app verification remains the root task's responsibility. The combined runtime/fixture changes are qualified without attributing an isolated timing benefit to either change.

Latest-dev integration at 12d4117b25: 240 passed and five failed in1290.94s. Two direct new-chat visibility waits and one skills output ownership assertion failed. Both original Home cases failed at the explicit exclusive snapshot worker before Resume, with WorkerCancelled masked by the parent's locale-dependent UTF8 log read. These are retained unresolved integration results; no final combined pass is claimed. All native processes settled before the tenth performance run. Production review found no remaining Critical/Important finding; strict JSON exact-integer predicates retain their original behavior with scoped lint explanations.


### Latest isolated native follow-ups and remaining RED gates

- Visible composer mirroring: real held-source RED2 failures/5passes -> fixed10passes184.90s. Actual unchanged composer/stash/session/config/runtime/prefill/attachments/evidence remain equal during the mirror; edit/edit-away-back/generation/store-only changes remain refused. Six source hashes are stable and held source leases plus owned turn tasks physically retire. Full existing module and final budget/matrix gates remain separate.
- UTF-8 private-child failure reporting: original locale reproduction1failure0.41s ->5passes0.49s with actual failed child stdout and JUnit detail. A queued ScreenResume barrier removes the exclusive-worker cancellation fixture defect; original cold/reused Home cases now reach two later product waits and remain RED in138.719s (cold target activation at8s; reused prior-composer restore at8s). No deadline or assertion was changed. Await/code stage diagnostics will identify actual causes before a fix.
- Qualified passive worker diagnostic: three native turns COMPLETED with3user/3assistant rows,3complete traces/3response links and0checkpoints, including streaming turn3. Original Send budget assertion remains RED. All-worker hook cost inflates phases159.682/69.901/116.572s; these are diagnostic only and are excluded from performance acceptance. Natural process exit1/435.67s; a later stop attempt found the verified processes already absent, so nothing was killed. Whole production/probe/helper/guard/plugin hashes stable, all selected spans complete, native code anchors present and profile defaults restored. Source receipts retain actual failure and counts.
- Critical worker native opens27623/26752/13680; redundant global root2000/1129/1129; first sensitive-input derivation13760 on each first two sends, second same-bind resolution688, third send both688. Costs overlap; do not add nested times as wall time. Scope excludes outer worker_guard admission. Native unowned worker opens94432/69244/76984 show substantial other work remains.
- Last committed Windows startup facade counts70488main+46401worker and3.632s max UI stall show seed-only progress is insufficient. Total syscall/content/unique-path, actual usable input, warm separate-process and final three-platform evidence are pending. Existing preflight source suggests warm assets are skipped; this must be observed before alleging repeated warm seed content I/O.
- Workflow source validation now confirms72explicit targets,3ordinary platform jobs and3unchanged required elevated Windows versions, exact-head checkout and no continue-on-error. No new final CI outcome is claimed.

TASK-34404 AC10/11 record whole-launch metrics and finite-source repair outcomes before further code. Tasks remain In Progress; final original performance, actual DeepSeek setup/conversation UAT, cross-platform jobs and combined PR againstdev remain required.

### 2026-10-05 scoped logging and whole-launch qualification

The scoped actual-root repair passed nine real native controls in 51.01 seconds after three late-dispatch failures in 17.39 seconds and a copied-context child failure in 8.38 seconds. The original five-control green was insufficient: later DB-return, source-check-return and containment-return faults could replace instance methods before dynamic dispatch. Qualified calls now retain the original bind/under-bind/migration functions and checked receiver, recheck service writer selection, and freeze source owners. Foreign copied metadata supplies no permission or root; the original worker independently reacquires admission and keeps survivor logging active. Same-actor retired or explicit foreign scope entry refuses. All nine controls positively retire ordinary resources and physically close the SQLite connection. A generator entry/resume counter of 2 is one actual-root execution interval, not two independent permission verdicts. Original scoped ordinary bind still records 15,993 native facade opens versus 17,479 before this leaf; the whole Send budget remains RED.

Existing affected regression bundle: 91 passed, 31 failed, 7 errors in 135.13 seconds. 22 failures are original Windows TOML fixture errors before agent checks; remaining survivor/service-wiring failures have zero provider calls and require an actual pre-provider error witness before classification or fixture repair. No all-regression green is claimed. Differential syntax/lint at the scoped source snapshot: 136 changed Python files, 577 existing baseline/current diagnostics, 0 new diagnostics; later config changes require fresh verification.

Whole startup liveness baseline 2 completes separate cold/warm processes with exact executable/parent-chain custody, actual Pilot key insertion, original normal shutdown and unchanged full production hashes. Cold input confirmation 21.578 seconds, warm 16.883 seconds from parent spawn. Cold import 3.203 seconds / constructor 7.716 seconds / mount to confirmation 10.370 seconds; warm 2.999 / 3.794 / 9.852 seconds. Maximum continuous heartbeat gap 11.359 seconds cold / 7.279 seconds warm spans import/constructor/mount and must not be summed once per overlapping stage. Post-usable settling has no 100 ms stalls in this pair. Both preserve two builtin packs, owner 0, no network/profile-guard refusals. Pilot return confirms insertion conservatively; these are ordinary Windows observations, not three-host qualification or final performance success.

The initial driver-only cold observation (21.328s) had normal exit 0 but rejected Popen identity because Windows venv Python is a redirector. The fresh tiny process qualifier proved the exact driver->redirector->interpreter chain, both executable byte identities, real exit 0/37 propagation and foreign parent/driver refusal. Failed original receipts remain preserved. No identity check was removed.

The separate global Python/C-profile startup I/O census failed the original screen-wait deadline before usable input: incomplete cold only, no warm. Import 13.820 / constructor 43.030 / mount 60.874 seconds are diagnostic overhead, not startup-budget evidence. Original callbacks/source anchors remained intact and profiler hooks restored; 92,806 facade entries versus 92,805 returns is incomplete coverage. No original shutdown-completion claim or whole-kernel/unique-object/read-byte claim is made. Replace broad observation with a qualified lower-overhead observer or scoped host audit; do not raise deadlines.

Cold sensitive-input grouping TDD: 4 expected failures, 11 passes in 151.22 seconds. The cold getters independently reacquire config; callback/helper drift can publish a raw memo; the final original key comparison is outside a finite scope. Warm, preinstalled custom/bound/proxy/helper, environment/cwd and additive-error contracts pass. Dedicated plan and ADR-126 amendment precede production grouping; all original getter/native checks and integrated limits remain required.

Combined draft PR #3023 is open against dev at 3f93fc69f5eb628221bf516b57457860d9dc69a0. Further verified work will be added before ready. Final original Send/UI/open/helper limits, actual three-message DeepSeek UAT, exact final-head native platform jobs and task acceptance criteria remain pending.


### 2026-10-05 current-source ownership and display follow-ups

Frozen actual Windows3.12.10 targeted receipts: cold-sensitive bundle plus body/tuple substitution controls19PASS135.39s; scoped logging plus body controls11PASS55.96s; literal Windows argv product/pure codec15PASS11.49s; original tab-strip18 plus six actual suspended-parent/replacement controls24PASS10.299s; corrected original activation effect-boundary3PASS14.049s (the other119 original logging cases passed in the preceding cohort). All preserve original guards and source hashes. The old setup failures and genuine intended RED receipts remain retained separately.

The standard context-capacity consumer bypassed the issued display snapshot through the real runtime config partial. Exact cold and warm consumers each made an extra original getter/load; direct calls retained their intended fresh route. The repair projects an ephemeral serving target from the issued mapping, pins actual stock functions/receivers/body/global/field identities, keeps the live target memo untouched and refuses post-body drift. Actual55PASS51.94s includes original checked-display controls, cold/warm/direct and17 custom/mutation controls. Raw CRLF pre/post installation hashes are retained separately from the agent's initially LF-normalized hashes; normalized text hashes are explicitly labelled and are not asserted as raw-original custody evidence.

A qualified five-code original SQL observer attributes a worker cache opened by unread_ids_for to the actual acquisition actor. A later unrelated browser callback happens to retire that cache only on executor reuse. Dedicated original unread-row producer RED is1FAIL/new-worker and1PASS/borrowed. The two stock unread producers now use the existing operation-owned worker boundary; in-memory/custom routes and original publication fences retain their existing behavior. Final focused2PASS16.97s directly inspect the retained native object on its owning actor before test cleanup; original manual-mark service/timestamp42PASS39.26s. A strict timestamp-only proof initially failed because adjacent Windows callbacks share a monotonic timestamp; that receipt remains excluded. The direct physical inspection and original callback return ordering qualify retirement, without closing foreign resources.

Passive original tab-navigation observation passes1/47.33s on the combined current-source cohort: all eight expected/live/DOM tab sets match, expected close controls have nonzero regions, all129 actual parent checks succeed, and all22 surface returns match the current store. One real rail refusal is followed by replay reset, a fresh successful surface call and the original wait completing. Selected13/global0/overflow0/unmatched0 and hooks/tool retired; original assertions/deadlines and captured sources remain unchanged. Seven production files differ from the old failed cohort, so this is combined-source recovery, not isolated display causality. The complete original three journeys without observation remain pending.

The original nearest40 CI-prefix cases pass66.66s with the source-cohort diagnostic: exact full app/Tests/profile-core hashes stable, process positively reaped and observer source valid. This does not yet identify the earlier shared CI startup obstruction. Original config compatibility remains15PASS34FAIL; source inspection identifies per-test retarget/early captured adapter aliases, POSIX pipe monitoring on Windows and two unchanged warm-path expectations, but these are not retrospectively counted as passing baseline outcomes. The remaining serialization-pause cause is unclassified.

The original six native acceptance jobs remain unchanged. A separate Ubuntu terminal-settlement diagnostic preserves original third-turn assertions, budgets and ownership; its four pure controls pass. Separate three-host supplemental controls and an actual Linux3.10-floor/bare remote-bundle job qualify the new leaf fixes without replacing original acceptance. Final exact-head matrix, unobserved whole Send/UI/native/helper budgets, whole cold/warm input budgets and fresh real DeepSeek setup/three-message UAT remain required. PR3023 stays draft and TASK34404 stays In Progress.


2026-10-05 source checkpoint: genuine wrapped-getter body2RED then2GREEN (9.040s/9.621s XML; source stable); integrated sensitive-config21PASS155.59s with original native5000 ceiling and physical/census retirement. Defining getter metadata now pins actual body/code/globals/defaults plus body/wrapper closure cells at decoration time; custom prior replacements retain the ungrouped original route and qualified mid-build changes refuse before execution/memo publication. Unread source provenance2FAIL/2PASS21.80s then4PASS20.172s XML/source stable proves pre-UI class replacement and same-function body mutation no longer inherit stock worker closure. The stock reader is retained in its defining module and rechecked at actual worker entry; memory and instance overrides retain their original lifetime. Both repairs reviewed without remaining concrete source blockers. First unread4setup errors are excluded and its parent-only bootstrap marker correction does not alter the independent real child profile or native guards.

Original whole tab journeys remain1PASS2FAIL236.63s; no GREEN is inferred from the prior observed navigation pass. At the original failure/retry ten-second deadline, actual original sync Task is alive, uncancelled, global0/source stable and awaiting character_context.refresh_presentation_if_scope_changed at the shielded owned-task await. Original task/await-chain attribution is qualified; the owned child stage remains pending. The actual source-bound observer retains original assertions/deadlines, strong task/actor/code refs and explicit unmatched exception gaps. Separate exact-prefix Ubuntu/shared-startup and ordinary cold/warm three-OS diagnostics are now tracked; original six acceptance jobs remain unchanged. Installed tiny startup helper/process qualifier passes8.08s/noApp/source unchanged. Forced timeout descendant custody remains unqualified, so no complete startup-custody acceptance is claimed. Differential AST/Ruff against dev179Python files:639baseline/639current diagnostics, zero new. All final integrated budgets, remaining original failures, native matrix and real private DeepSeek UAT remain required; PR3023 remains draft.

2026-10-05 scheduling and diagnostic-launch verification: original Linux/macOS supplemental controls each104PASS/1FAIL isolate refused readiness worker scheduling. Four actual refusal/cancellation cases fail before repair (3.50s), then all58 scheduling/original projection/checked display controls pass38.84s on genuine Windows3.12.10, full source stable. Refused refresh/publication coroutines close and pending/event state retires; cancellation propagates after actual worker retirement. Original hook controls41PASS262.73s source stable. Fourteen pure current diagnostic repairs pass0.379s; nine namespace controls pass with exact stock PEP420 identities. Original test assertions, guards, budgets and native acceptance jobs remain unchanged.

Valid unprofiled original whole startup remains RED: cold22.332s/warm16.716s to input, mounted heartbeat0.780s/1.124s; both normal exits, actual ancestry, complete original source receipts. Separate passive config/key diagnostic directly observes original insertion gaps2.784s/1.868s but its timing is diagnostic only. Direct three-getter startup ancestry shows464 native facade entries per existing independent getter. The whole-send observer itself breaks stock source qualification (stock before=True/during=False/after=True); passive replacement is in progress, without weakening source guards. Original tab transaction-stage observation fails its unchanged settle while awaiting readiness projection settlement, no live matched database frames; no current SQLite cause is inferred. PR3023 and TASK34404 remain draft/In Progress pending original integrated acceptance and real DeepSeek UAT.

### Workspace scope worker connection repair

Source-qualified Library cleanup attributed the remaining Workspace worker lease to LocalWorkspaceRegistryService.get_workspace_scope through the original executor callback. Three native new/error/cancel controls failed on the actual live SQLite object and exact lease; borrowed, memory and custom controls passed. The existing operation_owned_connection boundary now covers only that synchronous producer interval. All six native controls pass (17.35s), including borrowed transaction preservation and retained running-worker custody after waiter cancellation. The unchanged original Library delete/undo journey passes (37.73s); its bounded original factory witness observes all six declared owners, successful physical cleanup and no refusal, with current sources, no invalid evidence and retired local hooks. This proves that producer repair, not overall startup/Send budgets or the original complete-cohort drain. ADR required: no; existing finite ownership API.

Retained receipts: workspace-scope-owned-native-red-1 (3 FAIL/3 PASS), workspace-scope-owned-native-green-1 (6 PASS), original-library-workspace-producer-green-1 (original journey PASS and exact factory retirement). The previous popup interleaving failure is preserved and this combined-source pass does not establish an independent popup cause. Original whole startup, Send/UI budgets and final matrix remain pending.

Native follow-up evidence (uncommitted source checkpoint): Windows absent-process registry controls reproduced two failures/four passes before the one-expression repair and six passes after. Stock MCP callable metadata and earlier inner-await refusal controls pass all20; unchanged catalog/source/offload compatibility passes52. Workspace operation-owned producer lifetime passes six controls, and the receiver-retarget regression reproduces a query/ownership mismatch before the captured-local repair; all seven final controls pass. Original Library delete/undo passed on the prior stable-receiver producer repair; integrated final-source rerun remains required.

Automatic-work limit grouping was rejected by measurement: original warm six getters perform zero native opens/operations (.00022s), whereas explicit grouping adds1279 native opens (.18950s). Invalidated cold has1543 opens/two config operations originally vs1570/three with grouping. These are bounded getter-phase diagnostics, not whole-Send timing. No production grouping was applied.

Ordered original Fleet evidence establishes timeout cleanup consequence: original host start is followed exactly10s later by fixture waiting_run cleanup removing its cancel entry, then original worker observation78ms later and guard check109ms later. No production cancel pop occurs. Its later cancel guard refusal therefore cannot establish the cause of the preceding arming delay. Separately, two exact sibling AgentRunsDB native handles/leases remain open at sandbox deletion; fixture runtime ownership repair is still pending.

Prepared fixture isolated helper controls pass5 with exact physical new-handle retirement, borrowed-transaction preservation, request/disposal cancellation custody, terminal reopen refusal and source stability. Integrated async fixture rerun remains RED at the original arming timeout; it now observes exact request-worker and runtime creator close returns, retains the original AssertionError, and refuses sandbox deletion with a cleanup RuntimeError. No zero-before-delete or complete shutdown claim is supported yet.

The fifth initial-send observer positively qualifies the actual inherited Textual Button.press on exact ComposerControlButton and the original MessagePump parent property. It observes one complete press pair, mounted/displaytrue/disabledtrue at both edges and zero send-handler entries. Earlier zero-press observations were filter mistakes (exact base class and dictionary lookup for a property); they are excluded from Send-state attribution. Source/body/runner/bootstrap flags are current, global monitoring events zero and local callbacks retired. This establishes pre-dispatch disabled state for that original fixture, not the reason for disabling it or a real trace-capture error.

Initial-send state capture6 qualifies the original inherited button call and actual readiness owners. The selected model changes the settings revision while the previous setup-blocked readiness result is pending; the actual press is disabled before dispatch. The current checked control-bar refresh clears that setup block about0.22s later. No send, trace coordinator or provider error handler enters. This is intended pending-readiness projection; the local fixture must await its real completion within its existing startup allowance before issuing its original press. It does not establish a persistent product block or justify forcing the button enabled.

Prepared Fleet cleanup capture5 qualifies the exact owned Runtime.dispose return and creator-finalization entry. Both retained AgentRunsDB handles are physically closed and its connections/operations/pending acquisitions/path leases reach zero. The CharactersRAGDB creator remains open with two operations and two overlapping pending acquisitions admitted during disposal. The original static prepared_close_database_not_retired refusal preserves the sandbox and primary arming timeout. Exact origins and retirement of those pending chat operations remain unresolved; no deletion, full shutdown or final cohort success is claimed. The earlier capture4 missing-ast-import setup failure is excluded.

Current local selected-readiness fixture Native1 passes eleven pure ownership controls and two unchanged in-flight controls; all three original real Resend journeys remain RED at the preserved shared two-second startup allowance (37.94s driver, sources unchanged). No enabled press or provider dispatch is claimed. Requiring whole-sync completion may over-wait unrelated presentation; actual current checked readiness completion remains the prerequisite and the original budget remains unchanged.

Original Fleet request-stage capture6 qualifies both actual get_run and private connector entry with 309 bounded local events, current source/closure/alias ownership, zero global events/overflow and retired hooks. First get_run takes 10.141s inclusive; its repository admission acquire_storage boundary accounts for 5.766s and the nested private-connection admission boundary for 1.063s, while the actual raw SQLite connector body is 0.062s. A second get_run takes 0.047s. These nested spans overlap and are not additive or CPU measurements. The original arming timeout remains RED; the first ordinary admission/setup, rather than migration or query execution, is now attributed. All guards and waits remain original.

Final captured-Workspace Library Native2 passes the unchanged delete/undo journey (52.73s pytest,63.36s driver). Its exact constructor-owner observer qualifies all six declarations, six original settled-close True returns and empty connections/operations/pending/path leases/retiring state. No refusal or invalid evidence; global0, local hooks retired and original exit0. This validates final Library resource retirement; its timing is diagnostic and does not satisfy broad startup/Send performance.

Selected-readiness fixture v2 removes the unrelated whole-sync coroutine/flags prerequisite, retaining only the actual current projection refresh/pending and checked control-bar True. Eleven pure controls pass, including unrelated held whole-sync success. The real three journeys still fail within the same shared two-second startup budget (59.20s driver, sources stable), with no provider dispatch. One remaining value is -1.11s, so test-driver settlement inside the bounded pause must be qualified separately before calling this a product readiness duration. The original budgets and provider assertions remain intact.

New-test formatter check found ten unformatted files; all ten were formatted with complete AST equivalence proven per file in new-tests-format-checkpoint-1.json. Earlier native receipts retain their exact preformat source hashes. Full final-source native and matrix acceptance remain pending; no raw-byte equivalence is asserted.


Current selected-readiness v3 fixes a reproduced driver defect: original Pilot.pause awaits every screen descendant before its requested pause, so its two-second prerequisite can over-wait unrelated work. The fixture now yields with asyncio.sleep while checking only its actual current projection and checked control bar. The new control compiles and gates the installed original Pilot.pause body; v2 reproduces the over-wait and v3 plus existing controls pass12. Native3 reaches the unchanged Send path: two original bodies pass, while the click retry fails its unchanged expected-provider-error text assertion. Its aggregate DOM text contains a capture-blocked placeholder; later source-qualified diagnostics observe no capture-error entry and do not establish actual capture refusal. All three teardown paths remain RED at test_factory_database_not_retired. Sources remain unchanged; no final retry, cleanup or performance acceptance is claimed.

Prepared Fleet Characters-origin diagnostic7 is source-qualified with retired local hooks and zero global monitoring events, but its disposal-only window does not cover the two pending operations that already existed at disposal entry. Their origin flags remain false. It positively observes a separate dispose -> detach_view -> recompute_console_attention -> list_console_unseen_marks operation retire before disposal return. That retired operation cannot be blamed for the remaining live resources. The diagnostic preserves the original arming timeout and refuses deletion; earlier absence of pending operations at disposal entry was not stable. Observation must begin at the exact fixture owner handoff to cover earlier admissions.


Checkpoint309b27bbde is committed/pushed to the single draft PR3023 against dev. Differential static221files: baseline659/current658/zero new diagnostics; all25 new Python files formatted and diff whitespace clean. The unchanged original native probe14 preserves sources and returns3provider turns/3complete traces/3response links, but remains RED: Send30.911/28.730/21.941s vs15s; typing7.250s and startup heartbeat3.517s remain over their limits. These are original recorded budgets, not relaxed acceptance.

Exact non-chat fixture Runtime regression reproduces its constructor-owned watcher left active after the original context exit (14.61s). After retaining that exact fresh Runtime on its original loop and finishing one shielded disposal, all9 genuine native controls pass34.56s, including the original fixture route, repeated cancellation, physical watcher/read retirement and wrong owner/thread/loop refusal. No database is adopted or closed by this helper; the existing prepared DB helper bytes remain unchanged. Original integrated Fleet and complete-prefix outcomes remain pending.

The narrow capture-handler diagnostic7 records zero actual entries, current source/global0/retired hooks. Button state diagnostic8 positively observes a complete original enabled press with current selected projection, then actual send/queue/preparation/admission/runtime/controller entry. Its later row coverage overflows3275 events, so later span absence/durations/return state remain unclassified. This gateway intentionally lacks durable-capture support, and CAPTURE_OFF explains why zero durable-trace handlers does not exclude a later unrelated send refusal. No capture failure is attributed from hidden placeholder text.

New notification and typing observer first launches are excluded setup failures: original pytest assertion rewriting was not accepted by notification source qualification; Textual's lazy version field was incorrectly read from its module dictionary by typing qualification. They retain original failed receipts and restore the temporary isolated diagnostic path. No actual notification delivery/expiry or keyboard-idle timing is inferred from those runs.

Storage coordinator native proof repair: the three unchanged-body Windows controls reproduce the shared RLock blocking an independent original lease close before the edit (storage-coordinator-native-red-1). The raw registry read and nested operation path checks execute inside the outer coordinator; all three actual native reads return and physical SQL/native ownership retires before each causal RED assertion. The repair captures synchronized dependencies, performs original fresh I/O outside the coordinator and repeats exact actor, pending, installed operation, pause, selection, hold, continuation and evidence-epoch checks before publication. ADR-126 records the amendment before implementation.

Final formatted-source storage-coordinator-formatted-native-green-2 passes all3 controls (16.75s driver, source unchanged). storage-scope-publication-native-1 passes6 actual held-return races: cancellation, pause, selection change, retired continuation, a valid saved binding after unrelated retirement and revoked installed operation. The original compatibility selection passes53, skips41 platform cases and fails1 Windows related-symlink fixture. The bounded unchanged-body diagnostic original-related-path-refusal-native-2 qualifies the original fixture's Path.symlink_to exception errno22/winerror1314 before its containment proof; it does not establish a production refusal regression. POSIX and elevated Windows matrix verification remain required. No guard, callback, persisted enrollment, check freshness or budget was relaxed.

Integrated Fleet native8 preserves current source and reaches the original later fleet-close scenario. Its sole child failure is the unchanged cancelled.is_set() and round_task.done() settlement assertion at test_console_session_tab_close.py:1035; no constructor-resource cleanup failure is reported. The diagnostic selected no applicable fixture owner, so its false source/coverage flags and empty rows provide no origin attribution. This is not a Fleet pass or a timing pass.

Typing observer v2 qualifies actual Textual8.2.8 original bodies and probe screen-wait wrapper, current source, zero gaps/overflow/global events, all normal returns and retired monitoring/runtime path. Eight original Pilot key presses enter16 original wait_for_idle(min_sleep=0,max_sleep=1) waits. Those waits occupy9.389s inclusive within _press_keys9.394s; screen-settlement adds2.092s. The spans overlap and are not summed or exclusive CPU. This diagnostic ran concurrently with read-only differential static analysis and cannot qualify an overall performance limit. It proves the measured typing phase includes test-driver process-idle waits; it does not establish real terminal keyboard latency or excuse heartbeat/Send failures.

Exact309b CI RemoteBundle controls10pass/1 artifact reproduction failure identifies stale generated config anchors. Regenerate with the original tldw_chatbook.Tools.build_remote_worker_bundle, preserving the original artifact-reproduction assertion and bare-worker contract. Focused final-source reproduction and next platform checks are pending.

Original artifact reproduction node cannot collect on Windows because its unchanged POSIX-only module imports fcntl. The failed local selector is a collection limitation, not a bundle result. Original builder --check passes against the regenerated artifact; the unchanged Ubuntu RemoteBundle suite supplies the actual node/stdlib/bare-worker verification at the next pushed head.

Original boot census and scheduling-policy selection passes (98.797s driver, current sources unchanged). The observed conditional (_sync_native_console_chat_ui, console-sync) pair now names its same-owner readiness whole-state replay; mandatory membership, policy concurrency and every time/module budget remain unchanged. Earlier static225files reports baseline659/current658/zero new diagnostics; generated bundle and inventory final differential verification follows.

Follow-up differential static225files: baseline659/current658/zero new diagnostics, including the regenerated artifact. All four new native/helper Python files format clean. The two existing modified fixture/census files were formatted with complete AST equivalence proven in followup-format-checkpoint-2.json; native receipts retain their exact earlier test-source hashes. The original lock3 and publication6 controls exercised the final production coordinator bytes. Next-head CI and original unobserved whole probe remain pending.


### 2026-10-05 recovery repaint and exact-owner follow-ups

The original mounted recovery repaint reproduces three ensure/core/settings
entries for one display sync (23.922s, unchanged source). The established-controller
display accessor removes those three execution-state refreshes while preserving
the original getter results. Final formatted source passes the mounted control,
including replaced property, custom ensure and stale view-generation refusal;
explicit recovery actions still use the original live ensure path. The first
partial wiring attempt still entered ensure once and is retained as a failed
checkpoint, not success evidence. Existing live-binding and trace-action controls
pass. The original real-screen continuation node also passes after opting into
its required private bootstrap fixture (25.656s); the earlier failure occurred
in raw config selection before the recovery action.

Repeated historical cancellation reproduces early release/rearming on double,
triple and borrowed double cancellation after the unchanged original callback
executes real SQL. Single cancellation and normal publication pass the baseline.
The repair continues shielding the same callback until physical retirement,
then propagates cancellation. All five actual native cases pass33.141s. The
initial draft's blanket no-closure qualification rejected the original PEP695
generic run_owned_db_call before the causal assertion; that setup receipt is
excluded. The qualifier retains the exact original type-parameter closure and
compiled defining body instead.

The actual constructor EvalsDB was outside the factory-owned directory and
absent from its six-owner inventory. Two native controls retain its real handle
through original teardown and reproduce deletion starting with the handle open.
The factory now selects its private Evals path before construction and records
that exact seventh constructor owner. Retained, replaced-field and borrowed
negative controls pass29.968s. The unchanged original normal and queued MCP
wiring nodes followed by original accepted-catalog custody now all pass
66.234s in one real shared cohort, with source hashes unchanged. This verifies
the evaluation-resource leak rather than attributing every earlier cleanup
failure to it.

The Character rail control preserves the identity of its actual warmed issued
display projection. The corrected baseline fails both identity assertions; the
checked render callback passes both after routing rail calculation through the
existing display scope. An earlier equality-only control passed the original
source and does not qualify as RED. The formatted follow-up cohort passes52
cases but retains two original failures: private-bootstrap setup (subsequently
repaired and independently verified above), and a standalone final Enter event
that has not yet been causally attributed. No overall timing or completion
claim follows from these targeted results.


Notification source-qualified native v3 completes all original handler, enqueue,
reap, delivery and add spans with current sources, no global events/overflow and
retired hooks. The test inspects age0.342s before delivery at age0.499s; the
original12s timeout has not expired. It fails on the empty collection. The test
now awaits its actual original asynchronous delivery through the existing20s
poll helper before its unchanged message/privacy checks. Both original content
and modal callback journeys pass83.406s with unchanged sources and normal quit.
This repairs a test prerequisite, not a production notification timeout.

The original final-Enter standalone observer passes21.969s. Both original
action_press entries see active=false and invoke original press; all pairs and
hooks retire with current sources and no overflow. This does not attribute the
earlier sporadic missing second action to the Button timer; no timer or fixture
behavior was changed from that hypothesis. The broader original retry cohort
remains RED: two selected-readiness2s limits, a missing response and one actual
constructor-owner teardown refusal (15 cases,12PASS3FAIL1teardownerror,158.297s
driver, sources unchanged). Exact outstanding owner attribution is pending.

Exact8c cross-platform original whole receipts from push run37355120791 retain
all1325 loaded production hashes equal to8c blobs. macOS Send5.119/3.316/2.466s
and Ubuntu7.120/4.240/4.153s complete all three responses/traces/links, but complete
remainsfalse. Helper counts macOS38/15/15 and Ubuntu51/17/25 exceed16 where
shown; startup/main-loop gaps remain over their original limits. These are
actual platform results, not simulated Windows behavior. Windows original
whole and final-head acceptance remain pending.

Corrected binding qualifier native RED2 completes both original binding calls
and native retirement in8.343s. Opens152/152 exceed structural bound63 over24
actual resolved union nodes; source-current, local/global hooks retired, no
invalid observations or network attempts. Earlier native1 failed setup because
the draft pinned a different directory wrapper and is excluded. The production
candidate remains unapplied while its immediate-parent policy is corrected.


Windows binding batching is now installed and independently reviewed against the
actual private-path parent walk. The original native work-count plus all five
permission/source/custom-reader/uncertain-close controls pass (13.344s driver,
unchanged sources). Both complete binding calls stay within the original
structural union bound; this is leaf work-count evidence, not a whole-app timing
claim. A final defining-reader fence also refuses changes during parent-policy
checks. The existing compatibility selection passes82 cases with one custody
skip; two further cases stop at host limitations before relevant binding work
(WinError1314 symlink privilege and an unchanged POSIX chmod assertion). Native
Windows changed-ancestor, grouped-observation and handle-retirement controls
pass. Original end-to-end limits remain pending.

The original retry's exact seventh constructor owner reveals two genuine
Workspace worker handles opened by the hook-key consent/get_workspace readers;
the hook callback had only scoped the separate Chat persistence database. Nine
source-qualified native callback controls reproduce five ownership failures
(normal/error/cancel/custom-reader/retarget) while four supported ownership
controls pass (50.937s driver, unchanged sources, no guard violations). The
repair proposal is undergoing review and native acceptance. Review caught that
an outer lazy ownership context alone cannot make two nested readers share a
handle; the qualified stock composition must retain an actual supported
connection interval as well. The distinct availability-cancellation callback
lifetime is still a separate hypothesis.


### 2026-10-05 finite callback and readiness evidence

Hook grouping now retains one actual supported Workspace connection across the fresh original readers. `hook-workspace-one-handle-native-green-2` passes all12 cases (59.468s), including one stock physical handle, body failure/repeated cancellation/retarget, borrowed transactions and three inherited-alias fallbacks. Sources remain unchanged. Review caught the lazy-scope-only draft before acceptance.

Character finite batching retains initial capture/REFRESHING publication. Frozen-candidate entry tests reproduce five changed-body executions before refusal (19.140s). Exact captured readers with post-admission pre-entry fences repair the gap. `character-finite-batch-entry-source-native-green-1` passes all31 work-count, source/error, borrowed/custom and queued-entry controls (84.828s, current sources).

The first availability observer fails setup only: Python3.12 rejects local PY_UNWIND. With supported local events, original Native callback tests reproduce five early-release failures and two current successes (61.594s). Repeatedly shielding/draining the same captured task until physical callback retirement passes all7 in `workspace-availability-same-callback-native-green-1` (58.812s). Named/Default success, fresh retry and borrowed transactions remain.

Owned public Workspace readers expose another cold-composite cost. `workspace-cold-composite-native-red-1` reproduces two physical handles for admit/status/Inspector capture/save-binding, while three borrowed controls pass (30.250s). Original results/queries, native closes, leases and observer retirement qualify before each work-count assertion. Composite grouping remains pending; do not undo the public reader cleanup.

Original Character compatibility has91 passing bodies,16 bootstrap setup errors, one premature DOM assertion and one unassigned factory teardown error. Four existing-bootstrap marks and exact current idle widget/DOM readiness preserve original40-by-.05 bounds. `character-original-readiness-prerequisites-native-1` passes all18 selected bodies (27.875s); the mount node retains a separate constructor-owner teardown refusal. Its first adapted exact-owner witness fails selected-module setup and is excluded, not origin evidence.

Exact47a9 Ubuntu trace evidence reaches its durable oracle with the third settlement callback still owned; prepared/claimed/store-run finish31-34ms later and teardown drains tozero. The probe now awaits original pending work inside the unchanged15-second Send phase before rereading durable states/links. `trace-probe-original-retirement-native-green-1` passes success/expiry/cancel (22.468s) with actual native return held through physical handle/lease/registration retirement. The earlier three children completed their proof but lacked the shared harness's normal success marker; only that final marker was added. Original whole timing limits remain pending.


`original-retry-current-exact-owner-native-4` now passes the unchanged refused-echo retry and teardown (35.828s, sources unchanged). Its exact seven-constructor-owner v3 witness qualifies the actual bootstrap body, original code and close frames, zero invalid/global observations, retired hooks and no original close refusal. This verifies the original leak path after hook/availability corrections; it does not excuse other readiness or timing failures.


Current four-callback checkpoint checks242 changed/new files against dev: baseline734 diagnostics, current733, zero new diagnostics. The original persistent diagnostic inventory reports no drift (652 owners/1444 TASK492/56 TASK31551/7616 TASK494/16 sinks), and the original remote-worker bundle check passes. Formatter changes to the two performance tests preserve their complete ASTs. These static checks do not establish responsiveness acceptance.

The source-qualified original Character mount and canonical-writer exact-owner controls pass individually (21.328s and22.765s). All seven constructor owners are recorded and original close rows retire with no refusal. These individual runs do not explain the earlier cohort-only teardown failure; that remains unassigned.

Cold receipt controls1-3 are setup-only and excluded. The latest captures the actual Inspector-to-bridge-to-receipt constructor chain, but its exact-base-class prerequisite excludes the original admitted native SQLite subclass. A false exact-type test alone provides no facade attribution. No actual hold was qualified, so no startup causal RED or production startup repair is claimed. The next control must qualify the original admitted connection and lease lifetime.


### Fresh live DeepSeek UAT on ded197c0e18c

A new ordinary-user private profile completed initial setup, selected DeepSeek and exactly deepseek-chat, saved those defaults, and carried them into the unused Console chat. Three real user messages received Ready, cedar, and cedar THREE TURNS OK. The app quit through its normal Ctrl+Q action; the exact recorded host then exited normally. Read-only post-quit verification finds three complete DeepSeek/deepseek-chat calls, three durable assistant replies, three revision links with verified_equal, and zero pending dispatch checkpoints. Production sources and the original owner's config hash remain unchanged. Evidence and the proof image are retained under final-uat-ded197c0e18c. This positively verifies capture for this fresh conversation, not every refusal/retry path.

Existing cumulative Send diagnostics complete at13.483/10.000/9.422s. Provider entry begins at12.092/8.937/8.156s; much of the elapsed work therefore precedes the live provider call. The interval from durable-commit success to reservation entry is7.813/5.578/5.000s. These existing stage coordinates do not attribute exclusive cost to a particular helper. Overall original native/helper/heartbeat/startup limits and final-source platform acceptance remain pending.

Browser keyboard observations also retain unresolved transport behavior: lone Escape did not release the settings text field in this session, and Ctrl+2 did not navigate. F6 plus the existing save action and the navigation palette completed the journey. No keyboard product fix or cross-platform attribution is claimed from this single browser observation.


Workspace composite candidate4958 remains an uncommitted qualification checkpoint. The first seven additive controls fail only their outer harness argument and are excluded. After adding the required ownership outcome to both outer wrappers (embedded bodies unchanged), workspace-composite-captured-gaps-native-red-2 completes35.563s with six genuine product failures and one borrowed-handle pass. New-empty retarget retains A's native handle/lease; new-foreign additionally closes and modifies the unrelated B handle. Four custom Consent/Inspector lookup/descriptor routes each add one A connection despite preserving two original B readers and literal B outcomes. All seven receipts positively retire exact test-owned resources, leave zero worker leases/registrations, no invalid monitoring observations, current sources and retired Thread/hooks before product assertions. The candidate must be corrected before commit.

Cold startup attempts4 and5 remain observer-prerequisite failures. Diagnostic5 positively identifies the actual original close closure's additional RecoveryRequired cell; the observer expected four cells rather than the original five. Original admitted class/MRO/source and normal disposal qualify, but no held SQL callback does. Correct the qualifier against that exact defining exception class before any cold causal or startup repair claim.

The corrected original cold receipt control (`original-cold-receipts-native-red-6`) establishes the startup cause. It qualifies the original admitted native SQLite subclass and all five close-closure cells, then holds the actual receipt-schema callback reached from Console composition. The operation runs on the actual UI Thread/shared loop; the loop cannot progress during that hold. Exact physical handle and lease, Runtime disposal, source and observer retirement all pass before the original causal assertion fails (31.375s driver). Attempts1–5 failed observer prerequisites and remain excluded. Startup repair remains unaccepted pending independent review and actual controls.

Exact9d69 Linux Perf Guard run37373444655 passes its UI-latency and boot checks but fails the unchanged original Console storage-unit ratchet. Typing-pause helpers are5/7 against3, opens232/258 against206x1.05; credential-poll opens17.5 against6.75x1.05. These actual Linux effects establish additional cross-platform costs; source/callback attribution remains pending. The current ded197 platform runs are still pending, and these census phases do not establish exclusive elapsed-time attribution.

Captured Workspace correction a261c23b is now verified by `workspace-composite-captured-fix-native-green-1`: all21 actual native work-count/fallback/source/retarget/borrower controls pass (104.563s, unchanged App/Tests sources). The four stock cold routes each retain one physical connection; exact new-handle retirement occurs after the original counted interval, with replacement caches/foreign handles preserved. The three new test files pass formatting and lint checks. Existing service modules report79PASS7SKIP18FAIL (124.422s, current sources); all failures are in the file-inspector module, so this compatibility run is not green. An additional actual-stdlib/source receipt establishes that Windows lacks O_NOFOLLOW/O_DIRECTORY/O_NONBLOCK and the original no-follow descriptor methods are AST-identical to ded197. It explains that prerequisite refusal, not every Inspector failure; no filesystem authority was weakened or unsupported expectation suppressed.


### Qualified three-platform startup checkpoint at 9d69edcf

Actual ordinary macOS15, Ubuntu24.04 and Windows2022 jobs loaded the tested commit's source. Qualification compared every imported file's raw SHA256 against Git blobs, allowing only the declared LF/CRLF checkout variant, and separately recorded the single namespace package. All six hosts completed normal physical process exit with stable source and head; receipt: `ci-9d69-startup-qualified-summary.json`.

| Actual platform | Cold usable / limit15s | Warm usable / limit10s | Cold / warm mounted heartbeat gap / limit0.2s |
| --- | --- | --- | --- |
| macOS15 | 14.249s | 8.974s | 0.355s / 0.257s |
| Ubuntu24.04 | 14.808s | 8.290s | 0.373s / 0.350s |
| Windows2022 | 23.158s | 10.751s | 0.867s / 1.341s |

Startup duration passes on the two POSIX hosts, while all three platforms fail the original event-loop pause limit. Windows also fails both duration limits. These original measurements precede the uncommitted Workspace and proposed receipt startup repairs; they are evidence of shared pauses, not acceptance of the later candidate. The key-body latency gate was not attached in these pairs, so Pilot wait bounds are not keyboard latency evidence. Separate duration phases are inclusive observations and do not provide exclusive cause attribution.

The original Linux storage census at the same commit remains red. Its current observer replaces `config_participants.operation`, which invalidates production source-identity qualification and makes optional batching decline. A passive original-code observer is being qualified; this source effect alone does not attribute every measured helper/open. The Windows supplemental journey has117 passes and two failed original session-close tests; both remain unassigned rather than being hidden by passing leaf controls.


## Stock admission and startup native checkpoint

The finite stock run-turn fold passes all six actual native controls in51.000s with current source. Stock scope starts fall9→5 for the same five original owner/path keys and one physically retired worker SQLite resource; the custom selector retains9starts and its original arg-free call. Root/method/source/pause changes still prevent provider and log effects, and all original leases/operations/monitoring physically retire. The predecessor red-1 is a frame-selection setup exclusion; red-2 is the qualified causal failure. Original agent/log compatibility is running; this count change does not establish whole latency acceptance.

Startup v3 passes all24 native ownership/custom/binding/drift/cancellation/disposal leaf cases. The original cold control's receipt positively qualifies the actual startup-task-issued worker, original admitted native SQLite/lease, responsive UI callback under held SQL, normal Runtime disposal and exact physical resource retirement with current source and no invalid/overflow evidence. Its immediate ChatScreen assertion fires before the new asynchronous initial push completes; this prerequisite needs synchronization inside its original240-second outer deadline, followed by a fresh run of all unchanged assertions. The189.453s combined run is24PASS plus this one prerequisite failure, not an original gate pass. Original whole/platform budgets remain open.


### Accepted native stock/startup checkpoint

Implementation Notes: stock admission grouping passes6/6 native controls51.000s and all62 appropriate original activation/scoped-log/body-binding/service-wiring tests380.625s, with source unchanged. The original cold Console control passes53.750s after the separately reviewed completion-latch prerequisite, retaining its actual original SQL hold, source/currentness, typing, disposal, physical handle/lease retirement and all original assertions/deadlines. Its worker is issued by the actual original initial task and loop; UI progress occurs while the original admitted native SQL is held. The24 startup native custom/binding/owner/source/cancellation/disposal leaves already pass. No whole-startup/Send/heartbeat/helper/native-open acceptance is inferred from these controls. Census exceptional generator accounting remains an observer prerequisite under repair; status stays In Progress.


### Exact ded197 completed POSIX whole-probe qualification

Run37373809908 completed the original whole probe on Ubuntu24.04 and macOS15 at tested head `ded197c0e18cfc32ae2cc1806534e10bb11b7e31`. Each receipt records1,325 loaded application source files; all1,325 match that commit's Git blobs with only declared LF/CRLF checkout normalization. The reported GITHUB_SHA `b0c613671e1078a1efcafe2b45e8aedf0402f4c1` is merge-job metadata, not a substitute for loaded-source qualification. Both hosts report three provider calls, three complete trace states, three verified response links, zero pending dispatch checkpoints and a file-backed database. Each whole module is7PASS/1FAIL and `complete=false`.

| Host | Send1 /2 /3 seconds, original limit15s | Send heartbeat maxima, original limit1s | POSIX helpers, original limit16 per Send |
| --- | --- | --- | --- |
| Ubuntu24.04 | 3.074 /1.862 /1.533 | 0.147 /0.034 /0.027 | 37 /16 /17 |
| macOS15 | 4.621 /2.455 /1.999 | 0.301 /0.058 /0.061 | 41 /16 /18 |

The actual original failure is the first Send's helper-count cap. The recorded third Send also exceeds that unchanged cap on each POSIX host. Main-thread acquisitions are43/32/37 on Ubuntu and45/32/37 on macOS, below the original200 limit. Startup/idle/typing heartbeat maxima are0.980/0.209/0.183s on Ubuntu and0.880/0.356/0.202s on macOS. These are the recorded phase observations, not startup keyboard or physical cleanup proof. The Windows whole receipt is unavailable from this run; no completed Windows whole metrics are inferred.

Both observers record15 selected codes, zero global events and overflow, unchanged original bindings/bodies, and removal of local callbacks/freeing the monitoring tool. The receipts lack imported module-name/spec/raw-origin maps and an explicit final native-resource/physical-host-exit census. Their original producer resolves actual imported module file origins under the checkout and hashes those files; the source comparison verifies that stated file-level contract only. POSIX `_Native.open_handle=0` means the Windows seam is absent, not that OS file opens are zero.

Qualified helper callers account exactly for all phase counts (`ded197-helper-callers-qualified.json`,36 original defining source files). First Send includes six historical-fleet callbacks on each host, three Character metadata starts, two browser-scope starts, two/three badge starts, one/three files-availability starts, four distinct trace-maintenance callbacks and other named singletons. Third Send includes two browser-scope and two archive starts plus separately recorded history/badge/files, policy, durable-write and four trace-durability stages. Counts are partitioned actual helper entries; inclusive timing rows and concurrent global phase labels do not establish exclusive cost or a required Send dependency. Historical callback already owns one finite connection, so its six starts require schedule/invalidation evidence before a batching or cache proposal.

Receipts: `ci-ded197-whole-review-37373809908/ded197-whole-qualified-summary.json` and `ded197-helper-callers-qualified.json`. This preserves the historical checkpoint and claims no acceptance for the later Workspace/startup/stock-fold source or final whole/platform budgets.


### Passive original census observer checkpoint

The original storage census previously replaced the configuration admission callback, invalidating stock source selection and declining optional batching. Local original-code observation now preserves the captured callbacks and their defining sources. Python3.12 exceptional context-generator retirement is proved by the same still-live original generator having no frame, rather than inferred from a dead weak reference. `original-credential-passive-census-native-3` passes the eight unchanged original credential calls and their original ceilings in75.250s with current source: one configuration admission, two storage admissions, zero helper starts and zero audited os.open events. All628 spans complete (625 normal returns plus three positively closed generators); no unresolved spans, nonzero depth or invalid observations remain; global events stayzero, callbacks/tool retire, and source identities staycurrent. Receipts are retained in `original-credential-passive-census-native-3-bounded.json`. Native1 was an observer-prerequisite failure; Native2 passed observer completeness but failed only the new draft's unsupported positive-audit assumption. Both remain separately retained.

The unchanged full original census on Windows reports two completed real cleanup candidate queries (one per original evidence variant), two cleanup storage admissions each, and zero helper-process starts, then fails its original positive-helper assertion (217.235s driver, source unchanged). Windows registered SQLite admission selects its original native `_prepare_artifact` route; only the POSIX branch calls `prepare_in_helper`/`HelperLease.start`. The os.open audit does not count Windows native handle calls. These records therefore establish neither zero Windows I/O nor whole-census acceptance. The original cleanup assertion and all phase bodies/ceilings remain unchanged; fresh Linux CI must verify the original POSIX contract. Original whole performance and final-head platform budgets remain open.

Static verification of the19 new/changed Python files against ded197 finds126 baseline and126 current diagnostics, zero new. New helpers pass formatting after two explicitly AST-preserving formatter corrections. The original diagnostic inventory has no drift before the dev integration (652 owners/1444 TASK492/56 TASK31551/7616 TASK494/16 sinks). No full test sweep was run; all verification is targeted.


Original whole-probe prerequisite: await the actual `_initial_screen_pushed` completion latch inside its existing run_test before capturing the screen, as already qualified in the cold startup control. Observation/heartbeat remain active and all original900s timeout/phase counts/Send/UI/helper/native-open assertions remain unchanged; wrong completed screen types still fail. Removing only the wait restores the entire prior module AST. ADR required: no; ADR path: N/A; reason: test prerequisite synchronization preserves application boundaries and original performance oracles. Independently reviewed before application; final native whole verification follows.
