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
