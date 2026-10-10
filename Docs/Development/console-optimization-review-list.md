# Console optimization review list

Maintained for the user's request to preserve every considered optimization that
has not been used or implemented, for review after the main speed and stability
work. Scope: the current Console Send investigation and its integration lane.

Current integration (2026-10-09): combined PR #3050, including PR #3049, the final watch correction, OPT-79, OPT-81 the OPT-86 duplicate-flush cleanup and OPT-89 single-manifest parse. 99 stable optimization IDs and 16 follow-ups are retained. Fresh native comparisons and their limits are recorded below; OPT-31 was rejected and its implementation removed. The one-second Send target and consistent sub-100 ms rendered feedback remain open.

Historical checkpoint retained from the prior integration lane: Latest reported polling correctness evidence is `116628850f`: exact timer RED-to-GREEN, eleven attach passes, survivor pass, original late-FULL activation pass and actual two-saved-turn terminal/background pass. Broader polling narrowing and Send speed remain unqualified. Latest reported local publication evidence is `2d65a328ad` (seven status/recovery/action controls pass), following `5800bdc47d` (eleven targeted controls pass); original video controls pass at `41ab023e31`. Inspected prior failures are retained below. Census evidence is `435f948c56`, startup timing is `cee8faf944`, and the latest quiet whole-Send comparison is `36c6fb431a` to `fa3dad2d82` (below). Earlier `b4a284513b5837998017c12e146aea58b0356a7d`, task27/28 helper results and saved hook intervals remain historical evidence; none is an isolated comparison against the latest integrated sources.

Latest preparation follow-up: the integration owner reports ten focused runtime-flag controls pass with correction `b0bfd8344a`, including the next real Send. The earlier host-refresh-worker cleanup gap was subsequently absent in the aligned original rerun; retain each run's cleanup result separately. The repeated-poll control reached core counts 5 and 5 but its stable-view qualification failed; original-writer diagnostic `received-draft-origin-1` proves a fixture-induced session switch, with the original draft and revision intact. Source correction `d41973538e` selects the real registry workspace before mount and waits for original attachment completion; `aligned-warm-poll-controls-1` at integrated `4b984d` then reaches the intended core5-versus-zero polling RED with exact draft intact, one real-next-Send PASS and zero pending workers. OPT69 is integrated at `6cad32ed67`, following candidate `d7d0cc2d71` and clean causal baseline `c653f5445e`. The integration owner reports 22 isolated contracts and both actual Enter/button receipt controls passing; remaining managed/cancellation qualification and matched whole-Send timing remain pending.

Each ID stays stable. Record new ideas here as they arise. When an item is
implemented, move it to the resolved section with its commit/task and measured
outcome; do not silently delete it. A candidate is not a promised saving or a
scheduled implementation. Historical timings below identify the measured source;
re-measure after integration before treating them as current costs.

Sources: [phase-attribution report](2026-10-06-console-send-phase-attribution.md),
[approved architecture](../superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md),
[ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md),
and [qualification task](../../backlog/tasks/task-34563.4%20-%20Qualify-shared-preparation-Send-latency.md).

## Latest bounded cleanup and quiet comparison (2026-10-08)

Integration commit `fa3dad2d82d31920bd18fdb8db2b7abf74565faf` implements
OPT74 under TASK-34563.36. The original native count control fails at 54
preflight filename checks versus ten distinct components, then passes at ten;
all seven native count fields stay unchanged. The final targeted bundle has
71 passes and one capability skip: this Windows token cannot assign the
Administrators owner SID. Source review found no issue in the minimal change.

Matched quiet real DeepSeek conversations use the same private profile seed,
dialog flow and Enter/button/Enter routes, with three complete replies, three
verified trace links, six messages and no pending checkpoint in each run.
Sources and the owner configuration remain unchanged within each run; the
integration owner reports normal App/server/browser retirement.

| UI action to original provider adapter entry | First Send | Second Send | Third Send |
| --- | ---: | ---: | ---: |
| Before, `36c6fb431a` | 6.297 s | 5.093 s | 5.859 s |
| After, `fa3dad2d82` | 5.766 s | 4.687 s | 4.952 s |

These short paired samples improve by .531/.406/.907 s, but the two fresh
profiles finish startup at different exact times. They do not statistically
attribute the full difference to pure filename validation. The under-one-second
Send target remains unmet; this comparison does not measure physical-input to
terminal-flush latency. OPT73 and the larger serial-preparation investigation
remain open. Evidence: integration-owned `deepseek-uat/component-matched-before`,
`component-matched-after`, `component-matched-comparison.json` and
`component-preflight-green-1.xml`.

## Pre-cleanup Send baseline (2026-10-08)

Fresh setup and three real DeepSeek/deepseek-chat replies succeeded on
`36c6fb431a` (latest dev `a793acbef5` merged), with source unchanged. The
integration owner reports clean App/server retirement. Existing stage timestamps
give the following disjoint intervals in seconds; every column sums exactly to
its recorded adapter-entry time. These are stage boundaries, not exclusive CPU,
filesystem, database, lock or scheduler attribution.

| Recorded interval | First Send | Second Send | Third Send |
| --- | ---: | ---: | ---: |
| UI action scope to receipt | 0.094 | 0.016 | 0.015 |
| Receipt to controller submission | 1.125 | 1.484 | 1.282 |
| Controller submission to save start | 0.640 | 0.641 | 0.671 |
| Durable save interval | 0.563 | 0.218 | 0.172 |
| Saved turn to trace reservation start | 3.140 | 2.468 | 2.625 |
| Trace reservation | 0.219 | 0.142 | 0.110 |
| Trace completion to adapter entry | 0.203 | 0.172 | 0.202 |
| **Total to adapter entry** | **5.984** | **5.141** | **5.077** |

The largest unexplained interval remains saved turn to trace reservation on all
three Sends. The next diagnostic is restricted to that path and the earlier
received preparation. Receipt is not a rendered feedback timestamp: exact input
delivery and natural feedback frame times were not captured by this baseline.
There is no matched before/after speed result yet. Native execution stays with
the integration owner; unselected candidates and paused test drafts remain held.

Evidence: integration-owned `36c6-real-three-message-uat-1/uat-performance-verification.json`,
fields `commit`, `source_unchanged` and `send_stage_diagnostics`.

## Historical measured leads; current Send trace pending

`d221-quiet-three-send-1.measurement.json` records provider entry at
10.643/10.102/9.395 seconds and phase-intersected UI heartbeat maxima of
0.576/0.545/0.431 seconds. Three persisted exchanges/traces complete with zero
dispatch checkpoints, but the report is incomplete and budgets fail. Whole-phase
native open attempts are 68,203/60,296/53,591 across threads and background work;
these are neither unique files nor dispatch-only work. Attempts without an
observed return are 7,392/6,922/6,214, so their durations are unknown. Process
containment retires normally; ordinary app-native cleanup remains unproven.
OPT60 is the clearest repeated main-thread lead; OPT61 occurs at terminal
attention publication. No reduction in provider-entry time is claimed.

The unchanged startup census now passes at 1,032/1,033 in
`optional-dialogs-and-census-1` on `435f948c56` (19 cases: 13 pass, 6 fail).
This qualifies the count, not all dialog actions or responsiveness. The later
`video-platform-and-session-constructor-1` run on `39d98a735e` has five passes
and one video assertion failure: the final Save click lands, but the combined
error/choice predicate times out. The test had notification rendering disabled;
source-only `cbbda57fa6` enables it and adds bounded failure diagnostics. Its
native outcome was followed by a post-attempt gate assertion failure: the original
error/choice predicate passed, but the test incorrectly expected no retained
artifact gate. Test-only38839369a7 preserves that gate until discard. Integration
now reports all three original picker controls pass at41ab023e31 with frozen
sources and no pending workers after fixture teardown. Supported-platform
external-save success still requires its own native qualification. Separate
`ci-cee8-targeted` original startup liveness still fails usable-input and mounted
heartbeat budgets on all three platforms. Timing reports are diagnostic-only;
OS page cache is uncontrolled, async inclusive spans overlap, and Pilot return
is an input-delivery upper bound. Stage summaries can repeat one crossing
heartbeat gap. OPT57-59 retain source-backed startup leads without inventing
exclusive CPU or nested file-call attribution.

OPT60 source follow-up at d221: the 0.2-second transcript timer always awaits
full Console UI sync before deciding whether to stop. Receipt Preparing and
custody draft commit start it, so this repeats during preparation as well as
streaming. The transcript fingerprint skips redraw only after core/roleplay
admission. The existing full-sync coalescer collapses overlap, but does not
separate display requests from domain transitions; the control coalescer already
has a narrow path with explicit whole-sync escalation. Reusing those owners to
distinguish polling/display demand from full reconciliation is a separate
possible approach within OPT60, not permission to reuse display evidence for
live publication. Original poll ancestry accounts for 7/11/7 entries and
.811/1.280/.781 seconds; missing async origins prevent assigning the other full
sync calls to that timer. Keep settings/profile/session transitions, attach,
rewind, Stop and terminal reconciliation fresh. Narrowed polling must still
cover background runs, custody, wake/review publication, tabs/approvals and the
final browser-cache invalidation/survivor handoff. No product change selected.
Source: `chat_screen.py` d221 lines 17507-17842, 18219, 20682-20743;
`wiring.py:792`; `console_spend_projection.py:440`.

## Candidates preserved for review

| ID / status | Opportunity and evidence | Why held; condition for revisiting |
| --- | --- | --- |
| OPT-01 / Deferred on the measured route | **Prepare local catalog configuration once at composition.** `_default_specs` separately reads three exposure gates; enabled Ask User adds a prompt-config lookup. Enabled deep search reads nine settings to use one displayed timeout. Root-specific builders repeat these reads. The new `031b` detail puts the whole local-provider function at .026118/.001266/.001510s. That rules out this sampled function as the main delay; enabled multi-root/deep-search construction remains unmeasured. | Revisit only a materially slower enabled route demonstrated by new evidence. Capture fresh raw merged config at the current composition owner, project small immutable inputs once, and preserve conditional/custom getter calls, prompt precedence, and fresh invocation-time policy/timeout reads. Older turn values would change construction-time freshness. |
| OPT-02 / Deferred; optional shapes unmeasured | **Consolidate local root construction.** `WorkspaceToolExecutor` validates/resolves its root and captures each ancestor; `LocalToolProvider` separately resolves its redaction root. Multiple roots can repeat spec/executor construction. | The sampled local-provider function is tiny; it does not qualify all root shapes. Separate this from OPT-01's config cost. Reuse only an existing equivalent root proof within its valid ownership interval; preserve per-binding exclusions, remote-root separation, root replacement and invocation checks. Measure constructor cost before adding a new proof or cache. |
| OPT-03 / Deferred | **Separate hook path naming from default-directory establishment.** `HookPermissions._current` owns a config snapshot and then `default_hook_permissions_path` derives the user directory again. Current `031b` original spans total .184/.342/.189 s across five contexts per Send. | Raw `cfg.profile_data_dir` supplies a name, while the default getter also verifies/establishes the selected directory and cold fallback. Revisit only with equivalent finite ownership, retaining custom getters, config-source drift refusal, configured-base validation and fallback serialization. |
| OPT-04 / Deferred | **Reduce repeated hook preparation within equivalent boundaries.** Five hook-current contexts were observed per Send, associated with receipt, admission, initialization, capture policy and postcommit dispatch. Historical enclosing costs were seconds; JSON state reading was much smaller. | These contexts straddle real policy/effect boundaries. Current `031b` detail still records 1.282689/1.484787/1.579353s across five contexts per Send. Saved b4a reanalysis locates .814/1.118/1.777s in fresh config entry across the five contexts, versus .083/.077/.077s in hook-state reading; caller-held tails cannot explain the large lifetime. Identify a genuinely same-operation duplicate; keep consent reconciliation, resumed review, revocation and final launch serialization fresh. Empty inventory alone does not justify skipping admission. A naive master-off shortcut also skips persisted reconciliation: observing disable rotates grant tokens, and observed definition removal retires grants. Preserve disable/re-enable stale-target refusal and observed-removal behavior; readiness despite an unavailable disabled store is not proof that reconciliation is unnecessary. |
| OPT-05 / Deferred; current partition reviewed | **Streamline remaining agent startup and serial handoffs.** The d524 detail run has six guard entries totaling .508/.460/.665s to their first yield; reply-worker launch to trace reservation takes 1.838/.899/1.062s. Run-log binding is nested within that interval. | Finite metadata/run-log preparation is already implemented. Remaining guards cross real workers/awaits and include separate stores. Partition current owner setup, scheduling and required admission before selecting a contraction; avoid adding nested times or keeping a lease across an await. |
| OPT-06 / Deferred contract change | **Move auxiliary prompt-history persistence out of the pre-dispatch path.** On d524, awaited history took .262/.326/.470s. The approved design identifies it as auxiliary to conversation recovery. | Current ordering remains awaited. Revisit only with a concrete existing-owner queue/drain/failure contract covering Stop, shutdown, read-after-write and history ordering. A detached background write would trade latency for a lifetime bug. ADR-225 requires a separate explicit contract before changing this ordering. |
| OPT-07 / Deferred, low measured return | **Return checked raw-discovery state to immediate consumers.** Task34563.19 (retained in the `codex/console-send-preparation-plan` planning branch) proposes removing an immediate duplicate check after successful stock operation discovery. The task18 sample recorded 102 eligible checks costing only .118839s across all three Sends. | No product implementation or native RED was performed; its prepared test is outside active collection. Revisit if a current profile makes this material or the API simplifies another necessary change. Preserve custom discovery callbacks, actual file-effect checks and physical retirement. |
| OPT-08 / Unselected, lock-order risk | **Select/prepare an installed MCP source under its existing mutex.** Current task26 setup still observes the source before waiting on that mutex. Moving selection could avoid roughly one 38-open witness per read. | Small ceiling compared with the wider delay; it changes canonical-admission/source-mutex order. Requires a cancellation, recovery-maintenance and deadlock argument before implementation. Do not remove the four retained source/effect/publication checks or turn old observations into cached authority. |
| OPT-10 / Bounded manual-Preparing publication routing retained; wider changes deferred | **Avoid full reconciliation for a context-only publication during qualified manual Preparing.** The original e3ee observer records4/2/4 reads before provider entry; one warm status-only invalidation is younger than the one-second TTL, while payload/display changes correctly reject other reads. Reuse the existing guarded display route. | Keep context keys/TTL and live authority reads unchanged. Overlapping publication must retain the captured FULL callback and trailing demand; legacy/custom callbacks retain their captured identity. Original routing RED and the direct-swap overlap negative control qualify the removal. Integrated51pass and final strengthened6pass; sequential confirmation warm mean3.014297→2.799411s, with the original15.515540s baseline outlier retained and unexplained. No physical-terminal or one-second acceptance claim. Earlier task22 counts alone did not establish identical inputs or removable time. |
| OPT-11 / Not selected, low return and freshness boundary | **Merge live preaccept/dispatch message-version reads.** Five actual live reads in the task22 integrated sample total only 2.662ms across three Sends. | Reads straddle the commit and cover changing requested IDs. No unconditional duplicate was found. Revisit only if a future profile demonstrates expensive same-state duplicates; retain changed-request and postcommit freshness. |
| OPT-17 / Deferred, equivalence unproven | **Batch more Windows metadata work within a full raw check.** Remaining retained-identity and parent-pin observations reopen overlapping ancestor chains; `stat_many_for_admission` is an existing possible primitive. | This goes beyond the implemented parent-walk/control batching. Revisit only with current per-check evidence and equivalent alias, DACL/posture, handle-association and uncertain-close behavior. The existing batch is a stronger two-pass tree proof with exact security metadata, not a lightweight drop-in. A wider batch can add work or broaden ownership. |
| OPT-18 / Deferred, low earlier priority | **Combine small agent lifecycle DB writes.** Earlier attribution measured run creation and context lifecycle rows separately; their bodies were much smaller than admission/run-log preparation. | These rows record distinct lifecycle facts. Revisit if current transaction/admission overhead is material and the original event order, failure visibility and recovery semantics can be retained. Existing per-entry preparation has already been improved. |
| OPT-19 / Deferred contract change | **Defer other auxiliary audits/projections after dispatch.** They were considered as possible reductions to the serial postcommit path. | Classify each actual effect first: workspace binding may depend on projection, and audit policy belongs to its owner. No generic postcommit callback is presumed optional. Requires an explicit completion, failure, recovery and shutdown contract, as with history. |
| OPT-20 / Not selected, tiny warm costs | **Further optimize warm run-budget, personal-context service and profile helpers.** Earlier measured maxima were .213ms for budget, .008ms for warm service, .033ms for profile-tool composition and .044ms for profile snapshot; one cold personal-context construction took .140s. | Warm bodies cannot explain the multi-second delay. The current bridge computes prompt/schema token budgets before personal-context eligible records are known; lazy budget computation after the authorized empty-record check is a possible extension, but the entire measured first-request plan was only 73 ms in combined-real-diagnostic-1. No change selected. Revisit only changed-source or cold-path evidence, keeping inclusive timings distinct from attainable savings. |
| OPT-21 / Deferred, startup scope | **Defer or streamline cold Notes initialization before first input.** Integration lane's original d524 timing: Windows Notes 4.643s/join 4.641s; Linux 1.406s/1.403s; macOS 1.083s/1.081s. Media/prompts were smaller. | These are overlapping inclusive spans, not additive savings. No constructor deferral/reordering is implemented. Preserve mandatory schema/seeding, admission, safe first use and shutdown ownership. Evidence: integration lane's `ci-d524-startup` original timing receipts. |
| OPT-22 / Not adopted for live authority | **Use display caches in main-loop `sync_live_state`.** The integration lane measured .133–.296s in this path, within a full native Send census of 72,874/64,429/59,370 opens on d524 (first Send 13,664 main-thread and 59,210 worker opens). | Core/roleplay reads here intentionally feed live store mutations/projections. Disposable display cache substitution was rejected because it would change authority/freshness. Revisit narrower same-operation consolidation with current whole-Send attribution; task26's isolated reduction does not qualify that whole budget. |
| OPT-23 / Deferred, ownership equivalence unproven | **Fuse the control-read initial traversal with the native snapshot forward pass.** Task24 batches control reads but still leaves initial traversal plus independent forward/reverse completion passes for each fresh witness. An earlier six-ancestor layout suggests an approximate ceiling of 54 opens at nine witnesses, or 48 at eight; neither is a measured saving. | Initial pins and snapshot handles have different lifetimes and uncertain-close handling. Partition original passes first, then prove an equivalent finite handoff preserving named associations, stamps/posture, registry locking and retirement. Sources: `bootstrap._control_observation`, `WindowsOS.stat_many_for_admission`, task34563.24 and its control-parent/lifetime tests. |
| OPT-24 / Not adopted under current proof | **Use an already retained raw parent FD instead of a full private parent walk.** This could avoid traversing the same ancestors again. | Raw parent pins prove the current parent identity, not the private walk's full fresh ancestor ownership/mode/no-follow policy. A duplicated FD is insufficient. Revisit only when an existing domain supplies equivalent current ancestor and named-association evidence. Sources: task34563.22; `_check_parent_pins`, `_walk_verified_parent` and `_prepared_parent_walk`. |
| OPT-25 / Deferred; small observed ceiling and diagnostic limit | **Use identity-only native metadata for raw parent association checks.** Non-companion checks consume device/inode/type, while `_stat_handle` also queries owner/ACL and timestamps. In the instrumented warm hook read, 64 checks take .226s; final metadata totals .0305s, including .0179s in security queries. Retained `fstat` opens no paths; named `stat` accounts for 442 opens. Nested times overlap. | No drop-in identity-only pair exists, and removing existing security-query failures changes behavior. Preserve named ancestry/reparse checks, companion owner/private-mode checks and replacement refusal. The diagnostic totals 1,375 attempts versus 1,008 with diagnostics off on the same isolated node; its cause is unresolved. These are instrumented-sample ceilings, not baseline counts or expected savings. Defer this shortcut; do not replace fresh named-path checks with a retained handle. |
| OPT-26 / Deferred, optional route unmeasured | **Share project-binding list and automatic selection projection.** With enabled project instructions and no explicit binding ID, capture lists/validates eligible bindings and resolution lists/validates them again; local validation repeats ancestor lstat. | Current Send samples do not establish this route's cost. Revisit with enabled named-workspace counts, then derive sole-local selection from one finite capture while retaining remote first-selection checks, current root enforcement and custom routes. Sources: tasks34563.9/.14 and controller `capture_project_instruction_authority`/`resolve_project_instruction_binding`/`list_project_instruction_bindings`. |
| OPT-27 / Deferred, optional route unmeasured | **Combine imported-skill trust status and fingerprint projection within one scan.** For trusted imported rows, `capture_skill_context_maximum` asks for summary trust status and then current fingerprint digest; both can call `_scan_skill`. Builtins and empty catalogs do not pay this cost. | Measure an actual imported-skill route first. A named finite single-scan result could share work; invocation trust/revocation and custom owner behavior must stay current. Sources: task34563.14, `console_configuration_capture.capture_skill_context_maximum`, local skill summary/trust fields and `skill_trust_service._scan_skill`. |
| OPT-28 / Evaluated; not adopted | **Consolidate nested catalog source preparation within one store-owned read.** The new `031b` `local_external_catalog` span is .540950/.366745/.360997s. JSON is read once, through guarded `get_catalog_bundle`/`load`/`_read_payload`; the repeated work is native source observation. | A named operation would need an initial owner and final publication check while retaining readable/recovery approval, missing defaults, file entry, migration writes and custom routes. Replacing two guards with a new outer/final pair yields no net postcommit reduction. Establish actual original counts and a shared body/API without duplicated qualification machinery before selecting it. Initial maximum and postcommit catalog reads stay independently fresh. |
| OPT-29 / Deferred sibling route | **Use the same private lock-stream operation for default data-root selection.** `config._default_data_root_lock` also attempts empty creation before opening its stable lock inode. It does not establish an application-owned parent, so task34563.27's causal duplicate-parent case does not measure this route. | Keep its current pair while qualifying the config/hook owners. Revisit with actual ADR-127 default-root/fallback and concurrent-start controls, preserving selected inode, wait/maintenance checks, cross-process serialization and cold creation durability. No faster root-selection or startup claim is established. |
| OPT-30 / Cold receipt attribution pending | **Locate remaining synchronous first receipt work.** On current b4a all three Sends use accepted early receipts, so the prior first-Send legacy fallback is no longer observed. Quiet first dispatch takes115.6ms; a separate detail sample puts92.1ms inside `receive_console_visible_intent`, versus3-4ms later. | No selected descendant or paint timestamp identifies the cold cost. Next smallest observation is the original `ConsoleRuntime.accept_received_intent` body, separating custody/projection from earlier selection/qualification. Heartbeat maxima do not identify layout, scheduling or a specific callback. No optimization is selected from this gap. |
| OPT-31 / Not adopted after native comparison | **Overlap independent MCP maximum and non-MCP configuration preparation after hook review is ready.** Source inspection finds no data dependency until final snapshot assembly; the current b4a non-MCP worker lasts .285-.606 s in the detailed sample. | This is only an overlap ceiling. Shared source/admission locks may serialize both. Any selected design must join physically retained reads, revalidate receipt/config/source revisions, drain cancelled siblings and preserve optional MCP versus required-config failure policy and custom callback order/affinity. Review must remain first because later capture may migrate/audit stores. No new queue/ledger or lease across the join is justified. |
| OPT-33 / Unmeasured, source-supported candidate | **Share finite preparation in the directory-establishment walker.** `secure_private_directory` has its own root/component walk; it never enters task22's prepared `_open_verified_parent` walk. Each original component open can discover the raw owner and perform another complete parent-pin check. Windows stat then traverses absolute parent names again. This is repeated layered validation, not a recursive raw-check loop. | Measure full-check/native-open descendants on existing versus creation paths first. Any contraction must preserve source/actor/pause custody, component owner/type/mode checks, mkdir/chmod effect gates, final identity/posture and uncertain cleanup. The exact target is the selected directory, unlike the parent walker; require its retained pin and actual final-FD association before close, and full checks at any mkdir/chmod effect. Custom callbacks, visual/voice owners and unsupported routes retain ordinary behavior. Task27 removes one entire duplicate establishment but leaves this separate walk unchanged; OPT17/23-25 address different operations. |
| OPT-34 / Deferred, unmeasured warm-path alternative | **Try an existing lock before exclusive creation.** The implemented OPT09 uses actual exclusive-create outcome, so a warm lock incurs a failed exclusive attempt before opening the existing file. | No measured latency benefit; removing this one attempt adds disappearance/replacement/concurrent-first-creation branches. Keep the simple qualified first version until its remaining cost is material. Actual cold creation must still fsync its own FD; a generic append open is not equivalent. |
| OPT-35 / Deferred; default-root route unmeasured | **Avoid repeated default-root selection inside raw source checks when an equivalent finite boundary exists.** Each hook source-custody check calls `_hook_permissions_selection` and `profile_paths.user_data_dir`; default-root selection freshly checks fallback/conventional entries with `lstat`, reopening their Windows path chains. The explicit-data-directory census does not exercise this branch. | Selection is read-only but is not free of filesystem work. A competing fallback can appear while the config writer lock is held because the separate ADR-127 root-selection lock is not held. Do not cache that choice for an entire hook operation without equivalent freshness/serialization. Measure the actual default-root route before selecting a change; keep custom data directories and existing root-selection behavior distinct. |
| OPT-37 / Deferred; early-receipt scheduling alignment unmeasured | **Coalesce unissued hook-indicator refresh while the initial received-Send snapshot is being prepared.** Legacy `ConsoleHooksController.dispatch` already sets its Send flag before scheduling; refresh defers new readers, shares an existing flight and replays once. Early received-intent wiring returns through the runtime before that legacy flag is set. | First prove actual early-receipt overlap, not just two RUNNING worker labels. Consult existing exact runtime/session custody instead of another state map. Keep the fresh Send snapshot and all already-issued native reads. Visits may reconcile disabled/removed grants on a cache miss, so do not substitute visit data for Send authority or extend suppression across arbitrary approval/provider waits. Source review supports a scheduling gap, not a measured saving or the cause of the peer legacy-test stall. |
| OPT-38 / Unmeasured, source-supported candidate | **Batch the hook visit stamp's overlapping directory postures.** `HookPermissions._visit_stamp` makes two scalar lock-file posture observations and one scalar observation for every component of both config/store parent chains. Each scalar delegates to the same full ancestor snapshot used by task28. A warm visit still pays these metadata scans even when it reuses the published result. | Measure the actual original visit-stamp/warm-visit work before selection. One fresh ordered `_observe_stamps` call could reuse the existing native tree while preserving the exact returned tuple, missing/error behavior, DACL/owner sensitivity, file stamps, selection/generation, maintenance and in-memory sealing checks. Keep Send/review on fresh authority reads and preserve the before/after and settled-file rules. This is distinct from OPT37 scheduling; no reduced main-Send latency or duplicate UI issuance is established. See task33642 and `Tests/Agents/test_hook_permissions.py`. |
| OPT-39 / Deferred; mutation cost and equivalence unmeasured | **Avoid unnecessary permission hardening of an already-private warm lock.** `_open_private_text_stream` calls `fchmod(0600)` for each existing file; Windows `fchmod` queries owner/security, reopens for DACL writes and applies its canonical private descriptor. Some sibling file helpers harden only when projected mode differs. | Measure the actual warm mutation cost before selection. A projected mode of 0600 does not prove exact canonical DACL equality, so copying a sibling mode-only condition may change hardening behavior. Preserve current owner checks, actual descriptor identity, postconditions and uncertain cleanup. No metadata, antivirus, timestamp-invalidation or whole-Send delay cause is established by this source finding. |
| OPT-40 / Deferred; bound companion route unmeasured | **Share a fresh control-record view inside config companion validation.** `config_participants.companion_guard` holds the registry's shared lock, reads `_records`, calls `storage._scope` (which rereads `_records` and `_registry`), then reads `_registry` again. A pure derivation from one finite observed view could remove the repeated control reads. | First prove the current measured route reaches this body: no hold or no matching profile returns early. Preserve `_ScopeProof`, binding/fingerprint/containment checks, live authority and source checks, cleanup and applicable callback behavior. The registry lock alone does not establish that every record or native posture can be reused; verify each writer/record boundary. No new retained cache or cross-operation proof is proposed. |
| OPT-41 / Deferred; narrower read contract unmeasured | **Declare only the config members required by a locked hook-config read.** The generic config operation declares config, its lock, the advanced backup and two temporary destinations even when `locked_hooks_config_snapshot` only reads config while retaining its writer lock. Companion validation may inspect those extra members. | Current route/count cost is unmeasured. A narrower named operation would still need lock creation/hardening, owned-parent establishment, source selection, failure restoration and existing read/write serialization. It must not silently narrow a generic writer or break nested/custom routes. Prefer an existing equivalent read contract before introducing a new cross-module route; any selected contract requires task/ADR review. |
| OPT-44 / Deferred; source anchors and startup scheduling need proof | **Load optional display-pricing support at its actual finite display read.** Resident `console_spend_projection` eagerly imports `pricing_display`, which imports `models_dev_catalog`; both were added to the reported ready snapshot. | `pricing_display` captures original module/class/function anchors, subclasses the pricing catalog and later revalidates them. The projection independently captures that capsule/class. Preserve those original-versus-current checks, fallback behavior and first-frame output. A worker that starts before readiness may still load both modules; moving an import into mount is not a saving. Current source review establishes no timing improvement. |
| OPT-45 / Deferred; several eager roots and evaluated defaults | **Defer MCP result projection until tool-result handling.** `MCP.tool_results` is newly resident, and most uses are invocation, wire decoding or result formatting. | Four existing roots import it: `Agents.mcp_tool_provider`, `MCP.client`, `MCP.local_control_service` and `MCP.unified_control_plane_service`. Changing just one edge cannot remove it. The client's bounded-copy default arguments evaluate exported limits at definition time; preserve them, public aliases, the shared dispatch ContextVar and class identity. No type-only cleanup or module-count saving is established. |
| OPT-48 / Deferred; initial browser read may require residency | **Delay loading the stock browser-read helper until the initial browser read.** Workspace imports `console_browser_read` to pin `_CONSOLE_BROWSER_FACTORY`, although invocation is through the browser read owner. | Default/global browser initialization already uses this helper before readiness, so simple local import may save nothing. Preserve the direct original-factory/source check and current fallback behavior; lazy capture must not bless a replaced implementation. No new proxy/cache or constructor indirection is justified for census alone. |
| OPT-49 / Integrated style/rewind residency verified; behavior and lifetime pending | Task34563.31 uses original style/rewind values at their existing actions. At integrated `cee8faf944`, unchanged Windows census is 1,036 versus 1,042 at `16f0bda38b`, still above 1,033. The six newly absent modules are both modals plus `Media_Creation`, its templates, image-generation service and SwarmUI client. | Two Personas style controls (stubbed picker result), rewind row opening and visible-Send cancellation pass. Two real Console style controls and two rewind controls fail; no import causality is established. Style has a concrete off-screen-click fixture lead pending reproduction. Original log also retains Canvas-policy watch/read tasks, despite clean driver process retirement. Scope/character/library-search alternatives stay deferred; no first-use or Send latency gain is claimed. |
| OPT-51 / Deferred; action-only picker candidate | **Load DictionaryPicker when an attach/detach action opens it.** Its two eager product edges are `chat_screen` and `personas_screen`; actual uses found are Console attach/detach workers and the Personas character-attach worker. No package barrel export or before-ready class use was found. | Both edges must move, preserving original picker class, action results and supported alias/patch routes. The existing Personas attach/detach-through-picker control covers one route; no Console worker action control was located, so cold first use there needs explicit qualification. Modal-only tests are insufficient. This module is already in the pinned census, making it a possible offset, not a new regression cause or measured Send cost. |
| OPT-54 / Deferred; two prompt-variable action routes | **Load the shared prompt-variable dialog after the variable-free fast path.** `UI.Console_Modules.prompts` and `UI.Screens.library_screen` eagerly import both its modal and request type; their actual application actions construct the request immediately before opening. No barrel export was found. | Both routes and both names must defer. Keep the original request/modal classes, captured composer/system guards and Library navigation. Existing Console command and Library cancellation/authorization/application/Use-original journeys cover actions, but cold module absence needs the unchanged isolated census. Preserve supported alias routes; no owner or authority movement. |
| OPT-56 / Source fix saved; first-use qualification pending | **Avoid loading image-generation clients when only style templates are needed.** `console_style_picker_modal` imports the template leaf, but `Media_Creation.__init__` first eagerly imports `SwarmUIClient` and `ImageGenerationService`. The task31 census confirms the package plus all three descendants leave first-ready together when the style picker is deferred. | Startup deferral is already implemented by OPT49; this separate candidate concerns unnecessary service imports on the first template-only use. Measure that actual first-action cost before changing exports. Preserve documented package re-exports, class identity, import-order/cycle behavior and real generation callers; the integration owner saved generation-import deferral in `2998e5ab57`, together with startup census newline normalization. This entry remains pending isolated first-use/original generation-caller qualification; no result is inferred merely from the saved source change. This observation proves fanout, not network activity or an additional latency saving. |

| OPT-57 / Deferred; measured synchronous startup boundary | **Streamline initial Console route loading.** Original diagnostic navigation-target resolution at `cee8faf944` takes cold/warm 1.513/1.471 s on Windows, 1.502/0.847 s on macOS and 0.929/0.945 s on Ubuntu. The synchronous chain is `_push_initial_screen` → `_resolve_screen_navigation_target` → `ScreenRoute.load_screen_class` → `import_module(ChatScreen)`; same-child heartbeat intervals overlap this source-backed synchronous boundary. | Partition the actual transitive work at this original boundary before selecting more import changes. The witness does not identify one descendant or file operation. Keep original class identity, required first-frame owners, source anchors and thread affinity; do not move mandatory initialization or disguise count by merging modules. Separate startup from Send and first optional-action cost. Evidence: each host's `ci-cee8-targeted/console-startup-timing/startup-timing-{cold,warm}.json`. |
| OPT-58 / Deferred; constructor remainder unpartitioned | **Reduce synchronous constructor work after initializer workers finish.** Windows `cee8faf944` timing leaves 3.132 s cold / 2.259 s warm after the last selected initializer completes. Source includes later service wiring and worker-handler initialization; the existing witness cannot assign that remainder to one owner. | Observe the original post-initializer boundaries before selecting deferral/consolidation. Do not equate remainder with CPU or assign it to Notes: parallel worker spans overlap, and warm Notes/other initializer bodies do not explain the entire remainder. Preserve mandatory service availability, recovery/schema setup, source selection and shutdown ownership. This is separate from OPT21's measured cold Notes candidate. |
| OPT-59 / Deferred; awaited screen mounting needs partition | **Streamline the original initial-screen push and mounting work.** `cee8faf944` diagnostic awaited `screen_push` takes cold/warm 3.680/3.582 s Windows, 2.419/2.004 s macOS and 2.789/2.735 s Ubuntu. Later macOS/Ubuntu warm heartbeat stalls overlap this boundary; the selected screen-owned CSS region is negligible (Windows 0.149 ms both runs). | Awaited inclusive time includes suspension and is not one continuous stall or exclusive rendering cost. Partition original mount callbacks/layout/workers before a fix; no stack sampler identifies their individual cost. Keep usable controls, first frame, source-current publication and original shutdown behavior. Do not claim the CSS region explains this delay or add nested `_push_initial_screen`/push durations. |
| OPT-60 / Source-only plan proposed; no product implementation | **Investigate repeated fresh configuration admission during Console synchronization.** In `d221-quiet-three-send-1`, four main-thread `sync_live_state` caller groups enter `config_participants.operation` 43/48/39 times across the three Sends. Core refresh counts are 28/23/22 with entry intervals 1.171/1.556/1.479 s; roleplay 8/14/10 with .814/1.307/.725 s; transcript-poll core 4/6/4 with .575/.844/.458 s; poll roleplay 3/5/3 with .236/.435/.323 s. Entry ends at first yield or supported pre-yield return, before the callback body. Sampled ancestry reaches fresh native stamp/path observation. | `sync_live_state` deliberately omits `checked_projection`, unlike the decorated rail wrapper. Core callback publishes workspace/provider/controller/runtime state, including kill switch; roleplay may prepare persisted name/message refresh. ADR-126 keeps these live. First isolate presentation-only work or a provably unchanged publication before choosing coalescing; do not pass display evidence wholesale, reuse old authority, or bypass final live gates. Main `load_settings` invocation counts 19/25/21 match cache hits, not proven disk rereads. Worker hook scopes are separate. Do not add nested admission durations or attribute all whole-phase native opens to this owner. The roleplay key reads the current in-memory session ID/name, but excludes source path/generation, session incarnation and pause state. Its dedicated name-change generation is not general source provenance; tuple equality alone cannot justify bypass. A candidate effect-free no-op needs already-current standard proof for the exact owner/source and must preserve pending persistence drain; forced/custom/unproven/effectful routes keep live admission. Source-only [task34563.33](../../backlog/tasks/task-34563.33%20-%20Plan-Console-polling-and-full-state-reconciliation.md), [plan](../superpowers/plans/2026-10-08-console-polling-reconciliation-plan.md) and proposed [ADR226](../../backlog/decisions/226-console-polling-and-full-state-reconciliation.md) require origin/transition failure controls before narrowing. The integration owner approved Phase 1 source/test preparation. [TASK-34563.34](../../backlog/tasks/task-34563.34%20-%20Verify-Console-polling-reconciliation-boundaries.md) now holds two source-validated mounted controls (Preparing timer repetition and runtime disable to next real Send); the original two-control run at `557385e36e` failed fixture qualification before either intended assertion. Passive ancestry found real synchronous preparation fallback from cold trust; the first control also lacked a durable database. Test-only `551e43de6b` qualifies real warm sources before Send. At integrated `5b915c5eba`, rows 5/5/1 included a deferred last pass; subsequent `6fad0e9617` records all rows and requires two healthy callbacks within the unchanged five-second bound. `warm-repeated-poll-original-red-2` reached two full/core5 passes, but failed the draft assertion first. `received-draft-origin-1` attributes the empty visible composer to original attach reconciliation switching away from the fixture's directly modified workspace/session; the original received draft remains 23 characters/revision1 and current. Fixture correction `d41973538e` uses the real registry selection and original attach completion, retaining every draft assertion; `aligned-warm-poll-controls-1` at integrated `4b984d` then establishes the intended stable-view polling RED: two deferred rows followed by two healthy full/core5 rows, original draft assertion passes, no pending workers and clean source/custody. The in-place runtime-disable control separately established causal RED from the stale raw cache; correction `b0bfd8344a` has ten focused passes, including actual next Send and sibling builders. Earlier runs had a remaining host refresh worker; the aligned rerun subsequently records zero pending workers, without retroactively qualifying earlier cleanup. Stable-view polling causal RED is now established. Integration `116628850f` separately preserves a deferred final refresh before stopping polling, with original timer, attach, survivor, late-FULL and saved-turn/background controls qualified. Product polling narrowing and elapsed-time gain remain unimplemented/unqualified. |
| OPT-61 / Deferred; terminal attention read measured | **Streamline terminal attention/unseen-mark publication if equivalent fresh data already exists.** Send 1 of the d221 witness records a .356 s main-thread `_acquire_storage` beneath `_finish_custodied_turn` → `recompute_console_attention` → `list_console_unseen_marks`. | This is terminal attention work, not pre-provider hook preparation, so it is not an explanation of provider-entry delay. Establish actual invalidation/read requirements and existing owned data before considering coalescing or an off-loop read. Keep unseen-mark correctness, current chat/workspace identity and safe shutdown. No repeated-identical-query or cache-equivalence proof exists. |

| OPT-63 / Presentation addressed; generation suppression deferred | **Avoid unnecessary workspace-attention generation changes.** The original inventory proposed reducing generation churn. `2d65a328ad` instead guards only unchanged modal attention text; the newer generation is still accepted through the original gate. | Preserve generations as invalidation signals. The seven combined status/recovery/action controls pass, but no generation suppression was implemented or qualified. Separate this local publication guard from OPT61 terminal native reads and from measured Send speed. |
| OPT-64 / Speech, inbox and conversation text repaired | **Coalesce unchanged Buddy modal, speech and workspace presentation updates.** Integration commit `5800bdc47d` guards unchanged speech status and inbox title/error using current renderable values. Five new original mounted controls reproduced redundant writes; eleven targeted controls pass after the combined Buddy/progress/Skills changes, with no pending workers and clean custody. | Changed speech text still grows geometry; actual reads/currentness/action ownership remain. AC72 additionally guards conversation title/activity/notice/empty text and fixes unchanged transcript recovery after unavailability. Three original causal failures become13 targeted passes including five actual decision-owner controls; no source checks or timer intervals change. Remaining decision-card candidates require their own equivalence and lifecycle evidence. No isolated elapsed-time saving is measured. These surfaces remain outside the poll/full-state plan. |
| OPT-67 / Reported candidate; deferred | **Avoid repeated unchanged conversation-card publication.** The integration owner reports this surface remains in its pending inventory. | No original trigger/effect count or implementation is qualified here. Establish exact card inputs and preserve selection, conversation identity, changed content and accessibility before suppressing writes. No saving assumed. |
| OPT-68 / Reported candidate; deferred | **Avoid repeated unchanged decision-card publication.** The integration owner reports this separately from the implemented text surfaces. | Decision/approval state is authoritative: preserve new choices, expiry, resolution, ownership and required actions. First demonstrate redundant presentation writes without suppressing state checks. No timing or implementation claim. |
| OPT-70 / Duplicate native pause query repaired; remaining census excess open | **Remove redundant credential-poll file admissions only after attributing the fresh accesses.** The integration owner reports the Linux original guard records 81 opens across eight ticks versus its existing 54-open ceiling. | Integrated `fd32f2e7e6` removes the demonstrated duplicate pause query for one exact native hold while preserving both leases. The owner reports six final controls and ten broader passes with one POSIX-only skip; this qualifies that duplicate only. The remaining whole-census excess is unattributed. Preserve the unchanged ceiling, fresh credential/profile/source checks and native ownership. Read-only investigation remains in the integration lane; no cache, permission reuse or measured Send saving is selected. |
| OPT-71 / Source-backed alternative; deferred beyond cold Send | **Avoid unconditional trust construction during builtin-only skill discovery if its native path ownership can be preserved.** Original `LocalSkillsService._content_sources` reads `owner.trust_service` to enumerate its native source paths before original `get_context`, even when the later catalog is builtin-only. The original eager-host trace at `82b2bb1685` demonstrates this trigger; stock default-worker placement is a separate contract. | This changes discovery/content-lifetime source selection, beyond OPT69's Send receipt and one-catalog scope. First determine how to prove every actually consumed store/trust/marker path and managed override within the original finite admission, without a preliminary scan, unowned read, lost policy check or stale absence cache. No implementation, count reduction or latency saving is qualified. Do not suppress original discovery to make a Send control pass; retain this alternative for the later architecture review. |
| OPT-72 / Original lock contention attributed; design unselected | **Shorten the configuration writer-lock hold during a hook permission snapshot.** Integration reports an unchanged original Windows tab journey with 42 `ConfigOperationBusy` refusals, all `settings_rebuild` RLock attempts, each joined to its observed original holder. The dominant holder is `HookPermissions.visit_snapshot` → `snapshot` → `_current` → `locked_hooks_config_snapshot` → `_config_write_lock`; its original executor Future remained held over 10 seconds, with `_current` at lines 426/428. Initial `read_current` and prompt-history holders also occur. | Candidate choices require tracing which work needs writer serialization and which can use a checked detached input. Preserve atomic config/grant selection, fresh authority and source checks, writer ordering and physical retirement. Do not add a cross-operation permission cache or simply release the lock around unvalidated work. The diagnostic reports frozen sources, zero overflow, all refusals joined and restoration; the exact run revision and a causal repair remain integration-owned and pending. Related narrower ideas remain OPT03, OPT40 and OPT41. No saving or product fix is claimed. |
| OPT-73 / Evaluated; not adopted | **Consolidate nested admission in one stock local catalog read, if the existing finite owner supplies equivalent fresh checks.** On integrated `36c6fb431a`, `36c6-send-tree-delayed-1` observes warm catalog callbacks of .369/.378 s, each with nine generation witnesses, eight binding selections and seven participant-state binding checks. Witness descendants occupy .218/.227 s (inclusive, not additive). Catalog already loads its JSON state once. | Windows `_ordinary_hold` explicitly disables derived evidence reuse, so the missing-companion POSIX reuse candidate does not explain these repeated Windows reads. Do not enable that reuse or add a permission cache without its own equivalence proof. A finite consolidation could follow the existing owned permission-load contract, but catalog load may migrate/write; retain original custom routes, effect checks, fresh source/recovery gates and physical retirement. These instrumented wall spans include observer overhead and scheduling, not CPU or promised savings. The outer catalog scope retires on exit but has no unconditional final `raw._check`; an owned-read consolidation must supply an explicit final native check and preserve fresh guards on migration writes. No product patch is selected for this entry. |
| OPT-75 / Considered; not selected | **Share same-handle metadata when Windows snapshot helpers immediately reread it.** `open_handle` validates type/reparse status, `named_handle` obtains handle identity, and the reverse `_stat_handle` obtains a fresh metadata sample. Existing native schedule controls expose five `info` calls per tree node across the two passes. | Reusing a prior observation changes timing and failure coverage. Require equivalent fresh reparse/type checks, exact identity on uncertain close, owner/DACL handling and final named association before considering an API change. Do not simply drop `info`, cache native authority, or weaken cleanup. No implementation or quiet timing gain is established; preserve this option for later review. |

| OPT-76 / Refined candidate qualified, not adopted | **Consolidate the late raw-source validation immediately before operation publication and the full check before yielding.** `raw._scope` performs `_participant_state` after preflight/pause checks, then only coordinator/attempt/identity/closed checks and actor-local activation, then `_check(operation)` repeats full source validation before the caller body. The warm catalog tree has two direct scope-to-participant validations totaling .0609 s inclusive; individual gate costs are not isolated. | Replacing the late gate with `_participant_identity` moves full binding validation after the attempt/closed-gate checks and into active operation context. A late retarget plus concurrent pause/close can therefore change which refusal wins. Root and peer source reviews agree this prevents treating the one-line change as equivalent. Preserve the earlier full pre-lock gate and its real source/pending refusal tests. Review active-but-not-yielded observers, concurrent pause/source replacement, exact refusal priority and retirement; do not infer equivalence merely from adjacent calls. No product change, test or saving is selected. A short draft plan was preserved but stopped before tests; priority returns to the full-Send native census. This remains a central finite-operation alternative for later contract review, not Windows witness caching. |

| OPT-77 / Integrated at 67a7f526fb; focused hosts and three real turns pass; latency still unmet | **Skip derived-evidence construction when the existing owner rules prohibit its publication.** `raw._pin_parent` builds native `_path_evidence` for any retained hold, then `_note_derived` immediately refuses Windows via `_ordinary_hold`. The evidence/posture comparison only controls future metadata publication; actual pin validation is separate. `config_participants.companion_guard` has the same collector-before-eligibility shape on its successful full branch. | Prefer the existing exact ordinary-hold eligibility at these two evidence producers, as generation-witness and pause-query owners already do. Keep real native hold/lease ownership, original derivation, source/pause/parent checks and final publication eligibility unchanged. Do not enable Windows derived reuse or disable the distinct Windows-capable admission evidence. Prove actual discarded collection, original native count reduction and exact retirement under a warmed original hook snapshot; the companion success branch needs separate qualification. This is a simpler alternative to OPT33, not an established timing improvement. |

| OPT-78 / Queue measured; pool change deferred | **Check contention between Windows timer waits and native preparation in the default executor.** Installed Textual 8.2.8 submits each Windows timer wait to the default worker pool, also used by finite preparation reads; the app installs FreshContextExecutor with the default capacity. | frames-b5 observes transient saturation during cold feedback; causal queue delay remains unmeasured. Record original submit/start bounds before selecting a fix. Preserve timer cancellation, context isolation, shutdown and ordinary timer behavior. A larger pool or another executor is not justified from source inspection alone. |


| OPT-79 / Implemented; focused controls and natural frames pass | **Load question validation only when a live question can consume the draft.** `_answer_pending_question_with_draft` imports `Agents.ask_user_questions` before checking whether any matching mounted card exists. Ordinary cold Send samples in frames-b4/b5 observe transitive imports before the Preparing frame, including Pydantic and console_snapshot. | Move this existing local import below the no-card/stale-card/attachment guards, without prewarming or changing question validation/command routing. Preserve real question interception and staged attachments. No exclusive import duration or speed gain is yet claimed; qualify original controls and a natural cold frame independently of the catalog experiment. |
| OPT-80 / Evaluated; not adopted | **Coalesce concurrent verification of one exact installed watch.** send-watch-full-1 observes warm overlap clusters containing 11/19 and 14/20 full observations of the same unchanged tuple. | Per-watch finite serialization may let followers reuse a completed fresh verification. Preserve .5-second expiry, mutation generation, drive-root checks, pause, failure retirement, caller final checks and custom routes. Native three-caller RED3 to GREEN1 and ten controls qualified, but quiet ABBA warm mean worsened from 4.307 to 5.767 s with 46 Send stalls above 100 ms in each arm. Removed; exact tested source and controls retained outside the checkout. No eviction or capacity change justified. |
| OPT-81 / Implemented; separate-drive correctness | **Include every watched content drive root in selector evidence.** Qualification JSON is content, but its drive root is absent when code and profile live on separate drives; `_watch_directories` then correctly refuses arming. | Code on D:/private profile on C: reproduces missing anchor RED; two-line Windows anchor inclusion yields 22 passed/one privilege skip. Final same-drive native selection: 65 passed/10 expected skips. Existing guard, freshness and native retirement stay intact; root counts follow actual drives. Remote CI still needs confirmation; no same-drive speedup claimed. |
| OPT-82 / Deferred, policy-sensitive | **Distinguish flush-only by-ID access from actual mutation invalidation.** File/directory flush request write rights and invalidate every watch on entry/exit. | ADR-126 currently explicitly invalidates on any write-capable by-ID reopen. Native flush can update metadata; no safe exemption established. Do not add a caller-controlled skip flag or change epochs based on source inspection alone. |
| OPT-83 / Deferred, cold-only bound | **Retain one constructor-owned canonical permission lease through select and bind.** Both current witness reads acquire independently; outer `_Acquisition` is only a pause/drain barrier. | Possible finite retained mapping, never cached witnesses; retire before final raw scope and preserve custom/source/pause/cancellation/recovery checks. The measured .659 s enclosing pre-read cost is not a predicted saving, and this cannot resolve recurring warm seconds. |
| OPT-84 / Deferred; cold bridge setup only | **Avoid constructing plugin services for a turn that has no plugin consumer, if original demand and source contracts permit.** Original bridge checkpoints attribute .403024 s to the cold plugin-service getter; warm getter is 10.1/13.5 microseconds. | This is initialization within one cold setup, not recurring warm delay. Original continuation/plugin admission and later consumers must still receive their service. No implementation selected; current priority remains configuration capture and warm post-save orchestration. |


| OPT-85 / Deferred; low current return | **Group dictionary and world-input reads in one existing finite Notes operation.** Both use app.chachanotes_db while WorkspaceDB owns the surrounding capture. Stock independent admissions could be grouped without changing SQL transactions. Current original prompt capture is only .053301/.064308 s warm, with the entire second/world read .010324/.013343 s. | Not selected: the measured upper bound is too small to justify scope and compatibility work while saved-to-trace exceeds 1.4 s. Preserve original independent optional errors, exact receiver, borrowed/new connection retirement and counted-callback pause semantics. A narrow entry belongs inside the first dictionary try, never a catch-and-replay wrapper. The external source-only plan remains in prompt-transform-scope-candidate/plan.md; no product or test candidate was implemented. |

| OPT-86 / Retained redundant-work cleanup; no whole-Send gain | **Remove the second flush of the same raw-file descriptor.** Original native append/rewrite performs two same-descriptor barriers with no intervening write; the retained canonical flush already provides Windows/POSIX/Darwin durability. | Native RED is two barriers for append/rewrite; candidate is one, and reads remain zero. Ten distinct targeted controls qualify actual content/inode, directory barrier, cancellation, refusal and resource custody. Canonical first-flush failure preserves the old file and uncertain exclusion until process exit. Quiet ABBA warm mean3.698793s before versus3.720342s after does not demonstrate a speed gain. Retained because it deletes a provable duplicate and obsolete comments without adding machinery; it does not meet the latency target. |

| OPT-87 / Deferred; source-only history selection | **Avoid repeated default-history path selection inside one raw operation if original source/fallback equivalence can be retained.** The existing PromptHistory instance and loaded cache already avoid per-Send construction/reload, but raw scope selection and participant construction both select the default path. | Existing receipt has two paired getter bodies in each history interval; their second enclosing raw scopes take78.169/51.892/22.565ms. Thread/time containment is verified, exact receiver is not. Registration-only reuse breaks a supported custom resolver that changes its second result; config getter provenance exists but no history resolver anchor does. A new qualification seam is not justified by this bounded estimate. No implementation, history deferral or permission reuse. |
| OPT-88 / Implemented; measured local improvement | **Skip unused external-catalog reads for builtin-only stock composition.** Shared preparation and the stock ordinary fallback use the frozen maximum to omit this dependency. | User-approved ADR-225 refinement: unused external catalog errors/audits no longer affect this composition. Initial capture, common policy and actual invocation remain fresh; unknown/mixed/custom routes retain their checks. Original read-count RED1→0; quiet full-profile ABBA warm mean3.742618→3.398968s (9.18%, small local sample), cold5.334600→4.497071s. All12 turns save/settle; the overall one-second target is still unmet. |

| OPT-89 / Retained source-read consolidation; no whole-Send gain | **Read and parse installed server source once per builtin capability manifest.** The current synchronous inventory costs21.7/23.2ms; tools, resources and prompts each independently parse the same source. | One request-local AST now serves the three existing extractors. Original RED reads six times across two requests; final count is two, with all17 focused compatibility/schema/freshness cases passing. Isolated first-manifest sample16.557ms before and6.388ms after is descriptive only. Quiet ABBA warm3.531s before/3.727s after shows no whole-Send improvement. Retained as a small coherent source-read consolidation, not latency acceptance; no process cache or permission change. |

| OPT-90 / Deferred; measured low return | **Investigate repeated NTFS volume checks while walking a path.** Per-open checks apply to different handle incarnations and cannot be merged by drive name. `native_identity` does repeat GetVolumeInformationByHandleW for the same held handle after `ntfs`. | Original e3ee native_identity bodies total only4.024/16.870ms in the two warm Send windows (79/89 calls across all threads); the duplicate is a subset. Inclusive overlapping bodies and observer overhead make this a screening upper bound, not a removable wall-time claim. Keep fresh per-handle local/NTFS/reparse checks; no implementation or authority cache. Receipt native-identity-attribution/native-identity-current-1/native-identity.json sha25694e6af4a93a0edd5291a6033f6bf1804afb70f53c16753e1c7a9104cfc6538bc. |
| OPT-91 / Qualified native-work reduction; whole-Send improvement unproved | **Omit determined stock character-display prechecks.** Absent or differing resident identity now goes directly through the original complete refresh worker; unchanged/unknown/custom routes retain the outer fresh metadata comparison. | Original control2→1 callbacks,3→2 metadata pairs and663→405 native opens, with identical results and physical retirement;59 distinct targeted cases qualify. Quiet ABBA warm4.553→10.964s includes an unexplained29.089s candidate turn and proves no elapsed-time improvement. Retained on the draft branch as a bounded work reduction; see the character-refresh-precheck plan for all samples and limitations. |

| OPT-93 / Deferred; observed miss cost does not justify change | **Filter Windows evidence-watch notifications to actual dependency children.** Current DirectoryWatch watches each parent nonrecursively and treats any completed notification as dirty; a sibling write can invalidate a WATCH_CONTENT parent even when the exact config/hook dependency is unchanged. | Distinct from OPT80 observation coalescing and OPT82 process-wide by-ID mutation generations. First attribute actual misses to notification dirtiness, generation changes, expiry or absent evidence. Deep writes outside an immediate watched parent do not automatically invalidate it. Any filtering must preserve target/ancestor rename and security changes, overflow/errors, cancellation and the final admission gate. No current receipt proves unrelated sibling events cause Send delay; no filter implementation is selected. |
| OPT-94 / Deferred; measured checkpoint cost small in current run | **Distinguish finite worker retirement from final-owner WAL housekeeping.** CharactersRAGDB._close_connection_handle queries journal mode and attempts wal_checkpoint(TRUNCATE) before every finite worker close, including metadata-only reads. Failures are logged and physical close proceeds; the busy/result row is not an acceptance verdict. | No implementation selected. Measure complete journal-mode/checkpoint/close intervals before assigning Send cost. Preserve WAL/NORMAL commits, public final Close, maintenance checkpoints, source-thread physical retirement, borrowed handles and uncertain-close custody. Generic finite callbacks may write; any retirement-policy distinction must be justified by the actual finite-versus-final contract, not inferred from callback names or in_transaction=False. |

| OPT-95 / Implemented, qualified, then rejected: no whole-Send gain | **Share repeated parent preparation within one finite SQLite open.** The tested candidate halves ordinary parent walks (4 to 2), removes 16 native opens per setup, retains real mutation/refusal/churn checks and uncertain-close custody, and qualifies 55 distinct targeted cases with six privilege skips. | Quiet ABBA warm mean 3.030 to 3.224s does not justify the additional ownership/source machinery; cold 5.228 to 4.941s. All samples retained. Exact code/tests archived and removed from active source. See the finite-Windows-SQLite-preparation plan. Existing four walks remain; no pool, timeout or checkpoint-policy change. |
| OPT-96 / Deferred architecture alternative; attribution incomplete | **Give retained SQLite connections an explicit application-owned worker and shutdown/pause protocol.** This could amortize repeated opens across separate operations without leaving anonymous executor-thread handles alive. | Not selected or implemented. Current finite operations already reuse their existing thread handle; no duplicate open inside one finite owner was found. A dedicated owner needs proven cancellation, queue, pause/drain, thread-affine close and uncertain-retirement contracts. The current setup observer lacks DB/async-owner identities, so 24/17 warm setups do not quantify this alternative's benefit. Do not silently retain handles or remove finite cleanup. |
| OPT-97 / Deferred; cost unmeasured and callbacks intervene | **Share the immediate execution-selection proof between an owned MCP observation and its witness reader.** `_mcp_observation` and `_witnesses` both call `lease.execution_context(canonical)`, recomputing current selector/source resolutions and checking the existing lease. | This is not a second admission or control-record read. Supported selector/observer callbacks lie between the checks; sharing would require an explicit finite selection contract and stock qualification. No meaningful saving is established, and the dominant witness/control proof would remain. Do not simply remove the first canonical/source refusal. |
| OPT-98 / Retained; qualified duplicate-stage removal | **Establish stock tool definitions once at existing composition.** Qualified owned capture omits initial policy/catalog/inventory reads; successful consumer publication freezes its IDs/hashes. | ADR225 explicitly moves eligibility to each existing execution consumer; invocation authority stays fresh. 259 distinct targeted cases pass. Quiet ABBA warm3.031 to2.718s, cold4.885 to3.945s, with all12 turns saved/settled and source/no-overlap checks. Initial a1 excluded for detected external evidence-reader overlap, replaced before candidate runs. See composition-tool-ceiling plan; one-second and physical-feedback targets remain open. |
| OPT-99 / Deferred; source invalidation owner missing | **Retain reusable versioned MCP definition metadata on LocalMCPStore.** This could amortize inventory reads across consumers. | Not implemented: hand edits, other processes, recovery and cold empty snapshots need a complete invalidation contract. OPT98 removes the duplicate initial stage using existing owners; no cross-attempt cache is needed for that gain. A shared mutable attempt ceiling is also unselected under OPT14 because hooks, previews and live runs have distinct lifetimes. |
| OPT-100 / Deferred; startup config-lock consumer identified | **Review scheduler heartbeat default-path naming's full config admission within its existing finite worker.** The original failure-time trace identifies `default_heartbeat_path` holding both config RLocks through recovery metadata observation while Console FULL retries. Explicit `heartbeat_path` skips default naming, but no elapsed saving is measured. | Latency shrink is paused. The producer/default route pre-exists dev; no matched dev contention measurement proves unchanged cost or permits cached/constructor-path substitution. Heartbeat is already off-loop. Revisit only after establishing live profile/source currentness and a compatible existing destination owner; preserve selected-root changes, cold establishment/refusal, native recovery/pause evidence, atomic durable heartbeat write and awaited shutdown retirement. Any chosen change needs causal correctness and separate timing evidence; no geometry, timer, executor, TTL or budget shortcut. See archived `pr3050-closeout/geometry-readiness-consumer-evidence/evidence-note.md` (SHA17358203) and `final-geometry-consumer-evidence-2`. |

## Architecture alternatives retained with their decision

These were considered and deliberately not adopted under the current design.
They remain visible for later review rather than becoming implicit future work.

| ID / status | Alternative | Decision and revisit condition |
| --- | --- | --- |
| OPT-12 / Not adopted | A profile-wide storage actor, worker, shared cache or execution lock. | ADR-225 favors finite domain owners; a global mechanism adds invalidation/shutdown rules and can serialize unrelated chats. Revisit only with measured cross-operation reuse/contention and an explicit lifecycle design. |
| OPT-13 / Not adopted | Keep a native snapshot/lease across preparation awaits, approval waits or later dispatch; use one permission observation for the whole Send. | Initial narrowing maximum, live composition and actual invocation have different freshness/authority roles. Native leases stay within finite operations. Any future redesign must explicitly replace those contracts rather than treating prepared data as permission. |
| OPT-14 / Not adopted | Add a generic pipeline engine, dependency bag, scheduler, revision registry or duplicate preparation-state ledger. | Existing runtime/controller/store owners and named domain APIs already provide those responsibilities. Revisit only if concrete extension requirements justify the added owner/abstraction and simpler alternatives fail. |
| OPT-15 / Not adopted | Dispatch a saved chat before required acceptance/checkpoint/consent/trace/context facts, weaken persistence, or silently switch to unsaved work. | The user chose save failure to stop Send and preserve the draft; existing temporary chats remain. No schema/PRAGMA/durability-mode change is selected. Current measurements put much of the delay in repeated preparation, not the minimum commit. Any change would require revisiting that explicit product decision. |
| OPT-16 / Not adopted as the overall solution | Continue only with per-helper shortcuts or move unchanged repeated work into workers. | The approved design instead shares preparation within existing domain boundaries. A small change can still be retained when independently useful, but worker placement alone does not reduce the repeated I/O or total Send delay. |
| OPT-50 / Not adopted; required before first frame | **Move new Console owner imports into constructors or postpone required receipt/controller construction.** Receipt intent/turn modules provide store bases and synchronous admission; the hook-review host is an interrupt-host base; shared preparation reads serve startup workers. Attach/context/spend/hooks controllers and the chat-start coordinator are installed during construction or initial reconciliation. | Constructor-local imports still run before the first-ready snapshot. Delaying these responsibilities would change lifecycle/first-frame behavior or risk moving work ahead of the first Send receipt. Preserve the owners; any genuine later-work boundary requires its own compatibility and timing proof. Do not merge modules or add forwarding proxies solely to disguise residency. |

## Related finding to retain separately

**FOLLOWUP-01 — startup readiness completion accounting.** The integration lane
reports that the first attach can be marked complete while native full sync is
still deferred by replay/maintenance. The integration owner implemented exact runtime/view/generation completion
and deferred full-refresh replay in commit c5b1396daa, with33distinct targeted
controls passing. This is a correctness and perceived-readiness repair with no
isolated measured cost reduction. Keep it separate from catalog optimizations.


**FOLLOWUP-02 — unrelated backup roundtrip fixture.** An original retained-MCP-
history backup/restore control fails before app code on Windows: its copied
child environment omits SYSTEMROOT and other interpreter inputs, causing
asyncio import error10106. Reusing test_complete_roundtrip's existing tested
allowlist is a reviewed candidate correction, but it has not been qualified or
included in task27. The four-child full recovery run is broader than the changed
append path; actual installed history append/rotation/pause and helper-scope
controls passed separately. Also review TMPDIR versus inherited TEMP/TMP before
claiming complete fixture-local temporary storage. This is test portability,
not a demonstrated Console latency optimization.


**FOLLOWUP-03 — first-ready import regression is not Send attribution.**
Integration CI run `37757653003` on `8a4ed1e926` reported 1,046 resident
`tldw_chatbook` modules against the unchanged 1,033 limit: 18 additions and two
removals from the pinned 1,030 snapshot. The original guard snapshots at the
first `_ui_ready = True` assignment in a fresh second process with a warmed
profile. It also forbids `Agents.run_hooks` before ready; the earlier count
assertion can hide that separate failure. Source review identifies import edges,
not runtime savings. Preserve the budget and original test. OPT42 is integrated; task34563.30 additionally selects OPT43/46/47. Other
reviewed import candidates remain deferred or rejected. Original Windows
verification at `7b728190` records 59 passes and one census failure (1,045 versus
1,033); the prior Linux result is not a same-host baseline. Task30 at `16f0bda38b`
then measures 1,042 on Windows, with its three selected modules absent. Three
preparation and three census-helper controls pass; four UI action tests fail
before their actions in raw-source fixture setup. The guard remained red at that revision; the later combined 435f run passes
at 1,032/1,033. No Send-time improvement follows from the module count. A later
bootstrap-profile fixture run qualifies all three original feedback actions;
comparison remains pending its gateway-double correction (see OPT46/47).

**FOLLOWUP-04 — original action failures require real visibility and lifetime evidence.**
At `cee8faf944`, the style tests use the production harness and click the seventh
row without filtering/scrolling or asserting the click result. Shipping CSS puts
that row below the initial results viewport; installed Pilot does not scroll it
into view. This is a source-supported fixture lead, not recorded hit/geometry
proof. Preserve real actions and original dismissal/draft assertions when
reproducing. Separately, `style-rewind-imports-and-prompt-1.log` reports pending
`console-canvas-policy-watch/read` tasks and an unawaited `to_thread`; clean
process/pipe retirement does not qualify app-internal cleanup.

The existing action/lifetime follow-up also retains **early unscoped legacy
approval delivery before reconciliation**. `request_mcp_approvals(...,
session_id=None)` can register a live round before its view is answerable;
the original detached branch can announce attention without calling the attached
setter, and typed remount does not establish replay of that legacy card. This
behavior and the mounted-counts fixture exist in dev a190. The later focused
cc499 group passed without failed-delivery facts; earlier Linux/native first-card
failures remain unclassified. The fixture's reconciled-attachment precondition
is separate from any replay fix. No replay implementation or speed benefit is
selected. Revisit only with a deterministic original early-delivery/settled-
before-attachment control or a qualified failing-branch receipt, preserving round
ownership, settlement, cancellation and typed-decision precedence. Source owners:
`InterruptRoundHost._approval_view_is_detached/run_round` and
`ConsoleRuntime.has_answerable_view/finish_view_reconciliation/remount_pending_approval`.
The external `required-fixture-triage/legacy-mounted-ready/deferred-ledger-note.md`
retains the unselected design. This extends FOLLOWUP-04; it adds no optimization ID.

**FOLLOWUP-05 — video save expectation must match native capabilities.**
The original Windows cancel-control failure at `435f948c56` is a timeout after
its third picker selection; both prior cancels and quit-Stay succeeded. The
copy path requires directory-relative/no-follow primitives before touching the
destination, and intentionally refuses unsupported native platforms. An
unconditional successful-save assertion is incompatible with that contract.
Test-only `c288ea18fa` keeps the original product gate/copy and all earlier
assertions, checks a landed click and fresh visible original failure UI on an
unsupported platform, retains the artifact, then discards only explicitly.
Supported platforms retain byte/file/opener assertions. Source checks pass. Integrated 39d98 reaches the real Save click but times out
on the combined error/choice predicate. Source inspection finds Textual test
notification rendering disabled by default, preventing any ToastRack. Follow-up
cbbda57fa6 enables the original test option and adds bounded failure notes,
without changing the predicate/deadline; its native outcome awaits integration. The upfront capability check selects the
expected branch; it does not prove which later copy check caused a generic
error. This is test portability, not an optimization or established import
regression cause.

**FOLLOWUP-06 — separately owned Skills stale text, implemented.** The integration
owner reports `5800bdc47d` fixes the stale skill-review text alongside bounded
current-renderable guards. Its eleven targeted controls pass with no pending
workers after fixture teardown and clean custody. Keep this as correctness
evidence, not an isolated Send-latency saving or a polling-architecture change.

**FOLLOWUP-07 — fresh-profile UAT observations remain unattributed.** At41ab,
the integration owner reports three real DeepSeek/deepseek-chat user/reply flows,
retained CEDAR context, three complete traces/links and zero dispatch checkpoints,
with configuration/source unchanged and normal terminal quit. That is functional
evidence, not responsiveness qualification. Before the second message, typed
text visibly appeared in stages (blank, then a partial word, then the full text).
Canvas also repeatedly fell back to Terminal only. Retain both observations for
attribution; neither proves a Send regression or a cause in OPT60. Do not infer
acceptance of input/render or provider-entry budgets from completed replies.

**FOLLOWUP-08 — CI budget result lacks attribution.** Integration reports the
PerfGuard run at `a1bea` failed its boot-budget step, but annotations expose only
exit 1 and no artifact. Log endpoints returned HTTP 403, so the exact failing
budget is unknown. This is not evidence of a module-census regression; retain
the prior qualified census separately until an accessible detailed result exists.

**FOLLOWUP-09 — warning lost during layout, fixed in the integration lane.**
The recovery text controls exposed an outdated `DestinationRailSectionHeader.title`
that could overwrite the visible Model warning during allocation. `2d65a328ad`
updates the shared title along with the visible label. Its seven relevant original
controls pass; the separate allocator fixture failure does not erase this result
or establish a broad layout pass. Keep the correctness outcome distinct from speed.

**FOLLOWUP-10 — held draft assertion exposed a fixture session switch.**
`warm-repeated-poll-original-red-2` reached two successful full passes (core5 each)
but failed before its no-repeat oracle because the visible composer was empty.
The integration owner's original-writer diagnostic `received-draft-origin-1`
shows attach/resume registry reconciliation creating/selecting another session;
ordinary draft sync then loads that new session's empty draft. The original
received session remains length23/revision1 with current inputs and request=None.
This does not establish product draft loss. Test-only `d41973538e` selects the
workspace through the registry before mount, waits original attachment completion
within the same10s bound, and asserts exact active/visible/composer/store ownership.
No forced full sync or weakened draft assertion. At integrated `4b984d`, the aligned rerun passes the draft assertion and reaches the intended repeated-core polling RED, with zero pending workers and clean source/custody.

**FOLLOWUP-11 — cold trust setup failure is attributed to the eager host.**
The first two cold receipt controls fail their initial cold assertion at
`be8115f315`; no Send was attempted. `cold-trust-origin-1` at `82b2bb1685`
observes the original builder 10.3534304s after observer installation, on
MainThread beneath skill discovery's `_content_sources`, with no ensure/preparer
entry. Frozen sources, zero after-fixture workers and clean 22.235s driver
retirement qualify that attribution. They do not qualify normal default-worker
startup or any cold Send timing. Reuse the existing real TldwCli/default-scheduler
native fixture, holding its actual builder before publication; never reset a
live service or remove the cold qualification assertion. Startup discovery and
Send catalog demand remain separate scopes in OPT69.

**FOLLOWUP-12 — final refresh must finish before polling stops.**
The integration owner reports an original causal correctness failure:
`_poll_transcript` ignores False from a maintenance-deferred final full sync and
stops its timer when idle. Saved integration `116628850f` makes a minimal explicit-False
return. The integration owner reports the exact original timer failure is now
GREEN, eleven attach controls and the survivor control pass, actual coalesced
late-FULL activation passes, and two actual saved turns exercise native receipt
acknowledgment, a still-active background turn and final tail settlement. This
closes a correctness boundary for OPT60/ADR226 narrowing; it does not establish
that stable polling is streamlined or that Send is faster.

**FOLLOWUP-13 — authentic cold Send reaches the original blocking path; a worker cache remains.**
`stock-cold-send-original-1` uses original TldwCli/default scheduling at
`116628850f` with frozen helper/integration changes. Enter and button both observe
configuration SQL on the input thread, receipt absent, no loop progress and no
Preparing frame during the hold. Cold slots/factory, active session and draft
remain exact; the original lease is live, no probe timeout occurs and no remote
call starts. The original startup builder Future and singleflight physically
retire, with no observer invalidity. Fixture creator retirement then fails:
ordinary leases remain 3, core/pending/raw are zero, and the original one-second
drain returns False. Outer driver 59.159s/pytest 64.203s, frozen sources, zero
parent workers and normal diagnostic containment do not erase that native failure.
`stock-cold-send-retirement-origin-2` identifies the failed owner as
`db.workspaces`: one live, nontransactional cache remains on `asyncio_11`.
`WorkspaceDB.close` closes its calling thread's cache; a later main-thread close
cannot prove retirement of that worker's handle. The subsequent exact connection
trace `stock-cold-send-workspace-origin-3` identifies its birth in
`ServiceWiringMixin._compose_tool_pack_service_off_thread` →
`ToolPackService.reconcile_receipts` → `_WorkspaceReferences.capture` →
`LocalWorkspaceRegistryService.list_workspaces` → original Workspace transaction/
connection, within its original Textual worker and concurrent-futures WorkItem
(`asyncio_7` in that run). Every other observed Workspace creator connection
retired. This attributes the remaining cache to startup tool-pack composition,
separate from the helper's Send cancellation. The integration owner is correcting
that finite worker's ownership; no fix/pass is recorded yet. Do not broaden
closure, relax drain or infer that the held main-thread SQL created the cache;
original `get_workspace` already has an operation-owned scope. Cold product
implementation remains pending.

**FOLLOWUP-14 — worker profiler output mixed threads; retain only valid evidence.**

`36c6-send-worker-diagnostic-1` on `36c6fb431a` completed three real replies
and retired the App/server with unchanged source. Its original local monitoring
records 889 events, no overflow and unchanged selected bodies. However the
supposed worker-only cProfile outputs include original UI/event-loop functions;
`worker-2-4-_CapturedSources.read_permission_payload.pstats` totals .519 s while
the observed callback lasts .393 s, and several entries report self time greater
than cumulative time. These pstats totals/counts cannot select a worker cause.
The profiler/runtime interaction is not independently diagnosed. Keep original
thread/task spans separate and use thread-filtered original-body observation
for further attribution. Do not rerun broad profiling or erase the failed
measurement. Original callback lifetime can include blocking and scheduling;
it is not exclusive CPU time. The earlier span-only run records natural
supplied Preparing frames .151/.042/.023 s after the UI action scope, not physical
input or terminal-flush latency. No isolated speed improvement is claimed.

The replacement `36c6-send-tree-delayed-1` arms original-code monitoring at the
first actual UI action, avoiding the two earlier diagnostic startup failures.
It records nine expected native callbacks across three real replies, with zero
stack errors and zero unfinished roots; the owner reports normal App/server
retirement. Per-root exclusive sums reconcile within four microseconds. Warm
composition permission callbacks take .286/.289 s; catalog callbacks take
.369/.378 s. These are scoped wall observations, including observer overhead
and unselected/native waiting. They establish repeated Windows metadata work,
not a quiet speed improvement. Its supplied Preparing frames occur .088/.790/.012 s after the UI action scope; retain the slow second frame and do not declare feedback acceptance from this instrumented run. Earlier startup probes produced no Send evidence
and remain diagnostic failures, not product regressions.

## Ruled-out premise

The proposed removal of five supposedly eager database opens during configuration
capture was ruled out by source review. `operation_owned_connection` owns cleanup;
it does not eagerly open each database. The explicit Workspace connection is the
sole forced scope and its nonempty-workspace lookup consumes it. Removing that
scope could increase admissions. Preserve this conclusion from the phase report's
"Domain priority after task18" rather than reopening the same premise as an
unmeasured optimization.

## Implemented entries and remaining qualification

| ID / status | Implementation and evidence | Remaining qualification |
| --- | --- | --- |
| OPT-74 / Implemented; main latency target remains open | **Validate each node name once in Windows metadata preflight.** Integration commit `fa3dad2d82` under TASK-34563.36 replaces repeated full ancestor-chain scans with each non-root `node.name`, since the existing node set already contains every ancestor. The original native count control goes from 54 validations to ten, with all seven native count fields unchanged; four invalid-ancestor cases preserve refusal before native access. | Final targeted bundle: 71 passes, one owner-SID capability skip. Matched quiet three-turn samples improve from 6.297/5.093/5.859 s to 5.766/4.687/4.952 s to adapter entry, with complete replies/traces and normal retirement. Different exact startup settling and the small sample prevent assigning the full difference to this change. All drive checks, complete preflight, both native tree passes, ACL/posture checks and physical retirement remain. This does not close OPT73 or the overall latency target. |
| TASK-34601 iteration 1 / Implemented (branch `claude/console-send-admission-cost`, stacked on #3023; its node-tree rewrite also covers OPT-74) | Windows native security observations read the object's own descriptor with `NtQuerySecurityObject` (GetSecurityInfo re-read the PARENT descriptor for app-created directories: 40–54 µs → 3–4 µs), read TokenOwner only when the projection depends on it, and build the snapshot node tree/component validation once. 40-node snapshot 10.2–10.9 → 4.8–5.1 ms; observer-free warm Send→provider entry 6.0 → 4.5 s (interleaved). ChaChaNotes now finishes its `PRAGMA journal_mode` statement (a retained cursor refused commits). | ADR-126 TASK-34601 amendment. |
| OPT-60 (narrowed) + TASK-34601 L2 / Implemented | Live-turn transcript poll ticks run a light publication pass; the full config-locked reconciliation runs on direct requests, at most every 2 s and on the stopping tick, through #3023's Preparing-narrowed full pass (`UI/Console_Modules/poll_cadence.py`). No measurable Send-latency change; UI stalls >100 ms per three sends 57 → 43 (36 combined with all levers). | Rails/inspector rows may trail a turn start by ≤2 s by design. |
| OPT-72 (partial) + TASK-34601 L1 / Implemented | One hook-permission read serves a Send attempt's pre-commit preparation consumers; effect gates and the final pre-dispatch admission stay fresh (ADR-225 decision 3). Warm −0.76 s, cold −1.35 s vs iteration 1 (two interleaved rounds). | The writer-lock hold itself (OPT-72) is unchanged. |
| TASK-34601 L3 + L4 / Implemented | One native operation-path fence per acquisition; on Windows, confirmed evidence is reused while change notifications armed before its confirming observation stay quiet (0.5 s backstop, drive root re-stamped, single-link content, in-process by-id writes invalidate; another process’s by-id writes are bounded by the backstop). Native opens per warm Send 37.9k → 20.3k/19.6k; the native pause probe's open budget passes. | ADR-126 TASK-34601 amendment; watches block renaming watched ancestors while held (owner-accepted). |
| OPT-32 / Implemented by integration owner | Commit `4874767fa357bfa18919c29dd6b3dbb579b3aa50` avoids startup persistence when rail preferences have no saved source. The eight distinct targeted controls pass, preserving saved-layout adoption, explicit edits and the real first-chat generation race. | Original compact compose/mount samples now retain generation 1 throughout, with no original config publisher, loop-progress checks passing and normal shutdown. The peer reports current model compose/mount controls now pass on Windows and model groups pass on Linux/macOS. Startup heartbeat/input limits still fail on all three hosts; this repair does not establish whole-budget acceptance. Historical combined b4a Send was 8.530/8.031/7.938s; this is not an isolated causal comparison. |
| OPT-09 / Implemented and qualified within its task | Root commit `b14185060f48523b51462c7e084534f67a251ee1`, integrated in `7824bc6850b017760b55b6bfc60295fa34861085`, shares config/hook lock creation and stream preparation. Parent establishment2-to1; whole-read native attempts cold/warm config1623/1622-to1484/1495, hooks2117/1125-to1980/1008. All22new controls pass in root and integrated checkout; actual cold fsync, current posture, locking and retirement retained. | Quiet b4a Send remains8.530/8.031/7.938s, slower than the older031b point sample; count reduction has not established latency gain. That historical timing is combined-source evidence, not an isolated causal comparison. CI run37739739905 passes107/107 on Windows/Linux/macOS, including all22new controls; task34563.27 is Done, while the main latency task remains open. Unimplemented existing-first alternative is retained as OPT34. |
| OPT-36 / Implemented and qualified | Task34563.28 batches `_user_data_dir_stamps` through the existing fresh multi-path snapshot. Native causal proof: 11 requested entries over 10 objects, 11 trees/126 opens become one tree/20 opens; exact ordered/DACL results and all physical handle retirements pass. Seven new and eight existing distinct controls pass on Windows, including actual warm memo invalidation and hardening. | Preserve empty-input zero work, missing/error fallback and original cache keys/brackets. No new cache or owner. The same complete warm hook read with optional diagnostics off drops 1,008-to-952 attempts (56, 5.6%), with one parent establishment and 34 descriptor closes unchanged. Integrated directory/count checks pass eight cases with normal native retirement. CI on Windows passes all seven new cases; Linux/macOS each pass four portable new cases and all twelve original POSIX memo cases, with platform-only cases skipped. Task34563.28 is complete. Later combined d221 whole-Send timing still fails its budgets; it is not an isolated comparison and the helper reduction is not an established whole-Send gain. The early bound-companion return remains unchanged. |
| OPT-42 / Original targeted behavior verified; combined census now passes | Task34563.29 moves the annotation-only `HookPermissions` import into existing `TYPE_CHECKING` with postponed annotations. Integrated `7b728190` passes 56 original hook/interrupt checks and three census helpers; its actual census was 1,045/1,033. | The later combined 435f census is 1,032/1,033. Historical Linux 1,046 is not a same-host baseline. Source bodies/owners are unchanged; task closure still needs its recorded integration hygiene. No isolated count or Send-time saving is inferred. |
| OPT-43 / Integrated; observer behavior and residency verified | Task34563.30 uses the canonical `observe_preparation_reads` function in controller wiring. On integrated `16f0bda38b`, three original ownership/observer controls pass and the hook facade is absent from the first-ready additions. | The unchanged Windows census falls from 1,045 at `7b728190` to 1,042 with OPT43/46/47, exactly their three removed residents, but at that revision still exceeded 1,033. Module residency is measured; Send latency is not. Existing attribution and native ownership remain intact. |
| OPT-46 / Integrated; residency verified, behavior still pending | Task34563.30 moves the original comparison dialog import to its opening action and quotes only the type-only result annotation. The original first-ready census at `16f0bda38b` confirms absence. | The original action initially failed raw-source fixture setup. With existing bootstrap-profile setup it reaches app construction, where its gateway double lacks `cached_context_window`; the integration owner is correcting that test double. No action pass is claimed. Source review preserves original class, arguments, guard and callback; first-use timing remains unmeasured. |
| OPT-47 / Integrated; residency and original actions verified | Task34563.30 imports the original feedback modal at its existing opening action. Original census `optional-imports-original-1` confirms absence at first-ready; after existing bootstrap-profile fixture setup, all three original real-modal submission/empty-submission/Escape controls pass in `cost-absence-red-and-fixtures-1`. | The later receipt records HEAD `16f0bda38b` with local fixture changes, frozen source/HEAD, normal driver retirement and no force/overflow/lookup race. The first suite's fixture refusal is not erased; corrected controls now exercise the actual actions. That run still exceeded the census budget; the later combined 435f guard passes. No first-action/Send latency gain is established. |
| OPT-52 / Integrated; original Windows capacity and picker controls pass | Task34563.32 imports the original video-capacity modal inside its existing ownership loop. Original 435f census passes1,032/1,033 and capacity/discard/keep controls pass. The integration owner reports all three original video-picker tests pass at41ab023e31, with frozen sources and no pending workers after fixture teardown. | Preserve historical test failures: unsupported Windows save was assumed successful; the new Toast oracle initially lacked notifications=True; the post-attempt helper wrongly expected an empty artifact gate. Test-only c288/cbb/388 correct those premises while retaining real gate/copy, both cancels, quit-Stay and exact cleanup. Windows verifies refusal/retention/explicit discard. Supported-platform successful external save remains separate qualification. No import cause or latency saving is inferred. |
| OPT-53 / Integrated; original editor controls pass | Task34563.32 loads the original system-prompt editor at its existing opening action after settings/callback preparation. Combined census at `435f948c56` passes 1,032/1,033. | Four original editor controls failed before their actions with `raw_source_selection_changed` in the legacy 435f fixture. Integration subsequently reports seven original/new editor controls pass at fab660f878 after fixture and immediate sidebar-publication fixes; that later receipt is retained by the integration owner. Preserve original class, callback and settings semantics; a count pass does not qualify these actions or prove startup/Send improvement. |
| OPT-55 / Integrated; source/lifetime and selection controls pass | Task34563.32 imports original `ReactionOption` and picker classes at existing construction points, with annotation-only names under postponed `TYPE_CHECKING`. At `435f948c56`, combined census passes 1,032/1,033; original profile-DB replacement (open/selection), dismissed-preview cancellation/nonretention and screen-unmount drain controls pass. | The original avatar control initially saw Automatic instead of Relief. Integration commit `e00f230fbe` publishes the accepted reaction label immediately; the integration owner reports eight relevant original controls pass. Preserve that historical failure and separate first-action latency qualification. Class identities/current-source checks/weakrefs/preview ownership remain original. No measured first-action or Send gain, and process containment is not proof of app-internal cleanup. |
| OPT-62 / Implemented; original status and recovery controls pass | `2d65a328ad` guards unchanged recovery title/context/default text. Original controls also exposed the layout allocator reading stale `header.title`; the fix updates that shared title so the model warning survives allocation. Seven combined relevant controls pass after five causal REDs. | An additional pre-existing allocator control failed before mount, including alone; its diagnostic fixture marker was reverted and the separate fixture/asynchronous-contract issue is documented. No broad allocator qualification or Send latency saving is claimed. |
| OPT-65 / Implemented; targeted status/action controls pass | `2d65a328ad` compares current llama launch-preview title, connection status and bind warning text before widget updates. Seven combined recovery/status/action controls pass and original setup actions remain exercised. | Availability reads, model/runtime ownership, enabled states and changed/error publication stay original. Three-OS supplemental qualification and isolated elapsed-time saving are not established by this local receipt. |
| OPT-66 / Implemented; original targeted publication controls pass | `5800bdc47d` compares current progress-modal count/body/refusal text before calling the original widget update. Five new mounted controls were RED on redundant writes before the combined changes; the final eleven targeted controls pass, with no pending workers after fixtures and clean custody. | Real changed progress/error publication remains covered. Added to the three-OS supplemental matrix, whose outcome is not yet recorded here. No isolated redraw or whole-Send timing improvement is established. |
| OPT-69 / Integrated; receipt controls pass, remaining qualification open | **Keep cold skill-trust initialization off the next Send's input loop.** At `557385e36e`, passive original SQL ancestry follows actual Send preparation through `capture_turn_configuration_snapshot(selection=None)` into the synchronous session builder. Original exception observation identifies `_ready_attribute` refusing `LocalSkillsService._trust_service` while its factory remains cold (configuration preparation:72/154). | Existing `ensure_local_skill_trust_service` provides async retained initialization, and the original local `trust_service` property publishes it. A product design must preserve source identity, cancellation/physical retirement, trust/permission authority and custom-source compatibility; do not simply bypass the guard or silently drop trust. The new warm-poll tests initialize it before their action only to qualify that distinct scenario. The probe-induced eight-second hold is artificial, not a measured cold-start cost. The [bounded source plan](../superpowers/plans/2026-10-08-console-cold-trust-preparation-plan.md) reuses the existing stock service proof and finite initializer, keeps initial receipt resident-only, carries cold eligibility through existing received custody, initializes after hook readiness, preserves app/local winners and reruns the unchanged strict source guard. Existing App-worker skill-discovery policy/scheduling remains a separate adapter. Focused original native cancellation, source replacement, navigation, concurrent initializer, draft and real Send controls are approved for Task1 source/control preparation under [TASK-34563.35](../../backlog/tasks/task-34563.35%20-%20Receive-Console-Sends-while-stock-skill-trust-is-cold.md). The original causal RED is now qualified after `c653f5445e`; candidate `d7d0cc2d71` is integrated as `6cad32ed67`; the owner reports all 22 isolated contracts and both actual Enter/button receipt controls pass with full native retirement. A second option under this same candidate is to avoid initialization when original finite skill capture proves trust is unused: built-in records and later no-digest validation skip trust. This needs an explicit source/capture contract; do not infer absence from a UI cache or initialize unused trust just to satisfy a guard. The required-trust finite initializer remains a branch to qualify. `cold-sql-receipt-baseline-1` at `be8115f315` fails before Send because the aligned eager fixture is already warm; this is not causal RED. `cold-trust-origin-1` at `82b2bb1685` attributes construction to original discovery: `_refresh_console_skill_candidates` → `_fetch_console_skill_context` → `get_context` → `local_content_lifetime` → `_content_sources` → the lazy trust factory, on MainThread, with no finite ensure/preparer entry. The separate eager host triggers this compatibility fallback; it does not establish normal default-worker behavior. Original `_content_sources` requests trust even for builtin-only discovery, so zero construction over whole startup is an invalid no-demand Send oracle. Helper `414e6ec491` and the narrow real-App fixture patch were then exercised as frozen local changes on integration `116628850f`: both Send routes show receipt absent, original configuration SQL on the input thread, no loop progress or Preparing frame during the hold, exact cold sources/draft retained, live native lease and zero remote calls. Actual original builder Future/singleflight retire, but fixture-wide retirement fails; this is a qualified blocking observation with incomplete cleanup qualification, not an integrated control pass (FOLLOWUP13). The reviewed one-catalog/no-demand candidate preserves legitimate concurrent cold-to-ready publication and explicit skill workspace scope. Source compilation and changed-code lint pass. The actual receipt controls establish feedback during held preparation, not an uninstrumented latency improvement. Real managed-demand completion, received cancellation/source-change controls and matched current-source whole-Send timing remain open; their unrelated draft expansion is paused while the current Send trace takes priority. |

Implemented work that predates this list (shared tool preparation,
run-log and activation preparation, retained saves, early receipt, configuration
workers, parent/control batching and the task26 owned permission load) remains in
its existing tasks and evidence report; it is not a pending optimization here.

## Review discipline

- Add evidence, source revision and the reason for deferral when preserving an idea.
- Keep operation-count reductions separate from measured Send/input improvement.
- Update status as soon as an item is selected, ruled out, implemented or superseded.
- Preserve the user's speed/stability goals and required durability/permission rules.
- Use targeted checks and sequential native timing. The list does not authorize a full sweep.


### 2026-10-08 integration evidence update for OPT69/70 and FOLLOWUP13

Checkpoint c653f5445e repairs the finite list_workspaces read that stranded Tool Pack startup reconciliation's worker connection. Both original Enter/button cases now retire all14 App creators, all native lease/operation counters reach zero, and the unchanged one-second drain succeeds. The original tests then fail precisely at cold Send entering configuration SQL on the input thread before receipt/Preparing. This supersedes FOLLOWUP13's cleanup blocker while retaining its historical findings; OPT69's one-catalog product repair is now integrated as `6cad32ed67`; remaining qualification is recorded below.

OPT70's original count control proves two counted leases point to one exact hold and issue two original native pause queries. The finite deduplication changes that to one query while preserving both leases and their retirement. Six final controls cover original count, custom callback/body, real maintenance between acquisitions, mid-call receiver replacement and custom instance dictionary refusal. Ten broader native controls passed with one POSIX-only skip locally. Linux/macOS results and the unchanged whole census budget remain pending; the previous81/54 count is not declared fixed, nor is the separate304 typing-pause excess. Original companion_guard early-return work remains under investigation.

### 2026-10-08 cold preparation source candidate

`d7d0cc2d71` implements the reviewed OPT69 candidate across the existing receipt,
runtime, configuration and skill owners. It captures the original catalog once,
keeps optional unavailability distinct, initializes only for managed demand and
reuses the immutable result for that exact received turn. The no-demand branch
accepts legitimate independent publication without treating unused trust as a
consumed dependency. The managed branch rechecks the current winner through the
existing initializer and strict guard. The selected skill workspace is explicit.

Source compilation of nine files, catalog descriptor-name checks and changed-code
Ruff checks pass; sixty unrelated controller lint findings match the baseline.
The integration owner subsequently ran `cold-trust-contracts-candidate-1`
(22 passes) and `stock-cold-send-candidate-1` (both original Enter/button cases
pass), with frozen candidate sources, original receipt/Preparing/loop progress,
complete native retirement and zero after-fixture workers. Product commit
`6cad32ed67` is integrated. These are reported causal controls, not a fresh timing
run by this lane or proof that whole-Send budgets are met. Eight existing cancellation/replacement/ready/custom regression controls also passed.
Real managed completion and the additional received cancellation/source-change
qualification remain open. Their extra drafts
are preserved and paused.

The active priority is the integration owner's current-source fresh setup to
DeepSeek/deepseek-chat and three real messages. Record actual input to first
naturally supplied feedback frame and original adapter entry; distinguish the
first cold Send from later Sends. Separate quiet timestamps from instrumented
critical-path attribution, retain unexplained time, and do not add overlapping
spans. Native runs remain sequential. Select the next correction from the largest
measured cause, then compare matched before/after absolute timings. This ledger
preserves other ideas without authorizing their implementation or displacing
that work.


### 2026-10-08 whole-Send native census: OPT04/05/06/72/73

At integrated `3ddb59e8fe85`, `3ddb-whole-send-census-1` observes three real
Enter/button/Enter Sends through original provider entry. All three DeepSeek
replies and response trace links verify, six messages persist, and no dispatch
checkpoint remains pending. Source/HEAD stay unchanged. The external observer
arms at the first actual Send and counts only four original native entry points;
its counts are entry attempts, not distinct files, successful opens, or CPU time.

The nine-root permission/catalog tree was too narrow to rank the whole Send.
This census records the following repeated qualified Send work on both warm turns:

| Original owner/family | Open-handle entries per warm Send | Interpretation |
| --- | ---: | --- |
| Five `HookPermissions._current` reads combined | 5,400 | Each read has 1,080 opens, 1,758 metadata entries and six multi-path snapshots. The family therefore totals 8,790 metadata entries and 30 snapshots. |
| Configuration maximum capture | 1,165 | Includes its captured source work; do not add a nested permission span again. |
| Remaining configuration capture | 821 | Separate original preparation callback. |
| Awaited history file body | 709 | Joined through the original FileJob creator task and route; the executor does not copy diagnostic context. |
| Local catalog composition | 608 | Same repeated native validation identified by OPT73. |
| Composition permission read | 457 | The owned permission path remains distinct from the earlier maximum capture. |

All qualified Send owner/read/history buckets total 12,278 open entries on each
warm turn. Separate inherited-context-only counts are 7,564/4,234; background
counts are 10,835/10,657. These unqualified buckets are not proven blockers and
must not be assigned to a Send owner or added as exclusive duration. Main-loop
background work and worker-result delivery need attribution alongside the finite
hook operation; this census does not by itself identify every background caller.

OPT04/72 remain unimplemented. Three reads are original readiness snapshots;
one is v2 configuration capture; the remaining original `_current` lifetime
occurs in the precommit hook-input interval (source review points to legacy
UserPromptSubmit target selection; that immediate caller was not monitored).
The five `_current` wall lifetimes total 1.080/2.032/2.075 seconds in this
instrumented run. Nested `snapshot` lifetimes are excluded from those sums.
Distinct live policy/effect gates must remain fresh; receipt-time authority is
not a reusable permission cache. Investigate duplicate work inside the finite
hook operation first. A possible admission/v2-initialization consolidation needs
an explicit existing-owner result contract preserving refusal, reconciliation,
custom callbacks and effect order before implementation.

A concrete OPT04 internal candidate is an owned raw-config read/lock body:
`locked_hooks_config_snapshot` enters `_config_write_lock`, the interprocess
lock enters another config operation, and guarded `_read_raw_cli_config_unlocked`
enters another. Their nested wrappers share the same operation but repeat source
and parent checks. An existing-owner body seam could retain standalone/custom
wrappers while sharing preparation, analogous to the implemented owned permission
load. This is not selected: count the actual removable work and preserve native
file-entry/final checks, lock ordering, source refusal, rollback/failure semantics
and physical retirement first. Consumer-level admission/v2 consolidation also
crosses awaited reference expansion on supported routes, so it is not an
await-free duplicate that can simply receive the earlier permission result.

Diagnostic action-to-provider times are 10.827/11.140/11.750 seconds, materially
slower than the quiet 4.687-5.766-second reference. They cannot establish speed
acceptance or expected savings. The disjoint diagnostic intervals in milliseconds
are:

| Send | Action to receipt | Receipt to controller | Controller to save | Save | Saved to trace | Trace | Trace to provider |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 94 | 1,890 | 907 | 671 | 6,547 | 298 | 420 |
| 2 | 0 | 3,280 | 1,345 | 390 | 5,750 | 171 | 204 |
| 3 | 15 | 3,437 | 1,203 | 203 | 6,079 | 390 | 423 |

Natural supplied Preparing frames occur .094/1.686/.054 seconds after the
original UI action. The slow second frame remains a failure of the feedback
budget in this instrumented run, not a quiet terminal-flush measurement. Its
initial hook callback returns at +1.003 seconds, while configuration capture
starts at +1.840 seconds; that gap is not additional hook-body execution.
Awaited history lasts .300/.613/.487 seconds; this does not authorize deferral.

Observation integrity: 904 selected events and 321 aggregate count rows, zero
native/read/ancestry/event overflow, zero address collisions or context mismatch,
unchanged selected originals, and monitoring retired. All 25 preparation reads
have unique monotonic ordinals and physically retired producers. Only 23 raw
read/future address pairs occur: address reuse actually happened, so raw IDs
alone would have merged separate reads. No active observed read or running
FileJob remains. Expected received spans and one background read span cross the
recording cutoff; their omitted return events do not contradict the independent
original retirement records.

Cleanup limit: normal Ctrl+Q, server exit zero, unchanged deadlines, no forced
process retirement and an empty final process tree are verified. The final
in-process sample still contains six live leases (pending/raw/core/retiring
counts zero), and the custody receipt explicitly does not claim complete
App-native cleanup. Preserve that unresolved ownership qualification separately
from successful observed-read and process retirement. No product change or
completed latency target is claimed by this diagnostic.


### 2026-10-08 isolated hook-read partition and source limits

The integration owner's `3ddb-hook-snapshot-native-probe.json` measures one
original warmed `HookPermissions.snapshot` on the disposable UAT profile, with
original native entry counters restricted to the exact actor and `_current`
receiver ancestry. Its 1,080 open / 1,758 metadata / six multi-path-snapshot
entries exactly reproduce the whole-Send count for each hook read.

| Disjoint native stage | Open entries | Metadata entries | Multi-path snapshots |
| --- | ---: | ---: | ---: |
| `locked_hooks_config_snapshot` | 572 | 965 | 3 |
| `HookPermissions._store_lock` | 383 | 602 | 2 |
| `HookPermissions._read_state` | 75 | 105 | 0 |
| `default_hook_permissions_path` | 50 | 86 | 1 |

This does not select a product change. OPT03's directory-establishment candidate
accounts for only 50 opens (4.6 percent) on this route, rather than most of the
hook read. Keep default-root and custom-source variants distinct.

Within those same stage totals, nearest `raw._check` callers account for 257
opens at `_runtime_operation`, 72 at `_scope`, 48 at the prepared-parent finish,
and 21 at config `operation`. These are overlapping classifications of the table,
not additional opens. The 565 entries without a `raw._check` ancestor include
original preparation and actual native access; they are not automatically
removable. This probe counts entries, not elapsed cost or possible savings.

OPT07 is narrower than the 257 discovery-associated opens. The retained original
TASK-34563.19 plan selects only immediate duplicate consumers: admitted file,
admitted reader, Windows binary-reader branch and trusted-directory verification.
The admitted-file and admitted-reader buckets identify 9 and 16 opens at
immediate second checks. The 16 entries labelled `open_private_binary` cannot be
assigned to the eligible fallback branch from this caller-level trace: the
native guarded branch also checks after `os.fdopen`, and that check remains
required. The observed prepared-parent finish and current native Windows facade
support the guarded branch. Thus 25 opens are directly identified immediate
duplicates; the initial 41-open interpretation included a potentially required
post-allocation boundary and was corrected before any product change. Actual
allocation, stream-specific path checks and checks after native work remain
required. The earlier .118839 seconds over three Sends remains historical
evidence, not a current speed gain.

Source review also rules out a supposed `_prepare_config_parent` plus config-lock
parent-establishment duplicate on this read path. `_prepare_config_parent` is
called by config writing, snapshot writing and restore, not by the hook snapshot.
The measured profile explicitly sets `TLDW_CONFIG_PATH`, so
`application_owned_config_directory` returns None after its companion check and
the config lock's application-owned-parent helper returns immediately. The hook
store lock does request its parent, once through the already implemented lock
stream body (OPT09). Do not select a parent-creation removal from these counts.
The remaining candidate is consolidation of actual finite configuration/lock
preparation, with its source/effect checks and ownership preserved.

Qualification: the observer reports ready state, zero ancestry overflow, current
selected stage/counter bodies, retired monitoring, and raw/pending counts zero.
The integration owner reports frozen source/HEAD, normal process retirement and
4.64 seconds for the whole isolated driver (not snapshot execution time). The
initial probe did not pin `_current`/`_check` in the final identity list, compare
all native owner baselines, or explicitly close the HookPermissions owner. The
review arrived as the run finished. These counts remain diagnostic attribution;
no stronger internal cleanup claim or product pass follows from the process exit.
A future causal control must include those small ownership checks.


## Retirement-qualified isolated hook read

The integration owner repeated the one warmed original snapshot at frozen
`3ddb59e8fe` with the small missing ownership qualification added. The original
`_current`, `raw._check` and `HookPermissions.close` identities are pinned along
with all measured stage/counter callbacks. Before and after, the exact issued
identity sets for leases, pending acquisitions, raw/core operations, retiring
holds and raw states are equal. The baseline has one pre-existing live lease;
the other sets are empty. After observation retirement, the original owner close
sets its closed flag. No source change, overflow or forced process retirement
occurs. Driver exit is zero; all counter buckets match the earlier snapshot.

This qualifies that one read against its observed baseline, while preserving
the separate six-live-lease limitation in the post-App census sample. It is not
a claim of zero global native ownership. Evidence: integration-owned
`deepseek-uat/3ddb-hook-snapshot-retirement-probe.json` and
`3ddb-one-hook-retirement-census-1.{source,custody}.json`.

The qualified partition remains 572 configuration/locking, 383 hook-store lock,
75 permission-state read and 50 default-directory native open attempts. The
main latency repair remains unimplemented; no permission snapshot is reused
across awaited stages and no guard, reconciliation or directory effect is
removed on this evidence alone. Quiet three-Send timings remain the acceptance
comparison. The independent source lane is reviewing a finite preparation
contract under existing owners rather than selecting the small ruled-out
shortcuts.

### 2026-10-08 OPT33 finite establishment contract (source-only, unselected)

The independent source review finds a concrete repeated operation inside the
383-open hook store-lock stage: `secure_private_directory` performs its original
root/component trust walk, but every original `_native_open` also discovers the
raw owner and repeats the full selected-parent proof. Task34563.22 already
contracts this duplication for `_open_verified_parent`; it does not cover the
separate directory-establishment loop. The saved hook partition attributes 144
opens to runtime discovery in this stage (16 full checks at nine opens each),
but does not identify every discovery's direct caller. Do not claim all 144 as
removable, or convert that aggregate into a predicted latency improvement.

The smallest proposed contract reuses the existing finite-walk machinery with
an explicit actual directory: parent preparation supplies `selected.parent`,
while establishment supplies `selected`. This is a private factoring of the
same source/custody and descriptor owner, not a new service or general cache.
Eligibility remains exact stock config/hook-permission/history ownership,
unchanged directly consumed callbacks, native guards and an already retained
pin for that exact directory. Standalone/custom, explicit open/close,
voice/visual and unpinned/new-directory routes retain the original path.

Preserve the original establishment loop, including every component's owner,
type, mode, trusted-symlink and private-directory postcondition. Perform full
source/parent checks at entry and completion, current source/actor/participant/
lease checks before each native allocation, and a full gate before any actual
mkdir/chmod effect. At completion the actual final FD must match the original
still-retained selected-directory pin; repeat the source/custody fence after
those native identity reads. Creator-owned cleanup must remain possible after
source drift and must retain uncertain closes. This proposal does not change
config/hook lock order, their lifetimes, reconciliation, token epochs, lock
creation/fsync, per-Send hook observations or Windows derived-evidence eligibility.

The integration owner should select a bounded causal control before product
work: observe original `HookPermissions._current` and its actual establishment
walk, attribute original full checks/native allocations to that walk, and prove
exact before/after lease/raw/descriptor retirement. Existing-directory depth
must increase real traversal while leaving the proposed full-check count
constant. Keep actual component/native work visible rather than suppressing it.
Additional controls should cover selected-directory rename/restore and final
FD substitution, native source/lease revocation during final fstat, actual
hardening/creation effect refusal, unsafe ancestor/link/owner cases, custom
callback/default drift, unchanged unpinned fallback, and uncertain-close custody.
Re-run task22 parent-walk and task27 config/hook lock controls after both product
and control source are frozen, followed by sequential quiet integrated timing.
No new whole-App census is needed to choose this candidate.

ADR required: no new ADR if implemented within the above existing boundaries.
ADR paths: `backlog/decisions/225-console-send-preparation-and-io-ownership.md`
and `backlog/decisions/126-complete-local-backup-and-recovery.md`.
Reason: the same finite source/descriptor contract already used by task22;
creation, storage authority and observable hook semantics remain owner-defined.
This is a review proposal, not an implementation plan or a qualified fix. Root
must record the atomic task/plan and causal result before releasing product work.
If the actual contraction is small, retain OPT33 for later and prioritize the
existing OPT60 polling work; do not widen this into storage/registry caching.

The larger 572-open config stage is not explained by duplicate raw leases in
nested config wrappers: they already reuse the current raw operation. Storage
acquisition also has an existing confirmed-evidence route. Source-level repeated
`startup_permission`/`_scope` control reads therefore cannot be called a measured
warm-path bottleneck from the current stage-only census. OPT40 remains subject
to its recorded companion-guard branch qualification; no new generic control
snapshot, cross-store lease or stale permission snapshot is proposed.


### 2026-10-08 OPT77 source contract: avoid unpublishable evidence collection

`storage._ordinary_hold` refuses native Windows, disabled evidence reuse,
maintenance and non-serving owners. `_note_derived` applies that gate before
examining or storing its collected evidence. `generation_witnesses._witnesses`
and `AdmissionAuthority.pause_requested` already select their evidence hold
through this gate; their collector returns immediately for `None`.

Two producers instead pass their actual raw-operation hold directly:

- `raw._pin_parent` builds `_path_evidence(hold.names, anchor)` after the real
  guarded parent walk. Its evidence/posture comparison only clears the optional
  evidence value; the actual FD still returns through the existing identity and
  scope gates. On Windows this collected value cannot be published.
- `config_participants.companion_guard` builds `_metadata_evidence` after the
  successful original companion derivation, then the same publication gate
  discards it. Its early-return/no-profile branch does not collect that evidence;
  do not assume the successful branch explains the measured config stage.

Proposed smallest contract: retain the real raw hold and every actual operation
unchanged, but select a separate evidence-eligible hold through the existing
`_ordinary_hold` rule and require exact identity with the retained hold. Use
only that eligible hold for derived before/reuse/collection/publication. Keep
`_note_derived`'s final eligibility gate because eligibility may change during
native work. An initially ineligible operation may simply miss an optional
future cache publication; it must still perform its complete original authority
and physical-parent derivation. No cross-operation evidence or permission is
introduced, and Windows derived-reuse eligibility stays disabled.

Do not change `_path_evidence` globally. The separate `_reuse_evidence` and
`_note_evidence` admission path also consumes it and supports native Windows.
Do not pass the optional hold into the actual scope, pin or companion lifetime,
and do not skip directory acquisition, fstat, parent association, source/pause,
registry-lock, foreign-owner or final checks. No new reflection framework or
shared mutable cache is needed for this producer-side selection.

Before selecting product work, one integration-owner control should observe an
original warmed `HookPermissions._current`, attribute `_path_evidence` and its
native entries specifically to `_pin_parent`, and show that its publication
reaches `_note_derived` with an ineligible actual hold. Keep both warmed snapshot
outputs, all actual source/lock/effect work and exact before/after native custody
observable. The desired failing assertion is absence of publication-only native
collection on an ineligible owner, after proving that the original collectors
really ran and retired. Cover the successful companion branch independently,
and retain existing supported-platform eligible publication/reuse controls.
Original failures in actual pin/companion preparation must still refuse and
retire identically. No arbitrary replacement callback that removes real native
work can establish the causal reduction. After implementation, root owns the
integrated targeted checks and sequential quiet Send timing.

ADR required: no new ADR under this scope; existing ADR-225 and ADR-126 apply.
Reason: avoid preparing data the existing optional publication contract will
reject, with no change to actual admission or derived-reuse policy. This remains
a source-only proposal with unmeasured savings; OPT33 is retained as the separate
finite directory-establishment alternative, not silently discarded.


OPT77 selection checkpoint: the integration owner reserved TASK-34563.37 and
selected only `raw._pin_parent` producer eligibility. The three-line candidate
uses the existing `_ordinary_hold` gate and requires the exact original hold;
it was source-reviewed but remains unapplied pending causal RED. The companion
success branch and OPT33 remain separate, unselected follow-ups. The source-only
`Tests/Backup_Recovery/test_raw_pin_evidence_eligibility.py` draft uses an actual
warmed hook snapshot, original local monitoring, returned-evidence/publication
identity joins, real pin closes and exact issued ownership baselines before its
zero-unpublishable-collection assertion. Source compilation and Ruff checks pass;
no native test or product import was run by this review lane. Root owns baseline,
product edits, final integrated checks and sequential timing.

OPT77 first baseline `c375-pin-evidence-red-1` is not causal RED. Both Windows
cases failed the draft observer's assumption that collected `_path_evidence`
reaches `_note_derived` unchanged. The original `_pin_parent` may clear that
optional object after comparing its metadata posture with the independently read
pin posture. The corrected observer requires the publisher argument to be the
actual pin frame's evidence and records either exact forwarding or the original
posture-mismatch discard before the unchanged ordinary-hold rejection. The
zero-collection assertion and all source/pin/ownership/physical-close checks stay
unchanged. Source and HEAD remained current; the 12.640-second driver retired
normally, with no forced retirement or identity overflow/races and no remaining
fixture worker rows. The early observer failure did not qualify the later
in-test ownership/count assertions. Product remains unchanged pending a clean
baseline. The control now includes default Windows plus the actual disabled-reuse
setting on all hosts; eligible POSIX reuse remains a separate existing control.


### 2026-10-08 OPT77 causal qualification

The corrected `c375-pin-evidence-red-2` qualifies causal RED in both Windows
cases: only the final two-versus-zero pin-evidence assertion fails after actual
payload, source, pin, physical-close, exact-issued-custody and owner-close checks.
The original code collects two optional metadata objects, discards both by its
posture comparison, and reaches the existing ineligible publisher. Each snapshot
uses 36 native open attempts solely for that discarded collection in this fixture.
The source-current driver retires normally in 11.781 seconds.

The integration owner's exact three-line `_pin_parent` eligibility change passes
`c375-pin-evidence-green-1`: 42 passes and two existing Windows capability skips,
66.747 seconds in pytest / 72.750 seconds in the contained driver. The skipped
cases need Administrators owner-SID assignment and an actual native parent rename
in the installed config-scope fixture. They are not passes. Source/HEAD remain
frozen within the run; native containment retires normally with zero force,
identity overflow or lookup races, and the after-fixture worker report is empty.

Independent offline comparison of both XML properties confirms that every count
and qualification field outside the following changes is identical:

| Observed field | Original | Candidate |
| --- | ---: | ---: |
| Raw-pin publication-only evidence collections | 2 | 0 |
| Native attempts inside that collection | 36 | 0 |
| Default-mode metadata snapshots | 6 | 4 |
| Disabled-reuse metadata snapshots | 3 | 1 |
| Original storage acquisitions, both cases | 3 | 3 |
| Full source checks / parent checks, both cases | 66 / 66 | 66 / 66 |
| Actual raw pins / physical descriptor closes, both cases | 2 / 86 | 2 / 86 |
| Default admission-native / other-native attempts | 198 / 970 | 198 / 970 |
| Disabled-reuse admission-native / other-native attempts | 1242 / 970 | 1242 / 970 |

The expected observer disposition moves from post-collection discard plus
publisher eligibility rejection to pre-collection eligibility rejection and
skipped collection. Both cases retain identical payload, source and positive
retirement qualification. No authority check was replaced by a cached result.

This establishes the bounded raw-pin contraction locally; it does not establish
whole-Send elapsed savings, immediate rendered feedback or POSIX qualification.
The integration owner is saving the candidate and arranging focused supported-host
controls. Companion-guard evidence collection and OPT33 stay unselected. The
main quiet Send reference remains approximately 4.7-5.8 seconds pending a fresh
integrated measurement; the separate six App-sample leases remain unresolved.

The original UAT-profile isolated snapshot also reproduces the contraction:
`3ddb-hook-snapshot-retirement-probe.json` versus
`c375-pin-candidate-hook-snapshot.json` falls from 1080 to 1048 open attempts,
1758 to 1678 info calls and six to four metadata snapshots. Independent offline
bucket reconciliation finds exactly six deltas: configuration no-raw-check
metadata loses 14 opens / 35 info calls / one snapshot, and hook-lock
no-raw-check metadata loses 18 / 45 / one. Every other native bucket is identical.
Both retain original callbacks, zero overflow/pending/raw work, the exact starting
custody baseline and normal owner close. The UAT-profile reduction is 32 opens;
the separate regression fixture reduction is 36. Neither count comparison is a
quiet whole-Send elapsed-time measurement.


OPT77 integration checkpoint: the reviewed three-line product change, frozen
causal control and focused Linux/macOS/Windows job are committed and pushed at
`67a7f526fb0ff3e3e4b270e7d04de3f3b6bffdea` in PR3023. The job covers the actual
disabled-reuse policy on all hosts, the Windows default rejection, separate
Windows admission evidence, and existing eligible POSIX pin/source/retirement
controls. Supported-host outcomes and the sequential three-message quiet Send
run are still pending. TASK-34563.37 remains In Progress; integration alone does
not qualify the latency budget or the separate App cleanup concern.


OPT77 supported-host qualification: focused workflow run `37838634736` at
exact `67a7f526fb` passes on all three hosts. Independent offline XML review
confirms macOS 8 passes / 17 Windows-only skips, Linux 8 / 17, and Windows
18 / 7 POSIX-only skips, with no failure or error. The eligible POSIX controls
actually execute. Every executed new causal case records zero optional pin
collections, two actual pins, three acquisitions, positive full/parent checks
and physical descriptor closes, exact issued custody, current original source,
owner close and observer retirement. macOS records 68 checks / 104 closes and
Linux 63 / 59; the two Windows cases reproduce the local GREEN counts exactly.
Artifacts are `67a7-pin-ci-{macos-15,windows-2022,ubuntu-24.04}/raw-pin-evidence.xml`
in the integration evidence directory. The wider workflow status is separate
from these focused jobs; quiet Send timing and App cleanup remain open.


OPT77 latest real-provider sample: `67a7-pin-final-uat-1/uat-performance-verification.json`
records exact `67a7f526fb`, unchanged sources and owner configuration, three
completed DeepSeek calls, six persisted messages, three `verified_equal` revision
links and zero pending dispatch checkpoints. Recorded UI-action-to-provider
entry is **7.967 / 5.530 / 5.422 seconds**. This sample establishes no whole-Send
speed improvement; retain the earlier 5.766 / 4.687 / 4.952 sample rather than
replacing it with an implied improvement. The bounded native-open reduction
remains qualified, but neither a causal elapsed regression nor an elapsed gain
is established by these two short runs.

Durable commit spans 0.423 / 0.125 / 0.218 seconds in the latest sample. The
interval from successful durable commit to trace reservation spans 4.469 /
2.265 / 2.485 seconds. Those are stage gaps, not attribution to one function or
proof of removable work. Preserve save-before-dispatch and draft retention;
repeated preparation and UI polling remain the larger open architecture work.
This verification report does not itself establish final process/App retirement
or actual rendered feedback within 100 ms. Those concerns remain separate.

The integration owner subsequently verified normal Ctrl+Q exit and full server/
browser retirement for this sample. That establishes process containment for
this run, not a new audit of every App-owned native lease. Task 34563.37 can
close on its bounded count, custody, host and functional criteria while the
overall Send-delay task and broader architecture remain open. No wider OPT33
change is selected by this checkpoint.


Root completion checkpoint: TASK-34563.37 is Done after the scoped unused-evidence reduction, original count/custody/source controls, executed three-platform qualification and final real-provider UAT. Product checkpoint67a7f526fb remains the exact code exercised by those artifacts; later workflow path/documentation changes do not alter product bodies. The initial fresh setup flow was already verified on integrated dev, and the final Enter/button/Enter conversation has three complete calls and equal revision links with no checkpoints. Source and original configuration remained unchanged and normal App/server/browser retirement is verified.

Latest quiet provider-entry7967/5530/5422ms is retained without a whole-Send gain claim. This task completion does not close the overall latency/architecture objective. The five fresh hook consumers, required consent epochs, background/main-loop work and unqualified intervals remain part of that investigation; no wider OPT33 or polling product change was made in task37.

## Parallel Claude Send-admission candidate (2026-10-08)

**FOLLOWUP-15 — review the local Claude implementation before duplicating or combining it.**
The user identified a parallel Claude Code session. Its clean comparison checkout
is `C:/Users/GDesktop-1/.claude/worktrees/send-baseline/tldw_tui`, detached at
`c504ff3b1b`; its clean implementation checkout is
`C:/Users/GDesktop-1/.claude/worktrees/send-admission/tldw_tui`, branch
`claude/console-send-admission-cost`, at `c9288d31a8`. Eight commits are stacked on
the current integration head. This is the relevant parallel work; the initially
supplied PR 3018 belongs to separate Eval/recovery work and is not its identity.
A Codex chat's inactivity says nothing about activity in Claude Code.

The candidate contains probe keyword-forwarding (`7b76e9c29f`), cheaper native
Windows descriptor reads, conditional fresh TokenOwner projection and snapshot
bookkeeping (`4921fadd28`), explicit journal_mode statement completion
(`d332694045`), one attempt-scoped pre-commit hook read (`ac7f7eb2aa`), light live
polling with a two-second full cadence (`0f0fe2b87d`), one native operation-path
fence per acquisition (`6ef8699675`), notification-backed Windows evidence reuse
(`158cb6992a`) and documentation (`c9288d31a8`). It therefore overlaps OPT60,
OPT72 and native observation work already recorded here. Keep these as external
candidates until integrated and verified; do not mark our deferred rows complete
or recreate parallel implementations.

TASK-34601 reports interleaved native, observer-free, full-App measurements with
only the provider adapter stubbed: primitive changes reduce warm Send from about
6.0 to 4.5 seconds; hook sharing reduces another .76 seconds warm and 1.35 seconds
cold against its preceding candidate. Poll cadence reports fewer >100 ms UI
stalls (57 to 43 across three Sends) with no measurable Send-duration change.
Instrumented native opens drop from about 38,000 to about 20,000 with watched
reuse. These are author-reported results; this support lane has reviewed source
and notes, not rerun the scenarios or independently inspected raw receipts.
The task remains In Progress with acceptance boxes open, and the notes do not
supply linked raw receipts or a final integrated feedback/whole-Send result.

Contract changes require explicit attention during integration: hook sharing can
omit a hook enabled/approved only in another process during the attempt, and can
move refusal to after durable acceptance; rails and inspector updates can trail
by two seconds; watched native evidence has a .5-second observation backstop for
unnotified external changes and its handles prevent ancestor renames while held.
The branch records these as accepted ADR amendments; they are not equivalent to
preserving every former per-call fresh observation. Review their actual controls
and existing approval context before adopting them. Keep native timing runs
sequential across both sessions, and use matched source/receipt evidence when
comparing against the separate real-provider UAT.


FOLLOWUP-15 source review found a final-gate race in L4 at `c9288d31a8`.
`_quiet_watch` checks the mutation generation, verification marker and .5-second
backstop before lease counting and drive-root observation. The final fast branch
checks only the root and notification event, so a notification-silent by-ID write,
expiry or another observer's invalidation during that interval can be missed.
A proposed correction shares the existing in-memory predicate and rechecks it
under the coordinator lock after the root/event checks; it adds no native reads
on the unchanged path. The existing ADR-126 TASK-34601 contract already requires
these invalidations, so this is a correctness repair rather than a new policy.

The review bundle is in this task's `claude-watch-final-gate-review` evidence
directory. `watch-final-gate.patch` SHA256 is
`36de43cbc2ed1f76962bc47614c2f39f24fc36ca23020ad31af782857eb64030`.
It dry-applies to the unchanged Claude candidate. An isolated check executing
only AST-selected original function bodies, with controlled observation seams
and no product imports/native I/O, fails all three invalidation cases on the
original and passes all four controls on the proposed source (including the
unchanged fast path and balanced lease retirement). Three actual Windows
regression cases are prepared but have not run. Native correctness and timings
remain with the integration owner; this logic proof establishes no speed gain.
Default Ruff reports the same eight pre-existing E721 diagnostics outside the
changed functions on both source versions; other targeted static checks pass.
Neither the Claude nor integration checkout was edited by this support review.


FOLLOWUP-15 raw receipt reconciliation: the Claude session's saved runs were
located under its task-specific temporary `scratchpad/runs` directory and read
independently. `c1/c2-base` and `c1/c2-new` are complete, exit 0, guard CLEAN,
with no sampler/spans/census. Their recorded product hashes differ only in
`windows_files.py` and `qualification.py`, and each arm is stable across rounds.
Warm samples average 6.059 -> 4.541 s; cold samples average 7.173 -> 6.104 s.
The matched `m1/m2-A0` -> `m1/m2-L1` pairs differ only in the six hook-sharing
files: warm mean 5.116 -> 4.359 s, cold mean 6.883 -> 5.531 s. L2 alone averages
5.236 s warm versus 5.116 s without it, while Send-phase stalls over 100 ms total
57 -> 43. The combined m-ALL warm mean is 3.946 s (3.494-4.338 s), with
1.623-3.404 s still between durable success and trace reservation.

The later L4 off/on pairs average 6.002 -> 5.003 s warm but 7.572 -> 8.505 s
cold, with Send-phase stalls 63 -> 68. These samples do not establish a uniform
latency/responsiveness gain from watch reuse, despite the instrumented open-count
reduction. Incomplete sampler runs `ab1/ab2` (exit 1) and overlapped `l2b-after`
are excluded. All receipts report `36c6fb431a` plus source-file hashes of then
uncommitted changes, before the c504 rebase; they are historical evidence, not
final-source validation. The probe itself explicitly does not establish rendered
feedback. The read-only 16-run, 10-comparison extraction is saved as
`claude-native-receipt-review.json` in the same review bundle, including raw
receipt paths/hashes, differing product hashes, functional counts and limitations.

The older completed `s3-new` sampler also records synchronous Windows metadata
work on the main thread in `run_console_config_sync` during the saved-to-trace
interval (including both direct FULL and transcript-poll origins). It predates
the hook/poll/watch fixes. Use it to focus the next current-source observation;
thread overlap/sample counts are neither exclusive time nor proof of contention.

Integration ownership transferred to this chat by the user on 2026-10-08.
The isolated `codex/console-send-lag-integration` checkout starts at the Claude
candidate `c9288d31a8`; other working checkouts are preserved. This chat now owns
product integration and sequential native verification/timing. The earlier
source-only limitations above describe the state at review time. Native final-gate
RED/GREEN and the rebased polling controls are the next checks, followed by
matched final-source timing. No completion or under-one-second claim is made.

FOLLOWUP-15 integrated qualification (2026-10-08, user-transferred native owner):
PR #3049 is the reviewed Claude branch at `bdff2a2d957c9b3ee4cf532589562b901c193df3`,
stacked on #3023. The integration checkout now uses that exact head plus the
final-watch-gate correction. All other sessions' working copies are preserved.
The original native final-gate controls failed in all three intended cases;
after the correction, the new and existing watch suite reports 21 passed,
1 skipped (this account cannot assign the required owner SID). Combined native
primitive/snapshot/WAL/hook/poll/reconciliation/source-boundary checks report
92 passed. Product/test fingerprints and the revision stayed fixed in each run.
The unchanged fast path retains its original native-read count.

Fresh full-app ABBA measurements use baseline `c504ff3b1b` with only the matching
probe keyword forwarding, versus the integrated PR plus correction. Each process
uses a fresh private file-backed profile and three Sends (two non-streaming, one
streaming); only the provider adapter is stubbed. Receipts `timing-a2`, `timing-b1`,
`timing-b2`, `timing-a3` and `integrated-comparison.json` are in the task's
`claude-watch-final-gate-review` evidence directory. No concurrent native Python
run was detected by the two-second guard; source fingerprints stayed unchanged.
Loaded common product source hashes also match between repeats of each arm.

Across two processes per arm, warm mean is 7.809 -> 3.429 s (baseline range
6.449-10.449; candidate 3.301-3.744), cold mean 8.324 -> 5.460 s. Send-period
heartbeat intervals over 100 ms total 87 -> 16. Those intervals include response
settlement; they are not all pre-provider stalls. All 12 Sends completed with
saved user/assistant rows, complete traces, response links and zero remaining
dispatch checkpoints. These are small descriptive samples, not percentile or
physical terminal feedback qualification. The under-one-second target remains
unmet. The first candidate still spends 3.330 s cold and 1.509/1.273 s warm after
durable success and before trace entry; current phase attribution follows.

Keep OPT-19/OPT-22 open: source review identified a possible stock roleplay no-op
check before live config admission, plus synchronous postcommit Library-policy
and workspace projection reads. Neither is implemented or attributed yet.
Core reconciliation changes live provider/runtime state and cannot consume a
disposable presentation cache. Preserve roleplay drain, changed/forced/custom
routes, durable ordering, current policy, cancellation and recovery when revisiting.

Current-source diagnosis after PR #3049 integration (81c70f4a10, 2026-10-08):
`phase-b1` keeps original startup ordering and observes postcommit history at
70/160/140 ms, fresh hook admission at 310/170/160 ms, and tool/provider
composition at 1.11/0.43/0.60 s. Identity and workspace publication are only
0-4 ms and 20-30 ms. These nested spans are not additive.

`detail-b1` identifies the serial owned-configuration producers: the complete
capture is 1.18/1.37/1.13 s, MCP maximum 0.85/0.96/0.43 s, and the independent
configuration worker 0.32/0.40/0.65 s. This diagnostic imports selected owners
early and cannot qualify cold timing; native scope lifetimes include awaits.
The 0.32-0.43 s ideal overlap ceiling motivates OPT-31's bounded experiment.
The real-body control `overlap-red-1` fails only its final overlap assertion on
the serial source, after original payload/SQL and native retirement checks pass.
No overlap performance gain or adopted implementation is claimed yet.

`frames-b2` observes the original compositor supplying Preparing cells for the
same screen, session, runtime and live received claim, without forcing a frame
or prewarming product imports. Action-to-frame samples are 49.27/24.32/244.26 ms.
The third exceeds the 100 ms goal and reproduces `frames-b1`'s 219.76 ms; the
first report is preserved as incomplete because its owner metadata lookup was
wrong. `frames-b2` completes with the corrected lookup, source unchanged and no
detected overlapping native run. This is headless supplied-frame evidence, not
physical terminal flush or keyboard-queue latency. OPT-22 remains investigation:
attribute the delayed window before changing synchronous live reconciliation.

OPT-28/OPT-73 current-source review: the initial MCP maximum already surrounds
three stock catalog wrappers with an owned entry and final native proof; bypassing
only qualified nested wrappers could remove three inner entry proofs there.
Postcommit currently has three guarded entries and no unconditional final native
proof on outer-scope exit. An owned catalog operation needs its own final proof,
so the net postcommit saving is one, not two. It cannot reduce JSON reads (already
one) or reuse the payload across these distinct phases. Preserve original reader
operation/file checks, missing/corrupt cases, migration write/effect approval,
sticky persistence-error marking including uncertain final validation, custom
callbacks and exact native retirement. The current qualifier does not anchor
_read_payload; any route that bypasses it must first qualify that skipped body.
No implementation selected. Detailed composition/catalog timings are inclusive
and do not establish the proposed saving.

Feedback follow-up `frames-b3` on clean 81c70f4a10 supplied Preparing at
42.81/16.25/42.55 ms. The three selected original synchronous code points are
current; eight spans retire with zero overflow or unmatched returns. Layout took
5-8 ms and config-sync did not enter any of these three action-to-frame windows.
This run did not reproduce the earlier 244 ms window. Both slow receipts remain
open evidence; one fast run does not establish the 100 ms bound or justify a
config-sync fix. A bounded main-thread sample of the pending-frame window will
distinguish other synchronous work from scheduling delay if it recurs.

OPT-31 outcome: **not adopted**. The real-body RED became GREEN and all 107
focused configuration/callback/source/lifecycle cases are covered: 106 passed
in `overlap-controls-1`, with its one faulty test observer corrected; all ten
new controls then passed in `overlap-controls-2`. The defect was in the test:
wrapping its LINE callback changed the frame seen by the original observer,
and an error in Workspace profile lookup is optional. The corrected required
failure is injected at the original capture return after actual SQL retirement.
Product code was unchanged through those checks and independent review.

Quiet sequential ABBA `overlap-a1/b1/b2/a2` compares clean 81c70f4a10 against
the one-file parallel-read experiment. All 12 Sends persist and settle, with
source stable and no detected native overlap. Serial warm mean is 3.255 s
(range 2.847-3.715), parallel 3.332 s (3.140-3.447); cold mean 5.355 -> 6.153 s;
Send-period >100 ms heartbeat intervals 17 -> 29. These small samples establish
no useful whole-Send gain; they do not isolate contention as the cause. Restore
the simpler serial source. The candidate, corrected tests, tracked patch and
raw receipts remain in the task evidence directory; `overlap-comparison.json`
records source hashes, stage timings and limits. No concurrency API or runtime
coordination was retained. OPT-28/OPT-73 remain candidates for removing actual
duplicate admission work within each fresh domain operation.


## Finite catalog count RED and Windows pool observation (2026-10-08)

On unchanged integration product source `81c70f4a10`, `catalog-red-1` fails
only the final intended count assertion in both routes, after actual payload,
one original acquisition, canonical/member lease and descriptor retirement
checks succeed. Initial maximum performs four scope checks plus one final check;
composition performs three scope checks without a final check. Each reads JSON
once. Native opens are 573/473, fresh selections 10/8 and witnesses 11/9.
The candidate keeps one scope plus one final check in each route, preserving
the original operation/file/readable gates. Initial and composition reads stay
separately fresh. Three maximum-attribution ancestor-limit hits limit attribution,
not those raw totals; no observer overflow. The two-file body consolidation is
under review and must pass behavior controls and quiet matched timing before
adoption. This supersedes OPT-28's earlier two-guard/net-zero estimate; the current
count supports three fewer initial checks and one fewer postcommit check.

`frames-b5` on the same unchanged product source records natural supplied
Preparing frames at 115.42/27.16/56.11 ms. Bindings and owner claims remain current,
all three frames complete, the sampler joins, no span gaps/overflow and no
detected native overlap. On the cold attempt the actual FreshContextExecutor has
16 workers. At 79.93 ms its queue has nine items; at 94.47 ms it has one queued
item and the sampled workers comprise 12 Textual timer waits and four other
operations, with no idle worker. The UI thread is in the Windows event-loop
poll at both samples. The four other tops are native file metadata/open/security
and pathlib.read_text. This is evidence of transient queueing/saturation, not
its duration, queued-work identity, or causal contribution to a particular frame.
OPT-78 remains a hypothesis pending original enqueue-to-start attribution. The
warm frames are fast here; the earlier 219/244 ms receipts remain unresolved.
No pool resizing, replacement or product timer change has been selected.
Evidence: task artifacts `catalog-red-1`, `frames-b5`, and the frozen observer
SHA256 `38f3e08f3820b1daf950a2fbb6d037e364714cadff2dc0de2047777d16c01a98`.


### Unchanged comparison failure retained: catalog-a1

The first new quiet baseline at exact `81c70f4a10` fails the existing provider
trace-settlement deadline after two completed replies. Adapter entry was
23.104/15.291 seconds; the result is incomplete, so it is excluded from any
accepted speed comparison. Sources stayed unchanged and the native-run guard
found no overlap. This happened before applying the catalog candidate. The
preceding instrumented frames-b5 was also slower (8.050/5.663/9.255 seconds),
but its observers prevent treating those totals as a quiet comparison.

After the failure, process inventory found only server-classified Python
processes; a point-in-time host counter showed 47 percent CPU, 34,849 MB available
memory, no page reads and no disk queue. These later counters cannot attribute
the failed run. Do not raise the settlement deadline or label it environmental
without further evidence; keep the failure and capture original queue/start
timing in the next diagnostic. No competing process was stopped.


## Catalog candidate correctness (2026-10-08; timing pending)

`catalog-green-1` passes both count controls and all eleven new behavior cases
(13 passes, 49.17 seconds). `catalog-controls-1` passes 26 existing cases,
including held-read cancellation/close and migration. Six drift controls fail
before their intended mutation because their observer targets the bypassed
public bundle body. Moving only that passive barrier to the shared original
load body retains actual native admission before payload work, exact receiver,
deadlines, mutations, refusal and retirement assertions. Those six then pass
in `catalog-controls-2` (31.98 seconds); independent review confirms equivalent
coverage. The targeted selection totals 45 passes. Product final hashes:
local_store `ee32f9c34ea4de7e07494a3c102ac552eafe69c9d9b70c542f6e8efcd980a727`,
console_snapshot `084e453829538d419a09dae3ba2cbd075d5c00b38477d65ce01223d7586f9d74`.
The source extraction is AST-equivalent except for the payload invocation;
formatting passes and scoped lint passes with the existing four intentional
local-store E721 exact-type checks excluded. Source review finds no blocker.

Each catalog phase now has one scope and one final check, plus its retained
operation/file/readable gates and one JSON read. Selections become 7/7, witnesses
8/8, parent walks 9/9, and descriptor closes remain ten. Native total opens are
573→482 for maximum and 473→482 for composition: these are not uniform gains.
Exact ancestry reconciliation is maximum -150 removed-check opens +59 member
acquisition opens; composition -100 removed-check opens +50 final-check opens
+59 member acquisition opens. RED reused evidence with a quiet watch, GREEN
reused confirmed evidence without one; acquisition therefore costs 2→61 opens.
The artifacts do not establish why the watch was unavailable. All other normalized
buckets match; maximum retains the same three ancestry-depth misses, composition
has none. Quiet whole-Send comparison remains the adoption gate.

Further analysis of the unchanged `catalog-a1` failure narrows its meaning:
the probe's action-relative 15-second allowance had already expired before it
entered the trace-settlement helper on Send 2 (turn completion 16.207 seconds;
helper failure at 17.005). Thus that failure does not prove a slow or hung
settlement worker. The actual large recorded interval is durable-success to
trace-reservation: 14.673/6.951 seconds. Original queue/start attribution is the
next diagnostic; no deadline is extended.


## Catalog decision and worker queue attribution (2026-10-08)

OPT-28/73 are not adopted. Quiet ABBA initially favored consolidation (warm mean 6.433 -> 5.323 s), but the final BA confirmation reversed it (candidate 4.273 s, baseline 3.245 s; cold 7.221 versus 6.346 s). All 18 completed Sends across the six accepted runs persisted and settled, with stable sources and no detected native overlap. Large between-run variance precludes a causal percentage claim. Restore the simpler 81c70f4a10 catalog implementation; preserve candidate code, tests, source review and receipts under catalog-candidate/rejected-final, catalog-comparison.json and catalog-confirmation.json. The earlier catalog-a1 failure remains excluded, not erased.

OPT-78 diagnostic queue-a1 on unchanged 81c70f4a10 records all original submit/start/return pairs with zero gaps, overflow, unstarted submissions or unmatched starts; observer and sampler retire. Preparation jobs waited at most 9.064/7.409/1.096 ms by Send, while their longest actual worker bodies took 567/448/486 ms. Catalog queue upper bounds were 0.062/1.032/0.124 ms with worker durations 789/249/248 ms. Permission reads likewise waited under 0.160 ms. A larger pool cannot address the dominant multi-second native work in this sample; no pool change is selected. Natural Preparing frames were 66.875/17.100/54.890 ms; this diagnostic does not explain the earlier 219/244 ms frames or establish physical terminal latency. Timer jobs did sometimes queue (cold maximum 95.55 ms), so the transient saturation observation is retained. Two worker_guard calls per Send took up to 1.216/0.551/0.524 s inclusive, with queue below 0.2 ms: original operation attribution is the next useful investigation. Do not sum overlapping inclusive job spans.


## Ordinary Send question-import cleanup (2026-10-08)

OPT-79 moves the existing question-validation import past empty/slash/controller/card/session/attachment early returns. Ordinary Send no longer imports that feature solely to discover no question is mounted; actual answers still use identical validation. No cache or prewarming was added. Updated two existing fake-screen routing fixtures to traverse the current real diagnostic/observed Send methods and carry the exact session argument. question-controls-3 passes all 18 selected tests, including the real mounted card/controller round and ordinary full-app Send. Earlier question-controls-1 recorded two stale-fixture failures before interception; question-controls-2 was an accidental retry without the app-module collection and exposed config-source fixture lifetime errors (1 pass/16 setup errors), retained rather than counted as product evidence.

question-feedback-1 supplies natural Preparing frames at 85.877/21.035/30.115 ms in a fresh process, with all three messages/replies/traces complete and no source changes or detected native overlap. The observer bindings remain current and retire, but two worker starts lack matching enqueue observation; queue attribution from this run is incomplete. Frame observations remain complete. No causal latency percentage, physical-terminal flush or universal 100 ms bound is claimed. The previously recorded 219/244 ms frames and under-one-second application target remain open.


## Combined-source real-provider qualification (2026-10-08)

The fresh ordinary setup and Enter/button/Enter run on `dd80a40dfb40`
completes three real DeepSeek/deepseek-chat calls, six durable messages,
three verified trace links and zero pending dispatch checkpoints. All three
calls are non-streaming. The separate headless third-turn streaming check
does not qualify live streaming. The owner config and production sources
are unchanged; custody proves the launched tree empty and its native identity
released without forced retirement. This is functional qualification, not
latency acceptance.

| Real Send stage, seconds from UI action | First | Second | Third |
| --- | ---: | ---: | ---: |
| Durable save succeeded | 4.295 | 3.983 | 3.812 |
| Saved to trace reservation (interval) | 7.000 | 2.719 | 4.328 |
| Provider adapter entry | **11.812** | **7.343** | **8.422** |

Do not transfer the stub-provider warm mean of 3.429 s to ordinary use.
The environments have matching Python 3.12.10 and dependency versions,
but differ in configuration shape (five-section 220-byte logical seed versus
ordinary generated configuration, observed post-run at 42,351 bytes/206
sections), provider, browser transport, background startup behavior and
inter-Send pacing. These are diagnostic variables, not established causes.
The Library 'Agent access off' chip concerns Library access, not the native
agent runtime; five ready tools is not evidence of that switch being bypassed.
Watch fallback remains unproved: its supported fallback has no log message.
No new optimization is selected from those differences.

The earlier `combined-real-deepseek-1` browser reconnection failed before any
Send; preserve it as an excluded setup attempt. Evidence:
`claude-watch-final-gate-review/real-uat-summary.json` and
`combined-real-deepseek-2/{uat-performance-verification,owned-server-1.custody}.json`.
The next diagnostic partitions an original real served Send, then isolates
full-profile seeding if needed. Native runs remain sequential.

Prior original-body config attribution bounds a possible nested-scope cleanup:
all hook config entry work together is only 0.319/0.246/0.338 s in
`detail-b1`, versus 5.282/4.353/3.970 s total. Nested scope entry alone is
9.4/10.1/11.4 ms; these inclusive diagnostics do not establish removable cost.
Do not add a coordinator or relax source/failure ordering for this bound.


## Current combined-source attribution and CI follow-through (2026-10-08)

All receipts below used dd80a40dfb40. Native runs were sequential, sources stayed unchanged, and guarded headless runs detected no overlapping native test/UAT. These findings do not close the one-second goal.

| Receipt | Observed result | Interpretation |
|---|---|---|
| combined-real-diagnostic-1 | One real DeepSeek turn persisted and settled; normal app exit, native identity released, process tree empty without forced retirement. Action to provider approximately 5.657 s; saved to trace approximately 2.250 s. | Original-body observer preimports owners, so this is attribution only. Named preparation/native-entry spans cover about 1.744 s of the saved gap without summing nested spans. |
| profile-full-a1 | Full stock default config shape; three Sends 8.955/6.657/6.973 s. | Compared sequentially with profile-minimal-b1; no evidence config file size explains the real-provider delay. |
| profile-minimal-b1 | Original minimal config shape, same effective settings; three Sends 10.677/7.054/7.322 s. | One pair only; no causal percentage, statistically established improvement, or diagnosis of between-run variation. |
| send-import-full-1 | 115/2/0 loader bodies; disjoint import union 1.697620/.018540/.000018 s across three Sends. All 247 observed import spans returned; no overflow/unmatched/unfinished entries. | Original imports account for cold work, essentially none of the warm saved-to-trace gap. The observer's 227,275 unrelated global unwind callbacks add unmeasured overhead; do not compare its whole-Send durations with quiet runs. Loaded modules are not file-open counts. |
| raw-pin-ci-local-1 | 65 passed, 9 skipped in 31.13 s for the exact failed Windows CI selection. | Seven POSIX-only cases and two unavailable owner-privilege cases skipped. Local success does not explain or clear remote failures. |

Deferred refinements of existing candidates: cold permission-owner construction enters independent canonical admissions for recovery selection and source binding before its final raw scope. Its outer construction acquisition is only a pending-operation barrier, not a lease these readers can share. The measured pre-read cost was about .659 s; a finite constructor-owned lease may consolidate some work, but must preserve recovery witnesses, source binding, pause/cancellation and final native validation. No constructor optimization has been adopted. OPT-20 also records the smaller lazy personal-context token-budget option. The bridge's uncovered .427 s setup interval still needs narrower attribution; progress-inbox preparation already avoids SQL when its cached inbox matches, so repeated calls alone are not proof of duplicate reads.

Clock calibration: application stage logs use Windows Python 3.12 GetTickCount64 monotonic time, nominal resolution 15.625 ms; original-body spans use QPC. The served diagnostic aligns those clocks approximately, including an unmeasured body-entry-to-stage interval. Import receipts instead copy all 42 original stage callbacks on the same QPC clock and need no log alignment.

Remote targeted CI run 37884217062 has unresolved failed jobs. Logs/artifacts were unavailable through the initial connection path at that checkpoint. Identifier-only summaries were added without changing tests, deadlines, or authentication. Later, completed job logs were retrieved through the normal permitted GitHub CLI connection, saved as data and stripped of terminal controls for review. Keep the PR draft and task In Progress until the relevant failure sets are understood.


## Watch dependency coverage and rejected overlap experiment (2026-10-08)

The bounded watch diagnostic at ff707f8ffd observed 489 evidence-reuse calls across three complete saved Sends, 63 full observations and no watch evictions. Warm overlap clusters contained 11 of 19 and 14 of 20 observations, all of the same unchanged tuple. This justifies investigating overlapping verification, not increasing cache capacity. Inclusive observation durations are not additive or predicted savings.

**OPT-81 retained.** Installed qualification content could be on D: while profile posture covered only C:. The existing completeness guard correctly disabled watching. A real D-source/C-private-profile control failed on the original missing D: anchor, then the original watch/final-race suite passed (22 passed, one unavailable-owner-privilege skip) with the missing root included. Same-drive inputs deduplicate the existing root; POSIX is unchanged. Final current-source native integration passed 65 cases with ten expected skips: seven POSIX-only, two unavailable owner privileges, one cross-drive case already qualified in D:/C:. Eight existing exact-type Ruff E721 diagnostics are unchanged from ff707; modified tests pass lint and all three files pass format checks.

**OPT-80 removed.** An exact-watch RLock could coalesce three real acquisitions from three original full observations to one. Running-loop callbacks used nonblocking fallback; workers retained a bounded wait. Native custom, invalidation, pause, exception, root and retirement controls passed; all 31 combined watch cases passed with two expected skips. The real security-mutation control permits either follower to require an additional observation and explicitly requires the affected caller to recheck. The unchanged-input case still requires exactly one observation.

Quiet full-default-profile ABBA, three saved turns per fresh process:

| Run | First Send (s) | Warm Sends (s) |
|---|---:|---:|
| watch-base-a2 | 6.963129 | 5.072459 / 4.046076 |
| watch-candidate-b1 | 5.018038 | 3.874526 / 3.571796 |
| watch-candidate-b2 | 5.726287 | 6.061276 / 9.558618 |
| watch-base-a3 | 8.347027 | 4.561976 / 3.548717 |

Warm mean: baseline 4.307307 s; candidate 5.766554 s. First-Send mean: 7.655078 / 5.372162 s. Each arm had 46 Send-period heartbeat stalls above 100 ms. All 12 turns persisted and settled, sources stayed unchanged, optional profilers were off, and the external native-process guard detected no overlap. This small, variable sample does not establish a causal regression percentage; it does not justify adding synchronization. Candidate product and tests were archived then removed. The initial new-checkout watch-base-a1 receipt (11.454268 / 9.653072 / 7.538105 s) is also retained; it precedes the four-run comparison and is not silently discarded or attributed to code.

An earlier same-drive run had an enrollment setup timeout and an unverified watch after the arming thread exited. Both cases then passed on untouched ff707, in the candidate combined suite, and in the final retained-source native selection. The cause remains unproved; no retry, deadline extension, or lifetime correction was added. The first cross-drive setup attempt put the private profile on an unsafe shared drive and correctly failed before collection; only the subsequent D-source/C-private-profile assertion is causal RED.

Stage partitions of four completed quiet profile/baseline receipts put 78–89% of warm time in action-to-controller and saved-to-trace entry. Trace-entry-to-adapter is only .205–.408 s; actual durable commit .372–.915 s. Focus remains initial hook/config preparation and post-save orchestration. Finer original bridge setup checkpoints are the next diagnostic, not another cache or permission-policy change.

Receipts remain under the external claude-watch-final-gate-review directory: watch-coalescing-comparison.json, send-watch-full-1, watch-drive-red-2, watch-drive-green-1, watch-integrated-1, watch-final-native-1 and watch-coalescing-candidate/rejected-final. Original same-drive failures remain in watch-drive-same-1. Remote run 37887161924 reports failures in Windows watch cases and Preparing controls; the local three-case Windows tab-boundary selection passed separately. Remote failures are not cleared by local success, and the PR remains draft.

Final current-source Console consumer checks passed all 85 cases in 535.28 s: hook consent sharing and native lifetime, original polling/reconciliation/cadence, tab-source changes, and full configuration-sync refusal/lifetime. No test deadline changed. Combined with the 65 native passes, this is 150 final targeted passes with ten expected native-only skips. Receipt: watch-final-consumers-1. One owned-process stack capture during verification confirmed real app tests were progressing behind buffered output; it was not a Send timing run.


## Original bridge setup partition (2026-10-08)

The focused current-source receipt bridge-setup-attribution/bridge-setup-final-1 at 9936f409c6 completed three saved Sends and all three traces with zero pending checkpoints. All 42 original QPC stage callbacks match the parent probe; 183 local LINE events yielded 54 checkpoint rows, zero gaps/overflow/unfinished setups, current source/bindings, and fully retired hooks. No global event stream, product preimport or callback replacement was used.

Adjacent original bridge setup intervals sum to .468187800 / .000562500 / .000385500 s. The separately measured START-to-first-line observation interval is .000307500 / .000273700 / .000264300 s. Cold plugin-source lookup costs .403023600 s and the progress inbox call .059713000 s; the marker cannot split plugin imports, native setup and service construction. The warm bridge prefix is below one millisecond while saved-to-trace still takes 1.409817 / 1.454788 s. It cannot explain recurring warm seconds; do not redesign progress inbox custody from the earlier uncovered .427 s interval.

The historical detail-b1 receipt attributes 1.369435 / 1.134962 s to warm configuration capture, about 75% / 72% of precontroller time. Those exact capture spans are paired and their source modules still match, but underlying admission changed since that receipt; six unmatched starts and 92 raw ancestor-depth misses elsewhere remain limitations. A focused current-source capture partition is the next diagnostic before another change is selected.

Remote targeted run 37890968596 verifies 9936f409c6. This chat's superseded ff707 run 37887161924 was cancelled after its completed Windows/Linux Preparing logs were preserved; no peer PR run was cancelled. The temporary D: source-only reproduction checkout was retired after preserving its patch and hash manifest.

Recovered CI logs identify the Linux Preparing failures as direct fixture reads of the lazily created _console_control_bar_replay_whole_sync flag (three cases). Both fixture reads now use the original product's absent=False contract. Four affected real Windows polling scenarios pass in55.22s, with lint/format checks passing and all original deadlines preserved. The remote Windows log has one failed membership precondition before the deliberate mutation: the initial FULL did not reach the held Surface lock within10s. Its cause remains unproved; local final85-pass coverage does not clear that remote observation.


## Current configuration partitions and CI qualification (5ec3db629a)

`configuration-capture-attribution/capture-final-1` completes three saved turns with all42 original QPC stages matched, six returned capture bodies, zero overflow/gaps and retired monitoring. Warm owned capture is .860/.677s, including MCP .245/.257s and worker .608/.412s. Common capture's .467/.163s cannot alone be attributed to database or skills work.

The narrower `capture-domain-1` retains original callbacks, all42 matching stages,12 returned bodies,161 checkpoints, zero issues/overflow and unchanged sources. Actual warm common capture is .071001/.080616s: review roots .012966/.013260, prompt .053301/.064308, skills .002682/.001863. Dictionary reads take .042861/.050846s; world reads .010324/.013343s; snapshot assembly and config setting are small. This supports deferring OPT85, not assigning the previous run's variable common cost to those consumers. Owned capture still spans1.017/1.006s; saved-to-trace is1.726/1.414s. Worker intervals are nested, never added to their enclosing await. These are attribution runs, not quiet speed claims.

Observer-free `capture-followup-base-a1` on unchanged5ec3 completes all three saved turns in4.885104/3.139172/3.495077s to the stubbed adapter. Three complete traces, three response links, zero checkpoints, no detected overlapping native run and unchanged sources. This is a fresh baseline, not an optimization comparison or physical-terminal qualification.

Exact-head CI run37891900210 passes the raw storage/watch selection on Windows, Linux and macOS. Windows has68 passes/seven platform-or-privilege skips and confirms actual profile=C / qualification=D coverage. Each Preparing job passes38 polling,18 question and47 hook/lifetime cases; the lazy-flag fixture correction and the prior Windows membership precondition now pass on their actual CI hosts. Preserve earlier failures rather than claiming they never occurred.

Performance gates remain red. Startup receipts are complete with unchanged sources and ordinary shutdown, but usable startup times (cold/warm seconds) are Ubuntu16.837/9.550, macOS16.466/8.114 and Windows23.087/10.555; heartbeat gaps .361/.322, .342/.210 and .455/.365s exceed the unchanged .200s ceiling. Limits remain15s cold and10s warm. The trace diagnostic completes three replies, messages and traces with zero checkpoints and valid source/retirement, then fails the original POSIX helper-start count:38 versus16 on Send1. That all-thread phase census does not establish direct Send causality or introducedness. No deadline/count limit is weakened. Logs/artifacts are preserved under ci-37891900210 in the task artifact directory.


## Single canonical raw-file flush (OPT86)

Original5ec3db629a fails the new real-descriptor count at two flushes for append and rewrite; the read case passes with zero. The candidate removes only the trailing fsync and obsolete bookkeeping comments. First canonical flush, its uncertainty path, native descriptor ownership and replacement-directory flush are unchanged. Ten distinct targeted cases pass across single-flush-green-1 and final single-flush-controls-2. The failure child proves original bytes remain, the exact descriptor and uncertain leases stay owned, subsequent publication refuses and independent maintenance remains excluded until physical process exit.

Two old publication spies intercepted stdlib fsync, which misses Windows FlushFileBuffers. They now call and observe the canonical barrier for the exact published inode. The second-publisher fixture also wrote unescaped Windows paths into TOML; single-flush-controls-1 retains that failure, while JSON-encoded fixture paths let final controls reach real publication. The obsolete trailing-flush error test is replaced by an isolated first-barrier failure control without altering production uncertainty behavior. The existing three-platform raw-pin CI selection includes these seven cases; no new timeout or weakened budget.

Quiet, sequential full-profile ABBA uses two fresh processes per arm and three saved Sends per process. Baseline cold6.933481/6.230813s and warm3.734034/3.627038/3.612453/3.821648s; candidate cold5.136916/6.058842s and warm3.865354/3.666466/3.177661/4.171889s. Warm means3.698793 versus3.720342s show no demonstrated gain; cold means6.582147 versus5.597879s and heartbeat-stall counts26 versus20 are descriptive small-sample observations, not causal or percentile claims. All12 turns save and settle, three complete traces per run, zero checkpoints, sources stable and no detected overlapping native run. Receipt single-flush-comparison.json records exact hashes and the baseline storage_admission.py CRLF-versus-LF-only difference; normalized contents match. The only product-code difference is the duplicate barrier. The cleanup is retained for less work and simpler code, not presented as a latency fix or subsecond acceptance.

Current pre-cleanup postcommit-basic-1 attributes warm saved-to-trace1.584/1.530s with .0288/.0147s residual to disjoint named boundaries. Provider composition is .383/.485s; history .257/.158s; hook admission .173/.193s; checkpoint .176/.103s. Nested guard/worker intervals are not added. All496 observer events pair,42 original QPC stages match and monitoring retires; selected preimports make this diagnostic only. Existing detailed provider-preparation observation is the next discriminator. No new tracing framework or storage policy is introduced.

Final OPT86 combined selection (single-flush-final-integrated-1):13passed in19.84s, including the three original watch final-gate races. Review found no blocking product/test/workflow issue. The new test module passes lint/format; the three existing files retain51 lint findings versus52 at the unchanged baseline, with no added finding. No full suite was run.


## Warm preparation and ordinary CI follow-through (2026-10-09)

`tool-preparation-attribution/tool-prepare-partition-1` on 5ce21f8b35 completes three saved turns and traces, zero checkpoints, all42 QPC stages, six paired bodies and60 checkpoints. Source/bindings remain current, monitoring retires, no global events/overflow/unfinished calls or detected native overlap. Natural late import leaves cold preparation unobserved. Warm preparation .416/.525s contains permission workers .072/.064s, catalog workers .264/.378s, and inventory .022/.023s. Both results have zero external records,32 inventory tools, five prepared tools and no changed-definition audit. Catalog return-to-outer-resumption .052/.055s includes original caller-loop validation/projection plus possible scheduling wait; it is not solely queue delay. This is attribution, not observer-free speed evidence. Existing nested MCP calls already reuse the same operation, leases and pins; their remaining source/parent freshness check has actual drift-refusal controls.

Completed ordinary Ubuntu CI at5ec3 confirms the helper-start budget failure is not confined to added instrumentation: original probe33 >16 on Send1, diagnostic38 >16. Three calls/messages/traces and zero checkpoints complete, but performance acceptance fails. Phase counts do not establish introducedness or direct Send causality. Broad controls have1443 passes, six failures,42 skips; finite callback controls108 passes/three skips. Five failures originate in incomplete hand-built Workspace controllers; the sixth cancellation test is being aligned with the current drain-before-cancellation contract. The first local fixture retry supplied _screen but exposed a second missing constructor field, _preparation_reads; availability-fixture-final-2 preserves all five failures. The initial manual attempt failed before collection because its temporary root was shared; subsequent runs use the existing private-profile runner. No product behavior or deadline is changed to hide these setup errors.


Final fixture qualification `fixture-final-integrated-1`: all six identified cases pass in10.87s with unchanged sources and original deadlines. Workspace fixtures now carry their real constructor-owned screen, read set and immutable caches. Registry/database replacement requires the exact prior cache to remain untouched; generation change still publishes the new default-only result. The presentation cancellation control checks live custody while the actual reader is held, releases it, then awaits cancellation and verifies retirement/retry. Its old ordering waited for cancellation before releasing a worker the current contract deliberately drains, consuming the barrier timeout before inspecting custody. Both test files pass lint/format. This corrects test contracts, not runtime latency; cross-platform requalification remains pending.


## One source parse per builtin manifest (OPT89, 2026-10-09)

The existing manifest independently read/parsed installed server.py for tools, resources and prompts. It now supplies one invocation-local AST to the same extractors; no-argument helpers still read fresh source. Subsequent manifest requests observe source changes and SyntaxError, and returned dictionaries/schemas never alias earlier results or Library descriptors. Original manifest-single-parse-red-1 completes content/freshness checks but fails at six reads instead of two for two requests; the source-change/error case passes. The first isolated manifest sample is16.557ms before versus6.388ms after, not a statistical latency claim. Initial GREEN is7passes with two optional mcp-unified modules skipped; final manifest-final-integrated-1 is17passes in1.81s, covering standalone helpers, custom provider, exact registered surface and Library contracts. The new module passes lint/format; server.py retains its two existing E402 findings and adds none. Independent review found no actionable issue.

Sequential full-profile quiet ABBA (manifest-a1/b1/b2/a2) completes all12 turns with three traces per process, zero checkpoints, unchanged source/HEAD and no detected native overlap. Baseline cold mean5.500424s, warm mean3.531417s; candidate cold4.832818s, warm3.726698s. Heartbeat stalls over100ms are22/26. This small variable sample establishes no whole-Send improvement or causal regression percentage; the target remains unmet. Only server.py differs in normalized loaded product code; manifest-comparison.json preserves the storage_admission.py newline-only difference. Exact samples/hashes and stage intervals remain in the external receipts. Retain the simple three-to-one source-read reduction for coherent request preparation, without presenting it as a successful elapsed-time iteration.

Existing CI provenance cannot establish whether the POSIX helper-start failure was introduced here. c504's original Ubuntu run fails before reaching its budget because its receive wrapper rejects _configuration_preparation; bdff/81c have no Actions runs. The later5ec receipt records33/26/24 against16. Loaded product hashes validate the respective source revisions; no baseline failure is invented. See ci-pause-provenance/summary.json.

## OPT-88 owner-approved refinement (2026-10-09)

The user approved omission of unused external catalog checks for builtin-only stock Send composition. The previous not-selected policy disposition is superseded by the ADR-225 Builtin-only composition refinement and Docs/superpowers/plans/2026-10-09-console-builtin-only-preparation.md. Implementation/qualification are in progress; no whole-Send improvement is claimed yet. Initial maximum and actual invocation remain fresh, as do composition checks whenever external tools are eligible.

### Completed c223 platform follow-through

Run37900514204 at c223d2c91dc852d1667b0545c8a0f0ec1fae71fe completed. The six corrected availability/cancellation fixtures pass on all three OSes (18/18 instances); local manifest controls pass8/8 on each (24/24). Ubuntu broad controls1449P0F42skip, macOS1452P2F37skip, Windows1436P13F42skip. Finite callback controls pass Ubuntu/macOS108P3skip and Windows109P2F. The downloaded reports contain27 failures across original probe budgets and UI/lifetime/skill/supplemental cases; they must not be reduced to a known timing-only failure.

Original helper-start counts remain above16 on Ubuntu32 and macOS33. Windows original native probe passes8/8, but its three Sends take12.824/11.828/11.291s. Those instrumented CI samples are not interchangeable with local quiet full-profile comparisons. Startup jobs fail on all OSes; raw-pin jobs pass on all OSes (job metadata, not newly downloaded numeric qualification here). No budget was changed.

Local receipts: claude-watch-final-gate-review/ci-c223/summary.json (sha256 fa9f1f9af9073de2fac2c96395fce7d66e9b53de5810c111ad0dca0c599f882b) and summary.md (dc84ad9041cbd16d0da22981daeacc075bb814a01da2e60c016ee40c4ab9111b), with62 JUnit reports and all three probes recording c223. Later OPT88 is not qualified by this run.

Read-only lifetime review identifies a supported stock-skill fixture cleanup-order race: creator retirement is attempted before the default-executor join delegated to asyncio.Runner.close; Ubuntu shows unrelated FTS backfill still active. Finite history/context failures have an unidentified remaining admitted operation after their selected callback tasks finish; current evidence does not establish product defect versus fixture race. Preserve that distinction pending exact-owner qualification. Receipt lifetime-review.md sha2560343247c2364c30d49d91cc79bfe0186636c144cd3cae163fd6b907b055fac0c.

### OPT-88 implementation and final local qualification

Root retained integration/native ownership; controller/provider implementation and baseline review ran in parallel with distinct files. The shared pure namespace rule accepts only exact frozen sets of nonempty builtin IDs for omission. It is anchored at its defining module, preserving pre-first-import substitution refusal. The issued preparation records whether external definitions were included, so an external-capable consumer cannot adopt incomplete data. Existing stock source/factory qualification, custom adapter routes, finite worker retention, policy reads and actual invocation stay in force. Builtin hash-free permission policy is unchanged.

The original controller regression fails on one unnecessary external read after verifying correct builtin catalog, one common permission read, loop inventory and native retirement (builtin-only-red-2, stable source). The earlier red-1 attempt observed the same failure but detected concurrent test formatting, so it is not source-stable qualification. Integrated selection:248 passed,7 failed in504.45s (builtin-only-integrated-1). All seven were new test assumptions: five attempted maintenance drain without first closing the producer; one invented a hash downgrade for the already hash-free builtin namespace; one expected stock loop inventory on the preserved custom callback route. Corrected tests retain original native lease-retirement checks and the actual policy/affinity contract. The affected selection and neighboring custom controls then passed12/12 in35.52s (builtin-only-integrated-2); product logic was unchanged. Together these qualify255 distinct cases, including original invocation, source, lifetime, first-import and initial-capture controls. A final provider-range formatter change is AST-identical to that tested product (builtin-only-format.json). No native correctness runs overlapped timing.

Shared code/tests pass Ruff; controller/provider have the same60 pre-existing diagnostics as c223, no added diagnostic. Changed ranges are formatted, YAML parses and git diff --check passes. Existing three-platform Preparing job now includes the four preparation/provider test modules under unchanged180s case and15min job budgets. Remote qualification of this new candidate is still pending.

Observer-free full-profile A/B/B/A (one cold plus two warm turns per process):

| Run | Send-to-adapter seconds (cold / warm / warm) |
| --- | --- |
| builtin-only-a1 | 5.205569 / 3.682034 / 3.569225 |
| builtin-only-a2 | 5.463631 / 3.927549 / 3.791663 |
| builtin-only-b1 | 4.501672 / 3.473575 / 3.078784 |
| builtin-only-b2 | 4.492469 / 3.595962 / 3.447551 |

Baseline c223 cold mean5.334600s, warm3.742618s (range3.569225–3.927549); candidate cold4.497071s, warm3.398968s (range3.078784–3.595962). Warm reduction9.18%. Send-period heartbeat stalls over100ms24→20, including response settlement. Each run has3 saved user/assistant pairs,3 complete trace links,0 checkpoints, stable source and NO_DETECTED_OVERLAP; third turn streams. Loaded product differences are exactly the four intended files after accounting for newline-only copies. Small headless/stub-provider sample; action return is not rendered feedback, physical terminal and real-provider acceptance remain unqualified. The result supports retaining the approved dependency refinement, not declaring the latency goal achieved. Receipt builtin-only-comparison.json sha256 8266d146b5409a8f9b1801cc460b813b4abffadf53679390061b4f822b9de745.


## Remaining-cost attribution after OPT88 (2026-10-09)

Source e3ee73c9d2409678aef0c959a647887ca113b2b9. The original configuration observer was reused without source changes: capture-after-builtin-1 completes three saved turns/traces, zero checkpoints, all42 QPC stages,12 returned bodies and161 checkpoints, source/bindings current, monitoring retired and no detected native overlap. Warm owned capture is0.717416/0.914793s: initial MCP maximum0.481953/0.329161s and its separate worker0.226709/0.577225s. Worker resource-scope setup is0.045492/0.280190s; RAG defaults0.088163/0.158039s; common capture0.070997/0.093896s. Nested intervals are not additive. RAG already has a singleton profile manager and warm config lookup; inclusive wall time does not prove repeated disk reads or exclude scheduling interference. No new optimization is selected from those labels alone.

Exact prior c223 POSIX helper census reconciles Ubuntu32/30/26 and macOS33/30/23 starts per Send. Both hosts have12 lifecycle/preparation starts on each warm Send. Repeated context, character and other presentation reads account for much of the remainder; these phase-wide counts include background work and do not prove redundant inputs or introducedness. Original preparation already shares nested same-owner admission. Cross-store durable effects and async boundaries do not justify one retained helper/lease. OPT10 remains the next bounded investigation: identify exact context snapshot invalidation/expiry reasons before changing presentation refreshes. Raw c223 receipts remain under ci-c223/console-pause-{ubuntu-24.04,macos-15}.

Exact e3ee CI37942366228 passes all four builtin-only preparation/provider modules on Windows, Linux and macOS:195 cases per OS,585 executions, no failures/skips. Manifest and Preparing/hook/question controls also pass. Broader CI remains red: Windows tab-strip/setup cases, creator/finite-context retirement, POSIX helper count and startup budgets. The partial snapshot does not qualify unfinished jobs. Parameter-free exact failures and artifact hashes are in ci-e3ee/summary.json and summary.md. The one-second Send and consistent sub-100ms physical-feedback targets remain open.


### Context-publication routing follow-through

The original e3ee context observer completes two source-stable, overlap-free three-Send runs, with42 matched QPC stages each and no observer gaps. Version1 records4/3/3 reads, but cannot classify key index20's changed enum; version2 qualifies the exact resident enum and records4/2/4 reads with zero unknown changed fields. Version2 clipped awaits are763.010/841.152/1073.460ms. One warm status-only invalidation at age0.625s awaits106.184ms; two longer reads are correctly rejected after payload/display changes. These inclusive async intervals do not establish removable critical-path cost. No key/TTL weakening is selected. The bounded next experiment routes changed context publication through the existing manual-Preparing display path, retaining captured FULL callbacks on overlap and legacy/custom behavior. Plan and ADR226 record the trailing-demand regression that a direct callback swap would introduce.

The e3ee startup-cohort CI artifact adds two OPT88-related failures in test_console_snapshot_source_contracts.py::test_empty_mcp_maximum_preserves_plugin_route. Both still construct the provider and refuse unavailable plugin authority as required, but assert an external catalog read for an empty MCP ceiling. Contract reconciliation is pending;585 preparation-module passes do not erase these omitted controls. The Windows context/skill reports complete240pass/12skip/0fail. Three ordinary probes were still running at the last snapshot. Preserve ci-e3ee/follow-through-summary.{json,md} alongside the earlier partial reports.


### OPT10 retained result
The snapshot captures FULL and the optional existing poll-display callback once. Its small wrapper retains captured FULL when a sync is running or the screen has replaced that callback; exact bound receiver/body identity recognizes ordinary repeated attribute access without invoking custom equality. Otherwise it enters the existing guarded manual-Preparing route. No intervening await, context key/TTL change, new coordinator or native lifetime owner was added.

Original source: context-routing-red-1 fails the two intended routing checks after successful original transcript/control publication; four compatibility/overlap cases pass. The temporary direct-swap negative control fails exactly on lost _console_sync_requested demand (context-routing-overlap-negative-1). Final integrated selection passes51 cases in167.63s, including changed runtime configuration, FULL replay, final deferral, owner changes, exact context-host retirement and both corrected plugin controls. The primary publication test was then strengthened to require successful True completion; all six final routing cases pass in24.98s. Source/HEAD stable in each run. Ruff checks pass; independent review found no actionable issue.

Quiet full-profile sequential ABBA plus one BA confirmation, cold/warm/warm seconds:

| Run | Seconds |
| --- | --- |
| baseline A1 | 4.601554 / 2.763011 / 3.315037 |
| candidate B1 | 4.463632 / 2.873153 / 2.757184 |
| candidate B2 | 3.975742 / 2.817478 / 2.755896 |
| baseline A2 | 4.110389 / 15.515540 / 4.042185 |
| candidate B3 | 4.433306 / 2.731259 / 2.867563 |
| baseline A3 | 4.329425 / 3.398638 / 2.629956 |

All18 saved turns complete their traces with zero checkpoints, unchanged sources and NO_DETECTED_OVERLAP. The only normalized loaded product difference is console_spend_projection.py; raw_participants.py, storage_admission.py and server.py differ only in line endings. Candidate source is the uncommitted, fingerprinted change above c9fb3fbc6a; baseline is e3ee73c9d2, with identical product code to c9fb before this experiment.

The unexplained15.515540s baseline outlier is retained. Its broad stage slowdown prevents treating the initial56.3% mean difference as a causal speedup. The separate confirmation pair is3.014297s baseline versus2.799411s candidate warm mean (7.13%); all six candidate warm samples are2.731259–2.873153s. Six-sample warm medians are3.356838/2.787331s. This small headless/stub-provider sample supports retaining the bounded work reduction, not a percentile, cold-speed, consistent heartbeat or physical-terminal claim. One-second Send and sub100ms rendered-feedback acceptance remain open. Receipt context-routing-comparison.json sha256 2a887cd1214ed79e4e3a06506ec321e06524664fdceae55b9c4bec0d24b0924e.

The existing three-OS Preparing job now includes the six publication cases and the two empty-ceiling plugin controls with unchanged case/job budgets. Exact final-source remote qualification is pending. Broader context-only routing outside manual Preparing, narrower invalidation dependencies and longer TTLs are not implemented; the wider source/transition proof obligations in ADR226 remain in force.

The OPT88 plugin-contract correction is pushed as c9fb3fbc6a. Its13 focused native policy/custom-route controls pass with stable source; final integrated routing also reruns both plugin refusal cases successfully. This resolves the local read-expectation mismatch, not unrelated CI failures.


## Current MCP partition and feedback (bae450ce5e)

The unchanged configuration observer records warm initial MCP maximum0.583085/0.501208s (capture-after-context-routing-1, SHA183471c5a4bdf08dff2a64a92ad6c2a8efdb611af43ea45db1530327c537564a). Source review rules out a duplicate common permission payload in this initial stage; the later shared provider policy read belongs to a separately fresh dispatch boundary.

The bounded MCP extension then records permission checked-read0.350446/0.116482/0.132144s and catalog checked-read0.110222/0.535214/0.446020s, cold/warm/warm. Inventory costs5.758/5.888/7.528ms; projection is small and no audit body enters. mcp-maximum-detail-1 completes three saved turns,18 returned bodies,245 checkpoints and42 exact original QPC stages; sources/current bindings and monitor retirement qualify, with zero issues/overflow/unfinished captures and no detected native overlap. Inclusive worker intervals are not removable-work estimates. SHA709edd6d4015c3cccd4a6891cf38f18bc9b72d2ed9863bcb7c7859bd4a5b127a.

OPT23 remains deferred after source review: initial verified control reads and final named-association validation occur at different moments. A fused walk would need all ancestor handles retained, reader descriptor adaptation and unchanged final bottom-up validation. A retained root descriptor alone is insufficient; no small duplicate traversal was found. No native estimate is claimed.

OPT28/73 remains rejected in its original two-path form. A distinct initial-only experiment is recorded in the new 2026-10-09 plan because OPT88 removes ordinary composition's catalog read and the new experiment leaves that consumer unchanged. It still needs roughly200 net product lines and fresh whole-Send evidence; no adoption or saving is claimed.

frames-after-context-routing-1 supplies actual Preparing compositor cells22.413/15.820/15.186ms after original action entry. All three have current received claims and runtime custody; bindings, monitor retirement and sampler join qualify with no span/queue gaps or overflow. Three saved replies/traces complete, source stays stable and no native overlap is detected. SHA8239cf0e351a71a5f12f81b20e594d90fa1634eb86df746ec54556782e76a87c. This is headless supplied-frame evidence, not physical terminal flush or a percentile guarantee.


## OPT28/73 initial-only experiment rejected (2026-10-09)

Rejected; no initial-catalog implementation retained. The original source fails only the causal four-versus-one scope assertion (initial-catalog-red-1), after real payload and ownership/retirement proof. Candidate integrated controls pass77 with two new test expectation failures; both expected only the external governance gates copied from composition. Initial capture also calls the inventory gate. The corrected exact ordered three-gate expectation passes both originalbae and candidate2/2, preserving all custom argument/thread/result assertions. Candidate total is79 distinct passing cases. Source/HEAD stayed stable, static checks pass and independent review found no actionable correctness issue.

Quiet sequential ABBA followed by BA confirmation, cold/warm/warm seconds:

| Run | Seconds |
| --- | --- |
| A1 baseline | 4.176367 / 2.848485 / 2.315082 |
| B1 candidate | 3.874678 / 2.672952 / 2.362119 |
| B2 candidate | 3.719994 / 2.800344 / 2.378922 |
| A2 baseline | 4.398010 / 2.730120 / 2.522113 |
| B3 candidate | 3.728209 / 2.768415 / 2.341416 |
| A3 baseline | 6.314893 / 4.866397 / 4.480464 |

All18 saved turns settle with zero checkpoints; source hashes/HEAD are stable and no native overlap is detected. Every run within each arm loads identical sources. The only normalized product differences are the two candidate files; three other raw hash differences are line endings only.

Initial ABBA warm means2.603950/2.553584s differ by50.37ms (1.93%). Cold means4.287188/3.797336s are suggestive but too few to establish reliable cold gain. A3 slows broadly across preparation, commit and postcommit phases; its cause is unproven. The six-sample aggregate22.46% is not a causal saving. These results do not justify approximately213 net product lines of qualification and effect-handling machinery. Independent review agrees. Preserve the count/behavior proof separately from the adoption decision.

Exact rejected source/tests and manifest are retained in task artifacts initial-catalog-candidate/rejected-final; initial-catalog-comparison.json SHA329f9cc41a657c190d93d4c6071b2ebc0041eac31e37c73022e9437495fa43c1 includes every raw run and interpretation. The two product files and existing source-contract test are restored to publishedbae, and both experimental test files removed after verifying the archive. OPT88 and the retained context-publication route remain shipped. OPT28/73 remains available for later review, not scheduled work or a promised saving.


## Exact retained-source CI follow-through (bae450ce5e)

Run37952590183 completes the Preparing selection with314 passes on Windows,
314 on Ubuntu and313 passes/one failure on macOS. All197 builtin-preparation
cases and both empty-ceiling plugin controls pass on each host. Five of six
new routing cases pass on macOS; the primary successful-publication assertion
receives `[False]`. The context read and issued publication worker exist;
the boolean belongs to the original UI refresh callback. Original refresh
permits deferral on overlap, maintenance/replay, owner drift and configuration
contention. The fixture discarded its initial FULL result and checked only
idle/request/current-record state. The artifact does not identify the exact
False branch, so neither a product regression nor a fixture-only cause is
established. Preserve final successful-publication proof while investigating.
Receipts: ci-bae/summary.json and preparing-poll-{windows-2022,ubuntu-24.04,macos-15}.

The earlier completed e3ee ordinary probes all save three replies with three
complete traces and zero checkpoints, but fail performance acceptance:
Windows full Send phase15.833s exceeds15s; Ubuntu helper starts38/28/24 and
macOS35/31/25 exceed16. Helper totals include response settlement and are not
pre-provider counts. Exact source comparisons and individual failures remain
in ci-e3ee/final-ordinary-summary.json. Current bae ordinary jobs remain pending;
startup and other broader jobs remain red. No timing deadline, helper budget,
source gate or ownership assertion is relaxed.


### Current post-save attribution and OPT05 review

The existing basic observer was reused unchanged on bae450ce5e. All three
saved turns settle, with zero checkpoints, stable source and no native overlap.
All502 events balance into242 START/RETURN pairs and18 guard-entry pairs;
42 original QPC stages match,60 targets remain current, monitoring retires,
and there are no pending/unmatched/overflow events. This preimported diagnostic
does not qualify cold-start or observer-free speed.

Warm saved-to-trace intervals are1.330787/1.350263s. Disjoint named work covers
98.77%/98.43%: composition0.318423/0.096424, history0.103592/0.279377,
fresh hook admission0.204611/0.156006, checkpoint0.120570/0.069991,
initial chain-maintenance await0.109660/0.152951, five later guard entries
0.117094/0.169743, run-log binding0.136874/0.120938, memory preflight
0.031399/0.032464 and other named work0.172231/0.251132 seconds.
Residual is0.016332/0.021237s. Nested spans are counted once; inclusive awaits
include scheduling and are not estimates of removable work. Cold hook admission
takes1.526082s here, without a native-versus-wait partition.

OPT05 source review found no new duplicate inside RunLogWriter.bind: one bind
per run, immediate return on repeat binding, one acquired root and one shared
sensitive-path context for its two path checks. Existing path-preparation tests
already cover this sharing and custom callbacks. The remaining checks surround
distinct directory/migration effects; repeated scoped-source checks are pure
metadata. No new guard pool, history deferral, permission cache or run-log change
is selected. Preserve this ruled-out investigation with the other candidates.

Receipt: postcommit-after-routing-1/attribution-summary.json,
SHA380e76be8d7f2ce8889de1331e675f764605d22cb6aa500cbb71cebeb592b063.


## First same-owner display synchronization and publication qualification

The original first display sync invokes the draft-change callback even when
the visible and active session IDs are unchanged. That callback unconditionally
cancelled the hook Send generation. Two original-body regressions fail on bae:
the generation changes from 0 to 1 after the real initial same-owner sync.
The successor-session control passes and demonstrates the distinct invalidation
that must remain. The correction checks those existing owner IDs at the callback
boundary; an absent store or changed owner still cancels. Draft, undo, first
indicator refresh, permission checks and durable dispatch retain their behavior.
Existing ADR-225 applies; no new coordinator, cache or native read is introduced.

The causal run same-owner-red-1 has two intended generation failures, 11 passes,
and one separate exploratory review-setup failure. The exploratory setup holds
the first FULL and consequently leaves runtime attachment unreconciled before
the modal can open; it was removed rather than represented as a reproduced live
continuation race. Existing actual modal/review controls are used for final
integration. The deterministic generation bug does not establish the cause of
the historical intermittent handed-off review refusal.

Original context-publication observations also demonstrate natural FULL deferral
while maintenance/replay and readiness/context publishers settle. The final
controlled publication succeeds after they settle. This does not identify the
exact macOS CI False branch. The fixture now awaits an actual changed FULL
completion token and those existing publishers within its original five-second
budget. It still requires True completion, transcript/control publication, no
core/roleplay reconciliation, the held received record, retained draft and zero
provider calls. No product flags are cleared by the test. Reader-drift test
cleanup now removes its override only if installed, so failed publication cannot
be obscured by a secondary AttributeError.

Diagnostic limitations are retained: the first context and hook observers did
not install in the ordinary bootstrap process; their passing tests are not
observation evidence. A second hook observer failed before tests because local
PY_UNWIND is unsupported on this Python 3.12. The corrected START/RETURN observer
runs two passing tests, but each broad receipt has one unmatched snapshot after
an exception and is incomplete. Its paired same-owner callback observations are
useful source facts, not a complete failure explanation. The corrected context
observer supplies six complete, retired receipts with actual callback events.
See context-publication-diagnostic-2 and hook-continuation-diagnostic-{1,2,3}.


Final frozen-source integration passes all 54 cases in 416.97 seconds
(same-owner-final-integrated-1), covering both changed fixtures, real hook review
accept/decline/failure/navigation, current permission-owner/reader refusal,
captured review lifetime, cancellation retirement and actual successor switching.
Source hashes and HEAD remain unchanged. Ruff check and format check pass for
all four modified Python files; diff whitespace check and independent reviews
pass. Only existing dependency deprecations appear; the failed exploratory
teardown warning does not recur. These correctness checks are not speed evidence.


## Current real-provider qualification (a3f5ba2517, 2026-10-09)

The existing owned-server procedure was reused with only its revision pin and
fresh private-profile prefix changed. Normal provider setup selected
DeepSeek/deepseek-chat. Three real replies retain CEDAR across the conversation;
six messages, three complete calls and three response links are saved with zero
pending checkpoints. Product/Test source hashes and HEAD stay unchanged. The
original user configuration is unchanged. The App exits through Ctrl+Q and the
server through its exact owned stop request; its tree is empty, native identity
is released and no forced retirement occurs.

| Original stage timing | First Send | Second Send | Third Send |
| --- | ---: | ---: | ---: |
| UI action to provider adapter | 5.359 s | 3.547 s | 3.719 s |
| UI action to controller entry | 1.375 s | 1.344 s | 1.452 s |
| Durable save completed | 1.859 s | 1.687 s | 1.719 s |
| Saved turn to trace reservation start | 3.063 s | 1.657 s | 1.563 s |

This is one functional real conversation, not a matched causal comparison with
the historical dd80 run. These rounded original stage timestamps do not measure
physical input-to-render latency. The one-second target remains unmet. The
already qualified headless Preparing frames remain separate evidence. Initial
preparation and post-save orchestration are still the main remaining intervals.
Receipt: current-real-deepseek-1/real-uat-summary.json,
SHA300683f6b3925f272cf63ac127fd19713ef0d9898c415bf8bcec4eb01b7edef3.

**FOLLOWUP-16 — Enter on a collapsed pasted draft.** The first turn was sent
through Enter and the second through the Send button. The third collapsed
pasted draft did not dispatch on Enter in the served browser terminal; the UI
showed its expansion control. Explicitly expanding the text and clicking Send
completed the exact third message. Preserve this observed focus/paste issue;
its cause and introducedness are unproved. The run qualifies Enter/button/button,
not Enter/button/Enter. No fourth provider call or unrelated UI change was added.

### Completed bae ordinary CI

Run37952590183 has completed with failures. Unique test-ID totals are Windows
1684 passed/18 failed/42 skipped, Ubuntu1694/2/45 and macOS1696/5/40. Each host
also executes five repeated passing IDs; raw pass totals are1689/1699/1701.
No duplicate ID has conflicting outcomes. Windows first-Send native opens are
41,360 against40,000 (later24,256/23,198); Ubuntu helper starts43/23/22 and macOS
35/29/24 exceed16. Counts cover the full Send phase, including response settlement,
and are neither distinct-file counts nor pre-provider-only counts. All other
computed phase budgets pass. All three hosts save three replies/traces/links
with zero checkpoints. Both permission-owner publication variants and corrected
empty-ceiling plugin controls pass everywhere; hook-review continuation failures
remain on Windows/macOS. The current a3f5 local54 passes do not clear remote failures.

Selected original bindings remain current and monitoring retires with zero
overflow. Exceptional timing gaps remain; report-time-only source manifests do
not prove whole-run source stability or application native retirement. Same test
coverage on earlier commits does not establish introducedness. Receipt:
ci-bae/final-ordinary-summary.json,
SHA4d339ac6dc4154af587fcdf83b51f7c7e1cdc929c9e7dfa4869093a76e05ad82.


### Remaining repetition: count interpretation and bounded source review

The bae Windows receipt partitions native-open attempts into main/worker counts
5,386/35,974 on Send1, 4,164/20,092 on Send2 and 4,936/18,262 on Send3.
Native rows intentionally omit caller ancestry, so they cannot identify exact
owners, distinct files or which guard is redundant. WindowsOS._named_stat calls
_parent to open the drive root and every ancestor before the named leaf
(windows_files.py); repeated metadata proofs amplify the number of handles.
The separate startup census budget of 1,033 counts resident project modules in
sys.modules, not native opens. Neither count means that many different content
files are read per Send.

Higher-level original caller observations record 23/17/23 main-thread
run_console_config_sync entries (inclusive entry time0.907/0.534/0.723s) and
7/6/5 worker spend-projection read_current entries (1.814/0.507/0.251s).
Storage acquisitions number265/196/204. These independent full-phase counts
include reply settlement and cannot be joined into native-open attribution or
added as serial savings. Repeated display reconciliation remains the strongest
recorded logical-operation lead under existing OPT10/OPT60; broader narrowing
still needs its existing source/transition proof, not a looser timer or budget.

OPT03 source review confirms that the saved-config profile name and cached
runtime selection are distinct inputs. get_user_data_dir already shares nested
config operations, uses a proven companion when available and batches warm
posture observations; cold creation/hardening and fallback remain meaningful.
The comparison cannot be replaced by cfg.profile_data_dir. Custom getter
retarget refusal is explicitly tested. No new naming-only shortcut is selected.

OPT05 controller/provider review likewise finds no new bounded duplicate:
run_turn and _run_one guards surround actual source selection, callbacks and
effects, while worker guards reacquire inside their real execution boundary.
Run-log binding and within-entry batching already share their finite owner.
Do not remove the later guards or add another eligibility framework for an
unmeasured small return. No product change followed these two source reviews.

### Current caller partition and selected experiment (2026-10-09)

The bae fallback-entry partition is context publication4/4/6, recovered-image completion2/2/2, transcript poll4/1/5, FULL coroutine with scheduling origin unavailable12/10/10, independent controls1/0/0. Inclusive entry seconds total0.907/0.534/0.723; this is native configuration fallback entry work, not FULL-pass counts or removable pre-provider delay. Source/limitations are retained in ci-bae/windows-config-fallback-origins.{json,md}; JSON SHA4255b0047790da78dec2ad9a755238ed8abd8f9a310027d0289174bb403005f0.

**OPT92 — recovered-image completion routing (experiment pending).** New message selection publishes even for an empty real metadata result, which is required to release transient/remote suppression. Route that one completion through the existing live-poll guards and display/exclusion/replay owners, preserving FULL on overlap/custom/refusal. Keep actual native reads and other image actions unchanged. See Docs/superpowers/plans/2026-10-09-console-recovered-image-publication.md. Phase-wide work counts cannot predict whole-Send benefit.

**OPT10 context-key follow-up (retained for correctness; speed unproved).** The stock reader consumes the held compaction echo but not run status or active_run_id. A narrow stock-only dependency correction could remove status-only invalidations and add the actual held-echo dependency; custom/replaced readers must retain conservative lifecycle invalidation. Preserve payload/display/context/settings/repository/config fences and TTL. Requires causal reuse, same-status echo-change and stale-result controls; keep separate from OPT92.

Within-FULL source review found no new safe consolidation: pricing equality compares values and its stable models.dev label, nested getters already share one native operation, and core/roleplay gates are separated by real awaits. No cross-await lease, extra config cache or pricing-fingerprint change selected.


OPT92 result: candidate22 product lines and candidate-only tests archived outside the checkout at image-publication-rejected-final; original product restored. All8 final image controls and6 maintenance controls pass, including the overlap negative control. The stable A2 baseline warm mean4.700s and B1/B2 combined4.731s are tied; subsequent BA favors candidate4.516s vs6.674s, while unexplained slow A1/A3 samples dominate the32.17% aggregate. This is inconclusive, not evidence of a32% attributable gain. All six process samples and hashes remain in the plan and image-publication-final-comparison.json (SHAb62b4ad323aef28b54e04f17f6cb993da22d6fb9ee0ed75c11c8898800fef27f). No further timing retries selected. Keep the independently justified maintenance fixture repair.

OPT10 dependency follow-up is now planned under AC17 with distinct reader-anchor/key/test owners. Real resume clears the held echo before provider-resolution await without another revision/status change. Plan: Docs/superpowers/plans/2026-10-09-console-context-presentation-dependencies.md. No performance adoption claim pending causal tests and sequential measurement.

OPT60 additional unimplemented caller: fenced citation-count completion requests FULL around chat_screen.py:17010. No measured native origin was attributed to it in bae; defer until actual runtime attribution rather than adding another speculative callback route.


OPT10 dependency correction qualified: the actual held-resume interval reproduces stale cache and stale in-flight publication on unchanged a017; replacing a warmed reader/getter also reuses stale data. Exact-stock echo dependency plus conservative tagged custom fallback fixes these without changing1sTTL, other owner/revision/config fences or action/dispatch/native lifetime. Final57 integrated cases and2 rendered-feedback cases pass; Preparing31/44ms and input frames6/8ms are headless-only samples. Context dependency plan carries original RED and all limits.

Quiet ABBA warm means worsen4.108-to6.057s; baseline cold varies4.043-to21.416s across every stage. No causal speed improvement or47.43% regression is established. Retain this small correction for independently proved correctness, not a claimed latency gain. No favorable retries selected. The existing detail observer is run separately on both arms for logical count/consumer attribution.

Current source review confirms multi-path Windows snapshots already deduplicate ancestors and use retained relative parent handles in two proof passes; L3/L4 are integrated. Remaining scalar raw._check_parent_pins named stats restart the drive/ancestor walk per directory: this is existing OPT17/25, not a new missed batch. Historical64checks/442opens/.226s is an instrumented hook ceiling, not current whole-Send savings. A fresh batch must still prove FD association, error/order/DACL and uncertain-close equivalence; no implementation selected.

Controller review finds first-request/run-log/schema data already shared through the existing plan. Personal-context preview is an optional other caller. Remaining live gates cross source/callback/effect boundaries. Reuse current original AgentService/run-log/lifecycle/request-builder observations before another consolidation; no new owner/cache framework selected.


## Current original-function diagnostic

The same existing83-code DetailSpans observer ran sequentially on the candidate and baseline, each with3 saved replies/traces/links and42 stages, source/bindings current and monitoring retired. This preimports selected owners and is diagnostic-only. Pre-provider logical version reads are baseline2/5/5 and candidate2/6/6. Critical reads stay one dispatch per Send plus one preaccept per warm Send; all five original version calls total1.90ms baseline and3.44ms candidate. Extra candidate calls are presentation reads, with differing elapsed/TTL windows; no causal invalidation regression follows.

Warm saved-to-trace is baseline1.129/1.904s versus candidate2.022/1.872s. Recurring postcommit hook_admission_reason is0.183/0.237s versus0.331/0.358s, with nested locked_hooks_config_snapshot entry-to-first-yield0.088/0.125s versus0.171/0.191s. These are original inclusive wall intervals, not removable costs. Each Send retains six complete admission first-yield pairs. Context records have no gaps; hook/raw ancestry has92 depth misses per arm and seven unpaired ordinary starts (exception unwinds were not observed). Both generator-label unfinished counts are0. No blanket observer-completeness or speed claim.

A host snapshot during the candidate diagnostic shows54% aggregate CPU across12 logical processors; this isolated observation cannot explain the earlier quiet timing. Analysis and full limits: context-dependencies-detail-comparison.{json,md}, JSON SHA19c40b940b5a4b862514cfdc5daee8917a0719db466ac8737155b4521b47902c. Source reviewed at committed7d4039f6ac; all50 files captured by integrated qualification match its frozen bytes (context-dependencies-committed-source.json).

No new safe snapshot/L3/L4 contraction was found. Ordinary raw config/hook/MCP scopes normally carry one selected-parent pin, so replacing its scalar named stat with the existing forward/reverse multi-path proof would roughly double ancestor opens. Multi-pin directory-creation operations are a separate unmeasured case with error/close-contract differences. OPT17/25 remain deferred. Current evidence prioritizes OPT41's hook/config read declarations and OPT06's explicitly owned auxiliary prompt-history ordering for source review; no new authority or persistence contract is implemented here.


### 2026-10-09 history ordering and finite config follow-through

OPT06 remains deferred after tracing the actual accepted-turn owner. PromptHistory
publishes recall entries only after its original FileJob retires. Its safe point
waits on the append lock; it is not an owner for a scheduled but unstarted task.
After history, preparation publication, commit-owner validation/release, final
hook reconciliation and trace/provider preparation are ordered effects. There is
no measured independent window for useful overlap without moving these effects.
The earlier suggestion of a small structured overlap is withdrawn. A post-dispatch
queue would require explicit recall, ordering, failure, Stop and drain semantics;
the current .1–.35s inclusive history interval does not justify that new contract.
OPT31 tested a different pair but the same finite-native overlap approach and
was removed after no observed benefit (warm3.255→3.332s); it is not repeated here.

OPT41 cannot narrow the generic config operation globally. Stock hook snapshot
entry consumes config under the writer lock, but nested cold data-root resolution
can initialize config and custom writers need backup/temporary membership. The
next bounded observation partitions original companion member validation before
considering a named narrower read. OPT04 still has repeated full checks at nested
config-helper entrance (classification, raw scope classification, final selected
entry); actual file/lock-effect boundaries must remain separate. Neither source
observation is a demonstrated whole-Send saving or a selected new API.

### 2026-10-09 exact companion-loop diagnostic

At unchanged product2b5e6eef1b, companion-members-detail-1 completes three saved
replies and42 stages in the original full-default profile, source/bindings current,
no detected overlap and monitoring retired. All eight attributed hook companion
calls return None before the member loop (224 scoped LINE callbacks; zero member
rows, overflow, unmatched or unfinished companion calls). The two warm final-hook
companion prefixes take5.888/5.849ms. No unused-member-loop saving exists in these
observed calls. Ten separate ancestry-depth misses and4116 unscoped LINE callbacks
remain explicitly unattributed; this is not complete process/native attribution.
The stock member-role path was not exercised. The one-off observer is archived
with the receipt and restored out of active source.

Do not generalize this to all OPT41 costs: companion fallback performs a second
related-member acquisition. Source review proves fresh random temporaries bypass
L4 only for bound holds; unbound holds skip per-path evidence. The actual bound
bit and fresh-child branch must be observed before choosing a narrower contract.
Summary SHA256:6639e4f7334828148767067aa7fdc399f3b4ae95ff3d28d8c912d4646d4abcf8.
This diagnostic is not a quiet latency comparison; no product change follows.

### 2026-10-09 actual watch-miss attribution

The unchanged-product evidence-misses-detail-1 completes three saved replies,
three linked complete traces and zero checkpoints; source/bindings stay current,
monitoring retires and no overlap is detected. Its4096-row cap starts56.95ms AFTER
the third provider entry; every recorded pre-provider call finishes. Whole-run
settlement attribution is incomplete:210 overflow/unmatched and294 ancestry
misses, with no unfinished recorded call. All observed reusable holds are
unbound and fresh_children=0; random temporary names do not defeat L4 here.

Warm Send windows contain165/129 reuse calls. Respectively140/113 have no observed
miss and take.319/.149s inclusive. Of25/16 full observations, expiry accounts for
14/7 (.744/.178s), generation7/6 (.095/.063s), and quiet-false4/3 (.037/.074s).
These cross-thread nested intervals are not additive savings. On the original
dispatch thread, all reuse totals.204/.110s; its three full observations per Send
are.028/.035s. Final hook reads have only two observations each (.022/.016s).
Quiet-false returns are notification/error, not closed/empty, but their actual
file cause is not observed. Current evidence does not justify OPT41/82/93 policy
changes. OPT82 selective by-ID invalidation also lacks an existing dependency
owner; it would add indexing/history, absence/race/overflow and retirement rules.
Preserve current invalidation rather than introduce that machinery on this cost.

Next focus is the issuing domains behind the remaining calls, distinguishing
critical work from activity merely inside a Send window. The same observer gains
only bounded code-location ancestry and task IDs; the earlier capped receipt is
retained. Analysis: evidence-misses-detail-1/attribution-summary.json, SHA256:
134995b42ef3fb759912ce38ffc2093738ecdb9798c933882cc19be51d32db9d.


### 2026-10-09 admission caller ownership

The final one-off observer at unchanged product2b5e6eef1b records124/129 warm
pre-provider reuse entries in2.903/3.162s Send windows. Reuse union occupies
.248/.487s; saved-to-trace remains1.096/1.508s and action-to-controller1.441/1.230s.
Connection-creation ancestry accounts for17/20 entries (.038/.162s inside reuse),
including Characters11/13, AgentRuns4/5 and Workspace2/2. These are admission
entries, not successful handles, complete connection costs or removable delay.
Generic run_owned_db_call7/9 and worker_guard10/10 entries remain unclassified.
Version-connection entries link to presentation, not dispatch:2 then4 IDs on
Send2,4 then6 then6 on Send3; the last two start1.299s apart. No violated TTL
or duplicate revision is proved. FULL config entry counts8/10 and readiness6/6
likewise do not establish avoidable reconciliation.

All three saved replies/traces/links complete with zero checkpoints; sources
remain current, no native overlap is detected and monitoring retires. Zero watch
overflow/unmatched/unfinished;305 ancestry-depth misses and285 eight-frame
truncations remain explicit. Worker task0 cannot identify its awaiting owner.
Inclusive cross-thread spans are not additive or a quiet speed comparison. Exact
observer source/patch are archived beside admission-callers-detail-1 and the
normal helper is restored. Summary SHA256: e3b7779986e3b0510fa3841e259e9df24855190b95914bbacb9f53ce9e7325bd.

DB connection reuse alternative is not selected. operation_owned_connection
borrows an existing handle but retires newly opened finite-worker handles. The
all-thread quiescence registry supports exclusive maintenance; normal close()
only retires the calling thread, and _core_closing refuses foreign-thread cache
retirement. Simply retaining executor handles would lose the existing App
shutdown guarantee. A new shutdown/settled-work contract is not justified by
these partial costs. Prefer grouping adjacent reads inside one existing owner;
OPT91 character-display scope capture is an existing candidate for that review.

OPT60 follow-through: publication of a prepared request does not make every
later poll FULL. Off-cadence ticks already use the light display route. The
accepted turn does not prove that live UI reconciliation has no pending effects;
promotion changes receipt ownership and draft/staged revisions. Preserve ADR226
periodic/direct/settlement/backstop work rather than extending Preparing by
removing a condition without origin evidence.


OPT94 source follow-through: the useful distinction may be finite operation
retirement versus final owner Close, including finite writers. A best-effort
checkpoint failure already permits accepted work with WAL retained; ADR225
promises application-crash recovery with WAL/NORMAL, not latest-commit survival
of power loss. Keep public Close and explicit maintenance/capture checkpoint
behavior, transaction outcome and exact physical retirement. Changed checkpoint
frequency, WAL growth and incidental power-loss protection are real policy
considerations; they require an explicit bounded plan/ADR clarification and
measurement before implementation. No per-finite-operation truncation requirement
was found in the reviewed contracts; final-close template tests do rely on it.


### OPT91 bounded resident mismatch result (2026-10-09)

The stock display now omits its outer DB precheck only when absent/different
resident identity already determines that refresh is required. Existing stock
reader qualification and the complete refresh worker remain; same-resident
revision/authority checks, custom sources, TTL, live actions and retirement stay.
No captured metadata or handle is transferred. Original causal RED establishes
2→1 callbacks,3→2 pairs,2→1 retired handles and663→405 native opens; direct action
remains4 callbacks/3pairs/1904opens.59 distinct integrated cases qualify after
one error-retry count fixture correction (58pass first run,6pass rerun); source
stability, lint/format and independent reviews pass.

Quiet ABBA does not establish whole-Send improvement: warm4.553→10.964s, including
a29.089s candidate turn that remains unexplained. The other candidate warm
samples4.495/4.515/5.755s also do not establish improvement beyond baseline
variation. No favorable retry or performance-regression-free claim. The small
change remains on the draft branch for reduced redundant native work; the
long-turn investigation continues separately. All12 turns persist/settle with
current source and no detected overlap; conditional loaded-module membership
is explicit in the comparison. See the character-refresh-precheck plan for
full samples, source differences and limitations.


OPT94 blocking hypothesis: CharactersRAGDB opens connections with timeout=15.
SQLite documents that TRUNCATE uses RESTART waiting semantics and may return a
busy status row rather than raise; the current close does not fetch that result.
The29.089s quiet turn includes a14.723s saved-to-trace interval, but timing
resemblance is not causality. Next original-body measurement isolates the actual
checkpoint statement and close before product selection. Official contract:
https://www.sqlite.org/pragma.html#pragma_wal_checkpoint.


### OPT94 measured disposition and OPT95 setup lead (2026-10-09)

OPT94 diagnostic sqlite-close-detail-1 completes on b5f9bc4c8f: 1 passed,
three saved/settled turns, 42 stages, unchanged source/HEAD and no detected
native overlap. All 596 selected SQLite rows finish with zero overflow,
unmatched or unfinished rows; original bindings and monitoring retirement pass.
Pre-provider checkpoint counts 19/16/15 take 47.795/43.559/36.656ms inclusive;
warm saved-to-trace checkpoints total only 10.881/11.621ms. This run does not
reproduce a long checkpoint stall or explain the earlier 29s outlier. OPT94 is
deferred rather than changing retirement policy on this evidence. Fresh
connection setup is larger: 20/16/15 branches and 1.353/.541/.952s inclusive,
with setup journal queries .175/.213/.139s. Concurrent/nested intervals are not
additive savings. OPT95 investigates the original connection preparation.
Exact observer/patch and attribution-summary.json/md are archived beside the
receipt; helper SHA256 cad7b899319329ae464e7237d4543ee123b31b60e926e3781ab8bf942405513a.
The tracked helper is restored; the overall latency target remains unmet.


### OPT95 original setup partition (2026-10-09)

OPT95 original setup diagnostic sqlite-setup-detail-1 completes with three saved
turns/42 stages, unchanged source and no detected overlap. All888 setup rows pair;
zero caps/unmatched/unfinished/error-cleanup markers; original bindings/source
and monitoring retirement pass. The separate raw ancestry observer retains92
misses. Warm24/17 setup operations have admission prefixes .2025/.0992s,
directory verification .2569/.1572s, main preparation .4076/.0857s, sidecars
.7783/.2656s and actual SQLite connect .0295/.0106s. Components partition each
operation; operations can overlap, so these totals are not Send savings.
The actual SQLite open is under2% of setup work; fixed artifact preparation is
55.8–69.9% and directory verification another15.1–24.9%. Warm saved-to-trace
has8/5 setups and .3322/.0975s artifact preparation. The earlier29s outlier is
still unexplained. Exact observer47a7cffb69565c08ff31f0ad2a83fc8b28c0c3d0a91db07a2c5cf4e3e963e695,
patch and attribution-summary.json/md are archived beside the receipt; tracked
observer restored. This supports the bounded fixed-inventory preparation
experiment under the OPT95 plan/ADR125 amendment, not a connection pool,
checkpoint-policy change or timeout adjustment. Product qualification is pending.

### Connection-owner review after OPT95 integration (2026-10-09)

The source review found no further repeated open within one existing stock
finite owner. Characters reuses its live registered thread connection;
operation_owned_connection borrows existing handles and retires only newly
opened/replacement handles. Configuration capture already deduplicates cleanup
owners, dictionary/world reads share its Notes handle, and character refresh
groups its metadata and group reads. Commit, context, preflight and trace work
occur in separate callbacks or effect/freshness checkpoints.

The setup diagnostic records immediate caller/thread/task, not database or
awaiting-owner identity. Its 24/17 warm opens and 8/5 saved-to-trace opens therefore
do not prove same-owner duplication. OPT96 records the larger lifecycle
alternative explicitly for later review; it has no measured saving or selected
implementation. Existing finite retirement and shutdown guarantees remain.

### OPT95 disposition (2026-10-09)

The qualified shared-parent experiment was rejected and removed. Native work
fell by 16 opens per unchanged setup, but quiet sequential ABBA warm means
3.030 to 3.224s did not establish a whole-Send gain; cold means 5.228 to 4.941s.
All twelve turns saved and settled with current source and no detected overlap.
The candidate's extra custody/source machinery is not retained on count evidence
alone. Fifty-five distinct targeted cases qualified after five fixture fixes;
six owner-privilege cases remain skipped. Exact source, negative samples and
full verification limitations are preserved in the
[plan](../superpowers/plans/2026-10-09-finite-windows-sqlite-preparation.md).

### Larger-domain attribution after rejected OPT95 (2026-10-09)

Offline original-body analysis of sqlite-setup-detail-1 retains the measured
source: warm configuration capture 1.280/1.032s includes initial MCP maximum
.587/.858s and the non-MCP worker .682/.163s. Within maximum capture, the original
read_sources body is .559/.845s: permission .147/.681s followed by catalog
.412/.164s. The slower permission scope enters in .103s, then spends .578s in
its original body/final-check/retirement lifetime. Current targets do not divide
that remainder. Pre-worker envelopes .0015/.0006s and return envelopes
.0268/.0122s do not support blaming inventory projection or initial dispatch,
and there is no explicit cross-thread submit/awaiter token for queue attribution.
These are nested diagnostic intervals, not additive or quiet latency savings.
Next observation is restricted to the original permission body and cleanup.

OPT96 source follow-through: Writing/Research scope services demonstrate serial
executors, ProducerLifetime, tracked physical futures and maintenance-thread
cache retirement. Console's serial stream/trace executors join at teardown but
depend on finite retirement; preparation uses the default executor. Current
Characters/Workspace close cannot retire foreign-thread cached handles, and
maintenance counts idle connections as live custody. Reusing an executor alone
is insufficient: retained ownership needs intake fencing, physical-job drain,
thread-owned close, uncertain-close handling and resume rules. No such new
contract or implementation is selected, and no saving is established.

### Original permission-load partition (2026-10-09)

permission-load-detail-1 on f87328a3a3 passes with three saved replies, complete
traces, no dispatch checkpoints, 42 stages, stable source and no detected overlap.
All six original permission bodies return at line795: readable is false OR the
file is absent. The discriminator is not observed. No reader, text read or JSON
parse is entered, so a read/parse optimization cannot explain this result.

| Send | Initial permission root | Composition permission root | Composition admission | Composition final check |
| --- | ---: | ---: | ---: | ---: |
| 1 | .272210s | 1.252503s | .975834s | .131439s |
| 2 | .119292s | .146343s | .105897s | .019220s |
| 3 | .153356s | .488207s | .102227s | .356420s |

Columns nest and are not additive. The slow warm final raw check is .356368s:
.352231s before parent proof, .003959s parent proof and .000177s after it.
The slow cold admission raw check is .807082s, including .128099s parent proof.
Mutation-fence acquisition is 9.2-27.1 microseconds; scope-exit cleanup is
.101-.585ms across all six. The original .578s remainder is not reproduced in
the same initial capture; the new slow composition check is independently
localized. No disk/CPU/GIL/lock explanation follows from wall intervals.

All50 permission rows finish;108 saved checkpoints from162 LINE callbacks;
zero permission cap/unmatched/unfinished, but200 ancestry-limit misses.
Base events have800 complete pairs and7 unclosed starts; separate hook/raw
ancestry has92 misses. Initial maxima have same-thread read_sources containment;
composition has unique prepare_console_tools containment and stock source
support, without an async-to-worker identity token. Heartbeat summaries have
no per-gap timestamps, so overlap cannot establish scheduling causality.
Original bindings/current source and monitor retirement pass; global masks0.
Exact observer ca0f2fe98823b47c7a34e1af5b59b29c55834c32ae97bc65560154a4789e2865,
patch and repeatable attribution-summary.json/md are archived beside the receipt.
Tracked helper restored. Diagnostic timings are not quiet acceptance evidence.

Source review finds each witness already uses one control-record/registry read
for both startup and source-scope derivation. Initial read pins and final named
tree validation serve different proof moments. Windows derived-witness reuse
remains disabled independently of acquisition watch reuse. OPT97 preserves the
small adjacent selection-proof alternative; the next diagnostic partitions the
larger shared source proof using wall and current-thread CPU time. No product
change, cache, weakened gate, retained handle or favorable timing retry follows.

### Shared recovery proof and native cost follow-through (2026-10-09)

permission-source-detail-1 passes at f87328a3a3 with unchanged source, no detected
overlap and three saved/settled turns. All six loads take line795 without reader
or parse. All42 witnesses perform a full control observation:12 for cold initial,
six for each other root. Each warm load observes four times during admission,
once in its readable/default body and once at final publication. Initial control
preparation totals31-66ms and final named trees42-74ms per warm root; record bodies
are1.5-8.8ms and registry9.7-24.9ms, nested within initial preparation.
Warm initial roots wall/CPU are244/203ms and261/203ms; composition150/47ms and
202/141ms. These are diagnostic observations, not quiet Send savings. Windows
thread counters quantize to15.625ms despite advertised100ns resolution; short
CPU deltas can exceed wall or report zero. Process CPU includes other threads.
Prior807/356ms check outliers did not recur, and their cause remains unknown.

All276 recorded rows complete with zero caps/unmatched/unfinished;562 ancestry
misses, seven unclosed base starts and92 separate hook/raw ancestry misses remain
explicit. Source/current bindings, local-mask retirement and original test
completion pass. Exact observer948d528dc735044b973db42609f8fc685fd648fc3668d5d3ea3717e5c9abf4bb
and repeatable analysis/summary are archived beside the receipt; helper restored.

OPT90 refinement considered, unselected: inherit local-NTFS qualification only
through a synchronous metadata tree's held relative parents. Both source reviews
identify an unresolved held-parent junction/reparse ABA case; checking roots and
final parent identities alone does not establish equivalence. Do not implement
this shortcut. Another unmeasured alternative keeps per-handle checks but obtains
the filesystem name through the already-used NtQueryVolumeInformationFile API
(FileFsAttributeInformation), retaining device/remoteness checks. Original ntfs
cost must justify any experiment; the earlier native_identity ceiling does not
measure every metadata-tree child qualification. Step37 uses the existing native
schedule/retirement test without modifying its assertions or product behavior.
Primary API references: [NtCreateFile](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntcreatefile),
[volume query](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntqueryvolumeinformationfile),
[filesystem attributes](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_fs_attribute_information).


## Original native metadata cost discriminator (2026-10-09)

OPT90 remains deferred. The original positive Windows snapshot control passed
1/1 in1.10s on f87328a3a3, with the product unchanged. The ten-node tree took
5.973ms including its temporary original-code observer: twenty open_handle
bodies totaled2.744ms, twenty ntfs bodies0.433ms, fifty info bodies1.223ms,
ten security bodies0.445ms and ten _stat_handle bodies1.058ms. These are nested
inclusive totals, not additive costs or predicted Send savings. This one small
operation does not bound all profile workloads. The direct per-handle volume
query alternative is not selected; repetition of whole admission observations
remains the stronger architectural lead.

The existing exact schedule/identity/source and actual invalid-HANDLE retirement
assertions all passed. All123 observed calls paired, with zero caps, unmatched
returns or unfinished starts, zero global monitoring events, and confirmed local
monitor retirement. CPU stamps quantized to15.625ms despite GetThreadTimes'
advertised100ns resolution; per-primitive zero/over-wall CPU readings are not
usable attribution. No disk/GIL/lock cause is inferred.

Receipt: native-metadata-detail-1/{timing.json,run.json,pytest.xml,observer.py,
observer.patch} in the external review directory. Installed observer byte SHA256
c3f84fb9651467e2f94b5d71336b262c81ddfd35aa1119dfbd5ae99502ff88f1;
the manifest initially confused LF-normalized text with CRLF bytes, and the
install guard refused before editing or running. Metadata was corrected before
the only run. Exact original helper ca3bf8074f6be4a18e545fcdd1bf030cd24220d271d2ca7c54bb0c2d01c5025c
was restored afterward. Product hashes/HEAD remained stable; the runner checked
for competing native processes before execution (no during-run overlap guard).


## Finite MCP admission contract refinement (2026-10-09)

OPT76 is reconsidered as an explicit boundary change under ADR126/222, not as
the previously rejected equivalent one-line deletion. Root and independent
source reviews found metadata/resource preparation before the late proof has no
payload effect. Initial source selection plus one late source-and-parent proof
before activation can replace four full admission observations. Every effect and
Console final checked-read proof remains fresh. Resident identity, selection,
closed/pause and exact lease custody still gate setup and activation.

The deliberate tradeoff is later recovery refusal: external pending drift may
acquire the source mutex and construct inactive State before refusal, including
waiting behind that mutex. Cancellation/pause remain checked while waiting. The
prior test's zero-lock/zero-State assertion changes; no body/payload/default/
backup effect and exact physical retirement remain mandatory. Supported custom
callbacks and custom locks keep their original route. No reusable witness or new
authority token. The small source plan is
Docs/superpowers/plans/2026-10-09-finite-mcp-source-admission.md, TASK34601AC20.
This is a candidate only; native regressions, integrated checks and sequential
quiet whole-Send evidence determine adoption. No gain is claimed yet.


### OPT76 measured disposition

The finite admission refinement qualified86 distinct targeted cases and reduced
empty permission-scope witnesses4→2/native opens349→208. It added118 net product
lines and qualified only the permission store. Quiet sequential ABBA warm means
4.534→5.691s establish no whole-Send gain; cold6.715→5.207s and all12 saved/settled
turns remain in the evidence. Both arms vary substantially, so no causal slowdown
is claimed. Source/HEAD/loaded membership and overlap checks pass. The candidate
and exact tests are archived and original product/tests restored; no runtime
contract change ships. One old early-State assertion was legitimately migrated;
pending-3 is excluded for source drift and final pending-4 qualifies the corrected
boundary. Full raw samples, control details, lint limits and archive paths:
Docs/superpowers/plans/2026-10-09-finite-mcp-source-admission.md. ADR126 records the
rejected alternative. The next architectural investigation consolidates initial
tool selection with existing composition, subject to an explicit ADR225 refinement;
it must not repeat the rejected initial-only catalog extraction or add a cache by
default. All97 optimization IDs remain available for later review.


## Composition-owned ceiling outcome (2026-10-09)

OPT98 is retained: initial original native permission/catalog bodies fall1/1 to0/0,
with1/1 fresh observation at demanded composition. Root integrated all three lanes;
259 distinct targeted controls qualify the boundary, saved draft policy and native
retirement. Quiet sequential ABBA warm mean3.031 to2.718s (10.34% lower), cold4.885
to3.945s; all12 turns saved/settled, source stable, no detected overlap. The first
a1 baseline is excluded for an external evidence-reader process and replaced by
a3 before candidate runs. All raw samples and limits are in the composition-tool-
ceiling plan. Candidate warm saved-to-trace remains1.133-1.562s; no one-second or
physical-feedback acceptance claim. Source review found no additional removable
post-save whole-domain duplicate; history, final hook admission and trace effects
retain their ordered contracts. OPT99 preserves the unimplemented metadata owner.
All99 optimization IDs remain available; task stays In Progress.


### Committed OPT98 feedback and post-save costs

At24d3d773fd, two current held-reader feedback cases pass: Preparing34.671/41.205ms
and input frames26.718/9.178ms (headless supplied frames only). The targeted total
is261. The separate current original-function diagnostic completes all three
saved/settled turns, source stable/no detected overlap. Warm post-save gap2.140/
1.828s is instrumented; inclusive composition .537/.455s, final hook admission
.229/.274s, history .111/.221s, initial maintenance .174/.249s, checkpoint .151/
.136s and run-log bind .164/.064s are not additive savings. No further duplicate
whole stage is established. Seven ordinary starts and92 hook/raw ancestry misses
remain explicit; monitoring retires and recorded context/generator maps are
complete. The composition plan links exact receipts. Existing OPT96/99 require
new lifecycle/invalidation decisions; neither is silently implemented here.

### OPT30 deferred follow-up: first feedback and input paint after rebase

Historical post-rebase acknowledgement samples painted Sending and a pending
USER row before received preparation. `post-rebase-smoke-3` (1eddc7a4a8) and
`post-rebase-ui-final-1` (1867ef2fb9) retain different outcomes with unchanged
production bytes; the latter has 37 functional passes and 2 timing failures.

| Natural supplied-frame/input milliseconds, Enter/button | Smoke3 | UI final1 |
| --- | ---: | ---: |
| Sending feedback | 87.21 / 92.11 | 112.28 / 67.86 |
| Exact pending USER cells | 86.88 / 196.47 | 111.91 / 67.60 |
| Typing mutation | 2.55 / 3.16 | 96.06 / 226.03 |
| Typing supplied frame | 11.78 / 8.52 | 112.74 / 263.75 |

The 100 ms Sending/input gates remain unchanged and consistent qualification is
open. The pending USER endpoint is separate from receipt/Preparing and the
older stored-USER measurements; there is no button-USER <=100 ms claim. Review is
deferred while latency reduction is paused. No cause, new cache or scheduler
is selected from these samples. Raw XML/log/run receipts remain under the task's
`claude-watch-final-gate-review` directory. These headless/stub-provider samples
do not establish physical terminal flush, real provider TTFT or current whole-
Send latency; the older OPT98 2.718 s ABBA result remains historical.

The custom-fallback newer-draft REDs led to an original pressed-snapshot
ownership correction. Before the final rebase, two visible/hidden draft controls,
ten physical saved-acceptance controls and three custom-worker controls passed.
The first correction exposed duplicate repeated Sends and lost switch-window
typing; both REDs are retained. Its bounded follow-on passed all eleven focused
repeat, switch, newer-draft and callback-boundary controls in 96.71 seconds.
One Textual Worker ContextVar warning remains recorded. This does not close
consistent 100 ms feedback or final post-rebase qualification. See the phase
report for Windows Close and early legacy-approval limits; no further latency
work is selected.
