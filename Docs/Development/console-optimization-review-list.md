# Console optimization review list

Maintained for the user's request to preserve every considered optimization that
has not been used or implemented, for review after the main speed and stability
work. Scope: the current Console Send investigation and its integration lane.
Last reconciled: 2026-10-07, candidate checkpoint `98cb24aad5` and task26 evidence.

Each ID stays stable. Record new ideas here as they arise. When an item is
implemented, move it to the resolved section with its commit/task and measured
outcome; do not silently delete it. A candidate is not a promised saving or a
scheduled implementation. Historical timings below identify the measured source;
re-measure after integration before treating them as current costs.

Sources: [phase-attribution report](2026-10-06-console-send-phase-attribution.md),
[approved architecture](../superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md),
[ADR-222](../../backlog/decisions/222-console-send-preparation-and-io-ownership.md),
and [qualification task](../../backlog/tasks/task-34563.4%20-%20Qualify-shared-preparation-Send-latency.md).

## Candidates preserved for review

| ID / status | Opportunity and evidence | Why held; condition for revisiting |
| --- | --- | --- |
| OPT-01 / Awaiting attribution | **Prepare local catalog configuration once at composition.** `_default_specs` separately reads three exposure gates; enabled Ask User adds a prompt-config lookup. Enabled deep search reads nine settings to use one displayed timeout. Root-specific builders repeat these reads. The d524 detail sample leaves .733/.742/.748s of composition outside its permission call, but that remainder is not yet assigned to these lookups. | The new optional original-body spans must identify the local-builder cost. Capture fresh raw merged config at the current composition owner, project small immutable inputs once, and preserve conditional/custom getter calls, prompt precedence, and fresh invocation-time policy/timeout reads. Older turn values would change construction-time freshness. |
| OPT-02 / Unmeasured | **Consolidate local root construction.** `WorkspaceToolExecutor` validates/resolves its root and captures each ancestor; `LocalToolProvider` separately resolves its redaction root. Multiple roots can repeat spec/executor construction. | Separate this from OPT-01's config cost. Reuse only an existing equivalent root proof within its valid ownership interval; preserve per-binding exclusions, remote-root separation, root replacement and invocation checks. Measure constructor cost before adding a new proof or cache. |
| OPT-03 / Deferred | **Separate hook path naming from default-directory establishment.** `HookPermissions._current` already owns a config snapshot, then `default_hook_permissions_path` derives the user directory again. Earlier original spans put that lookup at roughly 7–19% of hook-context time (.163–.591s across five contexts per Send). | Those samples predate later raw/private-path changes. Cold/default selection also owns serialized fallback establishment and configured-base validation. Revisit using current timing and an equivalent finite owner; retain custom getter/snapshot callbacks and refusal behavior. |
| OPT-04 / Deferred | **Reduce repeated hook preparation within equivalent boundaries.** Five hook-current contexts were observed per Send, associated with receipt, admission, initialization, capture policy and postcommit dispatch. Historical enclosing costs were seconds; JSON state reading was much smaller. | These contexts straddle real policy/effect boundaries. Later storage improvements already affect their cost. Identify a genuinely same-operation duplicate; keep consent reconciliation, resumed review, revocation and final launch serialization fresh. Empty hook inventory alone does not justify skipping admission. |
| OPT-05 / Awaiting attribution | **Streamline remaining agent startup and serial handoffs.** The d524 detail run has six guard entries totaling .508/.460/.665s to their first yield; reply-worker launch to trace reservation takes 1.838/.899/1.062s. Run-log binding is nested within that interval. | Finite metadata/run-log preparation is already implemented. Remaining guards cross real workers/awaits and include separate stores. Partition current owner setup, scheduling and required admission before selecting a contraction; avoid adding nested times or keeping a lease across an await. |
| OPT-06 / Deferred contract change | **Move auxiliary prompt-history persistence out of the pre-dispatch path.** On d524, awaited history took .262/.326/.470s. The approved design identifies it as auxiliary to conversation recovery. | Current ordering remains awaited. Revisit only with a concrete existing-owner queue/drain/failure contract covering Stop, shutdown, read-after-write and history ordering. A detached background write would trade latency for a lifetime bug. ADR-222 requires a separate explicit contract before changing this ordering. |
| OPT-07 / Deferred, low measured return | **Return checked raw-discovery state to immediate consumers.** [Task34563.19](../../backlog/tasks/task-34563.19%20-%20Return-checked-raw-discovery-state-to-immediate-consumers.md) proposes removing an immediate duplicate check after successful stock operation discovery. The task18 sample recorded 102 eligible checks costing only .118839s across all three Sends. | No product implementation or native RED was performed; its prepared test is outside active collection. Revisit if a current profile makes this material or the API simplifies another necessary change. Preserve custom discovery callbacks, actual file-effect checks and physical retirement. |
| OPT-08 / Unselected, lock-order risk | **Select/prepare an installed MCP source under its existing mutex.** Current task26 setup still observes the source before waiting on that mutex. Moving selection could avoid roughly one 38-open witness per read. | Small ceiling compared with the wider delay; it changes canonical-admission/source-mutex order. Requires a cancellation, recovery-maintenance and deadlock argument before implementation. Do not remove the four retained source/effect/publication checks or turn old observations into cached authority. |
| OPT-09 / Unmeasured | **Avoid repeated empty lock-file precreation work when an existing file is sufficient.** The original private-text observer sees creation attempts whose ordinary FileExistsError exits are not paired, so its completed-span list understates this route. | First obtain original normal/exceptional-exit cost. Preserve secure first creation, existing-file identity, lock acquisition and required durability. The current evidence does not show that precreation is free or that an existence check would be cheaper. |
| OPT-10 / Deferred, attribution incomplete | **Coalesce redundant context-display refreshes.** The task22 integrated sample has 16 presentation version-reader calls before adapter entry; worker elapsed totals .303/.542/.825s by Send. The current display owner already serializes warming with owner/revision checks and a one-second TTL. | Counts do not prove identical message IDs/versions, explain invalidation, or establish that worker time delays Send. Revisit only after identifying redundant invalidations or actual contention. Keep live acceptance/dispatch reads separate and reject stale display publication. |
| OPT-11 / Not selected, low return and freshness boundary | **Merge live preaccept/dispatch message-version reads.** Five actual live reads in the task22 integrated sample total only 2.662ms across three Sends. | Reads straddle the commit and cover changing requested IDs. No unconditional duplicate was found. Revisit only if a future profile demonstrates expensive same-state duplicates; retain changed-request and postcommit freshness. |
| OPT-17 / Deferred, equivalence unproven | **Batch more Windows metadata work within a full raw check.** Remaining retained-identity and parent-pin observations reopen overlapping ancestor chains; `stat_many_for_admission` is an existing possible primitive. | This goes beyond the implemented parent-walk/control batching. Revisit only with current per-check evidence and equivalent alias, DACL/posture, handle-association and uncertain-close behavior. A wider batch can add work or broaden ownership. |
| OPT-18 / Deferred, low earlier priority | **Combine small agent lifecycle DB writes.** Earlier attribution measured run creation and context lifecycle rows separately; their bodies were much smaller than admission/run-log preparation. | These rows record distinct lifecycle facts. Revisit if current transaction/admission overhead is material and the original event order, failure visibility and recovery semantics can be retained. Existing per-entry preparation has already been improved. |
| OPT-19 / Deferred contract change | **Defer other auxiliary audits/projections after dispatch.** They were considered as possible reductions to the serial postcommit path. | Classify each actual effect first: workspace binding may depend on projection, and audit policy belongs to its owner. No generic postcommit callback is presumed optional. Requires an explicit completion, failure, recovery and shutdown contract, as with history. |
| OPT-20 / Not selected, tiny warm costs | **Further optimize warm run-budget, personal-context service and profile helpers.** Earlier measured maxima were .213ms for budget, .008ms for warm service, .033ms for profile-tool composition and .044ms for profile snapshot; one cold personal-context construction took .140s. | Warm bodies cannot explain the multi-second delay. Revisit only changed-source or cold-path evidence, keeping inclusive timings distinct from attainable savings. See the early agent-entry attribution in the phase report. |
| OPT-21 / Deferred, startup scope | **Defer or streamline cold Notes initialization before first input.** Integration lane's original d524 timing: Windows Notes 4.643s/join 4.641s; Linux 1.406s/1.403s; macOS 1.083s/1.081s. Media/prompts were smaller. | These are overlapping inclusive spans, not additive savings. No constructor deferral/reordering is implemented. Preserve mandatory schema/seeding, admission, safe first use and shutdown ownership. Evidence: integration lane's `ci-d524-startup` original timing receipts. |
| OPT-22 / Not adopted for live authority | **Use display caches in main-loop `sync_live_state`.** The integration lane measured .133–.296s in this path, within a full native Send census of 72,874/64,429/59,370 opens on d524 (first Send 13,664 main-thread and 59,210 worker opens). | Core/roleplay reads here intentionally feed live store mutations/projections. Disposable display cache substitution was rejected because it would change authority/freshness. Revisit narrower same-operation consolidation with current whole-Send attribution; task26's isolated reduction does not qualify that whole budget. |
| OPT-23 / Deferred, ownership equivalence unproven | **Fuse the control-read initial traversal with the native snapshot forward pass.** Task24 batches control reads but still leaves initial traversal plus independent forward/reverse completion passes for each fresh witness. An earlier six-ancestor layout suggests an approximate ceiling of 54 opens at nine witnesses, or 48 at eight; neither is a measured saving. | Initial pins and snapshot handles have different lifetimes and uncertain-close handling. Partition original passes first, then prove an equivalent finite handoff preserving named associations, stamps/posture, registry locking and retirement. Sources: `bootstrap._control_observation`, `WindowsOS.stat_many_for_admission`, task34563.24 and its control-parent/lifetime tests. |
| OPT-24 / Not adopted under current proof | **Use an already retained raw parent FD instead of a full private parent walk.** This could avoid traversing the same ancestors again. | Raw parent pins prove the current parent identity, not the private walk's full fresh ancestor ownership/mode/no-follow policy. A duplicated FD is insufficient. Revisit only when an existing domain supplies equivalent current ancestor and named-association evidence. Sources: task34563.22; `_check_parent_pins`, `_walk_verified_parent` and `_prepared_parent_walk`. |
| OPT-25 / Unmeasured | **Use identity-only native metadata for raw parent association checks.** Some non-companion checks consume only device/inode/type, while Windows stat/fstat also calculates owner, ACL and timestamps. Potential metadata/CPU reduction; open savings are unproven. | First measure `_Native.security` and `_stat_handle` separately. Preserve full companion posture, named ancestry/reparse checks and uncertain-close behavior. No compatible reduced-metadata primitive has been selected. Sources: raw `_check_parent_pins`, `WindowsOS._named_stat`/`_stat_handle`, `_Native.info`. |
| OPT-26 / Deferred, optional route unmeasured | **Share project-binding list and automatic selection projection.** With enabled project instructions and no explicit binding ID, capture lists/validates eligible bindings and resolution lists/validates them again; local validation repeats ancestor lstat. | Current Send samples do not establish this route's cost. Revisit with enabled named-workspace counts, then derive sole-local selection from one finite capture while retaining remote first-selection checks, current root enforcement and custom routes. Sources: tasks34563.9/.14 and controller `capture_project_instruction_authority`/`resolve_project_instruction_binding`/`list_project_instruction_bindings`. |
| OPT-27 / Deferred, optional route unmeasured | **Combine imported-skill trust status and fingerprint projection within one scan.** For trusted imported rows, `capture_skill_context_maximum` asks for summary trust status and then current fingerprint digest; both can call `_scan_skill`. Builtins and empty catalogs do not pay this cost. | Measure an actual imported-skill route first. A named finite single-scan result could share work; invocation trust/revocation and custom owner behavior must stay current. Sources: task34563.14, `console_configuration_capture.capture_skill_context_maximum`, local skill summary/trust fields and `skill_trust_service._scan_skill`. |

## Architecture alternatives retained with their decision

These were considered and deliberately not adopted under the current design.
They remain visible for later review rather than becoming implicit future work.

| ID / status | Alternative | Decision and revisit condition |
| --- | --- | --- |
| OPT-12 / Not adopted | A profile-wide storage actor, worker, shared cache or execution lock. | ADR-222 favors finite domain owners; a global mechanism adds invalidation/shutdown rules and can serialize unrelated chats. Revisit only with measured cross-operation reuse/contention and an explicit lifecycle design. |
| OPT-13 / Not adopted | Keep a native snapshot/lease across preparation awaits, approval waits or later dispatch; use one permission observation for the whole Send. | Initial narrowing maximum, live composition and actual invocation have different freshness/authority roles. Native leases stay within finite operations. Any future redesign must explicitly replace those contracts rather than treating prepared data as permission. |
| OPT-14 / Not adopted | Add a generic pipeline engine, dependency bag, scheduler, revision registry or duplicate preparation-state ledger. | Existing runtime/controller/store owners and named domain APIs already provide those responsibilities. Revisit only if concrete extension requirements justify the added owner/abstraction and simpler alternatives fail. |
| OPT-15 / Not adopted | Dispatch a saved chat before required acceptance/checkpoint/consent/trace/context facts, weaken persistence, or silently switch to unsaved work. | The user chose save failure to stop Send and preserve the draft; existing temporary chats remain. No schema/PRAGMA/durability-mode change is selected. Current measurements put much of the delay in repeated preparation, not the minimum commit. Any change would require revisiting that explicit product decision. |
| OPT-16 / Not adopted as the overall solution | Continue only with per-helper shortcuts or move unchanged repeated work into workers. | The approved design instead shares preparation within existing domain boundaries. A small change can still be retained when independently useful, but worker placement alone does not reduce the repeated I/O or total Send delay. |


## Related finding to retain separately

**FOLLOWUP-01 — startup readiness completion accounting.** The integration lane
reports that the first attach can be marked complete while native full sync is
still deferred by replay/maintenance. This remains an unimplemented correctness
and perceived-readiness issue, with no demonstrated cost reduction. Track the
actual full-sync completion before presenting startup as ready. Keep it separate
from both catalog optimizations and the in-progress initial-draft/switch repair.


## Ruled-out premise

The proposed removal of five supposedly eager database opens during configuration
capture was ruled out by source review. `operation_owned_connection` owns cleanup;
it does not eagerly open each database. The explicit Workspace connection is the
sole forced scope and its nonempty-workspace lookup consumes it. Removing that
scope could increase admissions. Preserve this conclusion from the phase report's
"Domain priority after task18" rather than reopening the same premise as an
unmeasured optimization.

## Resolved entries

None moved yet. Implemented work that predates this list (shared tool preparation,
run-log and activation preparation, retained saves, early receipt, configuration
workers, parent/control batching and the task26 owned permission load) remains in
its existing tasks and evidence report; it is not a pending optimization here.

## Review discipline

- Add evidence, source revision and the reason for deferral when preserving an idea.
- Keep operation-count reductions separate from measured Send/input improvement.
- Update status as soon as an item is selected, ruled out, implemented or superseded.
- Preserve the user's speed/stability goals and required durability/permission rules.
- Use targeted checks and sequential native timing. The list does not authorize a full sweep.
