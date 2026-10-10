# Incremental Send speed improvements

Date: 2026-10-05
Status: Revised written design approved on 2026-10-06. Implementation plan review remains pending. User clarified that long-term stability and speed, with immediate terminal feedback, govern completion.

## Intent and scope

Reduce application work before provider dispatch while preserving the existing enabled tool set, hook consent, capture, persistence, cancellation, recovery, and resource ownership. Keep the main thread's approval display work isolated. The initial implementation consists of small changes to existing shared functions, with a separate measurement after each change. Broader dispatch restructuring is reserved for a later design.

The terminal must visibly acknowledge Send within 100 ms across normal, enabled-feature, approval-wait, and refused/error paths, and remain responsive to input while preparation runs. This received/preparing feedback must not claim durable acceptance before commit. Ordinary chat targets less than one second of application overhead before provider-adapter entry. Long-term stability and speed are the objective: preserve durable storage, fresh consent, ownership, cancellation, and recovery while removing repeated work and main-loop stalls. These are acceptance targets, not a claim that this first pass will achieve them. Provider/network response time is separate. Cold initialization and enabled feature paths must be measured separately; a faster empty-feature path cannot establish performance for enabled tools.

## Evidence

The source-current isolated three-send diagnostic retained 7,676 Python source hashes, original guards, original test assertions, and normal native process retirement. Three traces completed with three reply links and zero dispatch checkpoints. All selected postcommit observations qualified at each durable success boundary; no global monitoring, overflow, unknown exits, or source drift was reported.

Measured diagnostic ranges:

- Durable turn commit: 0.527–0.812 s.
- Commit completion to trace reservation: 20.552–21.243 s.
- MCP composition: 2.873–4.420 s.
- Local-tool composition: 1.273–1.718 s.
- Final hook admission: 1.548–2.430 s, with no hooks section in the selected config.
- Two agent worker handoffs within the critical interval: 9.291–11.937 s, including agent preparation.
- Prompt-history append: 1.387–2.696 s.

The new diagnostic includes extra observation overhead and a separate filesystem location. Nested intervals overlap; neither sums nor promised savings are derived from them. The earlier exact-bc740 Windows capture reached the provider in 20.854–22.062 s. A Linux diagnostic on the same actual sources reached it in 3.095–5.023 s, establishing a cross-platform issue with greater Windows cost.

Evidence: [timing summary](C:/Users/GDesktop-1/.codex/visualizations/2026/10/06/01a10f94-4fb9-7b22-9267-66bf71531824/send-investigation/timing-summary.json) and isolated-current-send-2.{critical,probe,source,custody}.json in the same retained evidence directory. The rejected first probe is excluded.

## Architectural governance

ADR required: no for the initial stock MCP early return; yes before introducing an admission-only hook interface or a new snapshot-sharing/worker callback lifetime.
ADR path: existing backlog/decisions/126-complete-local-backup-and-recovery.md, 197-console-hook-configuration-review.md, and 134-fleet-admission-and-automatic-work-budgets.md. Amend ADR-197 before a later hook interface is implemented, and the applicable existing ADR before any new lifetime.
Reason: the initial MCP change preserves existing signatures and qualified ownership. Hook reconciliation has durable consent effects and cannot be removed merely because current configuration requests no authority. Source/current-owner boundaries must remain intact.

## First pass

### 1. Hook admission and required reconciliation

Current `_current()` work is more than an admission verdict: `_reconcile()` rotates grant tokens when an approved hook becomes disabled, deletes grants when definitions disappear, and publishes current targets/review state. Returning solely because a fresh lossless inventory has `requires_authority == False` can skip those durable effects. If the same hook is reenabled or readded before another original reconciliation, a queued target or previous grant may survive when the existing Send would have retired it. Final launch checks alone cannot reconstruct a disable/remove transition that was never recorded.

Retain the current admission/store path in the first executable slice. Count the original config/store reads and reconciliation writes, including absent/empty configuration and enabled-to-disabled-to-reenabled or remove-to-readd transitions. A later shortcut must establish both no requested authority and preservation of the current reconciliation/publication effects within an existing qualified lifetime. A genuinely absent store is a separate candidate requiring current, source-bound absence proof; the no-hooks diagnostic may already have an existing store, so absence cannot be assumed. Do not introduce a consent cache or mark state reconciled without doing the required work.

A future admission-only query must preserve Settings/review rows and truthful store state, enabled/pending/invalid/malformed/custom paths, next-Send review, cross-process revocation, and final launch-time definition/consent checks. It must avoid an additional config read on enabled paths, preserve original lock order, and never manufacture a review snapshot with a fictitious store revision. Amend ADR-197 before changing that shared interface.

Likely seams: Agents/hook_permissions.py, Chat/console_chat_controller.py, and all callers of the admission query. Reuse the lossless hook config parser. Tests must cover both v1 and v2 definitions, legacy and named identities, and queued notification/launch epochs. The existing `test_disable_reenable_retains_consent_but_retires_queued_epoch` is a required original control.

### 2. Empty MCP maximum

An explicitly empty frozen maximum cannot advertise any MCP tools. At the existing qualified stock composition seam, publish the same empty inspector result and return without catalog construction or native catalog reads. Distinguish empty from an unset maximum. Preserve nonempty filtering, disconnect counts, kill switches, source/owner changes, cancellation, and customized factory behavior.

Likely seams: Chat/console_chat_controller.py and Agents/mcp_tool_provider.py. Place the early return where all relevant stock consumers share it; do not duplicate guards across unrelated callers. Preserve the current live empty inspector outcome `(None, None)`; preview composition with `publish_counts=False` must leave live inspector state untouched. Initially retain the ordinary plugin path whenever `plugin_maximum` is supplied, including an explicitly empty mapping; skip only a qualified stock route with `plugin_maximum is None`. A no-tool bound does not establish plugin selection/owner validity.

### 3. Local-tool and enabled catalog preparation

Reuse the existing static tool definitions and one completed preparation result within the same owned composition invocation. Find and remove duplicate definition building and repeated lookups before adding any cache. Keep workspace/scratch selection, exclusions, access modes, persona floors, and current permission checks. `_default_specs` includes dynamic gate reads, environment/settings-derived descriptions, service construction, and root-bound handler closures. Reuse only data proven static or the exact invocation-owned result; never share root-bound handlers or mutable schemas across roots, sessions, or later sends. Check catalog IDs, order, descriptions, schemas, and dispatch routing for parity. Its repeated multi-root construction is a candidate to count, not a proven ordinary single-root bottleneck.

Share data only where an existing source/generation/current-owner contract already covers its lifetime. Native admission proofs cannot cross an await or worker boundary. If the existing result does not cover the second consumer, retain the current read and record that work for a later design; this pass does not invent a new cache manager or permission memo.

### 4. Existing worker callbacks

Optimize repeated reads inside current finite callbacks using the existing counted repository and connection-ownership mechanisms. Measure the actual callback and actor before grouping any operations. Capture the exact database and preserve foreign/borrowed transaction behavior, physical retirement, cancellation drain, and source/owner rejection.

Keep chain creation, accounting, admission, and callback ordering unchanged in this pass. Moving chain creation into a different worker or splitting the dispatch lifecycle is reserved for option 2 because it can alter failure/recovery behavior.

### 5. Prompt history

Inspect the existing append path for repeated path resolution, file loading, and participant construction within one append. `_load_locked` already avoids successful repeat loads; do not add another load cache without evidence. Measure cold load, warm append, duplicate suppression, and cap-triggered rewrite independently. Reuse current helpers and eliminate duplicated work only within their existing lifetime. Preserve append ordering, deduplication, bounds, durable outcomes, and error behavior.

Keep history as an awaited postcommit effect in this first pass. Moving it concurrently with provider dispatch changes failure and shutdown semantics and is reserved for option 2. This is a deliberate scope refinement of the earlier brainstorming proposal, consistent with preserving current behavior.

### 6. UI and startup

Use the measured synchronous configuration fallback as a targeted follow-up after the main thread's approval display change is integrated. Keep original expiry/current-owner checks and direct/custom routes. Reuse existing workers, published display state, and coalesced refreshes. Read-only display must not cause unrelated database housekeeping.

Do not change the main thread's screen/widget files while it works on them. Avoid duplicate implementation of its pending fixes. Startup import/constructor work and mounted UI stalls require separate attribution; startup stage maxima cannot be called mounted pauses.

## Verification and delivery

For each change:

1. Reproduce the exact original work cost with real private config or SQLite and the relevant original callback/guard.
2. Verify reduced original operation counts, not only elapsed time.
3. Keep enabled/custom/source-change/owner-change/revocation/cancellation controls and original physical retirement checks.
4. Run the directly affected existing regression tests and scoped lint/format checks. No full suite without explicit approval.
5. Compare unchanged-source cold/warm three-send measurements separately from diagnostic observation overhead. Confirm all three trace completions and reply links.
6. Measure from the real Enter or mouse Send action to both actual provider-adapter entry and the first rendered acknowledgment that the app received the action. A status assignment or task submission is not rendered acknowledgment, and preparation acknowledgment must not claim durable acceptance. Use one monotonic clock for each process; report Send-to-ack, Send-to-provider, and event-loop stalls separately. Retain the sub-second dispatch target; the legacy 15-second regression assertion is not the desired user experience.
7. Verify on Windows, Linux, and macOS before claiming cross-platform performance acceptance. For each host, run a private unchanged baseline and candidate with matching interpreter, dependency build, instrumentation, configuration shape, and filesystem placement; alternate repeated runs and report the individual samples. Preserve cold first Send and warm later Sends separately. Report feature-enabled paths separately and record missing host evidence explicitly. Prefer operation-count improvements when wall times overlap.
8. Exercise no-hooks-to-enabled, approved-to-disabled, disable-to-reenable, consent revocation, nonempty-to-empty MCP composition, and source/owner replacement between sends. Preserve original cancellation and native physical-retirement controls. Permission-store corruption may remain unobserved by an admission-only no-authority query, while Settings must still report it truthfully.

Implement one change at a time in an isolated branch based on the integrated current development/fix state. Record any source drift and repeat affected measurements. No production change or new test assertion has been applied by this design step.

## Self-review

- No placeholders or estimated additive savings.
- No new plain-chat mode or reduction in enabled capabilities.
- No global cache, scheduler, dependency, schema migration, longer TTL, weaker storage checks, or raised performance ceiling.
- Authority remains fresh; shared snapshots contain data rather than reusable permission verdicts.
- Worker and history ordering changes are explicitly reserved for option 2.
- The design is saved in this side conversation's artifact directory. It will be incorporated into the implementation branch without modifying the main thread's current worktree.
- The reviewed copy lives in this task's side artifact directory. Original evidence remains in the preceding side conversation's send-investigation directory; its absolute timing-summary link was verified.

## Review findings incorporated — 2026-10-06

The six directly relevant files in the active main worktree (controller, hook owner, MCP provider, local provider, prompt history, and config) matched the retained source hashes during this review. Neither active checkout was edited. Recheck the integrated execution base because later changes invalidate that comparison.

The first executable slice is the empty stock MCP maximum, followed by a matched measurement and hook reconciliation attribution. The unconditional no-active-hooks storage shortcut is withdrawn because the current store visit has durable consent effects. The local/catalog, worker, history, and UI/startup sections remain evidence-led opportunities in this incremental effort: do not force an edit where no duplicate work within a qualified existing lifetime is established. Keep the wider Send-performance objective open if the slice improves counts but still misses 100 ms acknowledgment or one-second dispatch.

The diagnostic evidence does not establish that this initial change alone can achieve the target. Existing callback and history costs remain measured opportunities; their intervals cannot be added to predict a final latency. Broader worker/chain or history scheduling remains option 2 and requires its later design.

Review severity: the hook reconciliation issue is a correctness blocker for the proposed unconditional hook shortcut, not a blocker for incremental Send optimization. Preserve the capability/consent constraint and lead with the concrete MCP change. This is a refinement of option 1; option 2 remains reserved.
