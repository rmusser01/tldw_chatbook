# Console file-review and workspace-tool performance recovery

Date: 2026-08-21
Status: approved by owner after independent review
Target: `dev` refreshed to `549a26b7a5828e5ddeed1051dbff5f4fa634a003`

## 1. Purpose

Restore reliable multi-turn Console chat after Agent Change Review and the
expanded local-tool catalog introduced latency and lifecycle gaps. The repair
must make ordinary chat independent of filesystem snapshot work, make review
an explicit workspace choice, keep queued text honest and private, and ensure
local file tools operate on the workspace the user selected.

This is an incident-recovery umbrella design. Delivery is split into atomic,
independently testable Backlog tasks; it is not one cross-cutting rewrite.

## 2. Evidence and confirmed failure modes

The original investigation was performed in a clean worktree at
`ab8eb1e7`, not against the user's dirty primary worktree. Before planning,
the branch was fetched, rebased, and revalidated at `690435d0a`. The intervening
74 commits did not change the synchronous Change Review tracker or its bridge
call sites, but they did add ADR-069 project-instruction authority and tool
preflight. During final review, `dev` advanced again to `549a26b7a`; that delta
contains workspace-create tests, docstrings, and small Settings/create-handler
follow-ups, not Change Review, agent runtime, tracker, or tool-provider changes.
Sections 4 and 5 incorporate the current authority and Settings boundaries.

### 2.1 Change Review holds a completed response's turn open

`ConsoleAgentBridge.run_reply` starts a baseline snapshot in the background,
but every terminal path synchronously calls `ChangeTurnTracker.end_turn`.
`end_turn` first joins the baseline thread for up to 120 seconds and then takes
the end snapshot. The prompt coordinator does not clear
`accepted_live_turn` or drain the next prompt until the bridge returns.

The assistant response can therefore already be durable and visible while the
turn still owns the session slot. A third prompt entered in that interval is
accepted only into the process-memory prompt queue. This is the observed
"file checker" stall.

Production-shaped measurements on a 9,263-file, 258.7 MB root were:

- initial shadow snapshot: 11.93 seconds;
- warm baseline: 292 ms median;
- warm end snapshot: 218 ms median;
- clean tracked turn: approximately 510 ms after model completion;
- three filesystem walks and roughly 22 Git subprocesses per clean turn.

An isolated shadow-repo probe was repeated after the refresh on the current
9,301-tracked-file source tree: initial B was 6.62 seconds, warm B median was
309 ms, warm E median was 230 ms, and a clean turn still paid about 539 ms
after model completion. The exact cold value varies with disk cache; the
synchronous warm tax and completion coupling reproduce unchanged.

Change Review is currently default-on when configuration is absent, when a
workspace has no toggle row, and even when the toggle read fails. Shadow Git
stores file contents under application data for 30 days.

### 2.2 The default catalog makes common file tasks take extra model turns

The refreshed default catalog still contains 19 tools.
`DIRECT_DISCLOSE_THRESHOLD` is 16, so
the runtime discloses no real tool schema initially and requires
`find_tools` followed by `load_tools`. Catalog search is one contiguous
substring comparison: natural phrases such as `list files` and `read files`
return no results.

A recorded file-listing request needed five model calls after one failed
search. Disclosing all 19 schemas would remove the round trips but add roughly
11,000 serialized characters (about 2.7k tokens; exact size depends on wire
serialization) to every provider request.

### 2.3 Local file authority is correct only for instruction-enabled sessions

The built-in file-tool family resolves the run's workspace bindings through
`allowed_file_roots`. ADR-069 now correctly composes the newer local `fs_*`
and `git_*` family at one explicitly selected binding when project instructions
are enabled. It pins binding ID, canonical-locator fingerprint, access mode,
and filesystem identity and rechecks that authority before invocation.

Disabled and legacy-disabled sessions still close the local provider over
`[console] workspace_root`, falling back to the live process working directory.
A named workspace can therefore visibly bind one project while those sessions'
local tools operate on another. The repair must preserve ADR-069's enabled
single-root contract while removing only this legacy fallback.

The access-root source must not be `folder_binding_roots`: that function is a
Change Review gate and returns no roots when review is disabled. File-tool
authorization and review policy are independent.

### 2.4 Console mount time has a measured framework-dominated floor

The known warm Console switch cost is about 1.45 seconds after the only
material application-side composition hotspot was memoized. Profiles attribute
the remaining cost to mounting, CSS, and compositor work over roughly 600
widgets. Reusing a cached `Screen` is prohibited by reproduced teardown/remount
freeze failures. Any further work must be a measured subtree reduction or
deferral, not screen-instance caching.

### 2.5 Host saturation amplified the incident but is not yet a dev regression

The host had a 14-hour pytest process at approximately 99% CPU, more than 5 GB
RSS, and over 1,000 threads, plus an old application process from a stale
worktree. The original pytest process exited before its active test and stacks
could be captured. Latest `dev` already has per-test pytest timeouts and CI job
timeouts. No AgentService thread leak has yet been reproduced; the refreshed
code review found no new ownership evidence that would justify changing its
lifecycle speculatively.

## 3. Product decisions

1. Change Review is opt-in per workspace.
2. Missing or unreadable workspace review state means disabled.
3. Review work never owns the chat-turn completion boundary.
4. An enabled review conservatively gates potentially dispatchable calls so B
   precedes any mutation; the wait is bounded and cannot become a stuck turn.
5. Review failure is advisory: the tool proceeds and the transcript visibly
   states that the turn was not tracked.
6. The prompt queue remains bounded process-memory state. This design does not
   persist unsent private prompt text or replay it after restart.
7. Local path tools use run-bound workspace authority, never Change Review
   state and never an implicit process-CWD fallback. Project-instruction-
   enabled sessions retain ADR-069's one selected root; disabled named
   workspaces may use their admitted bindings. Default and binding-less named
   workspaces omit only local filesystem/Git specs; unrelated local web,
   Watchlists, and todo tools remain available, and built-in file tools retain
   sandbox access.
8. Large catalogs use a small directly disclosed core plus progressive
   discovery; all schemas are not injected into every request.
9. Console mount changes ship only when profiling demonstrates a meaningful
   first-interactive improvement without lifecycle regressions.
10. Unreproduced thread leaks receive diagnostics before lifecycle changes.

## 4. Change Review lifecycle

### 4.1 Opt-in and storage disclosure

The global `[change_review] enabled` setting remains a master availability
switch. Per-workspace state changes as follows:

- an explicit stored `true` remains enabled;
- an explicit stored `false` remains disabled;
- no row means disabled;
- a storage/read failure means disabled and emits a content-free diagnostic.

This changes existing default behavior deliberately. No database migration is
needed because the existing Boolean row can represent both explicit states.

The registry exposes a tri-state read result: `enabled`, `disabled`, or
`unavailable`. A missing row maps to `disabled`. A database failure maps to
`unavailable`, never to a Boolean that a caller might invert. Settings shows an
error and offers no toggle while state is unavailable. An available read also
returns an opaque durable revision derived from the existing `updated_at`
column; missing-row disabled uses a sentinel revision. Writes guarantee a new
revision distinct from the prior value, even under an injected/frozen clock.
The toggle handler may write only after an `enabled` or `disabled` read and
always writes the explicit opposite state. The write is compare-and-set against
both observed state and revision in one registry transaction. A concurrent or
ABA change refreshes Settings and reports a state conflict instead of
accidentally inverting another actor's choice. No schema migration is required.

The global capability reader is tri-state too. `[change_review] enabled`
remains `enabled` when the setting is simply absent, because it is a capability
master rather than workspace consent. An environment/config read or coercion
failure returns `unavailable`, fails runtime tracking off, is diagnosed without
content, and is rendered as unavailable in Settings. An explicit false returns
`disabled`. The per-workspace opt-in remains required even when the global
capability is available.

Settings copy must state that enabling review creates shadow Git history under
application data and retains file contents for the configured retention period
(30 days by default). Enabling an existing workspace starts initial snapshots
in the background and shows `Preparing change history` without disabling chat.
Readiness is per canonical root: `preparing`, `ready`, or `failed`. Turns do not
reserve or wait on a preparing/failed root; they proceed with a visible
untracked-root warning while ready roots in the same workspace remain eligible.
A failed root exposes a bounded retry action. On restart, enabled workspaces
rebuild this transient readiness state and initialize in the background.

One app-owned workspace-consent lock serializes local tracker/initializer
admission, toggle compare-and-set plus readiness publication, and initializer
completion. Admission linearizes at its transactional tri-state/revision read
under that lock. A toggle linearizes only when its compare-and-set write commits
under the same lock. An admission that reads enabled before a disable commit
may finish and publish; every admission ordered after that commit reads disabled
and is rejected. A failed or conflicting toggle changes no runtime state.

Each initializer captures the enabled revision under the lock. On completion it
reacquires the lock and publishes `ready` only after a transactional re-read
still reports that exact enabled revision. Disable→re-enable therefore cannot
let an old initializer survive an ABA transition, including a revision changed
by another process. The Settings copy states that disabling is not retroactive
erasure of already admitted turn results.

No toggle transition silently deletes existing history. Deleting retained
history is a separate follow-up design: a canonical root may be shared across
workspaces, and deletion must coordinate with initializers, baselines,
finalizers, reverts, retention, transcript markers, and partial failures. It is
not part of the nonblocking-turn P0.

### 4.2 App-owned root-ordering coordinator

Asynchronous E is not a detached thread added to `run_reply`. One app-owned,
thread-safe `ChangeReviewFinalizationCoordinator` owns all review windows and
pure filesystem work. It is created beside the app-owned Console runtime and
disposed before `AgentRunsDB` closes.

At tracked-turn registration, the coordinator reserves a window independently
in every canonical-root lane. Each lane is FIFO across all conversations that
share the root, not merely per conversation. A lane reservation covers the
whole B-to-E window; a later B cannot pass an earlier turn whose E has not
settled. Registration happens before model execution, so a successor is known
even if the prior E is still running.

One turn may reserve multiple roots, but the coordinator never holds more than
one root-lane lock while doing filesystem or Git work. Per-root B/E results are
aggregated only after each independent operation settles. Dynamically
auto-registered nested roots are enrolled as their own lanes before they can
contribute a result. This prevents multi-root lock-order deadlocks.

The existing sub-agent survivor-window contract is preserved inside each root
lane. If children remain live at primary E, the lane records a post-turn
survivor window. The next registered turn's B closes that window at the same
SHA before opening its own window; if no successor arrives, the last child
settlement closes it. Successor registration is therefore not a callback that
can race ahead of window creation—it is already queued in the lane.

The coordinator uses a fixed small worker set and a bounded pending-operation
queue. Exhausted admission disables tracking for the new turn with an honest
warning; it never creates another unbounded thread or delays the model request.
Workers receive only immutable filesystem inputs and return pure review
results. They never receive an `AgentRunsDB`, Console store, widget, or UI
callback.

Reservation is all-or-nothing from the caller's perspective. A multi-root
registration that cannot reserve every lane tombstones any reservations it
already placed; lane executors skip those IDs and wake successors. Admission
exhaustion rejects before returning a live handle. Shutdown atomically
tombstones all pending reservations, wakes baseline/finalizer waiters, and
closes any in-memory survivor lineage as unavailable. A fresh coordinator
never inherits an in-memory reservation for work that was not admitted.

These FIFO guarantees cover every review-enabled Console conversation in one
application process. A disabled workspace sharing the same canonical root does
not reserve a window, take snapshots, or enter the Change Review dispatch path.
Its writes are therefore an attribution limit equivalent to another local
writer and are disclosed in review help. Adding a global mutation-observer
protocol solely to label this rare overlap is deferred unless incident evidence
shows the limitation is materially confusing.

Another Chatbook process can operate on the same canonical root. The existing
portable shadow-repo lock continues to protect repository integrity, but
cross-process write attribution is imperfect and remains disclosed in review
help; this coordinator does not claim a distributed total order.

### 4.3 Turn completion and durable publication

The agent outcome and assistant-message persistence remain the authoritative
chat completion. Once those succeed or reach another terminal status:

1. the bridge schedules review finalization;
2. the bridge returns the run outcome without awaiting filesystem snapshots;
3. the prompt coordinator clears the accepted turn and may drain the next
   prompt; and
4. finalization later stores any change rows by `run_id` and requests a
   content-free refresh.

Publication needs no in-memory result/anchor handshake. `AgentRunsDB` already
stores the run's `assistant_message_id`, and change rows already reference the
run. The assistant-anchor writer and finalizer writer commit independently and
each requests the same idempotent refresh. Transcript rendering joins durable
state and derives a marker only when both the assistant anchor and review rows
exist, inserting it immediately after the originating assistant message. A
result-first race renders on the later anchor refresh; an anchor-first race
renders on the later result refresh. If no screen is mounted, the next mount
derives the same placement. If a run never obtains an assistant anchor, its
review rows remain inspectable in run history but never invent a transcript
position. The coordinator retains no widget, store callback, anchor listener,
or pending publication handshake.

Every terminal path still requests an end snapshot, including failed and
cancelled turns. A normal bounded failure produces a tracking-error result. An
application shutdown that has already invalidated persistence may drop an
unfinished result with a content-free diagnostic; the design does not promise
a database write after persistence has become unavailable.

### 4.4 Preparation, existing permission review, and conservative mutation gate

Baseline B still begins in parallel with the model request for a review-enabled
workspace. The existing runtime and permission architecture remains intact:

1. `prepare_tool_calls` resolves project-instruction scope and may defer the
   whole batch for another model turn;
2. the current combined `review_tool_calls` hooks run unchanged, including the
   app-global kill switch, prompts, stamps, pinned refusal text, and auditing;
3. calls carrying an explicit non-`proceed` review verdict skip both dispatch
   and the baseline wait;
4. every remaining potentially dispatchable call waits for B across all roots
   tracked by the turn, bounded to three seconds; and
5. dispatch enters the existing invocation path, which remains the sole owner
   of final permission resolution and refusal auditing.

A project-context-deferred batch never reaches permission review or waits for
B. Change Review adds no post-review permission projection, provider audit
protocol, effect classifier, or root-target grammar. A configured deny that is
enforced only inside `invoke()` may pay an unnecessary bounded wait; preserving
one permission owner is safer and smaller than predicting that outcome.

If an existing permission-review hook raises, the conservative wrapper performs
the same bounded all-roots wait before allowing the runtime's existing hook-
failure policy to continue. A raised hook therefore cannot bypass B, but Change
Review still does not reinterpret the permission failure or become an
authorization mechanism.

The only bypass is a fixed app-runtime table for special tools whose handlers
cannot write workspace files: `find_tools`, `load_tools`, `skill_file`,
`search_run_log`, `run_log_stats`, `run_log_slice`, `wait_agents`, and
`check_agents`. These calls never wait for B. `spawn_subagent`, `install_skill`,
`run_skill_script`, `send_to_agent`, every provider call, every skill call, and
every unknown call remain conservative and wait across all tracked roots.
Common catalog discovery therefore cannot consume the mutation ceiling, while
new provider protocols are avoided.

When the three-second wait expires:

- only unresolved roots are irrevocably invalidated for this turn;
- any late baseline result for an invalidated root is ignored and cannot
  restore validity or create a misleading record;
- the tool executes under its normal permission and path gates;
- completed roots remain trackable;
- a content-free warning is persisted and shown naming aliases, not absolute
  paths: change review could not establish a baseline for those roots, so this
  turn's changes there are not tracked.

Timeout while a predecessor survivor window is open has a stricter recovery
protocol. For each affected root, the lane atomically:

1. marks the unresolved current window attribution-invalid;
2. marks the predecessor survivor window attribution-invalid too, because the
   successor mutation can no longer be separated honestly from late child
   writes;
3. persists tracking-error results for both originating runs when their run
   records remain available, without claiming file counts or a B/E diff;
4. closes that logical lineage and discards every late B/E result from it;
5. enters a degraded epoch in which further mutations may proceed but no turn
   claims attribution;
6. after all known survivor children and mutation-capable windows in the epoch
   settle, takes one un-attributed resynchronization snapshot and uses that SHA
   only as the starting point for future windows.

If quiescence never arrives, the lane remains visibly degraded and future
mutations remain untracked; chat and tools still work. Disabling/re-enabling
review may request a new initialization only after the same lane reports no
live operation. A late baseline can never close the predecessor window or
retroactively restore either run. Deterministic coverage holds a survivor
open, times out the successor B, performs a successor mutation, and proves
that neither run receives the mutation as a valid diff before resynchronizing.

Review remains observation, not authorization. It cannot deny a tool that the
permission system allowed.

### 4.5 Shutdown lifecycle

Disposal ordering is explicit:

1. stop accepting review windows and increment a coordinator generation;
2. prevent new B/E work from being scheduled;
3. bounded-drain already-running pure filesystem workers;
4. discard every result whose captured generation no longer matches;
5. dispose the coordinator; and
6. only then close `AgentRunsDB` and the rest of the Console runtime.

Workers may outlive the bounded drain because Python threads cannot be killed.
They are daemon-owned, Git subprocesses retain their existing timeouts, and a
late worker can only finish shadow-repo filesystem work and return a pure value
that the disposed generation rejects. No worker can write a closed database or
touch UI/store state. Owner-thread persistence already in progress is allowed
to finish before the database-close step.

### 4.6 Queue boundary

No prompt-queue schema or database persistence is added. ADR-046's privacy and
restart semantics remain intact:

- queued prompts survive tab and workspace switching inside Console;
- leaving Console or quitting with unsent queue entries requires the existing
  revision-stable confirmation;
- confirmed exit may discard process-memory queue text;
- forced process termination may lose it.

The incident is fixed at its source: asynchronous review finalization cannot
keep `accepted_live_turn` true. A deterministic test holds the finalizer behind
a barrier, submits a third prompt after the second assistant response becomes
durable, and proves that the third turn starts without waiting for review.

Content-free diagnostics record button receipt, dispatch outcome, queue count,
worker start, durable turn acceptance, terminal release, review scheduling,
and any confirmed queue discard. Prompt bodies, previews, file contents, and
absolute paths are never logged.

## 5. Workspace file-tool authority

### 5.1 Preserve ADR-069 for project-instruction-enabled sessions

ADR-069 remains authoritative when project instructions are enabled. The
session selects exactly one workspace folder binding; that selected root is
the local provider's authority boundary and working directory. Local tools do
not gain the app sandbox or any other binding in this mode. Built-in file tools
retain their existing workspace/sandbox authority and emit the existing
outside-instruction-scope warning when they target another admitted root.

The immutable run context captures all of the selected authority tuple:

- workspace ID;
- stable binding ID;
- canonical-locator fingerprint;
- access mode; and
- root/ancestor filesystem identity.

Binding ID alone is explicitly insufficient because the registry can retarget
its locator. Every local-tool invocation re-reads the binding from the same
workspace and compares membership, status, binding ID, canonical-locator
fingerprint, current `ro`/`rw`, and filesystem identity. Removal, retargeting,
or registry failure refuses the call and requires re-selection. An `rw` to
`ro` downgrade takes effect immediately. A new binding never expands an
already-running provider.

This design does not duplicate the atomic execution lease required to close
rename/replacement check-use races. `TASK-19637`, "Atomically pin local-tool
workspace execution", is a prerequisite for shipping the new root-selection
behavior. It was renumbered from the later duplicate `TASK-16324`; the
unrelated completed task keeps that historical ID. Its ambiguous `TASK-16320`
dependency was removed because the intended AGENTS.md delivery is already
complete and merged. Its record links the exact
`task-16320 - Add-startup-AGENTS.md-project-context-to-Console.md` filename and
`069-console-project-instruction-local-state-and-preflight.md`. Its
cross-platform confinement ADR is separate from this incident's Change Review
ADR.

### 5.2 Disabled named workspaces and Default

For a named workspace whose project instructions are explicitly or
legacy-disabled, local path tools use the bindings admitted at run start. The
run captures each binding's ID and canonical-locator fingerprint; call-time
validation applies the same membership, retargeting, access-downgrade, and
filesystem-identity checks described above. This is the smallest compatible
extension: it repairs the hidden CWD/config root without changing ADR-069's
instruction-enabled single-root model.

Local path tools gain a stable optional `root` string argument:

- in project-instruction-enabled mode, omission selects the one selected
  binding and only that binding's exact ID is accepted;
- in a disabled named workspace with one admitted binding, omission selects
  that binding;
- in a disabled named workspace with multiple admitted bindings, omission
  returns a bounded actionable error listing their aliases;
- an unknown, stale, newly added, or nonmember alias is refused;
- reads accept `ro` or `rw`; writes accept only current `rw`;
- paths remain relative to the selected root and cannot escape it; and
- schemas never enumerate current roots, so ordinary binding changes do not
  churn tool definition hashes every turn.

The collision-free model-facing alias is the exact stable `binding_id`; the
workspace context note pairs it with the human label and current access.
Absolute locators remain governed by the existing workspace-context contract;
errors and telemetry use aliases.

`LocalToolProvider` also owns unrelated web, Watchlists, and session-todo
capabilities, so the provider itself is never dropped merely because path
authority is absent. Composition gains an explicit no-path-authority/spec-
filter mode. The Default workspace and a disabled named workspace with no valid
binding compose the provider with only its non-filesystem/Git specs; `fs_*` and
`git_*` are absent while web fetch/search/crawl/deep-search, Watchlists, and todo
availability remains governed by its existing gates and session wiring. In both
cases, the built-in file family keeps its existing ADR-028 sandbox access. This
is what "sandbox-only" means in this design; local Git/patch tools do not gain
sandbox authority merely as a side effect of removing the fallback.

`[console] workspace_root` is deprecated and ignored by the in-app Console.
The live process CWD is never an implicit root. Standalone external MCP
exposure continues to require its explicit configured root and cannot borrow
an in-app active workspace. Migration and Settings copy explain that users who
relied on in-app CWD/config-root access must bind the folder to a named
workspace.

The one-time local schema change correctly invalidates stored persistent
approvals through the existing definition-hash guard. Documentation warns that
first use after upgrade may request approval again.

### 5.3 Shared filtering and execution confinement

The existing duplicated binding filtering should first be consolidated through
TASK-17067 so context notes, Change Review roots, built-in enforcement, and
local-provider composition cannot drift. This helper supplies validated
binding facts; it does not make `folder_binding_roots` an authorization source.
That function remains Change Review-specific and returns no roots when review
is disabled.

Project-instruction preflight and dispatch continue resolving a model tool name
through the registry's same immutable first-owner snapshot and reuse that
owner's existing `path_targets` result. The atomic execution lease then carries
the validated authority through the actual filesystem or Git operation; a
call-time `Path.resolve()` check alone is not claimed to close TOCTOU races.
Change Review does not parse or selectively map tool targets in this recovery;
its bounded conservative gate covers every tracked root.

## 6. Tool disclosure and discovery

When the allowed catalog exceeds `DIRECT_DISCLOSE_THRESHOLD`, initial
disclosure contains a bounded core instead of zero real tools. The default core
is the available subset of:

- `fs_list`
- `fs_read`
- `fs_glob`
- `fs_grep`
- `git_status`

Write tools, web tools, Watchlists, skills, MCP tools, and other long-tail
entries remain discoverable through `find_tools`/`load_tools`. The core is
deduplicated from later loads and counts against `max_active_tools`.

Catalog search ranks results by:

1. exact tool name;
2. normalized name or prefix match;
3. all normalized query tokens present across name and description;
4. token coverage and deterministic catalog order.

Normalization handles underscores, punctuation, case, and simple singular /
plural forms. Results are capped and deterministic. A query that shares no
token returns no result rather than an arbitrary guess.

Acceptance tests use production-shaped catalogs and natural phrases including
`list files in directory`, `read a file`, `search files for text`, and
`git status`. They prove the intended tool is found and executed under the
normal permission gate. Existing exact-name and collision precedence remain
unchanged.

Performance evidence must compare:

- provider-call count for a common file listing;
- serialized initial tool-schema bytes;
- estimated prompt tokens;
- end-to-end time under a deterministic fake provider.

The change is retained only if common file tasks avoid the find/load round-trip
without injecting the full 19-tool schema set.

## 7. Console mount performance

This slice begins with subtree instrumentation, not a redesign. Record for at
least 30 warm navigations from an isolated profile:

- first interactive composer paint;
- full screen ready time;
- widget count and mount time by top-level Console subtree;
- focus restoration and outgoing-screen teardown time;
- key-to-composer-echo latency after the first visible paint;
- Enter-to-send-worker scheduling latency;
- median and p95 for every latency metric.

If one secondary subtree accounts for enough cost, defer only that subtree
until `call_after_refresh`. Inspector/right-rail content is the first candidate,
not a preselected outcome. Every eager `query_one`, mount hook, focus path,
restore-state path, and view-hook binding that touches the subtree must tolerate
the deferred state.

Keep the change only when all conditions hold:

- median first-interactive time improves by at least 15%;
- full-ready median regresses by no more than 5%;
- p95 key-to-echo and Enter-to-worker latency regress by no more than 10%;
- widget count or measured mount work decreases before first interaction;
- screen-navigation freeze gates, rapid-switch soak, focus restoration,
  state restore, and unmount cleanup remain green.

Cached `Screen` instances remain prohibited. If no subtree meets the threshold,
the task closes with measurements and no production refactor. A later wrapper
flattening project requires its own design and ADR.

## 8. Runaway-process diagnostics

Do not change AgentService or fleet shutdown ownership without a reproduction.
The diagnostic task will:

1. run the exact suspected phase-zero test set with a shorter per-test timeout,
   verbose progress, RSS samples, current-test identity, and project-owned
   thread names/counts;
2. prefix-bisect on a reproduction;
3. capture Python stacks before termination;
4. assert suspected fixtures return project-owned thread counts near baseline;
5. add a bounded producer-signal wait only where a concrete unbounded wait is
   demonstrated.

Existing pytest and CI timeouts remain. A broad autouse thread-leak failure is
not introduced until thread names and legitimate survivors are classified;
otherwise the diagnostic itself would make the suite flaky.

Stale processes from old worktrees are operational residue. Cleanup resolves
the exact PID after revalidation; the application must not gain a general
process killer.

## 9. Delivery slices and acceptance gates

Before implementation, the new atomic-confinement task was renumbered from its
duplicate `TASK-16324` to unique `TASK-19637`; its already-satisfied ambiguous
`TASK-16320` dependency was removed, and its record links the exact completed
AGENTS.md task filename plus ADR-069. Latest `dev` had seven duplicated Backlog
task IDs; this incident changed only the task that blocked an explicit
dependency and records the other six as repository hygiene rather than silently
broadening scope. ADR references below always include full filenames because
the repository also contains historical duplicate numeric ADR IDs.

Change Review delivery is ordered A1 → A2 → A3: consent/default posture
first, nonblocking ownership second, and mutation attribution last. Slice B may
prepare in parallel only after its shared-filtering and atomic-confinement
prerequisites have unique task identities and accepted plans.

### Slice A1 — Change Review opt-in and disclosure (P0)

- Missing/error workspace state is disabled; explicit true remains enabled.
- Unavailable state cannot be inverted into an enable write.
- Admission and toggle commits obey the transactional linearization rule in
  section 4.1; a failed toggle changes no runtime state.
- Initializer readiness is revision-bound and cannot survive disable/re-enable
  ABA.
- Disabled workspaces create no shadow snapshot thread on binding or turn.
- Settings disclose retention; history deletion is a separate follow-up.
- ADR-079 is accepted before this slice changes the existing default.

### Slice A2 — nonblocking finalization and publication (P0)

- Review finalization cannot hold `accepted_live_turn` or queue drain.
- Finalizers are root-ordered across conversations; rendering joins durable
  change rows to the existing assistant anchor without in-memory handshake
  state.
- Successor turns with surviving sub-agents preserve the shared B/E boundary.
- A deterministic mounted three-turn test holds finalization and completes the
  third turn.
- Shutdown rejects late worker results before `AgentRunsDB` closes.

### Slice A3 — conservative bounded mutation gating (P0)

- Project preparation and permission review precede any baseline wait.
- Existing permission hooks, refusal copy, stamps, audit, and invocation
  ownership remain unchanged.
- Explicit hook refusals skip the wait; remaining provider, skill, script,
  opaque runtime, and unknown calls wait across every tracked root.
- Only the fixed pure runtime discovery/status table bypasses B.
- Mutation gating is bounded and late baselines are invalidated.
- A survivor plus successor-baseline timeout invalidates both windows and
  resynchronizes without attributing the successor mutation to either run.

### Slice B — workspace-root authority (P0)

- Complete or depend on TASK-17067's shared filtering foundation.
- Renumber the duplicate-ID atomic-confinement task and complete its execution
  lease before shipping new multi-root local authority.
- ADR-069 remains unchanged for project-instruction-enabled single-root runs.
- Disabled named workspaces use only their run-admitted bindings for local path
  tools; multi-root aliases and ro/rw enforcement use real directories.
- Active-workspace changes mid-run cannot retarget tools.
- Binding removal, locator retargeting, and `rw` to `ro` downgrade revoke access
  mid-run.
- Default and binding-less named workspaces filter out local `fs_*`/`git_*`
  specs while preserving local web, Watchlists, and todo capabilities; built-in
  file tools keep sandbox access, and no process-CWD/config-root fallback
  remains in-app.
- Standalone MCP retains only its explicit documented root.

### Slice C — tool reachability (P1)

- Common read schemas bootstrap in a production-sized catalog.
- Natural-phrase discovery ranks the expected tool.
- File listing reaches the real tool without find/load round trips.
- Full-catalog prompt inflation is avoided and measured.

### Slice D — Console mount experiment (P1)

- Subtree measurements are reproducible from an isolated profile.
- A production change lands only if every threshold in section 7 passes.
- Existing navigation freeze and lifecycle gates pass.

### Slice E — runaway-test investigation (diagnostic)

- A reproduction identifies a concrete owner and wait/thread leak, or the task
  closes with captured evidence and no speculative runtime change.
- Any resulting fix receives its own atomic task and acceptance criteria.

## 10. Testing and verification

Implementation follows TDD. Each bug fix begins with a failing test at the
lowest useful seam and adds joined/mounted coverage for the lifecycle claim.

Required gates include:

- real-Git tracker tests;
- settings registry and mounted settings tests;
- controller/bridge queue lifecycle tests with deterministic barriers;
- local provider and two-workspace integration tests;
- production-shaped agent catalog tests;
- focused Console navigation/freeze tests;
- shared-root concurrent-conversation ordering tests;
- successor-turn tests with surviving sub-agents;
- survivor-plus-successor-timeout degraded-lineage/resynchronization tests;
- shutdown-during-finalization tests with a deliberately late pure worker;
- result-before-anchor and anchor-before-result tests proving the same durable
  query renders correctly without coordinator-held publication state;
- missing/unavailable opt-in state and failed-toggle tests;
- one barrier-controlled send/disable race test proving both linearization
  orders and that a failed toggle changes no admission state;
- one initializer-completion barrier test spanning disable→re-enable, proving
  the stale revision cannot publish `ready` under local or cross-process ABA;
- binding removal, locator-retargeting, and access-downgrade mid-run tests;
- project-preparation deferral and explicit permission-refusal tests proving no
  baseline wait occurs;
- configured-deny tests proving the conservative wait may occur but existing
  invocation remains the sole refusal/audit owner;
- raised-review-hook tests proving the bounded all-roots wait still occurs
  before the runtime's existing failure policy continues;
- runtime-special tests proving `find_tools`/`load_tools` do not wait for a cold
  B while provider, skill, spawn, install, script, message, and unknown calls
  conservatively wait across all tracked roots;
- deterministic root rename/replacement confinement tests from the prerequisite
  execution-lease task;
- provider-composition tests proving no-path-authority mode removes only
  `fs_*`/`git_*` while preserving web, Watchlists, and todo availability;
- one-time approval definition-hash invalidation tests;
- isolated live verification with scratch config, data, cache, and database
  roots against a real provider;
- `git diff --check`, focused lint/static checks, and the relevant test suites.

Live verification must complete at least three sequential turns, including one
file read and one file mutation, with Change Review disabled and then enabled.
It records turn-release timing, review-finalizer timing, tool root alias, and
queue outcome without recording prompt or file content.

## 11. ADR impact

ADR required: yes.

- Create ADR-079 for Change Review opt-in state, shadow-content ownership and
  retention disclosure, ordered asynchronous finalization, durable publication
  joins, and bounded conservative mutation gating. It supersedes the default-on
  and synchronous completion portions of the 2026-08-02 design.
- Keep ADR-069 authoritative for project-instruction-enabled sessions. Amend
  only its disabled/legacy provider-root consequence: those named sessions use
  admitted workspace bindings instead of `[console] workspace_root` or CWD.
- Amend
  `backlog/decisions/028-settings-workspaces-category-and-folder-roots.md` so
  binding filtering, Default sandbox behavior, revocation, and disabled-session
  multi-root aliases are explicit without widening enabled-session authority.
- Amend
  `backlog/decisions/032-local-agent-tool-permission-boundary.md` to remove
  implicit process-CWD/config-root confinement for in-app Console path tools
  and document the one-time schema/approval change plus non-path spec
  preservation. Existing permission and kill-switch ownership is unchanged.
- The atomic-confinement task writes its own ADR for the cross-platform
  execution lease. ADR-079 references but does not duplicate that runtime and
  security boundary.
- ADR-046 remains unchanged unless implementation discovers that its existing
  confirmed-discard navigation contract is not actually enforced. Queue
  persistence is not part of this design.
- Console subtree deferral requires an ADR only if the measured implementation
  introduces a new long-lived composition/lifecycle boundary. Measurement and
  a local `call_after_refresh` deferral alone do not.

## 12. Explicit non-goals

- Persisting or replaying unsent queue prompts after restart.
- Deleting retained Change Review history in the incident P0.
- Making Change Review an authorization or approval mechanism.
- Tracking disabled workspaces, even to pre-warm history.
- Replacing shadow Git with a new diff engine.
- Caching/remounting Textual `Screen` instances.
- Rewriting the Console runtime, queue, or workspace registry wholesale.
- Killing arbitrary host processes from the application.
- Fixing an AgentService leak without a reproducible owner and failing test.
