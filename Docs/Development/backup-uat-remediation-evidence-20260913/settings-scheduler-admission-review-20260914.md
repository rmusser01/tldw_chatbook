# Settings scheduler admission diagnosis

Read-only source review against exact 0811768fa4c4685613dddc194dfa7900d4089dd1. Current scheduler loop, raw participants and storage admission have no diff from that commit. Original Windows evidence remains under /private/tmp/uat-windows-081176-support. No repository files or keyboard fixtures changed.

## Confirmed

SchedulerLoop._maintenance_drain (loop.py213–225) expired while an accepted heartbeat offload remained alive. _offload233–247 retains the actual to_thread task and signals its completion; cancellation does not falsely retire its worker. RuntimeMaintenance closes scheduler intake at producer stage324, drains at376, and starts the local storage pause only after settle_producers returns (monitor_app710–711). Thus this failure is not explained by the coordinator locally pausing storage before draining the scheduler.

The four Windows rolling samples show the same heartbeat thread6508 progressing from raw scope699 (config anchor acquisition) to703 (member acquisition). It alternates native ACL/registry verification and same-root initialization waits. Concurrent Console thread5148 also advances from its operation acquisition to the private SQLite connection acquisition. Capture waits at admission.py600; the startup hold remains because producer settlement failed. These observations do not establish a permanent lock cycle. The available:false settlement metadata cannot identify the contemporaneous lock holder or its phase duration. Samples are rolling, not a timestamped acquisition trace.

## Disposable probe

/private/tmp/uat-scheduler-native-probe-9rci7b_o/probe.py and home/probe-result.json: exit0 with explicit qualified_for(admission) and non-null native lease key assertions. The ordinary default heartbeat path lookup invokes original acquire_storage five times and original bootstrap._registry20times, measured11.975ms on this macOS host. No result/exception or native gate was substituted. This is a per-call count, not a Windows speed estimate.

Holding the existing configuration rebuild RLock before scheduling the real _record_heartbeat worker causes the actual scheduler drain to return false at the bounded probe deadline. Releasing that lock lets the same retained worker complete; the heartbeat is read back and scheduler task accounting empties. This proves correct dependency retention, not that Windows was waiting on that particular lock (its captured worker had already passed it).

The first probe used a /var alias and obtained fallback admission only; its5/15count is explicitly excluded from native evidence. It remains at /var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/uat-scheduler-admission-probe-fn0vh3l4.

## Recommendation

No scheduler product fix is justified by the current evidence. Keep accepted-work drain, final tick heartbeat guarantee, fresh selector/native checks, and existing deadline. Do not cancel or omit an already accepted heartbeat, cache its default path, or move the storage pause before scheduler settlement.

Smallest next diagnostic is bounded metadata-only timing of the existing acquisition boundaries in the next already-planned Windows support run: initializing wait versus leader authority open; first scope; native hold-ready wait; post-acquisition validation. Record thread ID, fixed caller category (heartbeat/Console/other), count and elapsed only, plus the active initializing leader thread at wait entry. Also timestamp scheduler close/drain/worker completion. Preserve calls and errors. A deterministic eventual correction should be based on the dominant measured boundary and retain real selector-change/pending/native-pause refusals.

Passing a pre-resolved heartbeat_path is an existing constructor option, but the normal writer itself uses direct filesystem publication. Treating that option as a production cache would alter current selection/refusal behavior; it is not an established safe fix for this finding. No such change is recommended.
