---
target: Console-driven Watchlists workflow UAT remediation design specification
total_score: 24
max_score: 40
na_heuristics:
p0_count: 0
p1_count: 5
timestamp: 2026-08-27T04-00-31Z
slug: console-driven-watchlists-workflow-uat-remediation-design-md
---
# Design Health Score

## Planning disposition

The approved design and implementation plans resolve the critique's open
contracts as follows: TASK-22860 owns the focused provenance/claim migration;
TASK-22861 and TASK-22863 own exact receipt polling and Console-only briefing
detail; TASK-22862 pins collision and partial-result behavior; TASK-22864 pins
86,400-second interval semantics and reload acknowledgement; TASK-613 and
TASK-22867 own single-flight classification; TASK-22868 owns the First Run and
cross-surface closeout. This pre-hardening score is retained as review history,
not as the expected post-implementation score.

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of system status | 3 | Durable receipts are strong, but accepted work lacks an exact receipt-keyed follow path. |
| 2 | Match system / real world | 2 | “Daily” currently means an elapsed 86,400-second interval, not a calendar day at a chosen time. |
| 3 | User control and freedom | 2 | Cancellation, partial-result continuation, and existing-collection behavior are not pinned. |
| 4 | Consistency and standards | 2 | State vocabularies and Console/UI collection-collision behavior can diverge. |
| 5 | Error prevention | 2 | Bounds are good, but unsafe cadence values and race-prone idempotency remain possible. |
| 6 | Recognition rather than recall | 2 | The human must infer which receipt, screen, or follow-up tool owns recovery. |
| 7 | Flexibility and efficiency | 3 | Console, UI, bulk, canonical-ID, and keyboard paths are strong foundations. |
| 8 | Aesthetic and minimalist design | 2 | The proposed start state and operations snapshot still carry too many equal-weight choices. |
| 9 | Error recognition and recovery | 3 | Classified errors and stale-while-refreshing are strong; exact retry/cancel/deep-link paths are incomplete. |
| 10 | Help and documentation | 3 | Documentation is planned, but setup blocking, schedule semantics, and consent scope need contextual copy. |
| **Total** |  | **24/40** | **Acceptable direction; significant contract hardening required before implementation planning.** |

# Design Specificity Verdict

The design is unmistakably Chatbook: local-first authority, Console orchestration, ADR-032 approvals, app-open scheduling, Watchlists receipts, Artifacts recovery, briefing provenance, and the explicit no-hunt boundary are product-specific and coherent.

Its interaction and persistence contracts are less specific than its architecture. Critical choices are still left to implementers: exact receipt polling, cancellation, partial mutation behavior, collection collisions, schedule anchoring, external briefing privacy, fresh-profile blocking, and what “read latest briefing” presents. Those gaps could yield several technically valid but incompatible implementations.

The deterministic detector returned zero findings for the Markdown target. This is a false-clean signal for this review: the detector had no applicable markup/UI rules for an architecture specification. Repository-contract inspection found the issues below.

# Overall Impression

The product direction is right and should be preserved. The single biggest opportunity is to make every promised outcome traceable to an existing durable owner—or explicitly authorize the schema/owner work needed to make it true. Once that is done, the workflow can feel unusually trustworthy rather than merely feature-complete.

# What's Working

1. The domain façade and exposure boundary are excellent: tools express user intent, not SQL or widget actions, and reuse the established permission principal.
2. Operational honesty is a strong design value: receipt-before-acknowledgement, app-open limitations, safe failures, interrupted-work reconciliation, and stale-while-refreshing all support user trust.
3. Briefing consumption is scoped thoughtfully: generated/untrusted labels, selected-versus-cited evidence, bounded JSON, redacted URLs, and no hunt coupling directly answer the user's need.

# Priority Issues

## [P1] Several guarantees are impossible under the claimed no-schema design

**Why it matters:** The existing `briefing_items` junction stores briefing ID, item ID, and featured state, but not selection order or an immutable citation/source snapshot. Item deletion cascades remove the relationship. Source-check runs have no uniqueness constraint preventing concurrent active receipts. Source URLs also have no atomic uniqueness constraint. The specification currently promises ordered provenance and duplicate-free accepted work as though those are already representable.

**Fix:** Choose and document one of two honest contracts for each case. Recommended: permit a focused migration for ordered briefing provenance and atomic active-run/idempotency claims. Store the minimum immutable snapshot needed to explain a generated briefing after source/item edits or deletion. If migration is rejected, explicitly downgrade the promise to unordered, live, best-effort provenance and allow duplicate queued receipts that later resolve as skipped. Remove the blanket “No new database schema” assertion.

**Suggested command:** `$impeccable harden`

## [P1] “Accepted” operations do not yet form a closed user journey

**Why it matters:** The proposed check and generation tools return durable IDs, but the status tool cannot be queried by that exact operation ID. Existing `generate_briefing` returns only after the model call and does not expose a public accept/execute split. A background coordinator also lacks specified strong task ownership, cancellation, shutdown, exception consumption, and stuck-receipt reconciliation.

**Fix:** Introduce an owner-level accept/execute API and exact `watchlists_get_operation_status(operation_id)`. Every accepted result returns `poll_tool`, exact `poll_arguments`, a bounded suggested poll interval/backoff, and terminal states. Use one Console receipt card that transitions from Queued to Running to Complete/Empty/Failed and deep-links to the exact Runs or Artifacts record. Define app shutdown and cancellation behavior. Prefer UI/store-driven receipt refresh over indefinite model-call polling.

**Suggested command:** `$impeccable harden`

## [P1] Mutation, idempotency, and consent semantics are ambiguous

**Why it matters:** “Create or resolve” can mutate an existing case-insensitive collection even though the incumbent UI auto-suffixes collisions. Bulk source partial success can silently feed an incomplete collection. “Normalized URL” is undefined and can accidentally merge distinct signed/query endpoints. External persistent Allow for full briefing detail expands access to private content beyond the stated Console-agent need.

**Fix:** Make collection creation strict by default with an explicit `if_exists` policy; never mutate an existing collection without that choice. Validate all collection membership changes before one all-or-nothing transaction. Pause follow-on workflow after partial source creation until the user chooses to continue with valid rows or fix failures. Reuse exact configured-source identity with only outer-whitespace cleanup unless a separately specified canonicalization policy is approved. Add fail-closed exposure metadata with no permissive default. Recommended privacy posture: list/receipt reads may be externally exposable, while full briefing Markdown remains Console-only unless the user explicitly chooses the broader boundary. Approval cards should show read/modify/network effect, entity count, sanitized destinations, approval duration, and revocation location.

**Suggested command:** `$impeccable clarify`

## [P1] “Daily” and scheduler acknowledgement are not honest enough

**Why it matters:** Current storage/projection semantics mean “every 24 hours from the latest attempt,” with a never-run schedule due immediately. That differs from “once each calendar day at a local time.” The advanced cadence accepts one second and has no practical maximum, enabling runaway LLM/network spend. `request_reload()` only sets a flag; it cannot truthfully report that the queue has reloaded.

**Fix:** For the minimal existing-schema path, rename the user contract to “Every 24 hours” and state immediate first run, attempt-anchored next run, reopen/catch-up behavior, timezone display, and app-open requirement. Bound advanced cadence values; recommended minimum one hour and a documented maximum. Pin weekly to 604,800 seconds. Distinguish `reload_requested` from `reload_acknowledged`; either add an acknowledgement token/future with timeout or report only the request. If calendar-daily behavior is desired, specify local time, timezone/DST, and missed-run policy and treat it as a separate scheduling/storage expansion.

**Suggested command:** `$impeccable clarify`

## [P1] The first and final moments of the Console journey remain underspecified

**Why it matters:** A fresh user cannot ask an agent to configure Watchlists until a provider/model works. The proposed six equal Console suggestions ignore that dependency. At the other end, “read latest briefing” does not define whether “latest” means the newest receipt or newest completed readable briefing, how truncated content continues, or how provenance appears without ending on raw JSON.

**Fix:** Add a blocked Console state with one primary action, “Configure provider and model,” preserve the user's draft intent across the setup round trip, and read persisted readiness back before resuming. Specify provider/model state transitions rather than treating highlight as commit. Define latest-readable traversal as list completed briefings first, while separately showing newer generating/failed receipt context. Render a briefing result card with freshness, generated/untrusted label, readable body, truncation badge, source disclosure, and Open in Artifacts. Bound provenance arrays and add continuation so metadata cannot consume the whole body budget.

**Suggested command:** `$impeccable onboard`

# Persona Red Flags

## Jordan — first-time user

- Six Console suggestions appear before the prerequisite provider/model is resolved.
- “Cadence,” “preset,” “selection mode,” “reload acknowledgement,” and canonical IDs are implementation terms at decision points.
- Partial bulk success does not say whether the system proceeds with fewer feeds.
- “Read latest” hides multiple tool calls and has no pinned human presentation.

## Alex — power user

- Returned run IDs cannot directly query the proposed broad status tool.
- Repeated read approvals and model-driven polling can make the fast path slow.
- Multi-selection lacks range, select-all, filtered-selection, and shortcut semantics.
- Console and UI collection collision behavior can differ.

## Security-conscious operator

- Persistent external Allow for full briefing detail may expose future private briefings without per-call consent.
- Network effects are descriptive rather than enforced by a dedicated risk floor.
- Partial mutations can land before the operator understands failed rows.
- Generated/untrusted labels do not yet constrain how assistant prose distinguishes summary from verified evidence.

# Minor Observations

- Split `watchlists_get_operations_status` into overview and exact receipt lookup.
- Define a single default source sort and stable cursor order.
- Define canonical run ID shape and date/time normalization.
- Normalize state vocabulary across tool envelopes, receipts, first run, and UI status.
- Add a surface/state responsibility matrix covering empty, loading, partial, queued, complete, failed, unsupported, and narrow layouts.
- Specify exact supported terminal floors, focus order, Escape behavior, bulk-entry syntax, range/select-all behavior, and draft preservation.
- Make TASK-613 an explicit prerequisite or incorporate its acceptance criteria into the skill-import slice.
- Split the current delivery slices further: exposure filtering; bounded reads; transactional commands; async operation coordination; feed transport; scheduler semantics; Watchlists UI; First Run; and Library import.
- A wrapper-skill recovery option must state that classification/import does not execute repository code.
- Suppress local Watchlists action suggestions in server runtime mode.

# Questions to Consider

- Is “daily” an elapsed interval or a calendar event in the user's mental model?
- Should a command named “Create Watchlist” ever modify an existing one without an explicit policy?
- Is external MCP access to complete briefing bodies necessary, or does Console-only detail better match the stated goal?
- If 48 of 50 sources are valid, should the user or the agent decide whether to continue?
- Is immutable ordered provenance worth a small migration, or should the UI clearly label it as live best-effort data?
