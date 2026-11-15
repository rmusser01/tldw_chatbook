# Console attachment native waits implementation plan

> Execute one measured correction at a time using systematic debugging, causal tests, and independent review.

**Goal:** Explain the retained baseline attachment timeout, reduce the next measured Console preparation cost, and update PR3572 with auto-merge off.

**Architecture:** Existing finite-read owners, source qualification and App/Console shutdown retain authority. Diagnostics observe original code and task identities; no storage lease crosses an await and no permission or source check is relaxed.

**Spec:** User request; TASK-35679.2; previous evidence in `Docs/Development/evidence/windows-console-2026-11-15/`.

**ADR required:** No for investigation or a routine correction within the existing finite-read contract. **ADR path:** N/A; existing ADR126 and ADR230 apply. Reassess before changing ownership, scheduling contracts or storage authority.

## Constraints and review focus

- Keep native admission, Windows ownership/path checks, source-currentness, optional policy fallback and physical retirement intact.
- Retain custom callback behavior, persisted memory/lineage reads, repeated cancellation and Close semantics.
- Keep the existing six-return journey, ten-second attachment bound and 300-second external process deadline. Retain every failure.
- Separate original-code diagnostic spans from identically sampled headless comparison runs. No general latency claim from callback counts alone.
- Targeted tests only; use the pinned Ruff0.16.6 in an isolated tool installation. Preserve raw evidence bytes and exact source/harness maps.

## Steps

- [x] Read retained A3 failure and current reconciliation/read ownership.
- [x] Extend original-code observations with preparation callback identity/CPU and attachment await-chain/phase metadata; run a fresh unchanged-source diagnostic.
- [x] Record measured cost, one hypothesis and exact bounded correction before product edits.
- [x] Add a causal RED control, implement the minimal correction, and run GREEN plus connected source/custom/persisted/cancellation/Close coverage.
- [x] Freeze actual baseline/candidate sources and run serial comparison cohorts with unchanged guards and bounds; record incomplete outcomes without aggregating them into a speed claim.
- [x] Review source and evidence, complete scoped static checks, clean only owned temporary roots, and update PR3572 with auto-merge off.

## Evidence-led hypotheses

Retained A3 resume1 is predominantly awaiting native reads: the timeout dump catches active-workspace reconciliation opening a checked WorkspaceDB connection and a context presentation worker validating admission evidence. It does not establish executor starvation or a readiness deadlock. Resume0 separately sampled a synchronous subscriptions boot admission check on the UI thread during its large input delay.

One bounded candidate is the context memory-presence probe. It currently enters captured database scopes before calling the stock helper, even for a captured unpersisted conversation or absent repository, where that helper immediately returns False without database access. Any correction must qualify that module-global helper's original identity/code; unknown helpers retain full scopes. Optional policy reads remain finite and retain their own checks. This is a hypothesis until current native timings and causal controls qualify it.

### Selected correction after fresh diagnostic

The unchanged-source diagnostic completes six returns (attachment2.36–5.28s), and measures the subscriptions starter's original execution_allowed call at637ms wall/125ms main-thread CPU during resume2. Retained A3 separately samples the same boundary during its5.647s input stall. The timeout dump's context/workspace reads remain real native preparation costs, not proof of executor deadlock. UI-thread native boot admission can also age prepared display results while the loop cannot publish them.

Reject the empty-chat database-scope skip: existing scopes enforce pause, source/path and native-authority refusal even when the computed result is RAW. Keep all original scopes.

Move subscriptions' complete informational admission check plus original backfill callback into one existing-style finite App-owned invocation. Reuse the existing App read collection/drain and fresh Context/active Worker pattern; retain name/group/exclusivity and body error policy. The original backfill independently enters fresh execution_scope. Gate failures keep nonfatal warning or exact local-pause retry, distinguished from body errors; resolve the body callback only after a true gate. A denied gate creates only a short-lived metadata carrier instead of starter None, releasing its slot through terminal/backstop. No allow value or native lease crosses an await. Leave the actor-pack helper unchanged; a subscriptions-specific helper handles its different error policy.

ADR required:no new ADR; this applies ADR126/230's existing App-owned finite lifetime and boot admission without changing authority, effect gates, transaction policy or service boundaries. Require causal off-loop admission, denial, fresh effect refusal, physical drain under repeated cancellation, queued shutdown, callback result/error and gate retry controls.

### Implementation and review adjustments

The finite invocation lives in `app_boot_workers.py`, behind the existing App service-wiring method. Existing startup wiring stays byte-identical; all existing size ceilings are preserved or lowered and the new helper is born governed. This is an internal helper extraction, not a new lifetime or authority boundary.

The connected Actor Pack fixture originally cancelled the inner lifecycle method after one loop turn. The current method first awaits project-index shutdown, which deliberately rethrows delivered cancellation; the shipping outer `_shutdown` retries it. The fixture now observes the original native drain yielding with the exact App read before cancelling. It retains both cancellation requests, actual native ownership, physical retirement and reopen checks and existing deadlines. No shutdown implementation changed.

All failed controls remain in the evidence packet, including an incorrect test node (collection error), 60-second child startup/exit timeouts, and the old Actor fixture race. The earlier thread case reached its success marker before the whole-process deadline; this does not establish prompt process exit. The immediate diagnostic rerun retained the same deadlines.

Further native runs exposed the 60-second **whole-process test** budget as insufficient on this host: one subscription control completed assertions just before expiry, while the Actor child was still seeding built-in content before its ownership exercise. Raise only these two files' process budget to 120 seconds, retaining their internal held-operation, checkpoint and cancellation deadlines. The Console comparison's ten-second attachment bound and 300-second external deadline stay unchanged. Keep the failed packets; report the budget change explicitly rather than presenting those reruns as unchanged-deadline evidence.

### Verification outcome

Existing ADRs [126](../../../backlog/decisions/126-complete-local-backup-and-recovery.md) and [230](../../../backlog/decisions/230-console-send-preparation-and-io-ownership.md) govern the unchanged authority and physical lifetime. No new ADR is required.

Product commit: `ea5a104807a53c2ba69c2434f6a4a1b22118b139`. Original-code diagnostics measure the selected UI starter at 637.492 → 7.308 ms; a positively paired 4.1-second background call permits a 78.9 ms key observation while active. The existing original gate/body checks still run. Five other startup UI admission spans remain.

Final comparison order is before-1 / after-3 / after-4 / before-2, with all 24 returns complete and source/guard qualification passing. Preliminary after-1/after-2 are retained but excluded because restoring the unchanged startup module had changed its CRLF bytes to LF. The replacement cohorts restore its exact baseline bytes and verify all product paths before launch. Frozen executed harness bytes are preserved in Git.

Median navigation-to-draft is 120.993 → 102.127 ms, attachment 3683.187 → 3460.905 ms. First-return attachment is slower in both candidate processes (4924/4022 ms versus 3299/2565 ms baseline), and worst sampled loop overshoot is 360.003 → 367.724 ms. This is not a uniform or statistically established latency gain. Main-thread samples predominantly poll; diagnostics show existing CSS and awaited native preparation, with the selected subscription worker starting after the diagnostic first return. The exact historical timeout cause remains unproven.

Verification includes 391 policy/ratchet/P0 checks, native contracts across retained runs, pinned Ruff 0.16.6 and a passing 1040/1040 startup census. Independent source review finds no remaining Critical/Important issue. Evidence review strengthened actual-harness hashing and the normalized-byte/manifest witness. All outcomes and limits are indexed in `Docs/Development/evidence/windows-console-attachment-2026-11-15/README.md`. Required remote CI remains separate; auto-merge must stay off.
