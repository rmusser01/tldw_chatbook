# TASK-40608 Windows admission reuse

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: extend the existing admission authority rules to Windows without changing storage or lock ownership.

1. Amend ADR-126 for ordinary qualified NTFS admission. Keep the existing derivation and use it as the verdict oracle.
2. Exercise an engaged native reuse route and mutations of control, registry, qualification, selector, paths and local ownership. Check cleanup on native observation failure.
3. Reuse twice-derived allowed scope on the same counted hold, with fresh Windows metadata and fresh authority inputs. Preserve current qualification, path/provenance, publication and lease checks.
4. Run three serial baseline/candidate pairs on the same source and with the same PR #3496 overlay. Use a fresh private Windows profile and the same scripted README and approved edit journeys in every run. Alternate run order by pair.
5. Compare each journey's median Send-to-completed-reply interval after subtracting measured visible disclosure and approval waits. Require at least 30% reduction in both journeys. Retain first-paint, first-provider, approval, tool, verification and work-count evidence. Check exact README code, read-only binding, three Once approvals, sole expected main.py diff and all clamp cases.

## Approved follow-up: reduce admission frequency

The first candidate met the README target but missed the edit target. The user approved operation-owned admission and revision-driven Console presentation on 2026-11-12. Continue in the existing isolated worktree, retaining the baseline, fixture and approval semantics.

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: bounded operation custody and detached display data must preserve live authority checks.

Architecture: use existing related-path admission for an exact MCP operation's members, retain its canonical observation during preparation and nested reads, and consume existing checked display snapshots between Console reconciliations. New/later/foreign operations and tool dispatch remain fresh; no lease spans a provider or human wait.

### Task 1: bounded MCP admission

- [x] Establish RED using real-source work-count tests in `Tests/Backup_Recovery/test_raw_related_source_preparation.py`; add canonical preparation reuse/refusal/retirement coverage.
- [x] Modify `Backup_Recovery/raw_participants.py` to group exact ordered members and retain only operation-owned canonical evidence. Preserve same-group, generation, source, process/thread/task, path, pin, maintenance and close-failure checks. No change to `mcp_source_participants.py` was needed.
- [x] Run focused native Windows participant, related-path, coordinator and admission tests. Count whole-operation admissions and native opens, including selection/preflight.

### Task 2: bounded Console presentation refresh

- [x] Extend existing character presentation tests with unchanged-refresh counts and owner/revision invalidation before implementation.
- [x] Use the existing checked display memo and owner/revision checks, with reconciliation cadence measured after a completed native read. Preserve external-change reconciliation, final settled refresh, context switches and custom readers. Display data grants no dispatch authority.
- [x] Run targeted refresh tests and review every changed invalidation path.

### Task 3: integrated verification

- [x] Review both changes and resolve findings; run focused tests and changed-file lint/format checks.
- [x] Freeze source and harness hashes after tests. Reuse the qualified harness. No concurrent tests or edits during timing.
- [x] Run three sequential baseline/candidate pairs, alternating order. Compare both latency medians against 30% and retain admission counts plus all original correctness assertions.
- [x] Update task notes; mark Done only after all acceptance criteria and required checks pass.

Review focus: canonical versus recovery-destination authority; foreign source/task rejection; remapping or ACL mutation during preparation; uncertain native retirement; stale display after owner/revision/external changes.

Execution ledger: the earlier Windows evidence-reuse implementation and its six successful timings remain the first-stage evidence. Storage and Console work have separate implementers; root owns integration and benchmarks. The user has authorized implementation of this follow-up. No integration or commit was requested.

Storage verification: 24 focused native tests passed (`storage-green-1-tests.xml`). Permission read admissions fell 12 to 2 and native opens 3,594 to 2,664; catalog admissions fell 11 to 2 and opens 3,111 to 2,274. These are complete-read counts, not latency promises. Console verification passed 70 native tests (`console-green-1-tests.xml`). Independent read-only review found no actionable storage or Console defects. Changed-region formatting passes; lint comparison records no new findings (15 pre-existing findings remain). Final matched timings use batch `operations` in the existing verification directory.

Final evidence: all six completed runs passed. Median adjusted README latency fell 236.8475 to 84.4768 seconds (64.33%); edit fell 282.0649 to 162.0053 seconds (42.56%). Median admissions fell 2,139 to 1,418 and 2,259 to 1,754 respectively; full scope derivations fell 26,400 to zero. A host restart interrupted the first pair-2 candidate; preserve/exclude that incomplete attempt and rerun it before continuing. Frozen source/harness hashes and completed pair-1 receipts were verified after the restart. This is the combined candidate versus the original baseline, not an isolated estimate for either optimization.

## Reopened: absolute latency requirement (2026-11-13)

The owner rejected the remaining latency as unusable. Percentage improvement is historical evidence, not the final acceptance gate. Both original journeys must complete in under five seconds of application time with the deterministic loopback provider, excluding only measured human disclosure/approval waits. Preserve the original authority, approval and exact file assertions. TASK-40608 is In Progress again with this additional criterion.

ADR required: yes for any change to admission ownership or authority freshness.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: removing repeated work must preserve the existing security and recovery contracts; a wider lifetime cannot be introduced silently.

1. Attribute remaining admissions to their complete caller chains, with native-open and security-query counts, using the frozen candidate and an observation-only diagnostic harness. Keep its results separate from the matched timing series.
2. Identify the smallest architectural boundary that removes repeated work. Distinguish display-only observations, bounded local operations and actual authority/effect boundaries; preserve fresh checks at the latter.
3. Record the chosen design and its invalidation/retirement rules in ADR-126 before implementation. Add causal work-count and refusal regressions, then make and review the focused change.
4. Qualify a single candidate journey against the absolute budget before committing time to another three-pair series. Repeat the unchanged sequential baseline/candidate protocol only when that candidate meets the budget; never claim sub-five-second completion from call counts alone.

Diagnostic outcome: the unchanged candidate passed both journeys with additional observation-only call attribution. Each labelled diagnostic window includes post-reply verification, and instrumentation increases periodic work; these are not acceptance timings. README/edit windows contained 2,050/2,098 admissions, all reuse hits, and 607,678/628,569 successful native opens. The largest identified display roots are readiness config, character scope/recents, context-memory probing and workspace availability. Their native callbacks and repeated SQLite opens create a duration-dependent polling workload. Generic `base_db.invoke` frames need submitted-callable attribution before choosing additional command batches.

Two independent read-only design checks recommend extending existing `config_sync._Cadence` and section-specific revision keys, separating native memory-presence facts from streaming projection inputs, and batching consecutive local DB calls under existing finite ownership. Keep bounded external reconciliation and all native action checks. Exact same-participant ancestral reuse can support concrete multi-DB batches; persistent descriptor-only path evidence and sharing primary-only leases across DB members are not valid shortcuts. No additional product changes have been made, and five-second acceptance remains open. See the chat verification directory's `absolute-target-diagnosis.md` and `after-callers-1/caller-summary.json` for evidence and proposed next work.
