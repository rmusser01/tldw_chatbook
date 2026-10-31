# Task 6 integration report

## Result and revision boundaries

Final source HEAD: `ec3d10170103dc0f4817f848ee66646410a27f31`. It descends from pinned dev `67fc5310471823262eb6ce344a539e699784bed8`. Source and qualification are halted for root's metadata checkpoint and independent integration review. Current-head external CI and publication remain pending; no push or merge was performed.

The latest rebase replayed the existing 25-commit chain from ca2992 onto67fc531 after clean root checkpoint `dc4df3dc80f767de7c94a833b0402dd880af235c`, yielding `9ce98df2e7cda8b444b3bed90f27c1d382677871`. Recovery refs and mode0600 bundles remain intact. Only the ADR index and Backlog lessons conflicted: both upstream and owned additions were retained, including upstream212/217 and our mechanical219 rename. Source/test paths merged without conflicts. [Recovery](task-6-latestdev-recovery.json), [conflict decisions](task-6-latestdev-conflicts.json), [mapping and preservation](task-6-latestdev-final-preservation.json).

Earlier reviewed BASE `b265d5bd2c2f5a28e4f56c2115f6247988d29ac2` and source `e7cc5337617781e9a4b08b9e324297b537fcad07` were integrated onto ca2992, followed by the independently scoped Task6 corrections. [Prelatest report](task-6-prelatestdev-report.md) contains that full chronology, RED/GREEN receipts and original21-commit mapping; none of its historical receipts was rewritten. Receipt `dev` means the observed shared ref, not tested ancestry. In particular52a integratesca;9ce integrates67fc, even after the shared ref moved tof1f808 during final diagnostics. No f1f808 ancestry is claimed.

## Preservation and corrective scope

The final proof retains104 upstream non-overlap blobs,11,572 historical QA blobs and the25 old/new commit mapping. All production source remains byte-identical to the prelatest checkpoint plus the approved upstream delta. Of164 owned Python files, only the named navigation fixture differs from that checkpoint; its original assertions and all other functions are preserved. Initial integration proof separately distinguishes1,878 owned PR QA artifacts from27 upstream-only QA scripts intentionally reformatted on dev.

PR2999 round-robin UI sharding and fail-closed aggregation, ADR212 BaseAppScreen opt-in Tab/arrival behavior and Library aliases, and the34-path docs-only upstream merge remain intact. [Budget and document proof](task-6-latestdev-budget-and-doc-proof.json) retains all latest-dev thresholds; the pre-import limit557 is upstream's ADR097 exception. The stronger reviewed `console_compaction_failure` absence assertion stays unchanged. An initial whole-file budget-proof assumption failed because of that assertion; the diagnostic remains recorded rather than weakening the test.

Earlier source corrections remain within ADR092 fork-transition admission, ADR069 project controls, ADR094 physical lifetime and ADR219 bounded starts: six canonical mutation boundaries, exact display-plan custody through physical drain, typed constructor-owned fresh controls, and narrow census DTO/delegate/branch-owner modeling. ADR097 and upstream212 govern this final integration. No new storage, schema, routing, charging or authority policy was introduced. [Boundary proof](task-6-boundary-preservation.json), [ADR219 mechanical proof](task-6-adr-renumber-proof.json).

The navigation fixture first reproduced `raw_source_selection_changed` on integrated and immutable pinned-dev source. Existing private-profile enrollment exposes a second real assertion failure: the typed `half` draft is empty after public Library→Console navigation. Isolated9ce diagnostics prove two mounted content Consoles; the top/runtime-owned composer contains `half`, but navigation correctly selects stack[1] and serializes the older empty composer. Claiming manual initial-screen ownership before run_test repairs that harness invariant; added preconditions verify one content Console, current runtime owner and actual typed text. All original navigation/session/draft assertions remain. [Ownership evidence](task-6-latestdev-navigation-ownership-proof.json), [fixture AST proof](task-6-latestdev-navigation-fixture-final-proof.json).

## Latest targeted qualification

Each receipt links exact argv, cwd, hashes, exit and corresponding log/XML. Labels containing `green` with a nonzero exit remain failures.

| Receipt | Exit | Result |
|---|---:|---|
| [CI owners](task-6-latestdev-ci.json) | 0 | 34 passed; sharding and aggregate contracts |
| [UI census checker](task-6-latestdev-census.json) | 0 |126 files, floor125 |
| [BaseAppScreen owners](task-6-latestdev-baseapp.json) | 0 |43 passed; one FD warning retained |
| [Startup ratchets](task-6-latestdev-startup.json) | 0 |25 passed; three budget warnings retained |
| [Console startup/navigation](task-6-latestdev-console-navigation.json) | 1 |Real Console launch passed; navigation failed config admission |
| [Immutable dev navigation](task-6-latestdev-navigation-baseline.json) | 1 |Same preconstruction admission failure |
| [Helper-only navigation](task-6-latestdev-navigation-green.json) | 1 |Reached original empty-draft assertion |
| [Immutable dev helper overlay](task-6-latestdev-navigation-draft-baseline.json) | 1 |Earlier mount `NoMatches #console-left-rail`; not evidence of draft behavior |
| [Ownership diagnostic](task-6-latestdev-navigation-owner-values.json) | 1 |Duplicate screen measured; original draft assertion fails |
| [Save-owner diagnostic](task-6-latestdev-navigation-save-owner.json) | 1 |Actual wrong harness content owner serialized; original assertion fails |
| [Final navigation](task-6-latestdev-navigation-final-green.json) | 0 |1 passed; original public-navigation assertions and four ownership preconditions |
| [Fatal Ruff](task-6-latestdev-navigation-final-fatal.json), [formatter](task-6-latestdev-navigation-final-format.json), [source whitespace](task-6-latestdev-navigation-final-whitespace.json) |0|All pass |

Carried qualification is bound by exact source identity, not ref labels: native5, CRUD3, queue75, egress112,315 unaffected provider/Anthropic cases, repaired provider plus security controls10, ten Anthropic UI cases, profile59 with two existing platform skips; corrective boundaries/caller/census81, extra validation/lock-order5 and final adversarial census43. No completed broad runtime batch was repeated. The prelatest report links all exact commands, genuine REDs, setup-diagnostic failures and final static/derived checks.

Warnings: imports681/686, UI ready1033/1033, preimport557/557; LOC415329/425347 and largest route127522/135111. FD growth206(start14,end220,limit200) occurred in the source-identical mounted BaseAppScreen owner. The nested host fixture constructs a separate TldwCli app without running its lifetime; it is a plausible owner, not a proven descriptor attribution. All43 assertions passed. No warning filter/threshold or production leak repair was added. [FD disposition](task-6-latestdev-fd-disposition.md).

## Self-review and handoff

Review the two document conflict unions, immutable source/QA mapping, six canonical boundaries and unchanged nested preparation/promotion order, cancellation/sibling/pre-start physical custody, typed initial project controls, narrow fail-closed census modeling, and the single navigation fixture repair. Reviewers can use existing receipts without repeating closed owners. External current-head CI remains mandatory for the historical native timeout. Root identified subsequent dev f1f808 as PR2989’s11-path Library reader freshness/Console delete-undo/census/diagnostic delta. That next integration remains pending the root metadata checkpoint; this report makes no claim about its qualification. Root must also recheck ADR219 allocation before publication.

No full sweep, install, admission reset, new skip/xfail, guard waiver, feature budget increase or Git housekeeping occurred. Private profiles/config/DBs remain outside retained report artifacts; only child test log/XML evidence is copied with hashes. Root's plan metadata remains separate from the test-source commit.
