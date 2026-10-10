# Recovered-image result publication

Task: TASK-34601, OPT92. Status: experiment archived; not adopted after inconclusive timing.

ADR required: no new ADR; bounded amendment to ADR226.
ADR path: backlog/decisions/226-console-polling-and-full-state-reconciliation.md; preserves ADR225/126.
Reason: reuse existing live-poll eligibility and exclusion/replay for disposable image publication. No new cache, scheduler, authority or lifetime owner.

## Evidence

Historical bae Windows original caller chains attribute two native config-operation fallback entries per Send to _load_recovered_images completion. Inclusive entry durations are 194/96/44ms across the full cold/warm/warm phases. These are not unique-file counts, complete refresh counts, pre-provider attribution or removable-time estimates. Receipt ci-bae/windows-config-fallback-origins.json SHA4255b0047790da78dec2ad9a755238ed8abd8f9a310027d0289174bb403005f0.

New message selection causes publication even when the real metadata result is empty. That publication remains required: it releases transient/remote-image suppression. Fresh catalog/payload checks, recovered status handling and native cancellation draining stay unchanged.

## Agreed API and ownership

- Shared preparation owns only UI/Console_Modules/image.py: add optional sync_recovered_images callback, used only by changed recovered lookup completion. Omission falls back to the existing late-bound FULL callback. Generation, editing and other image actions retain FULL.
- Integration owner (root) owns UI/Screens/chat_screen.py and UI/Console_Modules/wiring.py: wire a late-bound _sync_console_recovered_image_ui method. Use original FULL for custom FULL callbacks, overlap, or existing poll_wants_full_sync refusal; otherwise run existing sync_console_poll_display. No await between eligibility selection and entry. Overlap must retain trailing FULL demand even if old image state was already consumed.
- Baseline verification owns new Tests/UI/test_console_recovered_image_publication.py and necessary optional-callback fixture adjustment in Tests/Backup_Recovery/test_recovered_media.py. Root runs all tests and native timing; agents must not launch them.

## Sequence

1. Complete source review and tests before product mutation. Establish RED on original image publication with real metadata and original mounted consumers. Count FULL-only core/roleplay body entries for the exact completion task, preserving their bodies.
2. Integrate both product edits. Verify empty lookup permits transient image fallback, recovered ready/missing/deleted/failed states remain current, and concurrent publication preserves a trailing pass. Retain native cancellation/retirement, maintenance, teardown, custom callback, current owner and terminal settlement controls.
3. Run targeted integrated controls only, source frozen; lint/format changed files and review diff. Do not relax budgets or TTLs.
4. Freeze original a017660377 baseline in the existing comparison worktree. Run full-profile quiet A/B/B/A sequentially with existing source/overlap guards and original budgets. Remove candidate if benefit is not established; retain source and evidence in the optimization ledger.

The separate context-key proposal (replace stock run status/id invalidations with actual held-echo dependency, keeping custom readers conservative) remains a distinct unimplemented lead. It must not contaminate this comparison.


## Qualification and decision

Not adopted. The product callback, screen/wiring helper, candidate-specific tests and optional fixture change were archived under image-publication-rejected-final and restored to a017. The separate maintenance fixture repair remains. No new product route ships from this experiment.

Original metadata/mounted-publication RED reached the intended FULL-only core/roleplay count failure (image-publication-red-1, 27.51s). Final image tests preserve real native reads and isolate their actual completion task after other publishers settle. Removing only the overlap FULL guard fails the exact image-task FULL assertion (image-publication-overlap-negative-1, 19.13s); valid source was restored and verified before further execution.

The initial integrated selection was61 passed/8 failed: three new tests had overlapping unrelated publication, three maintenance fixtures failed, and two feedback budgets missed. The unchanged a017 baseline reproduces the maintenance active-case timeout and Enter Preparing262.906ms miss; its button case passes. This does not establish both initial feedback misses as pre-existing. After test corrections, all16 changed/control cases pass in129.73s (eight image, six maintenance, two feedback); combined with unchanged53 passing cases, all69 selected cases have passing evidence, not a single clean69-case run. Final headless Preparing/input-frame times are53.699/9.550ms Enter and42.295/7.548ms button. Earlier misses remain; this is not a percentile or physical-terminal guarantee.

Quiet sequential ABBA then BA, three saved turns per private full-profile process, provider adapter stubbed:

| Run | Cold seconds | Warm1 seconds | Warm2 seconds |
| --- | ---: | ---: | ---: |
| A1 original |15.694181|9.223584|9.240624|
| B1 candidate |6.348491|4.339475|5.170080|
| B2 candidate |6.150805|4.234366|5.179198|
| A2 original |6.572589|4.790580|4.609786|
| B3 candidate |8.967547|4.618050|4.414562|
| A3 original |9.364590|8.302567|5.044530|

Every run completes three saved replies/traces/links, zero checkpoints, nonstream/nonstream/stream, no detected native overlap and unchanged measured sources/HEAD. The same-arm loaded source manifests match; cross-arm extra raw_participants/storage_admission/server differences are CRLF-only, leaving the intended three product files. The two-second overlap guard is not proof of an idle host. No native census/stack/phase probes were enabled.

Combined warm means6.868612 versus4.659288s suggest32.17%, but that aggregate is strongly influenced by the unexplained A1/A3 slow samples. A2 warm mean4.700183s is effectively tied with B1/B2 combined4.730780s; BA confirmation favors the candidate. The inconsistent result does not establish a repeatable benefit sufficient for this extra route. Preserve all samples and revisit only with stronger attribution/controlled host evidence. The sub-one-second target remains open. Final receipt image-publication-final-comparison.json SHAb62b4ad323aef28b54e04f17f6cb993da22d6fb9ee0ed75c11c8898800fef27f.

Maintenance fixture repair: supply current attachment/control-bar seams, race its held-entry witness against actual task completion to surface errors, and assert current pause/deferred-FULL semantics. No production lifetime change or extended deadline. Ruff/format pass for this retained test file; source diff whitespace check passes. Its final six cases must also be included after image rollback.
