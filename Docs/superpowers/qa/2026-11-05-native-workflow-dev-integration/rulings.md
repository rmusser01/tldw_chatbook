# Integration decisions and trade-offs

These are the twelve decisions recorded in the PR-preparation ledger through the
first draft, runtime integration and current Windows checkout correction. Readiness gates
remain in progress.

1. Prepare the topic tree through a temporary Git index and an ignored source
   directory inside the existing worktree. This preserves the user's branch,
   staging entries and unrelated files. Exported-source tests can accidentally
   borrow the parent checkout, so source origins and exact blob identities need
   verification; the export is not a separate Git checkout.
2. Preserve the shipped AgentRuns review-boundary v25 and allocate workflow
   stages v26–v30. Recognize older local workflow v25–v29 only by exact frozen
   catalogs and stamps, then bridge them transactionally through existing owners.
   Version-only renumbering could corrupt or strand history; a wrong catalog or
   physical column-order declaration could still strand a recognized old profile.
   Constructor, restore, rollback, row-preservation and inertness checks are required.
3. Keep current upstream request/reply/streaming/Stop owners and port workflow
   checks into their actual receiving frames. Restoring older monoliths could
   silently remove delivered behavior or attach proof to an unused wrapper.
   Missing a receiver could break execution or cleanup despite a clean merge,
   so targeted native checks and independent integration review remain required.
4. Correct active ADR084 window/ref allocations to v29, retain predecessor
   catalogs v25–v28 and resolve its two prompt-queue links to ADR098. This aligns
   active governance without rewriting historical QA. An incorrect correction
   would require documentation rework; it changes no runtime authority.
5. Publish the syntax-qualified scoped checkpoint as a draft before final
   behavioral/static verification and independent review finish. The user asked
   for the PR, and the draft makes the actual work visible. Provisional CI may be
   red; that cost does not waive any gate before marking the PR ready. No merge,
   auto-merge, force push or checkout change is authorized by this decision.
6. Preserve the current stream watchdog's Stop behavior and unknown-usage
   retention. A provider worker's physical return does not prove terminal usage
   was consumed. Update the obsolete shutdown-success expectation to an actual
   uncertainty-retention control, alongside a separate successful worker-return
   case. This may retain recovery/disposal conservatively; inventing usage or
   clearing custody would falsely certify settlement. Retained process exit is
   never successful drain evidence.
7. Extend the original wave launch worker's existing no-entry cancellation branch
   to consult the original prepared item's cancellation predicate before calling
   the native child helper. This fixes a known-cancel path inherited from the
   reviewed feature. Original source/storage/execution scopes and exact unused
   worker return remain required. A broader predicate or weaker return proof could
   falsely classify work as unused; predicate errors, races after helper entry,
   changed sources and unproven returns must retain custody.
8. Carry only an ephemeral `review_possible` restriction from the original clean
   draft and automatic review boundary into ordinary admission. Reuse the native
   parser and its closed dependencies, conservatively include existing affirmative
   offer replies, and add the authenticated review family's closed OPAQUE maximum
   only on that path. Keep actual composition in its existing preparation owner;
   the boolean grants neither trust nor review acceptance. False negatives could
   admit uncounted review effects, while false positives reduce concurrency for
   some short replies. Pre-effect source/input checks, authenticated provider
   binding and native physical-return tests remain required.
9. Factor existing prior-command comparison and workflow zero-limit rules into
   shared ledger-owned helpers. Use the read-only preflight before the original
   request callback marks preparation entered, and reuse exactly the same checks
   in the existing writer transaction. This avoids duplicate policy and per-error
   cleanup guesses. Divergent checks could reject valid work early or retain it
   unnecessarily; the writer remains the only admission authority, and all races
   or failures after actual entry retain the original conservative handling.
10. Preserve an already active ordinary child's owner by exact registered-member
    identity, and finish that child only after its original retained source and
    worker-owned database scopes return. Reuse the existing finish flag and
    unchanged finish routine, as the workflow-child path already does. A weaker
    identity check or premature finish could misattribute child work or release
    protection too early; source/unwind failures retain custody and foreign
    owners cannot be adopted.
11. Let ordinary notification/event cleanup leave its original source scope before
    the unchanged durable readback opens and returns its own database scope, then
    finish the original owner. Apply this order to primary and child finish only.
    The observed nested activation retained a newly opened handle despite matching
    durable rows. A failed cleanup exit must skip downstream finish, and uncertain
    readback still retains the claim; warming caches or overriding durability
    would conceal the actual return obligation.
12. Reuse the repository's existing Windows Git long-path bootstrap before the
    initial checkout in the two failing qualification jobs. Both current-head jobs
    failed on the same historical receipt path before tests started, and the main
    test workflow already uses this setting. A wrong bootstrap could leave CI
    blocked; changing raw receipts would instead damage provenance. Historical
    paths and bytes stay intact, and this makes no Windows runtime-support claim.

The existing ADR285 governs these runtime/storage decisions. Its current-dev
integration amendment and Stop clarification were recorded before the respective
implementation or regression changes. Initial failures and separate source epochs
remain evidence of their original outcomes.

13. **Preserve immutable input provenance at completion.** Retain and recheck disjoint read-only input manifests in the original task-evidence/checkpoint owner; wholly write-covered inputs can retain intentional edits. Changed, unavailable or partially overlapping inputs remain unverified through the existing unavailable manifest. Cost: partial directory/alias overlap is conservatively unverified; a completed native task alone does not certify fresh evidence.

14. **Expand references once under original admission.** The extracted plugin helper selects metadata/candidates before admission; the existing reference-read owner expands that exact set and returns offsets from the same read. Preserve literal/manual-rule exclusions, raw invocation source and resumed text/offsets. Cost: an internal preparation return-shape adjustment, requiring connected plugin/reference caller and guard coverage.

15. **Use the existing private SQLite seam for installed workflow catalogs.** Four new memory connections receive one literal module-owned memory-only policy and explicit close, with exact owner-inventory rows. Preserve the raw-call guard; unchanged upstream detached-layout violations stay disclosed. Cost: one restricted ownership entry, with no file/backup/capture grant.

16. **Resolve newly advanced dev through a topic-only rebase.** Dev advanced from3c392 to29592 while the draft gates ran and created conflicts. Preserve the complete reviewed old-base tree/receipts first, compose the new upstream changes without changing the shared checkout/index, and replay the scoped feature onto the new base. Use a lease-protected update of this draft’s owned topic ref; no merge commit or plain force push. Re-review actual conflicts and run affected checks on the new source epoch. Cost: earlier receipts remain qualified to3c rather than being relabeled as new-base proof.

17. **Publish completed ordinary MCP selection as captured.** Set the existing capture-mode field in the original admitted owner’s metadata publication, matching workflow preparation. Preserve strict unresolved-composition validation and all original source/retirement/permission gates. Cost: one explicit state field; incomplete metadata retains OPAQUE treatment rather than authorizing a second selection.

18. **Qualify the real-send UI cadence test through the supported direct-provider fixture.** Original-frame observation proved managed capture rejected the synthetic service graph and the fallback raised at turn_admission189. Reuse the existing private file-backed cited-sources app setup, persist and select console.agent_runtime=false, supply its host to the existing acknowledgement timeline/gateway helper, and verify zero actual agent-loop replies. Opt in the real-send caller; preserve shared memory fixture defaults and native source guards. Cost: this qualifies provider-direct behavior with native storage, not memory/no-bridge construction. No product compatibility redesign or new ADR is implied.

Ruling18 fixture follow-up: the direct fixture’s first attempt also exposed a separate factory-directory mismatch. Its MCP store was created under an invented app directory while the installed source compared against the selected config’s real private data directory, so original binding correctly remained absent. Forward an optional caller-supplied user_data_dir through the existing cited-sources helper and opt this caller into config.get_user_data_dir(), following existing native UI fixtures. Defaults remain unchanged. Assert the original installed binding before sending; do not write a registry or infer permission from path equality in test code. The original constructor/installer performs qualification.

19. **Restore the original legacy-hook handoff and remove a duplicate import.** The scoped name check found unreachable added handoff code using an undefined outcome after an immediate return. Inspect the existing handoff owner, retain the actual hook outcome for normal consumption and complete only its documented no-block/no-context path. Remove the repeated MappingProxyType import. Compare remaining diagnostics against exact upstream source; do not clean up unrelated inherited findings or loosen checks. Cost: one small legacy submission correction requiring real outcome/retirement coverage; no architectural change.

20. **Carry the native MCP permission publication through the original provider-direct turn.** Direct dispatch intentionally has no agent storage source, but its ordinary capture must consume the native publisher’s exact identity update. Retain its existing six-item tuple on the exact live admitted ordinary turn and validate the same turn’s controller, app, store, session, service and active operation before the existing witness update. Preserve original candidate/method/binding/path/init-lock checks and all agent/workflow behavior. Reject foreign, stale or replaced carriers/sources. Cost: one ephemeral slot and one narrow alternative carrier, requiring first cold direct send, follow-up and adverse source/owner controls; no new authority or persisted state. ADR285 records this clarification before implementation.

21. **Test failed required-save retention without assuming shutdown returns.** The exact old adverse assertion expected zero tool workers, while the original host handoff truthfully retained one worker and one execution after a required save failed. Its unconditional shutdown then hid that assertion by waiting for unresolved custody. Run this one deliberate uncertainty case in a private child, verify the retained original claims and pending shutdown, emit an explicit retention marker, and end the child. Parent completion proves only those negative assertions; it never proves native retirement. Positive drain controls stay separate and unchanged. Cost: a bounded process test for an intentionally unresolved owner; no product cleanup relaxation or new ADR.

22. **Qualify the separate hook-footprint read by its actual stage.** Original-frame observation identified the additional full read at ordinary `_capture_maximum -> v2_configuration`, before fresh legacy target selection. The first shared authority read does not supply this whole-hook effect maximum, including legacy hooks. Keep ADR230’s received-read sharing and all fresh target, launch and final-admission checks. Record and assert exact caller stages and durable flags in affected read-sharing controls, preserving external revocation and foreign-owner refusals. Cost: one explicit fresh footprint read; optimizing it requires separate equivalence and freshness proof. No product behavior is changed by this test correction.

Ruling21 fixture follow-up: the first child passed original retained-event/capacity/physical-target assertions but assumed shutdown had already created its legacy cleanup task. Shutdown may stop at an earlier ordinary cleanup boundary. Observe the existing permanent runtime/controller shutdown fences and the still-pending original dispose task instead; retain the same event and capacity assertions and require a nonempty original target set. This proves attempted shutdown plus retained custody without asserting a later stage or cleanup. The initial failed child remains evidence.

Ruling22 revocation follow-up: the fresh effect-maximum read now observes a canonical revocation made after the received read and refuses before durable acceptance. Assert that earlier refusal explicitly. Retain a separate final-admission control by revoking through the existing real custody-acceptance callback after actual durable acceptance, then require provider refusal and recovery. Use original API callbacks and actual persisted acceptance evidence; do not weaken either revocation fence, replace guarded receivers or accept an unspecified error/result alternative. This remains a test-only timing correction.

Ruling22 batch follow-up: the legacy foreign-owner unit supplied a nonexistent session, so native source capture refused before matching and fail-open legacy behavior hid that fixture error. Supply the actual existing session to both calls and keep the required fresh owner read. Pin the early native refusal’s actual initialization-recovery copy. Narrowed hooks require another `v2_configuration` observation because the footprint refresh invalidates the original shared authority; assert that exact refreshed-preparation stage rather than dropping fresh validation. These corrections retain the security outcomes and change no production code.
