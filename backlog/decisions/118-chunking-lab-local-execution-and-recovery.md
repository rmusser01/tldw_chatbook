# ADR-118: Chunking Lab local execution, authoring, and recovery

- Status: Accepted and implemented; final targeted acceptance and independent follow-up review complete
- Date: 2026-09-04
- Spec: [Chunking Lab design](../../Docs/superpowers/specs/2026-09-04-chunking-lab-design.md)
- Plan: [Chunking Lab implementation](../../Docs/superpowers/plans/2026-09-04-chunking-lab.md)
- Related: ADR-003, ADR-029, ADR-031 (keybindings), ADR-073, ADR-076 (Library discovery), ADR-078
- Allocation evidence: fetched remotes; swept 320 local/remote refs and 61 worktrees
  on 2026-09-04; highest existing ADR 117, task 31420, no scan errors. Recheck at merge.

## Context

Users need to try one chunking recipe, pin its result, compare a second recipe,
and save either as a reusable template. They explicitly require full configuration
execution, lossless advanced editing, local execution/saving, and automatic recovery
of sample text and completed results after a crash.

The initial checkout predates ADR-078's completed convergence. Inspection of dev at
`91757b61e9c7e9f920d80a0ce282261b4161ffff` found the canonical flat body, versioned
Media DB records, validate-on-write, and `Chunking/template_runtime.py`. Planning
must build on those contracts, not revive the deleted file store or pipeline model.
TASK-24404's unimplemented Settings form overlaps this work and conflicts with
ADR-003 ownership; its placement proposal is superseded by this decision.

## Decision

1. **Library owns Chunking Lab.** A dedicated lazy-loaded screen is reachable from
   Library, local-item extracted text, and the command palette. It returns to its
   opener. No global destination, Settings editor, or inert Evals action is added.
   Reconcile TASK-24404 before landing the UI; retain its useful validation and
   ingest-picker-refresh requirements in the Lab work.
2. **Adopt ADR-078 unchanged as the canonical template shape/store.** Author the
   flat pre/chunk/post body, classifier, and metadata; record fields contain
   name/description/tags. Preserve arbitrary raw drafts, but do not execute or save
   unsupported legacy shapes, inheritance, or unknown executable extensions.
   Existing saved records and imports become detached drafts, never a second store.
3. **Separate preservation, server validation, and preview capability.** Keep the
   parity validator's contract intact. Add a headless local Lab preflight, used by
   Run and Lab Save, which rejects ignored operations/options, unavailable assets,
   network-capable methods, and unsupported structured-output combinations. Unknown
   metadata remains intact. Classifier rules are selection metadata, not an implicit
   classifier run during direct preview. This is not new server-parity behavior.
4. **Reuse one runtime seam and the vendored algorithms.** Add structured reporting
   outside `Chunking/engine/`; preserve the existing saved-apply list adapter.
   Carry available metadata and verified coordinate spaces. Never invent source
   alignment or a quality score. Refuse a combination if the pinned processor
   cannot execute it faithfully; vendor fixes require the existing sync workflow.
5. **One bounded local preview process, immutable run inputs.** v1 has at most two
   stable candidate IDs. Persist a Run both manifest before starting A then B;
   neither can read newer drafts or template records. Record backend, execution
   versions, exact body/defaults/assets, sample hash, and terminal outcome. No
   automatic network access, model downloads, or LLM invocation. Cancel terminates
   active work and the queue; epoch guards reject late replies.
6. **Dedicated profile-local SQLite recovery storage.** Use the private-path/SQLite
   owner registry (ADR-029), with a versioned schema independent of the Media DB.
   A serialized writer transactionally publishes immutable results with checkpoint
   references. Keep current/previous checkpoints and active undo references. Use
   cross-instance compare-and-swap plus profile/epoch/revision acknowledgments.
   Autosave includes invalid JSON and incomplete controls, not merely parsed models.
7. **Recovery is automatic; reusable-template saving is explicit.** Reopen restores
   the last durable state without execution. Writes failing leave the draft/result
   in memory and expose Retry/Export. Versioned bounded JSON export has a paired
   validated restore, atomic replacement, and one-level Undo restore. Clear cancels
   work, removes all recovery references, and prevents late resurrection. Neither
   deletion nor POSIX private storage is an encryption/secure-erasure promise.
8. **Extend existing template updates with atomic expected-version checks.** Lab
   updates carry ID/UUID/version, recheck builtin/live state in the same transaction,
   and retain the draft on conflict. Use existing live-name uniqueness for creates.
   No Media DB schema migration is required. A successful save refreshes the ingest
   picker but never re-chunks source content or changes defaults.

## Limits and evolution

Initial v1 limits: 2 MiB UTF-8 sample, 10,000 chunks, 32 MiB serialized result,
60 seconds per preview; test resource behavior, including intermediate allocations.
Lab preflight additionally admits at most 16 pre/post operation entries combined
and 2 MiB for each canonical authored/effective recipe document. Sample-dependent
admission estimates at most 32 MiB of intermediate working payload, including
bounded section-capture amplification. This estimate is not a process RSS cap:
Python objects and intermediate copies can consume substantially more memory.
Record measured child-lifetime peaks and actual successful OS limits separately;
in particular, macOS address-space enforcement cannot be assumed. The fresh
subprocess uses bounded framed JSON and explicit stderr DEVNULL, without changing
the application's global streams. It is not a security sandbox.
Autosave targets a 300 ms trailing debounce and at most one second between normal
continuous-edit checkpoints. These are engineering defaults, not benchmark claims
or guarantees of zero-keystroke loss. Display Saved locally only for the latest
committed revision.

v2 may expand the candidate collection to 3+. v3 assigns corpus evaluation,
judgments, aggregate metrics, scheduling, and history to Evals; immutable snapshots
form the future handoff. Neither release is implemented by this ADR.

## Alternatives rejected

- Settings-hosted creation (TASK-24404): violates Library ownership and mixes
  experiment/sample state into settings. Keep its useful save/cache requirements.
- Rebuild the legacy pipeline/inheritance system: contradicts completed ADR-078.
- Tighten the global parity validator: silently changes an existing service contract.
  A clearly named Lab capability gate supplies the stricter promise where needed.
- Method/options-only preview: cannot honor advanced pre/post operations.
- Manual-only recovery or repeated whole-session JSON autosaves: fails automatic
  crash recovery or rewrites large immutable outputs on each small edit.
- Thread-only cancellation: cannot stop non-cooperative regex/CPU execution.
- Immediate server adapters or corpus scheduler: unnecessary for the accepted v1.

## Consequences and verification

This adds one private recovery DB and one process lifecycle, with migration,
content-retention, and platform-test responsibilities. Restores and exports contain
full source text; explain that locally and never log it. Larger inputs must be
explicit excerpts, not silent truncations.

Capabilities may initially expose fewer operations than the engine advertises;
the reason must be visible. Exact metadata/alignment is either test-backed or
unavailable, never guessed. Preserve global server-validator tests and vendor sync
checks while adding real execution fixtures, temporary-SQLite crash/concurrency
tests, and Textual keyboard/resizing tests under an isolated profile.

Implementation records: TASK-31421–TASK-31428 (created in dependency order; see plan).
TASK-24404 was archived as superseded. The final correction wave directly implements
this ADR: exact in-transaction catalog acknowledgments, validated replacement
inspection, readable historical evidence, explicit Previous selection, and lossless
raw tags. It introduces no store, schema, runtime, or policy owner.
Scoped independent re-review accepted all twelve corrections. The user-requested
follow-up reconciled runtime admission with ADR-078 without widening guards, and
corrected only the local Lab fixtures' manual initial-screen ownership. Final
targeted verification passed473 tests; independent follow-up review found no issues.
TASK-31428 is complete. The original
non-green integration evidence and platform/privacy limits remain in
[Chunking Lab verification](../../Docs/Chunking_Lab_Verification.md).
