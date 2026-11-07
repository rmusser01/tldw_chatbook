# Documentation integration gate

Scope: the frozen 13-file `docs-review-package.diff` / `docs-review-current` snapshot, current dev `3c39215749a96d5c1b8c05972fdd2aeb7588b002`, scoped feature `9b0faa`, and the nine `docs-resolution.json` entries. This is a documentation/spec gate, not production/test review. The active writer remains independent.

**Verdict: SPEC changes required; QUALITY changes required for this documentation scope. C=0, I=1, M=1.** No blocking defect was found in the new legacy-compatibility policy itself. I1 is an active-governance numbering inconsistency, not a claim that the implementation uses the wrong migration.

## I1 — ADR-084 still presents workflow window v28 as its active allocation

References: `backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md:326` and `:346`; contrast `backlog/decisions/285-native-phased-agent-workflows.md:1326`–`:1343` and `Docs/Architecture/agent-runtime.md:456`/`:458`.

The resolved ADR-084 calls the workflow window/ref records and snapshot-count refinement “v28”, and says that refinement does not change v25–v27 catalogs. Current release allocation is shipped review-boundary v25, workflow core v26, admission v27, checkpoints v28, windows v29, repair input v30. Thus active ADR-084 assigns the same v28 stage to different contents than the current amendment and living architecture. ADR-285’s header says its amendment supersedes historical numbers “below”; it does not clearly classify the separate ADR-084 paragraphs as historical. This leaves maintainers with two governing allocations for migration/recovery work.

Required resolution: either update these active ADR-084 allocation statements to canonical v29 and the corresponding predecessor range, or explicitly classify the preserved Task8 allocation paragraphs as historical and link their current release mapping to the ADR-285 amendment. Do not rewrite immutable QA/source receipts. No new migration or feature scope is implied by this finding.

## M1 — Reintroduced canonical queue reference points at a missing ADR

Reference: `backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md:274` (the same upstream-existing stale link also remains at `:195`).

The resolved Links section includes `046-visible-bounded-console-prompt-queue.md`, which is absent. The canonical accepted queue decision is `098-visible-bounded-console-prompt-queue.md`, indexed at `backlog/decisions/README.md:92`. The feature additions duplicate this unresolved link while other renumbered ownership links were correctly preserved. Correct the active link/label to ADR-098. This is a navigation problem, not an execution contract change.

## Preserved contracts and satisfactory decisions

- Every original current-dev line is preserved in seven of the nine resolved files. In `agent-runtime.md`, the two replaced lines deliberately identify `Agents/agent_request_execution.py` as the actual closure owner while preserving AgentService composition and survivor semantics. In ADR-084, the five replaced lines deliberately move fallible work outside bookkeeping locks and distinguish immutable snapshot inputs from fresh original worker-local storage/source scopes; they preserve consent CAS, original thread custody and UI separation. No accidental upstream-section loss was identified.
- Upstream accepted trace/review/capture/continuation, prompt-mixin and extracted reply ownership descriptions remain present. `console.md:15` and `:713`–`:721` retain the real `console_agent_reply.py` delegation and existing authority/cancellation owners; the new ADR-285 amendment preserves current request/reply/streaming/pre-provider Stop receivers rather than granting authority to obsolete wrappers. `agent-runtime.md:11`/`:32` name the current request execution owner. Actual implementation of those guards remains for the separate code review.
- Canonical ADR-317 project instructions, ADR-325 workspace defaults, ADR-336 turn lifetime, ADR-344 routing, ADR-345 legacy hooks and ADR-348 fork/spawn references survive the resolutions. ADR-345’s renumbering provenance and accepted legacy behavior are retained. All upstream ADR-index rows remain, including independently shipped ADR-286; the feature adds only its ADR-285 row.
- `database-layer.md:67`/`:76` and ADR-285’s header/amendment correctly state schema30 and canonical stages26–30 after shipped review25. The amendment requires recognition by complete frozen catalog plus matching stamp before constructor DDL; it preserves original SQL/comments and actual column-order lineages, refuses unknown hybrids, uses the existing SQLite owner with FULL BEGIN IMMEDIATE, adds only missing published review-boundary storage, retains prior stamps and validates exact result catalogs/rows. Staged restore uses the same declarations under its original temporary migration transaction; read-only validation is inert and refusal/cancellation rolls back. No table rebuild, profile scan, version-only inference or runtime capability is authorized. Preservation of data, charges, uncertainty, startup audit and explicit Resume is explicit. No unresolved bridge-policy ambiguity was found at this documentation level.
- The diff package contains no historical QA paths. Existing source-specific result qualifications and adverse receipts remain labeled historically; the current integration policy expressly forbids rewriting them. This check does not claim a whole-tree immutable-archive byte audit.
- TASK-38943 remains In Progress with unchecked acceptance criteria covering scoped publication, upstream migration/ownership preservation, clean-source targeted validation, independent review and original checkout/index protection. Its integration plan names the existing ADR-285 amendment before schema work, excludes unrelated local designs, retains shipped25 and staged support bytes, disallows a full sweep and leaves publication/review pending. Scope is appropriate; neither document declares current integration complete.

## Limits

No source, test, Git/index/ref, profile or network changes were made; only this requested report was written. No tests were rerun. Named receiver locations were checked only for documentation ownership references, without reviewing production logic. Old immutable raw QA bodies were excluded. The separate full integration code review and targeted current-source verification remain required.

## Snapshot identities

Package SHA256: `459010a39a145de3041b322ee6183dadcbe35ea2a30f86538852804f9f5a69a5`.

```json
{
  "Docs/Architecture/agent-runtime.md": "308c7fbbea6c871e87605c067d96ec3132a78b63f9c82068320322f298c1447f",
  "Docs/Architecture/console-file-authority.md": "f62ad1c4050d95fc225e3a3124b9d3d2ec4ab76faac62f1bdd2d44352062fdf1",
  "Docs/Architecture/console.md": "62d9c53126f219479c625f1ff92ae393898fd565cc6621ca8a85d63a1319645b",
  "backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md": "33f3db2a486ebf58ef5da56a8e83faaa2630ac50fd6b8f803e25a8b07c1ee0a8",
  "backlog/decisions/163-expanded-console-hook-runtime.md": "ffe81db822e2e85b64c61e9a692846d01532227bc5373c8e7a01444ad7646aaa",
  "backlog/decisions/345-console-run-hooks.md": "6f26fcdd59368816a60b4573e5b2d245901792104c268fe1c922a606d8ce2b5b",
  "backlog/decisions/README.md": "83513a57df4d62053e874a145cb3cde4eae05e49e26efd1376e78053a5c3b055",
  "backlog/docs/lessons-testing-evidence.md": "b4f89e0c92514674030c3fd667a9fec7cca9c59015ce1e8e35079487aba69a8e",
  "backlog/docs/lessons-textual.md": "dfbe181447836e0eb67b7e670fde41a293d8c46785bdb0aa1edd4867c960f914",
  "Docs/Architecture/database-layer.md": "e566b48c4af363c116b95c9ecc3e5a162658420b76ae52f63cb1d651722457b0",
  "backlog/decisions/285-native-phased-agent-workflows.md": "0f2ef6cb0842675c65c3763a698d75f751130bd0eb2f5b48200da324fc1e1e87",
  "Docs/superpowers/plans/2026-11-05-native-workflow-pr-integration.md": "f16b9f67b8b07a6edaf5ee9f4aaeb213061ab037b128b069ba619f11b977c6bc",
  "backlog/tasks/task-38943 - Integrate-native-shared-worktree-workflows-with-current-dev-and-open-PR.md": "97d7110f415268eb39fd568eb986ba65e7eca955d2c2a4dc8302f846ffbf600e"
}
```
