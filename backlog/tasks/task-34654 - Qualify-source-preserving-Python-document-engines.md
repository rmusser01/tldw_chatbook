---
id: TASK-34654
title: Qualify source-preserving Python document engines
status: In Progress
created_date: 2026-10-07 03:09
dependencies:
- TASK-34565
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editing.md
updated_date: 2026-10-07 12:08
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide local strict YAML, JSON and JSONL checks and reversible formatting without changing document meaning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Supported shared fixtures agree on strict validity/categories and scalar-offset diagnostics; malformed and unqualified inputs never produce success.
- [x] #2 JSON and JSONL formatting preserves every numeric/string token and key order, detects decoded duplicate keys, preserves records/newlines and is idempotent.
- [x] #3 Qualified YAML formatting preserves comments, directives, markers, anchors, aliases, merges, scalar tokens/styles and semantic graph for nontrivial streams; unsupported constructs are explicitly refused.
- [x] #4 Depth/source limits and source-safe error normalization are tested using real parsers plus meaningful generated JSON cases.
- [x] #5 Candidate version/license, shared fixture hash, supported/refused constructs and startup observations are recorded before dependency adoption; targeted tests and static/security checks pass.
- [x] #6 Relevant governing ADR is assessed/linked and repo-local completion notes name actual evidence and remaining limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md (existing programme decision)
Reason: qualify source-preserving local engines and dependency/runtime choices under the approved language contract; amend the existing ADR before accepting a parser dependency, without a new UI/save/runtime owner.
1. Read task8 brief/binding constraints/context, applicable instructions and relevant testing lessons, ADR194, actual E1 request-aware models and byte-identical shared fixtures. Trace existing/native facilities before choosing a YAML candidate; capture immutable base b7141cb0d381fbce4cb4a9f09b723beba5e5837d.
2. Add meaningful failing real-engine JSON/JSONL, YAML graph/source preservation and Unicode/CRLF/EOF controls, including duplicates and resource boundaries.
3. Implement strict stdlib JSON validation with bounded source token positions and token-preserving whitespace formatting; per-physical-record JSONL preserves records, newline convention and final terminator facts.
4. Qualify the candidate YAML event/token and source facilities without application constructors or alias expansion. Require successful nontrivial comments/multi-document/anchor/block-scalar formatting plus explicit unsupported refusals; preserve version semantics, scalar tokens and graph. Record version/license/hash/startup/memory before packaging adoption and update existing ADR for an accepted tooling choice.
5. Run the four explicit contract/JSON/YAML/position files with meaningful generated JSON cases, scoped Ruff/Black/Bandit and immutable-base no-introduced checks. No full sweep, UI integration, scheduler or shared environment rebind.
6. Write durable qualification report, concise task notes and full task-8-report.md with exact RED/GREEN/final commands/output, candidate and resource evidence, supported/refused constructs and limitations. Self-review and commit explicit unit paths; leave In Progress for independent controller review.
Fix round1 (task-8-review.md I1): correct the shared YAML1.2 Core integer tag matcher so signs apply only to decimal spellings; retain distinct native1.1 resolution. Add meaningful signed hex/octal distinct-versus-quoted-duplicate and explicit1.1 positive controls, then run whole YAML test file plus scoped static/security/format/diff checks against actual reviewed4f42519c8ee85b6b53a70d4f29f19b666084de96. Append captured RED/GREEN/final evidence, update relevant durable qualification evidence, self-review and commit explicit paths. Existing ADR194 applies to this contract correction, no new boundary/dependency; remain In Progress for scoped review.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the pure JSON/JSONL and YAML document engines with E1 request-aware contracts; source tokens, comments and symbolic YAML graph are verified before edits. Qualified and pinned ruamel.yaml 0.19.1 MIT event/token APIs after source/graph/fixture/license experiments; generic round-trip dumping is rejected. ADR-194 amendment and durable evidence: Docs/superpowers/reviews/2026-09-27-document-engine-python.md. Four targeted files pass 120 tests including generated JSON and actual YAML positives/refusals; scoped Ruff, Black and Bandit pass. Source/depth/diagnostic limits and cooperative deadlines are tested. No hard termination or heap cap is claimed: dense fixtures reached about 190 MB RSS and deadline cleanup about 2.02 seconds; E4 remains the process owner. Unknown tags/mixed versions/complex or alias keys/indentless sequences/token-changing multiline layouts explicitly decline. No UI/save/network/filesystem changes; fixtures unchanged. Self-review found and fixed tag/merge admission, candidate-error normalization, comment invariants, EOF indentation and line clamping; testing lesson recorded. Status remains In Progress pending controller review.
Fix round1 / I1: narrowed optional signs to the decimal branch of the YAML 1.2 Core integer matcher, preserving the separate 1.1 resolver. Added both-sign/both-base default/explicit1.2 distinct-key and quoted-duplicate controls, plus explicit1.1 signed hex/legacy-octal compatibility cases. RED: 16 failed/56 passed; whole YAML covering GREEN: 72 passed in 0.14s, exit 0. Changed-file Ruff/Black/Bandit and reviewed-base diff checks pass. Durable qualification correction recorded; unchanged JSON/position/contracts not rerun. Existing ADR194 applies; no new dependency or boundary. Self-reviewed one-line root-cause fix; remains In Progress for fresh review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
