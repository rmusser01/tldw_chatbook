---
id: TASK-399.7
title: B1a Build journaled create save and autosave foundation
status: To Do
assignee: []
created_date: '2026-07-23 14:24'
labels:
  - notes
  - filesystem
  - recovery
dependencies:
  - TASK-399.5
  - TASK-399.6
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build the gated create/save/autosave substrate for verified APFS-backed notes while preserving every version that a future Chatbook write could otherwise displace.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Read/write upgrade acquires exclusive mutation ownership only after draining cooperative shared legacy passes; passive processes then start neither file mutation/reconciliation nor legacy filesystem sync.
- [ ] #2 First writable activation pairs owner-protected notes_recovery.db through an exact durable UUID/generation marker, recovery-first identity commit, projection-side commit, verification, and marker removal; only that exact crash state resumes, while orphaned, missing, corrupt, or mismatched evidence fails closed.
- [ ] #3 Blank create in an existing directory and body save/autosave use fresh hashes, durable intent, verified safety bytes, atomic displaced-target preservation, and complete-last ordering.
- [ ] #4 Frontmatter, BOM, uniform newline/final-newline state, and versioned supported-security metadata manifests/fingerprints round-trip; every operation persists expected/intended metadata baselines for crash classification, and mixed/lone-CR normalization requires hash-bound acknowledgment plus verified prior bytes.
- [ ] #5 Projection commit precedes journal completion; later FTS failure leaves indexing pending, permits later writes, and reports search index updating.
- [ ] #6 A conflict never overwrites observed disk bytes and durably retains the draft and every displaced side.
- [ ] #7 On startup, only the elected owner holding exclusive mutation ownership classifies interrupted create/save operations and cleans exact-owned artifacts after durable byte/metadata capture; passive processes inspect/report only.
- [ ] #8 Debounce, 30-second maximum persistence, editor-generation races, no-op saves, exact draft-export fallback, and capacity reservation covering compressed content plus encoded manifests pass fault tests; manifests over 64 KiB remain read-only.
- [ ] #9 Recovery-only access enumerates, verifies, and exact-exports retained content without opening ChaChaNotes or file_notes.db; Recovery items (N) appears only for genuine retained evidence.
- [ ] #10 Changed this session records only successful current-process working-tree changes; Delete, rename, and move remain unavailable.
- [ ] #11 Unsafe or retained drafts and save failures outrank generic root state, name Retry/Export actions, and veto navigation that would discard the only in-memory copy.
- [ ] #12 The writable mode transition and all create/save/autosave controls remain hidden until the complete B1 release gate supplies conflict resolution, relocation recovery, and Unlink/Forget barriers.
<!-- AC:END -->
