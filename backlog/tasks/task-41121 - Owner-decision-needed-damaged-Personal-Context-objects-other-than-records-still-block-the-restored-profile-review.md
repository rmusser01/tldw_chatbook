---
id: TASK-41121
title: "Owner decision needed: damaged Personal Context objects other than records still block the restored-profile review"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- personal-context
- backup
- owner-decision
priority: medium
references:
- tldw_chatbook/Personal_Context/repository_keys.py
- tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The owner ruled on 2026-11-14 that damaged records are counted, shown and removed during re-encryption (TASK-40140 AC#5). A damaged object of any other kind (manifest, scope, proposal, local body, outbox body, undo entry, binding) still makes re-encryption refuse, so the restored-profile review can never be confirmed and the profile stays unusable. What to do with those objects is a product decision this task asks for; it does not decide it.

**Evidence.**
- `repository_keys.py:243-250`: `_reads` returns None only for a damaged record ("a record only; any other damaged object still refuses"); `damaged_record_versions` (`:225-241`) counts only `object_type = 'record'`.
- `reencrypt` (`:290-318`) re-encrypts every row of `encrypted_objects`; a non-record that fails authentication raises.
- The panel (`personal_context_panel.py:1003-1024`) shows the mutation's generic failure copy and reloads; it does not name the damaged object or offer a way forward.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision on damaged non-record objects (for example: drop them with a count, keep the profile read-only, or another option) is recorded verbatim in the notes
- [ ] #2 The restored-profile review follows that decision, and when re-encryption is refused for a damaged object, the user is told which kind of object and what they can do instead of the generic failure copy
- [ ] #3 Tests cover each object kind the decision names, red before the change
<!-- AC:END -->
