---
id: TASK-2707
title: 'Settings RAG: the Delete button''s state tracks the active profile, not the previewed one'
status: To Do
assignee: []
created_date: '2026-08-01'
labels: [settings, rag, bug, ui]
dependencies: []
---

## Description (the why)

In Settings ▸ RAG the profile picker lets you *browse* profiles without
activating them (the banner says "Previewing '<name>' (read-only) — press
Set active to edit it"). The **Delete** button does not follow that
preview:

- Its label and `disabled` state are computed once, at compose time, from
  the **active** profile — `"Delete — built-in" if info["read_only"] else
  "Delete"` and `delete_button.disabled = bool(info["read_only"])`
  (`settings_screen.py:~10119-10125`), where `info` describes the active
  profile.
- `_sync_library_rag_profile_widgets` re-syncs the read-only banner and
  the editable fields from `info["read_only"]`, but **never touches the
  Delete button**.
- The handler, however, acts on the **selected/previewed** profile:
  `profile_id = self._library_rag_selected_profile_id()`
  (`settings_screen.py:13847`).

So the control and its action disagree whenever you browse away from the
active profile:

1. **Active is a built-in, previewing your own profile** → Delete still
   reads "Delete — built-in" and stays disabled, so you cannot delete
   your own profile without first activating something else. This is the
   likely case on a fresh install, where a built-in is active by default.
2. **Active is your own profile, previewing a built-in** → Delete stays
   enabled and pressing it tries to delete a built-in, failing with
   "Couldn't delete profile: …".

Found while writing the G4 Settings user-guide page (dev @ fb2df0c8a);
the page documents it as a quirk.

## Acceptance Criteria (the what)

- [ ] The Delete button's label and enabled state follow the **previewed**
      profile (the one the handler would act on), re-synced whenever the
      picker selection changes.
- [ ] Deleting your own profile works while a built-in is active.
- [ ] A built-in can never be reached by an enabled Delete.
- [ ] A test covers both directions of the mismatch above.
- [ ] Update the quirk note in `Docs/User_Guide/settings/rag.md` once fixed.
