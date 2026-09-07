---
id: TASK-1650
title: 'Lore: a deleted world book keeps its name reserved forever, with no way to see or recover it'
status: To Do
assignee: []
created_date: '2026-07-31'
labels: [roleplay, lore, bug]
dependencies: []
---

## Description (the why)

Deleting a world book removes it from every list, but its **name stays
reserved**. Creating a new book with that name then fails with
"A lore book with that name already exists." — naming a record the user
cannot see, reopen, or restore anywhere in the UI. The only workaround is
to pick a different name forever.

Reproduced deterministically on dev @ 207053253 (G3 user-guide session,
2026-07-31) against a scratch profile:

```
create_world_book("Soft Delete Probe Book")      -> id=1
list_world_books(include_disabled=True)          -> ['Soft Delete Probe Book']
delete_world_book(1)                             -> ok
list_world_books(include_disabled=True)          -> []            # gone from the UI
create_world_book("Soft Delete Probe Book")      -> ConflictError:
        "World book with name 'Soft Delete Probe Book' already exists"
```

So the delete is a soft delete (`WorldBookManager.delete_world_book`) while
the uniqueness check behind `create_world_book` still counts soft-deleted
rows. The list path filters them out, so the two disagree.

Prior art in this codebase: the **prompts** editor hit the same class of
problem and solved the UX half of it with dedicated copy — "A deleted
prompt holds this name — restore it or choose another."
(`LIBRARY_PROMPT_SAVE_STATUS_COPY`, `library_screen.py:324-329`). Lore has
no equivalent message and, unlike prompts, offers no restore path at all.

## Acceptance Criteria (the what)

- [ ] Creating a world book whose name is held only by a deleted record
      either succeeds (name released on delete) or fails with copy that
      says a *deleted* book holds the name and states the way forward.
- [ ] Whichever behavior is chosen is covered by a test that creates,
      deletes, and re-creates under the same name.
- [ ] If names stay reserved, the user has some way to see or purge the
      deleted record (or the message tells them the name is permanently
      taken).
- [ ] Check the same create/delete/re-create path for chat dictionaries
      and characters and file follow-ups if they share the defect.
