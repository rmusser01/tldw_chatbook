---
id: task-1651
title: 'Character editor: validation messages show internal field ids to the user'
status: To Do
assignee: []
created_date: '2026-07-31'
labels: [roleplay, ux, polish]
dependencies: []
---

## Description (the why)

The character editor's validation footer prints raw widget ids. Clear the
Name field and the footer reads:

```
Validation errors:
personas-char-editor-name: required
```

and an oversized avatar reads `personas-char-editor-avatar-status: image
exceeds 5 MB`, a blank greeting `personas-char-editor-greetings-table:
greeting 2 is blank`.

Source (dev @ 207053253): `validate()` returns `(field_id, message, level)`
tuples (`personas_character_editor_widget.py:1143-1173`) and `_run_validation`
renders them as-is — `show_validation(tuple(f"{fid}: {msg}" …))`
(`:1219`) — so the internal id becomes the user-facing label. The
offending row is already outlined in red by the same pass, so the id adds
nothing a user can act on.

Found while writing the G3 user-guide page for this screen; the page had
to describe the messages by their readable half only.

## Acceptance Criteria (the what)

- [ ] Validation lines name the field the way the form labels it (e.g.
      "Name: required", "Avatar: image exceeds 5 MB", "Alternate
      greetings: greeting 2 is blank") — no widget ids in user-visible text.
- [ ] The red row-outline behavior is unchanged.
- [ ] A test pins the rendered text so ids cannot leak back in.
- [ ] Check the persona editor and the screen-side `_validate_character`
      findings for the same leak and fix them together.
