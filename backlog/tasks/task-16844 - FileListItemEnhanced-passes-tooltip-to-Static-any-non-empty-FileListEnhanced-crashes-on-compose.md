---
id: TASK-16844
title: 'FileListItemEnhanced passes tooltip= to Static: any non-empty FileListEnhanced crashes on compose'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - bug
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found live during the TASK-15771 review (PR #1699) and still present at dev `ee741cf10`:
`Widgets/file_list_item_enhanced.py:123-127` yields
`Static(self._metadata["name"], classes="file-name", tooltip=str(self.file_path))`, but
Textual 8.2.8's `Static.__init__` takes only
`content, *, expand, shrink, markup, name, id, classes, disabled` — **no `tooltip`
parameter**. The review reproduced it deterministically: mounting `FileListEnhanced` and
setting a one-element `files` list raises

```
TypeError: FileListItemEnhanced(id='file-item-...') compose() method returned an invalid
result; Static.__init__() got an unexpected keyword argument 'tooltip'
```

Any non-empty `FileListEnhanced` hits it — `files` is a `recompose=True` reactive, so the
first real row triggers the crash (the irony noted by the review: it is one of the four
recompose sites 15771 fixed, and the one that cannot be exercised). That no test caught
it says the widget currently mounts nowhere with data in the tested surface — so
establish reachability first: if something real feeds it, fix the tooltip (set the
`.tooltip` attribute after construction, or on the row widget) and pin with a born-red
non-empty-list test; if nothing does, this is a wire-or-retire candidate rather than a
one-line patch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Reachability is established and stated: what (if anything) mounts `FileListEnhanced` with a non-empty list in the live app
- [ ] #2 A mounted `FileListEnhanced` with at least one file composes without error, and the intended tooltip behavior actually works (test born-red against the current code)
- [ ] #3 If the widget is dead, it is retired with reachability evidence instead of patched
<!-- AC:END -->
