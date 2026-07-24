---
id: TASK-221
title: Carry images in Chatbook export
status: Done
assignee:
  - '@claude'
created_date: '2026-07-13 09:30'
updated_date: '2026-07-16 15:44'
labels:
  - chatbooks
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Save Chatbook currently omits message images: a Chatbook exported from a Console conversation with image messages (PR #621) loses them. Extend the Chatbook schema/packaging to carry image attachments and restore them on import (adjacent to task-19's attachment-availability work).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exporting a conversation with image messages includes the image bytes in the Chatbook
- [x] #2 Importing that Chatbook restores messages with working chips and Save Image
- [x] #3 Chatbook schema/version bump documented
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Export: _export_message_attachments writes position 0 (legacy image_data/image_mime_type columns) + positions >=1 (one batched get_attachments_for_messages call per conversation) as files under content/conversations/attachments/<message_id>-<position><ext> (mime->ext map + mimetypes fallback to .bin), and adds an attachments list (position/file/mime_type/display_name) to each message's JSON. Import: _load_message_attachments restores position 0 via add_message's image kwargs and positions >=1 via set_message_attachments — the app's live read contract; resolved-path zip-slip guard skips entries escaping the extraction root (warning, message still imports); missing files warn+skip; pre-TASK-221 chatbooks (no attachments key) import unchanged. RED-first round-trip tests over real SQLite (Tests/Chatbooks/test_chatbook_image_round_trip.py, 4): export file+manifest shape with byte equality, both-tier import restoration, traversal-tampered chatbook skips only the hostile entry, backward-compat import. Note: fixture initially exposed that add_conversation reads 'title' (not 'conversation_name') — test fixture corrected, no production title bug. Sweep 300/1 skip across Chatbooks+store+DB. Files: Chatbooks/chatbook_creator.py, Chatbooks/chatbook_importer.py, Tests/Chatbooks/test_chatbook_image_round_trip.py.
<!-- SECTION:NOTES:END -->
