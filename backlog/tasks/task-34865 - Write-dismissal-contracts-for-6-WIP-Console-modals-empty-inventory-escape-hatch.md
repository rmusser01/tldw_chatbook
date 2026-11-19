---
id: TASK-34865
title: >-
  Write dismissal contracts for 6 WIP Console modals (empty inventory escape
  hatch)
status: Done
assignee: []
created_date: '2026-09-12 00:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-32475: six WIP Console modals rode the inventory_only_types escape hatch with launch sites but no dismissal contracts. Write real contracts using latest dev branch as reference.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: used latest dev (4a14b3f36f) as reference. Findings: dev already converted both project-instruction modals to SafeModalDismissMixin but still hatches them (no contracts); agent history/progress modals are local-only. Work: (1) ported dev's SafeModalDismissMixin conversion for ProjectInstructionSetupModal/NoticeModal into the local file, keeping DEFAULT_CSS (dev's BUNDLED_CSS requires harvested bundle entries absent from this branch's tcss); (2) contracts: ConsoleAgentProgressModal -> TASK2 (cancel None, inspection-only); ConsoleAgentHistoryModal/ConsoleAppearancePickerModal/WorkspaceArchiveReceiptModal -> TASK3 (success str/ConsoleConversationAppearance/str) with new factories; both PI modals -> TASK567 exceptional contracts (sentinel cancel results cannot ride task-2/3 tables); (3) emptied inventory_only_types, removed the 6 manual root-edge extras (now flow via contracts), kept the ConsoleWorkspaceRenameModal extra with explanation; (4) updated completeness tests (TASK2 15->16, TASK3 13->16). Caught a real crash: the restored appearance picker passed a sync cancel_filter_timer effect to run_cancel_effect_once which awaits it -- every Escape on the picker raised TypeError (present on dev too, uncaught because dev has no contract for the modal); fixed by making the effect async. Verified: 168 passed (122 dismissal + 46 appearance), 49 passed (project-instructions + fleet), headless boot x2 reaches ChatScreen (skip 2.6s, auto-close 7.2s). ADR: not required - test-contract additions + adoption of reviewed dev conversion.
<!-- SECTION:NOTES:END -->
