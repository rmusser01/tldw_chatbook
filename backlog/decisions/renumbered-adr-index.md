# Renumbered ADR index

Maps each ADR number that was once shared by several files (TASK-36943.3, 2026-11-03) to the files that held it and the number each holds now. The oldest file (Date field, else add-commit time) kept the number; every younger file took the next free number. See the Numbering rule in [README.md](README.md). Older references to a shared number mean the keeper unless the context names a renamed file.

| Old number | File | Now |
| --- | --- | --- |
| ADR-004 | [004-settings-storage-defaults-restart-boundary.md](004-settings-storage-defaults-restart-boundary.md) | ADR-004 (keeper) |
| ADR-004 | [287-personas-destination-native-workbench.md](287-personas-destination-native-workbench.md) (was `004-personas-destination-native-workbench.md`) | ADR-287 |
| ADR-005 | [005-console-workspace-server-readiness.md](005-console-workspace-server-readiness.md) | ADR-005 (keeper) |
| ADR-005 | [288-invest-in-local-rag-mirroring-tldw-server.md](288-invest-in-local-rag-mirroring-tldw-server.md) (was `005-invest-in-local-rag-mirroring-tldw-server.md`) | ADR-288 |
| ADR-013 | [013-library-ingest-ownership-and-job-lifecycle.md](013-library-ingest-ownership-and-job-lifecycle.md) | ADR-013 (keeper) |
| ADR-013 | [289-media-search-plain-text-fts-boundary.md](289-media-search-plain-text-fts-boundary.md) (was `013-media-search-plain-text-fts-boundary.md`) | ADR-289 |
| ADR-014 | [014-library-ingest-service-authority-and-recovery.md](014-library-ingest-service-authority-and-recovery.md) | ADR-014 (keeper) |
| ADR-014 | [290-retire-legacy-navigation-chrome.md](290-retire-legacy-navigation-chrome.md) (was `014-retire-legacy-navigation-chrome.md`) | ADR-290 |
| ADR-018 | [018-watchlists-tui-screen.md](018-watchlists-tui-screen.md) | ADR-018 (keeper) |
| ADR-018 | [291-local-server-hybrid-scheduled-tasks.md](291-local-server-hybrid-scheduled-tasks.md) (was `018-local-server-hybrid-scheduled-tasks.md`) | ADR-291 |
| ADR-026 | [026-retire-chat-tab-conversation-entry-chain.md](026-retire-chat-tab-conversation-entry-chain.md) | ADR-026 (keeper) |
| ADR-026 | [292-brand-asset-source-and-export-boundary.md](292-brand-asset-source-and-export-boundary.md) (was `026-brand-asset-source-and-export-boundary.md`) | ADR-292 |
| ADR-027 | [027-default-workspace-chats-in-chats-section.md](027-default-workspace-chats-in-chats-section.md) | ADR-027 (keeper) |
| ADR-027 | [293-portable-database-note-session-coordinator.md](293-portable-database-note-session-coordinator.md) (was `027-portable-database-note-session-coordinator.md`) | ADR-293 |
| ADR-028 | [028-settings-workspaces-category-and-folder-roots.md](028-settings-workspaces-category-and-folder-roots.md) | ADR-028 (keeper) |
| ADR-028 | [294-character-tts-generation-profile-ownership.md](294-character-tts-generation-profile-ownership.md) (was `028-character-tts-generation-profile-ownership.md`) | ADR-294 |
| ADR-028 | [295-structured-prompts-and-auxiliary-improvement-calls.md](295-structured-prompts-and-auxiliary-improvement-calls.md) (was `028-structured-prompts-and-auxiliary-improvement-calls.md`) | ADR-295 |
| ADR-029 | [029-local-private-data-boundary.md](029-local-private-data-boundary.md) | ADR-029 (keeper) |
| ADR-029 | [296-file-notes-disk-authority.md](296-file-notes-disk-authority.md) (was `029-file-notes-disk-authority.md`) | ADR-296 |
| ADR-029 | [297-versioned-prompt-artifacts-and-safe-improvement-transactions.md](297-versioned-prompt-artifacts-and-safe-improvement-transactions.md) (was `029-versioned-prompt-artifacts-and-safe-improvement-transactions.md`) | ADR-297 |
| ADR-030 | [030-derived-index-lifecycle-and-atomic-media-migrations.md](030-derived-index-lifecycle-and-atomic-media-migrations.md) | ADR-030 (keeper) |
| ADR-030 | [298-local-library-agent-tool-boundary.md](298-local-library-agent-tool-boundary.md) (was `030-local-library-agent-tool-boundary.md`) | ADR-298 |
| ADR-031 | [031-bounded-evaluation-and-tool-worker-execution.md](031-bounded-evaluation-and-tool-worker-execution.md) | ADR-031 (keeper) |
| ADR-031 | [299-tui-keybinding-and-footer-hint-conventions.md](299-tui-keybinding-and-footer-hint-conventions.md) (was `031-tui-keybinding-and-footer-hint-conventions.md`) | ADR-299 |
| ADR-032 | [032-immutable-installed-distribution-assets.md](032-immutable-installed-distribution-assets.md) | ADR-032 (keeper) |
| ADR-032 | [300-local-agent-tool-permission-boundary.md](300-local-agent-tool-permission-boundary.md) (was `032-local-agent-tool-permission-boundary.md`) | ADR-300 |
| ADR-032 | [301-textual-diff-view-for-tool-call-diff-rendering.md](301-textual-diff-view-for-tool-call-diff-rendering.md) (was `032-textual-diff-view-for-tool-call-diff-rendering.md`) | ADR-301 |
| ADR-033 | [033-application-session-state-ownership.md](033-application-session-state-ownership.md) | ADR-033 (keeper) |
| ADR-033 | [302-local-agent-process-execution-boundary.md](302-local-agent-process-execution-boundary.md) (was `033-local-agent-process-execution-boundary.md`) | ADR-302 |
| ADR-033 | [303-settings-commit-models-three-honestly-labeled.md](303-settings-commit-models-three-honestly-labeled.md) (was `033-settings-commit-models-three-honestly-labeled.md`) | ADR-303 |
| ADR-039 | [039-file-notes-guarded-session-push.md](039-file-notes-guarded-session-push.md) | ADR-039 (keeper) |
| ADR-039 | [304-global-and-studio-tts-settings-ownership.md](304-global-and-studio-tts-settings-ownership.md) (was `039-global-and-studio-tts-settings-ownership.md`) | ADR-304 |
| ADR-040 | [040-profile-owned-state-and-shared-asset-paths.md](040-profile-owned-state-and-shared-asset-paths.md) | ADR-040 (keeper) |
| ADR-040 | [305-versioned-prompt-artifacts-and-safe-improvement-transactions.md](305-versioned-prompt-artifacts-and-safe-improvement-transactions.md) (was `040-versioned-prompt-artifacts-and-safe-improvement-transactions.md`) | ADR-305 |
| ADR-040 | [306-speech-lab-current-result-and-auto-play.md](306-speech-lab-current-result-and-auto-play.md) (was `040-speech-lab-current-result-and-auto-play.md`) | ADR-306 |
| ADR-043 | [043-console-rail-compact-collapse-yields-to-explicit-toggle.md](043-console-rail-compact-collapse-yields-to-explicit-toggle.md) | ADR-043 (keeper) |
| ADR-043 | [307-opml-watchlist-folder-mapping.md](307-opml-watchlist-folder-mapping.md) (was `043-opml-watchlist-folder-mapping.md`) | ADR-307 |
| ADR-050 | [050-external-parakeet-roots-with-managed-vad.md](050-external-parakeet-roots-with-managed-vad.md) | ADR-050 (keeper) |
| ADR-050 | [308-audio-cpp-generated-model-setup-ownership.md](308-audio-cpp-generated-model-setup-ownership.md) (was `050-audio-cpp-generated-model-setup-ownership.md`) | ADR-308 |
| ADR-052 | [052-console-conversation-memory-and-compaction-policy.md](052-console-conversation-memory-and-compaction-policy.md) | ADR-052 (keeper) |
| ADR-052 | [309-comfyui-h3-image-edit-provider-boundary.md](309-comfyui-h3-image-edit-provider-boundary.md) (was `052-comfyui-h3-image-edit-provider-boundary.md`) | ADR-309 |
| ADR-053 | [053-mcp-unified-standalone-runtime-boundary.md](053-mcp-unified-standalone-runtime-boundary.md) | ADR-053 (keeper) |
| ADR-053 | [310-prompt-variable-grammar-and-guarded-insertion.md](310-prompt-variable-grammar-and-guarded-insertion.md) (was `053-prompt-variable-grammar-and-guarded-insertion.md`) | ADR-310 |
| ADR-066 | [066-console-trajectory-view-and-trace-metadata.md](066-console-trajectory-view-and-trace-metadata.md) | ADR-066 (keeper) |
| ADR-066 | [311-local-provider-thinking-controls.md](311-local-provider-thinking-controls.md) (was `066-local-provider-thinking-controls.md`) | ADR-311 |
| ADR-067 | [067-library-top-level-pagination-contracts.md](067-library-top-level-pagination-contracts.md) | ADR-067 (keeper) |
| ADR-067 | [312-bundled-samira-visual-identity-pack.md](312-bundled-samira-visual-identity-pack.md) (was `067-bundled-samira-visual-identity-pack.md`) | ADR-312 |
| ADR-067 | [313-trajectory-export-format.md](313-trajectory-export-format.md) (was `067-trajectory-export-format.md`) | ADR-313 |
| ADR-067 | [314-indefinite-human-approval-waits.md](314-indefinite-human-approval-waits.md) (was `067-indefinite-human-approval-waits.md`) | ADR-314 |
| ADR-068 | [068-console-text-selection-and-annotations.md](068-console-text-selection-and-annotations.md) | ADR-068 (keeper) |
| ADR-068 | [315-local-research-execution-engine.md](315-local-research-execution-engine.md) (was `068-local-research-execution-engine.md`) | ADR-315 |
| ADR-068 | [316-console-project-instruction-context-boundary.md](316-console-project-instruction-context-boundary.md) (was `068-console-project-instruction-context-boundary.md`) | ADR-316 |
| ADR-069 | [069-project-skills-folder-convention.md](069-project-skills-folder-convention.md) | ADR-069 (keeper) |
| ADR-069 | [317-console-project-instruction-local-state-and-preflight.md](317-console-project-instruction-local-state-and-preflight.md) (was `069-console-project-instruction-local-state-and-preflight.md`) | ADR-317 |
| ADR-073 | [073-notes-sync-round-trip-and-interoperability-constraints.md](073-notes-sync-round-trip-and-interoperability-constraints.md) | ADR-073 (keeper) |
| ADR-073 | [318-vendored-chunking-engine-parity.md](318-vendored-chunking-engine-parity.md) (was `073-vendored-chunking-engine-parity.md`) | ADR-318 |
| ADR-077 | [077-server-offloaded-scheduled-agent-tasks.md](077-server-offloaded-scheduled-agent-tasks.md) | ADR-077 (keeper) |
| ADR-077 | [319-console-bounded-rail-section-scrolling.md](319-console-bounded-rail-section-scrolling.md) (was `077-console-bounded-rail-section-scrolling.md`) | ADR-319 |
| ADR-078 | [078-chunking-template-convergence.md](078-chunking-template-convergence.md) | ADR-078 (keeper) |
| ADR-078 | [320-defer-console-context-rail-content-until-first-refresh.md](320-defer-console-context-rail-content-until-first-refresh.md) (was `078-defer-console-context-rail-content-until-first-refresh.md`) | ADR-320 |
| ADR-078 | [321-structured-agent-tool-outcome-provenance.md](321-structured-agent-tool-outcome-provenance.md) (was `078-structured-agent-tool-outcome-provenance.md`) | ADR-321 |
| ADR-078 | [322-research-workspace-authority-and-screen-boundaries.md](322-research-workspace-authority-and-screen-boundaries.md) (was `078-research-workspace-authority-and-screen-boundaries.md`) | ADR-322 |
| ADR-079 | [079-console-library-conversation-authority.md](079-console-library-conversation-authority.md) | ADR-079 (keeper) |
| ADR-079 | [323-daily-reports-surface-and-demo-seeding.md](323-daily-reports-surface-and-demo-seeding.md) (was `079-daily-reports-surface-and-demo-seeding.md`) | ADR-323 |
| ADR-079 | [324-network-tls-trust-policy.md](324-network-tls-trust-policy.md) (was `079-network-tls-trust-policy.md`) | ADR-324 |
| ADR-079 | [325-workspace-assistant-defaults.md](325-workspace-assistant-defaults.md) (was `079-workspace-assistant-defaults.md`) | ADR-325 |
| ADR-079 | [326-workspace-file-inspector-direct-user-authority-and-save-publication.md](326-workspace-file-inspector-direct-user-authority-and-save-publication.md) (was `079-workspace-file-inspector-direct-user-authority-and-save-publication.md`) | ADR-326 |
| ADR-080 | [080-trace-v2-exhaustive-event-projection-and-collaboration.md](080-trace-v2-exhaustive-event-projection-and-collaboration.md) | ADR-080 (keeper) |
| ADR-080 | [327-model-machine-memory-fit-estimation.md](327-model-machine-memory-fit-estimation.md) (was `080-model-machine-memory-fit-estimation.md`) | ADR-327 |
| ADR-080 | [328-permission-request-context-summaries.md](328-permission-request-context-summaries.md) (was `080-permission-request-context-summaries.md`) | ADR-328 |
| ADR-084 | [084-change-review-consent-and-asynchronous-finalization.md](084-change-review-consent-and-asynchronous-finalization.md) | ADR-084 (keeper) |
| ADR-084 | [329-library-media-reader-ia.md](329-library-media-reader-ia.md) (was `084-library-media-reader-ia.md`) | ADR-329 |
| ADR-084 | [330-mcp-profile-driven-rag-search-contract.md](330-mcp-profile-driven-rag-search-contract.md) (was `084-mcp-profile-driven-rag-search-contract.md`) | ADR-330 |
| ADR-089 | [089-console-per-turn-change-review-ownership.md](089-console-per-turn-change-review-ownership.md) | ADR-089 (keeper) |
| ADR-089 | [331-console-full-semantic-capture-policy.md](331-console-full-semantic-capture-policy.md) (was `089-console-full-semantic-capture-policy.md`) | ADR-331 |
| ADR-090 | [090-console-thinking-block-ownership-and-replay.md](090-console-thinking-block-ownership-and-replay.md) | ADR-090 (keeper) |
| ADR-090 | [332-permission-request-context-summaries.md](332-permission-request-context-summaries.md) (was `090-permission-request-context-summaries.md`) | ADR-332 |
| ADR-092 | [092-console-full-semantic-capture-policy.md](092-console-full-semantic-capture-policy.md) | ADR-092 (keeper) |
| ADR-092 | [333-console-live-child-write-path-boundaries.md](333-console-live-child-write-path-boundaries.md) (was `092-console-live-child-write-path-boundaries.md`) | ADR-333 |
| ADR-092 | [334-console-chat-fork-copy-and-authority-boundary.md](334-console-chat-fork-copy-and-authority-boundary.md) (was `092-console-chat-fork-copy-and-authority-boundary.md`) | ADR-334 |
| ADR-093 | [093-raw-and-virtual-cli-execution-boundaries.md](093-raw-and-virtual-cli-execution-boundaries.md) | ADR-093 (keeper) |
| ADR-093 | [335-offline-tiktoken-runtime-assets.md](335-offline-tiktoken-runtime-assets.md) (was `093-offline-tiktoken-runtime-assets.md`) | ADR-335 |
| ADR-094 | [094-raw-and-virtual-cli-execution-boundaries.md](094-raw-and-virtual-cli-execution-boundaries.md) | ADR-094 (keeper) |
| ADR-094 | [336-console-turn-lifetime-and-navigation-boundary.md](336-console-turn-lifetime-and-navigation-boundary.md) (was `094-console-turn-lifetime-and-navigation-boundary.md`) | ADR-336 |
| ADR-097 | [097-boot-budget-ratchets.md](097-boot-budget-ratchets.md) | ADR-097 (keeper) |
| ADR-097 | [337-console-reference-backed-semantic-trace-ledger.md](337-console-reference-backed-semantic-trace-ledger.md) (was `097-console-reference-backed-semantic-trace-ledger.md`) | ADR-337 |
| ADR-098 | [098-visible-bounded-console-prompt-queue.md](098-visible-bounded-console-prompt-queue.md) | ADR-098 (keeper) |
| ADR-098 | [338-low-latency-speculative-duplex-voice-pipeline.md](338-low-latency-speculative-duplex-voice-pipeline.md) (was `098-low-latency-speculative-duplex-voice-pipeline.md`) | ADR-338 |
| ADR-099 | [099-persistent-terminal-session-runtime-boundary.md](099-persistent-terminal-session-runtime-boundary.md) | ADR-099 (keeper) |
| ADR-099 | [339-schedule-editor-shape.md](339-schedule-editor-shape.md) (was `099-schedule-editor-shape.md`) | ADR-339 |
| ADR-102 | [102-personal-context-profile-authority-sync-and-encryption.md](102-personal-context-profile-authority-sync-and-encryption.md) | ADR-102 (keeper) |
| ADR-102 | [340-console-run-admitted-local-path-authority.md](340-console-run-admitted-local-path-authority.md) (was `102-console-run-admitted-local-path-authority.md`) | ADR-340 |
| ADR-104 | [104-token-budgeted-agent-tool-disclosure.md](104-token-budgeted-agent-tool-disclosure.md) | ADR-104 (keeper) |
| ADR-104 | [341-library-media-return-settlement-boundary.md](341-library-media-return-settlement-boundary.md) (was `104-library-media-return-settlement-boundary.md`) | ADR-341 |
| ADR-115 | [115-local-versioned-canvas-artifacts-and-browser-sandbox.md](115-local-versioned-canvas-artifacts-and-browser-sandbox.md) | ADR-115 (keeper) |
| ADR-115 | [342-personas-demand-mounted-center-views.md](342-personas-demand-mounted-center-views.md) (was `115-personas-demand-mounted-center-views.md`) | ADR-342 |
| ADR-146 | [146-independent-buddy-petdex-publication-and-character-creation.md](146-independent-buddy-petdex-publication-and-character-creation.md) | ADR-146 (keeper) |
| ADR-146 | [343-console-custom-endpoint-registry.md](343-console-custom-endpoint-registry.md) (was `146-console-custom-endpoint-registry.md`) | ADR-343 |
| ADR-147 | [147-conversation-archive-and-exact-resume.md](147-conversation-archive-and-exact-resume.md) | ADR-147 (keeper) |
| ADR-147 | [344-agent-provider-routing.md](344-agent-provider-routing.md) (was `147-agent-provider-routing.md`) | ADR-344 |
| ADR-148 | [148-mcp-hub-rail-ia-and-responsive-triad.md](148-mcp-hub-rail-ia-and-responsive-triad.md) | ADR-148 (keeper) |
| ADR-148 | [345-console-run-hooks.md](345-console-run-hooks.md) (was `148-console-run-hooks.md`) | ADR-345 |
| ADR-149 | [149-offer-first-canvas-guide-and-inline-skill.md](149-offer-first-canvas-guide-and-inline-skill.md) | ADR-149 (keeper) |
| ADR-149 | [346-console-persona-session-identity.md](346-console-persona-session-identity.md) (was `149-console-persona-session-identity.md`) | ADR-346 |
| ADR-150 | [150-design-token-system-and-design-language.md](150-design-token-system-and-design-language.md) | ADR-150 (keeper) |
| ADR-150 | [347-mcp-hub-bulk-permission-actions.md](347-mcp-hub-bulk-permission-actions.md) (was `150-mcp-hub-bulk-permission-actions.md`) | ADR-347 |
| ADR-150 | [348-agent-chat-fork-and-spawn.md](348-agent-chat-fork-and-spawn.md) (was `150-agent-chat-fork-and-spawn.md`) | ADR-348 |
| ADR-172 | [172-library-artifacts-browse-and-navigation.md](172-library-artifacts-browse-and-navigation.md) | ADR-172 (keeper) |
| ADR-172 | [349-skill-eval-sub-harness.md](349-skill-eval-sub-harness.md) (was `172-skill-eval-sub-harness.md`) | ADR-349 |
| ADR-184 | [184-agent-task-server-execution-message-refs-and-approval-escalation.md](184-agent-task-server-execution-message-refs-and-approval-escalation.md) | ADR-184 (keeper) |
| ADR-184 | [350-local-console-prompt-draft-shelf.md](350-local-console-prompt-draft-shelf.md) (was `184-local-console-prompt-draft-shelf.md`) | ADR-350 |
| ADR-218 | [218-in-repo-merge-queue.md](218-in-repo-merge-queue.md) | ADR-218 (keeper) |
| ADR-218 | [351-file-notes-replica-retention.md](351-file-notes-replica-retention.md) (was `218-file-notes-replica-retention.md`) | ADR-351 |
| ADR-220 | [220-console-human-decision-coordination-ownership.md](220-console-human-decision-coordination-ownership.md) | ADR-220 (keeper) |
| ADR-220 | [352-evaluation-defaults-and-private-overrides.md](352-evaluation-defaults-and-private-overrides.md) (was `220-evaluation-defaults-and-private-overrides.md`) | ADR-352 |
| ADR-228 | [228-config-encryption-derived-key-memoization.md](228-config-encryption-derived-key-memoization.md) | ADR-228 (keeper) |
| ADR-228 | [353-sqlite-store-kernel.md](353-sqlite-store-kernel.md) (was `228-sqlite-store-kernel.md`) | ADR-353 |
