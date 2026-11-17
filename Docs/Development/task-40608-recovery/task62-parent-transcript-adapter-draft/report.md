# Parent transcript adoption proposal

Artifact-only new module: `tldw_chatbook/UI/Console_Modules/execution_projection.py`. Its absence was captured before drafting and checked again when the manifest was written. No live source changed. Existing ADR-126/Task62 contract applies; no new architectural decision.

`apply_console_process_projection(screen, client, projection, *, visit)` requires the original exact receiver types, client/view/visit/projection identity, unchanged source revision, the actual parent app, original current/mounted screen, and live mounted original transcript. Closed, detached, superseded, shutdown, or channel-lost work returns `False`. Changed original classes, bound methods, properties, or definitions refuse explicitly. Direct synchronous consumer definitions are pinned with the existing `_definition`/`_definition_current` helpers.

The consumer queries the existing `#console-native-transcript`, transfers the original pending swipe-selection UI value at its original boundary, then invokes only `set_model_thinking_visible`, `set_messages(..., session_id=...)`, and `sync_jump_indicator`. The original `_show_model_thinking` policy reads device-local app configuration and does not create or query a Console runtime/store/controller. The existing `_message` receiver is checked only to read and clear its UI handoff field; none of its service callbacks run.

This module does not establish operation retirement. Central integration must install `client._projection` only after the actual positive retirement receipt, bind the exact view/visit, and establish all other screen context. `True` means the synchronous presentation calls ran; it does not acknowledge row mounting, paint, native cleanup, or a complete ChatScreen update. No composer, draft, session switch, title, activity, decision, cost, inspector, or lifecycle behavior is supplied here.

Integration should import this adapter lazily from the established parent update path after `ChatScreen` has finished defining its class: the module pins the actual screen methods at import time. No callback registration or alternate fallback is added.

Source anchors inspected and copied into `source-dependencies/`:

- `ChatScreen._show_model_thinking` at line4212; swipe proxy at9868; original transcript ingest at17441 and jump update at17504.
- `ConsoleTranscript.set_messages` at4180, `set_model_thinking_visible` at4451, and `sync_jump_indicator` at3231.
- Original screen teardown helper at1431, plus actual Textual app/current/mounted/screen descriptors.
- Existing definition helpers and current `ConsoleProjection`/`ConsoleExecutionClient` fields.

Read both required UI design documents. This change reuses existing widgets, styles, tokens, and behavior; it adds no visual values or layouts.

No Python/import, lint, tests, native launch, process census, or new test matrix ran. Central integration retains formatting/static validation and actual attached process/UI evidence.

Manifest SHA256: `1817f84a20b02fc43c256a81c62dd6f659a7848d04db3d3d04fb85d45bfe53ef`.

Proposed module SHA256: `3075eb9860ce5b408fa5406acf1fa15f6b40afad1b55b68305cc0ec139ea4e25`.
