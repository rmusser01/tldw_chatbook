# Same-screen Canvas import refusal: native proof

**Reproduced on unchanged d6406d4dc86720f9916e5416bf3e8de8cc7e647a.** Two finite successful proof executions; the second retained a small child receipt. No repository changes, app boot, guard replacement, capture mutation or timeout change.

The probe reuses the real native-bound private configuration fixture, `ConsoleRuntime`, its real `ConsoleChatStore` and `ConsoleCanvasController`, and native authority. It creates a legitimate ephemeral session/message. A minimal attached view supplies live scope from that actual store and ordinary callback slots; it invokes the real `ChatScreen._ensure_console_chat_store` body. The direct config policy reader and all native import/capture/owner/compiler/apply checks remain original.

Sequence and evidence:

1. Capture an import and run the initial real `_apply_import` validation: returns None with no mutation. Compile with the authority's real compilation helper.
2. Invoke the actual screen store accessor once. It replaces the binding and increases authority view generation by exactly one.
3. Original capture commit raises exact `RuntimeError('canvas_scope_unavailable')`; no Canvas created. Current policy remains enabled, authority is live, session/conversation/branch/selection unchanged, and original `validate_interactive_owner` accepts the captured owner.
4. Fresh `_capture_import` differs from the original dataclass in **only `view_generation`**. The unchanged public `import_html` subsequently creates one Canvas successfully. No historical capture or generation was edited to make this pass.

This proves a semantic consequence of repeated same-view binding, independently of startup timing. It supports the caller idempotence correction while preserving genuine rebind invalidation and all direct binder/effect policy gates. It does not prove the Windows60s startup failure is resolved or predict elapsed-time savings.

Scope limit: the view is a narrow holder using the real screen store accessor and a live store-derived scope callback; it is not a mounted Textual screen. Canvas uses the supported ephemeral owner/controller path, not durable SQLite Canvas publication. The native selected-config admission is real. Delivery callbacks do not deliver UI output. This is the requested finite import lifecycle proof, not full UI acceptance.

Probe: `/private/tmp/uat-canvas-import-same-view-probe.py`.
Final fixture: `/private/tmp/uat-canvas-import-same-view-lb4qrk05`; original fixture: `/private/tmp/uat-canvas-import-same-view-bz2yi1sy`.
Hashes and exact child receipt: `/private/tmp/uat-canvas-import-same-view-receipt.json`.
