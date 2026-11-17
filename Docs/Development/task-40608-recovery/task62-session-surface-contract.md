# Task62 selected session-surface values and presenter

Artifact-only helper implementation; no live writes or execution during the accepted-process freeze. Existing ADR-126 and Task62 whole-service plan apply. No new architecture or production cohort is introduced.

Exclusive intended paths:
- `tldw_chatbook/Chat/console_execution_session_surface.py`
- `tldw_chatbook/UI/Console_Modules/execution_sessions.py`

Central owns backend capture, protocol/client adoption, App/ChatScreen wiring, Send/ACK, all lifecycle and native execution. Do not edit their files or the original widget. Return proposed files, exact absent/preimage/dependency pins and concise report under `task62-session-surface-draft`. No additional test module; central will cover the new values in the affected projection regression and actual attached surface path.

## Fixed inert value API

`ConsoleSessionSurfaceProjection` is a frozen dataclass with exactly:
- `view: ConsoleProjectionView` (the exact already adopted read-only view; never a store)
- `streaming_session_id: str | None`
- `run_markers: tuple[tuple[str, ConsoleRunMarker], ...] | None`
- `queue_counts: tuple[tuple[str, int], ...] | None`

Use original `ConsoleRunMarker` from console_chat_models. Reject nonexact types, duplicate/unknown IDs, boolean/negative counts, and a non-null streaming ID absent from view rows. Non-null marker/count sets cover exactly all published row IDs, including real NONE/zero values. Preserve None distinctly; never invent values for absent observations. The exact view already validates its source/row/frame identities.

`encode_session_surface_projection(value) -> dict` returns plain JSON with exactly streaming_session_id, run_markers and queue_counts. Marker/count collections use arrays of fixed objects `{session_id, marker}` and `{session_id, count}`; marker is the original enum value. None remains null. No view serialization or authority field.

`decode_session_surface_projection(view: ConsoleProjectionView, value: dict) -> ConsoleSessionSurfaceProjection` validates only this finite plain data and returns original enum leaves bound to that exact view. Reuse existing finite validation helpers only where their plain/frozen representation contract fits; do not change protocol internals.

Central captures these values beside the original ReadProjection boundary from original controller streaming_session_id/run_marker_for/activity_for and actual unseen-mark policy. Central must preserve original fleet.prepare_session_run_markers viewed-mark/wake/current-screen checks through the named parent checkpoint; no missing markers or fabricated defaults may stand in for that behavior. The helper derives no backend policy and clears no durable marks.

## Actual parent presenter

`async apply_console_process_sessions(screen: ChatScreen, client: ConsoleExecutionClient, publication: ConsoleSessionSurfaceProjection, *, visit: int) -> bool`.

Central will install `client.session_surface` as this exact record only alongside its retired, checked projection_view. Reject/refuse unless publication is that current record and publication.view is client.projection_view; client._view/visit/source, actual screen App/current/mounted/not-torn-down, nonfailed/nonclosing client, original definitions and receiver association must all match. Use the existing transcript draft's finite source/method/property qualification pattern. Lazy import after ChatScreen class definition is required; no import-time UI construction.

Use actual `#console-session-surface` / exact original ConsoleSessionSurface. Under the original screen `_console_session_tabs_sync_lock`, capture the exact current surface/publication, revalidate after lock acquisition, preserve `_console_session_tabs_sync_calls` increment/finally decrement, and invoke original `surface.sync_sessions` with publication.view.sessions(), active_session_id, streaming_session_id, actual marker/count values. Preserve the original actual `_console_send_ack.overlay_run_markers` only when that exact original UI owner exists, with original method/receiver checks; do not construct a substitute acknowledgement owner. Await mounting outside native/settings custody. After await, recheck actual App/screen/visit/client/source/current publication and same current queried surface before True. A stale or torn-down route returns False; definition/receiver changes refuse. Cancellation propagates after the original finally counter cleanup. No retry against a later publication.

Do not invoke legacy `_sync_console_native_session_tabs`, ensure/creation/settings/workflow/fleet/coaching methods or local runtime/controller/store. Those are central's named behavior integration. Do not call this adapter a complete full-screen refresh. True means the original surface call completed for the exact still-current publication, not native retirement or paint.

## Existing source anchors

- ChatScreen `_sync_console_native_session_tabs` currently 18229: lock/counter, original marker overlay, surface await and currentness reconciliation.
- ConsoleSessionSurface.sync_sessions currently 836: original render/mount ownership and internal attachment checks.
- UI/Console_Modules/fleet.py prepare_session_run_markers240: original viewed unseen-mark side effects are outside this pure presenter.
- UI/Console_Modules/send_acknowledgement.py overlay_run_markers230: exact UI overlay, not backend acceptance.
- `task62-parent-transcript-adapter-draft/proposed/execution_projection.py`: reviewed original-receiver/current-visit source checks.
- Current ConsoleProjectionView and ConsoleSessionView: reuse, no parallel row schema.
