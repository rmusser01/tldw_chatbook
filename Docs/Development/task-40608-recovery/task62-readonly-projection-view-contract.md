# Accepted read-only projection adapter

Implement only new `tldw_chatbook/Chat/console_execution_projection_view.py` as an artifact proposal during the native freeze. No client/protocol/worker/ChatScreen edits or execution. Existing canonical design section3 explicitly requires a read-only adapter for sessions/messages/fingerprint; this is not another runtime/store.

Use a frozen `ConsoleProjectionView` containing only:

- `source_revision: int`
- `session_rows: tuple[ConsoleSessionView, ...]`, in original creation order (existing decoded frozen type)
- `active_session_id: str | None`
- `fingerprints: tuple[tuple[str, tuple[int, int]], ...]`, original backend `store.session_fingerprint` values captured with this publication
- `projections: tuple[ConsoleProjection, ...]`, only explicitly supplied original seven-field decoded projections, at most one per represented session/incarnation. Central owns capture/retirement/epoch/visit/adoption and keeps current frames, never per-poll history.

Exact read API:

- `sessions() -> list[ConsoleSessionView]` preserves row creation order.
- `session(session_id) -> ConsoleSessionView` returns the exact frozen row or raises KeyError.
- `messages_for_session(session_id) -> list[ConsoleChatMessage]` returns original `copy_console_message` disposable rendering copies from that explicitly supplied frame; missing frame raises `ConsoleProjectionUnavailable`, never silently returns empty or fetches native state.
- `session_fingerprint(session_id) -> tuple[int,int]` returns the exact supplied original cheap fingerprint, preserving its documented limitations; do not use it as source or complete snapshot authority.
- `display_projection_revision(session_id) -> int` returns the original supplied frame revision; missing frame refuses.

Validate exact classes, nonnegative integer revisions/counts excluding bool, unique row/frame/fingerprint identities, active ID belongs to rows when non-None, frame source revision equals adapter source, and each frame's incarnation matches its row. Frames may cover only a subset of rows; fingerprints cover exactly all rows. No fabricated session/initial empty frame. Original messages remain inert mutable rendering records; return disposable copies and do not mutate stored frames. No config, commands, provider methods, received claims, live source refs, listeners/callables, lazy native reads, ensure/create/update methods or duck-typed store compatibility. Parent refresh notification belongs to the actual client/attachment code, not an arbitrary callback registry here.

Keep source scope finite and reuse exact original records/copy helper. No new component matrix; central actual process/projection/attached UI tests will exercise use. Report source-only/unrun status, exact preimages/absence and pins. This adapter does not itself provide currently absent rail/ancillary or decision-modal values; those remain explicit central capture work.
