# Read-only projection view proposal

Artifact-only new target `tldw_chatbook/Chat/console_execution_projection_view.py`; its absence was captured before drafting and rechecked at finalization. No live source changed. Existing ADR-126 and the accepted Task62 read-only-adapter contract apply.

The frozen, slotted `ConsoleProjectionView` contains exactly the five contracted fields. Construction validates original finite session rows with the existing `_SESSION` schema, nonnegative integer revisions/counts excluding bool, unique session/fingerprint/frame identities, complete fingerprint coverage, active-session membership, matching frame source/incarnation, and exact decoded frame/message/run/decision record classes. Missing transcript frames are permitted; invented empty frames are not supplied.

The five public read methods preserve creation order and exact frozen rows, return only supplied original fingerprints/revisions, and use the original `copy_console_message` for disposable message records. That helper intentionally makes shallow copies: nested field values retain their existing sharing semantics. An absent frame raises `ConsoleProjectionUnavailable`; missing rows/fingerprints raise `KeyError`. No read fetches state or creates a session.

Lookup walks the finite supplied rows/frames. Construction checks message classes without serializing or copying message bodies; only an explicit `messages_for_session` call creates the requested rendering copies. There is no callback registry, history cache, native reader, command surface, configuration reference, or authoritative store. Epoch/visit/currentness/retirement, capture, and adoption remain with the actual client and central integration. The cheap original fingerprint keeps its equal-length/non-final-edit limitations and is not complete snapshot authority.

Original source anchors: `ConsoleSessionView` and its finite `_SESSION` schema in `console_execution_projection_values.py:561`; `ConsoleProjection` in `console_control_dispatch.py:236`; `copy_console_message` in `console_chat_models.py:1357`; original store reads at `console_chat_store.py:5609`, `:12303`, `:12324`, and `:21903`. Seven source dependencies, including the contract, are copied and pinned by the manifest.

No Python/import, lint, native execution, tests, or additional test matrix ran. Central integration owns adoption, static validation, and the actual process/projection/attached-UI evidence. Ancillary rail and decision-modal capture is not implemented by this view.

Manifest SHA256: `072f1403abbf115b85df7bb6dbe5129e7373001414c9ef56ffe689085859a754`.

Proposed module SHA256: `7263fff0f9f8c742b4342b03524e9d68adedbeda2031c5705c77dcba8518027d`.
