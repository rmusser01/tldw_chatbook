# Task10 store-cap preflight proposal

Status: FIT, proposal only. Recommend existing `tldw_chatbook/Chat/console_chat_fork.py` for eight pure fork projection/validation/fingerprint functions. Keep mutable authority and public store wrappers in `ConsoleChatStore`. This is a placement proposal for root selection, not implementation authorization.

Source pin: `d3443b9e4297fa20897cf99e77a3ae2c8c562b10` (reviewed I1); incoming dev: `5e0341d1ec701865e019eb2fd8a5e2028ab2d474`. All source extraction and count comparisons use Git object contents at this pin. Worktree HEAD was `7c5528654742` when the preflight began; no checkout was changed.

## Actual formatted projection

| Measurement | Lines |
| --- | ---: |
| Incoming store | 22,499 |
| Reviewed store | 22,655 |
| Unchanged store cap | 22,344 |
| First non-fitting proposal | 22,345 |
| Final proposed store, including all wrappers/imports/live callback wiring | **22,334** |
| Reduction from reviewed store | **321** |
| Remaining headroom | **10** |
| Existing fork owner | 1,135 |
| Final proposed fork owner | 1,563 |
| Joint source LOC increase | 107 |

These are `len(text.splitlines())` counts after actual installed Ruff **0.16.6** formatting through stdin with each real source filename/config. Formatting the untouched pinned store also yields 22,655 lines and identical AST. No formatter or dependency was installed. The unchanged cap has 10 lines of slack, within the ratchet's 50-line slack tolerance; neither cap nor exceptions change.

The first proposal missed by one line. Its exact 22,345-line store and 1,532-line owner sketches/map are retained as `task-10-store-cap-first-nonfit-*`, reproduced from the same source pin and formatter. Adding the coherent video tuple validator gives 22,332; preserving fingerprint callback lookup at invocation time gives the final 22,334. There is no assumed allowance or comment compression.

## Exact placement

All spans below are one-based inclusive in the reviewed store; standalone-method spans include their decorator. The map records proposed qualified names, dependencies and concrete wiring; the complete formatted sketches supply the projected bytes.

| Original owner/span | Proposed owner function |
| --- | --- |
| `ConsoleChatStore._fork_message_state_is_eligible`, 7004–7014 | `console_fork_message_state_is_eligible` |
| `_fork_visible_selection`, 7075–7095 | `console_fork_visible_selection` |
| `_fork_attachment_fingerprint`, 7097–7187 | `fingerprint_console_fork_attachments` |
| `_validate_fork_image_selections`, 7189–7230 | `validate_console_fork_image_selections` |
| `_fork_video_fingerprint`, 7232–7260 | `fingerprint_console_fork_video` |
| `_validate_video_projection_tuple`, 7262–7282 | `validate_console_fork_video_projection` |
| `_stage_fork_snapshot` per-message projection, 7761–7872 | `project_console_fork_message` |
| `_stage_fork_snapshot` candidate comparison, 7909–7968 | `console_fork_candidate_matches_fence` |

The six original method names, signatures and static/class method decorators remain callable wrappers. `_fork_media_fingerprint` remains in the store and dispatches through its original class methods. The stage loop retains UUID allocation, turn IDs, predecessor IDs, append order and projected-image alias writers. Its pure per-message helper receives explicit already-allocated identities, one borrowed message, one frozen lineage entry, a read-only image-alias map, and one narrow fingerprint callback; it returns an immutable projection and a boolean. No helper receives the store or gains a state writer. Candidate comparison receives projected immutable messages plus lineage and read-only maps and returns a boolean.

AST comparison finds changes in exactly seven store methods: the six wrappers plus `_stage_fork_snapshot`. Every other store method AST and all method names are preserved. The eight new owner functions have no attribute or subscript assignments. Local construction lists remain local to one invocation.

## Dependency and lookup census

The existing fork owner already defines the immutable projection contracts, canonical image/configuration fingerprinting, image-byte validation and source-prefix refusal. Moving their pure companions there is direct ownership alignment. Exact enum/type checks, exception messages, hash domains, JSON sorting/bounds, byte retention and tuple ordering follow the original bodies.

Existing eager imports are reused. The store's existing fork import adds eight symbols; `math` moves from the store into the existing fork module. `MAX_ATTACHMENT_BYTES` is added to the owner's existing `attachment_core` import. Video metadata and marker imports are function-local; their module identities are already loaded by the store. The saved ready census contains fork/store/video metadata/video store at lines 148/150/868/869. No new eager project-module identity is proposed. Source top-level import statements are store 67→66 and owner 21→22; all import statements including guarded/local ones are store 92→91 and owner 22→26. This is a static dependency proof, not a newly executed runtime census.

Ready 1,033 and preimport 557 module caps carry unchanged. The Chat closure is excluded from the preimport pass's marginal payload by its documented prewarm contract. Joint source LOC grows 107 lines due to honest helper signatures/wrappers; do not describe this as an overall code-size reduction or as fresh startup performance evidence.

Original store class/method patch paths remain. Known fork tests patch `_validate_fork_fence`/`validate_fork_fence`; both paths and two-stage validation remain intact. Prefix-refusal callbacks still call store eligibility/selection wrappers. Lambda callbacks re-read `self._fork_video_fingerprint`, `self._fork_attachment_fingerprint` and `cls._fork_video_fingerprint` at invocation, preserving live method lookup rather than capturing a bound method ahead of image validation. No borrowed external class-method use or literal computed-name patch of the moved six names was located by bounded source/test census; all six names are retained anyway. The pure global image validator and selected-image fingerprint resolve in the fork owner after relocation; no store-global monkeypatch of those names was located. Arbitrary dynamically constructed names are not proven absent by text search.

## Mutable boundaries and feature carry

`stage_fork_snapshot` retains `_fork_source_lock`; its initial and final `validate_fork_fence` calls remain in `_stage_fork_snapshot` and use the exact fence/image selections. Configuration snapshotting and fingerprinting, destination identity validation, citation-owner lookup, candidate snapshot assembly, final refusal and return order remain store-owned. Pure helpers are synchronous and do not yield, schedule, read SQLite, write files, register a session, mutate a message, or operate on a video store. `_validate_video_projection_tuple` keeps its store path for generation reload callers.

The incoming→reviewed feature patch does not overlap any extracted span. Handoff metadata parsing/launch state, revision writers/tasks/callbacks, draft custody/consumption, session incarnation, machine-start receipt fingerprint fields, source Close decisions, actor ownership and durable acceptance cutoff retain their original owner and AST. No schema, provider, continuation policy, field ownership, lock protocol, retry behavior or persistence contract changes.

## Governance and later verification scope

ADR required: **no new ADR**. Governing existing ADR: `backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md` (pure canonical projections, immutable copy and mutable currentness/persistence authority); carry `backlog/decisions/219-console-chat-destinations-and-bounded-starts.md` and the current spec for draft/start/source-Close/acceptance ownership. This directly implements existing boundaries through a mechanical pure placement. Root must record exact architecture/source scope before separate sole-writer work.

Affected source scope is exactly `Chat/console_chat_store.py` and `Chat/console_chat_fork.py`. Proposed targeted behavioral coverage after implementation is the existing fork store/projection suite (`Tests/Chat/test_console_chat_fork.py`), fork public mutation fences (`test_console_fork_public_boundaries.py`), and fork lineage contracts (`test_console_trace_fork_lineage.py`), plus focused new call-time callback and no-input-mutation probes at the helper boundary if needed. Apply both size-ratchet tests only to the store row. Source/AST ownership and import-derived receipts should be refreshed for these paths without inventing a runtime census result. Semantic persistence writer inventory is carried: no indexed writer moved. Previously qualified creation/start/cutoff/metadata and unrelated subsystem suites are carried, with no blanket rerun proposed.

No tests, production imports/execution, installs, agents, tracked source/test/plan edits or Git mutations were performed. Only task-10-prefixed proposal artifacts were saved in the specified SDD folder. The sketches are review artifacts and require root selection, later implementation review and targeted verification before any correctness claim.

Concerns: 10-line store headroom is real but narrow; later wrapper/signature additions must be remeasured. Eight helpers enlarge the existing pure owner to 1,563 lines, below the governed modules in this ratchet and without adding another module. Static dependency equivalence carries existing startup evidence but does not establish fresh timing/census. Live callback lookup is explicitly preserved; pure global patch ownership moves as described above.
