# Collections controller extraction history

These docstrings describe the original September 2 extraction and are preserved
from the pre-review source for historical reference. Their method counts and
byte-for-byte statements describe that extraction, not the current implementation.
Current ownership and coupling rationale remain in the controller source.

## Module

Library Collections canvas controller.

Controller PR of the Collections extraction series (wave-2 task 6 of
``.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio``;
collections series 2/3; recipe:
``backlog/docs/library-decomposition-recipe.md`` §13; export series --
``library_export_controller.py`` -- is the template this mirrors
byte-for-byte in shape). Owns the entire Collections capture-reader
cluster: rail-scope/filter/sort/paging, quick-capture (open/close/draft
retention/save/retry/refresh), the reader (mode switch, highlights,
freeform/linked notes, content actions, archive/hard-delete/favorite/
mark-read/open-original), the legacy-JSON-recovery export mechanism (an
unrelated feature to the chatbook Export canvas -- see below), and the
adaptive-reader-shell layout sync/preference mirror pair. ``LibraryScreen``
keeps one-line delegators under every one of these original names.

**Cluster derivation -- ownership.** A mechanical ``ast`` scan of
``LibraryScreen`` for method names containing ``"collection"`` (case-
insensitive) finds 67 methods (matches wave-2 task 5's own census exactly,
re-derived fresh at this task's execution time per the recipe's own
"never trust a carried-over count" rule -- §6). Reading each of the 67
bodies (not trusting the name match, per the recipe's own documented
substring-match trap, §2/§11) finds **3 are Prompts-owned, not Collections-
owned**: ``handle_library_prompts_collection`` (``@on``),
``_apply_library_prompt_collection``, ``_sync_library_prompt_collection_
label`` -- an entirely different feature (saved-prompt grouping), using
``_library_prompt_collections_controller``/``_library_prompt_browse_
controller``, not this cluster's ``_library_collections_capture_
controller``. Task 5's own report already excluded these from the field
census; this task reconfirms the same 3 are excluded from the METHOD
census for the identical reason.

**No further exclusions were found.** Unlike the export series (29 of 51
candidates excluded across three rounds: other-subsystem ownership, a
``@work`` framework-decorator hazard, and 9 unbound-fake-self/silent-Mock
test bypasses), this cluster's remaining 64 candidates all: (a) have no
``@work`` decorator (a full ``ast`` decorator-list scan over all 64 found
zero -- confirmed, not assumed, before committing to this move); (b) are
never called via ``LibraryScreen.<name>(fake, ...)`` unbound, in any test
file under ``Tests/`` (a repo-wide grep for every one of the 64 exact
names as ``LibraryScreen.<name>(`` found zero hits -- neither
``Tests/UI/`` nor ``Tests/Library/``, matching the export series' own
forward note to widen the search); (c) are never monkeypatched via
``monkeypatch.setattr(screen, "<name>", ...)``/``monkeypatch.setattr(
LibraryScreen, "<name>", ...)`` nor assigned directly as an instance
attribute (``screen.<name> = ...``) anywhere in ``Tests/`` (a script-driven
regex sweep over every ``.py`` file under ``Tests/`` for both shapes, all
64 names, found zero); and (d) are none of recipe §3's four known
screen-routed monkeypatch names (``_list_local_source_snapshot``,
``_refresh_local_source_snapshot``, ``_apply_local_source_snapshot``,
``_refresh_library_note_detail``). All 64 move onto this controller.

**A fourth, scattered group the naive "one contiguous block" read would
miss**: 4 of the 64 live far from the other 60 in the pre-move file --
``_sync_library_collections_reader_layout_from_shell`` (was line 6886),
``_mirror_library_collections_reader_preference`` (was line 6926),
``_restore_library_collections_page`` (was line 9554, a ``@staticmethod``),
and ``_library_collections_capture_presentation`` (was line 13922) -- each
sitting beside its sibling subsystems' own same-shaped methods (the
adaptive-reader-shell layout-sync/preference-mirror family every browse
subsystem has one of, and the RAG panel-state builder family). All four
are genuinely Collections-owned (confirmed by body content, not position)
and move here alongside the other 60. The first two are called by name
from ``_toggle_library_media_reader_pane`` (a FOUR-subsystem shell
dispatcher that stays on ``LibraryScreen``, unmoved) and from
``_sync_library_reader_preference_layout``/``_persist_library_reader_
preference``'s literal-string-keyed dispatch dicts (``"collections":
self._mirror_library_collections_reader_preference``) -- both call sites
resolve ``self.<name>`` on the SCREEN at call time, so a same-named screen
delegator satisfies them exactly like every other cluster method's
external callers, with no special-casing needed.

**Already-extracted-wiring check (this series' own new bypass-adjacent
shape, per the task brief): does any candidate already delegate to an
existing controller, making it dead-on-arrival for a full-body move?**
None do. Every one of the 64 candidates is a REAL, full-bodied
``LibraryScreen`` method -- none is a bare one-line forward to
``LibraryCollectionsCaptureController`` (the pre-existing headless
orchestration engine this cluster depends on, distinct from the
Textual-adjacent controller this file defines) or to any other
already-existing controller. 28 of the 64 REFERENCE that headless engine
via ``self._library_collections_capture_controller`` as a collaborator
(building requests, calling ``controller.load_page``/``select_item``/
``scope_service.<op>``, etc.), which is a data/business-logic dependency,
not a wiring shortcut -- confirmed by reading every one of those 28
bodies: each still carries its own request-building, validation, status-
line, and recompose-scheduling logic around the calls into that engine.
(Test guard, confirmed still green: ``Tests/UI/
test_product_maturity_phase39_library_collections.py::
test_collections_route_has_no_generic_container_controller_or_panel``
asserts the literal string ``"LibraryCollectionsBrowseController"`` --
note the DIFFERENT name -- never appears in ``library_screen.py``; this
controller is named ``LibraryCollectionsController``, matching the
``LibraryCollectionsState``/``LibraryExportController``/
``LibraryExportState`` naming convention, and does not touch that guard.)

**Dynamic-dispatch census (recipe §11 lesson 3, generalized), confirmed
BEFORE moving anything:** a full grep for ``getattr(self,``/``getattr(
screen,``/``setattr(self,``/``setattr(screen,`` using an f-string or
dict-literal argument, across ``tldw_chatbook/``, found none touching any
Collections field or method name. The one PRE-EXISTING dynamic-dispatch
site that DOES reach a Collections name --
``_replace_library_reader_preference``'s/``_persist_library_reader_
preference``'s 7-destination ``{"collections": "_library_collections_
reader_preferences", ...}`` string-keyed dict (``library_screen.py``,
shared across every browse subsystem) -- resolves through
``operator.attrgetter``/``_assign_library_reader_preferences_attribute``
to the SCREEN's own ``_library_collections_reader_preferences`` property
shim (installed by task 5's state PR, unaffected by this controller
move -- that shim still lives on ``LibraryScreen``, reading through
``self._collections_state``) rather than to any method this PR moves, so
it is not a hazard for this task. An AST Store-context scan over all 64
moved bodies additionally confirms ``_library_selected_row_id`` (the
recipe's own canonical >=2-subsystems field, 226 refs) is read-only in
this cluster -- no moved body writes it -- so only a read accessor is
bound below, mirroring the export controller's identical treatment of the
same field.

**Byte-for-byte canon** (moved bodies never edited -- every name they
reference that is not this controller's own state is rebound under the
SAME name, per the two binding kinds; see
``ConsoleDictationController.__init__``,
``tldw_chatbook/UI/Console_Modules/dictation.py``, and
``LibraryExportController.__init__`` for the sibling worked examples):

1. **Framework services** (``app_instance``, ``app``, ``call_after_
   refresh``, ``is_mounted``, ``query_one``, ``refresh``) are live-read
   from the screen via ``@property`` on every access -- never snapshotted.
2. **Everything else** the cluster depends on that is not its own state is
   a NAMED constructor dependency. This cluster's dependencies: (a) one
   general Library-wide shell helper a moved body calls with an explicit
   argument (``_library_adaptive_reader_allocation_is_current``, shared
   with Notes/File Notes/Media -- ``_sync_library_collections_reader_
   layout_from_shell`` uses it to guard a stale-allocation shell resize);
   (b) one piece of shared shell state this cluster only READS
   (``_library_selected_row_id``, read-only per the Store-context scan
   above -- ``_refresh_library_collections_capture_reader`` uses it to
   gate the destination-owned recompose); and (c) the ONE screen-resident
   wiring field this series' own state PR (task 5) deliberately kept OFF
   ``LibraryCollectionsState`` (``_library_collections_capture_
   controller``, holding a live ``LibraryCollectionsCaptureController``
   instance -- the ``_conversation_reader_controller`` precedent task 5's
   report named in advance), bound here as a GET+SET accessor PAIR (not a
   read-only accessor like (b)) because ``_ensure_library_collections_
   capture_controller`` both reads AND lazily WRITES it (confirmed by an
   AST Store-context scan: exactly one moved body, this one, assigns to
   it).

This subsystem's OWN state (every ``_library_collections_<field>`` name
the moved bodies reference) is exposed through generated properties
reading ``self._collections_state_accessor().<field>`` -- the same
generator shape task 5 installed on ``LibraryScreen`` and the export
controller installed on itself, applied here. Collections uses a single
``_library_collections_`` prefix for every field (task 5's report: no
field needed a plural variant, since "collections" is already plural), so
there is no per-field prefix-selection logic in the generator loop,
matching the export controller's own precedent exactly. No ``_safe_text``
class-binding is needed here (unlike Conversations/Export): no moved body
in this cluster calls ``self._safe_text(...)``.


## Class

Owns the entire Collections capture-reader cluster (64 methods).

    Holds no state of its own beyond what it reads and writes through
    ``LibraryCollectionsState`` (via the injected accessor), the shared
    shell attributes bound below, and the ``LibraryCollectionsCaptureController``
    headless-engine instance it borrows via a get+set accessor pair.
    ``LibraryScreen`` constructs exactly one of these, in ``__init__``
    right after ``self._export_controller``, and keeps one-line delegators
    for every original name this cluster moved (64 -- see the module
    docstring for the full derivation).


## Constructor

Build the controller and bind everything its moved bodies need.

        Every one of the 64 method bodies below is a byte-for-byte copy of
        the pre-extraction ``LibraryScreen`` method: no internal line was
        edited to retarget a call or an attribute. That is possible
        because this constructor binds every name those bodies reference
        that is not this controller's own state, under the SAME name the
        original method used. See the module docstring for the binding
        kinds this follows.

        Args:
            screen: The Library screen. Used ONLY for the six framework
                services below (``app_instance``, ``app``, ``call_after_
                refresh``, ``is_mounted``, ``query_one``, ``refresh``) --
                this cluster owns no DOM of its own.
            collections_state_accessor: Returns the live
                ``LibraryCollectionsState`` (``LibraryScreen.
                _collections_state``, task 5). Backs every generated
                ``_library_collections_<field>`` property below.
            library_adaptive_reader_allocation_is_current: ``LibraryScreen.
                _library_adaptive_reader_allocation_is_current`` -- the
                shared stale-allocation guard every adaptive-reader-shell
                layout sync uses (Notes/File Notes/Media/Collections
                alike); ``_sync_library_collections_reader_layout_from_
                shell`` calls it before resolving a fresh layout.
            library_selected_row_id_accessor: Reads ``LibraryScreen.
                _library_selected_row_id`` -- the recipe's own canonical
                >=2-subsystems shared field (226 refs). Read-only in this
                cluster: confirmed by an AST Store-context check that no
                moved body writes it directly, so no setter is bound.
            library_collections_capture_controller_accessor: Reads
                ``LibraryScreen._library_collections_capture_controller``
                -- the live headless-engine instance task 5 kept OFF
                ``LibraryCollectionsState`` ("wiring, not state").
            set_library_collections_capture_controller: Writes that same
                screen attribute -- ``_ensure_library_collections_capture_
                controller`` lazily constructs and caches the engine the
                first time it is needed, so both a getter and a setter are
                bound (unlike the read-only accessor above).
