"""Library Notes canvas controller.

Controller PR of the Notes extraction series (wave-8 task 2 of
``.superpowers/sdd/2026-09-08-library-decomposition-wave8-notes``; notes
series 2/N; recipe: ``backlog/docs/library-decomposition-recipe.md``;
``library_media_controller.py`` -- the newest and largest prior
single-cluster move -- is the template this mirrors in shape). Owns the
Notes cluster: the database-notes list canvas and its editor/work pane, the
Folder-Files workspace seams the screen owns, the note-import canvas, the
lasting-sync ("Add from files" / sync roots) canvases, the notes footer and
stage/responsive presentation, and the notes focus/scroll restore
machinery. This is the FINAL controller move of the eight-wave Library
decomposition program.

**Cluster derivation.** An ``ast`` census of every ``LibraryScreen``
class-body method whose name contains ``"note"`` (case-insensitive), run
fresh at this task's own execution time (the recipe's "never trust a
carried-over count" rule, SS6): **291 raw ``FunctionDef`` matches, 285
unique names**. The 6-name gap is NOT a property/setter pair, as in every
prior series -- it is a **byte-identical DUPLICATE block dev shipped twice**
(see the duplicate-block paragraph below). Of the 285, 93 carry an ``@on``
decorator, 6 are ``action_*``, 7 are ``@staticmethod``, 5 are ``@property``,
and **0 are ``@work``**.

**Recipe SS4's third whitelist member -- ``on_<message>`` NAME-dispatched
handlers -- is INERT for notes, and this is the first series to resolve it
rather than assume it.** The wave-8 plan predicted the opposite ("notes
likely OWNS name-dispatched handlers... the census must enumerate them
explicitly and the prune whitelist must be exercised with evidence"), and
task 1's own hand-off repeated the prediction. Both are wrong, and the
evidence is mechanical: the 93 ``@on``-decorated notes handlers split **63
selector-bound decorators / 36 Message-typed decorators (35 unique method
names**, because ``handle_library_notes_lasting_back`` carries two). Every
one of those 36 Message classes was imported and its ``handler_name``
READ -- ``LibraryNoteImportCanvas.ImportRequested`` computes
``on_library_note_import_canvas_import_requested``,
``LibraryNoteWorkPane.EditorReady`` computes
``on_library_note_work_pane_editor_ready``, and so on -- and **not one
matches the name of the method decorated with it**. Run the other way, over
every ``Message`` subclass in ``Widgets/Library`` (113 distinct
``handler_name``s), the only ones that ARE ``LibraryScreen`` methods are 4
Textual builtins (``on_resize``, ``on_key``, ``on_mouse_down``,
``on_descendant_focus``) and the 7 prompts-owned
``on_prompt_block_editor_*``. ``LibraryScreen`` carries **17** ``on_*``
methods in total: 10 lifecycle hooks and those 7. Zero are notes-owned.
``test_no_notes_handler_is_name_dispatched_by_textual`` pins both directions
so a later notes widget that DOES introduce a name-dispatched handler fails
loudly instead of being pruned silently.

Two completeness checks ran alongside the census:

- a decorator scan for any NON-notes-named method carrying a
  ``#library-note*``/``.library-note*``/``*Note*`` ``@on`` selector: **one
  hit, a false positive** -- ``set_library_collection_capture_mode``'s
  four-selector list includes ``#library-collections-mode-notes``, a
  Collections button;
- a reverse call-graph fixpoint for NON-notes-named methods whose every
  in-class caller is (transitively) notes-named -- the "bare-named cluster
  member" shape the conversations exemplar's own ``_conversation_records``
  miss made the recipe warn about. It returns **6 names**, every one
  SHELL-named and none moved: ``_apply_library_emergency_geometry`` with its
  two helpers ``_capture_library_emergency_restore_receipt`` and
  ``_restore_library_emergency_receipt`` (the emergency-geometry family),
  ``_library_compose_scoped_ref``, ``_library_landing_control_row_id`` and
  ``_library_landing_focus_control_id`` (landing/compose shell helpers).
  Each is a general shell primitive that happens to have only one live
  caller today; moving them would relocate shell infrastructure into a
  subsystem controller, which is the One Home Rule violation the media
  series' review-set ruling refused at the same shape. The one a mover calls
  (``_apply_library_emergency_geometry``, from
  ``_apply_library_notes_stage_legs``) is bound as a named late-binding
  dependency instead.

**The duplicate block, disclosed rather than tidied away.** Six notes
methods were defined TWICE on ``LibraryScreen`` at the parent
(``2641ff0a0``): ``_library_notes_work_session_reader_width``,
``_dispatch_library_notes_work_session``,
``_library_notes_work_first_preferences``, ``_set_library_notes_source``,
``_dispatch_database_note_identity_cleared`` and
``_activate_database_note_work_session``, at ``library_screen.py``
``:6557-6662`` and ``:6663-6768``. The two 106-line blocks are **byte-for-byte
identical** (verified by string comparison, not by eye), so the second
definition silently won at class creation and behaviour never depended on
which. It is a dev-side merge artefact, not this program's: ``git log -L``
attributes the first block to ``7cf89de6c`` ("consolidate long-lived branch
changes") and the second to ``fd505637f`` ("apply work-first Notes
sessions"). A move cannot preserve it -- two delegators cannot share one
name -- so **both copies are removed and ONE delegator per name is
installed**; the definition that lands on this controller is byte-identical
to both. That is why this commit removes 4,034 source lines from the screen
for 3,934 lines of controller body: the 100-line difference is the duplicate
block plus its own blank lines, and it is stated here so a reviewer reading
the two numbers does not have to reconstruct why they differ.

**Single vs. split controller: SINGLE, decided by connected components, not
by feel.** The decision was taken at the point the split had to be made --
over the **196-name candidate MOVE set as it stood then**, before the later
hazard classes (2/3's ``LibraryScreen.<x>(self)`` family and everything after
it) trimmed it to 185. Building the ``self.<name>`` reference graph over
those 196 yields **one connected component of 135 names, one of 9 (the
note-import canvas handlers), one of 2, and 50 isolated singletons**
(135+9+2+50 = 196); adding an edge between any two of them that touch the
same ``LibraryNotesState`` field collapses that to **one component of 154
plus one of 9, one of 2 and 31 singletons** (154+9+2+31 = 196).

**Re-derived over the FINAL 185 movers with the identical instrument** (a
``self.<attr>`` walk PLUS the ``getattr(self, "<literal>")`` spelling, since
that is the same spelling the binding census has to honour): **126 + 9 + 2 +
48 singletons**, and **146 + 9 + 2 + 28** once shared-field edges are added.
An independent re-derivation that counts only plain ``self.<attr>`` edges,
without the literal-``getattr`` spelling, measures **142 + 9 + 3 + 2 + 29**;
both are recorded because the difference is entirely that one spelling, and
both say the same thing about the seam. **The verdict is unaffected by which
set or which instrument is used** -- every profile is one dominant component,
the same 9-member note-import group, one 2-member pair, and singletons. The
only candidate seam of any size is that 9-member
note-import group -- but four of the five ``LibraryNoteImportCanvas``
handlers outside it (``...add_source``, ``...cancel``, ``...collision_name``,
``...page``) are singletons only because they touch nothing else, and every
one of the nine reaches the same ``_library_note_import_controller`` WIRING
instance and the same ``_library_note_import_*`` state fields, so splitting
them out would produce a second controller of ~13 methods sharing a wiring
handle and a state object with the first. That buys nothing the ratchet's
per-file governance (SS17) does not already give. The feared size did not
materialise either: the hazard exclusions below remove 100 of the 285
candidates, so the 185 that move carry **3,934 source lines** of body,
putting this file within a few hundred lines of
``library_media_controller.py``'s 4,630.

**100 of the 285 candidates excluded, not moved (185 move).** Counted as
DISJOINT classes, in the order applied, so the numbers sum. Classes 1-9
come from the static censuses; class 10 came from this task's own battery,
which recipe SS3 records as the EXPECTED shape of the work rather than a
smell ("a battery-found hazard shrinking the mover set legitimately amends
the RED tuple"):

1. **57 unbound-fake-self test-bypass exclusions** (recipe SS3's first
   documented shape). Found by an ``ast`` census over ALL of ``Tests/``
   (3,372 files; never a ``-k``-filtered subset, per the seventh shape's
   filter-blindness rule) for every reference to
   ``LibraryScreen.<cluster-name>`` -- as a direct
   ``LibraryScreen.<name>(fake, ...)`` call, as a
   ``types.MethodType(LibraryScreen.<name>, fake)`` binding, and as
   ``getattr(LibraryScreen, "<name>")`` with a quoted-string name (the
   spelling SS3 requires every census to search for independently of the call
   that consumes it). The population concentrates in the notes tree/folder
   navigator suites -- ``_execute_library_notes_tree_mutation`` alone has 14
   sites, ``_load_library_notes_tree_slice`` 11,
   ``_reconcile_library_notes_tree_mutation`` 11. All 57 stay on
   ``LibraryScreen``, UNMOVED, full-bodied -- the recipe's established
   accommodation.

2. **11 further in-file ``LibraryScreen.<name>(self, ...)`` TARGETS -- a
   shape no prior series in this program has had to exclude.** Twenty
   notes methods are called, from inside ``library_screen.py`` itself, as
   an UNBOUND class call with ``self`` passed explicitly
   (``LibraryScreen._sync_library_notes_tree_canvas_if_present(self)``, 11
   sites; ``LibraryScreen._load_library_notes_tree_slice(self, ...)``, 7;
   ``LibraryScreen._supersede_library_notes_navigation(self)``, 6; and 17
   more targets). Nine were already excluded under class 1; these 11 are
   the remainder. The shape is written that way precisely BECAUSE those
   methods are unit-tested against ``SimpleNamespace`` fakes, and it is
   undeferrable: if such a target became a one-line delegator, the call
   ``LibraryScreen._target(self)`` would run the delegator body with a
   ``self`` that is whatever object the caller handed it, reaching for a
   ``_notes_controller`` attribute that object does not have.

3. **8 further CALLERS of that same shape.** Twenty-four candidates spell an
   unbound ``LibraryScreen.<x>(self, ...)`` call in their own body; 16 were
   already excluded above, and these 8 are the remainder
   (``_begin_library_note_create``, ``_create_library_note``,
   ``_delete_library_note_claimed``, ``_discard_new_library_note``,
   ``_invalidate_library_notes_tree_for_unmount``,
   ``_request_library_notes_tree_initial_load``,
   ``_schedule_library_notes_tree_mutation``,
   ``handle_library_note_delete_confirm``). Moved, their ``self`` would be
   the CONTROLLER, and the screen-resident target would then run against an
   object that is not the screen -- the same class of silent identity
   substitution recipe SS3's sixth bypass shape records, one indirection out.
   Excluding caller and target together keeps the whole unbound-call family
   in one place.

4. **8 further instance-attribute-monkeypatch exclusions** (recipe SS3's
   second shape), on top of the 5 the classes above already cover. Each is
   patched on a REAL, ``__init__``-constructed ``LibraryScreen`` -- the
   receiver of every candidate site was resolved to its binding statement
   rather than assumed, which matters here because this cluster's tests use
   the identifier ``screen`` for both real screens and ``SimpleNamespace``
   fakes (``test_library_notes_rename_propagation_t31796.py``'s
   ``_wiring_screen`` and ``test_library_notes_reader.py``'s ``tree_screen``
   are fakes; ``test_library_inspection_admission.py``'s
   ``library``-fixture ``screen`` and ``test_library_shell.py``'s
   ``_active_library_screen(host)`` are real). The 13 real-screen names are
   ``_acquire_file_notes_transition``,
   ``_apply_library_notes_stage_visibility``, ``_arm_library_note_editor``,
   ``_capture_library_notes_focus_identity``, ``_flush_active_file_notes``,
   ``_flush_library_note_save``, ``_focus_library_notes_tree_after_page``,
   ``_push_library_note_import_picker``, ``_save_library_note``,
   ``_schedule_library_note_autosave``,
   ``_sync_library_notes_tree_canvas_if_present``,
   ``_transition_library_notes_presentation`` and
   ``refresh_notes_sync_runtime``. The governing rule is recipe SS3's
   OPENING one, not a narrower dependency test: a name a test patches on a
   real ``LibraryScreen`` keeps its whole in-cluster call graph
   screen-routed until that subsystem's cleanup PR retargets the fixtures.
   Under a mover-caller map re-derived at the parent tree, **6 of the 13 do
   have a direct MOVER calling them as ``self.<name>(...)``**
   (``_apply_library_notes_stage_visibility``, ``_arm_library_note_editor``,
   ``_capture_library_notes_focus_identity``, ``_flush_library_note_save``,
   ``_save_library_note``, ``_schedule_library_note_autosave``), so for
   those the patch is directly bypassable by a moved body; the other 7 are
   held by the conservative rule rather than by a demonstrated bypass. Both
   halves are stated, per SS3's own "an exclusion's justification is a claim"
   rule.

5. **5 not-notes-owned exclusions** -- substring false positives of the
   ``"note"`` census, each owned by another subsystem by name and by
   caller: ``_resolve_library_export_chachanotes_db`` (Export; the DB name
   ``chachanotes`` contains the substring) and the four Collections
   quick-capture methods ``link_library_collection_capture_note``,
   ``retain_library_collection_quick_capture_note``,
   ``save_library_collection_capture_note`` and
   ``unlink_library_collection_capture_note``, all four of which
   ``Tests/Architecture/test_library_collections_wiring.py`` already pins by
   name as Collections cluster members.

6. **4 further members of the ``_library_note_session`` projection-property
   family.** ``LibraryScreen`` carries exactly five ``@property`` members in
   this cluster, defined as one contiguous block at ``library_screen.py``
   ``:3944-3985``: ``_library_note_detail``, ``_library_note_version``,
   ``_library_note_dirty``, ``_library_note_conflict_snapshot`` and
   ``_library_note_preview_snapshot``. Every body is a 4-to-15-line
   read-only projection of ``self._library_note_session.snapshot`` -- the
   WIRING coordinator instance task 1 deliberately kept OFF
   ``LibraryNotesState``. ``_library_note_dirty`` is independently excluded
   as a class-monkeypatch (class 7), so moving the other four would split a
   five-member family across two homes for no gain -- the One Home Rule
   ruling the media series applied to its own review-set family. Movers that
   read ``_library_note_dirty``/``_library_note_version`` get getter-only
   accessors, exactly as any other shared shell read.

7. **3 shared-shell-helper exclusions, by the >=2-subsystems rule applied to
   a METHOD rather than a field** -- the ``_sanitize_media_field``
   precedent, three times over. ``_sanitize_note_content`` is called by
   ``_run_library_prompts_import`` and ``_stage_library_prompt_for_console``
   and is bound by BOTH ``library_prompts_controller.py`` and
   ``library_skills_controller.py``; ``_library_note_keywords_from_input``
   is called by ``_write_library_prompt_export_file`` and is bound by
   ``library_prompts_controller.py``;
   ``_sync_library_notes_reader_layout_from_shell`` is called by the
   multi-subsystem pane dispatcher ``_toggle_library_media_reader_pane``, is
   bound by ``library_media_controller.py``, and is called by the shared
   ``_sync_library_canvas`` dispatcher off the bare ``self`` this cluster's
   own movers hand it. All three stay on the screen; this controller binds
   them the same way its siblings do.

8. **1 further class-monkeypatch exclusion** (recipe SS3's opening rule).
   ``_library_note_dirty`` (``monkeypatch.setattr(LibraryScreen, ...)`` at
   ``test_library_shell.py:2419`` and ``monkeypatch.setattr(type(screen),
   "_library_note_dirty", property(...))`` at ``:17652-17656``) is the only one of the three class-patched names
   (``_focus_library_notes_tree_after_page``,
   ``_refresh_library_note_detail``) not already covered above.
   ``_refresh_library_note_detail`` is additionally one of the FOUR names
   recipe SS3 records as permanently screen-routed shared shell
   infrastructure.

9. **1 module-globals-coupling exclusion** (recipe SS3's oldest shape,
   restated as the eighth numbered shape's mechanical census). The census
   ran to completion: all **82 bare module-global names** the 185 mover
   bodies read, crossed against every ``library_screen``-scoped patch shape
   (direct-attribute, fully-qualified string, two-argument
   ``monkeypatch.setattr``/``patch.object``) across ALL of ``Tests/``, under
   every alias tests actually import the module as -- ``library_screen``,
   ``library_screen_module``, ``library_module``, ``screen_module``, the
   list DERIVED by ``ast`` rather than assumed, per wave-5 task 3's own
   lesson. 10 names had hits; each hitting test was read:

   - ``validate_path_simple`` -> **ACTIVE** (3 sites, 2 files).
     ``test_library_shell.py::test_library_shell_note_write_export_file_
     rejects_invalid_path`` patches ``library_screen_module.
     validate_path_simple`` with a raising stub, then calls
     ``screen._write_library_note_export_file(...)`` on a REAL screen and
     asserts the destination file was NOT written and a warning fired.
     Moved, the body's bare name would resolve through THIS module's
     ``__globals__``, the real validator would accept the ``tmp_path``
     destination, and the file would be written.
     ``_write_library_note_export_file`` excluded; its one mover caller
     reaches it through a named late-binding dependency.
   - ``_sync_library_canvas`` -> **LATENT, kept** (33 sites across 10
     files; "site" per the ingest controller's own definition -- one match,
     one line, of the 3-shape pattern set, deduplicated by line number
     within a file). 31 call sites in 26 movers pass bare ``self`` here.
     Every one of the 33 sites was mapped to its enclosing test function;
     none patches the dispatcher in a way a notes mover's own call path
     observes. Same systemic shape and same verdict every sibling
     controller already carries.
   - ``_LibraryNotesRestoreGuard`` (3 sites), ``LIBRARY_ROW_BROWSE_NOTES``
     (2), ``LIBRARY_NOTES_SOURCE_DATABASE``, ``LIBRARY_NOTES_SOURCE_FILES``,
     ``NoteFlushOutcomeKind``, ``build_library_shell_state``,
     ``resolve_adaptive_reader_layout``, ``asyncio`` -> **LATENT**: every
     one is a plain module-attribute READ (a test constructing a guard for
     an assertion, or naming a constant), never a patch. ``asyncio``'s hit
     patches an attribute OF the shared ``asyncio`` module object, which is
     the same object in every importer.

10. **1 ``partial(LibraryScreen.<name>, self, ...)`` target -- a spelling of
    classes 2/3 that this task's FIRST census could not see, and that its own
    BATTERY caught rather than its static sweep.** The census behind classes 2
    and 3 walked every bare ``self`` ``ast.Name`` and asked what CALL it was a
    direct argument of; where that call's own ``func`` spelled
    ``LibraryScreen.<x>``, both ends were excluded. That question is answered
    "``partial``" for ``partial(LibraryScreen._restore_library_notes_browse_
    return_receipt, self, browse_receipt)`` -- three arguments, not four -- in
    ``handle_library_notes_filter``
    and ``handle_library_notes_filter_clear`` (both themselves excluded), so
    the TARGET was scored as an ordinary mover. It shipped into the paired
    sweep and produced 4 branch-unique failures, 2 of them
    ``AttributeError: 'types.SimpleNamespace' object has no attribute
    '_notes_controller'`` from
    ``test_library_notes_folder_navigator.py::test_clearing_filter_restores_
    same_epoch_browse_receipt_without_touching_ranges`` -- the delegator
    running with a fake ``self``.
    ``_restore_library_notes_browse_return_receipt`` is excluded; no mover
    calls it, so it needed no binding. **The standing correction, and it is
    both stronger and cheaper than enumerating call shapes: census EVERY
    ``LibraryScreen.<name>`` ``ast.Attribute`` reference anywhere in
    ``library_screen.py``, whatever expression encloses it.** That form is
    exhaustive by construction -- it cannot be defeated by ``partial``, by a
    variable assignment, by a list literal or by whatever wrapper the next
    refactor introduces. Run that way over the parent tree -- the WHOLE
    module, not only the class body -- it returns **27 targets**, of which 22
    are notes candidates and exactly one was a mover: this name. Both figures
    belong in the record: the direct-call-argument census found **20 of the
    22**, and the two it missed were ``_focus_library_notes_tree_after_page``
    (already excluded as a monkeypatch, so its miss was harmless and
    over-determined) and this one, which was not.

11. **1 test-bound-and-captured exclusion.** ``_exit_library_note_editor_
    guarded`` is awaited directly on a REAL screen by
    ``test_library_notes_reader.py:1532`` (``assert await screen._exit_
    library_note_editor_guarded() is False``, inside a test that also
    instance-patches ``screen._save_library_note``), and
    ``LibraryScreen.__init__`` injects it into ANOTHER controller as a named
    dependency (``library_screen.py:3223-3225``). It is held by recipe SS3's
    conservative OPENING rule -- **a name a test binds or captures keeps its
    call graph screen-routed until that subsystem's cleanup PR retargets the
    fixtures** -- not by a demonstrated bypass, and it is a MOVE candidate
    for a later series once those fixtures retarget, exactly as the media
    series disclosed for 7 of its own 16.

    **CORRECTION, review round -- the reason this exclusion originally
    carried was WRONG, and recipe SS3's "a wrong reason is worse than a thin
    one" is why it is rewritten here rather than quietly left standing.**
    The draft justified it as recipe SS3's TENTH shape (the media series'
    Form E, callback identity): that
    ``Tests/UI/test_screen_navigation.py:3225``'s ``focus_calls == [screen.
    _restore_library_notes_focus_identity]`` would compare against a
    CONTROLLER-bound method once the scheduling body moved. **It would not.**
    Read at the tree rather than inferred from the assertion text, the
    callback that assertion actually receives is ``finish_list_projection``
    -- a CLOSURE defined inside ``_exit_library_note_editor_guarded`` -- so
    the comparison never involved a bound method of anything, and no move
    could have changed which object it captures. **And the test is RED at the
    parent, on that exact assertion**, byte-identically (verified in an
    isolated worktree at ``afaf2320c``: ``At index 0 diff: <function
    LibraryScreen._exit_library_note_editor_guarded.<locals>.finish_list_
    projection> != <bound method ...._restore_library_notes_focus_
    identity>``). ``test_screen_navigation.py`` is on recipe SS7's documented
    list with a stable 32-name failure set across trees, so this red was
    never evidence about this move at all. The exclusion is retained -- on
    the rule above, which does govern -- and the Form-E mechanism claim is
    withdrawn. **Notes has ZERO Form-E callback-identity carriers**; the
    census that would find one (a test comparing a captured callback against
    ``<receiver>.<cluster-name>`` as a bare attribute) returned exactly this
    one candidate, and reading it disqualified it.

**The bare-``self`` census, both figures.** Recipe SS3's standing correction
after the media series' ``ancestors`` incident is to census EVERY bare
``self`` ``ast.Name`` in a moved body that is not the receiver of an
attribute access -- exhaustive by construction, rather than enumerating
comparison operators. Run over all **285 candidates** it returns **174
occurrences** in 74 methods; run over the **final 185 movers** it returns
**37**: 31 ``_sync_library_canvas(self, "notes", ...)`` duck-typed forwards
and 6 ``getattr(self, "<literal>")`` reads, every one of them bound. Both
figures are stated because only the pair shows the census working. The
difference is entirely classes 2 and 3 above -- the unbound
``LibraryScreen.<x>(self)`` family -- plus the 47 ``getattr(self, ...)``
reads inside methods excluded on other grounds. **Zero identity comparisons
(``is``/``is not``) and zero ``ancestors`` membership tests survive into the
mover set**: the screen carries 11 such sites, and every one of them lives
in a NON-candidate method (``_library_emergency_return_eligibility``,
``_library_ref_is_live``, ``on_descendant_focus``, and eight
media/prompts/skills-named methods). The four ``event.control is
self._library_file_notes_workspace`` comparisons that DO sit inside movers
compare against a ``LibraryNotesState`` FIELD, which resolves through this
controller's own shim loop to the identical object.

**Byte-for-byte canon** (moved bodies never edited -- every name they
reference that is not this controller's own state is rebound under the SAME
name, per the two binding kinds; see ``LibraryMediaController.__init__`` and
``ConsoleDictationController.__init__`` for the sibling worked examples).
The binding surface was derived MECHANICALLY, by walking all 185 moved
bodies for every ``self.<attr>`` load/store AND every ``getattr(self,
"<literal>")`` call, then subtracting this controller's own 100 state fields
and the movers themselves -- **99 names** -- and then adding **3 more the
walk cannot see**, for **102**, every one pinned in
``test_notes_controller_binds_every_name_its_moved_bodies_use``:

1. **Framework services** (``app``, ``app_instance``, ``call_after_refresh``,
   ``call_later``, ``focus_chain``, ``focused``, ``is_mounted``,
   ``is_running``, ``query``, ``query_one``, ``refresh``,
   ``register_footer_shortcuts``, ``run_worker``, ``set_focus``, ``watch``,
   and the base screen's ``_footer_shortcut_registration``) are live-read
   from the screen via ``@property`` on every access -- never snapshotted.
   ``focused`` is here for the reason the skills series established: movers
   reach it, and an unbound ``getattr(self, "focused", None)`` returns its
   default silently, forever.
2. **Everything else** is a NAMED constructor dependency: (b) 10 shared
   shell state accessors this cluster READS and never writes -- including
   two OTHER subsystems' state objects (``_media_state``, ``_prompts_state``,
   ``_rag_search_state``) that notes-named guards read without owning; (b')
   **6 it also WRITES** (``_library_canvas_resync_pending``,
   ``_library_navigation_context_generation``,
   ``_library_notes_programmatic_focus_target``,
   ``_library_notes_restoring_focus``, ``_library_selected_row_id``,
   ``_pending_library_source_open``) -- getter **and** setter. Two of those
   six are task 1's own BLOCKED fields: they carry a notes name and this
   cluster writes them, but ``library_media_controller.py`` binds each with a
   setter too, which is exactly why they stayed shell-owned; (c) 3 wiring
   accessors for the prior-extracted coordinator instances task 1 kept off
   ``LibraryNotesState`` (``_library_note_session`` -- 30 movers, this
   cluster's most-referenced name -- ``_library_notes_sync_controller`` at
   19 and ``_library_note_import_controller`` at 14); (a)/(e) 66 named
   late-binding callables for the general Library-wide shell helpers and for
   the exclusions above that a MOVER still calls internally -- each a
   ``lambda`` re-reading ``screen.<name>`` on every invocation, at CALL time,
   which is exactly why every bypass fixture named above keeps working
   unmodified after this move.

**The three bindings a ``self.<attr>`` walk cannot produce, and the
precedent that already knew about two of them.**
``_library_canvas_projection_depth`` and ``_library_canvas_resync_pending``
appear in NO moved body. They are read and written by the SHARED
``_sync_library_canvas`` dispatcher, off the bare ``self`` 26 movers hand it
-- so on the controller they resolve against the CONTROLLER, not the screen.
The first is read as ``getattr(screen,
"_library_canvas_projection_depth", 0)``: unbound, it returns 0 forever,
permanently disabling the projection-replay branch with no exception and no
red test anywhere in the standard battery. ``is_running`` is the third, for
the same reason (``screen.is_running`` at ``canvas_sync.py:535``). The
conversations and media controllers each found and bound the identical
``projection_depth``/``resync_pending`` pair, so this is a re-derivation of
a known lesson: **any controller whose movers forward bare ``self`` into
``_sync_library_canvas`` must read that dispatcher's own body for its
``kind``, because the recipe's ``self.<attr>`` census over the moved bodies
structurally cannot see it.** The full ``kind == "notes"`` path needs 16
names off the object it is handed; the other 13 are covered
(``query_one``, ``query``, ``refresh``, ``call_after_refresh``, ``app`` as
framework services; ``_library_notes_focus_intent_generation`` through this
controller's state shim loop; ``_library_notes_list_canvas_kwargs``,
``_library_note_work_pane_kwargs``,
``_restore_library_notes_after_targeted_sync`` as movers;
``_build_library_shell_input``, ``_library_selected_row_id``,
``_active_library_rail``, ``_library_lifecycle``,
``_library_onboarding_all_empty``, ``_capture_library_notes_focus_identity``
and ``_sync_library_notes_reader_layout_from_shell`` as named
dependencies). The sibling dispatcher ``_apply_library_row_toggle``, which
``handle_library_notes_row`` forwards into, is not a concern here:
``handle_library_notes_row`` is an unbound-fake-self exclusion and never
moves.

**Zero class-level constants relocated.** Unlike media, no moved body reads
a module-level assignment defined in ``library_screen.py``: all 82 bare
module globals the movers read are ordinary imports there, so this module
imports them from their own homes and ``library_screen.py``'s own import
surface is unchanged. In particular ``_LibraryNotesRestoreGuard`` and
``_LibraryNotesRecomposeCapture`` already live in ``screen_support_types.py``
-- had they been defined in ``library_screen.py`` this controller would have
needed a relocation, as the media series needed for its four
``_MEDIA_VIEW_*``/``_ANALYZE_ORIGIN_*`` constants.

**Construction order and import shape.** ``LibraryScreen.__init__`` builds
``self._notes_controller`` right after ``self._media_controller``, matching
every other controller in that file, and the controller class is imported
**function-locally**, inside ``__init__``'s established lazy-import block --
never at module level. That is a hard constraint of this wave, not a style
preference: ``library_screen.py`` is deliberately absent from the
``_ui_ready`` boot-budget snapshot
(``Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt``), and both
``Tests/Packaging/test_library_preimport_closure.py`` (whose deferred-suffix
tuple this commit extends with ``"notes"``) and
``Tests/Performance/test_ui_ready_module_census.py`` enforce it.

This subsystem's OWN state (every flat notes field name the moved bodies
reference -- all 100, across FOUR prefix families, resolved by task 1's
single-source ``notes_state_shim_attr()``) is exposed through a generated
property loop reading ``self._notes_state_accessor().<field>`` -- the same
generator shape task 1 installed on ``LibraryScreen`` (task 3 deletes that
screen block once this controller's own copy makes it dead).
"""
from __future__ import annotations

import asyncio
import dataclasses
import re
from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from loguru import logger
from textual import on
from textual.containers import Horizontal
from textual.css.query import NoMatches, QueryError
from textual.widget import Widget
from textual.widgets import Button, Input, Static, TextArea

from ...DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...config import get_cli_setting
from ...Library.library_browse_location import (
    claim_browse_directory,
    remember_browse_directory,
    validated_browse_directory,
)
from ...Library.library_export_scope import ExportScope
from ...Library.library_note_import_state import (
    LibraryNoteImportSnapshot,
    NoteImportPhase,
)
from ...Library.library_notes_lasting_sync_state import (
    LibraryNotesLastingSyncSnapshot,
)
from ...Library.library_notes_session import (
    ConflictAction,
    ConflictOutcomeKind,
    DestructiveAdmission,
    DestructiveAdmissionOutcomeKind,
    DestructiveKind,
    NoteFlushOutcomeKind,
    NoteSaveOutcome,
    NoteSaveOutcomeKind,
)
from ...Library.library_notes_state import (
    LibraryNoteDeleteReceipt,
    LibraryNoteEditorState,
    LibraryNotesFocusIdentity,
    LibraryNotesOperationState,
    build_library_note_editor_state,
    build_note_export_content,
    notes_autosave_status_text,
)
from ...Library.library_notes_tree_paging import NotesBranchKey
from ...Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_CREATE_NOTE,
    build_library_shell_state,
)
from ...Notes.note_folder_models import NoteFolder
from ...Notes.note_folder_repository import LocalNoteFolderRepository
from ...Third_Party.textual_fspicker import FileOpen, FileSave
from ...Utils.adaptive_reader_state import (
    AdaptiveReaderLayoutPreferences,
    PaneName,
    resolve_adaptive_reader_layout,
)
from ...Widgets.Library import (
    LibraryAdaptiveReaderShell,
    LibraryMediaCanvas,
    LibraryNoteWorkPane,
    LibraryNotesCanvas,
    LibraryRail,
)
from ...Widgets.Library.library_file_notes_events import (
    FileNotesEditableOpened,
    FileNotesIdentityCleared,
    FileNotesReloadConfirmationChanged,
    FileNotesRootChanged,
)
from ...Widgets.Library.library_note_import_canvas import LibraryNoteImportCanvas
from ...Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)
from ...Widgets.Library.library_notes_canvas import (
    LibraryNotePresentationState,
    resolve_database_note_status_channels,
)
from ...Widgets.Library.library_notes_sync_roots_canvas import (
    LibraryNotesSyncRootsCanvas,
)
from .canvas_sync import _sync_library_canvas
from .library_notes_state import LibraryNotesState, notes_state_shim_attr
from .library_notes_work_session import (
    NotesWorkSessionEvent,
    NotesWorkSessionPhase,
    reduce_notes_work_session,
)
from .screen_constants import (
    LIBRARY_FILE_NOTES_READER_PROFILE,
    LIBRARY_NOTES_COMPACT_BREAKPOINT,
    LIBRARY_NOTES_SOURCE_DATABASE,
    LIBRARY_NOTES_SOURCE_FILES,
    LIBRARY_NOTE_BLANK_SEED_TITLE,
)
from .screen_support_types import (
    LibraryEntryReconcileResult,
    _LibraryNotesRecomposeCapture,
    _LibraryNotesRestoreGuard,
)

if TYPE_CHECKING:
    from ...Notes.note_import_receipts import NoteImportReceiptRepository
    from ..Screens.library_screen import LibraryScreen


#: Commits one folder-tree mutation: patches the branches the operation
#: touched, reloads exactly those slices, and settles the selection.
#: ``(operation, payload, *, before, result, partial, destination_membership)``
#: -- see ``LibraryScreen._reconcile_library_notes_tree_mutation``, the only
#: implementation, which every screen wiring this controller must match.
LibraryNotesTreeMutationReconciler = Callable[..., Awaitable[None]]


class LibraryNotesController:
    """Owns the Library Notes cluster (185 methods).

    Holds no state of its own beyond what it reads and writes through
    ``LibraryNotesState`` (via the injected accessor) and the shared
    shell/framework/wiring bindings below. ``LibraryScreen`` constructs
    exactly one of these, in ``__init__`` right after
    ``self._media_controller``, and keeps a one-line delegator for every one
    of the 185 original names this cluster moved -- task 3 (the cleanup PR,
    notes series 3/N) prunes the ones nothing external reaches.
    """

    def __init__(
        self,
        screen: "LibraryScreen",
        *,
        notes_state_accessor,
        # -- shared shell state this cluster READS (group (b), getter-only).
        library_canvas_projection_depth_accessor,
        library_lifecycle_accessor,
        library_note_dirty_accessor,
        library_note_version_accessor,
        library_onboarding_all_empty_accessor,
        library_rail_collapsed_accessor,
        library_snapshot_state_generation_accessor,
        local_source_counts_accessor,
        local_source_records_accessor,
        media_state_accessor,
        prompts_state_accessor,
        rag_search_state_accessor,
        # -- shared shell state this cluster also WRITES (group (b'),
        #    getter + setter; the mechanical binding census found exactly
        #    6 such names.
        library_canvas_resync_pending_accessor,
        set_library_canvas_resync_pending,
        library_navigation_context_generation_accessor,
        set_library_navigation_context_generation,
        library_notes_programmatic_focus_target_accessor,
        set_library_notes_programmatic_focus_target,
        library_notes_restoring_focus_accessor,
        set_library_notes_restoring_focus,
        library_selected_row_id_accessor,
        set_library_selected_row_id,
        pending_library_source_open_accessor,
        set_pending_library_source_open,
        # -- the 3 prior-extracted notes WIRING coordinator instances task 1
        #    deliberately kept off `LibraryNotesState` (group (c)).
        library_note_import_controller_accessor,
        library_note_session_accessor,
        library_notes_sync_controller_accessor,
        # -- screen-resident methods a moved body still calls: general
        #    Library-wide shell helpers and named late-binding callables
        #    for this cluster's own exclusions (groups (a) and (e)).
        acknowledge_library_destination_change,
        active_library_rail,
        advance_library_stage_interaction,
        apply_library_emergency_geometry,
        apply_library_notes_stage_visibility,
        apply_library_notes_stage_visibility_for_resize,
        arm_library_list_entry_focus,
        arm_library_note_editor,
        begin_library_note_create,
        build_library_notes_state,
        build_library_notes_tree_projection,
        build_library_shell_input,
        capture_library_notes_browse_return_receipt,
        capture_library_notes_focus_identity,
        compose_library_rail_top_action,
        compose_workspaces_rail_body,
        create_library_note,
        delete_library_note_claimed,
        discard_new_library_note,
        exit_library_note_editor_guarded,
        file_notes_active,
        flush_library_note_save,
        focus_library_control,
        hide_library_adaptive_reader_rail_collapse,
        invalidate_library_workspace_depth_state,
        library_compose_scoped_ref,
        library_entry_route_key,
        library_header_line,
        library_layout_ref,
        library_note_keywords_from_input,
        library_note_template_fields,
        library_notes_compact_stage_applies,
        library_notes_compact_workflow_active,
        library_notes_focused_task_active,
        library_notes_footer_shortcuts,
        library_notes_restore_guard_is_current,
        library_notes_workflow_active,
        library_rail_preferences,
        library_rail_search_placeholder,
        library_rail_search_value,
        library_workspace_depth_state,
        locate_library_notes_tree_target,
        open_library_export_canvas,
        patch_library_note_list_from_session,
        project_library_media_stage_classes,
        push_library_note_import_picker,
        reconcile_library_notes_tree_mutation: LibraryNotesTreeMutationReconciler,
        refresh_library_note_detail,
        refresh_local_source_snapshot,
        register_footer_shortcuts,
        replace_library_canvas_child,
        request_library_notes_tree_initial_load,
        request_library_notes_tree_slice,
        return_to_library_database_notes,
        run_library_note_import_execution,
        run_library_service_call,
        save_library_note,
        schedule_library_note_autosave,
        select_library_rail_row,
        set_library_destination_with_conversation_fence,
        source_record_id,
        supersede_library_notes_navigation,
        sync_library_ingest_rail_for_width,
        sync_library_notes_reader_layout_from_shell,
        sync_library_ordinary_rail_width_contract,
        transition_library_notes_presentation,
        write_library_note_export_file,
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the 185 method bodies below is a byte-for-byte copy of
        the pre-extraction ``LibraryScreen`` method: no internal line was
        edited to retarget a call or an attribute. That is possible because
        this constructor binds every name those bodies reference that is not
        this controller's own state, under the SAME name the original method
        used. See the module docstring for the binding kinds this follows and
        the full per-parameter derivation.

        Args:
            screen: The Library screen. Used ONLY for the sixteen framework
                services below (``app``, ``app_instance``,
                ``call_after_refresh``, ``call_later``, ``focus_chain``,
                ``focused``, ``is_mounted``, ``is_running``, ``query``,
                ``query_one``, ``refresh``, ``register_footer_shortcuts``,
                ``run_worker``, ``set_focus``, ``watch``, and the base
                screen's own ``_footer_shortcut_registration``) -- this
                cluster owns no DOM of its own.
            notes_state_accessor: Returns the live ``LibraryNotesState``
                (``LibraryScreen._notes_state``, task 1). Backs every
                generated flat notes-field property below.
        """
        self._screen = screen
        self._notes_state_accessor = notes_state_accessor
        self._library_canvas_projection_depth_accessor = library_canvas_projection_depth_accessor
        self._library_lifecycle_accessor = library_lifecycle_accessor
        self._library_note_dirty_accessor = library_note_dirty_accessor
        self._library_note_version_accessor = library_note_version_accessor
        self._library_onboarding_all_empty_accessor = library_onboarding_all_empty_accessor
        self._library_rail_collapsed_accessor = library_rail_collapsed_accessor
        self._library_snapshot_state_generation_accessor = library_snapshot_state_generation_accessor
        self._local_source_counts_accessor = local_source_counts_accessor
        self._local_source_records_accessor = local_source_records_accessor
        self._media_state_accessor = media_state_accessor
        self._prompts_state_accessor = prompts_state_accessor
        self._rag_search_state_accessor = rag_search_state_accessor
        self._library_canvas_resync_pending_accessor = library_canvas_resync_pending_accessor
        self._set_library_canvas_resync_pending_fn = set_library_canvas_resync_pending
        self._library_navigation_context_generation_accessor = library_navigation_context_generation_accessor
        self._set_library_navigation_context_generation_fn = set_library_navigation_context_generation
        self._library_notes_programmatic_focus_target_accessor = library_notes_programmatic_focus_target_accessor
        self._set_library_notes_programmatic_focus_target_fn = set_library_notes_programmatic_focus_target
        self._library_notes_restoring_focus_accessor = library_notes_restoring_focus_accessor
        self._set_library_notes_restoring_focus_fn = set_library_notes_restoring_focus
        self._library_selected_row_id_accessor = library_selected_row_id_accessor
        self._set_library_selected_row_id_fn = set_library_selected_row_id
        self._pending_library_source_open_accessor = pending_library_source_open_accessor
        self._set_pending_library_source_open_fn = set_pending_library_source_open
        self._library_note_import_controller_accessor = library_note_import_controller_accessor
        self._library_note_session_accessor = library_note_session_accessor
        self._library_notes_sync_controller_accessor = library_notes_sync_controller_accessor
        self._acknowledge_library_destination_change_fn = acknowledge_library_destination_change
        self._active_library_rail_fn = active_library_rail
        self._advance_library_stage_interaction_fn = advance_library_stage_interaction
        self._apply_library_emergency_geometry_fn = apply_library_emergency_geometry
        self._apply_library_notes_stage_visibility_fn = apply_library_notes_stage_visibility
        self._apply_library_notes_stage_visibility_for_resize_fn = apply_library_notes_stage_visibility_for_resize
        self._arm_library_list_entry_focus_fn = arm_library_list_entry_focus
        self._arm_library_note_editor_fn = arm_library_note_editor
        self._begin_library_note_create_fn = begin_library_note_create
        self._build_library_notes_state_fn = build_library_notes_state
        self._build_library_notes_tree_projection_fn = build_library_notes_tree_projection
        self._build_library_shell_input_fn = build_library_shell_input
        self._capture_library_notes_browse_return_receipt_fn = capture_library_notes_browse_return_receipt
        self._capture_library_notes_focus_identity_fn = capture_library_notes_focus_identity
        self._compose_library_rail_top_action_fn = compose_library_rail_top_action
        self._compose_workspaces_rail_body_fn = compose_workspaces_rail_body
        self._create_library_note_fn = create_library_note
        self._delete_library_note_claimed_fn = delete_library_note_claimed
        self._discard_new_library_note_fn = discard_new_library_note
        self._exit_library_note_editor_guarded_fn = exit_library_note_editor_guarded
        self._file_notes_active_fn = file_notes_active
        self._flush_library_note_save_fn = flush_library_note_save
        self._focus_library_control_fn = focus_library_control
        self._hide_library_adaptive_reader_rail_collapse_fn = hide_library_adaptive_reader_rail_collapse
        self._invalidate_library_workspace_depth_state_fn = invalidate_library_workspace_depth_state
        self._library_compose_scoped_ref_fn = library_compose_scoped_ref
        self._library_entry_route_key_fn = library_entry_route_key
        self._library_header_line_fn = library_header_line
        self._library_layout_ref_fn = library_layout_ref
        self._library_note_keywords_from_input_fn = library_note_keywords_from_input
        self._library_note_template_fields_fn = library_note_template_fields
        self._library_notes_compact_stage_applies_fn = library_notes_compact_stage_applies
        self._library_notes_compact_workflow_active_fn = library_notes_compact_workflow_active
        self._library_notes_focused_task_active_fn = library_notes_focused_task_active
        self._library_notes_footer_shortcuts_fn = library_notes_footer_shortcuts
        self._library_notes_restore_guard_is_current_fn = library_notes_restore_guard_is_current
        self._library_notes_workflow_active_fn = library_notes_workflow_active
        self._library_rail_preferences_fn = library_rail_preferences
        self._library_rail_search_placeholder_fn = library_rail_search_placeholder
        self._library_rail_search_value_fn = library_rail_search_value
        self._library_workspace_depth_state_fn = library_workspace_depth_state
        self._locate_library_notes_tree_target_fn = locate_library_notes_tree_target
        self._open_library_export_canvas_fn = open_library_export_canvas
        self._patch_library_note_list_from_session_fn = patch_library_note_list_from_session
        self._project_library_media_stage_classes_fn = project_library_media_stage_classes
        self._push_library_note_import_picker_fn = push_library_note_import_picker
        self._reconcile_library_notes_tree_mutation_fn = (
            reconcile_library_notes_tree_mutation
        )
        self._refresh_library_note_detail_fn = refresh_library_note_detail
        self._refresh_local_source_snapshot_fn = refresh_local_source_snapshot
        self._register_footer_shortcuts_fn = register_footer_shortcuts
        self._replace_library_canvas_child_fn = replace_library_canvas_child
        self._request_library_notes_tree_initial_load_fn = request_library_notes_tree_initial_load
        self._request_library_notes_tree_slice_fn = request_library_notes_tree_slice
        self._return_to_library_database_notes_fn = return_to_library_database_notes
        self._run_library_note_import_execution_fn = run_library_note_import_execution
        self._run_library_service_call_fn = run_library_service_call
        self._save_library_note_fn = save_library_note
        self._schedule_library_note_autosave_fn = schedule_library_note_autosave
        self._select_library_rail_row_fn = select_library_rail_row
        self._set_library_destination_with_conversation_fence_fn = set_library_destination_with_conversation_fence
        self._source_record_id_fn = source_record_id
        self._supersede_library_notes_navigation_fn = supersede_library_notes_navigation
        self._sync_library_ingest_rail_for_width_fn = sync_library_ingest_rail_for_width
        self._sync_library_notes_reader_layout_from_shell_fn = sync_library_notes_reader_layout_from_shell
        self._sync_library_ordinary_rail_width_contract_fn = sync_library_ordinary_rail_width_contract
        self._transition_library_notes_presentation_fn = transition_library_notes_presentation
        self._write_library_note_export_file_fn = write_library_note_export_file

    # -- framework services: live-read properties, never snapshotted ------

    @property
    def _footer_shortcut_registration(self) -> Any:
        return self._screen._footer_shortcut_registration

    @property
    def app(self) -> Any:
        return self._screen.app

    @property
    def app_instance(self) -> Any:
        return self._screen.app_instance

    @property
    def call_after_refresh(self) -> Any:
        return self._screen.call_after_refresh

    @property
    def call_later(self) -> Any:
        return self._screen.call_later

    @property
    def focus_chain(self) -> Any:
        return self._screen.focus_chain

    @property
    def focused(self) -> Any:
        return self._screen.focused

    @property
    def is_mounted(self) -> Any:
        return self._screen.is_mounted

    @property
    def is_running(self) -> Any:
        return self._screen.is_running

    @property
    def query(self) -> Any:
        return self._screen.query

    @property
    def query_one(self) -> Any:
        return self._screen.query_one

    @property
    def refresh(self) -> Any:
        return self._screen.refresh

    @property
    def register_footer_shortcuts(self) -> Any:
        return self._screen.register_footer_shortcuts

    @property
    def run_worker(self) -> Any:
        return self._screen.run_worker

    @property
    def set_focus(self) -> Any:
        return self._screen.set_focus

    @property
    def watch(self) -> Any:
        return self._screen.watch


    # -- shared shell state, read-only (group (b)) -------------------------

    @property
    def _library_canvas_projection_depth(self) -> Any:
        return self._library_canvas_projection_depth_accessor()

    @property
    def _library_lifecycle(self) -> Any:
        return self._library_lifecycle_accessor()

    @property
    def _library_note_dirty(self) -> Any:
        return self._library_note_dirty_accessor()

    @property
    def _library_note_version(self) -> Any:
        return self._library_note_version_accessor()

    @property
    def _library_onboarding_all_empty(self) -> Any:
        return self._library_onboarding_all_empty_accessor()

    @property
    def _library_rail_collapsed(self) -> Any:
        return self._library_rail_collapsed_accessor()

    @property
    def _library_snapshot_state_generation(self) -> Any:
        return self._library_snapshot_state_generation_accessor()

    @property
    def _local_source_counts(self) -> Any:
        return self._local_source_counts_accessor()

    @property
    def _local_source_records(self) -> Any:
        return self._local_source_records_accessor()

    @property
    def _media_state(self) -> Any:
        return self._media_state_accessor()

    @property
    def _notes_state(self) -> Any:
        """This cluster's OWN state object, under the screen's own name.

        (wave-8 task 3.) No moved body spells ``self._notes_state`` -- every
        one of them reads a flat ``_library_notes_<field>`` property from the
        generated shim loop at the bottom of this file, and that is how the
        byte-for-byte canon keeps them unedited. This accessor exists for the
        SHARED dispatchers in ``canvas_sync.py``, which are handed a bare
        ``self`` by 26 of this cluster's methods (31 call sites), and so have
        to resolve the notes state on EITHER receiver: the screen
        (``LibraryScreen._notes_state``) or this controller. Without it, the
        dotted spellings the cleanup introduced (``_notes_state.row_selection`` in
        ``_apply_library_row_toggle``, ``_notes_state.focus_intent_generation``
        in ``_sync_library_canvas``) resolve on the screen and raise on the
        controller -- an ``AttributeError`` the dispatcher's own
        ``except Exception`` swallows into a full-screen recompose, with no
        exception and no red test. See this series' task-3 report for the
        measured precedent where exactly that happened.
        """
        return self._notes_state_accessor()

    @property
    def _prompts_state(self) -> Any:
        return self._prompts_state_accessor()

    @property
    def _rag_search_state(self) -> Any:
        return self._rag_search_state_accessor()


    # -- shared shell state this cluster also writes (group (b')) ----------

    @property
    def _library_canvas_resync_pending(self) -> Any:
        return self._library_canvas_resync_pending_accessor()

    @_library_canvas_resync_pending.setter
    def _library_canvas_resync_pending(self, value: Any) -> None:
        self._set_library_canvas_resync_pending_fn(value)

    @property
    def _library_navigation_context_generation(self) -> Any:
        return self._library_navigation_context_generation_accessor()

    @_library_navigation_context_generation.setter
    def _library_navigation_context_generation(self, value: Any) -> None:
        self._set_library_navigation_context_generation_fn(value)

    @property
    def _library_notes_programmatic_focus_target(self) -> Any:
        return self._library_notes_programmatic_focus_target_accessor()

    @_library_notes_programmatic_focus_target.setter
    def _library_notes_programmatic_focus_target(self, value: Any) -> None:
        self._set_library_notes_programmatic_focus_target_fn(value)

    @property
    def _library_notes_restoring_focus(self) -> Any:
        return self._library_notes_restoring_focus_accessor()

    @_library_notes_restoring_focus.setter
    def _library_notes_restoring_focus(self, value: Any) -> None:
        self._set_library_notes_restoring_focus_fn(value)

    @property
    def _library_selected_row_id(self) -> Any:
        return self._library_selected_row_id_accessor()

    @_library_selected_row_id.setter
    def _library_selected_row_id(self, value: Any) -> None:
        self._set_library_selected_row_id_fn(value)

    @property
    def _pending_library_source_open(self) -> Any:
        return self._pending_library_source_open_accessor()

    @_pending_library_source_open.setter
    def _pending_library_source_open(self, value: Any) -> None:
        self._set_pending_library_source_open_fn(value)


    # -- prior-extracted notes wiring coordinators (group (c)) -------------

    @property
    def _library_note_import_controller(self) -> Any:
        return self._library_note_import_controller_accessor()

    @property
    def _library_note_session(self) -> Any:
        return self._library_note_session_accessor()

    @property
    def _library_notes_sync_controller(self) -> Any:
        return self._library_notes_sync_controller_accessor()


    # -- screen-resident callables, late-bound (groups (a) and (e)) --------

    @property
    def _acknowledge_library_destination_change(self) -> Any:
        return self._acknowledge_library_destination_change_fn

    @property
    def _active_library_rail(self) -> Any:
        return self._active_library_rail_fn

    @property
    def _advance_library_stage_interaction(self) -> Any:
        return self._advance_library_stage_interaction_fn

    @property
    def _apply_library_emergency_geometry(self) -> Any:
        return self._apply_library_emergency_geometry_fn

    @property
    def _apply_library_notes_stage_visibility(self) -> Any:
        return self._apply_library_notes_stage_visibility_fn

    @property
    def _apply_library_notes_stage_visibility_for_resize(self) -> Any:
        return self._apply_library_notes_stage_visibility_for_resize_fn

    @property
    def _arm_library_list_entry_focus(self) -> Any:
        return self._arm_library_list_entry_focus_fn

    @property
    def _arm_library_note_editor(self) -> Any:
        return self._arm_library_note_editor_fn

    @property
    def _begin_library_note_create(self) -> Any:
        return self._begin_library_note_create_fn

    @property
    def _build_library_notes_state(self) -> Any:
        return self._build_library_notes_state_fn

    @property
    def _build_library_notes_tree_projection(self) -> Any:
        return self._build_library_notes_tree_projection_fn

    @property
    def _build_library_shell_input(self) -> Any:
        return self._build_library_shell_input_fn

    @property
    def _capture_library_notes_browse_return_receipt(self) -> Any:
        return self._capture_library_notes_browse_return_receipt_fn

    @property
    def _capture_library_notes_focus_identity(self) -> Any:
        return self._capture_library_notes_focus_identity_fn

    @property
    def _compose_library_rail_top_action(self) -> Any:
        return self._compose_library_rail_top_action_fn

    @property
    def _compose_workspaces_rail_body(self) -> Any:
        return self._compose_workspaces_rail_body_fn

    @property
    def _create_library_note(self) -> Any:
        return self._create_library_note_fn

    @property
    def _delete_library_note_claimed(self) -> Any:
        return self._delete_library_note_claimed_fn

    @property
    def _discard_new_library_note(self) -> Any:
        return self._discard_new_library_note_fn

    @property
    def _exit_library_note_editor_guarded(self) -> Any:
        return self._exit_library_note_editor_guarded_fn

    @property
    def _file_notes_active(self) -> Any:
        return self._file_notes_active_fn

    @property
    def _flush_library_note_save(self) -> Any:
        return self._flush_library_note_save_fn

    @property
    def _focus_library_control(self) -> Any:
        return self._focus_library_control_fn

    @property
    def _hide_library_adaptive_reader_rail_collapse(self) -> Any:
        return self._hide_library_adaptive_reader_rail_collapse_fn

    @property
    def _invalidate_library_workspace_depth_state(self) -> Any:
        return self._invalidate_library_workspace_depth_state_fn

    @property
    def _library_compose_scoped_ref(self) -> Any:
        return self._library_compose_scoped_ref_fn

    @property
    def _library_entry_route_key(self) -> Any:
        return self._library_entry_route_key_fn

    @property
    def _library_header_line(self) -> Any:
        return self._library_header_line_fn

    @property
    def _library_layout_ref(self) -> Any:
        return self._library_layout_ref_fn

    @property
    def _library_note_keywords_from_input(self) -> Any:
        return self._library_note_keywords_from_input_fn

    @property
    def _library_note_template_fields(self) -> Any:
        return self._library_note_template_fields_fn

    @property
    def _library_notes_compact_stage_applies(self) -> Any:
        return self._library_notes_compact_stage_applies_fn

    @property
    def _library_notes_compact_workflow_active(self) -> Any:
        return self._library_notes_compact_workflow_active_fn

    @property
    def _library_notes_focused_task_active(self) -> Any:
        return self._library_notes_focused_task_active_fn

    @property
    def _library_notes_footer_shortcuts(self) -> Any:
        return self._library_notes_footer_shortcuts_fn

    @property
    def _library_notes_restore_guard_is_current(self) -> Any:
        return self._library_notes_restore_guard_is_current_fn

    @property
    def _library_notes_workflow_active(self) -> Any:
        return self._library_notes_workflow_active_fn

    @property
    def _library_rail_preferences(self) -> Any:
        return self._library_rail_preferences_fn

    @property
    def _library_rail_search_placeholder(self) -> Any:
        return self._library_rail_search_placeholder_fn

    @property
    def _library_rail_search_value(self) -> Any:
        return self._library_rail_search_value_fn

    @property
    def _library_workspace_depth_state(self) -> Any:
        return self._library_workspace_depth_state_fn

    @property
    def _locate_library_notes_tree_target(self) -> Any:
        return self._locate_library_notes_tree_target_fn

    @property
    def _open_library_export_canvas(self) -> Any:
        return self._open_library_export_canvas_fn

    @property
    def _patch_library_note_list_from_session(self) -> Any:
        return self._patch_library_note_list_from_session_fn

    @property
    def _project_library_media_stage_classes(self) -> Any:
        return self._project_library_media_stage_classes_fn

    @property
    def _push_library_note_import_picker(self) -> Any:
        return self._push_library_note_import_picker_fn

    @property
    def _reconcile_library_notes_tree_mutation(
        self,
    ) -> LibraryNotesTreeMutationReconciler:
        """The screen's folder-tree mutation reconciler (task-32124).

        Returns:
            The awaitable injected at construction; see
            ``LibraryNotesTreeMutationReconciler`` for its contract.
        """
        return self._reconcile_library_notes_tree_mutation_fn

    @property
    def _refresh_library_note_detail(self) -> Any:
        return self._refresh_library_note_detail_fn

    @property
    def _refresh_local_source_snapshot(self) -> Any:
        return self._refresh_local_source_snapshot_fn

    @property
    def _register_footer_shortcuts(self) -> Any:
        return self._register_footer_shortcuts_fn

    @property
    def _replace_library_canvas_child(self) -> Any:
        return self._replace_library_canvas_child_fn

    @property
    def _request_library_notes_tree_initial_load(self) -> Any:
        return self._request_library_notes_tree_initial_load_fn

    @property
    def _request_library_notes_tree_slice(self) -> Any:
        return self._request_library_notes_tree_slice_fn

    @property
    def _return_to_library_database_notes(self) -> Any:
        return self._return_to_library_database_notes_fn

    @property
    def _run_library_note_import_execution(self) -> Any:
        return self._run_library_note_import_execution_fn

    @property
    def _run_library_service_call(self) -> Any:
        return self._run_library_service_call_fn

    @property
    def _save_library_note(self) -> Any:
        return self._save_library_note_fn

    @property
    def _schedule_library_note_autosave(self) -> Any:
        return self._schedule_library_note_autosave_fn

    @property
    def _select_library_rail_row(self) -> Any:
        return self._select_library_rail_row_fn

    @property
    def _set_library_destination_with_conversation_fence(self) -> Any:
        return self._set_library_destination_with_conversation_fence_fn

    @property
    def _source_record_id(self) -> Any:
        return self._source_record_id_fn

    @property
    def _supersede_library_notes_navigation(self) -> Any:
        return self._supersede_library_notes_navigation_fn

    @property
    def _sync_library_ingest_rail_for_width(self) -> Any:
        return self._sync_library_ingest_rail_for_width_fn

    @property
    def _sync_library_notes_reader_layout_from_shell(self) -> Any:
        return self._sync_library_notes_reader_layout_from_shell_fn

    @property
    def _sync_library_ordinary_rail_width_contract(self) -> Any:
        return self._sync_library_ordinary_rail_width_contract_fn

    @property
    def _transition_library_notes_presentation(self) -> Any:
        return self._transition_library_notes_presentation_fn

    @property
    def _write_library_note_export_file(self) -> Any:
        return self._write_library_note_export_file_fn

    # -- moved cluster methods (185), byte-for-byte, original file order --
    def _library_note_editor_state(self) -> LibraryNoteEditorState | None:
        """Project the one canonical session snapshot into incumbent canvas state."""
        snapshot = self._library_note_session.snapshot
        if snapshot is None:
            return None
        baseline_state = build_library_note_editor_state(
            {
                "id": snapshot.note_id,
                "title": snapshot.baseline.title,
                "content": snapshot.baseline.body,
                "keywords": snapshot.baseline.keywords,
                "version": snapshot.version,
                "created_at": snapshot.baseline.created_at,
                "last_modified": snapshot.baseline.modified_at,
            }
        )
        if snapshot.in_conflict:
            status = notes_autosave_status_text(
                "conflict", word_count=self._note_word_count(snapshot.body)
            )
            if snapshot.status_message != "Conflict — review the choices below.":
                status = f"{status} · {snapshot.status_message}"
        else:
            status = snapshot.status_message
        meta_line = baseline_state.meta_line
        if status:
            meta_line = f"{meta_line} · {status}" if meta_line else status
        return LibraryNoteEditorState(
            note_id=snapshot.note_id,
            title=snapshot.title,
            content=snapshot.body,
            keywords_text=snapshot.keywords_text,
            version=snapshot.version,
            meta_line=meta_line,
            has_note=True,
        )
    def _library_note_is_pending_blank(self) -> bool:
        """Whether the open note is still the untouched blank-GC candidate.

        Mirrors the condition ``_apply_library_note_presentation_state``
        already uses for ``title_placeholder_only`` -- a note this fresh
        was created (persisted) as a create-flow side effect, not because
        the user asked to keep anything, and gets silently deleted again if
        abandoned untouched (``_gc_pending_blank_note``). "Saved" is
        technically true and practically dishonest here: nothing the user
        typed is safe, because nothing has been typed yet.
        """
        return bool(
            self._library_note_pending_blank_gc_id
            and self._library_note_pending_blank_gc_id == self._selected_note_id
        )
    def _library_note_status_line(self) -> str:
        """Return the persistent, text-labeled save state for both regions."""
        snapshot = self._library_note_session.snapshot
        if snapshot is None:
            return "No note open"
        if self._library_note_shortcut_status:
            return self._library_note_shortcut_status
        # task-32133 AC#2: a blank note this fresh reads "Saved" before a
        # single character is typed -- true of the empty seed record, but
        # not of anything the user has asked to keep.
        if self._library_note_is_pending_blank():
            return "Draft — not saved yet"
        if self._library_note_autosave_state == "saving" or snapshot.saving:
            return "Saving…"
        if snapshot.in_conflict:
            return snapshot.status_message or "Conflict — review the choices below."
        if snapshot.status_message:
            return snapshot.status_message
        return "Unsaved changes" if snapshot.dirty else "Saved"
    def _library_note_presentation_state(self) -> LibraryNotePresentationState:
        """Project the canonical session into presentation-only canvas state."""
        snapshot = self._library_note_session.snapshot
        if snapshot is None:
            raise RuntimeError("A note presentation requires an active session.")
        base_meta = self._library_note_meta_base_line()
        word_count = self._note_word_count(snapshot.body)
        word_copy = f"{word_count} words" if word_count != 1 else "1 word"
        status_line = self._library_note_status_line()
        metadata_line = " · ".join(part for part in (base_meta, word_copy) if part)
        operation = self._library_notes_operation_for_active_region()
        status_channels = resolve_database_note_status_channels(
            conflict=snapshot.in_conflict,
            read_only=self._library_notes_select_mode,
            save_failed=self._library_note_autosave_state in {"error", "validation"},
            saving=snapshot.saving or self._library_note_autosave_state == "saving",
            dirty=snapshot.dirty,
        )
        if self._library_note_shortcut_status:
            status_channels = dataclasses.replace(
                status_channels,
                content_recovery=status_line,
                safe_next_action=None,
            )
        elif (
            snapshot.in_conflict
            or self._library_note_autosave_state in {"error", "validation"}
            or self._library_note_is_pending_blank()
        ):
            status_channels = dataclasses.replace(
                status_channels,
                content_recovery=status_line,
            )
        return LibraryNotePresentationState(
            snapshot=snapshot,
            metadata_line=metadata_line,
            status_line=status_line,
            region="context" if self._library_note_context else "editor",
            presentation="preview" if self._library_note_preview else "edit",
            compact=self._library_notes_compact,
            validation=self._library_note_autosave_state == "validation",
            conflict=snapshot.in_conflict,
            conflict_running=self._library_note_session.conflict_resolution_running,
            confirming_delete=self._library_note_confirming_delete,
            destructive_running=self._library_note_session.destructive_running,
            discard_new_note=(
                self._library_note_session.untouched_create_token is not None
            ),
            transfer_status=operation.status_line if operation is not None else "",
            transfer_running=bool(operation and operation.running),
            bulk_read_only=self._library_notes_select_mode,
            bulk_included=self._library_notes_row_selection.is_selected(
                snapshot.note_id
            ),
            status_channels=status_channels,
        )
    def _library_notes_active_region(
        self,
    ) -> Literal["navigator", "editor", "context", ""]:
        """Return the currently visible Notes operation-status region."""
        if (
            self._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
            and self._library_notes_view == "editor"
        ):
            return "context" if self._library_note_context else "editor"
        if (
            self._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
            and self._library_notes_view == "list"
        ):
            return "navigator"
        return ""
    def _library_notes_operation_for_active_region(
        self,
    ) -> LibraryNotesOperationState | None:
        operation = self._library_notes_operation
        if operation is None or operation.region != self._library_notes_active_region():
            return None
        return operation
    def _library_note_import_execution_active(self) -> bool:
        """Return whether reviewed import currently owns Notes mutations."""
        controller = getattr(self, "_library_note_import_controller", None)
        snapshot = getattr(controller, "snapshot", None)
        return getattr(snapshot, "phase", None) is NoteImportPhase.IMPORTING
    def _library_notes_mutation_fenced(self) -> bool:
        """Keep Database Notes single-writer while reviewed import executes."""
        return bool(
            getattr(self, "_library_notes_mutation_in_flight", False)
            or self._library_note_import_execution_active()
        )
    def _library_notes_operation_is_current_and_active(
        self, operation: LibraryNotesOperationState
    ) -> bool:
        """Return whether a running token still owns the visible region."""
        current = self._library_notes_operation
        return bool(
            current is not None
            and current.running
            and current.token == operation.token
            and current.kind == operation.kind
            and current.region == self._library_notes_active_region()
        )
    def _apply_library_notes_operation_state(self) -> None:
        """Synchronize the one active Notes operation status surface."""
        if not self.is_mounted:
            return
        if self._library_notes_active_region() == "navigator":
            _sync_library_canvas(self, "notes")
            return
        if self._library_notes_active_region() in {"editor", "context"}:
            try:
                canvas = self.query_one("#library-note-work-pane", LibraryNoteWorkPane)
            except (NoMatches, QueryError):
                return
            if canvas.mode != "editor":
                self.call_after_refresh(self._apply_library_notes_operation_state)
                return
            self._apply_library_note_presentation_state()
    def _begin_library_notes_operation(
        self, kind: Literal["import", "export", "copy", "console"]
    ) -> LibraryNotesOperationState | None:
        """Claim one typed transfer token before invoking an external seam."""
        if self._library_notes_mutation_fenced():
            return None
        current = self._library_notes_operation
        if current is not None and current.running:
            return None
        region = self._library_notes_active_region()
        if region not in {"navigator", "editor", "context"}:
            return None
        if kind not in {"import", "export", "copy", "console"}:
            return None
        self._library_notes_notice = ""
        self._library_notes_operation_counter += 1
        operation = LibraryNotesOperationState(
            kind=kind,
            token=self._library_notes_operation_counter,
            phase="running",
            region=region,
        )
        self._library_notes_operation = operation
        self._apply_library_notes_operation_state()
        return operation
    def _move_library_notes_operation(
        self,
        token: int,
        region: Literal["navigator", "editor", "context"],
    ) -> bool:
        """Move a current operation to the region its success opened."""
        operation = self._library_notes_operation
        if (
            operation is None
            or operation.token != token
            or region not in {"navigator", "editor", "context"}
        ):
            return False
        self._library_notes_operation = dataclasses.replace(operation, region=region)
        return True
    def _finish_library_notes_operation(
        self,
        operation: LibraryNotesOperationState,
        *,
        success: bool,
        completion_next_action: str = "",
        failure_next_action: str = "try again",
    ) -> bool:
        """Terminalize the current token and render only in its active region."""
        current = self._library_notes_operation
        active_region = self._library_notes_active_region()
        if (
            current is None
            or current.token != operation.token
            or current.kind != operation.kind
        ):
            return False
        terminal = dataclasses.replace(
            current,
            phase="complete" if success else "failed",
            completion_next_action=completion_next_action,
            failure_next_action=failure_next_action,
        )
        if current.region != active_region:
            # The side effect belongs to the matching token, but its former
            # status surface is no longer visible. Release the global claim
            # without leaking completion copy into the user's new region.
            self._library_notes_operation = None
            return False
        self._library_notes_operation = terminal
        self._apply_library_notes_operation_state()
        return True
    def _apply_library_note_presentation_state(self) -> None:
        """Synchronize the mounted canvas without changing its composition."""
        if not self.is_mounted or self._library_note_session.snapshot is None:
            return
        try:
            canvas = self.query_one("#library-note-work-pane", LibraryNoteWorkPane)
        except (NoMatches, QueryError):
            return
        canvas.title_placeholder_only = self._library_note_is_pending_blank()
        self._library_note_presentation_syncing = True
        try:
            canvas.apply_session_state(self._library_note_presentation_state())
        finally:
            self._library_note_presentation_syncing = False

        self._apply_library_notes_stage_visibility()
        self._apply_library_notes_footer_context()
    def _library_notes_focus_region(
        self,
    ) -> Literal[
        "",
        "navigator",
        "editor",
        "preview",
        "context",
        "create",
        "lasting_add",
        "lasting_roots",
        "import",
    ]:
        """Return the semantic Notes region currently presented by the host."""
        if self._library_selected_row_id == LIBRARY_ROW_CREATE_NOTE:
            return "create"
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_NOTES:
            return ""
        if self._library_notes_view in {"lasting_add", "lasting_roots"}:
            return self._library_notes_view
        if self._library_notes_view == "import":
            return "import"
        if self._library_notes_view == "list":
            return "navigator"
        if self._library_note_context:
            return "context"
        if self._library_note_preview:
            return "preview"
        return "editor"
    @staticmethod
    def _library_notes_widget_is_within(widget: Widget, ancestor: Widget) -> bool:
        """Return whether ``widget`` belongs to ``ancestor``'s live subtree."""
        return ancestor in widget.ancestors_with_self
    def _library_notes_authority_root(
        self, authority: Literal["database", "files"]
    ) -> Widget | None:
        """Resolve one retained Notes authority's mounted subtree."""
        if authority == "files":
            workspace = self._library_file_notes_workspace
            return (
                workspace if workspace is not None and workspace.is_attached else None
            )
        try:
            return self.query_one(
                ".library-notes-route", LibraryAdaptiveReaderShell
            )
        except (NoMatches, QueryError):
            return None
    def _remember_library_notes_authority_focus(self, focused: Widget | None) -> None:
        """Remember a reachable descendant without treating source chrome as work."""
        if focused is None or focused.id in {
            "library-notes-source-database",
            "library-notes-source-files",
            "library-notes-task-return",
        }:
            return
        for authority in ("database", "files"):
            root = self._library_notes_authority_root(authority)
            if root is not None and self._library_notes_widget_is_within(focused, root):
                self._library_notes_authority_focus[authority] = focused
                return
    def _evacuate_library_notes_authority_focus(
        self, authority: Literal["database", "files"]
    ) -> None:
        """Capture the outgoing authority and clear focus before hiding it."""
        focused = self.focused
        root = self._library_notes_authority_root(authority)
        if (
            focused is not None
            and root is not None
            and self._library_notes_widget_is_within(focused, root)
        ):
            self._library_notes_authority_focus[authority] = focused
        self.set_focus(None)
    def _restore_library_notes_authority_focus(
        self, authority: Literal["database", "files"]
    ) -> bool:
        """Restore the destination's reachable focus or its honest first target."""
        target = self._library_notes_authority_focus[authority]
        if target is not None and (
            not target.is_attached
            or not target.visible
            or target not in self.focus_chain
        ):
            target = None
        if target is None:
            selector = (
                "#file-notes-search"
                if authority == "files"
                else (
                    "#library-note-body" if self._library_notes_view == "editor" else ""
                )
            )
            if selector:
                try:
                    candidate = self.query_one(selector, Widget)
                except (NoMatches, QueryError):
                    candidate = None
                if (
                    candidate is not None
                    and candidate.visible
                    and candidate in self.focus_chain
                ):
                    target = candidate
        if target is None:
            return False
        self.set_focus(target, scroll_visible=False)
        return True
    def _library_notes_focus_stage(
        self, focused: Widget | None
    ) -> Literal["rail", "notes"]:
        """Map a live focused widget to the portable Library stage."""
        if focused is None:
            return self._library_notes_stage
        rail = self._active_library_rail()
        if rail is not None and self._library_notes_widget_is_within(focused, rail):
            return "rail"
        canvas = self._library_layout_ref("#library-canvas")
        if canvas is not None and self._library_notes_widget_is_within(focused, canvas):
            return "notes"
        work_pane = self._library_compose_scoped_ref("#library-note-work-pane")
        if work_pane is not None and self._library_notes_widget_is_within(
            focused, work_pane
        ):
            return "notes"
        return self._library_notes_stage
    def _library_notes_scroll_owner(self, region: str) -> Widget | None:
        """Resolve the one named scroll/content owner for a Notes region."""
        selector = {
            "rail": "#library-rail",
            "navigator": "#library-notes-list",
            "editor": "#library-note-body",
            "preview": "#library-note-preview-region",
            "context": "#library-note-context-region",
            "create": "#library-notes-create-viewport",
            "lasting_add": "#notes-sync-body",
            "lasting_roots": "#notes-sync-roots-body",
            "import": "#library-note-import-canvas",
        }.get(region)
        if selector is None:
            return None
        # TASK-23025: positive ref cache -- a mounted owner is one attribute
        # read; absence still costs one walk, so callers should not probe
        # regions their route cannot mount (see the observer installer).
        return self._library_layout_ref(selector)
    def _capture_library_notes_recompose_state(
        self,
    ) -> _LibraryNotesRecomposeCapture | None:
        """Capture coordinator identity and transient presentation, never draft text."""
        if not self.is_mounted or not self._library_notes_workflow_active():
            return None
        snapshot = self._library_note_session.snapshot
        return _LibraryNotesRecomposeCapture(
            focus=self._capture_library_notes_focus_identity(stage_from_focus=True),
            recompose_generation=self._library_notes_recompose_generation,
            scroll_generation=self._library_notes_scroll_intent_generation,
            focus_generation=self._library_notes_focus_intent_generation,
            session_generation=(
                snapshot.session_generation if snapshot is not None else None
            ),
            draft_revision=(snapshot.draft_revision if snapshot is not None else None),
            preview=self._library_note_preview,
            context=self._library_note_context,
            confirming_delete=self._library_note_confirming_delete,
        )
    def _commit_library_note_widgets_before_recompose(self) -> None:
        """Move queued editor values into the canonical coordinator before teardown."""
        if (
            not self.is_mounted
            or not self._library_note_editor_armed
            or self._library_note_presentation_syncing
            or self._library_notes_focus_region()
            not in {"editor", "preview", "context"}
            or self._library_note_session.snapshot is None
        ):
            return
        try:
            title = self.query_one("#library-note-title", Input).value
            body = self.query_one("#library-note-body", TextArea).text
            keywords_selector = (
                "#library-note-context-keywords"
                if self._library_note_context
                else "#library-note-keywords"
            )
            keywords_text = self.query_one(keywords_selector, Input).value
        except (NoMatches, QueryError):
            return
        if self._library_note_session.mutate(
            title=title,
            body=body,
            keywords_text=keywords_text,
        ):
            self._library_note_shortcut_status = ""
            self._schedule_library_note_autosave()
    def _library_notes_role_target(
        self, identity: LibraryNotesFocusIdentity
    ) -> Widget | None:
        """Resolve one portable semantic role against the current widget tree."""
        role = identity.semantic_role
        selector = {
            "filter": "#library-notes-filter",
            "select-toggle": "#library-notes-select-toggle",
            "title": "#library-note-title",
            "body": "#library-note-body",
            "preview-body": "#library-note-preview-region",
            "context": "#library-note-context-region",
            "save": "#library-note-save",
            "load-retry": "#library-note-load-retry",
            "conflict-callout": "#library-note-conflict-copy",
            "delete-cancel": "#library-note-delete-cancel",
            "lasting-display-name": "#notes-sync-display-name",
            "lasting-folder-choose": "#notes-sync-folder-choose",
            "lasting-check": "#notes-sync-check",
            "lasting-roots-back": "#notes-sync-roots-back",
            "import-add-source": "#note-import-add-source",
            "import-destination": "#note-import-destination",
            "import-check": "#note-import-check",
            "import-execute": "#note-import-import",
            "import-cancel": "#note-import-cancel",
            "import-retry": "#note-import-retry",
            "import-back": "#library-notes-import-back",
            "create-template:blank": "#library-notes-create-blank",
        }.get(role)
        if role.startswith("context-action:"):
            selector = f"#{role.removeprefix('context-action:')}"
        elif role.startswith("sync-direction:"):
            selector = f"#notes-sync-direction-{role.removeprefix('sync-direction:').replace('_', '-')}"
        elif role.startswith("region-back:"):
            selector = {
                "context": "#library-note-context-back",
                "create": "#library-notes-create-back",
                "lasting_add": "#notes-sync-back",
                "lasting_roots": "#notes-sync-roots-back",
            }.get(role.removeprefix("region-back:"), "#library-note-back")
        elif role.startswith("landing-control:"):
            control_id = role.removeprefix("landing-control:")
            if control_id:
                selector = f"#{control_id}"
        elif role.startswith("file-notes:"):
            widget_id = role.removeprefix("file-notes:")
            workspace = self._library_file_notes_workspace
            if workspace is None or not widget_id:
                return None
            try:
                return workspace.query_one(f"#{widget_id}", Widget)
            except (NoMatches, QueryError):
                return None
        elif role.startswith(("media-row:", "media-preview-open:")):
            media_id = role.split(":", 1)[1]
            if (
                role.startswith("media-preview-open:")
                and not self._library_notes_compact
            ):
                try:
                    return self.query_one("#library-media-open-viewer", Widget)
                except (NoMatches, QueryError):
                    return None
            for row in self.query(".library-media-row"):
                if str(getattr(row, "media_id", "") or "") == media_id:
                    return row
            return None
        if selector is not None:
            try:
                return self.query_one(selector, Widget)
            except (NoMatches, QueryError):
                return None

        if role.startswith("library-row:"):
            row_id = role.removeprefix("library-row:")
            for row in self.query(".library-rail-row"):
                if str(getattr(row, "row_id", "") or "") == row_id:
                    return row
        elif role.startswith("note-placement:"):
            placement_id = role.removeprefix("note-placement:")
            for row in self.query(".library-notes-row"):
                if str(getattr(row, "placement_id", "") or "") == placement_id:
                    return row
        elif role.startswith("folder-placement:"):
            placement_id = role.removeprefix("folder-placement:")
            for row in self.query(".library-notes-folder-row"):
                if str(getattr(row, "placement_id", "") or "") == placement_id:
                    return row
        elif role.startswith("tree-pager:"):
            pager_id = role.removeprefix("tree-pager:")
            for pager in self.query(".library-notes-tree-pager"):
                if pager.id == pager_id:
                    return pager
        elif role.startswith("note-row:"):
            note_id = role.removeprefix("note-row:")
            for row in self.query(".library-notes-row"):
                if str(getattr(row, "note_id", "") or "") == note_id:
                    return row
        elif role.startswith("create-template:"):
            template_key = role.removeprefix("create-template:")
            for row in self.query(".library-notes-template-row"):
                if str(getattr(row, "template_key", "") or "") == template_key:
                    return row
        return None
    def _library_notes_fallback_focus_target(
        self, identity: LibraryNotesFocusIdentity
    ) -> Widget | None:
        """Choose the safest visible focus target when an exact role vanished."""
        if self._library_notes_compact and identity.stage != self._library_notes_stage:
            identity = dataclasses.replace(
                identity,
                stage=self._library_notes_stage,
                semantic_role="",
            )
        if identity.stage == "rail":
            role = f"library-row:{self._library_selected_row_id}"
            return self._library_notes_role_target(
                dataclasses.replace(identity, semantic_role=role)
            )
        region = self._library_notes_focus_region()
        if region == "navigator":
            role = f"note-row:{identity.note_id}" if identity.note_id else "filter"
            return self._library_notes_role_target(
                dataclasses.replace(identity, semantic_role=role)
            ) or self._library_notes_role_target(
                dataclasses.replace(identity, semantic_role="filter")
            )
        if region == "context":
            return self._library_notes_role_target(
                dataclasses.replace(identity, semantic_role="context")
            )
        if region == "preview":
            return self._library_notes_role_target(
                dataclasses.replace(identity, semantic_role="preview-body")
            )
        if region == "editor":
            snapshot = self._library_note_session.snapshot
            role = "body"
            if self._library_note_load_state == "failed":
                role = "load-retry"
            elif snapshot is not None and snapshot.in_conflict:
                role = "conflict-callout"
            elif self._library_note_confirming_delete:
                role = "delete-cancel"
            elif self._library_note_session.untouched_create_token is not None:
                role = "title"
            return self._library_notes_role_target(
                dataclasses.replace(identity, semantic_role=role)
            )
        if region == "create":
            return self._library_notes_role_target(
                dataclasses.replace(identity, semantic_role="create-template:blank")
            )
        if region == "lasting_add":
            return self._library_notes_role_target(
                dataclasses.replace(identity, semantic_role="lasting-display-name")
            )
        if region == "lasting_roots":
            return self._library_notes_role_target(
                dataclasses.replace(identity, semantic_role="lasting-roots-back")
            )
        if region == "import":
            phase = self._library_note_import_controller.snapshot.phase
            role = {
                NoteImportPhase.SELECT: "import-add-source",
                NoteImportPhase.DESTINATION: "import-destination",
                NoteImportPhase.CHECKING: "import-cancel",
                NoteImportPhase.REVIEW: "import-execute",
                NoteImportPhase.IMPORTING: "import-cancel",
                NoteImportPhase.RECEIPT: "import-retry",
            }[phase]
            return self._library_notes_role_target(
                dataclasses.replace(identity, semantic_role=role)
            ) or self._library_notes_role_target(
                dataclasses.replace(identity, semantic_role="import-back")
            )
        return None
    def _restore_library_notes_focus_identity(
        self,
        identity: LibraryNotesFocusIdentity,
        guard: _LibraryNotesRestoreGuard | None = None,
    ) -> bool:
        """Restore semantic focus, retaining pending intent until its exact target exists."""
        if not self.is_mounted or not self._library_notes_restore_guard_is_current(
            guard
        ):
            return False
        target = None
        if not (
            self._library_notes_compact and identity.stage != self._library_notes_stage
        ):
            target = self._library_notes_role_target(identity)
        restored_exact_target = target is not None
        if target is None:
            target = self._library_notes_fallback_focus_target(identity)
        if target is None:
            return False
        if identity.semantic_role == "conflict-callout" and isinstance(target, Static):
            target.can_focus = True
        start = identity.body_selection_start
        end = identity.body_selection_end
        if start is not None and end is not None:
            try:
                body = self.query_one("#library-note-body", TextArea)
            except (NoMatches, QueryError):
                pass
            else:
                body.move_cursor(start)
                body.move_cursor(end, select=start != end)
        self._library_notes_restoring_focus = True
        try:
            if self.focused is not target:
                exact_scroll = identity.scroll_offset is not None
                self._library_notes_programmatic_focus_target = target
                self.set_focus(target, scroll_visible=not exact_scroll)
                if not exact_scroll:
                    target.scroll_visible(animate=False, force=True, immediate=True)
            self._restore_library_notes_scroll_offset(identity, guard)
        finally:
            self._library_notes_restoring_focus = False
        if (
            restored_exact_target
            and not self._library_notes_pending_focus_waits_for_snapshot
            and self._library_notes_pending_focus_identity == identity
        ):
            self._library_notes_pending_focus_identity = None
            self._library_notes_pending_focus_generation = None
        return restored_exact_target
    def _restore_library_notes_after_targeted_sync(
        self, identity: LibraryNotesFocusIdentity
    ) -> None:
        """Restore portable Notes focus + scroll after a canvas-scoped sync.

        task-15457 review round 1. The whole-screen seam does this through
        ``_rehydrate_library_notes_after_recompose``; a canvas-scoped sync
        never reaches ``LibraryScreen.refresh``, so it needs its own entry
        into the same restore. No ``_LibraryNotesRestoreGuard`` is passed:
        the guard exists to invalidate a DEFERRED restore that a newer
        navigation intent has superseded, and this one runs synchronously
        from the canvas's own ``recompose``, against the state that was
        captured moments earlier in the same handler.

        Args:
            identity: The portable identity captured before the sync.

        Returns:
            None.
        """
        # task-32052 AC#1: never replay an identity captured on a DIFFERENT
        # surface. ``_sync_library_canvas``'s own skip only watches the WORK
        # pane's mode, so a list -> create switch slipped through and the
        # navigator identity was replayed over the create canvas (Ctrl+N
        # landed on a notes-tree row, not Blank note).
        if identity.region and identity.region != self._library_notes_focus_region():
            return
        # Focus FIRST and synchronously: the callback runs from the canvas's
        # own ``recompose``, so focus is restored before any frame in which
        # it could be seen sitting outside the canvas.
        self._restore_library_notes_focus_identity(identity)
        # Scroll SECOND and deferred. The restore above already attempts it,
        # but at this point the freshly mounted children have not been laid
        # out yet -- the container's max scroll is still 0, so ``scroll_to``
        # clamps the offset away (measured: a 12-row offset restored as 0).
        # The whole-screen seam never hits this because it restores from
        # ``call_after_refresh``, i.e. after a display refresh; re-applying
        # from the same primitive here gives the offset a laid-out container
        # to land in. Idempotent, so the earlier attempt costs nothing.
        self.call_after_refresh(self._restore_library_notes_scroll_offset, identity)
    def _restore_library_notes_scroll_offset(
        self,
        identity: LibraryNotesFocusIdentity,
        guard: _LibraryNotesRestoreGuard | None = None,
    ) -> None:
        """Apply only the named owner's portable offset, without changing focus."""
        if not self._library_notes_restore_guard_is_current(guard):
            return
        owner_region = (
            "rail" if identity.stage == "rail" else self._library_notes_focus_region()
        )
        owner = self._library_notes_scroll_owner(owner_region)
        if identity.semantic_role.startswith(("media-row:", "media-preview-open:")):
            try:
                owner = self.query_one("#library-media-row-scroll", Widget)
            except (NoMatches, QueryError):
                owner = None
        if owner is not None and identity.scroll_offset is not None:
            owner.scroll_to(
                x=identity.scroll_offset[0],
                y=identity.scroll_offset[1],
                animate=False,
                force=True,
                immediate=True,
            )
    def _release_library_notes_focus_after_snapshot(self) -> None:
        """Complete a Back focus handoff after its source refresh settles."""
        if (
            self._library_notes_pending_focus_generation
            != self._library_notes_navigation_generation
        ):
            self._library_notes_pending_focus_identity = None
            self._library_notes_pending_focus_waits_for_snapshot = False
            self._library_notes_pending_focus_generation = None
            return
        identity = self._library_notes_pending_focus_identity
        self._library_notes_pending_focus_waits_for_snapshot = False
        if identity is not None:
            self._restore_library_notes_focus_identity(identity)
    def _library_note_session_is_unsafe(self) -> bool:
        """Return whether rail focus must not hide the active note session."""
        snapshot = self._library_note_session.snapshot
        return bool(
            self._library_note_load_state == "failed"
            or self._library_note_confirming_delete
            or self._library_note_session.destructive_running
            or self._library_note_session.destructive_admission is not None
            or (
                snapshot is not None
                and (
                    snapshot.dirty
                    or snapshot.saving
                    or snapshot.in_conflict
                    or self._library_note_autosave_state
                    in {"saving", "error", "validation", "conflict"}
                    or self._library_note_session.conflict_resolution_running
                )
            )
        )
    def _compact_library_notes_stage(
        self, identity: LibraryNotesFocusIdentity
    ) -> Literal["rail", "notes"]:
        """Resolve compact stage precedence without consulting layout children."""
        if self._library_note_session_is_unsafe():
            return "notes"
        if self._library_notes_explicit_stage_intent:
            return "notes"
        if identity.stage == "rail":
            return "rail"
        if (
            identity.stage == "notes"
            and self._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
        ):
            return "notes"
        if identity.stage == "notes" and self._library_notes_compact_workflow_active():
            return "notes"
        if self._library_notes_focus_region() in {
            "editor",
            "preview",
            "context",
            "create",
            "lasting_add",
            "lasting_roots",
        }:
            return "notes"
        return "rail"
    def _apply_library_notes_stage_legs(self) -> None:
        """Apply compact classes and mutually exclusive mounted stage visibility."""
        if not self.is_mounted:
            return
        try:
            shell = self.query_one("#library-shell-grid", Widget)
            rail = self.query_one("#library-rail", LibraryRail)
            canvas = self.query_one("#library-canvas", Widget)
        except (NoMatches, QueryError):
            return
        rail_handles = self.query("#library-rail-handle")
        rail_handle = rail_handles.first(Widget) if rail_handles else None
        adaptive_reader = bool(
            self.query(
                ".library-media-route, "
                "#library-collections-reader-shell, "
                "#library-conversations-reader-shell, "
                ".library-notes-route, "
                "#library-prompts-reader-shell, "
                "#library-skills-reader-shell"
            )
        )
        adaptive_notes = bool(self.query(".library-notes-route"))
        adaptive_media = bool(self.query(".library-media-route"))
        # Only the grid and canvas host participate in compact CSS selectors;
        # tagging the rail and inner Notes canvas forced two needless global
        # stylesheet matches on every breakpoint crossing. Apply these before
        # the emergency stage can return so a just-recomposed ordinary route
        # cannot retain wide padding/borders at the narrow stage. Adaptive
        # readers share only the shell box-model contract; their internal pane
        # geometry must not inherit the legacy Notes/Media compact selectors.
        legacy_compact = self._library_notes_compact and (
            not adaptive_reader or adaptive_notes
        )
        compact_widgets = (canvas,) if adaptive_media else (shell, canvas)
        for widget in compact_widgets:
            if widget.has_class("library-notes-compact") != legacy_compact:
                widget.set_class(legacy_compact, "library-notes-compact")
        adaptive_compact = self._library_notes_compact and adaptive_reader
        if adaptive_media:
            # Media's Task 1 projection is the single class writer and the
            # presentation-epoch/floor producer for its adaptive shell.
            self._project_library_media_stage_classes(shell)
        elif shell.has_class("library-adaptive-compact") != adaptive_compact:
            shell.set_class(adaptive_compact, "library-adaptive-compact")
        if self._apply_library_emergency_geometry(
            shell=shell,
            rail=rail,
            canvas=canvas,
            rail_handle=rail_handle,
        ):
            return
        try:
            notes_canvas = canvas.query_one("#library-notes-canvas", LibraryNotesCanvas)
        except (NoMatches, QueryError):
            pass
        else:
            if notes_canvas.compact != self._library_notes_compact:
                notes_canvas.apply_compact_presentation(self._library_notes_compact)
        try:
            notes_work = self.query_one("#library-note-work-pane", LibraryNoteWorkPane)
        except (NoMatches, QueryError):
            pass
        else:
            if notes_work.compact != self._library_notes_compact:
                notes_work.apply_compact_presentation(self._library_notes_compact)
        try:
            media_canvas = canvas.query_one("#library-media-canvas", LibraryMediaCanvas)
        except (NoMatches, QueryError):
            pass
        else:
            if (
                not self.query(".library-media-route")
                and media_canvas.compact != self._library_notes_compact
            ):
                media_canvas.apply_compact_presentation(self._library_notes_compact)
        if adaptive_reader:
            # The adaptive shell owns pane visibility. The legacy compact
            # stage toggles below predate the permanent reader work panes and
            # would hide retained siblings or move focus to an unrelated
            # legacy handle.
            self._sync_library_ordinary_rail_width_contract()
            return
        compact_single_stage = (
            self._library_notes_compact and self._library_notes_compact_stage_applies()
        )
        wide_focused_task = (
            not self._library_notes_compact
            and self._library_notes_focused_task_active()
        )
        single_stage = compact_single_stage or wide_focused_task
        manually_collapsed = self._library_rail_collapsed and not single_stage
        rail_display = (
            False
            if wide_focused_task
            else (not compact_single_stage or self._library_notes_stage == "rail")
            and not manually_collapsed
        )
        canvas_display = (
            True
            if wide_focused_task
            else (not compact_single_stage or self._library_notes_stage == "notes")
        )
        handle_display = manually_collapsed
        if rail_handle is not None and rail_handle.display != handle_display:
            rail_handle.display = handle_display
        if rail.display != rail_display:
            rail.display = rail_display
        if canvas.display != canvas_display:
            canvas.display = canvas_display
        try:
            collapse = rail.query_one("#library-rail-collapse", Button)
        except (NoMatches, QueryError):
            pass
        else:
            collapse.display = not single_stage
        for selector in (
            "#library-notes-source-database",
            "#library-notes-source-separator",
            "#library-notes-source-files",
        ):
            try:
                source_control = self.query_one(selector, Widget)
            except (NoMatches, QueryError):
                continue
            source_control.display = not wide_focused_task
        try:
            task_return = self.query_one("#library-notes-task-return", Button)
        except (NoMatches, QueryError):
            pass
        else:
            task_return.display = wide_focused_task
        try:
            note_back = self.query_one("#library-note-back", Button)
        except (NoMatches, QueryError):
            pass
        else:
            note_back.display = not wide_focused_task
        self._sync_library_ordinary_rail_width_contract()
    def _library_notes_work_session_reader_width(self) -> int | None:
        """Return the settled reader width for the active Notes authority."""
        try:
            if self._library_notes_source == LIBRARY_NOTES_SOURCE_FILES:
                workspace = self._library_file_notes_workspace
                if workspace is None or not workspace.is_mounted:
                    return None
                shell = workspace.query_one(
                    "#library-file-notes-reader-shell",
                    LibraryAdaptiveReaderShell,
                )
            else:
                shell = self.query_one(
                    ".library-notes-route",
                    LibraryAdaptiveReaderShell,
                )
        except (NoMatches, QueryError):
            return None
        return shell.region.width if shell.region.width > 0 else None
    def _dispatch_library_notes_work_session(
        self,
        event: NotesWorkSessionEvent,
        *,
        reader_width: int | None = None,
        sync_layout: bool = True,
    ) -> bool:
        """Apply one named transient Notes lifecycle event."""
        if event in {
            NotesWorkSessionEvent.DATABASE_IDENTITY_CLEARED,
            NotesWorkSessionEvent.FOLDER_IDENTITY_CLEARED,
            NotesWorkSessionEvent.AUTHORITY_CHANGED,
            NotesWorkSessionEvent.FOLDER_ROOT_CHANGED,
            NotesWorkSessionEvent.LEFT_NOTES,
        }:
            self._library_notes_work_session_activation_pending = False
        phase = reduce_notes_work_session(
            self._library_notes_work_session_phase,
            event,
            reader_width=(
                self._library_notes_work_session_reader_width()
                if reader_width is None
                and event is NotesWorkSessionEvent.EDITABLE_ITEM_OPENED
                else reader_width
            ),
        )
        if phase is self._library_notes_work_session_phase:
            return False
        self._library_notes_work_session_phase = phase
        if sync_layout:
            if self._library_notes_source == LIBRARY_NOTES_SOURCE_FILES:
                self._sync_library_file_notes_reader_layout_from_shell()
            else:
                self._sync_library_notes_reader_layout_from_shell()
        return True
    def _library_notes_work_first_preferences(
        self,
        preferences: AdaptiveReaderLayoutPreferences,
    ) -> AdaptiveReaderLayoutPreferences:
        """Derive the Library-only work-first override."""
        if self._library_notes_work_session_phase is NotesWorkSessionPhase.ACTIVE:
            return dataclasses.replace(preferences, library_open=False)
        return preferences
    def _set_library_notes_source(
        self,
        source: Literal["database", "files"],
    ) -> bool:
        """Set one admitted Notes authority and reset only on a real change."""
        if source == self._library_notes_source:
            return False
        self._library_notes_source = source
        self._dispatch_library_notes_work_session(
            NotesWorkSessionEvent.AUTHORITY_CHANGED,
            sync_layout=False,
        )
        return True
    def _dispatch_database_note_identity_cleared(self) -> bool:
        """Reset only when a Database selected/loaded identity is cleared."""
        if not self._selected_note_id and self._library_note_session.snapshot is None:
            return False
        return self._dispatch_library_notes_work_session(
            NotesWorkSessionEvent.DATABASE_IDENTITY_CLEARED
        )
    def _activate_database_note_work_session(self, note_id: str) -> None:
        """Activate after the admitted editor has settled its real shell width."""
        if (
            not self._library_notes_work_session_activation_pending
            or self._library_notes_source != LIBRARY_NOTES_SOURCE_DATABASE
            or self._library_notes_view != "editor"
            or self._selected_note_id != note_id
            or self._library_note_session.snapshot is None
        ):
            return
        reader_width = self._library_notes_work_session_reader_width()
        if reader_width is None:
            return
        self._library_notes_work_session_activation_pending = False
        self._dispatch_library_notes_work_session(
            NotesWorkSessionEvent.EDITABLE_ITEM_OPENED,
            reader_width=reader_width,
        )
    def _sync_library_file_notes_reader_layout_from_shell(
        self,
        priority: Literal["library", "items"] | None = None,
        *,
        manual_reopen: PaneName | None = None,
        automatic_priority: bool = True,
    ) -> None:
        """Resolve the settled Folder Files shell and patch it in place."""
        workspace = self._library_file_notes_workspace
        if workspace is None or not workspace.is_mounted:
            return
        try:
            shell = workspace.query_one(
                "#library-file-notes-reader-shell", LibraryAdaptiveReaderShell
            )
        except (NoMatches, QueryError):
            return
        width = shell.region.width
        if width <= 0:
            return
        previous = self._library_file_notes_reader_layout
        if (
            previous.reader_width == 0
            and previous.library_width == 0
            and previous.items_width == 0
        ):
            previous = None
        if priority is None and automatic_priority:
            if workspace.narrow and workspace._narrow_view == "navigator":
                priority = "items"
            elif previous is not None and previous.priority_pane == "items":
                previous = dataclasses.replace(previous, priority_pane=None)
        preferences = self._library_notes_work_first_preferences(
            self._library_file_notes_reader_preferences
        )
        if workspace.narrow and workspace._narrow_view == "editor":
            preferences = dataclasses.replace(preferences, items_open=False)
        layout = resolve_adaptive_reader_layout(
            width,
            preferences,
            LIBRARY_FILE_NOTES_READER_PROFILE,
            previous=previous,
            priority=priority,
        )
        workspace.sync_reader_layout(layout, manual_reopen=manual_reopen)
        self._library_file_notes_reader_layout = layout
    def _mirror_library_notes_reader_preference(
        self,
        key: Literal["library_open", "items_open"],
        value: bool,
    ) -> None:
        """Mirror one optimistic Notes pane choice into app config."""
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return
        library_config = app_config.setdefault("library", {})
        if not isinstance(library_config, dict):
            library_config = {}
            app_config["library"] = library_config
        section_name = "reader" if key == "library_open" else "notes_reader"
        section = library_config.setdefault(section_name, {})
        if not isinstance(section, dict):
            section = {}
            library_config[section_name] = section
        section[key] = value
    def _mirror_library_file_notes_reader_preference(
        self,
        key: Literal["library_open", "items_open"],
        value: bool,
    ) -> None:
        """Mirror one optimistic Folder Files pane choice into app config."""
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return
        library_config = app_config.setdefault("library", {})
        if not isinstance(library_config, dict):
            library_config = {}
            app_config["library"] = library_config
        section_name = "reader" if key == "library_open" else "notes_reader"
        config_key = "library_open" if key == "library_open" else "files_tree_open"
        section = library_config.setdefault(section_name, {})
        if not isinstance(section, dict):
            section = {}
            library_config[section_name] = section
        section[config_key] = value
    def _queue_library_notes_settled_focus_restore(
        self,
        expected: LibraryNotesFocusIdentity,
        guard: _LibraryNotesRestoreGuard | None = None,
    ) -> None:
        """Repeat restoration after resize layout settles, then sample it."""
        self.call_later(
            self._defer_library_notes_settled_focus_restore,
            expected,
            guard,
        )
    def _defer_library_notes_settled_focus_restore(
        self,
        expected: LibraryNotesFocusIdentity,
        guard: _LibraryNotesRestoreGuard | None = None,
    ) -> None:
        """Cross the extra turn Textual may need to finish scrollbar layout."""
        self.call_later(
            self._restore_library_notes_settled_focus,
            expected,
            guard,
        )
    def _restore_library_notes_settled_focus(
        self,
        expected: LibraryNotesFocusIdentity,
        guard: _LibraryNotesRestoreGuard | None = None,
    ) -> None:
        """Apply the tuple after geometry clamps and queue its actual position."""
        if not self._library_notes_restore_guard_is_current(guard):
            return
        self._restore_library_notes_focus_identity(expected, guard)
        self.call_later(
            self._restore_library_notes_final_scroll,
            expected,
            guard,
        )
    def _restore_library_notes_final_scroll(
        self,
        expected: LibraryNotesFocusIdentity,
        guard: _LibraryNotesRestoreGuard | None = None,
    ) -> None:
        """Reapply the exact offset after deferred focus visibility has settled."""
        if not self._library_notes_restore_guard_is_current(guard):
            return
        self._library_notes_restoring_focus = True
        try:
            self._restore_library_notes_scroll_offset(expected, guard)
        finally:
            self._library_notes_restoring_focus = False
        self.call_later(
            self._settle_library_notes_final_scroll,
            expected,
            guard,
        )
    def _settle_library_notes_final_scroll(
        self,
        expected: LibraryNotesFocusIdentity,
        guard: _LibraryNotesRestoreGuard | None = None,
    ) -> None:
        """Win the last focus-visibility clamp before recording the tuple."""
        if not self._library_notes_restore_guard_is_current(guard):
            return
        self._library_notes_restoring_focus = True
        try:
            self._restore_library_notes_scroll_offset(expected, guard)
        finally:
            self._library_notes_restoring_focus = False
        # A Markdown preview recalculates its virtual height on the refresh
        # after compact CSS is applied. The earlier immediate restores can
        # therefore see max_scroll_y == 0 and clamp a retained offset away.
        # Cross that refresh once, then restore and sample the laid-out owner.
        self.call_after_refresh(
            self._restore_library_notes_scroll_after_layout,
            expected,
            guard,
        )
    def _restore_library_notes_scroll_after_layout(
        self,
        expected: LibraryNotesFocusIdentity,
        guard: _LibraryNotesRestoreGuard | None = None,
    ) -> None:
        """Restore and record a portable offset after descendant layout settles."""
        if not self._library_notes_restore_guard_is_current(guard):
            return
        self._library_notes_restoring_focus = True
        try:
            self._restore_library_notes_scroll_offset(expected, guard)
        finally:
            self._library_notes_restoring_focus = False
        self._record_library_notes_presented_focus(expected, guard)
    def _record_library_notes_presented_focus(
        self,
        expected: LibraryNotesFocusIdentity,
        guard: _LibraryNotesRestoreGuard | None = None,
    ) -> None:
        """Remember the actual clamped position for user-change detection."""
        if not self._library_notes_restore_guard_is_current(guard):
            return
        self._install_library_notes_scroll_observers()
        current = self._capture_library_notes_focus_identity()
        if (
            current.region == expected.region
            and current.note_id == expected.note_id
            and current.semantic_role == expected.semantic_role
        ):
            self._library_notes_last_presented_focus = current
            self._library_notes_interaction_focus = current
    def _install_library_notes_scroll_observers(self) -> None:
        """Watch current named scroll owners only while their mounted tree lives.

        TASK-23025: probe only regions the current route can mount. Every
        non-rail region here is a Notes-workflow surface (Database Notes or
        File Notes canvases); outside those routes each probe was a
        guaranteed-failing whole-tree ``query_one`` walk, seven of them per
        focus change on the landing/list routes.
        """
        regions: tuple[str, ...] = ("rail",)
        if self._library_notes_compact_workflow_active():
            regions = (
                "rail",
                "navigator",
                "editor",
                "preview",
                "context",
                "create",
                "lasting_add",
                "lasting_roots",
            )
        owners: dict[int, Widget] = {}
        for region in regions:
            owner = self._library_notes_scroll_owner(region)
            if owner is not None:
                owners[id(owner)] = owner
        for owner in owners.values():
            if getattr(owner, "_library_notes_scroll_observed", False):
                continue
            owner._library_notes_scroll_observed = True
            self.watch(
                owner,
                "scroll_x",
                partial(self._queue_library_notes_scroll_interaction, owner),
                init=False,
            )
            self.watch(
                owner,
                "scroll_y",
                partial(self._queue_library_notes_scroll_interaction, owner),
                init=False,
            )
    def _queue_library_notes_scroll_interaction(
        self, owner: Widget, old_value: float, new_value: float
    ) -> None:
        """Defer scroll intent so a resize epoch can reject layout clamping."""
        del old_value, new_value
        epoch = self._library_notes_resize_epoch
        programmatic = bool(
            self._library_notes_restoring_focus or self._library_notes_resize_settling
        )
        stage = self._library_notes_focus_stage(self.focused)
        region = "rail" if stage == "rail" else self._library_notes_focus_region()
        relevant = self._library_notes_workflow_active() or stage == "rail"
        user_intent = bool(
            not programmatic
            and relevant
            and self._library_notes_scroll_owner(region) is owner
        )
        if user_intent:
            # Invalidate any older deferred resize/recompose restore before
            # its callback can overwrite this explicit scroll interaction.
            self._library_notes_scroll_intent_generation += 1
        self.call_later(
            self._record_library_notes_scroll_interaction,
            owner,
            epoch,
            user_intent,
        )
    def _record_library_notes_scroll_interaction(
        self, owner: Widget, epoch: int, user_intent: bool
    ) -> None:
        """Cache one settled user scroll without a global Idle hot path."""
        if epoch != self._library_notes_resize_epoch or not self.is_mounted:
            return
        stage = self._library_notes_focus_stage(self.focused)
        if not self._library_notes_workflow_active() and stage != "rail":
            return
        region = "rail" if stage == "rail" else self._library_notes_focus_region()
        if self._library_notes_scroll_owner(region) is not owner:
            return
        identity = self._capture_library_notes_focus_identity(stage_from_focus=True)
        self._library_notes_interaction_focus = identity
        if user_intent:
            self._advance_library_stage_interaction()
            self._library_notes_last_user_scroll_focus = identity
    def _record_library_notes_focus_interaction(
        self, expected: Widget, user_intent: bool
    ) -> None:
        """Capture a settled focus change and attach its current scroll owner."""
        if not self.is_mounted or self.focused is not expected:
            return
        if (
            not self._library_notes_workflow_active()
            and self._library_notes_focus_stage(self.focused) != "rail"
            and not (
                self._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
                and self._media_state.view == "list"
            )
        ):
            return
        self._install_library_notes_scroll_observers()
        identity = self._capture_library_notes_focus_identity(stage_from_focus=True)
        self._library_notes_interaction_focus = identity
        if user_intent:
            self._advance_library_stage_interaction()
            self._library_notes_last_user_focus = identity
    def _remember_library_notes_responsive_focus(
        self, identity: LibraryNotesFocusIdentity
    ) -> LibraryNotesFocusIdentity:
        """Keep offsets that an expanded viewport temporarily clamps away."""
        previous = self._library_notes_responsive_focus_memory
        same_target = bool(
            previous is not None
            and previous.region == identity.region
            and previous.note_id == identity.note_id
            and previous.semantic_role == identity.semantic_role
        )
        if (
            same_target
            and previous is not None
            and previous.scroll_offset is not None
            and identity.scroll_offset is not None
        ):
            user_scroll = self._library_notes_last_user_scroll_focus
            user_changed_since_transition = bool(
                self._library_notes_scroll_intent_generation
                != self._library_notes_transition_scroll_generation
                and user_scroll is not None
                and user_scroll.region == identity.region
                and user_scroll.note_id == identity.note_id
                and user_scroll.semantic_role == identity.semantic_role
            )
            current_x, current_y = identity.scroll_offset
            previous_x, previous_y = previous.scroll_offset
            if not user_changed_since_transition:
                current_x, current_y = previous_x, previous_y
            identity = dataclasses.replace(
                identity, scroll_offset=(current_x, current_y)
            )
        self._library_notes_responsive_focus_memory = identity
        return identity
    def _update_library_notes_responsive_state(self) -> None:
        """Measure the invariant shell allocation and transition only on crossing."""
        if not self.is_mounted:
            return
        try:
            width = self.query_one("#library-shell-grid").region.width
        except (NoMatches, QueryError):
            return
        if width <= 0:
            return
        self._sync_library_ingest_rail_for_width(width)
        self._sync_library_ordinary_rail_width_contract()
        self._apply_library_notes_stage_visibility_for_resize()
        compact = width < LIBRARY_NOTES_COMPACT_BREAKPOINT
        if compact == self._library_notes_compact:
            self._library_notes_pre_resize_focus = None
            return
        identity = self._library_notes_pre_resize_focus or (
            self._capture_library_notes_focus_identity(stage_from_focus=True)
        )
        self._library_notes_pre_resize_focus = None
        identity = self._remember_library_notes_responsive_focus(identity)
        self._transition_library_notes_presentation(compact, identity)
    def _mark_library_notes_user_interaction(self) -> None:
        """End resize suppression only when a real input event can own changes."""
        self._library_notes_resize_settling = False
    def _apply_library_notes_footer_context(self) -> None:
        """Persist region help and hide only compact Notes ancillary indicators."""
        shortcuts = self._library_notes_footer_shortcuts()
        registration = ("library", tuple(shortcuts))
        if self._footer_shortcut_registration != registration:
            self.register_footer_shortcuts(source="library", shortcuts=shortcuts)
        hide_ancillary = bool(
            self._library_notes_compact
            and self._library_notes_stage == "notes"
            and self._library_notes_workflow_active()
        )
        # TASK-23025: this method runs on EVERY screen refresh (see the
        # ``refresh`` override), several times per resize frame and focus
        # change -- resolve the stable footer chrome through the ref cache
        # and skip the no-op display writes.
        for selector in (
            "#footer-word-count",
            "#footer-token-count",
            "#internal-db-size-indicator",
        ):
            indicator = self._library_layout_ref(selector)
            if indicator is not None and indicator.display == hide_ancillary:
                indicator.display = not hide_ancillary
    def _rehydrate_library_notes_after_recompose(
        self, restore: _LibraryNotesRecomposeCapture
    ) -> None:
        """Reapply the live coordinator snapshot and portable presentation tuple."""
        if (
            not self.is_mounted
            or restore.recompose_generation != self._library_notes_recompose_generation
        ):
            return
        snapshot = self._library_note_session.snapshot
        session_matches = restore.session_generation is None or bool(
            snapshot is not None
            and snapshot.note_id == restore.focus.note_id
            and snapshot.session_generation == restore.session_generation
        )
        revision_matches = restore.draft_revision is None or bool(
            snapshot is not None and snapshot.draft_revision == restore.draft_revision
        )
        presentation_matches = session_matches and revision_matches
        pending_focus = (
            self._library_notes_pending_focus_identity
            if self._library_notes_pending_focus_generation
            == self._library_notes_navigation_generation
            else None
        )
        focus_identity = pending_focus or restore.focus
        may_restore_focus = pending_focus is not None or presentation_matches
        explicit_stage_intent = self._library_notes_explicit_stage_intent
        if (
            self._library_notes_compact
            and may_restore_focus
            and not explicit_stage_intent
        ):
            self._library_notes_stage = focus_identity.stage
        self._apply_library_notes_stage_visibility()
        if snapshot is not None and session_matches:
            if snapshot.in_conflict:
                self._library_note_preview = False
                self._library_note_context = False
            elif presentation_matches:
                self._library_note_preview = restore.preview
                self._library_note_context = restore.context
            if presentation_matches:
                self._library_note_confirming_delete = restore.confirming_delete
            self._apply_library_note_presentation_state()
            self.call_after_refresh(
                self._arm_library_note_editor, restore.recompose_generation
            )
        self._apply_library_notes_footer_context()
        if may_restore_focus:
            guard = _LibraryNotesRestoreGuard(
                recompose_generation=restore.recompose_generation,
                scroll_generation=restore.scroll_generation,
                focus_generation=restore.focus_generation,
            )
            self._restore_library_notes_focus_identity(focus_identity, guard)
            self.call_after_refresh(
                self._queue_library_notes_settled_focus_restore,
                focus_identity,
                guard,
            )
        if explicit_stage_intent:
            self._library_notes_explicit_stage_intent = False
    async def action_library_notes_new(self) -> None:
        """Open Create only after the active canonical draft flushes."""
        await self._select_library_rail_row(LIBRARY_ROW_CREATE_NOTE)
    def action_library_notes_focus_filter(self) -> None:
        """Focus Navigator Filter without claiming literal input keystrokes."""
        self._focus_library_notes_filter_input()
        try:
            self.query_one("#library-notes-filter", Input).scroll_visible(
                animate=False, force=True, immediate=True
            )
        except (NoMatches, QueryError):
            return
    def _show_library_note_shortcut_refusal(self, message: str) -> None:
        """Keep a blocked local action visible in status and notification."""
        self._library_note_shortcut_status = message
        self._apply_library_note_presentation_state()
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")
    async def action_library_notes_escape(self) -> None:
        """Follow the visible Notes Back hierarchy without bypassing vetoes."""
        snapshot = self._library_note_session.snapshot
        if self._library_note_confirming_delete:
            if self._library_note_session.destructive_running:
                self._show_library_note_shortcut_refusal(
                    "Delete is running — wait for it to finish."
                )
                return
            admission = self._library_note_session.destructive_admission
            if (
                admission is not None
                and not self._library_note_session.cancel_destructive(admission)
            ):
                self._show_library_note_shortcut_refusal(
                    "Delete is running — wait for it to finish."
                )
                return
            self._restore_library_note_delete_origin()
            return
        if snapshot is not None and (
            snapshot.in_conflict
            or self._library_note_session.conflict_resolution_running
        ):
            self._show_library_note_shortcut_refusal(
                "Conflict locked — choose Overwrite or Reload before leaving."
            )
            self._focus_library_note_conflict_callout()
            return
        if self._library_note_context:
            self._library_note_context = False
            self._apply_library_note_presentation_state()
            self._focus_library_note_control("#library-note-context")
            return
        if self._library_notes_select_mode:
            self._library_notes_select_mode = False
            self._library_notes_row_selection.clear()
            _sync_library_canvas(
                self,
                "notes",
                then=lambda: self._focus_library_note_control(
                    "#library-notes-select-toggle"
                ),
            )
            return
        if self._library_notes_view == "editor":
            # P0 (found independently by task-3315 and at dev 4d0232358):
            # this called `_back_from_library_note_editor()`, a method that
            # never existed in ANY commit (introduced dangling on the
            # notes-adaptive branch at e453e9099, "feat(notes): preserve
            # focus across compact stages") -- a real Escape keypress from
            # the note editor raised an uncaught AttributeError that
            # terminated the whole app. Route through the one shared Back
            # seam the "‹ Back to list" button and
            # `action_library_note_editor_back` already use (dirty-flush +
            # veto + session-blank GC + list restore) instead of inventing
            # a second exit path.
            #
            # task-32133 AC#1: a veto (invalid title, a failed/conflicted
            # flush, ...) used to return here silently -- live, with a
            # whitespace-padded title, Escape did nothing and said nothing,
            # and "Discard new note" had already disappeared, leaving no
            # visible way out at all. Fix round 1 Important 1: the notify
            # now lives IN ``_exit_library_note_editor_guarded`` itself (the
            # shared seam three other callers also use), not duplicated
            # here -- this call gets it for free.
            await self._exit_library_note_editor_guarded()
            return
        if self._library_selected_row_id == LIBRARY_ROW_CREATE_NOTE:
            if self._library_note_create_running:
                notify = getattr(self.app_instance, "notify", None)
                if callable(notify):
                    notify(
                        "Create is running — wait for it to finish.", severity="warning"
                    )
                return
            self._library_note_create_status = ""
            await self._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
            return
        if self._library_notes_view in {"lasting_add", "lasting_roots"}:
            await self._exit_library_notes_lasting_sync()
            return
        if self._library_notes_view == "import":
            phase = self._library_note_import_controller.snapshot.phase
            if phase in {NoteImportPhase.CHECKING, NoteImportPhase.IMPORTING}:
                self._library_note_import_controller.cancel()
                return
            self._library_notes_view = "list"
            _sync_library_canvas(
                self, "notes", then=self._focus_library_notes_filter_input
            )
            return
        if self._library_notes_sort_choices_visible:
            self._library_notes_sort_choices_visible = False
            _sync_library_canvas(
                self,
                "notes",
                then=lambda: self._focus_library_note_control("#library-notes-sort"),
            )
            return
        self._advance_library_stage_interaction()
        self._library_notes_stage = "rail"
        self._library_notes_explicit_stage_intent = False
        self._supersede_library_notes_navigation()
        if self.query(".library-notes-route"):
            # The adaptive shell keeps all three owners mounted, so Escape
            # from Navigator moves toward Library by granting that pane one
            # effective-layout priority. This is deliberately not persisted:
            # the user's requested pane preferences remain authoritative and
            # return on the next unconstrained layout. Resolve visibility
            # before restoring focus so a narrow shell never strands focus in
            # the auto-collapsed Library pane.
            self._sync_library_notes_reader_layout_from_shell(priority="library")
        self._apply_library_notes_stage_visibility()
        self._apply_library_notes_footer_context()
        identity = LibraryNotesFocusIdentity(
            stage="rail",
            region="navigator",
            note_id=None,
            semantic_role=f"library-row:{self._library_selected_row_id}",
        )
        self.call_after_refresh(self._restore_library_notes_focus_identity, identity)
    def _library_note_editor_active(self) -> bool:
        """True while the in-canvas DATABASE note editor is the live view (task-2856).

        Scoped to Database Notes so this never
        overlaps ``_file_notes_active()`` -- Files mode has its own
        dedicated Escape gate and never sets ``_library_notes_view``.
        """
        return (
            self._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
            and getattr(
                self,
                "_library_notes_source",
                LIBRARY_NOTES_SOURCE_DATABASE,
            )
            == LIBRARY_NOTES_SOURCE_DATABASE
            and getattr(self, "_library_notes_view", "list") == "editor"
        )
    @staticmethod
    def _note_word_count(content: str) -> int:
        """Count words in a Library note body for the meta line's status text.

        Counts whitespace-delimited runs via a regex scan instead of
        ``len(content.split())``, which would materialize a full list of
        every word just to throw it away -- wasteful for the very large
        note bodies this editor allows (see ``LIBRARY_NOTE_CONTENT_MAX_CHARS``).

        Args:
            content: The note body text to count words in.

        Returns:
            The number of whitespace-delimited words.
        """
        return sum(1 for _ in re.finditer(r"\S+", content))
    def _library_notes_user_id(self) -> str:
        """Return the notes-service user id, falling back to the shared default."""
        return getattr(self.app_instance, "notes_user_id", None) or "default_user"
    async def _notes_true_count_or_none(
        self, count_notes: Any, **kwargs: Any
    ) -> int | None:
        """Fetch the authoritative local notes total, degrading quietly on failure.

        Runs inside the same ``asyncio.gather`` as the paginated ``list_notes``
        fetch (see ``_list_local_source_snapshot``). ``list_notes`` on the
        real local backend returns a plain list with no total, so the rail
        badge would otherwise always show the "showing up to N" sample-cap
        suffix. When the seam is missing entirely, the caller never invokes
        this method (guarded by ``callable(count_notes)``); when it *is*
        present but raises here, the failure is swallowed and ``None`` is
        returned so the caller falls back to the paginated response's own
        record count instead of surfacing an error or failing the whole
        snapshot fetch.

        Args:
            count_notes: The bound ``count_notes`` callable to invoke.
            **kwargs: Forwarded to ``count_notes`` (``scope``, ``user_id``).

        Returns:
            The exact notes count, or ``None`` if the call failed or
            returned something other than an ``int``.
        """
        try:
            result = await self._run_library_service_call(
                count_notes, isolate_in_worker=True, **kwargs
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to fetch exact local notes count; using sample count."
            )
            return None
        return result if isinstance(result, int) else None
    def _selected_library_notes_tree_row(self):
        """Return the selected visible placement, if it still survives."""
        projection = self._build_library_notes_tree_projection()
        if projection is None:
            return None
        return projection.row(self._library_notes_tree_selected_placement_id)
    def _library_notes_folder_target_options(
        self, *, exclude_folder_id: str | None = None
    ) -> tuple[tuple[str, str], ...]:
        """Return bounded loaded folder choices, excluding a moved subtree."""
        folders = {
            folder.folder_id: folder
            for state in self._library_notes_tree_branches.values()
            if state.key.slice_kind == "folders"
            for folder in state.items
            if isinstance(folder, NoteFolder)
        }
        excluded_path = (
            folders[exclude_folder_id].normalized_path
            if exclude_folder_id in folders
            else ""
        )
        choices = [
            (folder.path.strip("/").replace("/", " / "), folder.folder_id)
            for folder in folders.values()
            if folder.folder_id != exclude_folder_id
            and not (
                excluded_path and folder.normalized_path.startswith(f"{excluded_path}/")
            )
        ]
        return tuple(sorted(choices, key=lambda item: (item[0].casefold(), item[1])))
    def _remove_library_note_source_record(self, note_id: str) -> None:
        """Patch one successful delete into the cached Notes rows and count."""
        target_id = str(note_id)
        records = self._local_source_records.get("notes", ())
        self._local_source_records["notes"] = tuple(
            record for record in records if self._source_record_id(record) != target_id
        )
        self._local_source_counts["notes"] = max(
            self._local_source_counts.get("notes", 0) - 1,
            0,
        )
    def _append_library_note_source_record(self, record: Mapping[str, Any]) -> bool:
        """Append one newly active note and increment the exact rail count."""
        note_id = self._source_record_id(record)
        if not note_id:
            return False
        records = self._local_source_records.get("notes", ())
        if any(self._source_record_id(item) == note_id for item in records):
            return False
        self._local_source_records["notes"] = records + (dict(record),)
        self._local_source_counts["notes"] = (
            self._local_source_counts.get("notes", 0) + 1
        )
        return True
    def _library_notes_canvas_kwargs(self) -> dict[str, Any]:
        """Return every compose input for the mounted Database Notes canvas."""
        tree_projection = self._build_library_notes_tree_projection()
        if tree_projection is not None and self._library_notes_sort_choices_visible:
            # task-32128 (review round 2): Sort exists only on the flat
            # fallback, so the tree arriving while the chooser is open must
            # close the MODE, not just stop rendering it -- otherwise the
            # footer keeps offering "choose sort" and the first Escape is
            # spent on a chooser nothing is painting.
            self._library_notes_sort_choices_visible = False
        values: dict[str, Any] = {
            "list_state": None,
            "sort_mode": self._library_notes_sort,
            "filter_value": self._library_notes_filter,
            "mode": "list",
            "presentation_state": None,
            "import_snapshot": self._library_note_import_snapshot,
            "import_receipt_available": (
                self._library_note_import_controller.snapshot.can_revisit_receipt
            ),
            "lasting_sync_snapshot": self._library_notes_lasting_sync_snapshot,
            "tree_projection": tree_projection,
            "tree_selected_placement_id": getattr(
                self, "_library_notes_tree_selected_placement_id", ""
            ),
            "tree_deleted_folder_available": (
                getattr(self, "_library_notes_deleted_folder_receipt", None) is not None
            ),
            "title_placeholder_only": False,
            "compact": self._library_notes_compact,
            # task-32127: the toolbar merges its two action groups only when
            # the pane can hold them; this is the width the reader layout
            # just resolved for the Items pane.
            "pane_width": getattr(
                self._notes_state.reader_layout, "items_width", 0
            ),
            "create_running": self._library_note_create_running,
            "create_status": self._library_note_create_status,
            "load_state": self._library_note_load_state,
            "load_message": self._library_note_load_message,
        }
        if self._library_selected_row_id == LIBRARY_ROW_CREATE_NOTE:
            values["mode"] = "create"
        elif self._library_notes_view == "import":
            values["mode"] = "import"
        elif self._library_notes_view == "lasting_add":
            values["mode"] = "lasting_add"
        elif self._library_notes_view == "lasting_roots":
            values["mode"] = "lasting_roots"
        elif self._library_notes_view == "editor":
            presentation_state = self._library_note_editor_state()
            if presentation_state is None:
                values["mode"] = "loading"
            else:
                values["mode"] = "editor"
                values["presentation_state"] = self._library_note_presentation_state()
                values["title_placeholder_only"] = (
                    self._library_note_pending_blank_gc_id is not None
                    and self._library_note_pending_blank_gc_id == self._selected_note_id
                )
        else:
            values["list_state"] = self._build_library_notes_state()
        return values
    def _library_notes_list_canvas_kwargs(self) -> dict[str, Any]:
        """Return list-only inputs for the retained Notes Items pane."""
        values = self._library_notes_canvas_kwargs()
        values.update(
            mode="list",
            list_state=self._build_library_notes_state(),
            presentation_state=None,
            title_placeholder_only=False,
        )
        return values
    def _library_note_work_pane_kwargs(self) -> dict[str, Any]:
        """Return active non-list content for the retained Notes work pane."""
        values = self._library_notes_canvas_kwargs()
        values["list_state"] = None
        return values
    async def _project_library_note_entry_result(
        self,
        *,
        entry_origin: bool,
        then: Callable[[], None] | None = None,
    ) -> LibraryEntryReconcileResult | None:
        """Project one note-load result without a Library-screen fallback."""
        if not self.is_mounted:
            return LibraryEntryReconcileResult.SUPERSEDED if entry_origin else None
        if entry_origin and not self.query("#library-notes-canvas"):
            generation = self._library_snapshot_state_generation
            route_key = self._library_entry_route_key()
            result = await self._replace_library_canvas_child(
                LibraryNotesCanvas(
                    **self._library_notes_list_canvas_kwargs(),
                    id="library-notes-canvas",
                ),
                generation=generation,
                route_key=route_key,
            )
            if result is LibraryEntryReconcileResult.APPLIED and then is not None:
                then()
            return result
        synced = _sync_library_canvas(
            self,
            "notes",
            then=then,
            allow_screen_fallback=not entry_origin,
        )
        if entry_origin:
            return (
                LibraryEntryReconcileResult.APPLIED
                if synced
                else LibraryEntryReconcileResult.FAILED
            )
        return None
    def _begin_library_note_load(
        self, note_id: str, *, entry_origin: bool = False
    ) -> None:
        """Reset presentation, invalidate old work, and start one editor load."""
        navigation_generation = self._supersede_library_notes_navigation()
        self._library_note_session.close_session()
        self._selected_note_id = note_id
        self._library_notes_view = "editor"
        self._library_note_load_state = "loading"
        self._library_note_load_message = ""
        self._library_note_autosave_state = "idle"
        self._library_note_shortcut_status = ""
        self._library_note_confirming_delete = False
        self._library_note_preview = False
        self._library_note_context = False
        self._library_note_delete_origin_context = False
        self._library_note_delete_origin_preview = False
        self._library_note_editor_armed = False
        self._apply_library_notes_stage_visibility()
        self.run_worker(
            self._refresh_library_note_detail(
                note_id,
                entry_origin=entry_origin,
            ),
            exclusive=True,
            group="library_note_detail",
        )
        self.run_worker(
            self._locate_library_notes_tree_target(
                note_id=note_id,
                focus=False,
                navigation_generation=navigation_generation,
            ),
            exclusive=True,
            group="library_notes_locator",
        )
    @on(LibraryNoteWorkPane.EditorReady)
    def handle_library_note_work_pane_editor_ready(
        self, event: LibraryNoteWorkPane.EditorReady
    ) -> None:
        """Arm dirty tracking only after the retained editor children mount."""
        event.stop()
        if (
            self._library_selected_row_id != LIBRARY_ROW_BROWSE_NOTES
            or self._library_notes_view != "editor"
            or self._library_note_session.snapshot is None
            or self._selected_note_id != self._library_note_session.snapshot.note_id
        ):
            return
        self._arm_library_note_editor()
        if self._library_note_session.untouched_create_token is not None:
            self._focus_library_note_control("#library-note-title")
    def _reset_library_note_editor_state(self) -> None:
        """Clear all in-canvas Library note editor/save state.

        Shared by the Back handler, note-row selection, and rail-row
        selection so every exit from the editor leaves save/autosave/
        conflict tracking in a clean ``idle`` state for the next note.
        """
        self._dispatch_database_note_identity_cleared()
        self._library_note_session.close_session()
        if self._library_notes_view != "import":
            self._library_notes_view = "list"
        self._selected_note_id = ""
        self._library_note_load_state = "idle"
        self._library_note_load_message = ""
        self._library_note_autosave_state = "idle"
        self._library_note_shortcut_status = ""
        self._library_note_confirming_delete = False
        self._library_note_preview = False
        self._library_note_context = False
        self._library_note_delete_origin_context = False
        self._library_note_delete_origin_preview = False
        self._library_note_editor_armed = False
        # Defense in depth: the normal exit path is ``_flush_library_note_
        # save`` (which GCs a still-pending blank note and clears both
        # flags itself, before this reset ever runs); this catches the
        # other "note is gone" fallback paths that reach this reset
        # without going through that flush first, and every full editor
        # exit generally, so neither flag can outlive the editor session
        # it was armed for.
        self._library_note_pending_blank_gc_id = None
        self._library_note_session_blank_id = None
        # (P0) A full editor reset ends the session the pristine-title
        # marker belonged to; the next note starts untouched again.
        self._library_note_title_user_edited = False
        self._invalidate_library_note_autosave()
    def _invalidate_library_note_autosave(self) -> None:
        """Cancel the debounce and fence off any autosave already queued."""
        self._library_note_autosave_generation += 1
        if self._library_notes_autosave_timer is not None:
            self._library_notes_autosave_timer.stop()
            self._library_notes_autosave_timer = None
    @on(Input.Changed, "#library-note-title")
    def handle_library_note_title_changed(self, event: Input.Changed) -> None:
        """Mark the open note dirty and (re)arm the autosave debounce.

        Args:
            event: Input change event emitted by the editor's title field.
        """
        if (
            self._library_note_presentation_syncing
            or not self._library_note_editor_armed
        ):
            return
        # (P0) Recorded BEFORE (and independently of) the mutate result:
        # the two guards above already exclude every programmatic echo, so
        # reaching this line means the human changed the title widget --
        # even when the resulting string happens to equal the seed and the
        # coordinator reports "no change". A blank-seeded note renders its
        # title Input placeholder-only (empty value, draft still holding
        # the seed), so pasting "Untitled" into it is exactly that case:
        # a real touch that mutate() sees as a no-op. Gating the marker on
        # mutate() therefore left the GC free to destroy the note.
        self._library_note_title_user_edited = True
        if self._library_note_session.mutate(title=event.value):
            self._library_note_pending_blank_gc_id = None
            self._library_note_shortcut_status = ""
            self._schedule_library_note_autosave()
            self._apply_library_note_presentation_state()
    @on(TextArea.Changed, "#library-note-body")
    def handle_library_note_body_changed(self, event: TextArea.Changed) -> None:
        """Mark the open note dirty and (re)arm the autosave debounce.

        Args:
            event: Text change event emitted by the editor's body ``TextArea``.
        """
        if (
            self._library_note_presentation_syncing
            or not self._library_note_editor_armed
        ):
            return
        if self._library_note_session.mutate(body=event.text_area.text):
            self._library_note_pending_blank_gc_id = None
            self._library_note_shortcut_status = ""
            self._schedule_library_note_autosave()
            self._apply_library_note_presentation_state()
    @on(Input.Changed, "#library-note-keywords")
    def handle_library_note_keywords_changed(self, event: Input.Changed) -> None:
        """Mark the open note dirty and (re)arm the autosave debounce.

        Args:
            event: Input change event emitted by the editor's keywords field.
        """
        if (
            self._library_note_presentation_syncing
            or not self._library_note_editor_armed
        ):
            return
        if self._library_note_session.mutate(keywords_text=event.value):
            self._library_note_pending_blank_gc_id = None
            self._library_note_shortcut_status = ""
            self._schedule_library_note_autosave()
            self._apply_library_note_presentation_state()
    @on(Input.Changed, "#library-note-context-keywords")
    def handle_library_note_context_keywords_changed(
        self, event: Input.Changed
    ) -> None:
        """Mutate the canonical keyword draft from the Context properties field."""
        if (
            self._library_note_presentation_syncing
            or not self._library_note_editor_armed
        ):
            return
        if self._library_note_session.mutate(keywords_text=event.value):
            # PR #2547 review (Qodo finding 5): the sibling wide-keywords
            # handler above clears this before scheduling autosave; this
            # handler didn't, so a keyword typed only through Info could
            # autosave while the status kept claiming "Draft — not saved
            # yet" (``_library_note_is_pending_blank`` stayed true).
            self._library_note_pending_blank_gc_id = None
            self._library_note_shortcut_status = ""
            self._schedule_library_note_autosave()
            self._apply_library_note_presentation_state()
    def _fire_library_note_autosave(self, generation: int) -> None:
        """Debounce-timer callback: kick the actual autosave worker."""
        if generation != self._library_note_autosave_generation:
            return
        self._library_notes_autosave_timer = None
        if not self._library_note_dirty:
            return
        self.run_worker(
            self._save_library_note(
                explicit=False,
                autosave_generation=generation,
            ),
            exclusive=True,
            group="library_note_save",
        )
    @on(Button.Pressed, "#library-note-save")
    def handle_library_note_save(self, event: Button.Pressed) -> None:
        """Explicitly save the open note, bypassing the autosave debounce."""
        event.stop()
        # LIB-14: an explicit Save is the user's own deliberate "keep it"
        # act, even if the note is still blank -- never GC a note the user
        # just told the app to save (unlike autosave, which is not a
        # deliberate signal -- see ``_library_note_session_blank_id``'s
        # docstring for why autosave alone must NOT clear it).
        self._library_note_pending_blank_gc_id = None
        self._library_note_session_blank_id = None
        self._invalidate_library_note_autosave()
        self.run_worker(
            self._save_library_note(explicit=True),
            exclusive=True,
            group="library_note_save",
        )
    def _library_note_meta_base_line(self) -> str:
        """The static Created/Modified/version portion of the meta line."""
        snapshot = self._library_note_session.snapshot
        if snapshot is None:
            return ""
        return build_library_note_editor_state(
            {
                "id": snapshot.note_id,
                "version": snapshot.version,
                "created_at": snapshot.baseline.created_at,
                "last_modified": snapshot.baseline.modified_at,
            }
        ).meta_line
    def _update_library_note_meta_static(self, *, content: str) -> None:
        """Synchronize persistent metadata and status without recomposition.

        Args:
            content: Compatibility input from save callers. Canonical content
                is read from the coordinator snapshot.
        """
        del content
        self._apply_library_note_presentation_state()
    def _focus_library_note_validation_field(self, field: str) -> None:
        """Restore keyboard focus to the field named by a validation veto."""
        selector = {
            "title": "#library-note-title",
            "body": "#library-note-body",
            "keywords": "#library-note-context-keywords",
        }.get(field)
        if selector is None or not self.is_mounted:
            return

        def focus_field() -> None:
            try:
                self.query_one(selector).focus()
            except (NoMatches, QueryError):
                return

        self.call_after_refresh(focus_field)
    def _route_library_note_validation_field(self, field: str) -> None:
        """Expose the region that owns a vetoed field without recomposing."""
        if field in {"title", "body"}:
            self._library_note_preview = False
            self._library_note_context = False
        elif field == "keywords":
            # Keywords now belong to Info -> Properties at every width.
            # Route there before focusing so validation never targets the
            # retained-but-hidden legacy inline field.
            self._library_note_context = True
    def _focus_library_note_conflict_callout(self) -> None:
        """Move keyboard focus to the explanatory conflict callout."""
        try:
            callout = self.query_one("#library-note-conflict-copy", Static)
        except (NoMatches, QueryError):
            return
        callout.can_focus = True
        callout.focus()
    def _apply_library_note_saved_presentation(
        self,
        outcome: NoteSaveOutcome,
    ) -> None:
        """Present saved only when the outcome still covers the latest draft."""
        snapshot = self._library_note_session.snapshot
        if snapshot is None or snapshot.note_id != self._selected_note_id:
            return
        if outcome.kind is NoteSaveOutcomeKind.SAVED:
            self._patch_library_note_list_from_session()
        is_current_clean_revision = (
            not snapshot.dirty
            and outcome.revision is not None
            and outcome.revision == snapshot.saved_revision
        )
        self._library_note_autosave_state = (
            "saved" if is_current_clean_revision else "idle"
        )
        self._update_library_note_meta_static(content=snapshot.body)
    def _apply_library_note_save_outcome(self, outcome: NoteSaveOutcome) -> None:
        """Translate one typed coordinator save outcome into screen presentation."""
        snapshot = self._library_note_session.snapshot
        if outcome.kind is NoteSaveOutcomeKind.STALE or snapshot is None:
            return
        if snapshot.note_id != self._selected_note_id:
            return
        if outcome.kind in {
            NoteSaveOutcomeKind.SAVED,
            NoteSaveOutcomeKind.ACKNOWLEDGED,
        }:
            self._apply_library_note_saved_presentation(outcome)
            return
        if outcome.kind is NoteSaveOutcomeKind.CONFLICTED:
            self._library_note_autosave_state = "conflict"
            self._library_note_preview = False
            self._library_note_context = False
            self._invalidate_library_note_autosave()
            if self.is_mounted:
                self._apply_library_note_presentation_state()
                self.call_after_refresh(self._focus_library_note_conflict_callout)
            return
        self._library_note_autosave_state = (
            "validation"
            if outcome.kind is NoteSaveOutcomeKind.VALIDATION_VETO
            else "error"
        )
        validation_field = outcome.veto.field if outcome.veto is not None else ""
        if outcome.kind is NoteSaveOutcomeKind.VALIDATION_VETO:
            self._route_library_note_validation_field(validation_field)
        elif self._library_note_preview:
            self._library_note_preview = False
        self._update_library_note_meta_static(content=snapshot.body)
        self._focus_library_note_validation_field(validation_field)
    async def _gc_pending_blank_note(self) -> None:
        """Delete this session's now-empty "Blank note" row before it is left behind (LIB-14).

        "Blank note" still commits its DB row immediately on click (the
        create-note seam has no create-on-first-edit branch -- see the
        AC#5 decision recorded on task-2858); this is the smaller-diff
        alternative the task allows instead: whenever the editor is left
        with this session's blank note in an effectively-empty final state
        (title/body/keywords all blank -- see the caller,
        ``_flush_library_note_save``, which covers both "never touched"
        and "typed then deleted everything"), the row is quietly removed
        here rather than surviving as a permanent literal "Untitled" row
        the user has to find and delete by hand.

        Reads ``_library_note_session_blank_id`` (not the narrower,
        edit-cleared ``_library_note_pending_blank_gc_id``) so this still
        fires after the note was typed into and then emptied out again --
        which, unlike the "never touched" case, may well have gone
        through one or more real autosaves in between (deliberately: see
        that flag's own docstring on why autosave alone must not exempt a
        session blank from GC). Each of those autosaves bumps the row's
        real DB version, so the delete below sends
        ``self._library_note_version`` (the screen's own up-to-date
        tracking of it, updated by every successful save) rather than a
        hardcoded 1 -- an earlier version of this fix hardcoded 1 and the
        delete silently failed on a version mismatch whenever a
        mid-session autosave had already bumped the row past v1 (found by
        the dedicated autosave-then-empty test). Falls back to 1 only for
        the genuinely-never-saved case, where ``_library_note_version`` is
        still ``None``. Best-effort: any failure (including a version
        conflict from a still-possible concurrent EXTERNAL change) is
        swallowed -- GC must never block the exit it runs inside, and the
        worst case on failure is the pre-existing behavior (the row
        survives), not a new regression. On success, patches the same cached
        Notes rows/count used by visible Delete, Undo, and Create while their
        one shared mutation interlock is held.

        Also clears ``_library_note_dirty`` unconditionally (review round
        1 fix): the caller, ``_flush_library_note_save``, is reached via
        the "typed then deleted everything" path with ``_library_note_
        dirty`` still ``True`` -- every exit seam (e.g.
        ``_exit_library_note_editor_guarded``) vetoes on that flag, so
        without clearing it here a successful GC would still leave the
        editor stuck open. Cleared regardless of whether the delete call
        itself succeeds, matching "GC must never block the exit it runs
        inside": on the rare failure path the row survives with its
        pre-exit content (the emptied edit is not separately persisted
        either), but the user is never trapped in the editor over it.
        """
        note_id = self._library_note_session_blank_id
        # The row's current version -- NOT hardcoded to 1 -- since a
        # mid-session autosave (deliberately still possible before GC;
        # see the docstring above) may already have bumped it past its
        # initial create-time version.
        current_version = self._library_note_version or 1
        self._library_note_session_blank_id = None
        self._library_note_pending_blank_gc_id = None
        if not note_id or self._library_notes_mutation_in_flight:
            return
        self._library_notes_mutation_in_flight = True
        service = getattr(self.app_instance, "notes_scope_service", None)
        delete_note = getattr(service, "delete_note", None)
        if not callable(delete_note):
            self._library_notes_mutation_in_flight = False
            return
        try:
            deleted = await self._run_library_service_call(
                delete_note,
                scope="local_note",
                note_id=note_id,
                version=current_version,
                user_id=self._library_notes_user_id(),
                isolate_in_worker=True,
            )
            if deleted:
                self._remove_library_note_source_record(note_id)
        except Exception:
            logger.opt(exception=True).debug(
                f"Could not GC untouched blank note {note_id!r}; leaving it in place."
            )
            return
        finally:
            self._library_notes_mutation_in_flight = False
    async def _resolve_library_note_conflict(self, *, overwrite: bool) -> None:
        """Resolve a conflict through the coordinator's token-gated action.

        Args:
            overwrite: ``True`` for Overwrite, ``False`` for Reload.
        """
        self._library_note_shortcut_status = ""
        action = ConflictAction.OVERWRITE if overwrite else ConflictAction.RELOAD
        operation = asyncio.create_task(
            self._library_note_session.resolve_conflict(action)
        )
        try:
            # Let the coordinator atomically claim its conflict token before
            # reflecting the running state. The operation's first service
            # boundary yields here; a synchronous rejection/finish needs no
            # intermediate presentation.
            await asyncio.sleep(0)
            if self._library_note_session.conflict_resolution_running:
                self._apply_library_note_presentation_state()
            outcome = await operation
        except asyncio.CancelledError:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            raise
        if outcome.kind in {
            ConflictOutcomeKind.STALE,
            ConflictOutcomeKind.ALREADY_RUNNING,
            ConflictOutcomeKind.NOT_IN_CONFLICT,
        }:
            return
        if outcome.kind is ConflictOutcomeKind.MISSING and not overwrite:
            self._dispatch_database_note_identity_cleared()
            self._reset_library_note_editor_state()
            self._notify_library_note_missing_warning()
            self._refresh_local_source_snapshot()
            if self.is_mounted:
                _sync_library_canvas(self, "notes")
            return
        snapshot = self._library_note_session.snapshot
        validation_field = ""
        if outcome.kind is ConflictOutcomeKind.OVERWRITTEN:
            if outcome.save_outcome is not None:
                self._apply_library_note_saved_presentation(outcome.save_outcome)
        elif outcome.kind is ConflictOutcomeKind.RELOADED:
            self._library_note_autosave_state = "idle"
        elif outcome.kind in {
            ConflictOutcomeKind.RENEWED_CONFLICT,
            ConflictOutcomeKind.DRAFT_CHANGED,
            ConflictOutcomeKind.MISSING,
        }:
            self._library_note_autosave_state = "conflict"
        elif outcome.kind is ConflictOutcomeKind.FAILED:
            self._library_note_autosave_state = (
                "conflict" if snapshot is not None and snapshot.in_conflict else "error"
            )
        elif outcome.kind is ConflictOutcomeKind.VALIDATION_VETO:
            self._library_note_autosave_state = "validation"
            if (
                outcome.save_outcome is not None
                and outcome.save_outcome.veto is not None
            ):
                validation_field = outcome.save_outcome.veto.field
        self._library_note_preview = False
        self._library_note_context = False
        if outcome.kind is ConflictOutcomeKind.VALIDATION_VETO:
            self._route_library_note_validation_field(validation_field)
        if self.is_mounted:
            self._apply_library_note_presentation_state()
            if snapshot is not None and snapshot.in_conflict:
                self.call_after_refresh(self._focus_library_note_conflict_callout)
            elif validation_field:
                self._focus_library_note_validation_field(validation_field)
    @on(Button.Pressed, "#library-note-conflict-overwrite")
    def handle_library_note_conflict_overwrite(self, event: Button.Pressed) -> None:
        """Resolve a shown save conflict by re-saving the kept local edits.

        Args:
            event: Button press event emitted by the conflict UI's
                "Overwrite" action.
        """
        event.stop()
        self.run_worker(
            self._resolve_library_note_conflict(overwrite=True),
            group="library_note_conflict",
        )
    @on(Button.Pressed, "#library-note-conflict-reload")
    def handle_library_note_conflict_reload(self, event: Button.Pressed) -> None:
        """Resolve a shown save conflict by discarding local edits and reloading.

        Args:
            event: Button press event emitted by the conflict UI's
                "Reload" action.
        """
        event.stop()
        self.run_worker(
            self._resolve_library_note_conflict(overwrite=False),
            group="library_note_conflict",
        )
    def _read_library_note_editor_fields(self) -> tuple[str, str, str] | None:
        """Project canonical draft fields for non-persistence utilities.

        Returns:
            ``(title, content, keywords_text)`` from the coordinator snapshot,
            or ``None`` when no Database Note session is active.
        """
        snapshot = self._library_note_session.snapshot
        if snapshot is None:
            return None
        return snapshot.title, snapshot.body, snapshot.keywords_text
    @on(Button.Pressed, "#library-note-edit")
    def handle_library_note_edit_mode(self, event: Button.Pressed) -> None:
        """Show the retained editable Database Note surface."""
        event.stop()
        if self._library_notes_view != "editor":
            return
        self._library_note_preview = False
        self._library_note_context = False
        self._apply_library_note_presentation_state()
    @on(Button.Pressed, "#library-note-preview")
    def handle_library_note_preview_toggle(self, event: Button.Pressed) -> None:
        """Show the retained read-only Markdown preview.

        The coordinator already owns every raw field, so recomposition can
        switch presentation without taking a second mutable draft snapshot.

        Args:
            event: Button press event emitted by the editor's Preview/Edit
                action.
        """
        event.stop()
        if (
            self._library_notes_view != "editor"
            or self._library_note_autosave_state == "conflict"
        ):
            return
        if self._library_note_session.snapshot is None:
            return
        self._library_note_context = False
        self._library_note_preview = True
        self._apply_library_note_presentation_state()
    @on(Button.Pressed, "#library-note-context")
    def handle_library_note_context_open(self, event: Button.Pressed) -> None:
        """Show the stable Context region without replacing editor widgets.

        Args:
            event: Button press emitted by the editor's Context action.

        Returns:
            None.
        """
        event.stop()
        snapshot = self._library_note_session.snapshot
        if (
            self._library_notes_view != "editor"
            or snapshot is None
            or snapshot.in_conflict
            or self._library_note_confirming_delete
        ):
            return
        self._library_note_context = True
        self._apply_library_note_presentation_state()
        try:
            context = self.query_one("#library-note-context-region")
        except (NoMatches, QueryError):
            return
        context.focus()
    @on(Button.Pressed, "#library-note-context-back")
    def handle_library_note_context_back(self, event: Button.Pressed) -> None:
        """Return to the Edit or Preview presentation that opened Context.

        Args:
            event: Button press emitted by the Context region's Back action.

        Returns:
            None.
        """
        event.stop()
        if not self._library_note_context:
            return
        self._library_note_context = False
        self._apply_library_note_presentation_state()
        try:
            self.query_one("#library-note-context", Button).focus()
        except (NoMatches, QueryError):
            return
    async def _export_library_note(self, export_format: str) -> None:
        """Push the Export dialog for the open Library note.

        Mirrors the retired standalone Notes screen's export dialog flow --
        a ``FileSave`` prompt pre-filled with a sanitized default filename,
        whose callback writes the built export content once a path is
        chosen. The export reads the *live* editor widgets (via
        ``_read_library_note_editor_fields``), never the DB, so unlike
        Save there is nothing to flush first.

        Note: the real ``Third_Party.textual_fspicker.FileSave`` dialog
        only accepts ``location``/``title``/``default_file`` (not the
        ``default_filename``/``context`` kwargs the retired screen passed
        it, which would have raised ``TypeError`` if that path ever
        actually ran) -- this uses the dialog's real constructor shape.

        Args:
            export_format: ``"markdown"`` for the frontmatter export
                (Export .md), or ``"text"`` for the plain-text export
                (Export .txt).
        """
        if self._library_notes_view != "editor" or not self._selected_note_id:
            return
        fields = self._read_library_note_editor_fields()
        if fields is None:
            return
        operation = self._begin_library_notes_operation("export")
        if operation is None:
            return
        title, content, keywords_text = fields
        note_id = self._selected_note_id
        safe_title = (
            "".join(
                char
                for char in (title.strip() or "note")
                if char.isalnum() or char in (" ", "-", "_")
            ).rstrip()
            or "note"
        )
        default_filename = (
            f"{safe_title}.md" if export_format == "markdown" else f"{safe_title}.txt"
        )
        dialog_title = (
            "Export Note as Markdown"
            if export_format == "markdown"
            else "Export Note as Text"
        )
        await self.app.push_screen(
            FileSave(
                location=str(Path.home()),
                title=dialog_title,
                default_file=default_filename,
            ),
            callback=lambda path: self.call_after_refresh(
                self._write_library_note_export_file,
                path,
                export_format,
                title,
                content,
                keywords_text,
                note_id,
                operation,
            ),
        )
    @on(Button.Pressed, "#library-note-context-export-md")
    @on(Button.Pressed, "#library-note-export-md")
    async def handle_library_note_export_markdown(self, event: Button.Pressed) -> None:
        """Export the open note as Markdown via a ``FileSave`` dialog.

        Args:
            event: Button press event emitted by the editor's "Export .md" action.
        """
        event.stop()
        await self._export_library_note("markdown")
    @on(Button.Pressed, "#library-note-context-export-txt")
    @on(Button.Pressed, "#library-note-export-txt")
    async def handle_library_note_export_text(self, event: Button.Pressed) -> None:
        """Export the open note as plain text via a ``FileSave`` dialog.

        Args:
            event: Button press event emitted by the editor's "Export .txt" action.
        """
        event.stop()
        await self._export_library_note("text")
    @on(Button.Pressed, "#library-note-context-copy")
    @on(Button.Pressed, "#library-note-copy")
    def handle_library_note_copy(self, event: Button.Pressed) -> None:
        """Copy the open Library note to the clipboard as markdown.

        Uses the app's ``copy_to_clipboard`` seam (the same
        ``getattr``-gated pattern every other Library handoff/action in
        this screen already uses) rather than importing ``pyperclip``
        directly the way the retired standalone Notes screen's copy action
        did -- that makes this testable via a recorded fake the way the
        rest of this screen's actions are, and doesn't add a hard runtime
        dependency on a package that isn't always installed/working.

        Args:
            event: Button press event emitted by the editor's Copy action.
        """
        event.stop()
        notify = getattr(self.app_instance, "notify", None)
        if self._library_notes_view != "editor" or not self._selected_note_id:
            return
        fields = self._read_library_note_editor_fields()
        if fields is None:
            return
        operation = self._begin_library_notes_operation("copy")
        if operation is None:
            return
        title, content, keywords_text = fields
        note_id = self._selected_note_id
        export_content = build_note_export_content(
            title, content, keywords_text, note_id, "markdown"
        )
        copy_to_clipboard = getattr(self.app_instance, "copy_to_clipboard", None)
        if not callable(copy_to_clipboard):
            if callable(notify):
                notify(
                    "Clipboard copy is unavailable in this runtime.", severity="warning"
                )
            self._finish_library_notes_operation(
                operation,
                success=False,
                failure_next_action="enable clipboard access and try again",
            )
            return
        try:
            copy_to_clipboard(export_content)
        except Exception as exc:
            logger.opt(exception=True).warning(
                f"Failed to copy Library note {note_id!r} to clipboard."
            )
            if callable(notify):
                notify(f"Error copying note: {type(exc).__name__}", severity="error")
            self._finish_library_notes_operation(
                operation,
                success=False,
                failure_next_action="check clipboard access and try again",
            )
            return
        if callable(notify):
            notify("Note copied to clipboard as markdown!", severity="information")
        self._finish_library_notes_operation(operation, success=True)
    def _selected_library_note_handoff_payload(self) -> ChatHandoffPayload | None:
        """Build the Console handoff payload for the open Library note.

        Mirrors ``_selected_media_handoff_payload``: reads the currently
        open note's live (possibly unsaved) editor fields rather than a
        list-row selection, matching the local-note handoff shape the
        retired standalone Notes screen staged.

        Returns:
            A ``ChatHandoffPayload`` staging the open note as Console
            context, or ``None`` when no note is currently open.
        """
        if self._library_notes_view != "editor" or not self._selected_note_id:
            return None
        fields = self._read_library_note_editor_fields()
        if fields is None:
            return None
        title, content, keywords_text = fields
        note_id = self._selected_note_id
        keywords = self._library_note_keywords_from_input(keywords_text) or []
        return ChatHandoffPayload.from_source_content(
            source="notes",
            item_type="note",
            title=title.strip() or "Untitled Note",
            body=content,
            source_id=str(note_id),
            suggested_prompt="Use this note as context and help me work with it.",
            runtime_backend="local",
            source_owner="local",
            source_selector_state="local",
            discovery_owner="notes",
            discovery_entity_id=str(note_id),
            scope_type="global",
            metadata={
                "note_version": self._library_note_version,
                "keywords": keywords,
            },
        )
    def _open_selected_library_note_handoff(self) -> bool:
        """Stage the note open in the editor into Console via the shared handoff.

        Mirrors ``_open_selected_media_handoff``: builds the payload, then
        guards on having an open note, the workspace context-handoff gate,
        and the app exposing ``open_chat_with_handoff`` at all.
        """
        workspace_state = self._library_workspace_depth_state()
        payload = self._selected_library_note_handoff_payload()
        notify = getattr(self.app_instance, "notify", None)
        if payload is None:
            if callable(notify):
                notify("Open a note before using it in Console.", severity="warning")
            return False
        if not workspace_state.context_handoff_enabled:
            if callable(notify):
                notify(workspace_state.context_handoff_tooltip, severity="warning")
            return False
        open_chat_with_handoff = getattr(
            self.app_instance, "open_chat_with_handoff", None
        )
        if not callable(open_chat_with_handoff):
            if callable(notify):
                notify(
                    "Console handoff is unavailable for Library Notes.",
                    severity="warning",
                )
            return False
        try:
            open_chat_with_handoff(payload, action_label="Use in Console")
        except Exception:
            logger.opt(exception=True).warning("Library note Console handoff failed.")
            if callable(notify):
                notify("Could not use this note in Console.", severity="warning")
            return False
        return True
    @on(Button.Pressed, "#library-note-context-use-in-console")
    @on(Button.Pressed, "#library-note-use-in-console")
    def handle_library_note_use_in_console(self, event: Button.Pressed) -> None:
        """Hand the open note off to Console as chat context.

        Args:
            event: Button press event emitted by the editor's
                "Use in Console" action.
        """
        event.stop()
        operation = self._begin_library_notes_operation("console")
        if operation is None:
            return
        handed_off = self._open_selected_library_note_handoff()
        self._finish_library_notes_operation(
            operation,
            success=handed_off,
            failure_next_action="check Console readiness and try again",
        )
    @on(Button.Pressed, "#library-notes-source-files")
    async def _show_library_file_notes(self, event: Button.Pressed) -> None:
        """Leave Database Notes safely and lazily mount File Notes."""
        event.stop()
        if self._prompts_state.mutation_in_flight:
            return
        if self._library_notes_source == LIBRARY_NOTES_SOURCE_FILES:
            return
        await self._flush_library_note_save()
        if self._library_note_dirty:
            return
        if self._library_notes_view == "list":
            self._library_notes_browse_return_receipt = (
                self._capture_library_notes_browse_return_receipt()
            )
        # task-32142 AC#4: a delete-undo receipt is scoped to the Database
        # Notes list session -- it survived a whole Add-from-files/Folder-
        # files journey in the critique, stale by the time the user got
        # back to it (its Undo target may no longer even be the visible
        # list). Leaving the list for another workflow dismisses it, same
        # as pressing Dismiss would.
        self._library_note_delete_receipt = None
        self._evacuate_library_notes_authority_focus("database")
        self._supersede_library_notes_navigation()
        # Database Notes and Folder Files are independent retained authorities.
        # The database coordinator has already flushed above; keep its loaded
        # selection/editor projection so returning from Folder Files resumes it.
        if self._library_file_notes_workspace is None:
            self._library_file_notes_workspace = (
                self._library_file_notes_workspace_factory()
            )
        workspace = self._library_file_notes_workspace
        self._acknowledge_library_destination_change()
        self._set_library_notes_source(LIBRARY_NOTES_SOURCE_FILES)
        # task-2850: the Files-mode Escape binding only fires while
        # ``_file_notes_active()`` is true, so the footer's advertised keys
        # must follow this same transition or Escape works unadvertised.
        self._register_footer_shortcuts()
        try:
            database_shell = self.query_one(
                ".library-notes-route", LibraryAdaptiveReaderShell
            )
            shell_grid = self.query_one("#library-shell-grid", Horizontal)
        except (NoMatches, QueryError):
            self.refresh(recompose=True)
            self.call_after_refresh(
                self._restore_library_notes_authority_focus,
                "files",
            )
        else:
            if workspace._reader_shell is None:
                shell_state = build_library_shell_state(
                    self._build_library_shell_input(),
                    selected_row_id=self._library_selected_row_id,
                )
                workspace.configure_reader_shell(
                    library_pane=LibraryRail(
                        shell_state,
                        self._library_rail_preferences(),
                        query=self._library_rail_search_value(),
                        search_placeholder=self._library_rail_search_placeholder(),
                        workspaces_body_factory=self._compose_workspaces_rail_body,
                        top_action_factory=self._compose_library_rail_top_action,
                        lifecycle=self._library_lifecycle,
                        onboarding_all_empty=self._library_onboarding_all_empty,
                        id="library-file-notes-rail",
                        classes="destination-workbench-pane",
                    ),
                    layout=self._library_file_notes_reader_layout,
                )
            if not workspace.is_attached:
                await shell_grid.mount(workspace, before=database_shell)
            database_shell.display = False
            database_shell.library.display = False
            workspace.display = True
            self._restore_library_notes_authority_focus("files")
            self._sync_library_notes_source_controls()
            self.call_after_refresh(
                self._sync_library_file_notes_reader_layout_from_shell
            )
            self.call_after_refresh(self._hide_library_adaptive_reader_rail_collapse)
        # The production app can recompose Library after the initial mount-time
        # responsive callback. Measure again once the Files canvas owns its
        # final shell geometry so a compact terminal cannot retain stale wide
        # presentation state from startup.
        self.call_after_refresh(self._update_library_notes_responsive_state)
    def _sync_library_notes_source_controls(self) -> None:
        """Patch the retained Notes source strip after an in-place switch."""
        database_selected = self._library_notes_source == LIBRARY_NOTES_SOURCE_DATABASE
        # task-32136: mirrors the compose-time rule -- wide Folder files is
        # a mode of Notes, so its two switches stay on the strip.
        wide_focused_task = (
            not database_selected
            and not self._file_notes_active()
            and not self._library_notes_compact
            and self._library_notes_focused_task_active()
        )
        try:
            database = self.query_one("#library-notes-source-database", Button)
            files = self.query_one("#library-notes-source-files", Button)
            separator = self.query_one("#library-notes-source-separator", Widget)
            task_return = self.query_one("#library-notes-task-return", Button)
        except (NoMatches, QueryError):
            return
        database.set_class(database_selected, "-selected")
        files.set_class(not database_selected, "-selected")
        for control in (database, files, separator):
            control.display = not wide_focused_task
        task_return.display = wide_focused_task
    @on(Button.Pressed, "#library-notes-task-return")
    async def _return_from_library_notes_task(self, event: Button.Pressed) -> None:
        """Return through the active Notes owner's existing guarded exit."""
        event.stop()
        if self._file_notes_active():
            await self._return_to_library_database_notes()
            return
        if (
            self._library_notes_workflow_active()
            and self._library_notes_view == "editor"
        ):
            await self._exit_library_note_editor_guarded()
    @on(Button.Pressed, "#library-notes-source-database")
    async def _show_library_database_notes(self, event: Button.Pressed) -> None:
        """Return to Database Notes only after the File Notes leave guard."""
        event.stop()
        if self._prompts_state.mutation_in_flight:
            return
        await self._return_to_library_database_notes()
    @on(FileNotesReloadConfirmationChanged)
    def _handle_file_notes_reload_confirmation_changed(
        self,
        event: FileNotesReloadConfirmationChanged,
    ) -> None:
        """Keep footer and F1 help truthful for the inline decision state."""
        event.stop()
        if event.control is self._library_file_notes_workspace:
            self._register_footer_shortcuts()
    @on(FileNotesEditableOpened)
    def _handle_file_notes_editable_opened(
        self,
        event: FileNotesEditableOpened,
    ) -> None:
        """Activate work-first only for the current admitted Folder identity."""
        event.stop()
        if (
            event.control is self._library_file_notes_workspace
            and self._library_notes_source == LIBRARY_NOTES_SOURCE_FILES
        ):
            self._dispatch_library_notes_work_session(
                NotesWorkSessionEvent.EDITABLE_ITEM_OPENED
            )
    @on(FileNotesIdentityCleared)
    def _handle_file_notes_identity_cleared(
        self,
        event: FileNotesIdentityCleared,
    ) -> None:
        """Reset Folder work-first only at an explicit identity clear."""
        event.stop()
        if event.control is self._library_file_notes_workspace:
            self._dispatch_library_notes_work_session(
                NotesWorkSessionEvent.FOLDER_IDENTITY_CLEARED
            )
    @on(FileNotesRootChanged)
    def _handle_file_notes_root_changed(
        self,
        event: FileNotesRootChanged,
    ) -> None:
        """Reset Folder work-first after the current root binding changes."""
        event.stop()
        if event.control is self._library_file_notes_workspace:
            self._dispatch_library_notes_work_session(
                NotesWorkSessionEvent.FOLDER_ROOT_CHANGED
            )
    def _try_switch_retained_library_notes_route(self, row_id: str) -> bool:
        """Switch between retained Notes list/Create routes in place.

        Navigator New and Create Back are in-reader mode transitions, not
        destination replacements. All leave guards have already admitted the
        transition before this method runs. Return ``False`` when the retained
        Notes owners are unavailable so the caller can use the legacy route
        fallback.

        Args:
            row_id: Admitted Library destination row.

        Returns:
            ``True`` when the retained Notes shell accepted the transition.
        """
        entering_create = (
            row_id == LIBRARY_ROW_CREATE_NOTE
            and self._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
            and self._library_notes_view == "list"
        )
        leaving_create = (
            row_id == LIBRARY_ROW_BROWSE_NOTES
            and self._library_selected_row_id == LIBRARY_ROW_CREATE_NOTE
        )
        reentering_browse = (
            row_id == LIBRARY_ROW_BROWSE_NOTES
            and self._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        )
        if not (
            (entering_create or leaving_create or reentering_browse)
            and self._library_notes_source == LIBRARY_NOTES_SOURCE_DATABASE
            and self.is_mounted
        ):
            return False
        try:
            self.query_one(".library-notes-route", LibraryAdaptiveReaderShell)
            rail = self.query_one("#library-rail", LibraryRail)
            header = self.query_one("#library-header-line", Static)
        except (NoMatches, QueryError):
            return False

        self._acknowledge_library_destination_change()
        self._library_navigation_context_generation += 1
        self._pending_library_source_open = None
        self._library_notes_operation = None
        self._supersede_library_notes_navigation()
        if not self._library_note_create_running:
            self._library_note_create_status = ""
        self._dispatch_database_note_identity_cleared()
        self._reset_library_note_editor_state()
        self._set_library_destination_with_conversation_fence(row_id)
        if reentering_browse:
            self._library_notes_filter = ""
            self._library_notes_filter_records = None
            self._library_notes_filter_generation += 1
            self._library_notes_tree_filter_state = None
            self._library_notes_browse_return_receipt = None
            self._request_library_notes_tree_initial_load()
        self._library_notes_explicit_stage_intent = True
        self._library_notes_stage = "notes"
        self._invalidate_library_workspace_depth_state()
        self._register_footer_shortcuts()

        shell = build_library_shell_state(
            self._build_library_shell_input(),
            selected_row_id=self._library_selected_row_id,
        )
        self._library_selected_row_id = shell.selected_row_id
        rail.apply_selection(
            shell,
            lifecycle=self._library_lifecycle,
            onboarding_all_empty=self._library_onboarding_all_empty,
        )
        header.update(self._library_header_line(shell.header_line))
        identity = LibraryNotesFocusIdentity(
            stage="notes",
            region="create" if entering_create else "navigator",
            note_id=None,
            semantic_role="create-template:blank" if entering_create else "filter",
        )
        _sync_library_canvas(
            self,
            "notes",
            then=lambda: self._restore_library_notes_focus_identity(identity),
        )
        return True
    @on(Button.Pressed, "#library-notes-sort")
    def handle_library_notes_sort(self, event: Button.Pressed) -> None:
        """Toggle the direct Library notes sort chooser.

        Args:
            event: Button press event emitted by the notes sort control.
        """
        event.stop()
        if self._library_notes_mutation_fenced():
            return
        self._library_notes_sort_choices_visible = (
            not self._library_notes_sort_choices_visible
        )
        _sync_library_canvas(self, "notes")
    @on(Button.Pressed, ".library-notes-sort-choice")
    def handle_library_notes_sort_choice(self, event: Button.Pressed) -> None:
        """Apply the exact sort value carried by one direct choice."""
        event.stop()
        if self._library_notes_mutation_fenced():
            return
        # task-14902: the shared strip composer stashes the payload as
        # ``choice_value`` (the sync panel's convention).
        requested = str(getattr(event.button, "choice_value", "") or "")
        if requested not in {"newest", "oldest", "title"}:
            return
        self._library_notes_sort = requested
        self._library_notes_sort_choices_visible = False
        self._library_notes_select_mode = False
        self._library_notes_row_selection.clear()
        _sync_library_canvas(self, "notes")
    @on(Button.Pressed, "#library-notes-new")
    async def handle_library_notes_new(self, event: Button.Pressed) -> None:
        """Open the in-Library Create surface from Navigator."""
        event.stop()
        if self._library_notes_mutation_fenced():
            return
        await self._select_library_rail_row(LIBRARY_ROW_CREATE_NOTE)
    def _focus_library_notes_filter_input(self) -> None:
        """Restore focus to the notes filter box after a filter recompose."""
        try:
            self.query_one("#library-notes-filter", Input).focus()
        except (NoMatches, QueryError):
            pass
    def _focus_library_notes_lasting_control(self) -> None:
        """Focus the safe primary control after a lasting-sync recompose."""
        try:
            self.query_one(LibraryNotesAddFromFilesCanvas).focus_first_safe_control()
        except (NoMatches, QueryError):
            pass
    def _focus_library_note_import_control(self) -> None:
        """Focus the enabled primary action after an import recompose."""
        try:
            canvas = self.query_one(LibraryNoteImportCanvas)
            for control in canvas.query(".note-import-primary"):
                if isinstance(control, Button) and not control.disabled:
                    control.focus()
                    return
        except (NoMatches, QueryError):
            pass
    async def _reconcile_library_notes_list_canvas(self) -> None:
        """Publish a completed editor-to-list transition before returning."""
        self._request_library_notes_tree_initial_load()
        _sync_library_canvas(
            self,
            "notes",
            then=self._focus_library_notes_filter_input,
        )
    def _library_note_import_database(self) -> CharactersRAGDB:
        """Return the current local Database Notes authority or fail closed."""
        database = getattr(self.app_instance, "chachanotes_db", None)
        if not isinstance(database, CharactersRAGDB):
            raise RuntimeError("Local Database Notes are unavailable.")
        return database
    def _library_note_import_folder_repository(self) -> LocalNoteFolderRepository:
        """Return the current app-owned local folder repository."""
        service = getattr(self.app_instance, "notes_scope_service", None)
        repository = getattr(service, "folder_repository", None)
        if not isinstance(repository, LocalNoteFolderRepository):
            raise RuntimeError("Local Database Notes folders are unavailable.")
        if repository.db is not self._library_note_import_database():
            raise RuntimeError("Local Database Notes authority changed.")
        return repository
    @staticmethod
    def _build_library_note_import_executor(
        database: CharactersRAGDB,
        repository: LocalNoteFolderRepository,
        receipts: NoteImportReceiptRepository,
    ) -> NoteImportExecutor:
        """Build one executor from exact current local authorities."""
        from ...Notes.note_import_executor import (
            LocalNoteImportTarget,
            NoteImportExecutor,
        )

        return NoteImportExecutor(
            target=LocalNoteImportTarget(db=database, folder_repository=repository),
            receipt_repository=receipts,
        )
    def _publish_library_note_import_snapshot(
        self, snapshot: LibraryNoteImportSnapshot
    ) -> None:
        """Retain completion while patching the DOM only on its visible route."""
        self._library_note_import_snapshot = snapshot
        if (
            self.is_mounted
            and self._library_notes_source == LIBRARY_NOTES_SOURCE_DATABASE
            and self._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
            and self._library_notes_view == "import"
        ):
            _sync_library_canvas(self, "notes")
    def _refresh_after_library_note_import(self) -> None:
        """Refresh local list/count and folder tree after execution settles."""
        self._refresh_local_source_snapshot()
        self._request_library_notes_tree_initial_load()
    def _publish_library_notes_lasting_sync_snapshot(
        self, snapshot: LibraryNotesLastingSyncSnapshot
    ) -> None:
        """Retain one typed projection and patch only a visible inert route."""

        self._library_notes_lasting_sync_snapshot = snapshot
        if self.is_mounted:
            self._apply_library_notes_footer_context()
        if (
            self.is_mounted
            and self._library_notes_source == LIBRARY_NOTES_SOURCE_DATABASE
            and self._library_notes_view in {"lasting_add", "lasting_roots"}
        ):
            _sync_library_canvas(self, "notes")
    def _notify_library_note_import_failure(self) -> None:
        """Show one bounded recovery notification without leaking backend detail."""
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(
                "Notes import needs attention. Review the import panel and try again.",
                severity="warning",
            )
    @on(Button.Pressed, "#library-notes-add-from-files")
    async def handle_library_notes_add_from_files(self, event: Button.Pressed) -> None:
        """Open the one authority chooser before any source picker."""
        event.stop()
        if self._library_note_import_execution_active():
            self._library_notes_view = "import"
            self._apply_library_notes_footer_context()
            _sync_library_canvas(
                self, "notes", then=self._focus_library_note_import_control
            )
            return
        if self._library_notes_mutation_fenced():
            return
        note_flush = await self._flush_library_note_save()
        if note_flush.kind is not NoteFlushOutcomeKind.PERMITTED:
            return
        self._supersede_library_notes_navigation()
        # task-32142 AC#4: see the matching comment in
        # ``_show_library_file_notes`` -- Add from files is the other
        # workflow the critique caught a stale delete receipt surviving.
        self._library_note_delete_receipt = None
        self._library_notes_lasting_origin = "setup"
        self._library_notes_view = "lasting_add"
        self._apply_library_notes_footer_context()
        _sync_library_canvas(
            self, "notes", then=self._focus_library_notes_lasting_control
        )
    @on(LibraryNotesAddFromFilesCanvas.RelationshipRequested)
    def handle_library_notes_relationship_choice(
        self, event: LibraryNotesAddFromFilesCanvas.RelationshipRequested
    ) -> None:
        """Keep the inert chooser honest and reuse the incumbent import flow."""

        event.stop()
        route = self._library_notes_sync_controller.choose_relationship(
            event.relationship
        )
        if route == "import":
            self._library_notes_lasting_origin = None
            self._library_notes_view = "import"
            self._apply_library_notes_footer_context()
            _sync_library_canvas(self, "notes")
            self._push_library_note_import_picker()
            return
        _sync_library_canvas(self, "notes")
    @on(LibraryNotesAddFromFilesCanvas.SetupChanged)
    def handle_library_notes_lasting_setup_changed(
        self, event: LibraryNotesAddFromFilesCanvas.SetupChanged
    ) -> None:
        event.stop()
        self._library_notes_sync_controller.set_setup(event.field, event.value)
    @on(LibraryNotesAddFromFilesCanvas.FolderRequested)
    def handle_library_notes_lasting_folder_requested(
        self, event: LibraryNotesAddFromFilesCanvas.FolderRequested
    ) -> None:
        event.stop()

        async def selected(path: Path | None) -> None:
            if path is None or not path.is_dir():
                return
            self._persist_library_notes_sync_location(path)
            controller = self._library_notes_sync_controller
            controller.set_setup("folder", str(path))
            if not controller.snapshot.setup.display_name:
                controller.set_setup("display_name", path.name)

        self.app.push_screen(
            FileOpen(
                title="Choose a folder to keep synced",
                offer_select_folder=True,
                location=self._library_notes_sync_browse_location(),
            ),
            selected,
        )

    def _library_notes_sync_browse_location(self) -> str:
        """Return where "Keep a folder synced" should open (task-32174 AC#2).

        Keyed independently (``library.notes_sync``) from Import once and
        the ingest browser -- each picker context remembers its own
        last-used directory. The stored value is persisted user state, so it
        is validated in ``library_browse_location`` before it is used.
        """
        remembered = validated_browse_directory(
            get_cli_setting("library.notes_sync", "last_directory", None)
        )
        return str(remembered) if remembered is not None else str(Path.home())

    def _persist_library_notes_sync_location(self, selected_path: Path) -> None:
        """Off the event loop: remember the picked sync-folder directory."""
        generation = claim_browse_directory("library.notes_sync", "last_directory")
        self.run_worker(
            lambda: remember_browse_directory(
                "library.notes_sync", "last_directory", selected_path, generation
            ),
            thread=True,
        )
    @on(LibraryNotesAddFromFilesCanvas.CheckRequested)
    async def handle_library_notes_lasting_check(
        self, event: LibraryNotesAddFromFilesCanvas.CheckRequested
    ) -> None:
        event.stop()
        if event.source is None:
            await self._library_notes_sync_controller.check_setup()
            return
        await self._library_notes_sync_controller.recheck_review(
            event.root_id, event.observation_token, event.source
        )
    @on(LibraryNotesAddFromFilesCanvas.ApplyRequested)
    async def handle_library_notes_lasting_apply(
        self, event: LibraryNotesAddFromFilesCanvas.ApplyRequested
    ) -> None:
        event.stop()
        await self._library_notes_sync_controller.apply_reviewed(
            event.root_id, event.observation_token
        )
        if self._library_notes_sync_controller.snapshot.phase == "roots":
            self._library_notes_lasting_origin = None
            self._library_notes_view = "lasting_roots"
            self._apply_library_notes_footer_context()
            _sync_library_canvas(self, "notes")
    @on(LibraryNotesAddFromFilesCanvas.ActivateRequested)
    async def handle_library_notes_lasting_activate(
        self, event: LibraryNotesAddFromFilesCanvas.ActivateRequested
    ) -> None:
        event.stop()
        await self._library_notes_sync_controller.activate_root(
            event.root_id, event.observation_token
        )
        if self._library_notes_sync_controller.snapshot.phase == "roots":
            self._library_notes_view = "lasting_roots"
            _sync_library_canvas(self, "notes")
    @on(LibraryNotesAddFromFilesCanvas.ChoiceRequested)
    def handle_library_notes_lasting_choice(
        self, event: LibraryNotesAddFromFilesCanvas.ChoiceRequested
    ) -> None:
        event.stop()
        self._library_notes_sync_controller.stage_attention_choice(
            event.root_id,
            event.observation_token,
            event.binding_id,
            event.choice,
        )
    @on(LibraryNotesAddFromFilesCanvas.ViewRequested)
    def handle_library_notes_lasting_comparison(
        self, event: LibraryNotesAddFromFilesCanvas.ViewRequested
    ) -> None:
        """Run the private comparison away from the Textual message pump."""

        event.stop()
        self.run_worker(
            self._library_notes_sync_controller.show_conflict_comparison(
                event.root_id,
                event.observation_token,
                event.binding_id,
            ),
            exclusive=True,
            group="library_notes_sync_comparison",
        )
    @on(LibraryNotesAddFromFilesCanvas.ReturnRequested)
    def handle_library_notes_lasting_comparison_return(
        self, event: LibraryNotesAddFromFilesCanvas.ReturnRequested
    ) -> None:
        event.stop()
        self._library_notes_sync_controller.return_to_conflict_choices(
            event.root_id,
            event.observation_token,
            event.binding_id,
        )
    @on(LibraryNotesAddFromFilesCanvas.UndoRequested)
    async def handle_library_notes_lasting_undo(
        self, event: LibraryNotesAddFromFilesCanvas.UndoRequested
    ) -> None:
        event.stop()
        await self._library_notes_sync_controller.undo_conflict_resolution(
            event.root_id,
            event.observation_token,
            event.operation_id,
            history_page=event.page,
        )
        if self._library_notes_sync_controller.snapshot.phase == "roots":
            self._library_notes_lasting_origin = None
            self._library_notes_view = "lasting_roots"
            self._apply_library_notes_footer_context()
            _sync_library_canvas(self, "notes")
    @on(LibraryNotesAddFromFilesCanvas.DismissRequested)
    async def handle_library_notes_lasting_dismiss(
        self, event: LibraryNotesAddFromFilesCanvas.DismissRequested
    ) -> None:
        event.stop()
        await self._library_notes_sync_controller.dismiss_conflict_receipt(
            event.root_id, event.observation_token, event.operation_id
        )
    @on(LibraryNotesAddFromFilesCanvas.HistoryRequested)
    async def handle_library_notes_lasting_history(
        self, event: LibraryNotesAddFromFilesCanvas.HistoryRequested
    ) -> None:
        event.stop()
        await self._library_notes_sync_controller.open_resolution_history(
            event.root_id, event.observation_token
        )
    @on(LibraryNotesAddFromFilesCanvas.HistoryPageRequested)
    async def handle_library_notes_lasting_history_page(
        self, event: LibraryNotesAddFromFilesCanvas.HistoryPageRequested
    ) -> None:
        event.stop()
        await self._library_notes_sync_controller.page_resolution_history(
            event.root_id,
            event.observation_token,
            event.from_page,
            event.page,
        )
    @on(LibraryNotesAddFromFilesCanvas.HistoryReturnRequested)
    def handle_library_notes_lasting_history_return(
        self, event: LibraryNotesAddFromFilesCanvas.HistoryReturnRequested
    ) -> None:
        event.stop()
        self._library_notes_sync_controller.return_from_resolution_history(
            event.root_id, event.observation_token, event.from_page
        )
    @on(LibraryNotesAddFromFilesCanvas.PageRequested)
    def handle_library_notes_lasting_review_page(
        self, event: LibraryNotesAddFromFilesCanvas.PageRequested
    ) -> None:
        event.stop()
        self._library_notes_sync_controller.page_review(
            event.root_id,
            event.observation_token,
            event.from_page,
            event.page,
        )
    @on(LibraryNotesSyncRootsCanvas.PageRequested)
    def handle_library_notes_lasting_root_page(
        self, event: LibraryNotesSyncRootsCanvas.PageRequested
    ) -> None:
        event.stop()
        page = self._library_notes_sync_controller.snapshot.root_page + event.delta
        self._library_notes_sync_controller.set_root_page(page)
    @on(LibraryNotesSyncRootsCanvas.RootActionRequested)
    async def handle_library_notes_lasting_root_action(
        self, event: LibraryNotesSyncRootsCanvas.RootActionRequested
    ) -> None:
        event.stop()
        controller = self._library_notes_sync_controller
        if event.action == "check":
            await controller.sync_now(event.root_id)
        elif event.action in {"migration", "review"}:
            self._library_notes_lasting_origin = "roots"
            self._library_notes_view = "lasting_add"
            self._apply_library_notes_footer_context()
            _sync_library_canvas(self, "notes")
            if event.action == "migration":
                await controller.check_migration(event.root_id)
            else:
                await controller.check_root(event.root_id)
        elif event.action == "pause":
            await controller.pause_root(event.root_id)
        elif event.action == "resume":
            await controller.resume_root(event.root_id)
        elif event.action == "recover":
            root = next(
                (
                    row
                    for row in controller.snapshot.roots
                    if row.root_id == event.root_id
                ),
                None,
            )
            if root is not None and root.action_id is not None:
                await controller.resolve_cleanup(event.root_id, root.action_id)
            else:
                controller.stage_root_action(event.root_id, "recover")
        elif event.action == "disconnect":
            await controller.disconnect_root(
                event.root_id, keep_folder_organization=True
            )
        else:
            controller.stage_root_action(event.root_id, "retarget")
    @on(LibraryNotesAddFromFilesCanvas.BackRequested)
    @on(LibraryNotesSyncRootsCanvas.BackRequested)
    async def handle_library_notes_lasting_back(self, event: Any) -> None:
        """Return from a directly mounted inert surface without mutation."""

        event.stop()
        if (
            isinstance(event, LibraryNotesAddFromFilesCanvas.BackRequested)
            and self._library_notes_lasting_origin == "roots"
        ):
            self._library_notes_sync_controller.return_to_roots()
            self._library_notes_lasting_origin = None
            self._library_notes_view = "lasting_roots"
            self._apply_library_notes_footer_context()
            _sync_library_canvas(self, "notes")
            return
        await self._exit_library_notes_lasting_sync()
    async def _exit_library_notes_lasting_sync(self) -> bool:
        """Settle provisional authority before leaving a lasting-sync canvas."""

        phase = self._library_notes_sync_controller.snapshot.phase
        if phase in {"checking", "activating"}:
            self._apply_library_notes_footer_context()
            return False
        await self._library_notes_sync_controller.abandon_setup()
        self._supersede_library_notes_navigation()
        self._library_notes_lasting_origin = None
        self._library_notes_view = "list"
        _sync_library_canvas(self, "notes", then=self._focus_library_notes_filter_input)
        return True
    @on(Button.Pressed, "#library-notes-import-back")
    def handle_library_notes_import_back(self, event: Button.Pressed) -> None:
        """Return to Notes while any running import continues off-canvas."""
        event.stop()
        self._library_notes_view = "list"
        _sync_library_canvas(self, "notes", then=self._focus_library_notes_filter_input)
    @on(LibraryNoteImportCanvas.AddSourceRequested)
    def handle_library_note_import_add_source(
        self, event: LibraryNoteImportCanvas.AddSourceRequested
    ) -> None:
        """Open the picker to add one more file to a file selection.

        Args:
            event: The canvas request raised by *Add another file*.
        """
        event.stop()
        self._push_library_note_import_picker()
    @on(LibraryNoteImportCanvas.ChangeSourceRequested)
    def handle_library_note_import_change_source(
        self, event: LibraryNoteImportCanvas.ChangeSourceRequested
    ) -> None:
        """Reopen the picker; the selection changes only if one comes back.

        Args:
            event: The canvas request raised by *Change selection*.
        """
        event.stop()
        self._push_library_note_import_picker(replace=True)
    @on(LibraryNoteImportCanvas.ClearSourceRequested)
    def handle_library_note_import_clear_source(
        self, event: LibraryNoteImportCanvas.ClearSourceRequested
    ) -> None:
        """Drop the chosen source without leaving the import workflow.

        Args:
            event: The canvas request raised by *Clear*.
        """
        event.stop()
        self._library_note_import_controller.clear_selection()
    @on(LibraryNoteImportCanvas.GroupActionRequested)
    def handle_library_note_import_group_action(
        self, event: LibraryNoteImportCanvas.GroupActionRequested
    ) -> None:
        """Apply one group's bulk action, reporting a rejected payload.

        Args:
            event: The canvas request carrying the group's classification and
                the action to apply. Values that name no enum member are
                refused by the controller and surface as a failure notice.
        """
        event.stop()
        try:
            self._library_note_import_controller.set_group_action(
                event.classification, event.action
            )
        except (TypeError, ValueError):
            self._notify_library_note_import_failure()
    @on(LibraryNoteImportCanvas.DestinationChanged)
    def handle_library_note_import_destination(
        self, event: LibraryNoteImportCanvas.DestinationChanged
    ) -> None:
        event.stop()
        try:
            self._library_note_import_controller.set_destination(event.destination)
        except (TypeError, ValueError):
            self._notify_library_note_import_failure()
    async def _run_library_note_import_check(self) -> None:
        try:
            await self._library_note_import_controller.check()
        except Exception as exc:  # noqa: BLE001 - controller publishes safe recovery
            logger.warning(
                "Notes import check failed; error_type={}", type(exc).__name__
            )
            self._notify_library_note_import_failure()
    @on(LibraryNoteImportCanvas.CheckRequested)
    def handle_library_note_import_check(
        self, event: LibraryNoteImportCanvas.CheckRequested
    ) -> None:
        event.stop()
        self.run_worker(
            self._run_library_note_import_check(),
            exclusive=True,
            group="library_note_import_check",
        )
    @on(LibraryNoteImportCanvas.ObsidianModeToggled)
    def handle_library_note_import_obsidian_mode(
        self, event: LibraryNoteImportCanvas.ObsidianModeToggled
    ) -> None:
        """Re-run the read-only check with the requested vault-reading mode.

        Args:
            event: The canvas toggle carrying the requested `enabled` mode.
        """
        event.stop()
        try:
            self._library_note_import_controller.set_obsidian_mode(event.enabled)
        except (TypeError, ValueError):
            self._notify_library_note_import_failure()
            return
        self.run_worker(
            self._run_library_note_import_check(),
            exclusive=True,
            group="library_note_import_check",
        )
    @on(LibraryNoteImportCanvas.CollisionNameChanged)
    def handle_library_note_import_collision_name(
        self, event: LibraryNoteImportCanvas.CollisionNameChanged
    ) -> None:
        event.stop()
        self._library_note_import_controller.set_collision_name(event.name)
    @on(LibraryNoteImportCanvas.CollisionChoiceRequested)
    def handle_library_note_import_collision_choice(
        self, event: LibraryNoteImportCanvas.CollisionChoiceRequested
    ) -> None:
        event.stop()
        try:
            self._library_note_import_controller.set_collision_choice(event.choice)
        except (TypeError, ValueError):
            self._notify_library_note_import_failure()
    @on(LibraryNoteImportCanvas.UncertainMatchConfirmed)
    def handle_library_note_import_confirm_match(
        self, event: LibraryNoteImportCanvas.UncertainMatchConfirmed
    ) -> None:
        event.stop()
        try:
            self._library_note_import_controller.confirm_uncertain(event.item_id)
        except (TypeError, ValueError):
            self._notify_library_note_import_failure()
    @on(LibraryNoteImportCanvas.ItemActionRequested)
    def handle_library_note_import_item_action(
        self, event: LibraryNoteImportCanvas.ItemActionRequested
    ) -> None:
        event.stop()
        try:
            self._library_note_import_controller.set_item_action(
                event.item_id, event.action
            )
        except (TypeError, ValueError):
            self._notify_library_note_import_failure()
    @on(LibraryNoteImportCanvas.ItemChoiceRequested)
    def handle_library_note_import_item_choice(
        self, event: LibraryNoteImportCanvas.ItemChoiceRequested
    ) -> None:
        event.stop()
        try:
            self._library_note_import_controller.set_item_choice(
                event.item_id, event.choice, event.enabled
            )
        except (TypeError, ValueError):
            self._notify_library_note_import_failure()
    @on(LibraryNoteImportCanvas.PageRequested)
    def handle_library_note_import_page(
        self, event: LibraryNoteImportCanvas.PageRequested
    ) -> None:
        event.stop()
        current = self._library_note_import_controller.snapshot.page.page_number
        self._library_note_import_controller.set_page(current + event.delta)
    @on(LibraryNoteImportCanvas.RetryRequested)
    def handle_library_note_import_retry(
        self, event: LibraryNoteImportCanvas.RetryRequested
    ) -> None:
        event.stop()
        try:
            execution = self._library_note_import_controller.admit_execution(retry=True)
        except Exception as exc:  # noqa: BLE001 - publish only bounded recovery
            logger.warning(
                "Notes import retry admission failed; error_type={}",
                type(exc).__name__,
            )
            self._notify_library_note_import_failure()
            return
        if execution is None:
            return
        self.run_worker(
            self._run_library_note_import_execution(execution),
            exclusive=True,
            group="library_note_import_execute",
        )
    @on(LibraryNoteImportCanvas.CancelRequested)
    def handle_library_note_import_cancel(
        self, event: LibraryNoteImportCanvas.CancelRequested
    ) -> None:
        event.stop()
        self._library_note_import_controller.cancel()
    @on(Button.Pressed, "#library-notes-export")
    async def handle_library_notes_export(self, event: Button.Pressed) -> None:
        """Open the export canvas scoped to Notes.

        Args:
            event: Button press event emitted by the notes list canvas's
                "Export…" action.
        """
        event.stop()
        if self._library_notes_mutation_fenced():
            return
        await self._open_library_export_canvas(ExportScope(kind="notes"))
    @on(Button.Pressed, ".library-notes-folder-row")
    def handle_library_notes_folder_row(self, event: Button.Pressed) -> None:
        """Expand or collapse one real folder through stable folder identity."""
        event.stop()
        folder_id = str(getattr(event.button, "folder_id", "") or "")
        if not folder_id:
            return
        self._library_notes_tree_selected_placement_id = str(
            getattr(event.button, "placement_id", "") or ""
        )
        if folder_id in self._library_notes_tree_expanded_ids:
            self._library_notes_tree_expanded_ids.remove(folder_id)
            _sync_library_canvas(self, "notes")
            return
        self._library_notes_tree_expanded_ids.add(folder_id)

        def _load_expanded_folder() -> None:
            if folder_id not in self._library_notes_tree_expanded_ids:
                return
            for kind in ("folders", "placements"):
                key = NotesBranchKey(folder_id, kind)
                state = self._library_notes_tree_branches.get(key)
                if state is None or state.freshness == "uninitialized":
                    self._request_library_notes_tree_slice(key)

        _sync_library_canvas(self, "notes", then=_load_expanded_folder)
    @on(Button.Pressed, "#library-notes-select-toggle")
    async def handle_library_notes_select_toggle(self, event: Button.Pressed) -> None:
        """Enter/exit notes select mode; clears the selection set (both on enter and exit).

        Flushes any dirty edit first (awaited) so entering select mode
        never strands a dirty note edit mid-save -- select mode is only
        reachable from the notes list view, but the flush stays
        unconditional here to mirror ``handle_library_notes_row``'s
        always-flush-first behavior.

        TASK-15457 renames the unrelated sync-panel constructor field and
        routes this structural list change through the Notes canvas's own
        ``sync_state`` hook.

        Args:
            event: Button press event emitted by the notes canvas's
                Select/Done toggle.
        """
        event.stop()
        if self._library_notes_mutation_fenced():
            return
        note_flush = await self._flush_library_note_save()
        if note_flush.kind is not NoteFlushOutcomeKind.PERMITTED:
            return
        self._library_notes_select_mode = not self._library_notes_select_mode
        self._library_notes_row_selection.clear()
        _sync_library_canvas(self, "notes")
    @on(Button.Pressed, "#library-notes-select-clear")
    def handle_library_notes_select_clear(self, event: Button.Pressed) -> None:
        """Clear the current notes selection without leaving select mode.

        Rebuilds only the mounted Notes canvas.
        """
        event.stop()
        if self._library_notes_mutation_fenced():
            return
        self._library_notes_row_selection.clear()
        _sync_library_canvas(self, "notes")
    @on(Button.Pressed, "#library-note-back")
    async def handle_library_note_back(self, event: Button.Pressed) -> None:
        """Return the Library notes canvas from the editor to its list view.

        Args:
            event: Button press event emitted by the "‹ Back to list" action.
        """
        event.stop()
        await self._exit_library_note_editor_guarded()
    async def action_library_note_editor_back(self) -> None:
        """Escape: leave the note editor while honoring the dirty guard."""
        await self._exit_library_note_editor_guarded()
    @on(Button.Pressed, "#library-note-load-retry")
    def handle_library_note_load_retry(self, event: Button.Pressed) -> None:
        """Retry the current typed note-load failure without leaving Editor."""
        event.stop()
        note_id = self._selected_note_id
        if not note_id or self._library_notes_view != "editor":
            return
        self._library_note_load_state = "loading"
        self._library_note_load_message = ""
        self.run_worker(
            self._refresh_library_note_detail(note_id),
            exclusive=True,
            group="library_note_detail",
        )
        _sync_library_canvas(self, "notes")
    @on(Button.Pressed, "#library-note-context-delete")
    @on(Button.Pressed, "#library-note-delete")
    async def handle_library_note_delete(self, event: Button.Pressed) -> None:
        """Enter the inline delete-confirmation state for the open note.

        Deleting is destructive, so the first press only swaps the normal
        action row for a "Delete" / "Cancel" confirm affordance (mirroring
        ``handle_library_media_delete``); the actual service call only
        happens once the confirm button (``#library-note-delete-confirm``)
        is pressed. Before confirmation is shown, the coordinator atomically
        admits this exact note/session/version tuple so typing, Save, and a
        duplicate destructive action cannot race the user's decision.

        Flushes a dirty edit first (awaited) so the version the confirmed
        delete sends is never stale; an unsaved edit surviving the flush
        aborts entering the confirm state, same as Back and note-row selection.

        Args:
            event: Button press event emitted by the editor's "Delete" action.
        """
        event.stop()
        if self._library_notes_mutation_fenced():
            return
        initiating_snapshot = self._library_note_session.snapshot
        if (
            initiating_snapshot is None
            or initiating_snapshot.note_id != self._selected_note_id
        ):
            return
        initiating_note_id = initiating_snapshot.note_id
        initiating_session_generation = initiating_snapshot.session_generation
        origin_context = self._library_note_context
        origin_preview = self._library_note_preview
        # The user is explicitly managing deletion now; disarm the legacy
        # blank-note GC path so it cannot race the admitted destructive flow.
        self._library_note_pending_blank_gc_id = None
        self._library_note_session_blank_id = None
        note_flush = await self._flush_library_note_save()
        if note_flush.kind is not NoteFlushOutcomeKind.PERMITTED:
            return
        snapshot = self._library_note_session.snapshot
        if (
            snapshot is None
            or snapshot.note_id != initiating_note_id
            or snapshot.session_generation != initiating_session_generation
            or snapshot.note_id != self._selected_note_id
            or self._library_notes_view != "editor"
        ):
            return
        admission_outcome = (
            await self._library_note_session.request_destructive_admission(
                DestructiveKind.DELETE,
                note_id=snapshot.note_id,
                session_generation=snapshot.session_generation,
                expected_version=snapshot.version,
            )
        )
        if (
            admission_outcome.kind is not DestructiveAdmissionOutcomeKind.ADMITTED
            or admission_outcome.admission is None
        ):
            self._library_note_shortcut_status = admission_outcome.message
            self._apply_library_note_presentation_state()
            return
        admission = admission_outcome.admission
        current = self._library_note_session.snapshot
        if (
            current is None
            or current.note_id != initiating_note_id
            or current.session_generation != initiating_session_generation
            or current.note_id != self._selected_note_id
            or self._library_notes_view != "editor"
        ):
            self._library_note_session.cancel_destructive(admission)
            return
        self._library_note_delete_origin_context = origin_context
        self._library_note_delete_origin_preview = origin_preview
        self._library_note_confirming_delete = True
        # task-32132: Delete is only reachable from Info, and the canvas
        # keeps Info showing while confirming -- forcing these off used to
        # snap the pane to Edit out from under the user. Leave the mode
        # exactly as the user left it; ``_restore_library_note_delete_
        # origin`` below is then a no-op restore, same as it always was.
        self._apply_library_note_presentation_state()
        self._focus_library_note_control("#library-note-delete-cancel")
    def _focus_library_note_control(self, selector: str) -> None:
        """Focus one stable note control when its presentation is visible."""
        self._focus_library_control(selector)
    def _restore_library_note_delete_origin(self) -> None:
        """Leave confirmation and restore its stable source presentation."""
        origin_context = self._library_note_delete_origin_context
        origin_preview = self._library_note_delete_origin_preview
        self._library_note_shortcut_status = ""
        self._library_note_confirming_delete = False
        self._library_note_context = origin_context
        self._library_note_preview = origin_preview
        self._apply_library_note_presentation_state()
        selector = (
            "#library-note-context-delete" if origin_context else "#library-note-delete"
        )
        self._focus_library_note_control(selector)
        self._library_note_delete_origin_context = False
        self._library_note_delete_origin_preview = False
    @on(Button.Pressed, "#library-note-delete-cancel")
    def handle_library_note_delete_cancel(self, event: Button.Pressed) -> None:
        """Discard the pending delete confirmation and restore the normal action row.

        Args:
            event: Button press event emitted by the confirm affordance's
                "Cancel" action.
        """
        event.stop()
        admission = self._library_note_session.destructive_admission
        if admission is not None and not self._library_note_session.cancel_destructive(
            admission
        ):
            self._show_library_note_shortcut_refusal(
                "Delete is running — wait for it to finish."
            )
            return
        self._restore_library_note_delete_origin()
    async def _delete_library_note(self, admission: DestructiveAdmission) -> None:
        """Run one preclaimed Notes delete and always release its interlock."""
        try:
            await self._delete_library_note_claimed(admission)
        finally:
            self._library_notes_mutation_in_flight = False
            if (
                self.is_mounted
                and self._library_notes_view == "list"
                and self._library_note_delete_receipt is not None
            ):
                _sync_library_canvas(self, "notes")
    def _notify_library_note_delete_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed Library note delete.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")
    @on(Button.Pressed, "#library-notes-delete-receipt-dismiss")
    def handle_library_note_delete_receipt_dismiss(self, event: Button.Pressed) -> None:
        """Dismiss the recovery receipt without changing persisted data.

        Args:
            event: Press event emitted by the receipt's Dismiss button.
        """
        event.stop()
        if self._library_notes_mutation_in_flight:
            return
        self._library_note_delete_receipt = None
        if self.is_mounted:
            _sync_library_canvas(
                self,
                "notes",
                then=self._arm_library_list_entry_focus,
            )
    async def _undo_library_note_delete(
        self, receipt: LibraryNoteDeleteReceipt
    ) -> None:
        """Undo one delete through ``NotesScopeService.restore_note``."""
        restored_record: Mapping[str, Any] | None = None
        failure_message = ""
        try:
            service = getattr(self.app_instance, "notes_scope_service", None)
            restore_note = getattr(service, "restore_note", None)
            if not callable(restore_note):
                failure_message = "Note restore is unavailable."
            else:
                try:
                    result = await self._run_library_service_call(
                        restore_note,
                        scope="local_note",
                        note_id=receipt.note_id,
                        version=receipt.expected_version,
                        user_id=self._library_notes_user_id(),
                        isolate_in_worker=True,
                    )
                    if (
                        isinstance(result, Mapping)
                        and self._source_record_id(result) == receipt.note_id
                    ):
                        restored_record = result
                    else:
                        failure_message = "The note could not be restored."
                except ConflictError:
                    failure_message = (
                        "This deleted note changed elsewhere — refresh and try again."
                    )
                except Exception:
                    logger.warning("Failed to restore a Library note")
                    failure_message = "Could not restore this note."

            if restored_record is not None:
                self._append_library_note_source_record(restored_record)
                if self._library_note_delete_receipt == receipt:
                    self._library_note_delete_receipt = None
            else:
                self._notify_library_note_delete_warning(
                    failure_message or "Could not restore this note."
                )
        finally:
            self._library_notes_mutation_in_flight = False
            if self.is_mounted:
                if restored_record is not None:
                    # task-32124: the folder tree is projected from paged
                    # branch state, not from the flat source records the
                    # restore just patched, so re-syncing the canvas alone
                    # brought the rail count back without the row. Reuse the
                    # seam a create already commits through -- a restore is
                    # "this note exists again" -- rather than adding a second
                    # refresh mechanism. It reloads exactly the affected
                    # branches (the note's folders, plus Unfiled) and selects
                    # the restored placement.
                    #
                    # NOT the deep-link locator: repainting the canvas here
                    # removes the receipt the pressed Undo button lives in,
                    # and the focus move that follows is read as user intent
                    # (`on_descendant_focus`), which supersedes the locator's
                    # navigation before its first await returns. Proved live:
                    # the count returned to 10 and the row never came back.
                    await self._reconcile_library_notes_tree_mutation(
                        "note_create",
                        {"note_id": receipt.note_id},
                        before=None,
                        result=restored_record,
                    )
                    identity = LibraryNotesFocusIdentity(
                        stage="notes",
                        region="navigator",
                        note_id=receipt.note_id,
                        semantic_role=f"note-row:{receipt.note_id}",
                    )
                    finish = partial(
                        self._restore_library_notes_focus_identity, identity
                    )
                else:
                    finish = partial(
                        self._focus_library_note_control,
                        "#library-notes-delete-undo",
                    )
                _sync_library_canvas(self, "notes", then=finish)
    def _notify_library_note_missing_warning(self) -> None:
        """Surface a quiet warning when an opened note is no longer available.

        Used by ``_refresh_library_note_detail`` when the fetched detail
        resolves to nothing -- the note was deleted elsewhere, or the row
        pressed was already a stale ghost row.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify("That note is no longer available.", severity="warning")
    @on(Button.Pressed, "#library-notes-create-blank")
    def handle_library_notes_create_blank(self, event: Button.Pressed) -> None:
        """Create a blank local note from the in-canvas Create view.

        Args:
            event: Button press event emitted by the "Blank note" action.
        """
        event.stop()
        create_token = self._begin_library_note_create()
        if create_token is None:
            return
        self.run_worker(
            self._create_library_note(
                title=LIBRARY_NOTE_BLANK_SEED_TITLE,
                content="",
                create_token=create_token,
                blank=True,
            ),
            exclusive=True,
            group="library_note_mutation",
        )
    @on(Button.Pressed, ".library-notes-template-row")
    def handle_library_notes_create_template(self, event: Button.Pressed) -> None:
        """Create a local note pre-filled from the pressed template row.

        Args:
            event: Button press event emitted by a template row button.
        """
        event.stop()
        template_key = str(getattr(event.button, "template_key", "") or "")
        from tldw_chatbook.Event_Handlers.notes_events import NOTE_TEMPLATES

        template = NOTE_TEMPLATES.get(template_key)
        title, content = self._library_note_template_fields(template)
        # Templates carry keywords ("meeting, notes") that the standalone
        # screen applies on create. Preserve the raw shape until the shared
        # lossless validator can either accept it or surface a typed veto.
        keywords: Any = ""
        if isinstance(template, Mapping):
            keywords = template.get("keywords", "")
        create_token = self._begin_library_note_create()
        if create_token is None:
            return
        self.run_worker(
            self._create_library_note(
                title=title,
                content=content,
                keywords=keywords,
                create_token=create_token,
            ),
            exclusive=True,
            group="library_note_mutation",
        )
    @on(Button.Pressed, "#library-notes-create-back")
    async def handle_library_notes_create_back(self, event: Button.Pressed) -> None:
        """Leave Create without creating anything."""
        event.stop()
        if self._library_note_create_running:
            return
        self._library_note_create_status = ""
        await self._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
    def _finish_library_note_create(
        self, create_token: str, *, status: str = ""
    ) -> bool:
        """Finish only the still-current Create attempt."""
        if create_token != self._library_note_create_token:
            return False
        self._library_note_create_running = False
        self._library_notes_mutation_in_flight = False
        self._library_note_create_status = status
        return True
    @on(Button.Pressed, "#library-note-discard-new")
    def handle_library_note_discard_new(self, event: Button.Pressed) -> None:
        """Discard only the coordinator-certified untouched new note."""
        event.stop()
        create_token = self._library_note_session.untouched_create_token
        if create_token is None or self._library_notes_mutation_in_flight:
            return
        self._library_notes_mutation_in_flight = True
        self.run_worker(
            self._discard_new_library_note(create_token),
            exclusive=True,
            group="library_note_mutation",
        )
    async def _discard_new_library_note_claimed(self, create_token: str) -> None:
        """Delete one untouched create through typed destructive admission."""
        snapshot = self._library_note_session.snapshot
        if snapshot is None:
            return
        admission_outcome = (
            await self._library_note_session.request_destructive_admission(
                DestructiveKind.DISCARD_NEW_NOTE,
                note_id=snapshot.note_id,
                session_generation=snapshot.session_generation,
                expected_version=snapshot.version,
                create_token=create_token,
            )
        )
        if (
            admission_outcome.kind is not DestructiveAdmissionOutcomeKind.ADMITTED
            or admission_outcome.admission is None
        ):
            self._library_note_shortcut_status = ""
            self._apply_library_note_presentation_state()
            return
        admission = admission_outcome.admission
        self._apply_library_note_presentation_state()
        if not self._library_note_session.mark_destructive_running(admission):
            self._library_note_session.cancel_destructive(admission)
            self._library_note_shortcut_status = ""
            self._apply_library_note_presentation_state()
            return
        self._apply_library_note_presentation_state()

        service = getattr(self.app_instance, "notes_scope_service", None)
        delete_note = getattr(service, "delete_note", None)
        deleted = False
        if callable(delete_note):
            try:
                deleted = bool(
                    await self._run_library_service_call(
                        delete_note,
                        scope="local_note",
                        note_id=admission.note_id,
                        version=admission.expected_version,
                        user_id=self._library_notes_user_id(),
                        isolate_in_worker=True,
                    )
                )
            except Exception:
                logger.opt(exception=True).warning(
                    f"Failed to discard new Library note {admission.note_id!r}."
                )
        self._library_note_session.finish_destructive(admission, success=deleted)
        self._library_note_shortcut_status = ""
        if not deleted:
            self._notify_library_note_delete_warning(
                "Could not discard this new note. Try again."
            )
            self._apply_library_note_presentation_state()
            return

        self._dispatch_database_note_identity_cleared()
        self._reset_library_note_editor_state()
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
        self._library_notes_filter = ""
        self._library_notes_filter_records = None
        self._library_notes_filter_generation += 1
        self._library_notes_tree_filter_state = None
        self._remove_library_note_source_record(admission.note_id)
        if self.is_mounted:
            await self._reconcile_library_notes_list_canvas()
    def _notify_library_note_create_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed Library note create.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")


# --- BEGIN generated notes-controller-state shims ---
# Permanent, not a cleanup-PR deletion target -- same reasoning as
# `LibraryMediaController`'s own identical block: the byte-for-byte canon
# (recipe SS1) forbids editing a moved body, so the attribute names those
# bodies already use have to keep resolving through *something*. Exposes
# every `LibraryNotesState` field under its original flat name via task 1's
# single-source `notes_state_shim_attr()` -- FOUR prefix families
# (`_library_notes_` default, `_library_note_` singular, `_library_file_notes_`
# for the Folder-Files group, bare `_` for `selected_note_id`).
for _lnc_field in dataclasses.fields(LibraryNotesState):
    setattr(
        LibraryNotesController,
        notes_state_shim_attr(_lnc_field.name),
        property(
            lambda self, _n=_lnc_field.name: getattr(
                self._notes_state_accessor(), _n
            ),
            lambda self, value, _n=_lnc_field.name: setattr(
                self._notes_state_accessor(), _n, value
            ),
        ),
    )
del _lnc_field
# --- END generated notes-controller-state shims ---
