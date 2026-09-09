"""Library Media canvas controller.

Controller PR of the Media extraction series (wave-7 task 2 of
``.superpowers/sdd/2026-09-06-library-decomposition-wave7-media``; media
series 2/3; recipe: ``backlog/docs/library-decomposition-recipe.md``;
``library_prompts_controller.py`` -- the largest prior single-cluster move
-- is the template this mirrors in shape). Owns the Media cluster: the list
canvas's browse/filter/sort/pager/selection surface, the reader/viewer
(content body, find, highlights, reading progress, image previews, the
adaptive reader shell and its return-settlement machinery), the Trash view,
the bulk Analyze run, and the console/chat handoff.

**Cluster derivation.** An ``ast`` census of every ``LibraryScreen``
class-body method whose name contains ``"media"`` (case-insensitive), run
fresh at this task's own execution time (the recipe's "never trust a
carried-over count" rule, §6): **251 raw ``FunctionDef`` matches, 251 unique
names** -- no property/setter-pair gap. Of those, 79 carry an ``@on``
decorator, 12 are ``action_*``, 5 are ``@staticmethod``, **0 are ``@work``**,
and -- re-verifying task 1's claim rather than trusting it -- **0 are
``on_<message>`` name-dispatched handlers**, so §4's third whitelist member
(the one the plan predicted "will bite media") is inert here. The only
``on_[a-z]`` methods on the class are 10 Textual lifecycle hooks and the 7
prompts-owned ``on_prompt_block_editor_*``.

Two completeness checks ran alongside the census:

- a decorator scan for any NON-media-named method carrying a
  ``#library-media*``/``.library-media*``/``*Media*`` ``@on`` selector:
  **zero**;
- a reverse call-graph fixpoint for NON-media-named methods whose every
  in-class caller is (transitively) media-named -- the "bare-named cluster
  member" shape the conversations exemplar's own ``_conversation_records``
  miss made the recipe warn about. This one is NOT zero: it returns **17
  names, every one of them a review-set member** (``_review_*``,
  ``_walk_active_review_set``, ``_activate_review_set``,
  ``_create_and_open_review_set``, ``_collect_review_*``,
  ``_push_review_set_picker``, ``_order_selected_review_pairs``,
  ``_toggle_reviewed_unguarded``, ``_active_review_set_banner``). **All 17
  are deliberately NOT moved**, and the reason is the One Home Rule rather
  than a hazard: review sets are a FEATURE FAMILY of ~25 methods that
  straddles the media/shell boundary, not a media sub-cluster. Eight more of
  its members (``_review_set_service``, ``_review_set_live_ids``,
  ``_notify_review_set``, ``_review_set_active``, ``_active_review_progress``,
  ``_active_review_loaded_at_last``, ``_review_footer_entries``,
  ``_maybe_auto_resume_review_set``/``_cancel_pending_review_set_resume``)
  are called by SHELL methods -- ``check_action``,
  ``_library_route_shortcuts_for_current_state``,
  ``_select_library_rail_row_after_source_admission``,
  ``_open_library_item_by_id`` -- so moving the media-reachable subset would
  split one family across two homes, exactly the split task 1's own BLOCKED
  ruling on ``_library_pending_list_entry_media_return`` refused at the field
  level. The five movers that call into the family reach it through named
  late-binding dependencies instead. (The same fixpoint run WITHOUT first
  removing the two media-NAMED generic dispatchers,
  ``_toggle_library_media_reader_pane`` and
  ``request_library_media_layout_refresh``, returns 31 names -- it drags in
  the whole shared reader-preference machinery, including three OTHER
  subsystems' ``_mirror_library_*_reader_preference`` methods. That
  contamination is why the dispatchers are removed from the seed first, and
  it is worth recording: an uncritical bare-name fixpoint over a cluster that
  owns a generic dispatcher will claim half the screen.)

**Single vs. split controller: SINGLE, based on connected components.**
The ``self.<name>`` graph of 251 candidates has one 213-name component,
28 singletons and four 2–3-member components. Shared ``LibraryMediaState``
field edges join these into one 238-name component and 11 singletons.
The proposed browse/viewer vs. trash/maintenance split has no independent
seam: 35 trash-named candidates have 16 cross-call edges and share six
state fields with the rest; only 20 of the 140 movers are trash-named.
Splitting those methods would retain the coupling without improving the
existing per-file governance. The 111 exclusions leave 140 movers with
3,166 body lines, not the predicted 8–9k-line controller.

**111 of the 251 candidates excluded, not moved (140 move):**

1. **73 unbound-fake-self test-bypass exclusions** (recipe §3's first
   documented shape, and by far the largest population any subsystem has
   produced -- the export series' own forward note, "expect this shape to
   scale with how much a subsystem's tests favor unbound-``SimpleNamespace``
   unit-style calls over full-harness ones", measured). Found by an ``ast``
   census over ALL of ``Tests/`` (never a ``-k``-filtered subset, per the
   seventh shape's filter-blindness rule) for every reference to
   ``LibraryScreen.<cluster-name>``, in four shapes:
   - **49 names / 132 sites** called directly as
     ``LibraryScreen.<name>(fake, ...)`` (``test_library_multiselect_media.py``,
     ``test_library_media_trash.py``, ``test_library_media_reader_flow.py``,
     ``test_review_set_walker.py``, ``test_library_shell.py``,
     ``test_library_media_bulk_delete_real_db.py``);
   - **34 names** bound onto a ``SimpleNamespace`` fake as
     ``types.MethodType(LibraryScreen.<name>, fake)`` /
     ``_press(fake, LibraryScreen.<name>)`` /
     ``LibraryScreen.<name>.__get__(fake)``;
   - **12 names** dispatched by STRING through a
     ``for name in ("...", ...): setattr(fake, name, types.MethodType(
     getattr(LibraryScreen, name), fake))`` loop -- the quoted-string
     spelling §3 requires every census to search for independently of the
     call that consumes it (``test_library_multiselect_media.py:2515``,
     ``:2734``; ``test_library_media_reader_flow.py:1362``);
   - a bare-``LibraryScreen.<name>`` reference in DOCSTRING PROSE
     (``test_local_media_reading_service.py``) is the one false positive the
     line-based pass produced and the ``ast`` pass rejected -- recorded
     because the line-based and ``ast`` passes were reconciled name-for-name
     rather than assumed to agree.

   All 73 stay on ``LibraryScreen``, UNMOVED, full-bodied -- the recipe's
   established accommodation (leave the method real; a mover reaches it,
   where needed, through a named late-binding dependency that re-reads
   ``screen.<name>`` at call time, which is why every fixture above keeps
   working unmodified after this move).

2. **16 instance-attribute-monkeypatch exclusions** (recipe §3's second
   shape). Each is patched on a REAL, ``__init__``-constructed
   ``LibraryScreen`` instance -- eight via
   ``monkeypatch.setattr(screen, "<name>", ...)``
   (``_arm_library_media_return_settlement``,
   ``_focus_library_media_grip_if_current``,
   ``_library_media_content_signature``, ``_library_media_layout_signature``,
   ``_library_media_settlement_tree``, ``_open_selected_media_handoff``,
   ``_reconcile_library_media_stage_presentation``,
   ``_request_library_media_browse``) and eight more via a bare
   ``screen.<name> = ...`` assignment. The governing rule is recipe §3's
   OPENING one, not a narrower dependency test: a name a test patches on a
   real ``LibraryScreen`` keeps its whole in-cluster call graph
   screen-routed until that subsystem's cleanup PR retargets the fixtures.
   Under a mover-caller map re-derived at the parent tree, **9 of the 16 do
   have a direct MOVER calling them as ``self.<name>(...)``** --
   ``_arm_library_media_return_settlement``,
   ``_library_media_content_signature``, ``_library_media_layout_signature``,
   ``_library_media_settlement_tree``, ``_open_selected_media_handoff``,
   ``_reconcile_library_media_stage_presentation``,
   ``_request_library_media_browse``, ``_sync_library_media_browse_state``,
   ``_sync_library_media_trash_state`` -- so for those the patch is directly
   bypassable by a moved body. The other **7**
   (``_analyze_one_library_media_item``,
   ``_exit_library_media_select_mode``,
   ``_focus_library_media_grip_if_current``,
   ``_library_media_unanalyzed_ids``,
   ``_notify_library_media_analysis_warning``,
   ``_request_library_media_type``, ``_start_library_media_analyze``) have
   only EXCLUDED in-cluster callers today, so they are held by §3's
   conservative rule rather than by a demonstrated bypass -- correct as
   exclusions, and MOVE candidates for a later series once their fixtures
   retarget. An earlier draft of this paragraph asserted the narrower
   condition for all 16; it was false for those 7 and is corrected here
   rather than left as precedent.

3. **8 source-census exclusions** -- the shape recipe §3 records as "a
   hardcoded-file-path census, not a monkeypatch bypass, ships red", here in
   its ``inspect.getsource`` spelling.
   ``test_library_multiselect_media.py::test_every_media_mutation_claims_
   the_interlock_at_one_audited_seam`` asserts
   ``"self._media_state.bulk_delete_in_flight = True" in inspect.getsource(
   LibraryScreen._claim_library_media_mutation)`` AND, for each of SIX named
   handlers, ``"_claim_library_media_mutation" in inspect.getsource(handler)``
   -- assertions a one-line delegator's source empties. It also asserts that
   ``inspect.getsource(library_screen_module)`` contains EXACTLY ONE
   whitespace-tolerant regex match for ``_media_state.bulk_delete_in_flight
   = True``, and one for the shared worker-group literal; both sites live
   inside ``_claim_library_media_mutation`` (``library_screen.py:15062``/
   ``:15066`` at this wave's tip; task 3 retargeted seam and guard together),
   which this exclusion keeps in place, so both counts stay at 1.
   ``test_review_set_walker.py:1292`` (``inspect.getsource(
   LibraryScreen.handle_library_media_row)``) is the eighth. Five of the
   eight are independently excluded as unbound-fake-self anyway;
   ``handle_library_media_edit_save`` is excluded by this class alone.

4. **4 module-globals-coupling exclusions** (recipe §3's oldest shape,
   restated as the eighth numbered shape's mechanical census). The census ran
   to completion: all **71 bare module-global names** the 140 mover
   bodies read, crossed against every ``library_screen``-scoped patch shape
   (direct-attribute, fully-qualified string, two-argument
   ``monkeypatch.setattr``/``patch.object``) across ALL of ``Tests/``, under
   every alias tests actually import the module as -- ``library_screen``,
   ``library_screen_module``, ``library_module``, ``screen_module``, the list
   DERIVED by ``ast`` rather than assumed, per wave-5 task 3's own lesson.
   **The census had to be redone by ``ast``:** a first, line-anchored pass
   scored ``resolve_ingest_analysis_provider`` at ZERO hits because its five
   real patch sites spell the module alias and the name on DIFFERENT LINES of
   a multi-line ``monkeypatch.setattr(...)`` call. 13 names had hits; each
   hitting test was read:
   - ``chat_api_call`` -> **ACTIVE**. ``test_library_shell.py:10555`` patches
     ``library_screen_module.chat_api_call`` with a recorder, presses the real
     ``#library-media-analysis-generate`` button and asserts
     ``dispatched.get("api_endpoint") == "openai"``. Moved,
     ``_dispatch_library_media_analysis``'s own import would win, nothing
     would be recorded, and the assertion would fail loudly.
     ``_dispatch_library_media_analysis`` excluded.
   - ``analysis_unavailable_reason`` / ``resolve_ingest_analysis_provider``
     -> **ACTIVE** (8 and 5 sites). ``test_library_ingest_analyze_skipped.py``'s
     ``_ready_provider``/``_unready_provider`` patch them and drive a REAL
     pilot whose action-visibility gate is the patched reason;
     ``test_library_media_render_fixes.py`` patches both around real screens
     five more times. ``_library_media_analysis_provider_reason`` and
     ``handle_library_media_analysis_generate`` excluded.
   - ``build_library_media_viewer_state`` -> **ACTIVE**.
     ``test_library_media_reader_no_change_sync_t22208.py:161`` wraps it in a
     CALL COUNTER and asserts the count is ZERO for a pass-through
     traversal -- the green-but-vacuous shape exactly (a bypassed patch also
     counts zero), which is why the recipe classes a coincidence-satisfied
     assertion as ACTIVE. ``_library_media_viewer_state_cached`` excluded.
   - ``_sync_library_canvas`` -> **LATENT, kept** (33 sites across 10 files;
     "site" per the ingest controller's own definition -- one match, one
     line, of the 3-shape pattern set, deduplicated by line number within a
     file). Four movers forward bare ``self`` into this dispatcher
     (``handle_library_media_select_all``, ``...select_clear``, ``...sort``,
     ``...sort_choice``). Every one of the 33 sites was mapped to its
     ENCLOSING test function and compared against every function in the same
     file that names one of those four movers or presses one of their
     selectors: **zero overlap, in all ten files**. Same systemic shape and
     same verdict every sibling controller already carries.
   - ``_ANALYZE_ORIGIN_MEDIA``, ``_ANALYZE_ORIGIN_IMPORT``,
     ``LIBRARY_ROW_BROWSE_NOTES``/``_PROMPTS``/``_SKILLS``,
     ``LIBRARY_NOTES_SOURCE_DATABASE``, ``set_mode`` -> **LATENT**: plain
     module-attribute READS (a test asserting a constant or computing an
     expected value), never a patch.
   - ``asyncio`` -> **LATENT**: the one hit patches an attribute OF the
     shared ``asyncio`` module object, which is the same object in every
     importer, not a rebinding in ``library_screen``'s own globals.

5. **2 class-monkeypatch exclusions** (recipe §3's opening rule: a name a
   test patches on ``LibraryScreen`` keeps its whole call graph
   screen-routed until cleanup). ``_refresh_library_media_detail``
   (``monkeypatch.setattr(LibraryScreen, ...)`` in
   ``test_library_entry_compose_once.py:779`` and
   ``test_library_shell.py:9416``) and
   ``_sync_library_media_reader_layout_from_shell``
   (``patch.object(type(screen), ...)`` in
   ``test_library_reader_press_scope_t22228.py:135``/``:177``). Both have
   mover callers, so both stay.

6. **1 shared-shell-helper exclusion, by the >=2-subsystems rule applied to
   a METHOD rather than a field.** ``_sanitize_media_field`` is media-NAMED
   but is called by ``_run_library_prompts_import`` (Prompts) and
   ``_library_note_keywords_from_input`` (Notes), and
   ``library_prompts_controller.py`` already binds it as one of ITS OWN
   general shell helpers (``test_library_prompts_wiring.py:511`` pins that).
   It stays on the screen; this controller binds it the same way.

7. **1 generic-dispatcher exclusion.** ``_toggle_library_media_reader_pane``
   is media-named but is the multi-subsystem reader-pane dispatcher that
   reads FOUR state objects -- the wave-6-verified ruling, restated by task
   1's own forward note. Stays screen-resident.

8. **2 bypassed-construction-lifecycle exclusions -- a NINTH bypass shape,
   found by this task's OWN battery rather than by any of its censuses.**
   ``_stop_library_media_selection_debounce`` and
   ``_stop_library_media_filter_timer`` are called by ``on_screen_suspend``,
   a SCREEN-RESIDENT lifecycle hook, and
   ``Tests/UI/test_library_screen_reuse.py::test_on_screen_suspend_stops_
   every_timer_in_isolation`` invokes that hook directly on an
   ``object.__new__(LibraryScreen)`` instance -- ``__init__`` never ran, so
   ``self._media_controller`` does not exist and the delegator raises
   ``AttributeError: 'LibraryScreen' object has no attribute
   '_media_controller'``. This is the CONTROLLER-PR analogue of recipe §3's
   seventh shape (which bites a STATE PR through the generated property
   shim): same bypassed-construction cause, different attribute, and equally
   undeferrable -- it goes red at the move commit itself, so the no-red-ships
   rule fixes it here rather than at cleanup. Unlike the state PR's version
   there is no cheap one-line fixture seed available, because seeding a
   controller means constructing one with 80+ keyword arguments; the
   accommodation is therefore the ordinary one, exclusion. Task 1's own
   forward note had already flagged both names as exclusion candidates for
   exactly this call path, which is the reason to read a predecessor's
   forward notes as hypotheses to TEST rather than as either facts or noise.
   The census that generalises: for every mover, ask whether a
   SCREEN-RESIDENT method calls it AND is invoked on a bypassed screen. This
   task ran it empirically -- all 9 files in the repo containing
   ``object.__new__(LibraryScreen)``/``LibraryScreen.__new__``, run together
   after the two exclusions: **1 failed / 426 passed**, the one failure being
   ``test_library_ingest_retry_last.py::test_registry_ticks_only_reflow_
   footer_when_retry_availability_changes``, documented pre-existing in
   recipe §7 since wave-5 task 3.
   ``_stop_library_media_filter_timer`` has three MOVER callers and so also
   becomes a named late-binding dependency below;
   ``_stop_library_media_selection_debounce`` has none.
9. **3 screen-identity exclusions in a FOURTH form -- recipe §3's sixth
   bypass shape, as a MEMBERSHIP test rather than an ``is`` comparison.**
   ``_commit_library_media_return``, ``_library_media_focus_target_matches_
   receipt`` and ``_resolve_library_media_settlement_target`` each spell
   ``self in <widget>.ancestors`` / ``self not in <widget>.ancestors``.
   ``Widget.ancestors`` is the DOM parent chain up to the screen, so a moved
   body's ``self`` -- the CONTROLLER -- is never a member: the guard flips to
   permanently False (or permanently True for the negated form) and silently
   changes behaviour, exactly like Forms A/B/C. No accommodation exists;
   identity cannot be satisfied by a proxy.

   **How it was found, and the census gap it exposes.** The recipe's own
   sixth-shape census, as written, looks for ``... is self`` / ``... is not
   self`` (an ``ast.Compare`` whose operator is ``Is``/``IsNot``) and for
   bare ``self`` passed as a call ARGUMENT. This shape is neither: it is an
   ``ast.Compare`` with an ``In``/``NotIn`` operator, and a first pass of
   that census over all 251 candidates returned **zero** identity hits. The
   move shipped into this task's own battery and **19 of 163 tests failed in
   the return-settlement cluster against a 2-of-163 isolated baseline** --
   17 branch-unique, the signature assertion being ``assert
   'layout-settlement-failed' == 'exact-settled'``. **The standing
   correction: census bare ``self`` as an operand of ANY comparison
   operator, not just ``Is``/``IsNot`` -- and, stronger and cheaper, census
   EVERY bare ``self`` ``ast.Name`` in a moved body that is not the receiver
   of an attribute access.** That stronger form is exhaustive by
   construction. Run at the moment of the finding -- over the then-current
   mover set, the three ``ancestors`` methods still in it -- it returned 10
   occurrences: 3 ``getattr(self, "<literal>")`` reads, 4
   ``_sync_library_canvas(self, ...)`` duck-typed forwards, and the 3
   ``ancestors`` membership tests it exists to catch. Re-run over the FINAL
   140 movers, with those 3 excluded, it returns **7** -- the 3 ``getattr``
   reads and the 4 canvas forwards, every one of them bound. Both figures
   are stated because only the pair shows the census working: 10 is what it
   found, 7 is what survives.

   **The shape is more common than one incident suggests: there are FOUR
   ``self in/not in <widget>.ancestors`` sites in the 251-candidate
   cluster, not three.** The fourth is ``_library_media_settlement_tree``
   (``library_screen.py:5872`` at this commit, ``:5646`` at the parent
   ``3dd7a745a``), which is already excluded under class 2 above and is
   therefore not reclassified -- but it means a subsystem can carry this
   shape in a method whose exclusion happens to be over-determined, and a
   census that stops at the methods it had to newly exclude will undercount
   how widespread the shape is.

10. **1 callback-identity exclusion -- a TENTH bypass shape, also found by
   this task's own battery.** ``_exit_library_media_viewer``'s body schedules
   its continuation as ``self.call_next(self._apply_library_media_list_
   return, None)``, and ``test_screen_navigation.py:3109`` asserts
   ``continuation == screen._apply_library_media_list_return`` on a REAL
   screen. Moved, ``self`` is the controller, so the captured callback is the
   CONTROLLER's bound method and the equality fails -- and note that
   excluding the CALLEE would not have fixed it either: the controller's
   late-binding property returns the injected ``lambda``, which is a third
   object again. The CALLER is what has to stay, so that ``self.<name>``
   resolves to a screen-bound method whose ``__self__``/``__func__`` match
   the assertion's own. The census: any test comparing a captured callback
   against ``<receiver>.<cluster-name>`` as a BARE attribute (not a call, not
   an assignment target). Run over all 251 candidates across ``Tests/`` it
   returns 10 names, 9 of them already excluded on other grounds and 1
   (``_navigate_to_media``) where the receiver holds a ``MagicMock`` set on
   the instance, which a delegator cannot disturb.

**140 of the 251 candidates move onto this controller** (48 ``@on`` handlers
+ 5 ``action_*`` + 3 ``@staticmethod`` + 84 plain).

**Byte-for-byte canon** (moved bodies never edited -- every name they
reference that is not this controller's own state is rebound under the SAME
name, per the two binding kinds; see ``LibraryPromptsController.__init__``
and ``ConsoleDictationController.__init__`` for the sibling worked
examples). The binding surface was derived MECHANICALLY, by walking all 140
moved bodies for every ``self.<attr>`` load/store AND every
``getattr(self, "<literal>")`` call, then subtracting this controller's own
state fields and the movers themselves -- **90 names** -- and then adding
**2 more the walk cannot see**, for **92**, every one pinned in
``test_media_controller_binds_every_name_its_moved_bodies_use``:

1. **Framework services** (``app``, ``app_instance``, ``call_after_refresh``,
   ``call_next``, ``focused``, ``is_mounted``, ``is_running``,
   ``post_message``, ``query``, ``query_one``, ``refresh``, ``run_worker``,
   ``set_focus``, ``set_timer``, ``size``) are live-read from the screen via
   ``@property`` on every access -- never snapshotted. ``focused`` is here
   for the reason the skills series established: six movers reach it, and an
   unbound ``getattr(self, "focused", None)`` returns its default silently,
   forever.
2. **Everything else** is a NAMED constructor dependency: (a) 24 general
   Library-wide shell helpers a moved body calls with explicit arguments --
   including the five review-set entry points the One Home Rule kept on the
   screen (``_active_review_set_banner``, ``_review_dismiss_receipt_name``,
   ``_review_selected_worker``, ``_review_set_picker_worker``,
   ``_review_these_worker``, ``_walk_active_review_set``); (b) 11 shared
   shell state accessors this cluster READS and never writes; (b') **5 it
   also WRITES** (``_library_selected_row_id``,
   ``_library_notes_programmatic_focus_target``,
   ``_library_notes_restoring_focus``, ``_library_list_entry_focus_timer``,
   ``_library_canvas_resync_pending``) -- getter **and** setter, a shape the
   prompts series did not need (its census found zero stores to a
   non-own-state name) and this one does; (c) 2 wiring accessors for the
   prior-extracted media controller instances task 1 kept off
   ``LibraryMediaState`` (``_library_media_browse_controller``,
   ``_library_media_trash_browse_controller`` -- 11 movers each, the
   cluster's most-referenced names); (e) 35 named late-binding callables for
   the exclusions above that a MOVER still calls internally -- each a
   ``lambda`` re-reading ``screen.<name>`` on every invocation, at CALL time,
   which is exactly why every bypass fixture named above keeps working
   unmodified after this move.

**The two bindings a ``self.<attr>`` walk cannot produce, and the precedent
that already knew about them.** ``_library_canvas_projection_depth`` and
``_library_canvas_resync_pending`` appear in NO moved body. They are read and
written by the SHARED ``_sync_library_canvas`` dispatcher, off the bare
``self`` those four movers hand it -- so on the controller they resolve
against the CONTROLLER, not the screen. The first is read as
``getattr(screen, "_library_canvas_projection_depth", 0)``: unbound, it
returns 0 forever, permanently disabling the projection-replay branch with no
exception and no red test anywhere in the standard battery -- the skills
series' ``focused`` shape, one indirection further out. This wave's own
mechanical binding census does NOT produce either name, and would have shipped
without them; what caught it was tracing the dispatcher's own body for the
media ``kind``. **The conversations exemplar had already found the identical
pair and bound it** (``library_conversations_controller.py``'s own module
docstring, group (c)/(d), and its
``_library_canvas_projection_depth``/``_library_canvas_resync_pending``
properties) -- so the finding here is a re-derivation, and the standing
lesson is the one that precedent implies: **any controller whose movers
forward bare ``self`` into ``_sync_library_canvas`` must read that
dispatcher's own body for its ``kind``, because the recipe's ``self.<attr>``
census over the moved bodies structurally cannot see it.** The full media path
of ``_sync_library_canvas`` needs 15
names off the object it is handed; the other 13 are covered (``query_one``,
``refresh``, ``call_after_refresh``, ``app``, ``is_running`` as framework
services; ``_selected_media_id`` through the state shim loop;
``_build_library_media_state``, ``_library_media_canvas_presentation``,
``_capture_library_media_focus_identity``, ``_restore_library_media_focus``,
``_library_media_trash_canvas_presentation`` as movers;
``_build_library_media_trash_state`` and
``_sync_library_skills_reader_layout_from_shell`` as named dependencies).

**Four class-level constants relocated, not duplicated.** ``_MEDIA_VIEW_LIST``,
``_MEDIA_VIEW_VIEWER``, ``_ANALYZE_ORIGIN_MEDIA`` and ``_ANALYZE_ORIGIN_IMPORT``
were module-level assignments in ``library_screen.py`` that moved bodies read
as bare globals. They move verbatim (comments included) to
``screen_constants.py`` -- the established home, the ingest series'
dead-zone-constant precedent -- and ``library_screen.py`` imports them back,
so ``library_screen_module._ANALYZE_ORIGIN_MEDIA``/``_ANALYZE_ORIGIN_IMPORT``
still resolve for ``test_library_ingest_analyze_skipped.py:611``/``:612``,
the two assertions task 1's forward note flagged. Re-spelling ``"media"`` in
a second file would have broken the single-source guarantee those constants
exist to provide.

**Construction order and import shape.** ``LibraryScreen.__init__`` builds
``self._media_controller`` right after ``self._prompts_controller``, matching
every other controller in that file, and the controller class is imported
**function-locally**, inside ``__init__``'s established lazy-import block --
never at module level. That is a hard constraint of this wave, not a style
preference: ``library_screen.py`` is deliberately absent from the
``_ui_ready`` boot-budget snapshot
(``Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt``), and both
``Tests/Packaging/test_library_preimport_closure.py`` and
``Tests/Performance/test_ui_ready_module_census.py`` enforce it.

This subsystem's OWN state (every flat media field name the moved bodies
reference -- all 82, across TWO prefix families, resolved by task 1's
single-source ``media_state_shim_attr()``) is exposed through a generated
property loop reading ``self._media_state_accessor().<field>`` -- the same
generator shape task 1 installed on ``LibraryScreen`` (task 3 deleted that
screen block once this controller's own copy made it dead).
"""
from __future__ import annotations

import asyncio
import dataclasses
import time
import webbrowser
from collections.abc import Mapping
from functools import partial
from typing import Any, Literal, TYPE_CHECKING

from loguru import logger
from textual import on
from textual.containers import Vertical
from textual.css.query import NoMatches, QueryError
from textual.widget import Widget
from textual.widgets import Button, Input, OptionList, TextArea

from ...Chat.Chat_Functions import chat_api_call, extract_response_content
from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Library.ingest_analysis import (
    analysis_unavailable_reason,
    resolve_ingest_analysis_provider,
)
from ...Library.library_export_scope import ExportScope
from ...Library.library_media_reader_state import (
    SELECTION_SETTLE_SECONDS,
    LibraryMediaReaderSessionState,
    begin_selection,
    enter_external_detail,
    leave_external_detail,
    set_mode,
    set_more_open,
)
from ...Library.library_media_state import (
    MEDIA_SORT_CHOICES,
    LibraryMediaCanvasState,
    MediaBrowseScope,
    MediaTrashScope,
    build_library_media_browse_state,
    build_library_media_state,
    # (wave-7 round-2 merge) `handle_library_media_review_selected`'s ported
    # body reads this as a BARE module global, so it resolves against THIS
    # module's `__globals__` -- recipe SS3's module-globals-coupling shape. It
    # must be imported here, not reached through the screen.
    library_media_int_backing_id,
)
from ...Library.library_media_viewer_state import (
    build_library_media_highlight_rows,
    build_library_media_viewer_state,
)
from ...Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_BROWSE_PROMPTS,
    LIBRARY_ROW_BROWSE_SKILLS,
    LIBRARY_ROW_CREATE_NOTE,
    LIBRARY_ROW_CREATE_PROMPT,
    LIBRARY_ROW_CREATE_SKILL,
    library_disabled_action_label,
)
from ...Widgets.Library import (
    LIBRARY_ADAPTIVE_READER_GRIP_CLASS,
    LibraryMediaCanvas,
    LibraryBrowseReaderShell,
    LibraryMediaTrashCanvas,
    LibraryMediaViewer,
    MediaShellResized,
)
from ...Widgets.Library.library_media_canvas import (
    LibraryMediaRowGeometry,
    LibraryMediaRowGeometryChanged,
    LibraryMediaRowScroll,
)
from ...Widgets.Library.library_media_content import LibraryMediaContentBody
from ..Navigation.main_navigation import NavigateToScreen
from .canvas_sync import _sync_library_canvas
from .library_media_state import LibraryMediaState, media_state_shim_attr
from .screen_constants import (
    _ANALYZE_ORIGIN_IMPORT,
    _ANALYZE_ORIGIN_MEDIA,
    _MEDIA_VIEW_LIST,
    _MEDIA_VIEW_VIEWER,
    LIBRARY_MEDIA_HANDOFF_EXCERPT_CHARS,
    LIBRARY_MEDIA_PREVIEW_CACHE_LIMIT,
    LIBRARY_NOTES_SOURCE_DATABASE,
)
from .screen_support_types import (
    _LibraryMediaReturnReceipt,
    _LibraryMediaReturnSettlement,
    _LibraryMediaSettlementOutcome,
    _LibraryMediaSuccessfulFocusOwnership,
)

if TYPE_CHECKING:
    from ..Screens.library_screen import LibraryScreen


class LibraryMediaController:
    """Owns the Library Media cluster (140 methods).

    Holds no state of its own beyond what it reads and writes through
    ``LibraryMediaState`` (via the injected accessor) and the shared
    shell/framework/wiring bindings below. ``LibraryScreen`` constructs
    exactly one of these, in ``__init__`` right after
    ``self._prompts_controller``, and keeps a one-line delegator for 118 of
    the 140 original names this cluster moved -- task 3 (the cleanup PR,
    media series 3/3) pruned the 22 nothing external reaches.
    """

    def __init__(
        self,
        screen: "LibraryScreen",
        *,
        media_state_accessor,
        # -- shared shell state this cluster READS (group (b), getter-only).
        library_canvas_projection_depth_accessor,
        library_compose_generation_accessor,
        library_list_entry_focus_deadline_accessor,
        library_list_entry_focus_generation_accessor,
        library_notes_compact_accessor,
        library_notes_focus_intent_generation_accessor,
        library_notes_source_accessor,
        library_pending_list_entry_focus_accessor,
        library_pending_list_entry_focus_anchor_accessor,
        library_pending_list_entry_media_return_accessor,
        local_source_records_accessor,
        # -- shared shell state this cluster also WRITES (group (b'),
        #    getter + setter; the mechanical binding census found exactly
        #    five such names).
        library_canvas_resync_pending_accessor,
        set_library_canvas_resync_pending,
        library_list_entry_focus_timer_accessor,
        set_library_list_entry_focus_timer,
        library_notes_programmatic_focus_target_accessor,
        set_library_notes_programmatic_focus_target,
        library_notes_restoring_focus_accessor,
        set_library_notes_restoring_focus,
        library_selected_row_id_accessor,
        set_library_selected_row_id,
        # -- the 2 prior-extracted media WIRING controller instances task 1
        #    deliberately kept off `LibraryMediaState` (group (c)).
        library_media_browse_controller_accessor,
        library_media_trash_browse_controller_accessor,
        # -- general Library-wide shell helpers, not moved (group (a)).
        acknowledge_library_destination_change,
        active_review_set_banner,
        apply_library_open_item_surface,
        arm_library_list_entry_focus,
        disarm_library_list_entry_focus,
        focus_library_control,
        focus_library_list_entry,
        library_entry_route_key,
        open_library_export_canvas,
        open_library_item_by_id,
        register_footer_shortcuts,
        review_dismiss_receipt_name,
        review_selected_worker,
        review_set_picker_worker,
        review_these_worker,
        run_library_service_call,
        safe_text,
        source_record_id,
        sync_library_conversation_reader_layout_from_shell,
        sync_library_notes_reader_layout_from_shell,
        sync_library_prompts_reader_layout_from_shell,
        sync_library_skills_reader_layout_from_shell,
        walk_active_review_set,
        request_library_reader_layout_refresh,
        # -- named late-binding callables for the media exclusions (group
        #    (e)) that a MOVER still calls internally.
        after_library_media_viewer_sync,
        arm_library_media_return_settlement,
        build_library_media_trash_state,
        cancel_library_media_bulk_delete,
        capture_library_media_loaded_progress,
        clear_library_media_selection_for_scope_change,
        close_library_media_find,
        commit_library_media_return,
        decorate_library_media_reviewed,
        exit_library_media_trash,
        exit_library_media_viewer,
        library_media_analysis_provider_reason,
        library_media_backing_id,
        library_media_can_rename_speakers,
        library_media_content_matches,
        library_media_content_signature,
        library_media_focus_target_matches_receipt,
        library_media_layout_signature,
        library_media_list_unselectable,
        library_media_reader_exit_available,
        library_media_return_candidate,
        library_media_selected_backing_id,
        library_media_settlement_tree,
        library_media_trash_action_disabled_reason,
        library_media_trash_focus_selectors,
        library_media_trash_retry_visible,
        library_media_type_options,
        library_media_viewer_state_cached,
        open_selected_media_handoff,
        queue_after_library_media_viewer_recompose,
        reconcile_library_media_stage_presentation,
        refresh_library_media_detail,
        request_library_media_browse,
        reset_library_media_search_on_mode_change,
        restore_library_media_loaded_progress,
        restore_library_media_reader_width_on_open,
        sanitize_media_field,
        save_library_media_analysis,
        stop_library_media_filter_timer,
        sync_library_media_browse_state,
        sync_library_media_reader_layout_from_shell,
        sync_library_media_surfaces_or_recompose,
        sync_library_media_trash_state,
        toggle_library_media_select_mode,
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the 140 method bodies below is a byte-for-byte copy of
        the pre-extraction ``LibraryScreen`` method: no internal line was
        edited to retarget a call or an attribute. That is possible because
        this constructor binds every name those bodies reference that is not
        this controller's own state, under the SAME name the original method
        used. See the module docstring for the binding kinds this follows and
        the full per-parameter derivation.

        Args:
            screen: The Library screen. Used ONLY for the fifteen framework
                services below (``app``, ``app_instance``,
                ``call_after_refresh``, ``call_next``, ``focused``,
                ``is_mounted``, ``is_running``, ``post_message``, ``query``,
                ``query_one``, ``refresh``, ``run_worker``, ``set_focus``,
                ``set_timer``, ``size``) -- this cluster owns no DOM of its
                own.
            media_state_accessor: Returns the live ``LibraryMediaState``
                (``LibraryScreen._media_state``, task 1). Backs every
                generated flat media-field property below.
        """
        self._screen = screen
        self._media_state_accessor = media_state_accessor
        self._library_canvas_projection_depth_accessor = library_canvas_projection_depth_accessor
        self._library_compose_generation_accessor = library_compose_generation_accessor
        self._library_list_entry_focus_deadline_accessor = library_list_entry_focus_deadline_accessor
        self._library_list_entry_focus_generation_accessor = library_list_entry_focus_generation_accessor
        self._library_notes_compact_accessor = library_notes_compact_accessor
        self._library_notes_focus_intent_generation_accessor = library_notes_focus_intent_generation_accessor
        self._library_notes_source_accessor = library_notes_source_accessor
        self._library_pending_list_entry_focus_accessor = library_pending_list_entry_focus_accessor
        self._library_pending_list_entry_focus_anchor_accessor = library_pending_list_entry_focus_anchor_accessor
        self._library_pending_list_entry_media_return_accessor = library_pending_list_entry_media_return_accessor
        self._local_source_records_accessor = local_source_records_accessor
        self._library_canvas_resync_pending_accessor = library_canvas_resync_pending_accessor
        self._set_library_canvas_resync_pending_fn = set_library_canvas_resync_pending
        self._library_list_entry_focus_timer_accessor = library_list_entry_focus_timer_accessor
        self._set_library_list_entry_focus_timer_fn = set_library_list_entry_focus_timer
        self._library_notes_programmatic_focus_target_accessor = library_notes_programmatic_focus_target_accessor
        self._set_library_notes_programmatic_focus_target_fn = set_library_notes_programmatic_focus_target
        self._library_notes_restoring_focus_accessor = library_notes_restoring_focus_accessor
        self._set_library_notes_restoring_focus_fn = set_library_notes_restoring_focus
        self._library_selected_row_id_accessor = library_selected_row_id_accessor
        self._set_library_selected_row_id_fn = set_library_selected_row_id
        self._library_media_browse_controller_accessor = library_media_browse_controller_accessor
        self._library_media_trash_browse_controller_accessor = library_media_trash_browse_controller_accessor
        self._acknowledge_library_destination_change_fn = acknowledge_library_destination_change
        self._active_review_set_banner_fn = active_review_set_banner
        self._apply_library_open_item_surface_fn = apply_library_open_item_surface
        self._arm_library_list_entry_focus_fn = arm_library_list_entry_focus
        self._disarm_library_list_entry_focus_fn = disarm_library_list_entry_focus
        self._focus_library_control_fn = focus_library_control
        self._focus_library_list_entry_fn = focus_library_list_entry
        self._library_entry_route_key_fn = library_entry_route_key
        self._open_library_export_canvas_fn = open_library_export_canvas
        self._open_library_item_by_id_fn = open_library_item_by_id
        self._register_footer_shortcuts_fn = register_footer_shortcuts
        self._review_dismiss_receipt_name_fn = review_dismiss_receipt_name
        self._review_selected_worker_fn = review_selected_worker
        self._review_set_picker_worker_fn = review_set_picker_worker
        self._review_these_worker_fn = review_these_worker
        self._run_library_service_call_fn = run_library_service_call
        self._safe_text_fn = safe_text
        self._source_record_id_fn = source_record_id
        self._sync_library_conversation_reader_layout_from_shell_fn = sync_library_conversation_reader_layout_from_shell
        self._sync_library_notes_reader_layout_from_shell_fn = sync_library_notes_reader_layout_from_shell
        self._sync_library_prompts_reader_layout_from_shell_fn = sync_library_prompts_reader_layout_from_shell
        self._sync_library_skills_reader_layout_from_shell_fn = sync_library_skills_reader_layout_from_shell
        self._walk_active_review_set_fn = walk_active_review_set
        self.request_library_reader_layout_refresh_fn = request_library_reader_layout_refresh
        self._after_library_media_viewer_sync_fn = after_library_media_viewer_sync
        self._arm_library_media_return_settlement_fn = arm_library_media_return_settlement
        self._build_library_media_trash_state_fn = build_library_media_trash_state
        self._cancel_library_media_bulk_delete_fn = cancel_library_media_bulk_delete
        self._capture_library_media_loaded_progress_fn = capture_library_media_loaded_progress
        self._clear_library_media_selection_for_scope_change_fn = clear_library_media_selection_for_scope_change
        self._close_library_media_find_fn = close_library_media_find
        self._commit_library_media_return_fn = commit_library_media_return
        self._decorate_library_media_reviewed_fn = decorate_library_media_reviewed
        self._exit_library_media_trash_fn = exit_library_media_trash
        self._exit_library_media_viewer_fn = exit_library_media_viewer
        self._library_media_analysis_provider_reason_fn = library_media_analysis_provider_reason
        self._library_media_backing_id_fn = library_media_backing_id
        self._library_media_can_rename_speakers_fn = library_media_can_rename_speakers
        self._library_media_content_matches_fn = library_media_content_matches
        self._library_media_content_signature_fn = library_media_content_signature
        self._library_media_focus_target_matches_receipt_fn = library_media_focus_target_matches_receipt
        self._library_media_layout_signature_fn = library_media_layout_signature
        self._library_media_list_unselectable_fn = library_media_list_unselectable
        self._library_media_reader_exit_available_fn = library_media_reader_exit_available
        self._library_media_return_candidate_fn = library_media_return_candidate
        self._library_media_selected_backing_id_fn = library_media_selected_backing_id
        self._library_media_settlement_tree_fn = library_media_settlement_tree
        self._library_media_trash_action_disabled_reason_fn = library_media_trash_action_disabled_reason
        self._library_media_trash_focus_selectors_fn = library_media_trash_focus_selectors
        self._library_media_trash_retry_visible_fn = library_media_trash_retry_visible
        self._library_media_type_options_fn = library_media_type_options
        self._library_media_viewer_state_cached_fn = library_media_viewer_state_cached
        self._open_selected_media_handoff_fn = open_selected_media_handoff
        self._queue_after_library_media_viewer_recompose_fn = queue_after_library_media_viewer_recompose
        self._reconcile_library_media_stage_presentation_fn = reconcile_library_media_stage_presentation
        self._refresh_library_media_detail_fn = refresh_library_media_detail
        self._request_library_media_browse_fn = request_library_media_browse
        self._reset_library_media_search_on_mode_change_fn = reset_library_media_search_on_mode_change
        self._restore_library_media_loaded_progress_fn = restore_library_media_loaded_progress
        self._restore_library_media_reader_width_on_open_fn = restore_library_media_reader_width_on_open
        self._sanitize_media_field_fn = sanitize_media_field
        self._save_library_media_analysis_fn = save_library_media_analysis
        self._stop_library_media_filter_timer_fn = stop_library_media_filter_timer
        self._sync_library_media_browse_state_fn = sync_library_media_browse_state
        self._sync_library_media_reader_layout_from_shell_fn = sync_library_media_reader_layout_from_shell
        self._sync_library_media_surfaces_or_recompose_fn = sync_library_media_surfaces_or_recompose
        self._sync_library_media_trash_state_fn = sync_library_media_trash_state
        self._toggle_library_media_select_mode_fn = toggle_library_media_select_mode

    # -- framework services: live-read properties, never snapshotted ------

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
    def call_next(self) -> Any:
        return self._screen.call_next

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
    def post_message(self) -> Any:
        return self._screen.post_message

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
    def run_worker(self) -> Any:
        return self._screen.run_worker

    @property
    def set_focus(self) -> Any:
        return self._screen.set_focus

    @property
    def set_timer(self) -> Any:
        return self._screen.set_timer

    @property
    def size(self) -> Any:
        return self._screen.size


    # -- this cluster's OWN state, under the screen's own name -------------

    @property
    def _media_state(self) -> Any:
        """This cluster's OWN state object, under the screen's own name.

        (wave-8 close, a BEHAVIOUR FIX -- the wave's one deliberate behaviour
        change.) No moved body spells ``self._media_state``: every one of the
        140 of them reads a flat ``_library_media_<field>`` property from the
        generated shim loop at the bottom of this file, and that is how the
        byte-for-byte canon keeps them unedited. This accessor exists for the
        SHARED dispatchers in ``canvas_sync.py``, which are handed a bare
        ``self`` by this cluster's own methods and so have to resolve the
        media state on EITHER receiver: the screen
        (``LibraryScreen._media_state``) or this controller.

        Without it the wave-7 dotted retarget was only half done, and the
        missing half was live in production: ``_sync_library_canvas``'s media
        leg assigns through ``screen._media_state.selected_media_id``
        (``canvas_sync.py``), and ``handle_library_media_select_all`` /
        ``handle_library_media_select_clear`` below call
        ``_sync_library_canvas(self, "media")`` with a CONTROLLER ``self``, so
        every media "Select all"/"Clear" press raised ``AttributeError`` into
        that dispatcher's own ``except Exception`` and degraded into the
        whole-screen recompose the Tier-1 targeted-sync design exists to
        avoid -- with no exception surfaced and no red test. Found by the
        wave-8 notes cleanup's receiver census (the notes controller declares
        the same accessor for the same reason) and fixed here.

        Mutation-verified: removing this property reds
        ``test_media_row_toggle_resolves_the_dotted_state_path[controller]``
        and leaves the ``[screen]`` leg green; reverting ``canvas_sync.py``'s
        media branch to the computed flat name reds the ``[screen]`` leg and
        leaves ``[controller]`` green (the flat name still resolves through
        the shim loop below).
        """
        return self._media_state_accessor()


    # -- shared shell state, read-only (group (b)) -------------------------

    @property
    def _library_canvas_projection_depth(self) -> Any:
        return self._library_canvas_projection_depth_accessor()

    @property
    def _library_compose_generation(self) -> Any:
        return self._library_compose_generation_accessor()

    @property
    def _library_list_entry_focus_deadline(self) -> Any:
        return self._library_list_entry_focus_deadline_accessor()

    @property
    def _library_list_entry_focus_generation(self) -> Any:
        return self._library_list_entry_focus_generation_accessor()

    @property
    def _library_notes_compact(self) -> Any:
        return self._library_notes_compact_accessor()

    @property
    def _library_notes_focus_intent_generation(self) -> Any:
        return self._library_notes_focus_intent_generation_accessor()

    @property
    def _library_notes_source(self) -> Any:
        return self._library_notes_source_accessor()

    @property
    def _library_pending_list_entry_focus(self) -> Any:
        return self._library_pending_list_entry_focus_accessor()

    @property
    def _library_pending_list_entry_focus_anchor(self) -> Any:
        return self._library_pending_list_entry_focus_anchor_accessor()

    @property
    def _library_pending_list_entry_media_return(self) -> Any:
        return self._library_pending_list_entry_media_return_accessor()

    @property
    def _local_source_records(self) -> Any:
        return self._local_source_records_accessor()

    @property
    def _library_media_unfiltered_total(self) -> int | None:
        """The rail's whole-Media count, for the scope line's "N of M".

        task-32350. Read straight off the screen: the accessor-injection
        site lives in a ``library_screen.py`` range this branch does not
        own, and these three names are getter-only shell state of exactly
        the kind ``local_source_records_accessor`` already carries.

        Returns:
            The Library's Media count, or ``None`` while the snapshot is
            not trustworthy -- the count's ``0`` default would otherwise
            render as "1 of 0".
        """
        screen = self._screen
        if not screen._library_loaded or screen._library_lookup_error:
            return None
        # Review finding 2 (fix round 1): the count is sometimes a LOWER
        # BOUND -- a bounded preview page with no `total` -- which is why
        # the screen keeps this flag and why every other consumer renders
        # "5+" rather than "5". A scope line stating a flat "of 5" beside a
        # rail saying "5+" is exactly the claim this line exists to avoid,
        # so an unknown total drops the "of M" half instead.
        if not screen._local_source_total_known.get("media", True):
            return None
        return screen._local_source_counts.get("media")


    # -- shared shell state this cluster also writes (group (b')) ----------

    @property
    def _library_canvas_resync_pending(self) -> Any:
        return self._library_canvas_resync_pending_accessor()

    @_library_canvas_resync_pending.setter
    def _library_canvas_resync_pending(self, value: Any) -> None:
        self._set_library_canvas_resync_pending_fn(value)

    @property
    def _library_list_entry_focus_timer(self) -> Any:
        return self._library_list_entry_focus_timer_accessor()

    @_library_list_entry_focus_timer.setter
    def _library_list_entry_focus_timer(self, value: Any) -> None:
        self._set_library_list_entry_focus_timer_fn(value)

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


    # -- prior-extracted media wiring controllers (group (c)) --------------

    @property
    def _library_media_browse_controller(self) -> Any:
        return self._library_media_browse_controller_accessor()

    @property
    def _library_media_trash_browse_controller(self) -> Any:
        return self._library_media_trash_browse_controller_accessor()


    # -- general Library-wide shell helpers (group (a)) --------------------

    @property
    def _acknowledge_library_destination_change(self) -> Any:
        return self._acknowledge_library_destination_change_fn

    @property
    def _active_review_set_banner(self) -> Any:
        return self._active_review_set_banner_fn

    @property
    def _apply_library_open_item_surface(self) -> Any:
        return self._apply_library_open_item_surface_fn

    @property
    def _arm_library_list_entry_focus(self) -> Any:
        return self._arm_library_list_entry_focus_fn

    @property
    def _disarm_library_list_entry_focus(self) -> Any:
        return self._disarm_library_list_entry_focus_fn

    @property
    def _focus_library_control(self) -> Any:
        return self._focus_library_control_fn

    @property
    def _focus_library_list_entry(self) -> Any:
        return self._focus_library_list_entry_fn

    @property
    def _library_entry_route_key(self) -> Any:
        return self._library_entry_route_key_fn

    @property
    def _open_library_export_canvas(self) -> Any:
        return self._open_library_export_canvas_fn

    @property
    def _open_library_item_by_id(self) -> Any:
        return self._open_library_item_by_id_fn

    @property
    def _register_footer_shortcuts(self) -> Any:
        return self._register_footer_shortcuts_fn

    @property
    def _review_dismiss_receipt_name(self) -> Any:
        return self._review_dismiss_receipt_name_fn

    @property
    def _review_selected_worker(self) -> Any:
        return self._review_selected_worker_fn

    @property
    def _review_set_picker_worker(self) -> Any:
        return self._review_set_picker_worker_fn

    @property
    def _review_these_worker(self) -> Any:
        return self._review_these_worker_fn

    @property
    def _run_library_service_call(self) -> Any:
        return self._run_library_service_call_fn

    @property
    def _safe_text(self) -> Any:
        return self._safe_text_fn

    @property
    def _source_record_id(self) -> Any:
        return self._source_record_id_fn

    @property
    def _sync_library_conversation_reader_layout_from_shell(self) -> Any:
        return self._sync_library_conversation_reader_layout_from_shell_fn

    @property
    def _sync_library_notes_reader_layout_from_shell(self) -> Any:
        return self._sync_library_notes_reader_layout_from_shell_fn

    @property
    def _sync_library_prompts_reader_layout_from_shell(self) -> Any:
        return self._sync_library_prompts_reader_layout_from_shell_fn

    @property
    def _sync_library_skills_reader_layout_from_shell(self) -> Any:
        return self._sync_library_skills_reader_layout_from_shell_fn

    @property
    def _walk_active_review_set(self) -> Any:
        return self._walk_active_review_set_fn

    @property
    def request_library_reader_layout_refresh(self) -> Any:
        return self.request_library_reader_layout_refresh_fn


    # -- named late-binding callables for the exclusions (group (e)) -------
    #
    # The last EIGHT are dev-side screen methods this wave's reconciliation
    # merge introduced: the 11 dev-edited mover bodies ported in here call
    # them, and they stay screen-resident (they are dev's new methods, not
    # this wave's movers). Same late-binding shape as every other row.

    @property
    def _after_library_media_viewer_sync(self) -> Any:
        return self._after_library_media_viewer_sync_fn

    @property
    def _arm_library_media_return_settlement(self) -> Any:
        return self._arm_library_media_return_settlement_fn

    @property
    def _build_library_media_trash_state(self) -> Any:
        return self._build_library_media_trash_state_fn

    @property
    def _cancel_library_media_bulk_delete(self) -> Any:
        return self._cancel_library_media_bulk_delete_fn

    @property
    def _capture_library_media_loaded_progress(self) -> Any:
        return self._capture_library_media_loaded_progress_fn

    @property
    def _clear_library_media_selection_for_scope_change(self) -> Any:
        return self._clear_library_media_selection_for_scope_change_fn

    @property
    def _close_library_media_find(self) -> Any:
        return self._close_library_media_find_fn

    @property
    def _commit_library_media_return(self) -> Any:
        return self._commit_library_media_return_fn

    @property
    def _decorate_library_media_reviewed(self) -> Any:
        return self._decorate_library_media_reviewed_fn

    @property
    def _exit_library_media_trash(self) -> Any:
        return self._exit_library_media_trash_fn

    @property
    def _exit_library_media_viewer(self) -> Any:
        return self._exit_library_media_viewer_fn

    @property
    def _library_media_analysis_provider_reason(self) -> Any:
        return self._library_media_analysis_provider_reason_fn

    @property
    def _library_media_backing_id(self) -> Any:
        return self._library_media_backing_id_fn

    @property
    def _library_media_can_rename_speakers(self) -> Any:
        return self._library_media_can_rename_speakers_fn

    @property
    def _library_media_content_matches(self) -> Any:
        return self._library_media_content_matches_fn

    @property
    def _library_media_content_signature(self) -> Any:
        return self._library_media_content_signature_fn

    @property
    def _library_media_focus_target_matches_receipt(self) -> Any:
        return self._library_media_focus_target_matches_receipt_fn

    @property
    def _library_media_layout_signature(self) -> Any:
        return self._library_media_layout_signature_fn

    @property
    def _library_media_list_unselectable(self) -> Any:
        return self._library_media_list_unselectable_fn

    @property
    def _library_media_reader_exit_available(self) -> Any:
        return self._library_media_reader_exit_available_fn

    @property
    def _library_media_return_candidate(self) -> Any:
        return self._library_media_return_candidate_fn

    @property
    def _library_media_selected_backing_id(self) -> Any:
        return self._library_media_selected_backing_id_fn

    @property
    def _library_media_settlement_tree(self) -> Any:
        return self._library_media_settlement_tree_fn

    @property
    def _library_media_trash_action_disabled_reason(self) -> Any:
        return self._library_media_trash_action_disabled_reason_fn

    @property
    def _library_media_trash_focus_selectors(self) -> Any:
        return self._library_media_trash_focus_selectors_fn

    @property
    def _library_media_trash_retry_visible(self) -> Any:
        return self._library_media_trash_retry_visible_fn

    @property
    def _library_media_type_options(self) -> Any:
        return self._library_media_type_options_fn

    @property
    def _library_media_viewer_state_cached(self) -> Any:
        return self._library_media_viewer_state_cached_fn

    @property
    def _open_selected_media_handoff(self) -> Any:
        return self._open_selected_media_handoff_fn

    @property
    def _queue_after_library_media_viewer_recompose(self) -> Any:
        return self._queue_after_library_media_viewer_recompose_fn

    @property
    def _reconcile_library_media_stage_presentation(self) -> Any:
        return self._reconcile_library_media_stage_presentation_fn

    @property
    def _refresh_library_media_detail(self) -> Any:
        return self._refresh_library_media_detail_fn

    @property
    def _request_library_media_browse(self) -> Any:
        return self._request_library_media_browse_fn

    @property
    def _reset_library_media_search_on_mode_change(self) -> Any:
        return self._reset_library_media_search_on_mode_change_fn

    @property
    def _restore_library_media_loaded_progress(self) -> Any:
        return self._restore_library_media_loaded_progress_fn

    @property
    def _restore_library_media_reader_width_on_open(self) -> Any:
        return self._restore_library_media_reader_width_on_open_fn

    @property
    def _sanitize_media_field(self) -> Any:
        return self._sanitize_media_field_fn

    @property
    def _save_library_media_analysis(self) -> Any:
        return self._save_library_media_analysis_fn

    @property
    def _stop_library_media_filter_timer(self) -> Any:
        return self._stop_library_media_filter_timer_fn

    @property
    def _sync_library_media_browse_state(self) -> Any:
        return self._sync_library_media_browse_state_fn

    @property
    def _sync_library_media_reader_layout_from_shell(self) -> Any:
        return self._sync_library_media_reader_layout_from_shell_fn

    @property
    def _sync_library_media_surfaces_or_recompose(self) -> Any:
        return self._sync_library_media_surfaces_or_recompose_fn

    @property
    def _sync_library_media_trash_state(self) -> Any:
        return self._sync_library_media_trash_state_fn

    @property
    def _toggle_library_media_select_mode(self) -> Any:
        return self._toggle_library_media_select_mode_fn

    # -- moved cluster methods (140), byte-for-byte, original file order --
    def _project_library_media_stage_classes(self, shell_grid: Widget) -> bool:
        """Project effective Media stage classes; return whether they changed."""
        changed = False
        if shell_grid.has_class("library-notes-compact"):
            shell_grid.remove_class("library-notes-compact")
            changed = True
        if shell_grid.has_class("library-adaptive-compact") != self._library_notes_compact:
            shell_grid.set_class(
                self._library_notes_compact,
                "library-adaptive-compact",
            )
            changed = True
        if changed:
            current_owner: LibraryMediaRowScroll | None = None
            if shell_grid.is_attached:
                tree = self._library_media_settlement_tree()
                if tree is not None and shell_grid in tree[0].ancestors:
                    current_owner = tree[2]
            self._advance_library_media_presentation_epoch(current_owner)
        return changed

    def _adopt_library_media_row_owner(self, owner: LibraryMediaRowScroll) -> bool:
        """Advance presentation authority when the row owner is replaced."""
        owner_identity = id(owner)
        if self._library_media_current_owner is owner:
            return False
        self._library_media_current_owner = owner
        self._library_media_presentation_epoch += 1
        self._library_media_geometry_floor_owner_identity = owner_identity
        self._library_media_geometry_floor = 0
        self._library_media_return_settlement = None
        return True

    def _advance_library_media_presentation_epoch(
        self,
        owner: LibraryMediaRowScroll | None,
        *,
        rearm: bool = True,
    ) -> None:
        """Fence geometry already emitted before a Media presentation change."""
        self._library_media_presentation_epoch += 1
        if owner is None:
            self._library_media_current_owner = None
            self._library_media_geometry_floor_owner_identity = None
            self._library_media_geometry_floor = 0
        else:
            self._library_media_current_owner = owner
            self._library_media_geometry_floor_owner_identity = id(owner)
            latest = owner.latest_geometry
            self._library_media_geometry_floor = (
                latest.revision if latest is not None else 0
            )
        self._library_media_return_settlement = None
        receipt = self._library_pending_list_entry_media_return
        if (
            rearm
            and owner is not None
            and self._library_media_return_candidate(receipt)
        ):
            self._arm_library_media_return_settlement(receipt)

    def _library_media_exact_return_candidate(
        self,
        receipt: _LibraryMediaReturnReceipt | None,
    ) -> bool:
        """Return whether one retained Media coordinate system is unchanged."""
        return bool(
            self._library_media_return_candidate(receipt)
            and receipt is not None
            and receipt.content_signature == self._library_media_content_signature()
            and receipt.layout_signature == self._library_media_layout_signature()
        )

    def _library_media_semantic_row_is_current(
        self,
        receipt: _LibraryMediaReturnReceipt,
        items_host: Vertical,
    ) -> bool:
        """Return whether the receipt's attached row belongs to current Items."""
        return any(
            row.is_attached
            and str(getattr(row, "media_id", "") or "") == receipt.stable_id
            and items_host in row.ancestors
            for row in self.query(".library-media-row")
        )

    def _library_media_request_matches_current_authority(
        self,
        request: _LibraryMediaReturnSettlement,
        receipt: _LibraryMediaReturnReceipt,
        tree: tuple[LibraryBrowseReaderShell, Vertical, LibraryMediaRowScroll],
    ) -> bool:
        """Compare one immutable request with every non-geometry fence."""
        shell, items_host, owner = tree
        return bool(
            request.receipt is receipt
            and request.final_focus_policy == receipt.final_focus_policy
            and request.final_focus_identity == receipt.final_focus_identity
            and request.focus_intent_generation
            == self._library_notes_focus_intent_generation
            and request.compose_generation == self._library_compose_generation
            and request.media_lifecycle_generation
            == self._library_media_lifecycle_generation
            and request.presentation_epoch == self._library_media_presentation_epoch
            and request.content_signature == self._library_media_content_signature()
            and request.layout_signature == self._library_media_layout_signature()
            and request.route_identity == self._library_entry_route_key()
            and request.media_view_identity == self._library_media_view == "list"
            and request.shell_identity == id(shell)
            and request.items_host_identity == id(items_host)
            and request.owner_identity == id(owner)
            and request.focus_anchor
            is self._library_pending_list_entry_focus_anchor
            and self._library_media_live_focus_is_allowed(
                request.focus_anchor,
                receipt.stable_id,
            )
            and self._library_media_current_owner is owner
            and self._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
            and self._library_media_reader_layout.items_open
            and items_host.display
        )

    def _library_media_live_focus_is_allowed(
        self,
        focus_anchor: Widget | None,
        stable_id: str,
        target: Widget | None = None,
    ) -> bool:
        """Return whether live focus still belongs to one exact return."""
        focused = self.focused
        if focused is None or focused is focus_anchor:
            return True
        guarded_target = target or self._library_notes_programmatic_focus_target
        receipt = self._library_pending_list_entry_media_return
        transient_focus_allowed = bool(
            guarded_target is not None
            and focused is guarded_target
            and self._library_notes_programmatic_focus_target is guarded_target
            and receipt is not None
            and self._library_media_focus_target_matches_receipt(
                focused,
                receipt,
                stable_id,
            )
        )
        return transient_focus_allowed or bool(
            receipt is not None
            and self._library_media_successful_focus_is_allowed(
                receipt,
                stable_id,
            )
        )

    def _library_media_successful_focus_is_allowed(
        self,
        receipt: _LibraryMediaReturnReceipt,
        stable_id: str,
    ) -> bool:
        """Recognize exact-settled focus for this outer arm only."""
        ownership = self._library_media_successful_focus_ownership
        if ownership is None:
            return False
        request = ownership.request
        target = ownership.target
        return bool(
            self._library_pending_list_entry_focus
            and self._library_pending_list_entry_media_return is receipt
            and ownership.outer_generation
            == self._library_list_entry_focus_generation
            and request.receipt is receipt
            and request.request_id <= self._library_media_return_request_id
            and request.focus_intent_generation
            == self._library_notes_focus_intent_generation
            and request.route_identity == self._library_entry_route_key()
            and request.media_view_identity == self._library_media_view == "list"
            and request.final_focus_policy == receipt.final_focus_policy
            and request.final_focus_identity == receipt.final_focus_identity
            and request.receipt.stable_id == stable_id
            and ownership.selected_media_id == self._selected_media_id == stable_id
            and self.focused is target
            and self._library_media_focus_target_matches_receipt(
                target,
                receipt,
                stable_id,
            )
        )

    def _bind_library_media_settlement_deadline(self, request_id: int) -> None:
        """Bind the outer deadline to its Media request and focus generation."""
        deadline = self._library_list_entry_focus_deadline
        if deadline is None or not self._library_pending_list_entry_focus:
            return
        if self._library_list_entry_focus_timer is not None:
            self._library_list_entry_focus_timer.stop()
        remaining = max(0.0, deadline - time.monotonic())
        self._library_list_entry_focus_timer = self.set_timer(
            remaining,
            partial(
                self._expire_library_media_return_settlement,
                request_id,
                self._library_list_entry_focus_generation,
            ),
        )

    def _settle_library_media_return_from_geometry(
        self,
        request: _LibraryMediaReturnSettlement,
        owner: LibraryMediaRowScroll,
        geometry: LibraryMediaRowGeometry,
    ) -> bool:
        """Commit one honest scroll/focus outcome from current-owner geometry."""
        if (
            self._library_media_return_settlement is not request
            or self._library_media_return_settlement.request_id != request.request_id
        ):
            return False
        tree = self._library_media_settlement_tree()
        receipt = self._library_pending_list_entry_media_return
        if (
            tree is None
            or receipt is None
            or tree[2] is not owner
            or not self._library_media_request_matches_current_authority(
                request,
                receipt,
                tree,
            )
            or geometry.revision <= request.exclusive_geometry_floor
            or owner.latest_geometry != geometry
            or (owner.size, owner.virtual_size, owner.container_size)
            != (geometry.size, geometry.virtual_size, geometry.container_size)
        ):
            return False
        last_outcome = self._library_media_last_settlement_outcome
        if (
            last_outcome is not None
            and last_outcome[0] == request.request_id
            and last_outcome[2] == geometry.revision
        ):
            return True
        last_exact = self._library_media_last_exact_settlement
        if (
            last_exact is not None
            and last_exact[0].request_id == request.request_id
            and last_exact[1] >= geometry.revision
        ):
            return True
        desired = request.receipt.scroll_offset
        if desired is None:
            return False
        revised = not self._library_media_exact_return_candidate(request.receipt)
        if not revised and request.receipt.stable_id != self._selected_media_id:
            return False
        if revised:
            desired = (
                min(desired[0], int(owner.max_scroll_x)),
                min(desired[1], int(owner.max_scroll_y)),
            )
            outcome: _LibraryMediaSettlementOutcome = "clamped-after-revision"
        else:
            if (
                desired[0] > int(owner.max_scroll_x)
                or desired[1] > int(owner.max_scroll_y)
            ):
                return False
            outcome = "exact-settled"
        committed = self._commit_library_media_return(
            request,
            owner,
            geometry,
            desired,
            outcome,
        )
        if committed and revised:
            self._disarm_library_list_entry_focus()
        return committed

    def _expire_library_media_return_settlement(
        self,
        request_id: int,
        outer_generation: int,
    ) -> None:
        """Finish one ABA-bound outer arm without inferring layout readiness."""
        if (
            request_id != self._library_media_return_request_id
            or outer_generation != self._library_list_entry_focus_generation
        ):
            return
        request = self._library_media_return_settlement
        if request is None:
            self._disarm_library_list_entry_focus()
            return
        if request.request_id != request_id:
            return
        tree = self._library_media_settlement_tree()
        receipt = self._library_pending_list_entry_media_return
        if (
            tree is None
            or receipt is None
            or not self._library_media_request_matches_current_authority(
                request,
                receipt,
                tree,
            )
        ):
            self._disarm_library_list_entry_focus()
            return

        owner = tree[2]
        geometry = owner.latest_geometry
        committed = False
        if (
            geometry is not None
            and geometry.revision > request.exclusive_geometry_floor
            and (owner.size, owner.virtual_size, owner.container_size)
            == (geometry.size, geometry.virtual_size, geometry.container_size)
        ):
            desired = request.receipt.scroll_offset
            if desired is not None:
                committed = self._commit_library_media_return(
                    request,
                    owner,
                    geometry,
                    (
                        min(desired[0], int(owner.max_scroll_x)),
                        min(desired[1], int(owner.max_scroll_y)),
                    ),
                    "clamped-after-settlement-failure",
                    allow_reused_geometry=True,
                )
        if not committed:
            self._library_media_last_settlement_outcome = (
                request.request_id,
                "layout-settlement-failed",
                geometry.revision if geometry is not None else None,
            )
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(
                "Media return layout did not settle before its deadline.",
                severity="warning",
            )
        self._disarm_library_list_entry_focus()

    @on(LibraryMediaRowGeometryChanged)
    def _handle_library_media_row_geometry_changed(
        self,
        event: LibraryMediaRowGeometryChanged,
    ) -> None:
        """Settle only from the current attached Media row-scroll owner."""
        event.stop()
        tree = self._library_media_settlement_tree()
        if tree is None or tree[2] is not event.owner:
            return
        self._adopt_library_media_row_owner(event.owner)
        receipt = self._library_pending_list_entry_media_return
        if not self._library_media_return_candidate(receipt):
            return
        request = self._arm_library_media_return_settlement(
            receipt,
            consume_latest_geometry=False,
        )
        if request is not None:
            self._settle_library_media_return_from_geometry(
                request,
                event.owner,
                event.geometry,
            )

    def _mirror_library_media_reader_preference(
        self,
        key: Literal["library_open", "items_open"],
        value: bool,
    ) -> None:
        """Mirror an optimistic manual pane choice into the live app config."""
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return
        library_config = app_config.get("library")
        if not isinstance(library_config, dict):
            library_config = {}
            app_config["library"] = library_config
        section_name = "reader" if key == "library_open" else "media_reader"
        section = library_config.get(section_name)
        if not isinstance(section, dict):
            section = {}
            library_config[section_name] = section
        section[key] = value

    def request_library_media_layout_refresh(self, generation: int) -> None:
        """Compatibility alias for callers using the former Media-specific name."""
        self.request_library_reader_layout_refresh(generation)

    @on(MediaShellResized)
    def _resize_library_media_reader_shell(self, event: MediaShellResized) -> None:
        """Resolve a mounted shell after Textual assigns its real width."""
        event.stop()
        if self._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS:
            self._sync_library_conversation_reader_layout_from_shell()
        elif (
            self._library_selected_row_id
            in (LIBRARY_ROW_BROWSE_NOTES, LIBRARY_ROW_CREATE_NOTE)
            and self._library_notes_source == LIBRARY_NOTES_SOURCE_DATABASE
        ):
            self._sync_library_notes_reader_layout_from_shell()
        elif self._library_selected_row_id in {
            LIBRARY_ROW_BROWSE_PROMPTS,
            LIBRARY_ROW_CREATE_PROMPT,
        }:
            self._sync_library_prompts_reader_layout_from_shell()
        elif self._library_selected_row_id in {
            LIBRARY_ROW_BROWSE_SKILLS,
            LIBRARY_ROW_CREATE_SKILL,
        }:
            self._sync_library_skills_reader_layout_from_shell()
        else:
            self._reconcile_library_media_stage_presentation()
            self._sync_library_media_reader_layout_from_shell()
            receipt = self._library_pending_list_entry_media_return
            if self._library_media_return_candidate(receipt):
                # Shell lifecycle establishes the replacement identity fences;
                # only owner Resize geometry may complete the request.
                self._arm_library_media_return_settlement(receipt)

    def _library_media_viewer_substate_active(self) -> bool:
        """True while the media viewer's edit/delete-confirm/analysis-edit
        sub-state is the live view (task-2856 review round 3).

        All three sub-states leave ``_library_media_view == "viewer"``
        unchanged (see ``handle_library_media_edit`` etc.), so they are
        otherwise indistinguishable from the plain read-only viewer to
        ``_register_footer_shortcuts`` -- but ``action_library_media_viewer_
        back`` only steps back ONE level while any of these is set (to the
        plain viewer, not the list), so the footer must not claim "back to
        list" while this is true.
        """
        return (
            self._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
            and self._library_media_view == "viewer"
            and (
                self._library_media_editing
                or self._library_media_confirming_delete
                or self._library_media_editing_analysis
            )
        )

    def _disarm_library_media_return_for_route_change(self) -> None:
        """Clear all transient Media-return state at an admitted route exit."""
        self._disarm_library_list_entry_focus()
        self._library_media_last_settlement_outcome = None
        self._library_notes_programmatic_focus_target = None

    async def _apply_library_media_active_surface(self) -> None:
        """Synchronize the permanent Media Reader without replacing Items.

        Shared open/detail completion seam (task-21116). Skip unrelated routes
        to prevent the TASK-15706 transition stomp; fetched detail stays retained.

        The Trash view is one of those "no longer owns it" cases (review
        round, M1): it is a distinct surface with its own updater
        (``_sync_library_canvas(self, "media-trash")``), so a media-detail
        worker landing after the user pressed "Trash" must leave it alone
        rather than remount it and drop its selection.

        Returns:
            None.
        """
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_MEDIA:
            return
        self._sync_library_media_surfaces_or_recompose()

    async def _apply_library_media_list_return(
        self,
        media_return: _LibraryMediaReturnReceipt | None,
    ) -> None:
        """Project the viewer-to-list return and arm the list entry focus.

        Args:
            media_return: The exited viewer's origin row id and list scroll
                offset, captured by ``_open_library_media_viewer``.

        Returns:
            None.
        """
        if (
            self._library_selected_row_id != LIBRARY_ROW_BROWSE_MEDIA
            or self._library_media_view == "trash"
        ):
            return
        prearmed = self._mounted_library_media_viewer() is None
        if prearmed:
            self._arm_library_list_entry_focus(media_return=media_return)
        await self._apply_library_open_item_surface(
            self._build_library_media_active_child,
            then=partial(self._finish_library_media_list_return, media_return, prearmed),
        )

    def _finish_library_media_list_return(
        self, media_return: _LibraryMediaReturnReceipt | None, prearmed: bool
    ) -> None:
        """Sync chrome and arm only the retained viewer's replacement row owner."""
        if (
            self._library_selected_row_id != LIBRARY_ROW_BROWSE_MEDIA
            or self._library_media_view != "list"
        ):
            return
        self._register_footer_shortcuts()
        viewer = self._mounted_library_media_viewer()
        if viewer is not None:
            self._sync_library_media_viewer_state(viewer)
        if not prearmed:
            self._arm_library_list_entry_focus(media_return=media_return)

    def _selected_media_handoff_payload(self) -> ChatHandoffPayload | None:
        """Build the Console handoff payload for the open Library media item.

        Mirrors ``_selected_conversation_handoff_payload``, but reads the
        currently loaded media detail (``_library_media_detail``) instead of
        a selected browser row -- the media viewer's "Use in Console" action
        stages whatever item is open in the in-canvas viewer, not a row
        selection from the list.

        Returns:
            A ``ChatHandoffPayload`` staging the open media item as Console
            context, or None when no media item is currently loaded.
        """
        detail = self._library_media_detail
        if not isinstance(detail, Mapping):
            return None
        # task-22208: memoized -- this runs inside the viewer sync's
        # unchanged compare (console representation) on every interaction.
        viewer = self._library_media_viewer_state_cached(detail)
        external_detail = self._library_media_reader_session.external_detail
        media_id = (
            self._library_media_reader_session.loaded_id
            if external_detail
            else viewer.media_id or self._safe_text(self._selected_media_id)
        )
        if not media_id:
            return None
        title = viewer.title or "Untitled source"
        media_type = (
            self._safe_text(detail.get("type"))
            or self._safe_text(detail.get("media_type"))
            or "unknown"
        )
        content = viewer.content
        excerpt = content[:LIBRARY_MEDIA_HANDOFF_EXCERPT_CHARS]
        body_truncated = len(content) > LIBRARY_MEDIA_HANDOFF_EXCERPT_CHARS
        body_lines = [f"Media: {title}", f"Media ID: {media_id}"]
        body_lines.extend(
            line
            for line in viewer.metadata_lines
            if line.startswith(("Type:", "Author:", "Keywords:"))
        )
        body_lines.append(
            f"Source authority: {'server' if external_detail else 'local'}"
        )
        body_lines.append("")
        body_lines.append("Content excerpt:")
        body_lines.append(excerpt if excerpt else "No stored content.")
        body = "\n".join(body_lines)
        return ChatHandoffPayload(
            source="library",
            item_type="media",
            title=title,
            body=body,
            body_truncated=body_truncated,
            source_id=media_id,
            display_summary=f"Media staged: {title}",
            suggested_prompt="Use this media as source context for my next question.",
            runtime_backend="server" if external_detail else "local",
            source_owner="server" if external_detail else "local",
            source_selector_state="server" if external_detail else "local",
            discovery_owner="media",
            discovery_entity_id=media_id,
            metadata={
                "media_id": media_id,
                "media_title": title,
                "media_type": media_type,
                "source_authority": "server" if external_detail else "local",
            },
        )

    def _library_media_console_representation(self) -> str:
        """Describe the exact media representation chosen by the handoff builder."""
        payload = self._selected_media_handoff_payload()
        if payload is None:
            return "Unavailable until a media item is loaded"
        if "No stored content." in payload.body:
            return "Metadata only (no stored text)"
        return (
            "Stored text excerpt (truncated)"
            if payload.body_truncated
            else "Complete stored text excerpt"
        )

    def _build_library_media_state(self) -> LibraryMediaCanvasState:
        """Build Media rows only from the controller's retained exact page."""
        browse = self._library_media_browse_controller.state
        reader = self._library_media_reader_session
        # Initial page projection may clear the Items cursor while a deep link
        # loads. The local Reader retains that request's canonical identity.
        selected_id = (
            reader.selected_id or self._selected_media_id
            if self._library_media_view == "viewer" and not reader.external_detail
            else self._selected_media_id
        )
        if browse.applied_result is None:
            return build_library_media_state(
                (),
                active_type=self._library_media_type_filter,
                selected_id=selected_id,
                select_mode=self._library_media_select_mode,
                selected_ids=self._library_media_row_selection.ids,
                confirming_bulk_delete=self._library_media_confirming_bulk_delete,
                delete_receipt_count=len(self._library_media_delete_receipt_ids),
                delete_receipt_undo_failure=(
                    self._library_media_delete_receipt_undo_failure
                ),
                type_choices_visible=self._library_media_type_choices_visible,
                review_dismiss_receipt_name=self._review_dismiss_receipt_name(),
                **self._library_media_analyze_receipt_fields(),
            )
        state = build_library_media_browse_state(
            browse.applied_result,
            type_options=browse.type_options,
            retained_items=self._decorate_library_media_reviewed(
                browse.retained_items
            ),
            selected_id=selected_id,
            select_mode=self._library_media_select_mode,
            selected_ids=self._library_media_row_selection.ids,
            confirming_bulk_delete=self._library_media_confirming_bulk_delete,
            delete_receipt_count=len(self._library_media_delete_receipt_ids),
            delete_receipt_undo_failure=(
                self._library_media_delete_receipt_undo_failure
            ),
            type_choices_visible=self._library_media_type_choices_visible,
            sort_choices_visible=self._library_media_sort_choices_visible,
            review_dismiss_receipt_name=self._review_dismiss_receipt_name(),
            **self._library_media_analyze_receipt_fields(),
            loading_id=(
                (self._library_media_reader_session.selected_id or "")
                if self._library_media_reader_session.pending_request is not None
                else ""
            ),
            loaded_id=self._library_media_reader_session.loaded_id or "",
            # task-32350: the rail's unfiltered Media total, for the scope
            # line's "N of M".
            unfiltered_total=self._library_media_unfiltered_total,
        )
        if browse.loading and browse.inflight_scope is not None:
            state = dataclasses.replace(state, query=browse.inflight_scope.query)
        if self._library_media_select_mode:
            self._library_media_row_selection.reconcile(r.media_id for r in state.rows)
        return state

    def _library_media_analyze_receipt_fields(self) -> dict[str, Any]:
        """Canvas inputs for the bulk-Analyze receipt (task-28007 AC#3/AC#4).

        (final review, I-2) An Import-origin run ("Analyze N skipped")
        drives the SAME screen-owned counters, so without this guard its
        progress rendered as a Media receipt on a canvas the user never
        started it from -- and that receipt's "Retry failed" re-ran those
        ids as a MEDIA run, leaving the Import rows still saying
        "analysis failed" and still counting in ``N``. The Import surface
        reports itself, per row, with its own disabled action for
        progress; the Media canvas shows nothing for that run.
        """
        if (
            getattr(self, "_library_media_analyze_origin", _ANALYZE_ORIGIN_MEDIA)
            == _ANALYZE_ORIGIN_IMPORT
        ):
            return {
                "analyze_receipt_total": 0,
                "analyze_receipt_done": 0,
                "analyze_receipt_failed": 0,
                "analyze_receipt_running": False,
                "analyze_choice_count": 0,
            }
        choice = self._library_media_analyze_choice
        return {
            # On the armed-choice path the total IS the pressed selection
            # ("N of M already analyzed"); no run has set a total yet.
            "analyze_receipt_total": (
                len(choice[0])
                if choice is not None
                else self._library_media_analyze_total
            ),
            "analyze_receipt_done": self._library_media_analyze_done,
            "analyze_receipt_failed": len(self._library_media_analyze_failed_ids),
            "analyze_receipt_running": self._library_media_analyze_running,
            "analyze_choice_count": (
                0 if choice is None else len(choice[0]) - len(choice[1])
            ),
        }

    def _library_media_analyze_reason(self) -> str:
        """Why bulk Analyze is off, memoised for one select-mode session.

        task-28007 AC#4: the bulk action wears the same sentence the
        Reader's Generate does (AC#5), but this one is read on EVERY media
        canvas sync, and ``resolve_ingest_analysis_provider`` is not free
        (Anthropic ``claude_subscription`` readiness shells out to the
        macOS keychain behind a 5s TTL). So it is resolved once per
        select-mode entry and dropped on exit -- the gesture itself
        re-resolves, so a provider configured mid-session is never refused
        on a stale memo.

        Returns:
            The resolver's reason while select mode is on and no provider
            is ready; "" otherwise.
        """
        if not self._library_media_select_mode:
            self._library_media_analyze_reason_cache = None
            return ""
        if self._library_media_analyze_reason_cache is None:
            self._library_media_analyze_reason_cache = (
                self._library_media_analysis_provider_reason()
            )
        return self._library_media_analyze_reason_cache

    def _library_media_canvas_presentation(self) -> dict[str, Any]:
        """Return controller-owned inputs shared by every Media canvas path."""
        browse = self._library_media_browse_controller.state
        backing_id = self._library_media_selected_backing_id()
        db = getattr(self.app_instance, "media_db", None)
        can_rename = self._library_media_can_rename_speakers(db, backing_id)
        return {
            "pager": browse.pager,
            "type_options": self._library_media_type_options(),
            # Final review M-3: not ``stale_copy`` -- a failed Retry
            # overwrites that with "Couldn't retry · <reason>", which does
            # not explain why every OTHER gated action (Export, sort, ...)
            # is disabled. ``stale_reason`` names why the page went stale
            # and a failed retry never touches it.
            "stale_action_reason": (
                browse.stale_reason if browse.freshness == "stale" else ""
            ),
            "mutation_action_reason": (
                "Media change in progress."
                if self._library_media_bulk_delete_in_flight
                else ""
            ),
            "analysis_action_reason": self._library_media_analyze_reason(),
            # task-31632: the page-or-facet load failure the canvas paints as
            # ONE recovery callout, Retry inside it. ``None`` whenever the
            # last load of each fence succeeded.
            "load_failure": browse.failure,
            # task-31635 fix round 1: the NARROWER predicate the list-wide
            # actions gate on -- a failure with no rows behind it. The
            # callout above is broader on purpose (a page failure retains
            # its rows); those rows still export fine.
            "list_unselectable": self._library_media_list_unselectable(),
            "compact": False,
            "show_preview": False,
            # Task 8 (meeting diarization spec): whether the selected item is
            # a finished meeting recording whose speakers can still be
            # renamed, plus what a canvas needs to actually do that (the
            # real DB and the resolved backing id -- `media_db`/
            # `speaker_rename_media_id` are harmless when `can_rename` is
            # False; the canvas only uses them when it's True).
            "can_rename_speakers": can_rename,
            "media_db": db,
            "speaker_rename_media_id": backing_id,
        }

    @staticmethod
    def _bounded_library_media_trash_title(title: str, limit: int = 80) -> str:
        """Bound mutation receipts without changing confirmation identity."""
        readable = str(title).strip()
        if len(readable) <= limit:
            return readable
        return f"{readable[: limit - 3].rstrip()}..."

    @staticmethod
    def _valid_library_media_trash_delete_ack(
        result: Any,
        *,
        media_id: int,
    ) -> bool:
        """Accept only the local delete contract for the captured identity."""
        if not isinstance(result, Mapping):
            return False
        acknowledged_id = result.get("media_id")
        return (
            result.get("ok") is True
            and type(acknowledged_id) is int
            and acknowledged_id == media_id
        )

    @staticmethod
    def _restore_library_media_scope(state: Mapping[str, Any]) -> MediaBrowseScope:
        """Return one strict dispatch-safe applied Media scope from saved state."""
        saved = state.get("library_media_scope")
        raw = saved if type(saved) is dict else {}
        query = raw.get("query", "")
        media_type = raw.get("media_type")
        sort_by = raw.get("sort_by", "last_modified_desc")
        page = raw.get("page", 1)
        if type(query) is not str:
            query = ""
        if media_type is not None and type(media_type) is not str:
            media_type = None
        if type(sort_by) is not str:
            sort_by = "last_modified_desc"
        if type(page) is not int or page < 1:
            page = 1
        else:
            try:
                MediaBrowseScope(page=page)
            except ValueError:
                page = 1
        try:
            return MediaBrowseScope(
                query=query,
                media_type=media_type,
                sort_by=sort_by,
                page=page,
            )
        except (TypeError, ValueError):
            return MediaBrowseScope()

    def _focus_library_media_page_control(self, invoked: str) -> None:
        """Restore pager focus without ever landing on a disabled control."""
        opposite = {
            "#library-media-previous": "#library-media-next",
            "#library-media-next": "#library-media-previous",
            "#library-media-retry": "#library-media-type-filter",
        }[invoked]
        for selector in (invoked, opposite, "#library-media-type-filter"):
            try:
                control = self.query_one(selector)
            except (NoMatches, QueryError):
                continue
            if not getattr(control, "disabled", False):
                control.focus()
                return

    def _cancel_library_media_selection_settlement(self) -> None:
        """Cancel delayed traversal and invalidate its request generation."""
        if self._library_media_selection_timer is not None:
            self._library_media_selection_timer.stop()
            self._library_media_selection_timer = None
        session = self._library_media_reader_session
        if session.pending_request is None:
            return
        self._library_media_reader_session = LibraryMediaReaderSessionState(
            selected_id=session.loaded_id,
            selected_backing_id=session.loaded_backing_id,
            selected_title=session.loaded_title,
            loaded_id=session.loaded_id,
            loaded_backing_id=session.loaded_backing_id,
            loaded_title=session.loaded_title,
            request_generation=session.request_generation + 1,
            mode=session.mode,
        )

    def _dispatch_library_media_detail_request(
        self, generation: int, requested_id: str, media_id: str
    ) -> None:
        """Dispatch only the still-current canonical-id/generation request."""
        self._library_media_selection_timer = None
        pending = self._library_media_reader_session.pending_request
        if (
            pending is None
            or pending.generation != generation
            or pending.requested_id != requested_id
            or self._library_media_select_mode
        ):
            return
        self.run_worker(
            self._refresh_library_media_detail(
                media_id,
                request_generation=generation,
                requested_id=requested_id,
            ),
            group="library_media_detail",
        )

    def _select_library_media_reader_row(
        self,
        media_id: str,
        title: str,
        *,
        immediate: bool,
        sync_surfaces: bool = True,
    ) -> None:
        """Select a row now and schedule its generation-fenced detail load."""
        identity = self._library_media_reader_identity(media_id)
        if identity is None:
            return
        self._capture_library_media_loaded_progress()
        self._library_media_progress_restored_id = None
        canonical_id, backing_id = identity
        current = self._library_media_reader_session
        if (
            not immediate
            and current.selected_id == canonical_id
            and current.pending_request is not None
        ):
            return
        if self._library_media_selection_timer is not None:
            self._library_media_selection_timer.stop()
            self._library_media_selection_timer = None
        self._selected_media_id = canonical_id
        self._library_media_view = "viewer"
        self._library_media_reader_session = begin_selection(
            current,
            canonical_id,
            backing_id,
            title,
            immediate=immediate,
        )
        pending = self._library_media_reader_session.pending_request
        assert pending is not None
        if sync_surfaces:
            # M-2: the row patch is the sync wrapper's own tail now
            # (``_sync_library_media_surfaces_or_recompose``); only the
            # no-canvas fallback is still this call site's to make.
            if not self.query("#library-media-canvas"):
                self._sync_library_media_browse_state(None)
            self._sync_library_media_viewer_or_recompose()
        # task-31979: selecting flips the view to "viewer", so the Reader now
        # holds a document (or its loading banner) and must reclaim the width
        # the list-view widening handed to the Items list. The in-place viewer
        # patch above does not re-run the resolver.
        self._restore_library_media_reader_width_on_open()
        if immediate:
            self._dispatch_library_media_detail_request(
                pending.generation, pending.requested_id, canonical_id
            )
            return
        self._library_media_selection_timer = self.set_timer(
            SELECTION_SETTLE_SECONDS,
            partial(
                self._dispatch_library_media_detail_request,
                pending.generation,
                pending.requested_id,
                canonical_id,
            ),
        )

    def _request_library_media_facets(self) -> Any | None:
        return self._library_media_browse_controller.request_facets(
            fingerprint=self._library_media_browse_controller.state.requested_scope.fingerprint
        )

    def _request_library_media_filter(
        self, query: str, *, clear_type: bool = False
    ) -> None:
        """Request authoritative search while preserving the unfiltered anchor.

        Args:
            query: The filter text to apply; "" restores the unfiltered
                anchor scope and its previous selection.
            clear_type: Also drop the applied type facet (task-32350). The
                scope line's Clear clears the whole scope it states; the
                toolbar's "Clear filter" clears only the filter it names.
        """
        query = self._safe_text(query, max_length=200).strip()
        browse = self._library_media_browse_controller.state
        applied = browse.applied_scope or browse.mutation_refresh_scope
        # Qodo review: the scope line and its Clear are derived from the
        # APPLIED result, so a failed browse leaves a visible Clear whose
        # REQUESTED scope is already the target. Testing the requested scope
        # alone made every later press of that still-visible button a no-op,
        # with no way to retry. Suppress only when the page the user is
        # actually looking at is already there too.
        def _is_target(scope: Any) -> bool:
            return query == scope.query and not (
                clear_type and scope.media_type is not None
            )

        if _is_target(browse.requested_scope) and _is_target(applied):
            return
        self._clear_library_media_selection_for_scope_change()
        if query:
            if not applied.query:
                self._library_media_unfiltered_scope = applied
                self._library_media_unfiltered_selected_id = self._selected_media_id
            self._library_media_filter_select_first = True
            scope = dataclasses.replace(applied, query=query, page=1)
        else:
            self._library_media_filter_restore_id = (
                self._library_media_unfiltered_selected_id
            )
            self._library_media_filter_select_first = False
            scope = self._library_media_unfiltered_scope
        if clear_type:
            scope = dataclasses.replace(scope, media_type=None, page=1)
        self._request_library_media_browse(
            scope,
            focus_identity="#library-media-filter",
        )

    @on(Input.Changed, "#library-media-filter")
    def handle_library_media_filter_changed(self, event: Input.Changed) -> None:
        """Debounce Media search through the authoritative browse controller."""
        self._stop_library_media_filter_timer()
        self._library_media_filter_timer = self.set_timer(
            SELECTION_SETTLE_SECONDS,
            partial(self._request_library_media_filter, event.value),
        )

    @on(Input.Submitted, "#library-media-filter")
    def handle_library_media_filter_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._stop_library_media_filter_timer()
        self._request_library_media_filter(event.value)

    def _clear_library_media_filter(self, *, clear_type: bool) -> None:
        """Drop the applied filter, and the type facet when asked.

        Args:
            clear_type: True for the scope line's Clear, which clears the
                whole scope that line states; False for the toolbar's
                "Clear filter", which clears only the filter it names.
        """
        self._stop_library_media_filter_timer()
        # task-32350 AC#2: clearing the APPLIED filter clears the box too --
        # leaving a draft behind is exactly the split-brain this task closes.
        box = self.query("#library-media-filter")
        if box:
            box.first(Input).value = ""
        self._request_library_media_filter("", clear_type=clear_type)

    @on(Button.Pressed, "#library-media-filter-clear")
    def handle_library_media_filter_clear(self, event: Button.Pressed) -> None:
        event.stop()
        self._clear_library_media_filter(clear_type=False)

    @on(Button.Pressed, "#library-media-scope-clear")
    def handle_library_media_scope_clear(self, event: Button.Pressed) -> None:
        """Clear the whole scope the scope line states (task-32350).

        Args:
            event: Press of the scope line's "Clear", routed here by the
                canvas. Stopped so the shared list-row handlers never see
                it; the toolbar's "Clear filter" has its own handler
                because the two clear different things.
        """
        event.stop()
        self._clear_library_media_filter(clear_type=True)

    def _load_library_media_list_if_needed(self) -> None:
        """Load the exact list after a direct viewer had no applied page."""
        browse = self._library_media_browse_controller.state
        if browse.applied_result is not None:
            return
        self._request_library_media_browse(
            browse.mutation_refresh_scope,
            focus_identity="#library-media-row-0",
        )
        self._request_library_media_facets()

    def _request_library_media_page(
        self, page: int, *, focus_identity: str | None
    ) -> Any | None:
        """Request one page from the complete last-applied Media scope."""
        self._clear_library_media_selection_for_scope_change()
        return self._request_library_media_browse(
            self._library_media_browse_controller.state.scope_for_page(page),
            focus_identity=focus_identity,
        )

    def _retry_library_media_browse(self, *, focus_identity: str | None) -> Any | None:
        """Retry the latest requested Media draft with a fresh generation."""
        return self._library_media_browse_controller.retry(
            focus_identity=focus_identity
        )

    def _library_media_trash_canvas_presentation(self) -> dict[str, Any]:
        """Return screen-owned Trash controls without re-deriving page authority."""
        controller = self._library_media_trash_browse_controller
        state = controller.state
        applied = state.applied_result
        applied_scope = applied.scope if applied is not None else MediaTrashScope()
        scope_parts: list[str] = []
        if applied_scope.query:
            scope_parts.append(f"Query: {applied_scope.query}")
        if applied_scope.media_type is not None:
            scope_parts.append(f"Type: {applied_scope.media_type}")

        mutation_in_flight = bool(
            state.mutation_pending or self._library_media_bulk_delete_in_flight
        )
        action_disabled_reason = self._library_media_trash_action_disabled_reason()
        return {
            "pager": controller.pager,
            "types": state.types,
            "query_draft": self._library_media_trash_query_draft,
            "applied_scope_label": " · ".join(scope_parts),
            "applied_type": applied_scope.media_type,
            "type_choices_visible": self._library_media_trash_type_choices_visible,
            "action_disabled_reason": action_disabled_reason,
            "retry_visible": self._library_media_trash_retry_visible(),
            "controls_disabled_reason": (
                "Trash is refreshing." if mutation_in_flight else ""
            ),
            "confirmation_target": state.confirmation_target,
            "commit_pending": state.mutation_pending,
        }

    def _focus_library_media_trash_intent(self) -> None:
        """Validate the mounted target, then defer through one paint cycle."""
        self._library_notes_restoring_focus = False
        if (
            self._library_selected_row_id != LIBRARY_ROW_BROWSE_MEDIA
            or self._library_media_view != "trash"
            or self._library_media_trash_focus_authority_generation
            != self._library_notes_focus_intent_generation
        ):
            return
        identity = self._library_media_trash_focus_identity
        generation = self._library_media_trash_focus_authority_generation
        if self._resolve_library_media_trash_focus_target(identity) is None:
            return
        self.call_after_refresh(
            self._focus_library_media_trash_after_paint,
            identity,
            generation,
        )

    def _resolve_library_media_trash_focus_target(self, identity: str) -> Widget | None:
        """Reacquire the first current, enabled control for an identity."""
        for selector in self._library_media_trash_focus_selectors(identity):
            try:
                target = self.query_one(selector, Widget)
            except (NoMatches, QueryError):
                continue
            if not getattr(target, "disabled", False) and target.can_focus:
                return target
        return None

    def _focus_library_media_trash_after_paint(
        self, identity: str, generation: int
    ) -> None:
        """Reacquire immediately before focusing after compositor paint."""
        if (
            self._library_selected_row_id != LIBRARY_ROW_BROWSE_MEDIA
            or self._library_media_view != "trash"
            or identity != self._library_media_trash_focus_identity
            or generation != self._library_media_trash_focus_authority_generation
            or generation != self._library_notes_focus_intent_generation
        ):
            return
        target = self._resolve_library_media_trash_focus_target(identity)
        if target is None:
            return
        self._library_notes_programmatic_focus_target = target
        target.focus()

    def _library_media_image_preview_capable(self) -> bool:
        """Return whether this runtime can present terminal image previews."""
        if self._library_media_preview_factory_injected:
            return True
        try:
            return not bool(self.app.is_headless)
        except Exception:
            return False

    def _schedule_library_media_image_preview(
        self,
        *,
        request_generation: int,
        canonical_id: str,
        backing_id: int | str,
        detail: Mapping[str, Any],
        force: bool = False,
    ) -> None:
        """Schedule an additive local-original preview without delaying text."""
        if not self._library_media_image_preview_capable():
            return
        if canonical_id in self._library_media_preview_images and not force:
            return
        if self._library_media_preview_loading.get(canonical_id) == request_generation:
            return
        if force:
            self._library_media_preview_images.pop(canonical_id, None)
            self._library_media_preview_status.pop(canonical_id, None)
        self._library_media_preview_loading[canonical_id] = request_generation
        self.run_worker(
            self._load_library_media_image_preview(
                request_generation=request_generation,
                canonical_id=canonical_id,
                backing_id=backing_id,
                detail=detail,
            ),
            group="library_media_image_preview",
        )

    def _library_media_preview_request_is_current(
        self, request_generation: int, canonical_id: str
    ) -> bool:
        """Fence preview projection to the local item that still owns Reader."""
        session = self._library_media_reader_session
        return (
            not session.external_detail
            and session.request_generation == request_generation
            and session.loaded_id == canonical_id
        )

    def _cache_library_media_preview(self, canonical_id: str, image: Any) -> None:
        """Store one decoded preview and evict the oldest cached entry."""
        self._library_media_preview_images.pop(canonical_id, None)
        self._library_media_preview_images[canonical_id] = image
        while (
            len(self._library_media_preview_images) > LIBRARY_MEDIA_PREVIEW_CACHE_LIMIT
        ):
            evicted_id = next(iter(self._library_media_preview_images))
            self._library_media_preview_images.pop(evicted_id, None)
            self._library_media_preview_status.pop(evicted_id, None)
            self._library_media_preview_hidden.discard(evicted_id)
            self._library_media_preview_loading.pop(evicted_id, None)

    async def _load_library_media_image_preview(
        self,
        *,
        request_generation: int,
        canonical_id: str,
        backing_id: int | str,
        detail: Mapping[str, Any],
    ) -> None:
        """Load and decode one eligible local original behind a stale fence."""
        try:
            from ...Widgets.Library.library_media_image_preview import (
                decode_media_image,
                image_preview_eligibility,
                image_preview_expected,
            )

            # Remote details are rejected before even consulting local file
            # seams. This keeps previews strictly local and side-effect free.
            preliminary = image_preview_eligibility(detail, None, backend="local")
            if preliminary.reason == "remote":
                return
            service = getattr(self.app_instance, "media_reading_scope_service", None)
            check_media_file = getattr(service, "check_media_file", None)
            download_media_file = getattr(service, "download_media_file", None)
            if not callable(check_media_file) or not callable(download_media_file):
                return
            file_check = await self._run_library_service_call(
                check_media_file,
                mode="local",
                media_id=backing_id,
                file_type="original",
                isolate_in_worker=True,
            )
            eligibility = image_preview_eligibility(
                detail,
                file_check if isinstance(file_check, Mapping) else None,
                backend="local",
            )
            if not eligibility.eligible:
                # task-30044 (critique P2): the failure chrome is honest only
                # when an image was actually EXPECTED -- a plain document
                # with no image type hint reads as a document, never as a
                # failed image with a Retry button above its text.
                if not self._library_media_preview_request_is_current(
                    request_generation, canonical_id
                ):
                    return
                if eligibility.reason == "unavailable" and image_preview_expected(
                    detail
                ):
                    self._library_media_preview_status[canonical_id] = (
                        "Image preview unavailable — showing complete stored text"
                    )
                    self._sync_library_media_viewer_or_recompose()
                elif (
                    self._library_media_preview_status.pop(canonical_id, None)
                    is not None
                ):
                    # Qodo #2351: an item that STOPPED being image-expected
                    # (edited type, refreshed detail) must also shed any
                    # previously stamped failure text.
                    self._sync_library_media_viewer_or_recompose()
                return
            receipt = await self._run_library_service_call(
                download_media_file,
                mode="local",
                media_id=backing_id,
                file_type="original",
                isolate_in_worker=True,
            )
            content = receipt.get("content") if isinstance(receipt, Mapping) else None
            image = await asyncio.to_thread(decode_media_image, content)
            if not self._library_media_preview_request_is_current(
                request_generation, canonical_id
            ):
                return
            self._cache_library_media_preview(canonical_id, image)
            self._library_media_preview_status.pop(canonical_id, None)
            self._sync_library_media_viewer_or_recompose()
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to load Library media image preview for {}.", canonical_id
            )
            if self._library_media_preview_request_is_current(
                request_generation, canonical_id
            ):
                self._library_media_preview_images.pop(canonical_id, None)
                self._library_media_preview_status[canonical_id] = (
                    "Image preview failed — showing complete stored text"
                )
                self._sync_library_media_viewer_or_recompose()
        finally:
            if (
                self._library_media_preview_loading.get(canonical_id)
                == request_generation
            ):
                self._library_media_preview_loading.pop(canonical_id, None)

    def _recompose_library_media_detail_if_unrendered(self) -> None:
        """Project the media surface unless a compose already rendered the
        current media detail.

        task-21116: the projection is the targeted media canvas-child swap
        (``_apply_library_media_active_surface``), not the retired
        whole-screen recompose -- and it skips entirely when the media
        surface no longer owns the route (the TASK-15706 rule: an async
        completion must not rebuild a surface that is transitioning away;
        the fetched detail stays cached for the next media entry).
        """
        if not self.is_mounted:
            return
        if (
            self._library_media_view == "viewer"
            and self._library_media_detail is not None
            and self._library_media_composed_detail is self._library_media_detail
        ):
            return
        self.call_next(self._apply_library_media_active_surface)

    async def _fetch_library_media_highlights(
        self, media_id: str
    ) -> list[dict[str, Any]]:
        """Fetch reading highlights for a Library media item from the local scope service.

        Guards against an unavailable ``list_highlights`` service or a failed
        call the same way ``_refresh_library_media_detail`` guards the detail
        fetch: any failure yields an empty list rather than raising, so a
        highlights outage never blocks the rest of the viewer from loading.

        Args:
            media_id: The Library media item id to fetch highlights for.

        Returns:
            The fetched highlights list, or an empty list when the service
            is unavailable or the fetch fails.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        list_highlights = getattr(service, "list_highlights", None)
        if not callable(list_highlights):
            return []
        service_media_id = self._library_media_backing_id(media_id)
        try:
            highlights = await self._run_library_service_call(
                list_highlights,
                mode="local",
                item_id=service_media_id,
                isolate_in_worker=True,
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to load Library media highlights for {media_id!r}."
            )
            return []
        return list(highlights) if isinstance(highlights, list) else []

    async def _reload_library_media_highlights(self, media_id: str) -> None:
        """Re-fetch and store highlights after a highlight mutation, then recompose.

        Args:
            media_id: The Library media item id whose highlights changed.
        """
        self._library_media_highlights = await self._fetch_library_media_highlights(
            media_id
        )
        if self.is_mounted:
            self._sync_library_media_viewer_or_recompose()

    @on(Button.Pressed, "#library-media-sort")
    def handle_library_media_sort(self, event: Button.Pressed) -> None:
        """Open (or close) the media browse sort chooser's direct-pick strip.

        task-28013: mirrors the type chooser and the Prompts/Notes sort
        choosers -- Sort is one control family across the list canvases. Inert
        while a bulk delete is armed or in flight (task-2853 AC3: nothing may
        drift the list state under an armed confirm).
        """
        event.stop()
        if (
            self._library_media_bulk_delete_in_flight
            or self._library_media_confirming_bulk_delete
        ):
            return
        # Mutually exclusive with the type chooser: only one strip at a time.
        self._library_media_type_choices_visible = False
        self._library_media_sort_choices_visible = (
            not self._library_media_sort_choices_visible
        )
        # task-31235: the chooser is a vertical OptionList (like the type
        # chooser) with the active value pre-highlighted at compose time,
        # so focusing the list is enough. The focus rides the sync's
        # ``then`` hook -- ``call_after_refresh`` has no ordering against
        # a canvas-scoped recompose (see ``_sync_library_canvas``).
        def focus_open_chooser() -> None:
            self._focus_library_control("#library-media-sort-choices")

        def focus_sort_opener() -> None:
            self._focus_library_control("#library-media-sort")

        _sync_library_canvas(
            self,
            "media",
            then=(
                focus_open_chooser
                if self._library_media_sort_choices_visible
                else focus_sort_opener
            ),
        )

    @on(OptionList.OptionSelected, "#library-media-sort-choices")
    def handle_library_media_sort_choice(
        self, event: OptionList.OptionSelected
    ) -> None:
        """Apply the exact sort value carried by one chooser option.

        task-31235: the horizontal strip (task-28013) became a vertical
        OptionList -- it clipped options off-pane at the items pane's real
        width. Picking the already-active sort only closes the chooser --
        no service request. Any other value re-fetches page one under the
        new order.

        Args:
            event: Selection event emitted by the bounded sort chooser.
        """
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        requested = str(getattr(event.option, "choice_value", "") or "")
        self._library_media_sort_choices_visible = False
        current = self._library_media_browse_controller.state.mutation_refresh_scope.sort_by
        valid = {value for value, _ in MEDIA_SORT_CHOICES}
        if requested not in valid or requested == current:
            _sync_library_canvas(
                self,
                "media",
                then=lambda: self._focus_library_control("#library-media-sort"),
            )
            return
        self._request_library_media_sort(
            requested, focus_identity="#library-media-sort"
        )

    def _request_library_media_sort(
        self, sort_by: str, *, focus_identity: str | None
    ) -> Any | None:
        """Request page one after changing only the applied Media sort order."""
        self._clear_library_media_selection_for_scope_change()
        applied = self._library_media_browse_controller.state.mutation_refresh_scope
        return self._request_library_media_browse(
            dataclasses.replace(applied, sort_by=sort_by, page=1),
            focus_identity=focus_identity,
        )

    @on(Button.Pressed, "#library-media-previous")
    def handle_library_media_previous(self, event: Button.Pressed) -> None:
        """Request the previous exact Media page."""
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        applied = self._library_media_browse_controller.state.applied_scope
        if (
            applied is None
            or self._library_media_browse_controller.state.pager.previous_disabled
        ):
            return
        self._request_library_media_page(
            applied.page - 1,
            focus_identity="#library-media-previous",
        )

    @on(Button.Pressed, "#library-media-next")
    def handle_library_media_next(self, event: Button.Pressed) -> None:
        """Request the next exact Media page."""
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        applied = self._library_media_browse_controller.state.applied_scope
        if applied is None or self._library_media_browse_controller.state.pager.next_disabled:
            return
        self._request_library_media_page(
            applied.page + 1,
            focus_identity="#library-media-next",
        )

    @on(Button.Pressed, "#library-media-retry")
    def handle_library_media_retry(self, event: Button.Pressed) -> None:
        """Retry the latest failed Media request."""
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        self._retry_library_media_browse(
            focus_identity="#library-media-retry",
        )

    def action_library_media_toggle_select_mode(self) -> None:
        """Keyboard "s": enter/exit media select mode (task-28012)."""
        self._toggle_library_media_select_mode()

    @on(Button.Pressed, "#library-media-select-all")
    def handle_library_media_select_all(self, event: Button.Pressed) -> None:
        """Select every media row currently rendered by the canvas."""
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        rows = self._build_library_media_state().rows
        self._library_media_row_selection.select_all(r.media_id for r in rows)
        _sync_library_canvas(self, "media")

    @on(Button.Pressed, "#library-media-select-clear")
    def handle_library_media_select_clear(self, event: Button.Pressed) -> None:
        """Clear the current media selection without leaving select mode."""
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        self._library_media_row_selection.clear()
        _sync_library_canvas(self, "media")

    def action_library_media_bulk_delete_cancel(self) -> None:
        """Escape: cancel an ARMED bulk-delete confirmation (task-3020 AC2).

        ``check_action`` gates this to
        ``_library_media_confirming_bulk_delete``, so it only ever fires
        while the confirm row is showing -- parity with the single-item
        viewer confirm's own Escape-cancels behavior
        (``action_library_media_viewer_back``'s ``_library_media_
        confirming_delete`` branch), which already mirrors ITS Cancel
        button the same way. Reuses ``_cancel_library_media_bulk_delete``
        rather than duplicating the Cancel button's body.
        """
        if not self._library_media_bulk_delete_in_flight:
            self._cancel_library_media_bulk_delete()

    def _capture_library_media_trash_return(self) -> _LibraryMediaReturnReceipt:
        """Capture the normal-Media row, scroll, and semantic toolbar focus."""
        scroll_offset: tuple[int, int] | None = None
        try:
            scroll = self.query_one("#library-media-row-scroll", Widget)
        except (NoMatches, QueryError):
            pass
        else:
            scroll_offset = (int(scroll.scroll_x), int(scroll.scroll_y))
        return _LibraryMediaReturnReceipt(
            stable_id=self._selected_media_id,
            scroll_offset=scroll_offset,
            content_signature=self._library_media_content_signature(),
            layout_signature=self._library_media_layout_signature(),
            final_focus_policy="control",
            final_focus_identity="library-media-trash-open",
        )

    @on(Input.Changed, "#library-media-trash-search")
    def handle_library_media_trash_search_changed(self, event: Input.Changed) -> None:
        """Own the visible draft without applying it to the browse scope."""
        event.stop()
        self._library_media_trash_query_draft = event.value

    @on(Input.Submitted, "#library-media-trash-search")
    def handle_library_media_trash_search_submitted(
        self, event: Input.Submitted
    ) -> None:
        """Request page one for a bounded Trash query draft."""
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        query = event.value.strip()
        self._library_media_trash_query_draft = event.value
        if len(query) > 200:
            self._library_media_trash_input_error = (
                "Search is limited to 200 characters."
            )
            self._library_media_trash_focus_identity = "#library-media-trash-search"
            self._sync_library_media_trash_state("#library-media-trash-search")
            return
        try:
            applied = self._library_media_trash_applied_scope()
            scope = MediaTrashScope(
                query=query,
                media_type=applied.media_type,
                page=1,
            )
        except ValueError:
            self._library_media_trash_input_error = "Search contains invalid text."
            self._sync_library_media_trash_state("#library-media-trash-search")
            return
        self._library_media_trash_input_error = ""
        self._library_media_trash_browse_controller.request(
            scope,
            origin="search",
            focus_identity="#library-media-trash-search",
        )

    @on(Button.Pressed, "#library-media-trash-type-filter")
    def handle_library_media_trash_type_filter(self, event: Button.Pressed) -> None:
        """Toggle the bounded Trash type chooser owned by Task 5."""
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        self._library_media_trash_type_choices_visible = (
            not self._library_media_trash_type_choices_visible
        )
        target = (
            "#library-media-trash-type-choices"
            if self._library_media_trash_type_choices_visible
            else "#library-media-trash-type-filter"
        )
        self._library_media_trash_focus_identity = target
        # This focus intent is created by the same real Enter that advanced
        # the global focus generation.  Claim that generation before sync so
        # the opener is not mistaken for a newer intent and restored over the
        # chooser it just opened.
        self._library_media_trash_focus_authority_generation = (
            self._library_notes_focus_intent_generation
        )
        self._sync_library_media_trash_state(target)

    @on(OptionList.OptionSelected, "#library-media-trash-type-choices")
    def handle_library_media_trash_type_choice(
        self, event: OptionList.OptionSelected
    ) -> None:
        """Request page one for one source-provided Trash type."""
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        requested = getattr(event.option, "choice_value", None)
        if requested is not None and type(requested) is not str:
            return
        controller = self._library_media_trash_browse_controller
        if requested not in (None, *controller.state.types):
            return
        self._library_media_trash_type_choices_visible = False
        self._library_media_trash_input_error = ""
        applied = self._library_media_trash_applied_scope()
        controller.request(
            MediaTrashScope(
                query=applied.query,
                media_type=requested,
                page=1,
            ),
            origin="type",
            focus_identity="#library-media-trash-type-filter",
        )

    def _library_media_trash_applied_scope(self) -> MediaTrashScope:
        """Return only the scope that owns the currently visible Trash page."""
        applied = self._library_media_trash_browse_controller.state.applied_result
        return applied.scope if applied is not None else MediaTrashScope()

    def _request_library_media_trash_page(
        self, page: int, *, focus_identity: str
    ) -> Any | None:
        """Request one page from the complete visible applied Trash scope."""
        if self._library_media_bulk_delete_in_flight:
            return None
        applied = self._library_media_trash_browse_controller.state.applied_result
        if applied is None:
            return None
        self._library_media_trash_input_error = ""
        origin = "previous" if page < applied.scope.page else "next"
        return self._library_media_trash_browse_controller.request(
            applied.scope.with_page(page),
            origin=origin,
            focus_identity=focus_identity,
        )

    @on(Button.Pressed, "#library-media-trash-previous")
    def handle_library_media_trash_previous(self, event: Button.Pressed) -> None:
        """Request the previous page from the visible applied Trash scope."""
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        controller = self._library_media_trash_browse_controller
        applied = controller.state.applied_result
        if applied is None or controller.pager.previous_disabled:
            return
        self._request_library_media_trash_page(
            applied.scope.page - 1,
            focus_identity="#library-media-trash-previous",
        )

    @on(Button.Pressed, "#library-media-trash-next")
    def handle_library_media_trash_next(self, event: Button.Pressed) -> None:
        """Request the next page from the visible applied Trash scope."""
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        controller = self._library_media_trash_browse_controller
        applied = controller.state.applied_result
        if applied is None or controller.pager.next_disabled:
            return
        self._request_library_media_trash_page(
            applied.scope.page + 1,
            focus_identity="#library-media-trash-next",
        )

    @on(Button.Pressed, "#library-media-trash-retry")
    def handle_library_media_trash_retry(self, event: Button.Pressed) -> None:
        """Repeat the controller's retained failed Trash target."""
        event.stop()
        if self._library_media_bulk_delete_in_flight:
            return
        self._library_media_trash_input_error = ""
        self._library_media_trash_browse_controller.retry(
            focus_identity="#library-media-trash-retry"
        )

    def action_library_media_trash_back(self) -> None:
        """Escape: leave the Trash view for the media list (task-4025).

        ``check_action`` gates this to the media canvas genuinely showing
        its Trash view, so it only ever fires there.
        """
        controller = self._library_media_trash_browse_controller
        if controller.state.mutation_pending:
            return
        if controller.state.confirmation_target is not None:
            self._cancel_library_media_trash_delete_confirmation()
            return
        if self._library_media_trash_type_choices_visible:
            self._library_media_trash_type_choices_visible = False
            self._library_media_trash_focus_authority_generation = (
                self._library_notes_focus_intent_generation
            )
            self._sync_library_media_trash_state("#library-media-trash-type-filter")
            return
        self._exit_library_media_trash()

    def _focus_library_media_trash_entry(self) -> None:
        """Compatibility callback for restore paths; resolve semantic intent."""
        self._focus_library_media_trash_intent()

    def _cancel_library_media_trash_delete_confirmation(self) -> None:
        """Close confirmation and return focus to its permanent-delete opener."""
        self._library_media_trash_focus_identity = "#library-media-trash-delete"
        self._library_media_trash_focus_authority_generation = getattr(
            self, "_library_notes_focus_intent_generation", 0
        )
        self._library_media_trash_browse_controller.cancel_delete_confirmation()

    @on(Button.Pressed, "#library-media-trash-delete-cancel")
    def handle_library_media_trash_delete_cancel(self, event: Button.Pressed) -> None:
        """Cancel permanent deletion without invoking a service."""
        event.stop()
        if self._library_media_trash_browse_controller.state.mutation_pending:
            return
        self._cancel_library_media_trash_delete_confirmation()

    @on(Button.Pressed, "#library-media-open-viewer")
    def handle_library_media_open_viewer(self, event: Button.Pressed) -> None:
        """Open the browse summary's selected media item in the in-Library viewer.

        The browse canvas preview's primary action. Stays entirely inside
        Library and never posts a navigation message, unlike the full
        viewer's "Open in Library ▸ Media" action (task-2857) -- hence the
        "Open in viewer" label (2026-07 UAT relabel).

        Args:
            event: Button press event emitted by the "Open in viewer" action.
        """
        event.stop()
        self._open_library_media_viewer(self._selected_media_id)

    def _open_library_media_viewer(self, media_id: str) -> None:
        """Switch the media canvas to the in-canvas viewer for ``media_id``.

        Shared by media-row presses and the browse summary's "Open in
        viewer" action: resets per-item viewer state, kicks the async
        detail fetch, and recomposes into the viewer's loading line.

        Args:
            media_id: The media item to open; an empty id still switches to
                the viewer without kicking a fetch (mirrors the previous
                row-press behavior for a row missing its ``media_id``).
        """
        media_id = str(media_id or "")
        if self._library_pending_list_entry_focus:
            self._disarm_library_list_entry_focus()
        self._library_media_viewer_return = None
        if media_id and self._library_media_view == "list" and self.is_mounted:
            try:
                scroll = self.query_one("#library-media-row-scroll", Widget)
            except (NoMatches, QueryError):
                scroll_offset = None
            else:
                scroll_offset = (int(scroll.scroll_x), int(scroll.scroll_y))
            self._library_media_viewer_return = _LibraryMediaReturnReceipt(
                stable_id=media_id,
                scroll_offset=scroll_offset,
                content_signature=self._library_media_content_signature(),
                layout_signature=self._library_media_layout_signature(),
                final_focus_policy="row",
                final_focus_identity=None,
            )
        if media_id:
            self._acknowledge_library_destination_change()
            title = next(
                (
                    row.title
                    for row in self._build_library_media_state().rows
                    if row.media_id == media_id
                ),
                media_id,
            )
            self._select_library_media_reader_row(media_id, title, immediate=True)
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
        self._library_media_view = "viewer"
        # task-14902: an open type strip must not survive the trip through
        # the viewer and reappear (stale) on the way back.
        self._library_media_type_choices_visible = False
        self._library_media_editing = False
        self._library_media_confirming_delete = False
        self._library_media_highlights = []
        self._library_media_editing_analysis = False
        self._close_library_media_find()
        self._library_media_content_mode = "raw"
        self._sync_library_media_viewer_or_recompose()

    async def _open_library_external_media_detail(self, media_id: str) -> None:
        """Open one server detail without selecting or adding a local Items row."""
        self._cancel_library_media_selection_settlement()
        self._library_media_reader_session = enter_external_detail(
            self._library_media_reader_session, media_id, f"Server media {media_id}"
        )
        self._library_media_detail = None
        self._library_media_composed_detail = None
        self._library_media_highlights = []
        self._library_media_editing = False
        self._library_media_editing_analysis = False
        self._close_library_media_find()
        self._library_media_content_mode = "raw"
        self._library_media_view = "viewer"
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
        await self._apply_library_media_active_surface()
        await self._refresh_library_media_detail(
            media_id,
            request_generation=self._library_media_reader_session.request_generation,
            requested_id=self._library_media_reader_session.loaded_id,
        )

    def _library_media_reader_identity(
        self, media_id: str
    ) -> tuple[str, int | str] | None:
        """Normalize retained legacy ids for the canonical Reader session."""
        backing_id = self._library_media_backing_id(media_id)
        if type(backing_id) is int:
            return f"local:media:{backing_id}", backing_id
        if isinstance(backing_id, str) and backing_id.isdecimal():
            numeric_id = int(backing_id)
            if numeric_id > 0:
                return f"local:media:{numeric_id}", numeric_id
        if media_id.startswith("media-") and media_id[6:].isdecimal():
            backing_id = int(media_id[6:])
            if backing_id > 0:
                return f"local:media:{backing_id}", backing_id
        return None

    def _navigate_to_media(self, media_id: str | int) -> None:
        """Open the Library media viewer for ``media_id``.

        Thin synchronous seam so the ingest "Open in Library" fallback can
        reuse the same detail viewer path as Search/RAG result open actions
        without duplicating viewer state setup.
        """
        self.run_worker(self._open_library_item_by_id("media", str(media_id)))

    def _pop_library_media_arrival_note(self) -> str:
        note = self._library_media_arrival_note
        self._library_media_arrival_note = ""
        return note

    @on(Button.Pressed, "#library-media-export")
    async def handle_library_media_export(self, event: Button.Pressed) -> None:
        """Open the export canvas scoped to the media list's current type filter.

        Args:
            event: Button press event emitted by the media canvas's
                "Export…" action.
        """
        event.stop()
        await self._open_library_export_canvas(
            ExportScope(kind="media", media_type=self._library_media_type_filter)
        )

    @on(Button.Pressed, "#library-media-back")
    def handle_library_media_back(self, event: Button.Pressed) -> None:
        """Return the Library media canvas from the viewer to its list view.

        Args:
            event: Button press event emitted by the "‹ Back to list" action.
        """
        event.stop()
        self._exit_library_media_viewer()

    def _library_media_adjacent_row(
        self, direction: int
    ) -> tuple[str, str] | None:
        """Return the (id, title) of the browse row ``direction`` from the current.

        task-28005: neighbours come from the mounted rows -- they carry
        ``media_id`` in exactly the form ``_selected_media_id`` holds (so no
        id-format mismatch) and sit in browse order (newest first, so
        ``direction=+1`` is the next item DOWN the list, matching Down). No
        neighbour (at a list end, or the current item is off the loaded
        page) returns None.
        """
        rows = list(self.query(".library-media-row"))
        if not rows:
            return None
        selected = str(self._selected_media_id or "")
        index = next(
            (
                i
                for i, row in enumerate(rows)
                if str(getattr(row, "media_id", "") or "") == selected
            ),
            None,
        )
        if index is None:
            return None
        neighbour = index + direction
        if not 0 <= neighbour < len(rows):
            return None
        row = rows[neighbour]
        media_id = str(getattr(row, "media_id", "") or "")
        title = str(getattr(row, "_library_media_title", "") or media_id)
        return (media_id, title)

    def _library_media_item_traversal_active(self) -> bool:
        """True while the Reader is showing a plain item (no sub-state)."""
        return (
            self._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
            and self._library_media_view == "viewer"
            and not self._library_media_viewer_substate_active()
            and not self._library_media_select_mode
        )

    def action_library_media_next_item(self) -> None:
        """Open the next browse item in the Reader (task-28005)."""
        self._select_library_media_adjacent_item(1)

    def action_library_media_prev_item(self) -> None:
        """Open the previous browse item in the Reader (task-28005)."""
        self._select_library_media_adjacent_item(-1)

    def _select_library_media_adjacent_item(self, direction: int) -> None:
        if not self._library_media_item_traversal_active():
            return
        # task-28241: an active review set supersedes the browse-row traversal
        # -- ] / [ walk its pinned cursor over the whole set instead of the
        # mounted (current-page) rows. With no active set, fall through to
        # task-28005's browse-row behavior.
        if self._walk_active_review_set(direction):
            return
        neighbour = self._library_media_adjacent_row(direction)
        if neighbour is None:
            return
        media_id, title = neighbour
        self._select_library_media_reader_row(media_id, title, immediate=True)

    @on(Button.Pressed, "#library-media-review")
    def handle_library_media_review_these(self, event: Button.Pressed) -> None:
        """Pin the WHOLE filtered list as a review set and open it.

        Args:
            event: The "Review these" button press.
        """
        event.stop()
        self.run_worker(
            self._review_these_worker(),
            group="library_review_set",
            exclusive=True,
            exit_on_error=False,
        )

    @on(Button.Pressed, "#library-media-review-selected")
    def handle_library_media_review_selected(self, event: Button.Pressed) -> None:
        """Pin the current Select-mode selection as a review set and open it.

        Args:
            event: The "Review selected" bulk-action button press.
        """
        event.stop()
        if getattr(self, "_library_media_bulk_delete_in_flight", False):
            return
        selection = self._library_media_row_selection
        if not selection.count:
            return
        backing_ids = [
            backing_id
            for backing_id in (
                library_media_int_backing_id(canonical_id)
                for canonical_id in selection.ids
            )
            if backing_id is not None
        ]
        if not backing_ids:
            return
        self.run_worker(
            self._review_selected_worker(tuple(backing_ids)),
            group="library_review_set",
            exclusive=True,
            exit_on_error=False,
        )

    @on(Button.Pressed, "#library-media-review-sets")
    def handle_library_media_review_sets(self, event: Button.Pressed) -> None:
        """Open the saved review-set picker (resume / switch / dismiss).

        Args:
            event: The "Sets" toolbar button press.
        """
        event.stop()
        self.run_worker(
            self._review_set_picker_worker(),
            group="library_review_set",
            exclusive=True,
            exit_on_error=False,
        )

    def _focus_library_media_items_pane(self) -> None:
        """Focus the Items pane at its most useful control (task-28004).

        Prefers the loaded/selected item's ROW so the very next Down/Up
        walks the list -- Escape-then-Down is the sequential-review
        gesture (Down moves selection and auto-loads the adjacent item).
        Landing on the "Filter media" Input instead swallowed those
        keystrokes (typed characters fell into the filter, Down was inert
        until a Tab) -- live-verified 2026-09-02. Falls back to the first
        row, then to the filter when the list is empty.
        """
        rows = list(self.query(".library-media-row"))
        if rows:
            selected = str(self._selected_media_id or "")
            target = next(
                (
                    row
                    for row in rows
                    if str(getattr(row, "media_id", "") or "") == selected
                ),
                rows[0],
            )
            self.set_focus(target)
            return
        self._focus_library_control("#library-media-filter")

    def _library_media_list_surface_active(self, *, require_focus: bool = True) -> bool:
        """Whether the media LIST surface is live, Reader open or not.

        task-31272: with "‹ Back" gone from the three-pane shell,
        ``_library_media_view`` stays ``"viewer"`` for the whole visit, so
        every behaviour keyed on that flag (the ``s`` select gate, the list
        footer set, the type/sort choice strips, the background list
        refresh) would stay off while the Items pane is plainly on screen
        beside the Reader. In that layout the list is live too -- and for
        the KEY-owning behaviours only while the Items region actually
        holds focus, so the Reader keeps its own keys while the Reader is
        where the user is. ``require_focus=False`` asks the
        visibility-only question, for callers with no focus semantics
        (a background refresh of the list the user can see).

        Args:
            require_focus: Whether the Items region must hold focus for
                the three-pane Reader to count.

        Returns:
            True when the media list surface owns this behaviour.
        """
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_MEDIA:
            return False
        if self._library_media_view == _MEDIA_VIEW_LIST:
            return True
        if self._library_media_view != _MEDIA_VIEW_VIEWER:
            return False
        try:
            shell = self.query_one(
                ".library-media-route", LibraryBrowseReaderShell
            )
        except (NoMatches, QueryError):
            return False
        focused = self.focused
        items_focused = focused is not None and (
            focused is shell.items or shell.items in focused.ancestors
        )
        if items_focused:
            # task-32060: standing ON the list is the list surface, whatever
            # the Reader's exit says. The exit check below used to run first,
            # so in every layout with a real exit (100x30: Library collapsed,
            # Items beside the Reader) "s" was inert from a focused row and
            # the footer dropped its chip -- critique #8, forcing the mouse.
            return True
        if self._library_media_reader_exit_available(shell.effective_layout):
            # Not the three-pane shell: this Reader has a real exit, and
            # "list" is reachable again through it.
            return False
        return not require_focus

    @on(Button.Pressed, "#library-media-edit")
    def handle_library_media_edit(self, event: Button.Pressed) -> None:
        """Enter metadata edit mode for the open Library media viewer.

        Args:
            event: Button press event emitted by the viewer's "Edit" action.
        """
        event.stop()
        self._library_media_reader_session = set_mode(
            self._library_media_reader_session, "info"
        )
        self._library_media_editing = True
        self._sync_library_media_viewer_or_recompose()

    @on(Button.Pressed, "#library-media-edit-cancel")
    def handle_library_media_edit_cancel(self, event: Button.Pressed) -> None:
        """Discard in-progress metadata edits and return to the read-only viewer.

        Args:
            event: Button press event emitted by the edit form's "Cancel" action.
        """
        event.stop()
        self._library_media_editing = False
        self._sync_library_media_viewer_or_recompose()

    def _patch_local_media_record(
        self,
        media_id: str,
        *,
        title: str,
        author: str,
        url: str,
        keywords: list[str],
    ) -> None:
        """Update the cached media snapshot record after a saved metadata edit.

        The viewer re-fetches its detail from the backend, but the media
        *list* is rendered from ``_local_source_records['media']`` (seeded
        once at load), so without this the list keeps showing the pre-edit
        fields until a full snapshot refresh. Only the record whose id
        matches ``media_id`` is replaced; all others are passed through
        unchanged.

        Args:
            media_id: The edited media item's id.
            title: Saved title value.
            author: Saved author value.
            url: Saved URL value.
            keywords: Saved keywords list.
        """
        self._local_source_records["media"] = tuple(
            dict(record, title=title, author=author, url=url, keywords=list(keywords))
            if self._source_record_id(record) == media_id
            else record
            for record in self._local_source_records.get("media", ())
        )

    @on(Button.Pressed, "#library-media-delete-cancel")
    def handle_library_media_delete_cancel(self, event: Button.Pressed) -> None:
        """Discard the pending delete confirmation and restore the normal action row.

        Args:
            event: Button press event emitted by the confirm affordance's
                "Cancel" action.
        """
        event.stop()
        self._library_media_confirming_delete = False
        self._sync_library_media_viewer_or_recompose()

    @on(Button.Pressed, "#library-media-highlight-add")
    def handle_library_media_highlight_add(self, event: Button.Pressed) -> None:
        """Read the add-highlight form and hand the write off to a worker.

        Reads the three highlight inputs directly (before any recompose
        removes them) and silently ignores the press when the quote is
        blank -- mirroring how ``handle_library_media_edit_save`` reads its
        form inputs synchronously before deferring to a worker.

        Args:
            event: Button press event emitted by the highlights form's "Add
                highlight" action.
        """
        event.stop()
        media_id = self._selected_media_id
        if not media_id:
            return
        try:
            quote = self.query_one("#library-media-highlight-quote", Input).value
            note = self.query_one("#library-media-highlight-note", Input).value
            color = self.query_one("#library-media-highlight-color", Input).value
        except (NoMatches, QueryError):
            return
        # Validate/sanitize each user-entered field at the UI boundary before
        # it reaches the persistence service.
        quote = self._sanitize_media_field(quote, max_length=10000).strip()
        if not quote:
            return
        note = self._sanitize_media_field(note, max_length=10000).strip() or None
        color = self._sanitize_media_field(color, max_length=100).strip() or None
        # Exclusive in its own group so a rapid double-press cancels the first
        # add instead of inserting the same highlight twice (the add write is
        # not idempotent, unlike delete).
        self.run_worker(
            self._add_library_media_highlight(
                media_id,
                quote=quote,
                note=note,
                color=color,
            ),
            exclusive=True,
            group="library_media_highlight_add",
        )

    async def _add_library_media_highlight(
        self,
        media_id: str,
        *,
        quote: str,
        note: str | None,
        color: str | None,
    ) -> None:
        """Create a new reading highlight, then re-fetch the highlights list.

        Guards against a missing ``create_highlight`` service or a failed
        write by logging the failure and surfacing a quiet notice, but
        always re-fetches highlights afterwards so the section never shows a
        stale list.

        Args:
            media_id: The Library media item id to attach the highlight to.
            quote: The highlighted quote text.
            note: Optional note text, or None.
            color: Optional highlight color, or None.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        create_highlight = getattr(service, "create_highlight", None)
        service_media_id = self._library_media_backing_id(media_id)
        if callable(create_highlight):
            try:
                await self._run_library_service_call(
                    create_highlight,
                    mode="local",
                    item_id=service_media_id,
                    quote=quote,
                    note=note,
                    color=color,
                    isolate_in_worker=True,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    f"Failed to add Library media highlight for {media_id!r}."
                )
                self._notify_library_media_highlight_warning(
                    "Could not add this highlight."
                )
        else:
            self._notify_library_media_highlight_warning("Highlights are unavailable.")
        await self._reload_library_media_highlights(media_id)

    @on(Button.Pressed, ".library-media-highlight-delete")
    def handle_library_media_highlight_delete(self, event: Button.Pressed) -> None:
        """Read the pressed row's highlight id and hand the delete off to a worker.

        Args:
            event: Button press event emitted by a highlight row's delete
                action.
        """
        event.stop()
        media_id = self._selected_media_id
        highlight_id = getattr(event.button, "highlight_id", "")
        if not media_id or not highlight_id:
            return
        self.run_worker(self._delete_library_media_highlight(media_id, highlight_id))

    async def _delete_library_media_highlight(
        self, media_id: str, highlight_id: Any
    ) -> None:
        """Delete a reading highlight, then re-fetch the highlights list.

        Guards against a missing ``delete_highlight`` service or a failed
        write by logging the failure and surfacing a quiet notice, but
        always re-fetches highlights afterwards so the section never shows a
        stale list.

        Args:
            media_id: The Library media item id the highlight belongs to.
            highlight_id: The highlight id to delete.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        delete_highlight = getattr(service, "delete_highlight", None)
        if callable(delete_highlight):
            try:
                await self._run_library_service_call(
                    delete_highlight,
                    mode="local",
                    highlight_id=highlight_id,
                    isolate_in_worker=True,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    f"Failed to delete Library media highlight {highlight_id!r}."
                )
                self._notify_library_media_highlight_warning(
                    "Could not delete this highlight."
                )
        else:
            self._notify_library_media_highlight_warning("Highlights are unavailable.")
        await self._reload_library_media_highlights(media_id)

    def _notify_library_media_highlight_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed highlight mutation.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    @on(Input.Submitted, "#library-media-content-search")
    def handle_library_media_content_search_submitted(
        self, event: Input.Submitted
    ) -> None:
        """Set the in-content search query and jump to the first match.

        Submitted (rather than Changed) is used deliberately, so the query
        only takes effect on Enter. The mounted viewer coordinates its focused
        search controls and persistent body without rebuilding the screen.

        Args:
            event: Input submit event emitted by the content search box.
        """
        event.stop()
        # Strip once at the source so the status count, the body highlighting,
        # and prev/next navigation all search the exact same needle.
        submitted = event.value.strip()
        if submitted == self._library_media_content_query:
            # task-28011: re-pressing Enter on the same query walks to the
            # next match (find-bar convention) instead of no-opping.
            self._advance_library_media_content_match(1)
            return
        self._library_media_content_query = submitted
        self._library_media_content_match_index = 0
        matches = self._library_media_content_matches()
        viewer = self._mounted_library_media_viewer()
        if viewer is None:
            return
        try:
            viewer.sync_query_state(
                query=submitted,
                matches=matches,
                match_index=0,
            )
        except (NoMatches, QueryError):
            return
        self.call_after_refresh(self._focus_library_media_content_search_input)
        # Bring the first match into view after Rich text wrapping/layout settles.
        if matches:
            self.call_after_refresh(
                self._scroll_library_media_content_to_line, matches[0]
            )

    def _build_library_media_active_child(self) -> Widget:
        """Build the mounted Items child for the current Media subview.

        Covers all three of the media canvas's views. The "trash" branch is
        not decoration: this builder is the shared source of truth for
        "which media child is current", and before task-21116's review round
        it silently fell through to the LIST canvas for a screen sitting in
        the Trash view -- unlike ``compose_content`` and
        ``_build_library_entry_active_child``, which both checked "trash"
        first. A media-detail worker resolving after the user pressed
        "Trash" therefore mounted the list over the mounted
        ``LibraryMediaTrashCanvas`` while ``_library_media_view`` stayed
        "trash", diverging the DOM from the state and breaking the
        follow-on ``_sync_library_canvas(self, "media-trash")``. The
        route-key supersede guard cannot catch that: the view is INSIDE the
        route key, so the key stays self-consistent.
        """
        if self._library_media_view == "trash":
            trash_state = self._build_library_media_trash_state()
            return LibraryMediaTrashCanvas(
                trash_state,
                **self._library_media_trash_canvas_presentation(),
                id="library-media-trash-canvas",
            )
        media_state = self._build_library_media_state()
        if media_state.selected_id or self._library_media_view == "list":
            self._selected_media_id = media_state.selected_id
        return LibraryMediaCanvas(
            media_state,
            actions=self,
            **self._library_media_canvas_presentation(),
            id="library-media-canvas",
        )

    def _build_library_media_reader(self) -> LibraryMediaViewer:
        """Build the permanent Reader from loaded detail or its empty state."""
        detail = (
            self._library_media_detail
            if isinstance(self._library_media_detail, Mapping)
            else None
        )
        if detail is not None:
            self._library_media_composed_detail = detail
        arrival_note = self._pop_library_media_arrival_note()
        (
            preview_widget,
            preview_status,
            preview_hidden,
            preview_available,
            preview_source,
        ) = self._library_media_image_preview_projection()
        viewer = LibraryMediaViewer(
            self._build_library_media_viewer_display_state(
                detail, arrival_note=arrival_note
            ),
            editing=self._library_media_editing,
            confirming_delete=self._library_media_confirming_delete,
            highlights=build_library_media_highlight_rows(
                self._library_media_highlights
            ),
            editing_analysis=self._library_media_editing_analysis,
            generating_analysis=self._library_media_generating_analysis,
            # Review I1: gated on the tab that renders it -- see the twin
            # gate in ``_sync_library_media_viewer_state``.
            analysis_provider_reason=(
                self._library_media_analysis_provider_reason()
                if self._library_media_reader_session.mode == "analysis"
                else ""
            ),
            content_query=self._library_media_content_query,
            content_match_index=self._library_media_content_match_index,
            content_mode=self._library_media_content_mode,
            find_open=self._library_media_find_open,
            find_focus_pending=self._consume_library_media_find_focus(),
            loading=(
                self._library_media_reader_session.pending_request is not None
                and self._library_media_reader_session.error is None
            ),
            loading_message=(
                self._library_media_reader_session.pending_banner or "Loading media…"
            ),
            error_message=self._library_media_reader_session.error or "",
            reader_mode=self._library_media_reader_session.mode,
            more_open=self._library_media_reader_session.more_open,
            external_detail=self._library_media_reader_session.external_detail,
            console_representation=self._library_media_console_representation(),
            image_preview=preview_widget,
            image_preview_status=preview_status,
            image_preview_hidden=preview_hidden,
            image_preview_available=preview_available,
            image_preview_source=preview_source,
            review_banner=self._active_review_set_banner() or "",
            back_visible=self._library_media_reader_exit_available(),
            # task-31635 (critique #5 item 12): only the EMPTY Reader reads
            # this -- with the list load failed there is nothing to select.
            list_failed=self._library_media_list_unselectable(),
            # task-31635 (critique #5 item 11): the Items pane beside this
            # Reader is the Trash list, so the Reader names the list its own
            # (live) item actually belongs to.
            trash_list_open=self._library_media_view == "trash",
            # TASK-31745: what the speaker-rename legend needs to actually
            # persist a rename (harmless when the state says it cannot).
            media_db=getattr(self.app_instance, "media_db", None),
            speaker_rename_media_id=self._library_media_selected_backing_id(),
            id="library-media-viewer",
        )
        viewer._library_entry_arrival_note = arrival_note
        return viewer

    def _library_media_image_preview_projection(
        self, *, build_widget: bool = True
    ) -> tuple[Widget | None, str, bool, bool, Any]:
        """Build the current item's ephemeral preview projection.

        task-22208: with ``build_widget=False`` this returns only the cheap
        identity facts (status, hidden, available, source image) and never
        invokes the widget factory -- the sync's unchanged compare reads
        exactly those four fields, so the expensive PIL build must not run
        just to decide nothing changed. The widget slot is None in that
        mode; callers that conclude "changed" call again with the default
        to build (and to surface a build failure through the normal
        failure-status path).

        Args:
            build_widget: Whether to construct the preview widget for the
                visible case.

        Returns:
            (widget, status, hidden, available, source-image) projection.
        """
        session = self._library_media_reader_session
        canonical_id = session.loaded_id or ""
        if (
            not canonical_id
            or session.external_detail
            or not self._library_media_image_preview_capable()
        ):
            return None, "", False, False, None
        image = self._library_media_preview_images.get(canonical_id)
        status = self._library_media_preview_status.get(canonical_id, "")
        hidden = canonical_id in self._library_media_preview_hidden
        if image is None:
            return None, status, False, False, None
        if hidden:
            return None, status, True, True, image
        if not build_widget:
            return None, status, False, True, image
        try:
            factory = self._library_media_preview_factory
            if factory is None:
                from ...Widgets.Library.library_media_image_preview import (
                    build_media_image_widget,
                )

                factory = build_media_image_widget
            config = getattr(self.app_instance, "app_config", {})
            widget = factory(
                image,
                app_config=config if isinstance(config, Mapping) else {},
                box_cols=max(20, min(72, int(self.size.width or 72) - 4)),
                box_lines=18,
            )
            return widget, status, False, True, image
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to render Library media image preview for {}.", canonical_id
            )
            self._library_media_preview_images.pop(canonical_id, None)
            failure = "Image preview failed — showing complete stored text"
            self._library_media_preview_status[canonical_id] = failure
            return None, failure, False, False, None

    def _build_library_media_viewer_display_state(
        self, detail: Mapping[str, Any] | None, *, arrival_note: str = ""
    ):
        """Build display facts and qualify one-off server provenance."""
        session = self._library_media_reader_session
        return self._library_media_viewer_state_cached(
            detail,
            arrival_note=arrival_note,
            backend="server" if session.external_detail else "local",
            canonical_id=(session.loaded_id or "") if session.external_detail else "",
            force_raw=session.external_detail,
        )

    def _sync_library_media_viewer_state(self, viewer: LibraryMediaViewer) -> bool:
        """Re-read every viewer compose input after its mount await.

        task-21116: skips the viewer-scoped recompose when every compose
        input is unchanged from what the mounted viewer already rendered.
        The strict replacement seam re-syncs a freshly built viewer to
        catch state that changed while its mount awaited the message pump
        -- but in the common case nothing changed, and the unconditional
        ``viewer.refresh(recompose=True)`` re-mounted a brand-new
        ``Markdown`` body whose mount re-parses the full document (the
        exact double-parse task-15458 pinned for the open path).
        ``LibraryMediaViewerState`` is a frozen dataclass, so the
        comparison is structural.
        """
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_MEDIA:
            return False
        detail = (
            self._library_media_detail
            if isinstance(self._library_media_detail, Mapping)
            else None
        )
        viewer_state = self._build_library_media_viewer_display_state(
            detail,
            arrival_note=str(getattr(viewer, "_library_entry_arrival_note", "") or ""),
        )
        highlights = tuple(
            build_library_media_highlight_rows(self._library_media_highlights)
        )
        # task-22208: identity facts only -- the compare below reads status/
        # hidden/available/source, so the PIL preview widget must not be
        # built (and discarded) just to decide nothing changed.
        (
            _,
            preview_status,
            preview_hidden,
            preview_available,
            preview_source,
        ) = self._library_media_image_preview_projection(build_widget=False)
        loading = (
            self._library_media_reader_session.pending_request is not None
            and self._library_media_reader_session.error is None
        )
        loading_message = (
            self._library_media_reader_session.pending_banner or "Loading media…"
        )
        # task-22207: ``loading``/``loading_message`` are deliberately NOT
        # part of this comparison. Every traversal keystroke flips the
        # pending flag before this runs, so including it made the test
        # always fall through to a full viewer recompose -- re-parsing the
        # document being LEFT purely to paint "Loading…". The comparison
        # now covers only real compose identity (display state built from
        # the loaded detail, sub-state flags, highlights, search, mode,
        # preview identity); the loading placeholder is patched in place on
        # the unchanged path via ``viewer.sync_loading_state``.
        # task-22208: identity first -- ``viewer_state`` is memoized per
        # detail arrival (``_library_media_viewer_state_cached``), so on a
        # no-change sync it IS the object the mounted viewer already holds
        # and the O(document) structural compare (which memcmps the whole
        # content string) never runs. The ``==`` fallback stays: identity is
        # inconclusive exactly when a new detail arrived, and a re-fetch
        # with byte-identical values must still compare EQUAL so the
        # document is not rebuilt (pinned by task-22207's alternating-focus
        # probe).
        # task-30045: the review banner is a compose input like any other --
        # without it here, m/walk syncs kept the stale (or absent) banner.
        review_banner = self._active_review_set_banner() or ""
        # task-31272: a compose input like any other -- the layout that
        # decides it changes under resizes and pane toggles, and a viewer
        # attribute missing from this compare silently never updates.
        back_visible = self._library_media_reader_exit_available()
        # task-31635 item 11: a compose input like any other -- the Trash
        # entry recomposes the screen today, but a viewer-scoped sync landing
        # after it must not paint a Reader that has forgotten which list it
        # is standing beside.
        trash_list_open = self._library_media_view == "trash"
        # task-28007 AC#5: a compose input like any other -- resolved once
        # per sync and read by both halves below.
        # Review I1: only ``_compose_analysis`` consumes this, and
        # ``resolve_ingest_analysis_provider`` is not always cheap -- for
        # ``provider = "Anthropic"`` with ``auth_source =
        # "claude_subscription"`` readiness shells out to the macOS keychain
        # behind a 5s TTL, which human-paced Reader gestures would miss on
        # every sync. So it is resolved only on the tab that shows it. The
        # compare stays honest: ``reader_mode`` is itself a compose input,
        # so switching tabs recomposes and picks the reason up.
        analysis_provider_reason = (
            self._library_media_analysis_provider_reason()
            if self._library_media_reader_session.mode == "analysis"
            else ""
        )
        unchanged = (
            (viewer.viewer is viewer_state or viewer.viewer == viewer_state)
            and viewer.review_banner == review_banner
            and viewer.back_visible == back_visible
            and viewer.trash_list_open == trash_list_open
            and viewer.editing == self._library_media_editing
            and viewer.confirming_delete == self._library_media_confirming_delete
            and tuple(viewer.highlights) == highlights
            and viewer.editing_analysis == self._library_media_editing_analysis
            # Qodo #8 on PR #2601: assigned on the changed path below but
            # never compared here, so a generation that started with nothing
            # else changing took the unchanged path and the Reader kept
            # rendering the PREVIOUS reason -- "No analysis to search yet."
            # while one was generating, plus a stale "○" label and tooltip.
            and viewer.generating_analysis == self._library_media_generating_analysis
            and viewer.analysis_provider_reason == analysis_provider_reason
            and viewer.content_query == self._library_media_content_query
            and viewer.content_match_index == self._library_media_content_match_index
            and viewer.content_mode == self._library_media_content_mode
            and viewer.find_open == self._library_media_find_open
            and not self._library_media_find_focus_pending
            and viewer.error_message == (self._library_media_reader_session.error or "")
            and viewer.reader_mode == self._library_media_reader_session.mode
            and viewer.more_open == self._library_media_reader_session.more_open
            and viewer.external_detail
            == self._library_media_reader_session.external_detail
            and viewer.console_representation
            == self._library_media_console_representation()
            and viewer.image_preview_source is preview_source
            and viewer.image_preview_status == preview_status
            and viewer.image_preview_hidden == preview_hidden
            and viewer.image_preview_available == preview_available
        )
        if unchanged:
            # Re-anchor the memoized state object so the NEXT no-change sync
            # short-circuits on identity again -- after an identical
            # re-fetch settles (new detail dict, equal values), the mounted
            # viewer would otherwise hold the previous arrival's state
            # object and pay the structural compare on every sync until
            # something actually changed.
            viewer.viewer = viewer_state
            viewer.sync_loading_state(loading=loading, message=loading_message)
        else:
            # Something changed, so the viewer recomposes: build the
            # preview widget now (a fresh instance -- a removed widget
            # cannot be remounted; the expensive mosaic renderable inside is
            # memoized by ``build_media_image_widget``). Assign the
            # POST-build projection: a build failure flips the status caches
            # and this keeps the mounted state consistent with them.
            (
                preview_widget,
                preview_status,
                preview_hidden,
                preview_available,
                preview_source,
            ) = self._library_media_image_preview_projection()
            viewer.viewer = viewer_state
            viewer.editing = self._library_media_editing
            viewer.confirming_delete = self._library_media_confirming_delete
            viewer.highlights = highlights
            viewer.editing_analysis = self._library_media_editing_analysis
            viewer.generating_analysis = self._library_media_generating_analysis
            viewer.analysis_provider_reason = analysis_provider_reason
            viewer.content_query = self._library_media_content_query
            viewer.content_match_index = self._library_media_content_match_index
            viewer.content_mode = self._library_media_content_mode
            viewer.find_open = self._library_media_find_open
            viewer.find_focus_pending = self._consume_library_media_find_focus()
            viewer.loading = loading
            viewer.loading_message = loading_message
            viewer.error_message = self._library_media_reader_session.error or ""
            viewer.reader_mode = self._library_media_reader_session.mode
            viewer.more_open = self._library_media_reader_session.more_open
            viewer.external_detail = self._library_media_reader_session.external_detail
            viewer.console_representation = self._library_media_console_representation()
            viewer.image_preview = preview_widget
            viewer.image_preview_status = preview_status
            viewer.image_preview_hidden = preview_hidden
            viewer.image_preview_available = preview_available
            viewer.image_preview_source = preview_source
            viewer.review_banner = review_banner
            viewer.back_visible = back_visible
            viewer.trash_list_open = trash_list_open
            # TASK-31745: the backing id follows the selection like any other
            # compose input (the state's own rename fields are in the compare
            # above, so a stale id here could never outlive them).
            viewer.media_db = getattr(self.app_instance, "media_db", None)
            viewer.speaker_rename_media_id = (
                self._library_media_selected_backing_id()
            )
            # task-31567: this recompose replaces every child, so whatever
            # the user was standing on (the content box, the Find input, an
            # action button) is about to be removed and Textual will pick
            # the shell's first focusable widget -- a pane GRIP. The restore
            # rides the viewer's own post-recompose hook rather than
            # ``screen.call_after_refresh``: this rebuild is driven by the
            # VIEWER's message pump, and the two have no ordering (the same
            # trap ``PostRecomposeCallback`` exists for -- measured: the
            # screen-level callback ran before the new children mounted and
            # the grip still won).
            # ...unless the Find token owns this recompose. It was consumed
            # into ``find_focus_pending`` a few lines above, so the restore's
            # own token guard reads False by the time it runs (review I2);
            # read the viewer's copy here instead. Without this the seam
            # re-focused the pressed Find BUTTON on the way through -- a
            # third focus move, and a foreign ``DescendantFocus``, inside the
            # window the token channel owns.
            if not viewer.find_focus_pending:
                viewer.queue_after_recompose(
                    partial(
                        self._restore_library_media_focus,
                        self._capture_library_media_focus_identity(),
                    )
                )
            viewer.refresh(recompose=True)
        if detail is not None:
            self._library_media_composed_detail = detail
        # task-31950: both tail follow-ups read the viewer's children (the
        # edit Save, the content body's scroller) and the rebuild above runs
        # on the VIEWER's pump, so they ride the viewer's hook rather than
        # the screen's -- the same ordering PR H2 gave the focus follow-ups.
        # On the no-change path nothing was rebuilt and the seam falls back
        # to ``call_after_refresh``, exactly as before.
        self._queue_after_library_media_viewer_recompose(
            self._sync_library_media_viewer_mutation_gate, viewer
        )
        loaded_id = self._library_media_reader_session.loaded_id
        if (
            loaded_id is not None
            and self._library_media_progress_restored_id != loaded_id
        ):
            self._library_media_progress_restored_id = loaded_id
            self._queue_after_library_media_viewer_recompose(
                partial(self._restore_library_media_loaded_progress, loaded_id),
                viewer,
            )
        return True

    @on(Button.Pressed, "#library-media-reader-retry")
    def handle_library_media_reader_retry(self, event: Button.Pressed) -> None:
        """Retry the current failed Reader request immediately."""
        event.stop()
        session = self._library_media_reader_session
        if session.selected_id is None or session.error is None:
            return
        self._select_library_media_reader_row(
            session.selected_id,
            session.selected_title or session.selected_id,
            immediate=True,
        )

    @on(Button.Pressed, "#library-media-reader-more")
    def handle_library_media_reader_more(self, event: Button.Pressed) -> None:
        """Toggle the transient inline secondary-action region.

        Args:
            event: The More button press. Stopped here so the Reader's own
                toolbar handling never sees it.

        Returns:
            None.
        """
        event.stop()
        session = self._library_media_reader_session
        self._library_media_reader_session = set_more_open(
            session, not session.more_open
        )
        # task-31633 AC#3: the disclosure owns its own focus target -- PR F's
        # restore seam otherwise leaves focus wherever it already was (the
        # Items row that opened the Reader), so More could not be closed
        # again without hunting for it. The shared seam is what orders that
        # target against the viewer's own rebuild.
        self._after_library_media_viewer_sync("#library-media-reader-more")

    @on(Button.Pressed, "#library-media-image-preview-toggle")
    def handle_library_media_image_preview_toggle(self, event: Button.Pressed) -> None:
        """Hide or reveal the loaded item's cached preview for this session."""
        event.stop()
        canonical_id = self._library_media_reader_session.loaded_id
        if not canonical_id or canonical_id not in self._library_media_preview_images:
            return
        if canonical_id in self._library_media_preview_hidden:
            self._library_media_preview_hidden.remove(canonical_id)
        else:
            self._library_media_preview_hidden.add(canonical_id)
        self._sync_library_media_viewer_or_recompose()

    @on(Button.Pressed, "#library-media-image-preview-retry")
    def handle_library_media_image_preview_retry(self, event: Button.Pressed) -> None:
        """Retry only the loaded item's preview; never reload its detail."""
        event.stop()
        session = self._library_media_reader_session
        detail = self._library_media_detail
        if (
            session.external_detail
            or session.loaded_id is None
            or session.loaded_backing_id is None
            or not isinstance(detail, Mapping)
        ):
            return
        self._schedule_library_media_image_preview(
            request_generation=session.request_generation,
            canonical_id=session.loaded_id,
            backing_id=session.loaded_backing_id,
            detail=detail,
            force=True,
        )
        self._sync_library_media_viewer_or_recompose()

    @on(Button.Pressed, ".library-media-reader-mode")
    def handle_library_media_reader_mode(self, event: Button.Pressed) -> None:
        """Select one permanent Reader mode without reloading the item.

        Args:
            event: The Reader mode button press; its id names the target mode.
        """
        event.stop()
        button_id = str(event.button.id or "")
        prefix = "library-media-reader-select-"
        if not button_id.startswith(prefix):
            return
        mode = button_id.removeprefix(prefix)
        self._capture_library_media_loaded_progress()
        self._reset_library_media_search_on_mode_change(mode)
        self._library_media_reader_session = set_mode(
            self._library_media_reader_session,
            mode,  # type: ignore[arg-type]
        )
        # The one follow-up here is not a focus move -- the stored reading
        # position is restored against the rebuilt body, and reading it from
        # the OLD children (or after they are detached) is the same race.
        loaded_id = self._library_media_reader_session.loaded_id
        if mode == "read" and loaded_id is not None:
            # task-31954: ONE owner. This used to re-arm the sync tail's
            # arm-once guard (``_library_media_progress_restored_id = None``)
            # as well, so a single Analysis -> Read press restored twice --
            # harmless only while the restore stays an idempotent
            # ``scroll_to``. Claiming the id here instead keeps the tail
            # quiet for this sync and leaves it owning the LOAD path, where
            # the id genuinely changes.
            self._library_media_progress_restored_id = loaded_id
            self._after_library_media_viewer_sync(
                partial(self._restore_library_media_loaded_progress, loaded_id)
            )
        else:
            self._sync_library_media_viewer_or_recompose()

    def _consume_library_media_find_focus(self) -> bool:
        """Return and clear the one-shot Find-gesture focus token (task-31269)."""
        pending = self._library_media_find_focus_pending
        self._library_media_find_focus_pending = False
        return pending

    @on(Button.Pressed, "#library-media-open-original")
    def handle_library_media_open_original(self, event: Button.Pressed) -> None:
        """Open the stored original URL when the detail actually has one."""
        event.stop()
        detail = self._library_media_detail
        url = self._safe_text(detail.get("url")) if isinstance(detail, Mapping) else ""
        if url.startswith(("http://", "https://")):
            webbrowser.open(url)

    def _capture_library_media_focus_identity(self) -> str | None:
        """Record what holds focus just before a media-path recompose.

        task-31567. Returns an id selector (a media row, ``#library-media-
        viewer-content``, the Find input, a toolbar button, ...) or ``None``
        when nothing focusable-with-an-id owns focus. A pane GRIP is never
        captured, so the restore below can never put focus back on one.
        """
        focused = self.focused
        widget_id = getattr(focused, "id", None)
        if not widget_id or focused.has_class(LIBRARY_ADAPTIVE_READER_GRIP_CLASS):
            return None
        return f"#{widget_id}"

    def _restore_library_media_focus(self, previous: str | None) -> None:
        """Put focus back where a media recompose found it (task-31567 AC#1).

        Textual re-picks focus when the widget holding it is recomposed
        away, and the adaptive reader shell's two pane grips are the first
        focusable widgets in it -- so EVERY media recompose that did not
        name its own focus target landed the user on a grip, where Space
        collapses a pane (wave 4 PR B) and the reading position is gone.
        Measured before this seam, at 235x52 and 100x30 alike: a Reader
        recompose from the content or the Find input landed on
        ``library-browse-items-grip``; a receipt repaint or leaving select
        mode from a row landed on ``library-browse-library-grip``.

        An explicit target always WINS -- this only acts when nothing else
        claimed focus:

        * the two one-shot focus CHANNELS (task-31631's armed list-entry
          request and task-31269's Find token) are checked directly,
          because both land their focus in a LATER callback and would
          otherwise be disarmed by a foreign ``set_focus`` here;
        * any other explicit follow-up (PR E's ``focus_identity`` through
          ``_complete_library_media_mutation``, a ``then=`` focus callback)
          has already focused a real widget by the time this runs, which
          the "focus is not a grip" check below detects generically.

        Three choke points reach here, covering every media recompose: the
        ``kind == "media"`` branch of ``_sync_library_canvas`` and
        ``_sync_library_media_viewer_state`` (both canvas-scoped, via
        ``queue_after_recompose``), and -- for any WHOLE-screen
        ``refresh(recompose=True)`` -- ``restore_focus_after_recompose``,
        the shared ``BaseAppScreen`` seam this screen overrides (task-31946
        moved that hop off ``LibraryScreen.refresh`` itself). The last was
        added in the Qodo round: background workers and
        ``_sync_library_canvas``'s own failure fallback both take that bare
        path -- the fallback after CLEARING the follow-up it had queued --
        and left focus at ``None``.

        Args:
            previous: Identity captured by
                ``_capture_library_media_focus_identity`` before the
                recompose, or ``None`` to leave focus alone.
        """
        if previous is None:
            return
        if self._library_pending_list_entry_focus or (
            self._library_media_find_focus_pending
        ):
            return
        focused = self.focused
        if focused is not None and not focused.has_class(
            LIBRARY_ADAPTIVE_READER_GRIP_CLASS
        ):
            return
        try:
            target = self.query_one(previous, Widget)
        except (NoMatches, QueryError):
            # The widget itself is gone (a row the write removed, the Find
            # bar a mode switch closed): the list entry is the honest
            # landing spot -- never the grip Textual would pick.
            self._focus_library_list_entry()
            return
        if not target.focusable:
            # Same failure, quieter: ``Screen.set_focus`` NO-OPS on a widget
            # whose ``focusable`` is False, so treating this as success left
            # focus on the grip. The media canvas and the viewer both
            # compose gated actions disabled (Select all / Clear / Export /
            # Analyze / Delete, Find, Generate) and those gates flip on
            # exactly the recompose this seam wraps, so a captured id can
            # come back unfocusable (review I1).
            self._focus_library_list_entry()
            return
        # ``scroll_visible=False``: this seam restores FOCUS, never scroll
        # position -- Media owns that separately (the stored viewer-return
        # restore, and the wheel gesture that CANCELS it). Once the screen-
        # level hook made this run after every whole-screen recompose, the
        # default ``scroll_visible=True`` re-scrolled the focused row back
        # into view and reinstated exactly the restore the user's wheel had
        # just cancelled (``test_compact_media_viewer_back_wheel_scroll_
        # cancels_stored_restore``, scroll_y 45 where the pin wants 0).
        self.set_focus(target, scroll_visible=False)

    def _sync_library_media_viewer_or_recompose(self) -> None:
        """Viewer-scoped recompose for an in-viewer sub-state flip.

        task-21116: Escape's edit/delete-confirm/analysis Cancel branches
        flip one viewer flag; only the mounted ``LibraryMediaViewer``'s
        children change, so route the rebuild through the viewer itself
        instead of tearing down the whole screen. TASK-22228 (item 6)
        finished the conversion: the six BUTTON presses that make the same
        three flips (Edit metadata / Cancel, Move to trash / Cancel, Edit
        analysis / Cancel) were still whole-screen recomposes -- the same
        gesture through a different control paid the nav-bar + footer +
        rail + canvas teardown that Escape had already stopped paying.
        Mirrors
        ``_sync_library_canvas``'s mouse-capture release -- this path
        bypasses ``BaseAppScreen.refresh`` and its guard, and the viewer
        mounts ``Input``/``TextArea`` children whose removal mid-capture
        would otherwise strand ``App.mouse_captured`` on a dead widget.
        Any failure (viewer unmounted because a transition raced) falls
        back to the legacy whole-screen recompose.
        """
        if self.is_running:
            try:
                self.app.capture_mouse(None)
            except Exception:
                logger.debug(
                    "Mouse-capture release before media viewer sync skipped.",
                    exc_info=True,
                )
        # The retired whole-screen recompose re-ran ``compose_content``,
        # which re-registers the footer shortcut set on every pass. This
        # seam does not, and the set genuinely differs across the flip the
        # caller just made: ``_library_route_shortcuts_for_current_state``
        # branches on ``_library_media_viewer_substate_active()``, so
        # without this the footer kept advertising the sub-state's "back a
        # step" after Escape had already returned to the plain viewer,
        # where Escape means "back to list" (review round, M2). Registered
        # here rather than in each of Escape's three branches so no future
        # sub-state can be added without it.
        self._register_footer_shortcuts()
        # task-31567: the viewer-scoped branch queues its own restore from
        # inside ``_sync_library_media_viewer_state`` (it alone knows whether
        # a recompose actually happens). The whole-screen fallback needs
        # nothing here -- ``LibraryScreen.refresh`` captures and restores
        # around every screen recompose (Qodo round).
        self._sync_library_media_surfaces_or_recompose()

    def _sync_library_media_viewer_mutation_gate(self) -> None:
        """Disable a still-mounted edit Save while its write is unsettled."""
        try:
            save = self.query_one("#library-media-edit-save", Button)
        except (NoMatches, QueryError):
            return
        if self._library_media_bulk_delete_in_flight:
            save.label = library_disabled_action_label("Save", True)
            save.disabled = True
            save.tooltip = "Media change in progress."

    def _mounted_library_media_viewer(self) -> LibraryMediaViewer | None:
        """Return the mounted media viewer, or ``None`` during teardown."""
        try:
            return self.query_one("#library-media-viewer", LibraryMediaViewer)
        except (NoMatches, QueryError):
            return None

    def _focus_library_media_content_search_input(
        self, *, _retries: int = 4
    ) -> None:
        """Focus the mounted content search box after controls synchronize.

        task-31237: the input now MOUNTS during the viewer's recompose
        (the bar is collapsed until Find opens it), and screen-level
        ``call_after_refresh`` callbacks can flush before the widget-level
        recompose lands its children -- a short bounded re-defer chain
        covers the mount latency instead of silently losing the focus.
        """
        try:
            self.query_one("#library-media-content-search", Input).focus()
        except (NoMatches, QueryError):
            if _retries > 0:
                self.call_after_refresh(
                    partial(
                        self._focus_library_media_content_search_input,
                        _retries=_retries - 1,
                    )
                )

    @on(Button.Pressed, "#library-media-content-mode-rendered")
    async def handle_library_media_content_mode_rendered(
        self, event: Button.Pressed
    ) -> None:
        """Switch the open media item's Content section to the Rendered (Markdown) view.

        Args:
            event: Button press event emitted by the toggle strip's
                "Rendered" action.
        """
        event.stop()
        await self._set_library_media_content_mode("rendered")

    @on(Button.Pressed, "#library-media-content-mode-raw")
    async def handle_library_media_content_mode_raw(
        self, event: Button.Pressed
    ) -> None:
        """Switch the open media item's Content section to the Raw text view.

        Args:
            event: Button press event emitted by the toggle strip's "Raw"
                action.
        """
        event.stop()
        await self._set_library_media_content_mode("raw")

    async def _set_library_media_content_mode(self, mode: str) -> None:
        """Shared Rendered/Raw toggle: no-op when already in ``mode``.

        The persistent body lazily mounts each view once; no media write is
        involved because the viewer has no editable body while browsing.

        Args:
            mode: ``"rendered"`` or ``"raw"``.
        """
        if (
            self._library_media_view != "viewer"
            or self._library_media_content_mode == mode
        ):
            return
        self._library_media_content_mode = mode
        viewer = self._mounted_library_media_viewer()
        if viewer is None:
            return
        try:
            await viewer.sync_mode(mode)
        except (NoMatches, QueryError):
            return

    @on(Button.Pressed, "#library-media-content-search-next")
    def handle_library_media_content_search_next(self, event: Button.Pressed) -> None:
        """Advance to the next in-content search match and scroll it into view.

        Args:
            event: Button press event emitted by the "Next" search action.
        """
        event.stop()
        self._advance_library_media_content_match(1)

    @on(Button.Pressed, "#library-media-content-search-prev")
    def handle_library_media_content_search_prev(self, event: Button.Pressed) -> None:
        """Return to the previous in-content search match and scroll it into view.

        Args:
            event: Button press event emitted by the "Prev" search action.
        """
        event.stop()
        self._advance_library_media_content_match(-1)

    def _advance_library_media_content_match(self, step: int) -> None:
        """Move the current content-search match index and scroll to it.

        No-ops when there is no open item or the query has no matches
        (the status line already reads "No matches" in that case).

        Args:
            step: ``1`` to move to the next match, ``-1`` for the previous
                one; wraps around the match count either direction.
        """
        matches = self._library_media_content_matches()
        if not matches:
            return
        self._library_media_content_match_index = (
            self._library_media_content_match_index + step
        ) % len(matches)
        line_index = matches[self._library_media_content_match_index]
        viewer = self._mounted_library_media_viewer()
        if viewer is None:
            return
        try:
            viewer.sync_match_index(
                matches=matches,
                match_index=self._library_media_content_match_index,
            )
        except (NoMatches, QueryError):
            return
        self.call_after_refresh(self._scroll_library_media_content_to_line, line_index)

    def _scroll_library_media_content_to_line(self, line_index: int) -> None:
        """Scroll the content region so the given source line is visible.

        When Raw is the ACTIVE mode, ``line_index`` (a SOURCE line index)
        is mapped to its virtual row through the Raw view's wrap index --
        scrolling to the source-line index directly, as if it were already
        a screen row, drifts once any line wraps. Otherwise (Rendered mode)
        this falls back to scrolling the active scroller's Y axis by that
        index directly.

        Gated on ``body.active_mode`` rather than ``body.raw_view is not
        None``: the latter is a LIFETIME accessor -- once Raw has been
        mounted once it stays mounted (and non-``None``) for the rest of
        the body's life -- so it kept routing scroll requests to the Raw
        view even after the user had switched back to Rendered, silently
        scrolling a hidden widget while the visible one never moved.

        Args:
            line_index: 0-based line index within the content text to
                reveal.
        """
        try:
            body = self.query_one(
                "#library-media-viewer-content", LibraryMediaContentBody
            )
        except (NoMatches, QueryError):
            return
        raw_view = body.raw_view
        if body.active_mode == "raw" and raw_view is not None:
            raw_view.scroll_to_source_line(line_index)
            return
        body.scroller.scroll_to(y=line_index, animate=False)

    @on(Button.Pressed, "#library-media-read-later")
    def handle_library_media_read_later(self, event: Button.Pressed) -> None:
        """Toggle the open media item's read-it-later state via a worker.

        Reads the currently known saved state from ``_library_media_detail``
        (already reflecting ``is_read_it_later`` from the last fetch) to
        decide whether to save or remove, mirroring how
        ``handle_library_media_delete_confirm`` reads state synchronously
        before deferring to a worker.

        Args:
            event: Button press event emitted by the viewer's "Read it
                later" / "Remove from read-it-later" action.
        """
        event.stop()
        self._start_library_media_read_later_toggle()

    def _start_library_media_read_later_toggle(self) -> None:
        """Kick the read-it-later toggle worker for the open item (task-28027).

        Shared by the "Read later" button and the ``l`` accelerator so both
        read the last-fetched saved state and dispatch the same worker.
        """
        media_id = self._selected_media_id
        if not media_id:
            return
        detail = (
            self._library_media_detail
            if isinstance(self._library_media_detail, Mapping)
            else {}
        )
        currently_saved = bool(detail.get("is_read_it_later"))
        self.run_worker(
            self._toggle_library_media_read_later(
                media_id, currently_saved=currently_saved
            )
        )

    async def _toggle_library_media_read_later(
        self, media_id: str, *, currently_saved: bool
    ) -> None:
        """Save or remove the read-it-later state, then re-fetch detail.

        Guards against a missing ``save_to_read_it_later``/
        ``remove_from_read_it_later`` service or a failed write by logging
        the failure and surfacing a quiet notice, but always re-fetches
        detail afterwards so the button's label never shows a stale state.

        Args:
            media_id: The Library media item id to toggle.
            currently_saved: Whether the item is currently saved for
                read-it-later (determines whether to save or remove).
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        method_name = (
            "remove_from_read_it_later" if currently_saved else "save_to_read_it_later"
        )
        method = getattr(service, method_name, None)
        service_media_id = self._library_media_backing_id(media_id)
        if callable(method):
            try:
                await self._run_library_service_call(
                    method,
                    mode="local",
                    media_id=service_media_id,
                    isolate_in_worker=True,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    f"Failed to toggle Library media read-it-later state for {media_id!r}.",
                )
                self._notify_library_media_read_later_warning(
                    "Could not update read-it-later status."
                )
        else:
            self._notify_library_media_read_later_warning(
                "Read-it-later is unavailable."
            )
        await self._refresh_library_media_detail(media_id)

    def _notify_library_media_read_later_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed read-it-later toggle.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    @on(Button.Pressed, "#library-media-analysis-edit")
    def handle_library_media_analysis_edit(self, event: Button.Pressed) -> None:
        """Enter analysis edit mode for the open Library media viewer.

        Args:
            event: Button press event emitted by the analysis section's
                "Edit analysis" action.
        """
        event.stop()
        self._library_media_editing_analysis = True
        self._sync_library_media_viewer_or_recompose()

    @on(Button.Pressed, "#library-media-analysis-cancel")
    def handle_library_media_analysis_cancel(self, event: Button.Pressed) -> None:
        """Discard in-progress analysis edits and return to the read-only view.

        Args:
            event: Button press event emitted by the analysis edit form's
                "Cancel" action.
        """
        event.stop()
        self._library_media_editing_analysis = False
        self._sync_library_media_viewer_or_recompose()

    @on(Button.Pressed, "#library-media-analysis-save")
    def handle_library_media_analysis_save(self, event: Button.Pressed) -> None:
        """Read the analysis edit TextArea and hand the write off to a worker.

        Reads the edited analysis text directly (before any recompose
        removes the TextArea) and the current document content from the
        loaded detail -- ``save_analysis_version`` requires both, since it
        creates a new ``DocumentVersions`` row carrying the (unchanged)
        content alongside the edited analysis. Mirrors how
        ``handle_library_media_edit_save`` reads its form inputs
        synchronously before deferring to a worker.

        Args:
            event: Button press event emitted by the analysis edit form's
                "Save" action.
        """
        event.stop()
        media_id = self._selected_media_id
        if not media_id:
            self._library_media_editing_analysis = False
            self._sync_library_media_viewer_or_recompose()
            return
        try:
            analysis_content = self.query_one(
                "#library-media-analysis-edit-text", TextArea
            ).text
        except (NoMatches, QueryError):
            self._library_media_editing_analysis = False
            self._sync_library_media_viewer_or_recompose()
            return
        # Validate/sanitize the user-entered analysis at the UI boundary
        # before it reaches the persistence service.
        analysis_content = self._sanitize_media_field(
            analysis_content, max_length=100000
        )
        detail = (
            self._library_media_detail
            if isinstance(self._library_media_detail, Mapping)
            else {}
        )
        content = str(detail.get("content") or "")
        self.run_worker(
            self._save_library_media_analysis(
                media_id,
                content=content,
                analysis_content=analysis_content,
            )
        )

    async def _fetch_library_media_analysis_detail(
        self, media_id: str, *, include_content: bool
    ) -> Mapping[str, Any] | None:
        """Fetch one item's detail off the event loop for the bulk run.

        Deliberately NOT ``_refresh_library_media_detail``: that one owns
        the Reader's session state, and a bulk run must not move the
        Reader's selection forty times.

        Args:
            media_id: The canonical media id.
            include_content: Whether the document body is needed (the
                analyzed/not-analyzed pass does not need it).

        Returns:
            The detail mapping, or None when the service is unavailable or
            returned something else.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        get_media_item = getattr(service, "get_media_item", None)
        if not callable(get_media_item):
            return None
        detail = await self._run_library_service_call(
            get_media_item,
            mode="local",
            media_id=self._library_media_backing_id(media_id),
            include_content=include_content,
            include_versions=True,
            isolate_in_worker=True,
        )
        return detail if isinstance(detail, Mapping) else None

    @on(Button.Pressed, "#library-media-open")
    def handle_library_media_open(self, event: Button.Pressed) -> None:
        """Hand off to Library's own Media surface via the "media" route.

        Only the full in-Library viewer's action row carries this button.
        Before task-2851 this genuinely left Library for the separate
        legacy Media screen, which is why the button used to be labeled
        "Open in Media manager". Task-2851 retired that route -- the
        "media" route id now aliases to "library" in
        ``screen_registry._SCREEN_ALIASES`` -- so this now round-trips
        into Library's own restored state instead of a different screen.
        The label says what it actually opens now: "Open in Library ▸
        Media" (task-2857). The browse summary's in-Library action is
        ``#library-media-open-viewer`` ("Open in viewer") instead, and
        never posts a navigation message at all.

        Args:
            event: Button press event emitted by the "Open in Library ▸
                Media" action.
        """
        event.stop()
        self.post_message(NavigateToScreen("media"))

    @on(Button.Pressed, "#library-media-use-in-chat")
    def use_media_in_chat(self, event: Button.Pressed) -> None:
        """Handle the media viewer's "Use in Console" action.

        Args:
            event: Button press event emitted by the viewer's "Use in Console" action.
        """
        event.stop()
        self._open_selected_media_handoff()


# --- BEGIN generated media-controller-state shims ---
# Permanent, not a cleanup-PR deletion target -- same reasoning as
# `LibraryPromptsController`'s own identical block: the byte-for-byte canon
# (recipe §1) forbids editing a moved body, so the attribute names those
# bodies already use have to keep resolving through *something*. Exposes
# every `LibraryMediaState` field under its original flat name via task 1's
# single-source `media_state_shim_attr()` -- TWO prefix families
# (`_library_media_` default, bare `_` for `selected_media_id`).
for _lmc_field in dataclasses.fields(LibraryMediaState):
    setattr(
        LibraryMediaController,
        media_state_shim_attr(_lmc_field.name),
        property(
            lambda self, _n=_lmc_field.name: getattr(
                self._media_state_accessor(), _n
            ),
            lambda self, value, _n=_lmc_field.name: setattr(
                self._media_state_accessor(), _n, value
            ),
        ),
    )
del _lmc_field
# --- END generated media-controller-state shims ---
