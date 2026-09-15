# Dev698 merge verification: fixture-lifetime review

The retained run does not show147 failures of the merged feature behavior. It shows a consistent profile-selection refusal before that behavior is exercised:135 fixture setup errors and12 test-body failures while constructing their harness/app. The overall result remains12 failed/668 passed/135 errors in130.73s; these failures cannot be silently counted as passing. No tests or repository edits were performed for this review.

## Exact evidence

I parsed all147 error/failure sections of `/private/tmp/uat-dev-698-merge-focused.log`: each contains `raw_source_selection_changed`;135 headings are setup errors. The first executor failure is `target_harness`→NotesInteropService._get_db→load_console_library_migration_seed→guarded config load→raw._participant_state. The12 UI failures enter `_notes_host`/`_build_test_app` or `_build_fresh_wizard_app`→app_factory.build_test_app_config→load_settings, reaching the same refusal. These occur before the new link/rollback, note-count, graduation or provider-key actions/assertions. Calling the UI failures strictly "pre-body" would be inaccurate: the body begins, but fails inside its first setup helper.

The exact fresh single-node UI run `/private/tmp/uat-dev-698-ui-alone.log` also fails at this app-construction boundary in2.52s. That rules out preceding tests in the nine-file batch as a necessary trigger; it does not by itself prove every failure existed before the merge. Representative baseline execution remains needed for that classification.

## Mechanism and existing infrastructure

Root conftest creates and exports one private bootstrap HOME/config/data before collection. Collection imports install a source-bound config participant. Its autouse `isolate_test_environment` later selects per-test paths unless the exact case is an opted-in private-profile child (or one of the existing explicit MCP exceptions). It closes known lazy DBs but does not retire/rebind the config participant. At raw_participants.py127, current config binding must still match the participant's captured selected path. The late fixture retarget therefore correctly refuses instead of reusing authority from the earlier selection. Those guards and fixture implementations are unchanged from the premerge branch.

`Tests.private_profile.is_private_profile_child` requires BOTH the exact environment node ID and the decorated function's `_private_profile_test` flag. Simply exporting TLDW_TEST_PRIVATE_PROFILE_NODE for an undecorated new test will not fix this. Existing private_profile_test also expects a request fixture. Do not force its predicate true globally, clear participant registries, replace guards, or use noconftest for executing tests that depend on ordinary fixture isolation.

The wizard's own `_prepare_clean_environment` additionally changes HOME/XDG paths after collection; merely decorating every new test would not establish that all selected identity components remain consistent. A future narrowly authorized fixture conversion must inspect that helper too. This is why a broad fixture edit is not justified for merge verification.

## Smallest useful verification now

1. Retain the668 completed passes, with their exact selected nodes/JUnit if available, as real evidence for the tested Notes planner/receipts, session/state and other independent branches. They do not replace the135 unexecuted executor cases or12 unexercised UI behaviors.
2. Run only root's planned representative premerge baselines under identical fixture/import conditions: an unchanged executor target_harness case and a comparable app-factory case. For a new test absent from baseline, use an explicitly identified test-only overlay against exact old product sources and report the overlay. A matching source-selection failure classifies that representative fixture boundary, not all unobserved outcomes. Preserve any differing failure for investigation.
3. For merged backup integration, use the existing native_package/child infrastructure that selects the profile before imports: actual first-note Complete capture, saved/closed Library note handoff, and mounted Library Complete/resumed writes on a freshly built merged wheel. Run sequentially when the native slot is free. These validate the changed real ownership paths without discarding guards or adding arbitrary fixture exceptions.
4. Treat exact new upstream UI behavior as unverified on the merged product until either a separately reviewed narrow private-profile fixture conversion or a real fresh-profile journey exercises it. The existing upstream unit tests and source compatibility review are supporting evidence, not merged runtime acceptance. No need to rerun the entire failing batch unchanged now.

The smallest next action is baseline classification plus the existing installed native journeys root already scheduled. No product change, deadline adjustment, widespread opt-in, new framework or fixture weakening follows from these logs.
