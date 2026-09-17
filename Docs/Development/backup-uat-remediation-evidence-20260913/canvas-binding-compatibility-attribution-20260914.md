# Direct Canvas binding compatibility failure attribution

**All three ordinary pytest failures reproduce on exact pre-candidate d640 runtime source with the same original refusal.** Current:3failed/20deselected in3.26s; baseline:3failed/20deselected in3.05s. No repo edits, app boot, guard replacement or fixture change.

A private import overlay loads the byte-exact `git show d6406d4:.../console_runtime.py` module while retaining the original three test bodies and ordinary pytest fixtures. A bounded exception-only trace records the original exception at exact `_read_canvas_enabled` code identity and delegates all execution unchanged; it restores both current-thread and future-thread trace settings. Both runs record the same failure for each body:

`_read_canvas_enabled` → `config.get_canvas_execution_enabled:401` → `load_cli_config_and_ensure_existence:6261` → `config_participants.wrapped:419/operation:352` → `raw_participants._scope:611/_raw_participant:269/_participant_state:127` → **RecoveryRequired('raw_source_selection_changed')**.

At line127 the exact config raw participant rejects its current config binding being absent or differing from the selected source recorded by that participant. The existing Canvas reader converts this non-pause refusal to false; `_canvas_enabled` latches disable and direct binder returns before listener installation. This explains both empty listener assertions and why the third test's authority constructor is never entered. The new ownership query/UI accessor path is not used by these direct binder tests.

The ordinary root autouse fixture (`Tests/conftest.py:1020`, environment rewrite around1067) selects a new per-case config/home after collection unless a private-profile child was declared. These tests use imported global config/default reader and do not opt into the stable private profile. The observed config selection refusal is therefore concrete; the exact absent-versus-different binding branch was not further inspected and is unnecessary for baseline attribution. This is not evidence that Canvas configuration is disabled by the user's settings.

Affected cases: `test_canvas_view_binding_builds_authority_on_first_publication_and_rebinds`; `test_canvas_view_rebind_registers_listener_on_rebuilt_controller`; `test_concurrent_first_canvas_ensure_constructs_one_authority`.

No product fix or fixture guard weakening is justified by these three baseline failures. The parent's separate execution of the unchanged bodies in a fresh enrolled native fixture is the appropriate compatibility evidence; its result is not assumed here. The diagnostic trace has overhead, so the3-second totals are test outcomes, not performance measurements.

Evidence: `/private/tmp/uat-canvas-binding-compat-{current,baseline}.{json,log}`; exact hashes `/private/tmp/uat-canvas-binding-compatibility-attribution-hashes.json`; private observer `/private/tmp/uat_canvas_binding_attribution_plugin.py`.
