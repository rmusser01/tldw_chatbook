# Model Catalog Consent Startup Order Design

**Task:** TASK-21161
**ADR required:** no
**ADR path:** `backlog/decisions/020-automatic-model-catalog-refresh.md`
**Reason:** This fixes UI lifecycle ordering while preserving ADR-020's existing consent, persistence, and network boundary.

## Problem

Startup currently schedules the model-catalog consent decision from `TldwCli.on_mount`, while the splash widget can still be active. The scheduled modal may become topmost before the splash finishes. When splash close later pushes the initial app screen, that screen covers the still-mounted modal. This creates three user-visible failures: the intro appears skipped, the prompt appears inconsistently or briefly, and the prompt can become impossible to answer.

## Required sequence

For a normal interactive launch with unrecorded model-catalog consent:

1. The configured splash/intro completes.
2. The initial application screen is mounted.
3. The model-catalog consent modal is pushed topmost and requires Yes or No.
4. Either answer dismisses the modal and leaves the initial application screen usable.

When splash is disabled, steps 1 and 2 collapse to the existing no-splash initial-screen path; consent still appears only after that screen is mounted.

## Design

Move the existing initial startup call to `_schedule_startup_model_catalog_refresh()` from `on_mount` to the end of `_push_initial_screen()`. That method is already the shared choke point reached by both splash-enabled and no-splash launches, after the initial screen has been pushed.

Keep all existing gates unchanged:

- `setup_owns_startup_networking(...)` continues to defer the decision while first-run setup owns startup networking.
- `_startup_model_catalog_refresh_scheduled` continues to enforce one scheduling attempt.
- `_handle_first_run_wizard_result(...)` continues to schedule after completed setup.
- `_refresh_model_catalogs()` continues to enforce recorded consent at the network boundary.
- The consent modal and its persistence behavior remain unchanged.

Two existing startup competitors receive explicit precedence:

- When first-run setup completes with an exit route, route navigation finishes before the deferred catalog decision is scheduled. Navigation must never dismiss the consent modal with `None`, because only an explicit Yes or No may resolve this required decision.
- When the catalog decision is still unrecorded, optional project-skills discovery is not started on that launch. Its prompt may be offered on the next launch after consent has been recorded; it may not cover consent or appear immediately after consent instead of the usable initial screen.

Track only whether this launch required catalog consent so `_push_initial_screen()` can suppress the optional project-skills offer. No general modal queue, startup coordinator, or timing delay is introduced.

## Error handling

Existing error handling remains authoritative. A settings-load failure leaves startup unscheduled and logs a bounded error; a consent persistence failure preserves the existing warning and session behavior. This correction adds no new failure path.

## Test strategy

Add one splash-enabled full-app Textual Pilot regression. Override only the production headless suppression so the real consent modal can be exercised under `run_test()`; keep real scheduling and screen-stack behavior.

The completed-setup launch test records screen-stack transitions and requires:

- the initial splash/default screen is observed before consent;
- the initial app screen is directly below the topmost consent modal;
- consent is never present below another pushed app screen;
- pressing a consent action dismisses to the same usable initial screen.

The test must fail against the pre-fix call site by observing the modal below the initial screen, then pass after the lifecycle move.

Add focused regressions for the competing startup paths:

- an eligible project-skills offer is not started on an unconsented launch, so consent dismisses directly to the initial screen;
- completing first-run setup with a valid exit route settles on that destination before consent appears, and the callback receives no value until the user explicitly selects Yes or No.
