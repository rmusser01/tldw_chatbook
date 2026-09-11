"""Critique #10: onboarding lands where the profile's content says it should.

task-32349: ``coerce_library_lifecycle(raw=None, is_new_profile=False)`` returns
EXPANDED so a returning, populated profile does not flash the compact starter
rail while the six-source evidence read is still in flight. That default is
right; what was wrong is that nothing ever took it back. An empty profile whose
``config.toml`` merely pre-dated its first Library visit therefore opened the
full nine-destination rail and offered "Back to Get started" -- a return to a
view it had never shown.

The screen can tell the two EXPANDEDs apart where
``aggregate_library_lifecycle`` cannot: a stored ``lifecycle = "expanded"`` is a
real Explore press, an absent one is this default. These pin all three cases,
plus the in-session one (Explore, then a second evidence read).

task-32351 AC#2 pins the landing's attention card naming the counts it is
asking about.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_content_evidence import (
    LibraryContentEvidence,
    LibraryEvidenceStatus,
)
from tldw_chatbook.Library.library_rail_state import LibraryLifecycle
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _LibraryEvidenceGates,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_evidence_round,
    _wait_for_library_shell,
)


def _pre_written_config_app(gates: _LibraryEvidenceGates, *, lifecycle=None):
    """An existing profile: not created this session, so no admission fact.

    ``lifecycle=None`` leaves ``[library.rail_state]`` without the key, which is
    the state every real config written before its first Library visit is in.
    """
    app = _build_test_app()
    app.library_new_profile_admission = False
    rail_state = app.app_config.setdefault("library", {}).setdefault("rail_state", {})
    if lifecycle is None:
        rail_state.pop("lifecycle", None)
    else:
        rail_state["lifecycle"] = lifecycle
    gates.install(app)
    return app


@pytest.mark.asyncio
async def test_a_pre_written_config_with_no_content_lands_on_get_started() -> None:
    gates = _LibraryEvidenceGates()
    app = _pre_written_config_app(gates)
    screen = LibraryScreen(app)
    host = LibraryHarness(app, screen=screen)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            await _wait_for_evidence_round(pilot, gates)
            assert screen._library_lifecycle is LibraryLifecycle.EXPANDED
            gates.release_round(0)
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_onboarding_status is LibraryEvidenceStatus.SETTLED
                ),
                message="empty evidence did not settle",
            )
            assert screen._library_lifecycle is LibraryLifecycle.STARTER
            assert not screen.query("#library-rail-back-to-starter")
    finally:
        gates.release_all()


@pytest.mark.asyncio
async def test_a_returning_user_with_content_never_sees_the_starter_rail() -> None:
    """The reason the EXPANDED default exists: no compact-rail flash."""
    gates = _LibraryEvidenceGates(
        outcomes={"media": [LibraryContentEvidence.HAS_USER_CONTENT]}
    )
    app = _pre_written_config_app(gates)
    screen = LibraryScreen(app)
    host = LibraryHarness(app, screen=screen)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            await _wait_for_evidence_round(pilot, gates)
            # The full rail is already on screen while the read is in flight.
            assert screen._library_lifecycle is LibraryLifecycle.EXPANDED
            gates.release_round(0)
            await _wait_for_condition(
                pilot,
                lambda: screen._library_lifecycle is LibraryLifecycle.GRADUATED,
                message="positive evidence did not graduate",
            )
    finally:
        gates.release_all()


@pytest.mark.asyncio
async def test_an_explicit_explore_keeps_the_expanded_rail_when_empty() -> None:
    gates = _LibraryEvidenceGates()
    app = _pre_written_config_app(gates, lifecycle="expanded")
    screen = LibraryScreen(app)
    host = LibraryHarness(app, screen=screen)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            await _wait_for_evidence_round(pilot, gates)
            gates.release_round(0)
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_onboarding_status is LibraryEvidenceStatus.SETTLED
                ),
                message="empty evidence did not settle",
            )
            assert screen._library_lifecycle is LibraryLifecycle.EXPANDED
            assert screen.query_one("#library-rail-back-to-starter")
    finally:
        gates.release_all()


@pytest.mark.asyncio
async def test_explore_survives_the_next_evidence_read_in_the_same_session() -> None:
    """The demotion reads storage as it is NOW, not as it was at construction.

    Pressing Explore mirrors ``lifecycle = "expanded"`` into the config the
    screen reads, so the second all-empty read leaves the choice alone. Against
    the construction-time snapshot instead, this read yanked the user back to
    Get started the first time they returned to the Library.
    """
    gates = _LibraryEvidenceGates(rounds=2)
    app = _pre_written_config_app(gates)
    screen = LibraryScreen(app)
    host = LibraryHarness(app, screen=screen)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            await _wait_for_evidence_round(pilot, gates, 0)
            gates.release_round(0)
            await _wait_for_condition(
                pilot,
                lambda: screen._library_lifecycle is LibraryLifecycle.STARTER,
                message="empty evidence did not land on Get started",
            )
            await pilot.click("#library-rail-explore-all")
            await _wait_for_condition(
                pilot,
                lambda: screen._library_lifecycle is LibraryLifecycle.EXPANDED,
                message="Explore did not expand the rail",
            )

            screen._refresh_library_onboarding_evidence()
            await _wait_for_evidence_round(pilot, gates, 1)
            gates.release_round(1)
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_onboarding_status is LibraryEvidenceStatus.SETTLED
                ),
                message="second empty evidence round did not settle",
            )
            assert screen._library_lifecycle is LibraryLifecycle.EXPANDED
    finally:
        gates.release_all()


@pytest.mark.asyncio
async def test_the_landing_card_counts_the_failures_it_is_asking_about() -> None:
    """task-32351 AC#2: the queue was exact, the landing card was not."""
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        for index in range(4):
            job = app.library_ingest_jobs.submit(
                source_path=f"/tmp/inbox/fail-{index}.pdf",
                batch_id="local-inbox",
            )
            app.library_ingest_jobs.mark_failed(
                job.job_id,
                error="private failure",
                permanent=False,
            )
        for index in range(2):
            job = app.library_ingest_jobs.submit(
                source_path=f"/tmp/inbox/skip-{index}.pdf",
                batch_id="local-inbox",
            )
            app.library_ingest_jobs.mark_skipped(job.job_id, reason="duplicate")

        action = screen._library_landing_attention_action()

        assert action is not None
        assert action.message == "Last import: 4 files failed, 2 skipped."
        assert action.action_label == "Review"
        assert action.action_kind == "ingest-review"


@pytest.mark.asyncio
async def test_a_single_failure_is_named_in_the_singular() -> None:
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        job = app.library_ingest_jobs.submit(source_path="/tmp/solo.pdf")
        app.library_ingest_jobs.mark_failed(
            job.job_id,
            error="private failure",
            permanent=False,
        )

        action = screen._library_landing_attention_action()

        assert action is not None
        assert action.message == "Last import: 1 file failed."
