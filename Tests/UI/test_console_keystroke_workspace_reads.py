"""The Console keystroke path must not touch the Workspace DB.

TASK-21118. The holistic perf review (Docs/Design/2026-08-22-holistic-perf-review.md,
finding 21118) measured, live on dev: 20 printable keys in the configured
composer -> 25 x ``ensure_default_workspace`` + 25 x ``get_active_workspace``
(LocalWorkspaceRegistryService), ~1.25 synchronous SQLite round-trips per
keystroke on the UI thread, via DraftChanged ->
``_sync_console_workbench_actions_from_draft`` ->
``_build_console_control_state`` -> provider selection ->
``ConsoleWorkspaceController._current_console_workspace_context``.
``ensure_default_workspace`` additionally carries repair side-effects (a
probing SELECT plus a DELETE write-transaction when Default carries stale
runtime bindings), so the keystroke path was not even read-only.

These tests replicate the review's counter probe as a permanent gate:

* zero registry-service reads AND zero WorkspaceDB round-trips across 20
  printable keys on the mounted, configured Console;
* a control proving the counters still see real registry traffic (so "zero"
  can never pass against an unwired counter);
* the staleness guard: a cross-screen workspace switch mid-session (a
  registry write that does NOT go through any Console seam) must be
  reflected by the very next Console context read;
* the staged-launch evidence bundle is parsed once per launch
  (``EvidenceBundle.from_payload`` used to run >=2x per keystroke while a
  launch was staged) and re-parsed when the launch payload changes.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Chat.citation_evidence_models import EvidenceBundle
from tldw_chatbook.Chat.console_display_state import evidence_bundle_from_launch
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

APP_SIZE = (140, 42)

#: The review probe drove 20 printable keys; keep the same shape.
TWENTY_KEYS = tuple("abcdefghijklmnopqrst")


async def _console(host, pilot):
    """Mount the ready Console and focus its composer."""
    console = await _mounted_console(host, pilot)
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    await pilot.pause()
    return console, composer


class _WorkspaceReadCounter:
    """Count registry-service reads and WorkspaceDB round-trips.

    Patches at the class so every instance-bound call the app makes is
    counted, and counts BOTH layers: the two service methods the review
    probe counted, and the underlying ``WorkspaceDB.connection`` /
    ``WorkspaceDB.transaction`` context factories -- the latter is what
    makes "zero DB round-trips" a claim about the database rather than
    about two method names. Restored on exit.
    """

    def __init__(self) -> None:
        self.ensure_calls = 0
        self.get_active_calls = 0
        self.db_connections = 0
        self.db_transactions = 0
        self._original_ensure = LocalWorkspaceRegistryService.ensure_default_workspace
        self._original_get_active = LocalWorkspaceRegistryService.get_active_workspace
        self._original_connection = WorkspaceDB.connection
        self._original_transaction = WorkspaceDB.transaction

    def __enter__(self) -> "_WorkspaceReadCounter":
        counter = self
        original_ensure = self._original_ensure
        original_get_active = self._original_get_active
        original_connection = self._original_connection
        original_transaction = self._original_transaction

        def counting_ensure(service):
            counter.ensure_calls += 1
            return original_ensure(service)

        def counting_get_active(service):
            counter.get_active_calls += 1
            return original_get_active(service)

        def counting_connection(db):
            counter.db_connections += 1
            return original_connection(db)

        def counting_transaction(db):
            counter.db_transactions += 1
            return original_transaction(db)

        LocalWorkspaceRegistryService.ensure_default_workspace = counting_ensure
        LocalWorkspaceRegistryService.get_active_workspace = counting_get_active
        WorkspaceDB.connection = counting_connection
        WorkspaceDB.transaction = counting_transaction
        return self

    def __exit__(self, *_exc) -> None:
        LocalWorkspaceRegistryService.ensure_default_workspace = self._original_ensure
        LocalWorkspaceRegistryService.get_active_workspace = self._original_get_active
        WorkspaceDB.connection = self._original_connection
        WorkspaceDB.transaction = self._original_transaction

    @property
    def registry_reads(self) -> int:
        return self.ensure_calls + self.get_active_calls

    @property
    def db_round_trips(self) -> int:
        return self.db_connections + self.db_transactions


# ---------------------------------------------------------------------------
# 1. Twenty printable keys: zero Workspace-DB work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_twenty_printable_keystrokes_do_zero_workspace_db_reads():
    """The review's live probe, as a gate: 20 keys -> 0 registry reads.

    On the pre-fix code this counts ~25 ``ensure_default_workspace`` +
    ~25 ``get_active_workspace`` (and their SQLite round-trips) for the
    same 20 keys.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        _console_screen, composer = await _console(host, pilot)
        # Let the mount's own sync passes settle (they may legitimately
        # read the registry once to warm the memoized context).
        for _ in range(3):
            await pilot.pause()

        with _WorkspaceReadCounter() as counter:
            for key in TWENTY_KEYS:
                await pilot.press(key)
            for _ in range(3):
                await pilot.pause()

        assert composer.draft_text() == "".join(TWENTY_KEYS)
        assert counter.ensure_calls == 0
        assert counter.get_active_calls == 0
        assert counter.db_round_trips == 0


@pytest.mark.asyncio
async def test_the_counter_still_sees_real_registry_traffic():
    """Control: "zero" above must not be satisfiable by an unwired counter.

    Drives the same class-level counters with a real registry read AND a
    real registry write on the app's own service instance -- the exact
    call shapes the keystroke path used to make.
    """
    app, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        await _console(host, pilot)
        registry = app.workspace_registry_service
        assert registry is not None

        with _WorkspaceReadCounter() as counter:
            active = registry.get_active_workspace()
            registry.ensure_default_workspace()

        assert active is not None
        assert counter.get_active_calls >= 1
        assert counter.ensure_calls == 1
        assert counter.db_round_trips >= 2


# ---------------------------------------------------------------------------
# 2. The memo must not go stale
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_screen_workspace_switch_reflects_in_next_context_read():
    """A mid-session workspace switch must reach the next keystroke's state.

    The switch is performed directly on the registry service -- the way
    Settings' "Set active" or Library's create-modal do it, with no Console
    seam involved -- so only the memo's own invalidation can keep the
    Console context truthful.
    """
    app, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot)
        registry = app.workspace_registry_service
        assert registry is not None

        before = console._workspace._current_console_workspace_context()

        registry.create_workspace(workspace_id="workspace-beta", name="Beta")
        registry.set_active_workspace("workspace-beta")

        await pilot.press("a")
        for _ in range(3):
            await pilot.pause()

        after = console._workspace._current_console_workspace_context()
        assert composer.draft_text() == "a"
        assert before.active_workspace_id != "workspace-beta"
        assert after.active_workspace_id == "workspace-beta"


# ---------------------------------------------------------------------------
# 3. Staged-launch evidence bundle: parsed once per launch
# ---------------------------------------------------------------------------


def _reference_payload(evidence_id: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "source_id": f"source-{evidence_id}",
        "source_type": "media",
        "title": f"Title {evidence_id}",
        "snippet": "snippet text",
        "authority_label": "Local library",
        "status": "available",
    }


def _bundle_payload(bundle_id: str = "bundle-1", reference_count: int = 2) -> dict:
    return {
        "bundle_id": bundle_id,
        "query": "workspace reads",
        "references": [
            _reference_payload(f"ev-{bundle_id}-{index}")
            for index in range(reference_count)
        ],
    }


class _FromPayloadCounter:
    """Count ``EvidenceBundle.from_payload`` parses; restored on exit."""

    def __init__(self) -> None:
        self.calls = 0
        self._original = EvidenceBundle.__dict__["from_payload"].__func__

    def __enter__(self) -> "_FromPayloadCounter":
        counter = self
        original = self._original

        def counting(cls, payload):
            counter.calls += 1
            return original(cls, payload)

        EvidenceBundle.from_payload = classmethod(counting)
        return self

    def __exit__(self, *_exc) -> None:
        EvidenceBundle.from_payload = classmethod(self._original)


def test_evidence_bundle_is_parsed_once_per_launch():
    """Repeated reads of one staged launch must reuse one parse."""
    launch = ConsoleLiveWorkLaunch.from_values(
        source="library_rag",
        title="Staged evidence",
        payload={"evidence_bundle": _bundle_payload()},
    )

    with _FromPayloadCounter() as counter:
        first = evidence_bundle_from_launch(launch)
        second = evidence_bundle_from_launch(launch)
        third = evidence_bundle_from_launch(launch)

    assert first is not None
    assert second is first
    assert third is first
    assert len(first.references) == 2
    assert counter.calls == 1


def test_evidence_bundle_cache_invalidates_when_the_payload_changes():
    """A replaced evidence payload must be re-parsed, not served stale."""
    launch = ConsoleLiveWorkLaunch.from_values(
        source="library_rag",
        title="Staged evidence",
        payload={"evidence_bundle": _bundle_payload("bundle-1", 2)},
    )

    with _FromPayloadCounter() as counter:
        first = evidence_bundle_from_launch(launch)
        launch.payload["evidence_bundle"] = _bundle_payload("bundle-2", 3)
        second = evidence_bundle_from_launch(launch)
        again = evidence_bundle_from_launch(launch)

    assert first is not None and second is not None
    assert first.bundle_id == "bundle-1"
    assert second.bundle_id == "bundle-2"
    assert len(second.references) == 3
    assert again is second
    assert counter.calls == 2


def test_evidence_bundle_instance_payload_needs_no_parse():
    """A payload already carrying an ``EvidenceBundle`` is returned as-is."""
    bundle = EvidenceBundle.from_payload(_bundle_payload())
    launch = ConsoleLiveWorkLaunch(
        source="library_rag",
        title="Staged evidence",
        payload={"evidence_bundle": bundle},
    )

    with _FromPayloadCounter() as counter:
        resolved = evidence_bundle_from_launch(launch)

    assert resolved is bundle
    assert counter.calls == 0


def test_evidence_bundle_failed_parse_is_not_retried_per_read():
    """A malformed payload is diagnosed once, not once per keystroke."""
    launch = ConsoleLiveWorkLaunch.from_values(
        source="library_rag",
        title="Staged evidence",
        payload={"evidence_bundle": {"bundle_id": "", "query": "broken"}},
    )

    with _FromPayloadCounter() as counter:
        first = evidence_bundle_from_launch(launch)
        second = evidence_bundle_from_launch(launch)

    assert first is None
    assert second is None
    assert counter.calls == 1


@pytest.mark.asyncio
async def test_staged_launch_keystrokes_do_not_reparse_the_evidence_bundle():
    """While a launch is staged, keystrokes must not re-run the parse.

    Pre-fix, every keystroke's control-state build parsed the staged
    bundle at least twice (``console_staged_source_count`` + the workspace
    context's staged-sources leg).
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer = await _console(host, pilot)
        console._pending_console_launch_context = ConsoleLiveWorkLaunch.from_values(
            source="library_rag",
            title="Staged evidence",
            payload={"evidence_bundle": _bundle_payload()},
        )

        with _FromPayloadCounter() as counter:
            for key in ("a", "b", "c"):
                await pilot.press(key)
            for _ in range(3):
                await pilot.pause()

        assert composer.draft_text() == "abc"
        # The launch was staged after the mount's sync passes, so the FIRST
        # read inside this window may parse it once; keystrokes beyond that
        # must all reuse it.
        assert counter.calls <= 1
