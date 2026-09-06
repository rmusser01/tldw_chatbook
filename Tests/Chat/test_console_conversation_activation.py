from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    LocalCharacterConversationTarget,
    ResolvedLocalCharacterKey,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_conversation_activation import (
    CharacterConversationActivationRequest,
    ConsoleActivationCommit,
    ConsoleActivationResultKind,
    ConsoleConversationActivationCoordinator,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController

TARGET = LocalCharacterConversationTarget(
    ResolvedLocalCharacterKey("authority-A", 7), "conversation-X"
)
OTHER_TARGET = LocalCharacterConversationTarget(
    ResolvedLocalCharacterKey("authority-A", 7), "conversation-Y"
)


def test_activation_request_binds_target_authority_and_query_revision() -> None:
    """The Console receives the immutable authority/revision selected by Roleplay."""

    request = CharacterConversationActivationRequest(
        target=TARGET,
        data_authority_id="authority-A",
        data_revision=12,
    )

    assert request.target == TARGET
    assert request.data_authority_id == "authority-A"
    assert request.data_revision == 12
    with pytest.raises(ValueError, match="authority"):
        CharacterConversationActivationRequest(
            target=TARGET,
            data_authority_id="authority-B",
            data_revision=12,
        )


@dataclass
class _ConsoleState:
    active: str = "prior"
    sessions: tuple[str, ...] = ("prior",)
    visible: bool = False
    focused: bool = False


class _Harness:
    def __init__(self) -> None:
        self.state = _ConsoleState()
        self.commit_gate = asyncio.Event()
        self.finish_gate = asyncio.Event()
        self.open_calls = 0
        self.revalidation: ConsoleActivationResultKind | None = None
        self.fail = False
        self.force_invisible = False
        self.rolled_back_tokens: list[object] = []
        self.open_token = object()

    def capture(self) -> _ConsoleState:
        return _ConsoleState(**vars(self.state))

    async def revalidate(self, target: LocalCharacterConversationTarget):
        assert target in {TARGET, OTHER_TARGET}
        await self.commit_gate.wait()
        return self.revalidation

    async def open(self, target: LocalCharacterConversationTarget) -> object:
        assert target in {TARGET, OTHER_TARGET}
        self.open_calls += 1
        self.state.active = "target"
        self.state.sessions += ("target",)
        await self.finish_gate.wait()
        if self.fail:
            return ConsoleActivationCommit(False, self.open_token)
        self.state.visible = True
        self.state.focused = True
        return ConsoleActivationCommit(True, self.open_token)

    async def restore(self, prior: _ConsoleState) -> None:
        self.state.active = prior.active
        self.state.visible = prior.visible
        self.state.focused = prior.focused

    async def rollback_opened_target(self, token: object) -> None:
        self.rolled_back_tokens.append(token)
        self.state.sessions = tuple(
            session for session in self.state.sessions if session != "target"
        )

    def exact_visible(self, target: LocalCharacterConversationTarget) -> bool:
        return (
            target == TARGET
            and self.state.active == "target"
            and self.state.visible
            and self.state.focused
            and not self.force_invisible
        )

    def coordinator(self) -> ConsoleConversationActivationCoordinator[_ConsoleState]:
        return ConsoleConversationActivationCoordinator(
            capture_state=self.capture,
            revalidate=self.revalidate,
            open_target=self.open,
            rollback_opened_target=self.rollback_opened_target,
            restore_state=self.restore,
            exact_target_visible=self.exact_visible,
        )


@pytest.mark.asyncio
async def test_cancel_before_commit_changes_no_console_state() -> None:
    harness = _Harness()
    coordinator = harness.coordinator()
    cancellation = asyncio.Event()
    task = asyncio.create_task(coordinator.activate(TARGET, cancellation))
    cancellation.set()
    harness.commit_gate.set()

    result = await task

    assert result.kind is ConsoleActivationResultKind.CANCELLED_PRECOMMIT
    assert not result.commit_started
    assert harness.state == _ConsoleState()
    assert harness.open_calls == 0


@pytest.mark.asyncio
async def test_escape_after_commit_started_is_ignored() -> None:
    harness = _Harness()
    coordinator = harness.coordinator()
    cancellation = asyncio.Event()
    task = asyncio.create_task(coordinator.activate(TARGET, cancellation))
    harness.commit_gate.set()
    await coordinator.wait_until_commit_started(TARGET)
    cancellation.set()
    harness.finish_gate.set()

    result = await task

    assert result.kind is ConsoleActivationResultKind.OPENED
    assert result.commit_started


@pytest.mark.asyncio
async def test_failed_postcommit_open_rolls_back_to_exact_prior_session() -> None:
    harness = _Harness()
    harness.fail = True
    coordinator = harness.coordinator()
    harness.commit_gate.set()
    harness.finish_gate.set()

    result = await coordinator.activate(TARGET)

    assert result.kind is ConsoleActivationResultKind.FAILED
    assert result.commit_started
    assert harness.state == _ConsoleState()
    assert harness.rolled_back_tokens == [harness.open_token]


@pytest.mark.asyncio
async def test_double_activate_shares_one_attempt_and_one_runtime_session() -> None:
    harness = _Harness()
    coordinator = harness.coordinator()
    first = asyncio.create_task(coordinator.activate(TARGET))
    second = asyncio.create_task(coordinator.activate(TARGET))
    harness.commit_gate.set()
    await coordinator.wait_until_commit_started(TARGET)
    harness.finish_gate.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result == second_result
    assert harness.open_calls == 1
    assert harness.state.sessions == ("prior", "target")


@pytest.mark.asyncio
async def test_success_requires_exact_target_current_and_visible() -> None:
    harness = _Harness()
    harness.commit_gate.set()
    harness.finish_gate.set()
    harness.force_invisible = True
    coordinator = harness.coordinator()

    result = await coordinator.activate(TARGET)

    assert result.kind is ConsoleActivationResultKind.FAILED
    assert harness.state == _ConsoleState()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        ConsoleActivationResultKind.DATA_PROFILE_CHANGED,
        ConsoleActivationResultKind.CHARACTER_UNAVAILABLE,
        ConsoleActivationResultKind.NOT_FOUND,
    ],
)
async def test_profile_or_character_change_never_substitutes_a_target(
    failure: ConsoleActivationResultKind,
) -> None:
    harness = _Harness()
    harness.revalidation = failure
    harness.commit_gate.set()
    coordinator = harness.coordinator()

    result = await coordinator.activate(TARGET)

    assert result.kind is failure
    assert not result.commit_started
    assert harness.open_calls == 0
    assert harness.state == _ConsoleState()


@pytest.mark.asyncio
async def test_different_targets_are_globally_serialized() -> None:
    """A second target cannot enter revalidation while the first mutates Console."""

    first_revalidated = asyncio.Event()
    release_first = asyncio.Event()
    second_revalidated = asyncio.Event()
    calls: list[str] = []

    async def revalidate(target: LocalCharacterConversationTarget):
        calls.append(f"validate:{target.conversation_id}")
        if target == TARGET:
            first_revalidated.set()
            await release_first.wait()
        else:
            second_revalidated.set()

    async def open_target(target: LocalCharacterConversationTarget):
        calls.append(f"open:{target.conversation_id}")
        return target.conversation_id

    coordinator = ConsoleConversationActivationCoordinator(
        capture_state=lambda: None,
        revalidate=revalidate,
        open_target=open_target,
        rollback_opened_target=lambda _token: None,
        restore_state=lambda _prior: None,
        exact_target_visible=lambda _target: True,
    )
    first = asyncio.create_task(coordinator.activate(TARGET))
    await first_revalidated.wait()
    second = asyncio.create_task(coordinator.activate(OTHER_TARGET))
    await asyncio.sleep(0)

    assert not second_revalidated.is_set()
    release_first.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.kind is ConsoleActivationResultKind.OPENED
    assert second_result.kind is ConsoleActivationResultKind.OPENED
    assert calls == [
        "validate:conversation-X",
        "open:conversation-X",
        "validate:conversation-Y",
        "open:conversation-Y",
    ]


@pytest.mark.asyncio
async def test_disposable_console_workspaces_share_one_runtime_mutation_lane() -> None:
    """Two screen-owned coordinators cannot interleave through one app runtime."""

    runtime_lock = asyncio.Lock()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first_open(_target):
        first_entered.set()
        await release_first.wait()
        return "runtime-X"

    async def second_open(_target):
        second_entered.set()
        return "runtime-Y"

    def coordinator(open_target):
        return ConsoleConversationActivationCoordinator(
            capture_state=lambda: None,
            revalidate=lambda _target: None,
            open_target=open_target,
            rollback_opened_target=lambda _token: None,
            restore_state=lambda _prior: None,
            exact_target_visible=lambda _target: True,
            mutation_lock=runtime_lock,
        )

    first = asyncio.create_task(coordinator(first_open).activate(TARGET))
    await first_entered.wait()
    second = asyncio.create_task(coordinator(second_open).activate(OTHER_TARGET))
    await asyncio.sleep(0)
    assert not second_entered.is_set()

    release_first.set()
    first_result, second_result = await asyncio.gather(first, second)
    assert first_result.kind is ConsoleActivationResultKind.OPENED
    assert second_result.kind is ConsoleActivationResultKind.OPENED
    assert second_entered.is_set()


@pytest.mark.asyncio
async def test_same_target_with_new_revision_waits_for_fresh_revalidation() -> None:
    """Only exact target+revision duplicates may join one activation attempt."""

    first_request = CharacterConversationActivationRequest(TARGET, "authority-A", 12)
    changed_request = CharacterConversationActivationRequest(TARGET, "authority-A", 13)
    first_gate = asyncio.Event()
    validated: list[int] = []

    async def revalidate(request):
        validated.append(request.data_revision)
        if request.data_revision == 12:
            await first_gate.wait()

    coordinator = ConsoleConversationActivationCoordinator(
        capture_state=lambda: None,
        revalidate=revalidate,
        open_target=lambda request: f"runtime-{request.data_revision}",
        rollback_opened_target=lambda _token: None,
        restore_state=lambda _prior: None,
        exact_target_visible=lambda _request: True,
    )
    first = asyncio.create_task(coordinator.activate(first_request))
    await asyncio.sleep(0)
    second = asyncio.create_task(coordinator.activate(changed_request))
    await asyncio.sleep(0)
    assert validated == [12]

    first_gate.set()
    await asyncio.gather(first, second)

    assert validated == [12, 13]


@pytest.mark.asyncio
async def test_production_workspace_reports_global_revision_staleness_as_failed(
    tmp_path,
) -> None:
    """An unrelated mutation requests refresh without blaming the valid card."""

    db = CharactersRAGDB(tmp_path / "activation.sqlite", client_id="activation")
    try:
        character_id = db.add_character_card({"name": "Ada"})
        authority = db.get_local_authority_id()
        assert character_id
        assert db.add_conversation(
            {
                "id": "conversation-X",
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "assistant_authority_id": authority,
                "title": "Before mutation",
            }
        )
        assert db.add_conversation(
            {
                "id": "conversation-Y",
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "assistant_authority_id": authority,
                "title": "Unrelated",
            }
        )
        captured_revision = db.get_character_conversation_search_revision()
        request = CharacterConversationActivationRequest(
            LocalCharacterConversationTarget(
                ResolvedLocalCharacterKey(authority, character_id),
                "conversation-X",
            ),
            authority,
            captured_revision,
        )
        record = db.get_conversation_by_id("conversation-Y")
        assert record
        assert db.update_conversation(
            "conversation-Y", {"title": "After mutation"}, record["version"]
        )
        controller = ConsoleWorkspaceController.__new__(ConsoleWorkspaceController)
        controller.app_instance = SimpleNamespace(chachanotes_db=db)

        result = await controller._revalidate_character_conversation_target(request)

        assert result is ConsoleActivationResultKind.FAILED

        fresh_request = CharacterConversationActivationRequest(
            request.target,
            authority,
            db.get_character_conversation_search_revision(),
        )
        assert fresh_request.data_revision != request.data_revision
        assert (
            await controller._revalidate_character_conversation_target(fresh_request)
            is None
        )
    finally:
        db.close_connection()


def test_production_workspace_visibility_requires_settled_transcript_owner() -> None:
    """Store selection and composer focus cannot mask a stale transcript widget."""

    store = ConsoleChatStore()
    session = store.restore_persisted_session(
        title="Target",
        workspace_id=None,
        persisted_conversation_id="conversation-X",
        all_nodes=[],
    )
    composer = SimpleNamespace(ancestors=())
    stale_transcript = SimpleNamespace(
        _session_identity="different-session", display=True
    )

    class _Screen:
        is_mounted = True
        focused = composer
        _last_native_transcript_refresh_key = (1, ())
        _last_native_transcript_session_id = "different-session"

        @property
        def app(self):
            return SimpleNamespace(screen=self)

        def query_one(self, selector):
            if selector == "#console-native-composer":
                return composer
            if selector == "#console-native-transcript":
                return stale_transcript
            raise AssertionError(selector)

    controller = ConsoleWorkspaceController.__new__(ConsoleWorkspaceController)
    controller._screen = _Screen()
    controller._chat_store_accessor = lambda: store
    request = CharacterConversationActivationRequest(TARGET, "authority-A", 12)
    assert session.persisted_conversation_id == "conversation-X"

    assert controller._character_conversation_target_visible(request) is False

    from textual.css.query import NoMatches

    def missing_widget(_selector):
        raise NoMatches("transcript not mounted")

    controller._screen.query_one = missing_widget
    assert controller._character_conversation_target_visible(request) is False

    def broken_widget(_selector):
        raise RuntimeError("unexpected renderer failure")

    controller._screen.query_one = broken_widget
    with pytest.raises(RuntimeError, match="unexpected renderer failure"):
        controller._character_conversation_target_visible(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("focus_failure", ["", "missing_composer", "broken_focus"])
async def test_production_workspace_predicate_failure_rolls_back_exact_owned_session(
    focus_failure,
) -> None:
    """Production open/capture/predicate/rollback compose around the real store."""

    store = ConsoleChatStore()
    prior = store.create_session(title="Prior")
    composer = object()
    transcript = SimpleNamespace(_session_identity="stale-session")
    screen = SimpleNamespace(is_mounted=True, focused=composer)
    screen.app = SimpleNamespace(screen=screen)
    screen._last_native_transcript_refresh_key = ("stale-session", 1)

    def query(selector):
        if selector == "#console-native-composer":
            if focus_failure == "missing_composer":
                from textual.css.query import NoMatches

                raise NoMatches("composer unmounted")
            return composer
        return transcript

    def focus(widget):
        if focus_failure == "broken_focus":
            raise RuntimeError("focus presentation failed")
        screen.focused = widget

    screen.query_one = query
    screen.set_focus = focus
    controller = ConsoleWorkspaceController.__new__(ConsoleWorkspaceController)
    controller._screen = screen
    controller._chat_store_accessor = lambda: store

    async def hydrate(_conversation_id):
        store.restore_persisted_session(
            title="Target",
            workspace_id=None,
            persisted_conversation_id="conversation-X",
            all_nodes=[],
        )
        return True

    async def restore(prior_session_id):
        store.switch_session(prior_session_id)

    async def revalidate(_request):
        return None

    controller.open_console_workspace_conversation = hydrate
    controller._revalidate_character_conversation_target = revalidate
    controller._restore_character_conversation_prior_session = restore
    request = CharacterConversationActivationRequest(TARGET, "authority-A", 1)

    result = await controller.activate_character_conversation_after_commit(request)

    assert result.kind is ConsoleActivationResultKind.FAILED
    assert store.active_session_id == prior.id
    assert all(
        session.persisted_conversation_id != "conversation-X"
        for session in store.sessions()
    )


@pytest.mark.asyncio
async def test_workspace_final_screen_transfer_failure_rolls_back_exact_owned_session() -> (
    None
):
    """A failed final content-slot transfer remains inside runtime rollback."""

    store = ConsoleChatStore()
    prior = store.create_session(title="Prior")
    controller = ConsoleWorkspaceController.__new__(ConsoleWorkspaceController)
    controller._chat_store_accessor = lambda: store

    async def open_target(_request):
        owned = store.restore_persisted_session(
            title="Target",
            workspace_id=None,
            persisted_conversation_id="conversation-X",
            all_nodes=[],
        )
        return ConsoleActivationCommit(True, owned)

    async def restore(prior_session_id):
        store.switch_session(prior_session_id)

    async def fail_transfer():
        raise RuntimeError("content transfer failed")

    controller._open_character_conversation_activation = open_target
    controller._revalidate_character_conversation_target = AsyncMock(return_value=None)
    controller._restore_character_conversation_prior_session = restore
    controller._character_conversation_target_visible = lambda _request: True
    request = CharacterConversationActivationRequest(TARGET, "authority-A", 1)

    result = await controller.activate_character_conversation_after_commit(
        request, finalize_visible=fail_transfer
    )

    assert result.kind is ConsoleActivationResultKind.FAILED
    assert store.active_session_id == prior.id
    assert all(
        session.persisted_conversation_id != "conversation-X"
        for session in store.sessions()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    (
        "UPDATE conversations SET deleted = 1 WHERE id = 'conversation-X'",
        (
            "UPDATE conversations SET runtime_backend = 'server' "
            "WHERE id = 'conversation-X'"
        ),
        "UPDATE conversations SET character_id = 1 WHERE id = 'conversation-X'",
    ),
)
async def test_committed_workspace_revalidates_after_hydration_and_rolls_back_owned(
    tmp_path, mutation
) -> None:
    """Deletion, source, or link mutation after load cannot survive or open."""

    db = CharactersRAGDB(tmp_path / "post-hydration.sqlite", client_id="activation")
    try:
        character_id = db.add_character_card({"name": "Ada"})
        authority = db.get_local_authority_id()
        db.add_conversation(
            {
                "id": "conversation-X",
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "assistant_authority_id": authority,
                "title": "Target",
            }
        )
        revision = db.get_character_conversation_search_revision()
        request = CharacterConversationActivationRequest(
            LocalCharacterConversationTarget(
                ResolvedLocalCharacterKey(authority, character_id),
                "conversation-X",
            ),
            authority,
            revision,
        )
        store = ConsoleChatStore()
        prior = store.create_session(title="Prior")
        controller = ConsoleWorkspaceController.__new__(ConsoleWorkspaceController)
        controller.app_instance = SimpleNamespace(chachanotes_db=db)
        controller._chat_store_accessor = lambda: store

        async def open_then_mutate(_request):
            owned = store.restore_persisted_session(
                title="Target",
                workspace_id=None,
                persisted_conversation_id="conversation-X",
                all_nodes=[],
            )
            with db.transaction(immediate=True) as connection:
                connection.execute(mutation)
            return ConsoleActivationCommit(True, owned)

        async def restore(prior_session_id):
            store.switch_session(prior_session_id)

        controller._open_character_conversation_activation = open_then_mutate
        controller._restore_character_conversation_prior_session = restore
        controller._character_conversation_target_visible = lambda _request: True

        result = await controller.activate_character_conversation_after_commit(request)

        assert result.kind in {
            ConsoleActivationResultKind.NOT_FOUND,
            ConsoleActivationResultKind.CHARACTER_UNAVAILABLE,
        }
        assert store.active_session_id == prior.id
        assert all(
            session.persisted_conversation_id != "conversation-X"
            for session in store.sessions()
        )
    finally:
        db.close_connection()
