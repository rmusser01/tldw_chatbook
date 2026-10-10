"""One Send attempt shares one full hook consent read (ADR-225 decision 3).

Before this, a warm Send read hook consent in full five times -- received-
intent preparation, submission admission, v2 hook preparation, the legacy
UserPromptSubmit target selection and the final pre-dispatch admission --
each taking the config write lock, a fresh config read, the store lock and raw
admission. The attempt's first read is now passed explicitly to its later
pre-commit consumers. These controls pin what must still hold:

* exactly one full read before durable commit, plus one fresh final read;
* a grant revoked behind the owner's back (another process) after the shared
  read is still refused at the fresh final admission, before the provider;
* an in-process consent change, or a read bound to another session or owner,
  makes the consumer fall back to its fresh read;
* the shared read only ever answers "nothing to do". Anything it would have
  to select, build or refuse -- a UserPromptSubmit hook, a v2 handler, a
  turn carrying plugin-owned skills, a session with a retained hook engine,
  lifecycle or configured signature -- is decided by a fresh read, so a hook
  another process disabled after the shared read is omitted exactly as a
  fresh selection omits it, instead of being selected stale and then refused
  (blocking the Send) at its fresh launch guard.

Reads are counted at ``HookPermissions._current`` -- the one body every full
read (snapshot, v2 configuration, targets, launch guard) goes through.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Agents.test_hook_permissions import _approve, _edit, hook_file as _hook_file
from Tests.Chat.test_console_first_send_atomicity import _CheckpointObservingGateway
from Tests.Chat.test_console_received_intent_custody import _intent
from tldw_chatbook.Agents.hook_permissions import HookAuthorityRead, HookPermissions
from tldw_chatbook.Agents.run_hooks import RunHooksEngine, load_hooks_config
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_hook_preparation import ConsoleHookAttemptRead
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
from tldw_chatbook.Chat.prompt_history import PromptHistory
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
]
hook_file = _hook_file

_REVIEW_REQUIRED = "Review enabled hooks before sending."

#: Hand edits (another process) that narrow consent after the shared read.
_NARROWING_EDITS = {
    "row_disabled": lambda section: section["hook"][0].update(enabled=False),
    "master_disabled": lambda section: section.update(enabled=False),
    "removed": lambda section: section.update(hook=[]),
}

#: A v2 handler that needs no process or platform support to be configured.
_V2_HANDLER = {
    "id": "l1-v2",
    "event": "SessionStart",
    "type": "mcp_tool",
    "server": "local:fixture",
    "tool": "fixture",
    "effects": [],
}


def _as_user_prompt_hook(section):
    section["hook"][0]["event"] = "UserPromptSubmit"


def _with_v2_handler(section):
    section["handler"] = [dict(_V2_HANDLER)]


class _Rig:
    """A persisted Console with the production runtime-owned hook wiring."""

    def __init__(self, tmp_path, monkeypatch, owned_console_databases):
        self.db = CharactersRAGDB(tmp_path / "controller.sqlite", client_id="l1-hooks")
        self.store = ConsoleChatStore(persistence=ChatPersistenceService(self.db))
        self.session = self.store.create_session(session_id="session-1", title="Chat 1")
        self.gateway = _CheckpointObservingGateway(self.db)
        self.runtime = ConsoleRuntime(app=None)
        self.controller = ConsoleChatController(
            store=self.store,
            provider_gateway=self.gateway,
            provider="llama_cpp",
            model="test-model",
            hook_permissions_accessor=self.runtime.ensure_hook_permissions,
            ensure_run_hooks=self.runtime.ensure_run_hooks,
        )
        self.controller.prompt_history = PromptHistory(tmp_path / "history.jsonl")
        owned_console_databases(self.db, self.controller)
        self.runtime.set_chat_store(self.store)
        self.runtime.set_chat_controller(self.controller)
        # Warm: the owner exists and the configured hook is approved.
        self.owner = self.runtime.ensure_hook_permissions()
        assert _approve(self.owner).ready
        self.configuration = self.controller.resolve_turn_configuration_snapshot(
            self.session.id
        )
        #: Runs between the received read and submission admission.
        self.between = None
        rig = self

        async def captured_configuration(_session_id, *, selection=None):
            # The received path's capture is isolated here, as in the saved-
            # acceptance controls; it also marks "after the first read".
            if rig.between is not None:
                rig.between()
            return rig.configuration

        monkeypatch.setattr(
            self.controller,
            "capture_turn_configuration_snapshot",
            captured_configuration,
        )
        self.record = None
        self.phase = "send"
        self.reads: list[tuple[str, bool]] = []
        real = self.owner._current

        def counted(*args, **kwargs):
            record = rig.record
            durable = bool(record is not None and record.inputs.durable_accepted)
            rig.reads.append((rig.phase, durable))
            return real(*args, **kwargs)

        monkeypatch.setattr(self.owner, "_current", counted)

    def send_reads(self) -> list[bool]:
        """Durable-acceptance state at each full read the Send itself made."""
        return [durable for phase, durable in self.reads if phase == "send"]

    def key(self) -> str:
        snapshot = self.owner.snapshot()
        return next(row.entry.key for row in snapshot.rows if row.entry)

    def start_received(self, turn_id="received-l1"):
        self.store.set_session_draft(self.session.id, "share one hook read")
        request = ConsoleTurnCustodyRequest(
            turn_id=turn_id,
            session_id=self.session.id,
            draft="share one hook read",
            configuration=self.configuration,
        )
        intent = _intent(
            (self.runtime, self.store, self.session, request), turn_id=turn_id
        )
        self.reads.clear()
        self.runtime.accept_received_intent(intent)
        self.record = self.runtime._turn_custody[turn_id]
        return self.record

    def start_custodied(self, turn_id="custodied-l1"):
        request = ConsoleTurnCustodyRequest(
            turn_id=turn_id,
            session_id=self.session.id,
            draft="share one hook read",
            configuration=self.configuration,
        )
        self.reads.clear()
        self.runtime.accept_turn(request)
        self.record = self.runtime._turn_custody[turn_id]
        return self.record


@pytest.fixture
async def rig(hook_file, tmp_path, monkeypatch, owned_console_databases):
    state = _Rig(tmp_path, monkeypatch, owned_console_databases)
    try:
        yield state
    finally:
        await state.runtime.dispose()


@pytest.fixture
async def ups_rig(hook_file, tmp_path, monkeypatch, owned_console_databases):
    """The rig with its approved hook firing on UserPromptSubmit (blocking)."""
    _edit(hook_file, _as_user_prompt_hook)
    state = _Rig(tmp_path, monkeypatch, owned_console_databases)
    try:
        yield state
    finally:
        await state.runtime.dispose()


@pytest.fixture
async def v2_rig(hook_file, tmp_path, monkeypatch, owned_console_databases):
    """The rig with an approved v2 handler beside its legacy hook."""
    _edit(hook_file, _with_v2_handler)
    state = _Rig(tmp_path, monkeypatch, owned_console_databases)
    try:
        yield state
    finally:
        await state.runtime.close_hooks_v2()
        await state.runtime.dispose()


async def _outcome(record):
    (outcome,) = await asyncio.gather(
        asyncio.wait_for(asyncio.shield(record.task), 20), return_exceptions=True
    )
    return outcome


@pytest.mark.parametrize("entry", ["received", "custodied"])
async def test_warm_send_reads_once_before_commit_and_fresh_at_dispatch(rig, entry):
    """Received Send and runtime custody (queue/wake/legacy entry) alike."""
    record = rig.start_received() if entry == "received" else rig.start_custodied()

    result = await _outcome(record)

    assert not isinstance(result, BaseException), result
    assert result.accepted and rig.gateway.calls == 1
    # Before: received + admission + v2 + UserPromptSubmit + final = five
    # (custody: four). Now the attempt's one read, then the fresh final one.
    assert rig.send_reads() == [False, True], rig.reads


async def test_externally_revoked_grant_is_still_refused_at_final_admission(rig):
    """Another process revokes after the shared read; dispatch is refused."""
    key = rig.key()

    def revoke_elsewhere():
        # A second owner is another process's consent owner: it writes the
        # store through the canonical writer, unseen by this owner's memory.
        rig.phase = "elsewhere"
        other = HookPermissions()
        try:
            assert not other.revoke(other.snapshot(), key).ready
        finally:
            other.close()
            rig.phase = "send"

    rig.between = revoke_elsewhere
    record = rig.start_received()

    result = await _outcome(record)

    assert not isinstance(result, BaseException), result
    assert rig.gateway.calls == 0, "a revoked grant reached the provider"
    assert result.visible_copy == "Accepted turn is retained for recovery."
    # The shared read stood for the pre-commit consumers; the final admission
    # read fresh, saw the revocation and refused before provider entry.
    assert rig.send_reads() == [False, True], rig.reads
    assert not rig.owner.snapshot().ready


async def test_in_process_consent_change_falls_back_to_a_fresh_admission(rig):
    """A review action in this app moves the owner; admission reads fresh."""
    key = rig.key()

    def revoke_here():
        rig.phase = "review"
        try:
            assert not rig.owner.revoke(rig.owner.snapshot(), key).ready
        finally:
            rig.phase = "send"

    rig.between = revoke_here
    record = rig.start_received()

    result = await _outcome(record)

    assert isinstance(result, Exception), result
    assert rig.gateway.calls == 0
    assert rig.store.messages_for_session(rig.session.id) == []
    # The received read, then admission's own fresh read, which refused
    # before durable commit exactly as it did before sharing.
    assert rig.send_reads() == [False, False], rig.reads
    recovery = rig.runtime.recoveries_for_session(rig.session.id)
    assert recovery and _REVIEW_REQUIRED in recovery[-1].reason


async def test_foreign_session_read_falls_back_and_the_fresh_read_is_shared(rig):
    """A read bound to another session is never used for this attempt."""
    foreign = ConsoleHookAttemptRead("another-session", rig.owner.authority_read())
    accepted = []
    rig.reads.clear()

    result = await rig.controller.submit_draft(
        "share one hook read",
        session_id=rig.session.id,
        configuration=rig.configuration,
        custody_acceptance_hook=lambda: accepted.append(True),
        _hook_read=foreign,
    )

    assert result.accepted and rig.gateway.calls == 1
    # Admission read fresh (the foreign read was refused) and that one read
    # served v2 preparation and UserPromptSubmit; the final read is fresh.
    assert len(rig.send_reads()) == 2, rig.reads
    assert accepted == [True]


async def test_v2_preparation_uses_only_a_matching_current_read(rig):
    runtime, owner, session_id = rig.runtime, rig.owner, rig.session.id
    shared = ConsoleHookAttemptRead(session_id, owner.authority_read())
    other = HookPermissions()
    try:
        cases = {
            "matching": shared,
            "foreign_session": replace(shared, session_id="another-session"),
            "foreign_owner": ConsoleHookAttemptRead(session_id, other.authority_read()),
        }
        counts = {}
        for name, hook_read in cases.items():
            rig.reads.clear()
            assert (
                await runtime.prepare_hooks_v2(session_id, _hook_read=hook_read) is None
            )
            counts[name] = len(rig.reads)
    finally:
        other.close()
    # Only the matching read replaces the v2_configuration() re-read.
    assert counts == {"matching": 0, "foreign_session": 1, "foreign_owner": 1}


async def test_legacy_selection_uses_only_its_own_owners_current_read(rig):
    engine = rig.runtime.ensure_run_hooks()
    assert type(engine) is RunHooksEngine
    read = rig.owner.authority_read()
    rig.reads.clear()
    assert not engine.fire(
        "UserPromptSubmit", session_id="s", authority_read=read
    ).blocked
    assert rig.reads == []
    other = HookPermissions()
    try:
        foreign = other.authority_read()
        assert not engine.fire(
            "UserPromptSubmit", session_id="s", authority_read=foreign
        ).blocked
        assert len(rig.reads) == 1, "a foreign owner's read replaced a fresh one"
    finally:
        other.close()


def _shared_read(owner: HookPermissions) -> HookAuthorityRead:
    read = owner.authority_read()
    assert owner.attempt_read_current(read)
    return read


@pytest.mark.parametrize(
    "change",
    [
        "approve_in_process",
        "refresh_pending",
        "sealed",
        "closed",
        "config_identity",
        "unpublished_targets",
        "foreign_owner",
    ],
)
def test_a_read_stops_standing_on_any_in_memory_change(hook_file, monkeypatch, change):
    from tldw_chatbook.Agents import hook_permissions

    owner = HookPermissions()
    try:
        assert _approve(owner).ready
        read = _shared_read(owner)
        scope = (str(read.snapshot.store_path), str(read.snapshot.config.config_path))
        if change == "approve_in_process":
            # Any consent write publishes a new store revision.
            key = next(row.entry.key for row in read.snapshot.rows if row.entry)
            owner.revoke(owner.snapshot(), key)
        elif change == "refresh_pending":
            owner._refresh_pending.add(scope)
        elif change == "sealed":
            owner._seal(read.snapshot, ["any-definition"])
        elif change == "closed":
            owner.close()
        elif change == "config_identity":
            identity = read.config_identity
            monkeypatch.setattr(
                hook_permissions.config,
                "current_config_identity",
                lambda: (identity[0] + 1, identity[1]),
            )
        elif change == "unpublished_targets":
            read = replace(read, targets=None)
        else:
            other = HookPermissions()
            try:
                assert not other.attempt_read_current(read)
            finally:
                other.close()
            return
        assert not owner.attempt_read_current(read)
        assert owner.attempt_v2_configuration(read) is None
        assert owner.attempt_targets(read, "UserPromptSubmit", None) is None
    finally:
        owner.close()


def test_a_blocked_read_is_never_shared(hook_file):
    owner = HookPermissions()
    try:
        read = owner.authority_read()
        assert not read.snapshot.ready, "fixture hook starts unapproved"
        assert not owner.attempt_read_current(read)
    finally:
        owner.close()


def test_a_shared_read_answers_only_an_empty_selection(hook_file):
    """A selection taken from an earlier read could be stale; only "none" is.

    A target selected from the shared read would still reach its fresh launch
    guard after another process disabled or revoked it, and that refusal
    blocks a blocking event (UserPromptSubmit) where a fresh selection would
    simply have omitted the hook. So a non-empty selection is always left to
    the fresh read.
    """
    owner = HookPermissions()
    try:
        assert _approve(owner).ready
        read = _shared_read(owner)
        # The fixture hook fires on PostToolUse only.
        assert owner.attempt_targets(read, "UserPromptSubmit", None) == ()
        assert owner.attempt_targets(read, "PostToolUse", None) is None
        (target,) = owner.targets("PostToolUse", None)
        other = HookPermissions()
        try:
            other.revoke(other.snapshot(), target.key)
        finally:
            other.close()
        # The in-memory read still stands (the change was elsewhere), but it
        # never supplies the now-stale target: the fresh selection omits it.
        assert owner.attempt_read_current(read)
        assert owner.attempt_targets(read, "PostToolUse", None) is None
        assert owner.targets("PostToolUse", None) == ()
    finally:
        owner.close()


def test_a_refusing_shared_selection_is_left_to_the_fresh_read(hook_file, monkeypatch):
    """An earlier read never refuses either; the fresh read decides."""
    owner = HookPermissions()
    try:
        assert _approve(owner).ready
        read = _shared_read(owner)
        monkeypatch.setattr(owner, "_guard_reason", lambda *_args: "stale refusal")
        assert owner.attempt_targets(read, "UserPromptSubmit", None) is None
    finally:
        owner.close()


@pytest.mark.parametrize("edit", ["unchanged", *_NARROWING_EDITS])
def test_user_prompt_selection_with_a_shared_read_matches_a_fresh_one(
    hook_file, tmp_path, monkeypatch, edit
):
    """Engine level: a selected UserPromptSubmit hook is always chosen fresh.

    Another process disables, master-disables or removes the approved hook
    after the attempt's read. A fresh selection omits it and the prompt
    proceeds; the shared read must give the very same outcome rather than a
    stale target refused (and so blocked) at its launch guard.
    """
    _edit(hook_file, _as_user_prompt_hook)
    owner = HookPermissions()
    engine = RunHooksEngine(
        owner.targets,
        lambda: str(tmp_path),
        notification_targets=owner.notification_targets,
        launch_guard=owner.launch_guard,
    )
    try:
        assert _approve(owner).ready
        read = _shared_read(owner)
        if edit != "unchanged":
            _edit(hook_file, _NARROWING_EDITS[edit])
        reads = []
        real = owner._current

        def counted(*args, **kwargs):
            reads.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(owner, "_current", counted)
        data = {"prompt": "share one hook read"}
        shared = engine.fire(
            "UserPromptSubmit", session_id="s", data=data, authority_read=read
        )
        # The fresh selection, plus the launch guard when the hook still runs.
        assert len(reads) == (2 if edit == "unchanged" else 1), reads
        fresh = engine.fire("UserPromptSubmit", session_id="s", data=data)
        assert shared == fresh
        assert not shared.blocked, shared
    finally:
        engine.close()
        owner.close()


@pytest.mark.parametrize("edit", sorted(_NARROWING_EDITS))
async def test_a_user_prompt_hook_narrowed_elsewhere_lets_the_send_proceed(
    ups_rig, hook_file, edit
):
    """Full received Send: the hook another process disabled simply does not run.

    Before the shared selection was limited to "none", the stale target was
    refused at its launch guard and the whole Send was blocked before commit
    ("Send blocked by hook: Captured hook is disabled, changed or
    unapproved."); a fresh selection omits the hook and the Send dispatches.
    """
    rig = ups_rig
    rig.between = lambda: _edit(hook_file, _NARROWING_EDITS[edit])
    record = rig.start_received()

    result = await _outcome(record)

    assert not isinstance(result, BaseException), result
    assert result.accepted and rig.gateway.calls == 1
    messages = rig.store.messages_for_session(rig.session.id)
    assert not any("blocked by hook" in str(row.content) for row in messages)
    # The received read (shared by admission and v2 preparation), the fresh
    # UserPromptSubmit selection that omits the hook, the fresh final read.
    assert rig.send_reads() == [False, False, True], rig.reads


async def test_an_approved_user_prompt_hook_is_selected_fresh_and_runs(ups_rig):
    rig = ups_rig
    record = rig.start_received()

    result = await _outcome(record)

    assert not isinstance(result, BaseException), result
    assert result.accepted and rig.gateway.calls == 1
    # The received read, the fresh selection, the hook's own launch guard
    # (immediately before its process), then the fresh final read.
    assert rig.send_reads() == [False, False, False, True], rig.reads


async def test_v2_handlers_removed_elsewhere_are_prepared_from_a_fresh_read(
    v2_rig, hook_file
):
    """Another process removes an approved v2 handler after the shared read.

    A fresh read prepares nothing (no handler is configured any more).
    Preparing from the shared read would instead build the session's engine
    around the removed handler, whose fresh authority check then refuses it
    -- blocking the Send when the handler is required.
    """
    rig = v2_rig
    runtime, owner, session_id = rig.runtime, rig.owner, rig.session.id
    shared = ConsoleHookAttemptRead(session_id, owner.authority_read())
    assert owner.attempt_read_current(shared.authority)
    # A read that grants a v2 handler never answers v2 preparation.
    assert owner.attempt_v2_configuration(shared.authority) is None
    _edit(hook_file, lambda section: section.pop("handler"))
    rig.reads.clear()

    assert await runtime.prepare_hooks_v2(session_id, _hook_read=shared) is None

    assert len(rig.reads) == 1, rig.reads
    assert runtime.get_hooks_v2(session_id) is None
    assert session_id not in runtime._hooks_v2_lifecycles
    assert session_id not in runtime._hooks_v2_configured


async def test_v2_preparation_reads_fresh_for_retained_session_hook_state(rig):
    """A session that already holds hook state is never answered "nothing"."""
    runtime, owner, session_id = rig.runtime, rig.owner, rig.session.id
    runtime._hooks_v2_configured[session_id] = (
        load_hooks_config({"hooks": {}}),
        (),
        None,
    )
    shared = ConsoleHookAttemptRead(session_id, owner.authority_read())
    assert owner.attempt_read_current(shared.authority)
    rig.reads.clear()

    assert await runtime.prepare_hooks_v2(session_id, _hook_read=shared) is None

    # The fresh read saw the changed signature and retired the stale state.
    assert len(rig.reads) == 1, rig.reads
    assert session_id not in runtime._hooks_v2_lifecycles


def _fence_after_context(runtime, monkeypatch) -> list[str]:
    """Fence the session right after the fresh path reads its context.

    The same stop ``test_console_hook_preparation_demand`` uses: the fresh
    path past its "nothing to prepare" return captures the workspace context
    and then re-checks the session fence, so it raises "Console session is
    closed." before any engine or lifecycle starts. Reaching that point
    proves the shared read was not taken as the answer. Returns the sessions
    whose context was read.
    """
    contexts: list[str] = []
    original = runtime._hooks_v2_context_key

    def close_after_context(session_id):
        result = original(session_id)
        contexts.append(session_id)
        runtime._admission_fenced_sessions.add(session_id)
        return result

    monkeypatch.setattr(runtime, "_hooks_v2_context_key", close_after_context)
    return contexts


@pytest.mark.parametrize("retained", ["engine", "closed_engine", "lifecycle"])
async def test_v2_preparation_reads_fresh_for_a_retained_engine_or_lifecycle(
    rig, monkeypatch, retained
):
    """A session holding a hook engine or lifecycle is never answered "nothing".

    The rig's read grants no v2 handler, so the owner hands its (empty) v2
    selection over and only the runtime's session-state clauses stand between
    that read and the "nothing to prepare" answer -- which would leave the
    retained engine or lifecycle unchecked against current consent and
    workspace context. (The configured-signature clause is covered above.)
    """
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle

    runtime, owner, session_id = rig.runtime, rig.owner, rig.session.id
    engine = runtime.ensure_hooks_v2(session_id, (), lambda *_: True)
    detached = None
    if retained == "closed_engine":
        await engine.close()
    elif retained == "lifecycle":
        runtime._hooks_v2_lifecycles[session_id] = HookSessionLifecycle(
            engine, session_id
        )
        runtime._hooks_v2_engines.pop(session_id)
        detached = engine
    shared = ConsoleHookAttemptRead(session_id, owner.authority_read())
    assert owner.attempt_read_current(shared.authority)
    assert owner.attempt_v2_configuration(shared.authority) is not None
    contexts = _fence_after_context(runtime, monkeypatch)
    rig.reads.clear()
    try:
        with pytest.raises(RuntimeError, match="Console session is closed"):
            await runtime.prepare_hooks_v2(session_id, _hook_read=shared)
    finally:
        if detached is not None:
            await detached.close()
        await runtime.close_hooks_v2()

    assert len(rig.reads) == 1, rig.reads
    assert contexts == [session_id]


@pytest.mark.parametrize("has_definitions", [False, True], ids=["empty", "definitions"])
async def test_v2_preparation_consults_plugin_owned_skills_fresh(
    rig, monkeypatch, has_definitions
):
    """A turn carrying plugin-owned skills is never answered "nothing".

    Plugin-owned skills may contribute native hook definitions, which only
    the plugin service's ``hook_configuration`` can tell. The rig's read
    grants no v2 handler and its section configures none, so the plugin-owned
    clause of ``_hook_read_prepares_nothing`` is all that keeps the shared
    read from returning before the plugin service is asked: without it,
    native plugin hooks -- required ones included -- would silently not run
    on every warm Send with such a skill, no race needed. The production
    route passes exactly this pair (the turn configuration plus the attempt's
    read) from ``_prepare_submission_hooks``.

    Reads are counted at ``_current`` and ``v2_configuration`` is left as the
    stock method -- an instance-patched reader bypasses sharing, which is why
    the demand file's equivalent control cannot see this clause.
    """
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    runtime, owner, session_id = rig.runtime, rig.owner, rig.session.id
    maximum = {"available_skills": [{"plugin_owned": True}]}
    configuration = SimpleNamespace(skill_context_maximum=maximum)
    definitions = parse_handlers([dict(_V2_HANDLER)]) if has_definitions else ()
    native = SimpleNamespace(signature=("native",), definitions=definitions)
    captures = []

    async def hook_configuration(captured_maximum):
        captures.append(captured_maximum)
        return native

    # Only the plugin-service boundary is substituted (as in the demand file).
    service = SimpleNamespace(hook_configuration=hook_configuration)
    skills = SimpleNamespace(local_service=SimpleNamespace(plugin_service=service))
    monkeypatch.setattr(rig.controller, "_skills_service", skills)
    shared = ConsoleHookAttemptRead(session_id, owner.authority_read())
    assert owner.attempt_read_current(shared.authority)
    # The owner hands its empty v2 selection over: the runtime decides.
    assert owner.attempt_v2_configuration(shared.authority) is not None
    contexts = _fence_after_context(runtime, monkeypatch)
    rig.reads.clear()

    if has_definitions:
        # Native definitions need an engine: the fresh path goes on to read
        # the workspace context, where the fence stops it before any start.
        with pytest.raises(RuntimeError, match="Console session is closed"):
            await runtime.prepare_hooks_v2(
                session_id, configuration=configuration, _hook_read=shared
            )
    else:
        assert (
            await runtime.prepare_hooks_v2(
                session_id, configuration=configuration, _hook_read=shared
            )
            is None
        )

    assert len(captures) == 1 and captures[0] is maximum
    assert len(rig.reads) == 1, rig.reads
    assert contexts == ([session_id] if has_definitions else [])
    assert runtime.get_hooks_v2(session_id) is None
    assert session_id not in runtime._hooks_v2_lifecycles
    assert session_id not in runtime._hooks_v2_configured


async def test_v2_preparation_reads_fresh_for_configured_but_ungranted_handlers(
    v2_rig, hook_file
):
    """A configured v2 handler with no grant still goes to the fresh path.

    With the master switch off the read grants no v2 handler, so the owner
    would hand its (empty) selection over; but the section still configures a
    handler, and the fresh path does not take its "nothing" return for that
    state. Only a section with no handler at all may be answered from the
    shared read.
    """
    rig = v2_rig
    runtime, owner, session_id = rig.runtime, rig.owner, rig.session.id
    _edit(hook_file, lambda section: section.update(enabled=False))
    shared = ConsoleHookAttemptRead(session_id, owner.authority_read())
    assert owner.attempt_read_current(shared.authority)
    assert owner.attempt_v2_configuration(shared.authority) is not None
    assert shared.authority.snapshot.config.section.get("handler")
    rig.reads.clear()

    await runtime.prepare_hooks_v2(session_id, _hook_read=shared)

    assert len(rig.reads) == 1, rig.reads
