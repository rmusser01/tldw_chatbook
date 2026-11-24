"""The real trace planner admits a saved Kimi chat's turn 3+ (ADR-337 amendment).

TASK-34100.3 (folding in TASK-34100.3.1), review v5r1 F3/TE-1/TE-2/TE-7.
``Tests/Chat/test_console_trace_domain_append.py`` pins the pure rule with a
stand-in ``matches``. These tests drive ``ConsoleTraceService.
prepare_current_surface_delta`` itself, with a real ChaChaNotes database and
trace repository, over the request shape a saved chat with a reasoning-replay
model sends: every saved message as its semantic revision, and every earlier
reply's reasoning checkpoint as a typed attachment to that reply's revision.
Each reply and each checkpoint is distinct, so an admission that picks,
reorders or reattaches the wrong entry cannot reconstruct the request.

Turn N's request is the recorded messages, then the previous reply and the new
user message, then the recorded checkpoints, then the previous reply's
checkpoint. Nothing recorded changes, so the planner must admit exactly the
new entries of each domain, warm (the service still holds the previous turn's
surface checkpoint) and cold (a new service rebuilds it from the ledger), and
the ledger must then read back as the request that was sent. A request whose
saved history really changed must still be refused, and the hook's scope
guards (route, live native custody, completed project-context turn) must
leave every other send on the existing rules.

The Console's own sends use the ``agent_first`` route, which needs a real
actor chain in its provenance; that route is covered end to end by
``Tests/UI/test_console_moonshot_reasoning_reply.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_trace_final_values import (
    CompletedToolTurnWitness,
    build_verified_surface_delta,
    verify_provider_request_shadow,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    DerivedTraceProvenance,
    ProviderArtifactTraceProvenance,
    ProviderRequestProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenanceSource,
    TraceTransformKind,
    request_route_provenance,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService
from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

REFUSED = "unsupported_surface_change"


@pytest.fixture
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(str(tmp_path / "trace.sqlite"), "task-34100-3-planner")
    yield database
    database.close_connection()


@pytest.fixture
def repository() -> ConsoleTraceRepository:
    return ConsoleTraceRepository()


def _checkpoint(turn: int, reasoning: str | None = None) -> dict[str, object]:
    """A Kimi reasoning checkpoint whose reply and reasoning name its turn."""
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "moonshot",
            "protocol": "chat_completions",
            "model": "kimi-k3",
            "api_base_url": "https://api.moonshot.ai/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": f"Reply {turn}",
                    "reasoning_blocks": [reasoning or f"reasoning for turn {turn}"],
                    "calls": [],
                }
            ],
        }
    )
    encoded = dump_provider_continuation_json(checkpoint)
    assert encoded is not None
    value = json.loads(encoded)
    assert isinstance(value, dict)
    return value


def _revision_id(db: CharactersRAGDB, message_id: str) -> str:
    row = (
        db.get_connection()
        .execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
                 WHERE source_message_id = ? ORDER BY revision_sequence DESC LIMIT 1""",
            (message_id,),
        )
        .fetchone()
    )
    assert row is not None
    return str(row[0])


@dataclass
class _SavedKimiChat:
    """A saved conversation whose replies each carry a reasoning checkpoint."""

    db: CharactersRAGDB
    repository: ConsoleTraceRepository
    owner_id: str
    segment_id: str
    conversation_id: str
    policy: FrozenTracePolicy
    # (descriptor, provider value, message id) per saved message, in order.
    messages: list[tuple[SavedRevisionTraceProvenance, dict[str, object], str]]
    # (owner index into ``messages``, checkpoint value) per saved reply.
    checkpoints: list[tuple[int, dict[str, object]]]
    last_message_id: str | None = None

    def add(self, role: str, content: str, checkpoint=None) -> None:
        record: dict[str, object] = {
            "conversation_id": self.conversation_id,
            "sender": role,
            "content": content,
            "parent_message_id": self.last_message_id,
        }
        if checkpoint is not None:
            record["provider_continuation_json"] = json.dumps(checkpoint)
        message_id = self.db.add_message(record)
        assert message_id is not None
        self.last_message_id = message_id
        self.messages.append(
            (
                SavedRevisionTraceProvenance(_revision_id(self.db, message_id)),
                {"role": role, "content": content},
                message_id,
            )
        )
        if checkpoint is not None:
            self.checkpoints.append((len(self.messages) - 1, checkpoint))

    def attachment(self, owner: int) -> DerivedTraceProvenance:
        return DerivedTraceProvenance(
            TraceTransformKind.CONTINUATION_ATTACHMENT,
            (self.messages[owner][0],),
            artifact=ProviderArtifactTraceProvenance(
                TraceProvenanceSource.CONTINUATION, self.policy
            ),
        )

    def request(self, checkpoints=None, messages=None):
        """The provider surface a send of the current chat carries."""
        checkpoints = self.checkpoints if checkpoints is None else checkpoints
        messages = self.messages if messages is None else messages
        message_descriptors = tuple(item[0] for item in messages)
        provenance = ProviderRequestProvenance(
            messages=message_descriptors,
            messages_payload=message_descriptors,
            continuations=tuple(self.attachment(owner) for owner, _ in checkpoints),
            metadata=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        )
        values = tuple(item[1] for item in messages) + tuple(
            value for _, value in checkpoints
        )
        return provenance, values

    def sent(self) -> tuple[list, list]:
        """The messages and checkpoints a send of the current chat carries."""
        return (
            [item[1] for item in self.messages],
            [value for _, value in self.checkpoints],
        )


def _saved_chat(db, repository) -> _SavedKimiChat:
    conversation_id = db.add_conversation({"title": "saved kimi chat"})
    assert conversation_id is not None
    with db.transaction() as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
    return _SavedKimiChat(
        db=db,
        repository=repository,
        owner_id=owner.owner_id,
        segment_id=segment.segment_id,
        conversation_id=conversation_id,
        policy=FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None),
        messages=[],
        checkpoints=[],
    )


def _plan(service, cursor, chat, provenance, values, **kwargs):
    return service.prepare_current_surface_delta(
        cursor,
        owner_id=chat.owner_id,
        segment_id=chat.segment_id,
        route_identity=kwargs.pop("route_identity", ConsoleRequestRoute.FRESH.value),
        preparation_identity=new_opaque_id(),
        provenance=provenance,
        values=values,
        **kwargs,
    )


def _persist(service, cursor, chat, admission, boundary):
    """Bind the admitted boundary to the request and persist it."""
    actual: dict[str, object] = {"api_endpoint": "moonshot", "model": "kimi-k3"}
    actual.update(boundary._provider_request_surface_values())
    bundle = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=dict(actual),
        provenance=boundary.provenance,
        project_handler_kwargs=lambda kwargs: kwargs,
        endpoint_identity="https://api.moonshot.ai/v1",
        preparation_identity=admission.preparation_identity,
        surface_boundary=boundary,
    )
    assert bundle.available, bundle.omission_reason
    return service.persist_request(
        cursor,
        owner_id=chat.owner_id,
        segment_id=chat.segment_id,
        provenance=boundary.provenance,
        bundle=bundle,
        surface_delta=build_verified_surface_delta(
            boundary.provenance, bundle, admission=admission
        ),
    )


def _send(service, chat):
    """Plan and persist one send of the chat's current surface."""
    provenance, values = chat.request()
    with chat.db.transaction() as cursor:
        admission, boundary = _plan(service, cursor, chat, provenance, values)
        persisted = _persist(service, cursor, chat, admission, boundary)
    return admission, persisted


def _read_back(chat, persisted) -> tuple[list, list]:
    """Read one persisted request back from the ledger, as the Trace view does.

    ``ConsoleTraceNativeReader`` rebuilds a traced call's request from its
    surface head and request header; this hands it the ids a call row carries.
    """
    reader = ConsoleTraceNativeReader(chat.db)
    request, omissions = reader._reconstruct_request(  # noqa: SLF001
        chat.db.get_connection().cursor(),
        SimpleNamespace(
            surface_node_id=persisted.surface_head_id,
            request_header_id=persisted.header.header_id,
            segment_id=chat.segment_id,
            policy_id=chat.policy.policy_id,
        ),
    )
    assert omissions == ()
    return (
        list(request["messages_payload"]),
        list(request.get("provider_continuations") or ()),
    )


def _two_traced_turns(db, repository):
    """Turns 1 and 2 of a saved Kimi chat, each traced, plus turn 3's message."""
    chat = _saved_chat(db, repository)
    service = ConsoleTraceService(repository)
    chat.add("user", "Turn 1")
    _send(service, chat)
    chat.add("assistant", "Reply 1", _checkpoint(1))
    chat.add("user", "Turn 2")
    _, turn_two = _send(service, chat)
    chat.add("assistant", "Reply 2", _checkpoint(2))
    chat.add("user", "Turn 3")
    return chat, service, turn_two


def _node_count(chat) -> int:
    return len(
        chat.repository.read_surface_nodes(
            chat.db.get_connection().cursor(), chat.segment_id
        )
    )


@pytest.mark.parametrize("warm", [True, False], ids=["warm", "cold"])
def test_turn_three_appends_the_new_entries_of_each_domain(
    db, repository, warm: bool
) -> None:
    chat, service, turn_two = _two_traced_turns(db, repository)
    if not warm:
        service = ConsoleTraceService(repository)
    assert (
        service.current_surface_checkpoint(
            chat.segment_id,
            expected_head_id=turn_two.surface_head_id,
            expected_route=ConsoleRequestRoute.FRESH.value,
        )
        is not None
    ) is warm

    admission, persisted = _send(service, chat)

    # Exactly reply 2, turn 3 and reply 2's checkpoint: nothing recorded is
    # replaced, and the checkpoint is the NEW reply's, not an earlier one's.
    assert (admission.projection_checkpoint is not None) is warm
    assert admission.replacement_range is None
    assert admission.descriptors == (
        chat.messages[3][0],
        chat.messages[4][0],
        chat.attachment(3),
    )
    assert [node.component_kind for node in persisted.appended_nodes] == [
        "message",
        "message",
        "continuation",
    ]
    assert _read_back(chat, persisted) == chat.sent()


@pytest.mark.parametrize("warm", [True, False], ids=["warm", "cold"])
def test_every_later_turn_reads_back_as_the_request_sent(
    db, repository, warm: bool
) -> None:
    chat = _saved_chat(db, repository)
    service = ConsoleTraceService(repository)
    for turn in range(1, 6):
        if turn > 1:
            chat.add("assistant", f"Reply {turn - 1}", _checkpoint(turn - 1))
        chat.add("user", f"Turn {turn}")
        if not warm:
            service = ConsoleTraceService(repository)
        admission, persisted = _send(service, chat)
        if turn >= 3:
            assert admission.descriptors == (
                chat.messages[-2][0],
                chat.messages[-1][0],
                chat.attachment(len(chat.messages) - 2),
            ), f"turn {turn}"
        if turn >= 2:
            # The frozen policy is stored with the first checkpoint artifact
            # (a real call row stores it at turn 1); the reader needs it.
            assert _read_back(chat, persisted) == chat.sent(), f"turn {turn}"
    # Five turns: the read-back carries four distinct checkpoints, oldest first.
    _, checkpoints = chat.sent()
    assert [item["rounds"][0]["reasoning_blocks"] for item in checkpoints] == [
        [f"reasoning for turn {turn}"] for turn in range(1, 5)
    ]


def _foreign_revision(chat):
    """Turn 1 as an edited copy the ledger never recorded."""
    other = chat.db.add_message(
        {
            "conversation_id": chat.conversation_id,
            "sender": "user",
            "content": "Turn 1, edited",
        }
    )
    messages = list(chat.messages)
    messages[0] = (
        SavedRevisionTraceProvenance(_revision_id(chat.db, other)),
        {"role": "user", "content": "Turn 1, edited"},
        other,
    )
    return chat.request(messages=messages)


def _new_bytes_for_a_saved_revision(chat):
    messages = list(chat.messages)
    messages[0] = (messages[0][0], {"role": "user", "content": "Turn 1, edited"}, "")
    return chat.request(messages=messages)


def _changed_checkpoint_bytes(chat):
    owner, _ = chat.checkpoints[0]
    return chat.request(
        [(owner, _checkpoint(1, reasoning="tampered")), chat.checkpoints[1]]
    )


def _reordered_checkpoints(chat):
    return chat.request([chat.checkpoints[1], chat.checkpoints[0]])


def _new_checkpoint_on_an_earlier_reply(chat):
    owner, _ = chat.checkpoints[0]
    return chat.request(
        [chat.checkpoints[0], (owner, _checkpoint(1, reasoning="a second one"))]
    )


def _dropped_replay(chat):
    return chat.request([])


@pytest.mark.parametrize(
    "changed",
    [
        _foreign_revision,
        _new_bytes_for_a_saved_revision,
        _changed_checkpoint_bytes,
        _reordered_checkpoints,
        _new_checkpoint_on_an_earlier_reply,
        _dropped_replay,
    ],
    ids=lambda changed: changed.__name__.lstrip("_"),
)
@pytest.mark.parametrize("warm", [True, False], ids=["warm", "cold"])
def test_changed_history_is_still_refused(db, repository, changed, warm) -> None:
    chat, service, _ = _two_traced_turns(db, repository)
    if not warm:
        service = ConsoleTraceService(repository)
    before = _node_count(chat)
    provenance, values = changed(chat)

    with pytest.raises(ValueError, match=REFUSED):
        with db.transaction() as cursor:
            _plan(service, cursor, chat, provenance, values)

    assert _node_count(chat) == before


def test_an_edit_the_ledger_recorded_reads_back_as_sent(db, repository) -> None:
    """Dev's edit records a replacement node, so the next send is still exact.

    Editing a saved message appends a replacement for it to the trace surface
    (``update_message``), so the recorded history the planner compares with is
    the edited one. Turn 3 is then an append in each domain against that
    history, and the ledger reads back with the edited text.
    """
    chat, service, _ = _two_traced_turns(db, repository)
    message_id = chat.messages[0][2]
    row = db.get_message_by_id(message_id)
    nodes = _node_count(chat)
    db.update_message(
        message_id,
        {"content": "Turn 1, edited"},
        row["version"],
        preserve_descendants=True,
    )
    assert _node_count(chat) == nodes + 1
    chat.messages[0] = (
        SavedRevisionTraceProvenance(_revision_id(db, message_id)),
        {"role": "user", "content": "Turn 1, edited"},
        message_id,
    )

    admission, persisted = _send(ConsoleTraceService(repository), chat)

    assert len(admission.descriptors) == 3
    assert _read_back(chat, persisted) == chat.sent()


class _LiveNativeCustody:
    """Stands in for a host-issued live native custody that validates."""

    def validate(self, cursor, issued) -> None:
        return None


def _completed_project_turn(chat) -> dict[str, object]:
    return {
        "completed_tool_turn": CompletedToolTurnWitness(
            origin_call_id=new_opaque_id(),
            terminal_call_id=new_opaque_id(),
            assistant_revision_id=chat.messages[3][0].revision_id,
            user_revision_id=chat.messages[4][0].revision_id,
            project_context_count=0,
        )
    }


@pytest.mark.parametrize(
    "scope",
    [
        lambda chat: {"route_identity": ConsoleRequestRoute.TOOL_LOOP.value},
        lambda chat: {"route_identity": ConsoleRequestRoute.CONTINUE.value},
        lambda chat: {"route_identity": ConsoleRequestRoute.REGENERATE.value},
        lambda chat: {"live_native_custody": _LiveNativeCustody()},
        _completed_project_turn,
    ],
    ids=[
        "tool_loop_route",
        "continue_route",
        "regenerate_route",
        "live_native_custody",
        "completed_project_context_turn",
    ],
)
def test_sends_outside_the_amendment_keep_the_existing_rules(
    db, repository, scope
) -> None:
    """The same turn-3 surface is refused when the send is out of scope.

    The amendment covers FRESH and AGENT_FIRST sends without live native
    custody and outside a completed project-context transition; every other
    send falls through to the rules that refused turn 3 before it.
    """
    chat, _, _ = _two_traced_turns(db, repository)
    service = ConsoleTraceService(repository)
    before = _node_count(chat)
    provenance, values = chat.request()

    with pytest.raises(ValueError, match=REFUSED):
        with db.transaction() as cursor:
            _plan(service, cursor, chat, provenance, values, **scope(chat))

    assert _node_count(chat) == before


def test_a_completed_tool_turn_that_changed_nothing_appends_exactly(
    db, repository
) -> None:
    """A witnessed tool turn whose recorded entries all match needs no replacement.

    The hook runs before the completed-tool-turn replacement for a non-project
    witness. That replacement exists to swap a recorded tool-loop tail for the
    saved reply; when every recorded entry already matches, as for a plain
    append, the exact append reads back as the request sent.
    """
    chat, service, _ = _two_traced_turns(db, repository)
    witness = CompletedToolTurnWitness(
        origin_call_id=new_opaque_id(),
        terminal_call_id=new_opaque_id(),
        assistant_revision_id=chat.messages[3][0].revision_id,
        user_revision_id=chat.messages[4][0].revision_id,
    )
    provenance, values = chat.request()
    with db.transaction() as cursor:
        admission, boundary = _plan(
            service, cursor, chat, provenance, values, completed_tool_turn=witness
        )
        persisted = _persist(service, cursor, chat, admission, boundary)

    assert admission.completed_tool_turn is None
    assert admission.replacement_range is None
    assert _read_back(chat, persisted) == chat.sent()


def _three_traced_turns(db, repository):
    """Turns 1-3 traced, plus reply 3 and turn 4's message (two checkpoints)."""
    chat, service, _ = _two_traced_turns(db, repository)
    _send(service, chat)
    chat.add("assistant", "Reply 3", _checkpoint(3))
    chat.add("user", "Turn 4")
    return chat, service


def _spy_on_the_probe(monkeypatch):
    """Record the planner's retained values around each per-domain probe."""
    from tldw_chatbook.Chat import console_trace_domain_append, console_trace_service

    probes: list[dict[str, object]] = []

    def probe(**kwargs):
        retained = kwargs["retained"]
        record: dict[str, object] = {"before": dict(retained), "during": []}
        matches = kwargs["matches"]

        def watched(i: int, j: int) -> bool:
            matched = matches(i, j)
            record["during"].append(dict(retained))
            return matched

        record["result"] = console_trace_domain_append.domain_append_indices(
            **{**kwargs, "matches": watched}
        )
        record["after"] = dict(retained)
        record["retained"] = retained
        probes.append(record)
        return record["result"]

    monkeypatch.setattr(
        console_trace_service,
        "_domain_appends",
        lambda: SimpleNamespace(domain_append_indices=probe),
    )
    return probes


@pytest.mark.parametrize("warm", [True, False], ids=["warm", "cold"])
def test_a_refused_probe_leaves_the_planner_state_as_it_found_it(
    db, repository, monkeypatch, warm: bool
) -> None:
    """Pre-merge review M3, through the real planner and its real ``matches``.

    Turn 4 replays reply 1's checkpoint unchanged and reply 2's with new
    bytes. The probe compares the first (the real ``matches`` records its
    original value in the planner's retained artifact values), then refuses
    on the second. The retained values must be back to what they were before
    the probe, and the send is still refused by the existing rules.
    """
    chat, service = _three_traced_turns(db, repository)
    if not warm:
        service = ConsoleTraceService(repository)
    probes = _spy_on_the_probe(monkeypatch)
    owner, _ = chat.checkpoints[1]
    provenance, values = chat.request(
        [
            chat.checkpoints[0],
            (owner, _checkpoint(2, reasoning="tampered")),
            chat.checkpoints[2],
        ]
    )

    with pytest.raises(ValueError, match=REFUSED):
        with db.transaction() as cursor:
            _plan(service, cursor, chat, provenance, values)

    assert len(probes) == 1
    probe = probes[0]
    assert probe["result"] is None
    assert len(probe["during"]) == 2
    assert len(probe["during"][0]) == len(probe["before"]) + 1, (
        "the matched first checkpoint was recorded during the probe"
    )
    assert probe["after"] == probe["before"]


def test_an_admitted_probe_hands_its_values_to_the_boundary(
    db, repository, monkeypatch
) -> None:
    """The planner passes its own retained values; an admitted probe keeps them."""
    chat, service = _three_traced_turns(db, repository)
    probes = _spy_on_the_probe(monkeypatch)
    handed: list[object] = []
    prepare = ConsoleTraceService.prepare_surface_provenance

    def spy(self, *args, **kwargs):
        handed.append(kwargs["retained_artifact_values"])
        return prepare(self, *args, **kwargs)

    monkeypatch.setattr(ConsoleTraceService, "prepare_surface_provenance", spy)

    admission, persisted = _send(service, chat)

    assert [probe["result"] for probe in probes] == [(5, 6, 9)]
    assert handed == [probes[0]["retained"]]
    assert handed[0] is probes[0]["retained"]
    # Both recorded checkpoints were compared and kept for the boundary.
    assert len(probes[0]["after"]) == len(probes[0]["before"]) + 2
    assert _read_back(chat, persisted) == chat.sent()


@pytest.mark.parametrize("warm", [True, False], ids=["warm", "cold"])
def test_tool_loop_messages_keep_their_place_in_the_admitted_delta(
    db, repository, warm: bool
) -> None:
    """Pre-merge review M5: the delta's tool-loop overlay follows its indices.

    The overlay names provider messages by index. The per-domain delta is not
    a contiguous slice of the request, so each marked message must be named
    by its position in the admitted delta, not by an offset from the flat
    prefix.
    """
    from dataclasses import replace

    chat, service, _ = _two_traced_turns(db, repository)
    if not warm:
        service = ConsoleTraceService(repository)
    provenance, values = chat.request()
    # Mark reply 2 (incoming message 3), one of the two new messages.
    provenance = replace(provenance, tool_loop=(3,))

    with db.transaction() as cursor:
        admission, boundary = _plan(service, cursor, chat, provenance, values)

    assert admission.descriptors == (
        chat.messages[3][0],
        chat.messages[4][0],
        chat.attachment(3),
    )
    # Warm, the delta names reply 2 as its first message; either way the
    # verified surface reads it back as message 3 of the request.
    assert boundary.provenance.tool_loop == (3,)
