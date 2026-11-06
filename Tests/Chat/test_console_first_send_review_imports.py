"""A first agent send loads the code-review modules off the UI loop (TASK-33620.15.53).

Measured live on origin/dev e8eeae7db9 (Anthropic haiku through the review
proxy, 160x45, a fresh isolated profile per launch, the first send after the
idle first-send warm-up had finished, every cold module load recorded with its
thread): between Enter and the provider call, run start loaded
``Agents.agent_lesson_promotion`` (the managed-skill proposal gate, when the
Library provider is on), ``Chat.console_review_delivery`` and, inside
``compose_turn_review_provider``, ``Agents.code_review_models`` and
``Chat.console_code_review`` with ``code_review_service``, ``code_review_source``,
``code_review_store`` and ``code_review_guide`` on the UI loop. None is in the
idle warm-up (``Chat/console_first_send_modules.py``), and ADR-097's
``MAX_FIRST_SEND_WARMUP_ADDED_MODULES`` row has room for one module, not eight:
the owner declined a raise on 2026-10-31. So run start imports them on a worker
thread instead.

The send runs in a fresh interpreter (``sys.modules`` is process-global, and
an earlier test importing these modules would hide the import) after the
pre-import pass's screen routes and warm-up list, on a real controller with
the agent runtime on, a real agent bridge and a Library provider, so the
managed-skill gate is built as live.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.bootstrap_profile

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The modules the live first send loaded on the UI loop in run start's review
#: steps (the ones the warm-up does not hold).
REVIEW_FAMILY = (
    "tldw_chatbook.Agents.agent_lesson_promotion",
    "tldw_chatbook.Chat.console_review_delivery",
    "tldw_chatbook.Agents.code_review_models",
    "tldw_chatbook.Chat.console_code_review",
    "tldw_chatbook.Agents.code_review_service",
    "tldw_chatbook.Agents.code_review_source",
    "tldw_chatbook.Agents.code_review_store",
    "tldw_chatbook.Agents.code_review_guide",
)

_SEND_SCRIPT = r"""
import asyncio
import importlib
import json
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

from loguru import logger

logger.remove()

import tldw_chatbook.app as app_module
import tldw_chatbook.UI.Screens.chat_screen  # noqa: F401  (the initial route)
from tldw_chatbook.Chat import console_first_send_modules as warm

# The pre-import pass as it runs live, minus its pacing: the screen routes,
# then the first-send warm-up list.
cls = app_module.TldwCli
app = cls.__new__(cls)
app._shutting_down = False
app.loguru_logger = logger
cls._pause_between_preimports = lambda self, gap: None
app._preimport_screens(app._screen_preimport_route_order())
for name in warm.FIRST_SEND_MODULES:
    try:
        importlib.import_module(name)
    except Exception:
        pass

family = json.loads(sys.argv[1])
draft = sys.argv[2]

from Tests.console_provider_doubles import provider_resolution
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Agents.tool_catalog import LIBRARY_RESERVED_TOOL_NAMES
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_library_policy import ConsoleAssistantLibraryAccess
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnConfigurationSnapshot
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


class Gateway(ConsoleProviderGateway):
    def __init__(self):
        super().__init__(config_provider=dict, environ={})
        self.dispatched = []

    async def resolve_for_send(self, _selection):
        return provider_resolution(provider="llama_cpp", model="local")

    async def stream_chat(self, _resolution, messages, **_kwargs):
        self.dispatched.append(messages)
        yield "plain answer"


main = threading.main_thread()
loop_loads = {}
worker_loads = []
real_find_and_load = importlib._bootstrap._find_and_load


def recording(name, import_):
    cold = name not in sys.modules
    on_loop = threading.current_thread() is main
    if cold and on_loop:
        frame, stack = sys._getframe(1), []
        while frame is not None and len(stack) < 8:
            stack.append(f"{frame.f_globals.get('__name__')}:{frame.f_code.co_name}")
            frame = frame.f_back
    module = real_find_and_load(name, import_)
    if cold and on_loop:
        loop_loads.setdefault(name, " <- ".join(stack))
    elif cold:
        worker_loads.append(name)
    return module


async def main_send(tmp):
    gateway = Gateway()
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=ConsoleAgentBridge(
            agent_runs_db=AgentRunsDB(Path(tmp) / "runs.db", client_id="t"),
            store=store,
            provider_gateway=gateway,
        ),
        agent_runtime_enabled=True,
        provider="llama_cpp",
        model="local",
    )
    # A Library provider, as the live app's Assistant access gives run start,
    # so it builds the managed-skill proposal gate.
    library_provider = LibraryToolProvider(SimpleNamespace())
    library_authority = library_provider.issue_builtin_authority(
        reserved_names=LIBRARY_RESERVED_TOOL_NAMES,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
    )
    controller._library_provider_factory = lambda _context: library_provider
    controller._library_provider_for_context = lambda _context: (
        library_provider,
        library_authority,
    )
    # The Console's mount loads the automatic-work ledger (live:
    # ``ConsoleFleetWakeCoordinator.recover`` reads ``runs_db.automatic_work``
    # at mount), and with it ``code_review_models``, before any send.
    controller._agent_bridge.runs_db.automatic_work
    warmed.extend(sorted(m for m in family if m in sys.modules))
    base = controller.resolve_runtime_turn_configuration_snapshot(session.id)
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=base.provider_selection,
        scratch_space=base.scratch_space,
        session_settings=base.session_settings,
        project_authority=base.project_authority,
        tool_configuration={**base.tool_configuration, "agent_runtime_enabled": True},
    )
    importlib._bootstrap._find_and_load = recording
    try:
        await controller.submit_draft(
            draft, session_id=session.id, configuration=configuration
        )
    finally:
        importlib._bootstrap._find_and_load = real_find_and_load
    contents = [m.content for m in store.messages_for_session(session.id)]
    await controller.shutdown()
    return gateway, contents


warmed = []
with tempfile.TemporaryDirectory() as tmp:
    gateway, contents = asyncio.run(main_send(tmp))

print(
    "SEND_JSON:"
    + json.dumps(
        {
            "warmed": warmed,
            "dispatched": len(gateway.dispatched),
            "answered": "plain answer" in contents,
            "loop_loads": loop_loads,
            "worker_loads": worker_loads,
            "resident_after": sorted(m for m in family if m in sys.modules),
        }
    ),
    flush=True,
)
"""


def _send_in_fresh_interpreter(root: Path, draft: str) -> dict:
    """Run one first agent send of ``draft`` after the warm-up; return its record."""
    home, data, config_dir = (root / n for n in ("home", "data", "config"))
    for directory in (home, data, config_dir):
        directory.mkdir(mode=0o700)
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_DATA_HOME": str(data),
        "XDG_CONFIG_HOME": str(config_dir),
        "TLDW_TEST_MODE": "1",
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)
    result = subprocess.run(
        [sys.executable, "-c", _SEND_SCRIPT, json.dumps(REVIEW_FAMILY), draft],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = next(
        (x for x in result.stdout.splitlines() if x.startswith("SEND_JSON:")), None
    )
    assert line is not None, result.stdout[-2000:] + result.stderr[-2000:]
    return json.loads(line[len("SEND_JSON:") :])


def _loaded_on_the_loop(record: dict, names) -> dict[str, str]:
    """``names`` the send loaded on the UI loop that were not warm before it."""
    return {
        name: where
        for name, where in record["loop_loads"].items()
        if name in names and name not in record["warmed"]
    }


def _report(on_loop: dict[str, str]) -> str:
    return "\n".join(f"  {name}: {where}" for name, where in sorted(on_loop.items()))


@pytest.mark.integration
def test_a_first_ordinary_agent_send_loads_no_review_module_on_the_ui_loop(
    tmp_path,
):
    """The live case: an ordinary first send after the warm-up.

    A module the warm-up holds counts as warm; no other review module may
    load on the loop that runs the send. Red on origin/dev, where run start
    loaded the managed-skill gate's module, the review delivery seam and the
    whole review service there.
    """
    record = _send_in_fresh_interpreter(tmp_path, "hello")
    # Non-vacuity: the send reached the provider, and run start built the
    # managed-skill proposal gate (its module is loaded after the send).
    assert record["dispatched"] == 1 and record["answered"], record
    assert "tldw_chatbook.Agents.agent_lesson_promotion" in record["resident_after"]

    on_loop = _loaded_on_the_loop(record, REVIEW_FAMILY)
    assert not on_loop, (
        "an ordinary first send loaded these review modules on the UI loop:\n"
        + _report(on_loop)
    )


#: The review service's own modules, which an explicit review request needs.
REVIEW_SERVICE = (
    "tldw_chatbook.Chat.console_code_review",
    "tldw_chatbook.Agents.code_review_service",
    "tldw_chatbook.Agents.code_review_source",
    "tldw_chatbook.Agents.code_review_store",
    "tldw_chatbook.Agents.code_review_guide",
)


@pytest.mark.integration
def test_a_first_review_request_loads_the_review_service_on_a_worker_thread(
    tmp_path,
):
    """A request that names a review stage still composes, off the UI loop.

    Its run start imports the review service where it composes the turn's
    review tools, on that step's worker thread. Red on origin/dev, which
    imported the service on the loop before the hop.
    """
    record = _send_in_fresh_interpreter(tmp_path, "Review my changes")
    assert record["dispatched"] == 1 and record["answered"], record
    # Non-vacuity: the review composition really ran and loaded the service.
    assert set(REVIEW_SERVICE) <= set(record["resident_after"]), record
    assert set(REVIEW_SERVICE) <= set(record["worker_loads"]) | set(
        record["loop_loads"]
    ), record

    on_loop = _loaded_on_the_loop(record, REVIEW_SERVICE)
    assert not on_loop, (
        "a first review request loaded the review service on the UI loop:\n"
        + _report(on_loop)
    )


# --- The ordinary-turn check run start makes before the review composition ---

_STAGE_REQUESTS = (
    "review",
    "Review my changes",
    "Please review and verify my changes.",
    "Can you please do a multi-model code review?",
    "$multi-model-review review my changes",
    "verify",
    "Consolidate the findings.",
    "retry reviewer r2",
    "Retry the verification",
    "resume review",
)


@pytest.mark.parametrize("text", _STAGE_REQUESTS)
def test_every_stage_request_names_a_review_word(text):
    from tldw_chatbook.Agents.code_review_models import mentions_review_request
    from tldw_chatbook.Chat.console_code_review import _explicit_stages

    assert _explicit_stages(text, ("multi-model-review",)), text  # a real request
    assert mentions_review_request(text)


@pytest.mark.parametrize(
    ("text", "owner", "boundary", "possible"),
    [
        ("hello", None, None, False),
        (None, None, None, False),
        ("What changed in the weather today?", None, None, False),
        ("Review my resume", None, None, True),  # classify decides; may be chat
        ("VERIFY", None, None, True),
        ("yes", object(), None, True),  # an offer answer needs the owner's state
        (None, object(), None, True),
        ("hello", None, object(), True),  # an accepted review wake
    ],
)
def test_review_turn_possible_is_false_only_for_an_ordinary_turn(
    text, owner, boundary, possible
):
    from types import SimpleNamespace

    from tldw_chatbook.Agents.code_review_models import review_turn_possible

    wake = None if boundary is None else SimpleNamespace(review_boundary=boundary)
    assert (
        review_turn_possible(text, review_owner=owner, wake_authorization=wake)
        is possible
    )


_FRAGMENTS = (
    "review",
    "Review",
    "REVIEW",
    "verify",
    "Verify",
    "consolidate",
    "retry",
    "resume",
    "reviewer",
    "r2",
    "the",
    "this",
    "my",
    "changes",
    "patch",
    "verification",
    "please",
    "can you",
    "do a multi-model code",
    "and",
    "then",
    "not",
    "don't",
    "if",
    "?",
    ".",
    "!",
    '"',
    "'",
    "`",
    ">",
    "$multi-model-review",
    "hello",
    "findings",
    "reports",
    "\n",
)


def test_the_review_word_gate_changes_no_stage_the_parser_finds():
    """``_explicit_stages`` returns what it returned before its first gate.

    The gate is what lets run start skip a turn unparsed, so it must be a
    necessary condition of every stage, never a new refusal: with the gate
    forced open, the parser finds the same stages for every text.
    """
    from unittest import mock

    from hypothesis import given, settings
    from hypothesis import strategies as st

    from tldw_chatbook.Chat import console_code_review

    texts = st.one_of(
        st.text(max_size=60),
        st.lists(st.sampled_from(_FRAGMENTS), max_size=8).map(" ".join),
    )

    requests = []

    @settings(max_examples=400, deadline=None, derandomize=True)
    @given(text=texts, trusted=st.booleans())
    def check(text, trusted):
        bindings = ("multi-model-review",) if trusted else ()
        gated = console_code_review._explicit_stages(text, bindings)
        with mock.patch.object(
            console_code_review, "mentions_review_request", lambda _text: True
        ):
            ungated = console_code_review._explicit_stages(text, bindings)
        assert gated == ungated, text
        if gated:
            requests.append(text)

    check()
    assert len(requests) >= 5, "the generated texts held almost no stage request"
