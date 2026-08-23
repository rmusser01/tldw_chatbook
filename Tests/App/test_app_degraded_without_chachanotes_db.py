"""TASK-20970: the app still constructs when the ChaChaNotes DB cannot open.

`TldwCli.__init__` has always tolerated an absent ChaChaNotes database -- it
logs `ChaChaNotesDB (CharactersRAGDB) instance not found/assigned in
app.__init__` and sets `self.chachanotes_db = None`, and every other wiring
degrades around that (`_wire_chat_conversation_services` guards the same value
explicitly). TASK-19057's Actor Pack wiring did not: it built an
`ActorPackRepository(None)` and called `recover()` on it, so the repository
dereferenced `None` and raised `AttributeError` -- which walked straight
through the `except PersonaActorPackCoordinatorError` written to contain it,
because that error subclasses `ValueError`. Constructing the app object failed
outright, taking 294 tests and 4 whole test modules with it.

Two tests, deliberately not one:

* the first uses the shared factory, which patches `get_chachanotes_db_lazy`
  to `None` -- the cheap shape, and the one the 294 reds were measured on;
* the second is the same degraded start driven **end to end against a real
  unopenable file on disk**, with nothing patched out. That is the shape a
  user actually hits (a corrupt database, a permission error, or TASK-19860's
  migration `.sql` files missing from the wheel), and a test that only ever
  sees a mocked-away service cannot prove the real loader still returns
  `None` into the branch this fix guards.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

# The single fixed category the wiring records when the database is missing.
# Distinct from `actor_pack_recovery_failed`: recovery did not fail, it was
# never attempted, and an operator reading the log needs to see the
# difference.
_UNAVAILABLE = "actor_pack_recovery_unavailable"


def _assert_actor_pack_wiring_degraded(app) -> None:
    """Assert the explicit no-database decision the sibling wiring makes."""
    assert app.chachanotes_db is None
    # Explicit decision, not a half-built object holding a `None` database:
    # `_wire_chat_conversation_services` treats the same value the same way.
    assert app.actor_pack_repository is None
    assert app.persona_actor_pack_coordinator is None
    # `PersonasScreen` already guards this one -- it notifies "Actor Pack
    # creation is unavailable." on `None` -- so the degraded app has a real
    # user-facing story rather than a latent crash.
    assert app.actor_pack_creation_service is None
    assert app.actor_pack_recovery_error == _UNAVAILABLE


def _capture_errors():
    """Return (sink_id, messages) capturing ERROR-level loguru output."""
    from loguru import logger as loguru_logger

    messages: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: messages.append(message.record["message"]), level="ERROR"
    )
    return sink_id, messages


def test_app_constructs_with_no_chachanotes_database():
    """Building the app with no ChaChaNotes DB must not raise."""
    from loguru import logger as loguru_logger

    from Tests.UI.app_factory import _build_test_app

    sink_id, errors = _capture_errors()
    try:
        # Fails at origin/dev: this line raises AttributeError out of
        # ActorPackRepository.list_persona_intents.
        app = _build_test_app()
    finally:
        loguru_logger.remove(sink_id)

    _assert_actor_pack_wiring_degraded(app)

    # Operator-legible: names the subsystem, the cause, and the consequence --
    # not a traceback, and not silence.
    diagnostic = next((line for line in errors if _UNAVAILABLE in line), None)
    assert diagnostic is not None, errors
    assert "ChaChaNotes database" in diagnostic
    assert "Actor Pack" in diagnostic


def test_app_constructs_against_a_real_unopenable_chachanotes_file():
    """Degraded start, end to end, against a genuinely unopenable file.

    Nothing is patched: the file on disk is not a SQLite database, so the
    real `get_chachanotes_db_lazy()` fails to open it and returns `None`
    exactly as it does for a corrupt file, a permission error, or a
    migration that cannot complete.
    """
    from loguru import logger as loguru_logger

    from tldw_chatbook.config import get_chachanotes_db_lazy, get_chachanotes_db_path

    # The autouse `isolate_test_environment` fixture has already re-pointed
    # HOME/XDG at this test's own sandbox and cleared config.py's lazy
    # database singletons, so this path is never the user's real database.
    db_path = get_chachanotes_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"not a sqlite database\n" * 64)

    # Measured, not assumed: the loader really does return None here.
    assert get_chachanotes_db_lazy() is None

    from tldw_chatbook.app import TldwCli

    sink_id, errors = _capture_errors()
    try:
        # Fails at origin/dev with the same AttributeError.
        app = TldwCli()
    finally:
        loguru_logger.remove(sink_id)

    _assert_actor_pack_wiring_degraded(app)
    assert any(_UNAVAILABLE in line for line in errors), errors
