"""Pre-extraction characterization pins for the Library Ingest subsystem.

Wave-5 Task 1 (ingest series 1/3, state PR; recipe: ``backlog/docs/
library-decomposition-recipe.md``; collections/skills series precedent:
``Tests/UI/test_library_collections_characterization.py``). Before the
Ingest subsystem's state PR moves any ``_library_ingest_*`` field into
``LibraryIngestState``, this file pins the CURRENT behavior of every Ingest
``@on`` handler a per-id ``grep -rn "<id>" Tests/`` (the collections series'
own methodology, re-run here) reported as never actually
``.press()``-ed/``.click()``-ed/message-bubbled-through a real DOM/Pilot
interaction -- not merely mentioned for an existence/``.disabled``
assertion, and not merely reached via a raw ``screen.handle_x(event)`` call
with a hand-built ``Button.Pressed``-shaped fake (which exercises the
handler BODY but never proves the ``@on`` selector itself routes a real
DOM event there).

Enumeration: an ``ast`` walk of ``LibraryScreen`` for method names
containing "ingest" (case-insensitive) found 78 methods, of which 30 carry
a distinct ``@on`` decorator (29 unique methods --
``handle_library_ingest_browse`` carries two decorators,
``#ingest-preflight-choose`` and ``#library-ingest-browse``, over one
shared body). Each of the 29 was checked with a per-id ``grep -rn`` across
``Tests/UI``, ``Tests/Library``, ``Tests/Live``, ``Tests/App`` and
``Tests/integration`` (no dedicated ``Tests/Ingest/`` tree exists; ``Tests/
Live`` has zero ingest-field/selector references at all), followed by a
manual read of the surrounding lines. 24 of the 29 are already exercised
this way; 5 are genuine gaps, pinned below. (``handle_library_ingest_
browse``'s OTHER selector, ``#ingest-preflight-choose``, is not itself
pressed anywhere, but the SAME handler body IS pressed via ``#library-
ingest-browse`` in ``Tests/UI/test_library_ingest_inline_consent.py``, so
the method is genuinely covered -- this file does not double-pin it.)

- ``test_ingest_top_button_opens_ingest_canvas`` pins
  ``_on_library_ingest_top_button`` (``#library-ingest-top-button``). The
  one existing test that names this id
  (``test_ingest_button_opens_canvas``, ``Tests/integration/
  test_library_ingest_flow.py``) explicitly bypasses a real press ("button.
  press() is unreliable for async handlers in the test harness") and calls
  ``_select_library_rail_row`` directly instead. A real ``.press()`` DOES
  work for this handler -- Textual's message pump processes an async
  ``@on`` body across ``pilot.pause()`` the same as every other async
  handler in this file (e.g. ``_return_library_rail_to_starter``, pressed
  via ``#library-rail-back-to-starter`` at ``test_library_shell.py:2940``)
  -- so this pins the real dispatch path the old comment assumed away.
- ``test_tooling_fold_toggle_persists_across_recompose`` pins
  ``sync_library_ingest_tooling_detail_expanded``
  (``@on(LibraryIngestCanvas.ToolingDetailToggled)``). Its sibling,
  ``sync_library_ingest_type_group_expanded``, already has genuine
  full-screen coverage via
  ``test_library_shell_ingest_type_group_panel_expand_survives_recompose``
  (toggles a real ``#type-group-generic`` ``Collapsible`` and asserts
  ``screen._library_ingest_form.expanded_type_groups`` updates and
  survives a recompose) -- easy to miss with a literal
  ``grep -rn "OptionPanelToggled"``/handler-name search, since that test
  never names the message class or the handler. The tooling fold has no
  screen-level equivalent: the only existing tests toggling
  ``#ingest-preflight-tooling-detail`` mount the canvas standalone
  (``Tests/UI/test_library_ingest_canvas.py``'s ``_CanvasHost``/
  ``_MessageRecordingHost``), proving the CANVAS posts
  ``ToolingDetailToggled`` but never that the real ``LibraryScreen``
  handler receives it and persists ``tooling_detail_expanded``.
- ``test_view_on_server_button_opens_the_remote_media_detail`` pins
  ``handle_library_ingest_view_on_server``
  (``.library-ingest-view-server``, zero references anywhere in the
  suite).
- ``test_choose_gguf_button_opens_the_gguf_picker_for_the_failed_job``
  pins ``handle_library_ingest_choose_gguf``
  (``.library-ingest-choose-gguf`` -- only ever ``query_one``-d for a
  button-presence/removal assertion in ``test_library_ingest_canvas.py``,
  never pressed).

Five more per-job-row handlers found by the same census --
``handle_library_ingest_cancel``, ``handle_library_ingest_force_stop``,
``handle_library_ingest_retry_faster_whisper``, ``handle_library_ingest_
option_reset``, ``handle_library_ingest_directory_browse`` -- are each
exercised by an existing test, but only via a raw
``screen.handle_x(event)`` (or ``LibraryScreen.handle_x(screen, event)``)
call with a hand-built ``MagicMock`` event on a REAL, fully-``__init__``-ed
screen instance (``Tests/UI/test_library_screen.py``, ``Tests/UI/
test_library_ingest_inline_consent.py``, ``Tests/UI/
test_library_ingest_canvas.py``). That is not the same bypass shape the
recipe's own catalogue warns about (an unbound fake ``self``/a bare
``SimpleNamespace`` with only flat kwargs) -- the screen is real and the
handler's own LOGIC is genuinely exercised -- but it does not prove the
``@on`` CSS-selector-to-handler DISPATCH itself works, since the id/class
match Textual performs when routing a real ``Button.Pressed`` is never
exercised. Deliberately left un-pinned here (a "spot-check", not an
exhaustive re-drive of every already-somewhat-tested handler into a full
DOM press): each needs a specific backing ``LibraryIngestJob`` state
(an active local attempt for cancel/force-stop; a classified STT failure
for the retry variants) composed through the real registry + a real rendered
row, which is exactly the added machinery ``test_view_on_server_button_
opens_the_remote_media_detail``/``test_choose_gguf_button_opens_the_gguf_
picker_for_the_failed_job`` below demonstrate is tractable -- recorded here
as known, bounded coverage debt for a future task, not silently absorbed
into "24 of 29 covered."

No live bugs were found among the 5 pinned here -- every one is a coverage
gap, not a behavior bug (each handler's current behavior, once actually
driven through the DOM, is exactly what its body says it should be).

Every test below drives the screen only through DOM queries/presses/value
assignments and public screen attributes -- originally the pre-extraction
``_library_ingest_*`` names, which will resolve identically through the
state PR's generated property shim across the (future) controller PR. The
pinned BEHAVIOR these tests characterize is unaffected by that move.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Button, Collapsible, Input

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.ingest_types import PreflightResult
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    _LibraryIngestCanvasHarness,
    _open_library_ingest_canvas,
    _wait_for_library_shell,
    _wait_for_selector,
)

_POLL_ATTEMPTS = 250
_POLL_INTERVAL = 0.02


@pytest.mark.asyncio
async def test_ingest_top_button_opens_ingest_canvas() -> None:
    """Characterization (pre-extraction): a real press on the rail-top
    ``Import…`` button reaches ``_on_library_ingest_top_button`` and opens
    the Ingest canvas, exactly like the rail-row entry point already
    covered by ``_open_library_ingest_canvas``'s own ``.press()`` on
    ``#library-row-ingest-import-media``.
    """
    harness = _LibraryIngestCanvasHarness(None)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-ingest-top-button", Button).press()
        await pilot.pause()
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#library-ingest-path")

        assert screen.query_one("#library-ingest-path", Input) is not None


@pytest.mark.asyncio
async def test_tooling_fold_toggle_persists_across_recompose(tmp_path) -> None:
    """Characterization (pre-extraction): expanding the preflight summary's
    "What's missing" fold reaches ``sync_library_ingest_tooling_detail_
    expanded`` and the flag survives a full recompose -- the screen-level
    counterpart to ``test_library_shell_ingest_type_group_panel_expand_
    survives_recompose``'s own coverage of the sibling option-panel
    handler.
    """
    db = MediaDatabase(
        tmp_path / "ingest-canvas.db", client_id="wave5-tooling-fold"
    )
    source = tmp_path / "note.txt"
    source.write_text("warned preflight for the tooling fold.", encoding="utf-8")
    harness = _LibraryIngestCanvasHarness(db)

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        await _open_library_ingest_canvas(screen, pilot)

        # Prime a warned pre-flight result so the tooling-detail fold
        # renders (mirrors test_library_ingest_canvas.py's own
        # _warned_state, inlined here rather than importing a private
        # helper across test files).
        screen._library_ingest_form.preflight = PreflightResult(
            type_groups={"generic": [str(source)]},
            warnings=[
                {
                    "feature": "feature_0",
                    "label": "Backend 0",
                    "hint": "capability 0",
                    "command": 'pip install -e ".[extra0]"',
                }
            ],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=1,
        )
        screen.refresh(recompose=True)
        await _wait_for_selector(screen, pilot, "#ingest-preflight-tooling-detail")

        fold = screen.query_one("#ingest-preflight-tooling-detail", Collapsible)
        assert fold.collapsed is True, "the detail must start folded away"
        assert screen._library_ingest_form.tooling_detail_expanded is False

        fold.collapsed = False
        for _ in range(_POLL_ATTEMPTS):
            if screen._library_ingest_form.tooling_detail_expanded:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError(
                "Fold expand never synced back to tooling_detail_expanded."
            )

        # A full recompose must leave the fold expanded.
        screen.refresh(recompose=True)
        await _wait_for_selector(screen, pilot, "#ingest-preflight-tooling-detail")
        assert (
            screen.query_one(
                "#ingest-preflight-tooling-detail", Collapsible
            ).collapsed
            is False
        )


@pytest.mark.asyncio
async def test_view_on_server_button_opens_the_remote_media_detail() -> None:
    """Characterization (pre-extraction): pressing "View on server" on a
    finished SERVER-origin job reaches ``handle_library_ingest_view_on_
    server``, which resolves the job's ``remote_media_id`` and opens it via
    ``_open_library_external_media_detail``. Zero test references to this
    handler existed before this pin.
    """
    harness = _LibraryIngestCanvasHarness(None)
    job = harness.library_ingest_jobs.submit(
        source_path="https://example.test/report", origin="server"
    )
    harness.library_ingest_jobs.mark_remote_done(
        job.job_id, remote_media_id="server-media-9"
    )

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        await _open_library_ingest_canvas(screen, pilot)
        await _wait_for_selector(
            screen, pilot, f"#library-ingest-view-server-{job.job_id}"
        )

        opened = AsyncMock()
        screen._open_library_external_media_detail = opened

        screen.query_one(
            f"#library-ingest-view-server-{job.job_id}", Button
        ).press()
        await pilot.pause()
        await pilot.pause()

        opened.assert_called_once_with("server-media-9")


@pytest.mark.asyncio
async def test_choose_gguf_button_opens_the_gguf_picker_for_the_failed_job(
    monkeypatch,
) -> None:
    """Characterization (pre-extraction): pressing "Choose another GGUF…"
    on a job FAILED with an ``stt_failure``/``choose_another_gguf``
    classification reaches ``handle_library_ingest_choose_gguf``, which
    opens the real GGUF picker (mirrors ``test_library_ingest_inline_
    consent.py``'s own ``push_screen``-capturing pattern for ``#library-
    ingest-browse`` rather than driving a real filesystem dialog).
    """
    harness = _LibraryIngestCanvasHarness(None)
    job = harness.library_ingest_jobs.submit(source_path="/tmp/interview.wav")
    harness.library_ingest_jobs.mark_failed(
        job.job_id,
        error="transcribe.cpp model missing",
        error_detail={
            "category": "stt_failure",
            "actions": ["choose_another_gguf"],
        },
    )

    async with harness.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = harness.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        await _open_library_ingest_canvas(screen, pilot)
        await _wait_for_selector(
            screen, pilot, f"#library-ingest-choose-gguf-{job.job_id}"
        )

        callbacks: list = []
        monkeypatch.setattr(
            screen.app,
            "push_screen",
            lambda *args, **kwargs: callbacks.append(
                kwargs.get("callback", args[1] if len(args) > 1 else None)
            ),
            raising=False,
        )
        configure = Mock()
        screen._configure_transcribe_cpp_gguf = configure

        screen.query_one(
            f"#library-ingest-choose-gguf-{job.job_id}", Button
        ).press()
        await pilot.pause()

        assert callbacks and callbacks[0] is not None, "Choose GGUF pushed no picker"

        chosen = Path("/tmp/model.gguf")
        await callbacks[0](chosen)
        await pilot.pause()

        configure.assert_called_once_with(chosen, retry_job_id=job.job_id)
