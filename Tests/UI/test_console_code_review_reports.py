"""Native report readers retain complete originals outside model transport caps."""

import pytest

from Tests.Agents import code_review_helpers
from Tests.Agents.test_code_review_reports import finding
from tldw_chatbook.Agents.run_context import use_run_actor

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case

pytestmark = pytest.mark.bootstrap_profile


def test_native_report_preserves_original_larger_than_model_transport(review_case):
    c = review_case
    prepared = c.prepare()
    row = finding(c, prepared, "large", trigger="[bold]literal[/bold] " * 90)
    with c.participant(prepared) as provider:
        provider.service.submit(prepared.review_id, "findings", [row], 0, "large")
    c.primary.result_limit = 1024
    with use_run_actor(c.actor):
        text = c.primary.render_original_report(prepared.review_id, "R1")
    assert row["trigger"] in text
    assert row["impact"] in text
    assert prepared.review_id in text


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(48, 18), (100, 30)])
async def test_native_modal_literal_pages_focus_and_safe_close(review_case, size):
    from textual.widgets import Button, TextArea

    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.Widgets.Console.console_code_review_modal import (
        ConsoleCodeReviewModal,
    )

    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        rows = [
            finding(c, prepared, str(i), trigger="[bold]literal[/bold] " * 160)
            for i in range(3)
        ]
        provider.service.submit(prepared.review_id, "findings", rows, 0, "large")
    modal = ConsoleCodeReviewModal(service=c.primary, review_id=prepared.review_id)
    app = ConsolidatedCSSApp()
    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        for _ in range(100):
            await pilot.pause(0.01)
            if modal.query_one(TextArea).text:
                break
        assert "[bold]literal[/bold]" in modal.query_one(TextArea).text
        assert app.focused is modal.query_one(TextArea)
        assert not modal.query_one("#code-review-next", Button).disabled
        first = modal.query_one(TextArea).text
        await pilot.click("#code-review-next")
        await pilot.pause()
        assert modal.query_one(TextArea).text != first
        await pilot.press("tab")
        assert app.focused is not None
        await pilot.press("escape")
        await pilot.pause()
        assert modal not in app.screen_stack


def test_model_release_requires_native_discard_and_native_busy_is_preserved(
    review_case,
):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = review_case
    prepared = c.prepare()
    with use_run_actor(c.actor):
        result = c.provider.invoke(
            "code_review:code_review_release",
            {"review_id": prepared.review_id, "expected_revision": 0},
        )
    assert result.error == "native_discard_required"
    with c.store.lease(prepared.review_id):
        with pytest.raises(ReviewRefused, match="review_busy"):
            c.primary.release_native(prepared.review_id)
    receipt = c.primary.release_native(prepared.review_id)
    assert receipt.cleaned
    with pytest.raises(ReviewRefused):
        c.primary.render_original_report(prepared.review_id, "R1")


@pytest.mark.asyncio
async def test_mounted_controller_handoff_acknowledges_exact_original_and_closed_ui_refuses(
    review_case, tmp_path
):
    import asyncio
    from types import SimpleNamespace

    from textual.widgets import TextArea

    from Tests.Chat.test_console_agent_swap import _controller
    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.UI.Console_Modules import review_reports
    from tldw_chatbook.Widgets.Console.console_code_review_modal import (
        ConsoleCodeReviewModal,
    )

    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        provider.service.submit(
            prepared.review_id,
            "findings",
            [finding(c, prepared, "visible", trigger="literal mounted original")],
            0,
            "original",
        )
    controller, store, db = _controller(tmp_path, [])
    session = store.create_session()
    controller.code_review.report_service = lambda session_id, review_id: (
        prepared.review_id,
        c.primary,
    )
    controller.fleet_wake._loop = asyncio.get_running_loop()
    report = {
        "slot_id": "R1",
        "original_page": c.primary.native_report_page(prepared.review_id, "R1"),
    }
    app = ConsolidatedCSSApp()
    # Headless (the launch wake): the app is attached but no Console view, so
    # nothing presents and the original stays pending.
    controller.app = app
    try:
        async with app.run_test(size=(48, 18)):
            assert not await asyncio.to_thread(
                controller._present_review_report,
                session.id,
                prepared.review_id,
                report,
            )
            assert not isinstance(app.screen, ConsoleCodeReviewModal)
            # A view attach installs the presentation-layer presenter.
            review_reports.attach_app(controller, app)
            accepted = await asyncio.to_thread(
                controller._present_review_report,
                session.id,
                prepared.review_id,
                report,
            )
            assert accepted
            assert isinstance(app.screen, ConsoleCodeReviewModal)
            assert "literal mounted original" in app.screen.query_one(TextArea).text
            first_modal = app.screen
            assert await asyncio.to_thread(
                controller._present_review_report,
                session.id,
                prepared.review_id,
                report,
            )
            assert app.screen is first_modal
            assert not any(
                "literal mounted original" in message.content
                for message in store.messages_for_session(session.id)
            )
            controller._session_close_generations[session.id] = 1
            assert not await asyncio.to_thread(
                controller._present_review_report,
                session.id,
                prepared.review_id,
                report,
            )
            controller._session_close_generations.pop(session.id)
            controller.app = SimpleNamespace()
            assert not await asyncio.to_thread(
                controller._present_review_report,
                session.id,
                prepared.review_id,
                report,
            )
    finally:
        await controller.shutdown()
        store.persistence.db.close()
        db.close()


@pytest.mark.asyncio
async def test_native_export_denial_does_not_release_bundle(review_case, monkeypatch):
    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.UI.Console_Modules import markdown_export
    from tldw_chatbook.Widgets.Console.console_code_review_modal import (
        ConsoleCodeReviewModal,
    )

    c = review_case
    prepared = c.prepare()

    async def deny(path, body):
        raise PermissionError("destination denied")

    monkeypatch.setattr(markdown_export, "write_markdown_file", deny)
    modal = ConsoleCodeReviewModal(service=c.primary, review_id=prepared.review_id)
    app = ConsolidatedCSSApp()
    notices = []
    monkeypatch.setattr(
        app, "notify", lambda message, **kwargs: notices.append(message)
    )
    async with app.run_test(size=(80, 24)):
        await app.push_screen(modal)
        await modal._save("/not-admitted.md")
        assert notices == ["Export failed: PermissionError"]
        assert c.primary.native_report_slots(prepared.review_id) == ("R1",)


@pytest.mark.asyncio
async def test_close_reader_retains_native_read_lease_until_worker_finishes(
    review_case, monkeypatch
):
    import asyncio
    import threading

    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.Widgets.Console.console_code_review_modal import (
        ConsoleCodeReviewModal,
    )

    c = review_case
    prepared = c.prepare()
    started, finish = threading.Event(), threading.Event()
    native_read = c.primary.native_report_page

    def slow_read(review_id, slot, offset=0):
        with c.store.lease(review_id):
            started.set()
            assert finish.wait(5)
            return native_read(review_id, slot, offset)

    monkeypatch.setattr(c.primary, "native_report_page", slow_read)
    app = ConsolidatedCSSApp()
    modal = ConsoleCodeReviewModal(service=c.primary, review_id=prepared.review_id)
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            await app.push_screen(modal)
            assert await asyncio.to_thread(started.wait, 3)
            await pilot.press("escape")
            await pilot.pause()
            assert modal not in app.screen_stack
            with pytest.raises(ReviewRefused, match="review_busy"):
                c.primary.release_native(prepared.review_id)
        finally:
            finish.set()
        for _ in range(100):
            with c.store._tab.lock:
                if c.store._context(prepared.review_id)._leases == 0:
                    break
            await asyncio.sleep(0.01)
        assert c.primary.release_native(prepared.review_id).cleaned


@pytest.mark.parametrize(
    "operation", ["original", "submit", "source", "claim", "source_targets"]
)
def test_authority_callbacks_do_not_hold_publication_lock(review_case, operation):
    from concurrent.futures import ThreadPoolExecutor

    c = review_case
    prepared = c.prepare()
    available = []
    native_guard = c.authority.admitted.guard

    def probe():
        acquired = c.store._tab.lock.acquire(timeout=0.2)
        if acquired:
            c.store._tab.lock.release()
        return acquired

    def guard(write):
        with ThreadPoolExecutor(max_workers=1) as pool:
            available.append(pool.submit(probe).result(timeout=1))
        return native_guard(write)

    with c.participant(prepared) as provider:
        row = finding(c, prepared, "guard")
        if operation == "original":
            provider.service.submit(prepared.review_id, "findings", [row], 0, "guard")
        object.__setattr__(c.authority.admitted, "guard", guard)
        try:
            if operation == "original":
                c.primary.render_original_report(prepared.review_id, "R1")
            elif operation == "submit":
                provider.service.submit(
                    prepared.review_id, "findings", [row], 0, "guard"
                )
            elif operation == "claim":
                with use_run_actor(c.actor):
                    c.primary.claim_review_slot(prepared.review_id, "R1")
            elif operation == "source_targets":
                provider.service.source_instruction_targets(
                    prepared.review_id, document_ref=row["evidence"][0]["source_ref"]
                )
            else:
                provider.service.read(
                    prepared.review_id,
                    "source",
                    document_ref=row["evidence"][0]["source_ref"],
                )
        finally:
            object.__setattr__(c.authority.admitted, "guard", native_guard)
    assert available and all(available), "authority callback retained publication lock"


def test_native_original_does_not_expand_repeated_source_ranges(
    review_case, monkeypatch
):
    c = review_case
    c.file.write_text("large captured evidence line\n" * 8000)
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        rows = [finding(c, prepared, str(i)) for i in range(32)]
        with c.store.lease(prepared.review_id) as context:
            ref, source = next(
                (ref, source)
                for ref, source in context.sources.items()
                if source.line_count == 8000
            )
        for row in rows:
            row["evidence"][0].update(
                source_ref=ref, digest=source.digest, end_line=8000
            )
        for start in range(0, 32, 8):
            provider.service.submit(
                prepared.review_id,
                "findings",
                rows[start : start + 8],
                start // 8,
                str(start),
            )
    native_read = c.store.read_file
    source_reads = []
    with c.store.lease(prepared.review_id) as context:
        source_paths = {source.path for source in context.sources.values()}

    def read(context, path):
        if path in source_paths:
            source_reads.append(path)
        return native_read(context, path)

    monkeypatch.setattr(c.store, "read_file", read)
    text = c.primary.render_original_report(prepared.review_id, "R1")
    assert not source_reads, "native originals eagerly expanded retained evidence"
    assert all(f'"local_key": "{i}"' in text for i in range(32))
    pages, offset = [], 0
    for _ in range(100):
        page = c.primary.native_report_page(prepared.review_id, "R1", offset)
        assert len(page["body"]) <= 8192
        pages.append(page["body"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    else:
        pytest.fail("native original pagination did not terminate")
    assert all(f'"local_key": "{i}"' in "".join(pages) for i in range(32))
    exported = c.primary.render_all_reports(prepared.review_id)
    assert all(f'"local_key": "{i}"' in exported for i in range(32))
    assert len(exported) < 100_000 and not source_reads


@pytest.mark.asyncio
async def test_amplified_range_first_visible_page_does_bounded_work(
    review_case, monkeypatch
):
    import threading

    from textual.widgets import Select, TextArea

    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.Widgets.Console import console_code_review_modal as reader

    c = review_case
    c.file.write_text("large captured evidence line\n" * 8000)
    prepared = c.prepare()
    with c.store.lease(prepared.review_id) as context:
        ref, source = next(
            (ref, source)
            for ref, source in context.sources.items()
            if source.line_count == 8000
        )
    with c.participant(prepared) as provider:
        for chunk in range(4):
            rows = [
                finding(c, prepared, str(i)) for i in range(chunk * 8, chunk * 8 + 8)
            ]
            for row in rows:
                row["evidence"][0].update(
                    source_ref=ref, digest=source.digest, end_line=8000
                )
            provider.service.submit(
                prepared.review_id, "findings", rows, chunk, str(chunk)
            )
    reads, sanitized = [], []
    native_read, literal = c.store.read_file, reader._render_literal_text

    def read(context, path, **kwargs):
        data = native_read(context, path, **kwargs)
        reads.append((path, len(data)))
        return data

    def sanitize(body):
        sanitized.append((len(body), threading.get_ident()))
        return literal(body)

    monkeypatch.setattr(c.store, "read_file", read)
    monkeypatch.setattr(reader, "_render_literal_text", sanitize)
    modal = reader.ConsoleCodeReviewModal(
        service=c.primary, review_id=prepared.review_id
    )
    app = ConsolidatedCSSApp()
    owner = threading.get_ident()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(modal)
        for _ in range(100):
            await pilot.pause(0.01)
            if modal.query_one(TextArea).text:
                break
        assert modal.query_one(TextArea).text
        assert all(path != source.path for path, _ in reads)
        assert sum(size for _, size in reads) < 100_000
        assert sanitized and all(
            size <= 8192 and thread != owner for size, thread in sanitized
        )
        modal.query_one(Select).value = "source:" + ref
        for _ in range(100):
            await pilot.pause(0.01)
            if "large captured evidence line" in modal.query_one(TextArea).text:
                break
        assert "large captured evidence line" in modal.query_one(TextArea).text
        assert [(path, size) for path, size in reads if path == source.path] == [
            (source.path, 8192)
        ]


@pytest.mark.asyncio
async def test_automatic_second_original_syncs_selector_and_fences_stale_read(
    review_case, monkeypatch
):
    import asyncio
    import threading

    from textual.widgets import Select, TextArea

    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.Agents.code_review_models import ReviewTeam
    from tldw_chatbook.Widgets.Console.console_code_review_modal import (
        ConsoleCodeReviewModal,
    )

    c = review_case
    c.team = ReviewTeam(
        reviewers=c.team.reviewers * 2, verifier_preset_id=c.team.verifier_preset_id
    )
    c.primary = c.service()
    # prepare fixture uses the current primary and native actor.
    prepared = c.prepare()
    with c.participant(prepared, slot="R1") as provider:
        provider.service.submit(
            prepared.review_id,
            "findings",
            [finding(c, prepared, "first", trigger="FIRST ORIGINAL")],
            0,
            "first",
        )
    with c.participant(prepared, slot="R2") as provider:
        provider.service.submit(
            prepared.review_id,
            "findings",
            [finding(c, prepared, "second", trigger="SECOND ORIGINAL")],
            0,
            "second",
        )
    modal = ConsoleCodeReviewModal(service=c.primary, review_id=prepared.review_id)
    app = ConsolidatedCSSApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(modal)
        for _ in range(100):
            await pilot.pause(0.01)
            if "FIRST ORIGINAL" in modal.query_one(TextArea).text:
                break
        started, finish = threading.Event(), threading.Event()
        read_page = modal._read_page

        def slow(selection, offset):
            started.set()
            assert finish.wait(5)
            return read_page(selection, offset)

        monkeypatch.setattr(modal, "_read_page", slow)
        stale = asyncio.create_task(modal._read("R1", modal._generation))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            page = await asyncio.to_thread(
                lambda: modal.literal_page(
                    c.primary.native_report_page(prepared.review_id, "R2")
                )
            )
            assert modal.present_original("R2", page)
            assert modal.query_one(Select).value == "R2"
        finally:
            finish.set()
        await stale
        await pilot.pause()
        assert "SECOND ORIGINAL" in modal.query_one(TextArea).text
        assert "FIRST ORIGINAL" not in modal.query_one(TextArea).text
        monkeypatch.setattr(modal, "_read_page", read_page)
        modal.query_one(Select).value = "R1"
        for _ in range(100):
            await pilot.pause(0.01)
            if "FIRST ORIGINAL" in modal.query_one(TextArea).text:
                break
        assert "FIRST ORIGINAL" in modal.query_one(TextArea).text


def test_unified_verdict_order_precedes_severity(review_case):
    import json

    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        provider.service.submit(
            prepared.review_id,
            "findings",
            [
                finding(c, prepared, str(i), trigger=str(i), severity=severity)
                for i, severity in enumerate(("high", "low", "medium"))
            ],
            0,
            "findings",
        )
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        page = provider.service.next_verification_batch(
            prepared.review_id,
            remaining_input_tokens=10000,
            estimate=lambda text: len(text) // 4,
            max_result_chars=16000,
            cursor=None,
        )
        records = []
        for row in page.records:
            verdict = {"0": "refuted", "1": "unresolved", "2": "confirmed"}[
                row["claim"]["trigger"]
            ]
            records.append(
                {
                    "schema_version": 1,
                    "verification_unit_id": row["verification_unit_id"],
                    "verdict": verdict,
                    "reason": "source supported",
                    "evidence": row["claim"]["evidence"],
                    **(
                        {"next_check": "inspect boundary"}
                        if verdict == "unresolved"
                        else {}
                    ),
                }
            )
        provider.service.submit(
            prepared.review_id, "dispositions", records, 0, "decisions"
        )
    body = c.primary.render_unified_report(prepared.review_id)
    rows = json.loads(body.split("\n\n", 1)[1])["original_dispositions"]
    assert [row["disposition"]["verdict"] for row in rows] == [
        "confirmed",
        "unresolved",
        "refuted",
    ]
    page = c.primary.native_report_page(prepared.review_id, "unified")["body"]
    assert (
        page.index('"verdict": "confirmed"')
        < page.index('"verdict": "unresolved"')
        < page.index('"verdict": "refuted"')
    )


@pytest.mark.parametrize("operation", ["submit", "source"])
def test_authority_callback_revision_change_refuses_before_publication(
    review_case, operation
):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = review_case
    prepared = c.prepare()
    guard = c.authority.admitted.guard
    changed = False

    def change(write):
        nonlocal changed
        if not changed:
            changed = True
            with c.store._tab.lock:
                c.store._context(prepared.review_id).revision += 1
        return guard(write)

    with c.participant(prepared) as provider:
        row = finding(c, prepared)
        object.__setattr__(c.authority.admitted, "guard", change)
        try:
            with pytest.raises(ReviewRefused, match="source_authority_changed"):
                if operation == "submit":
                    provider.service.submit(
                        prepared.review_id, "findings", [row], 0, "changed"
                    )
                else:
                    provider.service.read(
                        prepared.review_id,
                        "source",
                        document_ref=row["evidence"][0]["source_ref"],
                    )
        finally:
            object.__setattr__(c.authority.admitted, "guard", guard)
        with c.store.lease(prepared.review_id) as context:
            assert not context.original_index
            assert not context.source_state.pages


@pytest.mark.asyncio
@pytest.mark.parametrize("read_error", [False, True])
async def test_deferred_report_read_does_not_publish_during_modal_removal(
    review_case, monkeypatch, read_error
):
    import asyncio
    import threading

    from textual.widgets import TextArea

    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.Widgets.Console.console_code_review_modal import (
        ConsoleCodeReviewModal,
    )

    c = review_case
    prepared = c.prepare()
    modal = ConsoleCodeReviewModal(service=c.primary, review_id=prepared.review_id)
    async with ConsolidatedCSSApp().run_test(size=(80, 24)) as pilot:
        await pilot.app.push_screen(modal)
        for _ in range(100):
            await pilot.pause(0.01)
            if modal._options_loaded:
                break
        await pilot.pause()
        body = modal.query_one(TextArea)
        before = body.text
        entered, finish = threading.Event(), threading.Event()
        removing, removed = asyncio.Event(), asyncio.Event()

        def deferred(*args):
            entered.set()
            assert finish.wait(5)
            if read_error:
                raise ValueError("late read failure")
            return {"body": "late read must not publish", "next_offset": None}

        async def hold_unmount(self):
            if self is modal:
                removing.set()
                await removed.wait()

        monkeypatch.setattr(modal, "_read_page", deferred)
        monkeypatch.setattr(
            ConsoleCodeReviewModal, "on_unmount", hold_unmount, raising=False
        )
        read = asyncio.create_task(modal._read("R1", modal._generation))
        removal = None
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            removal = asyncio.ensure_future(modal.remove())
            await asyncio.wait_for(removing.wait(), 3)
            assert modal.is_attached and modal._pruning
            finish.set()
            await read
            assert body.text == before
        finally:
            finish.set()
            removed.set()
            if removal is not None:
                await removal
            await read


def test_review_reports_key_is_advertised_and_inert_under_the_setup_card():
    """'u' joins the footer vocabulary and, like every workbench key, does
    nothing while the first-run setup card locks the composer."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_WORKBENCH_SHORTCUTS,
        ChatScreen,
    )

    assert {b.key: b.action for b in ChatScreen.BINDINGS}.get(
        "u"
    ) == "open_code_review_reports"
    assert ("U", "review reports") in CONSOLE_WORKBENCH_SHORTCUTS

    def unreachable():
        raise AssertionError("the review owner must not be touched")

    blocked = SimpleNamespace(
        _console_setup_modal_blocking=lambda: True,
        _ensure_console_chat_controller=unreachable,
    )
    assert ChatScreen.action_open_code_review_reports(blocked) is None
