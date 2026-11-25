"""Resumable listening service contracts using the installed SQLite store."""

from importlib import import_module, util
from dataclasses import replace

import pytest

from Tests.Subscriptions.test_article_listening_store import (
    prepare,
    records as _records_fixture,
    setup_store,
)
from Tests.Subscriptions.test_article_audio_assets import publish
from tldw_chatbook.Backup_Recovery.native_files import create_private_directory
from tldw_chatbook.TTS.audio_output_ownership import AudioOutputOwnership
from tldw_chatbook.TTS.controlled_wav_playback import CompletionKind, PlaybackCompletion
from tldw_chatbook.TTS.controlled_wav_progress import ProgressObservation
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from contextlib import asynccontextmanager
import os
import subprocess
import sys
import time

records = _records_fixture


class Output:
    def __init__(self):
        self.owner = AudioOutputOwnership()
        self.runs = []
        self.started = asyncio.Queue()

    async def prepare(self, wav, bookmark, *, before_release):
        run = Run(wav, bookmark, before_release, self.started)
        self.runs.append(run)
        return run

    async def acquire(self, ticket, run, current):
        assert current()
        return await self.owner.acquire("watchlists", run.retire, intent=ticket)


class Run:
    def __init__(self, wav, bookmark, before_release, starts):
        self.wav = wav
        self.bookmark = bookmark
        self.before_release = before_release
        self.starts = starts
        self.completion = asyncio.get_running_loop().create_future()
        self.lease = None
        self.already_at_end = bookmark.frame == wav.content_frames

    async def start(self, lease):
        assert lease.authorizes(self.retire)
        self.lease = lease
        self.starts.put_nowait(self)

    def position(self):
        return ProgressObservation(self.bookmark.frame, 1, self.lease.generation)

    async def finish(self, kind):
        result = PlaybackCompletion(
            kind,
            "interrupted" if kind == CompletionKind.STOPPED else "natural",
            self.bookmark,
            self.lease.generation if self.lease else None,
            None,
        )
        await self.before_release(result)
        if self.lease:
            self.lease.release()
        self.completion.set_result(result)
        return True

    def retire(self):
        return self.finish(CompletionKind.STOPPED)


def test_native_predecessor_format_is_strict_and_empty_is_legacy():
    name = "tldw_chatbook.Subscriptions.article_playback_receipt"
    assert util.find_spec(name), "missing private process predecessor format"
    codec = import_module(name)
    record = codec.encode_predecessor(123, 987654321)
    assert codec.decode_predecessor(record) == (123, 987654321)
    assert codec.decode_predecessor(b"") is None
    for invalid in (record[:-1], record + b"x", b"x" + record[1:]):
        with pytest.raises(ValueError):
            codec.decode_predecessor(invalid)


def service_type():
    name = "tldw_chatbook.Subscriptions.article_listening_service"
    assert util.find_spec(name), "missing app-owned resumable listening service"
    return import_module(name).ArticleListeningService


@pytest.mark.asyncio
async def test_restore_is_paused_and_server_cannot_read_local_item(records):
    _, binding, store, voice = setup_store(records)
    old = store.replace_session(
        prepare(store, binding[0], voice, brief=True), expected_revision=None
    )
    speech_calls = []

    async def forbidden_voice():
        speech_calls.append(1)
        raise AssertionError("restoration must not resolve speech")

    service = service_type()(
        store,
        records.db_path.parent,
        freeze_voice=forbidden_voice,
        authorize=lambda: binding[0],
    )
    try:
        await service.restore()
        assert service.snapshot().phase == "paused"
        assert service.snapshot().queue_count == 2
        await service.listen("server:watchlist_item:1", runtime_backend="local")
        await service.listen("local:watchlist_item:1", runtime_backend="server")
        assert not speech_calls
        assert store.load_session() == old
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("late", [False, True])
async def test_pending_article_uses_legacy_tts_admission_and_long_asset_holder(
    records, tmp_path, late
):
    from Tests.Subscriptions.test_article_listening_audio import mono, _snapshot
    from tldw_chatbook.Subscriptions.article_listening_audio import (
        freeze_listening_voice,
    )
    from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
    from tldw_chatbook.TTS.adapter_types import TTSProviderDescriptor, TTSProviderSpec
    from tldw_chatbook.TTS.legacy_bridge import LegacyBackendHost, LegacyTTSAdapter
    from tldw_chatbook.TTS.legacy_catalogs import legacy_catalog
    from tldw_chatbook.TTS.TTS_Generation import TTSService

    _, binding, store, _ = setup_store(records)
    requests = []
    entered, finish = asyncio.Event(), asyncio.Event()

    async def stream(request):
        requests.append(1)
        if late and len(requests) == 1:
            entered.set()
            await finish.wait()
        yield mono()

    backend = SimpleNamespace(
        set_progress_callback=lambda _: None, generate_speech_stream=stream
    )
    manager = SimpleNamespace(
        get_backend=AsyncMock(return_value=backend), close_all_backends=AsyncMock()
    )
    host = LegacyBackendHost(
        provider_id="openai", app_config={}, manager_factory=lambda _: manager
    )
    adapter = LegacyTTSAdapter("openai", host, legacy_catalog("openai"))
    spec = TTSProviderSpec(
        descriptor=TTSProviderDescriptor(
            provider_id="openai", display_name="Test", native=False
        ),
        factory=lambda _: adapter,
        initial_config={"app_config": {}},
        exclusive_reconfigure=False,
    )
    tts = TTSService(
        TTSAdapterRegistry(specs=(spec,), aliases={}),
        preferences_snapshot=_snapshot(
            provider_id="openai", model_id="tts-1", voice_mode="exact", voice_id="alloy"
        ),
    )

    async def speech(voice):
        return tts, None

    output = Output()
    root = tmp_path / "private"
    create_private_directory(root)
    service = service_type()(
        store,
        root,
        freeze_voice=lambda: freeze_listening_voice(tts_service=tts),
        speech_services=speech,
        authorize=lambda: binding[0],
        output=output.owner,
        prepare_output=output.prepare,
        acquire_output=output.acquire,
    )
    try:
        await service.restore()
        await service.listen("local:watchlist_item:1", runtime_backend="local")
        if late:
            await asyncio.wait_for(entered.wait(), 12)
            await asyncio.wait_for(service.stop(), 2)
            await service.resume()
            await asyncio.wait_for(service._active.task, 3)
            assert service.snapshot().error == "article_audio_busy"
            assert requests == [1]
            finish.set()
            await service._audio.settle()
            assert store.load_session().foreground.artifact is None
            assert output.started.empty()
            await service.retry()
        player = await asyncio.wait_for(output.started.get(), 12)
        assert requests == ([1, 1] if late else [1])
        assert store.load_session().foreground.artifact is not None
        assert service.snapshot().total_chunks == 1
        await service.stop()
        if not late:
            player.wav.path.unlink()
            await service.resume()
            await asyncio.wait_for(service._active.task, 4)
            assert service.snapshot().error == "article_audio_missing"
            assert "regenerate" in service.snapshot().actions
            assert requests == [1]
            await service.retry(regenerate=True)
            await asyncio.wait_for(output.started.get(), 12)
            assert requests == [1, 1]
    finally:
        finish.set()
        await service.close()
        await tts.wait_closed()


@pytest.mark.asyncio
async def test_interruption_checkpoints_queue_and_returns_after_natural_completion(
    records, tmp_path
):
    _, binding, store, voice = setup_store(records)
    from tldw_chatbook.Subscriptions.article_speech import CONVERTER, prepare_saved_text

    queue = store.prepare_brief(
        1,
        authority=binding[0],
        voice=voice,
        converter=CONVERTER,
        prepare_text=prepare_saved_text,
    )
    session = store.replace_session(queue, expected_revision=None)
    root = tmp_path / "private"
    create_private_directory(root)
    for entry in session.queue:
        publish(store, store.load_session(), root, entry, frames=480)

    async def freeze():
        return voice

    output = Output()
    service = service_type()(
        store,
        root,
        freeze_voice=freeze,
        authorize=lambda: binding[0],
        output=output.owner,
        prepare_output=output.prepare,
        acquire_output=output.acquire,
    )

    async def next_run():
        return await asyncio.wait_for(output.started.get(), 5)

    try:
        await service.restore()
        await service.resume()
        first = await next_run()
        first.bookmark = replace(first.bookmark, frame=80)
        await service.listen("local:watchlist_item:1", runtime_backend="local")
        second = await next_run()
        assert store.load_session().queue[0].bookmark.frame == 80
        assert store.load_session().resume_queue
        await service.listen("local:watchlist_item:1", runtime_backend="local")
        assert len(output.runs) == 2
        await second.finish(CompletionKind.NATURAL)
        returned = await next_run()
        assert returned.bookmark.frame == 80
        assert store.load_session().foreground is None
        await service.stop()
        assert service.snapshot().phase == "paused"
    finally:
        await service.close()


@asynccontextmanager
async def saved_service(records, root, *, pending=False):
    from tldw_chatbook.Subscriptions.article_speech import CONVERTER, prepare_saved_text

    _, binding, store, voice = setup_store(records)
    queue = store.prepare_brief(
        1,
        authority=binding[0],
        voice=voice,
        converter=CONVERTER,
        prepare_text=prepare_saved_text,
    )
    session = store.replace_session(queue, expected_revision=None)
    create_private_directory(root)
    if not pending:
        for entry in session.queue:
            publish(store, store.load_session(), root, entry, frames=480)
    output = Output()
    service = service_type()(
        store,
        root,
        freeze_voice=AsyncMock(return_value=voice),
        authorize=lambda: binding[0],
        output=output.owner,
        prepare_output=output.prepare,
        acquire_output=output.acquire,
    )
    try:
        await service.restore()
        yield service, store, binding, output
    finally:
        await service.close()


async def started(output):
    return await asyncio.wait_for(output.started.get(), 6)


@pytest.mark.asyncio
@pytest.mark.parametrize("cutover", ["prepared", "denied", "granted"])
async def test_unstarted_retirement_retains_run_holder_and_exact_grant(
    records, tmp_path, cutover
):
    from tldw_chatbook.Subscriptions.article_listening_lease import ListeningBusy
    from tldw_chatbook.TTS.audio_output_ownership import OutputBlocked

    root = tmp_path / "private"
    async with saved_service(records, root) as (service, store, _, output):
        retired = asyncio.Event()
        allow_retirement = False
        attempts = []
        grants = []

        async def retire():
            retired.set()
            return allow_retirement

        async def prepare(*args, **kwargs):
            attempts.append(1)
            if cutover == "prepared":
                output.owner.begin_intent()
            return SimpleNamespace(already_at_end=False, retire=retire)

        async def admit(ticket, run, current):
            assert current()
            if cutover == "denied":
                return None
            grant = await output.owner.acquire("watchlists", run.retire, intent=ticket)
            grants.append(grant)
            output.owner.begin_intent()
            return grant

        service._prepare_output, service._acquire_output = prepare, admit
        try:
            await service.resume()
            attempt = service._active
            await asyncio.wait_for(retired.wait(), 5)
            await attempt.task
            assert service.snapshot().phase == "retiring"
            assert service._active is attempt and attempt.run is not None
            assert service.snapshot().error == "player_retirement_unconfirmed"
            if cutover == "granted":
                assert grants == [attempt.lease] and output.owner.blocked
                with pytest.raises(OutputBlocked):
                    await output.owner.acquire("other", AsyncMock(return_value=True))
                assert output.owner.owner == "watchlists"
            await service.retry()
            with pytest.raises(ListeningBusy):
                await service.clear()
            with pytest.raises(ListeningBusy):
                records.delete_article(1)
            assert (
                attempts == [1]
                and store.load_session()
                and records.get_item_status(1) == "new"
            )
        finally:
            allow_retirement = True
        await service.stop()
        assert output.owner.owner is None and service.snapshot().phase == "paused"
        service._prepare_output, service._acquire_output = (
            output.prepare,
            output.acquire,
        )
        await service.retry()
        await started(output)


@pytest.mark.asyncio
async def test_stop_waits_for_inflight_output_preparation_to_return_its_pins(
    records, tmp_path
):
    from tldw_chatbook.Subscriptions.article_listening_lease import ListeningBusy

    async with saved_service(records, tmp_path / "private") as (service, _, _, _):
        entered, finish, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def retire():
            return settled.is_set()

        async def prepare(*args, **kwargs):
            entered.set()
            await finish.wait()
            return SimpleNamespace(already_at_end=False, retire=retire)

        service._prepare_output = prepare
        try:
            await service.resume()
            await asyncio.wait_for(entered.wait(), 5)
            stopping = asyncio.create_task(service.stop())
            await asyncio.sleep(0)
            assert not stopping.done() and service._active is not None
            finish.set()
            with pytest.raises(ListeningBusy, match="player_retirement_unconfirmed"):
                await stopping
            assert (
                service.snapshot().phase == "retiring" and service._active is not None
            )
        finally:
            finish.set()
            settled.set()
        await service.stop()
        assert service.snapshot().phase == "paused"


@pytest.mark.asyncio
@pytest.mark.parametrize("playing", [True, False])
async def test_two_foregrounds_keep_original_bookmark_and_return_policy(
    records, tmp_path, playing
):
    async with saved_service(records, tmp_path / "root") as (service, store, _, output):
        await service.resume()
        first = await started(output)
        first.bookmark = replace(first.bookmark, frame=109)
        if not playing:
            await service.pause()
        await service.listen("local:watchlist_item:1", runtime_backend="local")
        second = await started(output)
        second.bookmark = replace(second.bookmark, frame=31)
        await service.listen("local:watchlist_item:2", runtime_backend="local")
        third = await started(output)
        assert third.bookmark.frame == 0 and third.wav.sha256 == first.wav.sha256
        assert store.load_session().queue[0].bookmark.frame == 109
        assert store.load_session().resume_queue is playing
        await service.skip()
        assert store.load_session().foreground is None
        if playing:
            assert (await started(output)).bookmark.frame == 109
        else:
            assert service.snapshot().phase == "paused"
            assert output.started.empty()
        assert records.get_item_status(1) == records.get_item_status(2) == "new"


@pytest.mark.asyncio
async def test_stop_in_padding_exact_end_manual_resume_and_failure_do_not_return(
    records, tmp_path
):
    async with saved_service(records, tmp_path / "root") as (service, store, _, output):
        await service.resume()
        await started(output)
        await service.listen("local:watchlist_item:1", runtime_backend="local")
        fg = await started(output)
        fg.bookmark = replace(fg.bookmark, frame=fg.wav.content_frames)
        await service.stop()
        assert store.load_session().foreground.bookmark.frame == fg.wav.content_frames
        await service.resume()
        await asyncio.wait_for(service._active.task, 5)
        assert service.snapshot().phase == "paused"
        assert store.load_session().foreground.item_id == 1 and output.started.empty()
        await service.skip()
        back = await started(output)
        await back.finish(CompletionKind.FAILED)
        assert service.snapshot().phase == "paused"
        assert store.load_session().cursor == 0


@pytest.mark.asyncio
async def test_policy_away_and_back_fences_physical_owner_and_adopts_saved_roles(
    records, tmp_path
):
    async with saved_service(records, tmp_path / "root") as (
        service,
        store,
        binding,
        output,
    ):
        await service.resume()
        old = await started(output)
        old.bookmark = replace(old.bookmark, frame=90)
        binding[0] = replace(binding[0], policy_epoch=8)
        service.policy_changed()
        binding[0] = replace(binding[0], policy_epoch=9)
        service.policy_changed()
        await asyncio.wait_for(service._retiring_task, 5)
        assert store.load_session().authority.policy_epoch == 7
        assert store.load_session().queue[0].bookmark.frame == 90
        await service.resume()
        assert (await started(output)).bookmark.frame == 90
        assert store.load_session().authority.policy_epoch == 9


@pytest.mark.asyncio
async def test_foreground_survives_brief_delete_and_second_instance_cannot_mutate(
    records, tmp_path
):
    async with saved_service(records, tmp_path / "root") as (
        service,
        store,
        binding,
        output,
    ):
        await service.resume()
        await started(output)
        await service.listen("local:watchlist_item:1", runtime_backend="local")
        fg = await started(output)
        fg.bookmark = replace(fg.bookmark, frame=123)
        other = service_type()(
            store,
            tmp_path / "root",
            freeze_voice=AsyncMock(),
            authorize=lambda: binding[0],
        )
        await other.restore()
        assert (
            other.snapshot().error == "listening_busy" and not other.snapshot().actions
        )
        await other.close()
        await service.delete_brief(1)
        assert service.snapshot().phase == "paused"
        assert (
            not store.load_session().queue
            and store.load_session().foreground.bookmark.frame == 123
        )
        await service.delete_article(1)
        assert service.snapshot().current_title is None
        assert store.load_session().foreground.item_id is None


@pytest.mark.asyncio
async def test_foreign_audio_transfer_commits_before_successor_gets_lease(
    records, tmp_path
):
    async with saved_service(records, tmp_path / "root") as (service, store, _, output):
        await service.resume()
        old = await started(output)
        old.bookmark = replace(old.bookmark, frame=55)
        other = await output.owner.acquire("test-other", AsyncMock(return_value=True))
        assert store.load_session().queue[0].bookmark.frame == 55
        assert service.snapshot().phase == "paused"
        other.release()


@pytest.mark.skipif(os.name != "nt", reason="Windows NTFS process witness")
def test_live_predecessor_refuses_and_dead_predecessor_waits_device_settle(records):
    import _winapi
    from tldw_chatbook.TTS._controlled_wav_windows import _creation_time
    from tldw_chatbook.Subscriptions.article_listening_lease import ListeningBusy

    _, _, store, _ = setup_store(records)
    with store.session_lease(records.db_path.parent) as lease:
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            lease.seal_predecessor(child.pid, _creation_time(int(child._handle)))
            with pytest.raises(ListeningBusy, match="previous_player_unconfirmed"):
                lease.recover_predecessor()
            child.stdin.close()
            assert child.wait(10) == 0
            assert (
                _winapi.WaitForSingleObject(int(child._handle), 0)
                == _winapi.WAIT_OBJECT_0
            )
            after_death = time.monotonic()
            lease.recover_predecessor()
            assert time.monotonic() - after_death >= 0.250
            assert lease._read_predecessor() is None
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(10)


@pytest.mark.asyncio
async def test_stopping_during_slow_admission_preserves_previous_queue(
    records, tmp_path
):
    async with saved_service(records, tmp_path / "root") as (service, store, _, output):
        old = store.load_session()
        waiting, finish = asyncio.Event(), asyncio.Event()

        async def delayed_voice():
            waiting.set()
            await finish.wait()
            return old.voice

        service._freeze_voice = delayed_voice
        pending = asyncio.create_task(
            service.listen("local:watchlist_item:1", runtime_backend="local")
        )
        await asyncio.wait_for(waiting.wait(), 2)
        try:
            await asyncio.wait_for(service.stop(), 1)
        finally:
            finish.set()
            await pending
        assert store.load_session() == old
        assert not output.runs


@pytest.mark.asyncio
async def test_revision_targets_exact_service_even_for_equal_reopened_sqlite_revision(
    records, tmp_path
):
    async with saved_service(records, tmp_path / "root") as (first, store, binding, _):
        stale = first.snapshot().revision
        await first.close()
        other = service_type()(
            store,
            tmp_path / "root",
            freeze_voice=AsyncMock(),
            authorize=lambda: binding[0],
        )
        try:
            await other.restore()
            current = other.snapshot().revision
            assert current.revision == stale.revision and current != stale
            await other.clear(expected_revision=stale)
            assert store.load_session() is not None
            await other.clear(expected_revision=current)
            assert store.load_session() is None
        finally:
            await other.close()


@pytest.mark.asyncio
async def test_clock_coalescing_and_exact_terminal_flush(records, tmp_path):
    async with saved_service(records, tmp_path / "root") as (service, store, _, output):
        await service.resume()
        rendered = service.snapshot().revision
        player = await started(output)
        a = service._active
        due = a.next_checkpoint
        player.bookmark = replace(player.bookmark, frame=100)
        await service._save_progress(a, due - 0.1)
        assert store.load_session().queue[0].bookmark.frame == 0
        await service._save_progress(a, due)
        assert store.load_session().queue[0].bookmark.frame == 100
        player.bookmark = replace(player.bookmark, frame=150)
        await service._save_progress(a, due + 0.9)
        assert store.load_session().queue[0].bookmark.frame == 100
        await service.stop(expected_revision=rendered)
        assert store.load_session().queue[0].bookmark.frame == 150
        await service._save_progress(a, due + 2)
        assert store.load_session().queue[0].bookmark.frame == 150


@pytest.mark.asyncio
async def test_rendered_skip_cannot_advance_the_next_entry_but_session_stop_can(
    records, tmp_path
):
    async with saved_service(records, tmp_path / "root") as (service, store, _, output):
        await service.resume()
        first = await started(output)
        rendered = service.snapshot().revision
        await first.finish(CompletionKind.NATURAL)
        second = await started(output)
        await service.skip(expected_revision=rendered)
        assert not second.completion.done() and store.load_session().cursor == 1
        await service.stop(expected_revision=rendered)
        assert second.completion.done() and service.snapshot().phase == "paused"


@pytest.mark.asyncio
async def test_large_selection_validates_replacement_and_replay_resets_bookmarks(
    records, tmp_path
):
    from tldw_chatbook.Subscriptions.article_speech import CONVERTER, prepare_saved_text

    m, binding, store, voice = setup_store(records)
    with records.transaction() as c:
        c.executemany(
            "INSERT INTO subscription_items(id,subscription_id,url,title,content) VALUES (?,1,?,'title','saved')",
            [(i, str(i)) for i in range(4, 65)],
        )
        c.executemany(
            "INSERT INTO briefing_items(briefing_id,item_id,live_item_id,source_id,cited,citation_position,provenance_version) VALUES(1,?,?,1,1,?,2)",
            [(i, i, i - 2) for i in range(4, 65)],
        )
        c.execute(
            "INSERT INTO briefings(id,watchlist_id,status,body_markdown) VALUES (2,1,'complete','[item 3]')"
        )
    candidate = store.prepare_brief(
        1,
        authority=binding[0],
        voice=voice,
        converter=CONVERTER,
        prepare_text=prepare_saved_text,
    )
    session = store.replace_session(candidate, expected_revision=None)
    assert len(session.queue) == 63 and session.queue[-1].item_id == 64
    root = tmp_path / "root"
    create_private_directory(root)
    for e in session.queue[:3]:
        publish(store, store.load_session(), root, e, frames=480)
    session = store.load_session()
    for e in session.queue[:3]:
        session = store.checkpoint(
            m.ListeningCheckpoint(
                session.authority,
                session.incarnation,
                session.playback_generation,
                62,
                False,
                "completed",
                ((e.entry_id, replace(e.bookmark, frame=140)),),
            ),
            expected_revision=session.revision,
        )
    output = Output()
    service = service_type()(
        store,
        root,
        freeze_voice=AsyncMock(return_value=voice),
        authorize=lambda: binding[0],
        output=output.owner,
        prepare_output=output.prepare,
        acquire_output=output.acquire,
    )
    try:
        await service.restore()
        assert service.snapshot().phase == "completed"
        assert (
            await service.preview_brief(1, runtime_backend="local")
        ).playable_count == 63
        await service.play_brief(2, runtime_backend="local", replace=True)
        assert store.load_session() == session
        await service.resume()
        assert not output.runs
        await service.play_again()
        await started(output)
        restored = store.load_session()
        assert restored.cursor == 0 and [
            e.bookmark.frame for e in restored.queue[:3]
        ] == [0, 0, 0]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_policy_revoked_during_foreground_transfer_still_flushes_original_owner(
    records, tmp_path
):
    async with saved_service(records, tmp_path / "root") as (
        service,
        store,
        binding,
        output,
    ):
        await service.resume()
        original = await started(output)
        original.bookmark = replace(original.bookmark, frame=99)
        before = original.before_release
        refused_terminal = []

        async def switch_at_retirement(result):
            original.before_release = before
            binding[0] = replace(binding[0], policy_epoch=binding[0].policy_epoch + 1)
            service.policy_changed()
            try:
                await before(result)
            except Exception:
                refused_terminal.append(True)
                raise

        original.before_release = switch_at_retirement
        await service.listen("local:watchlist_item:1", runtime_backend="local")
        await asyncio.wait_for(service._retiring_task, 5)
        assert (
            not refused_terminal
        ), "physically retired output failed to checkpoint after authority changed"
        saved = store.load_session()
        assert saved.foreground is None and saved.queue[0].bookmark.frame == 99
        assert saved.authority.policy_epoch == 7 and not saved.resume_queue
        assert output.started.empty()


@pytest.mark.skipif(os.name != "nt", reason="Windows exact process identity")
def test_fresh_destructive_delete_and_prune_refuse_a_live_predecessor(
    records, tmp_path
):
    import _winapi
    from tldw_chatbook.Subscriptions.article_audio_assets import ArticleAudioAssets
    from tldw_chatbook.Subscriptions.article_listening_lease import ListeningBusy
    from tldw_chatbook.TTS._controlled_wav_windows import _creation_time

    _, _, store, _ = setup_store(records)
    root = tmp_path / "private-audio"
    create_private_directory(root)
    child = subprocess.Popen(
        [sys.executable, "-B", "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        with store.session_lease(root) as owner:
            owner.seal_predecessor(child.pid, _creation_time(int(child._handle)))
        # The old native DB/profile owner is gone, while the exact recorded child lives.
        with pytest.raises(ListeningBusy, match="previous_player_unconfirmed"):
            records.delete_article(1)
        with pytest.raises(ListeningBusy, match="previous_player_unconfirmed"):
            ArticleAudioAssets(store, profile_data_root=root).prune(int(time.time()))
        assert records.get_item_status(1) == "new"
        child.stdin.close()
        assert child.wait(10) == 0
        assert (
            _winapi.WaitForSingleObject(int(child._handle), 0) == _winapi.WAIT_OBJECT_0
        )
        observed = time.monotonic()
        assert records.delete_article(1)
        assert time.monotonic() - observed >= 0.250
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(10)
