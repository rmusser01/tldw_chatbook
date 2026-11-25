"""Manual Windows UI evidence under the normal guarded pytest parent.

Select this file explicitly with -s in a real terminal. The operator drives
the shipped terminal and browser UIs. This fixture has retained audio only.
"""

import asyncio
from contextlib import contextmanager
from hashlib import sha256
import io
from math import sin, tau
import os
from pathlib import Path
import socket
from struct import pack
import subprocess
import sys
import time
import wave

import pytest


_MODULE = "Tests.UI.native_watchlist_article_listening"
_APP = "tldw_chatbook.app"
_LOOPBACK = "127.0.0.1"
_BODY = "Synthetic saved article for the listening controls."


def _retained_tone(frequency):
    from tldw_chatbook.Subscriptions.article_listening_models import (
        GeneratedArticleAudio,
    )
    from tldw_chatbook.TTS.controlled_wav_profile import PADDING_FRAMES, RATE

    seconds = 32
    one_second = b"".join(
        pack("<hh", level, level)
        for frame in range(RATE)
        for level in [int(360 * sin(tau * frequency * frame / RATE))]
    )
    payload = io.BytesIO()
    with wave.open(payload, "wb") as audio:
        audio.setparams((2, 2, RATE, 0, "NONE", ""))
        audio.writeframes(one_second * seconds + bytes(PADDING_FRAMES * 4))
    data = payload.getvalue()
    return GeneratedArticleAudio(data, sha256(data).hexdigest(), RATE * seconds)


async def _prepare_private_profile(root, *, verify_only):
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.config import (
        get_cli_config_path,
        get_cli_setting,
        get_user_data_dir,
    )
    from tldw_chatbook.Subscriptions.article_audio_assets import ArticleAudioAssets
    from tldw_chatbook.Subscriptions.article_speech import CONVERTER, prepare_saved_text
    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item

    selector, data_root = get_cli_config_path(), get_user_data_dir()
    assert selector.resolve().is_relative_to(root)
    assert data_root.resolve().is_relative_to(root)
    app = TldwCli()
    db, service = app.subscriptions_db, app.services.watchlists.listening
    assert db.db_path.resolve().is_relative_to(root) and service is not None
    assert app.services.watchlists.subscriptions_db is db
    try:
        # Same immutable selector and DB authorization as the native crash fixture.
        store = service.store() if callable(service.store) else service.store
        if verify_only:
            with store.session_lease(data_root) as lease:
                lease.recover_predecessor()
            print("Checked fixture lease and remaining player custody.", flush=True)
            return
        assert store.load_session() is None
        preferences = app.tts_service.preferences_snapshot()
        assert preferences is not None and preferences.provider_id == "openai"
        assert not get_cli_setting("app_tts", "default_profile_id", None)
        voice = await service._freeze_voice()
        assert voice.provider == "openai"
        authority = service._authorize()
        bundle = app.services.watchlists.watchlist_bundle_service
        watchlist = bundle.create("Listening fixture")["id"]
        source = db.add_subscription(
            name="Listening fixture feed",
            type="rss",
            source="https://listening.invalid/feed",
        )
        bundle.add_source(watchlist, source)
        with db.transaction() as cursor:
            ids = tuple(
                persist_subscription_item(
                    cursor,
                    source,
                    {
                        "title": f"Fixture article {index}",
                        "url": f"https://listening.invalid/{index}",
                        "content": _BODY + f" Marker {index}." if index < 3 else None,
                        "content_hash": f"native-ui-{index}",
                        "content_kind": "article",
                        "content_format": "text",
                    },
                    None,
                    "2026-11-24T08:00:00+00:00",
                )
                for index in (1, 2, 3)
            )
        briefing = db.insert_briefing(watchlist)
        db.update_briefing(
            briefing,
            status="complete",
            body_markdown="\n\n".join(f"Fixture source [item {item}]." for item in ids),
            covers_through_item_id=ids[-1],
        )
        with db.transaction() as cursor:
            cursor.executemany(
                "INSERT INTO briefing_items"
                "(briefing_id,item_id,live_item_id,source_id,cited,citation_position,provenance_version)"
                " VALUES (?,?,?,?,1,?,2)",
                (
                    (briefing, item, item, source, position)
                    for position, item in enumerate(ids)
                ),
            )
        candidate = store.prepare_brief(
            briefing,
            authority=authority,
            voice=voice,
            converter=CONVERTER,
            prepare_text=prepare_saved_text,
        )
        session = store.replace_session(candidate, expected_revision=None)
        assert [entry.item_id for entry in session.queue] == [ids[0], ids[1], None]
        for index, frequency in enumerate((440, 554)):
            entry = session.queue[index]
            claim = store.reserve_audio(
                entry.entry_id, expected_revision=session.revision
            )
            assets = ArticleAudioAssets(
                store, profile_data_root=data_root, reservation=claim
            )
            assets.publish(
                _retained_tone(frequency),
                source_item_id=ids[index],
                generation=claim.generation,
            )
            session = store.load_session()
        assert [entry.availability for entry in session.queue] == [
            "ready",
            "ready",
            "unavailable",
        ]
        assert session.transport == "paused"
        print(
            "Seeded two retained marked WAVs and one unavailable citation.", flush=True
        )
    finally:
        try:
            try:
                await service.close()
            finally:
                await app.tts_service.wait_closed()
        finally:
            db.close()


@contextmanager
def _owned_child(directory, name, args, **options):
    from tldw_chatbook.Utils.windows_job import WindowsJobApi

    api = WindowsJobApi()
    job = api.create_kill_on_close_job()
    pid_file, release = directory / f"{name}.pid", directory / f"{name}.start"
    child = None
    try:
        child = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-m",
                _MODULE,
                "launch",
                str(pid_file),
                str(release),
                *args,
            ],
            **options,
        )
        deadline = time.monotonic() + 20
        while not pid_file.exists():
            assert child.poll() is None, f"{name} launcher exited before admission"
            assert (
                time.monotonic() < deadline
            ), f"{name} launcher did not reach admission"
            time.sleep(0.02)
        # The venv redirector's Popen PID can differ from the Python PID
        # that will launch the app. The waiting process reports its own PID.
        api.assign_process(job, int(pid_file.read_text(encoding="ascii")))
        release.touch(exist_ok=False)
        yield child
    finally:
        try:
            if not api.wait_for_job_empty(job, 0):
                api.terminate_job(job)
            assert api.wait_for_job_empty(
                job, 15
            ), f"{name} process tree did not retire"
        finally:
            api.close_handle(job)
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                child.wait(15)


def _fixture_step(root, name, mode):
    with _owned_child(root, name, ["-m", _MODULE, mode, str(root)]) as child:
        assert child.wait(120) == 0, f"{name} failed"


def _serve(root, port):
    from tldw_chatbook.config import get_cli_config_path
    from tldw_chatbook.Web_Server.serve import create_server

    assert get_cli_config_path().resolve().is_relative_to(root)
    server = create_server(host=_LOOPBACK, port=port)
    stop = root / "serve.stop"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def stop_when_requested():
        if stop.exists():
            server.request_exit()
        else:
            loop.call_later(0.15, stop_when_requested)

    loop.call_soon(stop_when_requested)
    server.serve()


@pytest.mark.skipif(os.name != "nt", reason="qualified Windows playback")
@pytest.mark.timeout(0)
@pytest.mark.loopback_network
def test_watchlist_listening_terminal_and_serve(tmp_path):
    """Keep one guarded profile alive while the operator uses both real surfaces."""
    assert (
        sys.stdin.isatty() and sys.stdout.isatty()
    ), "Use the guarded runner with -s in a PTY"
    root = tmp_path.resolve()
    selector = Path(os.environ["TLDW_CONFIG_PATH"]).resolve()
    assert selector.is_relative_to(root) and Path(
        os.environ["HOME"]
    ).resolve().is_relative_to(root)
    _fixture_step(root, "prepare", "seed")
    app_args = ["-m", _APP, "--config", str(selector)]
    print(
        f"Native terminal geometry: {os.get_terminal_size()}. Stop audio and Quit to proceed.",
        flush=True,
    )
    with _owned_child(root, "terminal", [*app_args, "--no-splash"]) as terminal:
        assert terminal.wait() == 0
    _fixture_step(root, "terminal-retirement", "retired")

    with socket.socket() as port_probe:
        port_probe.bind((_LOOPBACK, 0))
        port = port_probe.getsockname()[1]
    args = ["-m", _MODULE, "serve", str(root), str(port)]
    with (
        (root / "serve.log").open("wb") as log,
        _owned_child(
            root,
            "serve",
            args,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as server,
    ):
        deadline = time.monotonic() + 90
        while True:
            assert server.poll() is None, "Serve exited before binding"
            assert time.monotonic() < deadline, "Serve did not bind"
            try:
                with socket.create_connection((_LOOPBACK, port), timeout=1):
                    break
            except OSError:
                time.sleep(0.2)
        print(f"Served browser is ready on loopback port {port}.", flush=True)
        input(
            "Stop audio and Quit in the browser, then press Enter here to retire the server. "
        )
        (root / "serve.stop").touch(exist_ok=False)
        assert server.wait(40) == 0, "Serve did not shut down normally"
    _fixture_step(root, "serve-retirement", "retired")


def _main():
    mode, *args = sys.argv[1:]
    if mode == "launch":
        pid_file, release = map(Path, args[:2])
        staged_pid = pid_file.with_suffix(".tmp")
        staged_pid.write_text(str(os.getpid()), encoding="ascii")
        staged_pid.replace(pid_file)
        deadline = time.monotonic() + 20
        while not release.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("Native UI launch was never admitted")
            time.sleep(0.02)
        return subprocess.call([sys.executable, "-B", *args[2:]])
    if mode == "serve":
        _serve(Path(args[0]), int(args[1]))
        return 0
    assert mode in {"seed", "retired"}
    asyncio.run(_prepare_private_profile(Path(args[0]), verify_only=mode == "retired"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
