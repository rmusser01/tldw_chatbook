"""Opt-in native session witness for the private guarded pytest parent."""

import asyncio
from dataclasses import replace
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
import wave


def notify(path, **values):
    stage = path.with_suffix(".tmp")
    stage.write_text(json.dumps(values), encoding="utf-8")
    stage.replace(path)


async def exercise(root, mode):
    from loguru import logger

    logger.remove()
    logging.disable(logging.CRITICAL)
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Subscriptions.article_listening_models import (
        FrozenVoice,
        GeneratedArticleAudio,
        ListeningCheckpoint,
    )
    from tldw_chatbook.Subscriptions.article_speech import CONVERTER, prepare_saved_text
    from tldw_chatbook.Subscriptions.article_audio_assets import ArticleAudioAssets
    from tldw_chatbook.config import get_user_data_dir
    from tldw_chatbook.TTS.audio_output_ownership import AudioOutputOwnership
    from tldw_chatbook.Subscriptions.article_listening_lease import ListeningLease

    app = TldwCli()
    service = app.services.watchlists.listening
    db = app.subscriptions_db
    if not db.db_path.resolve().is_relative_to(
        root
    ) or not get_user_data_dir().resolve().is_relative_to(root):
        raise ValueError("probe_profile_unavailable")
    await service.restore()
    if mode == "seed":
        store = service.store
        with db.transaction() as c:
            source = c.execute(
                "INSERT INTO subscriptions(name,type,source) VALUES ('Synthetic','rss','local')"
            ).lastrowid
            item = c.execute(
                "INSERT INTO subscription_items(subscription_id,url,title,content) "
                "VALUES (?,'urn:synthetic','Synthetic','Stored synthetic text')",
                (source,),
            ).lastrowid
            watch = c.execute(
                "INSERT INTO watchlists(name) VALUES ('Synthetic')"
            ).lastrowid
            c.execute(
                "INSERT INTO watchlist_sources(watchlist_id,subscription_id) VALUES (?,?)",
                (watch, source),
            )
            brief = c.execute(
                "INSERT INTO briefings(watchlist_id,status,body_markdown) VALUES (?,'complete','')",
                (watch,),
            ).lastrowid
            c.execute(
                "INSERT INTO briefing_items(briefing_id,item_id,live_item_id,source_id,cited,citation_position,provenance_version) "
                "VALUES (?,?,?,?,1,0,2)",
                (brief, item, item, source),
            )
        candidate = store.prepare_item(
            item,
            authority=service._authorize(),
            voice=FrozenVoice("openai", "tts-1", "alloy"),
            converter=CONVERTER,
            prepare_text=prepare_saved_text,
        )
        queue = store.prepare_brief(
            brief,
            authority=service._authorize(),
            voice=candidate.voice,
            converter=CONVERTER,
            prepare_text=prepare_saved_text,
        )
        session = store.replace_session(
            replace(queue, foreground=candidate.foreground), expected_revision=None
        )
        frames = 8 * 48000
        dest = io.BytesIO()
        with wave.open(dest, "wb") as wav:
            wav.setparams((2, 2, 48000, 0, "NONE", ""))
            wav.writeframes(bytes(frames * 4 + 12000 * 4))
        data = dest.getvalue()
        audio = GeneratedArticleAudio(data, hashlib.sha256(data).hexdigest(), frames)
        claim = store.reserve_audio(
            session.foreground.entry_id, expected_revision=session.revision
        )
        assets = ArticleAudioAssets(
            store, profile_data_root=get_user_data_dir(), reservation=claim
        )
        assets.publish(audio, source_item_id=item, generation=claim.generation)
        session = store.load_session()
        store.checkpoint(
            ListeningCheckpoint(
                session.authority,
                session.incarnation,
                session.playback_generation,
                0,
                True,
                "paused",
            ),
            expected_revision=session.revision,
        )
        notify(root / "seed.json", frames=frames)
    else:
        initial = service.snapshot()
        if initial.phase != "paused" or service._active or service._audio is not None:
            raise ValueError("probe_restore_failed")
        saved_session = service.store.load_session()
        if (
            not saved_session.foreground
            or not saved_session.resume_queue
            or saved_session.cursor != 0
        ):
            raise ValueError("probe_restore_failed")
        old = saved_session.foreground
        point = old.bookmark
        notify(
            root / (mode + "-restore.json"),
            frame=point.frame,
            generation=old.artifact,
            sha256=point.sha256,
            rate=point.sample_rate,
            entry=old.entry_id,
        )
        original = ListeningLease.seal_predecessor

        def seal(holder, pid, created):
            original(holder, pid, created)
            notify(
                root / (mode + "-birth.json"),
                pid=pid,
                created=created,
                clock=time.perf_counter(),
            )
            if mode == "suspended":
                # Pause only this test's observer after the REAL durable seal.
                # The native creation flags and real resume/stop gates are intact.
                while True:
                    time.sleep(0.02)
            if mode == "startup-stop":
                if not service._active.run._stop.wait(4):
                    raise ValueError("probe_stop_missed_birth")

        ListeningLease.seal_predecessor = seal
        acquire_output = AudioOutputOwnership.acquire

        async def observe_grant(owner, *args, **kwargs):
            lease = await acquire_output(owner, *args, **kwargs)
            notify(root / (mode + "-grant.json"), clock=time.perf_counter())
            return lease

        AudioOutputOwnership.acquire = observe_grant
        await service.resume()
        async with asyncio.timeout(20):
            while True:
                snap = service.snapshot()
                if snap.error:
                    raise ValueError(snap.error)
                if mode == "startup-stop" and (root / "stop-now").exists():
                    await service.stop()
                    saved = service.store.load_session()
                    if (
                        not saved.foreground
                        or not saved.resume_queue
                        or app.audio_output.owner is not None
                    ):
                        raise ValueError("probe_stop_failed")
                    notify(
                        root / "stopped.json",
                        phase=service.snapshot().phase,
                        frame=saved.foreground.bookmark.frame,
                    )
                    break
                if snap.phase == "playing":
                    saved = (
                        await asyncio.to_thread(service.store.load_session)
                    ).foreground.bookmark.frame
                    if mode == "resume" and (snap.elapsed_frames or 0) > point.frame:
                        notify(
                            root / "resumed.json",
                            frame=saved,
                            observed=snap.elapsed_frames,
                        )
                        break
                    if (
                        mode == "playing"
                        and saved > 0
                        and (snap.elapsed_frames or 0) > saved + 200
                    ):
                        notify(
                            root / "checkpoint.json",
                            saved=saved,
                            observed=snap.elapsed_frames,
                        )
                        await asyncio.Event().wait()
                await asyncio.sleep(0.015)
        if mode == "resume":
            await service.stop()
    await service.close()
    await app._shutdown_app_owned_lifecycles()
    await app.tts_service.close()
    await app.tts_service.wait_closed()


if __name__ == "__main__":
    root, mode = Path(sys.argv[1]).resolve(strict=True), sys.argv[2]
    if mode not in {
        "seed",
        "playing",
        "suspended",
        "startup-stop",
        "resume",
    } or not all(
        Path(os.environ[key]).resolve().is_relative_to(root)
        for key in ("HOME", "TLDW_CONFIG_PATH")
    ):
        raise SystemExit(2)
    try:
        asyncio.run(exercise(root, mode))
    except Exception as error:
        code = (
            error.args[0]
            if error.args
            and error.args[0]
            in (
                "probe_profile_unavailable",
                "probe_restore_failed",
                "local_listening_authority_required",
                "listening_unavailable",
                "previous_player_unconfirmed",
                "player_containment_unavailable",
                "listening_busy",
                "storage_admission_unavailable",
                "unsupported_platform",
                "qualified_ffplay_unavailable",
                "article_audio_missing",
                "progress_unavailable",
            )
            else "probe_failed"
        )
        notify(root / "failure.json", category=type(error).__name__, code=code)
        raise SystemExit(1) from None
