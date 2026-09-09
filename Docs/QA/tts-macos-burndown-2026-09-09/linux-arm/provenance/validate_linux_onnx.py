"""Headless Linux ARM ONNX inference/decoding evidence, never physical playback.

--preflight imports dependencies and fingerprints inputs but never initializes
an inference session. --run-inference must be explicitly selected for live work.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback
from unittest.mock import patch


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-inference", action="store_true")
    result.add_argument("--source", type=Path, default=Path("/workspace"))
    result.add_argument("--model", type=Path, default=Path("/assets/kokoro.onnx"))
    result.add_argument("--voices", type=Path, default=Path("/assets/voices.bin"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--host-output", type=Path, required=True)
    result.add_argument("--source-revision", required=True)
    result.add_argument("--phase-timeout", type=float, default=180)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    inherited_environment_names = sorted(os.environ)
    credential_names = [name for name in inherited_environment_names if any(
        part in name.upper() for part in ("TOKEN", "PASSWORD", "SECRET", "API_KEY", "ACCESS_KEY", "PRIVATE_KEY"))]
    assert not credential_names, ("Flagged inherited environment names (values withheld): "
                                  f"{credential_names}")
    if args.output.exists():
        raise ValueError("A new evidence output directory is required")
    if not args.model.is_file() or not args.voices.is_file():
        raise ValueError("Both explicit local ONNX assets must already exist")
    if not 1 <= args.phase_timeout <= 600:
        raise ValueError("phase-timeout must be between 1 and 600 seconds")
    args.output.mkdir(mode=0o700)
    specification = importlib.util.spec_from_file_location(
        "live_helpers", args.source / "scripts/validate_live_tts.py")
    helpers = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helpers)
    helpers.setup_environment(args.output)
    os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    sys.addaudithook(helpers._deny_network)
    import toml
    configuration = {
        "general": {"users_name": "linux_onnx_validation"},
        "paths": {"data_dir": str(args.output / "data")},
        "first_run": {"setup_completed": True}, "model_catalog": {"enabled": False},
        "app_tts": {"KOKORO_USE_ONNX": True,
                    "KOKORO_ONNX_MODEL_PATH_DEFAULT": str(args.model),
                    "KOKORO_ONNX_VOICES_JSON_DEFAULT": str(args.voices),
                    "KOKORO_VOICE_BLENDS_DIR": str(args.output / "blends")},
    }
    config_path = args.output / "profile/config.toml"
    config_path.write_text(toml.dumps(configuration))
    config_path.chmod(0o600)
    from loguru import logger
    logger.remove()
    logger.add(args.output / "application.log", level="DEBUG")
    import tldw_chatbook
    from tldw_chatbook import config
    from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend
    from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
    import kokoro_onnx
    import onnxruntime
    package = Path(tldw_chatbook.__file__).resolve().parent
    helpers.check_package_root(package, args.source / "tldw_chatbook")
    assert config.get_user_data_dir().resolve().is_relative_to(args.output)
    assert platform.system() == "Linux" and platform.machine() in {"aarch64", "arm64"}
    assert importlib.util.find_spec("torch") is None, "This qualification intentionally excludes Torch"
    report = {
        "schema_version": 1, "status": "running", "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Headless Linux ARM production Kokoro backend inference and complete WAV decoding. No Lab, Console, physical playback, device drain or acoustic loopback claim.",
        "physical_playback_qualified": False, "dev_snd_present": Path("/dev/snd").exists(),
        "torch_present": False, "environment_names": sorted(os.environ),
        "inherited_environment_names": inherited_environment_names,
        "credential_environment_names": credential_names,
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "source_revision_supplied": args.source_revision, "imported_package": str(package),
        "source_hashes": helpers._source_hashes(package),
        "validator_files": {str(path): helpers.sha256(path) for path in
                            (Path(__file__), args.source / "scripts/validate_live_tts.py")},
        "assets": {str(path): helpers.sha256(path) for path in (args.model, args.voices)},
        "versions": {name: importlib.metadata.version(name) for name in ("kokoro-onnx", "onnxruntime", "numpy", "soundfile", "pydub", "av")},
        "runtime_files": {str(Path(module.__file__).resolve()): helpers.sha256(Path(module.__file__))
                          for module in (kokoro_onnx, onnxruntime, sys.modules["onnxruntime.capi.onnxruntime_pybind11_state"])},
        "available_execution_providers": onnxruntime.get_available_providers(),
        "phases": [],
    }
    report["package_lists"] = {}
    for name in ("pip-freeze.txt", "dpkg-packages.txt"):
        target = args.output / name
        target.write_bytes((Path("/opt/validation") / name).read_bytes())
        report["package_lists"][name] = helpers.sha256(target)
    lock = threading.RLock()

    def checkpoint():
        with lock:
            helpers.write_evidence(args.output / "evidence.json", report)

    checkpoint()
    if args.preflight:
        report["status"] = "preflight_ready_no_session_initialized"
        checkpoint()
        print(json.dumps({key: report[key] for key in ("status", "python", "machine", "versions", "dev_snd_present")}), flush=True)
        return 0

    async def run():
        backend = KokoroTTSBackend({
            "KOKORO_USE_ONNX": True, "KOKORO_DEVICE": "cpu",
            "KOKORO_MODEL_PATH": str(args.model), "KOKORO_VOICES_JSON_PATH": str(args.voices),
            "KOKORO_VOICE_DIR_PT": str(args.output / "blends"),
            "KOKORO_VOICE_BLENDS_DIR": str(args.output / "blends"),
            "KOKORO_ENABLE_VOICE_MIXING": False,
        })
        current = [None]
        main_thread = threading.get_ident()
        original = onnxruntime.InferenceSession.run

        def native(session, *positional, **keywords):
            row = current[0]
            assert Path(session._model_path).resolve() == args.model.resolve()
            details = {"request_id": row["id"], "class": type(session).__module__ + "." + type(session).__name__,
                       "execution_providers": session.get_providers(), "model_path": str(session._model_path)}
            return helpers.observe_native(original, row["native_calls"], details, lock=lock)(session, *positional, **keywords)

        def phase(name, text):
            row = {"id": name, "expected_text": text, "content_anchors": helpers.DEFAULT_ANCHORS,
                   "outcome": "running", "native_calls": [], "started_at": time.monotonic()}
            current[0] = row
            report["phases"].append(row)
            checkpoint()
            print("PHASE", name, flush=True)
            return row

        def resources():
            return {"native_tasks": sum(not task.done() for task in backend._native_tasks),
                    "onnx_workers": sum(not task.done() for task in backend._onnx_tasks),
                    "native_calls": sum("exit_at" not in call for row in report["phases"] for call in row["native_calls"])}

        async def bounded(task):
            done, _ = await asyncio.wait({task}, timeout=args.phase_timeout)
            if not done:
                raise TimeoutError("Headless phase deadline exceeded; outer launcher must enforce its process deadline")
            return task.result()

        async def consume(row):
            path = args.output / "audio" / (row["id"] + ".wav")
            request = OpenAISpeechRequest(model="kokoro", input=row["expected_text"], voice="af_heart", response_format="wav", speed=1.0)
            with path.open("wb") as output:
                async with aclosing(backend.generate_speech_stream(request)) as source:
                    async for chunk in source:
                        output.write(chunk)
                        if output.tell() > 64 * 1024 * 1024:
                            raise ValueError("Validation audio capture exceeds its bound")
            audio = helpers.inspect_wav(path)
            audio.update(container_path=audio["path"], path=str(args.host_output / path.relative_to(args.output)))
            row.update(audio=audio, settled_at=time.monotonic(), resources=resources())
            helpers.assert_quiescent(row["resources"])
            assert row["native_calls"] and all(call["thread_id"] != main_thread for call in row["native_calls"])
            row["outcome"] = "success"
            checkpoint()

        with patch.object(onnxruntime.InferenceSession, "run", native):
            try:
                await backend.initialize()
                assert backend.use_onnx and backend.kokoro_instance is not None
                report["active_execution_providers"] = backend.kokoro_instance.sess.get_providers()
                warmup = phase("linux-warmup", helpers.DEFAULT_TEXT)
                await bounded(asyncio.create_task(consume(warmup)))
                cancelled = phase("linux-cancel", "\n".join([helpers.DEFAULT_TEXT] * 4))
                operation = asyncio.create_task(consume(cancelled))
                deadline = time.monotonic() + args.phase_timeout
                while not any("exit_at" not in call for call in cancelled["native_calls"]):
                    if operation.done():
                        raise ValueError("Request completed before a live native interval was observed")
                    if time.monotonic() > deadline:
                        raise TimeoutError("No actual ONNX session.run interval observed")
                    await asyncio.sleep(0.005)
                await asyncio.sleep(0.01)
                with lock:
                    assert any("exit_at" not in call for call in cancelled["native_calls"]), "Inference finished before Stop"
                    cancelled.update(stop_requested_at=time.monotonic(), resources_at_stop=resources())
                operation.cancel()
                try:
                    await bounded(operation)
                except asyncio.CancelledError:
                    pass
                else:
                    raise ValueError("The cancelled generation returned success")
                cancelled.update(settled_at=time.monotonic(), resources=resources())
                helpers.validate_cancellation(cancelled)
                helpers.assert_quiescent(cancelled["resources"])
                partial = args.output / "audio" / "linux-cancel.wav"
                cancelled["partial_bytes"] = partial.stat().st_size
                assert cancelled["partial_bytes"] == 0, "Cancelled complete-WAV inference emitted bytes"
                cancelled["outcome"] = "cancelled"
                frozen = json.dumps(cancelled, sort_keys=True)
                successor = phase("linux-successor", helpers.DEFAULT_TEXT)
                await bounded(asyncio.create_task(consume(successor)))
                assert frozen == json.dumps(cancelled, sort_keys=True), "Old request changed after successor"
                report["status"] = "headless_runtime_passed_content_pending"
            finally:
                await bounded(asyncio.create_task(backend.close()))
                report["final_resources"] = resources()
                helpers.assert_quiescent(report["final_resources"])
                assert backend.kokoro_instance is None and backend._close_task.done()
                report["cleanup_joined"] = True
                report["source_hashes_after"] = helpers._source_hashes(package)
                report["source_unchanged"] = report["source_hashes_after"] == report["source_hashes"]
                assert report["source_unchanged"], "Source changed during the qualification"
                checkpoint()

    try:
        asyncio.run(run())
        return 0
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1
    finally:
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        checkpoint()


if __name__ == "__main__":
    raise SystemExit(main())
