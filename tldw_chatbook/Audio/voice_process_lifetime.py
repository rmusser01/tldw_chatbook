"""Dependency-light identity, deadlines and private process ownership.

This module never loads a device, model, app configuration or provider. Windows
containment is parent-only; the child does not call OwnedProcessTree.attach().
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import signal
import time

from .voice_process_io import PipeReader, PipeWriter
from .voice_process_protocol import (
    CreditWindow,
    Mailbox,
    ProtocolError,
    ReceiveStream,
    Record,
    StreamKey,
    _BUDGETS,
    _validate_header,
)


HANDSHAKE_SECONDS = 5.0
STT_READY_SECONDS = 120.0
LEASE_SECONDS = 5.0
RENEW_SECONDS = 0.25
NATIVE_CLOSE_SECONDS = 2.0
GRACE_SECONDS = 6.0
TERMINATE_SECONDS = 2.0
KILL_SECONDS = 2.0

_RUNTIME_SOURCE_PATHS = (
    "__init__.py",
    "Audio/__init__.py",
    "Audio/acoustic_isolation.py",
    "Audio/aec_backend.py",
    "Audio/duplex_contracts.py",
    "Audio/duplex_transport.py",
    "Audio/native_duplex_stream.py",
    "Audio/parakeet_voice_worker.py",
    "Audio/rolling_transcript.py",
    "Audio/voice_phrase_sequencer.py",
    "Audio/voice_preprocessor.py",
    "Audio/voice_process_core.py",
    "Audio/voice_process_entry.py",
    "Audio/voice_process_io.py",
    "Audio/voice_process_lifetime.py",
    "Audio/voice_process_protocol.py",
    "Audio/voice_process_types.py",
    "Audio/voice_transcription.py",
    "Audio/voice_turn_coordinator.py",
    "Chat/__init__.py",
    "Chat/console_provider_gateway.py",
    "Chat/console_speculative_voice.py",
    "Chat/console_speculative_voice_session.py",
    "Chat/console_turn_context.py",
    "Chat/console_voice_attempts.py",
    "Chat/console_voice_eligibility.py",
    "Chat/console_voice_preflight.py",
    "Chat/console_voice_process.py",
    "Chat/console_voice_process_effects.py",
    "Chat/console_voice_promotion.py",
    "Chat/console_voice_supervisor.py",
    "Chat/console_voice_trace_gateway.py",
    "Chat/console_voice_trace_promotion.py",
    "Chat/console_voice_tts_bridge.py",
    "Chat/console_voice_worker.py",
    "Chat/voice_phrase_sequencer.py",
    "Local_Ingestion/__init__.py",
    "Local_Ingestion/transcription_service.py",
    "STT/__init__.py",
    "STT/contracts.py",
    "STT/executor.py",
    "STT/executor_process_tree.py",
    "STT/executor_worker.py",
    "STT/parakeet_dispatch.py",
    "STT/parakeet_onnx.py",
    "TTS/__init__.py",
    "TTS/adapter_types.py",
    "TTS/audio_cpp_contract.py",
    "TTS/pcm_stream.py",
    "Utils/__init__.py",
    "Utils/fd_protection.py",
    "Utils/local_stt_providers.py",
    "Utils/persistent_diagnostics.py",
)


class VoiceProcessError(RuntimeError):
    """Content-free failure of the private process boundary."""

    def __init__(self, code: str = "startup_failed") -> None:
        if code not in {
            "startup_failed",
            "stale",
            "lease_expired",
            "transport_failed",
            "protocol_invalid",
            "shutdown_unconfirmed",
        }:
            code = "startup_failed"
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class SourceIdentity:
    root: str
    source: str
    native_abi: int = 1
    version: int = 1


def source_identity() -> SourceIdentity:
    """Fingerprint actual imported core sources in a checkout or installed wheel.

    Root is attestation only. No child uses the bootstrap root to select files.
    The native ABI here is the required ABI; native composition must also check
    the loaded bridge before opening a device.
    """
    from . import voice_process_core

    root = Path(voice_process_core.__file__).resolve().parents[2]
    if Path(__file__).resolve().parents[2] != root:
        raise VoiceProcessError()
    digest = hashlib.sha256(b"voice-process:protocol=1:native=1\0")
    try:
        for name in _RUNTIME_SOURCE_PATHS:
            digest.update(name.encode() + b"\0")
            digest.update((root / "tldw_chatbook" / name).read_bytes())
            digest.update(b"\0")
    except OSError:
        raise VoiceProcessError() from None
    return SourceIdentity(str(root), digest.hexdigest())


def child_environment() -> dict[str, str]:
    """Allow OS/runtime and local-cache facts, never provider secrets/config.

    PYTHONPATH/PYTHONHOME and credential-store variables are deliberately absent;
    the current interpreter resolves the package from the attested working root.
    """
    names = (
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
    )
    result = {name: os.environ[name] for name in names if name in os.environ}
    result["PYTHONUNBUFFERED"] = "1"
    return result


class ChildLease:
    """A child's own monotonic lease; expiry is sticky, including late renewal."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.deadline = clock() + LEASE_SECONDS
        self.expired = False

    @property
    def alive(self) -> bool:
        self.expired |= self.clock() >= self.deadline
        return not self.expired

    def renew(self) -> bool:
        if not self.alive:
            return False
        self.deadline = self.clock() + LEASE_SECONDS
        return True


class StartupGate:
    """Single-use hello, STT readiness and current-activation permit."""

    def __init__(
        self, identity: SourceIdentity, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.identity = identity
        self.clock = clock
        self.deadline = clock() + HANDSHAKE_SECONDS
        self.verified = False
        self.prepared = False
        self.permitted = False
        self.stopped = False

    def hello(self, identity: SourceIdentity) -> None:
        if (
            self.stopped
            or self.verified
            or self.clock() >= self.deadline
            or identity != self.identity
        ):
            self.stopped = True
            raise VoiceProcessError()
        self.verified = True
        self.deadline = self.clock() + STT_READY_SECONDS

    def stt_ready(self) -> None:
        if (
            self.stopped
            or not self.verified
            or self.prepared
            or self.clock() >= self.deadline
        ):
            self.stopped = True
            raise VoiceProcessError()
        self.prepared = True

    def permit(self, *, current: bool) -> None:
        if self.stopped or not self.prepared or self.permitted or not current:
            self.stopped = True
            raise VoiceProcessError("stale" if not current else "startup_failed")
        self.permitted = True


class OwnedProcessTree:
    """Retain an exact POSIX group or Windows Job until absence is observed."""

    def __init__(
        self, pid: int, *, platform: str | None = None, windows_api=None
    ) -> None:
        self.pid = pid
        self.platform = platform or os.name
        self.api = windows_api
        self.job = None
        self.attached = False
        self.absent = False

    def attach(self) -> None:
        try:
            if self.platform == "posix":
                if (
                    self.pid <= 0
                    or os.getpgid(self.pid) != self.pid
                    or self.pid == os.getpgrp()
                ):
                    raise VoiceProcessError()
            elif self.platform == "nt":
                if self.api is None:
                    # Parent-only lazy reuse; importing this module in the audio
                    # child never imports the STT package or ctypes controller.
                    from tldw_chatbook.STT.executor_process_tree import _WindowsJobApi

                    self.api = _WindowsJobApi()
                self.job = self.api.create_kill_on_close_job()
                self.api.assign_process(self.job, self.pid)
            else:
                raise VoiceProcessError()
        except Exception:
            if self.job is not None:
                self.api.close_handle(self.job)
                self.job = None
            raise VoiceProcessError() from None
        self.attached = True

    def observe_absence(self) -> bool:
        if self.absent:
            return True
        if not self.attached:
            return False
        try:
            if self.platform == "posix":
                os.killpg(self.pid, 0)
                return False
            if not self.api.wait_for_job_empty(self.job, 0):
                return False
            self.api.close_handle(self.job)
            self.job = None
        except ProcessLookupError:
            if self.platform != "posix":
                return False
        except OSError:
            return False
        self.absent = True
        return True

    def terminate(self, *, force: bool = False) -> None:
        if self.absent or not self.attached:
            return
        try:
            if self.platform == "posix":
                os.killpg(self.pid, signal.SIGKILL if force else signal.SIGTERM)
            else:
                self.api.terminate_job(self.job)
        except OSError:
            pass


class _SessionMailbox(Mailbox):
    def check_admission(self, header, payload_length):
        _validate_header(dict(header), payload_length, self.direction)
        if header["op"] in {"prepare", "preview", "draft"}:
            self.retain_turn(header["turn_id"])
        super().check_admission(header, payload_length)


class LifecyclePipe:
    """Bounded lifecycle/control transport with explicit retained data custody.

    The read fd transfers to PipeReader. The write fd stays owned by the caller
    until the writer exits. Fixed startup permits wait for the first ordinary
    cumulative control receipt. Credit follows consumption, never writes.
    """

    def __init__(
        self,
        read_fd: int,
        write_fd: int,
        *,
        generation: int,
        request_id: str,
        parent: bool,
        consume: Callable[[Record], None],
        fault: Callable[[ProtocolError], None],
        initial_record: Record | None = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        direction = "parent_to_child" if parent else "child_to_parent"
        self.generation = generation
        self.request_id = request_id
        self.output = Mailbox(direction, generation=generation, outbound=True)
        self.input = _SessionMailbox(
            "child_to_parent" if parent else "parent_to_child", generation=generation
        )
        self.window = CreditWindow(StreamKey(generation, request_id, "control"))
        self._windows = {"control": self.window}
        self._receivers = {}
        self._deliveries = {}
        self.sequence = 0

        def receive(record):
            if record.header["request_id"] != request_id:
                raise ProtocolError()
            lane = self.input._lane(record.header)
            if lane in _BUDGETS and lane not in {"provider", "pcm", "tts_closed"}:
                key = self._key(lane, record.header)
                receiver = self._receivers.setdefault(key, ReceiveStream(key))
                self._deliveries[id(record)] = receiver.receive(
                    record, mailbox=self.input
                )
            if (
                record.header["op"] == "credit"
                and record.header["lane"] in self._windows
            ):
                self._windows[record.header["lane"]].accept_credit(record)
                self.release(record)
            elif consume(record) is not True:
                self.release(record)

        self.reader = PipeReader(
            read_fd,
            self.input,
            loop.call_soon_threadsafe,
            receive,
            fault,
        )
        self.writer = PipeWriter(
            lambda data: os.write(write_fd, data),
            self.output,
            lambda error: loop.call_soon_threadsafe(fault, ProtocolError(error.code)),
        )
        if initial_record is not None:
            # Seed the already-read bootstrap before priority traffic can enter
            # the mailbox and displace it in take().
            self.input.put(initial_record)
            initial = self.input.take()
            key = self._key("control", initial.header)
            receiver = self._receivers.setdefault(key, ReceiveStream(key))
            self._deliveries[id(initial)] = receiver.receive(
                initial, mailbox=self.input
            )
            self.release(initial)
        self.reader.start()
        self.writer.start()

    def _key(self, lane, header):
        return StreamKey(
            self.generation,
            self.request_id,
            lane,
            header.get("turn_id", "") if lane.startswith("draft_") else "",
        )

    async def wait_draft_consumed(self, turn_id):
        """Observe original draft credit before proposing its terminal revision."""
        while any(
            window.key.turn_id == turn_id and window.outstanding != (0, 0)
            for window in self._windows.values()
        ):
            await asyncio.sleep(0.001)

    def retire_turn(self, turn_id):
        """Retire both fixed turn slots only after actual delivery/credit custody."""
        if any(
            window.key.turn_id == turn_id and window.outstanding != (0, 0)
            for window in self._windows.values()
        ):
            raise ProtocolError("voice_capacity_exceeded")
        self.input.retire_turn(turn_id)
        self.output.retire_turn(turn_id)
        for key in tuple(self._receivers):
            if key.turn_id == turn_id:
                self._receivers.pop(key)
        for lane, window in tuple(self._windows.items()):
            if window.key.turn_id == turn_id:
                self._windows.pop(lane)

    def credit(self, key, sequence, size):
        self.send(
            "credit",
            lane=key.lane,
            stream_id=key.request_id,
            stream_turn=key.turn_id,
            stream_revision=key.revision,
            stream_epoch=key.epoch,
            stream_phrase=key.phrase_id,
            ack_sequence=sequence,
            ack_bytes=size,
        )

    def release(self, record):
        """Release only after synchronous use or an explicitly retained consumer."""
        delivery = self._deliveries.pop(id(record), None)
        if delivery is None:
            self.input.release(record)
        else:
            key = delivery._owner.key
            sequence, size = delivery.consume()
            if record.header["op"] not in {
                "bootstrap",
                "hello",
                "stt_ready",
                "start",
                "session_ready",
            }:
                self.credit(key, sequence, size)

    def send_record(self, record):
        header = record.header
        if (header["generation"], header["request_id"]) != (
            self.generation,
            self.request_id,
        ):
            raise ProtocolError()
        self.send(
            header["op"],
            payload=record.payload,
            **{
                k: v
                for k, v in header.items()
                if k not in {"version", "op", "generation", "request_id", "sequence"}
            },
        )

    def send(self, op: str, *, payload=b"", **fields) -> None:
        self.sequence += 1
        permit = None
        sequence = self.sequence
        if op in {"prepare", "preview", "draft"}:
            self.output.retain_turn(fields["turn_id"])
        if op == "draft":
            slot = self.output._turns.index(fields["turn_id"])
            if "draft_slot" in fields and (
                type(fields["draft_slot"]) is not int or fields["draft_slot"] != slot
            ):
                raise ProtocolError()
            fields["draft_slot"] = slot
        lane = self.output._lane({"op": op, **fields})
        if lane in _BUDGETS:
            key = self._key(lane, fields)
            window = self._windows.get(lane)
            if window is None or window.key != key:
                if window is not None and window.outstanding != (0, 0):
                    raise ProtocolError("voice_capacity_exceeded")
                window = self._windows[lane] = CreditWindow(key)
            permit = window.try_reserve()
            if permit is None:
                raise ProtocolError("voice_capacity_exceeded")
            sequence = permit.sequence
        record = Record(
            dict(
                version=1,
                op=op,
                generation=self.generation,
                request_id=self.request_id,
                sequence=sequence,
                **fields,
            ),
            payload,
        )
        if permit is not None:
            permit.publish(record)
        self.output.put(record)

    def stop(self) -> None:
        self.reader.close()
        self.writer.close()

    async def join(self) -> bool:
        # Threads retain their own descriptors if peer death was not confirmed.
        await asyncio.to_thread(self.writer.join, 0.25)
        await asyncio.to_thread(self.reader.join, 0.25)
        return not self.writer.alive and not self.reader.alive
