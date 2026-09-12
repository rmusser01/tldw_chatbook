"""Blocking private-pipe threads with bounded owner-loop scheduling.

PipeReader takes exclusive ownership of its raw read descriptor. Its supervisor
owns the peer endpoint; closing that peer unblocks reads. PipeWriter borrows its
write callable. Join has a bounded deadline and exposes unconfirmed thread shutdown.
No thread calls application code except the supplied nonblocking scheduler/fault
sink. Neither PCM nor exception text is logged.
"""

from __future__ import annotations

from collections.abc import Callable
import os
import threading

from .voice_process_protocol import (
    Mailbox,
    ProtocolError,
    Record,
    _PREFIX,
    _header_bytes,
    read_record,
)


def write_record(
    write: Callable[[memoryview], int], record: Record, direction: str
) -> None:
    """Write one whole frame, handling EINTR/short writes without a body copy."""
    header = _header_bytes(record, direction)
    parts = (
        _PREFIX.pack(len(header), len(record.payload)),
        header,
        record.payload,
    )
    last_part = 2 if record.payload else 1
    permit = record._reservation
    for index, part in enumerate(parts):
        view = memoryview(part)
        while view:
            if permit is not None:
                permit._before_write(can_complete=index == last_part)
            try:
                count = write(view)
            except InterruptedError:
                if permit is not None:
                    permit._after_write(completed=False)
                continue
            except Exception:
                raise ProtocolError("voice_transport_failed") from None
            if type(count) is not int or not 0 < count <= len(view):
                raise ProtocolError("voice_transport_failed")
            if permit is not None:
                permit._after_write(completed=index == last_part and count == len(view))
            view = view[count:]


class PipeWriter:
    """One blocking writer; its current bounded frame keeps sender credit."""

    def __init__(
        self,
        write: Callable[[memoryview], int],
        mailbox: Mailbox,
        on_fault: Callable[[ProtocolError], None],
    ) -> None:
        if not mailbox.outbound:
            raise ProtocolError()
        self._write = write
        self.mailbox = mailbox
        self._on_fault = on_fault
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="voice-pipe-writer", daemon=True
        )
        self._condition = threading.Condition()
        self._written = 0
        self.failure: ProtocolError | None = None

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                record = self.mailbox.take(block=True)
                if record is None:
                    return
                permit = record._reservation
                try:
                    if permit is not None:
                        permit._begin_write()
                    write_record(self._write, record, self.mailbox.direction)
                except ProtocolError:
                    if permit is not None:
                        permit._fail_write()
                    raise
                finally:
                    self.mailbox.release(record)
                    del record
                if permit is not None:
                    permit._finish_write()
                del permit
                with self._condition:
                    self._written += 1
                    self._condition.notify_all()
        except ProtocolError as error:
            self.failure = ProtocolError(error.code)
            self.mailbox.close()
            try:
                self._on_fault(self.failure)
            except Exception:
                # The retained fixed category remains observable to supervision.
                pass

    def wait_written(self, count: int, timeout: float) -> bool:
        """Bounded progress observation for shutdown/transport supervision."""
        with self._condition:
            return self._condition.wait_for(lambda: self._written >= count, timeout)

    def close(self) -> None:
        self._stop.set()
        self.mailbox.close()

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)


class PipeReader:
    """One exclusively owned read fd and one scheduled bounded batch callback.

    Construction transfers the raw fd: the caller must never close it afterward.
    For a Popen/file wrapper, pass os.dup(wrapper.fileno()) and leave the original
    descriptor under that wrapper's ownership (or explicitly detach ownership).
    close() requests stop; only the running thread closes its fd in finally.
    The supervisor must close the peer to unblock a pending read before join().
    Closing before start closes immediately; repeated close/join is safe.
    """

    def __init__(
        self,
        fd: int,
        mailbox: Mailbox,
        schedule: Callable[[Callable[[], None]], None],
        consume: Callable[[Record], None],
        on_fault: Callable[[ProtocolError], None],
    ) -> None:
        self.fd = fd
        self.mailbox = mailbox
        self._schedule = schedule
        self._consume = consume
        self._on_fault = on_fault
        self._condition = threading.Condition()
        self._scheduled = False
        self._received = 0
        self._stopped = False
        self._failure: ProtocolError | None = None
        self.failure: ProtocolError | None = None
        self._fd_closed = False
        self._started = False
        self._thread = threading.Thread(
            target=self._run, name="voice-pipe-reader", daemon=True
        )

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        with self._condition:
            if self._stopped or self._started:
                raise ProtocolError("voice_transport_closed")
            self._thread.start()
            self._started = True

    def _wake(self) -> None:
        with self._condition:
            if self._scheduled:
                return
            self._scheduled = True
        try:
            self._schedule(self._drain)
        except Exception:
            with self._condition:
                self._scheduled = False
                self.failure = ProtocolError("voice_transport_failed")
            self._stopped = True
            self.mailbox.close()

    def _run(self) -> None:
        try:
            while not self._stopped:
                record = read_record(
                    lambda count: os.read(self.fd, count),
                    self.mailbox.direction,
                    admit=self.mailbox.check_admission,
                )
                self.mailbox.put(record)
                del record
                self._wake()
                with self._condition:
                    self._received += 1
                    self._condition.notify_all()
        except ProtocolError as error:
            with self._condition:
                self.failure = ProtocolError(error.code)
                self._failure = self.failure
                self._condition.notify_all()
            self._wake()
        finally:
            with self._condition:
                self._close_fd()

    def _drain(self) -> None:
        # Snapshot limits work even if the producer replenishes concurrently.
        budget = self.mailbox.queued
        try:
            for _ in range(budget):
                record = self.mailbox.take()
                if record is None:
                    break
                self._consume(record)
                self.mailbox.release_reserved(record)
                del record
            with self._condition:
                failure = None
                if self.mailbox.queued == 0:
                    failure, self._failure = self._failure, None
            if failure is not None:
                self._on_fault(failure)
        except Exception:
            self._stopped = True
            self.mailbox.close()
            self.failure = ProtocolError("voice_transport_failed")
            try:
                self._on_fault(self.failure)
            except Exception:
                pass
        finally:
            with self._condition:
                self._scheduled = False
                more = self.mailbox.queued > 0 or self._failure is not None
            if more and not self._stopped:
                self._wake()

    def wait_received(self, count: int, timeout: float) -> bool:
        """Wait for bounded receive progress without running the consumer loop."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._received >= count or self._failure is not None, timeout
            )
            return self._received >= count

    def close(self) -> None:
        """Stop admission; the read thread retains its fd until it actually exits."""
        with self._condition:
            self._stopped = True
            if not self._started:
                self._close_fd()
        self.mailbox.close()

    def _close_fd(self) -> None:
        if not self._fd_closed:
            self._fd_closed = True
            os.close(self.fd)

    def join(self, timeout: float) -> None:
        """Wait at most timeout; alive reports any still-owned blocking read."""
        with self._condition:
            started = self._started
        if started:
            self._thread.join(timeout)
