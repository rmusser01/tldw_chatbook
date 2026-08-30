"""Bounded byte-level gate in front of the persistent-terminal parser.

The gate deliberately knows less than the terminal emulator. Its job is to
bound every control sequence before a third-party parser sees it, discard host
integration controls, and expose only content-free diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass


MAX_CONTROL_BYTES = 4 * 1024
MAX_CSI_PARAMETERS = 32
MAX_CSI_PARAMETER_DIGITS = 4
MAX_CSI_PARAMETER_VALUE = 9_999
MAX_CSI_PRIVATE_INTERMEDIATE_BYTES = 16
MAX_ESCAPE_BYTES = 16

_C0_ADMITTED = frozenset(b"\a\b\t\n\v\f\r\x0e\x0f")
_STRING_INTRODUCERS = frozenset((ord("]"), ord("P"), ord("^"), ord("_")))
_C1_STRING_INTRODUCERS = frozenset((0x90, 0x9D, 0x9E, 0x9F))
_CSI_ONE_PARAMETER_FINALS = frozenset(b"@ABCDEFGJKLM PXadeg".replace(b" ", b""))
_CSI_TWO_PARAMETER_FINALS = frozenset(b"Hfr")
_CSI_VARIADIC_FINALS = frozenset(b"hlm")
_KNOWN_SIMPLE_ESCAPES = frozenset(
    (
        b"\x1b7",  # save cursor
        b"\x1b8",  # restore cursor
        b"\x1bD",  # index
        b"\x1bE",  # next line
        b"\x1bH",  # horizontal tab set
        b"\x1bM",  # reverse index
        b"\x1bc",  # reset
        b"\x1b=",  # application keypad
        b"\x1b>",  # normal keypad
    )
)
_CHARSET_INTERMEDIATES = frozenset(b"#()*+-./%")


@dataclass(frozen=True, slots=True)
class ProtocolGateSnapshot:
    """Content-free diagnostic state for the terminal protocol gate."""

    buffered_bytes: int = 0
    discarding: bool = False
    rejected_sequences: int = 0
    ignored_sequences: int = 0


class TerminalProtocolGate:
    """Incrementally admit bounded terminal bytes for the VT parser."""

    _TEXT = "text"
    _ESCAPE = "escape"
    _CSI = "csi"
    _STRING = "string"
    _DISCARD_ESCAPE = "discard_escape"
    _DISCARD_CSI = "discard_csi"
    _DISCARD_STRING = "discard_string"

    def __init__(self) -> None:
        self._state = self._TEXT
        self._buffer = bytearray()
        self._control_bytes = 0
        self._raw_c1 = False
        self._utf8_remaining = 0
        self._csi_parameters = 1
        self._csi_parameter_digits = 0
        self._csi_parameter_value = 0
        self._csi_private_intermediate_bytes = 0
        self._string_escape_pending = False
        self._string_utf8_remaining = 0
        self._discard_escape_pending = False
        self._rejected_sequences = 0
        self._ignored_sequences = 0

    def feed(self, data: bytes) -> bytes:
        """Return parser-admitted bytes from one output chunk.

        Bytes belonging to a control sequence are withheld until the sequence
        is complete and proven bounded. Once a limit is crossed, retained
        bytes are erased immediately and input is consumed through the
        sequence terminator.
        """
        admitted = bytearray()
        for byte in data:
            self._consume(byte, admitted)
        return bytes(admitted)

    def snapshot(self) -> ProtocolGateSnapshot:
        """Return content-free bounded parser state."""
        return ProtocolGateSnapshot(
            buffered_bytes=len(self._buffer),
            discarding=self._state.startswith("discard_"),
            rejected_sequences=self._rejected_sequences,
            ignored_sequences=self._ignored_sequences,
        )

    def finish(self) -> ProtocolGateSnapshot:
        """Discard any incomplete sequence and return final state."""
        if self._state != self._TEXT:
            if not self._state.startswith("discard_"):
                self._rejected_sequences += 1
            self._reset_sequence()
        return self.snapshot()

    def _consume(self, byte: int, admitted: bytearray) -> None:
        if self._state == self._TEXT:
            self._consume_text(byte, admitted)
        elif self._state == self._ESCAPE:
            self._consume_escape(byte, admitted)
        elif self._state == self._CSI:
            self._consume_csi(byte, admitted)
        elif self._state == self._STRING:
            self._consume_string(byte, admitted)
        elif self._state == self._DISCARD_ESCAPE:
            self._consume_discard_escape(byte, admitted)
        elif self._state == self._DISCARD_CSI:
            self._consume_discard_csi(byte, admitted)
        else:
            self._consume_discard_string(byte, admitted)

    def _consume_text(self, byte: int, admitted: bytearray) -> None:
        # A C1-valued byte is ordinary text when it is a continuation byte in
        # a UTF-8 sequence. Standalone C1 bytes retain their control meaning.
        if self._utf8_remaining:
            if 0x80 <= byte <= 0xBF:
                admitted.append(byte)
                self._utf8_remaining -= 1
                return
            self._utf8_remaining = 0
            self._consume_text(byte, admitted)
            return

        if 0xC2 <= byte <= 0xDF:
            self._utf8_remaining = 1
            admitted.append(byte)
            return
        if 0xE0 <= byte <= 0xEF:
            self._utf8_remaining = 2
            admitted.append(byte)
            return
        if 0xF0 <= byte <= 0xF4:
            self._utf8_remaining = 3
            admitted.append(byte)
            return

        if byte == 0x1B:
            self._start_escape()
        elif byte == 0x9B:
            self._start_csi(raw_c1=True)
        elif byte in _C1_STRING_INTRODUCERS:
            self._start_string(byte, raw_c1=True)
        elif 0x80 <= byte <= 0x9F:
            self._ignored_sequences += 1
        elif byte < 0x20 or byte == 0x7F:
            if byte in _C0_ADMITTED:
                admitted.append(byte)
        else:
            admitted.append(byte)

    def _start_escape(self) -> None:
        self._state = self._ESCAPE
        self._buffer = bytearray((0x1B,))
        self._raw_c1 = False

    def _start_csi(self, *, raw_c1: bool) -> None:
        self._state = self._CSI
        self._buffer = bytearray((0x9B,)) if raw_c1 else bytearray(b"\x1b[")
        self._raw_c1 = raw_c1
        self._csi_parameters = 1
        self._csi_parameter_digits = 0
        self._csi_parameter_value = 0
        self._csi_private_intermediate_bytes = 0

    def _start_string(self, introducer: int, *, raw_c1: bool = False) -> None:
        self._state = self._STRING
        self._buffer.clear()
        self._control_bytes = 1 if raw_c1 else 2
        self._raw_c1 = raw_c1
        self._string_escape_pending = False
        self._string_utf8_remaining = 0

    def _consume_escape(self, byte: int, admitted: bytearray) -> None:
        if len(self._buffer) == 1:
            if byte == ord("["):
                self._start_csi(raw_c1=False)
                return
            if byte in _STRING_INTRODUCERS:
                self._start_string(byte)
                return

        self._buffer.append(byte)
        is_final = 0x30 <= byte <= 0x7E
        if len(self._buffer) > MAX_ESCAPE_BYTES:
            self._reject_and_discard(self._TEXT if is_final else self._DISCARD_ESCAPE)
            return

        if byte in (0x18, 0x1A):
            self._ignored_sequences += 1
            self._reset_sequence()
        elif is_final:
            sequence = bytes(self._buffer)
            if self._is_known_escape(sequence):
                admitted.extend(sequence)
            else:
                self._ignored_sequences += 1
            self._reset_sequence()
        elif not 0x20 <= byte <= 0x2F:
            self._ignored_sequences += 1
            self._reset_sequence()

    @staticmethod
    def _is_known_escape(sequence: bytes) -> bool:
        if sequence in _KNOWN_SIMPLE_ESCAPES:
            return True
        return (
            len(sequence) == 3
            and sequence[1] in _CHARSET_INTERMEDIATES
            and 0x30 <= sequence[2] <= 0x7E
        )

    def _consume_csi(self, byte: int, admitted: bytearray) -> None:
        self._buffer.append(byte)
        is_final = 0x40 <= byte <= 0x7E

        crossed_limit = len(self._buffer) > MAX_CONTROL_BYTES
        if 0x30 <= byte <= 0x39:
            self._csi_parameter_digits += 1
            self._csi_parameter_value = self._csi_parameter_value * 10 + byte - 0x30
            crossed_limit = crossed_limit or (
                self._csi_parameter_digits > MAX_CSI_PARAMETER_DIGITS
                or self._csi_parameter_value > MAX_CSI_PARAMETER_VALUE
            )
        elif byte in (ord(";"), ord(":")):
            self._csi_parameters += 1
            self._csi_parameter_digits = 0
            self._csi_parameter_value = 0
            crossed_limit = crossed_limit or (self._csi_parameters > MAX_CSI_PARAMETERS)
        elif 0x20 <= byte <= 0x2F or 0x3C <= byte <= 0x3F:
            self._csi_private_intermediate_bytes += 1
            crossed_limit = crossed_limit or (
                self._csi_private_intermediate_bytes
                > MAX_CSI_PRIVATE_INTERMEDIATE_BYTES
            )

        if crossed_limit:
            self._reject_and_discard(self._TEXT if is_final else self._DISCARD_CSI)
            return

        if byte in (0x18, 0x1A):
            self._ignored_sequences += 1
            self._reset_sequence()
        elif is_final:
            sequence = bytes(self._buffer)
            if not self._raw_c1 and self._is_known_csi(sequence):
                admitted.extend(sequence)
            else:
                self._ignored_sequences += 1
            self._reset_sequence()
        elif not (0x20 <= byte <= 0x3F):
            self._ignored_sequences += 1
            self._reset_sequence()

    @staticmethod
    def _is_known_csi(sequence: bytes) -> bool:
        if len(sequence) < 3:
            return False

        body = sequence[2:-1]
        # pyte 0.8.2 does not parse colon-form subparameters. Withholding the
        # whole operation prevents the unparsed suffix becoming visible text.
        if b":" in body:
            return False
        private = b""
        if body and body[0] in b"<=>?":
            private, body = body[:1], body[1:]
        if any(byte not in b"0123456789;" for byte in body):
            return False

        parameters = tuple(int(value or b"0") for value in body.split(b";"))
        final = sequence[-1]
        if private:
            return private == b"?" and final in b"hl"
        if final in _CSI_ONE_PARAMETER_FINALS:
            return len(parameters) == 1
        if final in _CSI_TWO_PARAMETER_FINALS:
            return len(parameters) <= 2
        if final in _CSI_VARIADIC_FINALS:
            return True
        if final == ord("c"):
            return len(parameters) == 1 and parameters[0] == 0
        if final == ord("n"):
            return len(parameters) == 1 and parameters[0] in (5, 6)
        return False

    def _consume_string(self, byte: int, admitted: bytearray) -> None:
        self._control_bytes += 1
        if self._string_utf8_remaining:
            if 0x80 <= byte <= 0xBF:
                self._string_utf8_remaining -= 1
                self._discard_string_if_oversized()
                return
            self._string_utf8_remaining = 0
            self._control_bytes -= 1
            self._consume_string(byte, admitted)
            return
        if 0xC2 <= byte <= 0xDF:
            self._string_escape_pending = False
            self._string_utf8_remaining = 1
            self._discard_string_if_oversized()
            return
        if 0xE0 <= byte <= 0xEF:
            self._string_escape_pending = False
            self._string_utf8_remaining = 2
            self._discard_string_if_oversized()
            return
        if 0xF0 <= byte <= 0xF4:
            self._string_escape_pending = False
            self._string_utf8_remaining = 3
            self._discard_string_if_oversized()
            return

        terminator = byte in (0x07, 0x18, 0x1A, 0x9C)
        reset = self._string_escape_pending and byte == ord("c")
        string_terminator = self._string_escape_pending and byte == ord("\\")

        if self._control_bytes > MAX_CONTROL_BYTES:
            if terminator or string_terminator or reset:
                self._reject_and_discard(self._TEXT)
                if reset:
                    admitted.extend(b"\x1bc")
            else:
                self._reject_and_discard(self._DISCARD_STRING)
                self._discard_escape_pending = byte == 0x1B
            return

        if terminator or string_terminator or reset:
            self._reset_sequence()
            if reset:
                admitted.extend(b"\x1bc")
            return

        self._string_escape_pending = byte == 0x1B

    def _discard_string_if_oversized(self) -> None:
        """Cross into payload-free discard state at the string byte cap."""
        if self._control_bytes > MAX_CONTROL_BYTES:
            self._reject_and_discard(self._DISCARD_STRING)

    def _consume_discard_escape(self, byte: int, admitted: bytearray) -> None:
        if self._discard_reset(byte, admitted):
            return
        if byte in (0x18, 0x1A) or 0x30 <= byte <= 0x7E:
            self._reset_sequence()

    def _consume_discard_csi(self, byte: int, admitted: bytearray) -> None:
        if self._discard_reset(byte, admitted):
            return
        if byte in (0x18, 0x1A) or 0x40 <= byte <= 0x7E:
            self._reset_sequence()

    def _consume_discard_string(self, byte: int, admitted: bytearray) -> None:
        if self._string_utf8_remaining:
            if 0x80 <= byte <= 0xBF:
                self._string_utf8_remaining -= 1
                return
            self._string_utf8_remaining = 0
            self._consume_discard_string(byte, admitted)
            return
        if 0xC2 <= byte <= 0xDF:
            self._discard_escape_pending = False
            self._string_utf8_remaining = 1
            return
        if 0xE0 <= byte <= 0xEF:
            self._discard_escape_pending = False
            self._string_utf8_remaining = 2
            return
        if 0xF0 <= byte <= 0xF4:
            self._discard_escape_pending = False
            self._string_utf8_remaining = 3
            return
        if byte in (0x07, 0x18, 0x1A, 0x9C):
            self._reset_sequence()
            return
        if self._discard_escape_pending and byte == ord("\\"):
            self._reset_sequence()
            return
        if self._discard_escape_pending and byte == ord("c"):
            self._reset_sequence()
            admitted.extend(b"\x1bc")
            return
        self._discard_escape_pending = byte == 0x1B

    def _discard_reset(self, byte: int, admitted: bytearray) -> bool:
        if self._discard_escape_pending and byte == ord("c"):
            self._reset_sequence()
            admitted.extend(b"\x1bc")
            return True
        self._discard_escape_pending = byte == 0x1B
        return False

    def _reject_and_discard(self, next_state: str) -> None:
        self._rejected_sequences += 1
        self._buffer.clear()
        self._control_bytes = 0
        self._state = next_state
        self._raw_c1 = False
        self._string_escape_pending = False
        self._discard_escape_pending = False

    def _reset_sequence(self) -> None:
        self._state = self._TEXT
        self._buffer.clear()
        self._control_bytes = 0
        self._raw_c1 = False
        self._string_escape_pending = False
        self._string_utf8_remaining = 0
        self._discard_escape_pending = False
