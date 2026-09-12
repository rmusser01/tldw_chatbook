"""Behavioral coverage for the isolated, single-scrypt age v1 worker."""

import json
import subprocess
import sys
from pathlib import Path

WORKER = (
    Path(__file__).resolve().parents[2] / "tldw_chatbook/Backup_Recovery/age_worker.py"
)


def test_isolated_worker_reports_python_protocol():
    result = subprocess.run(
        [sys.executable, "-I", str(WORKER), "info"],
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "protocol": 2,
        "implementation": "python",
        "format": "age-v1",
    }
    assert result.stderr == b""


# Import by fixed filename, just as the isolated entry point does: no app imports.
import importlib.util
import io

import pytest


@pytest.fixture(scope="module")
def worker():
    spec = importlib.util.spec_from_file_location("tested_age_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def packet(password, data=b""):
    return len(password).to_bytes(4, "big") + password + data


def transform(worker, mode, data, password=b"worker-test-passphrase"):
    output = io.BytesIO()
    worker.run(mode, io.BytesIO(packet(password, data)), output)
    return output.getvalue()


@pytest.mark.parametrize("size", [0, 1, 65535, 65536, 65537, 196625])
def test_full_strength_stream_boundaries(worker, size):
    plaintext = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    ciphertext = transform(worker, "encrypt", plaintext)
    assert ciphertext.split(b"\n")[1].endswith(b" 18")
    assert transform(worker, "decrypt", ciphertext) == plaintext


@pytest.mark.parametrize("password", [b"\x00", b"\xff\xfe\x00\nsecret", b"x" * 4096])
def test_raw_binary_passwords(worker, password):
    encrypted = transform(worker, "encrypt", b"binary password payload", password)
    assert (
        transform(worker, "decrypt", encrypted, password) == b"binary password payload"
    )


@pytest.mark.parametrize(
    "wire",
    [b"", b"\x00\x00\x00", b"\x00" * 4, (4097).to_bytes(4, "big"), packet(b"abc")[:-1]],
)
def test_malformed_password_protocol(worker, wire):
    with pytest.raises(worker.WorkerError, match="^protocol_error$"):
        worker.run("encrypt", io.BytesIO(wire), io.BytesIO())


@pytest.fixture(scope="module")
def encrypted(worker):
    return transform(worker, "encrypt", b"protected payload")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.replace(b"age-encryption.org/v1", b"age-encryption.org/v2"),
        lambda value: value.replace(b"-> scrypt", b"-> X25519"),
        lambda value: value.replace(b"-> scrypt ", b"-> scrypt  "),
        lambda value: value.replace(b" 18\n", b" 19\n"),
        lambda value: value.replace(b" 18\n", b" 0\n"),
        lambda value: value.replace(b" 18\n", b" 018\n"),
        lambda value: value.replace(b" 18\n", b" +18\n"),
        lambda value: value.replace(b"\n", b"\r\n", 1),
        lambda value: value.replace(b"--- ", b"-> scrypt extra stanza\n--- "),
        lambda value: value.replace(b"--- ", b"---  "),
        lambda value: value.split(b"--- ")[0] + b"--- " + b"A" * 65536 + b"\n",
    ],
)
def test_noncanonical_headers_rejected_before_kdf(
    worker, encrypted, mutation, monkeypatch
):
    def forbidden_kdf(*args, **kwargs):
        pytest.fail("malformed header reached scrypt")

    monkeypatch.setattr(worker, "scrypt", forbidden_kdf)
    with pytest.raises(worker.WorkerError, match="^invalid_header$"):
        transform(worker, "decrypt", mutation(encrypted))


@pytest.mark.parametrize("part", [1, 2, 3])
def test_noncanonical_base64_rejected_before_kdf(worker, encrypted, part, monkeypatch):
    lines = encrypted.split(b"\n", 4)
    tokens = lines[part].split(b" ")
    position = 2 if part == 1 else (-1 if part == 3 else 0)
    tokens[position] += b"="
    lines[part] = b" ".join(tokens)
    monkeypatch.setattr(
        worker, "scrypt", lambda *a, **kw: pytest.fail("KDF before admission")
    )
    with pytest.raises(worker.WorkerError, match="^invalid_header$"):
        transform(worker, "decrypt", b"\n".join(lines))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value[:-1],
        lambda value: value[:-16],
        lambda value: value + b"trailing data",
        lambda value: value[:-1] + bytes([value[-1] ^ 1]),
        lambda value: value.split(b"\n", 4)[0] + b"\n",
    ],
)
def test_corruption_truncation_and_trailing_data_fail(worker, encrypted, mutation):
    with pytest.raises(worker.WorkerError):
        transform(worker, "decrypt", mutation(encrypted))


def test_wrong_password_releases_no_plaintext(worker, encrypted):
    output = io.BytesIO()
    with pytest.raises(worker.WorkerError, match="^transform_failed$"):
        worker.run("decrypt", io.BytesIO(packet(b"wrong", encrypted)), output)
    assert output.getvalue() == b""


class ShortReader(io.BytesIO):
    def read(self, size=-1):
        assert 0 <= size <= 65553
        return super().read(min(size, 7))


class ShortWriter(io.BytesIO):
    def write(self, data):
        return super().write(data[:3])


def test_short_binary_io_preserves_data(worker):
    plaintext = b"short pipes\x00\xff" * 6000
    encrypted = ShortWriter()
    worker.run("encrypt", ShortReader(packet(b"secret", plaintext)), encrypted)
    restored = ShortWriter()
    worker.run(
        "decrypt", ShortReader(packet(b"secret", encrypted.getvalue())), restored
    )
    assert restored.getvalue() == plaintext


@pytest.mark.parametrize("limit", [0, 1, 16, 147, 148])
def test_decrypt_input_budget_counts_header_and_payload(
    worker, encrypted, limit, monkeypatch
):
    monkeypatch.setattr(worker, "MAX_CONTAINER", limit)
    with pytest.raises(worker.WorkerError, match="^byte_limit$"):
        transform(worker, "decrypt", encrypted)


def test_output_budget_independent_of_input_budget(worker, monkeypatch):
    monkeypatch.setattr(worker, "MAX_CONTAINER", 200)
    with pytest.raises(worker.WorkerError, match="^byte_limit$"):
        transform(worker, "encrypt", b"p" * 30)


def test_cli_sanitizes_failure():
    result = subprocess.run(
        [sys.executable, "-I", str(WORKER), "decrypt"],
        input=packet(b"do-not-echo-this", b"invalid header"),
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert (result.returncode, result.stdout, result.stderr) == (
        1,
        b"",
        b"invalid_header\n",
    )


FIXTURES = Path(__file__).with_name("fixtures") / "age_v1"
LEGACY_CASES = [
    ("official.age", b"interop-synthetic-password", bytes(range(256)) * 2048),
    ("legacy-empty.age", b"\xff\x00" * 2048, b""),
    ("legacy-stream.age", b"test-only passphrase", b"synthetic recovery bytes" * 10000),
]


@pytest.mark.parametrize("filename,password,plaintext", LEGACY_CASES)
def test_decrypts_fixed_preexisting_age_archives(worker, filename, password, plaintext):
    assert (
        transform(worker, "decrypt", (FIXTURES / filename).read_bytes(), password)
        == plaintext
    )


def recorded_entropy(ciphertext, password):
    """Recover old file entropy with library primitives, independent of the worker."""
    import base64

    from Cryptodome.Cipher import ChaCha20_Poly1305
    from Cryptodome.Protocol.KDF import scrypt

    _, stanza, body, _, payload = ciphertext.split(b"\n", 4)
    _, _, salt64, factor = stanza.split(b" ")
    salt = base64.b64decode(salt64 + b"==")
    wrapped = base64.b64decode(body + b"=")
    wrapping_key = scrypt(
        password, b"age-encryption.org/v1/scrypt" + salt, 32, 2 ** int(factor), 8, 1
    )
    file_key = ChaCha20_Poly1305.new(
        key=wrapping_key, nonce=bytes(12)
    ).decrypt_and_verify(wrapped[:16], wrapped[16:])
    return file_key, salt, payload[:16]


@pytest.mark.parametrize("filename,password,plaintext", LEGACY_CASES)
def test_writer_matches_fixed_independent_ciphertext(
    worker, filename, password, plaintext, monkeypatch
):
    expected = (FIXTURES / filename).read_bytes()
    entropy = iter(recorded_entropy(expected, password))

    def fixed_entropy(size):
        assert size == 16
        return next(entropy)

    monkeypatch.setattr(worker, "get_random_bytes", fixed_entropy)
    assert transform(worker, "encrypt", plaintext, password) == expected


def test_canonical_but_corrupt_header_mac_releases_no_plaintext(worker, encrypted):
    import base64

    lines = encrypted.split(b"\n", 4)
    mac = bytearray(base64.b64decode(lines[3][4:] + b"="))
    mac[0] ^= 1
    lines[3] = b"--- " + base64.b64encode(mac).rstrip(b"=")
    output = io.BytesIO()
    with pytest.raises(worker.WorkerError, match="^transform_failed$"):
        worker.run(
            "decrypt",
            io.BytesIO(packet(b"worker-test-passphrase", b"\n".join(lines))),
            output,
        )
    assert output.getvalue() == b""


@pytest.mark.parametrize("part", [1, 2, 3])
def test_nonzero_base64_pad_bits_rejected_before_kdf(
    worker, encrypted, part, monkeypatch
):
    alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    lines = encrypted.split(b"\n", 4)
    tokens = lines[part].split(b" ")
    index = 2 if part == 1 else (-1 if part == 3 else 0)
    value = tokens[index]
    tokens[index] = value[:-1] + bytes([alphabet[alphabet.index(value[-1]) + 1]])
    lines[part] = b" ".join(tokens)
    monkeypatch.setattr(
        worker, "scrypt", lambda *a, **kw: pytest.fail("KDF before admission")
    )
    with pytest.raises(worker.WorkerError, match="^invalid_header$"):
        transform(worker, "decrypt", b"\n".join(lines))


def test_authenticated_empty_final_chunk_after_data_is_rejected(worker):
    from Cryptodome.Cipher import ChaCha20_Poly1305
    from Cryptodome.Hash import SHA256
    from Cryptodome.Protocol.KDF import HKDF

    ciphertext = (FIXTURES / "official.age").read_bytes()
    file_key, _, payload_nonce = recorded_entropy(
        ciphertext, b"interop-synthetic-password"
    )
    header_parts = ciphertext.split(b"\n", 4)
    header = b"\n".join(header_parts[:4]) + b"\n"
    first_chunk = header_parts[4][16 : 16 + 65552]
    key = HKDF(file_key, 32, payload_nonce, SHA256, context=b"payload")
    cipher = ChaCha20_Poly1305.new(key=key, nonce=(1).to_bytes(11, "big") + b"\x01")
    empty, tag = cipher.encrypt_and_digest(b"")
    forged = header + payload_nonce + first_chunk + empty + tag
    with pytest.raises(worker.WorkerError, match="^transform_failed$"):
        transform(worker, "decrypt", forged, b"interop-synthetic-password")


def test_nonfinal_chunk_at_eof_is_rejected(worker):
    ciphertext = (FIXTURES / "official.age").read_bytes()
    parts = ciphertext.split(b"\n", 4)
    truncated = b"\n".join(parts[:4]) + b"\n" + parts[4][: 16 + 65552]
    with pytest.raises(worker.WorkerError, match="^transform_failed$"):
        transform(worker, "decrypt", truncated, b"interop-synthetic-password")


@pytest.mark.parametrize("write_result", [0, None, -1])
def test_stalled_output_fails_instead_of_spinning(worker, write_result):
    class StalledWriter:
        def write(self, data):
            return write_result

    with pytest.raises(worker.WorkerError, match="^transform_failed$"):
        worker.run("info", io.BytesIO(), StalledWriter())


def test_exact_container_input_budget_still_probes_eof(worker, encrypted, monkeypatch):
    monkeypatch.setattr(worker, "MAX_CONTAINER", len(encrypted))
    assert transform(worker, "decrypt", encrypted) == b"protected payload"
    with pytest.raises(worker.WorkerError, match="^byte_limit$"):
        transform(worker, "decrypt", encrypted + b"x")


def test_payload_streaming_uses_bounded_python_memory(worker, tmp_path):
    import hashlib
    import tracemalloc

    size = 8 * 1024 * 1024

    class GeneratedInput:
        def __init__(self):
            self.prefix = io.BytesIO(packet(b"memory-test"))
            self.remaining = size

        def read(self, requested):
            assert 0 <= requested <= 65536
            prefix = self.prefix.read(requested)
            if prefix:
                return prefix
            length = min(requested, self.remaining)
            self.remaining -= length
            return b"z" * length

    class HashOutput:
        def __init__(self):
            self.digest = hashlib.sha256()
            self.size = 0

        def write(self, data):
            assert len(data) <= 65536
            self.digest.update(data)
            self.size += len(data)
            return len(data)

    class PrefixedFile:
        def __init__(self, file):
            self.prefix = io.BytesIO(packet(b"memory-test"))
            self.file = file

        def read(self, requested):
            assert 0 <= requested <= 65552
            return self.prefix.read(requested) or self.file.read(requested)

    restored = HashOutput()
    with (tmp_path / "memory.age").open("w+b") as encrypted_file:
        tracemalloc.start()
        try:
            worker.run("encrypt", GeneratedInput(), encrypted_file)
            encrypted_file.seek(0)
            worker.run("decrypt", PrefixedFile(encrypted_file), restored)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    # This bounds Python buffering; scrypt's separate native 256 MiB is intentional.
    assert peak < 2 * 1024 * 1024
    assert restored.size == size
    expected = hashlib.sha256()
    for _ in range(size // 65536):
        expected.update(b"z" * 65536)
    assert restored.digest.digest() == expected.digest()


@pytest.mark.parametrize("mode", ["info", "encrypt", "decrypt"])
def test_closed_stdout_pipe_has_only_fixed_failure(worker, encrypted, mode):
    import os

    wire = (
        b""
        if mode == "info"
        else packet(
            b"worker-test-passphrase",
            encrypted if mode == "decrypt" else b"pipe test payload",
        )
    )
    read_fd, write_fd = os.pipe()
    # Close the reader before launch, so failure does not depend on scheduling.
    os.close(read_fd)
    try:
        with subprocess.Popen(
            [sys.executable, "-I", str(WORKER), mode],
            stdin=subprocess.PIPE,
            stdout=write_fd,
            stderr=subprocess.PIPE,
        ) as process:
            _, stderr = process.communicate(input=wire, timeout=10)
    finally:
        os.close(write_fd)
    assert (process.returncode, stderr) == (1, b"transform_failed\n")
