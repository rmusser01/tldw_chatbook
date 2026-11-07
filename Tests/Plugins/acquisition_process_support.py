"""Owned fixtures for Git sources; never read real host authentication."""

import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path


def git(repository: Path, *args: str, input: bytes | None = None) -> bytes:
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
    return subprocess.run(
        [
            shutil.which("git"),
            "-c",
            "core.hooksPath=" + os.devnull,
            "-C",
            str(repository),
            *args,
        ],
        env=environment,
        input=input,
        capture_output=True,
        check=True,
        timeout=10,
    ).stdout


def repository(package: Path, destination: Path) -> Path:
    shutil.copytree(package, destination, symlinks=True)
    git(destination, "init", "-b", "main")
    git(destination, "add", ".")
    git(destination, "commit", "-m", "controlled fixture")
    return destination


def stubborn_git(path: Path, marker: Path) -> Path:
    """A controlled retained process tree with the ordinary child profile policy."""
    import shlex

    from Tests.hooks_v2_process_support import child_argv

    code = (
        "import json,os,signal,time;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "child=os.fork();"
        f"Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid(),'parent':os.getppid(),'child':child,'package':__import__('tldw_chatbook').__file__}})) if child else None;"
        "time.sleep(60)"
    )
    path.write_text("#!/bin/sh\nexec " + shlex.join(child_argv(code)) + ' "$@"\n')
    path.chmod(0o700)
    return path


def fake_ssh(path: Path, source: Path, log: Path) -> Path:
    """Inspect real OpenSSH effective config, then serve controlled Git locally."""
    import shlex

    from Tests.hooks_v2_process_support import child_argv

    ssh, git_binary = shutil.which("ssh"), shutil.which("git")
    code = (
        "import json,os,sys;from pathlib import Path;args=sys.argv[1:];"
        f"ssh={ssh!r};git={git_binary!r};"
        "os.execv(ssh,[ssh]+args) if '-G' in args else None;"
        f"Path({str(log)!r}).write_text(json.dumps({{'args':args,'config':Path(args[args.index('-F')+1]).read_text(),'package':__import__('tldw_chatbook').__file__}}));"
        f"os.execv(git,[git,'upload-pack',{str(source / '.git')!r}])"
    )
    path.write_text("#!/bin/sh\nexec " + shlex.join(child_argv(code)) + ' "$@"\n')
    path.chmod(0o700)
    return path


@contextmanager
def https_git(root: Path, *, redirect: str | None = None):
    """Real loopback TLS/HTTP Git protocol with an independently authored helper."""
    import base64
    import ipaddress
    import ssl
    import threading
    from datetime import UTC, datetime, timedelta
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlsplit

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "owned-loopback")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = root / "cert.pem", root / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    requests = []
    expected = (
        "Basic " + base64.b64encode(b"fixture:private-credential-sentinel").decode()
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.handle_git()

        def do_POST(self):
            self.handle_git()

        def handle_git(self):
            requests.append(
                (self.command, self.path, self.headers.get("Authorization"))
            )
            if redirect is not None:
                self.send_response(302)
                self.send_header("Location", redirect)
                self.end_headers()
                return
            if self.headers.get("Authorization") != expected:
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="owned-test"')
                self.end_headers()
                return
            path = urlsplit(self.path)
            environment = {
                "PATH": os.environ["PATH"],
                "HOME": os.environ["HOME"],
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_PROJECT_ROOT": str(root),
                "GIT_HTTP_EXPORT_ALL": "1",
                "REQUEST_METHOD": self.command,
                "PATH_INFO": path.path,
                "QUERY_STRING": path.query,
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "REMOTE_USER": "fixture",
                "REMOTE_ADDR": "127.0.0.1",
            }
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            process = subprocess.run(
                [shutil.which("git"), "http-backend"],
                input=body,
                env=environment,
                capture_output=True,
                check=False,
                timeout=10,
            )
            headers, _, output = process.stdout.partition(b"\r\n\r\n")
            values = [
                line.decode().split(": ", 1)
                for line in headers.split(b"\r\n")
                if b": " in line
            ]
            status = next(
                (
                    int(value.split()[0])
                    for key, value in values
                    if key.lower() == "status"
                ),
                200,
            )
            self.send_response(status)
            for key, value in values:
                if key.lower() != "status":
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"https://127.0.0.1:{server.server_port}", cert_path, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
