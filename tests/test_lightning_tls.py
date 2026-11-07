"""LightningRail TLS: certificates must be verified by default (audit F3)."""
from __future__ import annotations

import hashlib
import http.server
import json
import shutil
import ssl
import subprocess
import threading
from pathlib import Path

import pytest

from mnemopay.rails import LightningRail
from mnemopay.rails import lightning as lightning_mod

needs_openssl = pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI not available")


def test_source_never_disables_verification() -> None:
    src = Path(lightning_mod.__file__).read_text(encoding="utf-8")
    assert "_create_unverified_context" not in src


def test_default_context_verifies() -> None:
    rail = LightningRail("https://node.example.com:8080", "mac")
    assert rail._ssl_ctx.verify_mode == ssl.CERT_REQUIRED
    assert rail._ssl_ctx.check_hostname is True


def test_cert_path_context_verifies(tmp_path: Path) -> None:
    cert = _make_cert(tmp_path)
    rail = LightningRail("https://node.example.com:8080", "mac", tls_cert_path=str(cert))
    assert rail._ssl_ctx.verify_mode == ssl.CERT_REQUIRED
    assert rail._ssl_ctx.check_hostname is True


def test_bad_fingerprint_rejected() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        LightningRail("https://node.example.com:8080", "mac", tls_fingerprint="abcd")


def _make_cert(d: Path) -> Path:
    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI not available")
    key, crt = d / "k.pem", d / "c.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key),
         "-out", str(crt), "-days", "2", "-subj", "/CN=localhost",
         "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"],
        check=True, capture_output=True,
    )
    (d / "key").write_text(str(key))
    return crt


@pytest.fixture()
def tls_server(tmp_path: Path):
    crt = _make_cert(tmp_path)
    key = Path((tmp_path / "key").read_text())
    seen: list = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen.append(self.headers.get("Grpc-Metadata-macaroon"))
            body = json.dumps({"settled": True, "payment_request": "lnbc1"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a) -> None:  # silence
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    sctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    sctx.load_cert_chain(str(crt), str(key))
    srv.socket = sctx.wrap_socket(srv.socket, server_side=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, crt, seen
    srv.shutdown()
    srv.server_close()


def _rail(srv, **kw) -> LightningRail:
    rail = LightningRail("https://node.example.com:8080", "SECRETMAC", **kw)
    rail.base_url = f"https://127.0.0.1:{srv.server_address[1]}"  # bypass SSRF guard for loopback test server
    return rail


@needs_openssl
def test_self_signed_rejected_by_default(tls_server) -> None:
    srv, _crt, seen = tls_server
    with pytest.raises(RuntimeError, match="Request failed"):
        _rail(srv).capture_payment("abc", 1.0)
    assert seen == []  # macaroon never sent


@needs_openssl
def test_configured_cert_path_accepted(tls_server) -> None:
    srv, crt, seen = tls_server
    res = _rail(srv, tls_cert_path=str(crt)).capture_payment("abc", 1.0)
    assert res.status == "captured"
    assert seen == ["SECRETMAC"]


@needs_openssl
def test_pinned_fingerprint_accepted(tls_server) -> None:
    srv, crt, seen = tls_server
    der = ssl.PEM_cert_to_DER_cert(crt.read_text())
    fp = hashlib.sha256(der).hexdigest()
    res = _rail(srv, tls_fingerprint=fp.upper()).capture_payment("abc", 1.0)
    assert res.status == "captured"
    assert seen == ["SECRETMAC"]


@needs_openssl
def test_wrong_fingerprint_rejected_before_macaroon_sent(tls_server) -> None:
    srv, _crt, seen = tls_server
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        _rail(srv, tls_fingerprint="00" * 32).capture_payment("abc", 1.0)
    assert seen == []
