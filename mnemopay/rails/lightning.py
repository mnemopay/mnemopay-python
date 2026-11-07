"""
Lightning Payment Rail — Python port

Mirrors the TypeScript `@mnemopay/sdk` Lightning rail implementation.
Zero external dependencies, pure stdlib.
Integrates with LND REST APIs via HODL/standard invoice escrows. Features:
  - strict protocol checks
  - rigorous static SSRF block engine (`is_private_or_reserved`)
  - TLS certificate verification ON by default; self-signed LND nodes are
    supported by pointing ``tls_cert_path`` at LND's ``tls.cert`` and/or pinning
    its SHA-256 fingerprint via ``tls_fingerprint``
  - URL-safe base64 path mappings for LND compatibility
"""
from __future__ import annotations

import hashlib
import http.client
import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import warnings
from typing import Any, Dict, Optional

from . import HoldOptions, PaymentRailResult


def _normalize_fingerprint(fp: str) -> str:
    """Accept ``AA:BB:..``, ``aabb..`` or ``sha256:aabb..``; return lowercase hex."""
    h = fp.strip().lower()
    if h.startswith("sha256:"):
        h = h[len("sha256:"):]
    h = h.replace(":", "").replace(" ", "")
    if not re.fullmatch(r"[0-9a-f]{64}", h):
        raise ValueError("tls_fingerprint must be a SHA-256 hex digest (64 hex chars)")
    return h


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection that checks the server certificate's SHA-256 fingerprint
    immediately after the handshake and BEFORE any request bytes (the admin
    macaroon header) are sent."""

    _pin: str = ""

    def connect(self) -> None:
        super().connect()
        der = self.sock.getpeercert(binary_form=True)  # type: ignore[union-attr]
        got = hashlib.sha256(der or b"").hexdigest()
        if got != self._pin:
            self.close()
            raise ssl.SSLError("LND TLS certificate fingerprint mismatch")


class LightningRail:
    """Lightning payment rail conforming to PaymentRail protocol.

    Uses LND REST invoices for instant, sub-cent micropayments.

    TLS: certificates are verified by default (system CA store + hostname).
    LND nodes normally use a self-signed cert, so pass ``tls_cert_path`` (path to
    LND's ``tls.cert``; it becomes the only trusted CA, hostname still checked)
    and/or ``tls_fingerprint`` (SHA-256 of the DER cert; pins the exact cert and
    skips chain/hostname validation, since the pin is the trust anchor).
    There is intentionally no "skip verification" switch.
    """

    name: str = "lightning"

    @staticmethod
    def is_private_or_reserved(hostname: str) -> bool:
        """SSRF mitigation: block private, loopback, link-local, and reserved IPs.

        Protects LND REST requests from being coerced into hitting cloud metadata
        service endpoints (e.g. 169.254.169.254) or intranet subnets.
        Matches hex/octal/loopback IPv6-mapped IPv4 bypass tricks.
        """
        h = hostname.lower().strip("[]")

        # Direct blocked match
        blocked = {
            "localhost",
            "0.0.0.0",
            "::",
            "::1",
            "::ffff:127.0.0.1",
            "metadata.google.internal",
            "169.254.169.254",
        }
        if h in blocked:
            return True
        for b in blocked:
            if b in h:
                return True

        # Suffix matching
        if h.endswith(".internal") or h.endswith(".local"):
            return True

        # IPv6 loopback
        if h == "::1":
            return True

        # Raw numeric IP block (e.g., 2130706433 == 127.0.0.1)
        try:
            val = int(h)
            if val >= 0:
                return True
        except ValueError:
            pass

        # Hex IP strings (e.g., 0x7f000001)
        if re.match(r"^0x[0-9a-f]+$", h):
            return True

        # Octal IP strings (e.g., 0177.0000.0000.0001)
        if re.match(r"^0\d+(\.\d+){0,3}$", h):
            return True

        # IPv6-mapped IPv4
        if "::ffff:" in h:
            return True

        # Standard IPv4 dotted checks
        parts = h.split(".")
        if len(parts) == 4:
            try:
                nums = [int(p) for p in parts]
                if all(0 <= n <= 255 for n in nums):
                    # RFC1918 loopback and internal
                    if nums[0] == 127:
                        return True
                    if nums[0] == 10:
                        return True
                    if nums[0] == 172 and 16 <= nums[1] <= 31:
                        return True
                    if nums[0] == 192 and nums[1] == 168:
                        return True
                    if nums[0] == 169 and nums[1] == 254:
                        return True
                    if nums[0] == 0:
                        return True
            except ValueError:
                pass

        # Parse via standard ipaddress if it represents a direct IP
        import ipaddress

        try:
            ip = ipaddress.ip_address(h)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_reserved
                or ip.is_link_local
                or ip.is_multicast
            ):
                return True
        except ValueError:
            pass

        return False

    def __init__(
        self,
        lnd_rest_url: str,
        macaroon: str,
        btc_price_usd: float = 60000.0,
        tls_cert_path: Optional[str] = None,
        tls_fingerprint: Optional[str] = None,
    ) -> None:
        if not lnd_rest_url:
            raise ValueError("LND REST URL is required")
        if not macaroon:
            raise ValueError("Admin macaroon hex string is required")

        try:
            parsed = urllib.parse.urlparse(lnd_rest_url)
        except Exception as e:
            raise ValueError("Invalid LND REST URL") from e

        if parsed.scheme not in ("http", "https"):
            raise ValueError("LND URL must use http or https protocol")

        # Extract host ignoring port
        host_only = parsed.hostname or parsed.netloc.split(":")[0]
        if self.is_private_or_reserved(host_only):
            raise ValueError(
                "LND URL must not target private/internal network addresses",
            )

        self.base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        self.macaroon = macaroon
        self.btc_price_usd = btc_price_usd
        self.tls_cert_path = tls_cert_path
        self._pin: Optional[str] = (
            _normalize_fingerprint(tls_fingerprint) if tls_fingerprint else None
        )
        if parsed.scheme == "http":
            # The admin macaroon would travel in cleartext. LND REST is TLS-only
            # by default, so plain http is only sane through a trusted tunnel.
            warnings.warn(
                "LND URL uses http://: the admin macaroon is sent unencrypted",
                stacklevel=2,
            )
        self._ssl_ctx = self._build_ssl_context()

    def _build_ssl_context(self) -> ssl.SSLContext:
        """Verifying context. Pinned-only mode trusts the fingerprint instead of a CA."""
        if self.tls_cert_path:
            ctx = ssl.create_default_context(cafile=self.tls_cert_path)
        elif self._pin:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        else:
            ctx = ssl.create_default_context()
        return ctx

    def set_btc_price(self, usd: float) -> None:
        """Update active exchange rate for USD -> sats conversion."""
        self.btc_price_usd = usd

    def _usd_to_sats(self, usd: float) -> int:
        return round((usd / self.btc_price_usd) * 100000000.0)

    # ── PaymentRail Interface ───────────────────────────────────────────────

    def create_hold(
        self,
        amount: float,
        reason: str,
        agent_id: str,
        opts: Optional[HoldOptions] = None,  # noqa: ARG002
    ) -> PaymentRailResult:
        sats = self._usd_to_sats(amount)
        memo = f"mnemopay:{agent_id}:{reason[:100]}"

        payload = {
            "value": str(sats),
            "memo": memo,
            "expiry": "3600",  # 1 hour
        }

        res = self._lnd_request("/v1/invoices", "POST", payload)

        return PaymentRailResult(
            external_id=res.get("r_hash", ""),  # Base64 payment hash
            status="invoice_created",
            receipt_id=res.get("payment_request"),  # Lightning invoice string
        )

    def capture_payment(
        self,
        external_id: str,
        amount: float,  # noqa: ARG002
    ) -> PaymentRailResult:
        if not external_id:
            raise ValueError("LND capture requires an external r_hash reference")

        # Map standard base64 r_hash into URL-safe base64 expected by LND REST API paths
        r_hash_url_safe = (
            external_id.replace("+", "-").replace("/", "_").rstrip("=")
        )

        res = self._lnd_request(
            f"/v1/invoice/{r_hash_url_safe}",
            "GET",
        )

        settled = res.get("settled", False)
        return PaymentRailResult(
            external_id=external_id,
            status="captured" if settled else "pending",
            receipt_id=res.get("payment_request"),
        )

    def reverse_payment(
        self,
        external_id: str,
        amount: float,  # noqa: ARG002
    ) -> PaymentRailResult:
        # Lightning invoices cannot be cancelled or reversed once settled.
        # Mirror TypeScript expired status for unpaid hold cancellation.
        return PaymentRailResult(
            external_id=external_id,
            status="expired",
        )

    # ── HTTP Requests ───────────────────────────────────────────────────────

    def _lnd_request(
        self,
        path: str,
        method: str,
        body: Optional[Any] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        headers = {
            "Grpc-Metadata-macaroon": self.macaroon,
            "Content-Type": "application/json",
        }

        data_bytes = None
        if body is not None:
            data_bytes = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers=headers,
            method=method,
        )

        # Certificate verification is ON. Self-signed LND certs are handled by the
        # configured tls_cert_path (trusted root) and/or tls_fingerprint (pin).
        ctx = self._ssl_ctx

        try:
            if self._pin:
                opener = urllib.request.build_opener(
                    self._pinned_handler(ctx, self._pin),
                )
                res_cm = opener.open(req, timeout=30.0)
            else:
                res_cm = urllib.request.urlopen(req, timeout=30.0, context=ctx)
            with res_cm as res:
                resp_bytes = res.read()
                if not resp_bytes:
                    return {}
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Prevent raw LND internal data leaks — raise sanitized status
            raise RuntimeError(f"LND API error ({e.code})") from e
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}") from e

    @staticmethod
    def _pinned_handler(
        ctx: ssl.SSLContext, pin: str,
    ) -> urllib.request.HTTPSHandler:
        class _Conn(_PinnedHTTPSConnection):
            _pin = pin

        class _Handler(urllib.request.HTTPSHandler):
            def https_open(self, req: urllib.request.Request):  # type: ignore[override]
                return self.do_open(_Conn, req, context=ctx)

        return _Handler(context=ctx)
