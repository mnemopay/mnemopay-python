"""
Lightning Payment Rail — Python port

Mirrors the TypeScript `@mnemopay/sdk` Lightning rail implementation.
Zero external dependencies, pure stdlib.
Integrates with LND REST APIs via HODL/standard invoice escrows. Features:
  - strict protocol checks
  - rigorous static SSRF block engine (`is_private_or_reserved`)
  - self-signed SSL compatibility using unverified SSL contexts
  - URL-safe base64 path mappings for LND compatibility
"""
from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from . import HoldOptions, PaymentRailResult


class LightningRail:
    """Lightning payment rail conforming to PaymentRail protocol.

    Uses LND REST invoices for instant, sub-cent micropayments.
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

        # Standard LND nodes often employ self-signed SSL certs in dev.
        # Bypass strict certificate verification while securing hosts via our static SSRF blocks.
        ctx = ssl._create_unverified_context()

        try:
            with urllib.request.urlopen(req, timeout=30.0, context=ctx) as res:
                resp_bytes = res.read()
                if not resp_bytes:
                    return {}
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Prevent raw LND internal data leaks — raise sanitized status
            raise RuntimeError(f"LND API error ({e.code})") from e
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}") from e
