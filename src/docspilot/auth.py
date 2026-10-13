"""Authentication behind a tiny async interface.

Two implementations (and a chain): static API keys (service-to-service, simple), and
OIDC bearer tokens (end users, via any IdP: Clerk, Auth0, Okta, Entra, Keycloak...)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import jwt
from jwt import PyJWK

from docspilot.budget import RESERVED_IDS
from docspilot.logging import get_logger

log = get_logger("auth")


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str


class AuthUnavailableError(Exception):
    """The credential could not be *checked* (e.g. the IdP's keys are unreachable and
    none are cached). This is a 503, not a 401: clients must not log users out for it."""


class Authenticator(Protocol):
    @property
    def required(self) -> bool: ...
    async def authenticate(self, token: str | None) -> Principal | None: ...


def _digest(secret: str) -> bytes:
    return hashlib.sha256(secret.encode()).digest()


class ApiKeyAuthenticator:
    """API keys are stored as SHA-256 digests and compared in constant time against
    *every* configured key (no early exit that leaks which prefix matched)."""

    def __init__(self, keys: dict[str, str]) -> None:
        """keys: user_id -> raw API key."""
        self._digests = {user: _digest(key) for user, key in keys.items()}

    @property
    def required(self) -> bool:
        return bool(self._digests)

    async def authenticate(self, token: str | None) -> Principal | None:
        if not self._digests:
            return Principal("anonymous")  # auth not configured (dev only; strict envs validate)
        if not token:
            return None
        candidate = _digest(token)
        match: str | None = None
        for user, digest in self._digests.items():
            if hmac.compare_digest(candidate, digest):
                match = user
        return Principal(match) if match else None


def parse_api_keys(spec: str) -> dict[str, str]:
    """'alice:key-a,bob:key-b' -> {'alice': 'key-a', 'bob': 'key-b'}"""
    keys: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        user, sep, key = part.partition(":")
        if not sep or not user.strip() or len(key.strip()) < 16:
            raise ValueError("API_KEYS entries must be 'user:key' with keys of 16+ characters")
        user, key = user.strip(), key.strip()
        if user in RESERVED_IDS:
            raise ValueError(f"API_KEYS user id {user!r} is reserved")
        if user in keys or key in keys.values():
            raise ValueError("API_KEYS contains a duplicate user or key")
        keys[user] = key
    return keys


# ---- OIDC ----------------------------------------------------------------------------


# JOSE algorithm family -> JWK key type: a key of another type can never verify the token
# and must not reach the signature code (it would raise instead of rejecting).
_KTY = {"RS": "RSA", "PS": "RSA", "ES": "EC"}


class OIDCAuthenticator:
    """Verifies RS/ES-signed JWT access tokens against the issuer's JWKS.

    What is enforced, and why each matters:
    - signature against a key selected by `kid` from the issuer's JWKS (never from the token);
    - an *allow-list* of algorithms: blocks `alg: none` and RS256->HS256 key confusion
      (verifying an HMAC with the public key as the secret);
    - `iss`, `aud`, `exp` required; `nbf`/`iat` checked; small clock leeway;
    - the user id comes from a configured claim (`sub`), which must be a non-empty string.
    Unknown `kid` triggers one JWKS refresh (key rotation) at most every `min_refresh_s`,
    so a stream of garbage tokens cannot turn the IdP into a target."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str | None = None,
        algorithms: tuple[str, ...] = ("RS256",),
        user_claim: str = "sub",
        leeway_s: float = 30.0,
        jwks_ttl_s: float = 3600.0,
        min_refresh_s: float = 30.0,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if any(a.lower().startswith(("none", "hs")) for a in algorithms):
            raise ValueError("OIDC algorithms must be asymmetric (RS*/ES*/PS*/EdDSA)")
        self._issuer = issuer
        self._audience = audience
        self._jwks_url = jwks_url
        self._algorithms = list(algorithms)
        self._user_claim = user_claim
        self._leeway = leeway_s
        self._ttl = jwks_ttl_s
        self._min_refresh = min_refresh_s
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0))
        self._owns_client = client is None
        self._clock = clock
        self._keys: dict[str, PyJWK] = {}
        self._fetched_at: float | None = None
        self._last_refresh_ok = False
        self._lock = asyncio.Lock()
        self._kty = {_KTY[a[:2]] for a in algorithms if a[:2] in _KTY} | (
            {"OKP"} if "EdDSA" in algorithms else set()
        )

    @property
    def required(self) -> bool:
        return True

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def authenticate(self, token: str | None) -> Principal | None:
        if not token:
            return None
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            return None
        kid = header.get("kid")
        alg = header.get("alg")
        if alg not in self._algorithms or not isinstance(kid, str):
            return None
        key = await self._key_for(kid)
        if key is None:
            return None
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                key.key,
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={"require": ["exp", "iss", "aud"]},
            )
        except jwt.PyJWTError as exc:
            log.info("oidc_token_rejected", reason=type(exc).__name__)
            return None
        except Exception:
            log.warning("oidc_token_unverifiable", exc_info=True)
            return None
        user = claims.get(self._user_claim)
        if (
            not isinstance(user, str)
            or user != user.strip()
            or not user
            or len(user) > 200
            or any(ord(c) < 32 for c in user)
            or user in RESERVED_IDS
        ):
            return None
        # Namespaced: an IdP subject can never collide with an API-key user id.
        return Principal(f"oidc:{user}")

    # -- JWKS ------------------------------------------------------------------------

    async def _key_for(self, kid: str) -> PyJWK | None:
        stale = self._fetched_at is None or self._clock() - self._fetched_at > self._ttl
        if stale:
            await self._refresh()
        if kid not in self._keys:
            can_refresh = (
                self._fetched_at is None or self._clock() - self._fetched_at >= self._min_refresh
            )
            if can_refresh:
                await self._refresh()  # key rotation: the IdP published a new kid
        if not self._keys and not self._last_refresh_ok:
            raise AuthUnavailableError("identity provider keys are unavailable")
        return self._keys.get(kid)

    async def _jwks_url_resolved(self) -> str:
        if self._jwks_url:
            return self._jwks_url
        discovery = f"{self._issuer.rstrip('/')}/.well-known/openid-configuration"
        res = await self._client.get(discovery)
        res.raise_for_status()
        url = res.json()["jwks_uri"]
        if not isinstance(url, str):
            raise ValueError("discovery document has no jwks_uri")
        self._jwks_url = url
        return url

    async def _refresh(self) -> None:
        async with self._lock:
            # Waiters queued behind a refresh that just finished (or just failed) reuse it
            # instead of each repeating the fetch.
            if self._fetched_at is not None and self._clock() - self._fetched_at < 1.0:
                return
            try:
                res = await self._client.get(await self._jwks_url_resolved())
                res.raise_for_status()
                keys: dict[str, PyJWK] = {}
                for jwk in res.json().get("keys", []):
                    if jwk.get("use", "sig") != "sig" or not jwk.get("kid"):
                        continue
                    if self._kty and jwk.get("kty") not in self._kty:
                        continue
                    try:
                        keys[jwk["kid"]] = PyJWK(jwk)
                    except jwt.PyJWTError:
                        continue
                self._keys = keys
                self._last_refresh_ok = True
            except Exception:
                self._last_refresh_ok = False
                log.warning("jwks_refresh_failed", keys_cached=len(self._keys))
            finally:
                # stamped *after* the attempt: a slow failing fetch must still start the
                # cool-down, or every queued request repeats it
                self._fetched_at = self._clock()


class ChainAuthenticator:
    """JWT-shaped bearer tokens go to OIDC, everything else to API keys."""

    def __init__(self, oidc: Authenticator | None, api_keys: Authenticator | None) -> None:
        self._oidc = oidc
        self._keys = api_keys

    @property
    def required(self) -> bool:
        return bool((self._oidc and self._oidc.required) or (self._keys and self._keys.required))

    async def authenticate(self, token: str | None) -> Principal | None:
        if not self.required:
            return Principal("anonymous")
        if not token:
            return None
        looks_like_jwt = token.count(".") == 2 and token.startswith("eyJ")
        if looks_like_jwt and self._oidc is not None:
            return await self._oidc.authenticate(token)
        if self._keys is not None and self._keys.required:
            return await self._keys.authenticate(token)
        return None
