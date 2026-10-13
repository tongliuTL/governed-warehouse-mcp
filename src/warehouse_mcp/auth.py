"""Bearer-token authentication.

Tokens are JWTs signed with Ed25519 (``EdDSA``). The verifier pins the algorithm list, so
``alg: none`` and HS256-with-the-public-key confusion attacks fail, and it requires ``exp``,
``iat``, ``sub``, ``iss`` and ``aud``. Scopes come from the space-separated ``scope`` claim
(RFC 9068).

For production, point the verifier at your identity provider's public key. ``generate_keypair``
and ``issue_token`` exist for local development and the test suite.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from mcp.server.auth.provider import AccessToken

from warehouse_mcp.policy import Principal

ALGORITHMS = ["EdDSA"]
DEFAULT_ISSUER = "https://auth.example.test"
DEFAULT_AUDIENCE = "governed-warehouse-mcp"
MAX_TTL_S = 24 * 3600


class AuthError(Exception):
    pass


def generate_keypair(directory: Path | str) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    private_path, public_path = directory / "private.pem", directory / "public.pem"
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    private_path.chmod(0o600)
    public_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    return private_path, public_path


def issue_token(
    private_key_pem: bytes,
    subject: str,
    scopes: list[str] | tuple[str, ...],
    *,
    ttl_s: int = 3600,
    issuer: str = DEFAULT_ISSUER,
    audience: str = DEFAULT_AUDIENCE,
    client_id: str = "local-dev",
) -> str:
    if not 0 < ttl_s <= MAX_TTL_S:
        raise ValueError(f"ttl_s must be between 1 and {MAX_TTL_S}")
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    now = int(time.time())
    claims = {
        "sub": subject,
        "scope": " ".join(scopes),
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "nbf": now,
        "exp": now + ttl_s,
        "jti": uuid.uuid4().hex,
        "client_id": client_id,
    }
    return jwt.encode(claims, key, algorithm="EdDSA")  # type: ignore[arg-type]


class JwtVerifier:
    """Verifies tokens for both the stdio entry point and the HTTP transport.

    It implements the MCP SDK's ``TokenVerifier`` protocol (``verify_token``), so the same
    object plugs into ``MCPServer(token_verifier=...)``.
    """

    def __init__(
        self,
        public_key_pem: bytes,
        *,
        issuer: str = DEFAULT_ISSUER,
        audience: str = DEFAULT_AUDIENCE,
        leeway_s: int = 30,
    ) -> None:
        key = serialization.load_pem_public_key(public_key_pem)
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("expected an Ed25519 public key")
        self._key = key
        self.issuer = issuer
        self.audience = audience
        self.leeway_s = leeway_s

    def claims(self, token: str) -> dict[str, object]:
        try:
            return jwt.decode(
                token,
                self._key,
                algorithms=ALGORITHMS,
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway_s,
                options={"require": ["exp", "iat", "sub", "iss", "aud"]},
            )
        except jwt.PyJWTError as err:
            raise AuthError(f"invalid token: {err}") from None

    def principal(self, token: str) -> Principal:
        return self._principal(self.claims(token))

    @staticmethod
    def _principal(claims: dict[str, object]) -> Principal:
        scope = claims.get("scope", "")
        if not isinstance(scope, str):
            raise AuthError("invalid token: scope must be a space-separated string")
        client_id = claims.get("client_id")
        return Principal(
            subject=str(claims["sub"]),
            scopes=frozenset(scope.split()),
            client_id=str(client_id) if client_id is not None else None,
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            claims = self.claims(token)
            principal = self._principal(claims)
        except AuthError:
            return None
        exp = claims.get("exp")
        return AccessToken(
            token=token,
            client_id=principal.client_id or principal.subject,
            scopes=sorted(principal.scopes),
            expires_at=int(exp) if isinstance(exp, int) else None,
            subject=principal.subject,
            resource=self.audience,
        )
