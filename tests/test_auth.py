from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import serialization

from tests.util import run
from warehouse_mcp import auth
from warehouse_mcp.agents import AgentRegistry, load_agents
from warehouse_mcp.policy import ROLE_SCOPES, Principal


@pytest.fixture(scope="module")
def keys(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, bytes]:
    private, public = auth.generate_keypair(tmp_path_factory.mktemp("keys"))
    assert private.stat().st_mode & 0o777 == 0o600
    return private.read_bytes(), public.read_bytes()


@pytest.fixture(scope="module")
def verifier(keys: tuple[bytes, bytes]) -> auth.JwtVerifier:
    return auth.JwtVerifier(keys[1])


def test_round_trip(keys: tuple[bytes, bytes], verifier: auth.JwtVerifier) -> None:
    token = auth.issue_token(keys[0], "alice", ROLE_SCOPES["analyst"])
    principal = verifier.principal(token)
    assert principal == Principal("alice", frozenset(ROLE_SCOPES["analyst"]), client_id="local-dev")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _forge(claims: dict[str, object], alg: str, secret: bytes | None = None) -> str:
    """Hand-build a JWT, for attacks a well-behaved JWT library refuses to produce."""
    signing_input = (
        f"{_b64(json.dumps({'alg': alg, 'typ': 'JWT'}).encode())}.{_b64(json.dumps(claims).encode())}"
    )
    signature = _b64(hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()) if secret else ""
    return f"{signing_input}.{signature}"


def _claims(**overrides: object) -> dict[str, object]:
    now = int(time.time())
    return {
        "sub": "mallory",
        "scope": "warehouse:public warehouse:internal warehouse:restricted pii:read",
        "iss": auth.DEFAULT_ISSUER,
        "aud": auth.DEFAULT_AUDIENCE,
        "iat": now,
        "exp": now + 600,
        **overrides,
    }


def _sign(private_pem: bytes, claims: dict[str, object]) -> str:
    key = serialization.load_pem_private_key(private_pem, None)
    return jwt.encode(claims, key, algorithm="EdDSA")  # type: ignore[arg-type]


def test_rejects_bad_tokens(keys: tuple[bytes, bytes], tmp_path: Path) -> None:
    private, public = keys
    other_private, _ = auth.generate_keypair(tmp_path)
    now = int(time.time())
    bad = {
        "expired": _sign(private, _claims(iat=now - 120, exp=now - 60)),
        "not yet valid": _sign(private, _claims(nbf=now + 600)),
        "wrong key": auth.issue_token(other_private.read_bytes(), "a", ["warehouse:public"]),
        "wrong audience": auth.issue_token(private, "a", ["warehouse:public"], audience="someone-else"),
        "wrong issuer": auth.issue_token(private, "a", ["warehouse:public"], issuer="https://evil.example"),
        "alg none": _forge(_claims(), "none"),
        "HS256 keyed with the public key": _forge(_claims(), "HS256", secret=public),
        "missing exp": _sign(private, {k: v for k, v in _claims().items() if k != "exp"}),
        "missing sub": _sign(private, {k: v for k, v in _claims().items() if k != "sub"}),
        "tampered payload": auth.issue_token(private, "a", ["warehouse:public"])[:-8] + "AAAAAAAA",
        "garbage": "not.a.token",
    }
    verifier = auth.JwtVerifier(public, leeway_s=0)
    for name, token in bad.items():
        with pytest.raises(auth.AuthError):
            verifier.principal(token)
        assert run(verifier.verify_token, token) is None, name


def test_scope_must_be_a_string(keys: tuple[bytes, bytes], verifier: auth.JwtVerifier) -> None:
    token = _sign(keys[0], _claims(scope=["warehouse:public"]))
    with pytest.raises(auth.AuthError, match="scope"):
        verifier.principal(token)


def test_verify_token_for_http_transport(keys: tuple[bytes, bytes], verifier: auth.JwtVerifier) -> None:
    token = auth.issue_token(keys[0], "bob", ["warehouse:public"], ttl_s=60)
    access = run(verifier.verify_token, token)
    assert access is not None
    assert (
        access.subject == "bob" and access.scopes == ["warehouse:public"] and access.client_id == "local-dev"
    )
    assert access.expires_at is not None and access.expires_at > time.time()


def test_ttl_bounds(keys: tuple[bytes, bytes]) -> None:
    with pytest.raises(ValueError, match="ttl_s"):
        auth.issue_token(keys[0], "a", ["warehouse:public"], ttl_s=auth.MAX_TTL_S + 1)


def test_verifier_requires_ed25519(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives.asymmetric import rsa

    pem = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    with pytest.raises(ValueError, match="Ed25519"):
        auth.JwtVerifier(pem)


def test_principal_tiers() -> None:
    assert Principal.for_role("partner").max_tier is not None
    assert Principal("x").max_tier is None
    assert not Principal("x").can_read_pii
    assert Principal.for_role("support").can_read_pii


TWO_AGENTS = AgentRegistry.model_validate(
    {
        "agents": {
            "local-dev": {"owner": "dev", "permitted_scopes": ["warehouse:public", "warehouse:internal"]},
            "ops-assistant": {"owner": "operations", "permitted_scopes": ["warehouse:public"]},
            "retired-bot": {"owner": "nobody", "permitted_scopes": ["warehouse:public"], "enabled": False},
        }
    }
)


def test_second_agent_is_one_registry_entry_and_scopes_are_intersected(keys: tuple[bytes, bytes]) -> None:
    verifier = auth.JwtVerifier(keys[1], agents=TWO_AGENTS)
    # The identity provider over-issues finance scopes; each agent keeps only what it is permitted.
    finance = list(ROLE_SCOPES["finance"])
    ops = verifier.principal(auth.issue_token(keys[0], "svc", finance, client_id="ops-assistant"))
    assert ops.scopes == {"warehouse:public"} and ops.client_id == "ops-assistant"
    dev = verifier.principal(auth.issue_token(keys[0], "svc", finance, client_id="local-dev"))
    assert dev.scopes == {"warehouse:public", "warehouse:internal"}
    access = run(verifier.verify_token, auth.issue_token(keys[0], "svc", finance, client_id="ops-assistant"))
    assert access is not None and access.scopes == ["warehouse:public"]


@pytest.mark.parametrize("client_id", ["retired-bot", "never-registered"])
def test_unknown_or_disabled_agent_is_rejected(keys: tuple[bytes, bytes], client_id: str) -> None:
    verifier = auth.JwtVerifier(keys[1], agents=TWO_AGENTS)
    token = auth.issue_token(keys[0], "svc", ["warehouse:public"], client_id=client_id)
    with pytest.raises(auth.AuthError, match="not enabled"):
        verifier.principal(token)
    assert run(verifier.verify_token, token) is None


def test_token_without_client_id_is_rejected(keys: tuple[bytes, bytes], verifier: auth.JwtVerifier) -> None:
    signed = auth.issue_token(keys[0], "svc", ["warehouse:public"])
    claims = jwt.decode(signed, options={"verify_signature": False})
    del claims["client_id"]
    private = serialization.load_pem_private_key(keys[0], password=None)
    token = jwt.encode(claims, private, algorithm="EdDSA")  # type: ignore[arg-type]
    with pytest.raises(auth.AuthError, match="no client_id"):
        verifier.principal(token)


def test_packaged_registry_only_registers_local_dev() -> None:
    assert set(load_agents().agents) == {"local-dev"}
