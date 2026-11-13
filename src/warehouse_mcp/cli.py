"""``gwm``: generate the synthetic warehouse, manage dev keys and tokens, and serve MCP.

gwm generate                         # build data/warehouse.duckdb
gwm keys                             # write an Ed25519 dev keypair to .keys/
gwm token --role analyst             # print a signed dev token
GWM_TOKEN=$(gwm token --role analyst) gwm serve            # stdio (Claude Desktop, Claude Code)
gwm serve --transport http --port 8000                     # Streamable HTTP with bearer auth
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from warehouse_mcp import auth
from warehouse_mcp.agents import load_agents
from warehouse_mcp.policy import ROLE_SCOPES
from warehouse_mcp.synthetic import DEFAULT_SEED, build_warehouse

DEFAULT_DB = Path("data/warehouse.duckdb")
DEFAULT_KEYS = Path(".keys")


def _cmd_generate(args: argparse.Namespace) -> int:
    path = build_warehouse(args.db, seed=args.seed)
    print(f"wrote {path}", file=sys.stderr)
    return 0


def _cmd_keys(args: argparse.Namespace) -> int:
    private, public = auth.generate_keypair(args.dir)
    print(f"wrote {private} (keep secret) and {public}", file=sys.stderr)
    return 0


def _cmd_token(args: argparse.Namespace) -> int:
    scopes = list(ROLE_SCOPES[args.role]) if args.role else []
    scopes += args.scope or []
    if not scopes:
        print("give --role or --scope", file=sys.stderr)
        return 2
    token = auth.issue_token(
        (args.keys / "private.pem").read_bytes(),
        args.subject or f"dev-{args.role or 'user'}",
        scopes,
        ttl_s=args.ttl,
        issuer=args.issuer,
        audience=args.audience,
        client_id=args.client_id,
    )
    print(token)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from mcp.server.auth.settings import AuthSettings
    from mcp.server.transport_security import TransportSecuritySettings
    from pydantic import AnyHttpUrl

    from warehouse_mcp.server import ServerConfig, open_warehouse, principal_from_request

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    verifier = auth.JwtVerifier(
        (args.keys / "public.pem").read_bytes(),
        issuer=args.issuer,
        audience=args.audience,
        agents=load_agents(args.agents),
    )
    gw = open_warehouse(args.db, audit_path=args.audit_log, config=ServerConfig(max_rows=args.max_rows))
    if args.transport == "stdio":
        # stdio has no per-request auth, so the token is verified once at start-up.
        token = os.environ.get("GWM_TOKEN")
        if not token:
            print("set GWM_TOKEN to a signed token (see `gwm token`)", file=sys.stderr)
            return 2
        try:
            principal = verifier.principal(token)
        except auth.AuthError as err:
            print(str(err), file=sys.stderr)
            return 2
        gw.build_mcp(lambda: principal, log_level="INFO").run("stdio")
        return 0
    base_url = (args.public_url or f"http://{args.host}:{args.port}").rstrip("/")
    mcp = gw.build_mcp(
        principal_from_request,
        token_verifier=verifier,
        auth=AuthSettings(
            issuer_url=args.issuer,
            resource_server_url=AnyHttpUrl(f"{base_url}/mcp"),
            required_scopes=["warehouse:public"],
            validate_token_resource=False,
        ),
        log_level="INFO",
    )
    gw.add_health_routes(mcp)
    # DNS-rebinding protection checks the Host header. Behind a container port mapping or a
    # proxy the header is the public name, not the bind address, so deployments list it.
    allowed = [f"{args.host}:{args.port}", args.host, *(args.allowed_host or [])]
    mcp.run(
        "streamable-http",
        host=args.host,
        port=args.port,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=allowed
        ),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gwm", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="build the synthetic warehouse")
    gen.add_argument("--db", type=Path, default=DEFAULT_DB)
    gen.add_argument("--seed", type=int, default=DEFAULT_SEED)
    gen.set_defaults(func=_cmd_generate)

    keys = sub.add_parser("keys", help="write a dev Ed25519 keypair")
    keys.add_argument("--dir", type=Path, default=DEFAULT_KEYS)
    keys.set_defaults(func=_cmd_keys)

    for name, helptext in (("token", "issue a dev token"), ("serve", "run the MCP server")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--keys", type=Path, default=DEFAULT_KEYS)
        p.add_argument("--issuer", default=auth.DEFAULT_ISSUER)
        p.add_argument("--audience", default=auth.DEFAULT_AUDIENCE)
        if name == "token":
            p.add_argument("--role", choices=sorted(ROLE_SCOPES))
            p.add_argument("--scope", action="append", help="extra scope; repeatable")
            p.add_argument("--subject")
            p.add_argument("--ttl", type=int, default=3600)
            p.add_argument("--client-id", default="local-dev", help="the agent the token is for")
            p.set_defaults(func=_cmd_token)
        else:
            p.add_argument("--db", type=Path, default=DEFAULT_DB)
            p.add_argument("--transport", choices=("stdio", "http"), default="stdio")
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--max-rows", type=int, default=200)
            p.add_argument("--audit-log", type=Path, default=None, help="append audit records as JSONL")
            p.add_argument(
                "--agents", type=Path, default=None, help="agent registry YAML (default: local-dev only)"
            )
            p.add_argument(
                "--allowed-host",
                action="append",
                help="extra Host header to accept on HTTP, e.g. localhost:8000; repeatable",
            )
            p.add_argument("--public-url", help="the server's external base URL, for OAuth metadata")
            p.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
