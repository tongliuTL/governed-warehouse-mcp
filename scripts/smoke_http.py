"""Smoke-test a running HTTP server: probes, auth, and one governed query per role.

    uv run python scripts/smoke_http.py http://localhost:8000 --keys .keys

Exits non-zero on the first failed check. Used by CI against the Docker image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import anyio
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from warehouse_mcp import auth
from warehouse_mcp.policy import ROLE_SCOPES


def _token(keys: str, role: str) -> str:
    pem = (Path(keys) / "private.pem").read_bytes()
    return auth.issue_token(pem, f"smoke-{role}", ROLE_SCOPES[role], ttl_s=300)


async def _run_sql(url: str, token: str, sql: str) -> Any:
    async with (
        httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as http,
        Client(streamable_http_client(f"{url}/mcp", http_client=http)) as client,
    ):
        return await client.call_tool("run_sql", {"sql": sql})


def check(ok: bool, what: str) -> None:
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        raise SystemExit(1)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--keys", default=".keys")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    check(httpx2.get(f"{url}/healthz").json() == {"status": "ok"}, "/healthz")
    ready = httpx2.get(f"{url}/readyz")
    check(ready.status_code == 200 and ready.json()["status"] == "ready", "/readyz")
    unauth = httpx2.post(
        f"{url}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Accept": "application/json, text/event-stream"},
    )
    check(unauth.status_code == 401, "/mcp without a token is 401")

    sql = "select count(*) as n from fct_shipments"
    analyst = await _run_sql(url, _token(args.keys, "analyst"), sql)
    check(not analyst.is_error and analyst.structured_content["rows"][0][0] > 0, "analyst run_sql")
    partner = await _run_sql(url, _token(args.keys, "partner"), sql)
    check(partner.is_error and "UNKNOWN_MODEL" in partner.content[0].text, "partner is denied")
    write = await _run_sql(url, _token(args.keys, "finance"), "drop table fct_invoices")
    check(write.is_error and "READ_ONLY" in write.content[0].text, "writes are refused")


if __name__ == "__main__":
    try:
        anyio.run(main)
    except SystemExit:
        raise
    except Exception as err:  # a connection error is a failed smoke test, not a traceback
        print(f"FAIL {type(err).__name__}: {err}", file=sys.stderr)
        raise SystemExit(1) from None
