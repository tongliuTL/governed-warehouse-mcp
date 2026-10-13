"""The shipped entry points: `gwm` commands, stdio transport and Streamable HTTP with bearer auth."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from tests.util import run
from warehouse_mcp import cli


@pytest.fixture(scope="module")
def keys(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("keys")
    assert cli.main(["keys", "--dir", str(directory)]) == 0
    return directory


def token(keys: Path, role: str, capsys: pytest.CaptureFixture[str]) -> str:
    assert cli.main(["token", "--role", role, "--keys", str(keys)]) == 0
    return capsys.readouterr().out.strip()


def test_generate(tmp_path: Path) -> None:
    assert cli.main(["generate", "--db", str(tmp_path / "w.duckdb"), "--seed", "3"]) == 0
    assert (tmp_path / "w.duckdb").stat().st_size > 0


def test_token_needs_scopes(keys: Path) -> None:
    assert cli.main(["token", "--keys", str(keys)]) == 2


def test_token_extra_scopes(keys: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from warehouse_mcp.auth import JwtVerifier

    assert cli.main(["token", "--scope", "warehouse:public", "--scope", "pii:read", "--keys", str(keys)]) == 0
    principal = JwtVerifier((keys / "public.pem").read_bytes()).principal(capsys.readouterr().out.strip())
    assert principal.scopes == {"warehouse:public", "pii:read"}


def test_stdio_refuses_to_start_without_a_valid_token(
    db_path: Path, keys: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GWM_TOKEN", raising=False)
    args = ["serve", "--db", str(db_path), "--keys", str(keys)]
    assert cli.main(args) == 2
    monkeypatch.setenv("GWM_TOKEN", "forged")
    assert cli.main(args) == 2


def _env(**extra: str) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(Path(__file__).parents[1] / "src"), os.environ.get("PYTHONPATH", "")]
        ),
        **extra,
    }


def test_stdio_end_to_end(db_path: Path, keys: Path, capsys: pytest.CaptureFixture[str]) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "warehouse_mcp", "serve", "--db", str(db_path), "--keys", str(keys)],
        env=_env(GWM_TOKEN=token(keys, "partner", capsys)),
    )

    async def go() -> tuple[Any, Any]:
        async with Client(params) as client:
            ok = await client.call_tool("query_metric", {"metric": "avg_border_dwell_hours"})
            denied = await client.call_tool("run_sql", {"sql": "select count(*) from fct_shipments"})
            return ok, denied

    ok, denied = run(go)
    assert not ok.is_error and ok.structured_content["rows"][0][0] > 0
    assert denied.is_error and "UNKNOWN_MODEL" in denied.content[0].text


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def http_server(
    db_path: Path, keys: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[str, Path]]:
    port = _free_port()
    audit = tmp_path_factory.mktemp("audit") / "audit.jsonl"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "warehouse_mcp",
            "serve",
            "--transport",
            "http",
            "--port",
            str(port),
            "--db",
            str(db_path),
            "--keys",
            str(keys),
            "--audit-log",
            str(audit),
        ],
        env=_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(proc.stderr.read().decode() if proc.stderr else "server exited") from None
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/mcp", audit
    proc.terminate()
    proc.wait(timeout=10)


def test_http_requires_a_valid_bearer_token(http_server: tuple[str, Path]) -> None:
    url, _ = http_server
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    headers = {"Accept": "application/json, text/event-stream"}
    missing = httpx2.post(url, json=body, headers=headers)
    assert missing.status_code == 401 and "Bearer" in missing.headers["www-authenticate"]
    forged = httpx2.post(url, json=body, headers={**headers, "Authorization": "Bearer abc.def.ghi"})
    assert forged.status_code == 401


def test_http_scopes_follow_the_token(
    http_server: tuple[str, Path], keys: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url, audit = http_server
    tokens = {role: token(keys, role, capsys) for role in ("partner", "analyst")}

    async def go(role: str) -> Any:
        async with (
            httpx2.AsyncClient(headers={"Authorization": f"Bearer {tokens[role]}"}) as http,
            Client(streamable_http_client(url, http_client=http)) as client,
        ):
            return await client.call_tool("run_sql", {"sql": "select count(*) as n from fct_shipments"})

    partner, analyst = run(go, "partner"), run(go, "analyst")
    assert partner.is_error and "UNKNOWN_MODEL" in partner.content[0].text
    assert not analyst.is_error and analyst.structured_content["rows"] == [[12000]]
    time.sleep(0.2)
    lines = audit.read_text().splitlines()
    assert any('"principal": "dev-partner"' in line and '"deny"' in line for line in lines)
    assert any('"principal": "dev-analyst"' in line and '"allow"' in line for line in lines)
