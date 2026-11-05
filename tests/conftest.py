from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from warehouse_evals import cli as eval_cli
from warehouse_evals import tracing
from warehouse_evals.runner import EvalContext
from warehouse_mcp.catalog import Catalog, load_catalog
from warehouse_mcp.policy import Principal
from warehouse_mcp.server import GovernedWarehouse, ServerConfig
from warehouse_mcp.synthetic import build_warehouse
from warehouse_mcp.warehouse import Warehouse


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_warehouse(tmp_path_factory.mktemp("wh") / "warehouse.duckdb")


@pytest.fixture(scope="session")
def raw_db(db_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A copy for plain DuckDB connections.

    DuckDB shares one database instance per file within a process and refuses a second
    connection with a different configuration, so tests that inspect the file directly use
    their own copy instead of the hardened warehouse's.
    """
    copy = tmp_path_factory.mktemp("raw") / "warehouse.duckdb"
    shutil.copyfile(db_path, copy)
    return copy


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return load_catalog()


@pytest.fixture(scope="session")
def warehouse(db_path: Path) -> Iterator[Warehouse]:
    wh = Warehouse(db_path, timeout_s=5)
    yield wh
    wh.close()


@pytest.fixture
def governed(warehouse: Warehouse, catalog: Catalog) -> GovernedWarehouse:
    return GovernedWarehouse(warehouse=warehouse, catalog=catalog, config=ServerConfig())


@pytest.fixture(params=["partner", "analyst", "finance", "support"])
def any_role(request: pytest.FixtureRequest) -> Principal:
    return Principal.for_role(request.param)


@pytest.fixture
def analyst() -> Principal:
    return Principal.for_role("analyst")


@pytest.fixture
def finance() -> Principal:
    return Principal.for_role("finance")


@pytest.fixture
def partner() -> Principal:
    return Principal.for_role("partner")


@pytest.fixture
def support() -> Principal:
    return Principal.for_role("support")


@pytest.fixture(scope="session")
def ctx(db_path: Path) -> Iterator[EvalContext]:
    ctx = EvalContext(db_path)
    yield ctx
    ctx.close()


@pytest.fixture(autouse=True)
def _no_local_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never read a developer's .env or exported Langfuse keys: no live calls, no traces."""
    monkeypatch.setattr(eval_cli, "ENV_FILE", tmp_path / "absent.env")
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tracing._state, "live", None)
