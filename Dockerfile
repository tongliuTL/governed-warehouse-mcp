# syntax=docker/dockerfile:1
#
# The governed warehouse MCP server over Streamable HTTP.
#
#   docker build -t governed-warehouse-mcp .
#   uv run gwm keys --dir .keys            # or mount your own Ed25519 public key
#   docker run --rm -p 8000:8000 --read-only --tmpfs /tmp \
#     -v "$PWD/.keys:/keys:ro" governed-warehouse-mcp
#
# The image carries the synthetic warehouse at /data. Mount a different one read-only with
# -v /path/to/data:/data:ro. The process runs as a non-root user and writes nothing outside
# /tmp, so the root filesystem can be read-only.

ARG UV_VERSION=0.13.8
ARG PYTHON_IMAGE=python:3.13-slim-bookworm

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
# Dependencies first, so a source change does not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable \
 && /app/.venv/bin/gwm generate --db /data/warehouse.duckdb

FROM ${PYTHON_IMAGE}
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin gwm
COPY --from=build /app/.venv /app/.venv
# Owned by root and read-only to the server user: the engine opens it read_only as well.
COPY --from=build /data /data
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)"]
ENTRYPOINT ["gwm", "serve", "--transport", "http", "--host", "0.0.0.0", "--port", "8000", \
            "--db", "/data/warehouse.duckdb", "--keys", "/keys"]
# The Host headers the server accepts (DNS-rebinding protection). Replace them with your
# public name when the container sits behind a proxy, e.g. --allowed-host mcp.example.com.
CMD ["--allowed-host", "localhost:8000", "--allowed-host", "127.0.0.1:8000"]
