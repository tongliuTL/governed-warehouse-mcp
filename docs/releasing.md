# Releasing and directory listings

How a version gets to users: the container image on GitHub Container Registry, the official MCP
Registry entry, and the third-party directories that list the server.

## Cutting a release

1. On a branch, set the new version in:
   - `pyproject.toml`; then run `uv lock`.
   - `server.json`: `version`, and the tag at the end of `packages[0].identifier`.
   - `CHANGELOG.md`: move the `[Unreleased]` entries under the new version and date.
2. Open a pull request and merge it once CI passes.
3. Tag the merge commit on `main` and push the tag:

   ```bash
   git tag v0.2.2 && git push origin v0.2.2
   ```

The tag starts [`.github/workflows/release.yml`](../.github/workflows/release.yml). It runs two
jobs, in order:

1. **`image`** builds the `Dockerfile` for linux/amd64 and linux/arm64. It pushes
   `ghcr.io/tongliutl/governed-warehouse-mcp:<version>` and `:latest`, logging in with
   `GITHUB_TOKEN`.
2. **`mcp-registry`** publishes `server.json` with `mcp-publisher`, pinned and verified with cosign
   against the registry's release signature.
   - It sets `version` and the image tag from the git tag, so a stale `server.json` cannot
     publish the wrong image.
   - It logs in with GitHub OIDC (`mcp-publisher login github-oidc`), so no registry secret is
     stored.

To rerun a release for a tag that already exists, start the workflow by hand (**Actions > Release >
Run workflow**) and give it the tag, for example `v0.2.1`.

### How the registry checks ownership

- **The name.** `io.github.tongliuTL/...` is a GitHub namespace. The OIDC login proves that the
  workflow runs in a `tongliuTL` repository.
- **The image.** The registry also checks that the image carries the label
  `io.modelcontextprotocol.server.name="io.github.tongliuTL/governed-warehouse-mcp"`, which the
  `Dockerfile` sets. If you remove or change the label, publishing fails.
- **Visibility.** The ghcr package must stay public, so that the registry and users can pull it
  without logging in.

### Checking a release

```bash
docker pull ghcr.io/tongliutl/governed-warehouse-mcp:0.2.1
curl -s "https://registry.modelcontextprotocol.io/v0/servers?search=io.github.tongliuTL/governed-warehouse-mcp" | jq '.servers[].server.version'
```

## Directory listings

| Directory | Entry | How it is updated |
|---|---|---|
| [Official MCP Registry](https://registry.modelcontextprotocol.io) | `io.github.tongliuTL/governed-warehouse-mcp` | Automatically, on every `v*` tag |
| [Glama](https://glama.ai/mcp/servers/tongliuTL/governed-warehouse-mcp) | Claimed by the maintainer; score badge in the README | By hand after each release (below) |
| [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) | One line in the Databases section ([PR #20980](https://github.com/punkpeye/awesome-mcp-servers/pull/20980)) | Only when the description changes |
| [mcp.so](https://mcp.so) | Submitted as [chatmcp/mcpso#8246](https://github.com/chatmcp/mcpso/issues/8246) | Only when the description changes |

### Glama

Glama builds its own image from the repository and starts it, to confirm that the server works and
to list its tools. That build is configured in the Glama admin page for the server, not in this
repository. The current settings:

- **Build steps:** `["uv sync --locked --no-dev", "uv run --no-sync gwm generate", "uv run --no-sync gwm keys"]`.
  uv is already in Glama's base image; pip may not be.
- **Command:**

  ```json
  ["sh", "-c", "case $GWM_TOKEN in demo) export GWM_TOKEN=$(uv run --no-sync gwm token --role analyst --ttl 86400);; esac; exec uv run --no-sync gwm serve"]
  ```

- **Placeholder parameters:** `{"GWM_TOKEN": "demo"}`.

Glama's checks start the server with a placeholder value, so the build creates throwaway dev keys.
When the token is the literal `demo`, the build signs a real analyst token at startup. This applies
only to Glama's copy:
- The token grants analyst access to the synthetic demo warehouse.
- It does not change the published image or any real deployment, which always need a token signed
  by your own key.

After each release:
1. On the Glama admin page, run a new test build.
2. Create the matching release, so the score and tool list stay current.

### Listing copy

Use the same text in every directory.

- **One-liner (100 characters at most, the registry limit):** Governed MCP server for warehouse
  data: semantic layer, scoped access, PII controls and SQL guard.
- **Short description:** An MCP server that lets AI agents query an analytical warehouse safely:
  governed metrics, scoped access, PII controls, a read-only SQL guard and audit logs.
- **Category:** Databases. Also fits Developer tools and Security.
- **Tools:** `search_models`, `describe_model`, `list_metrics`, `query_metric`, `run_sql`; resource
  `catalog://version`.
- **awesome-mcp-servers line:**

  ```markdown
  - [tongliuTL/governed-warehouse-mcp](https://github.com/tongliuTL/governed-warehouse-mcp) [![tongliuTL/governed-warehouse-mcp MCP server](https://glama.ai/mcp/servers/tongliuTL/governed-warehouse-mcp/badges/score.svg)](https://glama.ai/mcp/servers/tongliuTL/governed-warehouse-mcp) 🐍 🏠 - Governed access to an analytical warehouse: semantic-layer metrics, scoped JWT access, PII controls, a read-only SQL guard and audit logs, with an offline eval harness for tool-calling agents.
  ```

When a submission asks for a disclosure, state that the maintainer is the author.
