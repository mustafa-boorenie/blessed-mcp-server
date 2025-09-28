# Blender MCP Router

FastMCP server that exposes two layers of functionality:

- **LLM routing** through [LiteLLM](https://github.com/BerriAI/litellm) so a single FastMCP tool can reach OpenAI, Anthropic, xAI, or any other LiteLLM-supported provider.
- **Blender bridge** that proxies to the Blender MCP add-on over a persistent TCP socket, providing scene inspection, PolyHaven / Sketchfab helpers, and Hyper3D automation.

The server is designed for FastMCP Hub distribution: `pyproject.toml` defines the package, the MCP endpoint is hosted via FastMCP, and an optional REST shim is exposed for the Blender add-on.

## Requirements

- Python 3.10+
- Blender MCP add-on running locally (for the Blender tools)
- API keys for any LLM providers you plan to route through LiteLLM

Install dependencies via:

```bash
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and fill in the values.

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Used by LiteLLM when routing to OpenAI models |
| `XAI_API_KEY` | Used for xAI (Grok) requests via LiteLLM |
| `ANTHROPIC_API_KEY` | Used for Anthropics models via LiteLLM |
| `OPENAI_MODEL` | Optional override for the `gpt-5` alias |
| `XAI_MODEL` | Optional override for the `grok-4-fast` alias |
| `ANTHROPIC_MODEL` | Optional override for the `claude-4` alias |
| `MCP_REST_TOKEN` | Shared secret for REST shim (`X-Token` header) |

All LLM-specific environment variables supported by LiteLLM can be passed through here as well (see LiteLLM docs for provider-specific keys).

## Running

After configuration, start the server via the script entry point:

```bash
blender-mcp-router
```

The process starts two services:

- FastMCP HTTP endpoint on `127.0.0.1:8974/mcp`
- REST bridge for the Blender add-on on `127.0.0.1:8975`

Both services are started inside `server.main()` so FastMCP Hub (or `pipx run blender-mcp-router`) can launch them.

## MCP Tools

`server.py` registers the following FastMCP tools:

- `generate_text`: Unified text generation routed through LiteLLM
- Blender tools: `get_scene_info`, `get_object_info`, `get_viewport_screenshot`, `execute_blender_code`, PolyHaven/Sketchfab helpers, and Hyper3D automation helpers

Each Blender tool forwards to the Blender MCP add-on using a JSON-over-TCP API. See that add-on for port configuration (default `localhost:9876`).

### REST Shim `/tools/call`

The REST API exposes a subset of the MCP tools so non-MCP clients (like the Blender add-on) can call them. Requests must include an `X-Token` header if `MCP_REST_TOKEN` is set. The response format mirrors MCP `content` objects (`text`, `json`, `image`).

### Health Check

`GET /health` returns `{ "ok": true }` so deployment targets can monitor the process.

## Development

- Run linting/formatting as desired (none enforced yet).
- The LiteLLM dependency keeps provider selection abstract; add more aliases in `MODEL_MAP` as needed.
- Additional tools can be exposed by adding `@mcp.tool()` functions and listing them in `_HTTP_EXPOSED_TOOL_NAMES` when required by the REST shim.


