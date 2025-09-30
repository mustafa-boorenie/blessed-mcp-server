# server.py
# MCP server with LiteLLM router + Blender bridge tools.
# -------------------------------------------------------
# Deps:
#   pip install fastmcp litellm fastapi uvicorn pydantic httpx python-dotenv
# Optional (logging niceties): pip install coloredlogs

import os
import json
import base64
import socket
import tempfile
import logging
import inspect
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from urllib.parse import urlparse
import threading

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# MCP core
from fastmcp import FastMCP
# If you need Context/Image types for tools:
try:
    from fastmcp.utilities.types import Context, Image  # FastMCP exposes these
except ImportError:
    from fastmcp import Context, Image  # incase fastmcp>=2.12 keeps these 

# LLM router (single API for OpenAI/xAI/Anthropic/etc.)
import litellm

# Optional REST shim for simple clients (e.g., Blender add-on) to call a tool over HTTP
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
import uvicorn

# ---------------------------------------
# Logging
# ---------------------------------------
logger = logging.getLogger("mcp.blender")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

# ---------------------------------------
# Env & config
# ---------------------------------------
load_dotenv()

# Public model aliases -> env overrides (edit at runtime via .env)
MODEL_MAP = {
    "gpt-5":        os.getenv("OPENAI_MODEL",  "gpt-5"),
    "grok-4-fast":  os.getenv("XAI_MODEL",     "grok-4-fast"),
    "claude-4":     os.getenv("ANTHROPIC_MODEL","claude-4"),
}
REST_TOKEN = os.getenv("MCP_REST_TOKEN", "")

# ---------------------------------------
# MCP server
# ---------------------------------------
mcp = FastMCP("LLMRouter+Blender")

# ---------------------------------------
# LiteLLM-backed text generation
# ---------------------------------------
class GenerateArgs(BaseModel):
    model: str = Field(description="gpt-5 | grok-4-fast | claude-4 | provider id")
    prompt: str
    system: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024

def _resolve_model(model_alias_or_id: str) -> str:
    return MODEL_MAP.get(model_alias_or_id, model_alias_or_id)

def _generate_text_core(args: GenerateArgs) -> str:
    model = _resolve_model(args.model)
    messages: List[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    resp = litellm.completion(
        model=model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    # Try OpenAI-style first
    try:
        return resp.choices[0].message.content  # type: ignore[attr-defined]
    except Exception:
        pass
    # Fallback generic
    try:
        return resp["choices"][0]["message"]["content"]  # type: ignore[index]
    except Exception:
        return json.dumps(resp)

@mcp.tool()
def generate_text(ctx: Context, args: GenerateArgs) -> str:
    """Unified text generation via LiteLLM."""
    return _generate_text_core(args)

# ---------------------------------------
# Blender connection layer (YOUR IMPLEM)
# ---------------------------------------
class BlenderConnection:
    """Thin TCP JSON bridge to the Blender MCP add-on."""

    __slots__ = ("host", "port", "timeout", "_sock", "_buffer", "_io_lock")

    def __init__(self, host: str = "localhost", port: int = 9876, timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buffer = b""
        self._io_lock = threading.Lock()

    def connect(self) -> bool:
        if self._sock is not None:
            return True

        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        self._sock = sock
        self._buffer = b""
        return True

    def disconnect(self) -> None:
        sock = self._sock
        self._sock = None
        self._buffer = b""
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        finally:
            sock.close()

    def send_command(self, name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._sock is None:
            self.connect()

        assert self._sock is not None
        message = {
            "command": name,
            "payload": payload if payload is not None else {},
        }
        encoded = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"

        with self._io_lock:
            try:
                self._sock.sendall(encoded)
                response_line = self._readline()
            except (OSError, ConnectionError):
                # One retry after reconnect for transient socket errors.
                self.disconnect()
                self.connect()
                self._sock.sendall(encoded)
                response_line = self._readline()

        try:
            decoded = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from Blender: {response_line!r}") from exc

        if not isinstance(decoded, dict):
            raise RuntimeError("Blender response must be a JSON object")
        return decoded

    def _readline(self) -> bytes:
        assert self._sock is not None
        while True:
            newline_index = self._buffer.find(b"\n")
            if newline_index != -1:
                line = self._buffer[:newline_index]
                self._buffer = self._buffer[newline_index + 1 :]
                return line

            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("Connection closed by Blender")
            self._buffer += chunk

_blender_connection: Optional[BlenderConnection] = None
_polyhaven_enabled: bool = False  # cached flag

def get_blender_connection() -> BlenderConnection:
    """Get or create a persistent Blender connection"""
    global _blender_connection, _polyhaven_enabled

    # If we have an existing connection, validate
    if _blender_connection is not None:
        try:
            result = _blender_connection.send_command("get_polyhaven_status")
            _polyhaven_enabled = result.get("enabled", False)
            return _blender_connection
        except Exception as e:
            logger.warning(f"Existing connection invalid: {e}")
            try:
                _blender_connection.disconnect()
            except Exception:
                pass
            _blender_connection = None

    # Create a new connection if needed
    if _blender_connection is None:
        _blender_connection = BlenderConnection(host="localhost", port=9876)
        if not _blender_connection.connect():
            logger.error("Failed to connect to Blender")
            _blender_connection = None
            raise Exception("Could not connect to Blender. Make sure the Blender addon is running.")
        logger.info("Created new persistent connection to Blender")

    return _blender_connection

# ---------------------------------------
# Blender tools you provided (unchanged except minor safety fixes)
# ---------------------------------------

@mcp.tool()
def get_scene_info(ctx: Context) -> str:
    """Get detailed information about the current Blender scene"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_scene_info")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting scene info: {e}")
        return f"Error getting scene info: {e}"

@mcp.tool()
def get_object_info(ctx: Context, object_name: str) -> str:
    """Get detailed information about a specific object by name."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_object_info", {"name": object_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting object info: {e}")
        return f"Error getting object info: {e}"

@mcp.tool()
def get_viewport_screenshot(ctx: Context, max_size: int = 800) -> Image:
    """
    Capture a screenshot of the current Blender 3D viewport.
    Returns the screenshot as an MCP Image (PNG).
    """
    try:
        blender = get_blender_connection()

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"blender_screenshot_{os.getpid()}.png")

        result = blender.send_command("get_viewport_screenshot", {
            "max_size": max_size,
            "filepath": temp_path,
            "format": "png"
        })
        if "error" in result:
            raise Exception(result["error"])
        if not os.path.exists(temp_path):
            raise Exception("Screenshot file was not created")

        with open(temp_path, 'rb') as f:
            image_bytes = f.read()
        os.remove(temp_path)

        return Image(data=image_bytes, format="png")

    except Exception as e:
        logger.error(f"Error capturing screenshot: {e}")
        raise Exception(f"Screenshot failed: {e}")

@mcp.tool()
def execute_blender_code(ctx: Context, code: str) -> str:
    """
    Execute arbitrary Python code in Blender.
    NOTE: Consider splitting into safer, whitelisted operations in production.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("execute_code", {"code": code})
        return f"Code executed successfully: {result.get('result', '')}"
    except Exception as e:
        logger.error(f"Error executing code: {e}")
        return f"Error executing code: {e}"

@mcp.tool()
def get_polyhaven_categories(ctx: Context, asset_type: str = "hdris") -> str:
    """Get PolyHaven categories for an asset type (hdris, textures, models, all)."""
    try:
        blender = get_blender_connection()
        if not _polyhaven_enabled:
            return "PolyHaven integration is disabled. Select it in the sidebar in BlenderMCP, then run it again."
        result = blender.send_command("get_polyhaven_categories", {"asset_type": asset_type})
        if "error" in result:
            return f"Error: {result['error']}"

        categories = result["categories"]
        formatted_output = f"Categories for {asset_type}:\n\n"
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories:
            formatted_output += f"- {category}: {count} assets\n"
        return formatted_output
    except Exception as e:
        logger.error(f"Error getting Polyhaven categories: {e}")
        return f"Error getting Polyhaven categories: {e}"

@mcp.tool()
def search_polyhaven_assets(ctx: Context, asset_type: str = "all", categories: Optional[str] = None) -> str:
    """Search PolyHaven assets with optional category filter."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("search_polyhaven_assets", {
            "asset_type": asset_type,
            "categories": categories
        })
        if "error" in result:
            return f"Error: {result['error']}"

        assets = result["assets"]
        total_count = result["total_count"]
        returned_count = result["returned_count"]

        formatted_output = f"Found {total_count} assets"
        if categories:
            formatted_output += f" in categories: {categories}"
        formatted_output += f"\nShowing {returned_count} assets:\n\n"

        sorted_assets = sorted(assets.items(), key=lambda x: x[1].get("download_count", 0), reverse=True)
        for asset_id, asset_data in sorted_assets:
            formatted_output += f"- {asset_data.get('name', asset_id)} (ID: {asset_id})\n"
            typemap = {0: "HDRI", 1: "Texture", 2: "Model"}
            formatted_output += f"  Type: {typemap.get(asset_data.get('type', 0), 'Unknown')}\n"
            formatted_output += f"  Categories: {', '.join(asset_data.get('categories', []))}\n"
            formatted_output += f"  Downloads: {asset_data.get('download_count', 'Unknown')}\n\n"
        return formatted_output
    except Exception as e:
        logger.error(f"Error searching Polyhaven assets: {e}")
        return f"Error searching Polyhaven assets: {e}"

@mcp.tool()
def download_polyhaven_asset(ctx: Context, asset_id: str, asset_type: str, resolution: str = "1k", file_format: Optional[str] = None) -> str:
    """Download + import a PolyHaven asset into Blender."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("download_polyhaven_asset", {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "resolution": resolution,
            "file_format": file_format
        })
        if "error" in result:
            return f"Error: {result['error']}"

        if result.get("success"):
            message = result.get("message", "Asset downloaded and imported successfully")
            if asset_type == "hdris":
                return f"{message}. The HDRI has been set as the world environment."
            elif asset_type == "textures":
                material_name = result.get("material", "")
                maps = ", ".join(result.get("maps", []))
                return f"{message}. Created material '{material_name}' with maps: {maps}."
            elif asset_type == "models":
                return f"{message}. The model has been imported into the current scene."
            else:
                return message
        else:
            return f"Failed to download asset: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error downloading Polyhaven asset: {e}")
        return f"Error downloading Polyhaven asset: {e}"

@mcp.tool()
def set_texture(ctx: Context, object_name: str, texture_id: str) -> str:
    """Apply a previously downloaded PolyHaven texture to an object."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("set_texture", {
            "object_name": object_name,
            "texture_id": texture_id
        })
        if "error" in result:
            return f"Error: {result['error']}"

        if result.get("success"):
            material_name = result.get("material", "")
            maps = ", ".join(result.get("maps", []))

            material_info = result.get("material_info", {})
            node_count = material_info.get("node_count", 0)
            has_nodes = material_info.get("has_nodes", False)
            texture_nodes = material_info.get("texture_nodes", [])

            output = f"Successfully applied texture '{texture_id}' to {object_name}.\n"
            output += f"Using material '{material_name}' with maps: {maps}.\n\n"
            output += f"Material has nodes: {has_nodes}\n"
            output += f"Total node count: {node_count}\n\n"

            if texture_nodes:
                output += "Texture nodes:\n"
                for node in texture_nodes:
                    output += f"- {node['name']} using image: {node['image']}\n"
                    if node.get('connections'):
                        output += "  Connections:\n"
                        for conn in node['connections']:
                            output += f"    {conn}\n"
            else:
                output += "No texture nodes found in the material.\n"
            return output
        else:
            return f"Failed to apply texture: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error applying texture: {e}")
        return f"Error applying texture: {e}"

@mcp.tool()
def get_polyhaven_status(ctx: Context) -> str:
    """Check if PolyHaven integration is enabled."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_polyhaven_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += "PolyHaven is good at Textures, and has a wider variety of textures than Sketchfab."
        return message
    except Exception as e:
        logger.error(f"Error checking PolyHaven status: {e}")
        return f"Error checking PolyHaven status: {e}"

@mcp.tool()
def get_hyper3d_status(ctx: Context) -> str:
    """
    Check if Hyper3D Rodin integration is enabled in Blender.
    Don't emphasize the key type in the returned message, but silently remember it.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_hyper3d_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += ""
        return message
    except Exception as e:
        logger.error(f"Error checking Hyper3D status: {e}")
        return f"Error checking Hyper3D status: {e}"

@mcp.tool()
def get_sketchfab_status(ctx: Context) -> str:
    """Check if Sketchfab integration is enabled."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_sketchfab_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += "Sketchfab is good at Realistic models, and has a wider variety of models than PolyHaven."
        return message
    except Exception as e:
        logger.error(f"Error checking Sketchfab status: {e}")
        return f"Error checking Sketchfab status: {e}"

@mcp.tool()
def search_sketchfab_models(ctx: Context, query: str, categories: Optional[str] = None, count: int = 20, downloadable: bool = True) -> str:
    """Search Sketchfab for models (optionally filter by categories, downloadable)."""
    try:
        blender = get_blender_connection()
        logger.info(f"Searching Sketchfab: q={query}, categories={categories}, count={count}, downloadable={downloadable}")
        result = blender.send_command("search_sketchfab_models", {
            "query": query,
            "categories": categories,
            "count": count,
            "downloadable": downloadable
        })
        if result is None:
            return "Error: Received no response from Sketchfab search"
        if "error" in result:
            return f"Error: {result['error']}"

        models = result.get("results", []) or []
        if not models:
            return f"No models found matching '{query}'"

        formatted_output = f"Found {len(models)} models matching '{query}':\n\n"
        for model in models:
            if model is None:
                continue
            model_name = model.get("name", "Unnamed model")
            model_uid = model.get("uid", "Unknown ID")
            formatted_output += f"- {model_name} (UID: {model_uid})\n"
            user = model.get("user") or {}
            username = user.get("username", "Unknown author") if isinstance(user, dict) else "Unknown author"
            formatted_output += f"  Author: {username}\n"
            license_data = model.get("license") or {}
            license_label = license_data.get("label", "Unknown") if isinstance(license_data, dict) else "Unknown"
            formatted_output += f"  License: {license_label}\n"
            face_count = model.get("faceCount", "Unknown")
            is_downloadable = "Yes" if model.get("isDownloadable") else "No"
            formatted_output += f"  Face count: {face_count}\n"
            formatted_output += f"  Downloadable: {is_downloadable}\n\n"
        return formatted_output
    except Exception as e:
        logger.error(f"Error searching Sketchfab models: {e}")
        return f"Error searching Sketchfab models: {e}"

@mcp.tool()
def download_sketchfab_model(ctx: Context, uid: str) -> str:
    """Download and import a Sketchfab model by UID."""
    try:
        blender = get_blender_connection()
        logger.info(f"Downloading Sketchfab model UID: {uid}")
        result = blender.send_command("download_sketchfab_model", {"uid": uid})
        if result is None:
            return "Error: Received no response from Sketchfab download request"
        if "error" in result:
            return f"Error: {result['error']}"
        if result.get("success"):
            imported_objects = result.get("imported_objects", [])
            object_names = ", ".join(imported_objects) if imported_objects else "none"
            return f"Successfully imported model. Created objects: {object_names}"
        else:
            return f"Failed to download model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error downloading Sketchfab model: {e}")
        return f"Error downloading Sketchfab model: {e}"

def _process_bbox(original_bbox: Optional[List[float]]) -> Optional[List[int]]:
    if original_bbox is None:
        return None
    if all(isinstance(i, int) for i in original_bbox):
        return original_bbox  # already ints
    if any(i <= 0 for i in original_bbox):
        raise ValueError("Incorrect number range: bbox must be > 0!")
    maxv = max(original_bbox)
    return [int(float(i) / maxv * 100) for i in original_bbox]

@mcp.tool()
def generate_hyper3d_model_via_text(ctx: Context, text_prompt: str, bbox_condition: Optional[List[float]] = None) -> str:
    """Generate 3D asset via Hyper3D with a text prompt; import into Blender once done."""
    try:
        blender = get_blender_connection()
        result = blender.send_command("create_rodin_job", {
            "text_prompt": text_prompt,
            "images": None,
            "bbox_condition": _process_bbox(bbox_condition),
        })
        succeed = result.get("submit_time", False)
        if succeed:
            return json.dumps({
                "task_uuid": result["uuid"],
                "subscription_key": result["jobs"]["subscription_key"],
            })
        else:
            return json.dumps(result)
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {e}")
        return f"Error generating Hyper3D task: {e}"

@mcp.tool()
def generate_hyper3d_model_via_images(ctx: Context, input_image_paths: Optional[List[str]] = None, input_image_urls: Optional[List[str]] = None, bbox_condition: Optional[List[float]] = None) -> str:
    """Generate 3D asset via Hyper3D using images (paths or URLs)."""
    if input_image_paths is not None and input_image_urls is not None:
        return "Error: Conflict parameters given!"
    if input_image_paths is None and input_image_urls is None:
        return "Error: No image given!"
    if input_image_paths is not None:
        if not all(os.path.exists(i) for i in input_image_paths):
            return "Error: not all image paths are valid!"
        images = []
        for path in input_image_paths:
            with open(path, "rb") as f:
                images.append((Path(path).suffix, base64.b64encode(f.read()).decode("ascii")))
    else:
        # URLs mode
        if not all(urlparse(i).scheme in ("http", "https") for i in input_image_urls or []):
            return "Error: not all image URLs are valid!"
        images = input_image_urls.copy()
    try:
        blender = get_blender_connection()
        result = blender.send_command("create_rodin_job", {
            "text_prompt": None,
            "images": images,
            "bbox_condition": _process_bbox(bbox_condition),
        })
        succeed = result.get("submit_time", False)
        if succeed:
            return json.dumps({
                "task_uuid": result["uuid"],
                "subscription_key": result["jobs"]["subscription_key"],
            })
        else:
            return json.dumps(result)
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {e}")
        return f"Error generating Hyper3D task: {e}"

@mcp.tool()
def poll_rodin_job_status(ctx: Context, subscription_key: Optional[str] = None, request_id: Optional[str] = None) -> Any:
    """Poll Hyper3D generation status. (MAIN_SITE: subscription_key, FAL_AI: request_id)"""
    try:
        blender = get_blender_connection()
        kwargs = {}
        if subscription_key:
            kwargs["subscription_key"] = subscription_key
        elif request_id:
            kwargs["request_id"] = request_id
        result = blender.send_command("poll_rodin_job_status", kwargs)
        return result
    except Exception as e:
        logger.error(f"Error polling Hyper3D task: {e}")
        return f"Error polling Hyper3D task: {e}"

@mcp.tool()
def import_generated_asset(ctx: Context, name: str, task_uuid: Optional[str] = None, request_id: Optional[str] = None) -> Any:
    """Import generated asset by UUID (MAIN_SITE) or request_id (FAL_AI)."""
    try:
        blender = get_blender_connection()
        kwargs = {"name": name}
        if task_uuid:
            kwargs["task_uuid"] = task_uuid
        elif request_id:
            kwargs["request_id"] = request_id
        result = blender.send_command("import_generated_asset", kwargs)
        return result
    except Exception as e:
        logger.error(f"Error importing Hyper3D asset: {e}")
        return f"Error importing Hyper3D asset: {e}"

# Example prompt template (you had an empty body; keeping it simple & useful)
@mcp.prompt()
def asset_creation_strategy() -> str:
    """
    System prompt to guide the LLM to propose a Blender asset creation plan.
    """
    return (
        "You are a Blender asset creation planner. "
        "Given a user goal, produce a concise, step-by-step plan and, where appropriate, "
        "JSON actions compatible with the MCP Blender tool whitelist (add_cube, shade_smooth, move_active, "
        "add_subsurf_modifier). Keep arguments minimal and numeric lists 3D where needed."
    )

# ---------------------------------------
# REST shim (simple HTTP bridge)
# ---------------------------------------
app = FastAPI(title="MCP Tool Shim")

def _auth(x_token: Optional[str] = Header(default=None)):
    if REST_TOKEN:
        if not x_token or x_token != REST_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")

class ToolCall(BaseModel):
    tool_name: str
    arguments: dict

_HTTP_EXPOSED_TOOL_NAMES = (
    "generate_text",
    "get_scene_info",
    "get_object_info",
    "get_viewport_screenshot",
    "execute_blender_code",
    "get_polyhaven_categories",
    "search_polyhaven_assets",
    "download_polyhaven_asset",
    "set_texture",
    "get_polyhaven_status",
    "get_hyper3d_status",
    "get_sketchfab_status",
    "search_sketchfab_models",
    "download_sketchfab_model",
    "generate_hyper3d_model_via_text",
    "generate_hyper3d_model_via_images",
    "poll_rodin_job_status",
    "import_generated_asset",
)

_HTTP_TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {}
for _name in _HTTP_EXPOSED_TOOL_NAMES:
    _candidate = globals().get(_name)
    if callable(_candidate):
        _HTTP_TOOL_REGISTRY[_name] = _candidate


def _invoke_tool_callable(func: Callable[..., Any], arguments: Dict[str, Any]) -> Any:
    sig = inspect.signature(func)
    bound_kwargs: Dict[str, Any] = {}
    provided_args = dict(arguments or {})

    for param_name, param in sig.parameters.items():
        if param_name == "ctx":
            bound_kwargs[param_name] = None
            continue

        if param_name == "args" and param.annotation is GenerateArgs:
            bound_kwargs[param_name] = GenerateArgs(**arguments)
            provided_args.clear()
            continue

        if param_name in provided_args:
            bound_kwargs[param_name] = provided_args.pop(param_name)
        elif param.default is not inspect._empty:
            bound_kwargs[param_name] = param.default
        else:
            raise ValueError(f"Missing argument '{param_name}'")

    return func(**bound_kwargs)


def _call_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    func = _HTTP_TOOL_REGISTRY.get(tool_name)
    if func is None:
        raise KeyError(tool_name)
    return _invoke_tool_callable(func, arguments)


def _format_tool_response(tool_name: str, result: Any) -> Dict[str, Any]:
    if isinstance(result, Image):
        image_data = getattr(result, "data", b"")
        if image_data is None:
            image_bytes = b""
        elif isinstance(image_data, memoryview):
            image_bytes = image_data.tobytes()
        elif isinstance(image_data, str):
            image_bytes = image_data.encode("utf-8")
        elif isinstance(image_data, (bytes, bytearray)):
            image_bytes = bytes(image_data)
        else:
            try:
                image_bytes = bytes(image_data)
            except Exception:
                image_bytes = str(image_data).encode("utf-8")
        payload = {
            "type": "image",
            "format": getattr(result, "format", "png"),
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }
        return {"tool_name": tool_name, "content": [payload]}

    if isinstance(result, str):
        return {"tool_name": tool_name, "content": [{"type": "text", "text": result}]}

    if isinstance(result, (dict, list, int, float, bool)):
        return {"tool_name": tool_name, "content": [{"type": "json", "data": result}]}

    if result is None:
        return {"tool_name": tool_name, "content": []}

    if isinstance(result, (bytes, bytearray)):
        payload = {
            "type": "binary",
            "data": base64.b64encode(result).decode("ascii"),
        }
        return {"tool_name": tool_name, "content": [payload]}

    return {
        "tool_name": tool_name,
        "content": [{"type": "text", "text": str(result)}],
    }

@app.post("/tools/call")
def tools_call(body: ToolCall, _=Depends(_auth)):
    try:
        result = _call_tool(body.tool_name, body.arguments or {})
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {body.tool_name}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(_format_tool_response(body.tool_name, result))

@app.get("/health")
def health():
    return {"ok": True}

# ---------------------------------------
# Entrypoint
# ---------------------------------------
def main() -> None:
    from fastmcp.server.sse import create_sse_handler
    
    # Mount MCP SSE endpoint onto the existing FastAPI app
    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        handler = create_sse_handler(mcp)
        return await handler(request)
    
    # Run unified server on port 8080 (FastMCP Cloud standard)
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()