"""Plugin tool wrapper — runs plugin-declared tools as subprocesses."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kosong.tooling import CallableTool, ToolError, ToolOk, ToolReturnValue
from kosong.tooling.error import ToolRuntimeError
from loguru import logger

from kimi_cli.plugin import PluginToolSpec
from kimi_cli.tools.display import (
    BackgroundTaskDisplayBlock,
    DiffDisplayBlock,
    ShellDisplayBlock,
    TodoDisplayBlock,
)
from kimi_cli.tools.utils import ToolRejectedError
from kimi_cli.utils.subprocess_env import get_clean_env
from kimi_cli.wire.types import ToolReturnValue as WireToolReturnValue

if TYPE_CHECKING:
    from kimi_cli.config import Config
    from kimi_cli.soul.approval import Approval


# Display block types that plugins are allowed to return.
# Unknown types are silently dropped for security.
_ALLOWED_PLUGIN_DISPLAY_BLOCKS: dict[str, type] = {
    "todo": TodoDisplayBlock,
    "diff": DiffDisplayBlock,
    "shell": ShellDisplayBlock,
    "background_task": BackgroundTaskDisplayBlock,
}


def _get_host_values(config: Config) -> dict[str, str]:
    """Extract current host values (api_key, base_url) from config.

    Reads the latest provider credentials, which may have been
    refreshed by OAuth since plugin install time.
    """
    from kimi_cli.auth.oauth import OAuthManager
    from kimi_cli.plugin.manager import collect_host_values

    oauth = OAuthManager(config)
    return collect_host_values(config, oauth)


class PluginTool(CallableTool):
    """A tool that executes a plugin command in a subprocess.

    Parameters are passed via stdin as JSON.
    stdout is captured as the tool result.
    Host credentials are injected as environment variables at runtime
    (not baked into config files) to handle OAuth token refresh.

    Plugin tools may return either:
    - Plain text (legacy): wrapped in ``ToolOk``.
    - JSON (extended): parsed into a full ``ToolReturnValue`` with optional
      ``display`` blocks (e.g., ``TodoDisplayBlock``). The JSON schema is::

        {
          "output": "string or list[ContentPart]",
          "message": "string (optional, defaults to output)",
          "display": [
            {"type": "todo", "items": [...]},
            {"type": "diff", "path": "...", ...}
          ]
        }
    """

    def __init__(
        self,
        tool_spec: PluginToolSpec,
        plugin_dir: Path,
        *,
        inject: dict[str, str],
        config: Config,
        approval: Approval | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=tool_spec.name,
            description=tool_spec.description,
            parameters=tool_spec.parameters or {"type": "object", "properties": {}},
            **kwargs,
        )
        self._command = tool_spec.command
        self._plugin_dir = plugin_dir
        self._inject = inject  # e.g. {"kimiCodeAPIKey": "api_key"}
        self._config = config
        self._approval = approval

    def _build_env(self) -> dict[str, str]:
        """Build env vars with fresh host credentials for the subprocess."""
        env = get_clean_env()
        if self._inject:
            host_values = _get_host_values(self._config)
            for target_key, source_key in self._inject.items():
                if source_key in host_values:
                    # Inject as env var using the plugin's config key name
                    # e.g. kimiCodeAPIKey=<fresh api_key>
                    env[target_key] = host_values[source_key]
        return env

    async def __call__(self, *args: Any, **kwargs: Any) -> ToolReturnValue:
        if self._approval is not None:
            description = f"Run plugin tool `{self.name}`."
            if not await self._approval.request(self.name, f"plugin:{self.name}", description):
                return ToolRejectedError()

        params_json = json.dumps(kwargs, ensure_ascii=False)

        try:
            proc = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._plugin_dir),
                env=self._build_env(),
            )
        except Exception as exc:
            return ToolRuntimeError(str(exc))

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=params_json.encode("utf-8")),
                timeout=120,
            )
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolError(
                message=f"Plugin tool '{self.name}' timed out after 120s.",
                brief="Timeout",
            )

        output = stdout.decode("utf-8", errors="replace").strip()
        err_output = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            error_msg = err_output or output or f"Exit code {proc.returncode}"
            return ToolError(
                message=f"Plugin tool '{self.name}' failed: {error_msg}",
                brief=f"Exit {proc.returncode}",
            )

        if err_output:
            logger.debug("Plugin tool {name} stderr: {err}", name=self.name, err=err_output)

        # Try to parse as extended JSON response (with display blocks)
        parsed = self._try_parse_json_response(output)
        if parsed is not None:
            return parsed

        # Fallback: plain text legacy response
        return ToolOk(output=output)

    def _try_parse_json_response(self, text: str) -> ToolReturnValue | None:
        """Attempt to parse plugin stdout as a structured JSON response.

        Returns ``None`` if the text is not valid JSON or does not contain
        the required ``output`` field, signalling the caller to fall back to
        plain text wrapping.
        """
        if not text.startswith("{"):
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict) or "output" not in data:
            return None

        output = data["output"]
        message = data.get("message", "")
        if not message:
            message = output if isinstance(output, str) else ""

        display_blocks: list[Any] = []
        raw_display = data.get("display")
        if isinstance(raw_display, list):
            for block_data in raw_display:
                if not isinstance(block_data, dict):
                    continue
                block_type = block_data.get("type")
                block_cls = _ALLOWED_PLUGIN_DISPLAY_BLOCKS.get(block_type)
                if block_cls is None:
                    logger.warning(
                        "Plugin tool '{name}' returned unknown display block type '{type}', skipping",
                        name=self.name,
                        type=block_type,
                    )
                    continue
                try:
                    display_blocks.append(block_cls(**block_data))
                except Exception:
                    logger.warning(
                        "Plugin tool '{name}' returned invalid {type} display block, skipping",
                        name=self.name,
                        type=block_type,
                    )

        return WireToolReturnValue(
            is_error=False,
            output=output,
            message=message,
            display=display_blocks,
        )


def load_plugin_tools(
    plugins_dir: Path, config: Config, *, approval: Approval | None = None
) -> list[PluginTool]:
    """Scan installed plugins and create PluginTool instances for declared tools."""
    from kimi_cli.plugin import PLUGIN_JSON, PluginError, parse_plugin_json

    if not plugins_dir.is_dir():
        return []

    tools: list[PluginTool] = []
    for child in sorted(plugins_dir.iterdir()):
        plugin_json = child / PLUGIN_JSON
        if not child.is_dir() or not plugin_json.is_file():
            continue
        try:
            spec = parse_plugin_json(plugin_json)
        except PluginError:
            continue
        for tool_spec in spec.tools:
            try:
                tool = PluginTool(
                    tool_spec,
                    plugin_dir=child,
                    inject=spec.inject,
                    config=config,
                    approval=approval,
                )
            except Exception:
                logger.warning(
                    "Skipping invalid plugin tool: {name} (from {plugin})",
                    name=tool_spec.name,
                    plugin=spec.name,
                )
                continue
            tools.append(tool)
            logger.info(
                "Loaded plugin tool: {name} (from {plugin})",
                name=tool_spec.name,
                plugin=spec.name,
            )
    return tools
