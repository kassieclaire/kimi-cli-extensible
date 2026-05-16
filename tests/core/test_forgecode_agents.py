"""Live verification that forgecode plugin agents load correctly."""

from __future__ import annotations

from pathlib import Path

import pytest

from kimi_cli.soul.agent import load_agent

FORGECODE_AGENTS_DIR = Path("/home/kassie/projects/kimi-code-firepass-setup/forgecode-tools/agents")


@pytest.mark.parametrize(
    "agent_file,expected_name,expected_subagents",
    [
        ("forgecode/agent.yaml", "Forgecode", {"forge", "sage"}),
        ("forge/agent.yaml", "Forge", {"sage"}),
        ("muse/agent.yaml", "Muse", {"sage"}),
        ("sage/agent.yaml", "Sage", set()),
    ],
)
async def test_load_forgecode_agent(
    runtime,
    agent_file: str,
    expected_name: str,
    expected_subagents: set[str],
):
    path = FORGECODE_AGENTS_DIR / agent_file
    agent = await load_agent(path, runtime, mcp_configs=[])

    assert agent.name == expected_name

    # Subagents are registered in the labor market, not on the Agent object.
    registered = set(agent.runtime.labor_market.builtin_types.keys())
    for sub in expected_subagents:
        # Subagent may be registered as bare name or plugin-namespaced
        assert sub in registered or f"forgecode-tools:{sub}" in registered

    tool_names = [tool.name for tool in agent.toolset.tools]

    # All agents should have core read/nav tools
    assert "ReadFile" in tool_names
    assert "Glob" in tool_names

    # Only non-leaf agents that delegate get the Agent tool
    if expected_subagents:
        assert "Agent" in tool_names

    # Forge and Forgecode should NOT have Grep/StrReplaceFile (plugin replaces them)
    if expected_name in ("Forgecode", "Forge"):
        assert "Grep" not in tool_names
        assert "StrReplaceFile" not in tool_names
        # But they should have EnterPlanMode and ExitPlanMode
        assert "EnterPlanMode" in tool_names
        assert "ExitPlanMode" in tool_names

    # Muse should NOT have WriteFile/StrReplaceFile (read-only planning agent)
    if expected_name == "Muse":
        assert "WriteFile" not in tool_names
        assert "StrReplaceFile" not in tool_names
        # Muse needs ExitPlanMode to hand back to Forge
        assert "ExitPlanMode" in tool_names
        assert "EnterPlanMode" in tool_names


async def test_forgecode_plugin_tools_available(runtime):
    """Plugin tools (multi_replace, search_files, plan_create, etc.) should be
    available on agents that don't have an allowed_tools whitelist."""
    path = FORGECODE_AGENTS_DIR / "muse/agent.yaml"
    agent = await load_agent(path, runtime, mcp_configs=[])

    tool_names = {tool.name for tool in agent.toolset.tools}

    # These are forgecode plugin tools — they must be present because Muse
    # uses exclude_tools, not allowed_tools.
    assert "multi_replace" in tool_names
    assert "search_files" in tool_names
    assert "plan_create" in tool_names
    assert "todo_write" in tool_names
    assert "todo_read" in tool_names


class TestPluginPlanModeAgentSwitch:
    async def test_toggle_plan_mode_switches_to_muse_and_back(
        self,
        runtime,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When runtime.plugin_plan_mode_agent is set, toggle_plan_mode()
        should reload the active agent to the specified plan agent and
        restore the original on exit.
        """
        from kosong.tooling.empty import EmptyToolset
        from kimi_cli.soul.agent import Agent
        from kimi_cli.soul.context import Context
        from kimi_cli.soul.kimisoul import KimiSoul
        from kimi_cli.subagents import AgentTypeDefinition, ToolPolicy

        # 1. Register "muse" in the labor market so it can be found
        muse_agent_file = FORGECODE_AGENTS_DIR / "muse/agent.yaml"
        runtime.labor_market.add_builtin_type(
            AgentTypeDefinition(
                name="muse",
                description="Planning agent",
                agent_file=muse_agent_file,
                tool_policy=ToolPolicy(mode="inherit"),
            )
        )

        # 2. Set the plugin plan mode override
        runtime.plugin_plan_mode_agent = "muse"
        monkeypatch.setattr(
            "kimi_cli.tools.plan.heroes.PLANS_DIR", tmp_path
        )

        # 3. Create a soul with Forge as the original agent
        forge_agent_file = FORGECODE_AGENTS_DIR / "forge/agent.yaml"
        forge_agent = await load_agent(forge_agent_file, runtime, mcp_configs=[])
        soul = KimiSoul(
            forge_agent,
            context=Context(file_backend=tmp_path / "history.jsonl"),
            original_agent_file=forge_agent_file,
        )
        assert soul.name == "Forge"

        # 4. Toggle plan mode ON → should switch to Muse
        await soul.toggle_plan_mode()
        assert soul.name == "Muse"
        assert soul._plugin_plan_mode_active is True

        # 5. Toggle plan mode OFF → should restore Forge
        await soul.toggle_plan_mode()
        assert soul.name == "Forge"
        assert soul._plugin_plan_mode_active is False


class TestPluginToolDisplayBlocks:
    async def test_todo_write_returns_todo_display_block(
        self,
        runtime,
        tmp_path: Path,
    ):
        """When todo_write returns JSON with a display block, the PluginTool
        wrapper should parse it into a TodoDisplayBlock.
        """
        from kimi_cli.plugin.tool import PluginTool, PluginToolSpec

        tool = PluginTool(
            tool_spec=PluginToolSpec(
                name="todo_write",
                description="Write todos",
                command=[
                    "python",
                    str(
                        FORGECODE_AGENTS_DIR.parent
                        / "scripts/todo_write.py"
                    ),
                ],
            ),
            plugin_dir=FORGECODE_AGENTS_DIR.parent,
            inject={},
            config=runtime.config,
        )

        result = await tool(todos=[{"content": "Display block test task", "status": "pending"}])
        assert result.is_error is False
        assert len(result.display) == 1
        assert result.display[0].type == "todo"
        # The todo list is persistent; just verify the newly-added item exists
        titles = [item.title for item in result.display[0].items]
        assert "Display block test task" in titles

    async def test_todo_read_returns_todo_display_block(
        self,
        runtime,
        tmp_path: Path,
    ):
        """When todo_read returns JSON with a display block, the PluginTool
        wrapper should parse it correctly.
        """
        from kimi_cli.plugin.tool import PluginTool, PluginToolSpec

        tool = PluginTool(
            tool_spec=PluginToolSpec(
                name="todo_read",
                description="Read todos",
                command=[
                    "python",
                    str(
                        FORGECODE_AGENTS_DIR.parent
                        / "scripts/todo_read.py"
                    ),
                ],
            ),
            plugin_dir=FORGECODE_AGENTS_DIR.parent,
            inject={},
            config=runtime.config,
        )

        result = await tool()
        assert result.is_error is False
        assert len(result.display) == 1
        assert result.display[0].type == "todo"
