from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ToolsConfig


class _DummyTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} test tool"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **_kwargs):
        return "ok"


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(max_tokens=1024)
    return provider


def _channels_config(**entries):
    cfg = MagicMock()
    cfg.model_extra = entries
    return cfg


def test_restricted_prompt_omits_privileged_context(tmp_path):
    (tmp_path / "AGENTS.md").write_text("private bootstrap rule", encoding="utf-8")
    builder = ContextBuilder(tmp_path)

    prompt = builder.build_system_prompt(is_privileged=False)

    assert "Restricted Mode" in prompt
    assert "private bootstrap rule" not in prompt
    assert str(tmp_path) not in prompt
    assert "Long-term memory" not in prompt
    assert "read its SKILL.md" not in prompt


def test_privilege_uses_strict_allow_from_policy(tmp_path):
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_provider(),
        workspace=tmp_path,
        channels_config=_channels_config(telegram={"allowFrom": ["123"]}),
    )

    assert loop._is_privileged("telegram", "123") is True
    assert loop._is_privileged("telegram", "123|Alice") is True
    assert loop._is_privileged("telegram", "456") is False
    assert loop._is_privileged("cli", "anyone") is True


@pytest.mark.asyncio
async def test_unprivileged_user_gets_filtered_tool_registry(tmp_path):
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_provider(),
        workspace=tmp_path,
        channels_config=_channels_config(telegram={"allowFrom": ["admin"]}),
        tools_config=ToolsConfig(admin_tools=["read_file", "exec"]),
    )
    registry = ToolRegistry()
    registry.register(_DummyTool("read_file"))
    registry.register(_DummyTool("exec"))
    registry.register(_DummyTool("message"))
    loop.tools = registry
    loop.runner.run = AsyncMock(return_value=AgentRunResult(final_content="ok", messages=[]))

    await loop._run_agent_loop([], channel="telegram", sender_id="guest")

    spec = loop.runner.run.await_args.args[0]
    assert "read_file" not in spec.tools.tool_names
    assert "exec" not in spec.tools.tool_names
    assert "message" in spec.tools.tool_names


@pytest.mark.asyncio
async def test_spawn_passes_privilege_to_subagent_manager() -> None:
    manager = MagicMock()
    manager.get_running_count.return_value = 0
    manager.max_concurrent_subagents = 1
    manager.spawn = AsyncMock(return_value="started")

    tool = SpawnTool(manager)
    tool.set_context(
        RequestContext(
            channel="telegram",
            chat_id="c1",
            session_key="telegram:c1",
            is_privileged=False,
        )
    )

    await tool.execute(task="background task")

    assert manager.spawn.await_args.kwargs["is_privileged"] is False


def test_unprivileged_subagent_filters_admin_tools(tmp_path):
    manager = SubagentManager(
        provider=_provider(),
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=1024,
        tools_config=ToolsConfig(admin_tools=["read_file", "exec"]),
    )

    tools = manager._build_tools(is_privileged=False)

    assert "read_file" not in tools.tool_names
    assert "exec" not in tools.tool_names
