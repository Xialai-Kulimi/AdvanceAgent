"""BaseAgent 單元測試。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

# 載入測試環境設定
load_dotenv(Path(__file__).parent / ".env")

from advance_agent.base_agent import AgentState, BaseAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:3131/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
MODEL = os.environ.get("OPENAI_MODEL", "gemini-3-flash-preview")


def _make_agent(**kwargs) -> BaseAgent:
    """建立一個連接到本地端點的 agent。"""
    defaults = {"base_url": BASE_URL, "model": MODEL, "api_key": API_KEY}
    defaults.update(kwargs)
    return BaseAgent(**defaults)


# ---------------------------------------------------------------------------
# Unit Tests (no LLM needed)
# ---------------------------------------------------------------------------

class TestAgentState:
    """AgentState 序列化/反序列化。"""

    def test_to_dict_and_from_dict(self) -> None:
        state = AgentState(
            history=[{"role": "user", "content": "hi"}],
            system_instruction="Be helpful.",
            model="test-model",
            model_config={"temperature": 0.7},
        )
        d = state.to_dict()
        restored = AgentState.from_dict(d)
        assert restored.history == state.history
        assert restored.system_instruction == state.system_instruction
        assert restored.model == state.model
        assert restored.model_config == state.model_config

    def test_from_dict_defaults(self) -> None:
        state = AgentState.from_dict({})
        assert state.history == []
        assert state.system_instruction is None
        assert state.model is None
        assert state.model_config == {}


class TestBaseAgentConfig:
    """BaseAgent 設定相關方法。"""

    def test_system_instruction(self) -> None:
        agent = _make_agent()
        agent.set_system_instruction("Test instruction.")
        assert agent.state.system_instruction == "Test instruction."

    def test_model_config(self) -> None:
        agent = _make_agent()
        agent.set_model_config(temperature=0.5, max_tokens=100)
        assert agent.state.model_config["temperature"] == 0.5
        assert agent.state.model_config["max_tokens"] == 100

    def test_history_management(self) -> None:
        agent = _make_agent()
        assert len(agent.state.history) == 0

        agent.add_history({"role": "user", "content": "hello"})
        assert len(agent.state.history) == 1
        assert agent.state.history[0]["content"] == "hello"

        agent.add_history({"role": "assistant", "content": "hi there"})
        assert len(agent.state.history) == 2

        agent.clear_history()
        assert len(agent.state.history) == 0


class TestToolManagement:
    """工具註冊/移除。"""

    def test_register_and_unregister(self) -> None:
        agent = _make_agent()
        agent.register_tool(
            name="greet",
            handler=lambda name: f"Hello, {name}!",
            description="Greet someone.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name to greet."}
                },
                "required": ["name"],
            },
        )
        assert "greet" in agent._tools
        tools_param = agent._build_tools_param()
        assert tools_param is not None
        assert len(tools_param) == 1
        assert tools_param[0]["function"]["name"] == "greet"

        agent.unregister_tool("greet")
        assert "greet" not in agent._tools
        assert agent._build_tools_param() is None

    def test_build_tools_param_empty(self) -> None:
        agent = _make_agent()
        assert agent._build_tools_param() is None


class TestMessageBuilding:
    """_build_messages 組裝邏輯。"""

    def test_with_system_and_prompt(self) -> None:
        agent = _make_agent()
        agent.set_system_instruction("You are a bot.")
        msgs = agent._build_messages("Hello!")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a bot."
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "Hello!"
        # prompt 應被加入 history
        assert len(agent.state.history) == 1

    def test_without_system(self) -> None:
        agent = _make_agent()
        msgs = agent._build_messages("Hi")
        assert msgs[0]["role"] == "user"

    def test_no_prompt(self) -> None:
        agent = _make_agent()
        agent.add_history({"role": "user", "content": "prev"})
        msgs = agent._build_messages()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "prev"


class TestPersistence:
    """dump / load 測試。"""

    def test_dump_and_load(self) -> None:
        agent = _make_agent()
        agent.set_system_instruction("Persist me.")
        agent.add_history({"role": "user", "content": "test"})
        agent.set_model_config(temperature=0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "agent_state.json")
            agent.dump(path)
            assert os.path.exists(path)

            loaded = BaseAgent.load(
                path, base_url=BASE_URL, model=MODEL, api_key=API_KEY
            )
            assert loaded.state.system_instruction == "Persist me."
            assert len(loaded.state.history) == 1
            assert loaded.state.model_config["temperature"] == 0.3

    def test_load_nonexistent(self) -> None:
        agent = BaseAgent.load(
            "/tmp/nonexistent_agent.json",
            base_url=BASE_URL, model=MODEL, api_key=API_KEY,
        )
        assert agent.state.history == []


class TestExtractText:
    """extract_text 靜態方法。"""

    def test_extracts_assistant_content(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
            {"role": "assistant", "content": "how are you?"},
        ]
        result = BaseAgent.extract_text(messages)
        assert "hello!" in result
        assert "how are you?" in result

    def test_ignores_non_assistant(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "result"},
        ]
        result = BaseAgent.extract_text(messages)
        assert result == ""


# ---------------------------------------------------------------------------
# Integration Tests (requires localhost:3131)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("SKIP_LLM_TESTS", "").lower() in ("1", "true"),
    reason="LLM integration tests disabled",
)
class TestLLMIntegration:
    """需要 localhost:3131 運行中的整合測試。"""

    def test_simple_generate(self) -> None:
        agent = _make_agent()
        agent.set_system_instruction("You are a test assistant. Reply in one sentence.")

        responses = list(agent.generate("What is 1+1? Answer briefly."))
        assert len(responses) >= 1
        text = BaseAgent.extract_text(responses)
        assert len(text) > 0

    def test_step(self) -> None:
        agent = _make_agent()
        agent.set_system_instruction("Reply very briefly.")

        result = agent.step("Say hello.")
        assert result["role"] == "assistant"
        assert result.get("content")

    def test_generate_with_tool(self) -> None:
        agent = _make_agent()
        agent.set_system_instruction(
            "You are a test assistant. When asked to add numbers, "
            "use the 'add' tool. Reply the result."
        )
        agent.register_tool(
            name="add",
            handler=lambda a, b: str(int(a) + int(b)),
            description="Add two integers together.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number."},
                    "b": {"type": "integer", "description": "Second number."},
                },
                "required": ["a", "b"],
            },
        )

        responses = list(agent.generate("What is 37 + 58? Use the add tool."))
        text = BaseAgent.extract_text(responses)
        assert "95" in text

    def test_history_persists_across_turns(self) -> None:
        agent = _make_agent()
        agent.set_system_instruction("You are a test assistant. Keep responses very short.")

        agent.step("My name is TestUser42.")
        result = agent.step("What is my name?")
        assert "TestUser42" in result.get("content", "")
