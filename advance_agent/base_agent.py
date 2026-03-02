"""BaseAgent -- 以 OpenAI SDK 為核心的基礎 Agent。"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Generator, Optional

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Agent 的完整可序列化狀態。"""

    history: list[dict[str, Any]] = field(default_factory=list)
    system_instruction: Optional[str] = None
    model: Optional[str] = None
    model_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "system_instruction": self.system_instruction,
            "model": self.model,
            "model_config": self.model_config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        return cls(
            history=data.get("history", []),
            system_instruction=data.get("system_instruction"),
            model=data.get("model"),
            model_config=data.get("model_config", {}),
        )


# ---------------------------------------------------------------------------
# Tool Declaration
# ---------------------------------------------------------------------------

@dataclass
class ToolDeclaration:
    """Client-side tool 聲明，包含 Python handler。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent:
    """以 OpenAI SDK 為核心的 Agent 基礎類別。

    取代原 IntimusAgent 的 subprocess 架構，所有 LLM 互動透過
    OpenAI Chat Completion API 完成。

    Parameters
    ----------
    base_url:
        OpenAI 相容 API 的 base URL (例如 ``http://localhost:3131/v1``)。
    model:
        模型名稱 (例如 ``gemini-3-flash-preview``)。
    api_key:
        API key，若端點不需驗證可傳任意值。
    state:
        可選的初始 AgentState。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3131/v1",
        model: str = "gemini-3-flash-preview",
        api_key: str = "not-needed",
        state: Optional[AgentState] = None,
    ) -> None:
        self.state: AgentState = state or AgentState()
        self.state.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._tools: dict[str, ToolDeclaration] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_system_instruction(self, instruction: str) -> None:
        """設定 system instruction。"""
        self.state.system_instruction = instruction

    def set_model_config(self, **kwargs: Any) -> None:
        """設定模型生成參數 (temperature, top_p, max_tokens 等)。"""
        self.state.model_config.update(kwargs)

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def add_history(self, message: dict[str, Any]) -> None:
        """追加一則訊息到對話歷史。

        Parameters
        ----------
        message:
            OpenAI 格式的 message dict，例如
            ``{"role": "user", "content": "Hello"}``。
        """
        self.state.history.append(message)

    def clear_history(self) -> None:
        """清除所有對話歷史。"""
        self.state.history = []

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
        parameters: Optional[dict[str, Any]] = None,
    ) -> None:
        """註冊一個 client-side tool。

        Parameters
        ----------
        name:
            工具名稱，必須唯一。
        handler:
            Python callable，接收 tool arguments 作為 keyword args。
        description:
            工具的自然語言描述，供 LLM 理解何時該使用。
        parameters:
            JSON Schema 格式的工具參數定義。
        """
        self._tools[name] = ToolDeclaration(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            handler=handler,
        )

    def unregister_tool(self, name: str) -> None:
        """移除已註冊的 tool。"""
        self._tools.pop(name, None)

    def _build_tools_param(self) -> list[dict[str, Any]] | None:
        """將已註冊的 tools 轉換為 OpenAI API 的 tools 參數格式。"""
        if not self._tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(
        self, prompt: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """組裝完整的 messages 列表（system + history + 可選的 user prompt）。"""
        messages: list[dict[str, Any]] = []

        if self.state.system_instruction:
            messages.append({
                "role": "system",
                "content": self.state.system_instruction,
            })

        messages.extend(self.state.history)

        if prompt is not None:
            user_msg: dict[str, Any] = {"role": "user", "content": prompt}
            messages.append(user_msg)
            self.state.history.append(user_msg)

        return messages

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(
        self, messages: list[dict[str, Any]]
    ) -> Any:
        """呼叫 OpenAI Chat Completion API。"""
        kwargs: dict[str, Any] = {
            "model": self.state.model,
            "messages": messages,
        }

        tools = self._build_tools_param()
        if tools:
            kwargs["tools"] = tools

        # 合併使用者自訂的模型參數
        kwargs.update(self.state.model_config)

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message

    def _execute_tool_calls(
        self, tool_calls: list[Any]
    ) -> list[dict[str, Any]]:
        """執行 tool calls 並回傳 tool response messages。"""
        tool_messages: list[dict[str, Any]] = []

        for tc in tool_calls:
            fn_name = tc.function.name
            tool_call_id = tc.id

            # 解析 arguments
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}

            # 執行 handler
            handler = self._tools.get(fn_name)
            if handler is None:
                result = f"Error: Tool '{fn_name}' not registered"
            else:
                try:
                    result = str(handler.handler(**args))
                except Exception:
                    result = f"Error: {traceback.format_exc()}"

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            })

        return tool_messages

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def step(self, prompt: Optional[str] = None) -> dict[str, Any]:
        """執行單步推理（一次 API call + 至多一輪 tool call）。

        Returns
        -------
        dict
            Assistant 的回應 message (OpenAI 格式)。
        """
        messages = self._build_messages(prompt)
        assistant_msg = self._call_llm(messages)

        # 將 assistant 回應序列化並加入歷史
        msg_dict = self._message_to_dict(assistant_msg)
        self.state.history.append(msg_dict)

        # 若有 tool calls，執行一輪
        if assistant_msg.tool_calls:
            tool_responses = self._execute_tool_calls(assistant_msg.tool_calls)
            self.state.history.extend(tool_responses)

            # 帶 tool results 再推理一次
            messages = self._build_messages()
            assistant_msg_2 = self._call_llm(messages)
            msg_dict_2 = self._message_to_dict(assistant_msg_2)
            self.state.history.append(msg_dict_2)
            return msg_dict_2

        return msg_dict

    def generate(
        self, prompt: Optional[str] = None
    ) -> Generator[dict[str, Any], None, None]:
        """完整的推理迴圈，自動處理所有 tool calls 直到模型停止呼叫。

        Yields
        ------
        dict
            每一步的 assistant message (OpenAI 格式)。
        """
        messages = self._build_messages(prompt)

        while True:
            assistant_msg = self._call_llm(messages)
            msg_dict = self._message_to_dict(assistant_msg)
            self.state.history.append(msg_dict)
            yield msg_dict

            # 沒有 tool calls -> 結束
            if not assistant_msg.tool_calls:
                break

            # 執行 tool calls
            tool_responses = self._execute_tool_calls(assistant_msg.tool_calls)
            self.state.history.extend(tool_responses)

            # 重建 messages 供下次 API call
            messages = self._build_messages()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def dump(self, path: str) -> None:
        """將 Agent 狀態序列化為 JSON 寫入檔案。"""
        data = self.state.to_dict()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        base_url: str = "http://localhost:3131/v1",
        model: str = "gemini-3-flash-preview",
        api_key: str = "not-needed",
    ) -> BaseAgent:
        """從 JSON 檔案載入 Agent 狀態。

        若檔案不存在，回傳一個全新的 Agent。
        """
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
            state = AgentState.from_dict(data)
            return cls(base_url=base_url, model=model, api_key=api_key, state=state)
        return cls(base_url=base_url, model=model, api_key=api_key)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _message_to_dict(msg: Any) -> dict[str, Any]:
        """將 OpenAI ChatCompletionMessage 物件轉換為可序列化的 dict。"""
        d: dict[str, Any] = {
            "role": msg.role,
        }
        if msg.content is not None:
            d["content"] = msg.content
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return d

    @staticmethod
    def has_tool_calls(msg: dict[str, Any]) -> bool:
        """檢查一則 assistant message 是否包含 tool calls。"""
        return bool(msg.get("tool_calls"))

    @staticmethod
    def extract_text(messages: list[dict[str, Any]]) -> str:
        """從 messages 列表中提取所有 assistant 的純文字回覆。"""
        texts: list[str] = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                texts.append(msg["content"])
        return "\n".join(texts)
