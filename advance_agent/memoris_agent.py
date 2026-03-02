"""MemorisAgent -- 具備仿生記憶能力的進階 Agent。"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Generator, Optional, Union

from .base_agent import AgentState, BaseAgent
from .memory_db import MemoryDB
from .models import (
    Fact,
    FactType,
    Goal,
    Memory,
)

# -- 工具 Parameter Schemas (OpenAI function calling 格式) --------------------

_SET_GOAL_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal_description": {
            "type": "string",
            "description": "The current short-term goal to focus on.",
        },
    },
    "required": ["goal_description"],
}

_SAVE_FACT_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fact": {
            "type": "string",
            "description": "The fact content to store.",
        },
        "reason": {
            "type": "string",
            "description": "The reasoning or evidence behind this fact.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score between 0.0 and 1.0.",
        },
        "type": {
            "type": "string",
            "enum": ["KNOWLEDGE", "PROCEDURE"],
            "description": "KNOWLEDGE for declarative facts, PROCEDURE for how-to instructions.",
        },
    },
    "required": ["fact", "reason", "confidence"],
}

_SAVE_MEMORY_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "context": {
            "type": "string",
            "description": (
                "The conversational background or summary of the current context. "
                "Include temporal context (e.g., precise time, morning, specific date) "
                "to aid chronological retrieval."
            ),
        },
        "reflection": {
            "type": "string",
            "description": "Your reflection or analysis of the current interaction.",
        },
        "observation": {
            "type": "string",
            "description": "The raw observation or factual record of the event.",
        },
    },
    "required": ["context", "reflection", "observation"],
}

_QUERY_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The question to search related memories and facts for.",
        },
    },
    "required": ["question"],
}

_DISPUTE_FACT_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fact_id": {
            "type": "string",
            "description": "The ID of the fact to mark as disputed.",
        },
    },
    "required": ["fact_id"],
}


# -- 壓縮/固化閾值 ----------------------------------------------------------

COMPRESS_THRESHOLD = 20   # 同一 level 未壓縮記憶達此數量時觸發壓縮
COMPRESS_BATCH_SIZE = 10  # 每次壓縮最舊的幾個
SOLIDIFY_THRESHOLD = 30   # 未固化記憶達此數量時觸發固化
AUTO_RECALL_TOP_K = 7     # Auto-Recall 取回的項目數
PERSONALITY_MAX_FACTS = 70      # Personality Context 最多注入的 Fact 數
RECALL_COOLDOWN_TURNS = 14      # Auto-Recall debounce 滑動窗口大小
FLASHBACK_TEMPLATE = "<flashback>\n{content}\n</flashback>"


class MemorisAgent(BaseAgent):
    """具備仿生記憶能力的進階 Agent。

    在 BaseAgent 的 OpenAI SDK 核心之上，加入：
    - Auto-Recall (Flashback)：每次推理前自動檢索相關記憶
    - Personality Context：將核心記憶注入 System Instruction
    - Compress：壓縮層級式的記憶濃縮
    - Solidify：從情節記憶中提取穩定事實

    Parameters
    ----------
    agent_id:
        此 Agent 的唯一識別。多個 MemorisAgent 可共用同一個 ``db``，
        各自以 ``agent_id`` 區分資料。
    db:
        記憶資料庫實例（符合 ``MemoryDB`` Protocol）。
    base_url:
        OpenAI 相容 API 的 base URL。
    model:
        模型名稱。
    api_key:
        API key。
    state:
        可選的 AgentState 初始狀態。
    """

    def __init__(
        self,
        agent_id: str,
        db: MemoryDB,
        base_url: str = "http://localhost:3131/v1",
        model: str = "gemini-3-flash-preview",
        api_key: str = "not-needed",
        state: Optional[AgentState] = None,
    ) -> None:
        self.agent_id = agent_id
        self.db = db
        self._base_system_instruction: Optional[str] = None
        self._last_recall_input: Optional[str] = None
        self._recent_recall_window: deque[set[str]] = deque(
            maxlen=RECALL_COOLDOWN_TURNS
        )

        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            state=state,
        )
        self._register_memory_tools()

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_memory_tools(self) -> None:
        """註冊 MemorisAgent 的記憶管理工具。"""
        self.register_tool(
            name="set_goal",
            handler=self._tool_set_goal,
            description="Set or update the current short-term goal that guides your behavior.",
            parameters=_SET_GOAL_PARAMS,
        )
        self.register_tool(
            name="save_fact",
            handler=self._tool_save_fact,
            description=(
                "Save a fact you have internalized. Use KNOWLEDGE for declarative facts "
                "(e.g. 'user prefers dark mode'), PROCEDURE for how-to instructions "
                "(e.g. 'always run tests before deployment')."
            ),
            parameters=_SAVE_FACT_PARAMS,
        )
        self.register_tool(
            name="save_memory",
            handler=self._tool_save_memory,
            description=(
                "Record an episodic memory of the current interaction. "
                "Use this to preserve important moments, making sure to include "
                "temporal information from the system info in the 'context'."
            ),
            parameters=_SAVE_MEMORY_PARAMS,
        )
        self.register_tool(
            name="query",
            handler=self._tool_query,
            description="Search your memory for related facts and past memories.",
            parameters=_QUERY_PARAMS,
        )
        self.register_tool(
            name="dispute_fact",
            handler=self._tool_dispute_fact,
            description=(
                "Mark a fact as disputed if you believe it may be inaccurate. "
                "This halves its confidence and flags it for review."
            ),
            parameters=_DISPUTE_FACT_PARAMS,
        )

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _tool_set_goal(self, goal_description: str) -> str:
        goal = Goal(content=goal_description, agent_id=self.agent_id)
        self.db.save_goal(goal)
        return f"Goal updated: {goal_description}"

    def _tool_save_fact(
        self,
        fact: str,
        reason: str,
        confidence: float,
        type: str = "KNOWLEDGE",
    ) -> str:
        fact_type = FactType(type) if type in ("KNOWLEDGE", "PROCEDURE") else FactType.KNOWLEDGE
        new_fact = Fact(
            fact=fact,
            reason=reason,
            confidence=max(0.0, min(1.0, confidence)),
            agent_id=self.agent_id,
            type=fact_type,
        )
        self.db.save_fact(new_fact)
        return f"Fact saved (id={new_fact.id}): {fact}"

    def _tool_save_memory(
        self,
        context: str,
        reflection: str,
        observation: str,
    ) -> str:
        memory = Memory(
            context=context,
            reflection=reflection,
            observation=observation,
            agent_id=self.agent_id,
        )
        self.db.save_memory(memory)

        # 檢查壓縮閾值
        status = self._check_compress_threshold()
        result = "Memory saved."
        if status:
            result += f" {status}"
        return result

    def _tool_query(self, question: str) -> str:
        results = self.db.query(question, self.agent_id, top_k=AUTO_RECALL_TOP_K)
        if not results:
            return "No related memories or facts found."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(f"--- Result {i} (score={r.score:.3f}) ---")
            lines.append(r.display_text())
        return "\n".join(lines)

    def _tool_dispute_fact(self, fact_id: str) -> str:
        fact = self.db.get_fact_by_id(fact_id, self.agent_id)
        if fact is None:
            return f"Fact not found: {fact_id}"
        fact.disputed = True
        fact.confidence /= 2.0
        self.db.update_fact(fact)
        return f"Fact marked as disputed (confidence halved to {fact.confidence:.2f}): {fact.fact}"

    # ------------------------------------------------------------------
    # Auto-Recall
    # ------------------------------------------------------------------

    def _update_personality_context(self) -> set[str]:
        """更新 System Instruction 中的 Personality Context，並回傳已注入的 ID。"""
        parts: list[str] = []
        if self._base_system_instruction:
            parts.append(self._base_system_instruction)

        recall_ids: set[str] = set()
        sections: list[str] = []

        # -- Personality Context --------------------------------------------
        top_facts = self.db.get_top_facts(
            self.agent_id, PERSONALITY_MAX_FACTS
        )
        if top_facts:
            recall_ids |= {f.id for f in top_facts}
            fact_lines = [f.display_text() for f in top_facts]
            sections.append(
                "[Personality Context: Core Facts]\n"
                + "\n---\n".join(fact_lines)
            )

        active_memories = self.db.get_all_active_memories(self.agent_id)
        if active_memories:
            recall_ids |= {m.id for m in active_memories}
            mem_lines = [m.display_text() for m in active_memories]
            sections.append(
                "[Personality Context: Active Memories]\n"
                + "\n---\n".join(mem_lines)
            )

        if sections:
            parts.append("\n\n".join(sections))

        # -- Goal -----------------------------------------------------------
        goal = self.db.get_latest_goal(self.agent_id)
        if goal:
            parts.append(f"[Current Goal]\n{goal.content}")

        # 更新 System Instruction
        combined = "\n\n".join(parts)
        super().set_system_instruction(combined)

        return recall_ids

    def _perform_flashback(
        self, prompt: Optional[str], exclude_ids: set[str]
    ) -> Optional[dict[str, Any]]:
        """執行 Auto-Recall 並注入 Flashback User Message。"""
        query_text = prompt or self._last_recall_input
        if not query_text:
            return None

        self._last_recall_input = query_text

        # -- Auto-Recall with debounce --------------------------------------
        debounce_ids = set().union(
            *self._recent_recall_window
        ) if self._recent_recall_window else set()

        # 排除 Personality Context 已有 + 最近 window 內的
        final_exclude = exclude_ids | debounce_ids

        results = self.db.query(
            query_text, self.agent_id, top_k=AUTO_RECALL_TOP_K
        )
        filtered = [r for r in results if r.item.id not in final_exclude]

        # 更新冷卻窗口
        self._recent_recall_window.append(
            {r.item.id for r in filtered}
        )

        if filtered:
            recall_lines = [r.display_text() for r in filtered]
            flashback_content = (
                "[Auto-Recall: Related Memories & Facts]\n"
                + "\n---\n".join(recall_lines)
            )
            flashback_msg_text = FLASHBACK_TEMPLATE.format(content=flashback_content)

            # 注入 User Message
            msg: dict[str, Any] = {"role": "user", "content": flashback_msg_text}
            self.add_history(msg)
            return msg

        return None

    def _inject_recall(self, prompt: Optional[str]) -> Optional[dict[str, Any]]:
        """注入記憶：Personality Context (System) + Flashback (User Message)。"""
        # 1. Update System Instruction (Personality + Goal)
        personality_ids = self._update_personality_context()

        # 2. Inject Flashback (Auto-Recall)
        return self._perform_flashback(prompt, personality_ids)

    # ------------------------------------------------------------------
    # Override: System instruction management
    # ------------------------------------------------------------------

    def set_system_instruction(self, instruction: str) -> None:
        """設定基礎的 system instruction。

        Auto-Recall 的結果會在每次推理前附加到此指令之後。
        """
        self._base_system_instruction = instruction
        super().set_system_instruction(instruction)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def before_generation(self, prompt: Optional[str]) -> None:
        """在 generate() 開始前呼叫的鉤子。預設實作動態時間注入。"""
        import datetime
        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S (%A)")
        msg: dict[str, Any] = {
            "role": "user",
            "content": f"[System Information: Current local time is {time_str}]",
        }
        self.add_history(msg)

    def after_generation(self) -> None:
        """在 generate() 完成（包含所有 tool call）後呼叫的鉤子。"""
        pass

    # ------------------------------------------------------------------
    # Override: Generation methods
    # ------------------------------------------------------------------

    def generate(
        self, prompt: Optional[str] = None
    ) -> Generator[dict[str, Any], None, None]:
        """完整的推理迴圈，自動注入 Auto-Recall 記憶。"""
        self.before_generation(prompt)
        try:
            # 1. 注入記憶
            flashback_msg = self._inject_recall(prompt)
            if flashback_msg:
                yield flashback_msg

            # 2. 委派給 BaseAgent.generate() 處理完整的 tool call loop
            yield from super().generate(prompt)
        finally:
            self.after_generation()

    def step(self, prompt: Optional[str] = None) -> dict[str, Any]:
        """執行一步推理，自動注入 Auto-Recall 記憶。"""
        flashback_msg = self._inject_recall(prompt)
        return super().step(prompt)

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------

    def _check_compress_threshold(self) -> Optional[str]:
        """檢查是否有任何 level 的未壓縮記憶達到閾值，若有則回傳提示。"""
        for level in range(10):
            uncompressed = self.db.get_uncompressed_memories(self.agent_id, level)
            if len(uncompressed) >= COMPRESS_THRESHOLD:
                return (
                    f"[Notice] {len(uncompressed)} uncompressed memories at level {level}. "
                    f"Consider running compress()."
                )
        return None

    def compress(self) -> str:
        """壓縮記憶：將同一 level 中最舊的記憶濃縮為更高層級的摘要。

        遍歷所有 level，若未壓縮記憶 >= COMPRESS_THRESHOLD，
        則取最舊的 COMPRESS_BATCH_SIZE 個交給 LLM 濃縮。
        """
        compressed_count = 0

        for level in range(10):
            uncompressed = self.db.get_uncompressed_memories(self.agent_id, level)
            if len(uncompressed) < COMPRESS_THRESHOLD:
                continue

            # 按建立時間排序，取最舊的
            uncompressed.sort(key=lambda m: m.created_at)
            batch = uncompressed[:COMPRESS_BATCH_SIZE]

            # 組裝 prompt 讓 LLM 濃縮
            memory_texts: list[str] = []
            for i, mem in enumerate(batch, 1):
                memory_texts.append(
                    f"Memory {i}:\n"
                    f"  Context: {mem.context}\n"
                    f"  Reflection: {mem.reflection}\n"
                    f"  Observation: {mem.observation}"
                )

            compress_prompt = (
                "You are performing memory compression. Below are episodic memories "
                "that need to be condensed into a single, higher-level summary memory.\n\n"
                "Preserve the key events, insights, and any important details, but "
                "merge redundant information and abstract away minor specifics.\n\n"
                "Respond with EXACTLY three lines in the following format:\n"
                "CONTEXT: <merged context summary>\n"
                "REFLECTION: <merged reflection and analysis>\n"
                "OBSERVATION: <merged key observations>\n\n"
                "Memories to compress:\n\n"
                + "\n\n".join(memory_texts)
            )

            try:
                # 使用 BaseAgent.generate() 進行壓縮推理
                responses = list(super().generate(compress_prompt))
                response_text = self.extract_text(responses)

                context, reflection, observation = self._parse_compress_response(
                    response_text, batch
                )

                # 儲存新的壓縮記憶
                new_memory = Memory(
                    context=context,
                    reflection=reflection,
                    observation=observation,
                    agent_id=self.agent_id,
                    level=level + 1,
                )
                self.db.save_memory(new_memory)

                # 保護判定：access_count > 平均 * 1.2 的記憶標記為 protected
                avg_access = sum(m.access_count for m in batch) / len(batch)
                protect_ids = [
                    m.id for m in batch
                    if m.access_count > avg_access * 1.2
                ]
                if protect_ids:
                    self.db.mark_protected(protect_ids, self.agent_id)

                # 標記原始記憶為已壓縮
                self.db.mark_compressed([m.id for m in batch], self.agent_id)
                compressed_count += len(batch)

            except Exception:
                # 壓縮失敗不應該中斷整個流程
                continue

        return f"Compressed {compressed_count} memories."

    # ------------------------------------------------------------------
    # Solidify
    # ------------------------------------------------------------------

    def solidify(self) -> str:
        """固化記憶：掃描未固化的記憶，提取值得內化的 Fact。"""
        unsolidified = self.db.get_unsolidified_memories(self.agent_id)
        if not unsolidified:
            return "No memories to solidify."

        memory_texts: list[str] = []
        for i, mem in enumerate(unsolidified, 1):
            memory_texts.append(
                f"Memory {i} (id={mem.id}):\n"
                f"  Context: {mem.context}\n"
                f"  Reflection: {mem.reflection}\n"
                f"  Observation: {mem.observation}"
            )

        solidify_prompt = (
            "You are performing memory solidification -- extracting lasting facts "
            "from episodic memories.\n\n"
            "Review the following memories and extract any stable, reusable facts "
            "(things that are likely to remain true and useful in the future).\n\n"
            "For each fact, respond with one line in the following format:\n"
            "FACT: <fact content> | REASON: <why this is worth remembering> | "
            "CONFIDENCE: <0.0-1.0> | TYPE: <KNOWLEDGE or PROCEDURE>\n\n"
            "If no facts worth extracting, respond with: NO_FACTS\n\n"
            "Memories to review:\n\n"
            + "\n\n".join(memory_texts)
        )

        try:
            responses = list(super().generate(solidify_prompt))
            response_text = self.extract_text(responses)

            facts = self._parse_solidify_response(response_text)
            for f in facts:
                self.db.save_fact(f)

            # 標記所有記憶為已固化
            self.db.mark_solidified([m.id for m in unsolidified], self.agent_id)

            return f"Solidified {len(unsolidified)} memories, extracted {len(facts)} facts."

        except Exception as e:
            return f"Solidification failed: {e}"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def dump(self, path: str, db_path: Optional[str] = None) -> None:
        """儲存 Agent 狀態與記憶資料庫。

        Parameters
        ----------
        path:
            Agent 狀態的 JSON 檔案路徑。
        db_path:
            記憶資料庫的 JSON 檔案路徑。若未指定，會在 ``path``
            旁邊建立一個 ``<basename>_memory.json``。
        """
        super().dump(path)

        if db_path is None:
            if path.endswith(".json"):
                db_path = path[:-5] + "_memory.json"
            else:
                db_path = path + "_memory.json"

        if hasattr(self.db, "dump"):
            self.db.dump(db_path)  # type: ignore[union-attr]

    @classmethod
    def load(  # type: ignore[override]
        cls,
        path: str,
        agent_id: str,
        db: MemoryDB,
        db_path: Optional[str] = None,
        base_url: str = "http://localhost:3131/v1",
        model: str = "gemini-3-flash-preview",
        api_key: str = "not-needed",
    ) -> MemorisAgent:
        """從檔案載入 Agent 狀態與記憶資料庫。"""
        import os

        state: Optional[AgentState] = None
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = AgentState.from_dict(data)

        if db_path is None:
            if path.endswith(".json"):
                db_path = path[:-5] + "_memory.json"
            else:
                db_path = path + "_memory.json"

        if os.path.exists(db_path) and hasattr(db, "load"):
            db.load(db_path)  # type: ignore[union-attr]

        return cls(
            agent_id=agent_id,
            db=db,
            base_url=base_url,
            model=model,
            api_key=api_key,
            state=state,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_compress_response(
        response: str, fallback_batch: list[Memory]
    ) -> tuple[str, str, str]:
        """解析壓縮 LLM 回應，回傳 (context, reflection, observation)。"""
        context = ""
        reflection = ""
        observation = ""

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("CONTEXT:"):
                context = line[len("CONTEXT:"):].strip()
            elif line.upper().startswith("REFLECTION:"):
                reflection = line[len("REFLECTION:"):].strip()
            elif line.upper().startswith("OBSERVATION:"):
                observation = line[len("OBSERVATION:"):].strip()

        # Fallback: 如果 LLM 沒有按格式回應，直接拼接原始記憶
        if not context and not reflection and not observation:
            context = " | ".join(m.context for m in fallback_batch)
            reflection = " | ".join(m.reflection for m in fallback_batch)
            observation = " | ".join(m.observation for m in fallback_batch)

        return context, reflection, observation

    def _parse_solidify_response(self, response: str) -> list[Fact]:
        """解析固化 LLM 回應，回傳提取的 Fact 列表。"""
        if "NO_FACTS" in response.upper():
            return []

        facts: list[Fact] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line.upper().startswith("FACT:"):
                continue
            try:
                # 解析格式: FACT: ... | REASON: ... | CONFIDENCE: ... | TYPE: ...
                parts_raw = line[5:].split("|")
                fact_text = parts_raw[0].strip()
                reason = ""
                confidence = 0.5
                fact_type = FactType.KNOWLEDGE

                for part in parts_raw[1:]:
                    part = part.strip()
                    if part.upper().startswith("REASON:"):
                        reason = part[7:].strip()
                    elif part.upper().startswith("CONFIDENCE:"):
                        try:
                            confidence = float(part[11:].strip())
                        except ValueError:
                            pass
                    elif part.upper().startswith("TYPE:"):
                        type_str = part[5:].strip().upper()
                        if type_str in ("KNOWLEDGE", "PROCEDURE"):
                            fact_type = FactType(type_str)

                if fact_text:
                    facts.append(Fact(
                        fact=fact_text,
                        reason=reason,
                        confidence=max(0.0, min(1.0, confidence)),
                        agent_id=self.agent_id,
                        type=fact_type,
                    ))
            except Exception:
                continue

        return facts
