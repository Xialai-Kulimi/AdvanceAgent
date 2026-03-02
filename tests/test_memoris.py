"""MemorisAgent 單元測試 -- 工具處理器、Auto-Recall、壓縮、持久化。"""

from __future__ import annotations

import os
import tempfile

import pytest

from advance_agent.base_agent import AgentState
from advance_agent.memoris_agent import (
    COMPRESS_THRESHOLD,
    MemorisAgent,
)
from advance_agent.memory_db import SimpleMemoryDB
from advance_agent.models import Fact, FactType, Goal, Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:3131/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
MODEL = os.environ.get("OPENAI_MODEL", "gemini-3-flash-preview")


def _dummy_embed(text: str) -> list[float]:
    """簡易 embedding 函數。"""
    n = len(text)
    c = ord(text[0]) if text else 0
    return [float(n), float(c % 10), float(n * 0.1), float(c * 0.01)]


def _make_agent(
    agent_id: str = "test-agent",
) -> tuple[MemorisAgent, SimpleMemoryDB]:
    db = SimpleMemoryDB(embed_fn=_dummy_embed)
    agent = MemorisAgent(
        agent_id=agent_id,
        db=db,
        base_url=BASE_URL,
        model=MODEL,
        api_key=API_KEY,
    )
    return agent, db


# ---------------------------------------------------------------------------
# Tool handler 測試
# ---------------------------------------------------------------------------

class TestToolHandlers:
    """直接呼叫 tool handler 方法，驗證 DB 操作正確。"""

    @pytest.fixture(autouse=True, scope="class")
    def setup_agent(self, request: pytest.FixtureRequest) -> None:
        agent, db = _make_agent()
        agent.set_system_instruction("Test assistant. Keep responses short.")
        request.cls.agent = agent  # type: ignore[union-attr]
        request.cls.db = db  # type: ignore[union-attr]

    def test_set_goal(self) -> None:
        result = self.agent._tool_set_goal("learn Python")  # type: ignore[attr-defined]
        assert "learn Python" in result
        goal = self.db.get_latest_goal("test-agent")  # type: ignore[attr-defined]
        assert goal is not None
        assert goal.content == "learn Python"

    def test_save_fact_knowledge(self) -> None:
        result = self.agent._tool_save_fact(  # type: ignore[attr-defined]
            fact="user prefers dark mode",
            reason="explicitly stated",
            confidence=0.95,
        )
        assert "saved" in result.lower()
        facts = [f for f in self.db._facts if f.fact == "user prefers dark mode"]  # type: ignore[attr-defined]
        assert len(facts) == 1
        assert facts[0].type == FactType.KNOWLEDGE
        assert facts[0].agent_id == "test-agent"

    def test_save_fact_procedure(self) -> None:
        self.agent._tool_save_fact(  # type: ignore[attr-defined]
            fact="always run tests before deploy",
            reason="best practice",
            confidence=0.8,
            type="PROCEDURE",
        )
        facts = [f for f in self.db._facts if f.fact == "always run tests before deploy"]  # type: ignore[attr-defined]
        assert len(facts) == 1
        assert facts[0].type == FactType.PROCEDURE

    def test_save_fact_clamps_confidence(self) -> None:
        self.agent._tool_save_fact(fact="over", reason="r", confidence=1.5)  # type: ignore[attr-defined]
        self.agent._tool_save_fact(fact="under", reason="r", confidence=-0.5)  # type: ignore[attr-defined]
        over = [f for f in self.db._facts if f.fact == "over"]  # type: ignore[attr-defined]
        under = [f for f in self.db._facts if f.fact == "under"]  # type: ignore[attr-defined]
        assert over[0].confidence == 1.0
        assert under[0].confidence == 0.0

    def test_save_memory(self) -> None:
        result = self.agent._tool_save_memory(  # type: ignore[attr-defined]
            context="discussing UI design",
            reflection="user seems frustrated",
            observation="user wants dark mode",
        )
        assert "saved" in result.lower()
        mems = [m for m in self.db._memories if m.context == "discussing UI design"]  # type: ignore[attr-defined]
        assert len(mems) == 1
        assert mems[0].agent_id == "test-agent"
        assert mems[0].level == 0
        assert mems[0].compress is False
        assert mems[0].embedding is not None

    def test_query_returns_results(self) -> None:
        result = self.agent._tool_query("dark mode")  # type: ignore[attr-defined]
        assert "Result" in result
        assert "score=" in result

    def test_dispute_fact(self) -> None:
        self.agent._tool_save_fact(  # type: ignore[attr-defined]
            fact="earth is flat", reason="mistake", confidence=0.6,
        )
        target = [f for f in self.db._facts if f.fact == "earth is flat"][0]  # type: ignore[attr-defined]
        result = self.agent._tool_dispute_fact(target.id)  # type: ignore[attr-defined]
        assert "disputed" in result.lower()
        assert target.disputed is True
        assert abs(target.confidence - 0.3) < 1e-6

    def test_dispute_fact_not_found(self) -> None:
        result = self.agent._tool_dispute_fact("nonexistent")  # type: ignore[attr-defined]
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# Auto-Recall 測試
# ---------------------------------------------------------------------------

class TestAutoRecall:
    @pytest.fixture(autouse=True, scope="class")
    def setup_agent(self, request: pytest.FixtureRequest) -> None:
        agent, db = _make_agent(agent_id="recall-agent")
        request.cls.agent = agent  # type: ignore[union-attr]
        request.cls.db = db  # type: ignore[union-attr]

    def test_recall_context_empty(self) -> None:
        initial_len = len(self.agent.state.history)  # type: ignore[attr-defined]

        self.agent.set_system_instruction("Base.")  # type: ignore[attr-defined]
        self.agent._update_personality_context()  # type: ignore[attr-defined]
        assert self.agent.state.system_instruction == "Base."  # type: ignore[attr-defined]

        self.agent._perform_flashback("hello", set())  # type: ignore[attr-defined]
        assert len(self.agent.state.history) == initial_len  # type: ignore[attr-defined]

    def test_recall_context_with_goal(self) -> None:
        self.agent.set_system_instruction("Base.")  # type: ignore[attr-defined]
        self.agent._tool_set_goal("build a website")  # type: ignore[attr-defined]

        self.agent._update_personality_context()  # type: ignore[attr-defined]
        sys_instr = self.agent.state.system_instruction  # type: ignore[attr-defined]

        assert sys_instr is not None
        assert "Base." in sys_instr
        assert "build a website" in sys_instr
        assert "[Current Goal]" in sys_instr

    def test_recall_context_with_memories(self) -> None:
        self.agent._tool_save_memory(  # type: ignore[attr-defined]
            context="web dev", reflection="user wants React", observation="prefers TS",
        )
        self.agent._update_personality_context()  # type: ignore[attr-defined]
        sys_instr = self.agent.state.system_instruction  # type: ignore[attr-defined]
        assert sys_instr is not None
        assert "Personality Context" in sys_instr
        assert "web dev" in sys_instr

        # Flashback -> User Message (History)
        initial_len = len(self.agent.state.history)  # type: ignore[attr-defined]
        self.agent._perform_flashback("web", set())  # type: ignore[attr-defined]

        assert len(self.agent.state.history) == initial_len + 1  # type: ignore[attr-defined]
        last_msg = self.agent.state.history[-1]  # type: ignore[attr-defined]
        assert last_msg["role"] == "user"
        assert "<flashback>" in last_msg["content"]
        assert "web dev" in last_msg["content"]

    def test_base_instruction_preserved(self) -> None:
        self.agent.set_system_instruction("Base instruction.")  # type: ignore[attr-defined]
        self.agent._inject_recall("test prompt")  # type: ignore[attr-defined]
        assert self.agent._base_system_instruction == "Base instruction."  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Compress threshold 測試
# ---------------------------------------------------------------------------

class TestCompressThreshold:
    @pytest.fixture(autouse=True, scope="class")
    def setup_agent(self, request: pytest.FixtureRequest) -> None:
        agent, db = _make_agent(agent_id="compress-agent")
        request.cls.agent = agent  # type: ignore[union-attr]

    def test_no_notice_below_threshold(self) -> None:
        for i in range(COMPRESS_THRESHOLD - 1):
            self.agent._tool_save_memory(  # type: ignore[attr-defined]
                context=f"ctx{i}", reflection=f"ref{i}", observation=f"obs{i}",
            )
        status = self.agent._check_compress_threshold()  # type: ignore[attr-defined]
        assert status is None

    def test_notice_at_threshold(self) -> None:
        self.agent._tool_save_memory(  # type: ignore[attr-defined]
            context="ctxN", reflection="refN", observation="obsN",
        )
        status = self.agent._check_compress_threshold()  # type: ignore[attr-defined]
        assert status is not None
        assert "compress" in status.lower()


# ---------------------------------------------------------------------------
# Agent ID 隔離
# ---------------------------------------------------------------------------

class TestAgentIdIsolation:
    @pytest.fixture(autouse=True, scope="class")
    def setup_agents(self, request: pytest.FixtureRequest) -> None:
        db = SimpleMemoryDB(embed_fn=_dummy_embed)
        a1 = MemorisAgent(
            agent_id="agent-1", db=db,
            base_url=BASE_URL, model=MODEL, api_key=API_KEY,
        )
        a2 = MemorisAgent(
            agent_id="agent-2", db=db,
            base_url=BASE_URL, model=MODEL, api_key=API_KEY,
        )
        request.cls.a1 = a1  # type: ignore[union-attr]
        request.cls.a2 = a2  # type: ignore[union-attr]
        request.cls.db = db  # type: ignore[union-attr]

    def test_isolation(self) -> None:
        self.a1._tool_save_memory(context="a1 ctx", reflection="r", observation="o")  # type: ignore[attr-defined]
        self.a2._tool_save_memory(context="a2 ctx", reflection="r", observation="o")  # type: ignore[attr-defined]
        self.a1._tool_save_fact(fact="a1 fact", reason="r", confidence=0.9)  # type: ignore[attr-defined]
        self.a2._tool_save_fact(fact="a2 fact", reason="r", confidence=0.9)  # type: ignore[attr-defined]

        a1_results = self.db.query("ctx", "agent-1", top_k=10)  # type: ignore[attr-defined]
        a2_results = self.db.query("ctx", "agent-2", top_k=10)  # type: ignore[attr-defined]

        assert all(r.item.agent_id == "agent-1" for r in a1_results)
        assert all(r.item.agent_id == "agent-2" for r in a2_results)


# ---------------------------------------------------------------------------
# Persistence 測試
# ---------------------------------------------------------------------------

class TestPersistence:
    @pytest.fixture(autouse=True, scope="class")
    def setup_agent(self, request: pytest.FixtureRequest) -> None:
        agent, db = _make_agent(agent_id="persist-test")
        request.cls.agent = agent  # type: ignore[union-attr]
        request.cls.db = db  # type: ignore[union-attr]

    def test_dump_and_load(self) -> None:
        self.agent._tool_save_memory(context="c", reflection="r", observation="o")  # type: ignore[attr-defined]
        self.agent._tool_save_fact(fact="f", reason="r", confidence=0.9)  # type: ignore[attr-defined]
        self.agent._tool_set_goal("test goal")  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "agent.json")
            self.agent.dump(state_path)  # type: ignore[attr-defined]

            db_path = os.path.join(tmpdir, "agent_memory.json")
            assert os.path.exists(state_path)
            assert os.path.exists(db_path)

            db2 = SimpleMemoryDB(embed_fn=_dummy_embed)
            agent2 = MemorisAgent.load(
                state_path, agent_id="persist-test", db=db2,
                base_url=BASE_URL, model=MODEL, api_key=API_KEY,
            )
            assert len(db2._memories) == 1
            assert len(db2._facts) == 1
            assert db2.get_latest_goal("persist-test") is not None


# ---------------------------------------------------------------------------
# Protected Memory 測試
# ---------------------------------------------------------------------------

class TestProtectedMemory:
    def test_mark_protected(self) -> None:
        db = SimpleMemoryDB(embed_fn=_dummy_embed)
        m = Memory(agent_id="a1", context="important", reflection="r", observation="o")
        db.save_memory(m)
        db.mark_compressed([m.id], "a1")
        db.mark_protected([m.id], "a1")

        results = db.query("important", "a1", top_k=5)
        mem_ids = [r.item.id for r in results if isinstance(r.item, Memory)]
        assert m.id in mem_ids

    def test_get_all_active_memories_includes_protected(self) -> None:
        db = SimpleMemoryDB(embed_fn=_dummy_embed)
        normal = Memory(agent_id="a1", context="normal", reflection="r", observation="o")
        protected = Memory(agent_id="a1", context="protected", reflection="r", observation="o")
        compressed = Memory(agent_id="a1", context="forgotten", reflection="r", observation="o")
        db.save_memory(normal)
        db.save_memory(protected)
        db.save_memory(compressed)

        db.mark_compressed([protected.id, compressed.id], "a1")
        db.mark_protected([protected.id], "a1")

        active = db.get_all_active_memories("a1")
        active_ids = {m.id for m in active}
        assert normal.id in active_ids
        assert protected.id in active_ids
        assert compressed.id not in active_ids


# ---------------------------------------------------------------------------
# get_top_facts 測試
# ---------------------------------------------------------------------------

class TestTopFacts:
    def test_get_top_facts_order_and_limit(self) -> None:
        db = SimpleMemoryDB(embed_fn=_dummy_embed)
        f1 = Fact(agent_id="a1", fact="low", reason="r", confidence=0.5)
        f2 = Fact(agent_id="a1", fact="high", reason="r", confidence=0.9)
        f3 = Fact(agent_id="a1", fact="mid", reason="r", confidence=0.7)
        f1.access_count = 1
        f2.access_count = 10
        f3.access_count = 5
        db.save_fact(f1)
        db.save_fact(f2)
        db.save_fact(f3)

        top = db.get_top_facts("a1", limit=2)
        assert len(top) == 2
        assert top[0].id == f2.id
        assert top[1].id == f3.id


# ---------------------------------------------------------------------------
# 解析器測試
# ---------------------------------------------------------------------------

class TestParsers:
    def test_parse_compress_response(self) -> None:
        response = (
            "CONTEXT: merged context here\n"
            "REFLECTION: merged reflection\n"
            "OBSERVATION: merged observation"
        )
        ctx, ref, obs = MemorisAgent._parse_compress_response(response, [])
        assert ctx == "merged context here"
        assert ref == "merged reflection"
        assert obs == "merged observation"

    def test_parse_compress_response_fallback(self) -> None:
        batch = [
            Memory(agent_id="a", context="c1", reflection="r1", observation="o1"),
            Memory(agent_id="a", context="c2", reflection="r2", observation="o2"),
        ]
        ctx, ref, obs = MemorisAgent._parse_compress_response("garbage", batch)
        assert "c1" in ctx and "c2" in ctx
        assert "r1" in ref and "r2" in ref

    def test_parse_solidify_response(self) -> None:
        agent, _ = _make_agent()
        response = (
            "FACT: user likes coffee | REASON: mentioned multiple times | "
            "CONFIDENCE: 0.9 | TYPE: KNOWLEDGE\n"
            "FACT: always backup before deploy | REASON: best practice | "
            "CONFIDENCE: 0.8 | TYPE: PROCEDURE\n"
        )
        facts = agent._parse_solidify_response(response)
        assert len(facts) == 2
        assert facts[0].fact == "user likes coffee"
        assert facts[0].type == FactType.KNOWLEDGE
        assert facts[1].type == FactType.PROCEDURE

    def test_parse_solidify_no_facts(self) -> None:
        agent, _ = _make_agent()
        facts = agent._parse_solidify_response("NO_FACTS")
        assert len(facts) == 0

# ---------------------------------------------------------------------------
# Integration Tests (requires localhost:3131)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("SKIP_LLM_TESTS", "").lower() in ("1", "true"),
    reason="LLM integration tests disabled",
)
class TestMemorisLLMIntegration:
    """需要 localhost:3131 運行中的整合測試。"""

    def test_end_to_end_memory_loop(self) -> None:
        agent, db = _make_agent(agent_id="e2e-agent")
        agent.set_system_instruction(
            "You are an assistant with bionic memory. "
            "When you learn something about the user, use 'save_fact' or 'save_memory'. "
            "Reply briefly."
        )

        # 1. 告訴它一些事情，期望它呼叫工具
        prompt = "My favorite color is emerald green. Remember this."
        responses = list(agent.generate(prompt))
        
        # 檢查是否呼叫了工具 (或者至少回應了)
        text = agent.extract_text(responses)
        assert len(text) > 0

        # 2. 驗證 DB 是否真的存入了
        facts = [f for f in db._facts if "emerald green" in f.fact.lower()]
        mems = [m for m in db._memories if "emerald green" in m.observation.lower()]
        assert len(facts) > 0 or len(mems) > 0

        # 3. 測試 Auto-Recall
        # 建立一個新的 Agent 實例，共用剛才的 DB
        agent2 = MemorisAgent(
            agent_id="e2e-agent", 
            db=db,
            base_url=BASE_URL,
            model=MODEL,
            api_key=API_KEY
        )
        agent2.set_system_instruction("Reply very briefly.")
        
        # 這次不提供資訊，直接問。Auto-Recall 應該會注入 context。
        responses2 = list(agent2.generate("What is my favorite color?"))
        text2 = agent2.extract_text(responses2)
        assert "emerald green" in text2.lower()

    def test_goal_persistence_in_prompt(self) -> None:
        agent, db = _make_agent(agent_id="goal-agent")
        agent.set_system_instruction("You are a helpful assistant.")
        agent._tool_set_goal("Always end your sentences with '!!!'.")
        
        responses = list(agent.generate("How are you?"))
        text = agent.extract_text(responses)
        assert "!!!" in text
