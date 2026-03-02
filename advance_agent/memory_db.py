"""MemoryDB Protocol 與 SimpleMemoryDB 實作。"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .models import (
    Fact,
    FactType,
    Goal,
    Memory,
    RecallResult,
    RecallSourceType,
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MemoryDB(Protocol):
    """記憶資料庫的介面定義。

    所有實作都必須在建構時接收 ``embed_fn``，
    並在 ``save_memory`` / ``save_fact`` / ``query`` 等方法內部處理 embedding。
    """

    def save_memory(self, memory: Memory) -> None: ...
    def save_fact(self, fact: Fact) -> None: ...
    def update_fact(self, fact: Fact) -> None: ...
    def get_fact_by_id(self, fact_id: str, agent_id: str) -> Optional[Fact]: ...
    def save_goal(self, goal: Goal) -> None: ...
    def get_latest_goal(self, agent_id: str) -> Optional[Goal]: ...
    def query(self, text: str, agent_id: str, top_k: int = 7) -> list[RecallResult]: ...
    def get_uncompressed_memories(self, agent_id: str, level: int) -> list[Memory]: ...
    def get_all_active_memories(self, agent_id: str) -> list[Memory]: ...
    def get_unsolidified_memories(self, agent_id: str) -> list[Memory]: ...
    def get_top_facts(self, agent_id: str, limit: int) -> list[Fact]: ...
    def mark_compressed(self, memory_ids: list[str], agent_id: str) -> None: ...
    def mark_solidified(self, memory_ids: list[str], agent_id: str) -> None: ...
    def mark_protected(self, memory_ids: list[str], agent_id: str) -> None: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """計算兩個向量的 cosine similarity。不依賴 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _access_weight(access_count: float) -> float:
    """access_count 對檢索分數的加權係數。"""
    return min(1.1 ** access_count, 3.0)


# ---------------------------------------------------------------------------
# SimpleMemoryDB
# ---------------------------------------------------------------------------

class SimpleMemoryDB:
    """基於 in-memory list 的 MemoryDB 實作，用於開發與測試。

    Parameters
    ----------
    embed_fn:
        接收一段文字、回傳 embedding 向量的函數。
        所有 embedding 計算都在 DB 內部透過此函數處理。
    """

    def __init__(self, embed_fn: Callable[[str], list[float]]) -> None:
        self._embed_fn = embed_fn
        self._memories: list[Memory] = []
        self._facts: list[Fact] = []
        self._goals: list[Goal] = []

    # -- save / update -------------------------------------------------------

    def save_memory(self, memory: Memory) -> None:
        memory.embedding = self._embed_fn(memory.to_embed_text())
        self._memories.append(memory)

    def save_fact(self, fact: Fact) -> None:
        fact.embedding = self._embed_fn(fact.to_embed_text())
        self._facts.append(fact)

    def update_fact(self, fact: Fact) -> None:
        for i, existing in enumerate(self._facts):
            if existing.id == fact.id:
                self._facts[i] = fact
                return
        raise KeyError(f"Fact not found: {fact.id}")

    def get_fact_by_id(self, fact_id: str, agent_id: str) -> Optional[Fact]:
        for f in self._facts:
            if f.id == fact_id and f.agent_id == agent_id:
                return f
        return None

    def save_goal(self, goal: Goal) -> None:
        self._goals.append(goal)

    def get_latest_goal(self, agent_id: str) -> Optional[Goal]:
        agent_goals = [g for g in self._goals if g.agent_id == agent_id]
        if not agent_goals:
            return None
        return agent_goals[-1]

    # -- query ---------------------------------------------------------------

    def query(self, text: str, agent_id: str, top_k: int = 7) -> list[RecallResult]:
        query_embedding = self._embed_fn(text)
        results: list[RecallResult] = []

        # 搜尋可檢索的 Memory（未壓縮 OR 受保護）
        for mem in self._memories:
            if mem.agent_id != agent_id:
                continue
            if mem.compress and not mem.protected:
                continue
            if mem.embedding is None:
                continue
            raw_score = _cosine_similarity(query_embedding, mem.embedding)
            weighted_score = raw_score * _access_weight(mem.access_count)
            results.append(RecallResult(
                source_type=RecallSourceType.MEMORY,
                item=mem,
                score=weighted_score,
            ))

        # 搜尋所有 Fact
        for fact in self._facts:
            if fact.agent_id != agent_id:
                continue
            if fact.embedding is None:
                continue
            raw_score = _cosine_similarity(query_embedding, fact.embedding)
            weighted_score = raw_score * _access_weight(fact.access_count)
            results.append(RecallResult(
                source_type=RecallSourceType.FACT,
                item=fact,
                score=weighted_score,
            ))

        # 排序取 top-k
        results.sort(key=lambda r: r.score, reverse=True)
        top_results = results[:top_k]

        # 更新 access_count (指數衰減)
        for i, r in enumerate(top_results):
            inc = 0.5 ** i
            r.item.access_count += inc

        return top_results

    # -- memory state queries ------------------------------------------------

    def get_uncompressed_memories(self, agent_id: str, level: int) -> list[Memory]:
        return [
            m for m in self._memories
            if m.agent_id == agent_id and m.level == level and not m.compress
        ]

    def get_unsolidified_memories(self, agent_id: str) -> list[Memory]:
        return [
            m for m in self._memories
            if m.agent_id == agent_id and not m.solidified
        ]

    def mark_compressed(self, memory_ids: list[str], agent_id: str) -> None:
        id_set = set(memory_ids)
        for mem in self._memories:
            if mem.id in id_set and mem.agent_id == agent_id:
                mem.compress = True

    def mark_solidified(self, memory_ids: list[str], agent_id: str) -> None:
        id_set = set(memory_ids)
        for mem in self._memories:
            if mem.id in id_set and mem.agent_id == agent_id:
                mem.solidified = True

    def mark_protected(self, memory_ids: list[str], agent_id: str) -> None:
        id_set = set(memory_ids)
        for mem in self._memories:
            if mem.id in id_set and mem.agent_id == agent_id:
                mem.protected = True

    def get_all_active_memories(self, agent_id: str) -> list[Memory]:
        """compress=false OR protected=true."""
        return [
            m for m in self._memories
            if m.agent_id == agent_id and (not m.compress or m.protected)
        ]

    def get_top_facts(self, agent_id: str, limit: int) -> list[Fact]:
        """access_count 降序取 limit 筆。"""
        agent_facts = [
            f for f in self._facts if f.agent_id == agent_id
        ]
        agent_facts.sort(key=lambda f: f.access_count, reverse=True)
        return agent_facts[:limit]

    # -- persistence ---------------------------------------------------------

    def dump(self, path: str) -> None:
        """將所有資料序列化為 JSON 寫入檔案。"""
        data = {
            "memories": [m.to_dict() for m in self._memories],
            "facts": [f.to_dict() for f in self._facts],
            "goals": [g.to_dict() for g in self._goals],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """從 JSON 檔案載入資料，取代目前的內容。"""
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        self._memories = [Memory.from_dict(d) for d in data.get("memories", [])]
        self._facts = [Fact.from_dict(d) for d in data.get("facts", [])]
        self._goals = [Goal.from_dict(d) for d in data.get("goals", [])]
