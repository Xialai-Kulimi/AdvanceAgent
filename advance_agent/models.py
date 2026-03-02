"""MemorisAgent 核心資料結構。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union


class FactType(str, Enum):
    """Fact 的類型。"""
    KNOWLEDGE = "KNOWLEDGE"
    PROCEDURE = "PROCEDURE"


class RecallSourceType(str, Enum):
    """RecallResult 的來源類型。"""
    MEMORY = "MEMORY"
    FACT = "FACT"


def _new_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    """情節記憶的原始單元。"""

    context: str
    reflection: str
    observation: str
    agent_id: str

    id: str = field(default_factory=_new_id)
    created_at: str = field(default_factory=_now_iso)
    access_count: float = 0.0
    level: int = 0
    compress: bool = False
    solidified: bool = False
    protected: bool = False
    embedding: Optional[list[float]] = None

    def to_embed_text(self) -> str:
        """回傳用於計算 embedding 的拼接文字。"""
        return f"{self.context}\n{self.reflection}\n{self.observation}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "context": self.context,
            "reflection": self.reflection,
            "observation": self.observation,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "level": self.level,
            "compress": self.compress,
            "solidified": self.solidified,
            "protected": self.protected,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            context=data["context"],
            reflection=data["reflection"],
            observation=data["observation"],
            created_at=data.get("created_at", _now_iso()),
            access_count=data.get("access_count", 0.0),
            level=data.get("level", 0),
            compress=data.get("compress", False),
            solidified=data.get("solidified", False),
            protected=data.get("protected", False),
            embedding=data.get("embedding"),
        )

    def display_text(self) -> str:
        """格式化為注入 Prompt 的文字。"""
        tag = "Protected | " if self.protected else ""
        return (
            f"[Memory | {tag}level={self.level}]\n"
            f"Context: {self.context}\n"
            f"Reflection: {self.reflection}\n"
            f"Observation: {self.observation}"
        )


# ---------------------------------------------------------------------------
# Fact
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    """經內化的穩定知識。"""

    fact: str
    reason: str
    confidence: float
    agent_id: str
    type: FactType = FactType.KNOWLEDGE

    id: str = field(default_factory=_new_id)
    disputed: bool = False
    access_count: float = 0.0
    last_updated: str = field(default_factory=_now_iso)
    embedding: Optional[list[float]] = None

    def to_embed_text(self) -> str:
        """回傳用於計算 embedding 的文字。"""
        return self.fact

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "fact": self.fact,
            "reason": self.reason,
            "confidence": self.confidence,
            "type": self.type.value,
            "disputed": self.disputed,
            "access_count": self.access_count,
            "last_updated": self.last_updated,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fact:
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            fact=data["fact"],
            reason=data["reason"],
            confidence=data["confidence"],
            type=FactType(data.get("type", "KNOWLEDGE")),
            disputed=data.get("disputed", False),
            access_count=data.get("access_count", 0.0),
            last_updated=data.get("last_updated", _now_iso()),
            embedding=data.get("embedding"),
        )

    def display_text(self) -> str:
        """格式化為注入 Prompt 的文字。"""
        prefix = "[存疑] " if self.disputed else ""
        return (
            f"{prefix}[Fact | {self.type.value} | confidence={self.confidence:.2f}]\n"
            f"{self.fact}\n"
            f"Reason: {self.reason}"
        )


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    """當前的工作記憶目標。"""

    content: str
    agent_id: str

    id: str = field(default_factory=_new_id)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Goal:
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            content=data["content"],
            updated_at=data.get("updated_at", _now_iso()),
        )


# ---------------------------------------------------------------------------
# RecallResult
# ---------------------------------------------------------------------------

@dataclass
class RecallResult:
    """Auto-Recall 檢索結果的統一結構。"""

    source_type: RecallSourceType
    item: Union[Memory, Fact]
    score: float

    def display_text(self) -> str:
        """格式化為注入 Prompt 的文字。"""
        return self.item.display_text()
