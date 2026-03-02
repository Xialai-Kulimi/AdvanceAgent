# AdvanceAgent

**AdvanceAgent** 是一個基於 OpenAI SDK 核心開發的高階 AI Agent 框架。它繼承了 [IntimusAdvance](https://github.com/Xialai-Kulimi/IntimusAdvance) 的優點，並將 LLM 通訊層完全重寫為 OpenAI Chat Completion 協議，提供極致的靈活性與仿生記憶能力。

## 🌟 核心特性

- **OpenAI 原生支持**：全面採用 `openai` Python SDK，可對接任何 OpenAI 相容的 API 端點（如本地端點 `http://localhost:3131/v1`）。
- **仿生記憶系統 (MemorisAgent)**：
  - **Auto-Recall (Flashback)**：推理前自動檢索相關記憶並注入上下文。
  - **Personality Context**：動態將長期事實與活躍記憶固化到系統提示詞中，型塑 Agent 人格。
  - **層級化壓縮 (Compress)**：模仿人類遺忘曲線，將舊記憶濃縮為高階摘要，節省 Context Window。
  - **事實固化 (Solidify)**：從情節記憶中自動提取穩定、可重用的 Facts。
- **純 Python 的 Tool Call Loop**：不再依賴二進制 server 處理工具，所有 Tool 調用均在 Python 層級閉環。
- **類型安全與序列化**：完善的資料模型與狀態持久化（`dump`/`load`）。

## 🚀 快速開始

### 安裝

```bash
git clone https://github.com/Xialai-Kulimi/AdvanceAgent.git
cd AdvanceAgent
pip install -e .
```

### 基礎使用

```python
from advance_agent import BaseAgent

agent = BaseAgent(
    base_url="http://localhost:3131/v1",
    model="gemini-3-flash-preview"
)
agent.set_system_instruction("你是一個簡短回應的助手。")

for msg in agent.generate("你好，請問你是？"):
    print(msg["content"])
```

### 使用備有記憶功能的 MemorisAgent

```python
from advance_agent import MemorisAgent, SimpleMemoryDB

# 定義你的 embedding 函數
def my_embed_fn(text):
    # 呼叫你的 embedding 模型
    return [0.1, 0.2, ...]

db = SimpleMemoryDB(embed_fn=my_embed_fn)
agent = MemorisAgent(agent_id="UserA", db=db)

# Agent 會自動使用 save_memory, query 等工具管理自身記憶
for msg in agent.generate("記住，我最喜歡的顏色是祖母綠。"):
    print(msg["content"])
```

## 📚 文件

- [MemorisAgent 設計與記憶模型](docs/MemorisAgent.md)

## 📜 授權協議

本專案採用 **MIT License** 授權。
你可以自由修改與使用，但請務必保留原作者的版權聲明與授權文件。詳見 [LICENSE](LICENSE) 文件。
