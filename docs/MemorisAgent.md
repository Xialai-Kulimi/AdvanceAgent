# MemorisAgent 設計與記憶模型

`MemorisAgent` 是 AdvanceAgent 框架中的核心，旨在模仿人類的生物記憶機制，讓 Agent 具備「長期學習」與「情境感知」的能力。

## 🧠 記憶結構

我們將記憶分為三種主要形式：

1.  **情節記憶 (Memory)**：記錄具體事件、對話背景與反射。具備 `level` (層級) 屬性，隨著時間推移被壓縮。
2.  **事實/知識 (Fact)**：從情節記憶中提取出的穩定規律或宣告式知識（例如：用戶的偏好）。
3.  **工作目標 (Goal)**：當前任務的核心導向，會注入到系統提示詞的最頂層。

## ⚙️ 核心機制

### 1. Auto-Recall & Flashback

在每次 `generate()` 呼叫時，Agent 會自動執行：

- **檢索**：根據用戶目前的輸入，在向量資料庫中搜尋最相關的 `Memory` 與 `Fact`。
- **注入 (Flashback)**：將檢索結果以 `<flashback>` 標籤包裹，作為最新的 User Message 注入到對話歷史中，讓模型在推理時「瞬間想起」相關往事。
- **冷卻機制 (Debounce)**：使用滑動視窗避免在短時間內重複注入相同的記憶片段。

### 2. Personality Context

Agent 會定期選取存取次數最高 (Access Count) 的 Facts 與活躍的 Memories，直接寫入 **System Instruction** 的結尾。這構成了 Agent 的「底色」，使其行為保持一致。

### 3. 記憶壓縮 (Compression)

當某一開發層級（Level 0）的未壓縮記憶達到閾值（預設 20 筆）時，系統會將其中最舊的 10 筆原始記憶交給 LLM 進行摘要，產生一筆 Level 1 的高品質記憶，並標記原始記憶為已壓縮。這模擬了睡眠中的記憶整合過程。

### 4. 事實固化 (Solidification)

掃描原始情節記憶，辨識其中具有長久價值的信息，轉化為 `Fact`。Fact 擁有 `confidence` (信心值) 與 `disputed` (存疑) 狀態，Agent 可透過工具自我修正。

## 🛠️ 管理工具 (Tools)

模型在推理過程中可以主動呼叫以下工具：

- `save_memory`: 記錄一段情節。
- `save_fact`: 內化一個知識點。
- `set_goal`: 更新目前的工作重心。
- `query`: 主動搜尋記憶庫（當 Auto-Recall 不足以應付複雜查詢時）。
- `dispute_fact`: 當發現之前的知識有誤時，標記為存疑並降低信心值。

## 💾 儲存層 (MemoryDB)

`MemorisAgent` 透過 `MemoryDB` 協議與後端介接。

- 預設提供 `SimpleMemoryDB`: 基於內存的存儲，支持 JSON 導出與導入。
- 可擴展至 `LanceDB` 或其他向量資料庫以處理大規模長程記憶。
