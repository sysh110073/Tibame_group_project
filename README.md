# AI 剩食主廚 (AI Leftover Chef)

> **智慧化剩食食材辨識與客製化食譜推薦系統**

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/26048e1c-517c-4566-9c0e-f8696c5a23d2" />


## 專案簡介 (About The Project)

本專案旨在解決現代家庭食材管理不善與食物浪費之痛點。透過整合邊緣運算層級的電腦視覺技術 (YOLO11n)、檢索增強生成 (RAG) 以及大語言模型 (LLM)，「AI 剩食主廚」能即時分析使用者提供的冰箱食材影像，並參照使用者歷史飲食偏好，自動生成具備高可行性與營養參考價值之專屬食譜。

### 核心解決方案
* **降低食材浪費**：精準辨識既有食材，提供最大化食材利用率之烹飪方案。
* **降低決策成本**：自動化食譜生成流程，免除人工搜尋與篩選之時間成本。
* **高度客製化**：導入向量檢索技術紀錄使用者飲食輪廓（如：過敏原、熱量控制），確保生成內容之安全性與個人化。

---

## 核心技術與特色 (Key Features)

* **高效能影像辨識模組**：採用自定義訓練之 `YOLO11n` 模型。透過導入 C2PSA 空間注意力機制，針對冰箱內部複雜背景與物件遮擋場景進行優化，於 Cloud Run 環境下推論延遲僅需 141ms。
* **高可靠性檢索增強生成 (RAG)**：透過 Dify 平台編排 RAG 工作流。設定嚴格之提示詞護欄 (Prompt Guardrails)，有效抑制 LLM 產生無中生有之「AI 幻覺」，確保食安規範。
* **非同步雙軌平行處理架構**：為解決生成式 AI 普遍存在之高延遲問題，系統於接收影像後，同步啟動 YOLO 視覺辨識與 Dify RAG 背景環境準備，大幅優化系統響應時間 (Response Time)。
* **前後端解耦設計 (Decoupling)**：強制 LLM 僅輸出結構化 JSON 資料，由 Python 後端統一接管 Line Flex Message 之 UI 組裝。此架構有效降低 API Token 消耗、消弭排版語法錯誤導致之系統崩潰，並保留未來擴展至 Web 或 App 之前端彈性。

---

## 技術堆疊 (Tech Stack)

### 雲端架構與部署 (Cloud & Deployment)
* **GCP Cloud Run**: 核心微服務部署與無伺服器運算
* **Docker**: 應用程式容器化與環境隔離
* **Dify**: RAG 工作流編排與 LLM 串接管理

### 人工智慧與資料服務 (AI & Data Services)
* **視覺模型**: YOLO11n (Precision 優先優化策略)
* **大語言模型**: OpenAI GPT-4o-mini / GPT-3.5-turbo
* **影像生成模型**: Gemini Imagen 4 (食譜成品圖預測生成)
* **向量資料庫**: Pinecone (使用者偏好與歷史輪廓儲存)

### 應用程式與介面 (Application & Interface)
* **後端框架**: Python (Flask / FastAPI)
* **前端介面**: Line Messaging API (Flex Message)

---

## 系統架構 (System Architecture)

本專案之系統架構設計著重於「穩定性」與「高擴充性」：

1. **視覺感知層**：前端傳入之影像交由優化後之 YOLO 模型進行物件偵測，輸出標準化之食材標籤陣列。
2. **決策中樞與檢索層**：食材陣列結合 Pinecone 中提取之使用者偏好向量，作為 Dify 系統之輸入上下文。
3. **資料流轉換與呈現層**：LLM 完成推理後輸出純 JSON 格式數據，由 Python 服務層進行資料清洗與 Line UI 模板綁定，最終將結果回傳至用戶端。

---

## 本地環境建置 (Getting Started)

### 1. 系統要求
確保開發環境已安裝 Python 3.9 或以上版本，並具備 Git 版本控制工具。

### 2. 專案複製
```bash
git clone [https://github.com/您的帳號/AI-Leftover-Chef.git](https://github.com/您的帳號/AI-Leftover-Chef.git)
cd AI-Leftover-Chef
```

### 3. 安裝依賴套件
```
pip install -r requirements.txt
```

### 4. 環境變數設定
請於專案根目錄建立 .env 檔案，並配置以下必要環境變數：
```
LINE_CHANNEL_ACCESS_TOKEN=your_token
LINE_CHANNEL_SECRET=your_secret
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
DIFY_API_KEY=your_dify_key
```

### 5. 啟動測試伺服器
```
python app.py
```

開發團隊 (Team Members)
本專案由「AI 智慧應用開發實戰養成班」第二組共同開發：

黃凡爵
王弘凱
黃冠傑
盧柏辰
黃郁如

指導老師 - 周志昂、郭惠民
