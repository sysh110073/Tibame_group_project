# 使用官方輕量級 Python 環境
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統層級的依賴 (OpenCV 運作需要這些 .so 檔)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 複製需求檔並安裝
COPY requirements.txt .
# --no-cache-dir 可以減少映像檔體積
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼到容器內
COPY . .

# 設定環境變數 (確保 Python 輸出直接顯示在 Log)
ENV PYTHONUNBUFFERED=1

# Cloud Run 會自動帶入 PORT 環境變數，預設 8080
# 使用 gunicorn 作為生產環境伺服器
# --workers 1 : 因為模型很吃記憶體，建議先設 1 個 worker 避免記憶體爆掉
# --threads 8 : 使用多執行緒來處理併發
# --timeout 0 : 防止載入模型時間過長被切斷
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app