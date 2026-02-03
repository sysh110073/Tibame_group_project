# 1. 使用官方瘦身版 Python
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 2. 安裝系統依賴 (補齊所有 OpenCV 缺少的依賴)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. 複製 requirements.txt
COPY requirements.txt .

# 4. 安裝套件
# (A) 強制安裝 CPU 版 PyTorch (瘦身關鍵)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# (B) 安裝其他套件
RUN pip install --no-cache-dir -r requirements.txt

# (C) 移除 NVIDIA 殘留
RUN pip uninstall -y nvidia-cublas-cu11 nvidia-cudnn-cu11 || true

# 5. 複製程式碼
COPY . .

# 6. 【關鍵修正】設定環境變數
# 讓 Python 輸出直接顯示 (Log 才看得到)
ENV PYTHONUNBUFFERED=1
# 解決 YOLO warning: 把設定檔指向暫存區
ENV YOLO_CONFIG_DIR='/tmp'
# 解決 Matplotlib 可能的報錯 (如果有的話)
ENV MPLCONFIGDIR='/tmp'

# 7. 啟動指令 (Timeout 設長一點)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "120", "main:app"]