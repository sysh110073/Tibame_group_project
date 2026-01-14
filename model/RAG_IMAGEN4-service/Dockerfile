# 使用最輕量的 Python 版本
FROM python:3.10-slim

WORKDIR /app

# 安裝基礎套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 建立圖片暫存區
RUN mkdir -p static

# 啟動命令
CMD ["sh", "-c", "exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 main:app"]