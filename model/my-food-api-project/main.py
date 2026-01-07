import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# --- 核心優化：在伺服器啟動時就預先載入模型 ---
# 這樣使用者傳照片來時，不用等模型載入，速度會快很多
print("正在載入模型，請稍候...")
try:
    model = YOLO('tibame_food_model_best_v1.pt')
    print("✅ 模型載入成功！")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    model = None

@app.route('/')
def health_check():
    return "Food AI API is running! (Tibame Project)"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 1. 讀取圖片
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # 2. 進行預測
        results = model(img)

        # 3. 整理結果 (JSON)
        predictions = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id] # 取得中文名稱
                confidence = float(box.conf)
                
                # 過濾信心度太低的 (例如低於 50%)
                if confidence > 0.5:
                    predictions.append({
                        "ingredient": class_name,
                        "confidence": round(confidence, 2)
                    })

        return jsonify({
            "status": "success",
            "data": predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 本地測試用，上雲端後會由 Gunicorn 啟動，不會跑這行
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)