import json
import os
from collections import Counter
from datetime import datetime
from ultralytics import YOLO

class SmartKitchenAI:
    def __init__(self, model_path=None, conf=0.6):
        # 1. 路徑防呆處理：如果沒有傳入路徑，預設抓取相對位置
        if model_path is None:
            # 取得當前檔案 (yolo_module.py) 的所在目錄
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # 組合出模型路徑 (請依據你的實際資料夾結構調整)
            model_path = os.path.join(base_dir, 'runs', 'detect', 'tibame_food_model2', 'weights', 'best.pt')
            
        print(f"[System] Loading YOLO Model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        
        # 2. 標籤翻譯字典 (根據你的類別自行擴充)
        self.translation_map = {
            "tomato": "番茄",
            "egg": "雞蛋",
            "onion": "洋蔥",
            "potato": "馬鈴薯",
            "carrot": "紅蘿蔔",
            "broccoli": "花椰菜",
            "cucumber": "小黃瓜",
            "apple": "蘋果",
            # ... 其他你在 roboflow 訓練的類別
        }

    def detect_and_parse(self, source, user_id):
        """
        執行偵測並回傳包含使用者 ID 的結構化資料
        """
        # 執行推論
        try:
            results = self.model.predict(source=source, conf=self.conf, save=False)
        except Exception as e:
            print(f"[Error] YOLO Prediction failed: {e}")
            return None

        detected_list = []
        
        # 解析 YOLO 結果
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                eng_name = names[class_id]
                confidence = float(box.conf[0])
                
                # 3. 進行翻譯 (如果字典找不到，就維持英文)
                cht_name = self.translation_map.get(eng_name, eng_name)
                
                detected_list.append({
                    "name": cht_name,       # 存中文
                    "original_name": eng_name, # 保留英文(若資料庫需要)
                    "confidence": round(confidence, 2)
                })

        # 統計數量
        item_names = [item['name'] for item in detected_list]
        counts = Counter(item_names)
        
        # 建立最終資料結構
        output_data = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "event_type": "fridge_scan",
            "total_items": len(detected_list),
            "ingredients": []
        }

        for name, count in counts.items():
            output_data["ingredients"].append({
                "ingredient_name": name,
                "quantity": count,
                "unit": "個" # 這裡可以統一單位
            })

        return output_data

    def get_json(self, data):
        return json.dumps(data, indent=4, ensure_ascii=False)

# --- 測試區 ---
if __name__ == "__main__":
    # 這裡如果不傳參數，會自動使用 __init__ 裡的預設路徑
    # 但建議測試時明確指定，避免路徑錯誤
    try:
        ai_chef = SmartKitchenAI(model_path='runs/detect/tibame_food_model2/weights/best.pt') 
        
        current_user_id = "U_TEST_001" 
        # 請確保這裡有一張真的圖片，不然會報錯
        img_source = "./egg.jpg" 
        
        if os.path.exists(img_source):
            raw_data = ai_chef.detect_and_parse(img_source, user_id=current_user_id)
            json_output = ai_chef.get_json(raw_data)
            print("--- 系統輸出 ---")
            print(json_output)
        else:
            print(f"❌ 找不到測試圖片: {img_source}")
            
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        print("建議檢查：模型路徑是否正確？是否有安裝 ultralytics？")