from ultralytics import YOLO

def train_custom_model():
    # 1. 載入預訓練模型 (作為基礎)
    model = YOLO('yolo11m.pt') 
    
    print("開始微調訓練 (Fine-tuning)...")
    
    # 2. 開始訓練
    # data: 指向你下載的資料集內的 data.yaml 路徑
    # epochs: 訓練輪數，建議先設 30-50 跑跑看 (大約需 1-2 小時，視顯卡而定)
    # imgsz: 圖片大小，通常 640
    results = model.train(
        data='Smart_Fridge/data.yaml', # 修改這裡！
        epochs=5, 
        imgsz=640,
        batch=16,
        name='tibame_food_model', # 訓練結果會存在 runs/detect/tibame_food_model
        device=0  # 使用 GPU 編號 0 (若只有一張 GPU 就是 0
    )
    
    print("訓練完成！最佳模型權重保存在 runs/detect/tibame_food_model/weights/best.pt")

if __name__ == '__main__':
    # 注意：Windows 用戶必須把訓練代碼放在 if __name__ == '__main__': 之下
    train_custom_model()