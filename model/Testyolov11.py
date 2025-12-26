from ultralytics import YOLO
import cv2

# --- 設定路徑 ---
# 請去你的專案資料夾找 'runs/detect/train(數字)/weights/best.pt'
# 注意：如果不確定是 train 幾號，請找日期最新的那個資料夾
custom_model_path = 'runs/detect/tibame_food_model2/weights/best.pt' 

# 載入你剛剛訓練好的模型
model = YOLO(custom_model_path)

# --- 測試圖片 ---
# 找一張網路上的番茄照片，或是你自己拍的照片路徑
image_path = './egg.jpg' 

# 執行預測
# conf=0.5 代表信心度大於 50% 才顯示，避免亂猜
results = model.predict(source=image_path, save=True, conf=0.5)

print(f"測試完成！結果圖片已儲存至: {results[0].save_dir}")

# 顯示偵測到的類別與信心度
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])
        print(f"➡️ 偵測到: {class_name} (信心度: {confidence:.2f})")