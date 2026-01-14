import os
import json
import requests
from flask import Flask, request, abort

# Line Bot SDK
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, 
    TextSendMessage, ImageSendMessage
)

app = Flask(__name__)

# ==========================================
# 1. 從 Cloud Run 環境變數讀取設定
# ==========================================
# Line 相關金鑰
LINE_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

# 其他微服務的網址 (請在 Cloud Run 變數中設定完整網址)
# 例如: https://yolo-service-xyz.run.app/detect
YOLO_API_URL = os.environ.get('YOLO_API_URL') 

# 例如: https://rag-service-xyz.run.app/api/chef
RAG_API_URL = os.environ.get('RAG_API_URL')

# 初始化 Line Bot
if not LINE_ACCESS_TOKEN or not LINE_SECRET:
    print("❌ 錯誤：未讀取到 Line Bot 環境變數，請檢查 Cloud Run 設定")

line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_SECRET)


# ==========================================
# 2. 輔助函式：呼叫 YOLO 服務 (眼睛)
# ==========================================
# ==========================================
# 2. 輔助函式：呼叫 YOLO 服務 (修正版)
# ==========================================
def call_yolo_service(image_binary):
    if not YOLO_API_URL:
        print("❌ YOLO_API_URL 未設定")
        return []
        
    print(f"🔍 正在呼叫 YOLO 辨識服務: {YOLO_API_URL}")
    try:
        # 1. 您的 YOLO 程式碼接收的是 'file' -> 這邊正確不用改
        files = {'file': ('line_image.jpg', image_binary, 'image/jpeg')}
        
        response = requests.post(YOLO_API_URL, files=files, timeout=30)
        
        if response.status_code == 200:
            json_response = response.json()
            
            # 2. 您的 YOLO 回傳的食材清單在 'data' 這個 key 裡面
            predictions = json_response.get('data', [])
            
            # 3. 解析結構：從 [{"ingredient": "egg", ...}] 提取出 "egg"
            # 這裡用 list comprehension 把食材名稱抓出來
            found_ingredients = [item.get('ingredient') for item in predictions if item.get('ingredient')]
            
            # 去除重複並回傳
            result_list = list(set(found_ingredients))
            print(f"🥦 YOLO 辨識成功: {result_list}")
            return result_list
            
        else:
            print(f"❌ YOLO 服務回傳錯誤 ({response.status_code}): {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ YOLO 連線失敗: {e}")
        return []


# ==========================================
# 3. 輔助函式：呼叫 RAG 服務 (大腦)
# ==========================================
def call_rag_service(ingredients_text):
    if not RAG_API_URL:
        print("❌ RAG_API_URL 未設定")
        return None
        
    print(f"🍳 正在呼叫 RAG 主廚服務 (食材: {ingredients_text})...")
    try:
        payload = {
            "message": f"我冰箱剩下這些食材：{ingredients_text}。請幫我設計一道食譜。",
            "need_image": True
        }
        
        # 發送 POST 請求
        response = requests.post(RAG_API_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            # 預期回傳: {"text": "食譜內容...", "image_url": "https://..."}
            return response.json()
        else:
            print(f"❌ RAG 服務回傳錯誤 ({response.status_code}): {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ RAG 連線失敗: {e}")
        return None


# ==========================================
# 4. Line Webhook 入口
# ==========================================
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("❌ Invalid Signature")
        abort(400)
        
    return 'OK'


# ==========================================
# 5. 處理圖片訊息 (核心流程)
# ==========================================
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_id = event.message.id
    
    try:
        # A. 取得圖片內容
        message_content = line_bot_api.get_message_content(message_id)
        image_binary = message_content.content
        
        # B. 呼叫 YOLO
        ingredients_list = call_yolo_service(image_binary)
        
        # 如果沒辨識到東西
        if not ingredients_list:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="主廚看不清楚照片裡的食材，請試著拍近一點，或是直接用文字告訴我！")
            )
            return
            
        # 轉成字串: ['egg', 'onion'] -> "egg, onion"
        ingredients_str = ", ".join(ingredients_list)
        
        # C. 呼叫 RAG
        # 這裡可以先傳一個 loading 訊息，但因為 reply_token 只能用一次，所以我們直接讓使用者等一下
        rag_result = call_rag_service(ingredients_str)
        
        if not rag_result:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=f"辨識出：{ingredients_str}，但主廚現在靈感枯竭 (RAG 服務連線失敗)，請稍後再試。")
            )
            return

        # D. 組合回覆
        recipe_text = rag_result.get("text", "沒有產生文字")
        image_url = rag_result.get("image_url")
        
        replies = []
        
        # 訊息 1: 辨識結果 + 食譜文字
        full_text = f"👁️ 辨識食材：{ingredients_str}\n\n{recipe_text}"
        # 限制文字長度避免 Line 報錯 (上限 5000 字)
        replies.append(TextSendMessage(text=full_text[:4900]))
        
        # 訊息 2: 生成圖 (如果有)
        if image_url and image_url.startswith("https"):
            replies.append(ImageSendMessage(
                original_content_url=image_url,
                preview_image_url=image_url
            ))
            
        line_bot_api.reply_message(event.reply_token, replies)

    except Exception as e:
        print(f"❌ 處理圖片流程發生錯誤: {e}")
        # 避免使用者空等，回傳錯誤提示 (如果 reply_token 還沒用掉)
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="系統發生預期外的錯誤，請稍後再試。")
            )
        except:
            pass


# ==========================================
# 6. 處理文字訊息 (手動輸入食材)
# ==========================================
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_input = event.message.text
    
    # 直接把文字丟給 RAG
    rag_result = call_rag_service(user_input)
    
    if rag_result:
        recipe_text = rag_result.get("text", "")
        image_url = rag_result.get("image_url")
        
        replies = [TextSendMessage(text=recipe_text)]
        if image_url and image_url.startswith("https"):
            replies.append(ImageSendMessage(
                original_content_url=image_url,
                preview_image_url=image_url
            ))
        line_bot_api.reply_message(event.reply_token, replies)
    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="主廚現在有點忙，請稍後再試。")
        )

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)