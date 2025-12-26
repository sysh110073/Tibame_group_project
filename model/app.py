from flask import Flask, request, abort
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent
import configparser
import requests
import json
import os
import cv2
import numpy as np

# 匯入我們自己寫的模組
from Chef_AI import SmartKitchenAI # 請確認你有將之前的 yolo code 存成 yolo_module.py
from rag_service import get_chef_response

# 讀取 Config
config = configparser.ConfigParser()
config.read('config.ini')

app = Flask(__name__)

# LINE 設定
LINE_ACCESS_TOKEN = config['LINE']['CHANNEL_ACCESS_TOKEN']
LINE_SECRET = config['LINE']['CHANNEL_SECRET']
handler = WebhookHandler(LINE_SECRET)

# 初始化 YOLO
ai_chef = SmartKitchenAI(model_path='runs/detect/tibame_food_model2/weights/best.pt', conf=0.5)

def reply_message_via_requests(reply_token, text_message):
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }
    payload = {
        "replyToken": reply_token,
        "messages": [
            {
                "type": "text",
                "text": text_message
            }
        ]
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        print(f"[Error] Line API 回傳錯誤: {response.text}")

@app.route("/callback", methods=['POST'])
def callback():
    # 取得 X-Line-Signature 標頭值
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)

    try:
        # 驗證簽章並處理事件
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# --- 處理文字訊息 ---
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    user_id = event.source.user_id
    user_text = event.message.text
    reply_token = event.reply_token
    
    print(f"[User Input] ID: {user_id}, Text: {user_text}")
    
    # 呼叫 RAG 系統 (包含 Pinecone 搜尋與 Gemini 生成)
    ai_response = get_chef_response(user_text, chat_history=[])
    
    # 用 requests 回傳結果
    reply_message_via_requests(reply_token, ai_response)

# --- 處理圖片訊息 (YOLO 核心) ---
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    user_id = event.source.user_id
    reply_token = event.reply_token
    message_id = event.message.id

    # 1. 透過 requests 下載圖片內容
    image_content_url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_ACCESS_TOKEN}"}
    
    print(f"[System] 正在下載圖片: {message_id}...")
    response = requests.get(image_content_url, headers=headers, stream=True)
    
    if response.status_code == 200:
        # 將二進制圖片轉為 OpenCV 格式
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # 2. 丟進 YOLO 進行偵測
        # 注意：這裡假設你的 yolo_module 還是維持原本的寫法
        vision_result = ai_chef.detect_and_parse(img, user_id=user_id)
        
        # 轉成文字描述
        ingredients = ", ".join([item['ingredient_name'] for item in vision_result['ingredients']])
        
        if not ingredients:
            reply_message_via_requests(reply_token, "看起來冰箱空空的，或者我沒看懂這張圖。請再試一次！")
            return

        print(f"[YOLO Result] {ingredients}")
        
        # (選擇性優化) 先回傳一句話讓使用者知道有偵測到，體驗會更好
        # reply_message_via_requests(reply_token, f"收到！我看到了 {ingredients}，正在為您翻閱 Sharp 食譜...")

        # 3. 構造 Prompt 並呼叫 RAG 生成食譜
        # 修改點：這裡的邏輯要配合新的 rag_service
        query_for_llm = f"我現在冰箱剩下這些食材：{ingredients}。請幫我推薦一道適合的料理，並告訴我怎麼做。"
        
        # 呼叫 RAG 系統
        try:
            ai_response = get_chef_response(user_input=query_for_llm, chat_history=[])
        except Exception as e:
            print(f"RAG Error: {e}")
            ai_response = "抱歉，我的大腦（RAG系統）暫時有點卡住，請稍後再試。"

        
        # 4. 回傳
        reply_message_via_requests(reply_token, ai_response)
        
    else:
        reply_message_via_requests(reply_token, "抱歉，圖片讀取失敗。")


if __name__ == "__main__":
    # 本地測試用 Port 5001
    app.run(port=5001, debug=True)