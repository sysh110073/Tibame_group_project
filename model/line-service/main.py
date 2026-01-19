import os
import json
import requests
from flask import Flask, request, abort

app = Flask(__name__)

# Line Bot SDK
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, PostbackEvent,
    TextSendMessage, FlexSendMessage, BubbleContainer, ImageComponent, BoxComponent,
    TextComponent, ButtonComponent, URIAction, PostbackAction, MessageAction,
    CarouselContainer
)
from urllib.parse import parse_qsl

# ... (existing imports)

# ==========================================
# 1. 從 Cloud Run 環境變數讀取設定
# ==========================================
LINE_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

# Check environment variables
if not LINE_ACCESS_TOKEN or not LINE_SECRET:
    print("❌ 錯誤：未讀取到 Line Bot 環境變數")

# Initialize Line Bot API and Handler
line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_SECRET)

# 微服務網址
YOLO_API_URL = os.environ.get('YOLO_API_URL') 
RAG_API_URL = os.environ.get('RAG_API_URL')
# 自動修正 trailing slash
if RAG_API_URL:
    RAG_API_URL = RAG_API_URL.rstrip('/')
    if RAG_API_URL.endswith('/api/chef'):
        RAG_API_URL = RAG_API_URL[:-9]


# ==========================================
# 2. Flex Message Helpers
# ==========================================

def create_recipe_bubble(recipe, ingredients=None):
    """建立單張食譜泡泡"""
    title = recipe.get('title', '美味食譜')
    summary = recipe.get('summary', '...')
    image_url = recipe.get('image_url')
    recipe_id = recipe.get('id')
    
    # Placeholder Logic
    if not image_url or not image_url.startswith("http"):
        image_url = "https://via.placeholder.com/1024x1024?text=Recipe"
    
    # Store ingredients in postback (limit checks omitted, keep it simple)
    data_str = f'action=view&id={recipe_id}'
    if ingredients:
        data_str += f'&ingr={ingredients}'

    return BubbleContainer(
        hero=ImageComponent(
            url=image_url,
            size='full',
            aspect_ratio='20:13',
            aspect_mode='cover',
            action=URIAction(uri=image_url, label='View Image')
        ),
        body=BoxComponent(
            layout='vertical',
            contents=[
                TextComponent(text=title, weight='bold', size='xl'),
                TextComponent(text=summary, margin='md', wrap=True, color='#666666', max_lines=3),
            ]
        ),
        footer=BoxComponent(
            layout='vertical',
            spacing='sm',
            contents=[
                ButtonComponent(
                    style='primary',
                    height='sm',
                    action=PostbackAction(label='View Recipe / 查看食譜', data=data_str)
                )
            ]
        )
    )

def create_recipe_carousel(recipes, ingredients=None):
    """建立 Carousel (旋轉木馬)"""
    bubbles = []
    # 最多顯示 10 張 (Line 限制 12，但我們抓 3 張)
    for r in recipes[:10]:
        bubbles.append(create_recipe_bubble(r, ingredients))
    
    return FlexSendMessage(
        alt_text="為您推薦多道食譜",
        contents=CarouselContainer(contents=bubbles)
    )

def create_feedback_flex(recipe_id, ingredients=None):
    """建立回饋按鈕 (想煮 / 不想煮 / 隨機推薦)"""
    # 1. Dislike Data
    dislike_data = f'action=dislike&id={recipe_id}'
    if ingredients:
        dislike_data += f'&ingr={ingredients}'
    
    # 2. Random Recommend Data
    rand_data = "action=recommend"
    if ingredients:
        rand_data += f"&ingr={ingredients}"
        
    bubble = BubbleContainer(
        body=BoxComponent(
            layout='vertical',
            contents=[
                TextComponent(text="🤔 覺得這道菜如何？", weight='bold', align='center'),
                BoxComponent(
                    layout='vertical', # Stack buttons vertically for better mobile view since there are 3
                    margin='md',
                    spacing='sm',
                    contents=[
                        # Row for Like/Dislike
                        BoxComponent(
                            layout='horizontal',
                            spacing='md',
                            contents=[
                                ButtonComponent(
                                    style='primary',
                                    action=PostbackAction(label='想煮這道菜', data=f'action=cook&id={recipe_id}')
                                ),
                                ButtonComponent(
                                    style='secondary',
                                    action=PostbackAction(label='不想煮這道菜', data=dislike_data)
                                )
                            ]
                        ),
                        # Button for Random Recommend
                        ButtonComponent(
                            style='link',
                            height='sm',
                            action=PostbackAction(label='🎲 再推薦一道菜', data=rand_data)
                        )
                    ]
                )
            ]
        )
    )
    return FlexSendMessage(alt_text="請給予回饋", contents=bubble)

def create_random_recommend_button(ingredients=None):
    """隨便推薦按鈕 (當都使用者不喜歡時)"""
    # Use Postback with ingredients context
    data_str = "action=recommend"
    if ingredients:
        data_str += f"&ingr={ingredients}"
        
    bubble = BubbleContainer(
        body=BoxComponent(
            layout='vertical',
            contents=[
                TextComponent(text="或是試試看別的？", align='center', size='sm', color='#aaaaaa'),
                ButtonComponent(
                    style='link',
                    height='sm',
                    action=PostbackAction(label='🎲 再推薦一道菜', data=data_str)
                )
            ]
        )
    )
    return FlexSendMessage(alt_text="試試別的推薦", contents=bubble)

# ==========================================
# 3. 服務呼叫
# ==========================================
def call_yolo_service(image_binary):
    if not YOLO_API_URL: return []
    try:
        files = {'file': ('line_image.jpg', image_binary, 'image/jpeg')}
        response = requests.post(YOLO_API_URL, files=files, timeout=30)
        if response.status_code == 200:
            predictions = response.json().get('data', [])
            found = [item.get('ingredient') for item in predictions if item.get('ingredient')]
            return list(set(found))
    except Exception as e:
        print(f"❌ YOLO Error: {e}")
    return []

def call_rag_chef(ingredients_text):
    if not RAG_API_URL: return []
    try:
        # 新版 API 回傳: {"recipes": [ {...}, {...}, {...} ]}
        # [FIX] Send 'ingredients' strictly for vector search, 'message' for LLM generation
        payload = {
            "message": f"我冰箱剩下：{ingredients_text}。請幫我設計一道食譜。", 
            "ingredients": ingredients_text, # Pure ingredients for search
            "need_image": True
        }
        # [Adjust] Timeout increased to 110s to tolerate Cold Start + Parallel Image Gen
        # Note: Line Reply Token expires in ~30s, so this may still result in user error on cold start, 
        # but prevents internal 500 error.
        response = requests.post(f"{RAG_API_URL}/api/chef", json=payload, timeout=110)
        if response.status_code == 200:
            return response.json().get('recipes', [])
    except Exception as e:
        print(f"❌ RAG Chef Error: {e}")
    return []

def call_random_recommend(user_id, ingredients=None):
    if not RAG_API_URL: return []
    try:
        payload = {"user_id": user_id}
        if ingredients:
            payload["ingredients"] = ingredients
            
        response = requests.post(f"{RAG_API_URL}/api/random_recommend", json=payload, timeout=110)
        if response.status_code == 200:
            return response.json().get('recipes', [])
    except Exception as e:
        print(f"❌ Random Recommend Error: {e}")
    return []

def get_full_recipe(recipe_id):
    if not RAG_API_URL: return None
    try:
        # Use query param for robust ID handling (Base64 contain /)
        response = requests.get(f"{RAG_API_URL}/api/recipe", params={"id": recipe_id}, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"❌ Get Recipe Error: {e}")
    return None

def send_like_feedback(user_id, recipe_id):
    if not RAG_API_URL: return
    try:
        # "想煮" = Positive Feedback (Like)
        requests.post(f"{RAG_API_URL}/api/like", json={"user_id": user_id, "recipe_id": recipe_id}, timeout=5)
    except Exception as e:
        print(f"❌ Like Error: {e}")

# ==========================================
# 4. Handlers
# ==========================================

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_id = event.message.id
    try:
        # 1. YOLO
        content = line_bot_api.get_message_content(message_id)
        ingredients = call_yolo_service(content.content)
        
        if not ingredients:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="看不清楚食材，請再試一次。"))
            return
            
        ingredients_str = ", ".join(ingredients)
        
        # 2. RAG Chef (Generate + Retrieve)
        recipes = call_rag_chef(ingredients_str) # returns list of recipes
        
        if not recipes:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="主廚暫時無法回應，請稍後再試。"))
            return

        # 3. Reply with Carousel (Pass ingredients_str!)
        flex_msg = create_recipe_carousel(recipes, ingredients=ingredients_str)
        line_bot_api.reply_message(event.reply_token, flex_msg)

    except Exception as e:
        print(f"❌ Handle Image Error: {e}")
        try:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="發生錯誤，請重試。"))
        except: pass

@handler.add(PostbackEvent)
def handle_postback(event):
    user_id = event.source.user_id
    data = event.postback.data # e.g., "action=view&id=123&ingr=Tomato"
    
    # Parse Query String (Robust Way)
    params = dict(parse_qsl(data))
    action = params.get('action')
    recipe_id = params.get('id')
    ingredients = params.get('ingr') # Extract ingredients context
    
    if action == 'view':
        # 取得完整食譜
        recipe_data = get_full_recipe(recipe_id)
        if recipe_data:
            text = recipe_data.get('text', 'No content')
            # 傳送: 圖文 + 回饋按鈕 (內含隨便推薦)
            replies = [
                TextSendMessage(text=text[:4900]),
                create_feedback_flex(recipe_id, ingredients)
            ]
            line_bot_api.reply_message(event.reply_token, replies)
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="找不到該食譜，可能已過期。"))
            
    elif action == 'cook':
        # 想煮 -> 正向回饋
        send_like_feedback(user_id, recipe_id)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="👨‍🍳 太棒了！已將您的偏好記錄下來！"))

    elif action == 'dislike':
        # 不想煮 -> 負向
        line_bot_api.reply_message(event.reply_token, [
            TextSendMessage(text="收到，下次幫您找別的！"),
            create_random_recommend_button(ingredients)
        ])
        
    elif action == 'recommend':
        # New Handler for Random Recommend Button
        recipes = call_random_recommend(user_id, ingredients)
        if recipes:
            flex_msg = create_recipe_carousel(recipes, ingredients)
            line_bot_api.reply_message(event.reply_token, flex_msg)
        else:
             line_bot_api.reply_message(event.reply_token, TextSendMessage(text="目前沒有更多推薦，請稍後再試！"))

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_input = event.message.text
    user_id = event.source.user_id

    if "隨便推薦" in user_input or "推薦" in user_input:
        # User typed it manually -> No ingredients context
        recipes = call_random_recommend(user_id)
        if recipes:
            flex_msg = create_recipe_carousel(recipes)
            line_bot_api.reply_message(event.reply_token, flex_msg)
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="目前沒有推薦的食譜，請多互動幾次讓我認識您！"))
        return

    # 當作食材輸入
    recipes = call_rag_chef(user_input)
    if recipes:
        flex_msg = create_recipe_carousel(recipes)
        line_bot_api.reply_message(event.reply_token, flex_msg)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)