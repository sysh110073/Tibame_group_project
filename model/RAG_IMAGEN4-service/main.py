import os
import base64
import uuid
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

app = Flask(__name__, static_folder='static')
if not os.path.exists('static'): os.makedirs('static')

import requests
import json

# 環境變數由 Cloud Run 注入
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

qa_chain = None

def init_rag_system():
    global qa_chain
    try:
        print("🚀 初始化 Google RAG 系統 (768維度)...")
        
        # 1. 使用 Google Embedding 模型 (不需下載，直接呼叫 API)
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        
        # 2. 連接 Pinecone (請確保 Index 是 768 維)
        index_name = "zero-waste-chef-recipes-csv"
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name, 
            embedding=embeddings
        )
        
        # 3. 建立 LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        
        # 4. 建立 RAG 鏈
        template = """你是零剩食主廚，請根據資料庫回答：{context} 
        問題：{question}
        回答："""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        print("✅ RAG 系統已就緒")
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")

def generate_image_prompt(recipe_text):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8)
        prompt = """
        You are an expert food photographer. 
        Extract the visual description from the recipe and create a prompt for Google Imagen 3.
        Recipe: {recipe}
        Requirements:
        1. Output ONLY the English prompt.
        2. Include tags: photorealistic, 8k, gourmet food photography, studio lighting, delicious, close-up.
        """
        return llm.invoke(prompt.format(recipe=recipe_text[:1000])).content.strip()
    except Exception as e:
        print(f"❌ Prompt 生成失敗: {e}")
        return "Delicious food, 4k, photorealistic"

def call_imagen_api(prompt):
    print(f"🎨 呼叫 Google Imagen 4 (Fast): {prompt[:30]}...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-fast-generate-001:predict?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1, "aspectRatio": "1:1"}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            b64_image = result.get('predictions', [{}])[0].get('bytesBase64Encoded')
            if b64_image:
                img_data = base64.b64decode(b64_image)
                filename = f"{uuid.uuid4()}.png"
                file_path = os.path.join('static', filename)
                with open(file_path, 'wb') as f:
                    f.write(img_data)
                print(f"✅ 圖片已儲存: {file_path}")
                # 強制使用 HTTPS (因為 Line Bot 要求圖片必須是 HTTPS)
                base_url = request.host_url.replace('http://', 'https://')
                return f"{base_url}static/{filename}"
        else:
            print(f"❌ Google Imagen 失敗 ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"❌ 連線異常: {e}")
    return None

@app.route('/api/chef', methods=['POST'])
def chef_api():
    if qa_chain is None: init_rag_system()
    
    data = request.json
    user_input = data.get('message', '')
    
    # 執行 RAG
    res = qa_chain.invoke({"question": user_input, "chat_history": []})
    answer = res["answer"]
    
    # 呼叫生圖 (Imagen 4)
    image_url = None
    if True: # Default to generating image
        img_prompt = generate_image_prompt(answer)
        image_url = call_imagen_api(img_prompt)
    
    return jsonify({"text": answer, "image_url": image_url})

if __name__ == "__main__":
    init_rag_system()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))